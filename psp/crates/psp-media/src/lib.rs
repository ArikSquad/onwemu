//! Linux media bridge for PSP MPEG streams.
//!
//! PSP PMF streams carry H.264 video and ATRAC3+ audio in an MPEG program
//! stream. The emulator feeds the exact bytes delivered through the guest MPEG
//! ring buffer into long-lived FFmpeg decoders. Keeping the decoders alive is
//! important: probing and rebuilding a codec for every access unit introduces
//! visible stalls and loses reference frames.

use std::io::{Read, Write};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use tracing::{debug, warn};

/// Native PSP video width in pixels.
pub const VIDEO_WIDTH: usize = 480;
/// Native PSP video height in pixels.
pub const VIDEO_HEIGHT: usize = 272;
/// RGBA8 bytes in one native PSP video frame.
pub const VIDEO_FRAME_BYTES: usize = VIDEO_WIDTH * VIDEO_HEIGHT * 4;
/// Size of one PCM chunk exposed by the MPEG HLE.
pub const AUDIO_CHUNK_BYTES: usize = 8_192;
/// ATRAC samples represented by one decoded frame.
pub const ATRAC_SAMPLES_PER_FRAME: u32 = 2_048;
/// Stereo signed-16 bytes represented by one ATRAC frame.
pub const ATRAC_OUTPUT_BYTES: usize = ATRAC_SAMPLES_PER_FRAME as usize * 2 * 2;

const INITIAL_VIDEO_WAIT: Duration = Duration::from_millis(500);
const AUDIO_FIRST_WAIT: Duration = Duration::from_millis(1_000);
const AUDIO_STEADY_WAIT: Duration = Duration::from_millis(50);
const FFMPEG_ENV: &str = "PSP_FFMPEG";

/// Container metadata needed by the managed PSP ATRAC API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtracFormat {
    /// Number of interleaved audio channels.
    pub channels: u32,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Decoded samples represented by one compressed frame.
    pub samples_per_frame: u32,
    /// Total decoded samples declared by the file.
    pub total_samples: u32,
    /// Offset of the compressed data in the input bytes.
    pub data_offset: usize,
    /// Number of compressed data bytes.
    pub data_bytes: usize,
    /// Size of one compressed frame in bytes.
    pub block_bytes: usize,
}

/// Inspect the RIFF/WAVE wrapper used by PSP `.AT3` files.
///
/// GTA's music and dialogue files are ordinary RIFF ATRAC3+ tracks. Keeping
/// this parser beside the FFmpeg bridge lets the HLE expose real buffer and
/// frame information before the host decoder emits its first PCM block.
pub fn inspect_atrac(bytes: &[u8]) -> Option<AtracFormat> {
    if bytes.get(..12)? != b"RIFF\0\0\0\0WAVE" {
        return None;
    }
    let mut cursor = 12usize;
    let mut channels = None;
    let mut sample_rate = None;
    let mut block_bytes = None;
    let mut total_samples = None;
    let mut data = None;
    while cursor.checked_add(8)? <= bytes.len() {
        let chunk = &bytes[cursor..cursor + 8];
        let size = usize::try_from(u32::from_le_bytes(chunk[4..8].try_into().ok()?)).ok()?;
        let payload = cursor.checked_add(8)?;
        let end = payload.checked_add(size)?.min(bytes.len());
        match &chunk[..4] {
            b"fmt " if end >= payload + 34 => {
                channels = Some(u32::from(u16::from_le_bytes(
                    bytes[payload + 2..payload + 4].try_into().ok()?,
                )));
                sample_rate = Some(u32::from_le_bytes(
                    bytes[payload + 4..payload + 8].try_into().ok()?,
                ));
                block_bytes = Some(usize::from(u16::from_le_bytes(
                    bytes[payload + 12..payload + 14].try_into().ok()?,
                )));
            }
            b"fact" if end >= payload + 4 => {
                total_samples = Some(u32::from_le_bytes(
                    bytes[payload..payload + 4].try_into().ok()?,
                ));
            }
            b"data" if payload <= bytes.len() => data = Some((payload, size)),
            _ => {}
        }
        cursor = payload.checked_add(size)?.checked_add(size & 1)?;
    }
    let (data_offset, data_bytes) = data?;
    let channels = channels?;
    let sample_rate = sample_rate?;
    let block_bytes = block_bytes?.max(1);
    let frame_count = data_bytes / block_bytes;
    let total_samples = total_samples.unwrap_or_else(|| {
        u32::try_from(frame_count)
            .unwrap_or(u32::MAX)
            .saturating_mul(ATRAC_SAMPLES_PER_FRAME)
    });
    (channels > 0 && sample_rate > 0).then_some(AtracFormat {
        channels,
        sample_rate,
        samples_per_frame: ATRAC_SAMPLES_PER_FRAME,
        total_samples,
        data_offset,
        data_bytes,
        block_bytes,
    })
}

#[derive(Clone, Copy, Debug)]
enum StreamKind {
    Video,
    Audio,
    Atrac,
}

impl StreamKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Atrac => "atrac",
        }
    }

    const fn output_size(self) -> usize {
        match self {
            Self::Video => VIDEO_FRAME_BYTES,
            Self::Audio | Self::Atrac => AUDIO_CHUNK_BYTES,
        }
    }
}

/// Long-lived host decoder for a managed PSP RIFF ATRAC3+ stream.
///
/// The guest still owns the circular input buffer and calls `submit` only for
/// bytes it has made available. The process stays alive across decode calls,
/// which is essential for ATRAC frame history and streaming dialogue.
pub struct AtracDecoder {
    decoder: FfmpegDecoder,
}

impl AtracDecoder {
    /// Start an ATRAC decoder using the configured FFmpeg executable.
    pub fn spawn() -> Option<Self> {
        Some(Self {
            decoder: FfmpegDecoder::spawn(StreamKind::Atrac)?,
        })
    }

    /// Submit one or more guest-owned ATRAC payload bytes.
    pub fn submit(&self, bytes: &[u8]) -> bool {
        self.decoder.submit(bytes).is_ok()
    }

    /// Take one decoded PCM block, waiting at most `wait` for the decoder.
    pub fn take(&self, wait: Duration) -> Option<Vec<u8>> {
        self.decoder.take(wait)
    }
}

/// Decoded PSP video and audio queues backed by the host's FFmpeg binary.
///
/// FFmpeg is deliberately an optional runtime dependency. A missing or unusable
/// binary leaves the HLE fallback available, which keeps boot and non-media
/// titles functional while Linux users with the normal multimedia stack receive
/// real H.264 and ATRAC3+ decoding.
#[derive(Default)]
pub struct MediaPipeline {
    video: Option<FfmpegDecoder>,
    audio: Option<FfmpegDecoder>,
    start_attempted: bool,
    audio_format: Option<(u32, usize)>,
    audio_stream: PsmfAudioStream,
    observed_video_frames: u64,
    observed_audio_chunks: u64,
    audio_underruns: u64,
    audio_mismatch_run: u32,
    /// Wire frames handed to FFmpeg minus PCM chunks taken by the game. Their
    /// difference is the decoded-audio backlog; gating new audio access units
    /// on it keeps the game from burning silent decodes for data the stream has
    /// not delivered yet.
    audio_frames_fed: u64,
    audio_chunks_taken: u64,
}

/// What to do with one confirmed wire frame given the latched bridge format.
#[derive(Debug, Eq, PartialEq)]
enum FormatDecision {
    Submit,
    Drop,
    Switch,
}

/// Pure policy for the format latch in [`MediaPipeline::submit`], kept
/// separate so the switching truth table is unit-testable without spawning
/// FFmpeg.
fn decide_audio_format(
    current: Option<(u32, usize)>,
    mismatch_run: u32,
    format: (u32, usize),
) -> (FormatDecision, u32) {
    const SWITCH_AFTER: u32 = 4;
    if current == Some(format) {
        (FormatDecision::Submit, 0)
    } else if current.is_none() || mismatch_run + 1 >= SWITCH_AFTER {
        (FormatDecision::Switch, 0)
    } else {
        (FormatDecision::Drop, mismatch_run + 1)
    }
}

impl MediaPipeline {
    /// Submit MPEG program-stream bytes in presentation order.
    pub fn submit(&mut self, packets: &[u8]) {
        if packets.is_empty() {
            return;
        }
        self.start_decoders();
        if let Some(decoder) = self.video.as_ref()
            && decoder.submit(packets).is_err()
        {
            self.video = None;
        }
        let audio_frames = self.audio_stream.extract(packets);
        for frame in audio_frames {
            // pmf audio frames declare their own size and channel count on
            // the wire, while the oma bridge re-frames its byte stream with
            // one constant size.  only confirmed, format-matching frames may
            // enter the pipe: one stray byte shifts every later frame and
            // stalls the decoder for good.  isolated strays are dropped; a
            // sustained new format (four in a row) switches the bridge.
            let format = (frame.channels, frame.bytes.len());
            let (decision, run) =
                decide_audio_format(self.audio_format, self.audio_mismatch_run, format);
            self.audio_mismatch_run = run;
            match decision {
                FormatDecision::Drop => {
                    debug!(
                        channels = frame.channels,
                        frame_bytes = frame.bytes.len(),
                        run = self.audio_mismatch_run,
                        "dropping off-format ATRAC frame"
                    );
                    continue;
                }
                FormatDecision::Switch => {
                    debug!(
                        channels = frame.channels,
                        frame_bytes = frame.bytes.len(),
                        "PSMF audio format observed; rebuilding ATRAC bridge"
                    );
                    self.audio = FfmpegDecoder::spawn_audio(frame.channels, frame.bytes.len());
                    self.audio_format = self.audio.as_ref().map(|_| format);
                    // chunks queued in the replaced decoder are lost with
                    // it; restart the backlog account so availability never
                    // promises a chunk that no longer exists.
                    self.audio_frames_fed = 0;
                    self.audio_chunks_taken = 0;
                }
                FormatDecision::Submit => {}
            }
            if let Some(decoder) = self.audio.as_ref() {
                if decoder.submit(&frame.bytes).is_err() {
                    self.audio = None;
                    self.audio_format = None;
                } else {
                    self.audio_frames_fed += 1;
                }
            }
        }
    }

    /// Take the next complete RGBA8888 frame, waiting briefly for the first
    /// asynchronous decode result to arrive.
    pub fn take_video_frame(&mut self) -> Option<Vec<u8>> {
        let wait = if self.observed_video_frames == 0 {
            INITIAL_VIDEO_WAIT
        } else {
            Duration::ZERO
        };
        let output = self.video.as_mut()?.take(wait);
        if output.is_some() {
            self.observed_video_frames += 1;
            if self.observed_video_frames == 1 {
                debug!("FFmpeg produced the first H.264 frame");
            } else if self.observed_video_frames.is_multiple_of(30) {
                debug!(
                    frames = self.observed_video_frames,
                    "FFmpeg video frames delivered"
                );
            }
        }
        output
    }

    /// Take the next 8192-byte stereo signed-16 PCM block at 44.1 kHz.
    ///
    /// always waits briefly: the ffmpeg bridge decodes asynchronously, so
    /// returning immediately whenever its output queue is momentarily empty
    /// splices 46 ms of silence into the game's audio on nearly every
    /// access unit.  a bounded wait matches hardware decode latency and
    /// keeps the stream continuous; only a genuinely starved decoder (end
    /// of stream) still yields `None`.
    pub fn take_audio_chunk(&mut self) -> Option<Vec<u8>> {
        let wait = if self.observed_audio_chunks == 0 {
            AUDIO_FIRST_WAIT
        } else {
            AUDIO_STEADY_WAIT
        };
        let output = self.audio.as_mut()?.take(wait);
        if output.is_some() {
            self.audio_chunks_taken += 1;
            self.observed_audio_chunks += 1;
            if self.observed_audio_chunks == 1 {
                debug!("FFmpeg produced the first ATRAC PCM chunk");
            }
        } else {
            self.audio_underruns += 1;
            debug!(
                underruns = self.audio_underruns,
                "MPEG audio underrun: FFmpeg had no chunk ready"
            );
        }
        output
    }

    /// Return whether at least one real host decoder started successfully.
    pub fn has_decoder(&self) -> bool {
        self.video.is_some() || self.audio.is_some()
    }

    /// Return whether the game may take another audio access unit.
    ///
    /// true while no atrac bridge exists (legacy ring behavior for decoder
    /// fallbacks and for streams whose first audio packet has not arrived
    /// yet), otherwise true only while fed-but-untaken frames back the
    /// request.
    pub fn audio_au_available(&self) -> bool {
        self.audio.is_none()
            || self.audio_format.is_none()
            || self.audio_frames_fed > self.audio_chunks_taken
    }

    fn start_decoders(&mut self) {
        if self.start_attempted {
            return;
        }
        self.start_attempted = true;
        self.video = FfmpegDecoder::spawn(StreamKind::Video);
        // the atrac bridge needs the observed frame size and channel count
        // for its oma header, so it is spawned lazily on the first frame.
    }
}

#[derive(Default)]
struct PsmfAudioStream {
    pending: Vec<u8>,
    audio: Vec<u8>,
}

/// One ATRAC access unit parsed from the program stream, ready for decoding.
///
/// `bytes` is the decoder payload with the 8-byte wire header (`0F D0` sync
/// plus channel/size codes) stripped, matching what ffmpeg's atrac3+
/// decoder expects after the oma container header.
struct AudioFrame {
    channels: u32,
    bytes: Vec<u8>,
}

impl PsmfAudioStream {
    fn extract(&mut self, packets: &[u8]) -> Vec<AudioFrame> {
        self.pending.extend_from_slice(packets);
        let mut pos = 0usize;
        while let Some(start) = find_start_code_at(&self.pending, pos) {
            let Some(code) = self.pending.get(start + 3).copied() else {
                pos = start;
                break;
            };
            match code {
                // pack header is a fixed 14 bytes.
                0xba => {
                    if start + 14 > self.pending.len() {
                        pos = start;
                        break;
                    }
                    pos = start + 14;
                }
                // system/padding/map headers and video pes packets are
                // length-prefixed; only 0xbd carries audio.
                0xbb | 0xbc | 0xbe | 0xbf | 0xe0..=0xef => {
                    if start + 6 > self.pending.len() {
                        pos = start;
                        break;
                    }
                    let length = usize::from(u16::from_be_bytes([
                        self.pending[start + 4],
                        self.pending[start + 5],
                    ]));
                    let Some(end) = start.checked_add(6 + length) else {
                        pos = start;
                        break;
                    };
                    if end > self.pending.len() {
                        pos = start;
                        break;
                    }
                    pos = end;
                }
                0xbd => {
                    if start + 6 > self.pending.len() {
                        pos = start;
                        break;
                    }
                    let length = usize::from(u16::from_be_bytes([
                        self.pending[start + 4],
                        self.pending[start + 5],
                    ]));
                    let Some(end) = start.checked_add(6 + length) else {
                        pos = start;
                        break;
                    };
                    if end > self.pending.len() {
                        pos = start;
                        break;
                    }
                    if let Some((payload_start, payload_end)) =
                        audio_payload_range(&self.pending[start..end])
                    {
                        self.audio.extend_from_slice(
                            &self.pending[start + payload_start..start + payload_end],
                        );
                    }
                    pos = end;
                }
                // unknown start code: step over it so the scan always
                // makes progress.
                _ => pos = start + 1,
            }
        }
        // bytes without a complete start code are a packet tail that may
        // complete on the next submit.  guard against unbounded growth if
        // the byte stream ever stops looking like mpeg.
        if self.pending.len() - pos.min(self.pending.len()) > 1_048_576 {
            warn!("PSMF audio parser dropped a megabyte of unparseable bytes");
            pos = self.pending.len().saturating_sub(3);
        }
        if pos != 0 {
            self.pending.drain(..pos.min(self.pending.len()));
        }
        pop_audio_frames(&mut self.audio)
    }
}

fn find_start_code_at(bytes: &[u8], from: usize) -> Option<usize> {
    let mut index = from;
    while index + 3 < bytes.len() {
        if bytes[index] == 0 && bytes[index + 1] == 0 && bytes[index + 2] == 1 {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Locate the ATRAC byte range inside one private-stream-1 PES packet.
///
/// After the standard PES timestamp header comes one substream (channel) byte
/// plus a 3-byte (or, for channels `0xb0..=0xbf`, 4-byte) subheader. The old
/// parser mistook the PES extension bytes (`1e 60 04`) found in some packets
/// for an ATRAC frame marker and dropped packets that did not contain it.
fn audio_payload_range(packet: &[u8]) -> Option<(usize, usize)> {
    let end = packet.len();
    let mut pos = 6usize;
    // skip stuffing bytes.
    while pos < end && packet[pos] == 0xff {
        pos += 1;
    }
    if pos >= end {
        return None;
    }
    let mut prefix = packet[pos];
    pos += 1;
    if (prefix & 0xc0) == 0x40 {
        if pos + 2 > end {
            return None;
        }
        pos += 1;
        prefix = packet[pos];
        pos += 1;
    }
    if (prefix & 0xe0) == 0x20 {
        pos += 4;
        if (prefix & 0x10) != 0 {
            pos += 5;
        }
    } else if (prefix & 0xc0) == 0x80 {
        if pos + 2 > end {
            return None;
        }
        let mut flags = packet[pos];
        let mut header_length = usize::from(packet[pos + 1]);
        pos += 2;
        if (flags & 0x80) != 0 {
            pos += 5;
            header_length = header_length.saturating_sub(5);
            if (flags & 0x40) != 0 {
                pos += 5;
                header_length = header_length.saturating_sub(5);
            }
        }
        if (flags & 0x3f) != 0 && header_length == 0 {
            flags &= 0xc0;
        }
        if (flags & 0x01) != 0 {
            if pos >= end {
                return None;
            }
            let extension = packet[pos];
            pos += 1;
            header_length = header_length.saturating_sub(1);
            let mut skip = usize::from((extension >> 4) & 0x0b);
            skip += skip & 0x09;
            if (extension & 0x40) != 0 || skip > header_length {
                skip = 0;
            }
            pos += skip;
            header_length = header_length.saturating_sub(skip);
            if (extension & 0x01) != 0 {
                if pos >= end {
                    return None;
                }
                let extension2_length = packet[pos];
                pos += 1;
                header_length = header_length.saturating_sub(1);
                if (extension2_length & 0x7f) != 0 {
                    // sub-stream extension id; only advances the cursor,
                    // channel selection accepts every audio sub-stream.
                    pos += 1;
                    header_length = header_length.saturating_sub(1);
                }
            }
        }
        pos += header_length;
    }
    if pos >= end {
        return None;
    }
    let channel = packet[pos];
    pos += 1;
    // standard psp audio sub-streams carry a 3-byte sub-header here (4
    // bytes for channels 0xb0..=0xbf); other sub-streams on this private
    // stream also reserve 3 bytes before their payload.
    pos += 3;
    if (0xb0..=0xbf).contains(&channel) {
        pos += 1;
    }
    if pos > end {
        return None;
    }
    Some((pos, end))
}
/// Split complete ATRAC frames off the continuous audio byte stream.
///
/// wire frames start with the `0F D0` sync header; the frame size (including the 8-byte
/// header) is `(((code1 & 3) << 8) | (code2 * 8)) + 0x10`.  frames span pes
/// packets, so the sync scan resynchronizes after garbage or chapter
/// transitions instead of assuming packet alignment.
///
/// A frame is emitted only when a second sync confirms it exactly
/// `frame_size` bytes later.  the oma bridge re-frames its byte stream with
/// one constant size, so a single false-sync hit (random `0F D0` bytes
/// inside compressed data) would shift every later frame and stall the
/// decoder permanently; the confirmation requirement makes such hits
/// harmless.  the trailing unconfirmed frame is held back for the next
/// submit rather than dropped.
fn pop_audio_frames(audio: &mut Vec<u8>) -> Vec<AudioFrame> {
    const WIRE_HEADER_BYTES: usize = 8;
    const MAX_FRAME_BYTES: usize = 8_192;
    let mut frames = Vec::new();
    let mut pos = 0usize;
    loop {
        let mut cursor = pos;
        let mut sync = None;
        while cursor + 4 <= audio.len() {
            if audio[cursor] == 0x0f && audio[cursor + 1] == 0xd0 {
                sync = Some(cursor);
                break;
            }
            cursor += 1;
        }
        let Some(header) = sync else {
            break;
        };
        let code1 = audio[header + 2];
        let code2 = audio[header + 3];
        let frame_size = (((usize::from(code1) & 0x03) << 8) | (usize::from(code2) * 8)) + 0x10;
        if frame_size <= WIRE_HEADER_BYTES || frame_size > MAX_FRAME_BYTES {
            // false sync inside compressed data; keep scanning past it.
            pos = header + 2;
            continue;
        }
        if header.saturating_add(frame_size + 2) > audio.len() {
            // incomplete trailing frame, or no room to confirm it yet;
            // wait for more bytes, dropping any garbage that preceded it.
            pos = header;
            break;
        }
        if audio[header + frame_size] != 0x0f || audio[header + frame_size + 1] != 0xd0 {
            // unconfirmed: a lone sync with no frame following it exactly
            // one frame later.  never feed these bytes to the decoder.
            pos = header + 2;
            continue;
        }
        let channels = if code1 == 0x24 { 1u32 } else { 2u32 };
        let payload = audio[header + WIRE_HEADER_BYTES..header + frame_size].to_vec();
        if !payload.is_empty() {
            frames.push(AudioFrame {
                channels,
                bytes: payload,
            });
        }
        pos = header + frame_size;
    }
    audio.drain(..pos.min(audio.len()));
    frames
}

fn atrac_oma_header(channels: u32, frame_bytes: usize) -> Vec<u8> {
    // ffmpeg's oma demuxer provides a stable streaming entry point for its
    // atrac3+ decoder. the psp pmf packets carry variable-size coded
    // frames (gta vcs uses 744-byte stereo frames), so the header is built
    // from the observed wire format instead of one hardcoded guess.
    let mut header = b"ea3\x03\x00\x00\x00\x00\x00\x0a".to_vec();
    header.extend_from_slice(&[0; 10]);
    let mut ea3 = [0; 96];
    ea3[..4].copy_from_slice(b"EA3\0");
    ea3[5] = 96;
    ea3[6..8].copy_from_slice(&0xffffu16.to_be_bytes());
    ea3[32] = 1;
    let codec_params = (1u32 << 13) | (channels << 10) | (frame_bytes as u32 / 8).saturating_sub(1);
    ea3[33..36].copy_from_slice(&codec_params.to_be_bytes()[1..]);
    header.extend_from_slice(&ea3);
    header
}

struct FfmpegDecoder {
    input: Sender<Vec<u8>>,
    output: Receiver<Vec<u8>>,
}

impl FfmpegDecoder {
    /// Spawn the OMA/ATRAC3+ bridge for one observed PMF audio format.
    fn spawn_audio(channels: u32, frame_bytes: usize) -> Option<Self> {
        let decoder = Self::spawn(StreamKind::Audio)?;
        if decoder
            .submit(&atrac_oma_header(channels, frame_bytes))
            .is_err()
        {
            warn!("FFmpeg ATRAC bridge rejected its OMA header");
            return None;
        }
        Some(decoder)
    }

    fn spawn(kind: StreamKind) -> Option<Self> {
        let program = std::env::var_os(FFMPEG_ENV).unwrap_or_else(|| "ffmpeg".into());
        let mut command = Command::new(&program);
        command
            .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        match kind {
            StreamKind::Video => {
                command.args([
                    "-f",
                    "mpeg",
                    "-i",
                    "pipe:0",
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    "scale=480:272:flags=bicubic",
                    "-pix_fmt",
                    "rgba",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ]);
            }
            StreamKind::Audio => {
                command.args([
                    "-f", "oma", "-i", "pipe:0", "-map", "0:a:0", "-vn", "-ac", "2", "-ar",
                    "44100", "-f", "s16le", "pipe:1",
                ]);
            }
            StreamKind::Atrac => {
                command.args([
                    "-f", "wav", "-i", "pipe:0", "-map", "0:a:0", "-vn", "-ac", "2", "-ar",
                    "44100", "-f", "s16le", "pipe:1",
                ]);
            }
        }

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                warn!(
                    decoder = kind.name(),
                    program = ?program,
                    %error,
                    "FFmpeg decoder unavailable; using HLE media fallback"
                );
                return None;
            }
        };
        let Some(stdin) = child.stdin.take() else {
            warn!(decoder = kind.name(), "FFmpeg decoder has no stdin");
            let _ = child.kill();
            return None;
        };
        let Some(stdout) = child.stdout.take() else {
            warn!(decoder = kind.name(), "FFmpeg decoder has no stdout");
            let _ = child.kill();
            return None;
        };
        let Some(stderr) = child.stderr.take() else {
            warn!(decoder = kind.name(), "FFmpeg decoder has no stderr");
            let _ = child.kill();
            return None;
        };
        let (input, input_rx) = mpsc::channel();
        let (output_tx, output) = mpsc::channel();
        let thread_name = format!("psp-mpeg-{}", kind.name());
        if thread::Builder::new()
            .name(thread_name)
            .spawn(move || decoder_worker(child, stdin, stdout, stderr, input_rx, output_tx, kind))
            .is_err()
        {
            warn!(decoder = kind.name(), "cannot start FFmpeg bridge thread");
            return None;
        }
        Some(Self { input, output })
    }

    fn submit(&self, packets: &[u8]) -> Result<(), ()> {
        self.input.send(packets.to_vec()).map_err(|_| ())
    }

    fn take(&self, wait: Duration) -> Option<Vec<u8>> {
        self.output.recv_timeout(wait).ok()
    }
}

fn decoder_worker(
    mut child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    stderr: ChildStderr,
    input_rx: Receiver<Vec<u8>>,
    output_tx: Sender<Vec<u8>>,
    kind: StreamKind,
) {
    let writer = thread::spawn(move || {
        let mut stdin = stdin;
        let mut submitted = false;
        for packet in input_rx {
            if !submitted {
                debug!(
                    decoder = kind.name(),
                    bytes = packet.len(),
                    "FFmpeg received MPEG input"
                );
                submitted = true;
            }
            if stdin.write_all(&packet).is_err() {
                break;
            }
        }
    });
    let reader = thread::spawn(move || {
        let mut stdout = stdout;
        let mut produced = false;
        loop {
            let mut output = vec![0; kind.output_size()];
            if stdout.read_exact(&mut output).is_err() {
                break;
            }
            if !produced {
                debug!(
                    decoder = kind.name(),
                    bytes = output.len(),
                    "FFmpeg emitted decoder output"
                );
                produced = true;
            }
            if output_tx.send(output).is_err() {
                break;
            }
        }
    });
    let error_reader = thread::spawn(move || {
        let mut stderr = stderr;
        // drain eagerly and keep only the tail: a chatty decoder must never
        // block on a full stderr pipe, which would freeze its stdin reader
        // and silently stall the whole bridge.
        let mut tail = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            match stderr.read(&mut chunk) {
                Ok(0) => break,
                Ok(read) => {
                    tail.extend_from_slice(&chunk[..read]);
                    const MAX_TAIL: usize = 8_192;
                    if tail.len() > MAX_TAIL {
                        tail.drain(..tail.len() - MAX_TAIL);
                    }
                }
                Err(_) => break,
            }
        }
        tail
    });
    let _ = writer.join();
    let _ = reader.join();
    let status = child.wait();
    let tail = error_reader.join().unwrap_or_default();
    if !tail.is_empty() {
        debug!(
            decoder = kind.name(),
            stderr = %String::from_utf8_lossy(&tail),
            "FFmpeg decoder diagnostics"
        );
    }
    debug!(decoder = kind.name(), ?status, "FFmpeg decoder stopped");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psp_media_geometry_is_stable() {
        assert_eq!(VIDEO_FRAME_BYTES, 522_240);
        assert_eq!(AUDIO_CHUNK_BYTES, 8_192);
        assert_eq!(ATRAC_OUTPUT_BYTES, AUDIO_CHUNK_BYTES);
        assert_eq!(StreamKind::Video.output_size(), VIDEO_FRAME_BYTES);
        assert_eq!(StreamKind::Audio.output_size(), AUDIO_CHUNK_BYTES);
        assert_eq!(StreamKind::Atrac.output_size(), ATRAC_OUTPUT_BYTES);
    }

    #[test]
    fn demuxes_wire_frames_across_both_pes_header_variants() {
        // wire frame: 0f d0 sync, code1/code2 sizing, 8-byte header.
        // code1 0x28 / code2 0x5c -> (((0) << 8) | (0x5c * 8)) + 0x10 = 752.
        fn wire_frame(fill: u8) -> Vec<u8> {
            let mut frame = vec![0x0f, 0xd0, 0x28, 0x5c, 0x11, 0x22, 0x33, 0x44];
            frame.extend(std::iter::repeat_n(fill, 752 - 8));
            frame
        }
        let first = wire_frame(0x61);
        let second = wire_frame(0x62);
        let third = wire_frame(0x63);
        let fourth = wire_frame(0x64);
        // split the first frame across two packets: 200 payload bytes in
        // the short-header packet, the rest plus two full frames and half
        // of a fourth in the long-header packet.
        let head = &first[..200];
        let tail = [&first[200..], &second[..], &third[..], &fourth[..752 / 2]].concat();

        // short pes header variant: flags 81 80, header length 5.
        let mut short = vec![0, 0, 1, 0xbd];
        let short_payload = [
            vec![
                0x81, 0x80, 0x05, 0x21, 0x00, 0x63, 0x88, 0xef, 0x00, 0x00, 0x02, 0x3e,
            ],
            head.to_vec(),
        ]
        .concat();
        short.extend_from_slice(&(short_payload.len() as u16).to_be_bytes());
        short.extend_from_slice(&short_payload);

        // long pes header variant with extension bytes (81 81 08 ... 1e).
        let mut long = vec![0, 0, 1, 0xbd];
        let long_payload = [
            vec![
                0x81, 0x81, 0x08, 0x21, 0x00, 0x65, 0xcf, 0x77, 0x1e, 0x60, 0x04, 0x00, 0x00, 0x00,
                0x1e,
            ],
            tail.clone(),
        ]
        .concat();
        long.extend_from_slice(&(long_payload.len() as u16).to_be_bytes());
        long.extend_from_slice(&long_payload);

        let mut stream = PsmfAudioStream::default();
        // in-order delivery reassembles the split frame.  each of the
        // first three frames is confirmed by the next sync; the trailing
        // half frame is held back, never emitted unconfirmed.
        let frames = stream.extract(&short);
        assert!(frames.is_empty());
        let frames = stream.extract(&long);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].channels, 2);
        assert_eq!(frames[0].bytes, first[8..]);
        assert_eq!(frames[1].bytes, second[8..]);
        assert_eq!(frames[2].bytes, third[8..]);

        // completing the fourth frame plus a fifth releases the fourth and
        // holds the fifth.
        let fifth = wire_frame(0x65);
        let rest_payload = [
            vec![
                0x81, 0x80, 0x05, 0x21, 0x00, 0x63, 0x88, 0xef, 0x00, 0x00, 0x02, 0x3e,
            ],
            [&fourth[752 / 2..], &fifth[..]].concat(),
        ]
        .concat();
        let mut rest = vec![0, 0, 1, 0xbd];
        rest.extend_from_slice(&(rest_payload.len() as u16).to_be_bytes());
        rest.extend_from_slice(&rest_payload);
        let frames = stream.extract(&rest);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, fourth[8..]);
    }

    #[test]
    fn unconfirmed_syncs_are_never_emitted() {
        // a lone sync (false positive inside compressed data, or a frame
        // whose successor has not arrived yet) must not reach the decoder:
        // one wrong byte shifts the oma bridge framing permanently.
        let mut audio = vec![0x0f, 0xd0, 0x28, 0x5c, 0x99];
        audio.extend(std::iter::repeat_n(0x77, 752 - 5));
        let frames = pop_audio_frames(&mut audio);
        assert!(frames.is_empty());
        // the bytes are retained, so a later submit can still confirm them.
        assert_eq!(audio.len(), 752);
        audio.extend_from_slice(&[0x0f, 0xd0, 0x28, 0x5c]);
        let frames = pop_audio_frames(&mut audio);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes.len(), 752 - 8);
    }

    #[test]
    fn frame_scanner_resynchronizes_after_garbage() {
        let mut audio = vec![0xde, 0xad, 0x0f, 0xbe, 0xef];
        audio.extend_from_slice(&[0x0f, 0xd0, 0x24, 0x10, 1, 2, 3, 4]);
        // code1 0x24 / code2 0x10 -> ((0) | 0x80) + 0x10 = 144.
        audio.extend(std::iter::repeat_n(0x77, 144 - 8));
        // confirming sync for the mono frame plus a held trailing frame.
        audio.extend_from_slice(&[0x0f, 0xd0, 0x24, 0x10, 9, 9, 9, 9]);
        audio.extend(std::iter::repeat_n(0x78, 144 - 8));
        let frames = pop_audio_frames(&mut audio);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].channels, 1);
        assert_eq!(frames[0].bytes.len(), 144 - 8);
        // the trailing unconfirmed frame stays buffered.
        assert_eq!(audio.len(), 144);
    }

    #[test]
    fn format_latch_drops_strays_and_switches_on_runs() {
        assert_eq!(
            decide_audio_format(None, 0, (2, 744)),
            (FormatDecision::Switch, 0)
        );
        assert_eq!(
            decide_audio_format(Some((2, 744)), 0, (2, 744)),
            (FormatDecision::Submit, 0)
        );
        assert_eq!(
            decide_audio_format(Some((2, 744)), 0, (2, 100)),
            (FormatDecision::Drop, 1)
        );
        assert_eq!(
            decide_audio_format(Some((2, 744)), 2, (2, 744)),
            (FormatDecision::Submit, 0)
        );
        assert_eq!(
            decide_audio_format(Some((2, 744)), 2, (1, 376)),
            (FormatDecision::Drop, 3)
        );
        assert_eq!(
            decide_audio_format(Some((2, 744)), 3, (1, 376)),
            (FormatDecision::Switch, 0)
        );
    }

    #[test]
    fn oma_header_describes_observed_format() {
        // gta vcs intro audio: stereo, 744-byte decoder frames.
        let header = atrac_oma_header(2, 744);
        assert_eq!(&header[..3], b"ea3");
        assert_eq!(&header[20..24], b"EA3\0");
        assert_eq!(header[52], 1);
        assert_eq!(
            u32::from_be_bytes([0, header[53], header[54], header[55]]),
            (1 << 13) | (2 << 10) | (744 / 8 - 1)
        );
        let mono = atrac_oma_header(1, 376);
        assert_eq!(
            u32::from_be_bytes([0, mono[53], mono[54], mono[55]]),
            (1 << 13) | (1 << 10) | (376 / 8 - 1)
        );
    }

    #[test]
    fn parses_psp_riff_atrac_metadata() {
        let mut wav = vec![0; 104];
        wav[..4].copy_from_slice(b"RIFF");
        wav[8..12].copy_from_slice(b"WAVE");
        wav[12..16].copy_from_slice(b"fmt ");
        wav[16..20].copy_from_slice(&52u32.to_le_bytes());
        wav[20..22].copy_from_slice(&0xfffeu16.to_le_bytes());
        wav[22..24].copy_from_slice(&2u16.to_le_bytes());
        wav[24..28].copy_from_slice(&44_100u32.to_le_bytes());
        wav[32..34].copy_from_slice(&280u16.to_le_bytes());
        wav[72..76].copy_from_slice(b"fact");
        wav[76..80].copy_from_slice(&8u32.to_le_bytes());
        wav[80..84].copy_from_slice(&4096u32.to_le_bytes());
        wav[88..92].copy_from_slice(b"data");
        wav[92..96].copy_from_slice(&560u32.to_le_bytes());
        wav.resize(656, 0);

        assert_eq!(
            inspect_atrac(&wav),
            Some(AtracFormat {
                channels: 2,
                sample_rate: 44_100,
                samples_per_frame: ATRAC_SAMPLES_PER_FRAME,
                total_samples: 4096,
                data_offset: 96,
                data_bytes: 560,
                block_bytes: 280,
            })
        );
    }
}
