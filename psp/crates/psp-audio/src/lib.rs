//! Guest audio queues and the optional SDL3 host sink.
//!
//! Guest code submits interleaved stereo PCM to this crate. The bounded guest
//! queue keeps accelerated execution from using unlimited host memory, while
//! the SDL stream is optional so headless tests and machines without an audio
//! device can still run the emulator.

use std::collections::VecDeque;
#[cfg(feature = "host-audio")]
use std::thread;
#[cfg(feature = "host-audio")]
use std::time::Duration;

#[cfg(feature = "host-audio")]
use sdl3::audio::{AudioFormat, AudioSpec, AudioStreamOwner};

mod sas;

pub use sas::SasMixer;

const DEFAULT_SAMPLE_CAPACITY: usize = 65_536;
#[cfg(feature = "host-audio")]
const HOST_QUEUE_LIMIT_BYTES: i32 = 44_100;

/// Stereo PCM queue shared by HLE audio services and the frontend.
pub struct AudioEngine {
    samples: VecDeque<[i16; 2]>,
    capacity: usize,
    #[cfg(feature = "host-audio")]
    host: Option<HostOutput>,
    /// Number of samples requested after the guest queue was empty.
    pub underruns: u64,
    /// Number of samples rejected because the guest queue was full.
    pub overruns: u64,
    /// Submitted PCM bytes per guest source tag. These counters help separate
    /// a silent game from a host sink that is being fed but not playing.
    ///
    /// The fields below correspond to the source tags used by the frontend.
    pub channel_bytes: u64,
    /// Bytes submitted by the generic `src` audio path.
    pub src_bytes: u64,
    /// Bytes submitted by the MPEG audio path.
    pub mpeg_bytes: u64,
    /// Bytes submitted by the SAS mixer.
    pub sas_bytes: u64,
    /// Bytes submitted by any other or unknown source.
    pub other_bytes: u64,
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self {
            samples: VecDeque::new(),
            capacity: DEFAULT_SAMPLE_CAPACITY,
            #[cfg(feature = "host-audio")]
            host: None,
            underruns: 0,
            overruns: 0,
            channel_bytes: 0,
            src_bytes: 0,
            mpeg_bytes: 0,
            sas_bytes: 0,
            other_bytes: 0,
        }
    }
}

impl AudioEngine {
    /// Queue one stereo sample. Returns `false` when the bounded queue is full.
    pub fn push(&mut self, sample: [i16; 2]) -> bool {
        if self.samples.len() == self.capacity {
            self.overruns += 1;
            false
        } else {
            self.samples.push_back(sample);
            true
        }
    }

    /// Pop the oldest stereo sample, returning silence and counting an
    /// underrun when the queue is empty.
    pub fn pop(&mut self) -> [i16; 2] {
        self.samples.pop_front().unwrap_or_else(|| {
            self.underruns += 1;
            [0, 0]
        })
    }

    /// Start the SDL3 playback stream used by the graphical frontend.
    ///
    /// SDL owns and drains the device stream independently. Submission waits
    /// at a bounded queue threshold so accelerated guest execution cannot build
    /// an unbounded audio backlog. Set `PSP_AUDIO=off` to disable host playback.
    #[cfg(feature = "host-audio")]
    pub fn enable_host_output(&mut self) -> bool {
        if self.host.is_some() {
            return true;
        }
        if std::env::var_os("PSP_AUDIO").is_some_and(|value| {
            let value = value.to_string_lossy();
            value.eq_ignore_ascii_case("off") || value.eq_ignore_ascii_case("false")
        }) {
            return false;
        }
        self.host = match HostOutput::open() {
            Ok(host) => Some(host),
            Err(error) => {
                eprintln!("SDL3 audio unavailable: {error}");
                None
            }
        };
        self.host.is_some()
    }

    /// Return whether the optional host stream is currently open.
    #[cfg(feature = "host-audio")]
    pub fn host_output_enabled(&self) -> bool {
        self.host.is_some()
    }

    /// Submit interleaved little-endian stereo signed-16 samples to the guest
    /// queue and, when enabled, the Linux host sink.
    pub fn submit_pcm(&mut self, pcm: &[u8]) {
        self.submit_pcm_from("unknown", pcm);
    }

    /// Submit PCM and attribute its bytes to a source counter.
    pub fn submit_pcm_from(&mut self, source: &'static str, pcm: &[u8]) {
        for stereo in pcm.as_chunks::<4>().0.iter() {
            let _ = self.push([
                i16::from_le_bytes([stereo[0], stereo[1]]),
                i16::from_le_bytes([stereo[2], stereo[3]]),
            ]);
        }
        match source {
            "channel" => self.channel_bytes += pcm.len() as u64,
            "src" => self.src_bytes += pcm.len() as u64,
            "mpeg" => self.mpeg_bytes += pcm.len() as u64,
            "sas" => self.sas_bytes += pcm.len() as u64,
            _ => self.other_bytes += pcm.len() as u64,
        }
        #[cfg(feature = "host-audio")]
        {
            let Some(host) = self.host.as_ref() else {
                return;
            };
            if host.submit(pcm).is_err() {
                self.host = None;
            }
        }
    }
}

#[cfg(feature = "host-audio")]
struct HostOutput {
    stream: AudioStreamOwner,
    _context: sdl3::Sdl,
}

#[cfg(feature = "host-audio")]
impl HostOutput {
    fn open() -> Result<Self, String> {
        let context =
            sdl3::init().map_err(|error| format!("SDL initialization failed: {error}"))?;
        let audio = context
            .audio()
            .map_err(|error| format!("SDL audio subsystem failed: {error}"))?;
        let spec = AudioSpec {
            freq: Some(44_100),
            channels: Some(2),
            format: Some(AudioFormat::s16_sys()),
        };
        let stream = audio
            .default_playback_device()
            .open_device_stream(Some(&spec))
            .map_err(|error| format!("SDL playback device failed: {error}"))?;
        stream
            .resume()
            .map_err(|error| format!("SDL playback resume failed: {error}"))?;
        Ok(Self {
            stream,
            _context: context,
        })
    }

    fn submit(&self, pcm: &[u8]) -> Result<(), ()> {
        loop {
            let queued = self.stream.queued_bytes().map_err(|_| ())?;
            if queued <= HOST_QUEUE_LIMIT_BYTES {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        self.stream.put_data(pcm).map_err(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_submission_preserves_stereo_samples_without_a_host() {
        let mut audio = AudioEngine::default();
        audio.submit_pcm(&[0x34, 0x12, 0xcd, 0xab, 0xff, 0xff, 0x00, 0x80]);
        assert_eq!(audio.pop(), [0x1234, -0x5433]);
        assert_eq!(audio.pop(), [-1, -0x8000]);
    }

    #[cfg(feature = "host-audio")]
    #[test]
    fn default_engine_does_not_spawn_a_host_process() {
        assert!(!AudioEngine::default().host_output_enabled());
    }
}
