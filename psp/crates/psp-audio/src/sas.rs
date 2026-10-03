use psp_memory::{GuestMemory, Memory, MemoryFault};

const VOICE_COUNT: usize = 32;
const PITCH_BASE: u32 = 0x1000;
const PITCH_MASK: u32 = 0x0fff;
const VOLUME_MAX: i32 = 0x1000;
const VAG_SAMPLES_PER_BLOCK: usize = 28;
const VAG_BLOCK_BYTES: u32 = 16;

const VAG_FILTERS: [[i32; 2]; 16] = [
    [0, 0],
    [60, 0],
    [115, 52],
    [98, 55],
    [122, 60],
    [0, 0],
    [0, 0],
    [52, 0],
    [55, 2],
    [60, 125],
    [0, 0],
    [0, 91],
    [0, 0],
    [2, 216],
    [125, 6],
    [0, 151],
];

/// HLE implementation of the PSP's software SAS mixer.
///
/// Keep this state separate from the hardware audio sink: games configure
/// voices through sceSasCore, mix one grain into guest memory, and only then
/// does the frontend submit the resulting stereo PCM to the host sink.
pub struct SasMixer {
    initialized: bool,
    grain_size: usize,
    output_mode: u32,
    voices: [SasVoice; VOICE_COUNT],
}

impl Default for SasMixer {
    fn default() -> Self {
        Self {
            initialized: false,
            grain_size: 0,
            output_mode: 0,
            voices: [SasVoice::default(); VOICE_COUNT],
        }
    }
}

impl SasMixer {
    /// Initialize the mixer for one of the PSP's supported 44.1 kHz modes.
    pub fn initialize(
        &mut self,
        grain_size: usize,
        max_voices: usize,
        output_mode: u32,
        sample_rate: u32,
    ) -> bool {
        if !(0x40..=0x800).contains(&grain_size)
            || grain_size & 0x1f != 0
            || max_voices == 0
            || max_voices > VOICE_COUNT
            || output_mode > 1
            || sample_rate != 44_100
        {
            return false;
        }
        self.initialized = true;
        self.grain_size = grain_size;
        self.output_mode = output_mode;
        for voice in &mut self.voices {
            voice.reset_runtime();
            voice.playing = false;
            voice.on = false;
        }
        true
    }

    /// Return whether `initialize` has accepted the current configuration.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Return the number of samples mixed into one guest grain.
    pub fn grain_size(&self) -> usize {
        self.grain_size
    }

    /// Return the configured mono/stereo output mode value.
    pub fn output_mode(&self) -> u32 {
        self.output_mode
    }

    /// Return guest-visible pacing for one SAS grain at 44.1 kHz.
    ///
    /// SAS mixes synchronously, but the audio hardware drains one grain per
    /// period. Pacing the guest by the real grain duration keeps an audio worker
    /// at real time instead of spinning and starving media or render threads.
    pub fn estimate_mix_us(&self) -> u64 {
        if std::env::var_os("PSP_SAS_PACING").is_some_and(|value| value.eq_ignore_ascii_case("cpu"))
        {
            let voices = self
                .voices
                .iter()
                .filter(|voice| voice.playing && !voice.paused)
                .count();
            return (20 + voices * 68 + self.grain_size * 60 / 100).min(1_200) as u64;
        }
        (self.grain_size as u64 * 1_000_000).div_ceil(44_100).max(1)
    }

    /// Change the grain size when it is a supported 32-sample multiple.
    pub fn set_grain_size(&mut self, grain_size: usize) -> bool {
        if !(0x40..=0x800).contains(&grain_size) || grain_size & 0x1f != 0 {
            return false;
        }
        self.grain_size = grain_size;
        true
    }

    /// Change the output mode. PSP SAS accepts mode values `0` and `1` here.
    pub fn set_output_mode(&mut self, output_mode: u32) -> bool {
        if output_mode > 1 {
            return false;
        }
        self.output_mode = output_mode;
        true
    }

    /// Configure a voice with little-endian signed-16 PCM samples in guest memory.
    ///
    /// `loop_position` is a sample index. A negative value disables looping.
    pub fn set_voice_pcm(
        &mut self,
        voice: usize,
        address: u32,
        sample_count: usize,
        loop_position: i32,
    ) -> bool {
        let Some(voice) = self.voices.get_mut(voice) else {
            return false;
        };
        if sample_count == 0 || sample_count > 0x10000 || loop_position >= sample_count as i32 {
            return false;
        }
        voice.kind = VoiceKind::Pcm;
        voice.pcm_address = address;
        voice.pcm_size = sample_count;
        voice.pcm_loop_position = loop_position.max(0) as usize;
        voice.loop_enabled = loop_position >= 0;
        voice.sample_position = 0;
        voice.playing = true;
        // setvoicepcm arms the voice; samples remain silent until the title
        // sends setkeyon. this separates the configured/playing state from
        // the key-on state, as on the psp.
        voice.on = false;
        voice.paused = false;
        true
    }

    /// Configure a voice with a PSP VAG/ADPCM stream in guest memory.
    pub fn set_voice_vag(
        &mut self,
        voice: usize,
        address: u32,
        byte_size: usize,
        loop_enabled: bool,
    ) -> bool {
        let Some(voice) = self.voices.get_mut(voice) else {
            return false;
        };
        if byte_size == 0 || byte_size & (VAG_BLOCK_BYTES as usize - 1) != 0 {
            return false;
        }
        voice.kind = VoiceKind::Vag;
        voice.vag_address = address;
        voice.vag_size = byte_size;
        voice.loop_enabled = loop_enabled;
        voice.reset_vag();
        voice.playing = voice.on;
        true
    }

    /// Set the left and right volume for one voice.
    pub fn set_volume(&mut self, voice: usize, left: i32, right: i32) -> bool {
        self.set_volume_with_effect(voice, left, right, VOLUME_MAX, VOLUME_MAX)
    }

    /// Set voice and effect-send volumes for one voice.
    pub fn set_volume_with_effect(
        &mut self,
        voice: usize,
        left: i32,
        right: i32,
        effect_left: i32,
        effect_right: i32,
    ) -> bool {
        let Some(voice) = self.voices.get_mut(voice) else {
            return false;
        };
        if left.unsigned_abs() > VOLUME_MAX as u32
            || right.unsigned_abs() > VOLUME_MAX as u32
            || effect_left.unsigned_abs() > VOLUME_MAX as u32
            || effect_right.unsigned_abs() > VOLUME_MAX as u32
        {
            return false;
        }
        voice.left_volume = left;
        voice.right_volume = right;
        voice.effect_left_volume = effect_left;
        voice.effect_right_volume = effect_right;
        true
    }

    /// Set a voice's PSP pitch value, where `0x1000` is normal speed.
    pub fn set_pitch(&mut self, voice: usize, pitch: u32) -> bool {
        let Some(voice) = self.voices.get_mut(voice) else {
            return false;
        };
        if pitch > 0x4000 {
            return false;
        }
        voice.pitch = pitch;
        true
    }

    /// Start or retrigger one voice from its configured beginning.
    pub fn key_on(&mut self, voice: usize) -> bool {
        let Some(voice) = self.voices.get_mut(voice) else {
            return false;
        };
        if voice.paused {
            return false;
        }
        // retrigger the voice on every keyon, even if it is already
        // on: gta re-keys living sfx voices without an intervening keyoff.
        voice.on = true;
        voice.playing = true;
        voice.sample_position = 0;
        voice.vag_pitch_acc = 0;
        voice.vag_pitch_current = 0;
        voice.vag_pitch_next = 0;
        voice.vag_pitch_have_next = false;
        if voice.kind == VoiceKind::Vag {
            voice.reset_vag();
        }
        true
    }

    /// Stop one voice and clear its key-on state.
    pub fn key_off(&mut self, voice: usize) -> bool {
        let Some(voice) = self.voices.get_mut(voice) else {
            return false;
        };
        voice.on = false;
        voice.playing = false;
        true
    }

    /// Pause or resume every voice selected by `voice_mask`.
    pub fn set_pause(&mut self, voice_mask: u32, pause: bool) {
        for (index, voice) in self.voices.iter_mut().enumerate() {
            if voice_mask & (1 << index) != 0 {
                voice.paused = pause;
            }
        }
    }

    /// Return a bit mask of voices currently paused.
    pub fn pause_flags(&self) -> u32 {
        self.voices
            .iter()
            .enumerate()
            .filter(|(_, voice)| voice.paused)
            .fold(0, |flags, (index, _)| flags | (1 << index))
    }

    /// Return a bit mask of voices that have stopped playing.
    pub fn end_flags(&self) -> u32 {
        self.voices
            .iter()
            .enumerate()
            .filter(|(_, voice)| !voice.playing)
            .fold(0, |flags, (index, _)| flags | (1 << index))
    }

    /// Mix one SAS grain, write it to the guest output buffer, and return the
    /// interleaved stereo PCM that should reach the host sink.
    pub fn mix(
        &mut self,
        memory: &mut Memory,
        output_address: u32,
        input_address: Option<u32>,
        left_volume: i32,
        right_volume: i32,
    ) -> Result<Vec<u8>, MemoryFault> {
        let grain = self.grain_size;
        let input = if let Some(address) = input_address {
            Some(memory.read_bytes(address, grain * 4)?)
        } else {
            None
        };
        let mut stereo = Vec::with_capacity(grain);
        let mut effect = Vec::with_capacity(grain);
        for index in 0..grain {
            let mut left = 0i32;
            let mut right = 0i32;
            let mut effect_left = 0i32;
            let mut effect_right = 0i32;
            for voice in &mut self.voices {
                if voice.paused || !voice.playing {
                    continue;
                }
                let sample = voice.next_sample(memory)?;
                left += (sample * voice.left_volume) >> 12;
                right += (sample * voice.right_volume) >> 12;
                effect_left += (sample * voice.effect_left_volume) >> 12;
                effect_right += (sample * voice.effect_right_volume) >> 12;
            }
            if let Some(input) = input.as_ref() {
                let offset = index * 4;
                let input_left = i16::from_le_bytes([input[offset], input[offset + 1]]) as i32;
                let input_right = i16::from_le_bytes([input[offset + 2], input[offset + 3]]) as i32;
                left += (input_left * left_volume) >> 12;
                right += (input_right * right_volume) >> 12;
            }
            stereo.push([clamp_i16(left), clamp_i16(right)]);
            effect.push([clamp_i16(effect_left), clamp_i16(effect_right)]);
        }

        let mut guest_output = Vec::with_capacity(if self.output_mode == 0 {
            grain * 4
        } else {
            grain * 8
        });
        if self.output_mode == 0 {
            for [left, right] in &stereo {
                guest_output.extend_from_slice(&left.to_le_bytes());
                guest_output.extend_from_slice(&right.to_le_bytes());
            }
        } else {
            for channel in 0..4 {
                for (index, [left, right]) in stereo.iter().enumerate() {
                    let sample = match channel {
                        0 => *left,
                        1 => *right,
                        2 => effect[index][0],
                        _ => effect[index][1],
                    };
                    guest_output.extend_from_slice(&sample.to_le_bytes());
                }
            }
        }
        memory.write_bytes(output_address, &guest_output)?;

        let mut host_pcm = Vec::with_capacity(grain * 4);
        for [left, right] in stereo {
            host_pcm.extend_from_slice(&left.to_le_bytes());
            host_pcm.extend_from_slice(&right.to_le_bytes());
        }
        Ok(host_pcm)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VoiceKind {
    Off,
    Pcm,
    Vag,
}

#[derive(Clone, Copy)]
struct SasVoice {
    kind: VoiceKind,
    playing: bool,
    paused: bool,
    on: bool,
    pcm_address: u32,
    pcm_size: usize,
    pcm_loop_position: usize,
    sample_position: u32,
    pitch: u32,
    loop_enabled: bool,
    left_volume: i32,
    right_volume: i32,
    effect_left_volume: i32,
    effect_right_volume: i32,
    vag_address: u32,
    vag_size: usize,
    vag_block: i32,
    vag_loop_start: i32,
    vag_loop_defined: bool,
    vag_loop_pending: bool,
    vag_sample_index: usize,
    vag_samples: [i16; VAG_SAMPLES_PER_BLOCK],
    vag_s1: i32,
    vag_s2: i32,
    vag_end: bool,
    /// Fractional resampling cursor for pitched VAG voices, using the same
    /// pitch scale as PCM. `vag_pitch_acc` accumulates `pitch` per output
    /// sample; whole steps advance the decoded VAG stream.
    vag_pitch_acc: u32,
    vag_pitch_current: i32,
    vag_pitch_next: i32,
    vag_pitch_have_next: bool,
}

impl Default for SasVoice {
    fn default() -> Self {
        Self {
            kind: VoiceKind::Off,
            playing: false,
            paused: false,
            on: false,
            pcm_address: 0,
            pcm_size: 0,
            pcm_loop_position: 0,
            sample_position: 0,
            pitch: PITCH_BASE,
            loop_enabled: false,
            left_volume: VOLUME_MAX,
            right_volume: VOLUME_MAX,
            effect_left_volume: VOLUME_MAX,
            effect_right_volume: VOLUME_MAX,
            vag_address: 0,
            vag_size: 0,
            vag_block: -1,
            vag_loop_start: -1,
            vag_loop_defined: false,
            vag_loop_pending: false,
            vag_sample_index: VAG_SAMPLES_PER_BLOCK,
            vag_samples: [0; VAG_SAMPLES_PER_BLOCK],
            vag_s1: 0,
            vag_s2: 0,
            vag_end: true,
            vag_pitch_acc: 0,
            vag_pitch_current: 0,
            vag_pitch_next: 0,
            vag_pitch_have_next: false,
        }
    }
}

impl SasVoice {
    fn reset_runtime(&mut self) {
        self.kind = VoiceKind::Off;
        self.playing = false;
        self.paused = false;
        self.on = false;
        self.pcm_address = 0;
        self.pcm_size = 0;
        self.pcm_loop_position = 0;
        self.sample_position = 0;
        self.pitch = PITCH_BASE;
        self.loop_enabled = false;
        self.left_volume = VOLUME_MAX;
        self.right_volume = VOLUME_MAX;
        self.effect_left_volume = VOLUME_MAX;
        self.effect_right_volume = VOLUME_MAX;
        self.vag_address = 0;
        self.vag_size = 0;
        self.reset_vag();
    }

    fn reset_vag(&mut self) {
        self.vag_block = -1;
        self.vag_loop_start = -1;
        self.vag_loop_defined = false;
        self.vag_loop_pending = false;
        self.vag_sample_index = VAG_SAMPLES_PER_BLOCK;
        self.vag_samples = [0; VAG_SAMPLES_PER_BLOCK];
        self.vag_s1 = 0;
        self.vag_s2 = 0;
        self.vag_end = self.vag_size == 0;
        self.vag_pitch_acc = 0;
        self.vag_pitch_current = 0;
        self.vag_pitch_next = 0;
        self.vag_pitch_have_next = false;
    }

    fn next_sample(&mut self, memory: &Memory) -> Result<i32, MemoryFault> {
        match self.kind {
            VoiceKind::Pcm => self.next_pcm_sample(memory),
            VoiceKind::Vag => self.next_vag_sample(memory),
            VoiceKind::Off => Ok(0),
        }
    }

    fn next_pcm_sample(&mut self, memory: &Memory) -> Result<i32, MemoryFault> {
        if self.pcm_size == 0 {
            self.playing = false;
            return Ok(0);
        }
        if !self.on {
            self.sample_position = 0;
            return Ok(0);
        }
        let index = (self.sample_position >> 12) as usize;
        if index >= self.pcm_size {
            if self.loop_enabled {
                self.sample_position = (self.pcm_loop_position as u32) << 12;
            } else {
                self.playing = false;
                self.on = false;
                return Ok(0);
            }
        }
        let index = (self.sample_position >> 12) as usize;
        let fraction = self.sample_position & PITCH_MASK;
        let first = read_pcm_sample(memory, self.pcm_address, index)? as i32;
        let next_index = if index + 1 < self.pcm_size {
            index + 1
        } else if self.loop_enabled {
            self.pcm_loop_position
        } else {
            index
        };
        let second = read_pcm_sample(memory, self.pcm_address, next_index)? as i32;
        self.sample_position = self.sample_position.wrapping_add(self.pitch);
        Ok((first * (PITCH_BASE - fraction) as i32 + second * fraction as i32) >> 12)
    }

    fn next_vag_sample(&mut self, memory: &Memory) -> Result<i32, MemoryFault> {
        if !self.on {
            return Ok(0);
        }
        // unpitched fast path preserves the existing sequential decode.
        if self.pitch == PITCH_BASE {
            return self.next_vag_raw_sample(memory);
        }
        // pitched voices resample the decoded vag stream with the same
        // 12-bit fractional stepping as pcm voices.
        if !self.vag_pitch_have_next {
            self.vag_pitch_current = self.next_vag_raw_sample(memory)?;
            self.vag_pitch_next = self.next_vag_raw_sample(memory)?;
            self.vag_pitch_have_next = true;
            if !self.playing {
                return Ok(0);
            }
        }
        let fraction = self.vag_pitch_acc & PITCH_MASK;
        let sample = (self.vag_pitch_current * (PITCH_BASE - fraction) as i32
            + self.vag_pitch_next * fraction as i32)
            >> 12;
        self.vag_pitch_acc = self.vag_pitch_acc.wrapping_add(self.pitch);
        while self.vag_pitch_acc >= PITCH_BASE {
            self.vag_pitch_acc -= PITCH_BASE;
            self.vag_pitch_current = self.vag_pitch_next;
            self.vag_pitch_next = self.next_vag_raw_sample(memory)?;
            if !self.playing {
                self.vag_pitch_have_next = false;
                break;
            }
        }
        Ok(sample)
    }

    fn next_vag_raw_sample(&mut self, memory: &Memory) -> Result<i32, MemoryFault> {
        if self.vag_end || self.vag_size < VAG_BLOCK_BYTES as usize {
            self.playing = false;
            return Ok(0);
        }
        if self.vag_sample_index == VAG_SAMPLES_PER_BLOCK {
            if self.vag_loop_pending && self.vag_loop_defined {
                self.vag_block = self.vag_loop_start;
                self.vag_loop_pending = false;
            }
            if self.vag_block >= (self.vag_size / VAG_BLOCK_BYTES as usize) as i32 - 1 {
                self.vag_end = true;
                self.playing = false;
                self.on = false;
                return Ok(0);
            }
            self.decode_vag_block(memory)?;
            if self.vag_end {
                self.playing = false;
                self.on = false;
                return Ok(0);
            }
        }
        let sample = self.vag_samples[self.vag_sample_index] as i32;
        self.vag_sample_index += 1;
        Ok(sample)
    }

    fn decode_vag_block(&mut self, memory: &Memory) -> Result<(), MemoryFault> {
        let block = (self.vag_block + 1) as u32;
        let address = self
            .vag_address
            .wrapping_add(block.saturating_mul(VAG_BLOCK_BYTES));
        let predictor_shift = memory.read_u8(address)?;
        let flags = memory.read_u8(address + 1)?;
        if flags == 7 {
            self.vag_end = true;
            return Ok(());
        }
        if flags == 6 {
            // the psp decoder records the loop block before advancing its
            // current-block counter. that makes a marker on block zero a
            // valid rewind target instead of skipping the first block.
            self.vag_loop_start = block as i32 - 1;
            self.vag_loop_defined = true;
        }
        if flags == 3 && self.loop_enabled {
            self.vag_loop_pending = true;
        }
        let predictor = usize::from(predictor_shift >> 4).min(15);
        let shift = u32::from(predictor_shift & 0xf);
        let filter = VAG_FILTERS[predictor];
        for index in (0..VAG_SAMPLES_PER_BLOCK).step_by(2) {
            let packed = memory.read_u8(address + 2 + (index / 2) as u32)?;
            let low = (i32::from(packed & 0xf) << 12) as i16 as i32;
            let high = (i32::from(packed & 0xf0) << 8) as i16 as i32;
            let first = (low >> shift) + ((self.vag_s1 * filter[0] - self.vag_s2 * filter[1]) >> 6);
            self.vag_s2 = self.vag_s1;
            self.vag_s1 = clamp_i16(first) as i32;
            self.vag_samples[index] = self.vag_s1 as i16;
            let second =
                (high >> shift) + ((self.vag_s1 * filter[0] - self.vag_s2 * filter[1]) >> 6);
            self.vag_s2 = self.vag_s1;
            self.vag_s1 = clamp_i16(second) as i32;
            self.vag_samples[index + 1] = self.vag_s1 as i16;
        }
        self.vag_block = block as i32;
        self.vag_sample_index = 0;
        Ok(())
    }
}

fn read_pcm_sample(memory: &Memory, address: u32, index: usize) -> Result<i16, MemoryFault> {
    let address = address.wrapping_add((index as u32).wrapping_mul(2));
    Ok(memory.read_u16(address)? as i16)
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_voice_mixes_into_guest_buffer_and_loops() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        let source = [1000i16, -1000, 2000, -2000];
        for (index, sample) in source.into_iter().enumerate() {
            memory
                .write_u16(0x1100 + index as u32 * 2, sample as u16)
                .unwrap();
        }
        let mut mixer = SasMixer::default();
        assert!(mixer.initialize(64, 1, 0, 44_100));
        assert!(mixer.set_voice_pcm(0, 0x1100, source.len(), 0));
        assert!(mixer.key_on(0));
        let pcm = mixer
            .mix(&mut memory, 0x1200, None, 0x1000, 0x1000)
            .unwrap();
        assert_eq!(i16::from_le_bytes([pcm[0], pcm[1]]), 1000);
        assert_eq!(i16::from_le_bytes([pcm[2], pcm[3]]), 1000);
        assert_eq!(i16::from_le_bytes([pcm[16], pcm[17]]), 1000);
        assert_eq!(memory.read_u16(0x1200).unwrap(), 1000);
    }

    #[test]
    fn mixed_output_scales_existing_stereo_input() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        for index in 0..64 {
            memory.write_u32(0x1100 + index * 4, 0x03e8_03e8).unwrap();
        }
        let mut mixer = SasMixer::default();
        assert!(mixer.initialize(64, 1, 0, 44_100));
        mixer
            .mix(&mut memory, 0x1200, Some(0x1100), 0x800, 0x1000)
            .unwrap();
        assert_eq!(memory.read_u16(0x1200).unwrap(), 500);
        assert_eq!(memory.read_u16(0x1202).unwrap(), 1000);
    }

    #[test]
    fn pcm_voice_waits_for_key_on() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        memory.write_u16(0x1100, 1234).unwrap();

        let mut mixer = SasMixer::default();
        assert!(mixer.initialize(64, 1, 0, 44_100));
        assert!(mixer.set_voice_pcm(0, 0x1100, 1, -1));
        let before_key_on = mixer
            .mix(&mut memory, 0x1200, None, 0x1000, 0x1000)
            .unwrap();
        assert_eq!(i16::from_le_bytes([before_key_on[0], before_key_on[1]]), 0);

        assert!(mixer.key_on(0));
        let after_key_on = mixer
            .mix(&mut memory, 0x1200, None, 0x1000, 0x1000)
            .unwrap();
        assert_eq!(i16::from_le_bytes([after_key_on[0], after_key_on[1]]), 1234);
    }

    #[test]
    fn mixer_pacing_matches_grain_playback_time() {
        let mut mixer = SasMixer::default();
        assert_eq!(mixer.estimate_mix_us(), 1);
        assert!(mixer.initialize(0x800, 1, 0, 44_100));
        // one 0x800-sample grain at 44.1 khz drains in ~46.4 ms.
        assert_eq!(mixer.estimate_mix_us(), 46_440);
        assert!(mixer.initialize(256, 1, 0, 44_100));
        assert_eq!(mixer.estimate_mix_us(), 5_805);
    }

    #[test]
    fn vag_loop_marker_rewinds_to_first_block() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        // block 0 is the loop start, block 1 requests a loop, and block 2 is
        // an end marker that should never be reached when looping is enabled.
        memory.write_u8(0x1100, 0x00).unwrap();
        memory.write_u8(0x1101, 0x06).unwrap();
        for offset in 0..14 {
            memory.write_u8(0x1102 + offset, 0x11).unwrap();
        }
        memory.write_u8(0x1110, 0x00).unwrap();
        memory.write_u8(0x1111, 0x03).unwrap();
        for offset in 0..14 {
            memory.write_u8(0x1112 + offset, 0x22).unwrap();
        }
        memory.write_u8(0x1120, 0x00).unwrap();
        memory.write_u8(0x1121, 0x07).unwrap();

        let mut mixer = SasMixer::default();
        assert!(mixer.initialize(64, 1, 0, 44_100));
        assert!(mixer.set_voice_vag(0, 0x1100, 48, true));
        assert!(mixer.key_on(0));
        let first_grain = mixer
            .mix(&mut memory, 0x1200, None, 0x1000, 0x1000)
            .unwrap();
        assert_ne!(i16::from_le_bytes([first_grain[0], first_grain[1]]), 0);
        let second_grain = mixer
            .mix(&mut memory, 0x1200, None, 0x1000, 0x1000)
            .unwrap();
        assert_ne!(i16::from_le_bytes([second_grain[0], second_grain[1]]), 0);
        assert_eq!(mixer.end_flags() & 1, 0);
    }
}
