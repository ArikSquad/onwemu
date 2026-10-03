//! PSP controller state, edge latches, and the 64-record sampling ring.
//!
//! Host polling updates the current button and analog values. Guest reads can
//! then consume either the current state, a short-lived press latch, or a
//! sampled record without racing the host event loop.

/// The neutral value used for both analog axes.
pub const ANALOG_CENTER: u8 = 128;

/// Button bits used by `sceCtrl` for the standard PSP controls.
pub const PSP_CTRL_SELECT: u32 = 0x0000_0001;
/// The Start button.
pub const PSP_CTRL_START: u32 = 0x0000_0008;
/// The D-pad up button.
pub const PSP_CTRL_UP: u32 = 0x0000_0010;
/// The D-pad right button.
pub const PSP_CTRL_RIGHT: u32 = 0x0000_0020;
/// The D-pad down button.
pub const PSP_CTRL_DOWN: u32 = 0x0000_0040;
/// The D-pad left button.
pub const PSP_CTRL_LEFT: u32 = 0x0000_0080;
/// The left shoulder trigger.
pub const PSP_CTRL_LTRIGGER: u32 = 0x0000_0100;
/// The right shoulder trigger.
pub const PSP_CTRL_RTRIGGER: u32 = 0x0000_0200;
/// The Triangle face button.
pub const PSP_CTRL_TRIANGLE: u32 = 0x0000_1000;
/// The Circle face button.
pub const PSP_CTRL_CIRCLE: u32 = 0x0000_2000;
/// The Cross face button.
pub const PSP_CTRL_CROSS: u32 = 0x0000_4000;
/// The Square face button.
pub const PSP_CTRL_SQUARE: u32 = 0x0000_8000;
/// The Home button.
pub const PSP_CTRL_HOME: u32 = 0x0001_0000;

/// The user-visible controller bits accepted by the PSP firmware.
///
/// Mask host input with this value before putting it in a sampled controller
/// record. Keeping the mask here prevents host-only bits from leaking into
/// guest input and gives negative reads the same complement semantics as the
/// hardware.
pub const PSP_CTRL_USER_MASK: u32 = 0x00ff_ffff;
/// Number of records in the controller sampling ring.
pub const CTRL_SAMPLE_BUFFER_COUNT: usize = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// One controller record in the PSP sampling ring.
pub struct ControllerSample {
    /// Scheduler frame or microsecond timestamp associated with the sample.
    pub frame: u32,
    /// PSP button bits captured in this sample.
    pub buttons: u32,
    /// Horizontal analog position, where [`ANALOG_CENTER`] is neutral.
    pub analog_x: u8,
    /// Vertical analog position, where [`ANALOG_CENTER`] is neutral.
    pub analog_y: u8,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Edge and level information returned by the controller latch API.
pub struct ControllerLatch {
    /// Buttons that changed from released to pressed.
    pub button_make: u32,
    /// Buttons that changed from pressed to released.
    pub button_break: u32,
    /// Buttons observed as pressed since the latch was cleared.
    pub button_press: u32,
    /// Buttons observed as released since the latch was cleared.
    pub button_release: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Current controller state plus guest-visible latches and samples.
pub struct InputState {
    /// Current host button state after applying [`PSP_CTRL_USER_MASK`].
    pub buttons: u32,
    latched_buttons: u32,
    button_make: u32,
    button_break: u32,
    button_press: u32,
    button_release: u32,
    /// Current horizontal analog position.
    pub analog_x: u8,
    /// Current vertical analog position.
    pub analog_y: u8,
    /// Host timestamp associated with the current state, in microseconds.
    pub timestamp_us: u64,
    analog_enabled: bool,
    samples: [ControllerSample; CTRL_SAMPLE_BUFFER_COUNT],
    sample_write: usize,
    sample_read: usize,
    sample_count: usize,
}

impl Default for InputState {
    fn default() -> Self {
        let initial_sample = ControllerSample {
            analog_x: ANALOG_CENTER,
            analog_y: ANALOG_CENTER,
            ..ControllerSample::default()
        };
        let mut samples = [ControllerSample::default(); CTRL_SAMPLE_BUFFER_COUNT];
        samples.fill(initial_sample);
        Self {
            buttons: 0,
            latched_buttons: 0,
            button_make: 0,
            button_break: 0,
            button_press: 0,
            button_release: u32::MAX,
            analog_x: ANALOG_CENTER,
            analog_y: ANALOG_CENTER,
            timestamp_us: 0,
            analog_enabled: false,
            samples,
            // match the psp startup state: one released sample is
            // available before the first vblank sample is produced.
            sample_write: 1,
            sample_read: 0,
            sample_count: 1,
        }
    }
}

impl InputState {
    /// Update the continuously sampled host state while retaining button edges
    /// until the guest consumes them. Host polling is intentionally decoupled
    /// from guest instruction execution, so a fast press cannot disappear
    /// between two guest samples.
    pub fn update_buttons(&mut self, buttons: u32) {
        let buttons = buttons & PSP_CTRL_USER_MASK;
        let changed = buttons ^ self.buttons;
        self.latched_buttons |= buttons & !self.buttons;
        self.button_make |= buttons & changed;
        self.button_break |= self.buttons & changed;
        self.button_press |= buttons;
        self.button_release |= !buttons;
        self.buttons = buttons;
    }

    /// Return the button word exposed by `sceCtrl`.
    ///
    /// Positive reads include and consume the edge latch. Negative reads
    /// describe only the continuously held state and do not consume a pending
    /// press.
    pub fn buttons_for_read(&mut self, negative: bool) -> u32 {
        if negative {
            !(self.buttons & PSP_CTRL_USER_MASK)
        } else {
            let buttons = (self.buttons | self.latched_buttons) & PSP_CTRL_USER_MASK;
            self.latched_buttons = 0;
            buttons
        }
    }

    /// Choose whether new samples use the current analog axes or center them.
    pub fn set_analog_enabled(&mut self, enabled: bool) {
        self.analog_enabled = enabled;
    }

    /// Capture the current host state in the PSP's sampling ring.
    ///
    /// `frame` is the PSP scheduler time in microseconds, matching the frame
    /// field written by the controller sampler.
    pub fn sample(&mut self, frame: u64) {
        let sample = ControllerSample {
            frame: frame as u32,
            buttons: self.buttons & PSP_CTRL_USER_MASK,
            analog_x: if self.analog_enabled {
                self.analog_x
            } else {
                ANALOG_CENTER
            },
            analog_y: if self.analog_enabled {
                self.analog_y
            } else {
                ANALOG_CENTER
            },
        };
        self.samples[self.sample_write] = sample;
        self.sample_write = (self.sample_write + 1) % CTRL_SAMPLE_BUFFER_COUNT;
        if self.sample_count == CTRL_SAMPLE_BUFFER_COUNT {
            self.sample_read = (self.sample_read + 1) % CTRL_SAMPLE_BUFFER_COUNT;
        } else {
            self.sample_count += 1;
        }
    }

    /// Return the number of queued records available to a consuming read.
    pub fn available_samples(&self) -> usize {
        self.sample_count
    }

    /// Read or peek sampled controller records.
    ///
    /// Positive reads consume the oldest records; peeks return the newest
    /// records without advancing the read head. The PSP preinitializes its
    /// ring, so a peek may request more records than have been sampled since
    /// boot; those records are filled from the initial released sample just
    /// like the firmware buffer.
    pub fn read_samples(
        &mut self,
        count: usize,
        negative: bool,
        peek: bool,
    ) -> Vec<ControllerSample> {
        let count = count.min(CTRL_SAMPLE_BUFFER_COUNT);
        if count == 0 {
            return Vec::new();
        }

        let mut samples = Vec::with_capacity(count);
        if peek {
            let available = self.sample_count.min(count);
            let start = (self.sample_write + CTRL_SAMPLE_BUFFER_COUNT - available)
                % CTRL_SAMPLE_BUFFER_COUNT;
            for offset in 0..available {
                samples.push(self.samples[(start + offset) % CTRL_SAMPLE_BUFFER_COUNT]);
            }
            let fill = self.samples[self.sample_read];
            while samples.len() < count {
                samples.push(fill);
            }
        } else {
            let available = self.sample_count.min(count);
            for _ in 0..available {
                samples.push(self.samples[self.sample_read]);
                self.sample_read = (self.sample_read + 1) % CTRL_SAMPLE_BUFFER_COUNT;
                self.sample_count -= 1;
            }
        }

        if negative {
            for sample in &mut samples {
                sample.buttons = !(sample.buttons & PSP_CTRL_USER_MASK);
            }
        } else {
            for sample in &mut samples {
                sample.buttons &= PSP_CTRL_USER_MASK;
            }
        }
        samples
    }

    /// Inspect edge latches without clearing them.
    pub fn peek_latch(&self) -> ControllerLatch {
        let mut latch = ControllerLatch {
            button_make: self.button_make & PSP_CTRL_USER_MASK,
            button_break: self.button_break & PSP_CTRL_USER_MASK,
            button_press: self.button_press & PSP_CTRL_USER_MASK,
            button_release: self.button_release,
        };
        if self.sample_count > 0 {
            latch.button_release |= !PSP_CTRL_USER_MASK;
        }
        latch
    }

    /// Read and clear the current edge latches.
    pub fn read_latch(&mut self) -> ControllerLatch {
        let latch = self.peek_latch();
        self.button_make = 0;
        self.button_break = 0;
        self.button_press = 0;
        self.button_release = 0;
        latch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pad_is_centered_and_released() {
        let input = InputState::default();
        assert_eq!(input.buttons, 0);
        assert_eq!(input.analog_x, ANALOG_CENTER);
        assert_eq!(input.analog_y, ANALOG_CENTER);
        assert_eq!(input.available_samples(), 1);
    }

    #[test]
    fn short_press_survives_until_positive_read() {
        let mut input = InputState::default();
        input.update_buttons(PSP_CTRL_CROSS);
        input.update_buttons(0);

        assert_eq!(input.buttons, 0);
        assert_eq!(input.buttons_for_read(false), PSP_CTRL_CROSS);
        assert_eq!(input.buttons_for_read(false), 0);
    }

    #[test]
    fn negative_read_uses_only_held_buttons() {
        let mut input = InputState::default();
        input.update_buttons(PSP_CTRL_CROSS);
        input.update_buttons(0);

        assert_eq!(input.buttons_for_read(true), u32::MAX);
        assert_eq!(input.buttons_for_read(false), PSP_CTRL_CROSS);
    }

    #[test]
    fn sampled_reads_consume_oldest_record() {
        let mut input = InputState::default();
        input.set_analog_enabled(true);
        input.update_buttons(PSP_CTRL_CROSS);
        input.analog_x = 200;
        input.analog_y = 40;
        input.sample(16_667);

        let initial = input.read_samples(1, false, false);
        assert_eq!(initial[0].buttons, 0);
        assert_eq!(initial[0].analog_x, ANALOG_CENTER);

        let current = input.read_samples(1, false, false);
        assert_eq!(current[0].frame, 16_667);
        assert_eq!(current[0].buttons, PSP_CTRL_CROSS);
        assert_eq!(current[0].analog_x, 200);
        assert_eq!(current[0].analog_y, 40);
        assert!(input.read_samples(1, false, false).is_empty());
    }

    #[test]
    fn peek_returns_recent_records_without_consuming() {
        let mut input = InputState::default();
        input.update_buttons(PSP_CTRL_CROSS);
        input.sample(1);
        input.update_buttons(PSP_CTRL_CIRCLE);
        input.sample(2);

        let peeked = input.read_samples(2, false, true);
        assert_eq!(peeked[0].buttons, PSP_CTRL_CROSS);
        assert_eq!(peeked[1].buttons, PSP_CTRL_CIRCLE);
        assert_eq!(input.available_samples(), 3);
    }

    #[test]
    fn sampled_ring_discards_oldest_record_at_capacity() {
        let mut input = InputState::default();
        for frame in 1..=CTRL_SAMPLE_BUFFER_COUNT {
            input.update_buttons(if frame == CTRL_SAMPLE_BUFFER_COUNT {
                PSP_CTRL_CROSS
            } else {
                0
            });
            input.sample(frame as u64);
        }

        let samples = input.read_samples(CTRL_SAMPLE_BUFFER_COUNT, false, false);
        assert_eq!(samples.len(), CTRL_SAMPLE_BUFFER_COUNT);
        assert_eq!(samples[0].frame, 1);
        assert_eq!(
            samples[CTRL_SAMPLE_BUFFER_COUNT - 1].buttons,
            PSP_CTRL_CROSS
        );
    }

    #[test]
    fn latch_tracks_button_edges_and_levels() {
        let mut input = InputState::default();
        input.update_buttons(PSP_CTRL_CROSS);
        input.update_buttons(0);

        let latch = input.read_latch();
        assert_eq!(latch.button_make, PSP_CTRL_CROSS);
        assert_eq!(latch.button_break, PSP_CTRL_CROSS);
        assert_eq!(latch.button_press, PSP_CTRL_CROSS);
        assert_eq!(latch.button_release & PSP_CTRL_CROSS, PSP_CTRL_CROSS);
        assert_eq!(input.read_latch().button_make, 0);
    }

    #[test]
    fn analog_axes_are_centered_until_sampling_is_enabled() {
        let mut input = InputState {
            analog_x: 1,
            analog_y: 254,
            ..InputState::default()
        };
        input.sample(1);
        assert_eq!(
            input.read_samples(1, false, false)[0].analog_x,
            ANALOG_CENTER
        );

        input.set_analog_enabled(true);
        input.sample(2);
        let sample = input.read_samples(1, false, false)[0];
        assert_eq!(sample.frame, 1);
        let sample = input.read_samples(1, false, false)[0];
        assert_eq!((sample.analog_x, sample.analog_y), (1, 254));
    }
}
