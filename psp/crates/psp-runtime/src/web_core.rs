use super::*;

const FRAME_INSTRUCTION_LIMIT: u64 = 8_000_000;
const MAX_JIT_BLOCK: u64 = 8_192;

/// Browser-facing controller buttons for the PSP adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControllerButton {
    Up,
    Down,
    Left,
    Right,
    Cross,
    Circle,
    Square,
    Triangle,
    Select,
    Start,
    LeftShoulder,
    RightShoulder,
    Home,
}

/// PSP execution state usable by non-SDL frontends.
///
/// The image and mounted disc are retained in memory. Each call to
/// [`run_frame`](Self::run_frame) advances one virtual PSP display interval,
/// subject to a guest-instruction limit so malformed software cannot lock the
/// browser event loop indefinitely.
pub struct WebCore {
    emulator: psp_core::Emulator,
    hle: HleState,
    imports: HashMap<u32, psp_loader::Import>,
    framebuffer: Vec<u8>,
    buttons: u32,
    instructions: u64,
    fast_forward_probe: u64,
    exited: bool,
    failure: Option<String>,
}

impl WebCore {
    /// Load an ELF, PBP, ISO, or CHD supplied as in-memory bytes.
    pub fn new(image: &[u8]) -> Result<Self> {
        let is_disc = image.starts_with(b"MComprHD") || image.get(0x8001..0x8006) == Some(b"CD001");
        let (executable, disc) = if is_disc {
            let mut disc = psp_loader::IsoImage::from_bytes(image.to_vec())?;
            let executable = psp_loader::load_executable_from_disc(&mut disc)?;
            (executable, Some(disc))
        } else {
            (psp_loader::load_executable_from_bytes(image)?, None)
        };
        let image_info = psp_loader::inspect(&executable)?;
        let partition_base = mapped_image_end(&image_info)?.next_multiple_of(256);
        let mut emulator = psp_core::Emulator::new(0);

        emulator
            .memory
            .map(VOLATILE_BASE, VOLATILE_SIZE, true, true)?;
        emulator.memory.map(EDRAM_BASE, EDRAM_SIZE, true, false)?;
        let entry = psp_loader::map_elf(&executable, &mut emulator.memory)?;
        let linked = psp_loader::link_prx_imports(&executable, &mut emulator.memory)?;
        emulator.cpu.gpr[28] = linked.gp;
        emulator.memory.map(
            partition_base,
            (USER_RAM_END - partition_base) as usize,
            true,
            true,
        )?;
        emulator.gpu.set_hardware_rendering(false);

        const THREAD_CONTEXT_BASE: u32 = 0x09f0_0000;
        emulator.cpu.gpr[26] = THREAD_CONTEXT_BASE;
        emulator.cpu.gpr[29] = USER_RAM_END;
        emulator.cpu.gpr[4] = 0;
        emulator.cpu.gpr[5] = 0;
        emulator.cpu.pc = entry;

        let mut hle = HleState::new(disc, partition_base, None)?;
        hle.attach_main_thread(&emulator.cpu);
        Ok(Self {
            emulator,
            hle,
            imports: linked
                .imports
                .into_iter()
                .map(|import| (import.syscall, import))
                .collect(),
            framebuffer: vec![0; DISPLAY_WIDTH as usize * DISPLAY_HEIGHT as usize * 4],
            buttons: 0,
            instructions: 0,
            fast_forward_probe: 0,
            exited: false,
            failure: None,
        })
    }

    pub fn width(&self) -> u32 {
        DISPLAY_WIDTH
    }

    pub fn height(&self) -> u32 {
        DISPLAY_HEIGHT
    }

    pub fn framebuffer(&self) -> &[u8] {
        &self.framebuffer
    }

    pub fn error(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    pub fn set_button(&mut self, button: ControllerButton, down: bool) {
        let mask = match button {
            ControllerButton::Up => PSP_CTRL_UP,
            ControllerButton::Down => PSP_CTRL_DOWN,
            ControllerButton::Left => PSP_CTRL_LEFT,
            ControllerButton::Right => PSP_CTRL_RIGHT,
            ControllerButton::Cross => PSP_CTRL_CROSS,
            ControllerButton::Circle => PSP_CTRL_CIRCLE,
            ControllerButton::Square => PSP_CTRL_SQUARE,
            ControllerButton::Triangle => PSP_CTRL_TRIANGLE,
            ControllerButton::Select => PSP_CTRL_SELECT,
            ControllerButton::Start => PSP_CTRL_START,
            ControllerButton::LeftShoulder => PSP_CTRL_LTRIGGER,
            ControllerButton::RightShoulder => PSP_CTRL_RTRIGGER,
            ControllerButton::Home => PSP_CTRL_HOME,
        };
        if down {
            self.buttons |= mask;
        } else {
            self.buttons &= !mask;
        }
        self.emulator.input.update_buttons(self.buttons);
    }

    pub fn set_analog(&mut self, x: f32, y: f32) {
        self.emulator.input.analog_x = analog_byte(x);
        self.emulator.input.analog_y = analog_byte(y);
        self.emulator.input.set_analog_enabled(true);
    }

    /// Advance the guest by one PSP vblank interval. Errors are retained for
    /// the wasm boundary to report without unwinding through JavaScript.
    pub fn run_frame(&mut self) {
        if self.failure.is_some() || self.exited {
            return;
        }
        if let Err(error) = self.advance_frame() {
            self.failure = Some(error.to_string());
            return;
        }
        self.update_framebuffer();
    }

    fn advance_frame(&mut self) -> Result<()> {
        let frame_start = self.emulator.scheduler.now();
        let mut frame_instructions = 0;
        while self.emulator.scheduler.now().saturating_sub(frame_start) < VBLANK_PERIOD_US
            && frame_instructions < FRAME_INSTRUCTION_LIMIT
            && !self.exited
        {
            self.hle.wake_timed_waiters(&mut self.emulator);
            self.hle.sample_controller_if_due(&mut self.emulator)?;

            if self.emulator.cpu.pc == 0 {
                if self.hle.return_from_interrupt(&mut self.emulator)? {
                    self.instructions = self.instructions.saturating_add(1);
                    frame_instructions += 1;
                    continue;
                }
                if !self.hle.finish_current_and_switch(&mut self.emulator) {
                    self.exited = true;
                    break;
                }
            }

            self.fast_forward_probe = self.fast_forward_probe.wrapping_add(1);
            if self.fast_forward_probe.is_multiple_of(64)
                && let Some(executed) = self
                    .emulator
                    .cpu
                    .try_fast_forward_memory_loop(&mut self.emulator.memory, self.hle.current_uid)?
            {
                self.emulator.scheduler.advance(executed);
                self.instructions = self.instructions.saturating_add(executed);
                frame_instructions = frame_instructions.saturating_add(executed);
                continue;
            }

            let micros_left = VBLANK_PERIOD_US
                .saturating_sub(self.emulator.scheduler.now().saturating_sub(frame_start));
            let cycle_budget = micros_left
                .saturating_mul(psp_kernel::Scheduler::CYCLES_PER_MICROSECOND)
                .max(1);
            let mut budget = cycle_budget.min(MAX_JIT_BLOCK);
            if let Some(next_wake) = self.hle.timed_waiters.values().copied().min() {
                let cycles_until_wake = next_wake
                    .saturating_sub(self.emulator.scheduler.now())
                    .saturating_mul(psp_kernel::Scheduler::CYCLES_PER_MICROSECOND)
                    .max(1);
                budget = budget.min(cycles_until_wake);
            }
            budget = budget.min(
                FRAME_INSTRUCTION_LIMIT
                    .saturating_sub(frame_instructions)
                    .max(1),
            );

            match self.hle.jit.step_guest(
                &mut self.emulator.cpu,
                &mut self.emulator.memory,
                self.hle.current_uid,
                budget,
            ) {
                Ok(psp_cpu::BlockOutcome::Flowing(executed)) => {
                    self.emulator.scheduler.advance(executed);
                    self.instructions = self.instructions.saturating_add(executed);
                    frame_instructions = frame_instructions.saturating_add(executed);
                }
                Ok(psp_cpu::BlockOutcome::Syscall { code, executed, .. }) => {
                    self.emulator.scheduler.advance(executed.saturating_sub(1));
                    self.instructions = self.instructions.saturating_add(executed);
                    frame_instructions = frame_instructions.saturating_add(executed);
                    let import = self
                        .imports
                        .get(&code)
                        .with_context(|| format!("unknown guest syscall {code}"))?;
                    self.hle
                        .dispatch(import, &mut self.emulator)
                        .with_context(|| {
                            format!(
                                "HLE {}::{:08x} failed at thread {}",
                                import.library, import.nid, self.hle.current_uid
                            )
                        })?;
                }
                Err(fault) => {
                    self.emulator.scheduler.advance(fault.executed);
                    return Err(fault.error.into());
                }
            }
        }
        Ok(())
    }

    fn update_framebuffer(&mut self) {
        self.framebuffer.fill(0);
        let display = &self.hle.display;
        if display.frame_buffer == 0 || display.width == 0 || display.height == 0 {
            return;
        }
        let Ok(pixels) = psp_gpu::Gpu::framebuffer_rgba(
            &self.emulator.memory,
            display.frame_buffer,
            display.buffer_width,
            display.width,
            display.height,
            display.pixel_format,
        ) else {
            return;
        };
        let width = display.width.min(DISPLAY_WIDTH) as usize;
        let height = display.height.min(DISPLAY_HEIGHT) as usize;
        let source_width = display.width as usize;
        let target_width = DISPLAY_WIDTH as usize;
        for y in 0..height {
            let source = y * source_width * 4;
            let target = y * target_width * 4;
            let bytes = width * 4;
            self.framebuffer[target..target + bytes]
                .copy_from_slice(&pixels[source..source + bytes]);
        }
    }
}

fn analog_byte(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) * 127.5) + f32::from(ANALOG_CENTER)).round() as u8
}
