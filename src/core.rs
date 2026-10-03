//! Frontend-facing contract shared by console cores.
//!
//! New systems only need to implement [`Core`];
//! frontends do not need to know a core's internal CPU, bus, or cartridge types.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum System {
    Nes,
    GameBoy,
    Psp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Indexed2,
    Indexed6,
    Rgba8888,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoFormat {
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Up,
    Down,
    Left,
    Right,
    Primary,
    Secondary,
    Select,
    Start,
    LeftShoulder,
    RightShoulder,
    Auxiliary1,
    Auxiliary2,
    Home,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Input {
    Button(Button, bool),
    /// Normalized stick coordinates in the inclusive range `-1.0..=1.0`.
    Analog {
        x: f32,
        y: f32,
    },
}

pub trait Core {
    fn system(&self) -> System;
    fn video_format(&self) -> VideoFormat;
    fn run_frame(&mut self);
    fn framebuffer(&self) -> &[u8];
    fn input(&mut self, input: Input);

    fn failure(&self) -> Option<&str> {
        None
    }

    fn save_data(&self) -> Option<Vec<u8>> {
        None
    }

    fn load_save_data(&mut self, _data: &[u8]) -> Result<(), String> {
        Err("this core has no persistent data".into())
    }
}

pub fn create(system: System, rom: &[u8]) -> Result<Box<dyn Core>, String> {
    match system {
        System::Nes => crate::nes::Emulator::new(rom).map(|core| Box::new(core) as Box<dyn Core>),
        System::GameBoy => crate::cartridge::Cartridge::from_bytes(rom.to_vec())
            .map(crate::Emulator::new)
            .map(|core| Box::new(core) as Box<dyn Core>)
            .map_err(|error| error.to_string()),
        #[cfg(feature = "web")]
        System::Psp => psp_runtime::WebCore::new(rom)
            .map(|core| Box::new(PspAdapter(core)) as Box<dyn Core>)
            .map_err(|error| error.to_string()),
        #[cfg(not(feature = "web"))]
        System::Psp => Err("PSP support is available with the web feature".into()),
    }
}

#[cfg(feature = "web")]
struct PspAdapter(psp_runtime::WebCore);

#[cfg(feature = "web")]
impl Core for PspAdapter {
    fn system(&self) -> System {
        System::Psp
    }

    fn video_format(&self) -> VideoFormat {
        VideoFormat {
            width: self.0.width(),
            height: self.0.height(),
            pixel_format: PixelFormat::Rgba8888,
        }
    }

    fn run_frame(&mut self) {
        self.0.run_frame();
    }

    fn framebuffer(&self) -> &[u8] {
        self.0.framebuffer()
    }

    fn input(&mut self, input: Input) {
        match input {
            Input::Button(button, down) => {
                let button = match button {
                    Button::Up => psp_runtime::ControllerButton::Up,
                    Button::Down => psp_runtime::ControllerButton::Down,
                    Button::Left => psp_runtime::ControllerButton::Left,
                    Button::Right => psp_runtime::ControllerButton::Right,
                    Button::Primary => psp_runtime::ControllerButton::Cross,
                    Button::Secondary => psp_runtime::ControllerButton::Circle,
                    Button::Select => psp_runtime::ControllerButton::Select,
                    Button::Start => psp_runtime::ControllerButton::Start,
                    Button::LeftShoulder => psp_runtime::ControllerButton::LeftShoulder,
                    Button::RightShoulder => psp_runtime::ControllerButton::RightShoulder,
                    Button::Auxiliary1 => psp_runtime::ControllerButton::Square,
                    Button::Auxiliary2 => psp_runtime::ControllerButton::Triangle,
                    Button::Home => psp_runtime::ControllerButton::Home,
                };
                self.0.set_button(button, down);
            }
            Input::Analog { x, y } => self.0.set_analog(x, y),
        }
    }

    fn failure(&self) -> Option<&str> {
        self.0.error()
    }
}
