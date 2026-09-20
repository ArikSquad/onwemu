//! Frontend-facing contract shared by console cores.
//!
//! New systems (including a future PSP core) only need to implement [`Core`];
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
        System::Psp => {
            Err("PSP core is not linked; implement core::Core and register it here".into())
        }
    }
}
