//! Small wasm-bindgen boundary. ROM bytes never leave the browser process.

use wasm_bindgen::prelude::*;

use crate::{Button, Emulator as GbEmulator, cartridge::Cartridge, nes::Emulator as NesEmulator};

const NES_PALETTE: [[u8; 3]; 64] = [
    [84, 84, 84],
    [0, 30, 116],
    [8, 16, 144],
    [48, 0, 136],
    [68, 0, 100],
    [92, 0, 48],
    [84, 4, 0],
    [60, 24, 0],
    [32, 42, 0],
    [8, 58, 0],
    [0, 64, 0],
    [0, 60, 0],
    [0, 50, 60],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [152, 150, 152],
    [8, 76, 196],
    [48, 50, 236],
    [92, 30, 228],
    [136, 20, 176],
    [160, 20, 100],
    [152, 34, 32],
    [120, 60, 0],
    [84, 90, 0],
    [40, 114, 0],
    [8, 124, 0],
    [0, 118, 40],
    [0, 102, 120],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [236, 238, 236],
    [76, 154, 236],
    [120, 124, 236],
    [176, 98, 236],
    [228, 84, 236],
    [236, 88, 180],
    [236, 106, 100],
    [212, 136, 32],
    [160, 170, 0],
    [116, 196, 0],
    [76, 208, 32],
    [56, 204, 108],
    [56, 180, 204],
    [60, 60, 60],
    [0, 0, 0],
    [0, 0, 0],
    [236, 238, 236],
    [168, 204, 236],
    [188, 188, 236],
    [212, 178, 236],
    [236, 174, 236],
    [236, 174, 212],
    [236, 180, 176],
    [228, 196, 144],
    [204, 210, 120],
    [180, 222, 120],
    [168, 226, 144],
    [152, 226, 180],
    [160, 214, 228],
    [160, 162, 160],
    [0, 0, 0],
    [0, 0, 0],
];

#[wasm_bindgen]
pub struct WebEmulator {
    machine: Machine,
    video: VideoFrame,
}

enum Machine {
    Nes(Box<NesEmulator>),
    GameBoy(Box<GbEmulator>),
}

#[derive(Clone, Copy)]
enum System {
    Nes,
    GameBoy,
}

impl System {
    fn parse(name: &str) -> Result<Self, JsError> {
        match name {
            "nes" => Ok(Self::Nes),
            "gb" => Ok(Self::GameBoy),
            _ => Err(JsError::new("Choose NES or Game Boy")),
        }
    }

    fn video_spec(self) -> VideoSpec {
        match self {
            Self::Nes => NES_VIDEO,
            Self::GameBoy => GAME_BOY_VIDEO,
        }
    }
}

impl Machine {
    fn new(system: System, rom: &[u8]) -> Result<Self, JsError> {
        match system {
            System::Nes => NesEmulator::new(rom)
                .map(|emulator| Self::Nes(Box::new(emulator)))
                .map_err(|error| JsError::new(&error)),
            System::GameBoy => Cartridge::from_bytes(rom.to_vec())
                .map(|cartridge| Self::GameBoy(Box::new(GbEmulator::new(cartridge))))
                .map_err(|error| JsError::new(&error.to_string())),
        }
    }

    fn run_frame(&mut self) {
        match self {
            Self::Nes(emulator) => emulator.run_frame(),
            Self::GameBoy(emulator) => emulator.run_frame(),
        }
    }

    fn framebuffer(&self) -> &[u8] {
        match self {
            Self::Nes(emulator) => emulator.framebuffer(),
            Self::GameBoy(emulator) => emulator.framebuffer(),
        }
    }

    fn set_button(&mut self, name: &str, down: bool) {
        let Some(button) = ButtonName::parse(name) else {
            return;
        };

        match self {
            Self::Nes(emulator) => emulator.set_button(button.nes_index(), down),
            Self::GameBoy(emulator) => emulator.set_button(button.game_boy(), down),
        }
    }
}

struct VideoFrame {
    spec: VideoSpec,
    rgba: Vec<u8>,
}

impl VideoFrame {
    fn new(spec: VideoSpec) -> Self {
        Self {
            rgba: vec![0; spec.pixel_count() * 4],
            spec,
        }
    }

    fn width(&self) -> u32 {
        self.spec.width
    }

    fn height(&self) -> u32 {
        self.spec.height
    }

    fn update(&mut self, pixels: &[u8]) {
        debug_assert_eq!(pixels.len(), self.spec.pixel_count());
        for (pixel, rgba) in pixels.iter().zip(self.rgba.chunks_exact_mut(4)) {
            let color = self.spec.palette[(*pixel & self.spec.pixel_mask) as usize];
            rgba[..3].copy_from_slice(&color);
            rgba[3] = 0xff;
        }
    }

    fn rgba(&self) -> Vec<u8> {
        self.rgba.clone()
    }
}

#[derive(Clone, Copy)]
struct VideoSpec {
    width: u32,
    height: u32,
    palette: &'static [[u8; 3]],
    pixel_mask: u8,
}

impl VideoSpec {
    fn pixel_count(self) -> usize {
        self.width as usize * self.height as usize
    }
}

#[derive(Clone, Copy)]
enum ButtonName {
    A,
    B,
    Select,
    Start,
    Up,
    Down,
    Left,
    Right,
}

impl ButtonName {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "a" => Self::A,
            "b" => Self::B,
            "select" => Self::Select,
            "start" => Self::Start,
            "up" => Self::Up,
            "down" => Self::Down,
            "left" => Self::Left,
            "right" => Self::Right,
            _ => return None,
        })
    }

    fn nes_index(self) -> usize {
        match self {
            Self::Right => 0,
            Self::Left => 1,
            Self::Up => 2,
            Self::Down => 3,
            Self::A => 4,
            Self::B => 5,
            Self::Select => 6,
            Self::Start => 7,
        }
    }

    fn game_boy(self) -> Button {
        match self {
            Self::Right => Button::Right,
            Self::Left => Button::Left,
            Self::Up => Button::Up,
            Self::Down => Button::Down,
            Self::A => Button::A,
            Self::B => Button::B,
            Self::Select => Button::Select,
            Self::Start => Button::Start,
        }
    }
}

#[wasm_bindgen]
impl WebEmulator {
    #[wasm_bindgen(constructor)]
    pub fn new(system: &str, rom: &[u8]) -> Result<WebEmulator, JsError> {
        console_error_panic_hook::set_once();
        let system = System::parse(system)?;
        Ok(Self {
            machine: Machine::new(system, rom)?,
            video: VideoFrame::new(system.video_spec()),
        })
    }

    pub fn width(&self) -> u32 {
        self.video.width()
    }

    pub fn height(&self) -> u32 {
        self.video.height()
    }

    pub fn run_frame(&mut self) -> Vec<u8> {
        self.machine.run_frame();
        self.video.update(self.machine.framebuffer());
        self.video.rgba()
    }

    pub fn set_button(&mut self, name: &str, down: bool) {
        self.machine.set_button(name, down);
    }
}

const NES_VIDEO: VideoSpec = VideoSpec {
    width: 256,
    height: 240,
    palette: &NES_PALETTE,
    pixel_mask: 0x3f,
};

const GAME_BOY_VIDEO: VideoSpec = VideoSpec {
    width: 160,
    height: 144,
    palette: &GAME_BOY_PALETTE,
    pixel_mask: 0x03,
};

const GAME_BOY_PALETTE: [[u8; 3]; 4] =
    [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]];
