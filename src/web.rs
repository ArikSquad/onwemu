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
    rgba: Vec<u8>,
}

enum Machine {
    Nes(Box<NesEmulator>),
    GameBoy(Box<GbEmulator>),
}

#[wasm_bindgen]
impl WebEmulator {
    #[wasm_bindgen(constructor)]
    pub fn new(system: &str, rom: &[u8]) -> Result<WebEmulator, JsError> {
        console_error_panic_hook::set_once();
        let (machine, pixels) = match system {
            "nes" => {
                let nes = NesEmulator::new(rom).map_err(|error| JsError::new(&error))?;
                (Machine::Nes(Box::new(nes)), 256 * 240)
            }
            "gb" => {
                let cart = Cartridge::from_bytes(rom.to_vec())
                    .map_err(|error| JsError::new(&error.to_string()))?;
                (Machine::GameBoy(Box::new(GbEmulator::new(cart))), 160 * 144)
            }
            _ => return Err(JsError::new("Choose NES or Game Boy")),
        };
        Ok(Self {
            machine,
            rgba: vec![0; pixels * 4],
        })
    }

    pub fn width(&self) -> u32 {
        match self.machine {
            Machine::Nes(_) => 256,
            Machine::GameBoy(_) => 160,
        }
    }
    pub fn height(&self) -> u32 {
        match self.machine {
            Machine::Nes(_) => 240,
            Machine::GameBoy(_) => 144,
        }
    }

    pub fn run_frame(&mut self) -> Vec<u8> {
        match &mut self.machine {
            Machine::Nes(nes) => {
                nes.run_frame();
                for (src, dst) in nes.framebuffer().iter().zip(self.rgba.chunks_exact_mut(4)) {
                    let color = NES_PALETTE[(*src & 0x3f) as usize];
                    dst.copy_from_slice(&[color[0], color[1], color[2], 255]);
                }
            }
            Machine::GameBoy(gb) => {
                gb.run_frame();
                let shades = [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]];
                for (src, dst) in gb.framebuffer().iter().zip(self.rgba.chunks_exact_mut(4)) {
                    let c = shades[(*src & 3) as usize];
                    dst.copy_from_slice(&[c[0], c[1], c[2], 255]);
                }
            }
        }
        self.rgba.clone()
    }

    pub fn set_button(&mut self, name: &str, down: bool) {
        match &mut self.machine {
            Machine::Nes(nes) => {
                if let Some(button) = nes_button(name) {
                    nes.set_button(button, down);
                }
            }
            Machine::GameBoy(gb) => {
                if let Some(button) = gb_button(name) {
                    gb.set_button(button, down);
                }
            }
        }
    }
}

fn nes_button(name: &str) -> Option<usize> {
    Some(match name {
        "a" => 0,
        "b" => 1,
        "select" => 2,
        "start" => 3,
        "up" => 4,
        "down" => 5,
        "left" => 6,
        "right" => 7,
        _ => return None,
    })
}

fn gb_button(name: &str) -> Option<Button> {
    Some(match name {
        "a" => Button::A,
        "b" => Button::B,
        "select" => Button::Select,
        "start" => Button::Start,
        "up" => Button::Up,
        "down" => Button::Down,
        "left" => Button::Left,
        "right" => Button::Right,
        _ => return None,
    })
}
