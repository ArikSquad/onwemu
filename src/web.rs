//! Small wasm-bindgen boundary. ROM bytes never leave the browser process.

use wasm_bindgen::prelude::*;

use crate::core::{self, Button, Core, Input, PixelFormat, System};

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
    machine: Box<dyn Core>,
    video: VideoFrame,
}

impl System {
    fn parse(name: &str) -> Result<Self, JsError> {
        match name {
            "nes" => Ok(Self::Nes),
            "gb" => Ok(Self::GameBoy),
            "psp" => Ok(Self::Psp),
            _ => Err(JsError::new("Choose NES, Game Boy, or PSP")),
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
        if self.spec.pixel_format == PixelFormat::Rgba8888 {
            debug_assert_eq!(pixels.len(), self.rgba.len());
            self.rgba.copy_from_slice(pixels);
        } else {
            debug_assert_eq!(pixels.len(), self.spec.pixel_count());
            let (rgba, remainder) = self.rgba.as_chunks_mut::<4>();
            debug_assert!(remainder.is_empty());
            for (pixel, rgba) in pixels.iter().zip(rgba) {
                let color = self.spec.palette[(*pixel & self.spec.pixel_mask) as usize];
                rgba[..3].copy_from_slice(&color);
                rgba[3] = 0xff;
            }
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
    pixel_format: PixelFormat,
}

impl VideoSpec {
    fn pixel_count(self) -> usize {
        self.width as usize * self.height as usize
    }

    fn from_core(format: core::VideoFormat) -> Self {
        match format.pixel_format {
            PixelFormat::Indexed2 => Self {
                width: format.width,
                height: format.height,
                palette: &GAME_BOY_PALETTE,
                pixel_mask: 0x03,
                pixel_format: format.pixel_format,
            },
            PixelFormat::Indexed6 => Self {
                width: format.width,
                height: format.height,
                palette: &NES_PALETTE,
                pixel_mask: 0x3f,
                pixel_format: format.pixel_format,
            },
            PixelFormat::Rgba8888 => Self {
                width: format.width,
                height: format.height,
                palette: &[],
                pixel_mask: 0xff,
                pixel_format: format.pixel_format,
            },
        }
    }
}

fn parse_button(name: &str) -> Option<Button> {
    Some(match name {
        "a" => Button::Primary,
        "b" => Button::Secondary,
        "select" => Button::Select,
        "start" => Button::Start,
        "up" => Button::Up,
        "down" => Button::Down,
        "left" => Button::Left,
        "right" => Button::Right,
        _ => return None,
    })
}

#[wasm_bindgen]
impl WebEmulator {
    #[wasm_bindgen(constructor)]
    pub fn new(system: &str, rom: &[u8]) -> Result<WebEmulator, JsError> {
        console_error_panic_hook::set_once();
        let system = System::parse(system)?;
        let machine = core::create(system, rom).map_err(|error| JsError::new(&error))?;
        let video = VideoFrame::new(VideoSpec::from_core(machine.video_format()));
        Ok(Self { machine, video })
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
        if let Some(button) = parse_button(name) {
            self.machine.input(Input::Button(button, down));
        }
    }

    /// Returns battery-backed cartridge data, or an empty vector for systems
    /// without persistent storage.
    pub fn save_data(&self) -> Vec<u8> {
        self.machine.save_data().unwrap_or_default()
    }

    pub fn load_save_data(&mut self, data: &[u8]) -> Result<(), JsError> {
        self.machine
            .load_save_data(data)
            .map_err(|error| JsError::new(&error))
    }
}

const GAME_BOY_PALETTE: [[u8; 3]; 4] =
    [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]];

#[cfg(test)]
mod tests {
    use super::parse_button;
    use crate::core::Button;

    #[test]
    fn maps_nes_buttons_to_serial_controller_order() {
        assert_eq!(parse_button("a"), Some(Button::Primary));
        assert_eq!(parse_button("right"), Some(Button::Right));
        assert_eq!(parse_button("unknown"), None);
    }
}
