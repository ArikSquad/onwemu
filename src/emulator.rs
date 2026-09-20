use crate::{
    bus::Bus,
    cartridge::Cartridge,
    cpu::Cpu,
    ppu::{HEIGHT, WIDTH},
};

pub const FRAME_CYCLES: u32 = 70_224;

#[derive(Clone, Copy, Debug)]
pub enum Button {
    Right = 0,
    Left = 1,
    Up = 2,
    Down = 3,
    A = 4,
    B = 5,
    Select = 6,
    Start = 7,
}

pub struct Emulator {
    pub cpu: Cpu,
    pub bus: Bus,
    cycles: u64,
}
impl Emulator {
    pub fn new(cart: Cartridge) -> Self {
        Self {
            cpu: Cpu::default(),
            bus: Bus::new(cart),
            cycles: 0,
        }
    }
    pub fn step(&mut self) -> u8 {
        let c = self.cpu.step(&mut self.bus);
        self.cycles += c as u64;
        c
    }
    pub fn run_frame(&mut self) {
        loop {
            self.step();
            if self.bus.ppu.take_frame_ready() {
                break;
            }
        }
    }
    pub fn framebuffer(&self) -> &[u8; WIDTH * HEIGHT] {
        self.bus.ppu.framebuffer()
    }
    pub fn set_button(&mut self, button: Button, down: bool) {
        if self.bus.joypad.set(button as u8, down) {
            self.bus.interrupt_flags |= 0x10
        }
    }
    pub fn cycles(&self) -> u64 {
        self.cycles
    }
}

impl crate::core::Core for Emulator {
    fn system(&self) -> crate::core::System {
        crate::core::System::GameBoy
    }

    fn video_format(&self) -> crate::core::VideoFormat {
        crate::core::VideoFormat {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pixel_format: crate::core::PixelFormat::Indexed2,
        }
    }

    fn run_frame(&mut self) {
        Emulator::run_frame(self)
    }

    fn framebuffer(&self) -> &[u8] {
        Emulator::framebuffer(self)
    }

    fn input(&mut self, input: crate::core::Input) {
        use crate::core::{Button as FrontendButton, Input};
        let Input::Button(button, down) = input else {
            return;
        };
        let button = match button {
            FrontendButton::Right => Button::Right,
            FrontendButton::Left => Button::Left,
            FrontendButton::Up => Button::Up,
            FrontendButton::Down => Button::Down,
            FrontendButton::Primary => Button::A,
            FrontendButton::Secondary => Button::B,
            FrontendButton::Select => Button::Select,
            FrontendButton::Start => Button::Start,
            _ => return,
        };
        self.set_button(button, down);
    }

    fn save_data(&self) -> Option<Vec<u8>> {
        self.bus
            .cart
            .has_battery()
            .then(|| self.bus.cart.save_data())
    }

    fn load_save_data(&mut self, data: &[u8]) -> Result<(), String> {
        self.bus
            .cart
            .load_save_data(data)
            .map_err(|error| error.to_string())
    }
}
