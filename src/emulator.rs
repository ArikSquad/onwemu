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
