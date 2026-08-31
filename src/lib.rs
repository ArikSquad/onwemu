#![forbid(unsafe_code)]

pub mod bus;
pub mod cartridge;
pub mod cpu;
pub mod emulator;
pub mod joypad;
pub mod nes;
pub mod ppu;
pub mod timer;

#[cfg(feature = "web")]
mod web;

pub use emulator::{Button, Emulator, FRAME_CYCLES};
