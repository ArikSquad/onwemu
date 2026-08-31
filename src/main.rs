use anyhow::Result;
use clap::Parser;
use minifb::{Key, Scale, Window, WindowOptions};
use onwemu::{Button, Emulator, cartridge::Cartridge};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    rom: PathBuf,
    #[arg(long, default_value_t = 4)]
    scale: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let mut emu = Emulator::new(Cartridge::load(&a.rom)?);
    let mut win = Window::new(
        "gbsml",
        160,
        144,
        WindowOptions {
            scale: match a.scale {
                1 => Scale::X1,
                2 => Scale::X2,
                4 => Scale::X4,
                _ => Scale::X4,
            },
            ..Default::default()
        },
    )?;
    win.set_target_fps(60);
    let mut pixels = vec![0u32; 160 * 144];
    let keys = [
        (Key::Right, Button::Right),
        (Key::Left, Button::Left),
        (Key::Up, Button::Up),
        (Key::Down, Button::Down),
        (Key::Z, Button::A),
        (Key::X, Button::B),
        (Key::Backspace, Button::Select),
        (Key::Enter, Button::Start),
    ];
    let frame = Duration::from_nanos(16_742_706);
    while win.is_open() && !win.is_key_down(Key::Escape) {
        let t = Instant::now();
        for &(k, b) in &keys {
            emu.set_button(b, win.is_key_down(k))
        }
        emu.run_frame();
        for (o, &p) in pixels.iter_mut().zip(emu.framebuffer()) {
            *o = match p {
                0 => 0xffe0f8d0,
                1 => 0xff88c070,
                2 => 0xff346856,
                _ => 0xff081820,
            }
        }
        win.update_with_buffer(&pixels, 160, 144)?;
        if let Some(left) = frame.checked_sub(t.elapsed()) {
            std::thread::sleep(left)
        }
    }
    Ok(())
}
