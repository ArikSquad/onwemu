use anyhow::{Result, bail};
use clap::Parser;
use onwemu::{Emulator, cartridge::Cartridge};
use std::{path::PathBuf, time::Duration};

#[derive(Parser)]
struct Args {
    rom: PathBuf,
    #[arg(long, default_value_t = 30)]
    timeout: u64,
    #[arg(long)]
    frames: Option<u64>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let cart = Cartridge::load(&a.rom)?;
    let mut emu = Emulator::new(cart);
    let limit = a.frames.unwrap_or(a.timeout * 60);
    let started = std::time::Instant::now();
    for _ in 0..limit {
        emu.run_frame();
        let out = emu.bus.serial_output();
        if out.windows(b"Passed".len()).any(|part| part == b"Passed") {
            println!("{}", String::from_utf8_lossy(out));
            return Ok(());
        }
        if out.windows(b"Failed".len()).any(|part| part == b"Failed") {
            bail!("{}", String::from_utf8_lossy(out))
        }
    }
    bail!(
        "no machine-readable result after {limit} frames ({:?})",
        Duration::from_secs_f64(started.elapsed().as_secs_f64())
    )
}
