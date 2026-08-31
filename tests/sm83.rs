use onwemu::{Emulator, cartridge::Cartridge};

fn rom(program: &[u8]) -> Cartridge {
    let mut data = vec![0; 0x8000];
    data[0x147] = 0;
    data[0x148] = 0;
    data[0x149] = 0;
    data[0x100..0x100 + program.len()].copy_from_slice(program);
    Cartridge::from_bytes(data).unwrap()
}

#[test]
fn arithmetic_flags() {
    let mut e = Emulator::new(rom(&[0x3e, 0x0f, 0xc6, 0x01]));
    e.step();
    e.step();
    assert_eq!(e.cpu.a, 0x10);
    assert_eq!(e.cpu.f, 0x20)
}
#[test]
fn call_and_return() {
    let mut e = Emulator::new(rom(&[0xcd, 0x06, 0x01, 0x00, 0x00, 0x00, 0x3e, 0x42, 0xc9]));
    e.step();
    e.step();
    e.step();
    assert_eq!(e.cpu.a, 0x42);
    assert_eq!(e.cpu.pc, 0x103)
}
#[test]
fn cb_operations() {
    let mut e = Emulator::new(rom(&[0x06, 0x80, 0xcb, 0x00, 0xcb, 0x78]));
    e.step();
    e.step();
    assert_eq!(e.cpu.b, 1);
    assert_ne!(e.cpu.f & 0x10, 0);
    e.step();
    assert_ne!(e.cpu.f & 0x80, 0)
}
