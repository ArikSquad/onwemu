use crate::{cartridge::Cartridge, joypad::Joypad, ppu::Ppu, timer::Timer};

pub struct Bus {
    pub cart: Cartridge,
    pub ppu: Ppu,
    pub timer: Timer,
    pub joypad: Joypad,
    wram: [u8; 0x2000],
    hram: [u8; 0x7f],
    io: [u8; 0x80],
    pub interrupt_enable: u8,
    pub interrupt_flags: u8,
    serial: Vec<u8>,
    dma: Option<(u16, u16)>,
}

impl Bus {
    pub fn new(cart: Cartridge) -> Self {
        Self {
            cart,
            ppu: Ppu::default(),
            timer: Timer::default(),
            joypad: Joypad::default(),
            wram: [0; 0x2000],
            hram: [0; 0x7f],
            io: [0xff; 0x80],
            interrupt_enable: 0,
            interrupt_flags: 0xe1,
            serial: vec![],
            dma: None,
        }
    }
    pub fn serial_output(&self) -> &[u8] {
        &self.serial
    }
    #[allow(clippy::match_overlapping_arm)]
    pub fn read8(&self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x7fff | 0xa000..=0xbfff => self.cart.read(addr),
            0x8000..=0x9fff => self.ppu.vram[(addr - 0x8000) as usize],
            0xc000..=0xdfff => self.wram[(addr - 0xc000) as usize],
            0xe000..=0xfdff => self.wram[(addr - 0xe000) as usize],
            0xfe00..=0xfe9f => self.ppu.oam[(addr - 0xfe00) as usize],
            0xfea0..=0xfeff => 0xff,
            0xff00 => self.joypad.read(),
            0xff01 => self.io[1],
            0xff02 => self.io[2],
            0xff04 => self.timer.read_div(),
            0xff05 => self.timer.tima,
            0xff06 => self.timer.tma,
            0xff07 => self.timer.tac | 0xf8,
            0xff0f => self.interrupt_flags | 0xe0,
            0xff40 => self.ppu.lcdc,
            0xff41 => self.ppu.read_stat(),
            0xff42 => self.ppu.scy,
            0xff43 => self.ppu.scx,
            0xff44 => self.ppu.ly,
            0xff45 => self.ppu.lyc,
            0xff47 => self.ppu.bgp,
            0xff48 => self.ppu.obp0,
            0xff49 => self.ppu.obp1,
            0xff4a => self.ppu.wy,
            0xff4b => self.ppu.wx,
            0xff00..=0xff7f => self.io[(addr - 0xff00) as usize],
            0xff80..=0xfffe => self.hram[(addr - 0xff80) as usize],
            0xffff => self.interrupt_enable,
        }
    }
    #[allow(clippy::match_overlapping_arm)]
    pub fn write8(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..=0x7fff | 0xa000..=0xbfff => self.cart.write(addr, value),
            0x8000..=0x9fff => self.ppu.vram[(addr - 0x8000) as usize] = value,
            0xc000..=0xdfff => self.wram[(addr - 0xc000) as usize] = value,
            0xe000..=0xfdff => self.wram[(addr - 0xe000) as usize] = value,
            0xfe00..=0xfe9f => self.ppu.oam[(addr - 0xfe00) as usize] = value,
            0xfea0..=0xfeff => {}
            0xff00 => self.joypad.write(value),
            0xff01 => self.io[1] = value,
            0xff02 => {
                self.io[2] = value;
                if value == 0x81 {
                    self.serial.push(self.io[1]);
                    self.io[2] = 1;
                    self.interrupt_flags |= 8;
                }
            }
            0xff04 => self.timer.write_div(),
            0xff05 => self.timer.tima = value,
            0xff06 => self.timer.tma = value,
            0xff07 => self.timer.write_tac(value),
            0xff0f => self.interrupt_flags = value & 0x1f,
            0xff40 => self.ppu.write_lcdc(value),
            0xff41 => self.ppu.stat = value & 0x78,
            0xff42 => self.ppu.scy = value,
            0xff43 => self.ppu.scx = value,
            0xff44 => {}
            0xff45 => self.ppu.lyc = value,
            0xff46 => self.dma = Some(((value as u16) << 8, 0)),
            0xff47 => self.ppu.bgp = value,
            0xff48 => self.ppu.obp0 = value,
            0xff49 => self.ppu.obp1 = value,
            0xff4a => self.ppu.wy = value,
            0xff4b => self.ppu.wx = value,
            0xff00..=0xff7f => self.io[(addr - 0xff00) as usize] = value,
            0xff80..=0xfffe => self.hram[(addr - 0xff80) as usize] = value,
            0xffff => self.interrupt_enable = value,
        }
    }
    pub fn read16(&self, a: u16) -> u16 {
        self.read8(a) as u16 | (self.read8(a.wrapping_add(1)) as u16) << 8
    }
    pub fn write16(&mut self, a: u16, v: u16) {
        self.write8(a, v as u8);
        self.write8(a.wrapping_add(1), (v >> 8) as u8);
    }
    pub fn tick(&mut self, cycles: u8) {
        for _ in 0..cycles {
            if self.timer.tick() {
                self.interrupt_flags |= 4
            }
            let irq = self.ppu.tick();
            self.interrupt_flags |= irq;
            if let Some((src, n)) = self.dma {
                if n < 160 && n % 4 == 0 {
                    let b = self.read8(src + n / 4);
                    self.ppu.oam[(n / 4) as usize] = b;
                }
                self.dma = if n >= 639 { None } else { Some((src, n + 1)) };
            }
        }
    }
}
