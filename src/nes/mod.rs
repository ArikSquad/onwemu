//! Repository-owned NES (NTSC RP2A03/2C02) emulator core.
//! The current cartridge scope intentionally matches nesma: iNES mapper 0 (NROM).

const WIDTH: usize = 256;
const HEIGHT: usize = 240;

pub struct Emulator {
    cpu: Cpu,
    bus: Bus,
}

impl Emulator {
    pub fn new(rom: &[u8]) -> Result<Self, String> {
        let cart = Cartridge::new(rom)?;
        let mut bus = Bus::new(cart);
        let mut cpu = Cpu::default();
        cpu.reset(&mut bus);
        Ok(Self { cpu, bus })
    }

    pub fn run_frame(&mut self) {
        self.bus.ppu.frame_ready = false;
        while !self.bus.ppu.frame_ready {
            self.cpu.step(&mut self.bus);
        }
    }

    pub fn framebuffer(&self) -> &[u8] {
        &self.bus.ppu.frame
    }
    pub fn set_button(&mut self, button: usize, down: bool) {
        if button < 8 {
            self.bus.pad.buttons[button] = down;
        }
    }
    pub fn pc(&self) -> u16 {
        self.cpu.pc
    }
}

struct Cartridge {
    prg: Vec<u8>,
    chr: Vec<u8>,
    chr_ram: bool,
    vertical: bool,
}
impl Cartridge {
    fn new(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 || &data[..4] != b"NES\x1a" {
            return Err("Not an iNES ROM".into());
        }
        let mapper = (data[6] >> 4) | (data[7] & 0xf0);
        if mapper != 0 {
            return Err(format!(
                "Mapper {mapper} is unsupported; this core currently supports NROM (mapper 0)"
            ));
        }
        let trainer = if data[6] & 4 != 0 { 512 } else { 0 };
        let prg_len = data[4] as usize * 0x4000;
        let chr_len = data[5] as usize * 0x2000;
        let start = 16 + trainer;
        if prg_len == 0 || data.len() < start + prg_len + chr_len {
            return Err("Truncated iNES ROM".into());
        }
        Ok(Self {
            prg: data[start..start + prg_len].to_vec(),
            chr: if chr_len == 0 {
                vec![0; 0x2000]
            } else {
                data[start + prg_len..start + prg_len + chr_len].to_vec()
            },
            chr_ram: chr_len == 0,
            vertical: data[6] & 1 != 0,
        })
    }
}

#[derive(Default)]
struct Controller {
    buttons: [bool; 8],
    shift: u8,
    strobe: bool,
}
impl Controller {
    fn write(&mut self, value: u8) {
        self.strobe = value & 1 != 0;
        if self.strobe {
            self.latch();
        }
    }
    fn latch(&mut self) {
        self.shift = 0;
        for i in 0..8 {
            self.shift |= (self.buttons[i] as u8) << i;
        }
    }
    fn read(&mut self) -> u8 {
        if self.strobe {
            self.latch();
        }
        let r = self.shift & 1;
        if !self.strobe {
            self.shift = (self.shift >> 1) | 0x80;
        }
        r | 0x40
    }
}

struct Ppu {
    chr: Vec<u8>,
    chr_ram: bool,
    vertical: bool,
    nt: [u8; 0x800],
    pal: [u8; 32],
    oam: [u8; 256],
    frame: Vec<u8>,
    ctrl: u8,
    mask: u8,
    status: u8,
    oam_addr: u8,
    bus: u8,
    v: u16,
    t: u16,
    fine_x: u8,
    latch: bool,
    buffer: u8,
    scanline: i16,
    dot: u16,
    frame_ready: bool,
    nmi: bool,
}
impl Ppu {
    fn new(cart: &Cartridge) -> Self {
        Self {
            chr: cart.chr.clone(),
            chr_ram: cart.chr_ram,
            vertical: cart.vertical,
            nt: [0; 0x800],
            pal: [0; 32],
            oam: [0; 256],
            frame: vec![0; WIDTH * HEIGHT],
            ctrl: 0,
            mask: 0,
            status: 0,
            oam_addr: 0,
            bus: 0,
            v: 0,
            t: 0,
            fine_x: 0,
            latch: false,
            buffer: 0,
            scanline: 261,
            dot: 0,
            frame_ready: false,
            nmi: false,
        }
    }
    fn mirror(&self, a: u16) -> usize {
        let a = (a - 0x2000) & 0xfff;
        let table = a / 0x400;
        let table = if self.vertical {
            table & 1
        } else {
            (table >> 1) & 1
        };
        (table * 0x400 + (a & 0x3ff)) as usize
    }
    fn pal_addr(a: u16) -> usize {
        let mut a = ((a - 0x3f00) & 0x1f) as usize;
        if matches!(a, 0x10 | 0x14 | 0x18 | 0x1c) {
            a -= 0x10
        }
        a
    }
    fn read_mem(&self, a: u16) -> u8 {
        let a = a & 0x3fff;
        if a < 0x2000 {
            self.chr[a as usize]
        } else if a < 0x3f00 {
            self.nt[self.mirror(a)]
        } else {
            self.pal[Self::pal_addr(a)] & 0x3f
        }
    }
    fn write_mem(&mut self, a: u16, val: u8) {
        let a = a & 0x3fff;
        if a < 0x2000 {
            if self.chr_ram {
                self.chr[a as usize] = val
            }
        } else if a < 0x3f00 {
            let i = self.mirror(a);
            self.nt[i] = val
        } else {
            self.pal[Self::pal_addr(a)] = val & 0x3f
        }
    }
    fn read_reg(&mut self, a: u16) -> u8 {
        match a & 7 {
            2 => {
                let r = (self.status & 0xe0) | (self.bus & 0x1f);
                self.status &= !0x80;
                self.latch = false;
                self.bus = r;
                r
            }
            4 => {
                self.bus = self.oam[self.oam_addr as usize];
                self.bus
            }
            7 => {
                let old = self.v;
                let r = if old & 0x3fff >= 0x3f00 {
                    let r = self.read_mem(old);
                    self.buffer = self.read_mem(old - 0x1000);
                    r
                } else {
                    let r = self.buffer;
                    self.buffer = self.read_mem(old);
                    r
                };
                self.v = self.v.wrapping_add(if self.ctrl & 4 != 0 { 32 } else { 1 });
                self.bus = r;
                r
            }
            _ => self.bus,
        }
    }
    fn write_reg(&mut self, a: u16, val: u8) {
        self.bus = val;
        match a & 7 {
            0 => {
                self.ctrl = val;
                self.t = (self.t & 0xf3ff) | ((val as u16 & 3) << 10)
            }
            1 => self.mask = val,
            3 => self.oam_addr = val,
            4 => {
                self.oam[self.oam_addr as usize] = val;
                self.oam_addr = self.oam_addr.wrapping_add(1)
            }
            5 => {
                if !self.latch {
                    self.fine_x = val & 7;
                    self.t = (self.t & 0xffe0) | (val as u16 >> 3)
                } else {
                    self.t = (self.t & 0x8fff) | ((val as u16 & 7) << 12);
                    self.t = (self.t & 0xfc1f) | ((val as u16 & 0xf8) << 2)
                }
                self.latch = !self.latch
            }
            6 => {
                if !self.latch {
                    self.t = (self.t & 0xff) | ((val as u16 & 0x3f) << 8)
                } else {
                    self.t = (self.t & 0xff00) | val as u16;
                    self.v = self.t
                }
                self.latch = !self.latch
            }
            7 => {
                self.write_mem(self.v, val);
                self.v = self.v.wrapping_add(if self.ctrl & 4 != 0 { 32 } else { 1 })
            }
            _ => {}
        }
    }
    fn inc_x(&mut self) {
        if self.v & 0x1f == 31 {
            self.v &= !0x1f;
            self.v ^= 0x400
        } else {
            self.v += 1
        }
    }
    fn inc_y(&mut self) {
        if self.v & 0x7000 != 0x7000 {
            self.v += 0x1000
        } else {
            self.v &= !0x7000;
            let mut y = (self.v & 0x3e0) >> 5;
            if y == 29 {
                y = 0;
                self.v ^= 0x800
            } else if y == 31 {
                y = 0
            } else {
                y += 1
            }
            self.v = (self.v & !0x3e0) | (y << 5)
        }
    }
    fn bg(&self, px: usize) -> u8 {
        if self.mask & 8 == 0 || (px < 8 && self.mask & 2 == 0) {
            return 0;
        }
        let fx = (px + self.fine_x as usize) & 7;
        let tile = self.read_mem(0x2000 | (self.v & 0xfff));
        let attr =
            self.read_mem(0x23c0 | (self.v & 0xc00) | ((self.v >> 4) & 0x38) | ((self.v >> 2) & 7));
        let shift = ((self.v >> 4) & 4) | (self.v & 2);
        let hi = (attr >> shift) & 3;
        let pat = (if self.ctrl & 0x10 != 0 { 0x1000 } else { 0 })
            + tile as u16 * 16
            + ((self.v >> 12) & 7);
        let bit = 7 - fx;
        let p = ((self.read_mem(pat) >> bit) & 1) | (((self.read_mem(pat + 8) >> bit) & 1) << 1);
        if p == 0 { 0 } else { hi << 2 | p }
    }
    fn sprite(&mut self, px: usize, py: usize, bg: bool) -> u8 {
        if self.mask & 0x10 == 0 || (px < 8 && self.mask & 4 == 0) {
            return 0;
        }
        let h = if self.ctrl & 0x20 != 0 { 16 } else { 8 };
        let mut found = 0;
        for i in 0..64 {
            let sy = self.oam[i * 4] as i16;
            let mut tile = self.oam[i * 4 + 1];
            let attr = self.oam[i * 4 + 2];
            let sx = self.oam[i * 4 + 3] as i16;
            let mut row = py as i16 - sy - 1;
            if row < 0 || row >= h {
                continue;
            }
            found += 1;
            if found > 8 {
                break;
            }
            let mut col = px as i16 - sx;
            if !(0..8).contains(&col) {
                continue;
            }
            if attr & 0x40 != 0 {
                col = 7 - col
            }
            if attr & 0x80 != 0 {
                row = h - 1 - row
            }
            let base = if h == 16 {
                let b = (tile & 1) as u16 * 0x1000;
                tile &= 0xfe;
                if row >= 8 {
                    tile += 1;
                    row -= 8
                }
                b
            } else if self.ctrl & 8 != 0 {
                0x1000
            } else {
                0
            };
            let bit = 7 - col as u8;
            let p = ((self.read_mem(base + tile as u16 * 16 + row as u16) >> bit) & 1)
                | (((self.read_mem(base + tile as u16 * 16 + row as u16 + 8) >> bit) & 1) << 1);
            if p == 0 {
                continue;
            }
            if i == 0 && bg && px != 255 {
                self.status |= 0x40
            }
            if attr & 0x20 != 0 && bg {
                return 0;
            }
            return 0x10 | ((attr & 3) << 2) | p;
        }
        0
    }
    fn tick(&mut self) {
        let rendering = self.mask & 0x18 != 0;
        if (0..240).contains(&self.scanline) && (1..=256).contains(&self.dot) {
            let px = self.dot as usize - 1;
            let bg = self.bg(px);
            let sp = self.sprite(px, self.scanline as usize, bg != 0);
            let idx = if sp != 0 { sp } else { bg };
            self.frame[self.scanline as usize * WIDTH + px] =
                self.read_mem(0x3f00 + idx as u16) & 0x3f;
            if rendering && (px + self.fine_x as usize) & 7 == 7 {
                self.inc_x()
            }
            if rendering && self.dot == 256 {
                self.inc_y()
            }
        }
        if rendering && self.scanline < 240 && self.dot == 257 {
            self.v = (self.v & !0x41f) | (self.t & 0x41f)
        }
        if rendering && self.scanline == 261 && (280..=304).contains(&self.dot) {
            self.v = (self.v & !0x7be0) | (self.t & 0x7be0)
        }
        if self.scanline == 241 && self.dot == 1 {
            self.status |= 0x80;
            self.frame_ready = true;
            self.nmi = self.ctrl & 0x80 != 0
        }
        if self.scanline == 261 && self.dot == 1 {
            self.status &= !0xe0;
            self.frame_ready = false
        }
        self.dot += 1;
        if self.dot > 340 {
            self.dot = 0;
            self.scanline += 1;
            if self.scanline > 261 {
                self.scanline = 0
            }
        }
    }
}

struct Bus {
    ram: [u8; 0x800],
    prg: Vec<u8>,
    ppu: Ppu,
    pad: Controller,
    pad2: Controller,
    dma_stall: u16,
}
impl Bus {
    fn new(cart: Cartridge) -> Self {
        let ppu = Ppu::new(&cart);
        Self {
            ram: [0; 0x800],
            prg: cart.prg,
            ppu,
            pad: Controller::default(),
            pad2: Controller::default(),
            dma_stall: 0,
        }
    }
    fn read(&mut self, a: u16) -> u8 {
        match a {
            0..=0x1fff => self.ram[a as usize & 0x7ff],
            0x2000..=0x3fff => self.ppu.read_reg(a),
            0x4016 => self.pad.read(),
            0x4017 => self.pad2.read(),
            0x8000..=0xffff => self.prg[(a as usize - 0x8000) % self.prg.len()],
            _ => 0,
        }
    }
    fn write(&mut self, a: u16, v: u8) {
        match a {
            0..=0x1fff => self.ram[a as usize & 0x7ff] = v,
            0x2000..=0x3fff => self.ppu.write_reg(a, v),
            0x4014 => {
                let base = (v as u16) << 8;
                for i in 0..256 {
                    let b = self.read(base + i);
                    self.ppu.oam[self.ppu.oam_addr.wrapping_add(i as u8) as usize] = b
                }
                self.dma_stall = 513
            }
            0x4016 => {
                self.pad.write(v);
                self.pad2.write(v)
            }
            _ => {}
        }
    }
    fn read16(&mut self, a: u16) -> u16 {
        self.read(a) as u16 | ((self.read(a.wrapping_add(1)) as u16) << 8)
    }
    fn tick(&mut self, n: u16) {
        for _ in 0..n * 3 {
            self.ppu.tick()
        }
    }
}

#[derive(Default)]
struct Cpu {
    a: u8,
    x: u8,
    y: u8,
    s: u8,
    p: u8,
    pc: u16,
}
const C: u8 = 1;
const Z: u8 = 2;
const I: u8 = 4;
const D: u8 = 8;
const B: u8 = 16;
const U: u8 = 32;
const V: u8 = 64;
const N: u8 = 128;
impl Cpu {
    fn reset(&mut self, b: &mut Bus) {
        self.s = 0xfd;
        self.p = I | U;
        self.pc = b.read16(0xfffc)
    }
    fn flag(&self, f: u8) -> bool {
        self.p & f != 0
    }
    fn set(&mut self, f: u8, on: bool) {
        if on { self.p |= f } else { self.p &= !f }
    }
    fn zn(&mut self, v: u8) {
        self.set(Z, v == 0);
        self.set(N, v & 0x80 != 0)
    }
    fn fetch(&mut self, b: &mut Bus) -> u8 {
        let v = b.read(self.pc);
        self.pc = self.pc.wrapping_add(1);
        v
    }
    fn word(&mut self, b: &mut Bus) -> u16 {
        let l = self.fetch(b) as u16;
        let h = self.fetch(b) as u16;
        l | h << 8
    }
    fn push(&mut self, b: &mut Bus, v: u8) {
        b.write(0x100 | self.s as u16, v);
        self.s = self.s.wrapping_sub(1)
    }
    fn pop(&mut self, b: &mut Bus) -> u8 {
        self.s = self.s.wrapping_add(1);
        b.read(0x100 | self.s as u16)
    }
    fn addr(&mut self, b: &mut Bus, mode: u8) -> (u16, bool) {
        match mode {
            0 => (self.fetch(b) as u16, false),
            1 => (self.fetch(b).wrapping_add(self.x) as u16, false),
            2 => (self.fetch(b).wrapping_add(self.y) as u16, false),
            3 => (self.word(b), false),
            4 => {
                let a = self.word(b);
                (
                    a.wrapping_add(self.x as u16),
                    (a & 0xff00) != (a.wrapping_add(self.x as u16) & 0xff00),
                )
            }
            5 => {
                let a = self.word(b);
                (
                    a.wrapping_add(self.y as u16),
                    (a & 0xff00) != (a.wrapping_add(self.y as u16) & 0xff00),
                )
            }
            6 => {
                let z = self.fetch(b).wrapping_add(self.x);
                (
                    b.read(z as u16) as u16 | ((b.read(z.wrapping_add(1) as u16) as u16) << 8),
                    false,
                )
            }
            _ => {
                let z = self.fetch(b);
                let a = b.read(z as u16) as u16 | ((b.read(z.wrapping_add(1) as u16) as u16) << 8);
                (
                    a.wrapping_add(self.y as u16),
                    (a & 0xff00) != (a.wrapping_add(self.y as u16) & 0xff00),
                )
            }
        }
    }
    fn adc(&mut self, v: u8) {
        let sum = self.a as u16 + v as u16 + self.flag(C) as u16;
        let r = sum as u8;
        self.set(C, sum > 255);
        self.set(V, (!(self.a ^ v) & (self.a ^ r) & 0x80) != 0);
        self.a = r;
        self.zn(r)
    }
    fn cmp(&mut self, a: u8, v: u8) {
        self.set(C, a >= v);
        self.zn(a.wrapping_sub(v))
    }
    fn branch(&mut self, b: &mut Bus, take: bool) -> u16 {
        let off = self.fetch(b) as i8;
        if take {
            let old = self.pc;
            self.pc = self.pc.wrapping_add_signed(off as i16);
            if old & 0xff00 != self.pc & 0xff00 {
                4
            } else {
                3
            }
        } else {
            2
        }
    }
    fn interrupt(&mut self, b: &mut Bus, vector: u16, brk: bool) {
        self.push(b, (self.pc >> 8) as u8);
        self.push(b, self.pc as u8);
        self.push(b, (self.p & !B) | U | if brk { B } else { 0 });
        self.set(I, true);
        self.pc = b.read16(vector)
    }
    fn step(&mut self, b: &mut Bus) {
        if b.ppu.nmi {
            b.ppu.nmi = false;
            self.interrupt(b, 0xfffa, false);
            b.tick(7);
            return;
        }
        if b.dma_stall > 0 {
            let n = b.dma_stall;
            b.dma_stall = 0;
            b.tick(n);
            return;
        }
        let op = self.fetch(b);
        let cycles = self.exec(b, op);
        b.tick(cycles)
    }
    fn exec(&mut self, b: &mut Bus, op: u8) -> u16 {
        let mut cy = 2;
        match op {
            0x00 => {
                self.pc = self.pc.wrapping_add(1);
                self.interrupt(b, 0xfffe, true);
                cy = 7
            }
            0x40 => {
                self.p = (self.pop(b) & !B) | U;
                let l = self.pop(b) as u16;
                self.pc = l | ((self.pop(b) as u16) << 8);
                cy = 6
            }
            0x60 => {
                let l = self.pop(b) as u16;
                self.pc = (l | ((self.pop(b) as u16) << 8)).wrapping_add(1);
                cy = 6
            }
            0x20 => {
                let a = self.word(b);
                let ret = self.pc.wrapping_sub(1);
                self.push(b, (ret >> 8) as u8);
                self.push(b, ret as u8);
                self.pc = a;
                cy = 6
            }
            0x4c => {
                self.pc = self.word(b);
                cy = 3
            }
            0x6c => {
                let p = self.word(b);
                let lo = b.read(p) as u16;
                let hi = b.read((p & 0xff00) | p.wrapping_add(1) & 0xff) as u16;
                self.pc = lo | hi << 8;
                cy = 5
            }
            0x10 => cy = self.branch(b, !self.flag(N)),
            0x30 => cy = self.branch(b, self.flag(N)),
            0x50 => cy = self.branch(b, !self.flag(V)),
            0x70 => cy = self.branch(b, self.flag(V)),
            0x90 => cy = self.branch(b, !self.flag(C)),
            0xb0 => cy = self.branch(b, self.flag(C)),
            0xd0 => cy = self.branch(b, !self.flag(Z)),
            0xf0 => cy = self.branch(b, self.flag(Z)),
            0x18 => self.set(C, false),
            0x38 => self.set(C, true),
            0x58 => self.set(I, false),
            0x78 => self.set(I, true),
            0xb8 => self.set(V, false),
            0xd8 => self.set(D, false),
            0xf8 => self.set(D, true),
            0x48 => {
                self.push(b, self.a);
                cy = 3
            }
            0x68 => {
                self.a = self.pop(b);
                self.zn(self.a);
                cy = 4
            }
            0x08 => {
                self.push(b, self.p | B | U);
                cy = 3
            }
            0x28 => {
                self.p = (self.pop(b) & !B) | U;
                cy = 4
            }
            0xaa => {
                self.x = self.a;
                self.zn(self.x)
            }
            0x8a => {
                self.a = self.x;
                self.zn(self.a)
            }
            0xa8 => {
                self.y = self.a;
                self.zn(self.y)
            }
            0x98 => {
                self.a = self.y;
                self.zn(self.a)
            }
            0xba => {
                self.x = self.s;
                self.zn(self.x)
            }
            0x9a => self.s = self.x,
            0xe8 => {
                self.x = self.x.wrapping_add(1);
                self.zn(self.x)
            }
            0xca => {
                self.x = self.x.wrapping_sub(1);
                self.zn(self.x)
            }
            0xc8 => {
                self.y = self.y.wrapping_add(1);
                self.zn(self.y)
            }
            0x88 => {
                self.y = self.y.wrapping_sub(1);
                self.zn(self.y)
            }
            0xea | 0x1a | 0x3a | 0x5a | 0x7a | 0xda | 0xfa => {}
            0x0a => {
                self.set(C, self.a & 0x80 != 0);
                self.a <<= 1;
                self.zn(self.a)
            }
            0x2a => {
                let c = self.flag(C);
                self.set(C, self.a & 0x80 != 0);
                self.a = (self.a << 1) | c as u8;
                self.zn(self.a)
            }
            0x4a => {
                self.set(C, self.a & 1 != 0);
                self.a >>= 1;
                self.zn(self.a)
            }
            0x6a => {
                let c = self.flag(C);
                self.set(C, self.a & 1 != 0);
                self.a = (self.a >> 1) | ((c as u8) << 7);
                self.zn(self.a)
            }
            0xa0 | 0xa4 | 0xb4 | 0xac | 0xbc => {
                let (v, c) = self.read_indexed(b, op);
                self.y = v;
                self.zn(v);
                cy = c
            }
            0xa2 | 0xa6 | 0xb6 | 0xae | 0xbe => {
                let (v, c) = self.read_indexed(b, op);
                self.x = v;
                self.zn(v);
                cy = c
            }
            0x84 | 0x94 | 0x8c => {
                let (a, c) = self.store_addr(b, op);
                b.write(a, self.y);
                cy = c
            }
            0x86 | 0x96 | 0x8e => {
                let (a, c) = self.store_addr(b, op);
                b.write(a, self.x);
                cy = c
            }
            0x24 | 0x2c => {
                let (a, _) = if op == 0x24 {
                    self.addr(b, 0)
                } else {
                    self.addr(b, 3)
                };
                let v = b.read(a);
                self.set(Z, self.a & v == 0);
                self.set(V, v & 0x40 != 0);
                self.set(N, v & 0x80 != 0);
                cy = if op == 0x24 { 3 } else { 4 }
            }
            0xc0 | 0xc4 | 0xcc => {
                let (v, c) = self.read_indexed(b, op);
                self.cmp(self.y, v);
                cy = c
            }
            0xe0 | 0xe4 | 0xec => {
                let (v, c) = self.read_indexed(b, op);
                self.cmp(self.x, v);
                cy = c
            }
            _ => cy = self.memory_op(b, op),
        }
        cy
    }
    fn read_indexed(&mut self, b: &mut Bus, op: u8) -> (u8, u16) {
        if matches!(op, 0xa0 | 0xa2 | 0xc0 | 0xe0) {
            return (self.fetch(b), 2);
        }
        let (mode, base) = match op {
            0xa4 | 0xa6 | 0xc4 | 0xe4 => (0, 3),
            0xb4 => (1, 4),
            0xb6 => (2, 4),
            0xac | 0xae | 0xcc | 0xec => (3, 4),
            0xbc => (4, 4),
            0xbe => (5, 4),
            _ => (0, 3),
        };
        let (a, page) = self.addr(b, mode);
        (b.read(a), base + page as u16)
    }
    fn store_addr(&mut self, b: &mut Bus, op: u8) -> (u16, u16) {
        let (mode, c) = match op {
            0x84 | 0x86 => (0, 3),
            0x94 => (1, 4),
            0x96 => (2, 4),
            _ => (3, 4),
        };
        let (a, _) = self.addr(b, mode);
        (a, c)
    }
    fn memory_op(&mut self, b: &mut Bus, op: u8) -> u16 {
        let lo = op & 0x1f;
        let mode = match lo {
            0x05 | 0x06 => 0,
            0x15 | 0x16 => 1,
            0x0d | 0x0e => 3,
            0x1d | 0x1e => 4,
            0x19 => 5,
            0x01 => 6,
            0x11 => 7,
            0x09 => 8,
            _ => 255,
        };
        if mode == 255 {
            return 2;
        }
        let (addr, page) = if mode == 8 {
            (0, false)
        } else {
            self.addr(b, mode)
        };
        let mut v = if mode == 8 {
            self.fetch(b)
        } else {
            b.read(addr)
        };
        let group = op & 0xe0;
        if lo == 0x06 || lo == 0x16 || lo == 0x0e || lo == 0x1e {
            let kind = op & 0xe0;
            match kind {
                0x00 => {
                    self.set(C, v & 0x80 != 0);
                    v <<= 1
                }
                0x20 => {
                    let c = self.flag(C);
                    self.set(C, v & 0x80 != 0);
                    v = (v << 1) | c as u8
                }
                0x40 => {
                    self.set(C, v & 1 != 0);
                    v >>= 1
                }
                0x60 => {
                    let c = self.flag(C);
                    self.set(C, v & 1 != 0);
                    v = (v >> 1) | ((c as u8) << 7)
                }
                0xc0 => v = v.wrapping_sub(1),
                0xe0 => v = v.wrapping_add(1),
                _ => {}
            }
            b.write(addr, v);
            self.zn(v);
            return if mode == 3 {
                6
            } else if mode == 4 {
                7
            } else {
                5
            };
        }
        match group {
            0x00 => {
                self.a |= v;
                self.zn(self.a)
            }
            0x20 => {
                self.a &= v;
                self.zn(self.a)
            }
            0x40 => {
                self.a ^= v;
                self.zn(self.a)
            }
            0x60 => self.adc(v),
            0x80 => {
                if mode != 8 {
                    b.write(addr, self.a);
                    return match mode {
                        0..=2 => 3,
                        3 => 4,
                        4 | 5 => 5,
                        _ => 6,
                    };
                }
            }
            0xa0 => {
                self.a = v;
                self.zn(v)
            }
            0xc0 => self.cmp(self.a, v),
            0xe0 => self.adc(!v),
            _ => {}
        }
        match mode {
            8 => 2,
            0 => 3,
            1 => 4,
            3 => 4,
            4 | 5 => 4 + page as u16,
            6 => 6,
            7 => 5 + page as u16,
            _ => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rom() -> Vec<u8> {
        let mut r = vec![0; 16 + 0x4000];
        r[..6].copy_from_slice(&[b'N', b'E', b'S', 0x1a, 1, 0]);
        r[16] = 0x4c;
        r[17] = 0;
        r[18] = 0x80;
        r[16 + 0x3ffc] = 0;
        r[16 + 0x3ffd] = 0x80;
        r
    }
    #[test]
    fn rejects_bad_rom() {
        assert!(Emulator::new(b"bad").is_err())
    }
    #[test]
    fn executes_frame() {
        let mut e = Emulator::new(&rom()).unwrap();
        e.run_frame();
        assert_eq!(e.pc(), 0x8000);
        assert_eq!(e.framebuffer().len(), WIDTH * HEIGHT)
    }
}
