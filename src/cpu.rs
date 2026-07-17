use crate::bus::Bus;

const Z: u8 = 0x80;
const N: u8 = 0x40;
const H: u8 = 0x20;
const C: u8 = 0x10;

#[derive(Clone, Debug)]
pub struct Cpu {
    pub a: u8,
    pub f: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
    pub e: u8,
    pub h: u8,
    pub l: u8,
    pub sp: u16,
    pub pc: u16,
    pub ime: bool,
    halted: bool,
    ime_delay: u8,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            a: 0x01,
            f: 0xb0,
            b: 0,
            c: 0x13,
            d: 0,
            e: 0xd8,
            h: 0x01,
            l: 0x4d,
            sp: 0xfffe,
            pc: 0x0100,
            ime: false,
            halted: false,
            ime_delay: 0,
        }
    }
}

impl Cpu {
    pub fn af(&self) -> u16 {
        (self.a as u16) << 8 | self.f as u16
    }
    pub fn bc(&self) -> u16 {
        (self.b as u16) << 8 | self.c as u16
    }
    pub fn de(&self) -> u16 {
        (self.d as u16) << 8 | self.e as u16
    }
    pub fn hl(&self) -> u16 {
        (self.h as u16) << 8 | self.l as u16
    }
    pub fn set_af(&mut self, v: u16) {
        self.a = (v >> 8) as u8;
        self.f = v as u8 & 0xf0
    }
    fn set_bc(&mut self, v: u16) {
        self.b = (v >> 8) as u8;
        self.c = v as u8
    }
    fn set_de(&mut self, v: u16) {
        self.d = (v >> 8) as u8;
        self.e = v as u8
    }
    fn set_hl(&mut self, v: u16) {
        self.h = (v >> 8) as u8;
        self.l = v as u8
    }
    fn fetch8(&mut self, b: &Bus) -> u8 {
        let v = b.read8(self.pc);
        self.pc = self.pc.wrapping_add(1);
        v
    }
    fn fetch16(&mut self, b: &Bus) -> u16 {
        let lo = self.fetch8(b) as u16;
        lo | (self.fetch8(b) as u16) << 8
    }
    fn r8(&self, bus: &Bus, n: u8) -> u8 {
        match n {
            0 => self.b,
            1 => self.c,
            2 => self.d,
            3 => self.e,
            4 => self.h,
            5 => self.l,
            6 => bus.read8(self.hl()),
            _ => self.a,
        }
    }
    fn w8(&mut self, bus: &mut Bus, n: u8, v: u8) {
        match n {
            0 => self.b = v,
            1 => self.c = v,
            2 => self.d = v,
            3 => self.e = v,
            4 => self.h = v,
            5 => self.l = v,
            6 => bus.write8(self.hl(), v),
            _ => self.a = v,
        }
    }
    fn rr(&self, n: u8) -> u16 {
        match n {
            0 => self.bc(),
            1 => self.de(),
            2 => self.hl(),
            _ => self.sp,
        }
    }
    fn wr(&mut self, n: u8, v: u16) {
        match n {
            0 => self.set_bc(v),
            1 => self.set_de(v),
            2 => self.set_hl(v),
            _ => self.sp = v,
        }
    }
    fn push(&mut self, b: &mut Bus, v: u16) {
        self.sp = self.sp.wrapping_sub(2);
        b.write16(self.sp, v)
    }
    fn pop(&mut self, b: &Bus) -> u16 {
        let v = b.read16(self.sp);
        self.sp = self.sp.wrapping_add(2);
        v
    }
    fn cond(&self, n: u8) -> bool {
        match n {
            0 => self.f & Z == 0,
            1 => self.f & Z != 0,
            2 => self.f & C == 0,
            _ => self.f & C != 0,
        }
    }
    fn alu(&mut self, op: u8, v: u8) {
        let a = self.a;
        match op {
            0 => {
                let (r, c) = a.overflowing_add(v);
                self.a = r;
                self.f = if r == 0 { Z } else { 0 }
                    | if (a & 15) + (v & 15) > 15 { H } else { 0 }
                    | if c { C } else { 0 }
            }
            1 => {
                let ci = (self.f & C != 0) as u8;
                let (r, c1) = a.overflowing_add(v);
                let (r, c2) = r.overflowing_add(ci);
                self.a = r;
                self.f = if r == 0 { Z } else { 0 }
                    | if (a & 15) + (v & 15) + ci > 15 { H } else { 0 }
                    | if c1 || c2 { C } else { 0 }
            }
            2 => {
                let (r, c) = a.overflowing_sub(v);
                self.a = r;
                self.f = N
                    | if r == 0 { Z } else { 0 }
                    | if (a & 15) < (v & 15) { H } else { 0 }
                    | if c { C } else { 0 }
            }
            3 => {
                let ci = (self.f & C != 0) as u8;
                let (r, c1) = a.overflowing_sub(v);
                let (r, c2) = r.overflowing_sub(ci);
                self.a = r;
                self.f = N
                    | if r == 0 { Z } else { 0 }
                    | if (a & 15) < (v & 15) + ci { H } else { 0 }
                    | if c1 || c2 { C } else { 0 }
            }
            4 => {
                self.a &= v;
                self.f = if self.a == 0 { Z | H } else { H }
            }
            5 => {
                self.a ^= v;
                self.f = if self.a == 0 { Z } else { 0 }
            }
            6 => {
                self.a |= v;
                self.f = if self.a == 0 { Z } else { 0 }
            }
            _ => {
                let (r, c) = a.overflowing_sub(v);
                self.f = N
                    | if r == 0 { Z } else { 0 }
                    | if (a & 15) < (v & 15) { H } else { 0 }
                    | if c { C } else { 0 }
            }
        }
    }
    fn inc(&mut self, v: u8) -> u8 {
        let r = v.wrapping_add(1);
        self.f = (self.f & C) | if r == 0 { Z } else { 0 } | if v & 15 == 15 { H } else { 0 };
        r
    }
    fn dec(&mut self, v: u8) -> u8 {
        let r = v.wrapping_sub(1);
        self.f = (self.f & C) | N | if r == 0 { Z } else { 0 } | if v & 15 == 0 { H } else { 0 };
        r
    }

    pub fn step(&mut self, b: &mut Bus) -> u8 {
        let pending = b.interrupt_enable & b.interrupt_flags & 0x1f;
        if pending != 0 {
            self.halted = false;
            if self.ime {
                let bit = pending.trailing_zeros() as u8;
                self.ime = false;
                b.interrupt_flags &= !(1 << bit);
                self.push(b, self.pc);
                self.pc = 0x40 + bit as u16 * 8;
                b.tick(20);
                return 20;
            }
        }
        if self.halted {
            b.tick(4);
            return 4;
        }
        let op = self.fetch8(b);
        let cycles = self.execute(b, op);
        b.tick(cycles);
        if self.ime_delay > 0 {
            self.ime_delay -= 1;
            if self.ime_delay == 0 {
                self.ime = true
            }
        }
        cycles
    }

    fn execute(&mut self, b: &mut Bus, op: u8) -> u8 {
        // Dense regular blocks: LD r,r and ALU A,r.
        if (0x40..=0x7f).contains(&op) {
            if op == 0x76 {
                self.halted = true;
                return 4;
            }
            let d = (op >> 3) & 7;
            let s = op & 7;
            let v = self.r8(b, s);
            self.w8(b, d, v);
            return if d == 6 || s == 6 { 8 } else { 4 };
        }
        if (0x80..=0xbf).contains(&op) {
            let s = op & 7;
            let v = self.r8(b, s);
            self.alu((op >> 3) & 7, v);
            return if s == 6 { 8 } else { 4 };
        }
        if op & 0xc7 == 0x04 {
            let r = (op >> 3) & 7;
            let v = self.r8(b, r);
            let v = self.inc(v);
            self.w8(b, r, v);
            return if r == 6 { 12 } else { 4 };
        }
        if op & 0xc7 == 0x05 {
            let r = (op >> 3) & 7;
            let v = self.r8(b, r);
            let v = self.dec(v);
            self.w8(b, r, v);
            return if r == 6 { 12 } else { 4 };
        }
        if op & 0xc7 == 0x06 {
            let r = (op >> 3) & 7;
            let v = self.fetch8(b);
            self.w8(b, r, v);
            return if r == 6 { 12 } else { 8 };
        }
        if op & 0xcf == 0x01 {
            let n = (op >> 4) & 3;
            let v = self.fetch16(b);
            self.wr(n, v);
            return 12;
        }
        if op & 0xcf == 0x03 {
            let n = (op >> 4) & 3;
            self.wr(n, self.rr(n).wrapping_add(1));
            return 8;
        }
        if op & 0xcf == 0x0b {
            let n = (op >> 4) & 3;
            self.wr(n, self.rr(n).wrapping_sub(1));
            return 8;
        }
        if op & 0xcf == 0x09 {
            let v = self.rr((op >> 4) & 3);
            let hl = self.hl();
            let r = hl.wrapping_add(v);
            self.f = (self.f & Z)
                | if (hl & 0xfff) + (v & 0xfff) > 0xfff {
                    H
                } else {
                    0
                }
                | if (hl as u32 + v as u32) > 0xffff {
                    C
                } else {
                    0
                };
            self.set_hl(r);
            return 8;
        }
        if op & 0xe7 == 0x20 {
            let e = self.fetch8(b) as i8;
            if self.cond((op >> 3) & 3) {
                self.pc = self.pc.wrapping_add_signed(e as i16);
                return 12;
            }
            return 8;
        }
        if op & 0xe7 == 0xc0 {
            if self.cond((op >> 3) & 3) {
                self.pc = self.pop(b);
                return 20;
            }
            return 8;
        }
        if op & 0xe7 == 0xc2 {
            let a = self.fetch16(b);
            if self.cond((op >> 3) & 3) {
                self.pc = a;
                return 16;
            }
            return 12;
        }
        if op & 0xe7 == 0xc4 {
            let a = self.fetch16(b);
            if self.cond((op >> 3) & 3) {
                self.push(b, self.pc);
                self.pc = a;
                return 24;
            }
            return 12;
        }
        if op & 0xc7 == 0xc7 {
            self.push(b, self.pc);
            self.pc = (op & 0x38) as u16;
            return 16;
        }
        if op & 0xcf == 0xc1 {
            let v = self.pop(b);
            match (op >> 4) & 3 {
                0 => self.set_bc(v),
                1 => self.set_de(v),
                2 => self.set_hl(v),
                _ => self.set_af(v),
            }
            return 12;
        }
        if op & 0xcf == 0xc5 {
            let v = match (op >> 4) & 3 {
                0 => self.bc(),
                1 => self.de(),
                2 => self.hl(),
                _ => self.af(),
            };
            self.push(b, v);
            return 16;
        }
        match op {
            0x00 => 4,
            0x02 => {
                b.write8(self.bc(), self.a);
                8
            }
            0x12 => {
                b.write8(self.de(), self.a);
                8
            }
            0x0a => {
                self.a = b.read8(self.bc());
                8
            }
            0x1a => {
                self.a = b.read8(self.de());
                8
            }
            0x08 => {
                let a = self.fetch16(b);
                b.write16(a, self.sp);
                20
            }
            0x07 => {
                let c = self.a >> 7;
                self.a = self.a.rotate_left(1);
                self.f = if c != 0 { C } else { 0 };
                4
            }
            0x0f => {
                let c = self.a & 1;
                self.a = self.a.rotate_right(1);
                self.f = if c != 0 { C } else { 0 };
                4
            }
            0x17 => {
                let c = self.a >> 7;
                self.a = (self.a << 1) | ((self.f & C != 0) as u8);
                self.f = if c != 0 { C } else { 0 };
                4
            }
            0x1f => {
                let c = self.a & 1;
                self.a = (self.a >> 1) | if self.f & C != 0 { 0x80 } else { 0 };
                self.f = if c != 0 { C } else { 0 };
                4
            }
            0x10 => {
                self.fetch8(b);
                4
            }
            0x18 => {
                let e = self.fetch8(b) as i8;
                self.pc = self.pc.wrapping_add_signed(e as i16);
                12
            }
            0x22 => {
                let a = self.hl();
                b.write8(a, self.a);
                self.set_hl(a.wrapping_add(1));
                8
            }
            0x2a => {
                let a = self.hl();
                self.a = b.read8(a);
                self.set_hl(a.wrapping_add(1));
                8
            }
            0x32 => {
                let a = self.hl();
                b.write8(a, self.a);
                self.set_hl(a.wrapping_sub(1));
                8
            }
            0x3a => {
                let a = self.hl();
                self.a = b.read8(a);
                self.set_hl(a.wrapping_sub(1));
                8
            }
            0x27 => {
                let mut fix = 0;
                let mut carry = self.f & C != 0;
                if self.f & N == 0 {
                    if self.f & H != 0 || self.a & 15 > 9 {
                        fix |= 6
                    }
                    if carry || self.a > 0x99 {
                        fix |= 0x60;
                        carry = true
                    }
                    self.a = self.a.wrapping_add(fix)
                } else {
                    if self.f & H != 0 {
                        fix |= 6
                    }
                    if carry {
                        fix |= 0x60
                    }
                    self.a = self.a.wrapping_sub(fix)
                }
                self.f = (self.f & N) | if self.a == 0 { Z } else { 0 } | if carry { C } else { 0 };
                4
            }
            0x2f => {
                self.a = !self.a;
                self.f |= N | H;
                4
            }
            0x37 => {
                self.f = (self.f & Z) | C;
                4
            }
            0x3f => {
                self.f = (self.f & Z) | if self.f & C == 0 { C } else { 0 };
                4
            }
            0xc3 => {
                self.pc = self.fetch16(b);
                16
            }
            0xc6 => {
                let v = self.fetch8(b);
                self.alu(0, v);
                8
            }
            0xce => {
                let v = self.fetch8(b);
                self.alu(1, v);
                8
            }
            0xd6 => {
                let v = self.fetch8(b);
                self.alu(2, v);
                8
            }
            0xde => {
                let v = self.fetch8(b);
                self.alu(3, v);
                8
            }
            0xe6 => {
                let v = self.fetch8(b);
                self.alu(4, v);
                8
            }
            0xee => {
                let v = self.fetch8(b);
                self.alu(5, v);
                8
            }
            0xf6 => {
                let v = self.fetch8(b);
                self.alu(6, v);
                8
            }
            0xfe => {
                let v = self.fetch8(b);
                self.alu(7, v);
                8
            }
            0xc9 => {
                self.pc = self.pop(b);
                16
            }
            0xd9 => {
                self.pc = self.pop(b);
                self.ime = true;
                16
            }
            0xcd => {
                let a = self.fetch16(b);
                self.push(b, self.pc);
                self.pc = a;
                24
            }
            0xcb => self.cb(b),
            0xe0 => {
                let a = 0xff00 | self.fetch8(b) as u16;
                b.write8(a, self.a);
                12
            }
            0xf0 => {
                let a = 0xff00 | self.fetch8(b) as u16;
                self.a = b.read8(a);
                12
            }
            0xe2 => {
                b.write8(0xff00 | self.c as u16, self.a);
                8
            }
            0xf2 => {
                self.a = b.read8(0xff00 | self.c as u16);
                8
            }
            0xea => {
                let a = self.fetch16(b);
                b.write8(a, self.a);
                16
            }
            0xfa => {
                let a = self.fetch16(b);
                self.a = b.read8(a);
                16
            }
            0xe8 => {
                let e = self.fetch8(b) as i8;
                let u = e as i16 as u16;
                let sp = self.sp;
                self.sp = sp.wrapping_add(u);
                self.f = if (sp & 15) + (u & 15) > 15 { H } else { 0 }
                    | if (sp & 255) + (u & 255) > 255 { C } else { 0 };
                16
            }
            0xf8 => {
                let e = self.fetch8(b) as i8;
                let u = e as i16 as u16;
                let sp = self.sp;
                self.set_hl(sp.wrapping_add(u));
                self.f = if (sp & 15) + (u & 15) > 15 { H } else { 0 }
                    | if (sp & 255) + (u & 255) > 255 { C } else { 0 };
                12
            }
            0xf9 => {
                self.sp = self.hl();
                8
            }
            0xe9 => {
                self.pc = self.hl();
                4
            }
            0xf3 => {
                self.ime = false;
                self.ime_delay = 0;
                4
            }
            0xfb => {
                self.ime_delay = 2;
                4
            }
            _ => 4,
        }
    }
    fn cb(&mut self, b: &mut Bus) -> u8 {
        let op = self.fetch8(b);
        let r = op & 7;
        let v = self.r8(b, r);
        let group = op >> 6;
        let bit = (op >> 3) & 7;
        if group == 1 {
            self.f = (self.f & C) | H | if v & (1 << bit) == 0 { Z } else { 0 };
            return if r == 6 { 12 } else { 8 };
        }
        if group >= 2 {
            let out = if group == 2 {
                v & !(1 << bit)
            } else {
                v | (1 << bit)
            };
            self.w8(b, r, out);
            return if r == 6 { 16 } else { 8 };
        }
        let kind = (op >> 3) & 7;
        let (out, carry) = match kind {
            0 => (v.rotate_left(1), v >> 7),
            1 => (v.rotate_right(1), v & 1),
            2 => ((v << 1) | ((self.f & C != 0) as u8), v >> 7),
            3 => ((v >> 1) | if self.f & C != 0 { 0x80 } else { 0 }, v & 1),
            4 => (v << 1, v >> 7),
            5 => (((v as i8) >> 1) as u8, v & 1),
            6 => (v.rotate_left(4), 0),
            _ => (v >> 1, v & 1),
        };
        self.w8(b, r, out);
        self.f = if out == 0 { Z } else { 0 } | if carry != 0 { C } else { 0 };
        if r == 6 { 16 } else { 8 }
    }
}
