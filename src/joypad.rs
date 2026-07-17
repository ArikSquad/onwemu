#[derive(Default)]
pub struct Joypad {
    select: u8,
    buttons: u8,
    previous: u8,
}

impl Joypad {
    pub fn read(&self) -> u8 {
        0xc0 | self.select | self.lines()
    }
    pub fn write(&mut self, value: u8) {
        self.select = value & 0x30;
    }
    pub fn set(&mut self, bit: u8, down: bool) -> bool {
        self.previous = self.lines();
        if down {
            self.buttons |= 1 << bit
        } else {
            self.buttons &= !(1 << bit)
        }
        self.previous & !self.lines() != 0
    }
    fn lines(&self) -> u8 {
        let mut out = 0x0f;
        if self.select & 0x10 == 0 {
            out &= !(self.buttons & 0x0f);
        }
        if self.select & 0x20 == 0 {
            out &= !((self.buttons >> 4) & 0x0f);
        }
        out
    }
}
