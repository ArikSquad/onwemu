#[derive(Default)]
pub struct Timer {
    div: u16,
    pub tima: u8,
    pub tma: u8,
    pub tac: u8,
    reload: Option<u8>,
}

impl Timer {
    pub fn read_div(&self) -> u8 {
        (self.div >> 8) as u8
    }
    pub fn write_div(&mut self) {
        let old = self.signal();
        self.div = 0;
        if old && !self.signal() {
            self.increment();
        }
    }
    pub fn write_tac(&mut self, v: u8) {
        let old = self.signal();
        self.tac = v & 7;
        if old && !self.signal() {
            self.increment();
        }
    }
    fn signal(&self) -> bool {
        self.tac & 4 != 0
            && self.div
                & (match self.tac & 3 {
                    0 => 1 << 9,
                    1 => 1 << 3,
                    2 => 1 << 5,
                    _ => 1 << 7,
                })
                != 0
    }
    fn increment(&mut self) {
        if self.tima == 0xff {
            self.tima = 0;
            self.reload = Some(4);
        } else {
            self.tima += 1;
        }
    }
    pub fn tick(&mut self) -> bool {
        let old = self.signal();
        self.div = self.div.wrapping_add(1);
        if old && !self.signal() {
            self.increment();
        }
        if let Some(n) = self.reload {
            if n == 1 {
                self.tima = self.tma;
                self.reload = None;
                return true;
            }
            self.reload = Some(n - 1);
        }
        false
    }
}
