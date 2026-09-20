use std::{fs, path::Path};

use anyhow::{Context, Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mapper {
    Rom,
    Mbc1,
    Mbc2,
    Mbc3,
    Mbc5,
}

const CLOCK_HZ: u32 = 4_194_304;
const SAVE_MAGIC: &[u8; 8] = b"ONWGBS01";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Rtc {
    seconds: u8,
    minutes: u8,
    hours: u8,
    days: u16,
    halted: bool,
    carry: bool,
    subsecond_cycles: u32,
}

impl Rtc {
    fn tick(&mut self, cycles: u32) {
        if self.halted {
            return;
        }
        let total = self.subsecond_cycles as u64 + cycles as u64;
        self.subsecond_cycles = (total % CLOCK_HZ as u64) as u32;
        for _ in 0..total / CLOCK_HZ as u64 {
            self.increment_second();
        }
    }

    fn increment_second(&mut self) {
        self.seconds += 1;
        if self.seconds < 60 {
            return;
        }
        self.seconds = 0;
        self.minutes += 1;
        if self.minutes < 60 {
            return;
        }
        self.minutes = 0;
        self.hours += 1;
        if self.hours < 24 {
            return;
        }
        self.hours = 0;
        self.days += 1;
        if self.days > 511 {
            self.days = 0;
            self.carry = true;
        }
    }

    fn read(self, register: u8) -> u8 {
        match register {
            0x08 => self.seconds,
            0x09 => self.minutes,
            0x0a => self.hours,
            0x0b => self.days as u8,
            0x0c => {
                ((self.days >> 8) as u8 & 1)
                    | ((self.halted as u8) << 6)
                    | ((self.carry as u8) << 7)
            }
            _ => 0xff,
        }
    }

    fn write(&mut self, register: u8, value: u8) {
        match register {
            0x08 => self.seconds = value % 60,
            0x09 => self.minutes = value % 60,
            0x0a => self.hours = value % 24,
            0x0b => self.days = (self.days & 0x100) | value as u16,
            0x0c => {
                self.days = (self.days & 0xff) | ((value as u16 & 1) << 8);
                self.halted = value & 0x40 != 0;
                self.carry = value & 0x80 != 0;
            }
            _ => {}
        }
    }
}

pub struct Cartridge {
    rom: Vec<u8>,
    ram: Vec<u8>,
    mapper: Mapper,
    ram_enabled: bool,
    rom_bank: u16,
    ram_bank: u8,
    mode: u8,
    rtc_select: u8,
    rtc: Rtc,
    latched_rtc: Rtc,
    rtc_latch_armed: bool,
    has_rtc: bool,
    has_battery: bool,
    title: String,
}

impl Cartridge {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let data = fs::read(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        Self::from_bytes(data)
    }

    pub fn from_bytes(mut rom: Vec<u8>) -> Result<Self> {
        if rom.len() < 0x150 {
            bail!("ROM is too small to contain a cartridge header")
        }
        let cartridge_type = rom[0x147];
        let mapper = match cartridge_type {
            0x00 | 0x08 | 0x09 => Mapper::Rom,
            0x01..=0x03 => Mapper::Mbc1,
            0x05 | 0x06 => Mapper::Mbc2,
            0x0f..=0x13 => Mapper::Mbc3,
            0x19..=0x1e => Mapper::Mbc5,
            kind => bail!("unsupported cartridge type {kind:#04x}"),
        };
        let expected = match rom[0x148] {
            code @ 0x00..=0x08 => 0x8000usize << code,
            0x52 => 72 * 0x4000,
            0x53 => 80 * 0x4000,
            0x54 => 96 * 0x4000,
            _ => rom.len(),
        };
        if expected > rom.len() {
            rom.resize(expected, 0xff);
        }
        let ram_len = match rom[0x149] {
            0 => 0,
            1 => 0x800,
            2 => 0x2000,
            3 => 0x8000,
            4 => 0x20000,
            5 => 0x10000,
            _ => 0,
        };
        let title = rom[0x134..=0x143]
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as char)
            .collect();
        Ok(Self {
            rom,
            ram: vec![0xff; if mapper == Mapper::Mbc2 { 512 } else { ram_len }],
            mapper,
            ram_enabled: mapper == Mapper::Rom,
            rom_bank: 1,
            ram_bank: 0,
            mode: 0,
            rtc_select: 0,
            rtc: Rtc::default(),
            latched_rtc: Rtc::default(),
            rtc_latch_armed: false,
            has_rtc: matches!(cartridge_type, 0x0f | 0x10),
            has_battery: matches!(
                cartridge_type,
                0x03 | 0x06 | 0x09 | 0x0f | 0x10 | 0x13 | 0x1b | 0x1e
            ),
            title,
        })
    }

    pub fn title(&self) -> &str {
        &self.title
    }
    pub fn ram(&self) -> &[u8] {
        &self.ram
    }
    pub fn load_ram(&mut self, data: &[u8]) {
        let n = data.len().min(self.ram.len());
        self.ram[..n].copy_from_slice(&data[..n]);
    }

    pub fn has_battery(&self) -> bool {
        self.has_battery
    }

    /// Returns a versioned battery save containing external RAM and MBC3 RTC state.
    pub fn save_data(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(SAVE_MAGIC.len() + 4 + self.ram.len() + 11);
        data.extend_from_slice(SAVE_MAGIC);
        data.extend_from_slice(&(self.ram.len() as u32).to_le_bytes());
        data.extend_from_slice(&self.ram);
        data.extend_from_slice(&[
            self.rtc.seconds,
            self.rtc.minutes,
            self.rtc.hours,
            self.rtc.days as u8,
            (self.rtc.days >> 8) as u8,
            self.rtc.halted as u8,
            self.rtc.carry as u8,
        ]);
        data.extend_from_slice(&self.rtc.subsecond_cycles.to_le_bytes());
        data
    }

    /// Loads a versioned save, or treats legacy data as a raw external-RAM image.
    pub fn load_save_data(&mut self, data: &[u8]) -> Result<()> {
        if !data.starts_with(SAVE_MAGIC) {
            self.load_ram(data);
            return Ok(());
        }
        if data.len() < 12 {
            bail!("truncated Game Boy save header")
        }
        let ram_len = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let rtc_start = 12usize
            .checked_add(ram_len)
            .context("invalid Game Boy save length")?;
        if data.len() < rtc_start + 11 {
            bail!("truncated Game Boy save payload")
        }
        self.load_ram(&data[12..rtc_start]);
        let rtc = &data[rtc_start..rtc_start + 11];
        self.rtc = Rtc {
            seconds: rtc[0] % 60,
            minutes: rtc[1] % 60,
            hours: rtc[2] % 24,
            days: u16::from_le_bytes([rtc[3], rtc[4]]) & 0x1ff,
            halted: rtc[5] != 0,
            carry: rtc[6] != 0,
            subsecond_cycles: u32::from_le_bytes(rtc[7..11].try_into().unwrap()) % CLOCK_HZ,
        };
        self.latched_rtc = self.rtc;
        Ok(())
    }

    pub fn tick(&mut self, cycles: u32) {
        if self.has_rtc {
            self.rtc.tick(cycles);
        }
    }

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x3fff => {
                let bank = if self.mapper == Mapper::Mbc1 && self.mode != 0 {
                    ((self.ram_bank as usize) << 5) % self.rom_banks()
                } else {
                    0
                };
                self.rom[(bank * 0x4000 + addr as usize) % self.rom.len()]
            }
            0x4000..=0x7fff => {
                let bank = self.effective_rom_bank();
                self.rom[(bank * 0x4000 + addr as usize - 0x4000) % self.rom.len()]
            }
            0xa000..=0xbfff
                if self.ram_enabled
                    && self.mapper == Mapper::Mbc3
                    && (0x08..=0x0c).contains(&self.rtc_select) =>
            {
                if self.has_rtc {
                    self.latched_rtc.read(self.rtc_select)
                } else {
                    0xff
                }
            }
            0xa000..=0xbfff if self.ram_enabled && !self.ram.is_empty() => {
                let bank = if self.mapper == Mapper::Mbc1 && self.mode == 0 {
                    0
                } else {
                    self.ram_bank as usize
                };
                let i = if self.mapper == Mapper::Mbc2 {
                    (addr as usize) & 0x1ff
                } else {
                    bank * 0x2000 + addr as usize - 0xa000
                };
                self.ram.get(i % self.ram.len()).copied().unwrap_or(0xff)
            }
            _ => 0xff,
        }
    }

    pub fn write(&mut self, addr: u16, value: u8) {
        match (self.mapper, addr) {
            (Mapper::Rom, 0xa000..=0xbfff) => self.write_ram(addr, value),
            (Mapper::Mbc2, 0x0000..=0x3fff) if addr & 0x100 == 0 => {
                self.ram_enabled = value & 0x0f == 0x0a
            }
            (Mapper::Mbc2, 0x0000..=0x3fff) => self.rom_bank = (value & 0x0f).max(1) as u16,
            (_, 0x0000..=0x1fff) => self.ram_enabled = value & 0x0f == 0x0a,
            (Mapper::Mbc1, 0x2000..=0x3fff) => {
                self.rom_bank = (self.rom_bank & 0x60) | (value as u16 & 0x1f).max(1)
            }
            (Mapper::Mbc1, 0x4000..=0x5fff) => {
                self.ram_bank = value & 3;
                self.rom_bank = (self.rom_bank & 0x1f) | ((value as u16 & 3) << 5);
            }
            (Mapper::Mbc1, 0x6000..=0x7fff) => self.mode = value & 1,
            (Mapper::Mbc3, 0x2000..=0x3fff) => self.rom_bank = (value & 0x7f).max(1) as u16,
            (Mapper::Mbc3, 0x4000..=0x5fff) => {
                self.ram_bank = value & 3;
                self.rtc_select = value;
            }
            (Mapper::Mbc3, 0x6000..=0x7fff) => {
                if value == 0 {
                    self.rtc_latch_armed = true;
                } else if value == 1 && self.rtc_latch_armed {
                    self.latched_rtc = self.rtc;
                    self.rtc_latch_armed = false;
                } else {
                    self.rtc_latch_armed = false;
                }
            }
            (Mapper::Mbc5, 0x2000..=0x2fff) => {
                self.rom_bank = (self.rom_bank & 0x100) | value as u16
            }
            (Mapper::Mbc5, 0x3000..=0x3fff) => {
                self.rom_bank = (self.rom_bank & 0xff) | ((value as u16 & 1) << 8)
            }
            (Mapper::Mbc5, 0x4000..=0x5fff) => self.ram_bank = value & 0x0f,
            (_, 0xa000..=0xbfff) => self.write_ram(addr, value),
            _ => {}
        }
    }

    fn rom_banks(&self) -> usize {
        (self.rom.len() / 0x4000).max(1)
    }
    fn effective_rom_bank(&self) -> usize {
        (self.rom_bank as usize % self.rom_banks()).max((self.mapper != Mapper::Mbc5) as usize)
    }
    fn write_ram(&mut self, addr: u16, value: u8) {
        if !self.ram_enabled {
            return;
        }
        if self.mapper == Mapper::Mbc3 && (0x08..=0x0c).contains(&self.rtc_select) {
            if self.has_rtc {
                self.rtc.write(self.rtc_select, value);
            }
            return;
        }
        if self.ram.is_empty() {
            return;
        }
        let bank = if self.mapper == Mapper::Mbc1 && self.mode == 0 {
            0
        } else {
            self.ram_bank as usize
        };
        let i = if self.mapper == Mapper::Mbc2 {
            addr as usize & 0x1ff
        } else {
            bank * 0x2000 + addr as usize - 0xa000
        } % self.ram.len();
        self.ram[i] = if self.mapper == Mapper::Mbc2 {
            value | 0xf0
        } else {
            value
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cartridge(kind: u8, ram_size: u8) -> Cartridge {
        let mut rom = vec![0; 0x8000];
        rom[0x147] = kind;
        rom[0x148] = 0;
        rom[0x149] = ram_size;
        Cartridge::from_bytes(rom).unwrap()
    }

    #[test]
    fn mbc3_rtc_ticks_latches_and_halts() {
        let mut cart = cartridge(0x10, 2);
        cart.write(0x0000, 0x0a);
        cart.tick(CLOCK_HZ);
        cart.write(0x6000, 0);
        cart.write(0x6000, 1);
        cart.write(0x4000, 0x08);
        assert_eq!(cart.read(0xa000), 1);

        cart.write(0x4000, 0x0c);
        cart.write(0xa000, 0x40);
        cart.tick(CLOCK_HZ);
        cart.write(0x6000, 0);
        cart.write(0x6000, 1);
        cart.write(0x4000, 0x08);
        assert_eq!(cart.read(0xa000), 1);
    }

    #[test]
    fn versioned_save_round_trips_ram_and_rtc() {
        let mut source = cartridge(0x10, 2);
        source.write(0x0000, 0x0a);
        source.write(0xa000, 0x42);
        source.tick(CLOCK_HZ * 2);
        let save = source.save_data();

        let mut restored = cartridge(0x10, 2);
        restored.load_save_data(&save).unwrap();
        restored.write(0x0000, 0x0a);
        assert_eq!(restored.read(0xa000), 0x42);
        restored.write(0x6000, 0);
        restored.write(0x6000, 1);
        restored.write(0x4000, 0x08);
        assert_eq!(restored.read(0xa000), 2);
    }
}
