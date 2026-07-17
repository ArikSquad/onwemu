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

pub struct Cartridge {
    rom: Vec<u8>,
    ram: Vec<u8>,
    mapper: Mapper,
    ram_enabled: bool,
    rom_bank: u16,
    ram_bank: u8,
    mode: u8,
    rtc_select: u8,
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
        let mapper = match rom[0x147] {
            0x00 | 0x08 | 0x09 => Mapper::Rom,
            0x01..=0x03 => Mapper::Mbc1,
            0x05 | 0x06 => Mapper::Mbc2,
            0x0f..=0x13 => Mapper::Mbc3,
            0x19..=0x1e => Mapper::Mbc5,
            kind => bail!("unsupported cartridge type {kind:#04x}"),
        };
        let expected = 0x8000usize
            .checked_shl(rom[0x148] as u32)
            .unwrap_or(rom.len());
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
            ram_enabled: false,
            rom_bank: 1,
            ram_bank: 0,
            mode: 0,
            rtc_select: 0,
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
            0xa000..=0xbfff if self.ram_enabled && !self.ram.is_empty() => {
                if self.mapper == Mapper::Mbc3 && (0x08..=0x0c).contains(&self.rtc_select) {
                    return 0;
                }
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
        if !self.ram_enabled || self.ram.is_empty() {
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
