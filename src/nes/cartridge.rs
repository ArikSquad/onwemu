//! iNES cartridge parsing and mapper-controlled address translation.
//!
//! The CPU and PPU both access the same cartridge instance.  Keeping mapper
//! registers here avoids the duplicated PRG/CHR state that made bank switching
//! impossible in the original NROM-only core.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mirroring {
    Horizontal,
    Vertical,
    OneScreenLower,
    OneScreenUpper,
    FourScreen,
}

#[derive(Debug)]
enum Mapper {
    Nrom,
    Mmc1 {
        shift: u8,
        control: u8,
        chr_bank_0: u8,
        chr_bank_1: u8,
        prg_bank: u8,
    },
    Uxrom {
        bank: u8,
    },
    Cnrom {
        bank: u8,
    },
    Mmc3 {
        select: u8,
        registers: [u8; 8],
        mirroring: Mirroring,
        ram_enabled: bool,
        ram_write_protected: bool,
        irq_latch: u8,
        irq_counter: u8,
        irq_reload: bool,
        irq_enabled: bool,
        irq_pending: bool,
    },
    Axrom {
        bank: u8,
        upper_screen: bool,
    },
}

pub(super) struct Cartridge {
    prg_rom: Vec<u8>,
    prg_ram: Vec<u8>,
    chr: Vec<u8>,
    chr_ram: bool,
    mapper: Mapper,
    header_mirroring: Mirroring,
}

impl Cartridge {
    pub(super) fn new(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 || &data[..4] != b"NES\x1a" {
            return Err("Not an iNES ROM".into());
        }
        if data[7] & 0x0c == 0x08 {
            return Err("NES 2.0 ROMs are not supported yet".into());
        }
        let mapper_number = (data[6] >> 4) | (data[7] & 0xf0);
        let trainer = if data[6] & 4 != 0 { 512 } else { 0 };
        let prg_len = data[4] as usize * 0x4000;
        let chr_len = data[5] as usize * 0x2000;
        let start = 16 + trainer;
        if prg_len == 0 || data.len() < start + prg_len + chr_len {
            return Err("Truncated iNES ROM".into());
        }
        let header_mirroring = if data[6] & 8 != 0 {
            Mirroring::FourScreen
        } else if data[6] & 1 != 0 {
            Mirroring::Vertical
        } else {
            Mirroring::Horizontal
        };
        let mapper = match mapper_number {
            0 => Mapper::Nrom,
            1 => Mapper::Mmc1 {
                shift: 0x10,
                control: 0x0c | if data[6] & 1 != 0 { 2 } else { 3 },
                chr_bank_0: 0,
                chr_bank_1: 0,
                prg_bank: 0,
            },
            2 => Mapper::Uxrom { bank: 0 },
            3 => Mapper::Cnrom { bank: 0 },
            4 => Mapper::Mmc3 {
                select: 0,
                registers: [0; 8],
                mirroring: header_mirroring,
                ram_enabled: true,
                ram_write_protected: false,
                irq_latch: 0,
                irq_counter: 0,
                irq_reload: false,
                irq_enabled: false,
                irq_pending: false,
            },
            7 => Mapper::Axrom {
                bank: 0,
                upper_screen: false,
            },
            _ => {
                return Err(format!(
                    "Mapper {mapper_number} is unsupported (supported: 0, 1, 2, 3, 4, 7)"
                ));
            }
        };
        let ram_units = data[8].max(1) as usize;
        Ok(Self {
            prg_rom: data[start..start + prg_len].to_vec(),
            prg_ram: vec![0; ram_units * 0x2000],
            chr: if chr_len == 0 {
                vec![0; 0x2000]
            } else {
                data[start + prg_len..start + prg_len + chr_len].to_vec()
            },
            chr_ram: chr_len == 0,
            mapper,
            header_mirroring,
        })
    }

    pub(super) fn cpu_read(&self, address: u16) -> u8 {
        match address {
            0x6000..=0x7fff if self.ram_readable() => {
                self.prg_ram[(address as usize - 0x6000) % self.prg_ram.len()]
            }
            0x8000..=0xffff => {
                let offset = self.prg_offset(address);
                self.prg_rom[offset % self.prg_rom.len()]
            }
            _ => 0,
        }
    }

    pub(super) fn cpu_write(&mut self, address: u16, value: u8) {
        if (0x6000..=0x7fff).contains(&address) {
            if self.ram_writable() {
                let index = (address as usize - 0x6000) % self.prg_ram.len();
                self.prg_ram[index] = value;
            }
            return;
        }
        if address < 0x8000 {
            return;
        }
        match &mut self.mapper {
            Mapper::Nrom => {}
            Mapper::Mmc1 {
                shift,
                control,
                chr_bank_0,
                chr_bank_1,
                prg_bank,
            } => {
                if value & 0x80 != 0 {
                    *shift = 0x10;
                    *control |= 0x0c;
                    return;
                }
                let complete = *shift & 1 != 0;
                *shift = (*shift >> 1) | ((value & 1) << 4);
                if complete {
                    let register = match address {
                        0x8000..=0x9fff => control,
                        0xa000..=0xbfff => chr_bank_0,
                        0xc000..=0xdfff => chr_bank_1,
                        _ => prg_bank,
                    };
                    *register = *shift & 0x1f;
                    *shift = 0x10;
                }
            }
            Mapper::Uxrom { bank } => *bank = value & 0x0f,
            Mapper::Cnrom { bank } => *bank = value & 0x03,
            Mapper::Mmc3 {
                select,
                registers,
                mirroring,
                ram_enabled,
                ram_write_protected,
                irq_latch,
                irq_reload,
                irq_enabled,
                irq_pending,
                ..
            } => match address & 0xe001 {
                0x8000 => *select = value,
                0x8001 => registers[*select as usize & 7] = value,
                0xa000 if self.header_mirroring != Mirroring::FourScreen => {
                    *mirroring = if value & 1 == 0 {
                        Mirroring::Vertical
                    } else {
                        Mirroring::Horizontal
                    }
                }
                0xa001 => {
                    *ram_enabled = value & 0x80 != 0;
                    *ram_write_protected = value & 0x40 != 0;
                }
                0xc000 => *irq_latch = value,
                0xc001 => *irq_reload = true,
                0xe000 => {
                    *irq_enabled = false;
                    *irq_pending = false;
                }
                0xe001 => *irq_enabled = true,
                _ => {}
            },
            Mapper::Axrom { bank, upper_screen } => {
                *bank = value & 7;
                *upper_screen = value & 0x10 != 0;
            }
        }
    }

    pub(super) fn ppu_read(&self, address: u16) -> u8 {
        let offset = self.chr_offset(address & 0x1fff);
        self.chr[offset % self.chr.len()]
    }

    pub(super) fn ppu_write(&mut self, address: u16, value: u8) {
        if self.chr_ram {
            let offset = self.chr_offset(address & 0x1fff) % self.chr.len();
            self.chr[offset] = value;
        }
    }

    pub(super) fn mirroring(&self) -> Mirroring {
        match &self.mapper {
            Mapper::Mmc1 { control, .. } => match control & 3 {
                0 => Mirroring::OneScreenLower,
                1 => Mirroring::OneScreenUpper,
                2 => Mirroring::Vertical,
                _ => Mirroring::Horizontal,
            },
            Mapper::Mmc3 { mirroring, .. } => *mirroring,
            Mapper::Axrom { upper_screen, .. } => {
                if *upper_screen {
                    Mirroring::OneScreenUpper
                } else {
                    Mirroring::OneScreenLower
                }
            }
            _ => self.header_mirroring,
        }
    }

    /// MMC3 scanline IRQ approximation, clocked at the PPU's rendering fetch point.
    pub(super) fn scanline_tick(&mut self) {
        if let Mapper::Mmc3 {
            irq_latch,
            irq_counter,
            irq_reload,
            irq_enabled,
            irq_pending,
            ..
        } = &mut self.mapper
        {
            if *irq_counter == 0 || *irq_reload {
                *irq_counter = *irq_latch;
                *irq_reload = false;
            } else {
                *irq_counter -= 1;
            }
            if *irq_counter == 0 && *irq_enabled {
                *irq_pending = true;
            }
        }
    }

    pub(super) fn irq_pending(&self) -> bool {
        matches!(
            &self.mapper,
            Mapper::Mmc3 {
                irq_pending: true,
                ..
            }
        )
    }

    fn ram_readable(&self) -> bool {
        !matches!(
            &self.mapper,
            Mapper::Mmc3 {
                ram_enabled: false,
                ..
            } | Mapper::Mmc1 {
                prg_bank: 0x10..=0x1f,
                ..
            }
        )
    }

    fn ram_writable(&self) -> bool {
        !matches!(
            &self.mapper,
            Mapper::Mmc3 {
                ram_enabled: false,
                ..
            } | Mapper::Mmc3 {
                ram_write_protected: true,
                ..
            } | Mapper::Mmc1 {
                prg_bank: 0x10..=0x1f,
                ..
            }
        )
    }

    fn prg_offset(&self, address: u16) -> usize {
        let address = address as usize - 0x8000;
        let banks_16k = (self.prg_rom.len() / 0x4000).max(1);
        let banks_8k = (self.prg_rom.len() / 0x2000).max(1);
        match &self.mapper {
            Mapper::Nrom | Mapper::Cnrom { .. } => address % self.prg_rom.len(),
            Mapper::Uxrom { bank } => {
                let bank = if address < 0x4000 {
                    *bank as usize % banks_16k
                } else {
                    banks_16k - 1
                };
                bank * 0x4000 + (address & 0x3fff)
            }
            Mapper::Mmc1 {
                control, prg_bank, ..
            } => {
                let mode = (*control >> 2) & 3;
                let selected = *prg_bank as usize % banks_16k;
                let bank = match mode {
                    0 | 1 => (selected & !1) + (address / 0x4000),
                    2 if address < 0x4000 => 0,
                    2 => selected,
                    3 if address < 0x4000 => selected,
                    _ => banks_16k - 1,
                };
                bank % banks_16k * 0x4000 + (address & 0x3fff)
            }
            Mapper::Mmc3 {
                select, registers, ..
            } => {
                let last = banks_8k - 1;
                let second_last = last.saturating_sub(1);
                let r6 = registers[6] as usize % banks_8k;
                let r7 = registers[7] as usize % banks_8k;
                let slot = address / 0x2000;
                let bank = if select & 0x40 == 0 {
                    [r6, r7, second_last, last][slot]
                } else {
                    [second_last, r7, r6, last][slot]
                };
                bank * 0x2000 + (address & 0x1fff)
            }
            Mapper::Axrom { bank, .. } => {
                let banks = (self.prg_rom.len() / 0x8000).max(1);
                (*bank as usize % banks) * 0x8000 + address
            }
        }
    }

    fn chr_offset(&self, address: u16) -> usize {
        let address = address as usize;
        match &self.mapper {
            Mapper::Cnrom { bank } => *bank as usize * 0x2000 + address,
            Mapper::Mmc1 {
                control,
                chr_bank_0,
                chr_bank_1,
                ..
            } => {
                if control & 0x10 == 0 {
                    (*chr_bank_0 as usize & !1) * 0x1000 + address
                } else if address < 0x1000 {
                    *chr_bank_0 as usize * 0x1000 + address
                } else {
                    *chr_bank_1 as usize * 0x1000 + address - 0x1000
                }
            }
            Mapper::Mmc3 {
                select, registers, ..
            } => {
                let inverted = select & 0x80 != 0;
                let slot = address / 0x400;
                let normal = [
                    registers[0] & 0xfe,
                    registers[0] | 1,
                    registers[1] & 0xfe,
                    registers[1] | 1,
                    registers[2],
                    registers[3],
                    registers[4],
                    registers[5],
                ];
                let bank = if inverted {
                    [
                        normal[4], normal[5], normal[6], normal[7], normal[0], normal[1],
                        normal[2], normal[3],
                    ][slot]
                } else {
                    normal[slot]
                };
                bank as usize * 0x400 + (address & 0x3ff)
            }
            _ => address,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rom(mapper: u8, prg_banks: u8, chr_banks: u8) -> Vec<u8> {
        let mut data = vec![0; 16 + prg_banks as usize * 0x4000 + chr_banks as usize * 0x2000];
        data[..4].copy_from_slice(b"NES\x1a");
        data[4] = prg_banks;
        data[5] = chr_banks;
        data[6] = mapper << 4;
        for bank in 0..prg_banks as usize {
            data[16 + bank * 0x4000..16 + (bank + 1) * 0x4000].fill(bank as u8);
        }
        let chr_start = 16 + prg_banks as usize * 0x4000;
        for bank in 0..chr_banks as usize * 8 {
            data[chr_start + bank * 0x400..chr_start + (bank + 1) * 0x400].fill(bank as u8);
        }
        data
    }

    #[test]
    fn uxrom_switches_low_bank_and_fixes_high_bank() {
        let mut cart = Cartridge::new(&rom(2, 4, 0)).unwrap();
        assert_eq!(cart.cpu_read(0x8000), 0);
        assert_eq!(cart.cpu_read(0xc000), 3);
        cart.cpu_write(0x8000, 2);
        assert_eq!(cart.cpu_read(0x8000), 2);
        assert_eq!(cart.cpu_read(0xc000), 3);
    }

    #[test]
    fn mmc1_loads_register_lsb_first() {
        let mut cart = Cartridge::new(&rom(1, 4, 0)).unwrap();
        for bit in [0, 1, 0, 0, 0] {
            cart.cpu_write(0xe000, bit);
        }
        assert_eq!(cart.cpu_read(0x8000), 2);
        assert_eq!(cart.cpu_read(0xc000), 3);
    }

    #[test]
    fn chr_ram_is_writable_through_mapper() {
        let mut cart = Cartridge::new(&rom(0, 1, 0)).unwrap();
        cart.ppu_write(0x1234, 0x56);
        assert_eq!(cart.ppu_read(0x1234), 0x56);
    }

    #[test]
    fn cnrom_switches_eight_kib_chr_banks() {
        let mut cart = Cartridge::new(&rom(3, 2, 4)).unwrap();
        assert_eq!(cart.ppu_read(0), 0);
        cart.cpu_write(0x8000, 2);
        assert_eq!(cart.ppu_read(0), 16);
        assert_eq!(cart.ppu_read(0x1c00), 23);
    }

    #[test]
    fn mmc3_maps_prg_and_raises_scanline_irq() {
        let mut data = rom(4, 4, 1);
        for bank in 0..8 {
            let start = 16 + bank * 0x2000;
            data[start..start + 0x2000].fill(bank as u8);
        }
        let mut cart = Cartridge::new(&data).unwrap();
        cart.cpu_write(0x8000, 6);
        cart.cpu_write(0x8001, 3);
        assert_eq!(cart.cpu_read(0x8000), 3);
        assert_eq!(cart.cpu_read(0xc000), 6);
        assert_eq!(cart.cpu_read(0xe000), 7);

        cart.cpu_write(0xc000, 2);
        cart.cpu_write(0xe001, 0);
        for _ in 0..3 {
            cart.scanline_tick();
        }
        assert!(cart.irq_pending());
        cart.cpu_write(0xe000, 0);
        assert!(!cart.irq_pending());
    }

    #[test]
    fn axrom_switches_thirty_two_kib_banks_and_nametables() {
        let mut cart = Cartridge::new(&rom(7, 4, 0)).unwrap();
        cart.cpu_write(0x8000, 0x11);
        assert_eq!(cart.cpu_read(0x8000), 2);
        assert_eq!(cart.cpu_read(0xc000), 3);
        assert_eq!(cart.mirroring(), Mirroring::OneScreenUpper);
    }
}
