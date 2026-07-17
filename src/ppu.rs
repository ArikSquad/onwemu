pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 144;

pub struct Ppu {
    pub vram: [u8; 0x2000],
    pub oam: [u8; 0xa0],
    pub lcdc: u8,
    pub stat: u8,
    pub scy: u8,
    pub scx: u8,
    pub ly: u8,
    pub lyc: u8,
    pub bgp: u8,
    pub obp0: u8,
    pub obp1: u8,
    pub wy: u8,
    pub wx: u8,
    dots: u16,
    window_line: u8,
    frame_ready: bool,
    framebuffer: [u8; WIDTH * HEIGHT],
}

impl Default for Ppu {
    fn default() -> Self {
        Self {
            vram: [0; 0x2000],
            oam: [0; 0xa0],
            lcdc: 0x91,
            stat: 0x80,
            scy: 0,
            scx: 0,
            ly: 0,
            lyc: 0,
            bgp: 0xfc,
            obp0: 0xff,
            obp1: 0xff,
            wy: 0,
            wx: 0,
            dots: 0,
            window_line: 0,
            frame_ready: false,
            framebuffer: [0; WIDTH * HEIGHT],
        }
    }
}

impl Ppu {
    pub fn framebuffer(&self) -> &[u8; WIDTH * HEIGHT] {
        &self.framebuffer
    }
    pub fn take_frame_ready(&mut self) -> bool {
        std::mem::take(&mut self.frame_ready)
    }
    pub fn mode(&self) -> u8 {
        if self.lcdc & 0x80 == 0 {
            0
        } else if self.ly >= 144 {
            1
        } else if self.dots < 80 {
            2
        } else if self.dots < 252 {
            3
        } else {
            0
        }
    }
    pub fn read_stat(&self) -> u8 {
        0x80 | (self.stat & 0x78) | ((self.ly == self.lyc) as u8) << 2 | self.mode()
    }
    pub fn write_lcdc(&mut self, value: u8) {
        if self.lcdc & 0x80 != 0 && value & 0x80 == 0 {
            self.ly = 0;
            self.dots = 0;
            self.window_line = 0;
        }
        self.lcdc = value;
    }
    pub fn tick(&mut self) -> u8 {
        if self.lcdc & 0x80 == 0 {
            return 0;
        }
        let old_mode = self.mode();
        let old_equal = self.ly == self.lyc;
        self.dots += 1;
        if self.dots == 252 && self.ly < 144 {
            self.render_line();
        }
        let mut irq = 0;
        if self.dots >= 456 {
            self.dots = 0;
            self.ly += 1;
            if self.ly == 144 {
                self.frame_ready = true;
                irq |= 1;
            }
            if self.ly > 153 {
                self.ly = 0;
                self.window_line = 0;
            }
        }
        let mode = self.mode();
        if mode != old_mode
            && ((mode == 0 && self.stat & 8 != 0)
                || (mode == 1 && self.stat & 0x10 != 0)
                || (mode == 2 && self.stat & 0x20 != 0))
        {
            irq |= 2;
        }
        if !old_equal && self.ly == self.lyc && self.stat & 0x40 != 0 {
            irq |= 2;
        }
        irq
    }

    fn render_line(&mut self) {
        let y = self.ly as usize;
        if y >= HEIGHT {
            return;
        }
        let window = self.lcdc & 0x20 != 0 && self.ly >= self.wy && self.wx <= 166;
        let mut used_window = false;
        let mut bg_colors = [0u8; WIDTH];
        for (x, bg_color) in bg_colors.iter_mut().enumerate() {
            let use_window = window && x + 7 >= self.wx as usize;
            let (px, py) = if use_window {
                used_window = true;
                ((x + 7 - self.wx as usize) as u8, self.window_line)
            } else {
                (
                    (x as u8).wrapping_add(self.scx),
                    self.ly.wrapping_add(self.scy),
                )
            };
            let color = if self.lcdc & 1 == 0 {
                0
            } else {
                self.bg_pixel(px, py, use_window)
            };
            *bg_color = color;
            self.framebuffer[y * WIDTH + x] = palette(self.bgp, color);
        }
        if used_window {
            self.window_line = self.window_line.wrapping_add(1);
        }
        if self.lcdc & 2 != 0 {
            self.render_sprites(y, &bg_colors);
        }
    }

    fn bg_pixel(&self, x: u8, y: u8, window: bool) -> u8 {
        let map = if window {
            if self.lcdc & 0x40 != 0 {
                0x1c00
            } else {
                0x1800
            }
        } else if self.lcdc & 8 != 0 {
            0x1c00
        } else {
            0x1800
        };
        let tile = self.vram[map + (y as usize / 8) * 32 + x as usize / 8];
        let base = if self.lcdc & 0x10 != 0 {
            tile as usize * 16
        } else {
            (0x1000i32 + (tile as i8 as i32) * 16) as usize
        };
        tile_pixel(&self.vram, base, x & 7, y & 7)
    }

    fn render_sprites(&mut self, y: usize, bg: &[u8; WIDTH]) {
        let height = if self.lcdc & 4 != 0 { 16 } else { 8 };
        let mut sprites: Vec<(usize, u8)> = (0..40)
            .filter_map(|i| {
                let sy = self.oam[i * 4] as i16 - 16;
                ((y as i16) >= sy && (y as i16) < sy + height).then_some((i, self.oam[i * 4 + 1]))
            })
            .take(10)
            .collect();
        sprites.sort_by_key(|&(i, x)| (x, i));
        for &(i, _) in sprites.iter().rev() {
            let sy = self.oam[i * 4] as i16 - 16;
            let sx = self.oam[i * 4 + 1] as i16 - 8;
            let mut tile = self.oam[i * 4 + 2];
            let flags = self.oam[i * 4 + 3];
            let mut row = (y as i16 - sy) as u8;
            if flags & 0x40 != 0 {
                row = height as u8 - 1 - row;
            }
            if height == 16 {
                tile &= 0xfe;
            }
            let base = tile as usize * 16 + row as usize * 2;
            for col in 0..8u8 {
                let x = sx + col as i16;
                if !(0..WIDTH as i16).contains(&x) {
                    continue;
                }
                let bitx = if flags & 0x20 != 0 { 7 - col } else { col };
                let c = tile_pixel(&self.vram, base, bitx, 0);
                if c != 0 && !(flags & 0x80 != 0 && bg[x as usize] != 0) {
                    let pal = if flags & 0x10 != 0 {
                        self.obp1
                    } else {
                        self.obp0
                    };
                    self.framebuffer[y * WIDTH + x as usize] = palette(pal, c);
                }
            }
        }
    }
}

fn tile_pixel(vram: &[u8], base: usize, x: u8, y: u8) -> u8 {
    let i = base + y as usize * 2;
    let bit = 7 - x;
    ((vram[i] >> bit) & 1) | (((vram[i + 1] >> bit) & 1) << 1)
}
fn palette(p: u8, color: u8) -> u8 {
    (p >> (color * 2)) & 3
}
