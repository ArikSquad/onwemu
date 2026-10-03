//! PSP Graphics Engine command decoding and software rendering.
//!
//! The GPU crate keeps GE register state and command-list flow independent from
//! the host renderer. Headless runs rasterize into guest EDRAM so tests and
//! framebuffer reads stay deterministic; graphical runs can consume the
//! decoded [`RenderCommand`] stream in an SDL/Vulkan backend.

use psp_memory::{GuestMemory, Memory, MemoryFault};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

const MAX_LIST_WORDS: usize = 1_000_000;

#[derive(Debug, Error)]
/// Failures while decoding a GE list or touching its guest surfaces.
pub enum GpuError {
    #[error(transparent)]
    /// A guest-memory access needed by the GE failed.
    Memory(#[from] MemoryFault),
    #[error("GE command list at 0x{start:08x} did not terminate")]
    /// The list reached the safety limit without an END command.
    UnterminatedList {
        /// Guest address where list execution began.
        start: u32,
    },
    #[error("GE RET at 0x{pc:08x} has no matching CALL")]
    /// A return command was encountered without a matching call.
    EmptyCallStack {
        /// Guest address of the unmatched RET command.
        pc: u32,
    },
    #[error("unsupported PSP framebuffer pixel format {0}")]
    /// The software renderer does not know this framebuffer format.
    PixelFormat(u32),
    #[error("PSP framebuffer dimensions overflow host size")]
    /// The requested framebuffer would overflow a host allocation.
    FramebufferSize,
    #[error("PSP texture dimensions are too large for a host upload")]
    /// The requested texture is too large for a host allocation.
    TextureSize,
    #[error(
        "GE primitive at 0x{pc:08x} failed (framebuffer=0x{framebuffer:08x}, stride=0x{stride:08x}, texture=0x{texture:08x}): {source}"
    )]
    /// A primitive failed while reading its framebuffer or texture data.
    PrimitiveMemory {
        /// GE program counter of the primitive command.
        pc: u32,
        /// Guest framebuffer address used by the primitive.
        framebuffer: u32,
        /// Guest framebuffer stride.
        stride: u32,
        /// Guest texture address used by the primitive.
        texture: u32,
        #[source]
        /// Underlying memory fault that stopped the primitive.
        source: MemoryFault,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
/// Summary of one synchronously executed GE command list.
pub struct ListResult {
    /// Number of command words consumed.
    pub words: usize,
    /// Whether execution stopped at the requested stall address.
    pub stalled: bool,
    /// Whether the list reached an END command.
    pub finished: bool,
    /// Guest address of the final command, when one was recorded.
    pub finish_pc: u32,
    /// Token carried by the final FINISH command.
    pub finish_token: u32,
    /// Signal (0x0e) events encountered while executing the list. The PSP
    /// raises a GE interrupt for each event so the guest callback can run
    /// before the list continues.
    pub signals: Vec<SignalEvent>,
}

/// A GE signal event decoded from a display list.
///
/// `behavior` mirrors the PSP `SceGeSignalBehavior` values:
/// 0x01 suspends the list until the handler returns, 0x02 continues
/// immediately, 0x03 pauses at finish, and the remaining forms (0x08 sync,
/// 0x10+ jumps/calls/returns, 0x20+ address updates, 0xf0+ breaks) are
/// synchronization or flow-control forms that never run guest code.
/// `token` is the 16-bit value passed to the handler in a0.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// One signal callback opportunity observed while a list is running.
pub struct SignalEvent {
    /// Guest address of the SIGNAL command.
    pub pc: u32,
    /// PSP signal behavior value.
    pub behavior: u32,
    /// 16-bit token passed to the guest callback.
    pub token: u32,
}

/// GE state, command-list execution, and software framebuffer rendering.
pub struct Gpu {
    /// Last argument written to each of the 256 GE registers.
    pub registers: [u32; 256],
    /// Number of primitive commands submitted.
    pub draw_calls: u64,
    /// Number of command lists that reached completion.
    pub completed_lists: u64,
    /// Latched state from the most recently submitted primitive.
    pub last_primitive: Option<PrimitiveState>,
    /// U, V, X, and Y values for the first two vertices of the most recent draw.
    pub last_vertex_probe: Option<[i32; 8]>,
    /// Number of color-buffer pixels written by the most recent command list.
    /// This distinguishes a list that ran from one that produced a frame while
    /// bringing up a backend.
    pub last_render_pixels: u64,
    matrices: MatrixState,
    base_address: u32,
    offset_address: u32,
    vertex_address: Option<u32>,
    index_address: Option<u32>,
    hardware_rendering: bool,
    render_commands: Vec<RenderCommand>,
    /// A GE frame commonly draws the same texture from many command lists. Keep
    /// decoded pixels across lists and validate the source pages with memory's
    /// write generations so guest texture uploads still invalidate the cache
    /// without re-decoding unchanged surfaces.
    texture_snapshots: HashMap<Texture, TextureSnapshot>,
}

struct TextureSnapshot {
    pixels: Arc<[u8]>,
    content_hash: u64,
    generations: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, Debug)]
enum MatrixKind {
    World,
    View,
    Projection,
    Texture,
}

#[derive(Clone, Copy, Debug)]
struct MatrixState {
    world: [f32; 12],
    view: [f32; 12],
    projection: [f32; 16],
    texture: [f32; 12],
    bones: [[f32; 12]; 8],
    bone_cursor: usize,
    active: MatrixKind,
    cursor: usize,
}

impl Default for MatrixState {
    fn default() -> Self {
        Self {
            world: identity_matrix_43(),
            view: identity_matrix_43(),
            projection: identity_matrix_44(),
            texture: identity_matrix_43(),
            bones: [[0.0; 12]; 8],
            bone_cursor: 0,
            active: MatrixKind::World,
            cursor: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// GE state captured when a primitive command is submitted.
pub struct PrimitiveState {
    /// Packed primitive argument from the GE command.
    pub argument: u32,
    /// Guest address of the vertex stream.
    pub vertex_address: u32,
    /// PSP vertex format bitfield.
    pub vertex_type: u32,
    /// Guest EDRAM framebuffer address.
    pub framebuffer_address: u32,
    /// Framebuffer row stride in guest words.
    pub framebuffer_width: u32,
    /// PSP framebuffer pixel format.
    pub framebuffer_format: u32,
    /// Clear flags carried by the current primitive.
    pub clear_mode: u32,
    /// Packed minimum scissor coordinates.
    pub scissor_min: u32,
    /// Packed maximum scissor coordinates.
    pub scissor_max: u32,
    /// Guest texture address.
    pub texture_address: u32,
    /// Texture row stride.
    pub texture_width: u32,
    /// Texture dimensions encoding.
    pub texture_size: u32,
    /// PSP texture pixel format.
    pub texture_format: u32,
    /// Guest color-look-up-table address.
    pub clut_address: u32,
    /// Color-look-up-table format.
    pub clut_format: u32,
    /// Texture function and color-combine mode.
    pub texture_function: u32,
}

/// A vertex decoded from a PSP GE list for a hardware renderer.
///
/// coordinates remain in psp screen space here.  keeping the conversion to
/// host clip space in the backend means the same decoded stream can be sent
/// to sdl gpu, vulkan, or another host api without changing ge semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
/// A decoded vertex in PSP screen-space coordinates.
pub struct RenderVertex {
    /// Packed RGBA color.
    pub color: u32,
    /// Texture U coordinate in PSP fixed-point units.
    pub u: i32,
    /// Texture V coordinate in PSP fixed-point units.
    pub v: i32,
    /// Reciprocal homogeneous W used for perspective interpolation.
    pub inv_w: f32,
    /// Screen-space X coordinate.
    pub x: i32,
    /// Screen-space Y coordinate.
    pub y: i32,
    /// Depth value in PSP framebuffer units.
    pub depth: u16,
}

#[derive(Clone, Debug, PartialEq)]
/// A host-uploadable texture snapshot and its PSP sampling state.
pub struct RenderTexture {
    /// Guest address used as the stable cache key for the host texture.
    pub address: u32,
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
    /// Decoded RGBA8 pixels.
    pub pixels: Arc<[u8]>,
    /// FNV-1a hash of the decoded pixels used for cheap cache comparisons.
    pub content_hash: u64,
    /// Whether the S coordinate is clamped instead of repeated.
    pub clamp_s: bool,
    /// Whether the T coordinate is clamped instead of repeated.
    pub clamp_t: bool,
    /// PSP texture function used by the draw.
    pub function: u32,
    /// Texture environment color.
    pub environment_color: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
/// One decoded draw or clear command for a host renderer.
pub struct RenderCommand {
    /// PSP primitive type.
    pub primitive: u32,
    /// Vertices decoded for this primitive.
    pub vertices: Vec<RenderVertex>,
    /// Optional texture snapshot used by the primitive.
    pub texture: Option<RenderTexture>,
    /// Whether texture sampling is enabled.
    pub textured: bool,
    /// Whether vertex colors should be interpolated.
    pub gouraud: bool,
    /// PSP clear flags.
    pub clear_mode: u32,
    /// Guest EDRAM surface selected by FRAMEBUF.
    pub framebuffer_address: u32,
    /// Framebuffer row stride.
    pub framebuffer_width: u32,
    /// Framebuffer pixel format.
    pub framebuffer_format: u32,
    /// Guest EDRAM depth surface selected by ZBUF.
    pub depth_address: u32,
    /// Depth-buffer row stride.
    pub depth_width: u32,
    /// Depth value carried by a PSP clear primitive.
    pub clear_depth: u16,
    /// Scissor rectangle as left, top, right, bottom.
    pub scissor: [u32; 4],
    /// Whether depth comparisons are enabled.
    pub depth_test: bool,
    /// Whether passing depth tests writes the depth buffer.
    pub depth_write: bool,
    /// PSP depth comparison function.
    pub depth_function: u32,
    /// Whether color blending is enabled.
    pub blend: bool,
    /// Packed source/destination blend factors.
    pub blend_state: u32,
    /// Fixed source blend color.
    pub blend_fixed_a: u32,
    /// Fixed destination blend color.
    pub blend_fixed_b: u32,
    /// Whether alpha testing is enabled.
    pub alpha_test: bool,
    /// PSP alpha comparison function.
    pub alpha_function: u32,
    /// Alpha value compared against each fragment.
    pub alpha_reference: u32,
    /// Alpha comparison mask.
    pub alpha_mask: u32,
    /// Whether color testing is enabled.
    pub color_test: bool,
    /// PSP color comparison function.
    pub color_function: u32,
    /// Color value compared against each fragment.
    pub color_reference: u32,
    /// Color comparison mask.
    pub color_test_mask: u32,
    /// RGB write mask; set bits are preserved.
    pub color_write_mask_rgb: u32,
    /// Alpha write mask; set bits are preserved.
    pub color_write_mask_alpha: u32,
}

impl Default for Gpu {
    fn default() -> Self {
        Self {
            registers: [0; 256],
            draw_calls: 0,
            completed_lists: 0,
            last_primitive: None,
            last_vertex_probe: None,
            last_render_pixels: 0,
            matrices: MatrixState::default(),
            base_address: 0,
            offset_address: 0,
            vertex_address: None,
            index_address: None,
            hardware_rendering: false,
            render_commands: Vec::new(),
            texture_snapshots: HashMap::new(),
        }
    }
}

impl Gpu {
    /// Select the renderer used after GE vertices have been decoded.
    ///
    /// Headless runs keep the software renderer because it makes framebuffer
    /// contents observable to tests and diagnostics. A graphical run uses the
    /// decoded command stream and leaves rasterization to the host GPU.
    pub fn set_hardware_rendering(&mut self, enabled: bool) {
        self.hardware_rendering = enabled;
        self.render_commands.clear();
        self.texture_snapshots.clear();
    }

    /// Return whether host hardware rendering is enabled.
    pub fn hardware_rendering(&self) -> bool {
        self.hardware_rendering
    }

    /// Take and clear the decoded commands accumulated since the last drain.
    pub fn take_render_commands(&mut self) -> Vec<RenderCommand> {
        std::mem::take(&mut self.render_commands)
    }

    /// Submit one packed GE command word and update the latched register state.
    pub fn submit(&mut self, word: u32) {
        let command = (word >> 24) as usize;
        let argument = word & 0x00ff_ffff;
        self.registers[command] = argument;
        self.update_matrix(command, argument);
        // prim, bezier and spline all emit geometry.
        if matches!(command, 0x04..=0x06) {
            self.draw_calls += 1;
            self.last_primitive = Some(PrimitiveState {
                argument,
                vertex_address: self.registers[0x01],
                vertex_type: self.registers[0x12],
                framebuffer_address: self.registers[0x9c],
                framebuffer_width: self.registers[0x9d],
                framebuffer_format: self.registers[0xd2],
                clear_mode: self.registers[0xd3],
                scissor_min: self.registers[0xd4],
                scissor_max: self.registers[0xd5],
                texture_address: self.registers[0xa0],
                texture_width: self.registers[0xa8],
                texture_size: self.registers[0xb8],
                texture_format: self.registers[0xc3],
                clut_address: (self.registers[0xb0] & 0x00ff_fff0)
                    | ((self.registers[0xb1] << 8) & 0x0f00_0000),
                clut_format: self.registers[0xc5],
                texture_function: self.registers[0xc9],
            });
        }
    }

    fn update_matrix(&mut self, command: usize, argument: u32) {
        match command {
            0x2a => self.matrices.bone_cursor = argument as usize & 0x7f,
            0x2b => {
                let cursor = self.matrices.bone_cursor;
                if cursor < 96 {
                    self.matrices.bones[cursor / 12][cursor % 12] = float24(argument);
                }
                self.matrices.bone_cursor = (cursor + 1) & 0x7f;
            }
            0x3a => {
                self.matrices.active = MatrixKind::World;
                self.matrices.cursor = (argument as usize).min(11);
            }
            0x3b => self.write_matrix_value(argument),
            0x3c => {
                self.matrices.active = MatrixKind::View;
                self.matrices.cursor = (argument as usize).min(11);
            }
            0x3d => self.write_matrix_value(argument),
            0x3e => {
                self.matrices.active = MatrixKind::Projection;
                self.matrices.cursor = (argument as usize).min(15);
            }
            0x3f => self.write_matrix_value(argument),
            0x40 => {
                self.matrices.active = MatrixKind::Texture;
                self.matrices.cursor = (argument as usize).min(11);
            }
            0x41 => self.write_matrix_value(argument),
            _ => {}
        }
    }

    fn write_matrix_value(&mut self, argument: u32) {
        let value = float24(argument);
        match self.matrices.active {
            MatrixKind::World => {
                self.matrices.world[self.matrices.cursor] = value;
                self.matrices.cursor = (self.matrices.cursor + 1) % 12;
            }
            MatrixKind::View => {
                self.matrices.view[self.matrices.cursor] = value;
                self.matrices.cursor = (self.matrices.cursor + 1) % 12;
            }
            MatrixKind::Projection => {
                self.matrices.projection[self.matrices.cursor] = value;
                self.matrices.cursor = (self.matrices.cursor + 1) % 16;
            }
            MatrixKind::Texture => {
                self.matrices.texture[self.matrices.cursor] = value;
                self.matrices.cursor = (self.matrices.cursor + 1) % 12;
            }
        }
    }

    /// Execute a GE list synchronously.
    ///
    /// This models command flow and register state. A renderer can consume the
    /// resulting registers, draw calls, or decoded command stream afterward.
    pub fn execute_list(
        &mut self,
        memory: &mut Memory,
        start: u32,
        stall: u32,
    ) -> Result<ListResult, GpuError> {
        let mut pc = start;
        let mut calls = Vec::new();
        let mut result = ListResult::default();
        self.last_render_pixels = 0;
        // a newly enqueued psp list starts with its own origin. base is ge
        // state and persists until another base command changes it.
        self.offset_address = 0;
        let mut previous_command = 0u8;
        for _ in 0..MAX_LIST_WORDS {
            if stall != 0 && pc == stall {
                result.stalled = true;
                return Ok(result);
            }
            let command_pc = pc;
            let word = memory.read_u32(pc)?;
            pc = pc.wrapping_add(4);
            result.words += 1;
            let command = (word >> 24) as u8;
            let argument = word & 0x00ff_ffff;
            self.submit(word);
            if trace_draw(self.draw_calls) {
                eprintln!(
                    "GE_CMD draw={} pc={command_pc:#010x} word={word:#010x}",
                    self.draw_calls
                );
            }
            let draw_result = match command {
                0x04 => self.draw_primitive(memory, argument),
                0x05 => self.draw_curve(memory, argument, true),
                0x06 => self.draw_curve(memory, argument, false),
                _ => Ok(()),
            };
            if let Err(source) = draw_result {
                return Err(match source {
                    GpuError::Memory(source) => GpuError::PrimitiveMemory {
                        pc: command_pc,
                        framebuffer: self.registers[0x9c],
                        stride: self.registers[0x9d],
                        texture: self.registers[0xa0],
                        source,
                    },
                    source => source,
                });
            }
            match command {
                0x01 => self.vertex_address = Some(self.relative_address(argument)),
                0x02 => self.index_address = Some(self.relative_address(argument)),
                0x08 | 0x09 => {
                    // jump and bjump use the same relative-address packing;
                    // bjump is treated as not-bounded when no bbox engine is
                    // present, which is the safe behavior for this renderer.
                    pc = self.relative_address(argument & 0x00ff_fffc);
                }
                0x0a => {
                    calls.push((pc, self.offset_address)); // CALL
                    pc = self.relative_address(argument & 0x00ff_fffc);
                }
                0x0b => {
                    let (return_pc, offset_address) = calls
                        .pop()
                        .ok_or(GpuError::EmptyCallStack { pc: command_pc })?;
                    pc = return_pc;
                    self.offset_address = offset_address;
                }
                0x0e => {
                    // signal.  the argument packs a behavior in bits 16-23 and
                    // a 16-bit token for the guest signal handler.  only the
                    // handler forms (0x01 suspend, 0x02 continue, 0x03 pause)
                    // run guest code; the remaining behaviors are
                    // synchronization (0x08), flow control (0x10+), address
                    // updates (0x20+), or breaks (0xf0+).  the frontend
                    // delivers the recorded events through the
                    // scegesetcallback handlers.
                    let behavior = (argument >> 16) & 0xff;
                    if (0x01..=0x03).contains(&behavior) {
                        result.signals.push(SignalEvent {
                            pc: command_pc,
                            behavior,
                            token: argument & 0xffff,
                        });
                    }
                }
                0x0c => {
                    // end terminates a list, except immediately after signal:
                    // the pair carries enddata and execution continues with
                    // the next word.  finish immediately before end is the
                    // normal completion sequence, but end alone is valid.
                    if previous_command == 0x0e {
                        previous_command = command;
                        continue;
                    }
                    result.finished = self.registers[0x0f] != 0 || result.words != 0;
                    result.finish_token = self.registers[0x0f] & 0xffff;
                    result.finish_pc = command_pc.saturating_sub(4);
                    self.completed_lists += 1;
                    return Ok(result);
                }
                0x10 => self.base_address = argument, // BASE
                0x13 => self.offset_address = argument << 8, // OFFSETADDR
                0x14 => self.offset_address = command_pc, // ORIGIN
                _ => {}
            }
            previous_command = command;
        }
        Err(GpuError::UnterminatedList { start })
    }

    fn draw_primitive(&mut self, memory: &mut Memory, argument: u32) -> Result<(), GpuError> {
        let vertex_type = self.registers[0x12];
        let count = (argument & 0xffff) as usize;
        let primitive = (argument >> 16) & 7;
        let position_format = (vertex_type >> 7) & 3;
        if position_format == 0 {
            return Ok(());
        }

        let vertex_address = self
            .vertex_address
            .unwrap_or_else(|| self.relative_address(self.registers[0x01]));
        let texture_enabled = self.registers[0x1e] & 1 != 0;
        let texture = self.current_texture();
        let Some(layout) = VertexLayout::new(vertex_type) else {
            return Ok(());
        };
        let index_address = self
            .index_address
            .unwrap_or_else(|| self.relative_address(self.registers[0x02]));
        let mut vertices = Vec::with_capacity(count);
        for ordinal in 0..count {
            let index = read_vertex_index(memory, index_address, layout.index_format, ordinal)?;
            let address = vertex_address.wrapping_add(index * layout.stride_bytes as u32);
            if let Some(vertex) = self.read_vertex(memory, address, layout, texture)? {
                vertices.push(vertex);
            }
        }
        if trace_draw(self.draw_calls) {
            eprintln!(
                "GE_DRAW draw={} vtype={vertex_type:#x} primitive={primitive} addr={vertex_address:#x} layout={layout:?} regs={:?} matrices={:?} vertices={:?}",
                self.draw_calls,
                self.registers,
                self.matrices,
                &vertices[..vertices.len().min(8)]
            );
        }
        self.last_vertex_probe = vertices
            .first()
            .zip(vertices.get(1))
            .map(|(first, second)| {
                [
                    first.u, first.v, first.x, first.y, second.u, second.v, second.x, second.y,
                ]
            });
        let textured =
            texture_enabled && layout.texture_format != 0 && matches!(texture.format, 0..=10);
        let gouraud = self.registers[0x50] & 1 != 0;
        self.render_or_record(memory, &vertices, primitive, texture, textured, gouraud)?;
        if layout.index_format == 0 {
            let bytes_read = count.saturating_mul(layout.stride_bytes);
            self.vertex_address = Some(vertex_address.wrapping_add(bytes_read as u32));
        } else {
            let index_bytes = match layout.index_format {
                1 => 1,
                2 => 2,
                3 => 4,
                _ => unreachable!(),
            };
            self.index_address =
                Some(index_address.wrapping_add((count.saturating_mul(index_bytes)) as u32));
        }
        Ok(())
    }

    fn draw_curve(
        &mut self,
        memory: &mut Memory,
        argument: u32,
        bezier: bool,
    ) -> Result<(), GpuError> {
        let vertex_type = self.registers[0x12];
        let position_format = (vertex_type >> 7) & 3;
        let points_u = (argument & 0xff) as usize;
        let points_v = ((argument >> 8) & 0xff) as usize;
        let point_count = points_u.saturating_mul(points_v);
        if position_format == 0 || points_u < 4 || points_v < 4 {
            return Ok(());
        }
        let Some(layout) = VertexLayout::new(vertex_type) else {
            return Ok(());
        };
        let vertex_address = self
            .vertex_address
            .unwrap_or_else(|| self.relative_address(self.registers[0x01]));
        let index_address = self
            .index_address
            .unwrap_or_else(|| self.relative_address(self.registers[0x02]));
        let texture = self.current_texture();
        let texture_enabled = self.registers[0x1e] & 1 != 0;
        let textured =
            texture_enabled && layout.texture_format != 0 && matches!(texture.format, 0..=10);
        let gouraud = self.registers[0x50] & 1 != 0;
        let mut controls = Vec::with_capacity(point_count);
        for ordinal in 0..point_count {
            let index = read_vertex_index(memory, index_address, layout.index_format, ordinal)?;
            let address = vertex_address.wrapping_add(index * layout.stride_bytes as u32);
            controls.push(self.read_vertex(memory, address, layout, texture)?);
        }
        let mut probe_vertices = controls.iter().flatten();
        self.last_vertex_probe =
            probe_vertices
                .next()
                .zip(probe_vertices.next())
                .map(|(first, second)| {
                    [
                        first.u, first.v, first.x, first.y, second.u, second.v, second.x, second.y,
                    ]
                });

        let patch_count_u = if bezier {
            (points_u - 1) / 3
        } else {
            points_u - 3
        };
        let patch_count_v = if bezier {
            (points_v - 1) / 3
        } else {
            points_v - 3
        };
        let tess_u = ((self.registers[0x36] & 0x7f) as usize).clamp(1, 32);
        let tess_v = (((self.registers[0x36] >> 8) & 0x7f) as usize).clamp(1, 32);
        let patch_primitive = self.registers[0x37] & 3;
        if patch_primitive == 0 {
            for patch_v in 0..patch_count_v {
                for patch_u in 0..patch_count_u {
                    let base_u = if bezier { patch_u * 3 } else { patch_u };
                    let base_v = if bezier { patch_v * 3 } else { patch_v };
                    let mut vertices = Vec::with_capacity((tess_u + 1) * (tess_v + 1));
                    for tile_v in 0..=tess_v {
                        let v = tile_v as f32 / tess_v as f32;
                        let weights_v = if bezier {
                            bezier_weights(v)
                        } else {
                            spline_weights(v, patch_v, patch_count_v, (argument >> 18) & 3)
                        };
                        for tile_u in 0..=tess_u {
                            let u = tile_u as f32 / tess_u as f32;
                            let weights_u = if bezier {
                                bezier_weights(u)
                            } else {
                                spline_weights(u, patch_u, patch_count_u, (argument >> 16) & 3)
                            };
                            let Some(vertex) = interpolate_patch_vertex(
                                &controls, points_u, base_u, base_v, weights_u, weights_v,
                            ) else {
                                vertices.clear();
                                break;
                            };
                            vertices.push(vertex);
                        }
                        if vertices.is_empty() {
                            break;
                        }
                    }
                    if vertices.len() == (tess_u + 1) * (tess_v + 1) {
                        let mut triangles = Vec::with_capacity(tess_u * tess_v * 6);
                        for tile_v in 0..tess_v {
                            for tile_u in 0..tess_u {
                                let top_left = tile_v * (tess_u + 1) + tile_u;
                                let top_right = top_left + 1;
                                let bottom_left = top_left + tess_u + 1;
                                let bottom_right = bottom_left + 1;
                                triangles.extend_from_slice(&[
                                    vertices[top_left],
                                    vertices[bottom_left],
                                    vertices[top_right],
                                    vertices[top_right],
                                    vertices[bottom_left],
                                    vertices[bottom_right],
                                ]);
                            }
                        }
                        self.render_or_record(memory, &triangles, 3, texture, textured, gouraud)?;
                    }
                }
            }
        }
        if layout.index_format == 0 {
            self.vertex_address = Some(
                vertex_address.wrapping_add(point_count.saturating_mul(layout.stride_bytes) as u32),
            );
        } else {
            let index_bytes = match layout.index_format {
                1 => 1,
                2 => 2,
                3 => 4,
                _ => unreachable!(),
            };
            self.index_address =
                Some(index_address.wrapping_add(point_count.saturating_mul(index_bytes) as u32));
        }
        Ok(())
    }

    fn current_texture(&self) -> Texture {
        Texture {
            address: edram_address(texture_address(self.registers[0xa0], self.registers[0xa8])),
            stride: self.registers[0xa8] & 0xffff,
            width: texture_dimension(self.registers[0xb8], false),
            height: texture_dimension(self.registers[0xb8], true),
            format: self.registers[0xc3] & 0xf,
            clut_address_low: self.registers[0xb0],
            clut_address_high: self.registers[0xb1],
            clut_format: self.registers[0xc5],
            function: self.registers[0xc9],
            environment_color: self.registers[0xca],
            clamp_s: self.registers[0xc7] & 1 != 0,
            clamp_t: self.registers[0xc7] & 0x100 != 0,
            swizzled: self.registers[0xc2] & 1 != 0,
        }
    }

    fn render_or_record(
        &mut self,
        memory: &mut Memory,
        vertices: &[Vertex],
        primitive: u32,
        texture: Texture,
        textured: bool,
        gouraud: bool,
    ) -> Result<(), GpuError> {
        // the ge can execute setup/utility lists before a display target is
        // selected. they are not drawable work; forwarding them to the host
        // renderer would incorrectly modify the persistent sdl render target.
        if self.registers[0x9d] & 0x0000_07fc == 0 {
            return Ok(());
        }
        let textured = textured && self.registers[0xd3] & 1 == 0;
        let prepared;
        let (vertices, primitive) = if matches!(primitive, 3..=5) && self.registers[0xd3] & 1 == 0 {
            prepared = self.prepare_triangles(vertices, primitive, gouraud);
            (prepared.as_slice(), 3)
        } else {
            (vertices, primitive)
        };
        if vertices.is_empty() {
            return Ok(());
        }
        if self.hardware_rendering {
            let command = self
                .record_render_command(memory, vertices, primitive, texture, textured, gouraud)?;
            if trace_draw(self.draw_calls) {
                let (min_depth, max_depth) = vertices
                    .iter()
                    .fold((u16::MAX, u16::MIN), |(min, max), vertex| {
                        (min.min(vertex.depth), max.max(vertex.depth))
                    });
                eprintln!(
                    "GE_HW draw={} fb=0x{:08x}/{} depth=0x{:08x}/{} clear=0x{:x} \
                     ztest={} zfunc={} zwrite={} blend={} blend_state=0x{:x} \
                     alphatest={} scissor=[{},{},{},{}] verts={} depth_range=[{},{}]",
                    self.draw_calls,
                    command.framebuffer_address,
                    command.framebuffer_width,
                    command.depth_address,
                    command.depth_width,
                    command.clear_mode,
                    command.depth_test,
                    command.depth_function,
                    command.depth_write,
                    command.blend,
                    command.blend_state,
                    command.alpha_test,
                    command.scissor[0],
                    command.scissor[1],
                    command.scissor[2],
                    command.scissor[3],
                    command.vertices.len(),
                    min_depth,
                    max_depth,
                );
            }
            self.render_commands.push(command);
            Ok(())
        } else {
            // the cpu path remains a deliberate reference renderer for
            // headless tests and guest framebuffer readback.
            self.render_vertices(memory, vertices, primitive, texture, textured, gouraud)
        }
    }

    fn prepare_triangles(&self, vertices: &[Vertex], primitive: u32, gouraud: bool) -> Vec<Vertex> {
        let capacity = if primitive == 3 {
            vertices.len()
        } else {
            vertices.len().saturating_sub(2) * 3
        };
        let mut output = Vec::with_capacity(capacity);
        let mut emit = |triangle: [Vertex; 3]| {
            // a triangle clipped against one plane has at most four vertices.
            let mut polygon = [triangle[0]; 4];
            polygon[..3].copy_from_slice(&triangle);
            let mut polygon_len = 3;
            if triangle.iter().any(|v| v.outside_range) {
                return;
            }
            if let Some(clip) = triangle[0].clip {
                let clips = [clip, triangle[1].clip.unwrap(), triangle[2].clip.unwrap()];
                if clips.iter().all(|p| p[3] < 0.0) {
                    return;
                }
                let z = clips.map(|p| p[2] / p[3]);
                const Z_BOUND: f32 = 1.000_030_5;
                if z.iter().all(|z| *z >= Z_BOUND)
                    || z.iter().all(|z| *z <= -Z_BOUND)
                    || (self.registers[0x1c] & 1 == 0 && z.iter().any(|z| z.abs() >= Z_BOUND))
                {
                    return;
                }
                // the ge clips the near plane in homogeneous coordinates.
                // interpolate attributes there before the perspective divide.
                if clips.iter().any(|p| p[2] + p[3] < 0.0) {
                    polygon_len = 0;
                    let mut previous = triangle[2];
                    let mut previous_distance = clips[2][2] + clips[2][3];
                    for current in triangle {
                        let p = current.clip.unwrap();
                        let distance = p[2] + p[3];
                        if (distance < 0.0) != (previous_distance < 0.0) {
                            let t = previous_distance / (previous_distance - distance);
                            let a = previous.clip.unwrap();
                            let clip = std::array::from_fn(|i| a[i] + (p[i] - a[i]) * t);
                            let Some((x, y, depth, inv_w, outside_range)) =
                                self.clip_to_screen(clip, true)
                            else {
                                return;
                            };
                            if outside_range {
                                return;
                            }
                            polygon[polygon_len] = Vertex {
                                clip: Some(clip),
                                outside_range,
                                x,
                                y,
                                depth,
                                inv_w,
                                u: (previous.u as f32 + (current.u as f32 - previous.u as f32) * t)
                                    .round() as i32,
                                v: (previous.v as f32 + (current.v as f32 - previous.v as f32) * t)
                                    .round() as i32,
                                color: lerp_color(
                                    previous.color,
                                    current.color,
                                    0,
                                    1.0 - t,
                                    t,
                                    0.0,
                                ),
                            };
                            polygon_len += 1;
                        }
                        if distance >= 0.0 {
                            polygon[polygon_len] = current;
                            polygon_len += 1;
                        }
                        previous = current;
                        previous_distance = distance;
                    }
                }
            }
            for i in 2..polygon_len {
                let mut clipped = [polygon[0], polygon[i - 1], polygon[i]];
                // i128: through-mode triangles skip range checks (only
                // non-finite coordinates are flagged), so arbitrary i32
                // coordinates can reach this area computation and overflow
                // i64 products into horizontal banding streaks.
                let area = (clipped[1].x as i128 - clipped[0].x as i128)
                    * (clipped[2].y as i128 - clipped[0].y as i128)
                    - (clipped[1].y as i128 - clipped[0].y as i128)
                        * (clipped[2].x as i128 - clipped[0].x as i128);
                if area == 0
                    || (self.registers[0x1d] & 1 != 0
                        && (area > 0) != (self.registers[0x9b] & 1 != 0))
                {
                    continue;
                }
                if !gouraud {
                    for v in &mut clipped {
                        v.color = triangle[2].color;
                    }
                }
                output.extend(clipped);
            }
        };
        match primitive {
            3 => {
                for v in vertices.as_chunks::<3>().0.iter() {
                    emit([v[0], v[1], v[2]]);
                }
            }
            4 => {
                for (i, v) in vertices.windows(3).enumerate() {
                    emit(if i & 1 == 0 {
                        [v[0], v[1], v[2]]
                    } else {
                        [v[1], v[0], v[2]]
                    });
                }
            }
            5 => {
                for v in vertices.get(1..).unwrap_or_default().windows(2) {
                    emit([vertices[0], v[0], v[1]]);
                }
            }
            _ => {}
        }
        output
    }

    fn record_render_command(
        &mut self,
        memory: &Memory,
        vertices: &[Vertex],
        primitive: u32,
        texture: Texture,
        textured: bool,
        gouraud: bool,
    ) -> Result<RenderCommand, GpuError> {
        let framebuffer_address = edram_address(self.registers[0x9c] & 0x001f_fff0);
        let framebuffer_width = self.registers[0x9d] & 0x0000_07fc;
        let depth_address = edram_address(self.registers[0x9e] & 0x001f_fff0);
        let scissor_min = self.registers[0xd4];
        let scissor_max = self.registers[0xd5];
        let texture = if textured {
            Some(self.snapshot_texture(memory, texture)?)
        } else {
            None
        };
        let vertices = vertices
            .iter()
            .copied()
            .map(|vertex| RenderVertex {
                color: vertex.color,
                u: vertex.u,
                v: vertex.v,
                inv_w: vertex.inv_w,
                x: vertex.x,
                y: vertex.y,
                depth: vertex.depth,
            })
            .collect::<Vec<_>>();
        let clear_depth = vertices.first().map_or(u16::MAX, |vertex| vertex.depth);
        Ok(RenderCommand {
            primitive,
            vertices,
            texture,
            textured,
            gouraud,
            clear_mode: self.registers[0xd3],
            framebuffer_address,
            framebuffer_width,
            framebuffer_format: self.registers[0xd2] & 3,
            depth_address,
            depth_width: self.registers[0x9f] & 0x0000_07fc,
            clear_depth,
            scissor: [
                scissor_min & 0x3ff,
                (scissor_min >> 10) & 0x3ff,
                scissor_max & 0x3ff,
                (scissor_max >> 10) & 0x3ff,
            ],
            depth_test: self.registers[0x23] & 1 != 0,
            depth_write: self.registers[0xe7] & 1 == 0,
            depth_function: self.registers[0xde] & 7,
            blend: self.registers[0x21] & 1 != 0,
            blend_state: self.registers[0xdf],
            blend_fixed_a: self.registers[0xe0] & 0x00ff_ffff,
            blend_fixed_b: self.registers[0xe1] & 0x00ff_ffff,
            alpha_test: self.registers[0x22] & 1 != 0,
            alpha_function: self.registers[0xdb] & 7,
            alpha_reference: (self.registers[0xdb] >> 8) & 0xff,
            alpha_mask: (self.registers[0xdb] >> 16) & 0xff,
            color_test: self.registers[0x27] & 1 != 0,
            color_function: self.registers[0xd8] & 3,
            color_reference: self.registers[0xd9] & 0x00ff_ffff,
            color_test_mask: self.registers[0xda] & 0x00ff_ffff,
            color_write_mask_rgb: self.registers[0xe8] & 0x00ff_ffff,
            color_write_mask_alpha: self.registers[0xe9] & 0xff,
        })
    }

    fn snapshot_texture(
        &mut self,
        memory: &Memory,
        texture: Texture,
    ) -> Result<RenderTexture, GpuError> {
        let key = Texture {
            function: 0,
            environment_color: 0,
            clamp_s: false,
            clamp_t: false,
            ..texture
        };
        if let Some(snapshot) = self.texture_snapshots.get(&key)
            && snapshot
                .generations
                .iter()
                .all(|&(page, generation)| memory.page_generation(page) == generation)
        {
            return Ok(render_texture(
                texture,
                Arc::clone(&snapshot.pixels),
                snapshot.content_hash,
            ));
        }
        let snapshot = snapshot_texture(memory, texture)?;
        self.texture_snapshots.insert(
            key,
            TextureSnapshot {
                pixels: Arc::clone(&snapshot.pixels),
                content_hash: snapshot.content_hash,
                generations: texture_memory_generations(memory, texture),
            },
        );
        Ok(snapshot)
    }

    fn render_vertices(
        &mut self,
        memory: &mut Memory,
        vertices: &[Vertex],
        primitive: u32,
        texture: Texture,
        textured: bool,
        gouraud: bool,
    ) -> Result<(), GpuError> {
        // the framebuffer pointer is a 16-byte-aligned edram offset.  the
        // upper bits of framebufwidth are address/implementation bits and
        // must not become part of the row stride.
        let framebuffer = edram_address(self.registers[0x9c] & 0x001f_fff0);
        let stride = self.registers[0x9d] & 0x0000_07fc;
        if stride == 0 {
            return Ok(());
        }
        let format = self.registers[0xd2] & 3;
        let scissor_min = self.registers[0xd4];
        let scissor_max = self.registers[0xd5];
        let min_x = ((scissor_min & 0x3ff) as i32).clamp(0, stride as i32 - 1);
        let min_y = ((scissor_min >> 10) & 0x3ff) as i32;
        let max_x = ((scissor_max & 0x3ff) as i32).clamp(0, stride as i32 - 1);
        let max_y = ((scissor_max >> 10) & 0x3ff) as i32;
        let depth_address = edram_address(self.registers[0x9e] & 0x001f_fff0);
        let depth_stride = self.registers[0x9f] & 0x0000_07fc;
        let mut target = RenderTarget {
            memory,
            framebuffer,
            stride,
            format,
            min_x,
            min_y,
            max_x,
            max_y,
            clear_mode: self.registers[0xd3],
            depth_address,
            depth_stride,
            depth_test: self.registers[0x23] & 1 != 0,
            depth_write: self.registers[0xe7] & 1 == 0,
            depth_function: self.registers[0xde] & 7,
            alpha_test: self.registers[0x22] & 1 != 0,
            alpha_function: self.registers[0xdb] & 7,
            alpha_reference: (self.registers[0xdb] >> 8) & 0xff,
            alpha_mask: (self.registers[0xdb] >> 16) & 0xff,
            color_test: self.registers[0x27] & 1 != 0,
            color_function: self.registers[0xd8] & 3,
            color_reference: self.registers[0xd9] & 0x00ff_ffff,
            color_mask: self.registers[0xda] & 0x00ff_ffff,
            blend: self.registers[0x21] & 1 != 0,
            blend_state: self.registers[0xdf],
            blend_fixed_a: self.registers[0xe0] & 0x00ff_ffff,
            blend_fixed_b: self.registers[0xe1] & 0x00ff_ffff,
            color_mask_rgb: self.registers[0xe8] & 0x00ff_ffff,
            color_mask_alpha: self.registers[0xe9] & 0xff,
            pixels_written: &mut self.last_render_pixels,
        };
        if target.clear_mode & 1 != 0 {
            target.clear(vertices)?;
            return Ok(());
        }
        match primitive {
            6 => {
                for pair in vertices.as_chunks::<2>().0.iter() {
                    if textured {
                        target.fill_textured_rect(texture, pair[0], pair[1])?;
                    } else {
                        target.fill_rect(pair[0], pair[1])?;
                    }
                }
            }
            3 => {
                for triangle in vertices.as_chunks::<3>().0.iter() {
                    if textured {
                        target.fill_textured_triangle(texture, triangle)?;
                    } else {
                        target.fill_triangle(triangle, gouraud)?;
                    }
                }
            }
            4 => {
                for index in 2..vertices.len() {
                    let triangle = [vertices[index - 2], vertices[index - 1], vertices[index]];
                    if textured {
                        target.fill_textured_triangle(texture, &triangle)?;
                    } else {
                        target.fill_triangle(&triangle, gouraud)?;
                    }
                }
            }
            5 => {
                for index in 2..vertices.len() {
                    let triangle = [vertices[0], vertices[index - 1], vertices[index]];
                    if textured {
                        target.fill_textured_triangle(texture, &triangle)?;
                    } else {
                        target.fill_triangle(&triangle, gouraud)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn relative_address(&self, data: u32) -> u32 {
        relative_address(self.base_address, self.offset_address, data)
    }

    fn read_vertex(
        &self,
        memory: &Memory,
        address: u32,
        layout: VertexLayout,
        texture: Texture,
    ) -> Result<Option<Vertex>, GpuError> {
        let through = self.registers[0x12] & 0x0080_0000 != 0;
        let mut position = match layout.position_format {
            // through-mode positions skip the viewport transform and are
            // used as raw screen coordinates: 8-bit values
            // stay signed bytes, 16-bit values stay signed words with an
            // unsigned depth, and floats pass through untouched.
            1 if through => [
                memory.read_u8(address + layout.position_offset as u32)? as i8 as f32,
                memory.read_u8(address + layout.position_offset as u32 + 1)? as i8 as f32,
                memory.read_u8(address + layout.position_offset as u32 + 2)? as i8 as f32,
            ],
            1 => [
                memory.read_u8(address + layout.position_offset as u32)? as i8 as f32 / 128.0,
                memory.read_u8(address + layout.position_offset as u32 + 1)? as i8 as f32 / 128.0,
                memory.read_u8(address + layout.position_offset as u32 + 2)? as i8 as f32 / 128.0,
            ],
            2 => {
                let x = memory.read_u16(address + layout.position_offset as u32)?;
                let y = memory.read_u16(address + layout.position_offset as u32 + 2)?;
                let z = memory.read_u16(address + layout.position_offset as u32 + 4)?;
                if through {
                    [x as i16 as f32, y as i16 as f32, z as f32]
                } else {
                    [
                        x as i16 as f32 / 32768.0,
                        y as i16 as f32 / 32768.0,
                        z as i16 as f32 / 32768.0,
                    ]
                }
            }
            3 => [
                read_f32(memory, address + layout.position_offset as u32)?,
                read_f32(memory, address + layout.position_offset as u32 + 4)?,
                read_f32(memory, address + layout.position_offset as u32 + 8)?,
            ],
            _ => return Ok(None),
        };
        if !through && layout.weight_format != 0 {
            let mut skinned = [0.0; 3];
            for bone in 0..layout.weight_count as usize {
                let weight = match layout.weight_format {
                    1 => memory.read_u8(address + bone as u32)? as f32 / 128.0,
                    2 => memory.read_u16(address + bone as u32 * 2)? as f32 / 32768.0,
                    3 => read_f32(memory, address + bone as u32 * 4)?,
                    _ => unreachable!(),
                };
                let transformed = transform_matrix_43(self.matrices.bones[bone], position);
                for axis in 0..3 {
                    skinned[axis] += transformed[axis] * weight;
                }
            }
            position = skinned;
        }
        let clip = (!through).then(|| self.transform_to_clip(position));
        let (x, y, depth, inv_w, outside_range) = if let Some(clip) = clip {
            self.clip_to_screen(clip, false)
                .unwrap_or((0, 0, 0, 0.0, true))
        } else {
            (
                position[0].round() as i32,
                position[1].round() as i32,
                saturating_u16(position[2].round()),
                1.0,
                !position.into_iter().all(f32::is_finite),
            )
        };
        let (u, v) = self.read_texture_coordinates(memory, address, layout, texture, through)?;
        let color = layout
            .color_offset
            .map(|offset| read_vertex_color(memory, address + offset as u32, layout.color_format))
            .transpose()?
            .unwrap_or_else(|| self.default_vertex_color());
        Ok(Some(Vertex {
            clip,
            outside_range,
            color,
            u,
            v,
            inv_w,
            x,
            y,
            depth,
        }))
    }

    fn read_texture_coordinates(
        &self,
        memory: &Memory,
        address: u32,
        layout: VertexLayout,
        texture: Texture,
        through: bool,
    ) -> Result<(i32, i32), GpuError> {
        let Some(offset) = layout.texture_offset else {
            return Ok((0, 0));
        };
        let offset = address + offset as u32;
        let (mut u, mut v) = match layout.texture_format {
            1 => (
                memory.read_u8(offset)? as f32,
                memory.read_u8(offset + 1)? as f32,
            ),
            2 => (
                memory.read_u16(offset)? as f32,
                memory.read_u16(offset + 2)? as f32,
            ),
            3 => (read_f32(memory, offset)?, read_f32(memory, offset + 4)?),
            _ => return Ok((0, 0)),
        };
        if !through {
            let scale_u = if self.registers[0x48] == 0 {
                1.0
            } else {
                float24(self.registers[0x48])
            };
            let scale_v = if self.registers[0x49] == 0 {
                1.0
            } else {
                float24(self.registers[0x49])
            };
            u = match layout.texture_format {
                1 => u / 128.0,
                2 => u / 32768.0,
                _ => u,
            } * scale_u
                + float24(self.registers[0x4a]);
            v = match layout.texture_format {
                1 => v / 128.0,
                2 => v / 32768.0,
                _ => v,
            } * scale_v
                + float24(self.registers[0x4b]);
            u *= texture.width as f32;
            v *= texture.height as f32;
        }
        Ok((saturating_i32(u.round()), saturating_i32(v.round())))
    }

    fn transform_to_clip(&self, position: [f32; 3]) -> [f32; 4] {
        let world = transform_matrix_43(self.matrices.world, position);
        let view = transform_matrix_43(self.matrices.view, world);
        transform_matrix_44(self.matrices.projection, [view[0], view[1], view[2], 1.0])
    }

    fn clip_to_screen(
        &self,
        clip: [f32; 4],
        always_check_range: bool,
    ) -> Option<(i32, i32, u16, f32, bool)> {
        let w = clip[3];
        if !w.is_finite() || w.abs() < f32::EPSILON {
            return None;
        }
        let mut x_scale = float24(self.registers[0x42]);
        let mut y_scale = float24(self.registers[0x43]);
        let mut x_center = float24(self.registers[0x45]);
        let mut y_center = float24(self.registers[0x46]);
        let mut z_scale = float24(self.registers[0x44]);
        let mut z_center = float24(self.registers[0x47]);
        if x_scale == 0.0 && x_center == 0.0 {
            x_scale = 240.0;
            x_center = 240.0;
        }
        if y_scale == 0.0 && y_center == 0.0 {
            y_scale = 136.0;
            y_center = 136.0;
        }
        if z_scale == 0.0 && z_center == 0.0 {
            z_scale = 32767.5;
            z_center = 32767.5;
        }
        let offset_x = (self.registers[0x4c] & 0xffff) as f32 / 16.0;
        let offset_y = (self.registers[0x4d] & 0xffff) as f32 / 16.0;
        let x = clip[0] / w * x_scale + x_center - offset_x;
        let y = clip[1] / w * y_scale + y_center - offset_y;
        let z = clip[2] / w * z_scale + z_center;
        // psp drawing coordinates have a 12-bit range before xy offsets.
        // out-of-range vertices reject their primitive, rather than being
        // clamped into huge screen-covering triangles after projection.
        let screen_bound = 4095.0 + 15.5 / 16.0;
        let depth_clamp = self.registers[0x1c] & 1 != 0;
        // with depth clamp enabled, vertices behind the near plane must
        // reach the clipper even when their projected xy is out of range.
        // newly generated clip vertices are always range-checked.
        let check_xy = !depth_clamp || always_check_range || clip[2] > -w;
        let outside_range = (check_xy
            && (x + offset_x < 0.0
                || (if depth_clamp {
                    x + offset_x >= screen_bound
                } else {
                    x + offset_x > screen_bound
                })
                || y + offset_y < 0.0
                || y + offset_y >= screen_bound))
            || (!depth_clamp && !(0.0..65536.0).contains(&z));
        (x.is_finite() && y.is_finite() && z.is_finite()).then_some((
            x.round() as i32,
            y.round() as i32,
            saturating_u16(z.round()),
            1.0 / w,
            outside_range,
        ))
    }

    fn default_vertex_color(&self) -> u32 {
        // ge uses materialambient/materialalpha as the color for vertices
        // without an explicit color.  ambientcolor is a lighting input, not
        // the fallback vertex color; using it makes textured through-mode
        // quads black when a game relies on the default material color.
        if self.registers[0x55] == 0 && self.registers[0x58] == 0 {
            u32::MAX
        } else {
            (self.registers[0x55] & 0x00ff_ffff) | ((self.registers[0x58] & 0xff) << 24)
        }
    }

    /// Convert a PSP display framebuffer into host RGBA8 pixels.
    pub fn framebuffer_rgba(
        memory: &Memory,
        address: u32,
        stride: u32,
        width: u32,
        height: u32,
        format: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let bytes_per_pixel = match format {
            0..=2 => 2usize,
            3 => 4,
            _ => return Err(GpuError::PixelFormat(format)),
        };
        let stride = usize::try_from(stride).map_err(|_| GpuError::FramebufferSize)?;
        let width = usize::try_from(width).map_err(|_| GpuError::FramebufferSize)?;
        let height = usize::try_from(height).map_err(|_| GpuError::FramebufferSize)?;
        let row_bytes = stride
            .checked_mul(bytes_per_pixel)
            .ok_or(GpuError::FramebufferSize)?;
        let output_size = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(GpuError::FramebufferSize)?;
        let mut output = Vec::with_capacity(output_size);
        for y in 0..height {
            let row_address = address.wrapping_add((y * row_bytes) as u32);
            let row = memory.read_bytes(row_address, row_bytes)?;
            for x in 0..width {
                let pixel = x * bytes_per_pixel;
                let (r, g, b, a) = match format {
                    0 => {
                        let p = u16::from_le_bytes(row[pixel..pixel + 2].try_into().unwrap());
                        (expand5(p), expand6(p >> 5), expand5(p >> 11), 255)
                    }
                    1 => {
                        let p = u16::from_le_bytes(row[pixel..pixel + 2].try_into().unwrap());
                        (
                            expand5(p),
                            expand5(p >> 5),
                            expand5(p >> 10),
                            if p & 0x8000 != 0 { 255 } else { 0 },
                        )
                    }
                    2 => {
                        let p = u16::from_le_bytes(row[pixel..pixel + 2].try_into().unwrap());
                        (
                            expand4(p),
                            expand4(p >> 4),
                            expand4(p >> 8),
                            expand4(p >> 12),
                        )
                    }
                    3 => {
                        let p = u32::from_le_bytes(row[pixel..pixel + 4].try_into().unwrap());
                        (p as u8, (p >> 8) as u8, (p >> 16) as u8, (p >> 24) as u8)
                    }
                    _ => unreachable!("pixel format validated above"),
                };
                output.extend_from_slice(&[r, g, b, a]);
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug)]
struct Vertex {
    clip: Option<[f32; 4]>,
    outside_range: bool,
    color: u32,
    u: i32,
    v: i32,
    inv_w: f32,
    x: i32,
    y: i32,
    depth: u16,
}

#[derive(Clone, Copy, Debug)]
struct VertexLayout {
    weight_format: u32,
    weight_count: u32,
    texture_format: u32,
    texture_offset: Option<usize>,
    color_format: u32,
    color_offset: Option<usize>,
    position_format: u32,
    position_offset: usize,
    index_format: u32,
    stride_bytes: usize,
}

impl VertexLayout {
    fn new(vertex_type: u32) -> Option<Self> {
        let texture_format = vertex_type & 3;
        let color_format = (vertex_type >> 2) & 7;
        let normal_format = (vertex_type >> 5) & 3;
        let position_format = (vertex_type >> 7) & 3;
        let weight_format = (vertex_type >> 9) & 3;
        let index_format = (vertex_type >> 11) & 3;
        let weight_count = ((vertex_type >> 14) & 7) + 1;
        let morph_count = ((vertex_type >> 18) & 7) + 1;
        if position_format == 0 || (color_format != 0 && !(4..=7).contains(&color_format)) {
            return None;
        }
        let mut offset = 0usize;
        let mut biggest_alignment = 1usize;
        if weight_format != 0 {
            let (alignment, size) = match weight_format {
                1 => (1, 1),
                2 => (2, 2),
                3 => (4, 4),
                _ => unreachable!(),
            };
            offset = align_offset(offset, alignment);
            offset += size * weight_count as usize;
            biggest_alignment = biggest_alignment.max(alignment);
        }
        let texture_offset = if texture_format != 0 {
            let (alignment, size) = match texture_format {
                1 => (1, 2),
                2 => (2, 4),
                3 => (4, 8),
                _ => unreachable!(),
            };
            offset = align_offset(offset, alignment);
            let result = offset;
            offset += size;
            biggest_alignment = biggest_alignment.max(alignment);
            Some(result)
        } else {
            None
        };
        let color_offset = if color_format != 0 {
            let alignment = if color_format == 7 { 4 } else { 2 };
            offset = align_offset(offset, alignment);
            let result = offset;
            offset += if color_format == 7 { 4 } else { 2 };
            biggest_alignment = biggest_alignment.max(alignment);
            Some(result)
        } else {
            None
        };
        if normal_format != 0 {
            let (alignment, size) = match normal_format {
                1 => (1, 3),
                2 => (2, 6),
                3 => (4, 12),
                _ => unreachable!(),
            };
            offset = align_offset(offset, alignment);
            offset += size;
            biggest_alignment = biggest_alignment.max(alignment);
        }
        let (alignment, position_size) = match position_format {
            1 => (1, 3),
            2 => (2, 6),
            3 => (4, 12),
            _ => return None,
        };
        offset = align_offset(offset, alignment);
        let position_offset = offset;
        offset += position_size;
        biggest_alignment = biggest_alignment.max(alignment);
        Some(Self {
            weight_format,
            weight_count,
            texture_format,
            texture_offset,
            color_format,
            color_offset,
            position_format,
            position_offset,
            index_format,
            stride_bytes: align_offset(offset, biggest_alignment) * morph_count as usize,
        })
    }
}

fn align_offset(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

fn read_vertex_index(
    memory: &Memory,
    address: u32,
    format: u32,
    index: usize,
) -> Result<u32, MemoryFault> {
    let address = address.wrapping_add(
        index as u32
            * match format {
                1 => 1,
                2 => 2,
                3 => 4,
                _ => 0,
            },
    );
    match format {
        1 => memory.read_u8(address).map(u32::from),
        2 => memory.read_u16(address).map(u32::from),
        3 => memory.read_u32(address),
        _ => Ok(index as u32),
    }
}

fn read_f32(memory: &Memory, address: u32) -> Result<f32, MemoryFault> {
    memory.read_u32(address).map(f32::from_bits)
}

fn saturating_i32(value: f32) -> i32 {
    if value.is_nan() {
        0
    } else if value >= i32::MAX as f32 {
        i32::MAX
    } else if value <= i32::MIN as f32 {
        i32::MIN
    } else {
        value as i32
    }
}

fn saturating_u16(value: f32) -> u16 {
    if value.is_nan() {
        0
    } else {
        value.clamp(0.0, u16::MAX as f32) as u16
    }
}

fn perspective_interpolate(values: [i32; 3], inv_w: [f32; 3], weights: [f32; 3]) -> f32 {
    let denominator = inv_w[0] * weights[0] + inv_w[1] * weights[1] + inv_w[2] * weights[2];
    if denominator.is_finite() && denominator.abs() > f32::EPSILON {
        let numerator = values[0] as f32 * inv_w[0] * weights[0]
            + values[1] as f32 * inv_w[1] * weights[1]
            + values[2] as f32 * inv_w[2] * weights[2];
        numerator / denominator
    } else {
        values[0] as f32 * weights[0]
            + values[1] as f32 * weights[1]
            + values[2] as f32 * weights[2]
    }
}

fn bezier_weights(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    [
        one_minus_t * one_minus_t * one_minus_t,
        3.0 * t * one_minus_t * one_minus_t,
        3.0 * t * t * one_minus_t,
        t * t * t,
    ]
}

fn spline_weights(t: f32, patch: usize, patch_count: usize, edge_type: u32) -> [f32; 4] {
    // this is the uniform cubic spline basis, including the open-end
    // knot adjustments controlled by spline's two edge bits.
    let mut d30 = 1.0 / 3.0;
    let mut d41 = 1.0 / 3.0;
    let mut d52 = 1.0 / 3.0;
    let mut d31 = 1.0 / 2.0;
    let mut d42 = 1.0 / 2.0;
    if edge_type & 1 != 0 {
        if patch == 0 {
            d30 = 1.0;
            d41 = 1.0 / 2.0;
            d31 = 1.0;
        } else if patch == 1 && patch_count > 1 {
            d30 = 1.0 / 2.0;
        }
    }
    if edge_type & 2 != 0 {
        if patch + 1 == patch_count {
            d41 = 1.0 / 2.0;
            d52 = 1.0;
            d42 = 1.0;
        } else if patch + 2 == patch_count && patch_count > 1 {
            d52 = 1.0 / 2.0;
        }
    }
    let patch = patch as f32;
    let t = patch + t.clamp(0.0, 1.0);
    let f30 = (t - (patch - 2.0)) * d30;
    let f41 = (t - (patch - 1.0)) * d41;
    let f52 = (t - patch) * d52;
    let f31 = (t - (patch - 1.0)) * d31;
    let f42 = (t - patch) * d42;
    let f32 = t - patch;
    let a = (1.0 - f30) * (1.0 - f31);
    let b = f31 * f41;
    let c = (1.0 - f41) * (1.0 - f42);
    let d = f42 * f52;
    [
        a * (1.0 - f32),
        1.0 - a - b + (a + b + c - 1.0) * f32,
        b + (1.0 - b - c - d) * f32,
        d * f32,
    ]
}

fn interpolate_patch_vertex(
    controls: &[Option<Vertex>],
    points_u: usize,
    base_u: usize,
    base_v: usize,
    weights_u: [f32; 4],
    weights_v: [f32; 4],
) -> Option<Vertex> {
    let mut color = [0.0f32; 4];
    let mut u = 0.0;
    let mut v = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    let mut depth = 0.0;
    let mut inv_w = 0.0;
    for (control_v, &weight_v) in weights_v.iter().enumerate() {
        for (control_u, &weight_u) in weights_u.iter().enumerate() {
            let index = (base_v + control_v)
                .checked_mul(points_u)?
                .checked_add(base_u + control_u)?;
            let vertex = controls.get(index)?.as_ref()?;
            let weight = weight_u * weight_v;
            u += vertex.u as f32 * weight;
            v += vertex.v as f32 * weight;
            x += vertex.x as f32 * weight;
            y += vertex.y as f32 * weight;
            depth += f32::from(vertex.depth) * weight;
            inv_w += vertex.inv_w * weight;
            for (channel, value) in color.iter_mut().enumerate() {
                *value += ((vertex.color >> (channel * 8)) & 0xff) as f32 * weight;
            }
        }
    }
    if ![u, v, x, y, depth, inv_w].into_iter().all(f32::is_finite) {
        return None;
    }
    let color = color
        .into_iter()
        .enumerate()
        .fold(0, |value, (channel, value8)| {
            value | (value8.round().clamp(0.0, 255.0) as u32) << (channel * 8)
        });
    Some(Vertex {
        clip: None,
        outside_range: false,
        color,
        u: saturating_i32(u.round()),
        v: saturating_i32(v.round()),
        inv_w,
        x: saturating_i32(x.round()),
        y: saturating_i32(y.round()),
        depth: saturating_u16(depth.round()),
    })
}

fn transform_matrix_43(matrix: [f32; 12], position: [f32; 3]) -> [f32; 3] {
    [
        position[0] * matrix[0] + position[1] * matrix[3] + position[2] * matrix[6] + matrix[9],
        position[0] * matrix[1] + position[1] * matrix[4] + position[2] * matrix[7] + matrix[10],
        position[0] * matrix[2] + position[1] * matrix[5] + position[2] * matrix[8] + matrix[11],
    ]
}

fn transform_matrix_44(matrix: [f32; 16], position: [f32; 4]) -> [f32; 4] {
    [
        position[0] * matrix[0]
            + position[1] * matrix[4]
            + position[2] * matrix[8]
            + position[3] * matrix[12],
        position[0] * matrix[1]
            + position[1] * matrix[5]
            + position[2] * matrix[9]
            + position[3] * matrix[13],
        position[0] * matrix[2]
            + position[1] * matrix[6]
            + position[2] * matrix[10]
            + position[3] * matrix[14],
        position[0] * matrix[3]
            + position[1] * matrix[7]
            + position[2] * matrix[11]
            + position[3] * matrix[15],
    ]
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Texture {
    address: u32,
    stride: u32,
    width: u32,
    height: u32,
    format: u32,
    clut_address_low: u32,
    clut_address_high: u32,
    clut_format: u32,
    function: u32,
    environment_color: u32,
    clamp_s: bool,
    clamp_t: bool,
    swizzled: bool,
}

struct RenderTarget<'a> {
    memory: &'a mut Memory,
    framebuffer: u32,
    stride: u32,
    format: u32,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    clear_mode: u32,
    depth_address: u32,
    depth_stride: u32,
    depth_test: bool,
    depth_write: bool,
    depth_function: u32,
    alpha_test: bool,
    alpha_function: u32,
    alpha_reference: u32,
    alpha_mask: u32,
    color_test: bool,
    color_function: u32,
    color_reference: u32,
    color_mask: u32,
    blend: bool,
    blend_state: u32,
    blend_fixed_a: u32,
    blend_fixed_b: u32,
    color_mask_rgb: u32,
    color_mask_alpha: u32,
    pixels_written: &'a mut u64,
}

impl RenderTarget<'_> {
    fn clear(&mut self, vertices: &[Vertex]) -> Result<(), GpuError> {
        if self.max_x < self.min_x || self.max_y < self.min_y {
            return Ok(());
        }
        let clear_color = vertices.first().map_or(0, |vertex| vertex.color);
        let clear_depth = vertices.first().map_or(u16::MAX, |vertex| vertex.depth);
        let clear_rgb = self.clear_mode & 0x100 != 0;
        let clear_alpha = self.clear_mode & 0x200 != 0;
        let clear_depth_buffer = self.clear_mode & 0x400 != 0;
        for y in self.min_y..=self.max_y {
            for x in self.min_x..=self.max_x {
                if clear_rgb || clear_alpha {
                    let previous = self.read_color(x, y)?;
                    let color = (if clear_rgb {
                        clear_color & 0x00ff_ffff
                    } else {
                        previous & 0x00ff_ffff
                    }) | (if clear_alpha {
                        clear_color & 0xff00_0000
                    } else {
                        previous & 0xff00_0000
                    });
                    if self.write_color(x, y, color)? {
                        *self.pixels_written += 1;
                    }
                }
                if clear_depth_buffer {
                    self.write_depth(x, y, clear_depth)?;
                }
            }
        }
        Ok(())
    }

    fn fill_textured_rect(
        &mut self,
        texture: Texture,
        first: Vertex,
        second: Vertex,
    ) -> Result<(), GpuError> {
        let left = first.x.min(second.x).clamp(self.min_x, self.max_x);
        let right = first.x.max(second.x).clamp(self.min_x, self.max_x + 1);
        let top = first.y.min(second.y).clamp(self.min_y, self.max_y);
        let bottom = first.y.max(second.y).clamp(self.min_y, self.max_y + 1);
        let width = (second.x - first.x).abs().max(1);
        let height = (second.y - first.y).abs().max(1);
        for y in top..bottom {
            for x in left..right {
                let u = first.u as i64
                    + i64::from(x - first.x) * i64::from(second.u - first.u) / i64::from(width);
                let v = first.v as i64
                    + i64::from(y - first.y) * i64::from(second.v - first.v) / i64::from(height);
                let Some(texel) = texture.sample(self.memory, u as i32, v as i32)? else {
                    continue;
                };
                self.write_pixel(
                    x,
                    y,
                    second.depth,
                    combine_texture(texel, second.color, texture),
                )?;
            }
        }
        Ok(())
    }

    fn fill_textured_triangle(
        &mut self,
        texture: Texture,
        vertices: &[Vertex],
    ) -> Result<(), GpuError> {
        if vertices.len() != 3 {
            return Ok(());
        }
        let min_x = vertices
            .iter()
            .map(|vertex| vertex.x)
            .min()
            .unwrap()
            .clamp(self.min_x, self.max_x);
        let max_x = vertices
            .iter()
            .map(|vertex| vertex.x)
            .max()
            .unwrap()
            .clamp(self.min_x, self.max_x);
        let min_y = vertices
            .iter()
            .map(|vertex| vertex.y)
            .min()
            .unwrap()
            .clamp(self.min_y, self.max_y);
        let max_y = vertices
            .iter()
            .map(|vertex| vertex.y)
            .max()
            .unwrap()
            .clamp(self.min_y, self.max_y);
        let edge = |a: Vertex, b: Vertex, x: i32, y: i32| {
            // i128: through-mode triangles skip range checks, so arbitrary
            // i32 coordinates can reach this edge computation and overflow
            // i64 products into horizontal banding streaks.
            (i128::from(x) - i128::from(a.x)) * (i128::from(b.y) - i128::from(a.y))
                - (i128::from(y) - i128::from(a.y)) * (i128::from(b.x) - i128::from(a.x))
        };
        let area = edge(vertices[0], vertices[1], vertices[2].x, vertices[2].y);
        if area == 0 {
            return Ok(());
        }
        let area_f = area as f32;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let e0 = edge(vertices[1], vertices[2], x, y);
                let e1 = edge(vertices[2], vertices[0], x, y);
                let e2 = edge(vertices[0], vertices[1], x, y);
                if !((area >= 0 && e0 >= 0 && e1 >= 0 && e2 >= 0)
                    || (area < 0 && e0 <= 0 && e1 <= 0 && e2 <= 0))
                {
                    continue;
                }
                let w0 = e0 as f32 / area_f;
                let w1 = e1 as f32 / area_f;
                let w2 = e2 as f32 / area_f;
                let weights = [w0, w1, w2];
                let u = perspective_interpolate(
                    [vertices[0].u, vertices[1].u, vertices[2].u],
                    [vertices[0].inv_w, vertices[1].inv_w, vertices[2].inv_w],
                    weights,
                )
                .round() as i32;
                let v = perspective_interpolate(
                    [vertices[0].v, vertices[1].v, vertices[2].v],
                    [vertices[0].inv_w, vertices[1].inv_w, vertices[2].inv_w],
                    weights,
                )
                .round() as i32;
                let Some(texel) = texture.sample(self.memory, u, v)? else {
                    continue;
                };
                let vertex_color = lerp_color(
                    vertices[0].color,
                    vertices[1].color,
                    vertices[2].color,
                    w0,
                    w1,
                    w2,
                );
                let depth = lerp_depth(
                    vertices[0].depth,
                    vertices[1].depth,
                    vertices[2].depth,
                    w0,
                    w1,
                    w2,
                );
                self.write_pixel(x, y, depth, combine_texture(texel, vertex_color, texture))?;
            }
        }
        Ok(())
    }

    fn fill_rect(&mut self, first: Vertex, second: Vertex) -> Result<(), GpuError> {
        let left = first.x.min(second.x).clamp(self.min_x, self.max_x);
        let right = first.x.max(second.x).clamp(self.min_x, self.max_x + 1);
        let top = first.y.min(second.y).clamp(self.min_y, self.max_y);
        let bottom = first.y.max(second.y).clamp(self.min_y, self.max_y + 1);
        for y in top..bottom {
            for x in left..right {
                self.write_pixel(x, y, second.depth, second.color)?;
            }
        }
        Ok(())
    }

    fn fill_triangle(&mut self, vertices: &[Vertex], gouraud: bool) -> Result<(), GpuError> {
        let min_x = vertices
            .iter()
            .map(|v| v.x)
            .min()
            .unwrap()
            .clamp(self.min_x, self.max_x);
        let max_bound_x = vertices
            .iter()
            .map(|v| v.x)
            .max()
            .unwrap()
            .clamp(self.min_x, self.max_x);
        let min_y = vertices
            .iter()
            .map(|v| v.y)
            .min()
            .unwrap()
            .clamp(self.min_y, self.max_y);
        let max_bound_y = vertices
            .iter()
            .map(|v| v.y)
            .max()
            .unwrap()
            .clamp(self.min_y, self.max_y);
        let edge = |a: Vertex, b: Vertex, x: i32, y: i32| {
            // i128: same through-mode overflow rationale as above.
            (i128::from(x) - i128::from(a.x)) * (i128::from(b.y) - i128::from(a.y))
                - (i128::from(y) - i128::from(a.y)) * (i128::from(b.x) - i128::from(a.x))
        };
        let area = edge(vertices[0], vertices[1], vertices[2].x, vertices[2].y);
        for y in min_y..=max_bound_y {
            for x in min_x..=max_bound_x {
                let e0 = edge(vertices[1], vertices[2], x, y);
                let e1 = edge(vertices[2], vertices[0], x, y);
                let e2 = edge(vertices[0], vertices[1], x, y);
                if (area >= 0 && e0 >= 0 && e1 >= 0 && e2 >= 0)
                    || (area < 0 && e0 <= 0 && e1 <= 0 && e2 <= 0)
                {
                    let area_f = area as f32;
                    let w0 = e0 as f32 / area_f;
                    let w1 = e1 as f32 / area_f;
                    let w2 = e2 as f32 / area_f;
                    let color = if gouraud {
                        lerp_color(
                            vertices[0].color,
                            vertices[1].color,
                            vertices[2].color,
                            w0,
                            w1,
                            w2,
                        )
                    } else {
                        vertices[0].color
                    };
                    let depth = lerp_depth(
                        vertices[0].depth,
                        vertices[1].depth,
                        vertices[2].depth,
                        w0,
                        w1,
                        w2,
                    );
                    self.write_pixel(x, y, depth, color)?;
                }
            }
        }
        Ok(())
    }

    fn write_pixel(&mut self, x: i32, y: i32, depth: u16, color: u32) -> Result<(), GpuError> {
        if x < self.min_x || x > self.max_x || y < self.min_y || y > self.max_y {
            return Ok(());
        }
        if self.alpha_test
            && !compare_masked(
                (color >> 24) & 0xff,
                self.alpha_reference,
                self.alpha_mask,
                self.alpha_function,
            )
        {
            return Ok(());
        }
        if self.color_test
            && !compare_masked(
                color & 0x00ff_ffff,
                self.color_reference,
                self.color_mask,
                self.color_function,
            )
        {
            return Ok(());
        }
        if self.depth_test
            && self.depth_stride != 0
            && let Some(existing) = self.read_depth(x, y)?
            && !compare_depth(depth, existing, self.depth_function)
        {
            return Ok(());
        }
        let mask = self.color_mask_rgb | (self.color_mask_alpha << 24);
        let destination = if self.blend || mask != 0 {
            self.read_color(x, y)?
        } else {
            0
        };
        let mut output = if self.blend {
            blend_color(
                color,
                destination,
                self.blend_state,
                self.blend_fixed_a,
                self.blend_fixed_b,
            )
        } else {
            color
        };
        output = (output & !mask) | (destination & mask);
        if self.write_color(x, y, output)? {
            *self.pixels_written += 1;
        }
        if self.depth_write && self.depth_stride != 0 {
            self.write_depth(x, y, depth)?;
        }
        Ok(())
    }

    fn read_color(&mut self, x: i32, y: i32) -> Result<u32, GpuError> {
        let Some(address) = self.color_address(x, y) else {
            return Ok(0);
        };
        Ok(match self.format {
            0 => {
                let value = self.memory.read_u16(address)?;
                u32::from(expand5(value))
                    | (u32::from(expand6(value >> 5)) << 8)
                    | (u32::from(expand5(value >> 11)) << 16)
                    | 0xff00_0000
            }
            1 => {
                let value = self.memory.read_u16(address)?;
                u32::from(expand5(value))
                    | (u32::from(expand5(value >> 5)) << 8)
                    | (u32::from(expand5(value >> 10)) << 16)
                    | if value & 0x8000 != 0 { 0xff00_0000 } else { 0 }
            }
            2 => {
                let value = self.memory.read_u16(address)?;
                u32::from(expand4(value))
                    | (u32::from(expand4(value >> 4)) << 8)
                    | (u32::from(expand4(value >> 8)) << 16)
                    | (u32::from(expand4(value >> 12)) << 24)
            }
            3 => self.memory.read_u32(address)?,
            _ => return Err(GpuError::PixelFormat(self.format)),
        })
    }

    fn write_color(&mut self, x: i32, y: i32, color: u32) -> Result<bool, GpuError> {
        let Some(address) = self.color_address(x, y) else {
            return Ok(false);
        };
        match self.format {
            3 => self.memory.write_u32(address, color)?,
            0 => {
                let packed =
                    ((color >> 3) & 0x1f) | ((color >> 10) & 0x7e0) | ((color >> 19) & 0xf800);
                self.memory.write_u16(address, packed as u16)?;
            }
            1 => {
                let packed = ((color >> 3) & 0x1f)
                    | ((color >> 11) & 0x3e0)
                    | ((color >> 19) & 0x7c00)
                    | ((color >> 16) & 0x8000);
                self.memory.write_u16(address, packed as u16)?;
            }
            2 => {
                let packed = ((color >> 4) & 0xf)
                    | ((color >> 8) & 0xf0)
                    | ((color >> 12) & 0xf00)
                    | ((color >> 16) & 0xf000);
                self.memory.write_u16(address, packed as u16)?;
            }
            _ => return Err(GpuError::PixelFormat(self.format)),
        }
        Ok(true)
    }

    fn read_depth(&mut self, x: i32, y: i32) -> Result<Option<u16>, GpuError> {
        let Some(address) = self.depth_address(x, y) else {
            return Ok(None);
        };
        Ok(Some(self.memory.read_u16(address)?))
    }

    fn write_depth(&mut self, x: i32, y: i32, depth: u16) -> Result<(), GpuError> {
        if let Some(address) = self.depth_address(x, y) {
            self.memory.write_u16(address, depth)?;
        }
        Ok(())
    }

    fn color_address(&self, x: i32, y: i32) -> Option<u32> {
        if x < 0 || y < 0 || x >= self.stride as i32 {
            return None;
        }
        let bytes = if self.format == 3 { 4 } else { 2 };
        checked_edram_span(
            self.framebuffer,
            (y as u32)
                .checked_mul(self.stride)?
                .checked_add(x as u32)?
                .checked_mul(bytes)?,
            bytes,
        )
    }

    fn depth_address(&self, x: i32, y: i32) -> Option<u32> {
        if x < 0 || y < 0 || self.depth_stride == 0 || x >= self.depth_stride as i32 {
            return None;
        }
        checked_edram_span(
            self.depth_address,
            (y as u32)
                .checked_mul(self.depth_stride)?
                .checked_add(x as u32)?
                .checked_mul(2)?,
            2,
        )
    }
}

impl Texture {
    fn sample(&self, memory: &Memory, u: i32, v: i32) -> Result<Option<u32>, GpuError> {
        let width = self.width.max(1);
        let height = self.height.max(1);
        let Some(x) = wrap_coordinate(u, width, self.clamp_s) else {
            return Ok(None);
        };
        let Some(y) = wrap_coordinate(v, height, self.clamp_t) else {
            return Ok(None);
        };
        if (8..=10).contains(&self.format) {
            return self.sample_dxt(memory, x, y);
        }
        let address = self.texel_address(x, y);
        let bytes = if matches!(self.format, 4 | 5) {
            1
        } else if matches!(self.format, 0..=2 | 6) {
            2
        } else {
            4
        };
        let Some(address) =
            checked_edram_span(self.address, address.wrapping_sub(self.address), bytes)
        else {
            return Ok(None);
        };
        let color = match self.format {
            0 => {
                let value = memory.read_u16(address)?;
                u32::from(expand5(value))
                    | (u32::from(expand6(value >> 5)) << 8)
                    | (u32::from(expand5(value >> 11)) << 16)
                    | 0xff00_0000
            }
            1 => {
                let value = memory.read_u16(address)?;
                u32::from(expand5(value))
                    | (u32::from(expand5(value >> 5)) << 8)
                    | (u32::from(expand5(value >> 10)) << 16)
                    | if value & 0x8000 != 0 { 0xff00_0000 } else { 0 }
            }
            2 => {
                let value = memory.read_u16(address)?;
                u32::from(expand4(value))
                    | (u32::from(expand4(value >> 4)) << 8)
                    | (u32::from(expand4(value >> 8)) << 16)
                    | (u32::from(expand4(value >> 12)) << 24)
            }
            3 => memory.read_u32(address)?,
            4 => {
                let packed = memory.read_u8(address)?;
                let raw_index = if x & 1 == 0 {
                    packed & 0xf
                } else {
                    packed >> 4
                };
                self.palette_color(memory, u32::from(raw_index))?
            }
            5 => self.palette_color(memory, u32::from(memory.read_u8(address)?))?,
            6 => self.palette_color(memory, u32::from(memory.read_u16(address)?))?,
            7 => self.palette_color(memory, memory.read_u32(address)?)?,
            _ => return Ok(None),
        };
        Ok(Some(color))
    }

    fn palette_color(&self, memory: &Memory, raw_index: u32) -> Result<u32, GpuError> {
        let shift = (self.clut_format >> 2) & 0x1f;
        let mask = (self.clut_format >> 8) & 0xff;
        let start = ((self.clut_format >> 16) & 0x1f) << 4;
        let palette_format = self.clut_format & 3;
        let offset_mask = if palette_format == 3 { 0xff } else { 0x1ff };
        let palette_index = if shift != 0 || mask != 0xff || start != 0 {
            ((raw_index >> shift) & mask) | (start & offset_mask)
        } else {
            raw_index & 0xff
        };
        let palette =
            (self.clut_address_low & 0x00ff_fff0) | ((self.clut_address_high << 8) & 0x0f00_0000);
        read_palette_color(memory, palette, palette_index, palette_format)
    }

    fn texel_address(&self, x: u32, y: u32) -> u32 {
        if !self.swizzled {
            let index = y.saturating_mul(self.stride).saturating_add(x);
            return self.address
                + match self.format {
                    4 => index / 2,
                    5 => index,
                    0..=2 | 6 => index * 2,
                    _ => index * 4,
                };
        }
        let bits_per_pixel = match self.format {
            4 => 4,
            5 => 8,
            0..=2 | 6 => 16,
            _ => 32,
        };
        let block_width = (128 / bits_per_pixel).max(1);
        let blocks_per_row = self.stride.div_ceil(block_width);
        let block = (y / 8) * blocks_per_row + x / block_width;
        let within = (y % 8) * 16 + (x % block_width) * bits_per_pixel / 8;
        self.address + block * 128 + within
    }

    fn sample_dxt(&self, memory: &Memory, x: u32, y: u32) -> Result<Option<u32>, GpuError> {
        let blocks_per_row = self.stride.div_ceil(4).max(1);
        let block_x = x / 4;
        let block_y = y / 4;
        let bytes_per_block = if self.format == 8 { 8 } else { 16 };
        let block_index = block_y
            .saturating_mul(blocks_per_row)
            .saturating_add(block_x);
        let Some(address) = checked_edram_span(
            self.address,
            block_index.saturating_mul(bytes_per_block),
            bytes_per_block,
        ) else {
            return Ok(None);
        };
        let block = memory.read_bytes(address, bytes_per_block as usize)?;
        let local_x = x & 3;
        let local_y = y & 3;
        let color_offset = if self.format == 8 { 0 } else { 8 };
        let color0 = u16::from_le_bytes([block[color_offset], block[color_offset + 1]]);
        let color1 = u16::from_le_bytes([block[color_offset + 2], block[color_offset + 3]]);
        let colors = dxt_colors(color0, color1, self.format == 8);
        let selector_offset = color_offset + 4;
        let selector_word = u32::from_le_bytes(
            block[selector_offset..selector_offset + 4]
                .try_into()
                .unwrap(),
        );
        let selector = ((selector_word >> ((local_y * 4 + local_x) * 2)) & 3) as usize;
        let mut color = colors[selector];
        if self.format == 9 {
            let alpha = (block[local_y as usize * 2] >> ((local_x & 1) * 4)) & 0xf;
            color = (color & 0x00ff_ffff) | (u32::from(alpha * 0x11) << 24);
        } else if self.format == 10 {
            let alpha_bits = u64::from_le_bytes({
                let mut bytes = [0; 8];
                bytes[..6].copy_from_slice(&block[2..8]);
                bytes
            });
            let alpha_palette = dxt_alpha_palette(block[0], block[1]);
            let alpha_index = ((alpha_bits >> (3 * (local_y * 4 + local_x))) & 7) as usize;
            color = (color & 0x00ff_ffff) | (u32::from(alpha_palette[alpha_index]) << 24);
        }
        Ok(Some(color))
    }
}

fn snapshot_texture(memory: &Memory, texture: Texture) -> Result<RenderTexture, GpuError> {
    // a psp texture dimension is encoded as a power of two and can describe
    // an absurdly large sparse surface.  refuse pathological uploads rather
    // than allocating gigabytes while the ge list is being decoded.
    const MAX_HARDWARE_TEXTURE_DIMENSION: u32 = 4096;
    if texture.width > MAX_HARDWARE_TEXTURE_DIMENSION
        || texture.height > MAX_HARDWARE_TEXTURE_DIMENSION
    {
        return Err(GpuError::TextureSize);
    }
    let pixel_count = (texture.width as usize)
        .checked_mul(texture.height as usize)
        .ok_or(GpuError::TextureSize)?;
    let byte_count = pixel_count.checked_mul(4).ok_or(GpuError::TextureSize)?;
    let mut pixels = Vec::with_capacity(byte_count);
    for y in 0..texture.height {
        // linear rgba textures already have the host layout. copy a checked
        // row instead of resolving guest memory separately for every texel.
        if texture.format == 3 && !texture.swizzled && texture.address & 3 == 0 {
            let row_bytes = texture.width as usize * 4;
            let offset = texture.texel_address(0, y).wrapping_sub(texture.address);
            if let Some(address) = checked_edram_span(texture.address, offset, row_bytes as u32)
                && let Ok(row) = memory.read_slice(address, row_bytes)
            {
                pixels.extend_from_slice(row);
                continue;
            }
        }
        for x in 0..texture.width {
            let color = texture.sample(memory, x as i32, y as i32)?.unwrap_or(0);
            pixels.extend_from_slice(&color.to_le_bytes());
        }
    }
    let mut content_hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in &pixels {
        content_hash = (content_hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(render_texture(texture, pixels.into(), content_hash))
}

fn render_texture(texture: Texture, pixels: Arc<[u8]>, content_hash: u64) -> RenderTexture {
    RenderTexture {
        address: texture.address,
        width: texture.width,
        height: texture.height,
        pixels,
        content_hash,
        clamp_s: texture.clamp_s,
        clamp_t: texture.clamp_t,
        function: texture.function,
        environment_color: texture.environment_color,
    }
}

fn texture_memory_generations(memory: &Memory, texture: Texture) -> Vec<(u32, u32)> {
    let mut pages = Vec::new();
    let bytes_per_texel = match texture.format {
        4 => 1,
        5 => 1,
        0..=2 | 6 => 2,
        _ => 4,
    } as u64;
    let texture_bytes = if (8..=10).contains(&texture.format) {
        let blocks_per_row = u64::from(texture.stride.div_ceil(4).max(1));
        let block_rows = u64::from(texture.height.div_ceil(4).max(1));
        let bytes_per_block = if texture.format == 8 { 8 } else { 16 };
        blocks_per_row
            .saturating_mul(block_rows)
            .saturating_mul(bytes_per_block)
    } else if texture.swizzled {
        let bits_per_pixel = bytes_per_texel.saturating_mul(8);
        let block_width = (128 / bits_per_pixel).max(1);
        let blocks_per_row = u64::from(texture.stride).div_ceil(block_width);
        let block_rows = u64::from(texture.height).div_ceil(8);
        blocks_per_row
            .saturating_mul(block_rows)
            .saturating_mul(128)
    } else {
        u64::from(texture.stride.max(texture.width))
            .saturating_mul(u64::from(texture.height.max(1)))
            .saturating_mul(bytes_per_texel)
    };
    append_memory_pages(
        memory,
        &mut pages,
        texture.address,
        texture_bytes.max(bytes_per_texel),
    );

    if (4..=7).contains(&texture.format) {
        let palette = (texture.clut_address_low & 0x00ff_fff0)
            | ((texture.clut_address_high << 8) & 0x0f00_0000);
        let palette_format = texture.clut_format & 3;
        let palette_bytes = if palette_format == 3 { 1024 } else { 512 };
        append_memory_pages(memory, &mut pages, palette, palette_bytes);
    }
    pages
}

fn append_memory_pages(memory: &Memory, pages: &mut Vec<(u32, u32)>, address: u32, bytes: u64) {
    if bytes == 0 {
        return;
    }
    let first = address >> 12;
    let last_address = u64::from(address)
        .saturating_add(bytes.saturating_sub(1))
        .min(u64::from(u32::MAX));
    let last = (last_address as u32) >> 12;
    for page in first..=last {
        pages.push((page, memory.page_generation(page)));
    }
}

fn texture_dimension(value: u32, vertical: bool) -> u32 {
    let shift = if vertical {
        (value >> 8) & 0xf
    } else {
        value & 0xf
    };
    1u32 << shift
}

fn texture_address(texaddr: u32, texbufwidth: u32) -> u32 {
    (texaddr & 0x00ff_fff0) | ((texbufwidth << 8) & 0x0f00_0000)
}

fn relative_address(base: u32, offset: u32, data: u32) -> u32 {
    let base_extended = ((base & 0x000f_0000) << 8) | (data & 0x00ff_ffff);
    (offset.wrapping_add(base_extended)) & 0x0fff_ffff
}

fn wrap_coordinate(value: i32, size: u32, clamp: bool) -> Option<u32> {
    if size == 0 {
        return None;
    }
    if clamp {
        Some(value.clamp(0, size as i32 - 1) as u32)
    } else {
        Some(value.rem_euclid(size as i32) as u32)
    }
}

fn dxt_colors(first: u16, second: u16, dxt1: bool) -> [u32; 4] {
    let color0 = rgb565(first);
    let color1 = rgb565(second);
    let mut colors = [0; 4];
    colors[0] = color0 | 0xff00_0000;
    colors[1] = color1 | 0xff00_0000;
    if dxt1 && first <= second {
        colors[2] = average_color(color0, color1, 1, 1) | 0xff00_0000;
        colors[3] = 0;
    } else {
        colors[2] = average_color(color0, color1, 2, 1) | 0xff00_0000;
        colors[3] = average_color(color0, color1, 1, 2) | 0xff00_0000;
    }
    colors
}

fn rgb565(value: u16) -> u32 {
    u32::from(expand5(value))
        | (u32::from(expand6(value >> 5)) << 8)
        | (u32::from(expand5(value >> 11)) << 16)
}

fn average_color(first: u32, second: u32, first_weight: u32, second_weight: u32) -> u32 {
    let total = first_weight + second_weight;
    [0, 8, 16].into_iter().fold(0, |color, shift| {
        let value = (((first >> shift) & 0xff) * first_weight
            + ((second >> shift) & 0xff) * second_weight)
            / total;
        color | value << shift
    })
}

fn dxt_alpha_palette(first: u8, second: u8) -> [u8; 8] {
    let mut palette = [0; 8];
    palette[0] = first;
    palette[1] = second;
    if first > second {
        for (index, value) in palette.iter_mut().enumerate().skip(2) {
            *value = (((8 - index) as u16 * u16::from(first)
                + (index - 1) as u16 * u16::from(second))
                / 7) as u8;
        }
    } else {
        for (index, value) in palette.iter_mut().enumerate().skip(2).take(4) {
            *value = (((6 - index) as u16 * u16::from(first)
                + (index - 1) as u16 * u16::from(second))
                / 5) as u8;
        }
        palette[6] = 0;
        palette[7] = 255;
    }
    palette
}

fn read_vertex_color(memory: &Memory, address: u32, format: u32) -> Result<u32, MemoryFault> {
    Ok(match format {
        4 => {
            let value = memory.read_u16(address)?;
            u32::from(expand5(value))
                | (u32::from(expand6(value >> 5)) << 8)
                | (u32::from(expand5(value >> 11)) << 16)
                | 0xff00_0000
        }
        5 => {
            let value = memory.read_u16(address)?;
            u32::from(expand5(value))
                | (u32::from(expand5(value >> 5)) << 8)
                | (u32::from(expand5(value >> 10)) << 16)
                | if value & 0x8000 != 0 { 0xff00_0000 } else { 0 }
        }
        6 => {
            let value = memory.read_u16(address)?;
            u32::from(expand4(value))
                | (u32::from(expand4(value >> 4)) << 8)
                | (u32::from(expand4(value >> 8)) << 16)
                | (u32::from(expand4(value >> 12)) << 24)
        }
        7 => memory.read_u32(address)?,
        _ => u32::MAX,
    })
}

fn lerp_color(first: u32, second: u32, third: u32, w0: f32, w1: f32, w2: f32) -> u32 {
    [0, 8, 16, 24].into_iter().fold(0, |color, shift| {
        let value = (((first >> shift) & 0xff) as f32 * w0
            + ((second >> shift) & 0xff) as f32 * w1
            + ((third >> shift) & 0xff) as f32 * w2)
            .round()
            .clamp(0.0, 255.0) as u32;
        color | (value << shift)
    })
}

fn lerp_depth(first: u16, second: u16, third: u16, w0: f32, w1: f32, w2: f32) -> u16 {
    saturating_u16(f32::from(first) * w0 + f32::from(second) * w1 + f32::from(third) * w2)
}

fn checked_edram_span(base: u32, offset: u32, size: u32) -> Option<u32> {
    let address = base.checked_add(offset)?;
    let end = address.checked_add(size)?;
    if (0x0400_0000..0x0420_0000).contains(&base) && (address >= 0x0420_0000 || end > 0x0420_0000) {
        None
    } else {
        Some(address)
    }
}

fn compare_masked(value: u32, reference: u32, mask: u32, function: u32) -> bool {
    compare_values(value & mask, reference & mask, function)
}

fn compare_depth(value: u16, reference: u16, function: u32) -> bool {
    compare_values(u32::from(value), u32::from(reference), function)
}

fn compare_values(value: u32, reference: u32, function: u32) -> bool {
    match function & 7 {
        0 => false,
        1 => true,
        2 => value == reference,
        3 => value != reference,
        4 => value < reference,
        5 => value <= reference,
        6 => value > reference,
        7 => value >= reference,
        _ => false,
    }
}

fn blend_color(source: u32, destination: u32, state: u32, fixed_a: u32, fixed_b: u32) -> u32 {
    let source_factor = state & 0xf;
    let destination_factor = (state >> 4) & 0xf;
    let equation = (state >> 8) & 7;
    [0, 8, 16, 24].into_iter().fold(0, |color, shift| {
        let source_component = (source >> shift) & 0xff;
        let destination_component = (destination >> shift) & 0xff;
        let source_factor = blend_factor(source_factor, source, destination, shift, fixed_a, true);
        let destination_factor = blend_factor(
            destination_factor,
            source,
            destination,
            shift,
            fixed_b,
            false,
        );
        let value = match equation {
            0 => {
                (source_component * source_factor + destination_component * destination_factor)
                    / 255
            }
            1 => {
                (source_component * source_factor)
                    .saturating_sub(destination_component * destination_factor)
                    / 255
            }
            2 => {
                (destination_component * destination_factor)
                    .saturating_sub(source_component * source_factor)
                    / 255
            }
            3 => (source_component * source_factor / 255)
                .min(destination_component * destination_factor / 255),
            4 => (source_component * source_factor / 255)
                .max(destination_component * destination_factor / 255),
            _ => (source_component * source_factor / 255)
                .abs_diff(destination_component * destination_factor / 255),
        };
        color | value.min(255) << shift
    })
}

fn blend_factor(
    factor: u32,
    source: u32,
    destination: u32,
    shift: u32,
    fixed: u32,
    source_side: bool,
) -> u32 {
    let source_component = (source >> shift) & 0xff;
    let destination_component = (destination >> shift) & 0xff;
    let source_alpha = source >> 24 & 0xff;
    let destination_alpha = destination >> 24 & 0xff;
    let component = if source_side {
        match factor {
            0 => destination_component,
            1 => 255 - destination_component,
            2 => source_alpha,
            3 => 255 - source_alpha,
            4 => destination_alpha,
            5 => 255 - destination_alpha,
            6 => (source_alpha * 2).min(255),
            7 => ((255 - source_alpha) * 2).min(255),
            8 => (destination_alpha * 2).min(255),
            9 => ((255 - destination_alpha) * 2).min(255),
            10 => fixed_component(fixed, shift),
            _ => 255,
        }
    } else {
        match factor {
            0 => source_component,
            1 => 255 - source_component,
            2 => source_alpha,
            3 => 255 - source_alpha,
            4 => destination_alpha,
            5 => 255 - destination_alpha,
            6 => (source_alpha * 2).min(255),
            7 => ((255 - source_alpha) * 2).min(255),
            8 => (destination_alpha * 2).min(255),
            9 => ((255 - destination_alpha) * 2).min(255),
            10 => fixed_component(fixed, shift),
            _ => 255,
        }
    };
    component.min(255)
}

fn fixed_component(color: u32, shift: u32) -> u32 {
    if shift == 24 {
        255
    } else {
        (color >> shift) & 0xff
    }
}

fn combine_texture(texture: u32, vertex: u32, state: Texture) -> u32 {
    let function = state.function & 7;
    let use_texture_alpha = state.function & 0x100 != 0;
    let color_doubling = state.function & 0x1_0000 != 0;
    let texture_rgb = rgb_components(texture);
    let vertex_rgb = rgb_components(vertex);
    let mut rgb = [0; 3];
    match function {
        // the software sampler uses 256-based fixed-point arithmetic
        // here, including the +1 vertex bias. this matters for dark video
        // quads and for alpha-tested ui textures.
        0 => {
            rgb = componentwise_rgb(vertex_rgb, texture_rgb, |v, t| {
                (v + 1) * t * if color_doubling { 2 } else { 1 } / 256
            })
        }
        1 => {
            if use_texture_alpha {
                let texture_alpha = channel(texture, 24);
                let inverse = 255 - texture_alpha;
                rgb = componentwise_rgb(vertex_rgb, texture_rgb, |v, t| {
                    ((v + 1) * inverse + (t + 1) * texture_alpha)
                        / if color_doubling { 128 } else { 256 }
                });
            } else {
                rgb = scale_rgb(texture_rgb, color_doubling);
            }
        }
        2 => {
            let environment_rgb = rgb_components(state.environment_color);
            let denominator = if color_doubling { 128 } else { 256 };
            for index in 0..3 {
                let texture_component = texture_rgb[index];
                rgb[index] = (((255 - texture_component) * vertex_rgb[index]
                    + texture_component * environment_rgb[index]
                    + 255)
                    / denominator)
                    .min(255);
            }
        }
        3 => rgb = scale_rgb(texture_rgb, color_doubling),
        4..=7 => {
            rgb = componentwise_rgb(vertex_rgb, texture_rgb, |v, t| {
                (v + t) * if color_doubling { 2 } else { 1 }
            })
        }
        _ => unreachable!(),
    }
    let alpha = if use_texture_alpha {
        ((channel(vertex, 24) + 1) * channel(texture, 24) / 256).min(255)
    } else {
        channel(vertex, 24)
    };
    pack_rgb_alpha(rgb, alpha)
}

fn rgb_components(color: u32) -> [u32; 3] {
    [color & 0xff, (color >> 8) & 0xff, (color >> 16) & 0xff]
}

fn componentwise_rgb(
    first: [u32; 3],
    second: [u32; 3],
    operation: impl Fn(u32, u32) -> u32,
) -> [u32; 3] {
    [
        operation(first[0], second[0]).min(255),
        operation(first[1], second[1]).min(255),
        operation(first[2], second[2]).min(255),
    ]
}

fn scale_rgb(color: [u32; 3], doubled: bool) -> [u32; 3] {
    let scale = if doubled { 2 } else { 1 };
    [
        (color[0] * scale).min(255),
        (color[1] * scale).min(255),
        (color[2] * scale).min(255),
    ]
}

fn pack_rgb_alpha(rgb: [u32; 3], alpha: u32) -> u32 {
    rgb[0] | (rgb[1] << 8) | (rgb[2] << 16) | (alpha.min(255) << 24)
}

fn channel(color: u32, shift: u32) -> u32 {
    (color >> shift) & 0xff
}

fn read_palette_color(
    memory: &Memory,
    palette: u32,
    index: u32,
    format: u32,
) -> Result<u32, GpuError> {
    Ok(match format {
        0 => {
            let value = memory.read_u16(palette + index * 2)?;
            u32::from(expand5(value))
                | (u32::from(expand6(value >> 5)) << 8)
                | (u32::from(expand5(value >> 11)) << 16)
                | 0xff00_0000
        }
        1 => {
            let value = memory.read_u16(palette + index * 2)?;
            u32::from(expand5(value))
                | (u32::from(expand5(value >> 5)) << 8)
                | (u32::from(expand5(value >> 10)) << 16)
                | if value & 0x8000 != 0 { 0xff00_0000 } else { 0 }
        }
        2 => {
            let value = memory.read_u16(palette + index * 2)?;
            u32::from(expand4(value))
                | (u32::from(expand4(value >> 4)) << 8)
                | (u32::from(expand4(value >> 8)) << 16)
                | (u32::from(expand4(value >> 12)) << 24)
        }
        3 => memory.read_u32(palette + index * 4)?,
        _ => unreachable!(),
    })
}

fn edram_address(address: u32) -> u32 {
    if address < 0x0020_0000 {
        0x0400_0000 | address
    } else {
        address
    }
}

fn float24(value: u32) -> f32 {
    f32::from_bits((value & 0x00ff_ffff) << 8)
}

fn identity_matrix_43() -> [f32; 12] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
}

fn identity_matrix_44() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn expand4(value: u16) -> u8 {
    ((value & 0xf) * 0x11) as u8
}

fn expand5(value: u16) -> u8 {
    let value = (value & 0x1f) as u8;
    (value << 3) | (value >> 2)
}

fn expand6(value: u16) -> u8 {
    let value = (value & 0x3f) as u8;
    (value << 2) | (value >> 4)
}

fn trace_draw(draw: u64) -> bool {
    static RANGE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let &(start, end) = RANGE.get_or_init(|| {
        std::env::var("PSP_GE_TRACE_DRAWS")
            .ok()
            .and_then(|s| {
                let (a, b) = s.split_once(':')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((0, 0))
    });
    draw >= start && draw < end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clipping_gpu() -> Gpu {
        let mut gpu = Gpu::default();
        // center the psp viewport in its unsigned 12-bit coordinate range,
        // then subtract the drawing offset to get a small test framebuffer.
        for command in [0x42, 0x43, 0x44] {
            gpu.submit((command << 24) | (1.0f32.to_bits() >> 8));
        }
        for command in [0x45, 0x46] {
            gpu.submit((command << 24) | (2048.0f32.to_bits() >> 8));
        }
        gpu.submit(0x4700_0000 | (32768.0f32.to_bits() >> 8));
        gpu.submit(0x4c00_8000);
        gpu.submit(0x4d00_8000);
        gpu.submit(0x1c00_0001);
        gpu.submit(0x9d00_0008);
        gpu.submit(0xd200_0003);
        gpu.submit(0xd500_1c07);
        gpu
    }

    fn clip_vertex(gpu: &Gpu, clip: [f32; 4], color: u32, u: i32) -> Vertex {
        let (x, y, depth, inv_w, outside_range) = gpu.clip_to_screen(clip, false).unwrap();
        Vertex {
            clip: Some(clip),
            outside_range,
            x,
            y,
            depth,
            inv_w,
            color,
            u,
            v: 0,
        }
    }

    #[test]
    fn rgba_snapshot_rows_match_sampling_with_padding_and_edram_edges() {
        let gpu = Gpu::default();
        let mut memory = Memory::default();
        memory.map(0x0400_0000, 0x20_0000, true, false).unwrap();
        for address in [0x0400_0000, 0x041f_fff0] {
            for i in 0..4 {
                memory
                    .write_u32(address + i * 4, 0xff00_0000 | (i + 1))
                    .unwrap();
            }
            let texture = Texture {
                address,
                width: 2,
                height: 2,
                stride: 3,
                format: 3,
                ..gpu.current_texture()
            };
            let snapshot = snapshot_texture(&memory, texture).unwrap();
            let expected: Vec<u8> = (0..2)
                .flat_map(|y| (0..2).map(move |x| (x, y)))
                .flat_map(|(x, y)| {
                    texture
                        .sample(&memory, x, y)
                        .unwrap()
                        .unwrap_or(0)
                        .to_le_bytes()
                })
                .collect();
            assert_eq!(&*snapshot.pixels, expected);
        }
    }

    #[test]
    fn texture_cache_shares_pixels_across_draw_state_and_tracks_alias_writes() {
        let mut gpu = Gpu::default();
        let mut memory = Memory::default();
        memory.map(0x0800_0000, 4096, true, false).unwrap();
        memory.write_u32(0x0800_0000, 0xff00_00ff).unwrap();
        let mut texture = Texture {
            address: 0x8800_0000,
            width: 1,
            height: 1,
            stride: 1,
            format: 3,
            ..gpu.current_texture()
        };
        let first = gpu.snapshot_texture(&memory, texture).unwrap();
        texture.function = 3;
        texture.clamp_s = true;
        texture.environment_color = 123;
        let second = gpu.snapshot_texture(&memory, texture).unwrap();
        assert!(Arc::ptr_eq(&first.pixels, &second.pixels));
        assert_eq!(second.function, 3);
        assert!(second.clamp_s);
        assert_eq!(second.environment_color, 123);
        memory.write_u32(0x0800_0000, 0xffff_0000).unwrap();
        let third = gpu.snapshot_texture(&memory, texture).unwrap();
        assert_ne!(first.pixels, third.pixels);
        assert_eq!(&*third.pixels, &[0, 0, 255, 255]);
    }

    #[test]
    fn clear_primitives_do_not_sample_latched_textures() {
        let mut gpu = clipping_gpu();
        gpu.set_hardware_rendering(true);
        gpu.submit(0xd300_0101);
        let vertices = [[1.0, 1.0, 0.0, 1.0], [6.0, 6.0, 0.0, 1.0]]
            .map(|p| clip_vertex(&gpu, p, 0xff12_3456, 0));
        let mut memory = Memory::default();
        gpu.render_or_record(
            &mut memory,
            &vertices,
            6,
            gpu.current_texture(),
            true,
            false,
        )
        .unwrap();
        let commands = gpu.take_render_commands();
        assert_eq!(commands.len(), 1);
        assert!(commands[0].texture.is_none());
        assert!(!commands[0].textured);
    }

    #[test]
    fn rejects_invisible_primitives_in_software_and_recorded_streams() {
        for hardware in [false, true] {
            let mut gpu = clipping_gpu();
            gpu.set_hardware_rendering(hardware);
            let mut memory = Memory::default();
            memory.map(0x0400_0000, 256, true, false).unwrap();
            let visible = [
                [1.0, 1.0, 0.0, 1.0],
                [6.0, 1.0, 0.0, 1.0],
                [1.0, 6.0, 0.0, 1.0],
            ];
            let behind = visible.map(|p| [-p[0], -p[1], 0.0, -1.0]);
            let mut outside = visible;
            outside[0][0] = 5000.0;
            for (positions, should_draw) in [(behind, false), (outside, false), (visible, true)] {
                let vertices = positions.map(|p| clip_vertex(&gpu, p, u32::MAX, 0));
                gpu.render_or_record(
                    &mut memory,
                    &vertices,
                    3,
                    gpu.current_texture(),
                    false,
                    true,
                )
                .unwrap();
                if hardware {
                    assert_eq!(gpu.take_render_commands().len(), usize::from(should_draw));
                } else {
                    assert_eq!(gpu.last_render_pixels > 0, should_draw);
                }
            }
        }
    }

    #[test]
    fn clips_near_plane_before_dividing_and_preserves_provoking_color() {
        let gpu = clipping_gpu();
        let triangle = [
            clip_vertex(&gpu, [1.0, 1.0, -2.0, 1.0], 0xff00_00ff, 0),
            clip_vertex(&gpu, [7.0, 1.0, 0.0, 1.0], 0xff00_ff00, 12),
            clip_vertex(&gpu, [1.0, 7.0, 0.0, 1.0], 0xffff_0000, 24),
        ];
        let clipped = gpu.prepare_triangles(&triangle, 3, true);
        assert_eq!(clipped.len(), 6);
        assert!(
            clipped
                .iter()
                .all(|v| v.clip.unwrap()[2] >= -v.clip.unwrap()[3] && !v.outside_range)
        );
        assert!(clipped.iter().any(|v| (v.x, v.y, v.u) == (4, 1, 6)));
        assert!(clipped.iter().any(|v| (v.x, v.y, v.u) == (1, 4, 12)));
        assert!(
            gpu.prepare_triangles(&triangle, 3, false)
                .iter()
                .all(|v| v.color == triangle[2].color)
        );

        // xy outside the screen range cannot reject an original vertex
        // behind the near plane until it has been clipped.
        assert!(
            !gpu.clip_to_screen([5000.0, 1.0, -2.0, 1.0], false)
                .unwrap()
                .4
        );
        assert!(
            gpu.clip_to_screen([5000.0, 1.0, -1.0, 1.0], true)
                .unwrap()
                .4
        );
        let mut unclamped = gpu;
        unclamped.submit(0x1c00_0000);
        assert!(unclamped.prepare_triangles(&triangle, 3, true).is_empty());
    }

    #[test]
    fn triangle_strips_keep_winding_and_last_vertex_color_after_rejection() {
        let mut gpu = clipping_gpu();
        gpu.submit(0x1d00_0001);
        gpu.submit(0x9b00_0001); // Counterclockwise front faces in PSP coordinates.
        let strip = [(1.0, 1.0), (6.0, 1.0), (1.0, 6.0), (6.0, 6.0)]
            .map(|(x, y)| clip_vertex(&gpu, [x, y, 0.0, 1.0], x as u32 + y as u32 * 8, 0));
        let triangles = gpu.prepare_triangles(&strip, 4, false);
        assert_eq!(triangles.len(), 6);
        assert!(triangles[..3].iter().all(|v| v.color == strip[2].color));
        assert!(triangles[3..].iter().all(|v| v.color == strip[3].color));
        let mut invalid_first = strip;
        invalid_first[0].outside_range = true;
        assert_eq!(gpu.prepare_triangles(&invalid_first, 4, true).len(), 3);
        gpu.submit(0x9b00_0000);
        assert!(gpu.prepare_triangles(&strip, 4, true).is_empty());
        gpu.submit(0x1d00_0000);
        assert_eq!(gpu.prepare_triangles(&strip, 4, true).len(), 6);
    }

    #[test]
    fn decodes_bone_weights_and_applies_matrices_before_world_transform() {
        let mut memory = Memory::default();
        memory.map(0x2000, 128, true, false).unwrap();
        for format in 1..=3 {
            let mut gpu = clipping_gpu();
            let vertex_type = 0x180 | (format << 9) | (1 << 14);
            gpu.submit(0x1200_0000 | vertex_type);
            let layout = VertexLayout::new(vertex_type).unwrap();
            gpu.submit(0x2a00_0000);
            let mut bone = identity_matrix_43();
            for translation in [2.0f32, 6.0] {
                bone[9] = translation;
                for value in bone {
                    gpu.submit(0x2b00_0000 | (value.to_bits() >> 8));
                }
            }
            // uploading a world matrix must leave the bones intact.
            gpu.submit(0x3a00_0000);
            let mut world = identity_matrix_43();
            world[0] = 2.0;
            for value in world {
                gpu.submit(0x3b00_0000 | (value.to_bits() >> 8));
            }
            match format {
                1 => memory.write_bytes(0x2000, &[32, 96]).unwrap(),
                2 => {
                    memory.write_u16(0x2000, 8192).unwrap();
                    memory.write_u16(0x2002, 24576).unwrap();
                }
                3 => {
                    memory.write_u32(0x2000, 0.25f32.to_bits()).unwrap();
                    memory.write_u32(0x2004, 0.75f32.to_bits()).unwrap();
                }
                _ => unreachable!(),
            }
            let address = 0x2000 + layout.position_offset as u32;
            for (axis, value) in [1.0f32, 2.0, 0.0].into_iter().enumerate() {
                memory
                    .write_u32(address + axis as u32 * 4, value.to_bits())
                    .unwrap();
            }
            let vertex = gpu
                .read_vertex(&memory, 0x2000, layout, gpu.current_texture())
                .unwrap()
                .unwrap();
            assert_eq!((vertex.x, vertex.y), (12, 2), "weight format {format}");
            gpu.submit(0x1280_0000 | vertex_type);
            let through = gpu
                .read_vertex(&memory, 0x2000, layout, gpu.current_texture())
                .unwrap()
                .unwrap();
            assert_eq!((through.x, through.y), (1, 2));
        }
    }

    #[test]
    fn linear_indexed_eight_bit_textures_use_byte_stride_and_last_edram_byte() {
        let mut memory = Memory::default();
        memory.map(0x041f_fff0, 16, true, false).unwrap();
        memory.map(0x2000, 1024, true, false).unwrap();
        for index in 0..16 {
            memory.write_u8(0x041f_fff0 + index, index as u8).unwrap();
            memory
                .write_u32(0x2000 + index * 4, 0xff00_0000 | index)
                .unwrap();
        }
        let mut texture = Gpu::default().current_texture();
        texture.address = 0x041f_fff0;
        texture.format = 5;
        texture.stride = 8;
        texture.width = 8;
        texture.height = 2;
        texture.clut_address_low = 0x2000;
        texture.clut_format = 0xff03;
        for y in 0..2 {
            for x in 0..8 {
                assert_eq!(
                    texture.sample(&memory, x, y).unwrap(),
                    Some(0xff00_0000 | (y * 8 + x) as u32)
                );
            }
        }
        let snapshot = snapshot_texture(&memory, texture).unwrap();
        assert_eq!(&snapshot.pixels[60..64], &[15, 0, 0, 255]);
    }

    #[test]
    fn executes_primitive_and_end() {
        let mut memory = Memory::default();
        memory.map(0x1000, 12, true, false).unwrap();
        memory.write_u32(0x1000, 0x0400_0003).unwrap();
        memory.write_u32(0x1004, 0x0f00_0000).unwrap();
        memory.write_u32(0x1008, 0x0c00_0000).unwrap();
        let mut gpu = Gpu::default();
        let result = gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert_eq!(result.words, 3);
        assert_eq!(gpu.draw_calls, 1);
        assert_eq!(gpu.completed_lists, 1);
    }

    #[test]
    fn hardware_mode_records_ge_vertices_without_cpu_rasterization() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x100, true, false).unwrap();
        memory.map(0x2000, 0x100, true, false).unwrap();
        memory.map(0x0400_0000, 0x100, true, false).unwrap();
        for (index, (x, y)) in [(1u16, 1u16), (3, 1), (3, 3)].into_iter().enumerate() {
            let address = 0x2000 + index as u32 * 12;
            memory.write_u32(address, u32::MAX).unwrap();
            memory.write_u16(address + 4, x).unwrap();
            memory.write_u16(address + 6, y).unwrap();
            memory.write_u16(address + 8, 0).unwrap();
        }
        let commands = [
            0x1280_011c,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0004,
            0xd200_0003,
            0xd500_0c03,
            0x0403_0003,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.set_hardware_rendering(true);
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        let recorded = gpu.take_render_commands();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].vertices.len(), 3);
        assert_eq!(recorded[0].framebuffer_address, 0x0400_0000);
        assert_eq!(recorded[0].framebuffer_format, 3);
        assert_eq!(recorded[0].depth_address, 0x0400_0000);
        assert_eq!(recorded[0].clear_depth, 0);
        assert_eq!(gpu.last_render_pixels, 0);
        assert_eq!(memory.read_u32(0x0400_0000).unwrap(), 0);
    }

    #[test]
    fn default_vertex_color_uses_material_ambient() {
        let mut gpu = Gpu::default();
        assert_eq!(gpu.default_vertex_color(), u32::MAX);

        gpu.registers[0x55] = 0x0012_3456;
        gpu.registers[0x58] = 0x78;
        assert_eq!(gpu.default_vertex_color(), 0x7812_3456);
    }

    #[test]
    fn stops_at_stall_address() {
        let mut memory = Memory::default();
        memory.map(0x1000, 8, true, false).unwrap();
        memory.write_u32(0x1000, 0).unwrap();
        let mut gpu = Gpu::default();
        let result = gpu.execute_list(&mut memory, 0x1000, 0x1004).unwrap();
        assert!(result.stalled);
        assert_eq!(result.words, 1);
    }

    #[test]
    fn signal_handler_behaviors_raise_events_and_others_do_not() {
        // 0x01/0x02/0x03 run the
        // guest signal handler, while sync (0x08) and flow-control (0x10+)
        // forms never do.
        let mut memory = Memory::default();
        memory.map(0x1000, 32, true, false).unwrap();
        for (index, word) in [
            0x0e01_0034u32, // suspend, token 0x34
            0x0e08_0078u32, // sync: ignored
            0x0e02_00abu32, // continue, token 0xab
            0x0e10_00cdu32, // jump: ignored
            0x0f00_0000u32, // FINISH
            0x0c00_0000u32, // END
        ]
        .into_iter()
        .enumerate()
        {
            memory.write_u32(0x1000 + index as u32 * 4, word).unwrap();
        }
        let mut gpu = Gpu::default();
        let result = gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert!(result.finished);
        assert_eq!(
            result
                .signals
                .iter()
                .map(|signal| (signal.behavior, signal.token))
                .collect::<Vec<_>>(),
            [(1, 0x34), (2, 0xab)],
        );
    }

    #[test]
    fn through_mode_keeps_signed_8_bit_positions() {
        let mut memory = Memory::default();
        memory.map(0x2000, 16, true, false).unwrap();
        memory.write_u8(0x2000, 0xfe).unwrap(); // -2
        memory.write_u8(0x2001, 0x05).unwrap(); // 5
        memory.write_u8(0x2002, 0x7f).unwrap(); // 127
        let mut gpu = Gpu::default();
        gpu.registers[0x12] = 0x0080_0080;
        let layout = VertexLayout::new(0x0080_0080).unwrap();
        let texture = Texture {
            address: 0,
            stride: 0,
            width: 0,
            height: 0,
            format: 3,
            clut_address_low: 0,
            clut_address_high: 0,
            clut_format: 0,
            function: 0,
            environment_color: 0,
            clamp_s: true,
            clamp_t: true,
            swizzled: false,
        };
        let vertex = gpu
            .read_vertex(&memory, 0x2000, layout, texture)
            .unwrap()
            .expect("through vertex decodes");
        assert_eq!((vertex.x, vertex.y, vertex.depth), (-2, 5, 127));
    }

    #[test]
    fn ge_relative_addresses_follow_base_and_offset_packing() {
        assert_eq!(
            relative_address(0x000b_1234, 0x0000_2000, 0x0000_1234),
            0x0b00_3234
        );
        assert_eq!(texture_address(0x08ab_cdef, 0x000a_0200), 0x0aab_cde0);
        assert_eq!(texture_dimension(0x0000_000f, false), 32_768);
        assert_eq!(texture_dimension(0x0000_0f00, true), 32_768);
    }

    #[test]
    fn packed_16_bit_uv_and_position_vertices_keep_ten_byte_stride() {
        let layout = VertexLayout::new(0x0080_0102).unwrap();
        assert_eq!(layout.texture_offset, Some(0));
        assert_eq!(layout.position_offset, 4);
        assert_eq!(layout.stride_bytes, 10);
    }

    #[test]
    fn converts_8888_framebuffer() {
        let mut memory = Memory::default();
        memory.map(0x2000, 4, true, false).unwrap();
        memory.write_u32(0x2000, 0x8040_2010).unwrap();
        assert_eq!(
            Gpu::framebuffer_rgba(&memory, 0x2000, 1, 1, 1, 3).unwrap(),
            [0x10, 0x20, 0x40, 0x80]
        );
    }

    #[test]
    fn samples_native_texture_formats_and_wraps_coordinates() {
        let mut memory = Memory::default();
        memory.map(0x2000, 32, true, false).unwrap();
        memory.write_u16(0x2000, 0x001f).unwrap();
        memory.write_u16(0x2002, 0x07e0).unwrap();
        memory.write_u16(0x2004, 0x001f).unwrap();
        memory.write_u32(0x2008, 0xff11_2233).unwrap();
        let texture = Texture {
            address: 0x2000,
            stride: 2,
            width: 2,
            height: 1,
            format: 0,
            clut_address_low: 0,
            clut_address_high: 0,
            clut_format: 0,
            function: 3,
            environment_color: 0,
            clamp_s: false,
            clamp_t: true,
            swizzled: false,
        };
        assert_eq!(texture.sample(&memory, 0, 0).unwrap(), Some(0xff0000ff));
        assert_eq!(texture.sample(&memory, 2, 0).unwrap(), Some(0xff0000ff));

        let mut texture = texture;
        texture.format = 3;
        texture.address = 0x2008;
        texture.stride = 1;
        texture.width = 1;
        assert_eq!(texture.sample(&memory, 0, 0).unwrap(), Some(0xff112233));
    }

    #[test]
    fn samples_swizzled_texture_blocks() {
        let mut memory = Memory::default();
        memory.map(0x3000, 128, true, false).unwrap();
        memory.write_u32(0x3000 + 4 * 16, 0xff00_00ff).unwrap();
        let texture = Texture {
            address: 0x3000,
            stride: 4,
            width: 4,
            height: 8,
            format: 3,
            clut_address_low: 0,
            clut_address_high: 0,
            clut_format: 0,
            function: 3,
            environment_color: 0,
            clamp_s: true,
            clamp_t: true,
            swizzled: true,
        };
        assert_eq!(texture.sample(&memory, 0, 4).unwrap(), Some(0xff0000ff));
    }

    #[test]
    fn samples_dxt1_and_dxt5_blocks() {
        let mut memory = Memory::default();
        memory.map(0x4000, 32, true, false).unwrap();
        memory.write_u16(0x4000, 0x001f).unwrap();
        memory.write_u16(0x4002, 0x07e0).unwrap();
        memory.write_u32(0x4004, 0x0000_0200).unwrap();
        let texture = Texture {
            address: 0x4000,
            stride: 4,
            width: 4,
            height: 4,
            format: 8,
            clut_address_low: 0,
            clut_address_high: 0,
            clut_format: 0,
            function: 3,
            environment_color: 0,
            clamp_s: true,
            clamp_t: true,
            swizzled: false,
        };
        assert_eq!(texture.sample(&memory, 0, 0).unwrap(), Some(0xff0000ff));
        assert_eq!(texture.sample(&memory, 0, 1).unwrap(), Some(0xff007f7f));

        memory.write_u8(0x4010, 0x00).unwrap();
        memory.write_u8(0x4011, 0xff).unwrap();
        memory.write_bytes(0x4012, &[0; 6]).unwrap();
        memory.write_u8(0x4012, 0x38).unwrap();
        memory.write_u16(0x4018, 0x001f).unwrap();
        memory.write_u16(0x401a, 0x001f).unwrap();
        memory.write_u32(0x401c, 0).unwrap();
        let mut texture = texture;
        texture.address = 0x4010;
        texture.format = 10;
        assert_eq!(texture.sample(&memory, 0, 0).unwrap(), Some(0x000000ff));
        assert_eq!(texture.sample(&memory, 1, 0).unwrap(), Some(0xff0000ff));
    }

    #[test]
    fn texture_functions_match_fixed_point_rules() {
        let state = Texture {
            address: 0,
            stride: 1,
            width: 1,
            height: 1,
            format: 3,
            clut_address_low: 0,
            clut_address_high: 0,
            clut_format: 0,
            function: 0x100,
            environment_color: 0x8040_2010,
            clamp_s: true,
            clamp_t: true,
            swizzled: false,
        };
        let texture = 0x8040_2010;
        let vertex = 0x80c0_8040;
        assert_eq!(combine_texture(texture, vertex, state), 0x40_30_10_04);

        let mut state = state;
        state.function = 0x1_0000 | 3;
        assert_eq!(combine_texture(texture, vertex, state), 0x80_80_40_20);

        state.function = 1;
        assert_eq!(combine_texture(texture, vertex, state), 0x80_40_20_10);

        state.function = 0x100 | 1;
        assert_eq!(combine_texture(texture, vertex, state), 0x40_80_50_28);

        state.function = 0x100 | 2;
        assert_eq!(combine_texture(texture, vertex, state), 0x40_a0_74_3d);
    }

    #[test]
    fn rasterizes_triangle_fans() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x100, true, false).unwrap();
        memory.map(0x2000, 0x100, true, false).unwrap();
        memory.map(0x0400_0000, 0x100, true, false).unwrap();
        for (index, (x, y)) in [(1u16, 1u16), (3, 1), (3, 3), (1, 3)]
            .into_iter()
            .enumerate()
        {
            let address = 0x2000 + index as u32 * 12;
            memory.write_u32(address, 0xffff_ffff).unwrap();
            memory.write_u16(address + 4, x).unwrap();
            memory.write_u16(address + 6, y).unwrap();
            memory.write_u16(address + 8, 0).unwrap();
        }
        let commands = [
            0x1280_011c,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0004,
            0xd200_0003,
            0xd500_0c03,
            0x0405_0004,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert!(
            gpu.last_render_pixels > 0,
            "pixels={}",
            gpu.last_render_pixels
        );
        let pixels = (0..16)
            .map(|index| memory.read_u32(0x0400_0000 + index * 4).unwrap())
            .collect::<Vec<_>>();
        assert!(pixels.contains(&u32::MAX));
    }

    #[test]
    fn rasterizes_bezier_and_spline_patches() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x100, true, false).unwrap();
        memory.map(0x2000, 0x200, true, false).unwrap();
        memory.map(0x0400_0000, 0x1000, true, false).unwrap();
        for row in 0..4u16 {
            for column in 0..4u16 {
                let address = 0x2000 + u32::from(row * 4 + column) * 12;
                memory.write_u32(address, u32::MAX).unwrap();
                memory.write_u16(address + 4, 1 + column * 2).unwrap();
                memory.write_u16(address + 6, 1 + row * 2).unwrap();
                memory.write_u16(address + 8, 0).unwrap();
            }
        }
        let commands = [
            0x1280_011c,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0008,
            0xd200_0003,
            0xd500_1c07,
            0x3600_0202,
            0x3700_0000,
            0x0500_0404,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert!(gpu.last_render_pixels > 0);

        let commands = [
            0x1280_011c,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0008,
            0xd200_0003,
            0xd500_1c07,
            0x3600_0202,
            0x3700_0000,
            0x0600_0404,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert!(gpu.last_render_pixels > 0);
    }

    #[test]
    fn advances_internal_vertex_cursor_between_primitives() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x100, true, false).unwrap();
        memory.map(0x2000, 0x100, true, false).unwrap();
        memory.map(0x0400_0000, 0x100, true, false).unwrap();
        for (index, (x, y)) in [(0u16, 0u16), (1, 1), (2, 2), (3, 3)]
            .into_iter()
            .enumerate()
        {
            let address = 0x2000 + index as u32 * 12;
            memory.write_u32(address, 0xffff_ffff).unwrap();
            memory.write_u16(address + 4, x).unwrap();
            memory.write_u16(address + 6, y).unwrap();
            memory.write_u16(address + 8, 0).unwrap();
        }
        let commands = [
            0x1280_011c,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0004,
            0xd200_0003,
            0xd500_0c03,
            0x0406_0002,
            0x0406_0002,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert_eq!(
            memory.read_u32(0x0400_0000 + (2 * 4 + 2) * 4).unwrap(),
            u32::MAX
        );
    }

    #[test]
    fn transforms_float_vertices_into_the_viewport() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x100, true, false).unwrap();
        memory.map(0x2000, 0x100, true, false).unwrap();
        memory.map(0x0400_0000, 0x100, true, false).unwrap();
        for (index, position) in [
            (-1.0f32, -1.0f32, 0.0f32),
            (1.0f32, -1.0f32, 0.0f32),
            (0.0f32, 1.0f32, 0.0f32),
        ]
        .into_iter()
        .enumerate()
        {
            let address = 0x2000 + index as u32 * 12;
            memory.write_u32(address, position.0.to_bits()).unwrap();
            memory.write_u32(address + 4, position.1.to_bits()).unwrap();
            memory.write_u32(address + 8, position.2.to_bits()).unwrap();
        }
        let float24 = |value: f32| value.to_bits() >> 8;
        let commands = [
            0x1200_0180,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0004,
            0xd200_0003,
            0xd500_0c03,
            0x4240_0000 | float24(2.0),
            0x4340_0000 | float24(2.0),
            0x4540_0000 | float24(2.0),
            0x4640_0000 | float24(2.0),
            0x0403_0003,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert_eq!(memory.read_u32(0x0400_0000 + 40).unwrap(), u32::MAX);
        assert!(gpu.last_render_pixels > 0);
    }

    #[test]
    fn interpolates_texture_coordinates_in_projective_space() {
        let perspective = perspective_interpolate([0, 100, 0], [1.0, 0.5, 1.0], [0.25, 0.5, 0.25]);
        let affine = 100.0 * 0.5;
        assert!((perspective - 33.333_332).abs() < 0.001);
        assert!((perspective - affine).abs() > 10.0);
        assert!(
            (perspective_interpolate([1, 2, 3], [0.0, 0.0, 0.0], [0.2, 0.3, 0.5]) - 2.3).abs()
                < 0.001
        );
    }

    #[test]
    fn clears_color_and_depth_inside_the_scissor() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x100, true, false).unwrap();
        memory.map(0x2000, 0x40, true, false).unwrap();
        memory.map(0x0400_0000, 0x1000, true, false).unwrap();
        for (address, color, z, x, y) in [
            (0x2000, 0xff12_3456, 0x1234u16, 0, 0),
            (0x200c, 0xff12_3456, 0x1234u16, 2, 2),
        ] {
            memory.write_u32(address, color).unwrap();
            memory.write_u16(address + 4, x).unwrap();
            memory.write_u16(address + 6, y).unwrap();
            memory.write_u16(address + 8, z).unwrap();
        }
        let commands = [
            0x1280_011c,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0004,
            0xd200_0003,
            0x9e00_0040,
            0x9f00_0004,
            0xd400_0000,
            0xd500_0401,
            0xd300_0701,
            0x0406_0002,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert_eq!(memory.read_u32(0x0400_0000).unwrap(), 0xff12_3456);
        assert_eq!(memory.read_u16(0x0400_0040).unwrap(), 0x1234);
        assert_eq!(gpu.last_render_pixels, 4);
    }

    #[test]
    fn depth_test_keeps_nearer_rectangle() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x100, true, false).unwrap();
        memory.map(0x2000, 0x80, true, false).unwrap();
        memory.map(0x0400_0000, 0x1000, true, false).unwrap();
        for y in 0..2u32 {
            for x in 0..2u32 {
                memory
                    .write_u16(0x0400_0040 + (y * 4 + x) * 2, u16::MAX)
                    .unwrap();
            }
        }
        for (address, color, z, x, y) in [
            (0x2000, 0xff00_00ff, 0x1000u16, 0, 0),
            (0x200c, 0xff00_00ff, 0x1000u16, 2, 2),
            (0x2040, 0xffff_0000, 0x2000u16, 0, 0),
            (0x204c, 0xffff_0000, 0x2000u16, 2, 2),
        ] {
            memory.write_u32(address, color).unwrap();
            memory.write_u16(address + 4, x).unwrap();
            memory.write_u16(address + 6, y).unwrap();
            memory.write_u16(address + 8, z).unwrap();
        }
        let commands = [
            0x1280_011c,
            0x0100_2000,
            0x9c00_0000,
            0x9d00_0004,
            0xd200_0003,
            0x9e00_0040,
            0x9f00_0004,
            0xd400_0000,
            0xd500_0401,
            0x2300_0001,
            0xde00_0004,
            0x0406_0002,
            0x0100_2040,
            0x0406_0002,
            0x0c00_0000,
        ];
        for (index, command) in commands.into_iter().enumerate() {
            memory
                .write_u32(0x1000 + index as u32 * 4, command)
                .unwrap();
        }
        let mut gpu = Gpu::default();
        gpu.execute_list(&mut memory, 0x1000, 0).unwrap();
        assert_eq!(memory.read_u32(0x0400_0000).unwrap(), 0xff00_00ff);
        assert_eq!(memory.read_u16(0x0400_0040).unwrap(), 0x1000);
    }

    fn extreme_vertex(x: i32, y: i32) -> Vertex {
        // through-mode vertices skip range checks (only non-finite
        // coordinates are flagged), so i32-extreme screen coordinates reach
        // the edge math directly.
        Vertex {
            clip: None,
            outside_range: false,
            color: u32::MAX,
            u: 0,
            v: 0,
            inv_w: 1.0,
            x,
            y,
            depth: 0,
        }
    }

    #[test]
    fn extreme_coordinates_do_not_overflow_triangle_area() {
        let gpu = clipping_gpu();
        // i32-wide coordinate differences overflow i64 products (up to
        // ~1.8e19) and panic in debug builds; the area must use i128.
        let huge = [
            extreme_vertex(i32::MIN, 0),
            extreme_vertex(i32::MAX, 0),
            extreme_vertex(0, i32::MAX),
        ];
        assert_eq!(gpu.prepare_triangles(&huge, 3, true).len(), 3);
    }

    #[test]
    fn extreme_coordinates_do_not_overflow_rasterizer_edges() {
        let mut memory = Memory::default();
        memory.map(0x0400_0000, 0x1000, true, false).unwrap();
        let mut pixels = 0u64;
        let mut target = RenderTarget {
            memory: &mut memory,
            framebuffer: 0x0400_0000,
            stride: 8,
            format: 3,
            min_x: 0,
            min_y: 0,
            max_x: 7,
            max_y: 7,
            clear_mode: 0,
            depth_address: 0,
            depth_stride: 0,
            depth_test: false,
            depth_write: false,
            depth_function: 1,
            alpha_test: false,
            alpha_function: 0,
            alpha_reference: 0,
            alpha_mask: 0,
            color_test: false,
            color_function: 0,
            color_reference: 0,
            color_mask: 0,
            blend: false,
            blend_state: 0,
            blend_fixed_a: 0,
            blend_fixed_b: 0,
            color_mask_rgb: 0,
            color_mask_alpha: 0,
            pixels_written: &mut pixels,
        };
        // the bounding box clamps to the scissor, but every covered pixel
        // still evaluates edges against the extreme vertices.
        let huge = [
            extreme_vertex(i32::MIN, i32::MIN),
            extreme_vertex(i32::MAX, 0),
            extreme_vertex(0, i32::MAX),
        ];
        target.fill_triangle(&huge, true).unwrap();
    }
}
