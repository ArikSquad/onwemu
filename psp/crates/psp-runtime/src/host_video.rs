use anyhow::{Context, Result};
use psp_gpu::{RenderCommand, RenderTexture, RenderVertex};
use psp_input::InputState;
use sdl3::event::{Event, WindowEvent};
use sdl3::gpu::{
    BlendFactor, BlendOp, BlitInfo, Buffer, BufferBinding, BufferUsageFlags, ColorComponentFlags,
    ColorTargetBlendState, ColorTargetDescription, ColorTargetInfo, CompareOp, DepthStencilState,
    DepthStencilTargetInfo, Device, Filter, GraphicsPipeline, GraphicsPipelineTargetInfo, LoadOp,
    PrimitiveType, RasterizerState, Sampler, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode, Shader, ShaderFormat, ShaderStage, StoreOp, Texture, TextureCreateInfo,
    TextureFormat, TextureRegion, TextureSamplerBinding, TextureTransferInfo, TextureType,
    TextureUsage, TransferBuffer, TransferBufferLocation, TransferBufferUsage, VertexAttribute,
    VertexBufferDescription, VertexElementFormat, VertexInputState, Viewport,
};
use sdl3::keyboard::Scancode;
use sdl3::pixels::Color;
use sdl3::rect::Rect;
use std::borrow::Cow;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

const INITIAL_VERTEX_CAPACITY: usize = 64 * 1024;
const PSP_EDRAM_BASE: u32 = 0x0400_0000;
// psp ge textures use power-of-two backing dimensions.  a 512-wide surface
// with a 272-row display viewport is therefore still sampled as a 512x512
// texture when a game renders it to edram and uses it in a later pass.
const PSP_TEXTURE_BACKING_DIMENSION: u32 = 512;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HostVertex {
    position: [f32; 4],
    color: [f32; 4],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FragmentUniforms {
    function: u32,
    flags: u32,
    alpha_test: u32,
    alpha_function: u32,
    alpha_reference: u32,
    alpha_mask: u32,
    color_test: u32,
    color_function: u32,
    color_reference: u32,
    color_mask: u32,
    _padding: [u32; 2],
    environment: [f32; 4],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PipelineKey {
    depth_test: bool,
    depth_write: bool,
    depth_function: u32,
    blend_state: u32,
    color_write_mask: u32,
}

struct CachedTexture {
    texture: Texture<'static>,
    _transfer: TransferBuffer,
    content_hash: u64,
    width: u32,
    height: u32,
}

struct GpuRenderTarget {
    texture: Texture<'static>,
    logical_width: u32,
    logical_height: u32,
    storage_width: u32,
    storage_height: u32,
    initialized: bool,
}

struct GpuDepthTarget {
    texture: Texture<'static>,
    initialized: bool,
}

struct PreparedDraw {
    first_vertex: usize,
    vertex_count: usize,
    texture: Texture<'static>,
    sampler: usize,
    uniforms: FragmentUniforms,
    pipeline: PipelineKey,
    framebuffer_address: u32,
    depth_address: u32,
    /// SDL blend-constant color for fixed-factor blends (PSP factor 10).
    /// `None` when neither blend side uses the constant color.
    blend_constants: Option<[f32; 4]>,
    scissor: Rect,
}

struct PendingTextureUpload {
    texture: Texture<'static>,
    transfer: TransferBuffer,
    width: u32,
    height: u32,
}

/// Linux host window and GPU renderer.
///
/// SDL3 owns the Wayland/X11 window and its GPU device. GE lists are decoded by
/// `psp-gpu`, but their triangles and texture uploads are submitted here to SDL
/// GPU. The CPU rasterizer is reserved for headless/reference runs.
pub struct SdlHost {
    _context: sdl3::Sdl,
    window: sdl3::video::Window,
    event_pump: sdl3::EventPump,
    gpu: Device,
    frame_texture: Texture<'static>,
    video_texture: Texture<'static>,
    video_transfer: TransferBuffer,
    vertex_buffer: Buffer,
    vertex_transfer: TransferBuffer,
    vertex_capacity: usize,
    white_texture: Texture<'static>,
    white_transfer: TransferBuffer,
    white_uploaded: bool,
    samplers: Vec<Sampler>,
    vertex_shader: Shader,
    fragment_shader: Shader,
    pipelines: HashMap<PipelineKey, GraphicsPipeline>,
    render_targets: HashMap<u32, GpuRenderTarget>,
    depth_targets: HashMap<u32, GpuDepthTarget>,
    textures: HashMap<u32, CachedTexture>,
    width: u32,
    height: u32,
    scale: u32,
    render_width: u32,
    render_height: u32,
    open: bool,
    render_initialized: bool,
}

impl SdlHost {
    /// Create a host window and GPU render targets at the requested scale.
    pub fn new(scale: u32, width: u32, height: u32) -> Result<Self> {
        let context = sdl3::init().context("cannot initialize SDL3")?;
        let video = context.video().context("cannot initialize SDL3 video")?;
        let window = video
            .window(
                "psp-rs",
                width.saturating_mul(scale),
                height.saturating_mul(scale),
            )
            .position_centered()
            .resizable()
            .build()
            .map_err(|error| anyhow::anyhow!("cannot create SDL3 window: {error}"))?;

        // keep native video frames at psp resolution. ge draws use a scaled
        // offscreen target, and video frames are enlarged by a gpu blit into
        // that same target. this makes --scale an internal render-resolution
        // setting instead of only making the window larger.
        let render_width = width.saturating_mul(scale);
        let render_height = height.saturating_mul(scale);

        let gpu = Device::new(ShaderFormat::SPIRV, cfg!(debug_assertions))
            .context("cannot create SDL3 GPU device")?
            .with_window(&window)
            .context("cannot claim SDL3 window for GPU device")?;
        let frame_texture = gpu
            .create_texture(
                TextureCreateInfo::new()
                    .with_type(TextureType::_2D)
                    .with_format(TextureFormat::R8g8b8a8Unorm)
                    .with_usage(TextureUsage::SAMPLER | TextureUsage::COLOR_TARGET)
                    .with_width(render_width)
                    .with_height(render_height)
                    .with_layer_count_or_depth(1)
                    .with_num_levels(1),
            )
            .context("cannot create SDL3 PSP render target")?;
        let video_texture = gpu
            .create_texture(
                TextureCreateInfo::new()
                    .with_type(TextureType::_2D)
                    .with_format(TextureFormat::R8g8b8a8Unorm)
                    .with_usage(TextureUsage::SAMPLER)
                    .with_width(width)
                    .with_height(height)
                    .with_layer_count_or_depth(1)
                    .with_num_levels(1),
            )
            .context("cannot create SDL3 native video texture")?;
        let video_transfer = gpu
            .create_transfer_buffer()
            .with_usage(TransferBufferUsage::UPLOAD)
            .with_size(width.saturating_mul(height).saturating_mul(4))
            .build()
            .context("cannot create SDL3 native video upload buffer")?;
        let vertex_capacity = INITIAL_VERTEX_CAPACITY;
        let vertex_buffer = gpu
            .create_buffer()
            .with_usage(BufferUsageFlags::VERTEX)
            .with_size((vertex_capacity * size_of::<HostVertex>()) as u32)
            .build()
            .context("cannot create SDL3 GE vertex buffer")?;
        let vertex_transfer = gpu
            .create_transfer_buffer()
            .with_usage(TransferBufferUsage::UPLOAD)
            .with_size((vertex_capacity * size_of::<HostVertex>()) as u32)
            .build()
            .context("cannot create SDL3 GE vertex upload buffer")?;
        let white_texture = gpu
            .create_texture(
                TextureCreateInfo::new()
                    .with_type(TextureType::_2D)
                    .with_format(TextureFormat::R8g8b8a8Unorm)
                    .with_usage(TextureUsage::SAMPLER)
                    .with_width(1)
                    .with_height(1)
                    .with_layer_count_or_depth(1)
                    .with_num_levels(1),
            )
            .context("cannot create SDL3 white fallback texture")?;
        let white_transfer = gpu
            .create_transfer_buffer()
            .with_usage(TransferBufferUsage::UPLOAD)
            .with_size(4)
            .build()
            .context("cannot create SDL3 white texture upload buffer")?;
        let samplers = (0..4)
            .map(|index| {
                let mode = |clamp| {
                    if clamp {
                        SamplerAddressMode::ClampToEdge
                    } else {
                        SamplerAddressMode::Repeat
                    }
                };
                gpu.create_sampler(
                    SamplerCreateInfo::new()
                        .with_min_filter(Filter::Nearest)
                        .with_mag_filter(Filter::Nearest)
                        .with_mipmap_mode(SamplerMipmapMode::Nearest)
                        .with_address_mode_u(mode(index & 1 != 0))
                        .with_address_mode_v(mode(index & 2 != 0))
                        .with_address_mode_w(SamplerAddressMode::ClampToEdge),
                )
                .context("cannot create SDL3 texture sampler")
            })
            .collect::<Result<Vec<_>>>()?;
        let vertex_shader = gpu
            .create_shader()
            .with_code(
                ShaderFormat::SPIRV,
                include_bytes!(concat!(env!("OUT_DIR"), "/psp_vertex.spv")),
                ShaderStage::Vertex,
            )
            .with_entrypoint(c"vertex_main")
            .build()
            .context("cannot create SDL3 GE vertex shader")?;
        let fragment_shader = gpu
            .create_shader()
            .with_code(
                ShaderFormat::SPIRV,
                include_bytes!(concat!(env!("OUT_DIR"), "/psp_fragment.spv")),
                ShaderStage::Fragment,
            )
            .with_samplers(1)
            .with_uniform_buffers(1)
            .with_entrypoint(c"fragment_main")
            .build()
            .context("cannot create SDL3 GE fragment shader")?;
        let event_pump = context
            .event_pump()
            .context("cannot create SDL3 event pump")?;

        tracing::info!(
            backend = "SDL3 GPU",
            width,
            height,
            scale,
            render_width,
            render_height,
            "hardware host video initialized"
        );
        Ok(Self {
            _context: context,
            window,
            event_pump,
            gpu,
            frame_texture,
            video_texture,
            video_transfer,
            vertex_buffer,
            vertex_transfer,
            vertex_capacity,
            white_texture,
            white_transfer,
            white_uploaded: false,
            samplers,
            vertex_shader,
            fragment_shader,
            pipelines: HashMap::new(),
            render_targets: HashMap::new(),
            depth_targets: HashMap::new(),
            textures: HashMap::new(),
            width,
            height,
            scale,
            render_width,
            render_height,
            open: true,
            render_initialized: false,
        })
    }

    /// Return whether the host window is still open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Poll host events and update the guest controller state.
    pub fn poll(&mut self, input: &mut InputState) -> u32 {
        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::Window {
                    win_event: WindowEvent::CloseRequested,
                    ..
                } => self.open = false,
                _ => {}
            }
        }
        if !self.open {
            input.update_buttons(0);
            return 0;
        }
        let keyboard = self.event_pump.keyboard_state();
        let buttons = super::host_button_mask(|key| keyboard.is_scancode_pressed(key));
        input.update_buttons(buttons);
        input.analog_x = super::analog_axis(
            keyboard.is_scancode_pressed(Scancode::J),
            keyboard.is_scancode_pressed(Scancode::L),
        );
        input.analog_y = super::analog_axis(
            keyboard.is_scancode_pressed(Scancode::I),
            keyboard.is_scancode_pressed(Scancode::K),
        );
        buttons
    }

    /// Submit decoded GE commands to the persistent PSP render target.
    pub fn render_commands(&mut self, commands: &[RenderCommand]) -> Result<()> {
        if commands.is_empty() {
            return Ok(());
        }
        let render_started = Instant::now();
        let mut vertices = Vec::new();
        let mut draws = Vec::new();
        let mut pending_uploads = Vec::new();
        // rendertexture snapshots from one ge list share their arc-backed
        // pixels.  avoid hashing the same 256/512px texture once per draw;
        // the pointer is only used for this call, while the source arc keeps
        // the pixels alive for the whole batch.
        let mut prepared_texture_cache: HashMap<usize, Texture<'static>> = HashMap::new();
        let mut white_upload = false;
        for command in commands {
            let expanded = expand_vertices(command);
            if expanded.is_empty() {
                continue;
            }
            let required_height = expanded
                .iter()
                .map(|vertex| vertex.y.max(0) as u32)
                .fold(0, u32::max)
                .saturating_add(1)
                .max(self.height)
                .max(command.scissor[3].saturating_add(1));
            self.ensure_render_target(
                command.framebuffer_address,
                command.framebuffer_width,
                required_height,
            )?;
            self.ensure_depth_target(command.depth_address, command.depth_width, required_height)?;
            let target_width = self
                .render_targets
                .get(&command.framebuffer_address)
                .map_or(self.width, |target| target.logical_width);
            let target_height = self
                .render_targets
                .get(&command.framebuffer_address)
                .map_or(self.height, |target| target.logical_height);
            let (texture, sampler, uniforms) = if let Some(source) = command.texture.as_ref() {
                let aliased_target = (normalize_surface_address(source.address)
                    != command.framebuffer_address)
                    .then(|| {
                        self.render_targets
                            .get(&normalize_surface_address(source.address))
                            .filter(|target| {
                                source.width <= target.storage_width
                                    && source.height <= target.storage_height
                            })
                            .map(|target| target.texture.clone())
                    })
                    .flatten();
                let (texture, upload) = if let Some(texture) = aliased_target {
                    // keep framebuffer-backed textures on the gpu. this is
                    // the common psp render-to-texture path and avoids
                    // sampling stale guest ram after a target switch.
                    (texture, None)
                } else if let Some(texture) = prepared_texture_cache
                    .get(&(source.pixels.as_ptr() as usize))
                    .cloned()
                {
                    (texture, None)
                } else {
                    let prepared = self.prepare_texture(source)?;
                    prepared_texture_cache
                        .insert(source.pixels.as_ptr() as usize, prepared.0.clone());
                    prepared
                };
                if let Some(upload) = upload {
                    pending_uploads.push(upload);
                }
                (
                    texture,
                    sampler_index(source.clamp_s, source.clamp_t),
                    fragment_uniforms(Some(source), command),
                )
            } else {
                if !self.white_uploaded {
                    white_upload = true;
                }
                (
                    self.white_texture.clone(),
                    0,
                    fragment_uniforms(None, command),
                )
            };
            let first_vertex = vertices.len();
            vertices.extend(expanded.iter().map(|vertex| {
                host_vertex(
                    *vertex,
                    target_width,
                    target_height,
                    self.scale,
                    command.texture.as_ref(),
                )
            }));
            let last_x = command.scissor[2].min(target_width.saturating_sub(1));
            let last_y = command.scissor[3].min(target_height.saturating_sub(1));
            let first_x = command.scissor[0].min(last_x);
            let first_y = command.scissor[1].min(last_y);
            let pipeline = pipeline_key(command);
            draws.push(PreparedDraw {
                first_vertex,
                vertex_count: expanded.len(),
                texture,
                sampler,
                uniforms,
                pipeline,
                framebuffer_address: command.framebuffer_address,
                depth_address: command.depth_address,
                blend_constants: blend_constants(command),
                scissor: scaled_scissor(
                    [first_x, first_y, last_x, last_y],
                    self.scale,
                    target_width.saturating_mul(self.scale),
                    target_height.saturating_mul(self.scale),
                ),
            });
        }
        if vertices.is_empty() {
            return Ok(());
        }
        let mut target_summary = Vec::new();
        for draw in &draws {
            let summary = (
                draw.framebuffer_address,
                draw.depth_address,
                draw.pipeline.color_write_mask,
            );
            if !target_summary.contains(&summary) {
                target_summary.push(summary);
            }
        }
        tracing::debug!(
            command_count = commands.len(),
            draw_count = draws.len(),
            texture_cache = self.textures.len(),
            render_target_cache = self.render_targets.len(),
            depth_target_cache = self.depth_targets.len(),
            ?target_summary,
            "SDL3 GE batch"
        );
        self.ensure_vertex_capacity(vertices.len())?;
        {
            let mut mapped = self.vertex_transfer.map::<HostVertex>(&self.gpu, true);
            mapped.mem_mut()[..vertices.len()].copy_from_slice(&vertices);
            mapped.unmap();
        }
        if white_upload {
            let mut mapped = self.white_transfer.map::<u8>(&self.gpu, true);
            mapped.mem_mut()[..4].copy_from_slice(&[255, 255, 255, 255]);
            mapped.unmap();
        }

        let command_buffer = self
            .gpu
            .acquire_command_buffer()
            .context("cannot acquire SDL3 GE command buffer")?;
        let copy_pass = self
            .gpu
            .begin_copy_pass(&command_buffer)
            .context("cannot begin SDL3 GE upload pass")?;
        copy_pass.upload_to_gpu_buffer(
            TransferBufferLocation::new().with_transfer_buffer(&self.vertex_transfer),
            sdl3::gpu::BufferRegion::new()
                .with_buffer(&self.vertex_buffer)
                .with_size((vertices.len() * size_of::<HostVertex>()) as u32),
            true,
        );
        if white_upload {
            copy_pass.upload_to_gpu_texture(
                TextureTransferInfo::new()
                    .with_transfer_buffer(&self.white_transfer)
                    .with_pixels_per_row(1)
                    .with_rows_per_layer(1),
                TextureRegion::new()
                    .with_texture(&self.white_texture)
                    .with_width(1)
                    .with_height(1)
                    .with_depth(1),
                true,
            );
        }
        for upload in &pending_uploads {
            copy_pass.upload_to_gpu_texture(
                TextureTransferInfo::new()
                    .with_transfer_buffer(&upload.transfer)
                    .with_pixels_per_row(upload.width)
                    .with_rows_per_layer(upload.height),
                TextureRegion::new()
                    .with_texture(&upload.texture)
                    .with_width(upload.width)
                    .with_height(upload.height)
                    .with_depth(1),
                true,
            );
        }
        self.gpu.end_copy_pass(copy_pass);

        // a psp frame is not a single host render target. games commonly
        // alternate two edram color buffers, and menu/cutscene code can use a
        // third offscreen surface in the same batch. keep pass order intact,
        // but close and reopen the sdl pass whenever framebuf or zbuf changes.
        let mut group_start = 0;
        while group_start < draws.len() {
            let group_key = (
                draws[group_start].framebuffer_address,
                draws[group_start].depth_address,
            );
            let group_end = draws[group_start..]
                .iter()
                .position(|draw| (draw.framebuffer_address, draw.depth_address) != group_key)
                .map_or(draws.len(), |offset| group_start + offset);
            let color_texture = self
                .render_targets
                .get(&group_key.0)
                .context("missing SDL3 PSP color target")?
                .texture
                .clone();
            let color_initialized = self
                .render_targets
                .get(&group_key.0)
                .is_some_and(|target| target.initialized);
            let mut depth_texture = self
                .depth_targets
                .get(&group_key.1)
                .context("missing SDL3 PSP depth target")?
                .texture
                .clone();
            let depth_initialized = self
                .depth_targets
                .get(&group_key.1)
                .is_some_and(|target| target.initialized);
            let color_target = [ColorTargetInfo::default()
                .with_texture(&color_texture)
                .with_load_op(if color_initialized {
                    LoadOp::LOAD
                } else {
                    LoadOp::CLEAR
                })
                .with_store_op(StoreOp::STORE)
                .with_clear_color(Color::RGBA(0, 0, 0, 255))];
            let depth_target = DepthStencilTargetInfo::new()
                .with_texture(&mut depth_texture)
                .with_load_op(if depth_initialized {
                    LoadOp::LOAD
                } else {
                    LoadOp::CLEAR
                })
                .with_store_op(StoreOp::STORE)
                .with_clear_depth(0.0)
                .with_stencil_load_op(LoadOp::LOAD)
                .with_stencil_store_op(StoreOp::STORE);
            let render_pass = self
                .gpu
                .begin_render_pass(&command_buffer, &color_target, Some(&depth_target))
                .context("cannot begin SDL3 GE render pass")?;
            let target = self
                .render_targets
                .get(&group_key.0)
                .context("missing SDL3 PSP viewport target")?;
            self.gpu.set_viewport(
                &render_pass,
                Viewport::new(
                    0.0,
                    0.0,
                    target.logical_width.saturating_mul(self.scale) as f32,
                    target.logical_height.saturating_mul(self.scale) as f32,
                    0.0,
                    1.0,
                ),
            );
            // sdl blend constants are render-pass state, not pipeline
            // state. only constant-factor draws need them; reprogram on
            // change so back-to-back draws sharing a color skip the call.
            let mut active_constants: Option<[f32; 4]> = None;
            let mut active_pipeline = None;
            render_pass
                .bind_vertex_buffers(0, &[BufferBinding::new().with_buffer(&self.vertex_buffer)]);
            for draw in &draws[group_start..group_end] {
                if active_pipeline != Some(draw.pipeline) {
                    let pipeline = self.pipeline(draw.pipeline)?;
                    render_pass.bind_graphics_pipeline(&pipeline);
                    active_pipeline = Some(draw.pipeline);
                }
                if draw.blend_constants != active_constants {
                    if let Some(constants) = draw.blend_constants {
                        // sdl3 0.18 has no safe wrapper for
                        // sdl_setgpublendconstants yet; the raw handle from
                        // renderpass::raw() is valid for the pass lifetime.
                        unsafe {
                            sdl3::sys::gpu::SDL_SetGPUBlendConstants(
                                render_pass.raw(),
                                sdl3::sys::pixels::SDL_FColor {
                                    r: constants[0],
                                    g: constants[1],
                                    b: constants[2],
                                    a: constants[3],
                                },
                            );
                        }
                    }
                    active_constants = draw.blend_constants;
                }
                render_pass.set_scissor(draw.scissor);
                let sampler = &self.samplers[draw.sampler];
                render_pass.bind_fragment_samplers(
                    0,
                    &[TextureSamplerBinding::new()
                        .with_texture(&draw.texture)
                        .with_sampler(sampler)],
                );
                command_buffer.push_fragment_uniform_data(0, &draw.uniforms);
                render_pass.draw_primitives(draw.vertex_count, 1, draw.first_vertex, 0);
            }
            self.gpu.end_render_pass(render_pass);
            if let Some(target) = self.render_targets.get_mut(&group_key.0) {
                target.initialized = true;
            }
            if let Some(target) = self.depth_targets.get_mut(&group_key.1) {
                target.initialized = true;
            }
            group_start = group_end;
        }
        let submit_started = Instant::now();
        command_buffer
            .submit()
            .context("cannot submit SDL3 GE render pass")?;
        let submit_ms = submit_started.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = render_started.elapsed().as_secs_f64() * 1_000.0;
        tracing::debug!(submit_ms, total_ms, "SDL3 GE submit timing");
        if white_upload {
            self.white_uploaded = true;
        }
        Ok(())
    }

    /// Upload a CPU-visible framebuffer for paths that do not use GE.
    pub fn present(&mut self, rgba: &[u8]) -> Result<()> {
        let expected = self.width as usize * self.height as usize * 4;
        anyhow::ensure!(
            rgba.len() == expected,
            "SDL3 scanout received {} bytes for {}x{} RGBA8",
            rgba.len(),
            self.width,
            self.height
        );
        {
            let mut mapped = self.video_transfer.map::<u8>(&self.gpu, true);
            mapped.mem_mut()[..expected].copy_from_slice(rgba);
            mapped.unmap();
        }
        let command_buffer = self
            .gpu
            .acquire_command_buffer()
            .context("cannot acquire SDL3 scanout command buffer")?;
        let copy_pass = self
            .gpu
            .begin_copy_pass(&command_buffer)
            .context("cannot begin SDL3 scanout upload pass")?;
        copy_pass.upload_to_gpu_texture(
            TextureTransferInfo::new()
                .with_transfer_buffer(&self.video_transfer)
                .with_pixels_per_row(self.width)
                .with_rows_per_layer(self.height),
            TextureRegion::new()
                .with_texture(&self.video_texture)
                .with_width(self.width)
                .with_height(self.height)
                .with_depth(1),
            true,
        );
        self.gpu.end_copy_pass(copy_pass);
        command_buffer.blit_texture(
            BlitInfo::default()
                .with_source_texture(&self.video_texture)
                .with_source_region(0, 0, 0, self.width, self.height)
                .with_destination_texture(&self.frame_texture)
                .with_destination_region(0, 0, 0, self.render_width, self.render_height)
                .with_filter(Filter::Nearest),
        );
        command_buffer
            .submit()
            .context("cannot submit SDL3 CPU scanout upload")?;
        self.render_initialized = true;
        self.present_gpu(None).map(|_| ())
    }

    /// Scale the persistent PSP render target to the current swapchain size.
    ///
    /// Returns `false` when a requested framebuffer has not been rendered into
    /// a persistent host target yet. The caller must then use the guest-memory
    /// scanout path; retaining the previous `frame_texture` would make a
    /// framebuffer switch look like a frozen or stale game screen.
    pub fn present_gpu(&mut self, framebuffer_address: Option<u32>) -> Result<bool> {
        let scanout_address = framebuffer_address.map(normalize_surface_address);
        let source_texture = scanout_address.and_then(|address| {
            self.render_targets
                .get(&address)
                .map(|target| target.texture.clone())
        });
        let target_found = source_texture.is_some();
        tracing::trace!(
            framebuffer = scanout_address.map(|address| format!("0x{address:08x}")),
            target_count = self.render_targets.len(),
            target_found,
            "SDL GPU display scanout"
        );
        if scanout_address.is_some() && source_texture.is_none() {
            return Ok(false);
        }
        if let Some(source_texture) = source_texture.as_ref() {
            // resolve through the persistent scanout texture. sdl's gpu
            // backends have different restrictions on blitting a render
            // target directly to a swapchain image; this keeps that boundary
            // identical for ge and mpeg/cpu presentation.
            let resolve_buffer = self
                .gpu
                .acquire_command_buffer()
                .context("cannot acquire SDL3 framebuffer resolve command buffer")?;
            resolve_buffer.blit_texture(
                BlitInfo::default()
                    .with_source_texture(source_texture)
                    .with_source_region(0, 0, 0, self.render_width, self.render_height)
                    .with_destination_texture(&self.frame_texture)
                    .with_destination_region(0, 0, 0, self.render_width, self.render_height)
                    .with_filter(Filter::Nearest),
            );
            let resolve_submit_started = Instant::now();
            resolve_buffer
                .submit()
                .context("cannot submit SDL3 framebuffer resolve")?;
            tracing::debug!(
                resolve_submit_ms = resolve_submit_started.elapsed().as_secs_f64() * 1_000.0,
                "SDL3 framebuffer resolve timing"
            );
        }
        let mut command_buffer = self
            .gpu
            .acquire_command_buffer()
            .context("cannot acquire SDL3 present command buffer")?;
        let acquire_started = Instant::now();
        let Ok(swapchain) = command_buffer.wait_and_acquire_swapchain_texture(&self.window) else {
            command_buffer.cancel();
            return Ok(true);
        };
        let acquire_ms = acquire_started.elapsed().as_secs_f64() * 1_000.0;
        let destination_width = swapchain.width();
        let destination_height = swapchain.height();
        let blit = BlitInfo::default()
            .with_source_texture(&self.frame_texture)
            .with_source_region(0, 0, 0, self.render_width, self.render_height)
            .with_destination_texture(&swapchain)
            .with_destination_region(0, 0, 0, destination_width, destination_height)
            .with_filter(Filter::Nearest);
        drop(swapchain);
        command_buffer.blit_texture(blit);
        let present_submit_started = Instant::now();
        command_buffer
            .submit()
            .context("cannot submit SDL3 GPU present")?;
        tracing::debug!(
            acquire_ms,
            present_submit_ms = present_submit_started.elapsed().as_secs_f64() * 1_000.0,
            "SDL3 present timing"
        );
        Ok(true)
    }

    fn ensure_render_target(
        &mut self,
        address: u32,
        framebuffer_width: u32,
        required_height: u32,
    ) -> Result<()> {
        if self.render_targets.contains_key(&address) {
            return Ok(());
        }
        let logical_width = framebuffer_width.max(self.width);
        let logical_height = inferred_render_height(self.height, required_height);
        let storage_width = logical_width.max(PSP_TEXTURE_BACKING_DIMENSION);
        let storage_height = logical_height.max(PSP_TEXTURE_BACKING_DIMENSION);
        tracing::debug!(
            framebuffer = format_args!("0x{address:08x}"),
            width = logical_width,
            height = logical_height,
            storage_width,
            storage_height,
            "SDL3 PSP color target allocated"
        );
        let texture = self
            .gpu
            .create_texture(
                TextureCreateInfo::new()
                    .with_type(TextureType::_2D)
                    .with_format(TextureFormat::R8g8b8a8Unorm)
                    .with_usage(TextureUsage::SAMPLER | TextureUsage::COLOR_TARGET)
                    .with_width(storage_width.saturating_mul(self.scale))
                    .with_height(storage_height.saturating_mul(self.scale))
                    .with_layer_count_or_depth(1)
                    .with_num_levels(1),
            )
            .with_context(|| format!("cannot create SDL3 PSP framebuffer 0x{address:08x}"))?;
        self.render_targets.insert(
            address,
            GpuRenderTarget {
                texture,
                logical_width,
                logical_height,
                storage_width,
                storage_height,
                initialized: false,
            },
        );
        Ok(())
    }

    fn ensure_depth_target(
        &mut self,
        address: u32,
        depth_width: u32,
        required_height: u32,
    ) -> Result<()> {
        if self.depth_targets.contains_key(&address) {
            return Ok(());
        }
        let logical_width = depth_width.max(self.width);
        // size the depth attachment like its color sibling so tall
        // offscreen targets keep matching color/depth extents. a vulkan
        // render pass requires identical attachment sizes.
        let logical_height = inferred_render_height(self.height, required_height);
        let storage_width = logical_width.max(PSP_TEXTURE_BACKING_DIMENSION);
        let storage_height = logical_height.max(PSP_TEXTURE_BACKING_DIMENSION);
        tracing::debug!(
            depth = format_args!("0x{address:08x}"),
            width = logical_width,
            height = logical_height,
            storage_width,
            storage_height,
            "SDL3 PSP depth target allocated"
        );
        let texture = self
            .gpu
            .create_texture(
                TextureCreateInfo::new()
                    .with_type(TextureType::_2D)
                    .with_format(TextureFormat::D16Unorm)
                    .with_usage(TextureUsage::DEPTH_STENCIL_TARGET)
                    .with_width(storage_width.saturating_mul(self.scale))
                    .with_height(storage_height.saturating_mul(self.scale))
                    .with_layer_count_or_depth(1)
                    .with_num_levels(1),
            )
            .with_context(|| format!("cannot create SDL3 PSP depth buffer 0x{address:08x}"))?;
        self.depth_targets.insert(
            address,
            GpuDepthTarget {
                texture,
                initialized: false,
            },
        );
        Ok(())
    }

    fn ensure_vertex_capacity(&mut self, required: usize) -> Result<()> {
        if required <= self.vertex_capacity {
            return Ok(());
        }
        let capacity = required.next_power_of_two();
        self.vertex_buffer = self
            .gpu
            .create_buffer()
            .with_usage(BufferUsageFlags::VERTEX)
            .with_size((capacity * size_of::<HostVertex>()) as u32)
            .build()
            .context("cannot grow SDL3 GE vertex buffer")?;
        self.vertex_transfer = self
            .gpu
            .create_transfer_buffer()
            .with_usage(TransferBufferUsage::UPLOAD)
            .with_size((capacity * size_of::<HostVertex>()) as u32)
            .build()
            .context("cannot grow SDL3 GE vertex upload buffer")?;
        self.vertex_capacity = capacity;
        Ok(())
    }

    fn prepare_texture(
        &mut self,
        source: &RenderTexture,
    ) -> Result<(Texture<'static>, Option<PendingTextureUpload>)> {
        if let Some(cached) = self.textures.get(&source.address)
            && cached.content_hash == source.content_hash
            && cached.width == source.width
            && cached.height == source.height
        {
            return Ok((cached.texture.clone(), None));
        }
        let texture = self
            .gpu
            .create_texture(
                TextureCreateInfo::new()
                    .with_type(TextureType::_2D)
                    .with_format(TextureFormat::R8g8b8a8Unorm)
                    .with_usage(TextureUsage::SAMPLER)
                    .with_width(source.width)
                    .with_height(source.height)
                    .with_layer_count_or_depth(1)
                    .with_num_levels(1),
            )
            .with_context(|| {
                format!(
                    "cannot create SDL3 texture 0x{:08x} {}x{}",
                    source.address, source.width, source.height
                )
            })?;
        let transfer = self
            .gpu
            .create_transfer_buffer()
            .with_usage(TransferBufferUsage::UPLOAD)
            .with_size(source.pixels.len() as u32)
            .build()
            .context("cannot create SDL3 GE texture upload buffer")?;
        {
            let mut mapped = transfer.map::<u8>(&self.gpu, true);
            mapped.mem_mut()[..source.pixels.len()].copy_from_slice(&source.pixels);
            mapped.unmap();
        }
        let pending = PendingTextureUpload {
            texture: texture.clone(),
            transfer: transfer.clone(),
            width: source.width,
            height: source.height,
        };
        self.textures.insert(
            source.address,
            CachedTexture {
                texture: texture.clone(),
                _transfer: transfer,
                content_hash: source.content_hash,
                width: source.width,
                height: source.height,
            },
        );
        Ok((texture, Some(pending)))
    }

    fn pipeline(&mut self, key: PipelineKey) -> Result<GraphicsPipeline> {
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let vertex_buffer_descriptions = [VertexBufferDescription::new()
            .with_slot(0)
            .with_pitch(size_of::<HostVertex>() as u32)];
        let vertex_attributes = [
            VertexAttribute::new()
                .with_location(0)
                .with_buffer_slot(0)
                .with_format(VertexElementFormat::Float4)
                .with_offset(0),
            VertexAttribute::new()
                .with_location(1)
                .with_buffer_slot(0)
                .with_format(VertexElementFormat::Float4)
                .with_offset(size_of::<[f32; 4]>() as u32),
            VertexAttribute::new()
                .with_location(2)
                .with_buffer_slot(0)
                .with_format(VertexElementFormat::Float2)
                .with_offset(size_of::<[f32; 8]>() as u32),
        ];
        let blend = blend_state(key.blend_state)
            .with_color_write_mask(color_write_components(key.color_write_mask))
            .with_enable_color_write_mask(key.color_write_mask != 0);
        let color_targets = [ColorTargetDescription::new()
            .with_format(TextureFormat::R8g8b8a8Unorm)
            .with_blend_state(blend)];
        let depth_state = DepthStencilState::new()
            .with_compare_op(compare_op(key.depth_function))
            .with_enable_depth_test(key.depth_test)
            .with_enable_depth_write(key.depth_write);
        let pipeline = self
            .gpu
            .create_graphics_pipeline()
            .with_vertex_shader(&self.vertex_shader)
            .with_fragment_shader(&self.fragment_shader)
            .with_vertex_input_state(
                VertexInputState::new()
                    .with_vertex_buffer_descriptions(&vertex_buffer_descriptions)
                    .with_vertex_attributes(&vertex_attributes),
            )
            .with_primitive_type(PrimitiveType::TriangleList)
            .with_rasterizer_state(RasterizerState::new())
            .with_depth_stencil_state(depth_state)
            .with_target_info(
                GraphicsPipelineTargetInfo::new()
                    .with_color_target_descriptions(&color_targets)
                    .with_depth_stencil_format(TextureFormat::D16Unorm)
                    .with_has_depth_stencil_target(true),
            )
            .build()
            .context("cannot create SDL3 GE graphics pipeline")?;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    #[allow(dead_code)]
    /// Return the integer scale used by the host render target.
    pub fn scale(&self) -> u32 {
        self.scale
    }
}

impl crate::HostDisplay for SdlHost {
    fn is_open(&self) -> bool {
        SdlHost::is_open(self)
    }

    fn poll(&mut self, input: &mut InputState) -> u32 {
        SdlHost::poll(self, input)
    }

    fn render_commands(&mut self, commands: &[RenderCommand]) -> anyhow::Result<()> {
        SdlHost::render_commands(self, commands)
    }

    fn present_gpu(&mut self, framebuffer_address: Option<u32>) -> anyhow::Result<bool> {
        SdlHost::present_gpu(self, framebuffer_address)
    }

    fn present(&mut self, rgba: &[u8]) -> anyhow::Result<()> {
        SdlHost::present(self, rgba)
    }
}

fn inferred_render_height(display_height: u32, required_height: u32) -> u32 {
    if required_height <= display_height {
        display_height
    } else {
        required_height.next_power_of_two()
    }
}

fn normalize_surface_address(address: u32) -> u32 {
    if address < 0x0020_0000 {
        PSP_EDRAM_BASE | (address & 0x001f_fff0)
    } else {
        address & 0xffff_fff0
    }
}

/// Select the SDL sampler for a PSP texture.
///
// each axis has its own wrap bit; a set bit clamps to the texture edge.
fn sampler_index(clamp_s: bool, clamp_t: bool) -> usize {
    usize::from(clamp_s) | (usize::from(clamp_t) << 1)
}

fn host_vertex(
    vertex: RenderVertex,
    logical_width: u32,
    logical_height: u32,
    scale: u32,
    texture: Option<&RenderTexture>,
) -> HostVertex {
    let inv_w = if vertex.inv_w.is_finite() && vertex.inv_w.abs() > f32::EPSILON {
        vertex.inv_w
    } else {
        1.0
    };
    let reciprocal_w = 1.0 / inv_w;
    let x = ((vertex.x as f32 + 0.5) * scale as f32 / logical_width.saturating_mul(scale) as f32)
        * 2.0
        - 1.0;
    // sdl gpu follows vulkan clip convention (ndc y = -1 at the top),
    // unlike opengl. psp screen row 0 must therefore map to ndc -1 so ge
    // draws land upright, matching the blit-based cpu/video scanout path.
    let y = ((vertex.y as f32 + 0.5) * scale as f32 / logical_height.saturating_mul(scale) as f32)
        * 2.0
        - 1.0;
    let color = [
        (vertex.color & 0xff) as f32 / 255.0,
        ((vertex.color >> 8) & 0xff) as f32 / 255.0,
        ((vertex.color >> 16) & 0xff) as f32 / 255.0,
        ((vertex.color >> 24) & 0xff) as f32 / 255.0,
    ];
    let uv = texture.map_or([0.5, 0.5], |texture| {
        [
            (vertex.u as f32 + 0.5) / texture.width.max(1) as f32,
            (vertex.v as f32 + 0.5) / texture.height.max(1) as f32,
        ]
    });
    HostVertex {
        position: [
            x * reciprocal_w,
            y * reciprocal_w,
            (vertex.depth as f32 / 65535.0) * reciprocal_w,
            reciprocal_w,
        ],
        color,
        uv,
    }
}

fn scaled_scissor(scissor: [u32; 4], scale: u32, width: u32, height: u32) -> Rect {
    let max_width = width.saturating_sub(1);
    let max_height = height.saturating_sub(1);
    let first_x = scissor[0].saturating_mul(scale).min(max_width);
    let first_y = scissor[1].saturating_mul(scale).min(max_height);
    let last_x = scissor[2]
        .saturating_add(1)
        .saturating_mul(scale)
        .saturating_sub(1)
        .min(max_width);
    let last_y = scissor[3]
        .saturating_add(1)
        .saturating_mul(scale)
        .saturating_sub(1)
        .min(max_height);
    Rect::new(
        first_x as i32,
        first_y as i32,
        last_x.saturating_sub(first_x).saturating_add(1),
        last_y.saturating_sub(first_y).saturating_add(1),
    )
}

fn expand_vertices(command: &RenderCommand) -> Cow<'_, [RenderVertex]> {
    match command.primitive {
        3 => Cow::Borrowed(&command.vertices),
        4 => command
            .vertices
            .windows(3)
            .enumerate()
            .flat_map(|(index, triangle)| {
                if index & 1 == 0 {
                    [triangle[0], triangle[1], triangle[2]]
                } else {
                    [triangle[1], triangle[0], triangle[2]]
                }
            })
            .collect::<Vec<_>>()
            .into(),
        5 => command
            .vertices
            .get(1..)
            .unwrap_or_default()
            .windows(2)
            .flat_map(|triangle| [command.vertices[0], triangle[0], triangle[1]])
            .collect::<Vec<_>>()
            .into(),
        6 => command
            .vertices
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|pair| rectangle_vertices(pair[0], pair[1]))
            .collect::<Vec<_>>()
            .into(),
        _ => Cow::Borrowed(&[]),
    }
}

fn rectangle_vertices(first: RenderVertex, second: RenderVertex) -> [RenderVertex; 6] {
    // psp sprites are specified by a top-left and bottom-right vertex, but
    // either axis may run backwards. keep the original corner orientation;
    // sorting x/y independently changes the texture mapping for flipped
    // sprites.
    let mut top_right = second;
    top_right.y = first.y;
    top_right.v = first.v;
    let mut top_left = second;
    top_left.x = first.x;
    top_left.y = first.y;
    top_left.u = first.u;
    top_left.v = first.v;
    let mut bottom_left = second;
    bottom_left.x = first.x;
    bottom_left.u = first.u;

    // the psp has a special uv rotation rule when exactly one screen axis is
    // reversed. swap the middle corners' uvs before emitting two triangles.
    if (first.x < second.x && first.y > second.y) || (first.x > second.x && first.y < second.y) {
        std::mem::swap(&mut top_right.u, &mut bottom_left.u);
        std::mem::swap(&mut top_right.v, &mut bottom_left.v);
    }

    [second, top_right, top_left, bottom_left, second, top_left]
}

/// Host depth, test, and blend pipeline state for one GE draw.
///
/// The PSP depth-test enable (0x23) and depth-write disable (0xe7) are
/// independent bits, but SDL GPU (like Vulkan/D3D12) ignores depth writes
/// while the test is disabled. Any PSP draw that writes depth without
/// testing it (depth clears, sky priming the buffer, gta world's depth
/// pre-pass) is therefore routed through always: every covered fragment
/// passes and stores depth, exactly like the cpu reference rasterizer.
/// without this, the host depth buffer keeps its initial clear and every
/// gequal-tested world fragment fails while the cpu reference passes.
fn pipeline_key(command: &RenderCommand) -> PipelineKey {
    if command.clear_mode & 1 != 0 {
        // a color-only clear keeps both test and write off.
        let write_depth = command.clear_mode & 0x400 != 0;
        PipelineKey {
            depth_test: write_depth,
            depth_write: write_depth,
            depth_function: 1,
            blend_state: 0,
            color_write_mask: effective_color_write_mask(command),
        }
    } else if command.depth_write && !command.depth_test {
        PipelineKey {
            depth_test: true,
            depth_write: true,
            depth_function: 1,
            blend_state: if command.blend {
                command.blend_state
            } else {
                0
            },
            color_write_mask: effective_color_write_mask(command),
        }
    } else {
        PipelineKey {
            depth_test: command.depth_test,
            depth_write: command.depth_write,
            depth_function: command.depth_function,
            blend_state: if command.blend {
                command.blend_state
            } else {
                0
            },
            color_write_mask: effective_color_write_mask(command),
        }
    }
}

fn fragment_uniforms(texture: Option<&RenderTexture>, command: &RenderCommand) -> FragmentUniforms {
    let clear = command.clear_mode & 1 != 0;
    let function = texture.map_or(0, |texture| texture.function & 7);
    let flags = texture.map_or(0, |texture| {
        ((texture.function >> 8) & 1) | (((texture.function >> 16) & 1) << 1)
    });
    let environment_color = texture.map_or(0, |texture| texture.environment_color);
    FragmentUniforms {
        function,
        flags,
        // clear primitives bypass fragment tests on the psp.  applying the
        // latched alpha/color tests here can discard the clear rectangle and
        // leave a freshly allocated sdl target black.
        alpha_test: u32::from(command.alpha_test && !clear),
        alpha_function: command.alpha_function,
        alpha_reference: command.alpha_reference,
        alpha_mask: command.alpha_mask,
        color_test: u32::from(command.color_test && !clear),
        color_function: command.color_function,
        color_reference: command.color_reference,
        color_mask: command.color_test_mask,
        _padding: [0; 2],
        environment: [
            (environment_color & 0xff) as f32 / 255.0,
            ((environment_color >> 8) & 0xff) as f32 / 255.0,
            ((environment_color >> 16) & 0xff) as f32 / 255.0,
            ((environment_color >> 24) & 0xff) as f32 / 255.0,
        ],
    }
}

fn effective_color_write_mask(command: &RenderCommand) -> u32 {
    if command.clear_mode & 1 != 0 {
        clear_color_write_mask(command.clear_mode)
    } else {
        command.color_write_mask_rgb | (command.color_write_mask_alpha << 24)
    }
}

fn clear_color_write_mask(clear_mode: u32) -> u32 {
    let mut preserve = 0;
    if clear_mode & 0x100 == 0 {
        preserve |= 0x00ff_ffff;
    }
    if clear_mode & 0x200 == 0 {
        preserve |= 0xff00_0000;
    }
    preserve
}

fn compare_op(function: u32) -> CompareOp {
    match function & 7 {
        0 => CompareOp::Never,
        1 => CompareOp::Always,
        2 => CompareOp::Equal,
        3 => CompareOp::NotEqual,
        4 => CompareOp::Less,
        5 => CompareOp::LessOrEqual,
        6 => CompareOp::Greater,
        _ => CompareOp::GreaterOrEqual,
    }
}

/// SDL blend-constant color for a draw, if either blend side uses the PSP
/// fixed-color factor (10). The PSP carries one fixed RGB color in
/// blendcolorfixa/b (registers 0xe0/0xe1, alpha implicitly 1.0 per the cpu
/// `fixed_component` phi); sdl exposes a single rgba constant for both
/// sides, so the source-side color (fixa) wins when both sides are constant
/// and disagree. games set both to the same color in practice.
fn blend_constants(command: &RenderCommand) -> Option<[f32; 4]> {
    if command.clear_mode & 1 != 0 || !command.blend {
        return None;
    }
    let source = command.blend_state & 0xf;
    let destination = (command.blend_state >> 4) & 0xf;
    if source != 10 && destination != 10 {
        return None;
    }
    let fixed = if source == 10 {
        command.blend_fixed_a
    } else {
        command.blend_fixed_b
    };
    Some([
        (fixed & 0xff) as f32 / 255.0,
        ((fixed >> 8) & 0xff) as f32 / 255.0,
        ((fixed >> 16) & 0xff) as f32 / 255.0,
        1.0,
    ])
}

fn blend_state(state: u32) -> ColorTargetBlendState {
    if state == 0 {
        return ColorTargetBlendState::new();
    }
    ColorTargetBlendState::new()
        .with_src_color_blendfactor(source_blend_factor(state & 0xf))
        .with_dst_color_blendfactor(destination_blend_factor((state >> 4) & 0xf))
        .with_color_blend_op(blend_op((state >> 8) & 7))
        .with_src_alpha_blendfactor(source_blend_factor(state & 0xf))
        .with_dst_alpha_blendfactor(destination_blend_factor((state >> 4) & 0xf))
        .with_alpha_blend_op(blend_op((state >> 8) & 7))
        .with_enable_blend(true)
}

fn color_write_components(preserve_mask: u32) -> ColorComponentFlags {
    let mut components = ColorComponentFlags::R & ColorComponentFlags::G;
    if preserve_mask & 0xff == 0 {
        components = components | ColorComponentFlags::R;
    }
    if preserve_mask >> 8 & 0xff == 0 {
        components = components | ColorComponentFlags::G;
    }
    if preserve_mask >> 16 & 0xff == 0 {
        components = components | ColorComponentFlags::B;
    }
    if preserve_mask >> 24 & 0xff == 0 {
        components = components | ColorComponentFlags::A;
    }
    components
}

fn source_blend_factor(factor: u32) -> BlendFactor {
    match factor {
        0 => BlendFactor::DstColor,
        1 => BlendFactor::OneMinusDstColor,
        2 => BlendFactor::SrcAlpha,
        3 => BlendFactor::OneMinusSrcAlpha,
        4 => BlendFactor::DstAlpha,
        5 => BlendFactor::OneMinusDstAlpha,
        10 => BlendFactor::ConstantColor,
        _ => BlendFactor::One,
    }
}

fn destination_blend_factor(factor: u32) -> BlendFactor {
    match factor {
        0 => BlendFactor::SrcColor,
        1 => BlendFactor::OneMinusSrcColor,
        2 => BlendFactor::SrcAlpha,
        3 => BlendFactor::OneMinusSrcAlpha,
        4 => BlendFactor::DstAlpha,
        5 => BlendFactor::OneMinusDstAlpha,
        10 => BlendFactor::ConstantColor,
        _ => BlendFactor::One,
    }
}

fn blend_op(equation: u32) -> BlendOp {
    match equation {
        1 => BlendOp::Subtract,
        2 => BlendOp::ReverseSubtract,
        3 => BlendOp::Min,
        4 => BlendOp::Max,
        _ => BlendOp::Add,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_scale_keeps_psp_clip_coordinates_stable() {
        let vertex = RenderVertex {
            color: u32::MAX,
            u: 0,
            v: 0,
            inv_w: 1.0,
            x: 479,
            y: 271,
            depth: u16::MAX,
        };
        let native = host_vertex(vertex, 480, 272, 1, None);
        let scaled = host_vertex(vertex, 480, 272, 4, None);
        for (native, scaled) in native.position.into_iter().zip(scaled.position) {
            assert!((native - scaled).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn depth_write_without_test_routes_through_always() {
        // gta world's depth pre-pass / sky priming: psp writes depth with
        // the test disabled. the host must still store it, or every later
        // gequal fragment fails against the initial clear.
        let key = pipeline_key(&RenderCommand {
            depth_test: false,
            depth_write: true,
            depth_function: 7,
            ..Default::default()
        });
        assert!(key.depth_test);
        assert!(key.depth_write);
        assert_eq!(key.depth_function, 1);

        // regular depth-tested draws pass through untouched.
        let key = pipeline_key(&RenderCommand {
            depth_test: true,
            depth_write: true,
            depth_function: 7,
            ..Default::default()
        });
        assert!(key.depth_test);
        assert_eq!(key.depth_function, 7);

        // a depth clear writes via always even though clears never test.
        let key = pipeline_key(&RenderCommand {
            clear_mode: 0x401,
            ..Default::default()
        });
        assert!(key.depth_test);
        assert!(key.depth_write);

        // a color-only clear touches neither test nor depth.
        let key = pipeline_key(&RenderCommand {
            clear_mode: 0x101,
            ..Default::default()
        });
        assert!(!key.depth_test);
        assert!(!key.depth_write);
    }

    #[test]
    fn fixed_factor_blends_carry_the_psp_blend_color() {
        // 0xaa: src=10 (fixed), dst=10 (fixed), equation=add.
        let command = RenderCommand {
            blend: true,
            blend_state: 0xaa,
            blend_fixed_a: 0x0011_2233,
            blend_fixed_b: 0x0044_5566,
            ..Default::default()
        };
        let constants = blend_constants(&command).expect("constant blend needs a color");
        // source-side fixa wins; bytes are r,g,b; alpha is implicitly 1.0.
        assert!((constants[0] - 0x33 as f32 / 255.0).abs() < f32::EPSILON);
        assert!((constants[1] - 0x22 as f32 / 255.0).abs() < f32::EPSILON);
        assert!((constants[2] - 0x11 as f32 / 255.0).abs() < f32::EPSILON);
        assert_eq!(constants[3], 1.0);

        // destination-only fixed blend uses fixb.
        let command = RenderCommand {
            blend: true,
            blend_state: (2) | (10 << 4),
            blend_fixed_b: 0x00ab_cdef,
            ..Default::default()
        };
        let constants = blend_constants(&command).expect("dst-constant blend needs a color");
        assert!((constants[0] - 0xef as f32 / 255.0).abs() < f32::EPSILON);

        // standard src-alpha/inv-src-alpha needs no constant.
        assert_eq!(
            blend_constants(&RenderCommand {
                blend: true,
                blend_state: (2) | (3 << 4),
                ..Default::default()
            }),
            None
        );
        // disabled blending and clears need none either.
        assert_eq!(blend_constants(&RenderCommand::default()), None);
        assert_eq!(
            blend_constants(&RenderCommand {
                blend: true,
                blend_state: 0xaa,
                clear_mode: 1,
                ..Default::default()
            }),
            None
        );
    }

    #[test]
    fn internal_scale_expands_inclusive_scissor_bounds() {
        let rect = scaled_scissor([1, 2, 3, 4], 2, 960, 544);
        assert_eq!(rect, Rect::new(2, 4, 6, 6));
    }

    #[test]
    fn psp_surface_addresses_accept_offsets_and_full_addresses() {
        assert_eq!(normalize_surface_address(0x0017_8000), 0x0417_8000);
        assert_eq!(normalize_surface_address(0x0417_8000), 0x0417_8000);
    }

    #[test]
    fn offscreen_geometry_can_extend_beyond_the_display_viewport() {
        assert_eq!(inferred_render_height(272, 272), 272);
        assert_eq!(inferred_render_height(272, 321), 512);
        assert_eq!(inferred_render_height(272, 512), 512);
    }

    #[test]
    fn clear_mode_uses_clear_bits_instead_of_latched_draw_mask() {
        assert_eq!(clear_color_write_mask(0x701), 0);
        assert_eq!(clear_color_write_mask(0x501), 0xff00_0000);
        assert_eq!(clear_color_write_mask(0x101), 0xff00_0000);
        assert_eq!(clear_color_write_mask(0x301), 0);
    }

    #[test]
    fn full_byte_color_masks_map_to_gpu_components() {
        let all = ColorComponentFlags::R
            | ColorComponentFlags::G
            | ColorComponentFlags::B
            | ColorComponentFlags::A;
        assert_eq!(color_write_components(0), all);
        assert_eq!(
            color_write_components(0x00ff_ffff),
            (ColorComponentFlags::R & ColorComponentFlags::G) | ColorComponentFlags::A
        );
        assert_eq!(
            color_write_components(0xff00_0000),
            ColorComponentFlags::R | ColorComponentFlags::G | ColorComponentFlags::B
        );
    }

    #[test]
    fn twrap_bits_select_each_axis_independently() {
        // twrap set means clamp. gta's full-screen blits
        // use wrap=00 (repeat both), ui strips often clamp.
        assert_eq!(sampler_index(false, false), 0);
        assert_eq!(sampler_index(true, false), 1);
        assert_eq!(sampler_index(false, true), 2);
        assert_eq!(sampler_index(true, true), 3);
    }

    #[test]
    fn psp_screen_top_maps_to_vulkan_clip_top() {
        // sdl gpu uses vulkan clip convention (ndc y = -1 at the top),
        // verified with a fullscreen-triangle probe: psp row 0 must map to
        // negative ndc y and the last row to positive ndc y.
        let top = host_vertex(
            RenderVertex {
                color: 0,
                u: 0,
                v: 0,
                inv_w: 1.0,
                x: 0,
                y: 0,
                depth: 0,
            },
            480,
            272,
            1,
            None,
        );
        let bottom = host_vertex(
            RenderVertex {
                color: 0,
                u: 0,
                v: 0,
                inv_w: 1.0,
                x: 479,
                y: 271,
                depth: 0,
            },
            480,
            272,
            1,
            None,
        );
        assert!(top.position[1] < 0.0);
        assert!(bottom.position[1] > 0.0);
        assert!(top.position[1] < bottom.position[1]);
    }

    #[test]
    fn sprite_expansion_preserves_psp_uv_rotation() {
        let first = RenderVertex {
            color: 0x1020_3040,
            u: 0,
            v: 200,
            inv_w: 1.0,
            x: 0,
            y: 20,
            depth: 1,
        };
        let second = RenderVertex {
            color: 0xa0b0_c0d0,
            u: 100,
            v: 0,
            inv_w: 1.0,
            x: 10,
            y: 0,
            depth: 2,
        };
        let expanded = rectangle_vertices(first, second);
        assert_eq!(
            expanded
                .iter()
                .map(|vertex| (vertex.x, vertex.y, vertex.u, vertex.v))
                .collect::<Vec<_>>(),
            vec![
                (10, 0, 100, 0),
                (10, 20, 0, 0),
                (0, 20, 0, 200),
                (0, 0, 100, 200),
                (10, 0, 100, 0),
                (0, 20, 0, 200),
            ]
        );
        assert!(expanded.iter().all(|vertex| vertex.color == second.color));
        assert!(expanded.iter().all(|vertex| vertex.depth == second.depth));
    }
}
