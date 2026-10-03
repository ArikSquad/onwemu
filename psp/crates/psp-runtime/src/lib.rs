//! PSP runtime and desktop frontend.
//!
//! The same HLE runtime powers the SDL desktop runner and the browser adapter.

use anyhow::{Context, Result};
#[cfg(feature = "desktop")]
use clap::{Parser, Subcommand};
use psp_input::{
    ANALOG_CENTER, CTRL_SAMPLE_BUFFER_COUNT, ControllerSample, InputState, PSP_CTRL_CIRCLE,
    PSP_CTRL_CROSS, PSP_CTRL_DOWN, PSP_CTRL_HOME, PSP_CTRL_LEFT, PSP_CTRL_LTRIGGER, PSP_CTRL_RIGHT,
    PSP_CTRL_RTRIGGER, PSP_CTRL_SELECT, PSP_CTRL_SQUARE, PSP_CTRL_START, PSP_CTRL_TRIANGLE,
    PSP_CTRL_UP,
};
use psp_media::{
    ATRAC_OUTPUT_BYTES, ATRAC_SAMPLES_PER_FRAME, AUDIO_CHUNK_BYTES, AtracDecoder, AtracFormat,
    MediaPipeline, VIDEO_HEIGHT, VIDEO_WIDTH, inspect_atrac,
};
use psp_memory::{GuestMemory, Memory};
#[cfg(feature = "desktop")]
use sdl3::keyboard::Scancode;
use std::collections::{HashMap, HashSet, VecDeque, hash_map::Entry};
#[cfg(feature = "desktop")]
use std::path::PathBuf;
#[cfg(feature = "desktop")]
use std::thread;
use std::time::Duration;
#[cfg(feature = "desktop")]
use std::time::Instant;
#[cfg(feature = "desktop")]
use tracing::{info, warn};
#[cfg(feature = "desktop")]
use tracing_subscriber::EnvFilter;

#[cfg(feature = "desktop")]
mod host_video;
#[cfg(feature = "desktop")]
use host_video::SdlHost;

#[cfg_attr(not(feature = "desktop"), allow(dead_code))]
trait HostDisplay {
    fn is_open(&self) -> bool;
    fn poll(&mut self, input: &mut InputState) -> u32;
    fn render_commands(&mut self, commands: &[psp_gpu::RenderCommand]) -> Result<()>;
    fn present_gpu(&mut self, framebuffer_address: Option<u32>) -> Result<bool>;
    fn present(&mut self, rgba: &[u8]) -> Result<()>;
}

#[cfg(feature = "desktop")]
fn timer_start() -> Option<std::time::Instant> {
    Some(std::time::Instant::now())
}

#[cfg(not(feature = "desktop"))]
fn timer_start() -> Option<std::time::Instant> {
    None
}

fn timer_elapsed(start: Option<std::time::Instant>) -> Duration {
    #[cfg(feature = "desktop")]
    {
        start.map_or(Duration::ZERO, |started| started.elapsed())
    }
    #[cfg(not(feature = "desktop"))]
    {
        let _ = start;
        Duration::ZERO
    }
}

mod web_core;
pub use web_core::{ControllerButton, WebCore};

const VOLATILE_BASE: u32 = 0x0840_0000;
const VOLATILE_SIZE: usize = 4 * 1024 * 1024;
const EDRAM_BASE: u32 = 0x0400_0000;
const EDRAM_SIZE: usize = 2 * 1024 * 1024;
const USER_RAM_END: u32 = 0x0a00_0000;
const USER_MODULE_BASE: u32 = 0x0880_4000;
const AUDIO_CHANNEL_COUNT: usize = 8;
const AUDIO_SAMPLE_MAX: u32 = 65_472;
const AUDIO_ERROR_CHANNEL_NOT_INIT: u32 = 0x8026_0001;
const AUDIO_ERROR_CHANNEL_BUSY: u32 = 0x8026_0002;
const AUDIO_ERROR_INVALID_CHANNEL: u32 = 0x8026_0003;
const AUDIO_ERROR_NO_CHANNELS: u32 = 0x8026_0005;
const AUDIO_ERROR_SAMPLE_SIZE: u32 = 0x8026_0006;
const AUDIO_ERROR_INVALID_FORMAT: u32 = 0x8026_0007;
const AUDIO_ERROR_INVALID_VOLUME: u32 = 0x8026_000b;
const AUDIO_ERROR_CHANNEL_ALREADY_RESERVED: u32 = 0x8026_8002;
const AUDIO_SAMPLE_RATE: u64 = 44_100;
const AUDIO_MIX_BLOCK_SAMPLES: u32 = 64;
/// PSP audio channels transfer in multiples of one hardware mixer block.
const AUDIO_MIX_BLOCK_MASK: u32 = AUDIO_MIX_BLOCK_SAMPLES - 1;
const SAS_ERROR_INVALID_GRAIN: u32 = 0x8042_0001;
const SAS_ERROR_INVALID_MAX_VOICES: u32 = 0x8042_0002;
const SAS_ERROR_INVALID_OUTPUT_MODE: u32 = 0x8042_0003;
const SAS_ERROR_INVALID_SAMPLE_RATE: u32 = 0x8042_0004;
const SAS_ERROR_BAD_ADDRESS: u32 = 0x8042_0005;
const SAS_ERROR_INVALID_VOICE: u32 = 0x8042_0010;
const SAS_ERROR_INVALID_PITCH: u32 = 0x8042_0012;
const SAS_ERROR_INVALID_PARAMETER: u32 = 0x8042_0014;
const SAS_ERROR_INVALID_LOOP_POS: u32 = 0x8042_0015;
const SAS_ERROR_VOICE_PAUSED: u32 = 0x8042_0016;
const SAS_ERROR_INVALID_VOLUME: u32 = 0x8042_0018;
const SAS_ERROR_INVALID_PCM_SIZE: u32 = 0x8042_001a;
const SAS_ERROR_NOT_INIT: u32 = 0x8042_0100;
const KERNEL_ERROR_ILLEGAL_ATTR: u32 = 0x8002_0191;
const KERNEL_ERROR_ILLEGAL_MODE: u32 = 0x8002_0195;
const KERNEL_ERROR_INVALID_SIZE: u32 = 0x8000_0104;
const KERNEL_ERROR_INVALID_VALUE: u32 = 0x8000_01fe;
const KERNEL_ERROR_WAIT_TIMEOUT: u32 = 0x8002_01a8;
const IO_ERROR_FILE_NOT_FOUND: u32 = 0x8001_0002;
const IO_ERROR_INVALID_ARGUMENT: u32 = 0x8001_0016;
const IO_ERROR_IO: u32 = 0x8001_0001;
const IO_ERROR_BAD_FD: u32 = 0x8001_0009;
const IO_ERROR_NOT_SUPPORTED: u32 = 0x8001_0081;
const IO_ERROR_ASYNC_BAD_FD: u32 = 0x8002_0323;
const IO_ERROR_ASYNC_BUSY: u32 = 0x8002_0329;
const IO_ERROR_NO_ASYNC: u32 = 0x8002_032a;
const MPEG_ERROR_INVALID_VALUE: u32 = 0x8061_01fe;
const MPEG_ERROR_AVC_DECODE_FATAL: u32 = 0x8062_8002;
const ATRAC_MAX_IDS: u32 = 6;
const ATRAC_ERROR_NO_ATRACID: u32 = 0x8063_0003;
const ATRAC_ERROR_BAD_ATRACID: u32 = 0x8063_0005;
const ATRAC_ERROR_UNKNOWN_FORMAT: u32 = 0x8063_0006;
const ATRAC_ERROR_ALL_DATA_LOADED: u32 = 0x8063_0009;
const ATRAC_ERROR_NO_DATA: u32 = 0x8063_0010;
const ATRAC_ERROR_INCORRECT_READ_SIZE: u32 = 0x8063_0013;
const ATRAC_ERROR_BAD_ALIGNMENT: u32 = 0x8063_0014;
const ATRAC_ERROR_ADD_DATA_TOO_BIG: u32 = 0x8063_0018;
const ATRAC_ERROR_ALL_DATA_DECODED: u32 = 0x8063_0024;
const MPEG_AVC_ES_SIZE: u32 = 2_048;
const MPEG_ATRAC_ES_SIZE: u32 = 2_112;
const MPEG_ATRAC_OUTPUT_SIZE: u32 = 8_192;
const MPEG_RING_PACKETS: u32 = 0;
const MPEG_RING_PACKETS_READ: u32 = 4;
const MPEG_RING_PACKETS_WRITTEN: u32 = 8;
const MPEG_RING_PACKETS_IN_BUFFER: u32 = 12;
const MPEG_RING_PACKET_SIZE: u32 = 16;
const MPEG_RING_DATA: u32 = 20;
const MPEG_RING_CALLBACK: u32 = 24;
const MPEG_RING_CALLBACK_PARAM: u32 = 28;
const MPEG_RING_DATA_END: u32 = 32;
const MPEG_RING_SEMAPHORE: u32 = 36;
const MPEG_RING_MPEG: u32 = 40;
const MPEG_RING_FIELD_COUNT: u32 = 11;
const MPEG_HANDLE_RING: u32 = 16;
const MPEG_HANDLE_DATA_END: u32 = 20;
const MPEG_PACKET_SIZE: u32 = 2_048;
const MPEG_PACKET_OVERHEAD: u32 = 104;
const MPEG_VIDEO_WIDTH: u32 = 480;
const MPEG_VIDEO_HEIGHT: u32 = 272;
const MPEG_DECODER_PREFETCH_PACKETS: u32 = 128;
#[cfg(feature = "desktop")]
const AUTO_CROSS_HOLD_INSTRUCTIONS: u64 = 100_000_000;
const DISPLAY_WIDTH: u32 = 480;
const DISPLAY_HEIGHT: u32 = 272;
const MPEG_AU_DTS_MSB: u32 = 8;
const MPEG_AU_DTS: u32 = 12;
const MPEG_AU_ES_BUFFER: u32 = 16;
const MPEG_AU_SIZE: u32 = 20;
#[cfg(feature = "desktop")]
const DEFAULT_HEADLESS_INSTRUCTIONS: u64 = 1_000_000;
#[cfg(feature = "desktop")]
const HOST_EVENT_POLL_INTERVAL: u64 = 4_096;
#[cfg(feature = "desktop")]
const HOST_FRAME_INTERVAL: Duration = Duration::from_micros(16_667);
const ISO_SECTOR_SIZE: u64 = 2_048;
const SCE_STM_FDIR: u32 = 0x1000;
const SCE_STM_FREG: u32 = 0x2000;
const SCE_STM_READ_ONLY: u32 = 0o444;
const SCE_STM_ISO_ACCESS: u32 = 0o555;
const SCE_IO_STAT_SIZE: usize = 88;
const SCE_IO_DIRENT_SIZE: usize = 352;
#[derive(Parser)]
#[cfg(feature = "desktop")]
#[command(
    name = "psp-rs",
    version,
    about = "A correctness-first PSP emulator foundation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
#[cfg(feature = "desktop")]
enum Command {
    Inspect {
        /// ISO, CHD, ELF, or PBP image to inspect.
        path: PathBuf,
    },
    Run {
        /// ISO, CHD, ELF, or PBP image to run.
        path: PathBuf,
        #[arg(long)]
        /// Run without opening a graphical window.
        headless: bool,
        /// Stop after this many guest instructions. Graphical runs without a
        /// limit continue until the guest exits or the host window closes;
        /// headless runs default to a bounded diagnostic session.
        #[arg(long)]
        instructions: Option<u64>,
        /// Integer internal render and host presentation scale for the PSP
        /// framebuffer. Supported values are 1x through 4x.
        #[arg(long, default_value_t = 2, value_parser = parse_scale)]
        scale: u32,
        /// Write the final guest display framebuffer as a PPM image.
        #[arg(long, value_name = "PATH")]
        dump_frame: Option<PathBuf>,
        /// Hold Cross after each listed guest instruction count.
        #[arg(long, value_name = "INSTRUCTIONS")]
        cross_at: Vec<u64>,
        /// Hold Cross for this many guest instructions.
        #[arg(long, value_name = "INSTRUCTIONS", default_value_t = AUTO_CROSS_HOLD_INSTRUCTIONS)]
        cross_for: u64,
        /// Write the GE off-screen EDRAM target as a PPM image.
        #[arg(long, value_name = "PATH")]
        dump_target: Option<PathBuf>,
        /// Write the guest display framebuffer here every `dump-frame-every`
        /// instructions for long-run progress tracking.
        #[arg(long, value_name = "DIR")]
        dump_frame_dir: Option<PathBuf>,
        /// Guest instruction interval between frame-directory dumps.
        #[arg(long, value_name = "INSTRUCTIONS", default_value_t = 50_000_000)]
        dump_frame_every: u64,
        /// Dump guest memory ranges at exit as `start:length:path`; repeatable.
        #[arg(long, value_name = "START:LEN:PATH")]
        dump_memory: Vec<String>,
        /// Log every HLE syscall from this instruction count onward. Unlike
        /// `trace --cpu`, the block JIT stays enabled for long runs. Set
        /// `PSP_TRACE_THREAD=<uid>` to record one guest thread.
        #[arg(long)]
        trace_syscalls: bool,
        /// Suppress syscall tracing until this instruction count.
        #[arg(long, default_value_t = 0)]
        trace_from: u64,
    },
    Trace {
        /// ISO, CHD, ELF, or PBP image to run under tracing.
        path: PathBuf,
        #[arg(long)]
        /// Log each CPU instruction through the interpreter.
        cpu: bool,
        /// Restrict `--cpu` logs to these instruction addresses. Repeatable or
        /// comma-separated; execution still uses the single-step interpreter.
        #[arg(long, value_parser = parse_u32, value_delimiter = ',', requires = "cpu")]
        pc: Vec<u32>,
        #[arg(long)]
        /// Log HLE syscall calls and returns.
        syscalls: bool,
        #[arg(long, default_value_t = 1_000_000)]
        /// Stop after this many guest instructions.
        instructions: u64,
        /// Suppress per-instruction CPU logs until this instruction count.
        #[arg(long, default_value_t = 0)]
        from: u64,
        /// Log every change to this guest address, in hexadecimal or decimal.
        #[arg(long, value_parser = parse_u32)]
        watch: Option<u32>,
        /// Dump guest memory ranges after the run as `start:length:path`.
        /// Repeatable.
        #[arg(long, value_name = "START:LEN:PATH")]
        dump_memory: Vec<String>,
        /// Hold Cross after each listed guest instruction count.
        #[arg(long, value_name = "INSTRUCTIONS")]
        cross_at: Vec<u64>,
        /// Hold Cross for this many guest instructions.
        #[arg(long, value_name = "INSTRUCTIONS", default_value_t = AUTO_CROSS_HOLD_INSTRUCTIONS)]
        cross_for: u64,
    },
}

#[cfg(feature = "desktop")]
struct RunOptions {
    path: PathBuf,
    headless: bool,
    limit: u64,
    trace_cpu: bool,
    trace_pcs: Vec<u32>,
    trace_syscalls: bool,
    trace_from: u64,
    watch_address: Option<u32>,
    host_scale: u32,
    frame_output: Option<PathBuf>,
    target_output: Option<PathBuf>,
    auto_cross_at: Vec<u64>,
    auto_cross_for: u64,
    frame_dir: Option<PathBuf>,
    frame_interval: u64,
    /// Guest memory ranges written to disk when execution ends.
    memory_dumps: Vec<(u32, u32, PathBuf)>,
}

#[cfg(feature = "desktop")]
pub fn run_cli() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    std::panic::set_hook(Box::new(|p| eprintln!("fatal emulator panic: {p}")));
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { path } => {
            let i = psp_loader::inspect_path(&path)
                .with_context(|| format!("cannot inspect {}", path.display()))?;
            println!("{i:#?}")
        }
        Command::Run {
            path,
            headless,
            instructions,
            scale,
            dump_frame,
            cross_at,
            cross_for,
            dump_target,
            dump_frame_dir,
            dump_frame_every,
            dump_memory,
            trace_syscalls,
            trace_from,
        } => {
            let limit = instructions.unwrap_or(if headless {
                DEFAULT_HEADLESS_INSTRUCTIONS
            } else {
                u64::MAX
            });
            run_image(RunOptions {
                path,
                headless,
                limit,
                trace_cpu: false,
                trace_pcs: Vec::new(),
                trace_syscalls,
                trace_from,
                watch_address: None,
                host_scale: scale,
                frame_output: dump_frame,
                target_output: dump_target,
                auto_cross_at: cross_at,
                auto_cross_for: cross_for,
                frame_dir: dump_frame_dir,
                frame_interval: dump_frame_every.max(1),
                memory_dumps: parse_memory_dumps(&dump_memory)?,
            })?;
        }
        Command::Trace {
            path,
            cpu,
            pc,
            syscalls,
            instructions,
            from,
            watch,
            dump_memory,
            cross_at,
            cross_for,
        } => run_image(RunOptions {
            path,
            headless: true,
            limit: instructions,
            trace_cpu: cpu,
            trace_pcs: pc,
            trace_syscalls: syscalls,
            trace_from: from,
            watch_address: watch,
            host_scale: 1,
            frame_output: None,
            target_output: None,
            auto_cross_at: cross_at,
            auto_cross_for: cross_for,
            frame_dir: None,
            frame_interval: 50_000_000,
            memory_dumps: parse_memory_dumps(&dump_memory)?,
        })?,
    }
    Ok(())
}

#[cfg(feature = "desktop")]
fn run_image(options: RunOptions) -> Result<()> {
    let RunOptions {
        path,
        headless,
        limit,
        trace_cpu,
        trace_pcs,
        trace_syscalls,
        trace_from,
        watch_address,
        host_scale,
        frame_output,
        target_output,
        auto_cross_at,
        auto_cross_for,
        frame_dir,
        frame_interval,
        memory_dumps,
    } = options;
    let executable = psp_loader::load_executable(&path)
        .with_context(|| format!("cannot load executable from {}", path.display()))?;
    let image_info = psp_loader::inspect(&executable)?;
    tracing::debug!(image = ?image_info, "boot executable decoded");
    let mut emulator = psp_core::Emulator::new(0);
    if !headless {
        if emulator.audio.enable_host_output() {
            info!("SDL3 host audio enabled");
        } else {
            warn!("SDL3 host audio unavailable; continuing without host playback");
        }
    }
    // the 4 mib volatile partition sits immediately below the user partition
    // and is exposed by scekernelvolatilememlock for games to use as a heap.
    emulator
        .memory
        .map(VOLATILE_BASE, VOLATILE_SIZE, true, true)?;
    emulator.memory.map(EDRAM_BASE, EDRAM_SIZE, true, false)?;
    let entry = psp_loader::map_elf(&executable, &mut emulator.memory)?;
    let linked = psp_loader::link_prx_imports(&executable, &mut emulator.memory)?;
    emulator.cpu.gpr[28] = linked.gp;
    tracing::debug!(
        gp = format_args!("0x{:08x}", linked.gp),
        "boot module global pointer"
    );
    let imports: HashMap<_, _> = linked
        .imports
        .into_iter()
        .map(|import| (import.syscall, import))
        .collect();
    let partition_base = mapped_image_end(&image_info)?.next_multiple_of(256);
    let disc = psp_loader::IsoImage::open(&path).ok();
    let window = if headless {
        None
    } else {
        Some(
            Box::new(SdlHost::new(host_scale, DISPLAY_WIDTH, DISPLAY_HEIGHT)?)
                as Box<dyn HostDisplay>,
        )
    };
    let mut hle = HleState::new(disc, partition_base, window)?;
    emulator.gpu.set_hardware_rendering(!headless);
    // `PSP_HW_RECORD=1` forces hardware command recording on a headless
    // run. nothing is submitted to a gpu; per-heartbeat
    // `log_hardware_batch` summaries drain the stream, giving the same
    // ge-trace view (targets, depth/blend distribution) the graphical
    // backend consumes, without needing a display.
    if std::env::var_os("PSP_HW_RECORD").is_some() {
        emulator.gpu.set_hardware_rendering(true);
    }
    // a psp user thread starts with a stack in the top of user ram.  real
    // firmware also places argc/argv and thread metadata here; zero arguments
    // are sufficient until those kernel services are connected.
    // map all user ram not occupied by the executable.  the partition
    // allocator below accounts for stacks and kernel memory objects from the
    // same range, as the psp kernel does.
    emulator.memory.map(
        partition_base,
        (USER_RAM_END - partition_base) as usize,
        true,
        true,
    )?;
    const THREAD_CONTEXT_BASE: u32 = 0x09f0_0000;
    emulator.cpu.gpr[26] = THREAD_CONTEXT_BASE;
    emulator.cpu.gpr[29] = USER_RAM_END;
    emulator.cpu.gpr[4] = 0;
    emulator.cpu.gpr[5] = 0;
    emulator.cpu.pc = entry;
    hle.attach_main_thread(&emulator.cpu);
    info!(
        ?path,
        headless,
        limit,
        entry = format_args!("0x{entry:08x}"),
        "guest started"
    );
    let mut watched_value =
        watch_address.and_then(|address| emulator.memory.read_u32(address).ok());
    let mut recent_hle = VecDeque::with_capacity(16);
    let mut guest_step = 0;
    let mut next_heartbeat = 50_000_000;
    let mut next_pc_sample = 1_000_000;
    let mut pc_samples: HashMap<(u32, u32), u64> = HashMap::new();
    let mut fast_memory_loops = 0u64;
    let mut fast_memory_instructions = 0u64;
    let mut auto_cross_active = false;
    let mut next_frame_dump = frame_interval;
    // strides the memset fast-forward probe: it fires rarely, and the probe
    // itself costs two instruction fetches. delaying detection by a few
    // dozen blocks is unobservable (the skipped bytes are filled
    // identically either way).
    let mut ff_counter = 0u64;
    while guest_step < limit {
        if guest_step >= next_heartbeat {
            let mut top_pc_samples: Vec<_> = pc_samples
                .iter()
                .map(|(key, count)| (*key, *count))
                .collect();
            top_pc_samples.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.1));
            top_pc_samples.truncate(12);
            tracing::info!(
                target: "psp_rs::pc_sample",
                instructions = guest_step,
                ?top_pc_samples,
                "guest PC samples"
            );
            tracing::debug!(
                instructions = guest_step,
                scheduler = emulator.scheduler.now(),
                pc = format_args!("0x{:08x}", emulator.cpu.pc),
                ra = format_args!("0x{:08x}", emulator.cpu.gpr[31]),
                sp = format_args!("0x{:08x}", emulator.cpu.gpr[29]),
                a0 = format_args!("0x{:08x}", emulator.cpu.gpr[4]),
                a1 = format_args!("0x{:08x}", emulator.cpu.gpr[5]),
                a2 = format_args!("0x{:08x}", emulator.cpu.gpr[6]),
                a3 = format_args!("0x{:08x}", emulator.cpu.gpr[7]),
                thread = hle.current_uid,
                draws = emulator.gpu.draw_calls,
                framebuffer = format_args!("0x{:08x}", hle.display.frame_buffer),
                sas_bytes = emulator.audio.sas_bytes,
                channel_bytes = emulator.audio.channel_bytes,
                src_bytes = emulator.audio.src_bytes,
                mpeg_bytes = emulator.audio.mpeg_bytes,
                audio_overruns = emulator.audio.overruns,
                audio_underruns = emulator.audio.underruns,
                audio_busy = hle.audio.busy_returns,
                sas_grain = hle.sas.grain_size(),
                sas_initialized = hle.sas.is_initialized(),
                fast_memory_loops,
                fast_memory_instructions,
                "guest progress heartbeat"
            );
            // headless hardware-record runs (psp_hw_record=1 with a wrapper
            // forcing set_hardware_rendering(true)) never present, so drain
            // the recorded stream here: both as an oom guard and as the
            // per-batch ge trace (targets, depth/blend distribution).
            if hle.window.is_none() && emulator.gpu.hardware_rendering() {
                let commands = emulator.gpu.take_render_commands();
                if !commands.is_empty() {
                    log_hardware_batch(guest_step, &commands);
                }
            }
            // headless audio continuity: per-source submitted bytes show
            // whether the game keeps mixing sas/channel audio through
            // heavy phases (world streaming) or goes silent itself.
            if hle.window.is_none() {
                tracing::info!(
                    instructions = guest_step,
                    scheduler = emulator.scheduler.now(),
                    draws = emulator.gpu.draw_calls,
                    sas_bytes = emulator.audio.sas_bytes,
                    channel_bytes = emulator.audio.channel_bytes,
                    src_bytes = emulator.audio.src_bytes,
                    mpeg_bytes = emulator.audio.mpeg_bytes,
                    audio_overruns = emulator.audio.overruns,
                    audio_busy = hle.audio.busy_returns,
                    sas_grain = hle.sas.grain_size(),
                    sas_initialized = hle.sas.is_initialized(),
                    "headless audio progress"
                );
            }
            next_heartbeat = next_heartbeat.saturating_add(50_000_000);
        }
        if frame_dir.is_some() && guest_step >= next_frame_dump {
            next_frame_dump = guest_step.saturating_add(frame_interval);
            dump_display_frame(
                &emulator,
                &hle.display,
                &frame_dir.clone().expect("frame dir checked above"),
                guest_step,
            )?;
        }
        hle.wake_timed_waiters(&mut emulator);
        if guest_step >= next_pc_sample {
            *pc_samples
                .entry((hle.current_uid, emulator.cpu.pc))
                .or_default() += 1;
            next_pc_sample = next_pc_sample.saturating_add(1_000_000);
        }
        if !auto_cross_at.is_empty() {
            let cross_active = auto_cross_at.iter().any(|&cross_at| {
                guest_step >= cross_at && guest_step < cross_at.saturating_add(auto_cross_for)
            });
            if cross_active != auto_cross_active {
                emulator
                    .input
                    .update_buttons(if cross_active { PSP_CTRL_CROSS } else { 0 });
                auto_cross_active = cross_active;
                tracing::info!(
                    instructions = guest_step,
                    pressed = cross_active,
                    "automatic Cross input changed"
                );
            }
        }
        hle.sample_controller_if_due(&mut emulator)?;
        if !headless
            && guest_step.is_multiple_of(HOST_EVENT_POLL_INTERVAL)
            && !hle.poll_window(&mut emulator.input)
        {
            info!(instructions = guest_step, "host display window closed");
            return Ok(());
        }
        if emulator.cpu.pc == 0 {
            if hle.return_from_interrupt(&mut emulator)? {
                guest_step = guest_step.saturating_add(1);
                continue;
            }
            if !hle.finish_current_and_switch(&mut emulator) {
                info!(instructions = guest_step, "all guest threads exited");
                return Ok(());
            }
        }
        ff_counter = ff_counter.wrapping_add(1);
        if !trace_cpu
            && watch_address.is_none()
            && ff_counter.is_multiple_of(64)
            && let Some(executed) = emulator
                .cpu
                .try_fast_forward_memory_loop(&mut emulator.memory, hle.current_uid)?
        {
            fast_memory_loops += 1;
            fast_memory_instructions = fast_memory_instructions.saturating_add(executed);
            emulator.scheduler.advance(executed);
            guest_step = guest_step.saturating_add(executed);
            continue;
        }
        if trace_cpu
            && guest_step >= trace_from
            && (trace_pcs.is_empty() || trace_pcs.contains(&emulator.cpu.pc))
        {
            let word = emulator.memory.fetch_u32(emulator.cpu.pc).ok();
            tracing::debug!(
                pc = format_args!("0x{:08x}", emulator.cpu.pc),
                instruction = word.map(|word| format!("0x{word:08x}")),
                ra = format_args!("0x{:08x}", emulator.cpu.gpr[31]),
                sp = format_args!("0x{:08x}", emulator.cpu.gpr[29]),
                t9 = format_args!("0x{:08x}", emulator.cpu.gpr[25]),
                k0 = format_args!("0x{:08x}", emulator.cpu.gpr[26]),
                at = format_args!("0x{:08x}", emulator.cpu.gpr[1]),
                a0 = format_args!("0x{:08x}", emulator.cpu.gpr[4]),
                a1 = format_args!("0x{:08x}", emulator.cpu.gpr[5]),
                a2 = format_args!("0x{:08x}", emulator.cpu.gpr[6]),
                v0 = format_args!("0x{:08x}", emulator.cpu.gpr[2]),
                s0 = format_args!("0x{:08x}", emulator.cpu.gpr[16]),
                s1 = format_args!("0x{:08x}", emulator.cpu.gpr[17]),
                s2 = format_args!("0x{:08x}", emulator.cpu.gpr[18]),
                s3 = format_args!("0x{:08x}", emulator.cpu.gpr[19]),
                s4 = format_args!("0x{:08x}", emulator.cpu.gpr[20]),
                s5 = format_args!("0x{:08x}", emulator.cpu.gpr[21]),
                s6 = format_args!("0x{:08x}", emulator.cpu.gpr[22]),
                instructions = guest_step,
                thread = hle.current_uid,
                "CPU step"
            );
        }
        let step_pc = emulator.cpu.pc;
        let step_instruction = if (trace_cpu && guest_step >= trace_from) || watch_address.is_some()
        {
            emulator.memory.fetch_u32(step_pc).ok()
        } else {
            None
        };
        // the block jit shares the interpreter's semantic core (pre-decoded
        // opcode handlers run through `Cpu::execute_predecoded`), so this is
        // purely an execution strategy. tracing and watchpoints need
        // per-instruction granularity and stay on the single-step path.
        let use_jit = hle.jit_enabled && !trace_cpu && watch_address.is_none();
        let mut jit_counted = false;
        let mut step_pc = step_pc;
        let step_result = if use_jit {
            let mut budget = limit.saturating_sub(guest_step).max(1);
            // chained jit execution may cross several guest scheduling
            // boundaries. stop at the next deadline and host-input poll so
            // timed waiters and window events retain the same granularity as
            // the single-block frontend loop.
            if let Some(next_wake) = hle.timed_waiters.values().copied().min() {
                budget = budget.min(next_wake.saturating_sub(emulator.scheduler.now()).max(1));
            }
            let until_host_poll = HOST_EVENT_POLL_INTERVAL
                .saturating_sub(guest_step % HOST_EVENT_POLL_INTERVAL)
                .max(1);
            budget = budget.min(until_host_poll);
            match hle.jit.step_guest(
                &mut emulator.cpu,
                &mut emulator.memory,
                hle.current_uid,
                budget,
            ) {
                Ok(psp_cpu::BlockOutcome::Flowing(executed)) => {
                    emulator.scheduler.advance(executed);
                    guest_step = guest_step.saturating_add(executed);
                    continue;
                }
                Ok(psp_cpu::BlockOutcome::Syscall { code, pc, executed }) => {
                    // the single-step path advances the scheduler clock for
                    // `Continue` ops only; a syscall op itself consumes no
                    // virtual cycles there (hle dispatch accounts time
                    // separately). the jit must match that exactly, or
                    // guest-visible clocks drift per syscall.
                    emulator.scheduler.advance(executed.saturating_sub(1));
                    // the shared handling below logs `guest_step` as the
                    // pre-syscall count (matching the interpreter, where the
                    // bottom-of-loop increment runs after dispatch) and then
                    // counts the syscall op itself.
                    guest_step = guest_step.saturating_add(executed.saturating_sub(1));
                    jit_counted = false;
                    step_pc = pc;
                    Ok(psp_cpu::Step::Syscall(code))
                }
                Err(fault) => {
                    emulator.scheduler.advance(fault.executed);
                    guest_step = guest_step.saturating_add(fault.executed);
                    jit_counted = true;
                    Err(fault.error)
                }
            }
        } else {
            emulator
                .cpu
                .step_with_thread_id(&mut emulator.memory, hle.current_uid)
        };
        if let Some(address) = watch_address {
            let new_value = emulator.memory.read_u32(address).ok();
            if new_value != watched_value {
                tracing::warn!(
                    address = format_args!("0x{address:08x}"),
                    old = ?watched_value.map(|value| format!("0x{value:08x}")),
                    new = ?new_value.map(|value| format!("0x{value:08x}")),
                    pc = format_args!("0x{step_pc:08x}"),
                    instruction = ?step_instruction.map(|word| format!("0x{word:08x}")),
                    instructions = guest_step,
                    thread = hle.current_uid,
                    "guest watchpoint changed"
                );
                watched_value = new_value;
            }
        }
        match step_result {
            Ok(psp_cpu::Step::Continue) => emulator.scheduler.advance(1),
            Ok(psp_cpu::Step::Syscall(code)) => {
                let import = imports
                    .get(&code)
                    .with_context(|| format!("unknown guest syscall {code}"))?;
                recent_hle.push_back(format!(
                    "{}:{} pc=0x{:08x} {}::{:08x} a0=0x{:08x} a1=0x{:08x} a2=0x{:08x} a3=0x{:08x}",
                    guest_step,
                    hle.current_uid,
                    step_pc,
                    import.library,
                    import.nid,
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                    emulator.cpu.gpr[6],
                    emulator.cpu.gpr[7]
                ));
                if recent_hle.len() > 16 {
                    recent_hle.pop_front();
                }
                if trace_syscalls
                    && guest_step >= trace_from
                    && trace_thread_matches(hle.current_uid)
                {
                    let a0_string = read_guest_string(&emulator.memory, emulator.cpu.gpr[4]).ok();
                    tracing::debug!(instructions=guest_step, thread=hle.current_uid, pc=format_args!("0x{:08x}", step_pc), stub=format_args!("0x{:08x}", step_pc.wrapping_sub(4)), library=%import.library, nid=format_args!("0x{:08x}", import.nid), a0=format_args!("0x{:08x}", emulator.cpu.gpr[4]), a1=format_args!("0x{:08x}", emulator.cpu.gpr[5]), a2=format_args!("0x{:08x}", emulator.cpu.gpr[6]), a3=format_args!("0x{:08x}", emulator.cpu.gpr[7]), t0=format_args!("0x{:08x}", emulator.cpu.gpr[8]), t1=format_args!("0x{:08x}", emulator.cpu.gpr[9]), ra=format_args!("0x{:08x}", emulator.cpu.gpr[31]), ?a0_string, "HLE call");
                }
                let hle_started = timer_start();
                let dispatch_result = hle.dispatch(import, &mut emulator);
                let hle_ms = timer_elapsed(hle_started).as_secs_f64() * 1_000.0;
                if hle_ms >= 1.0 {
                    tracing::debug!(
                        target: "psp_rs::hle_timing",
                        library = %import.library,
                        nid = format_args!("0x{:08x}", import.nid),
                        pc = format_args!("0x{:08x}", step_pc),
                        thread = hle.current_uid,
                        instructions = guest_step,
                        hle_ms,
                        "slow HLE dispatch"
                    );
                }
                dispatch_result.with_context(|| {
                    format!(
                        "HLE {}::{:08x} failed at pc=0x{:08x} thread={}",
                        import.library, import.nid, step_pc, hle.current_uid
                    )
                })?;
                if trace_syscalls
                    && guest_step >= trace_from
                    && trace_thread_matches(hle.current_uid)
                {
                    tracing::debug!(
                        instructions = guest_step,
                        thread = hle.current_uid,
                        library = %import.library,
                        nid = format_args!("0x{:08x}", import.nid),
                        v0 = format_args!("0x{:08x}", emulator.cpu.gpr[2]),
                        v1 = format_args!("0x{:08x}", emulator.cpu.gpr[3]),
                        pc = format_args!("0x{:08x}", emulator.cpu.pc),
                        "HLE return"
                    );
                }
            }
            Err(error) => {
                // `Cpu::step` advances pc before executing an instruction, so
                // the faulting instruction is one word behind the exposed pc.
                let fault_pc = emulator.cpu.pc.wrapping_sub(4);
                let instruction = emulator.memory.fetch_u32(fault_pc).ok();
                let nearby = (fault_pc.saturating_sub(96)..=fault_pc)
                    .step_by(4)
                    .filter_map(|address| {
                        emulator
                            .memory
                            .fetch_u32(address)
                            .ok()
                            .map(|word| format!("0x{address:08x}:0x{word:08x}"))
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let stack = (0..=0x80u32)
                    .step_by(4)
                    .filter_map(|offset| {
                        emulator
                            .memory
                            .read_u32(emulator.cpu.gpr[29].wrapping_add(offset))
                            .ok()
                            .map(|word| {
                                format!(
                                    "0x{:08x}:0x{word:08x}",
                                    emulator.cpu.gpr[29].wrapping_add(offset)
                                )
                            })
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                anyhow::bail!(
                    "guest halted after {} instructions: {error}\n\
                     CPU context: fault_pc=0x{fault_pc:08x} instruction={} ra=0x{:08x} \
                     sp=0x{:08x} gp=0x{:08x} a0=0x{:08x} a1=0x{:08x} a2=0x{:08x} \
                     a3=0x{:08x} v0=0x{:08x} v1=0x{:08x} t9=0x{:08x} \
                     s0=0x{:08x} s1=0x{:08x} s2=0x{:08x} s3=0x{:08x} \
                     s4=0x{:08x} s5=0x{:08x} s6=0x{:08x} s7=0x{:08x}\nNearby instructions: {nearby}\nStack words: {stack}\nRecent HLE calls: {}",
                    guest_step,
                    instruction.map_or_else(
                        || "<unavailable>".to_owned(),
                        |word| format!("0x{word:08x}")
                    ),
                    emulator.cpu.gpr[31],
                    emulator.cpu.gpr[29],
                    emulator.cpu.gpr[28],
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                    emulator.cpu.gpr[6],
                    emulator.cpu.gpr[7],
                    emulator.cpu.gpr[2],
                    emulator.cpu.gpr[3],
                    emulator.cpu.gpr[25],
                    emulator.cpu.gpr[16],
                    emulator.cpu.gpr[17],
                    emulator.cpu.gpr[18],
                    emulator.cpu.gpr[19],
                    emulator.cpu.gpr[20],
                    emulator.cpu.gpr[21],
                    emulator.cpu.gpr[22],
                    emulator.cpu.gpr[23],
                    recent_hle
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join("; "),
                )
            }
        }
        guest_step = if jit_counted {
            guest_step
        } else {
            guest_step.saturating_add(1)
        };
    }
    let nearby = (emulator.cpu.pc.saturating_sub(32)..=emulator.cpu.pc.wrapping_add(32))
        .step_by(4)
        .filter_map(|address| {
            emulator
                .memory
                .fetch_u32(address)
                .ok()
                .map(|word| format!("0x{address:08x}:0x{word:08x}"))
        })
        .collect::<Vec<_>>()
        .join(" ");
    let return_nearby = (emulator.cpu.gpr[31].saturating_sub(32)
        ..=emulator.cpu.gpr[31].wrapping_add(32))
        .step_by(4)
        .filter_map(|address| {
            emulator
                .memory
                .fetch_u32(address)
                .ok()
                .map(|word| format!("0x{address:08x}:0x{word:08x}"))
        })
        .collect::<Vec<_>>()
        .join(" ");
    info!(
        instructions = limit,
        pc = format_args!("0x{:08x}", emulator.cpu.pc),
        thread = hle.current_uid,
        a0 = format_args!("0x{:08x}", emulator.cpu.gpr[4]),
        a1 = format_args!("0x{:08x}", emulator.cpu.gpr[5]),
        a2 = format_args!("0x{:08x}", emulator.cpu.gpr[6]),
        a3 = format_args!("0x{:08x}", emulator.cpu.gpr[7]),
        t0 = format_args!("0x{:08x}", emulator.cpu.gpr[8]),
        t4 = format_args!("0x{:08x}", emulator.cpu.gpr[12]),
        t5 = format_args!("0x{:08x}", emulator.cpu.gpr[13]),
        s0 = format_args!("0x{:08x}", emulator.cpu.gpr[16]),
        vfpu_cc = format_args!("0x{:02x}", emulator.cpu.vfpu_cc),
        v0f = f32::from_bits(emulator.cpu.vfpu[0]),
        v1f = f32::from_bits(emulator.cpu.vfpu[1]),
        v28f = f32::from_bits(emulator.cpu.vfpu[28]),
        ra = format_args!("0x{:08x}", emulator.cpu.gpr[31]),
        %nearby,
        %return_nearby,
        "instruction limit reached"
    );
    hle.log_scheduler_state();
    if hle.jit_enabled {
        info!(
            compiled = hle.jit.compiled(),
            blocks = hle.jit.block_count(),
            hits = hle.jit.cache_hits(),
            misses = hle.jit.cache_misses(),
            invalidated = hle.jit.invalidated(),
            jit_ops = hle.jit.executed_ops(),
            jit_blocks = hle.jit.executed_blocks(),
            "block JIT cache statistics"
        );
    }
    for (start, length, path) in &memory_dumps {
        // fault-tolerant: unmapped holes are zero-filled so one sparse
        // range still dumps surrounding regions (user ram is mapped as
        // executable plus individually allocated stack/heap blocks).
        let mut bytes = Vec::new();
        let mut cursor = *start;
        let end = start.wrapping_add(*length);
        let mut holes = 0u32;
        while cursor < end {
            let page_end = (cursor | 0xfff).wrapping_add(1);
            let step = (end - cursor).min(page_end.saturating_sub(cursor)) as usize;
            if step == 0 {
                break;
            }
            match emulator.memory.read_bytes(cursor, step) {
                Ok(chunk) => {
                    bytes.extend_from_slice(&chunk);
                    cursor = cursor.wrapping_add(step as u32);
                }
                Err(_) => {
                    bytes.extend(std::iter::repeat_n(0, step));
                    cursor = cursor.wrapping_add(step as u32);
                    holes += 1;
                }
            }
        }
        std::fs::write(path, &bytes)
            .with_context(|| format!("cannot write memory dump {}", path.display()))?;
        info!(
            start = format_args!("0x{start:08x}"),
            bytes = bytes.len(),
            requested = length,
            holes,
            path = %path.display(),
            "guest memory dumped"
        );
    }
    if hle.display.frame_buffer != 0 && hle.display.width != 0 && hle.display.height != 0 {
        let pixels = psp_gpu::Gpu::framebuffer_rgba(
            &emulator.memory,
            hle.display.frame_buffer,
            hle.display.buffer_width,
            hle.display.width,
            hle.display.height,
            hle.display.pixel_format,
        )?;
        let nonzero_pixels = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|pixel| pixel[..3] != [0, 0, 0])
            .count();
        let checksum = pixels.iter().fold(0u64, |sum, byte| {
            sum.rotate_left(5).wrapping_add(u64::from(*byte))
        });
        info!(
            address = format_args!("0x{:08x}", hle.display.frame_buffer),
            width = hle.display.width,
            height = hle.display.height,
            stride = hle.display.buffer_width,
            format = hle.display.pixel_format,
            nonzero_pixels,
            checksum = format_args!("0x{checksum:016x}"),
            "display framebuffer inspected"
        );
        if let Some(path) = frame_output {
            write_ppm(&path, hle.display.width, hle.display.height, &pixels)
                .with_context(|| format!("cannot write framebuffer dump {}", path.display()))?;
            info!(path = %path.display(), "display framebuffer dumped");
        }
        if let Some(path) = target_output {
            let target = psp_gpu::Gpu::framebuffer_rgba(
                &emulator.memory,
                EDRAM_BASE | 0x0008_8000,
                512,
                hle.display.width,
                hle.display.height,
                hle.display.pixel_format,
            )?;
            write_ppm(&path, hle.display.width, hle.display.height, &target)
                .with_context(|| format!("cannot write render-target dump {}", path.display()))?;
            info!(path = %path.display(), "GE render target dumped");
        }
    }
    Ok(())
}

impl HleState {
    #[cfg(feature = "desktop")]
    fn log_scheduler_state(&self) {
        for (&uid, thread) in &self.threads {
            let state = if uid == self.current_uid {
                "running"
            } else if self.ready.contains(&uid) {
                "ready"
            } else if self.timed_waiters.contains_key(&uid) {
                "timed-wait"
            } else if thread.cpu.is_some() {
                "blocked"
            } else {
                "unrunnable"
            };
            tracing::debug!(
                uid,
                entry = format_args!("0x{:08x}", thread.entry),
                priority = thread.priority,
                state,
                pc = thread.cpu.as_ref().map_or(0, |cpu| cpu.pc),
                "guest thread state"
            );
        }
        for (&uid, semaphore) in &self.semaphores {
            if !semaphore.waiters.is_empty() {
                let waiters = semaphore
                    .waiters
                    .iter()
                    .map(|waiter| waiter.thread)
                    .collect::<Vec<_>>();
                tracing::debug!(uid, semaphore.count, ?waiters, "guest semaphore waiters");
            }
        }
        for (&uid, flag) in &self.event_flags {
            if !flag.waiters.is_empty() {
                let waiters = flag
                    .waiters
                    .iter()
                    .map(|waiter| {
                        format!(
                            "{}:bits=0x{:x}:mode=0x{:x}",
                            waiter.thread, waiter.bits, waiter.mode
                        )
                    })
                    .collect::<Vec<_>>();
                tracing::debug!(
                    uid,
                    bits = format_args!("0x{:08x}", flag.bits),
                    ?waiters,
                    "guest event flag waiters"
                );
            }
        }
        if !self.volatile_waiters.is_empty() {
            let waiters = self
                .volatile_waiters
                .iter()
                .map(|waiter| waiter.thread)
                .collect::<Vec<_>>();
            tracing::debug!(?waiters, "guest volatile waiters");
        }
        if !self.suspended.is_empty() {
            let mut suspended: Vec<u32> = self.suspended.iter().copied().collect();
            suspended.sort_unstable();
            tracing::debug!(?suspended, "guest suspended threads");
        }
        if !self.ctrl_waiters.is_empty() {
            let waiters = self
                .ctrl_waiters
                .iter()
                .map(|waiter| waiter.thread)
                .collect::<Vec<_>>();
            tracing::debug!(?waiters, "guest controller waiters");
        }
        for (&uid, deadline) in &self.timed_waiters {
            tracing::debug!(
                uid,
                deadline,
                has_timeout_result = self.timed_wait_results.contains_key(&uid),
                "guest timed waiter"
            );
        }
    }
}

/// Summarize one headless hardware-record batch: EDRAM target usage plus the
/// depth/blend state distribution consumed by the host GPU path. This is the
/// GE-trace proof for world-invisible divergences, such as every world draw
/// using depth-tested GEQUAL against a host depth buffer that was never
/// written. `PSP_HW_TRACE=1` additionally logs per-target vertex-depth ranges
/// for depth-tested draws.
#[cfg(feature = "desktop")]
fn log_hardware_batch(instructions: u64, commands: &[psp_gpu::RenderCommand]) {
    use std::collections::HashMap;
    let mut targets: HashMap<(u32, u32, u32, u32), usize> = HashMap::new();
    let mut depth: HashMap<(bool, u32, bool), usize> = HashMap::new();
    let mut blend: HashMap<u32, usize> = HashMap::new();
    let mut clears = 0usize;
    let mut clear_depths: Vec<u16> = Vec::new();
    for command in commands {
        *targets
            .entry((
                command.framebuffer_address,
                command.framebuffer_width,
                command.depth_address,
                command.depth_width,
            ))
            .or_default() += 1;
        *depth
            .entry((
                command.depth_test,
                command.depth_function,
                command.depth_write,
            ))
            .or_default() += 1;
        *blend
            .entry(if command.blend {
                command.blend_state
            } else {
                0
            })
            .or_default() += 1;
        if command.clear_mode & 1 != 0 {
            clears += 1;
            if command.clear_mode & 0x400 != 0 && !clear_depths.contains(&command.clear_depth) {
                clear_depths.push(command.clear_depth);
            }
        }
    }
    let mut targets: Vec<_> = targets.into_iter().collect();
    targets.sort_unstable();
    let mut depth: Vec<_> = depth.into_iter().collect();
    depth.sort_unstable();
    let mut blend: Vec<_> = blend.into_iter().collect();
    blend.sort_unstable();
    tracing::info!(
        instructions,
        draws = commands.len(),
        clears,
        ?clear_depths,
        ?targets,
        ?depth,
        ?blend,
        "hardware-record batch"
    );
    if std::env::var_os("PSP_HW_TRACE").is_some() {
        let mut ranges: HashMap<(u32, u32), (u16, u16, usize)> = HashMap::new();
        for command in commands {
            if command.clear_mode & 1 != 0 || !command.depth_test {
                continue;
            }
            let entry = ranges
                .entry((command.framebuffer_address, command.depth_address))
                .or_insert((u16::MAX, u16::MIN, 0));
            for vertex in &command.vertices {
                entry.0 = entry.0.min(vertex.depth);
                entry.1 = entry.1.max(vertex.depth);
            }
            entry.2 += command.vertices.len();
        }
        let mut ranges: Vec<_> = ranges.into_iter().collect();
        ranges.sort_unstable();
        tracing::info!(instructions, ?ranges, "hardware-record depth ranges");
    }
}

#[cfg(feature = "desktop")]
fn dump_display_frame(
    emulator: &psp_core::Emulator,
    display: &DisplayState,
    frame_dir: &std::path::Path,
    guest_step: u64,
) -> Result<()> {
    if display.frame_buffer == 0 || display.width == 0 || display.height == 0 {
        return Ok(());
    }
    let pixels = psp_gpu::Gpu::framebuffer_rgba(
        &emulator.memory,
        display.frame_buffer,
        display.buffer_width,
        display.width,
        display.height,
        display.pixel_format,
    )
    .with_context(|| format!("cannot read framebuffer 0x{:08x}", display.frame_buffer))?;
    std::fs::create_dir_all(frame_dir)
        .with_context(|| format!("cannot create {}", frame_dir.display()))?;
    let path = frame_dir.join(format!("frame_{guest_step:016}.ppm"));
    write_ppm(&path, display.width, display.height, &pixels)
        .with_context(|| format!("cannot write framebuffer dump {}", path.display()))?;
    tracing::info!(
        instructions = guest_step,
        path = %path.display(),
        "periodic framebuffer dumped"
    );
    Ok(())
}

#[cfg(feature = "desktop")]
fn write_ppm(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .context("framebuffer dump dimensions overflow")? as usize;
    anyhow::ensure!(
        rgba.len() == expected,
        "framebuffer dump received {} bytes for {}x{} RGBA8",
        rgba.len(),
        width,
        height
    );
    let header = format!("P6\n{width} {height}\n255\n");
    let mut ppm = Vec::with_capacity(header.len() + (width as usize * height as usize * 3));
    ppm.extend_from_slice(header.as_bytes());
    for pixel in rgba.as_chunks::<4>().0.iter() {
        ppm.extend_from_slice(&pixel[..3]);
    }
    std::fs::write(path, ppm).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

#[cfg(feature = "desktop")]
fn parse_u32(value: &str) -> std::result::Result<u32, String> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).map_err(|error| error.to_string())
    } else {
        value.parse::<u32>().map_err(|error| error.to_string())
    }
}

/// Parse `START:LENGTH:PATH` memory-dump specifications, with hexadecimal
/// addresses and decimal or `0x` lengths.
#[cfg(feature = "desktop")]
fn parse_memory_dumps(specs: &[String]) -> Result<Vec<(u32, u32, PathBuf)>> {
    specs
        .iter()
        .map(|spec| {
            let mut parts = spec.splitn(3, ':');
            let start_text = parts.next().context("memory dump needs START:LEN:PATH")?;
            let start: u32 =
                parse_u32(start_text).map_err(|_| anyhow::anyhow!("bad START in {spec}"))?;
            let length_text = parts.next().context("memory dump needs LENGTH")?;
            let length = parse_u32(length_text).map_err(|error| anyhow::anyhow!("{error}"))?;
            let path = PathBuf::from(parts.next().context("memory dump needs PATH")?);
            Ok((start, length, path))
        })
        .collect()
}

#[cfg(feature = "desktop")]
fn parse_scale(value: &str) -> std::result::Result<u32, String> {
    let scale = parse_u32(value)?;
    if (1..=4).contains(&scale) {
        Ok(scale)
    } else {
        Err("scale must be an integer from 1 to 4".to_owned())
    }
}

#[cfg(feature = "desktop")]
fn analog_axis(negative: bool, positive: bool) -> u8 {
    match (negative, positive) {
        (true, false) => 0,
        (false, true) => u8::MAX,
        _ => ANALOG_CENTER,
    }
}

#[cfg(feature = "desktop")]
fn host_button_mask(is_pressed: impl Fn(Scancode) -> bool) -> u32 {
    [
        (Scancode::Up, PSP_CTRL_UP),
        (Scancode::Right, PSP_CTRL_RIGHT),
        (Scancode::Down, PSP_CTRL_DOWN),
        (Scancode::Left, PSP_CTRL_LEFT),
        (Scancode::Return, PSP_CTRL_START),
        (Scancode::Backspace, PSP_CTRL_SELECT),
        (Scancode::Escape, PSP_CTRL_HOME),
        (Scancode::Z, PSP_CTRL_CIRCLE),
        (Scancode::X, PSP_CTRL_CROSS),
        (Scancode::A, PSP_CTRL_SQUARE),
        (Scancode::S, PSP_CTRL_TRIANGLE),
        (Scancode::Q, PSP_CTRL_LTRIGGER),
        (Scancode::W, PSP_CTRL_RTRIGGER),
    ]
    .into_iter()
    .filter(|(key, _)| is_pressed(*key))
    .fold(0, |buttons, (_, button)| buttons | button)
}

fn write_controller_sample(
    memory: &mut psp_memory::Memory,
    address: u32,
    sample: ControllerSample,
) -> Result<()> {
    memory.write_u32(address, sample.frame)?;
    memory.write_u32(address.wrapping_add(4), sample.buttons)?;
    memory.write_u8(address.wrapping_add(8), sample.analog_x)?;
    memory.write_u8(address.wrapping_add(9), sample.analog_y)?;
    memory.write_bytes(address.wrapping_add(10), &[0; 6])?;
    Ok(())
}

struct HleState {
    next_uid: u32,
    threads: HashMap<u32, GuestThread>,
    ready: VecDeque<u32>,
    suspended: HashSet<u32>,
    current_uid: u32,
    disc: Option<psp_loader::IsoImage>,
    next_fd: u32,
    directories: HashMap<u32, GuestDirectory>,
    files: HashMap<u32, GuestFile>,
    async_operations: HashMap<u32, AsyncFileOperation>,
    modules: HashSet<u32>,
    user_partition: PartitionAllocator,
    memory_blocks: HashMap<u32, u32>,
    fixed_pools: HashMap<u32, FixedPool>,
    semaphores: HashMap<u32, Semaphore>,
    event_flags: HashMap<u32, EventFlag>,
    mpeg: MpegState,
    atrac: AtracState,
    audio: AudioState,
    sas: psp_audio::SasMixer,
    ctrl_mode: u32,
    ctrl_cycle: u32,
    ctrl_last_sample: u64,
    ctrl_waiters: VecDeque<ControllerWaiter>,
    ctrl_idle_reset: i32,
    ctrl_idle_back: i32,
    umd_activated: bool,
    umd_callback: Option<u32>,
    ge_lists: HashMap<u32, GeList>,
    ge_callbacks: HashMap<u32, GeCallback>,
    display: DisplayState,
    callbacks: HashMap<u32, (u32, u32)>,
    unknown_imports: HashSet<(String, u32)>,
    volatile_locked: bool,
    /// Threads blocked in `sceKernelVolatileMemLock` (blocking form) while
    /// waiting for the current holder to call `sceKernelVolatileMemUnlock`.
    /// Each entry remembers the output pointers from its lock call so the
    /// transferred lock can write the base and size on wake.
    volatile_waiters: VecDeque<VolatileWaiter>,
    subinterrupts: HashMap<(u32, u32), SubInterrupt>,
    interrupt_stack: Vec<psp_cpu::Cpu>,
    pending_interrupts: VecDeque<PendingInterrupt>,
    pending_ringbuffer_put: Option<PendingRingbufferPut>,
    timed_waiters: HashMap<u32, u64>,
    /// Threads blocked in a synchronization wait with a guest timeout carry
    /// the v0 result delivered when the deadline expires (
    /// `SCE_KERNEL_ERROR_WAIT_TIMEOUT`).  plain delays use
    /// `wait_current_for` and never insert here; an explicit wake via
    /// `wake_thread` always cancels the pending deadline.
    timed_wait_results: HashMap<u32, u32>,
    /// GE lists may be submitted several times during one display interval.
    /// Keep their decoded commands ordered and submit them as one host batch at
    /// the next display boundary instead of forcing one GPU submission per list.
    #[cfg(feature = "desktop")]
    pending_render_commands: Vec<psp_gpu::RenderCommand>,
    #[cfg(feature = "desktop")]
    window: Option<Box<dyn HostDisplay>>,
    #[cfg(feature = "desktop")]
    last_host_present: Option<Instant>,
    #[cfg(feature = "desktop")]
    next_host_vblank: Option<Instant>,
    #[cfg(feature = "desktop")]
    last_host_buttons: u32,
    #[cfg(feature = "desktop")]
    host_started_at: Instant,
    #[cfg(feature = "desktop")]
    hardware_frame_valid: bool,
    /// A vblank callback runs before the waiting thread is released. The
    /// callback path used to skip the vblank delay entirely.
    vblank_wait_pending: bool,
    jit: psp_cpu::Jit,
    #[cfg(feature = "desktop")]
    jit_enabled: bool,
}

struct SubInterrupt {
    handler: u32,
    argument: u32,
    enabled: bool,
}

struct GeList {
    start: u32,
    stall: u32,
    completed: bool,
    /// The `sceGeSetCallback` id passed at enqueue time, if any.
    callback_id: u32,
}

#[derive(Clone, Copy, Debug)]
struct GeCallback {
    _signal_function: u32,
    _signal_argument: u32,
    finish_function: u32,
    finish_argument: u32,
}

/// A guest handler queued for execution at interrupt level.
///
/// GE lists can raise several events per execution (signal commands plus the
/// finish completion), and each one must run its registered guest handler to
/// completion before the interrupted thread resumes.  the queue keeps the
/// handlers ordered; `return_from_interrupt` dispatches the next entry until
/// the queue drains.
#[derive(Clone, Copy, Debug)]
struct PendingInterrupt {
    handler: u32,
    argument: u32,
    token: u32,
}

struct PendingRingbufferPut {
    ringbuffer_addr: u32,
    write_position: u32,
    target_packets: u32,
    current_packets: u32,
    added_packets: u32,
}

#[derive(Default)]
struct DisplayState {
    mode: u32,
    width: u32,
    height: u32,
    frame_buffer: u32,
    buffer_width: u32,
    pixel_format: u32,
    explicit_frame_buffer: bool,
}

struct Semaphore {
    count: u32,
    max_count: u32,
    waiters: VecDeque<SemaphoreWaiter>,
}

struct SemaphoreWaiter {
    thread: u32,
    count: u32,
}

struct EventFlag {
    bits: u32,
    waiters: VecDeque<EventFlagWaiter>,
}

struct EventFlagWaiter {
    thread: u32,
    bits: u32,
    mode: u32,
    out_bits: u32,
}

struct ControllerWaiter {
    thread: u32,
    data: u32,
    negative: bool,
}

struct VolatileWaiter {
    thread: u32,
    address_output: u32,
    size_output: u32,
}

#[derive(Default)]
struct AudioState {
    reserved: [bool; AUDIO_CHANNEL_COUNT],
    sample_counts: [u32; AUDIO_CHANNEL_COUNT],
    formats: [u32; AUDIO_CHANNEL_COUNT],
    left_volumes: [u32; AUDIO_CHANNEL_COUNT],
    right_volumes: [u32; AUDIO_CHANNEL_COUNT],
    queued_samples: [u32; AUDIO_CHANNEL_COUNT],
    output2_reserved: bool,
    output2_sample_count: u32,
    output2_queued_samples: u32,
    last_update_us: u64,
    sample_remainder: u64,
    /// Non-blocking outputs rejected with `CHANNEL_BUSY`. A rising count while
    /// SAS/channel submits stall means the guest outruns the virtual audio
    /// clock during a compute-bound phase rather than going silent.
    busy_returns: u64,
}

impl AudioState {
    fn advance_to(&mut self, now_us: u64) {
        if now_us <= self.last_update_us {
            return;
        }
        let elapsed = now_us - self.last_update_us;
        let work = elapsed
            .saturating_mul(AUDIO_SAMPLE_RATE)
            .saturating_add(self.sample_remainder);
        let drained = work / 1_000_000;
        self.sample_remainder = work % 1_000_000;
        self.last_update_us = now_us;
        if drained == 0 {
            return;
        }
        let drained = drained.min(u64::from(u32::MAX)) as u32;
        for queued in &mut self.queued_samples {
            *queued = queued.saturating_sub(drained);
        }
        self.output2_queued_samples = self.output2_queued_samples.saturating_sub(drained);
    }

    fn wait_duration_us(queued_samples: u32) -> u64 {
        // sceaudiooutputblocking waits until the previously queued
        // buffer has drained before the forcing thread continues. waiting
        // for only one 64-sample mixer block lets a high-priority audio
        // thread re-enqueue faster than real time, growing the emulated
        // queue and returning channel_busy to the game's double-buffer.
        let samples = u64::from(queued_samples);
        if samples == 0 {
            0
        } else {
            (samples * 1_000_000).div_ceil(AUDIO_SAMPLE_RATE)
        }
    }

    fn output2_wait_duration_us(queued_samples: u32) -> u64 {
        let samples = u64::from(queued_samples);
        if samples == 0 {
            0
        } else {
            (samples * 1_000_000).div_ceil(AUDIO_SAMPLE_RATE)
        }
    }

    fn enqueue(
        &mut self,
        channel: u32,
        sample_count: u32,
        has_buffer: bool,
        blocking: bool,
    ) -> Result<u64, u32> {
        let Some(queued) = self.queued_samples.get(channel as usize).copied() else {
            return Err(AUDIO_ERROR_INVALID_CHANNEL);
        };
        let delay = if queued != 0 && !blocking {
            self.busy_returns += 1;
            return Err(AUDIO_ERROR_CHANNEL_BUSY);
        } else if blocking {
            Self::wait_duration_us(queued)
        } else {
            0
        };
        if has_buffer {
            self.queued_samples[channel as usize] = queued.saturating_add(sample_count);
        }
        Ok(delay)
    }

    fn enqueue_output2(
        &mut self,
        sample_count: u32,
        has_buffer: bool,
        blocking: bool,
    ) -> Result<u64, u32> {
        let queued = self.output2_queued_samples;
        if queued != 0 && !blocking {
            self.busy_returns += 1;
            return Err(AUDIO_ERROR_CHANNEL_BUSY);
        }
        let delay = if blocking {
            // src output is a producer/consumer boundary.  waiting for only
            // one mixer block lets a high-priority audio thread enqueue a
            // full 0x800-sample buffer every 1.45 ms, growing the emulated
            // queue without giving the game worker time to run.
            Self::output2_wait_duration_us(queued)
        } else {
            0
        };
        if has_buffer {
            self.output2_queued_samples = queued.saturating_add(sample_count);
        }
        Ok(delay)
    }

    fn channel_rest(&self, channel: u32) -> Result<u32, u32> {
        let Some(queued) = self.queued_samples.get(channel as usize) else {
            return Err(AUDIO_ERROR_INVALID_CHANNEL);
        };
        if !self.reserved[channel as usize] {
            return Err(AUDIO_ERROR_CHANNEL_NOT_INIT);
        }
        Ok(*queued)
    }

    fn reserve(&mut self, requested: i32, sample_count: u32, format: u32) -> u32 {
        let channel = if requested < 0 {
            (1..AUDIO_CHANNEL_COUNT)
                .rev()
                .find(|&channel| !self.reserved[channel])
                .map_or(AUDIO_ERROR_NO_CHANNELS, |channel| channel as u32)
        } else {
            requested as u32
        };
        if channel >= AUDIO_CHANNEL_COUNT as u32 {
            return AUDIO_ERROR_INVALID_CHANNEL;
        }
        if sample_count == 0
            || sample_count & AUDIO_MIX_BLOCK_MASK != 0
            || sample_count > AUDIO_SAMPLE_MAX
        {
            return AUDIO_ERROR_SAMPLE_SIZE;
        }
        if format != 0 && format != 0x10 {
            return AUDIO_ERROR_INVALID_FORMAT;
        }
        let channel = channel as usize;
        if self.reserved[channel] {
            return AUDIO_ERROR_INVALID_CHANNEL;
        }
        self.reserved[channel] = true;
        self.sample_counts[channel] = sample_count;
        self.formats[channel] = format;
        self.left_volumes[channel] = u32::MAX;
        self.right_volumes[channel] = u32::MAX;
        self.queued_samples[channel] = 0;
        channel as u32
    }

    fn set_data_len(&mut self, channel: u32, sample_count: u32) -> u32 {
        if channel >= AUDIO_CHANNEL_COUNT as u32 {
            return AUDIO_ERROR_INVALID_CHANNEL;
        }
        if !self.reserved[channel as usize] {
            return AUDIO_ERROR_CHANNEL_NOT_INIT;
        }
        if sample_count == 0
            || sample_count & AUDIO_MIX_BLOCK_MASK != 0
            || sample_count > AUDIO_SAMPLE_MAX
        {
            return AUDIO_ERROR_SAMPLE_SIZE;
        }
        self.sample_counts[channel as usize] = sample_count;
        0
    }

    fn change_format(&mut self, channel: u32, format: u32) -> u32 {
        if channel >= AUDIO_CHANNEL_COUNT as u32 {
            return AUDIO_ERROR_INVALID_CHANNEL;
        }
        let channel = channel as usize;
        if !self.reserved[channel] {
            return AUDIO_ERROR_CHANNEL_NOT_INIT;
        }
        if format != 0 && format != 0x10 {
            return AUDIO_ERROR_INVALID_FORMAT;
        }
        self.formats[channel] = format;
        0
    }

    fn change_volume(&mut self, channel: u32, left: u32, right: u32) -> u32 {
        if channel >= AUDIO_CHANNEL_COUNT as u32 {
            return AUDIO_ERROR_INVALID_CHANNEL;
        }
        let channel = channel as usize;
        if !self.reserved[channel] {
            return AUDIO_ERROR_CHANNEL_NOT_INIT;
        }
        if left > 0xffff || right > 0xffff {
            return AUDIO_ERROR_INVALID_VOLUME;
        }
        self.left_volumes[channel] = left;
        self.right_volumes[channel] = right;
        0
    }

    fn output_volumes(&mut self, channel: u32, left: u32, right: u32) -> Result<(u32, u32), u32> {
        let Some(index) = usize::try_from(channel)
            .ok()
            .filter(|&index| index < AUDIO_CHANNEL_COUNT)
        else {
            return Err(AUDIO_ERROR_INVALID_CHANNEL);
        };
        if !self.reserved[index] {
            return Err(AUDIO_ERROR_CHANNEL_NOT_INIT);
        }
        if left != u32::MAX {
            self.left_volumes[index] = left;
        }
        if right != u32::MAX {
            self.right_volumes[index] = right;
        }
        Ok((self.left_volumes[index], self.right_volumes[index]))
    }

    fn release(&mut self, channel: u32) -> u32 {
        if channel >= AUDIO_CHANNEL_COUNT as u32 {
            return AUDIO_ERROR_INVALID_CHANNEL;
        }
        let channel = channel as usize;
        if !self.reserved[channel] {
            return AUDIO_ERROR_CHANNEL_NOT_INIT;
        }
        self.reserved[channel] = false;
        self.sample_counts[channel] = 0;
        self.formats[channel] = 0;
        self.left_volumes[channel] = u32::MAX;
        self.right_volumes[channel] = u32::MAX;
        self.queued_samples[channel] = 0;
        0
    }

    fn channel_output(&self, channel: u32) -> Option<(u32, u32)> {
        let channel = usize::try_from(channel).ok()?;
        (channel < AUDIO_CHANNEL_COUNT && self.reserved[channel])
            .then_some((self.sample_counts[channel], self.formats[channel]))
    }

    fn reserve_output2(&mut self, sample_count: u32) -> u32 {
        if !(17..=4_111).contains(&sample_count) {
            return AUDIO_ERROR_SAMPLE_SIZE;
        }
        if self.output2_reserved {
            return AUDIO_ERROR_CHANNEL_ALREADY_RESERVED;
        }
        self.output2_reserved = true;
        self.output2_sample_count = sample_count;
        self.output2_queued_samples = 0;
        0
    }

    fn release_output2(&mut self) -> u32 {
        if !self.output2_reserved {
            return AUDIO_ERROR_CHANNEL_NOT_INIT;
        }
        self.output2_reserved = false;
        self.output2_sample_count = 0;
        self.output2_queued_samples = 0;
        0
    }

    fn change_output2_length(&mut self, sample_count: u32) -> u32 {
        if !self.output2_reserved {
            return AUDIO_ERROR_CHANNEL_NOT_INIT;
        }
        if !(17..=4_111).contains(&sample_count) {
            return AUDIO_ERROR_SAMPLE_SIZE;
        }
        self.output2_sample_count = sample_count;
        0
    }
}

#[derive(Default)]
struct AtracState {
    contexts: HashMap<u32, AtracContext>,
}

struct AtracContext {
    buffer: u32,
    buffer_size: u32,
    write_offset: u32,
    buffered_bytes: u32,
    next_file_offset: u32,
    format: AtracFormat,
    decoder: Option<AtracDecoder>,
    decoded_samples: u32,
    loop_num: i32,
}

impl AtracState {
    fn set_halfway_buffer_and_get_id(
        &mut self,
        memory: &Memory,
        buffer: u32,
        read_size: u32,
        buffer_size: u32,
    ) -> Result<u32> {
        if buffer_size == 0 || read_size > buffer_size {
            return Ok(ATRAC_ERROR_INCORRECT_READ_SIZE);
        }
        let initial = memory.read_bytes(buffer, read_size as usize)?;
        let Some(format) = inspect_atrac(&initial) else {
            return Ok(ATRAC_ERROR_UNKNOWN_FORMAT);
        };
        let Some(id) = (0..ATRAC_MAX_IDS).find(|id| !self.contexts.contains_key(id)) else {
            return Ok(ATRAC_ERROR_NO_ATRACID);
        };
        let mut context = AtracContext {
            buffer,
            buffer_size,
            write_offset: read_size % buffer_size,
            buffered_bytes: read_size,
            next_file_offset: read_size,
            format,
            decoder: AtracDecoder::spawn(),
            decoded_samples: 0,
            loop_num: 0,
        };
        if let Some(decoder) = context.decoder.as_ref()
            && !decoder.submit(&initial)
        {
            context.decoder = None;
            tracing::warn!(id, "ATRAC host decoder rejected the initial track bytes");
        }
        tracing::info!(
            id,
            buffer = format_args!("0x{buffer:08x}"),
            read_size,
            buffer_size,
            total_samples = format.total_samples,
            bitrate = context.bitrate(),
            "ATRAC3+ track initialized"
        );
        self.contexts.insert(id, context);
        Ok(id)
    }

    fn context(&self, id: u32) -> Option<&AtracContext> {
        self.contexts.get(&id)
    }

    fn context_mut(&mut self, id: u32) -> Result<&mut AtracContext> {
        self.contexts
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("invalid ATRAC id {id}"))
    }

    fn add_stream_data(&mut self, memory: &Memory, id: u32, bytes_to_add: u32) -> Result<u32> {
        let context = match self.contexts.get(&id) {
            Some(context) => context,
            None => return Ok(ATRAC_ERROR_BAD_ATRACID),
        };
        if bytes_to_add == 0 {
            return Ok(0);
        }
        let writable = context.writable_bytes();
        if bytes_to_add > writable {
            return Ok(if writable == 0 {
                ATRAC_ERROR_ALL_DATA_LOADED
            } else {
                ATRAC_ERROR_ADD_DATA_TOO_BIG
            });
        }

        let (buffer, buffer_size, write_offset) =
            (context.buffer, context.buffer_size, context.write_offset);
        let first_len = bytes_to_add.min(buffer_size - write_offset);
        let first = memory.read_bytes(
            buffer.wrapping_add(write_offset),
            usize::try_from(first_len).unwrap_or(usize::MAX),
        )?;
        let second_len = bytes_to_add - first_len;
        let second = if second_len == 0 {
            Vec::new()
        } else {
            memory.read_bytes(buffer, usize::try_from(second_len).unwrap_or(usize::MAX))?
        };

        let context = self.context_mut(id)?;
        context.submit_to_decoder(&first);
        context.submit_to_decoder(&second);
        context.buffered_bytes = context.buffered_bytes.saturating_add(bytes_to_add);
        context.next_file_offset = context.next_file_offset.saturating_add(bytes_to_add);
        context.write_offset = (write_offset + bytes_to_add) % buffer_size;
        Ok(0)
    }

    fn get_stream_data_info(
        &self,
        memory: &mut Memory,
        id: u32,
        write_ptr_addr: u32,
        writable_bytes_addr: u32,
        read_offset_addr: u32,
    ) -> Result<u32> {
        let Some(context) = self.context(id) else {
            return Ok(ATRAC_ERROR_BAD_ATRACID);
        };
        if write_ptr_addr != 0 {
            memory.write_u32(
                write_ptr_addr,
                context.buffer.wrapping_add(context.write_offset),
            )?;
        }
        if writable_bytes_addr != 0 {
            memory.write_u32(writable_bytes_addr, context.writable_bytes())?;
        }
        if read_offset_addr != 0 {
            memory.write_u32(read_offset_addr, context.next_file_offset)?;
        }
        Ok(0)
    }

    fn decode(
        &mut self,
        memory: &mut Memory,
        id: u32,
        output: u32,
        samples_addr: u32,
        finish_addr: u32,
        remain_addr: u32,
    ) -> Result<u32> {
        let context = match self.contexts.get_mut(&id) {
            Some(context) => context,
            None => return Ok(ATRAC_ERROR_BAD_ATRACID),
        };
        if output & 1 != 0 {
            return Ok(ATRAC_ERROR_BAD_ALIGNMENT);
        }
        if context.decoded_samples >= context.format.total_samples {
            if samples_addr != 0 {
                memory.write_u32(samples_addr, 0)?;
            }
            if finish_addr != 0 {
                memory.write_u32(finish_addr, 1)?;
            }
            if remain_addr != 0 {
                memory.write_u32(remain_addr, 0)?;
            }
            return Ok(ATRAC_ERROR_ALL_DATA_DECODED);
        }

        let wait = if context.decoded_samples == 0 {
            Duration::from_millis(50)
        } else {
            Duration::from_millis(2)
        };
        let pcm = context
            .decoder
            .as_ref()
            .and_then(|decoder| decoder.take(wait));
        let Some(pcm) = pcm.or_else(|| {
            context
                .decoder
                .is_none()
                .then(|| vec![0; ATRAC_OUTPUT_BYTES])
        }) else {
            return Ok(ATRAC_ERROR_NO_DATA);
        };
        let available_samples = u32::try_from(pcm.len() / 4).unwrap_or(u32::MAX);
        let samples = available_samples
            .min(ATRAC_SAMPLES_PER_FRAME)
            .min(context.format.total_samples - context.decoded_samples);
        if samples == 0 {
            return Ok(ATRAC_ERROR_NO_DATA);
        }
        let output_bytes = usize::try_from(samples).unwrap_or(usize::MAX) * 4;
        if output != 0 {
            memory.write_bytes(output, &pcm[..output_bytes])?;
        }
        context.decoded_samples = context.decoded_samples.saturating_add(samples);
        context.buffered_bytes = context
            .buffered_bytes
            .saturating_sub(context.format.block_bytes as u32);
        let finished = u32::from(context.decoded_samples >= context.format.total_samples);
        if samples_addr != 0 {
            memory.write_u32(samples_addr, samples)?;
        }
        if finish_addr != 0 {
            memory.write_u32(finish_addr, finished)?;
        }
        if remain_addr != 0 {
            memory.write_u32(remain_addr, context.remaining_frames())?;
        }
        Ok(0)
    }

    fn get_buffer_info_for_resetting(
        &self,
        memory: &mut Memory,
        id: u32,
        sample: u32,
        info_addr: u32,
    ) -> Result<u32> {
        let Some(context) = self.context(id) else {
            return Ok(ATRAC_ERROR_BAD_ATRACID);
        };
        let frame = sample / context.format.samples_per_frame;
        let file_offset = (context.format.data_offset as u32)
            .saturating_add(frame.saturating_mul(context.format.block_bytes as u32));
        let write_ptr = context.buffer.wrapping_add(context.write_offset);
        let writable = context.writable_bytes();
        let min_write = file_offset
            .saturating_sub(context.next_file_offset)
            .min(writable);
        write_atrac_reset_info(
            memory,
            info_addr,
            write_ptr,
            writable,
            min_write,
            file_offset,
        )?;
        Ok(0)
    }

    fn reset_play_position(&mut self, id: u32, sample: u32) -> u32 {
        let Ok(context) = self.context_mut(id) else {
            return ATRAC_ERROR_BAD_ATRACID;
        };
        context.decoded_samples = sample.min(context.format.total_samples);
        0
    }

    fn set_loop_num(&mut self, id: u32, loop_num: i32) -> u32 {
        let Ok(context) = self.context_mut(id) else {
            return ATRAC_ERROR_BAD_ATRACID;
        };
        context.loop_num = loop_num;
        0
    }

    fn release(&mut self, id: u32) -> u32 {
        if self.contexts.remove(&id).is_some() {
            0
        } else {
            ATRAC_ERROR_BAD_ATRACID
        }
    }
}

impl AtracContext {
    fn submit_to_decoder(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if self
            .decoder
            .as_ref()
            .is_some_and(|decoder| !decoder.submit(bytes))
        {
            self.decoder = None;
            tracing::warn!("ATRAC host decoder stopped while accepting stream data");
        }
    }

    fn writable_bytes(&self) -> u32 {
        let remaining = (self.format.data_offset as u32)
            .saturating_add(self.format.data_bytes as u32)
            .saturating_sub(self.next_file_offset);
        let free = self.buffer_size.saturating_sub(self.buffered_bytes);
        let contiguous = self.buffer_size.saturating_sub(self.write_offset);
        remaining.min(free).min(contiguous)
    }

    fn remaining_frames(&self) -> u32 {
        let loaded = self
            .next_file_offset
            .saturating_sub(self.format.data_offset as u32)
            .min(self.format.data_bytes as u32);
        let frames = loaded / self.format.block_bytes.max(1) as u32;
        frames.saturating_sub(self.decoded_samples / self.format.samples_per_frame)
    }

    fn bitrate(&self) -> u32 {
        let numerator = (self.format.data_bytes as u64)
            .saturating_mul(8)
            .saturating_mul(self.format.sample_rate as u64);
        u32::try_from(numerator / u64::from(self.format.total_samples.max(1))).unwrap_or(u32::MAX)
    }
}

fn write_atrac_reset_info(
    memory: &mut Memory,
    address: u32,
    write_ptr: u32,
    writable_bytes: u32,
    min_write_bytes: u32,
    file_offset: u32,
) -> Result<()> {
    memory.write_u32(address, write_ptr)?;
    memory.write_u32(address.wrapping_add(4), writable_bytes)?;
    memory.write_u32(address.wrapping_add(8), min_write_bytes)?;
    memory.write_u32(address.wrapping_add(12), file_offset)?;
    memory.write_bytes(address.wrapping_add(16), &[0; 16])?;
    Ok(())
}

#[derive(Default)]
struct MpegState {
    initialized: bool,
    next_stream_id: u32,
    frame_width: u32,
    pixel_format: u32,
    decoded_frames: u32,
    contexts: HashMap<u32, MpegContext>,
    media: MediaPipeline,
    last_video_frame: Option<Vec<u8>>,
    decoder_prefetched: bool,
}

struct MpegContext {
    streams: HashSet<u32>,
    es_buffers: [bool; 4],
}

impl MpegState {
    fn init(&mut self) {
        self.initialized = true;
        if self.next_stream_id == 0 {
            self.next_stream_id = 1;
        }
        self.frame_width = 0;
        self.pixel_format = 3;
        self.decoded_frames = 0;
        self.media = MediaPipeline::default();
        self.last_video_frame = None;
        self.decoder_prefetched = false;
    }

    fn configure(&mut self, frame_width: u32) {
        self.frame_width = frame_width;
    }

    fn set_pixel_format(&mut self, pixel_format: u32) {
        self.pixel_format = pixel_format;
    }

    fn next_decoded_frame(&mut self) -> u32 {
        let frame = self.decoded_frames;
        self.decoded_frames = self.decoded_frames.wrapping_add(1);
        frame
    }

    fn create(&mut self, handle: u32) {
        self.contexts.insert(
            handle,
            MpegContext {
                streams: HashSet::new(),
                es_buffers: [false; 4],
            },
        );
    }

    fn register_stream(&mut self, handle: u32) -> Option<u32> {
        if !self.initialized {
            return None;
        }
        let stream_id = self.next_stream_id.max(1);
        self.next_stream_id = stream_id.wrapping_add(1).max(1);
        self.contexts.get_mut(&handle)?.streams.insert(stream_id);
        Some(stream_id)
    }

    fn unregister_stream(&mut self, handle: u32, stream_id: u32) -> Option<()> {
        let context = self.contexts.get_mut(&handle)?;
        context.streams.remove(&stream_id);
        Some(())
    }

    fn malloc_es_buffer(&mut self, handle: u32) -> Option<u32> {
        let context = self.contexts.get_mut(&handle)?;
        context
            .es_buffers
            .iter_mut()
            .position(|allocated| {
                if *allocated {
                    false
                } else {
                    *allocated = true;
                    true
                }
            })
            .map(|index| index as u32 + 1)
    }

    fn es_buffer_allocated(&self, handle: u32, buffer: u32) -> Option<bool> {
        let index = buffer.checked_sub(1)? as usize;
        Some(*self.contexts.get(&handle)?.es_buffers.get(index)?)
    }

    fn au_parameters(&self, handle: u32, buffer: u32) -> Option<(u32, u32)> {
        if !self.contexts.contains_key(&handle) {
            return None;
        }
        let is_avc =
            (1..=4).contains(&buffer) && self.es_buffer_allocated(handle, buffer) == Some(true);
        Some(if is_avc {
            (MPEG_AVC_ES_SIZE, 0)
        } else {
            (MPEG_ATRAC_ES_SIZE, u32::MAX)
        })
    }

    fn free_es_buffer(&mut self, handle: u32, buffer: u32) -> Option<bool> {
        let context = self.contexts.get_mut(&handle)?;
        let Some(allocated) = buffer
            .checked_sub(1)
            .and_then(|index| context.es_buffers.get_mut(index as usize))
        else {
            return Some(false);
        };
        *allocated = false;
        Some(true)
    }

    fn delete(&mut self, handle: u32) -> bool {
        self.contexts.remove(&handle).is_some()
    }

    fn finish(&mut self) {
        self.initialized = false;
        self.contexts.clear();
        self.media = MediaPipeline::default();
        self.last_video_frame = None;
        self.decoder_prefetched = false;
    }
}

struct GuestDirectory {
    entries: Vec<psp_loader::IsoEntry>,
    index: usize,
}

enum GuestFileData {
    Bytes(Vec<u8>),
    DiscRange { offset: u64, length: u64 },
}

struct GuestFile {
    data: GuestFileData,
    position: usize,
    start_sector: u32,
}

struct AsyncFileOperation {
    result: i64,
    completed: bool,
    close_pending: bool,
}

impl GuestFile {
    fn bytes(bytes: Vec<u8>, start_sector: u32) -> Self {
        Self {
            data: GuestFileData::Bytes(bytes),
            position: 0,
            start_sector,
        }
    }

    fn disc_range(offset: u64, length: u64, start_sector: u32) -> Self {
        Self {
            data: GuestFileData::DiscRange { offset, length },
            position: 0,
            start_sector,
        }
    }

    fn len(&self) -> usize {
        match &self.data {
            GuestFileData::Bytes(bytes) => bytes.len(),
            GuestFileData::DiscRange { length, .. } => {
                usize::try_from(*length).unwrap_or(usize::MAX)
            }
        }
    }

    fn size(&self) -> u64 {
        self.len() as u64
    }

    fn read(
        &mut self,
        disc: Option<&mut psp_loader::IsoImage>,
        requested: usize,
    ) -> Result<Vec<u8>> {
        let count = requested.min(self.len().saturating_sub(self.position));
        let position = self.position;
        let bytes = match &self.data {
            GuestFileData::Bytes(bytes) => bytes[position..position + count].to_vec(),
            GuestFileData::DiscRange { offset, .. } => {
                let disc = disc.context("raw disc file is no longer mounted")?;
                let offset = offset
                    .checked_add(position as u64)
                    .context("raw disc file position overflow")?;
                disc.read_raw(offset, count)?
            }
        };
        self.position += bytes.len();
        Ok(bytes)
    }

    fn read_remaining(&mut self, disc: Option<&mut psp_loader::IsoImage>) -> Result<Vec<u8>> {
        self.read(disc, self.len().saturating_sub(self.position))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DiscMetadata {
    extent: u32,
    size: u64,
    directory: bool,
    access: u32,
}

fn psp_io_stat(metadata: DiscMetadata) -> [u8; SCE_IO_STAT_SIZE] {
    // keep the unused timestamp/private fields in the same sentinel state as
    // the psp iso driver. titles generally ignore them, but some resource
    // loaders use them to distinguish a real stat record from a zeroed one.
    let mut stat = [0xfe; SCE_IO_STAT_SIZE];
    let mode = (if metadata.directory {
        SCE_STM_FDIR
    } else {
        SCE_STM_FREG
    }) | metadata.access;
    stat[0..4].copy_from_slice(&mode.to_le_bytes());
    stat[4..8].copy_from_slice(&(if metadata.directory { 0x10u32 } else { 0x20u32 }).to_le_bytes());
    stat[8..16].copy_from_slice(&metadata.size.to_le_bytes());
    stat[64..68].copy_from_slice(&metadata.extent.to_le_bytes());
    stat
}

struct GuestThread {
    entry: u32,
    priority: u32,
    stack_size: u32,
    stack_base: u32,
    stack_top: u32,
    attributes: u32,
    cpu: Option<psp_cpu::Cpu>,
}

struct FixedPool {
    address: u32,
    block_size: u32,
    free: Vec<bool>,
}

#[derive(Clone, Debug)]
struct PartitionBlock {
    size: u32,
    owner: u32,
    name: String,
    kind: &'static str,
}

#[derive(Clone, Copy)]
struct AllocationRequest {
    partition: u32,
    allocation_type: u32,
    attributes: u32,
    size: u32,
    alignment: u32,
}

struct PartitionAllocator {
    base: u32,
    end: u32,
    blocks: std::collections::BTreeMap<u32, PartitionBlock>,
}

impl HleState {
    fn new(
        disc: Option<psp_loader::IsoImage>,
        partition_base: u32,
        window: Option<Box<dyn HostDisplay>>,
    ) -> Result<Self> {
        #[cfg(not(feature = "desktop"))]
        let _ = window;
        let mut user_partition = PartitionAllocator::new(partition_base, USER_RAM_END);
        let main_stack = user_partition
            .allocate(1, "main-thread", "stack", 0x0010_0000, 256, true)
            .context("cannot allocate main thread stack")?;
        anyhow::ensure!(
            main_stack + 0x0010_0000 == USER_RAM_END,
            "main stack was not placed at top of user partition"
        );
        // the block jit is the default execution engine; `PSP_JIT=off`
        // keeps the single-step interpreter for debugging and differential
        // validation.
        #[cfg(feature = "desktop")]
        let jit_enabled = !std::env::var_os("PSP_JIT").is_some_and(|value| {
            let value = value.to_string_lossy();
            value.eq_ignore_ascii_case("off") || value.eq_ignore_ascii_case("false")
        });
        Ok(Self {
            next_uid: 0,
            threads: HashMap::new(),
            ready: VecDeque::new(),
            suspended: HashSet::new(),
            current_uid: 0,
            disc,
            next_fd: 3,
            directories: HashMap::new(),
            files: HashMap::new(),
            async_operations: HashMap::new(),
            modules: HashSet::new(),
            user_partition,
            memory_blocks: HashMap::new(),
            fixed_pools: HashMap::new(),
            semaphores: HashMap::new(),
            event_flags: HashMap::new(),
            mpeg: MpegState::default(),
            atrac: AtracState::default(),
            audio: AudioState::default(),
            sas: psp_audio::SasMixer::default(),
            ctrl_mode: 0,
            ctrl_cycle: 0,
            ctrl_last_sample: 0,
            ctrl_waiters: VecDeque::new(),
            ctrl_idle_reset: -1,
            ctrl_idle_back: -1,
            umd_activated: true,
            umd_callback: None,
            ge_lists: HashMap::new(),
            ge_callbacks: HashMap::new(),
            display: DisplayState::default(),
            callbacks: HashMap::new(),
            unknown_imports: HashSet::new(),
            volatile_locked: false,
            volatile_waiters: VecDeque::new(),
            subinterrupts: HashMap::new(),
            interrupt_stack: Vec::new(),
            pending_interrupts: VecDeque::new(),
            pending_ringbuffer_put: None,
            timed_waiters: HashMap::new(),
            timed_wait_results: HashMap::new(),
            #[cfg(feature = "desktop")]
            pending_render_commands: Vec::new(),
            #[cfg(feature = "desktop")]
            window,
            #[cfg(feature = "desktop")]
            last_host_present: None,
            #[cfg(feature = "desktop")]
            next_host_vblank: None,
            #[cfg(feature = "desktop")]
            last_host_buttons: 0,
            #[cfg(feature = "desktop")]
            host_started_at: Instant::now(),
            #[cfg(feature = "desktop")]
            hardware_frame_valid: false,
            vblank_wait_pending: false,
            jit: psp_cpu::Jit::default(),
            #[cfg(feature = "desktop")]
            jit_enabled,
        })
    }

    fn sample_controller_if_due(&mut self, emulator: &mut psp_core::Emulator) -> Result<()> {
        if self.ctrl_cycle == 0 {
            return Ok(());
        }
        let now = emulator.scheduler.now();
        if now.saturating_sub(self.ctrl_last_sample) >= u64::from(self.ctrl_cycle) {
            self.sample_controller(emulator, now)?;
        }
        Ok(())
    }

    fn sample_controller(&mut self, emulator: &mut psp_core::Emulator, frame: u64) -> Result<()> {
        emulator.input.sample(frame);
        if emulator.input.buttons != 0 {
            tracing::debug!(
                frame,
                buttons = format_args!("0x{:08x}", emulator.input.buttons),
                "controller sample captured"
            );
        }
        self.ctrl_last_sample = frame;

        while let Some(waiter) = self.ctrl_waiters.pop_front() {
            if !self.threads.contains_key(&waiter.thread) {
                continue;
            }
            let Some(sample) = emulator
                .input
                .read_samples(1, waiter.negative, false)
                .into_iter()
                .next()
            else {
                break;
            };
            if sample.buttons != 0 {
                tracing::debug!(
                    thread = waiter.thread,
                    buttons = format_args!("0x{:08x}", sample.buttons),
                    frame = sample.frame,
                    data = format_args!("0x{:08x}", waiter.data),
                    negative = waiter.negative,
                    "guest waiter received pressed controller sample"
                );
            }
            write_controller_sample(&mut emulator.memory, waiter.data, sample)?;
            self.wake_thread(waiter.thread, 1);
            break;
        }
        Ok(())
    }

    fn attach_main_thread(&mut self, cpu: &psp_cpu::Cpu) {
        self.next_uid += 1;
        self.current_uid = self.next_uid;
        self.threads.insert(
            self.current_uid,
            GuestThread {
                entry: cpu.pc,
                priority: 32,
                stack_size: 0x0010_0000,
                stack_base: USER_RAM_END - 0x0010_0000,
                stack_top: USER_RAM_END,
                attributes: 0,
                cpu: None,
            },
        );
    }

    fn disc_metadata(&mut self, path: &str) -> Option<DiscMetadata> {
        let normalized = disc_path(path);
        if let Some((sector, size)) = parse_lbn_path(normalized) {
            let extent = u32::try_from(sector).ok()?;
            let size = size as u64;
            self.disc
                .as_ref()?
                .validate_raw_range(sector.checked_mul(ISO_SECTOR_SIZE)?, size)
                .ok()?;
            return Some(DiscMetadata {
                extent,
                size,
                directory: false,
                access: SCE_STM_READ_ONLY,
            });
        }

        let entry = self.disc.as_mut()?.find(normalized).ok()?;
        Some(DiscMetadata {
            extent: entry.extent,
            size: u64::from(entry.size),
            directory: entry.directory,
            access: SCE_STM_ISO_ACCESS,
        })
    }

    fn open_disc_file(&mut self, path: &str) -> Option<GuestFile> {
        if !path.split_once(':').is_some_and(|(device, _)| {
            device.eq_ignore_ascii_case("disc0") || device.eq_ignore_ascii_case("umd0")
        }) {
            return None;
        }

        self.disc.as_mut().and_then(|disc| {
            let normalized = disc_path(path);
            if let Some((sector, size)) = parse_lbn_path(normalized) {
                let offset = sector.checked_mul(ISO_SECTOR_SIZE)?;
                disc.validate_raw_range(offset, size as u64).ok()?;
                Some(GuestFile::disc_range(
                    offset,
                    size as u64,
                    u32::try_from(sector).ok()?,
                ))
            } else {
                let entry = disc.find(normalized).ok()?;
                if entry.directory {
                    None
                } else {
                    let bytes = disc.read_file(normalized, 64 * 1024 * 1024).ok()?;
                    Some(GuestFile::bytes(bytes, entry.extent))
                }
            }
        })
    }

    fn insert_file(&mut self, file: GuestFile) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        self.files.insert(fd, file);
        fd
    }

    fn start_async_read(
        &mut self,
        emulator: &mut psp_core::Emulator,
        fd: u32,
        output: u32,
        requested: usize,
    ) -> Result<u32> {
        if !self.files.contains_key(&fd) {
            return Ok(IO_ERROR_ASYNC_BAD_FD);
        }
        if self.async_operations.contains_key(&fd) {
            return Ok(IO_ERROR_ASYNC_BUSY);
        }

        let file = self
            .files
            .get_mut(&fd)
            .expect("file existence was checked above");
        let result = match file.read(self.disc.as_mut(), requested) {
            Ok(bytes) => {
                let count = bytes.len() as i64;
                emulator.memory.write_bytes(output, &bytes)?;
                tracing::debug!(
                    fd,
                    requested,
                    returned = count,
                    destination = format_args!("0x{output:08x}"),
                    thread = self.current_uid,
                    "async disc bytes delivered to guest"
                );
                count
            }
            Err(error) => {
                tracing::warn!(fd, ?error, "async disc read failed");
                signed_io_result(IO_ERROR_IO)
            }
        };
        self.async_operations.insert(
            fd,
            AsyncFileOperation {
                result,
                // host-backed disc reads complete without a worker thread,
                // but the result is deliberately retained until poll/wait.
                // this preserves the psp async abi instead of collapsing the
                // call into a synchronous return value.
                completed: true,
                close_pending: false,
            },
        );
        Ok(0)
    }

    fn consume_async_result(
        &mut self,
        emulator: &mut psp_core::Emulator,
        fd: u32,
        output: u32,
    ) -> Result<u32> {
        let Some(operation) = self.async_operations.get(&fd) else {
            return Ok(IO_ERROR_NO_ASYNC);
        };
        if !operation.completed {
            return Ok(1);
        }

        let result = operation.result;
        let close_pending = operation.close_pending;
        write_async_result(&mut emulator.memory, output, result)?;
        self.async_operations.remove(&fd);
        if close_pending {
            self.files.remove(&fd);
        }
        Ok(0)
    }

    fn dispatch(
        &mut self,
        import: &psp_loader::Import,
        emulator: &mut psp_core::Emulator,
    ) -> Result<()> {
        match (import.library.as_str(), import.nid) {
            ("ThreadManForUser", 0x446d_8de6) => {
                let stack_size = emulator.cpu.gpr[7].max(0x4000).next_multiple_of(256);
                self.next_uid += 1;
                let stack_base = match self.user_partition.allocate(
                    self.next_uid,
                    "guest-thread",
                    "stack",
                    stack_size,
                    256,
                    true,
                ) {
                    Some(address) => address,
                    None => {
                        self.log_allocation_failure(
                            import,
                            emulator,
                            AllocationRequest {
                                partition: 2,
                                allocation_type: 1,
                                attributes: 0,
                                size: stack_size,
                                alignment: 256,
                            },
                        );
                        emulator.cpu.gpr[2] = 0x8002_0190;
                        return Ok(());
                    }
                };
                self.threads.insert(
                    self.next_uid,
                    GuestThread {
                        entry: emulator.cpu.gpr[5],
                        priority: emulator.cpu.gpr[6],
                        stack_size,
                        stack_base,
                        stack_top: stack_base + stack_size,
                        attributes: 0,
                        cpu: None,
                    },
                );
                tracing::debug!(
                    uid = self.next_uid,
                    name = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])
                        .unwrap_or_default(),
                    entry = format_args!("0x{:08x}", emulator.cpu.gpr[5]),
                    priority = emulator.cpu.gpr[6],
                    thread = self.current_uid,
                    "CreateThread"
                );
                emulator.cpu.gpr[2] = self.next_uid;
            }
            ("ThreadManForUser", 0x293b_45b8) => {
                emulator.cpu.gpr[2] = self.current_uid;
            }
            ("ThreadManForUser", 0xea74_8e31) => {
                let clear = emulator.cpu.gpr[4];
                let set = emulator.cpu.gpr[5];
                if clear & !0x4000 != 0 || set & !0x4000 != 0 {
                    emulator.cpu.gpr[2] = KERNEL_ERROR_ILLEGAL_ATTR;
                } else if let Some(thread) = self.threads.get_mut(&self.current_uid) {
                    thread.attributes = (thread.attributes & !clear) | set;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = 0x8002_01ff;
                }
            }
            ("ThreadManForUser", 0x809c_e29b | 0xaa73_c935) => {
                tracing::debug!(
                    uid = self.current_uid,
                    nid = format_args!("0x{:08x}", import.nid),
                    "guest requested thread exit"
                );
                emulator.cpu.pc = 0;
            }
            ("ThreadManForUser", 0x383f_7bcc) => {
                let requested_uid = emulator.cpu.gpr[4];
                let uid = if requested_uid == 0 {
                    self.current_uid
                } else {
                    requested_uid
                };
                if self.remove_thread(uid) {
                    emulator.cpu.gpr[2] = 0;
                    if uid == self.current_uid {
                        // the terminated thread does not resume at its
                        // return address. let the main loop perform the
                        // ordinary context switch.
                        emulator.cpu.pc = 0;
                    }
                } else {
                    emulator.cpu.gpr[2] = 0x8002_00d6;
                }
            }
            ("ThreadManForUser", 0x349d_6d6c) => {
                // scekernelcheckcallback is a polling point.  callback
                // delivery is modeled at scheduler/vblank boundaries, so a
                // direct poll has no work to report here.
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0xe81c_af8f) => {
                self.next_uid += 1;
                self.callbacks
                    .insert(self.next_uid, (emulator.cpu.gpr[5], emulator.cpu.gpr[6]));
                emulator.cpu.gpr[2] = self.next_uid;
            }
            ("ThreadManForUser", 0x369e_d59d) => {
                emulator.cpu.gpr[2] = emulator.scheduler.now() as u32;
            }
            ("ThreadManForUser", 0x82bc_5777) => {
                let clock = emulator.scheduler.now();
                emulator.cpu.gpr[2] = clock as u32;
                emulator.cpu.gpr[3] = (clock >> 32) as u32;
            }
            ("ThreadManForUser", 0xe161_9d7c) => {
                let clock = u64::from(emulator.cpu.gpr[4]) | (u64::from(emulator.cpu.gpr[5]) << 32);
                let (seconds, microseconds) = split_system_clock(clock);
                if emulator.cpu.gpr[6] != 0 {
                    emulator.memory.write_u32(emulator.cpu.gpr[6], seconds)?;
                }
                if emulator.cpu.gpr[7] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[7], microseconds)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0xdb73_8f35) => {
                let now = emulator.scheduler.now();
                emulator.memory.write_u32(emulator.cpu.gpr[4], now as u32)?;
                emulator
                    .memory
                    .write_u32(emulator.cpu.gpr[4].wrapping_add(4), (now >> 32) as u32)?;
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0xba6b_92e2) => {
                let clock = u64::from(emulator.memory.read_u32(emulator.cpu.gpr[4])?)
                    | (u64::from(
                        emulator
                            .memory
                            .read_u32(emulator.cpu.gpr[4].wrapping_add(4))?,
                    ) << 32);
                if emulator.cpu.gpr[5] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[5], (clock % 1_000_000) as u32)?;
                }
                if emulator.cpu.gpr[6] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[6], (clock / 1_000_000) as u32)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0xcead_eb47 | 0x68da_9e36) => {
                let delay = u64::from(emulator.cpu.gpr[4]);
                // trace: delay calls are extremely hot (pacing loops).
                tracing::trace!(delay, thread = self.current_uid, "DelayThread");
                emulator.cpu.gpr[2] = 0;
                self.wait_current_for(emulator, delay)?;
            }
            ("ThreadManForUser", 0xbd12_3d9e | 0x1181_e963) => {
                // delaysysclockthread takes a pointer to a
                // microsecond sysclock and delays by its value. missing this
                // (returning success without waiting) spins streaming loops
                // impossibly fast and starves lower-priority pump threads.
                let pointer = emulator.cpu.gpr[4];
                let delay = if pointer == 0 {
                    0
                } else {
                    let low = emulator.memory.read_u32(pointer).unwrap_or(0);
                    let high = emulator
                        .memory
                        .read_u32(pointer.wrapping_add(4))
                        .unwrap_or(0);
                    (u64::from(high) << 32) | u64::from(low)
                };
                tracing::trace!(delay, thread = self.current_uid, "DelaySysClockThread");
                emulator.cpu.gpr[2] = 0;
                self.wait_current_for(emulator, delay)?;
            }
            ("ThreadManForUser", 0x110d_ec9a) => {
                // scekernelusec2sysclock: write the 32-bit
                // microsecond value as a 64-bit sysclock. the fallback used
                // to return success without writing, leaving timeout clocks
                // as stack garbage (far-future deadlines that never expire
                // instead of short retries that re-issue umd pump reads).
                let usec = emulator.cpu.gpr[4];
                let clock_ptr = emulator.cpu.gpr[5];
                if clock_ptr != 0 {
                    emulator.memory.write_u32(clock_ptr, usec)?;
                    emulator.memory.write_u32(clock_ptr.wrapping_add(4), 0)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0x8282_6f70) => {
                self.block_current(emulator)?;
            }
            ("ThreadManForUser", 0x9944_f31f) => {
                let requested_uid = emulator.cpu.gpr[4];
                let uid = if requested_uid == 0 {
                    self.current_uid
                } else {
                    requested_uid
                };
                if !self.threads.contains_key(&uid) {
                    emulator.cpu.gpr[2] = 0x8002_00d6;
                } else if self.suspended.insert(uid) {
                    self.ready.retain(|thread| *thread != uid);
                    self.timed_waiters.remove(&uid);
                    self.timed_wait_results.remove(&uid);
                    if uid == self.current_uid {
                        if let Some(thread) = self.threads.get_mut(&uid) {
                            let mut cpu = emulator.cpu.clone();
                            cpu.gpr[2] = 0;
                            thread.cpu = Some(cpu);
                        }
                        self.current_uid = 0;
                        anyhow::ensure!(
                            self.switch_next_or_advance(emulator),
                            "guest deadlock: no runnable thread after suspend"
                        );
                    } else {
                        emulator.cpu.gpr[2] = 0;
                    }
                } else {
                    emulator.cpu.gpr[2] = 0x8002_00d5;
                }
            }
            ("ThreadManForUser", 0x7515_6e8f) => {
                let uid = emulator.cpu.gpr[4];
                if self.suspended.remove(&uid) {
                    if self
                        .threads
                        .get(&uid)
                        .is_some_and(|thread| thread.cpu.is_some())
                        && !self.ready.contains(&uid)
                    {
                        self.ready.push_back(uid);
                    }
                    emulator.cpu.gpr[2] = 0;
                    self.preempt_if_needed(emulator);
                } else {
                    emulator.cpu.gpr[2] = 0x8002_00d6;
                }
            }
            ("ThreadManForUser", 0xd59e_ad2f) => {
                self.wake_thread(emulator.cpu.gpr[4], 0);
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0xf475_845d) => {
                let uid = emulator.cpu.gpr[4];
                let argument_size = emulator.cpu.gpr[5];
                let argument = emulator.cpu.gpr[6];
                let thread = self
                    .threads
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelStartThread: unknown thread {uid}"))?;
                anyhow::ensure!(
                    argument_size <= thread.stack_size,
                    "sceKernelStartThread: {argument_size} argument bytes exceed thread {uid} stack"
                );
                let argument_bytes = (0..argument_size)
                    .map(|offset| emulator.memory.read_u8(argument.wrapping_add(offset)))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut cpu = psp_cpu::Cpu::new(thread.entry);
                cpu.gpr[28] = emulator.cpu.gpr[28];
                cpu.gpr[26] = emulator.cpu.gpr[26];
                let argument_address = thread
                    .stack_top
                    .saturating_sub(argument_size.next_multiple_of(16));
                anyhow::ensure!(
                    argument_address >= thread.stack_base,
                    "sceKernelStartThread arguments underflow thread {uid} stack"
                );
                emulator
                    .memory
                    .write_bytes(argument_address, &argument_bytes)?;
                cpu.gpr[29] = argument_address;
                cpu.gpr[4] = argument_size;
                cpu.gpr[5] = argument_address;
                thread.cpu = Some(cpu);
                self.ready.push_back(uid);
                emulator.cpu.gpr[2] = 0;
                tracing::info!(
                    uid,
                    entry = format_args!("0x{:08x}", thread.entry),
                    priority = thread.priority,
                    stack_size = thread.stack_size,
                    "guest thread ready"
                );
                self.preempt_if_needed(emulator);
            }
            ("ThreadManForUser", 0xd6da_4ba1) => {
                let initial = emulator.cpu.gpr[6];
                let maximum = emulator.cpu.gpr[7];
                if initial > maximum || maximum == 0 {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                } else {
                    self.next_uid += 1;
                    tracing::debug!(
                        uid = self.next_uid,
                        name = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])
                            .unwrap_or_default(),
                        initial,
                        maximum,
                        thread = self.current_uid,
                        "CreateSema"
                    );
                    self.semaphores.insert(
                        self.next_uid,
                        Semaphore {
                            count: initial,
                            max_count: maximum,
                            waiters: VecDeque::new(),
                        },
                    );
                    emulator.cpu.gpr[2] = self.next_uid;
                }
            }
            ("ThreadManForUser", 0x28b6_489c) => {
                let uid = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if self.semaphores.remove(&uid).is_some() {
                    0
                } else {
                    0x8002_00cb
                };
            }
            ("ThreadManForUser", 0x3f53_e640) => {
                let uid = emulator.cpu.gpr[4];
                let amount = emulator.cpu.gpr[5];
                // trace: signal calls are extremely hot (mpeg pacing loops).
                tracing::trace!(uid, amount, thread = self.current_uid, "SignalSema");
                let semaphore = self
                    .semaphores
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelSignalSema: bad semaphore {uid}"))?;
                if amount == 0 || semaphore.count.saturating_add(amount) > semaphore.max_count {
                    emulator.cpu.gpr[2] = 0x8002_01ae;
                } else {
                    semaphore.count += amount;
                    let mut awakened = Vec::new();
                    while let Some(waiter) = semaphore.waiters.front() {
                        if semaphore.count < waiter.count {
                            break;
                        }
                        let waiter = semaphore.waiters.pop_front().unwrap();
                        semaphore.count -= waiter.count;
                        awakened.push(waiter.thread);
                    }
                    for thread in awakened {
                        self.wake_thread(thread, 0);
                    }
                    emulator.cpu.gpr[2] = 0;
                    self.preempt_if_needed(emulator);
                }
            }
            ("ThreadManForUser", 0x4e3a_1105 | 0x6d21_2bac) => {
                let uid = emulator.cpu.gpr[4];
                let amount = emulator.cpu.gpr[5];
                tracing::debug!(uid, amount, thread = self.current_uid, "WaitSema");
                let semaphore = self
                    .semaphores
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelWaitSema: bad semaphore {uid}"))?;
                if amount == 0 || amount > semaphore.max_count {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                } else if semaphore.count >= amount {
                    semaphore.count -= amount;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    // three-arg waits take the timeout pointer in a2, not on
                    // the stack (mips o32 passes the first four arguments in
                    // registers); reading sp+16 here consumed stack garbage.
                    let timeout_us = read_timeout_us_at(&emulator.memory, emulator.cpu.gpr[6])?;
                    semaphore.waiters.push_back(SemaphoreWaiter {
                        thread: self.current_uid,
                        count: amount,
                    });
                    let what = format!("sema{uid}x{amount}");
                    self.block_current_with_timeout(emulator, timeout_us, &what)?;
                }
            }
            ("ThreadManForUser", 0x58b1_f937) => {
                let uid = emulator.cpu.gpr[4];
                let amount = emulator.cpu.gpr[5];
                let semaphore = self
                    .semaphores
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelPollSema: bad semaphore {uid}"))?;
                if amount != 0 && semaphore.count >= amount {
                    semaphore.count -= amount;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = 0x8002_01ad;
                }
            }
            ("ThreadManForUser", 0x55c2_0a00) => {
                self.next_uid += 1;
                let name =
                    read_guest_string(&emulator.memory, emulator.cpu.gpr[4]).unwrap_or_default();
                self.event_flags.insert(
                    self.next_uid,
                    EventFlag {
                        bits: emulator.cpu.gpr[6],
                        waiters: VecDeque::new(),
                    },
                );
                tracing::debug!(
                    uid = self.next_uid,
                    name,
                    attr = format_args!("0x{:08x}", emulator.cpu.gpr[5]),
                    bits = format_args!("0x{:08x}", emulator.cpu.gpr[6]),
                    thread = self.current_uid,
                    "CreateEventFlag"
                );
                emulator.cpu.gpr[2] = self.next_uid;
            }
            ("ThreadManForUser", 0xef9e_4c70) => {
                let uid = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if self.event_flags.remove(&uid).is_some() {
                    0
                } else {
                    0x8002_00cb
                };
            }
            ("ThreadManForUser", 0x1fb1_5a32) => {
                let uid = emulator.cpu.gpr[4];
                let set = emulator.cpu.gpr[5];
                let flag = self
                    .event_flags
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelSetEventFlag: bad event flag {uid}"))?;
                // prev distinguishes coalesced sets (bit already set, no
                // new wakeup) from fresh ones that must produce work.
                let prev = flag.bits;
                tracing::debug!(
                    instructions = emulator.cpu.instruction_count,
                    uid,
                    set = format_args!("0x{set:08x}"),
                    prev = format_args!("0x{prev:08x}"),
                    thread = self.current_uid,
                    pc = format_args!("0x{:08x}", emulator.cpu.pc),
                    ra = format_args!("0x{:08x}", emulator.cpu.gpr[31]),
                    "SetEventFlag"
                );
                flag.bits |= set;
                let mut awakened = Vec::new();
                let mut remaining = VecDeque::new();
                while let Some(waiter) = flag.waiters.pop_front() {
                    if event_flag_matches(flag.bits, waiter.bits, waiter.mode) {
                        let matched = flag.bits;
                        apply_event_flag_clear(&mut flag.bits, waiter.bits, waiter.mode);
                        awakened.push((waiter.thread, waiter.out_bits, matched));
                    } else {
                        remaining.push_back(waiter);
                    }
                }
                flag.waiters = remaining;
                for (thread, out_bits, matched) in awakened {
                    if out_bits != 0 {
                        emulator.memory.write_u32(out_bits, matched)?;
                    }
                    self.wake_thread(thread, 0);
                }
                emulator.cpu.gpr[2] = 0;
                self.preempt_if_needed(emulator);
            }
            ("ThreadManForUser", 0x8123_46e4) => {
                let uid = emulator.cpu.gpr[4];
                let keep = emulator.cpu.gpr[5];
                let flag = self
                    .event_flags
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelClearEventFlag: bad event flag {uid}"))?;
                let prev = flag.bits;
                tracing::debug!(
                    uid,
                    keep = format_args!("0x{keep:08x}"),
                    prev = format_args!("0x{prev:08x}"),
                    thread = self.current_uid,
                    pc = format_args!("0x{:08x}", emulator.cpu.pc),
                    "ClearEventFlag"
                );
                flag.bits &= keep;
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0x402f_cf22 | 0x328c_546a) => {
                let uid = emulator.cpu.gpr[4];
                let wanted = emulator.cpu.gpr[5];
                let mode = emulator.cpu.gpr[6];
                let out_bits = emulator.cpu.gpr[7];
                tracing::debug!(
                    uid,
                    wanted = format_args!("0x{wanted:08x}"),
                    mode = format_args!("0x{mode:08x}"),
                    thread = self.current_uid,
                    "WaitEventFlag"
                );
                let flag = self
                    .event_flags
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelWaitEventFlag: bad event flag {uid}"))?;
                if wanted == 0 {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                } else if event_flag_matches(flag.bits, wanted, mode) {
                    let matched = flag.bits;
                    apply_event_flag_clear(&mut flag.bits, wanted, mode);
                    if out_bits != 0 {
                        emulator.memory.write_u32(out_bits, matched)?;
                    }
                    emulator.cpu.gpr[2] = 0;
                } else {
                    // the fifth argument arrives
                    // in t0 (gpr[8]), not on the stack. the o32 stack slot at
                    // sp+16 holds caller garbage here; reading it produced
                    // spurious timeouts that woke streaming waits early.
                    let timeout_us = read_timeout_us_at(&emulator.memory, emulator.cpu.gpr[8])?;
                    flag.waiters.push_back(EventFlagWaiter {
                        thread: self.current_uid,
                        bits: wanted,
                        mode,
                        out_bits,
                    });
                    let what = format!("flag{uid}:{wanted:#x}");
                    self.block_current_with_timeout(emulator, timeout_us, &what)?;
                }
            }
            ("ThreadManForUser", 0x30fd_48f0) => {
                let uid = emulator.cpu.gpr[4];
                let wanted = emulator.cpu.gpr[5];
                let mode = emulator.cpu.gpr[6];
                let out_bits = emulator.cpu.gpr[7];
                let flag = self
                    .event_flags
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelPollEventFlag: bad event flag {uid}"))?;
                let current = flag.bits;
                if event_flag_matches(current, wanted, mode) {
                    apply_event_flag_clear(&mut flag.bits, wanted, mode);
                    if out_bits != 0 {
                        emulator.memory.write_u32(out_bits, current)?;
                    }
                    emulator.cpu.gpr[2] = 0;
                } else {
                    if out_bits != 0 {
                        emulator.memory.write_u32(out_bits, current)?;
                    }
                    emulator.cpu.gpr[2] = 0x8002_01af;
                }
                // polls are hot; enable psp_rs::sync=trace when investigating
                // them. include both the observed bits and any clear effect.
                tracing::trace!(
                    target: "psp_rs::sync",
                    uid,
                    wanted = format_args!("0x{wanted:08x}"),
                    mode = format_args!("0x{mode:08x}"),
                    current = format_args!("0x{current:08x}"),
                    remaining = format_args!("0x{:08x}", flag.bits),
                    result = format_args!("0x{:08x}", emulator.cpu.gpr[2]),
                    thread = self.current_uid,
                    pc = format_args!("0x{:08x}", emulator.cpu.pc),
                    "PollEventFlag"
                );
            }
            ("ThreadManForUser", 0x278c_0df5 | 0x840e_8133 | 0x9ace_131e) => {
                emulator.cpu.gpr[2] = 0;
                self.yield_current(emulator);
            }
            ("ThreadManForUser", 0x71bc_9871) => {
                let uid = if emulator.cpu.gpr[4] == 0 {
                    self.current_uid
                } else {
                    emulator.cpu.gpr[4]
                };
                let priority = if emulator.cpu.gpr[5] == 0 {
                    self.threads
                        .get(&self.current_uid)
                        .map_or(32, |thread| thread.priority)
                } else {
                    emulator.cpu.gpr[5]
                };
                if !(8..=119).contains(&priority) {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                } else if let Some(thread) = self.threads.get_mut(&uid) {
                    thread.priority = priority;
                    emulator.cpu.gpr[2] = 0;
                    self.preempt_if_needed(emulator);
                } else {
                    emulator.cpu.gpr[2] = 0x8002_00d6;
                }
            }
            ("IoFileMgrForUser", 0xace9_46e8) => {
                let path = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])?;
                let output = emulator.cpu.gpr[5];
                let is_disc_path = path.split_once(':').is_some_and(|(device, _)| {
                    device.eq_ignore_ascii_case("disc0") || device.eq_ignore_ascii_case("umd0")
                });
                if output == 0 {
                    emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                } else if is_disc_path {
                    if let Some(metadata) = self.disc_metadata(&path) {
                        emulator
                            .memory
                            .write_bytes(output, &psp_io_stat(metadata))?;
                        emulator.cpu.gpr[2] = 0;
                    } else {
                        emulator.cpu.gpr[2] = IO_ERROR_FILE_NOT_FOUND;
                    }
                } else {
                    emulator.cpu.gpr[2] = IO_ERROR_FILE_NOT_FOUND;
                }
            }
            ("IoFileMgrForUser", 0xb29d_df9c) => {
                let path = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])?;
                if path.split_once(':').is_some_and(|(device, _)| {
                    device.eq_ignore_ascii_case("disc0") || device.eq_ignore_ascii_case("umd0")
                }) {
                    let normalized = disc_path(&path);
                    if let Some(entries) = self
                        .disc
                        .as_mut()
                        .and_then(|disc| disc.entries(normalized).ok())
                    {
                        tracing::debug!(
                            path = normalized,
                            entries = entries.len(),
                            "disc directory opened"
                        );
                        let fd = self.next_fd;
                        self.next_fd += 1;
                        self.directories
                            .insert(fd, GuestDirectory { entries, index: 0 });
                        emulator.cpu.gpr[2] = fd;
                    } else {
                        emulator.cpu.gpr[2] = IO_ERROR_FILE_NOT_FOUND;
                    }
                } else {
                    // no memory stick is mounted yet.  firmware reports the
                    // missing path to the guest; games then use their normal
                    // first-run/no-save-data flow.
                    emulator.cpu.gpr[2] = 0x8001_0002;
                }
            }
            ("sceAtrac3plus", 0x0fae_370e) => {
                emulator.cpu.gpr[2] = self.atrac.set_halfway_buffer_and_get_id(
                    &emulator.memory,
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                    emulator.cpu.gpr[6],
                )?;
            }
            ("sceAtrac3plus", 0x2dd3_e298) => {
                emulator.cpu.gpr[2] = self.atrac.get_buffer_info_for_resetting(
                    &mut emulator.memory,
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                    emulator.cpu.gpr[6],
                )?;
            }
            ("sceAtrac3plus", 0x5d26_8707) => {
                emulator.cpu.gpr[2] = self.atrac.get_stream_data_info(
                    &mut emulator.memory,
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                    emulator.cpu.gpr[6],
                    emulator.cpu.gpr[7],
                )?;
            }
            ("sceAtrac3plus", 0x61eb_33f5) => {
                emulator.cpu.gpr[2] = self.atrac.release(emulator.cpu.gpr[4]);
            }
            ("sceAtrac3plus", 0x644e_5607) => {
                emulator.cpu.gpr[2] = self
                    .atrac
                    .reset_play_position(emulator.cpu.gpr[4], emulator.cpu.gpr[5]);
            }
            ("sceAtrac3plus", 0x6a8c_3cd5) => {
                // psp import stubs expose the fifth argument in t0, which is
                // gpr[8] at the hle boundary: remainframesaddr.
                emulator.cpu.gpr[2] = self.atrac.decode(
                    &mut emulator.memory,
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                    emulator.cpu.gpr[6],
                    emulator.cpu.gpr[7],
                    emulator.cpu.gpr[8],
                )?;
            }
            ("sceAtrac3plus", 0x7db3_1251) => {
                emulator.cpu.gpr[2] = self.atrac.add_stream_data(
                    &emulator.memory,
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                )?;
            }
            ("sceAtrac3plus", 0x8681_20b5) => {
                emulator.cpu.gpr[2] = self
                    .atrac
                    .set_loop_num(emulator.cpu.gpr[4], emulator.cpu.gpr[5] as i32);
            }
            ("sceAtrac3plus", 0x9ae8_49a7) => {
                let id = emulator.cpu.gpr[4];
                let remain_addr = emulator.cpu.gpr[5];
                emulator.cpu.gpr[2] = if let Some(context) = self.atrac.context(id) {
                    if remain_addr != 0 {
                        emulator
                            .memory
                            .write_u32(remain_addr, context.remaining_frames())?;
                    }
                    0
                } else {
                    ATRAC_ERROR_BAD_ATRACID
                };
            }
            ("sceAtrac3plus", 0xa2bb_a8be) => {
                let id = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if let Some(context) = self.atrac.context(id) {
                    if emulator.cpu.gpr[5] != 0 {
                        emulator
                            .memory
                            .write_u32(emulator.cpu.gpr[5], context.format.total_samples)?;
                    }
                    if emulator.cpu.gpr[6] != 0 {
                        emulator.memory.write_u32(emulator.cpu.gpr[6], u32::MAX)?;
                    }
                    if emulator.cpu.gpr[7] != 0 {
                        emulator.memory.write_u32(emulator.cpu.gpr[7], u32::MAX)?;
                    }
                    0
                } else {
                    ATRAC_ERROR_BAD_ATRACID
                };
            }
            ("sceAtrac3plus", 0xa554_a158) => {
                let id = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if let Some(context) = self.atrac.context(id) {
                    if emulator.cpu.gpr[5] != 0 {
                        emulator
                            .memory
                            .write_u32(emulator.cpu.gpr[5], context.bitrate())?;
                    }
                    0
                } else {
                    ATRAC_ERROR_BAD_ATRACID
                };
            }
            ("sceAtrac3plus", 0xe88f_759b) => {
                let id = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if self.atrac.context(id).is_some() {
                    if emulator.cpu.gpr[5] != 0 {
                        emulator.memory.write_u32(emulator.cpu.gpr[5], 0)?;
                    }
                    0
                } else {
                    ATRAC_ERROR_BAD_ATRACID
                };
            }
            ("sceAtrac3plus", 0xfaa4_f89b) => {
                let id = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if let Some(context) = self.atrac.context(id) {
                    if emulator.cpu.gpr[5] != 0 {
                        emulator
                            .memory
                            .write_u32(emulator.cpu.gpr[5], context.loop_num as u32)?;
                    }
                    if emulator.cpu.gpr[6] != 0 {
                        emulator.memory.write_u32(emulator.cpu.gpr[6], 0)?;
                    }
                    0
                } else {
                    ATRAC_ERROR_BAD_ATRACID
                };
            }
            ("sceAudio", 0x5ec8_1c55) => {
                let requested = emulator.cpu.gpr[4] as i32;
                let sample_count = emulator.cpu.gpr[5];
                let format = emulator.cpu.gpr[6];
                emulator.cpu.gpr[2] = self.audio.reserve(requested, sample_count, format);
            }
            ("sceAudio", 0xcb2e_439e) => {
                emulator.cpu.gpr[2] = self
                    .audio
                    .set_data_len(emulator.cpu.gpr[4], emulator.cpu.gpr[5]);
            }
            ("sceAudio", 0x95fd_0c2d) => {
                emulator.cpu.gpr[2] = self
                    .audio
                    .change_format(emulator.cpu.gpr[4], emulator.cpu.gpr[5]);
            }
            ("sceAudio", 0xb7e1_d8e7) => {
                emulator.cpu.gpr[2] = self.audio.change_volume(
                    emulator.cpu.gpr[4],
                    emulator.cpu.gpr[5],
                    emulator.cpu.gpr[6],
                );
            }
            ("sceAudio", 0x6fc4_6853) => {
                emulator.cpu.gpr[2] = self.audio.release(emulator.cpu.gpr[4]);
            }
            ("sceAudio", 0xe9d9_7901 | 0xb011_922f) => {
                self.audio.advance_to(emulator.scheduler.now());
                emulator.cpu.gpr[2] = self
                    .audio
                    .channel_rest(emulator.cpu.gpr[4])
                    .unwrap_or_else(|error| error);
            }
            ("sceAudio", 0x8c10_09b2 | 0x136c_af51 | 0xe2d5_6b2d | 0x13f5_92bc) => {
                self.audio.advance_to(emulator.scheduler.now());
                let channel = emulator.cpu.gpr[4];
                let blocking = matches!(import.nid, 0x136c_af51 | 0x13f5_92bc);
                let panned = matches!(import.nid, 0xe2d5_6b2d | 0x13f5_92bc);
                let left_volume = emulator.cpu.gpr[5];
                let right_volume = if panned {
                    emulator.cpu.gpr[6]
                } else {
                    left_volume
                };
                let buffer = if panned {
                    emulator.cpu.gpr[7]
                } else {
                    emulator.cpu.gpr[6]
                };
                let Some((sample_count, format)) = self.audio.channel_output(channel) else {
                    emulator.cpu.gpr[2] = AUDIO_ERROR_CHANNEL_NOT_INIT;
                    return Ok(());
                };
                let (left_volume, right_volume) =
                    match self
                        .audio
                        .output_volumes(channel, left_volume, right_volume)
                    {
                        Ok(volumes) => volumes,
                        Err(error) => {
                            emulator.cpu.gpr[2] = error;
                            return Ok(());
                        }
                    };
                let delay = match self
                    .audio
                    .enqueue(channel, sample_count, buffer != 0, blocking)
                {
                    Ok(delay) => delay,
                    Err(error) => {
                        emulator.cpu.gpr[2] = error;
                        return Ok(());
                    }
                };
                if buffer == 0 {
                    // a null buffer is the firmware's drain request. the
                    // host sink is asynchronous, so there is no guest pcm to
                    // copy in this case; report the channel's sample length.
                    emulator.cpu.gpr[2] = sample_count;
                } else {
                    let pcm = read_guest_pcm(
                        &emulator.memory,
                        buffer,
                        sample_count,
                        format,
                        left_volume,
                        right_volume,
                    )?;
                    emulator.audio.submit_pcm_from("channel", &pcm);
                }
                emulator.cpu.gpr[2] = sample_count;
                if blocking && delay != 0 {
                    self.wait_current_for(emulator, delay)?;
                }
            }
            ("sceAudio", 0x0156_2ba3) => {
                emulator.cpu.gpr[2] = self.audio.reserve_output2(emulator.cpu.gpr[4]);
            }
            ("sceAudio", 0x4319_6845) => {
                emulator.cpu.gpr[2] = self.audio.release_output2();
            }
            ("sceAudio", 0x63f2_889c) => {
                emulator.cpu.gpr[2] = self.audio.change_output2_length(emulator.cpu.gpr[4]);
            }
            ("sceAudio", 0x647c_ef33) => {
                self.audio.advance_to(emulator.scheduler.now());
                emulator.cpu.gpr[2] = if self.audio.output2_reserved {
                    self.audio
                        .output2_queued_samples
                        .min(self.audio.output2_sample_count)
                } else {
                    AUDIO_ERROR_CHANNEL_NOT_INIT
                };
            }
            ("sceAudio", 0x3855_3111) => {
                let frequency = emulator.cpu.gpr[5];
                let channels = emulator.cpu.gpr[6];
                emulator.cpu.gpr[2] = if frequency != 44_100 || channels != 2 {
                    AUDIO_ERROR_INVALID_FORMAT
                } else {
                    self.audio.reserve_output2(emulator.cpu.gpr[4])
                };
            }
            ("sceAudio", 0x5c37_c0ae) => {
                emulator.cpu.gpr[2] = self.audio.release_output2();
            }
            ("sceAudio", 0x2d53_f36e | 0xe072_7056) => {
                self.audio.advance_to(emulator.scheduler.now());
                let volume = emulator.cpu.gpr[4];
                let buffer = emulator.cpu.gpr[5];
                let sample_count = if self.audio.output2_reserved {
                    self.audio.output2_sample_count
                } else if buffer != 0 {
                    // the firmware also permits src output to auto-reserve
                    // when a title supplies its first buffer directly.
                    self.audio.output2_reserved = true;
                    self.audio.output2_sample_count = 0x800;
                    0x800
                } else {
                    0
                };
                if buffer != 0 && sample_count != 0 {
                    let pcm =
                        read_guest_pcm(&emulator.memory, buffer, sample_count, 0, volume, volume)?;
                    emulator.audio.submit_pcm_from("src", &pcm);
                }
                let delay = match self.audio.enqueue_output2(sample_count, buffer != 0, true) {
                    Ok(delay) => delay,
                    Err(error) => {
                        emulator.cpu.gpr[2] = error;
                        return Ok(());
                    }
                };
                emulator.cpu.gpr[2] = if buffer == 0 { 0 } else { sample_count };
                if delay != 0 {
                    self.wait_current_for(emulator, delay)?;
                }
            }
            ("sceSasCore", 0x4277_8a9f) => {
                let core = emulator.cpu.gpr[4];
                let grain_size = emulator.cpu.gpr[5];
                let max_voices = emulator.cpu.gpr[6];
                let output_mode = emulator.cpu.gpr[7];
                let sample_rate = emulator.cpu.gpr[8];
                emulator.cpu.gpr[2] = if core & 0x3f != 0 || emulator.memory.read_u8(core).is_err()
                {
                    SAS_ERROR_BAD_ADDRESS
                } else if !(0x40..=0x800).contains(&grain_size) || grain_size & 0x1f != 0 {
                    SAS_ERROR_INVALID_GRAIN
                } else if max_voices == 0 || max_voices > 32 {
                    SAS_ERROR_INVALID_MAX_VOICES
                } else if output_mode > 1 {
                    SAS_ERROR_INVALID_OUTPUT_MODE
                } else if sample_rate != 44_100 {
                    SAS_ERROR_INVALID_SAMPLE_RATE
                } else if self.sas.initialize(
                    grain_size as usize,
                    max_voices as usize,
                    output_mode,
                    sample_rate,
                ) {
                    0
                } else {
                    SAS_ERROR_INVALID_PARAMETER
                };
            }
            ("sceSasCore", 0xa358_9d81) => {
                let output = emulator.cpu.gpr[5];
                self.run_sas_mix(emulator, output, None, 0x1000, 0x1000)?;
            }
            ("sceSasCore", 0x50a1_4dfc) => {
                let output = emulator.cpu.gpr[5];
                let left_volume = emulator.cpu.gpr[6] as i32;
                let right_volume = emulator.cpu.gpr[7] as i32;
                self.run_sas_mix(emulator, output, Some(output), left_volume, right_volume)?;
            }
            ("sceSasCore", 0x68a4_6b95) => {
                emulator.cpu.gpr[2] = if self.sas.is_initialized() {
                    self.sas.end_flags()
                } else {
                    SAS_ERROR_NOT_INIT
                };
            }
            ("sceSasCore", 0x440c_a7d8) => {
                let voice = emulator.cpu.gpr[5] as usize;
                let left = emulator.cpu.gpr[6] as i32;
                let right = emulator.cpu.gpr[7] as i32;
                let effect_left = emulator.cpu.gpr[8] as i32;
                let effect_right = emulator.cpu.gpr[9] as i32;
                let valid = self.sas.is_initialized()
                    && self.sas.set_volume_with_effect(
                        voice,
                        left,
                        right,
                        effect_left,
                        effect_right,
                    );
                emulator.cpu.gpr[2] = if !self.sas.is_initialized() {
                    SAS_ERROR_NOT_INIT
                } else if voice >= 32 {
                    SAS_ERROR_INVALID_VOICE
                } else if !valid {
                    SAS_ERROR_INVALID_VOLUME
                } else {
                    0
                };
            }
            ("sceSasCore", 0xad84_d37f) => {
                let voice = emulator.cpu.gpr[5] as usize;
                let pitch = emulator.cpu.gpr[6];
                emulator.cpu.gpr[2] = if !self.sas.is_initialized() {
                    SAS_ERROR_NOT_INIT
                } else if voice >= 32 {
                    SAS_ERROR_INVALID_VOICE
                } else if !self.sas.set_pitch(voice, pitch) {
                    SAS_ERROR_INVALID_PITCH
                } else {
                    0
                };
            }
            ("sceSasCore", 0x9994_4089) => {
                let voice = emulator.cpu.gpr[5] as usize;
                let address = emulator.cpu.gpr[6];
                let size = emulator.cpu.gpr[7] as usize;
                let loop_enabled = emulator.cpu.gpr[8] != 0;
                emulator.cpu.gpr[2] = if !self.sas.is_initialized() {
                    SAS_ERROR_NOT_INIT
                } else if voice >= 32 {
                    SAS_ERROR_INVALID_VOICE
                } else if !self.sas.set_voice_vag(voice, address, size, loop_enabled) {
                    SAS_ERROR_INVALID_PARAMETER
                } else {
                    0
                };
            }
            ("sceSasCore", 0xe1cd_9561) => {
                let voice = emulator.cpu.gpr[5] as usize;
                let address = emulator.cpu.gpr[6];
                let size = emulator.cpu.gpr[7] as usize;
                let loop_position = emulator.cpu.gpr[8] as i32;
                emulator.cpu.gpr[2] = if !self.sas.is_initialized() {
                    SAS_ERROR_NOT_INIT
                } else if voice >= 32 {
                    SAS_ERROR_INVALID_VOICE
                } else if size == 0 || size > 0x10000 {
                    SAS_ERROR_INVALID_PCM_SIZE
                } else if loop_position >= 0 && loop_position >= size as i32 {
                    SAS_ERROR_INVALID_LOOP_POS
                } else if !self.sas.set_voice_pcm(voice, address, size, loop_position) {
                    SAS_ERROR_INVALID_PARAMETER
                } else {
                    0
                };
            }
            ("sceSasCore", 0xa0cf_2fa4 | 0x76f0_1aca) => {
                let voice = emulator.cpu.gpr[5] as usize;
                emulator.cpu.gpr[2] = if !self.sas.is_initialized() {
                    SAS_ERROR_NOT_INIT
                } else if voice >= 32 {
                    SAS_ERROR_INVALID_VOICE
                } else if import.nid == 0x76f0_1aca {
                    if self.sas.key_on(voice) {
                        0
                    } else {
                        SAS_ERROR_VOICE_PAUSED
                    }
                } else if self.sas.key_off(voice) {
                    0
                } else {
                    SAS_ERROR_VOICE_PAUSED
                };
            }
            ("sceSasCore", 0x2c8e_6ab3) => {
                emulator.cpu.gpr[2] = if self.sas.is_initialized() {
                    self.sas.pause_flags()
                } else {
                    SAS_ERROR_NOT_INIT
                };
            }
            ("sceSasCore", 0x787d_04d5) => {
                if self.sas.is_initialized() {
                    self.sas
                        .set_pause(emulator.cpu.gpr[5], emulator.cpu.gpr[6] != 0);
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = SAS_ERROR_NOT_INIT;
                }
            }
            ("sceSasCore", 0xbd11_b7c2) => {
                emulator.cpu.gpr[2] = if self.sas.is_initialized() {
                    self.sas.grain_size() as u32
                } else {
                    SAS_ERROR_NOT_INIT
                };
            }
            ("sceSasCore", 0xd1e0_a01e) => {
                emulator.cpu.gpr[2] = if !self.sas.is_initialized() {
                    SAS_ERROR_NOT_INIT
                } else if self.sas.set_grain_size(emulator.cpu.gpr[5] as usize) {
                    0
                } else {
                    SAS_ERROR_INVALID_GRAIN
                };
            }
            ("sceSasCore", 0xe175_ef66) => {
                emulator.cpu.gpr[2] = if self.sas.is_initialized() {
                    self.sas.output_mode()
                } else {
                    SAS_ERROR_NOT_INIT
                };
            }
            ("sceSasCore", 0xe855_bf76) => {
                emulator.cpu.gpr[2] = if !self.sas.is_initialized() {
                    SAS_ERROR_NOT_INIT
                } else if self.sas.set_output_mode(emulator.cpu.gpr[5]) {
                    0
                } else {
                    SAS_ERROR_INVALID_OUTPUT_MODE
                };
            }
            ("sceSasCore", 0x07f5_8c24) => {
                let output = emulator.cpu.gpr[5];
                if !self.sas.is_initialized() {
                    emulator.cpu.gpr[2] = SAS_ERROR_NOT_INIT;
                } else {
                    for index in 0..32u32 {
                        emulator.memory.write_u32(output + index * 4, 0)?;
                    }
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceSasCore", 0x33d4_ab37 | 0xf983_b186 | 0xd5a2_29c9 | 0x267a_6dd2) => {
                emulator.cpu.gpr[2] = if self.sas.is_initialized() {
                    0
                } else {
                    SAS_ERROR_NOT_INIT
                };
            }
            (
                "sceSasCore",
                0xb766_0a23 | 0x019b_25eb | 0x9ec3_676a | 0x5f95_29f6 | 0xcbcd_4f79 | 0xa232_cbe6
                | 0xd5eb_bbcd | 0x4aa9_ead6 | 0x7497_ea85 | 0xf610_7f00,
            ) => {
                emulator.cpu.gpr[2] = if self.sas.is_initialized() {
                    0
                } else {
                    SAS_ERROR_NOT_INIT
                };
            }
            ("IoFileMgrForUser", 0xe3eb_004c) => {
                let fd = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                let directory = self
                    .directories
                    .get_mut(&fd)
                    .with_context(|| format!("sceIoDread: bad fd {fd}"))?;
                if let Some(entry) = directory.entries.get(directory.index) {
                    directory.index += 1;
                    if entry.name.eq_ignore_ascii_case("ENGLISH.GXT") {
                        tracing::debug!(
                            output = format_args!("0x{output:08x}"),
                            extent = entry.extent,
                            size = entry.size,
                            instructions = emulator.cpu.instruction_count,
                            thread = self.current_uid,
                            "ENGLISH.GXT directory entry delivered"
                        );
                    }
                    let mut dirent = vec![0u8; SCE_IO_DIRENT_SIZE];
                    let metadata = DiscMetadata {
                        extent: entry.extent,
                        size: u64::from(entry.size),
                        directory: entry.directory,
                        access: SCE_STM_ISO_ACCESS,
                    };
                    dirent[..SCE_IO_STAT_SIZE].copy_from_slice(&psp_io_stat(metadata));
                    let name = entry.name.as_bytes();
                    let length = name.len().min(255);
                    dirent[88..88 + length].copy_from_slice(&name[..length]);
                    emulator.memory.write_bytes(output, &dirent)?;
                    emulator.cpu.gpr[2] = 1;
                } else {
                    // sceiodread terminates a listing by clearing the first
                    // byte of d_name.  leaving the previous entry in place
                    // makes callers process the last filename twice.
                    emulator.memory.write_u8(output.wrapping_add(88), 0)?;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("IoFileMgrForUser", 0xeb09_2469) => {
                self.directories.remove(&emulator.cpu.gpr[4]);
                emulator.cpu.gpr[2] = 0;
            }
            ("IoFileMgrForUser", 0x109f_50bc) => {
                let path = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])?;
                tracing::debug!(%path, "disc file open requested");
                let result = self.open_disc_file(&path);
                if let Some(file) = result {
                    let length = file.len();
                    let fd = self.insert_file(file);
                    tracing::debug!(fd, length, "disc file opened");
                    emulator.cpu.gpr[2] = fd;
                } else {
                    emulator.cpu.gpr[2] = IO_ERROR_FILE_NOT_FOUND;
                }
            }
            ("IoFileMgrForUser", 0x89aa_9906) => {
                let path = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])?;
                let fd = self.next_fd;
                self.next_fd += 1;
                if let Some(file) = self.open_disc_file(&path) {
                    self.files.insert(fd, file);
                    self.async_operations.insert(
                        fd,
                        AsyncFileOperation {
                            result: i64::from(fd),
                            completed: true,
                            close_pending: false,
                        },
                    );
                    tracing::debug!(fd, %path, "async disc file opened");
                } else {
                    // psp async open still returns an fd so the caller can
                    // collect the failure through sceiopollasync/waitasync.
                    self.files.insert(fd, GuestFile::bytes(Vec::new(), 0));
                    self.async_operations.insert(
                        fd,
                        AsyncFileOperation {
                            result: signed_io_result(IO_ERROR_FILE_NOT_FOUND),
                            completed: true,
                            close_pending: true,
                        },
                    );
                    tracing::debug!(fd, %path, "async disc file open failed");
                }
                emulator.cpu.gpr[2] = fd;
            }
            ("IoFileMgrForUser", 0x6363_2449) => {
                // the psp umd path exposes a small ioctl abi in addition to
                // ordinary reads. a number of loaders use these commands to
                // avoid parsing the iso directory a second time.
                let fd = emulator.cpu.gpr[4];
                let command = emulator.cpu.gpr[5];
                let input = emulator.cpu.gpr[6];
                let input_len = emulator.cpu.gpr[7];
                let output = emulator.cpu.gpr[8];
                let output_len = emulator.cpu.gpr[9];
                let Some((position, size, start_sector)) = self
                    .files
                    .get(&fd)
                    .map(|file| (file.position, file.size(), file.start_sector))
                else {
                    emulator.cpu.gpr[2] = IO_ERROR_BAD_FD;
                    return Ok(());
                };

                match command {
                    // get umd sector size.
                    0x0102_0003 => {
                        if output == 0 || output_len < 4 {
                            emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                        } else {
                            emulator.memory.write_u32(output, ISO_SECTOR_SIZE as u32)?;
                            emulator.cpu.gpr[2] = 0;
                        }
                    }
                    // get the current byte offset in the file.
                    0x0102_0004 => {
                        if output == 0 || output_len < 4 {
                            emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                        } else {
                            emulator.memory.write_u32(output, position as u32)?;
                            emulator.cpu.gpr[2] = 0;
                        }
                    }
                    // seek using the psp's 16-byte seek descriptor. the
                    // kernel accepts an input length of four but still reads
                    // the complete descriptor in practice.
                    0x0101_0005 => {
                        if input == 0 || input_len < 16 {
                            emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                        } else {
                            let offset_bits = u64::from(emulator.memory.read_u32(input)?)
                                | (u64::from(emulator.memory.read_u32(input.wrapping_add(4))?)
                                    << 32);
                            let offset = offset_bits as i64;
                            let whence = emulator.memory.read_u32(input.wrapping_add(12))?;
                            let base = match whence {
                                0 => 0i64,
                                1 => position as i64,
                                2 => size as i64,
                                _ => -1,
                            };
                            let new_position = base.saturating_add(offset);
                            if base < 0 || new_position < 0 || new_position as u64 > size {
                                emulator.cpu.gpr[2] = IO_ERROR_IO;
                            } else {
                                self.files
                                    .get_mut(&fd)
                                    .expect("file metadata was checked above")
                                    .position = new_position as usize;
                                emulator.cpu.gpr[2] = 0;
                            }
                        }
                    }
                    // get the iso start sector for this file.
                    0x0102_0006 => {
                        if output == 0 || output_len < 4 {
                            emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                        } else {
                            emulator.memory.write_u32(output, start_sector)?;
                            emulator.cpu.gpr[2] = 0;
                        }
                    }
                    // get the file size in bytes.
                    0x0102_0007 => {
                        if output == 0 || output_len < 8 {
                            emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                        } else {
                            emulator.memory.write_u32(output, size as u32)?;
                            emulator
                                .memory
                                .write_u32(output.wrapping_add(4), (size >> 32) as u32)?;
                            emulator.cpu.gpr[2] = 0;
                        }
                    }
                    // read a byte count supplied through the input pointer.
                    0x0103_0008 => {
                        if input == 0 || input_len < 4 || output == 0 {
                            emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                        } else {
                            let requested = emulator.memory.read_u32(input)? as usize;
                            if requested as u64 > u64::from(output_len) {
                                emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                            } else {
                                let file = self
                                    .files
                                    .get_mut(&fd)
                                    .expect("file metadata was checked above");
                                let read_started = timer_start();
                                let bytes = file.read(self.disc.as_mut(), requested)?;
                                let read_ms = timer_elapsed(read_started).as_secs_f64() * 1_000.0;
                                if read_ms >= 1.0 {
                                    tracing::debug!(
                                        target: "psp_rs::io_timing",
                                        fd,
                                        position = file.position.saturating_sub(bytes.len()),
                                        requested,
                                        returned = bytes.len(),
                                        read_ms,
                                        "slow ioctl disc read"
                                    );
                                }
                                let count = bytes.len();
                                emulator.memory.write_bytes(output, &bytes)?;
                                emulator.cpu.gpr[2] = count as u32;
                            }
                        }
                    }
                    // current sector position for a umd device file.
                    0x01d2_0001 => {
                        if output == 0 || output_len < 4 {
                            emulator.cpu.gpr[2] = IO_ERROR_INVALID_ARGUMENT;
                        } else {
                            emulator
                                .memory
                                .write_u32(output, (position as u64 / ISO_SECTOR_SIZE) as u32)?;
                            emulator.cpu.gpr[2] = 0;
                        }
                    }
                    _ => {
                        tracing::debug!(
                            fd,
                            command = format_args!("0x{command:08x}"),
                            "unsupported sceIoIoctl command"
                        );
                        emulator.cpu.gpr[2] = IO_ERROR_NOT_SUPPORTED;
                    }
                }
            }
            ("IoFileMgrForUser", 0x6a63_8d83) => {
                let fd = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                let requested = emulator.cpu.gpr[6] as usize;
                let file = self
                    .files
                    .get_mut(&fd)
                    .with_context(|| format!("sceIoRead: bad fd {fd}"))?;
                let position = file.position;
                let read_started = timer_start();
                let bytes = file.read(self.disc.as_mut(), requested)?;
                let read_ms = timer_elapsed(read_started).as_secs_f64() * 1_000.0;
                if read_ms >= 1.0 {
                    tracing::debug!(
                        target: "psp_rs::io_timing",
                        fd,
                        position,
                        requested,
                        returned = bytes.len(),
                        read_ms,
                        "slow disc read"
                    );
                }
                let count = bytes.len();
                if position == 0 {
                    let prefix = bytes
                        .iter()
                        .take(64)
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<Vec<_>>()
                        .join("");
                    tracing::debug!(fd, requested, %prefix, "disc read starts at byte zero");
                }
                emulator.memory.write_bytes(output, &bytes)?;
                let hash = fnv1a64(&bytes);
                emulator.cpu.gpr[2] = count as u32;
                // the emulated i/o timing uses the requested byte count,
                // with a 100 us floor. the delay blocks only the calling
                // thread, allowing other threads to run before it resumes.
                // advancing the global clock here without blocking lets a
                // high-priority reader deliver a completion callback before
                // the submitting thread can finish recording its request.
                let read_us = (requested as u64 / 100).max(100);
                tracing::debug!(
                    fd,
                    position,
                    requested,
                    returned = count,
                    destination = format_args!("0x{output:08x}"),
                    hash = format_args!("0x{hash:016x}"),
                    thread = self.current_uid,
                    instructions = emulator.cpu.instruction_count,
                    delay_us = read_us,
                    "disc bytes delivered to guest"
                );
                self.wait_current_for(emulator, read_us)?;
            }
            ("IoFileMgrForUser", 0xa0b5_a7c2) => {
                let fd = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                let requested = emulator.cpu.gpr[6] as usize;
                emulator.cpu.gpr[2] = self.start_async_read(emulator, fd, output, requested)?;
            }
            ("IoFileMgrForUser", 0x810c_4bc3) => {
                let fd = emulator.cpu.gpr[4];
                if self.async_operations.contains_key(&fd) {
                    emulator.cpu.gpr[2] = IO_ERROR_ASYNC_BUSY;
                } else {
                    self.files.remove(&fd);
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("IoFileMgrForUser", 0xff59_40b6) => {
                let fd = emulator.cpu.gpr[4];
                if !self.files.contains_key(&fd) {
                    emulator.cpu.gpr[2] = IO_ERROR_ASYNC_BAD_FD;
                } else {
                    emulator.cpu.gpr[2] = match self.async_operations.entry(fd) {
                        Entry::Occupied(_) => IO_ERROR_ASYNC_BUSY,
                        Entry::Vacant(entry) => {
                            entry.insert(AsyncFileOperation {
                                result: 0,
                                completed: true,
                                close_pending: true,
                            });
                            0
                        }
                    };
                }
            }
            ("IoFileMgrForUser", 0x6896_3324) => {
                let fd = emulator.cpu.gpr[4];
                let offset = emulator.cpu.gpr[5] as i32 as i64;
                let whence = emulator.cpu.gpr[6];
                let file = self
                    .files
                    .get_mut(&fd)
                    .with_context(|| format!("sceIoLseek32: bad fd {fd}"))?;
                let base = match whence {
                    0 => 0,
                    1 => file.position as i64,
                    2 => file.len() as i64,
                    _ => anyhow::bail!("sceIoLseek32: bad whence {whence}"),
                };
                file.position = usize::try_from(base.saturating_add(offset))
                    .context("negative file seek")?
                    .min(file.len());
                emulator.cpu.gpr[2] = file.position as u32;
                tracing::debug!(fd, offset, whence, position = file.position, "disc seek");
            }
            ("IoFileMgrForUser", 0x27eb_27b8) => {
                let fd = emulator.cpu.gpr[4];
                let offset = ((u64::from(emulator.cpu.gpr[7]) << 32)
                    | u64::from(emulator.cpu.gpr[6])) as i64;
                // the 64-bit
                // offset occupies a2/a3 (with a1 padding), so whence arrives
                // in t0 (gpr[8]), not on the stack. reading sp+16 consumed
                // caller garbage and sent umd pump seeks to the wrong base,
                // which re-read the same sectors forever.
                let whence = emulator.cpu.gpr[8];
                let file = self
                    .files
                    .get_mut(&fd)
                    .with_context(|| format!("sceIoLseek: bad fd {fd}"))?;
                let base = match whence {
                    0 => 0,
                    1 => file.position as i64,
                    2 => file.len() as i64,
                    _ => anyhow::bail!("sceIoLseek: bad whence {whence}"),
                };
                let position = base.saturating_add(offset);
                if position < 0 {
                    emulator.cpu.gpr[2] = 0x8001_0016;
                    emulator.cpu.gpr[3] = u32::MAX;
                } else {
                    file.position = (position as usize).min(file.len());
                    emulator.cpu.gpr[2] = position as u32;
                    emulator.cpu.gpr[3] = (position as u64 >> 32) as u32;
                    tracing::debug!(fd, offset, whence, position = file.position, "disc seek");
                }
            }
            ("IoFileMgrForUser", 0x3251_ea56 | 0xcb05_f8d6) => {
                let fd = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                emulator.cpu.gpr[2] = if self.files.contains_key(&fd) {
                    self.consume_async_result(emulator, fd, output)?
                } else {
                    IO_ERROR_ASYNC_BAD_FD
                };
            }
            ("IoFileMgrForUser", 0xe23e_ec33 | 0x35db_d746) => {
                let fd = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                emulator.cpu.gpr[2] = if self.files.contains_key(&fd) {
                    self.consume_async_result(emulator, fd, output)?
                } else {
                    IO_ERROR_ASYNC_BAD_FD
                };
            }
            ("ModuleMgrForUser", 0xb7f4_6618) => {
                let fd = emulator.cpu.gpr[4];
                let payload = self
                    .files
                    .get_mut(&fd)
                    .with_context(|| format!("sceKernelLoadModuleByID: bad fd {fd}"))?
                    .read_remaining(self.disc.as_mut())?;
                let magic = payload
                    .iter()
                    .take(16)
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<Vec<_>>()
                    .join("");
                tracing::debug!(fd, %magic, size = payload.len(), "module payload inspected");
                if payload.starts_with(b"\x7fELF") || payload.starts_with(b"~PSP") {
                    anyhow::bail!(
                        "sceKernelLoadModuleByID reached a real module; dynamic PRX mapping is required"
                    );
                } else if payload.starts_with(b"~SCE") {
                    // encrypted firmware modules use hle. register a module
                    // object without mapping
                    // the opaque kernel prx into user memory.
                    self.next_uid += 1;
                    self.modules.insert(self.next_uid);
                    emulator.cpu.gpr[2] = self.next_uid;
                } else {
                    emulator.cpu.gpr[2] = 0x8002_012d;
                }
            }
            ("ModuleMgrForUser", 0x50f0_c1ec) => {
                emulator.cpu.gpr[2] = if self.modules.contains(&emulator.cpu.gpr[4]) {
                    0
                } else {
                    0x8002_012e
                }
            }
            ("ModuleMgrForUser", 0xd1ff_982a | 0x2e09_11aa) => {
                // firmware-owned modules are already represented by hle and
                // do not have a user prx image to stop or unload.
                emulator.cpu.gpr[2] = 0;
            }
            ("sceUmdUser", 0xc618_3d47) => {
                let mode = emulator.cpu.gpr[4];
                if mode == 1 {
                    self.umd_activated = true;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = KERNEL_ERROR_ILLEGAL_MODE;
                }
            }
            ("sceUmdUser", 0x6b4a_146c) => {
                // present | ready | readable.  a mounted iso is available
                // immediately to the linux host, so there is no media spin-up
                // delay to model here.
                emulator.cpu.gpr[2] = 0x12 | if self.umd_activated { 0x20 } else { 0 };
            }
            ("sceUmdUser", 0x46eb_b729) => emulator.cpu.gpr[2] = 1,
            ("sceUmdUser", 0x8ef0_8fce) => {
                let wanted = emulator.cpu.gpr[4];
                let current = 0x12 | if self.umd_activated { 0x20 } else { 0 };
                emulator.cpu.gpr[2] = if wanted == 0 || current & wanted == wanted {
                    0
                } else {
                    0x8002_01a8
                };
            }
            ("sceUmdUser", 0xaee7_404d) => {
                self.umd_callback = Some(emulator.cpu.gpr[4]);
                emulator.cpu.gpr[2] = 0;
            }
            ("sceUmdUser", 0xbd2b_de07) => {
                let callback = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if self.umd_callback == Some(callback) {
                    self.umd_callback = None;
                    0
                } else {
                    0x8001_0004
                };
            }
            ("sceCtrl", 0x3e65_a0ea) => {
                emulator.cpu.gpr[2] = 0;
            }
            ("sceCtrl", 0x1f40_11e6) => {
                let mode = emulator.cpu.gpr[4];
                if mode > 1 {
                    emulator.cpu.gpr[2] = KERNEL_ERROR_ILLEGAL_MODE;
                } else {
                    let previous = self.ctrl_mode;
                    self.ctrl_mode = mode;
                    emulator.input.set_analog_enabled(mode == 1);
                    emulator.cpu.gpr[2] = previous;
                }
            }
            ("sceCtrl", 0x6a27_74f3) => {
                let cycle = emulator.cpu.gpr[4];
                if (cycle > 0 && cycle < 5_555) || cycle > 20_000 {
                    emulator.cpu.gpr[2] = KERNEL_ERROR_INVALID_VALUE;
                } else {
                    let previous = self.ctrl_cycle;
                    self.ctrl_cycle = cycle;
                    self.ctrl_last_sample = emulator.scheduler.now();
                    emulator.cpu.gpr[2] = previous;
                }
            }
            ("sceCtrl", 0x02ba_ad91) => {
                if emulator.cpu.gpr[4] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[4], self.ctrl_cycle)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("sceCtrl", 0xda6b_76a1) => {
                if emulator.cpu.gpr[4] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[4], self.ctrl_mode)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("sceCtrl", 0xa714_4800) => {
                let idle_reset = emulator.cpu.gpr[4] as i32;
                let idle_back = emulator.cpu.gpr[5] as i32;
                if !(-1..=128).contains(&idle_reset) || !(-1..=128).contains(&idle_back) {
                    emulator.cpu.gpr[2] = KERNEL_ERROR_INVALID_VALUE;
                } else {
                    self.ctrl_idle_reset = idle_reset;
                    self.ctrl_idle_back = idle_back;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceCtrl", 0x6876_60fa) => {
                if emulator.cpu.gpr[4] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[4], self.ctrl_idle_reset as u32)?;
                }
                if emulator.cpu.gpr[5] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[5], self.ctrl_idle_back as u32)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("sceCtrl", 0x3a62_2550 | 0xc152_080a | 0x1f80_3938 | 0x60b8_1f86) => {
                let pad_data = emulator.cpu.gpr[4];
                let requested = emulator.cpu.gpr[5];
                let negative = matches!(import.nid, 0xc152_080a | 0x60b8_1f86);
                let peek = matches!(import.nid, 0x3a62_2550 | 0xc152_080a);
                if emulator.input.buttons != 0 {
                    tracing::debug!(
                        thread = self.current_uid,
                        nid = format_args!("0x{:08x}", import.nid),
                        requested,
                        available = emulator.input.available_samples(),
                        peek,
                        negative,
                        "guest polled held controller"
                    );
                }
                if requested as usize > CTRL_SAMPLE_BUFFER_COUNT {
                    emulator.cpu.gpr[2] = KERNEL_ERROR_INVALID_SIZE;
                } else if requested == 0 {
                    emulator.cpu.gpr[2] = 0;
                } else if !peek && emulator.input.available_samples() == 0 {
                    self.ctrl_waiters.push_back(ControllerWaiter {
                        thread: self.current_uid,
                        data: pad_data,
                        negative,
                    });
                    self.block_current(emulator)?;
                } else {
                    let samples = emulator
                        .input
                        .read_samples(requested as usize, negative, peek);
                    if let Some(sample) = samples.iter().find(|sample| sample.buttons != 0) {
                        tracing::debug!(
                            thread = self.current_uid,
                            nid = format_args!("0x{:08x}", import.nid),
                            buttons = format_args!("0x{:08x}", sample.buttons),
                            frame = sample.frame,
                            requested,
                            peek,
                            negative,
                            "guest consumed pressed controller sample"
                        );
                    }
                    for (index, sample) in samples.iter().enumerate() {
                        write_controller_sample(
                            &mut emulator.memory,
                            pad_data.wrapping_add(index as u32 * 16),
                            *sample,
                        )?;
                    }
                    emulator.cpu.gpr[2] = samples.len() as u32;
                }
            }
            ("sceCtrl", 0xb1d0_e5cd | 0xb1d0_5ecd | 0x0b58_8501) => {
                let latch = if import.nid != 0x0b58_8501 {
                    emulator.input.peek_latch()
                } else {
                    emulator.input.read_latch()
                };
                if latch.button_make != 0 || latch.button_press != 0 {
                    tracing::debug!(
                        thread = self.current_uid,
                        nid = format_args!("0x{:08x}", import.nid),
                        make = format_args!("0x{:08x}", latch.button_make),
                        press = format_args!("0x{:08x}", latch.button_press),
                        "guest read controller latch"
                    );
                }
                let address = emulator.cpu.gpr[4];
                if address != 0 {
                    emulator.memory.write_u32(address, latch.button_make)?;
                    emulator
                        .memory
                        .write_u32(address.wrapping_add(4), latch.button_break)?;
                    emulator
                        .memory
                        .write_u32(address.wrapping_add(8), latch.button_press)?;
                    emulator
                        .memory
                        .write_u32(address.wrapping_add(12), latch.button_release)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("sceMpeg", 0x682a_619b) => {
                self.mpeg.init();
                emulator.cpu.gpr[2] = 0;
            }
            ("sceMpeg", 0xd7a2_9f46) => {
                emulator.cpu.gpr[2] =
                    emulator.cpu.gpr[4].wrapping_mul(MPEG_PACKET_OVERHEAD + MPEG_PACKET_SIZE)
            }
            ("sceMpeg", 0xc132_e22f) => emulator.cpu.gpr[2] = 0x0001_0000,
            ("sceMpeg", 0xf8dc_b679) => {
                if emulator.cpu.gpr[5] == 0 || emulator.cpu.gpr[6] == 0 {
                    emulator.cpu.gpr[2] = MPEG_ERROR_INVALID_VALUE;
                } else {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[5], MPEG_ATRAC_ES_SIZE)?;
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[6], MPEG_ATRAC_OUTPUT_SIZE)?;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceMpeg", 0x3729_5ed8) => {
                let ring = emulator.cpu.gpr[4];
                let packets = emulator.cpu.gpr[5];
                let data = emulator.cpu.gpr[6];
                let size = emulator.cpu.gpr[7];
                let required = packets.wrapping_mul(MPEG_PACKET_OVERHEAD + MPEG_PACKET_SIZE);
                if size < required {
                    emulator.cpu.gpr[2] = 0x8061_0022;
                } else {
                    initialize_mpeg_ringbuffer(
                        &mut emulator.memory,
                        ring,
                        packets,
                        data,
                        emulator.cpu.gpr[8],
                        emulator.cpu.gpr[9],
                    )?;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceMpeg", 0xd8c5_f121) => {
                let mpeg = emulator.cpu.gpr[4];
                let data = emulator.cpu.gpr[5];
                let size = emulator.cpu.gpr[6];
                let ring = emulator.cpu.gpr[7];
                if size < 0x0001_0000 {
                    emulator.cpu.gpr[2] = 0x8061_0022;
                } else {
                    let handle = data + 0x30;
                    emulator.memory.write_u32(mpeg, handle)?;
                    emulator.memory.write_bytes(handle, b"LIBMPEG\0")?;
                    emulator.memory.write_bytes(handle + 8, b"001\0")?;
                    emulator.memory.write_u32(handle + 12, u32::MAX)?;
                    let data_end = emulator.memory.read_u32(ring + MPEG_RING_DATA_END)?;
                    emulator.memory.write_u32(handle + MPEG_HANDLE_RING, ring)?;
                    emulator
                        .memory
                        .write_u32(handle + MPEG_HANDLE_DATA_END, data_end)?;
                    emulator.memory.write_u32(ring + MPEG_RING_MPEG, mpeg)?;
                    self.mpeg.configure(emulator.cpu.gpr[8]);
                    self.mpeg.create(handle);
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceMpeg", 0x4256_0f23) => {
                let mpeg = emulator.cpu.gpr[4];
                let handle = emulator.memory.read_u32(mpeg)?;
                emulator.cpu.gpr[2] = self.mpeg.register_stream(handle).unwrap_or(0x8061_8009);
            }
            ("sceMpeg", 0x591a_4aa2) => {
                let mpeg = emulator.cpu.gpr[4];
                let handle = emulator.memory.read_u32(mpeg)?;
                emulator.cpu.gpr[2] = if self
                    .mpeg
                    .unregister_stream(handle, emulator.cpu.gpr[5])
                    .is_some()
                {
                    0
                } else {
                    u32::MAX
                };
            }
            ("sceMpeg", 0xa780_cf7e) => {
                let mpeg = emulator.cpu.gpr[4];
                let handle = emulator.memory.read_u32(mpeg)?;
                emulator.cpu.gpr[2] = self.mpeg.malloc_es_buffer(handle).unwrap_or(u32::MAX);
            }
            ("sceMpeg", 0xceb8_70b1) => {
                let mpeg = emulator.cpu.gpr[4];
                let handle = emulator.memory.read_u32(mpeg)?;
                let buffer = emulator.cpu.gpr[5];
                emulator.cpu.gpr[2] = match self.mpeg.free_es_buffer(handle, buffer) {
                    Some(true) => 0,
                    Some(false) => 0x8061_01fe,
                    None => u32::MAX,
                };
            }
            ("sceMpeg", 0x606a_4649) => {
                let mpeg = emulator.cpu.gpr[4];
                let handle = emulator.memory.read_u32(mpeg)?;
                emulator.cpu.gpr[2] = if self.mpeg.delete(handle) {
                    0
                } else {
                    u32::MAX
                };
            }
            ("sceMpeg", 0x740f_ccd1) => {
                // stop and drop the long-lived host decoders when the title
                // leaves its pmf. dropping the pipes lets both ffmpeg worker
                // threads finish before the next game state starts.
                self.mpeg.media = MediaPipeline::default();
                self.mpeg.last_video_frame = None;
                self.mpeg.decoder_prefetched = false;
                emulator.cpu.gpr[2] = 0;
            }
            ("sceMpeg", 0x8746_24d6) => {
                self.mpeg.finish();
                emulator.cpu.gpr[2] = 0;
            }
            ("sceMpeg", 0x1340_7f13) => emulator.cpu.gpr[2] = 0,
            ("sceMpeg", 0x167a_fd9e) => {
                let mpeg = emulator.cpu.gpr[4];
                let handle = emulator.memory.read_u32(mpeg)?;
                let buffer = emulator.cpu.gpr[5];
                if let Some((es_size, dts)) = self.mpeg.au_parameters(handle, buffer) {
                    // avc uses the small allocator tokens returned by
                    // scempegmallocavcesbuf.  atrac callers pass a real
                    // guest buffer address here; the firmware accepts both
                    // forms and initializes the au accordingly.
                    write_mpeg_au(
                        &mut emulator.memory,
                        emulator.cpu.gpr[6],
                        buffer,
                        dts,
                        es_size,
                    )?;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = MPEG_ERROR_INVALID_VALUE;
                }
            }
            ("sceMpeg", 0xb5f6_dc87) => {
                let ring = emulator.cpu.gpr[4];
                let packets = emulator.memory.read_u32(ring + MPEG_RING_PACKETS)?;
                let packets_in_buffer = emulator
                    .memory
                    .read_u32(ring + MPEG_RING_PACKETS_IN_BUFFER)?;
                emulator.cpu.gpr[2] = packets.saturating_sub(packets_in_buffer);
            }
            ("sceMpeg", 0x769b_ebb6) => {
                emulator.cpu.gpr[2] =
                    emulator.cpu.gpr[4] / (MPEG_PACKET_OVERHEAD + MPEG_PACKET_SIZE);
            }
            ("sceMpeg", 0xb240_a59e) => {
                let ring = emulator.cpu.gpr[4];
                let packets = emulator.memory.read_u32(ring + MPEG_RING_PACKETS)?;
                let packets_in_buffer = emulator
                    .memory
                    .read_u32(ring + MPEG_RING_PACKETS_IN_BUFFER)?;
                let capacity = packets.saturating_sub(packets_in_buffer);
                // ffmpeg's mpeg-ps demuxer needs a small look-ahead before it
                // can expose the first h.264 access unit.  keep the guest
                // callback abi intact while allowing the first media refill to
                // stage enough sectors for low-latency decode.
                let requested_packets = emulator.cpu.gpr[5].min(emulator.cpu.gpr[6]);
                let prefetch = requested_packets != 0 && !self.mpeg.decoder_prefetched;
                let target_packets = if requested_packets == 0 {
                    0
                } else if prefetch {
                    requested_packets.max(MPEG_DECODER_PREFETCH_PACKETS)
                } else {
                    requested_packets
                };
                let target_packets = target_packets.min(capacity);
                let callback = emulator.memory.read_u32(ring + MPEG_RING_CALLBACK)?;
                if target_packets != 0 && callback != 0 && packets != 0 {
                    self.mpeg.decoder_prefetched |= prefetch;
                    let write_position =
                        emulator.memory.read_u32(ring + MPEG_RING_PACKETS_WRITTEN)?;
                    let sequential_packets = packets - (write_position % packets);
                    let requested = target_packets.min(sequential_packets);
                    let packet_size = emulator
                        .memory
                        .read_u32(ring + MPEG_RING_PACKET_SIZE)?
                        .max(MPEG_PACKET_SIZE);
                    let data = emulator.memory.read_u32(ring + MPEG_RING_DATA)?;
                    let buffer =
                        data.wrapping_add((write_position % packets).wrapping_mul(packet_size));
                    let callback_args =
                        emulator.memory.read_u32(ring + MPEG_RING_CALLBACK_PARAM)?;
                    self.interrupt_stack.push(emulator.cpu.clone());
                    self.pending_ringbuffer_put = Some(PendingRingbufferPut {
                        ringbuffer_addr: ring,
                        write_position,
                        target_packets,
                        current_packets: requested,
                        added_packets: 0,
                    });
                    emulator.cpu.pc = callback;
                    emulator.cpu.gpr[4] = buffer;
                    emulator.cpu.gpr[5] = requested;
                    emulator.cpu.gpr[6] = callback_args;
                    emulator.cpu.gpr[31] = 0;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceMpeg", 0xfe24_6728 | 0xe1ce_83a7) => {
                let mpeg = emulator.cpu.gpr[4];
                let ring = mpeg_ringbuffer(&emulator.memory, mpeg)?;
                let au = emulator.cpu.gpr[6];
                let attr = emulator.cpu.gpr[7];
                let es_size = if import.nid == 0xfe24_6728 {
                    MPEG_AVC_ES_SIZE
                } else {
                    MPEG_ATRAC_ES_SIZE
                };
                // audio access units additionally require a demuxed frame
                // backing them.  the ring holds interleaved video packets,
                // so ring occupancy alone lets the game burn silent decodes
                // faster than the audio track delivers; return
                // no_data until more audio arrives.
                let is_atrac_au = import.nid == 0xe1ce_83a7;
                let has_data = mpeg_ring_has_data(&emulator.memory, ring)?
                    && (!is_atrac_au || self.mpeg.media.audio_au_available());
                if has_data {
                    let es_buffer = emulator.memory.read_u32(au + MPEG_AU_ES_BUFFER)?;
                    write_mpeg_au(
                        &mut emulator.memory,
                        au,
                        es_buffer,
                        if import.nid == 0xfe24_6728 {
                            0
                        } else {
                            u32::MAX
                        },
                        es_size,
                    )?;
                }
                if attr != 0 {
                    emulator.memory.write_u32(attr, u32::from(has_data))?;
                }
                emulator.cpu.gpr[2] = if has_data { 0 } else { 0x8061_8001 };
            }
            ("sceMpeg", 0x0e3c_2e9d) => {
                let au = emulator.cpu.gpr[5];
                let output_address = emulator.cpu.gpr[7];
                let init = emulator.cpu.gpr[8];
                let es_size = emulator.memory.read_u32(au + MPEG_AU_SIZE).unwrap_or(0);
                if es_size == 0 {
                    emulator.cpu.gpr[2] = MPEG_ERROR_AVC_DECODE_FATAL;
                } else {
                    // pbuffer is a guest pointer to the actual output-buffer
                    // address. treating it as the frame address itself leaves
                    // the player decoding into its control structure.
                    let output = if output_address == 0 {
                        0
                    } else {
                        emulator.memory.read_u32(output_address)?
                    };
                    let frame_width = emulator.cpu.gpr[6].max(self.mpeg.frame_width);
                    let frame_width = frame_width.max(MPEG_VIDEO_WIDTH);
                    if output != 0 {
                        if let Some(frame) = self.mpeg.media.take_video_frame() {
                            self.mpeg.last_video_frame = Some(frame);
                        }
                        if let Some(frame) = self.mpeg.last_video_frame.as_deref() {
                            write_mpeg_video_frame(
                                &mut emulator.memory,
                                output,
                                frame_width,
                                self.mpeg.pixel_format,
                                frame,
                            )?;
                        } else {
                            write_mpeg_fallback_frame(
                                &mut emulator.memory,
                                output,
                                frame_width,
                                self.mpeg.pixel_format,
                                self.mpeg.next_decoded_frame(),
                            )?;
                        }
                    }
                    if init != 0 {
                        emulator.memory.write_u32(init, 1)?;
                    }
                    // a ring-backed au is consumed by the decode call. the
                    // next getavcau must parse a fresh access unit.
                    emulator.memory.write_u32(au + MPEG_AU_SIZE, 0)?;
                    let ring = mpeg_ringbuffer(&emulator.memory, emulator.cpu.gpr[4])?;
                    consume_mpeg_packet(&mut emulator.memory, ring)?;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceMpeg", 0x800c_44df) => {
                let au = emulator.cpu.gpr[5];
                let output = emulator.cpu.gpr[6];
                let es_size = emulator.memory.read_u32(au + MPEG_AU_SIZE).unwrap_or(0);
                if es_size == 0 {
                    emulator.cpu.gpr[2] = MPEG_ERROR_AVC_DECODE_FATAL;
                } else {
                    let decoded = self.mpeg.media.take_audio_chunk();
                    // the decoded pcm belongs to the game alone: it plays
                    // the buffer through its own reserved audio channel, and
                    // that channel output is what feeds the host sink.
                    // submitting it here as well played every video sound
                    // twice (once at bursty decode time, once at realtime
                    // play time).
                    write_mpeg_audio(
                        &mut emulator.memory,
                        output,
                        decoded.as_deref().unwrap_or(&[]),
                    )?;
                    emulator.memory.write_u32(au + MPEG_AU_SIZE, 0)?;
                    let ring = mpeg_ringbuffer(&emulator.memory, emulator.cpu.gpr[4])?;
                    consume_mpeg_packet(&mut emulator.memory, ring)?;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceMpeg", 0xa11c_7026) => {
                let mode = emulator.cpu.gpr[5];
                if mode != 0 {
                    let pixel_format = emulator.memory.read_u32(mode + 4)?;
                    if pixel_format <= 3 {
                        self.mpeg.set_pixel_format(pixel_format);
                    }
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("sceMpeg", 0x21ff_80e4) => {
                let header = emulator.cpu.gpr[5];
                let output = emulator.cpu.gpr[6];
                if emulator.memory.read_u32(header)? != 0x464d_5350 {
                    emulator.memory.write_u32(output, 0)?;
                    emulator.cpu.gpr[2] = 0x8061_01fe;
                } else {
                    let offset = u32::from_be(emulator.memory.read_u32(header + 8)?);
                    emulator.memory.write_u32(output, offset)?;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceMpeg", 0x611e_9e11) => {
                let header = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                if emulator.memory.read_u32(header)? != 0x464d_5350 {
                    emulator.memory.write_u32(output, 0)?;
                    emulator.cpu.gpr[2] = 0x8061_01fe;
                } else {
                    let size = u32::from_be(emulator.memory.read_u32(header + 12)?);
                    emulator.memory.write_u32(output, size)?;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceGe_user", 0xe47e_40e4) => emulator.cpu.gpr[2] = EDRAM_BASE,
            ("sceGe_user", 0x1f67_52ad) => emulator.cpu.gpr[2] = EDRAM_SIZE as u32,
            ("sceGe_user", 0xab49_e76a | 0x1c0d_95a6) => {
                let start = emulator.cpu.gpr[4];
                let stall = emulator.cpu.gpr[5];
                let callback_id = emulator.cpu.gpr[6];
                let execute_started = timer_start();
                let result = emulator
                    .gpu
                    .execute_list(&mut emulator.memory, start, stall)?;
                let execute_ms = timer_elapsed(execute_started).as_secs_f64() * 1_000.0;
                if execute_ms >= 5.0 {
                    tracing::debug!(target: "psp_rs::ge_timing",
                        start = format_args!("0x{start:08x}"),
                        stall = format_args!("0x{stall:08x}"),
                        words = result.words,
                        draws = emulator.gpu.draw_calls,
                        execute_ms,
                        "slow GE list decode"
                    );
                }
                self.next_uid += 1;
                self.ge_lists.insert(
                    self.next_uid,
                    GeList {
                        start,
                        stall,
                        completed: !result.stalled,
                        callback_id,
                    },
                );
                tracing::debug!(
                    list = self.next_uid,
                    start = format_args!("0x{start:08x}"),
                    words = result.words,
                    draws = emulator.gpu.draw_calls,
                    stalled = result.stalled,
                    vaddr = format_args!("0x{:06x}", emulator.gpu.registers[0x01]),
                    vtype = format_args!("0x{:06x}", emulator.gpu.registers[0x12]),
                    framebuffer = format_args!("0x{:06x}", emulator.gpu.registers[0x9c]),
                    framebuffer_width = format_args!("0x{:06x}", emulator.gpu.registers[0x9d]),
                    framebuffer_format = emulator.gpu.registers[0xd2],
                    clear_mode = format_args!("0x{:06x}", emulator.gpu.registers[0xd3]),
                    rendered_pixels = emulator.gpu.last_render_pixels,
                    primitive = ?emulator.gpu.last_primitive,
                    vertices = ?emulator.gpu.last_vertex_probe,
                    "GE list executed"
                );
                self.infer_display_from_ge(&emulator.gpu);
                self.present(emulator, false)?;
                emulator.cpu.gpr[2] = self.next_uid;
                self.deliver_ge_events(emulator, callback_id, &result);
            }
            ("sceGe_user", 0xe0d6_8148) => {
                let uid = emulator.cpu.gpr[4];
                let stall = emulator.cpu.gpr[5];
                let list = self
                    .ge_lists
                    .get_mut(&uid)
                    .with_context(|| format!("sceGeListUpdateStallAddr: bad list {uid}"))?;
                list.stall = stall;
                let callback_id = list.callback_id;
                if !list.completed {
                    let execute_started = timer_start();
                    let result =
                        emulator
                            .gpu
                            .execute_list(&mut emulator.memory, list.start, list.stall)?;
                    let execute_ms = timer_elapsed(execute_started).as_secs_f64() * 1_000.0;
                    if execute_ms >= 5.0 {
                        tracing::debug!(target: "psp_rs::ge_timing",
                            list = uid,
                            start = format_args!("0x{:08x}", list.start),
                            stall = format_args!("0x{:08x}", stall),
                            words = result.words,
                            draws = emulator.gpu.draw_calls,
                            execute_ms,
                            "slow GE list update"
                        );
                    }
                    list.completed = !result.stalled;
                    self.infer_display_from_ge(&emulator.gpu);
                    self.present(emulator, false)?;
                    self.deliver_ge_events(emulator, callback_id, &result);
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("sceGe_user", 0x0344_4eb4) => {
                let uid = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if self.ge_lists.get(&uid).is_some_and(|list| list.completed)
                {
                    0
                } else {
                    1
                };
            }
            ("sceGe_user", 0xb287_bd61 | 0xb448_ec0d | 0x4c06_e472) => {
                emulator.cpu.gpr[2] = 0;
            }
            ("sceGe_user", 0xa4fc_06a4) => {
                let address = emulator.cpu.gpr[4];
                let callback_id = (0..16)
                    .find(|id| !self.ge_callbacks.contains_key(id))
                    .context("sceGeSetCallback: callback table exhausted")?;
                self.ge_callbacks.insert(
                    callback_id,
                    GeCallback {
                        _signal_function: emulator.memory.read_u32(address)?,
                        _signal_argument: emulator.memory.read_u32(address + 4)?,
                        finish_function: emulator.memory.read_u32(address + 8)?,
                        finish_argument: emulator.memory.read_u32(address + 12)?,
                    },
                );
                emulator.cpu.gpr[2] = callback_id;
            }
            ("sceGe_user", 0x05db_22ce) => {
                emulator.cpu.gpr[2] = if self.ge_callbacks.remove(&emulator.cpu.gpr[4]).is_some() {
                    0
                } else {
                    0x8002_00cb
                }
            }
            ("InterruptManager", 0xca04_a2b9) => {
                let key = (emulator.cpu.gpr[4], emulator.cpu.gpr[5]);
                if emulator.cpu.gpr[6] == 0 || self.subinterrupts.contains_key(&key) {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                } else {
                    self.subinterrupts.insert(
                        key,
                        SubInterrupt {
                            handler: emulator.cpu.gpr[6],
                            argument: emulator.cpu.gpr[7],
                            enabled: false,
                        },
                    );
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("InterruptManager", 0xfb8e_22ec | 0x8a38_9411) => {
                let key = (emulator.cpu.gpr[4], emulator.cpu.gpr[5]);
                if let Some(handler) = self.subinterrupts.get_mut(&key) {
                    handler.enabled = import.nid == 0xfb8e_22ec;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = 0x8002_00cb;
                }
            }
            ("InterruptManager", 0xd61e_6961) => {
                emulator.cpu.gpr[2] = if self
                    .subinterrupts
                    .remove(&(emulator.cpu.gpr[4], emulator.cpu.gpr[5]))
                    .is_some()
                {
                    0
                } else {
                    0x8002_00cb
                };
            }
            ("sceDisplay", 0x0e20_f177) => {
                self.display.mode = emulator.cpu.gpr[4];
                self.display.width = emulator.cpu.gpr[5];
                self.display.height = emulator.cpu.gpr[6];
                emulator.cpu.gpr[2] = 0;
            }
            ("sceDisplay", 0x289d_82fe) => {
                self.display.frame_buffer = emulator.cpu.gpr[4];
                self.display.buffer_width = emulator.cpu.gpr[5];
                self.display.pixel_format = emulator.cpu.gpr[6];
                self.display.explicit_frame_buffer = self.display.frame_buffer != 0;
                // scedisplaysetmode is optional on real firmware.  titles
                // that render directly to a framebuffer still scan out the
                // fixed psp panel dimensions when they select a non-null
                // buffer.
                if self.display.frame_buffer != 0 {
                    if self.display.width == 0 {
                        self.display.width = DISPLAY_WIDTH;
                    }
                    if self.display.height == 0 {
                        self.display.height = DISPLAY_HEIGHT;
                    }
                }
                // selecting a scanout buffer does not itself start a display
                // scan.  the real display controller consumes this selection
                // at vblank.  deferring the host submission here is
                // important: gta changes the selected buffer while building
                // a frame, and presenting from this syscall turns one frame
                // into many expensive gpu submissions.
                emulator.cpu.gpr[2] = 0;
            }
            ("sceDisplay", 0xeeda_2e54) => {
                if emulator.cpu.gpr[4] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[4], self.display.frame_buffer)?;
                }
                if emulator.cpu.gpr[5] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[5], self.display.buffer_width)?;
                }
                if emulator.cpu.gpr[6] != 0 {
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[6], self.display.pixel_format)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("sceDisplay", 0x984c_27e7 | 0x46f1_86c3 | 0x36cd_fade | 0x8eb9_ec49) => {
                if self.ctrl_cycle == 0 {
                    self.sample_controller(emulator, emulator.scheduler.now())?;
                }
                self.pace_host_vblank();
                // pump the host compositor and scan out the current display
                // buffer at vblank.  this mirrors the psp display lifecycle
                // and works for ge-rendered as well as cpu-rendered frames.
                self.present(emulator, true)?;
                emulator.cpu.gpr[2] = 0;
                if self.trigger_subinterrupt(30, 15, emulator) {
                    // the callback is an interrupt handler for this same
                    // vblank.  delay until it returns so the handler can run
                    // without allowing the display thread to spin.
                    self.vblank_wait_pending = true;
                } else {
                    // a vblank wait blocks until the next display tick.  a
                    // plain priority-ordered yield would select this same
                    // high-priority display thread forever and starve media,
                    // input, and loader threads.
                    self.wait_current_for(emulator, 16_667)?;
                }
            }
            ("sceDisplay", 0x9c6e_aad7) => {
                let vcount = display_vcount(emulator.scheduler.now());
                tracing::debug!(vcount, thread = self.current_uid, "GetVcount");
                emulator.cpu.gpr[2] = vcount;
            }
            ("sceDisplay", 0x210e_ab3a) => {
                let hcount = display_hcount(emulator.scheduler.now());
                tracing::debug!(hcount, thread = self.current_uid, "GetAccumulatedHcount");
                emulator.cpu.gpr[2] = hcount;
            }
            ("sceUtility", 0xa5da_2406) => {
                let id = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                if let Some(value) = system_param_int(id) {
                    if output == 0 {
                        emulator.cpu.gpr[2] = 0x8011_0103;
                    } else {
                        emulator.memory.write_u32(output, value)?;
                        emulator.cpu.gpr[2] = 0;
                    }
                } else {
                    emulator.cpu.gpr[2] = 0x8011_0103;
                }
            }
            ("sceSuspendForUser", 0xa14f_40b2) => {
                let address_output = emulator.cpu.gpr[5];
                let size_output = emulator.cpu.gpr[6];
                if self.volatile_locked {
                    // sce_kernel_error_power_vmem_in_use
                    emulator.cpu.gpr[2] = 0x802b_0200;
                } else {
                    emulator.memory.write_u32(address_output, VOLATILE_BASE)?;
                    emulator
                        .memory
                        .write_u32(size_output, VOLATILE_SIZE as u32)?;
                    self.volatile_locked = true;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("sceSuspendForUser", 0xa569_e425) => {
                // scekernelvolatilememunlock only takes the lock type in a0.
                // returning an error for an unmatched unlock is important: it
                // catches ownership/state bugs instead of silently hiding them.
                if self.volatile_locked {
                    // wake the longest-waiting *runnable* waiter so a suspended
                    // waiter never inherits the lock while it cannot run.
                    let index = self
                        .volatile_waiters
                        .iter()
                        .position(|waiter| !self.suspended.contains(&waiter.thread));
                    if let Some(index) = index {
                        let waiter = self.volatile_waiters.remove(index).expect("waiter checked");
                        // transfer the lock to the woken thread instead
                        // of freeing it: the woken thread owns volatile from here.
                        // its lock output pointers were saved at wait time because
                        // unlock carries no address arguments.
                        if waiter.address_output != 0 {
                            emulator
                                .memory
                                .write_u32(waiter.address_output, VOLATILE_BASE)?;
                        }
                        if waiter.size_output != 0 {
                            emulator
                                .memory
                                .write_u32(waiter.size_output, VOLATILE_SIZE as u32)?;
                        }
                        self.wake_thread(waiter.thread, 0);
                        // volatile_locked stays true: ownership moved, not freed.
                    } else if self.volatile_waiters.is_empty() {
                        self.volatile_locked = false;
                    }
                    // else: all waiters suspended; keep the lock held until one
                    // resumes (it stays queued and will be woken by a later unlock).
                    emulator.cpu.gpr[2] = 0;
                    self.preempt_if_needed(emulator);
                } else {
                    // sce_kernel_error_power_vmem_not_locked
                    emulator.cpu.gpr[2] = 0x802b_0201;
                }
            }
            ("sceSuspendForUser", 0x3e02_71d3) => {
                // the blocking form waits until volatile memory
                // becomes available instead of reporting in_use like the
                // non-blocking form. queue fifo so an audio/worker thread that
                // arrives while world streaming holds the lock sleeps until the
                // holder unlocks, then receives the base/size and owns it.
                let address_output = emulator.cpu.gpr[5];
                let size_output = emulator.cpu.gpr[6];
                if self.volatile_locked {
                    self.volatile_waiters.push_back(VolatileWaiter {
                        thread: self.current_uid,
                        address_output,
                        size_output,
                    });
                    self.block_current(emulator)?;
                } else {
                    emulator.memory.write_u32(address_output, VOLATILE_BASE)?;
                    emulator
                        .memory
                        .write_u32(size_output, VOLATILE_SIZE as u32)?;
                    self.volatile_locked = true;
                    emulator.cpu.gpr[2] = 0;
                }
            }
            ("IoFileMgrForUser", 0x54f5_fb11) => {
                let device = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])?;
                let command = emulator.cpu.gpr[5];
                let output = emulator.cpu.gpr[8];
                if let Some(value) = devctl_u32_output(&device, command) {
                    // the emulated memory stick is permanently present.  this
                    // devctl returns its fat state through the output pointer;
                    // reporting success without writing it leaves callers
                    // using stale guest memory to select their resource root.
                    if output == 0 {
                        emulator.cpu.gpr[2] = 0x8001_0016;
                    } else {
                        emulator.memory.write_u32(output, value)?;
                        emulator.cpu.gpr[2] = 0;
                    }
                } else {
                    emulator.cpu.gpr[2] = 0x8001_0016;
                }
            }
            ("SysMemUserForUser", 0xa291_f107) => {
                emulator.cpu.gpr[2] = self.user_partition.largest_free()
            }
            ("SysMemUserForUser", 0xf919_f628) => {
                emulator.cpu.gpr[2] = self.user_partition.total_free()
            }
            ("SysMemUserForUser", 0x237d_bd4f) => {
                let partition = emulator.cpu.gpr[4];
                let allocation_type = emulator.cpu.gpr[6];
                let size = emulator.cpu.gpr[7];
                let alignment_or_address = emulator.cpu.gpr[8];
                let name = read_guest_string(&emulator.memory, emulator.cpu.gpr[5])?;
                let (alignment, high) = match allocation_type {
                    0 => (256, false),
                    1 => (256, true),
                    3 => (alignment_or_address.max(256), false),
                    4 => (alignment_or_address.max(256), true),
                    // address allocations need an exact-hole operation.  keep
                    // their failure explicit until a caller requires it.
                    2 => {
                        emulator.cpu.gpr[2] = 0x8002_00d8;
                        return Ok(());
                    }
                    _ => {
                        emulator.cpu.gpr[2] = 0x8002_00d8;
                        return Ok(());
                    }
                };
                if partition != 2 || size == 0 || !alignment.is_power_of_two() {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                } else {
                    self.next_uid += 1;
                    if let Some(address) = self.user_partition.allocate(
                        self.next_uid,
                        &name,
                        "partition",
                        size,
                        alignment,
                        high,
                    ) {
                        self.memory_blocks.insert(self.next_uid, address);
                        emulator.cpu.gpr[2] = self.next_uid;
                    } else {
                        self.log_allocation_failure(
                            import,
                            emulator,
                            AllocationRequest {
                                partition,
                                allocation_type,
                                attributes: 0,
                                size,
                                alignment,
                            },
                        );
                        emulator.cpu.gpr[2] = 0x8002_01b1;
                    }
                }
            }
            ("SysMemUserForUser", 0xb6d6_1d02) => {
                let uid = emulator.cpu.gpr[4];
                emulator.cpu.gpr[2] = if self.memory_blocks.remove(&uid).is_some() {
                    self.user_partition.free_owner(uid);
                    0
                } else {
                    0x8002_00cb
                };
            }
            ("SysMemUserForUser", 0x9d9a_5ba1) => {
                emulator.cpu.gpr[2] = self
                    .memory_blocks
                    .get(&emulator.cpu.gpr[4])
                    .copied()
                    .unwrap_or(0)
            }
            ("SysMemUserForUser", 0x7591_c7db | 0xf77d_77cb) => {
                // scekernelsetcompiledsdkversion and
                // scekernelsetcompilerversion are loader metadata setters.
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0xc07b_b470) => {
                let partition = emulator.cpu.gpr[5];
                let attributes = emulator.cpu.gpr[6];
                let block_size = emulator.cpu.gpr[7];
                // psp hle calls expose parameters beyond a3 in t0-t3.
                let block_count = emulator.cpu.gpr[8];
                let options = emulator.cpu.gpr[9];
                let mut alignment = 4;
                if options != 0 && emulator.memory.read_u32(options)? >= 8 {
                    alignment = emulator.memory.read_u32(options + 4)?.max(4);
                }
                let aligned_size = if alignment.is_power_of_two() {
                    block_size.next_multiple_of(alignment)
                } else {
                    0
                };
                let total_size = aligned_size.checked_mul(block_count).unwrap_or(0);
                tracing::debug!(
                    partition,
                    attributes = format_args!("0x{attributes:08x}"),
                    block_size = format_args!("0x{block_size:08x}"),
                    block_count,
                    options = format_args!("0x{options:08x}"),
                    alignment,
                    total_size = format_args!("0x{total_size:08x}"),
                    "sceKernelCreateFpl parameters"
                );
                self.next_uid += 1;
                let name = read_guest_string(&emulator.memory, emulator.cpu.gpr[4])?;
                if partition != 2 || total_size == 0 {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                } else if let Some(address) = self.user_partition.allocate(
                    self.next_uid,
                    &name,
                    "fpl",
                    total_size,
                    alignment.max(4),
                    attributes & 0x4000 != 0,
                ) {
                    self.fixed_pools.insert(
                        self.next_uid,
                        FixedPool {
                            address,
                            block_size: aligned_size,
                            free: vec![true; block_count as usize],
                        },
                    );
                    emulator.cpu.gpr[2] = self.next_uid;
                } else {
                    self.log_allocation_failure(
                        import,
                        emulator,
                        AllocationRequest {
                            partition,
                            allocation_type: 0,
                            attributes,
                            size: total_size,
                            alignment,
                        },
                    );
                    emulator.cpu.gpr[2] = 0x8002_0190;
                }
            }
            ("ThreadManForUser", 0xd979_e9bf) => {
                let uid = emulator.cpu.gpr[4];
                let output = emulator.cpu.gpr[5];
                let pool = self
                    .fixed_pools
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelAllocateFpl: bad pool {uid}"))?;
                if let Some(index) = pool.free.iter().position(|free| *free) {
                    pool.free[index] = false;
                    let address = pool.address + pool.block_size * index as u32;
                    emulator.memory.write_u32(output, address)?;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = 0x8002_0190;
                    self.block_current(emulator)?;
                }
            }
            ("ThreadManForUser", 0xf641_4a71) => {
                let uid = emulator.cpu.gpr[4];
                let address = emulator.cpu.gpr[5];
                let pool = self
                    .fixed_pools
                    .get_mut(&uid)
                    .with_context(|| format!("sceKernelFreeFpl: bad pool {uid}"))?;
                let offset = address.wrapping_sub(pool.address);
                if offset % pool.block_size == 0
                    && (offset / pool.block_size) < pool.free.len() as u32
                {
                    pool.free[(offset / pool.block_size) as usize] = true;
                    emulator.cpu.gpr[2] = 0;
                } else {
                    emulator.cpu.gpr[2] = 0x8002_00d8;
                }
            }
            ("IoFileMgrForUser", 0xb293_727f) => {
                // async priority is a per-file scheduling hint.  the
                // interpreter has one guest execution lane, so accepting the
                // hint is the observable firmware behavior we need.
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0x64d4_540e | 0x8218_b4dd) => {
                // both profiler queries return success from a short
                // query with no output pointer.  the counters themselves are
                // not observable by this interpreter yet.
                emulator.cpu.gpr[2] = 0;
            }
            ("sceWlanDrv", 0xd776_3699) => {
                // scewlangetswitchstate reports the physical switch.  the
                // linux runner has no psp radio, so the disabled state is 0.
                emulator.cpu.gpr[2] = 0;
            }
            ("UtilsForUser", 0x6ad3_45d7 | 0x79d1_c3fa | 0xb435_dec5) => {
                // gpo configuration and a full data-cache writeback have no
                // additional host-side effect in this memory model.
                emulator.cpu.gpr[2] = 0;
            }
            ("scePower", 0x04b7_766e) => {
                // the callback registration is retained by firmware only;
                // power events are not generated while the host is running
                // the guest in the foreground.
                emulator.cpu.gpr[2] = 0;
            }
            ("LoadExecForUser", 0x4ac5_7943) => {
                // registering an exit callback is harmless until the guest
                // exits; the callback is not needed to boot the title.
                emulator.cpu.gpr[2] = 0;
            }
            ("sceSuspendForUser", 0x090c_cb3f) => {
                // scekernelpowertick only refreshes the firmware power
                // watchdog, which is not modeled by the host runtime.
                emulator.cpu.gpr[2] = 0;
            }
            ("sceUtility", 0xc629_af26 | 0x50c4_cd57 | 0x8874_dbe0) => {
                // av-module loading and the no-save-data savedata state
                // machine are synchronous on this headless-first backend.
                emulator.cpu.gpr[2] = 0;
            }
            ("sceRtc", 0x3f7a_d767) => {
                // tick resolution is 1 mhz, matching the scheduler's
                // microsecond clock.
                let now = emulator.scheduler.now();
                emulator.memory.write_u32(emulator.cpu.gpr[4], now as u32)?;
                emulator
                    .memory
                    .write_u32(emulator.cpu.gpr[4].wrapping_add(4), (now >> 32) as u32)?;
                emulator.cpu.gpr[2] = 0;
            }
            ("sceRtc", 0xc41c_2853) => emulator.cpu.gpr[2] = 1_000_000,
            ("sceRtc", 0x6ff4_0acc | 0x4cfa_57b0) => {
                // scertcgettick / getcurrentclock: report scheduler time so
                // time structs are initialized instead of stale stack.
                let now = emulator.scheduler.now();
                if import.nid == 0x6ff4_0acc {
                    emulator.memory.write_u32(emulator.cpu.gpr[5], now as u32)?;
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[5].wrapping_add(4), (now >> 32) as u32)?;
                } else {
                    emulator.memory.write_u32(emulator.cpu.gpr[4], now as u32)?;
                    emulator
                        .memory
                        .write_u32(emulator.cpu.gpr[4].wrapping_add(4), (now >> 32) as u32)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("ThreadManForUser", 0x30c0_8374) => {
                emulator.cpu.gpr[2] =
                    (emulator.scheduler.now().saturating_mul(333) & 0xffff_ffff) as u32;
            }
            ("scePower", 0x2085_d15d) => emulator.cpu.gpr[2] = 100,
            ("scePower", 0xfee0_3a2f | 0xfdb5_bfe9) => emulator.cpu.gpr[2] = 333,
            ("scePower", 0x2b51_1a2c | 0xd307_c764) => emulator.cpu.gpr[2] = 222,
            ("scePower", 0x8744_0c69 | 0x34f9_c463) => emulator.cpu.gpr[2] = 111,
            // single-threaded mutex/lwmutex: creation hands out an id and
            // every lock/unlock succeeds immediately (this is a
            // cooperative interpreter without true contention).
            ("ThreadManForUser", 0xb7d0_98c6 | 0x19cf_f145 | 0x6010_7536) => {
                self.next_uid += 1;
                emulator.cpu.gpr[2] = self.next_uid;
            }
            (
                "ThreadManForUser",
                0xf817_0fbe | 0xb011_b11f | 0x5bf4_dd27 | 0x6b30_100f | 0x0ddc_d2c9 | 0xa9c2_cb9a
                | 0x1fc6_4e09 | 0x3132_7f19 | 0x3743_1849 | 0x15b6_446b | 0xf35a_f645 | 0xbeed_3a47,
            ) => emulator.cpu.gpr[2] = 0,
            // psmf container queries: report the fixed psp panel geometry
            // and a valid stream offset/size so the cutscene player can
            // advance instead of spinning on uninitialized structs.
            ("scePsmf", 0x0ba5_14e5) => {
                let out = emulator.cpu.gpr[5];
                if out != 0 {
                    emulator.memory.write_u32(out, 480)?;
                    emulator.memory.write_u32(out.wrapping_add(4), 272)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            ("scePsmf", 0xa83f_7113) => {
                let out = emulator.cpu.gpr[5];
                if out != 0 {
                    emulator.memory.write_u32(out, 2)?;
                    emulator.memory.write_u32(out.wrapping_add(4), 44_100)?;
                }
                emulator.cpu.gpr[2] = 0;
            }
            (
                "scePsmf",
                0x5b70_fcc1 | 0x9553_cc91 | 0xc22c_8327 | 0x4bc9_bde0 | 0x1e6d_9013 | 0x0c12_0e1d
                | 0x2824_0568 | 0x2673_646b | 0x43ac_7dbb,
            ) => {
                // querystreamoffset/size, setpsmf, specifystream, verifypsmf:
                // write a plausible offset/size pair when pointers are given.
                if emulator.cpu.gpr[5] != 0 {
                    let _ = emulator.memory.write_u32(emulator.cpu.gpr[5], 2_048);
                }
                if emulator.cpu.gpr[6] != 0 {
                    let _ = emulator.memory.write_u32(emulator.cpu.gpr[6], 2_048);
                }
                emulator.cpu.gpr[2] = 0;
            }
            // calls not yet requiring observable state return success. keeping
            // imports linked is important: execution now reaches the hle
            // boundary and each required behavior can be implemented by nid.
            _ => {
                if self
                    .unknown_imports
                    .insert((import.library.clone(), import.nid))
                {
                    tracing::warn!(
                        library = %import.library,
                        nid = format_args!("0x{:08x}", import.nid),
                        "unimplemented HLE import returning success"
                    );
                }
                emulator.cpu.gpr[2] = 0;
            }
        }
        Ok(())
    }

    fn run_sas_mix(
        &mut self,
        emulator: &mut psp_core::Emulator,
        output_address: u32,
        input_address: Option<u32>,
        left_volume: i32,
        right_volume: i32,
    ) -> Result<()> {
        if !self.sas.is_initialized() {
            emulator.cpu.gpr[2] = SAS_ERROR_NOT_INIT;
            return Ok(());
        }
        if input_address.is_some() && self.sas.output_mode() != 0 {
            emulator.cpu.gpr[2] = SAS_ERROR_INVALID_PARAMETER;
            return Ok(());
        }
        let mix_delay = self.sas.estimate_mix_us();
        let mix_started = timer_start();
        let pcm = match self.sas.mix(
            &mut emulator.memory,
            output_address,
            input_address,
            left_volume,
            right_volume,
        ) {
            Ok(pcm) => pcm,
            Err(_) => {
                // report a bad sas buffer to the guest. do not turn
                // an ordinary bad title pointer into a host-side emulator
                // fault and terminate the whole game.
                emulator.cpu.gpr[2] = SAS_ERROR_INVALID_PARAMETER;
                return Ok(());
            }
        };
        let mix_ms = timer_elapsed(mix_started).as_secs_f64() * 1_000.0;
        if mix_ms >= 1.0 {
            tracing::debug!(
                target: "psp_rs::audio_timing",
                mix_ms,
                grain = self.sas.grain_size(),
                "slow SAS mix"
            );
        }
        emulator.audio.submit_pcm_from("sas", &pcm);
        emulator.cpu.gpr[2] = 0;
        self.wait_current_for(emulator, mix_delay.max(1))
    }

    fn yield_current(&mut self, emulator: &mut psp_core::Emulator) {
        if self.current_uid != 0
            && let Some(thread) = self.threads.get_mut(&self.current_uid)
        {
            thread.cpu = Some(emulator.cpu.clone());
            self.ready.push_back(self.current_uid);
        }
        self.switch_next(emulator);
    }

    fn infer_display_from_ge(&mut self, gpu: &psp_gpu::Gpu) {
        if self.display.explicit_frame_buffer {
            return;
        }
        let framebuffer = gpu.registers[0x9c];
        let buffer_width = gpu.registers[0x9d] & 0x7fc;
        if framebuffer == 0 || buffer_width == 0 {
            return;
        }
        self.display.frame_buffer = if framebuffer < EDRAM_SIZE as u32 {
            EDRAM_BASE | framebuffer
        } else {
            framebuffer
        };
        self.display.width = 480;
        self.display.height = 272;
        self.display.buffer_width = buffer_width;
        self.display.pixel_format = gpu.registers[0xd2] & 3;
    }

    #[cfg(feature = "desktop")]
    fn present(&mut self, emulator: &mut psp_core::Emulator, scanout: bool) -> Result<()> {
        let Some(window) = self.window.as_mut() else {
            return Ok(());
        };
        if !window.is_open() {
            return Ok(());
        }
        let render_commands = emulator.gpu.take_render_commands();
        if !render_commands.is_empty() {
            self.pending_render_commands.extend(render_commands);
        }
        if !scanout {
            return Ok(());
        }
        if !self.pending_render_commands.is_empty() {
            window.render_commands(&self.pending_render_commands)?;
            self.pending_render_commands.clear();
            self.hardware_frame_valid = true;
        }
        if self.display.frame_buffer == 0 || self.display.width == 0 || self.display.height == 0 {
            return Ok(());
        }
        // the guest can call scedisplaywaitvblankstart much faster than a host
        // window can present.  host events are pumped by poll_window; only
        // convert and upload the scanout at the display refresh cadence here.
        let now = Instant::now();
        if self
            .last_host_present
            .is_some_and(|last| now.duration_since(last) < HOST_FRAME_INTERVAL)
        {
            return Ok(());
        }
        // in hardware mode the ge never writes guest ram, so a stale-ram
        // scanout would show black after every double-buffered flip (the
        // typical gta frame). always resolve the persistent gpu target
        // first; only fall back to ram for cpu-rendered frames.
        if window.present_gpu(Some(self.display.frame_buffer))? {
            if self.hardware_frame_valid {
                self.last_host_present = Some(now);
                return Ok(());
            }
            // the target exists but no fresh ge commands were submitted
            // this tick (e.g. a pure display flip). still scan it out:
            // retaining the previous frame_texture would freeze the image.
            self.last_host_present = Some(now);
            return Ok(());
        }
        // a display switch can point at a guest framebuffer that has not
        // been represented by an sdl target yet. do not leave the prior
        // ge surface on screen while that buffer is being populated.
        self.hardware_frame_valid = false;
        let rgba = psp_gpu::Gpu::framebuffer_rgba(
            &emulator.memory,
            self.display.frame_buffer,
            self.display.buffer_width,
            self.display.width,
            self.display.height,
            self.display.pixel_format,
        )
        .with_context(|| {
            format!(
                "cannot scan out framebuffer 0x{:08x} {}x{} stride={} format={}",
                self.display.frame_buffer,
                self.display.width,
                self.display.height,
                self.display.buffer_width,
                self.display.pixel_format
            )
        })?;
        window.present(&rgba)?;
        self.last_host_present = Some(now);
        Ok(())
    }

    #[cfg(not(feature = "desktop"))]
    fn present(&mut self, _emulator: &mut psp_core::Emulator, _scanout: bool) -> Result<()> {
        Ok(())
    }

    /// Keep windowed execution on the PSP's real 60 Hz display clock.
    ///
    /// Guest vblank waits advance scheduler time immediately, which is useful
    /// for headless diagnostics but otherwise lets display-paced content such
    /// as PMF/MPEG movies run as fast as the host CPU.  Sleeping here makes a
    /// graphical run observe the same wall-clock cadence as the hardware.
    #[cfg(feature = "desktop")]
    fn pace_host_vblank(&mut self) {
        if self.window.is_none() {
            return;
        }

        let now = Instant::now();
        let deadline = self
            .next_host_vblank
            .unwrap_or_else(|| now + HOST_FRAME_INTERVAL);
        if now < deadline {
            thread::sleep(deadline.duration_since(now));
        }

        let after_wait = Instant::now();
        self.next_host_vblank = Some(if after_wait < deadline + HOST_FRAME_INTERVAL {
            deadline + HOST_FRAME_INTERVAL
        } else {
            after_wait + HOST_FRAME_INTERVAL
        });
    }

    #[cfg(not(feature = "desktop"))]
    fn pace_host_vblank(&mut self) {}

    #[cfg(feature = "desktop")]
    fn poll_window(&mut self, input: &mut InputState) -> bool {
        let Some(window) = self.window.as_mut() else {
            return true;
        };
        if !window.is_open() {
            return false;
        }
        let buttons = window.poll(input);
        if buttons != self.last_host_buttons {
            tracing::debug!(
                old = format_args!("0x{:08x}", self.last_host_buttons),
                new = format_args!("0x{buttons:08x}"),
                "host controller state changed"
            );
            self.last_host_buttons = buttons;
        }
        input.timestamp_us = self.host_started_at.elapsed().as_micros() as u64;
        window.is_open()
    }

    fn block_current(&mut self, emulator: &mut psp_core::Emulator) -> Result<()> {
        let uid = self.current_uid;
        if matches!(uid, 4 | 47) {
            tracing::debug!(
                target: "psp_rs::sched_timing",
                uid,
                scheduler = emulator.scheduler.now(),
                ready = ?self.ready,
                timed_waiters = ?self.timed_waiters,
                "guest thread blocked"
            );
        }
        let thread = self
            .threads
            .get_mut(&uid)
            .with_context(|| format!("cannot block unregistered thread {uid}"))?;
        thread.cpu = Some(emulator.cpu.clone());
        self.current_uid = 0;
        anyhow::ensure!(
            self.switch_next_or_advance(emulator),
            "guest deadlock: all threads are waiting on synchronization objects"
        );
        Ok(())
    }

    /// Block the current thread in a synchronization wait with a guest
    /// timeout. A null timeout waits forever; a zeroed clock reports
    /// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` without blocking; otherwise the
    /// thread sleeps until the setter wakes it or the deadline expires,
    /// whichever comes first.
    fn block_current_with_timeout(
        &mut self,
        emulator: &mut psp_core::Emulator,
        timeout_us: Option<u64>,
        what: &str,
    ) -> Result<()> {
        let Some(micros) = timeout_us else {
            return self.block_current(emulator);
        };
        if micros == 0 {
            emulator.cpu.gpr[2] = KERNEL_ERROR_WAIT_TIMEOUT;
            return Ok(());
        }
        let uid = self.current_uid;
        let thread = self
            .threads
            .get_mut(&uid)
            .with_context(|| format!("cannot time-wait unregistered thread {uid}"))?;
        thread.cpu = Some(emulator.cpu.clone());
        self.timed_waiters
            .insert(uid, emulator.scheduler.now().saturating_add(micros.max(1)));
        self.timed_wait_results
            .insert(uid, KERNEL_ERROR_WAIT_TIMEOUT);
        tracing::debug!(
            uid,
            timeout_us = micros,
            what,
            "guest thread blocked with timeout"
        );
        self.current_uid = 0;
        anyhow::ensure!(
            self.switch_next_or_advance(emulator),
            "guest deadlock: no runnable thread during timed wait"
        );
        Ok(())
    }

    fn wait_current_for(&mut self, emulator: &mut psp_core::Emulator, ticks: u64) -> Result<()> {
        let uid = self.current_uid;
        if ticks >= 5_000 && matches!(uid, 4 | 47) {
            tracing::debug!(
                target: "psp_rs::sched_timing",
                uid,
                ticks,
                scheduler = emulator.scheduler.now(),
                ready = ?self.ready,
                timed_waiters = ?self.timed_waiters,
                "guest timed wait"
            );
        }
        let thread = self
            .threads
            .get_mut(&uid)
            .with_context(|| format!("cannot time-wait unregistered thread {uid}"))?;
        thread.cpu = Some(emulator.cpu.clone());
        self.timed_waiters
            .insert(uid, emulator.scheduler.now().saturating_add(ticks.max(1)));
        self.current_uid = 0;
        anyhow::ensure!(
            self.switch_next_or_advance(emulator),
            "guest deadlock: no runnable thread during timed wait"
        );
        Ok(())
    }

    fn wake_timed_waiters(&mut self, emulator: &mut psp_core::Emulator) {
        self.audio.advance_to(emulator.scheduler.now());
        if self.timed_waiters.is_empty() {
            return;
        }
        let now = emulator.scheduler.now();
        let mut awakened = Vec::new();
        self.timed_waiters.retain(|&uid, wake_tick| {
            if *wake_tick <= now {
                awakened.push(uid);
                false
            } else {
                true
            }
        });
        // sort by uid: `timed_waiters` is a hash map, and without this the
        // ready-queue order would depend on hash iteration order (and, under
        // the block jit, on how many waiters become due within one compiled
        // block versus separate interpreter steps).
        awakened.sort_unstable();
        for uid in awakened.iter().copied() {
            // a synchronisation wait whose deadline expired reports its
            // timeout result; plain delays preset v0 before sleeping.
            if let Some(result) = self.timed_wait_results.remove(&uid)
                && let Some(cpu) = self
                    .threads
                    .get_mut(&uid)
                    .and_then(|thread| thread.cpu.as_mut())
            {
                cpu.gpr[2] = result;
            }
            if !self.suspended.contains(&uid) && !self.ready.contains(&uid) {
                self.ready.push_back(uid);
            }
        }
        if awakened.iter().any(|uid| matches!(uid, 4 | 47)) {
            tracing::debug!(
                target: "psp_rs::sched_timing",
                scheduler = now,
                current = self.current_uid,
                awakened = ?awakened,
                ready = ?self.ready,
                "guest timed waiters woke"
            );
        }
        if !awakened.is_empty() && self.current_uid != 0 {
            self.preempt_if_needed(emulator);
        }
    }

    fn wake_thread(&mut self, uid: u32, result: u32) {
        // an explicit wake always cancels a pending wait deadline: the
        // synchronisation result wins over a later timeout expiry.
        self.timed_waiters.remove(&uid);
        self.timed_wait_results.remove(&uid);
        if let Some(cpu) = self
            .threads
            .get_mut(&uid)
            .and_then(|thread| thread.cpu.as_mut())
        {
            cpu.gpr[2] = result;
            if !self.suspended.contains(&uid) && !self.ready.contains(&uid) {
                self.ready.push_back(uid);
            }
        }
    }

    fn remove_thread(&mut self, uid: u32) -> bool {
        if self.threads.remove(&uid).is_none() {
            return false;
        }
        self.ready.retain(|thread| *thread != uid);
        self.suspended.remove(&uid);
        self.timed_waiters.remove(&uid);
        self.timed_wait_results.remove(&uid);
        for semaphore in self.semaphores.values_mut() {
            semaphore.waiters.retain(|waiter| waiter.thread != uid);
        }
        for flag in self.event_flags.values_mut() {
            flag.waiters.retain(|waiter| waiter.thread != uid);
        }
        self.volatile_waiters.retain(|waiter| waiter.thread != uid);
        self.user_partition.free_owner(uid);
        true
    }

    fn finish_current_and_switch(&mut self, emulator: &mut psp_core::Emulator) -> bool {
        if self.current_uid != 0 {
            let uid = self.current_uid;
            let entry = self.threads.get(&uid).map(|thread| thread.entry);
            self.remove_thread(uid);
            tracing::debug!(uid, entry = ?entry, "guest thread exited");
        }
        self.switch_next_or_advance(emulator)
    }

    fn switch_next_or_advance(&mut self, emulator: &mut psp_core::Emulator) -> bool {
        if self.switch_next(emulator) {
            return true;
        }
        let now = emulator.scheduler.now();
        let Some(next_wake) = self.timed_waiters.values().copied().min() else {
            return false;
        };
        if self.timed_waiters.keys().any(|uid| matches!(uid, 4 | 47)) {
            tracing::debug!(
                target: "psp_rs::sched_timing",
                scheduler = now,
                next_wake,
                delta = next_wake.saturating_sub(now),
                timed_waiters = ?self.timed_waiters,
                "scheduler advancing to timed waiter"
            );
        }
        emulator
            .scheduler
            .advance_micros(next_wake.saturating_sub(now));
        self.wake_timed_waiters(emulator);
        self.switch_next(emulator)
    }

    fn switch_next(&mut self, emulator: &mut psp_core::Emulator) -> bool {
        while !self.ready.is_empty() {
            let index = self
                .ready
                .iter()
                .enumerate()
                .min_by_key(|(_, uid)| {
                    self.threads
                        .get(uid)
                        .map_or(u32::MAX, |thread| thread.priority)
                })
                .map(|(index, _)| index)
                .unwrap();
            let uid = self.ready.remove(index).unwrap();
            if let Some(cpu) = self
                .threads
                .get_mut(&uid)
                .and_then(|thread| thread.cpu.take())
            {
                self.current_uid = uid;
                emulator.cpu = cpu;
                return true;
            }
        }
        self.current_uid = 0;
        false
    }

    fn preempt_if_needed(&mut self, emulator: &mut psp_core::Emulator) {
        let current_priority = self
            .threads
            .get(&self.current_uid)
            .map_or(u32::MAX, |thread| thread.priority);
        let ready_priority = self
            .ready
            .iter()
            .filter_map(|uid| self.threads.get(uid).map(|thread| thread.priority))
            .min()
            .unwrap_or(u32::MAX);
        if ready_priority < current_priority {
            self.yield_current(emulator);
        }
    }

    fn log_allocation_failure(
        &self,
        import: &psp_loader::Import,
        emulator: &psp_core::Emulator,
        request: AllocationRequest,
    ) {
        tracing::error!(
            instructions = emulator.cpu.instruction_count,
            pc = format_args!("0x{:08x}", emulator.cpu.pc),
            ra = format_args!("0x{:08x}", emulator.cpu.gpr[31]),
            gp = format_args!("0x{:08x}", emulator.cpu.gpr[28]),
            thread = self.current_uid,
            module = "main",
            library = %import.library,
            nid = format_args!("0x{:08x}", import.nid),
            partition = request.partition,
            allocation_type = request.allocation_type,
            attributes = format_args!("0x{:08x}", request.attributes),
            requested_size = format_args!("0x{:08x}", request.size),
            alignment = request.alignment,
            total_free = self.user_partition.total_free(),
            largest_free = self.user_partition.largest_free(),
            blocks = %self.user_partition.block_map(),
            "guest allocation failed"
        );
    }

    fn trigger_subinterrupt(
        &mut self,
        interrupt: u32,
        subinterrupt: u32,
        emulator: &mut psp_core::Emulator,
    ) -> bool {
        let Some(handler) = self.subinterrupts.get(&(interrupt, subinterrupt)) else {
            return false;
        };
        if !handler.enabled || handler.handler == 0 {
            return false;
        }
        let handler_address = handler.handler;
        let argument = handler.argument;
        self.interrupt_stack.push(emulator.cpu.clone());
        emulator.cpu.pc = handler_address;
        emulator.cpu.gpr[31] = 0;
        emulator.cpu.gpr[4] = subinterrupt;
        emulator.cpu.gpr[5] = argument;
        tracing::debug!(
            interrupt,
            subinterrupt,
            handler = format_args!("0x{handler_address:08x}"),
            "guest subinterrupt entered"
        );
        true
    }

    /// Queue the guest GE signal and finish handlers for a finished list.
    ///
    /// A PSP display list raises one GE interrupt per signal command plus one
    /// for its finish completion, and each runs the handler registered with
    /// scegesetcallback before the interrupted thread resumes.  games rely on
    /// those handlers to advance streaming and rendering handshakes, so every
    /// recorded event must be delivered even when several fire in one list.
    fn deliver_ge_events(
        &mut self,
        emulator: &mut psp_core::Emulator,
        callback_id: u32,
        result: &psp_gpu::ListResult,
    ) {
        let Some(callback) = self.ge_callbacks.get(&callback_id).copied() else {
            return;
        };
        if result.signals.is_empty() && !(result.finished && callback.finish_function != 0) {
            return;
        }
        let saved = emulator.cpu.clone();
        for signal in &result.signals {
            // 0x01 suspends until the handler returns, 0x02 continues right
            // away, 0x03 pauses at finish; all three run the registered
            // signal handler.  the
            // remaining forms are synchronization or flow-control only.
            if !matches!(signal.behavior, 0x01..=0x03) || callback._signal_function == 0 {
                continue;
            }
            self.interrupt_stack.push(saved.clone());
            self.pending_interrupts.push_back(PendingInterrupt {
                handler: callback._signal_function,
                argument: callback._signal_argument,
                token: signal.token,
            });
        }
        if result.finished && callback.finish_function != 0 {
            self.interrupt_stack.push(saved);
            self.pending_interrupts.push_back(PendingInterrupt {
                handler: callback.finish_function,
                argument: callback.finish_argument,
                token: result.finish_token,
            });
        }
        self.enter_pending_interrupt(emulator);
    }

    fn enter_pending_interrupt(&mut self, emulator: &mut psp_core::Emulator) {
        let Some(pending) = self.pending_interrupts.pop_front() else {
            return;
        };
        emulator.cpu.pc = pending.handler;
        emulator.cpu.gpr[31] = 0;
        emulator.cpu.gpr[4] = pending.token;
        emulator.cpu.gpr[5] = pending.argument;
        tracing::debug!(
            handler = format_args!("0x{:08x}", pending.handler),
            token = pending.token,
            "guest interrupt handler entered"
        );
    }

    fn return_from_interrupt(&mut self, emulator: &mut psp_core::Emulator) -> Result<bool> {
        if let Some(pending) = self.pending_ringbuffer_put.take() {
            let returned = emulator.cpu.gpr[2] as i32;
            let added = u32::try_from(returned)
                .unwrap_or(0)
                .min(pending.current_packets);
            tracing::debug!(
                requested = pending.current_packets,
                returned,
                added,
                ring = format_args!("0x{:08x}", pending.ringbuffer_addr),
                "MPEG ringbuffer callback completed"
            );
            if added != 0 {
                let ring = pending.ringbuffer_addr;
                let packets =
                    read_mpeg_packets(&emulator.memory, ring, pending.write_position, added)?;
                if pending.write_position == 0 {
                    let prefix = packets
                        .iter()
                        .take(16)
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>();
                    tracing::debug!(%prefix, "captured first MPEG ring packet");
                }
                self.mpeg.media.submit(&packets);
                add_mpeg_packets(&mut emulator.memory, ring, added)?;
            }

            let total_added = pending.added_packets.saturating_add(added);
            if added != 0 && total_added < pending.target_packets {
                let ring = pending.ringbuffer_addr;
                let packets = emulator.memory.read_u32(ring + MPEG_RING_PACKETS)?;
                let callback = emulator.memory.read_u32(ring + MPEG_RING_CALLBACK)?;
                if packets != 0 && callback != 0 {
                    let write_position =
                        emulator.memory.read_u32(ring + MPEG_RING_PACKETS_WRITTEN)?;
                    let remaining = pending.target_packets - total_added;
                    let sequential_packets = packets - (write_position % packets);
                    let requested = remaining.min(sequential_packets);
                    if requested != 0 {
                        let packet_size = emulator
                            .memory
                            .read_u32(ring + MPEG_RING_PACKET_SIZE)?
                            .max(MPEG_PACKET_SIZE);
                        let data = emulator.memory.read_u32(ring + MPEG_RING_DATA)?;
                        let buffer =
                            data.wrapping_add((write_position % packets).wrapping_mul(packet_size));
                        let callback_args =
                            emulator.memory.read_u32(ring + MPEG_RING_CALLBACK_PARAM)?;
                        self.pending_ringbuffer_put = Some(PendingRingbufferPut {
                            ringbuffer_addr: ring,
                            write_position,
                            target_packets: pending.target_packets,
                            current_packets: requested,
                            added_packets: total_added,
                        });
                        emulator.cpu.pc = callback;
                        emulator.cpu.gpr[4] = buffer;
                        emulator.cpu.gpr[5] = requested;
                        emulator.cpu.gpr[6] = callback_args;
                        emulator.cpu.gpr[31] = 0;
                        emulator.cpu.gpr[2] = 0;
                        return Ok(true);
                    }
                }
            }

            if let Some(cpu) = self.interrupt_stack.last_mut() {
                cpu.gpr[2] = if returned < 0 {
                    returned as u32
                } else {
                    total_added
                };
            }
        }
        // more ge handlers may be chained behind the one that just ran; each
        // must finish before control returns to the interrupted thread.
        if !self.pending_interrupts.is_empty() {
            self.enter_pending_interrupt(emulator);
            return Ok(true);
        }
        if let Some(cpu) = self.interrupt_stack.pop() {
            emulator.cpu = cpu;
            if self.vblank_wait_pending {
                self.vblank_wait_pending = false;
                self.wait_current_for(emulator, VBLANK_PERIOD_US)?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl PartitionAllocator {
    fn new(base: u32, end: u32) -> Self {
        Self {
            base,
            end,
            blocks: std::collections::BTreeMap::new(),
        }
    }

    fn free_ranges(&self) -> Vec<(u32, u32)> {
        let mut ranges = Vec::new();
        let mut cursor = self.base;
        for (&address, block) in &self.blocks {
            if cursor < address {
                ranges.push((cursor, address));
            }
            cursor = cursor.max(address.saturating_add(block.size));
        }
        if cursor < self.end {
            ranges.push((cursor, self.end));
        }
        ranges
    }

    fn allocate(
        &mut self,
        owner: u32,
        name: &str,
        kind: &'static str,
        size: u32,
        alignment: u32,
        high: bool,
    ) -> Option<u32> {
        if size == 0 || !alignment.is_power_of_two() {
            return None;
        }
        let reserved_size = size.next_multiple_of(256);
        let ranges = self.free_ranges();
        let selected = if high {
            ranges.into_iter().rev().find_map(|(start, end)| {
                let unaligned = end.checked_sub(reserved_size)?;
                let address = unaligned & !(alignment - 1);
                (address >= start).then_some(address)
            })
        } else {
            ranges.into_iter().find_map(|(start, end)| {
                let address = start.checked_add(alignment - 1)? & !(alignment - 1);
                address
                    .checked_add(reserved_size)
                    .is_some_and(|block_end| block_end <= end)
                    .then_some(address)
            })
        }?;
        self.blocks.insert(
            selected,
            PartitionBlock {
                size: reserved_size,
                owner,
                name: name.to_owned(),
                kind,
            },
        );
        Some(selected)
    }

    fn free_owner(&mut self, owner: u32) {
        self.blocks.retain(|_, block| block.owner != owner);
    }

    fn total_free(&self) -> u32 {
        self.free_ranges()
            .into_iter()
            .map(|(start, end)| end - start)
            .sum()
    }

    fn largest_free(&self) -> u32 {
        self.free_ranges()
            .into_iter()
            .map(|(start, end)| end - start)
            .max()
            .unwrap_or(0)
    }

    fn block_map(&self) -> String {
        if self.blocks.is_empty() {
            return "<empty>".to_owned();
        }
        self.blocks
            .iter()
            .map(|(address, block)| {
                format!(
                    "[0x{address:08x}..0x{:08x}) owner={} kind={} name={:?}",
                    address + block.size,
                    block.owner,
                    block.kind,
                    block.name
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn mapped_image_end(info: &psp_loader::ImageInfo) -> Result<u32> {
    let psp_loader::ImageInfo::Elf { segments, .. } = info else {
        anyhow::bail!("decoded boot executable is not an ELF")
    };
    let relative = segments
        .iter()
        .all(|segment| segment.address < USER_MODULE_BASE);
    segments
        .iter()
        .filter_map(|segment| {
            segment
                .address
                .checked_add(if relative { USER_MODULE_BASE } else { 0 })
                .and_then(|address| address.checked_add(segment.memory_size))
        })
        .max()
        .context("ELF contains no mapped segments")
}

fn event_flag_matches(current: u32, wanted: u32, mode: u32) -> bool {
    if mode & 1 != 0 {
        current & wanted != 0
    } else {
        current & wanted == wanted
    }
}

fn apply_event_flag_clear(current: &mut u32, wanted: u32, mode: u32) {
    if mode & 0x10 != 0 {
        *current = 0;
    } else if mode & 0x20 != 0 {
        *current &= !wanted;
    }
}

fn devctl_u32_output(device: &str, command: u32) -> Option<u32> {
    (device.eq_ignore_ascii_case("fatms0:") && command == 0x0242_5823).then_some(1)
}

fn system_param_int(id: u32) -> Option<u32> {
    // keep the host profile deterministic: english ui, automatic ad-hoc
    // channel, 24-hour clock, and no daylight-saving adjustment.  these are
    // the values exposed by the psp system-parameter ids used by games.
    match id {
        2 | 3 | 4 | 5 | 6 | 7 | 9 => Some(0),
        8 => Some(1),
        _ => None,
    }
}

fn split_system_clock(clock: u64) -> (u32, u32) {
    ((clock / 1_000_000) as u32, (clock % 1_000_000) as u32)
}

/// Thread filter for syscall tracing. When `PSP_TRACE_THREAD` names one or more
/// guest thread IDs (comma-separated), only those threads' HLE calls are
/// logged, which keeps traces of single-threaded loops small on long runs.
#[cfg(feature = "desktop")]
fn trace_thread_matches(uid: u32) -> bool {
    static FILTER: std::sync::OnceLock<Option<Vec<u32>>> = std::sync::OnceLock::new();
    FILTER
        .get_or_init(|| {
            std::env::var("PSP_TRACE_THREAD").ok().map(|text| {
                text.split(',')
                    .filter_map(|part| part.trim().parse().ok())
                    .collect()
            })
        })
        .as_ref()
        .is_none_or(|filtered| filtered.contains(&uid))
}

/// Display refresh period in microseconds (60 Hz). The vblank counter is a
/// hardware clock that advances with emulated time, not a tally of
/// `sceDisplayWaitVblank` calls. Counting waits instead would freeze titles
/// that pace streaming, audio, or loading progress from `getVcount` and
/// `getAccumulatedHcount` without waiting every tick.
const VBLANK_PERIOD_US: u64 = 16_667;
/// Scanlines per PSP frame: 272 visible lines plus vertical blanking.
const SCANLINES_PER_FRAME: u64 = 286;

fn display_vcount(now_us: u64) -> u32 {
    (now_us / VBLANK_PERIOD_US) as u32
}

fn display_hcount(now_us: u64) -> u32 {
    (now_us.saturating_mul(SCANLINES_PER_FRAME) / VBLANK_PERIOD_US) as u32
}

/// Read the timeout of a blocking ThreadMan wait.
///
/// hle arguments arrive in
/// a0-a3/t0-t3 (gpr[4-11]), so a timeout pointer is passed directly
/// (waitsema in a2, waiteventflag in t0). it points at a microsecond
/// `SceKernelSysClock`.  a null pointer means "wait forever"; a zeroed clock
/// means "do not wait".  expiry reports
/// `SCE_KERNEL_ERROR_WAIT_TIMEOUT`.
fn read_timeout_us_at(memory: &psp_memory::Memory, pointer: u32) -> Result<Option<u64>> {
    if pointer == 0 {
        return Ok(None);
    }
    // a handful of titles pass small sentinel values instead of a clock
    // pointer.  never let a dubious timeout kill the run: fall back to an
    // infinite wait and leave a trace for diagnosis.
    if pointer & 3 != 0 {
        tracing::debug!(
            pointer = format_args!("0x{pointer:08x}"),
            "wait timeout pointer is not a clock; waiting forever"
        );
        return Ok(None);
    }
    let clock = memory.read_u32(pointer).and_then(|low| {
        memory
            .read_u32(pointer.wrapping_add(4))
            .map(|high| (u64::from(high) << 32) | u64::from(low))
    });
    match clock {
        Ok(micros) => Ok(Some(micros)),
        Err(error) => {
            tracing::debug!(
                pointer = format_args!("0x{pointer:08x}"),
                ?error,
                "wait timeout clock unreadable; waiting forever"
            );
            Ok(None)
        }
    }
}

fn initialize_mpeg_ringbuffer(
    memory: &mut psp_memory::Memory,
    address: u32,
    packets: u32,
    data: u32,
    callback: u32,
    callback_param: u32,
) -> Result<()> {
    let fields = [
        (MPEG_RING_PACKETS, packets),
        (MPEG_RING_PACKETS_READ, 0),
        (MPEG_RING_PACKETS_WRITTEN, 0),
        (MPEG_RING_PACKETS_IN_BUFFER, 0),
        (MPEG_RING_PACKET_SIZE, MPEG_PACKET_SIZE),
        (MPEG_RING_DATA, data),
        (MPEG_RING_CALLBACK, callback),
        (MPEG_RING_CALLBACK_PARAM, callback_param),
        (
            MPEG_RING_DATA_END,
            data.wrapping_add(packets.wrapping_mul(MPEG_PACKET_SIZE)),
        ),
        (MPEG_RING_SEMAPHORE, 0),
        (MPEG_RING_MPEG, 0),
    ];
    debug_assert_eq!(fields.len() as u32, MPEG_RING_FIELD_COUNT);
    for (offset, value) in fields {
        memory.write_u32(address + offset, value)?;
    }
    Ok(())
}

fn mpeg_ringbuffer(memory: &psp_memory::Memory, mpeg: u32) -> Result<u32> {
    let handle = memory.read_u32(mpeg)?;
    memory
        .read_u32(handle + MPEG_HANDLE_RING)
        .context("MPEG handle has no ringbuffer")
}

fn add_mpeg_packets(memory: &mut psp_memory::Memory, ring: u32, packet_count: u32) -> Result<()> {
    let packets = memory.read_u32(ring + MPEG_RING_PACKETS)?;
    let packets_read = memory.read_u32(ring + MPEG_RING_PACKETS_READ)?;
    let packets_written = memory.read_u32(ring + MPEG_RING_PACKETS_WRITTEN)?;
    let packets_in_buffer = memory.read_u32(ring + MPEG_RING_PACKETS_IN_BUFFER)?;
    let accepted = packet_count.min(packets.saturating_sub(packets_in_buffer));
    memory.write_u32(
        ring + MPEG_RING_PACKETS_READ,
        packets_read.wrapping_add(accepted),
    )?;
    memory.write_u32(
        ring + MPEG_RING_PACKETS_WRITTEN,
        packets_written.wrapping_add(accepted),
    )?;
    memory.write_u32(
        ring + MPEG_RING_PACKETS_IN_BUFFER,
        packets_in_buffer.saturating_add(accepted).min(packets),
    )?;
    Ok(())
}

fn consume_mpeg_packet(memory: &mut psp_memory::Memory, ring: u32) -> Result<bool> {
    let packets_in_buffer = memory.read_u32(ring + MPEG_RING_PACKETS_IN_BUFFER)?;
    if packets_in_buffer == 0 {
        return Ok(false);
    }
    memory.write_u32(ring + MPEG_RING_PACKETS_IN_BUFFER, packets_in_buffer - 1)?;
    Ok(true)
}

fn mpeg_ring_has_data(memory: &psp_memory::Memory, ring: u32) -> Result<bool> {
    Ok(memory.read_u32(ring + MPEG_RING_PACKETS_IN_BUFFER)? != 0)
}

fn read_mpeg_packets(
    memory: &psp_memory::Memory,
    ring: u32,
    start_position: u32,
    packet_count: u32,
) -> Result<Vec<u8>> {
    let packets = memory.read_u32(ring + MPEG_RING_PACKETS)?;
    anyhow::ensure!(packets != 0, "MPEG ringbuffer has no packet slots");
    let packet_size = memory
        .read_u32(ring + MPEG_RING_PACKET_SIZE)?
        .max(MPEG_PACKET_SIZE);
    let packet_size = usize::try_from(packet_size).context("MPEG packet size overflows usize")?;
    let packet_count =
        usize::try_from(packet_count).context("MPEG packet count overflows usize")?;
    let capacity = packet_count
        .checked_mul(packet_size)
        .context("MPEG packet capture size overflows usize")?;
    let data = memory.read_u32(ring + MPEG_RING_DATA)?;
    let mut output = Vec::with_capacity(capacity);
    for index in 0..packet_count {
        let slot = start_position.wrapping_add(index as u32) % packets;
        let address = data.wrapping_add(slot.wrapping_mul(packet_size as u32));
        output.extend_from_slice(&memory.read_bytes(address, packet_size)?);
    }
    Ok(output)
}

fn write_mpeg_video_frame(
    memory: &mut psp_memory::Memory,
    address: u32,
    frame_width: u32,
    pixel_format: u32,
    frame: &[u8],
) -> Result<()> {
    let source_row_bytes = VIDEO_WIDTH
        .checked_mul(4)
        .context("MPEG source row size overflows usize")?;
    let source_size = source_row_bytes
        .checked_mul(VIDEO_HEIGHT)
        .context("MPEG source frame size overflows usize")?;
    anyhow::ensure!(
        frame.len() >= source_size,
        "decoded MPEG video frame is too short: {} < {source_size}",
        frame.len()
    );
    let pixel_format = if pixel_format <= 3 { pixel_format } else { 3 };
    let bytes_per_pixel = if pixel_format == 3 { 4 } else { 2 };
    let stride = usize::try_from(frame_width).context("MPEG frame width overflows usize")?;
    let row_bytes = stride
        .checked_mul(bytes_per_pixel)
        .context("MPEG output row size overflows usize")?;
    let image_width = stride.min(VIDEO_WIDTH);
    for y in 0..VIDEO_HEIGHT {
        let mut row = vec![0; row_bytes];
        for x in 0..image_width {
            let source = y * source_row_bytes + x * 4;
            let red = frame[source];
            let green = frame[source + 1];
            let blue = frame[source + 2];
            let alpha = frame[source + 3];
            let destination = x * bytes_per_pixel;
            match pixel_format {
                0 => row[destination..destination + 2].copy_from_slice(
                    &(u16::from(red >> 3)
                        | (u16::from(green >> 2) << 5)
                        | (u16::from(blue >> 3) << 11))
                        .to_le_bytes(),
                ),
                1 => row[destination..destination + 2].copy_from_slice(
                    &(u16::from(red >> 3)
                        | (u16::from(green >> 3) << 5)
                        | (u16::from(blue >> 3) << 10)
                        | (u16::from(alpha >> 7) << 15))
                        .to_le_bytes(),
                ),
                2 => row[destination..destination + 2].copy_from_slice(
                    &(u16::from(red >> 4)
                        | (u16::from(green >> 4) << 4)
                        | (u16::from(blue >> 4) << 8)
                        | (u16::from(alpha >> 4) << 12))
                        .to_le_bytes(),
                ),
                3 => row[destination..destination + 4].copy_from_slice(&[red, green, blue, alpha]),
                _ => unreachable!("pixel format normalized above"),
            }
        }
        let row_address = address.wrapping_add((y * row_bytes) as u32);
        memory.write_bytes(row_address, &row)?;
    }
    Ok(())
}

fn write_mpeg_audio(memory: &mut psp_memory::Memory, address: u32, decoded: &[u8]) -> Result<()> {
    let mut output = vec![0; AUDIO_CHUNK_BYTES];
    let count = decoded.len().min(output.len());
    output[..count].copy_from_slice(&decoded[..count]);
    memory.write_bytes(address, &output)?;
    Ok(())
}

fn write_mpeg_fallback_frame(
    memory: &mut psp_memory::Memory,
    address: u32,
    frame_width: u32,
    pixel_format: u32,
    frame_number: u32,
) -> Result<()> {
    let pixel_format = if pixel_format <= 3 { pixel_format } else { 3 };
    let bytes_per_pixel = if pixel_format == 3 { 4 } else { 2 };
    let stride =
        usize::try_from(frame_width).context("MPEG frame width does not fit host usize")?;
    let image_width = stride.min(MPEG_VIDEO_WIDTH as usize);
    let height = MPEG_VIDEO_HEIGHT as usize;
    let row_bytes = stride
        .checked_mul(bytes_per_pixel)
        .context("MPEG frame row size overflow")?;
    let frame_bytes = row_bytes
        .checked_mul(height)
        .context("MPEG frame size overflow")?;
    let mut pixels = vec![0; frame_bytes];
    let phase = frame_number as u8;
    for y in 0..height {
        for x in 0..image_width {
            let x8 = ((x * 255) / image_width.max(1)) as u8;
            let y8 = ((y * 255) / height.max(1)) as u8;
            let red = x8.wrapping_add(phase);
            let green = y8.wrapping_add(phase.rotate_left(1));
            let blue = x8.wrapping_add(y8).wrapping_add(phase.rotate_left(2));
            let offset = y * row_bytes + x * bytes_per_pixel;
            match pixel_format {
                0 => {
                    let pixel = (u16::from(red >> 3)
                        | (u16::from(green >> 2) << 5)
                        | (u16::from(blue >> 3) << 11))
                        .to_le_bytes();
                    pixels[offset..offset + bytes_per_pixel].copy_from_slice(&pixel);
                }
                1 => {
                    let pixel = (u16::from(red >> 3)
                        | (u16::from(green >> 3) << 5)
                        | (u16::from(blue >> 3) << 10)
                        | 0x8000)
                        .to_le_bytes();
                    pixels[offset..offset + bytes_per_pixel].copy_from_slice(&pixel);
                }
                2 => {
                    let pixel = (u16::from(red >> 4)
                        | (u16::from(green >> 4) << 4)
                        | (u16::from(blue >> 4) << 8)
                        | 0xf000)
                        .to_le_bytes();
                    pixels[offset..offset + bytes_per_pixel].copy_from_slice(&pixel);
                }
                3 => {
                    let pixel = u32::from(red)
                        .wrapping_add(u32::from(green) << 8)
                        .wrapping_add(u32::from(blue) << 16)
                        .wrapping_add(0xff00_0000)
                        .to_le_bytes();
                    pixels[offset..offset + bytes_per_pixel].copy_from_slice(&pixel);
                }
                _ => unreachable!("pixel format normalized above"),
            }
        }
    }
    memory.write_bytes(address, &pixels)?;
    Ok(())
}

fn write_mpeg_au(
    memory: &mut psp_memory::Memory,
    address: u32,
    es_buffer: u32,
    dts: u32,
    es_size: u32,
) -> Result<()> {
    memory.write_u32(address, 0)?;
    memory.write_u32(address + 4, 0)?;
    memory.write_u32(address + MPEG_AU_DTS_MSB, 0)?;
    memory.write_u32(address + MPEG_AU_DTS, dts)?;
    memory.write_u32(address + MPEG_AU_ES_BUFFER, es_buffer)?;
    memory.write_u32(address + MPEG_AU_SIZE, es_size)?;
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn signed_io_result(error: u32) -> i64 {
    i64::from(error as i32)
}

fn write_async_result(memory: &mut psp_memory::Memory, address: u32, result: i64) -> Result<()> {
    if address != 0 {
        memory.write_u32(address, result as u32)?;
        memory.write_u32(address.wrapping_add(4), (result >> 32) as u32)?;
    }
    Ok(())
}

fn read_guest_pcm(
    memory: &psp_memory::Memory,
    address: u32,
    sample_count: u32,
    format: u32,
    left_volume: u32,
    right_volume: u32,
) -> Result<Vec<u8>> {
    let samples = usize::try_from(sample_count).context("audio sample count overflows host")?;
    let mono = format == 0x10;
    let input_frame_bytes = if mono { 2 } else { 4 };
    let input = memory.read_bytes(
        address,
        samples
            .checked_mul(input_frame_bytes)
            .context("audio buffer size overflows host")?,
    )?;
    let left_volume = normalize_audio_volume(left_volume);
    let right_volume = normalize_audio_volume(right_volume);
    let mut output = Vec::with_capacity(samples.saturating_mul(4));
    for index in 0..samples {
        let offset = index * input_frame_bytes;
        let left = i16::from_le_bytes([input[offset], input[offset + 1]]);
        let right = if mono {
            left
        } else {
            i16::from_le_bytes([input[offset + 2], input[offset + 3]])
        };
        output.extend_from_slice(&scale_audio_sample(left, left_volume).to_le_bytes());
        output.extend_from_slice(&scale_audio_sample(right, right_volume).to_le_bytes());
    }
    Ok(output)
}

fn normalize_audio_volume(volume: u32) -> i32 {
    // psp_audio_volume_max is 0x8000. treat the all-ones value used by a few
    // titles as the default full-volume setting, then clip out-of-range
    // values as the hardware mixer does.
    if volume == u32::MAX {
        0x8000
    } else {
        (volume as i32).clamp(0, 0x8000)
    }
}

fn scale_audio_sample(sample: i16, volume: i32) -> i16 {
    (i64::from(sample) * i64::from(volume) / 0x8000).clamp(i64::from(i16::MIN), i64::from(i16::MAX))
        as i16
}

fn read_guest_string(memory: &psp_memory::Memory, address: u32) -> Result<String> {
    let mut bytes = Vec::new();
    for offset in 0..1024u32 {
        let byte = memory.read_u8(address.wrapping_add(offset))?;
        if byte == 0 {
            return String::from_utf8(bytes).context("guest string is not UTF-8");
        }
        bytes.push(byte);
    }
    anyhow::bail!("unterminated guest string at 0x{address:08x}")
}

fn disc_path(path: &str) -> &str {
    if let Some((device, rest)) = path.split_once(':')
        && (device.eq_ignore_ascii_case("disc0") || device.eq_ignore_ascii_case("umd0"))
    {
        rest.trim_start_matches('/')
    } else {
        path.trim_start_matches('/')
    }
}

fn parse_lbn_path(path: &str) -> Option<(u64, usize)> {
    let path = path.strip_prefix("sce_lbn0x")?;
    let (sector, size) = path.split_once("_size0x")?;
    Some((
        u64::from_str_radix(sector, 16).ok()?,
        usize::from_str_radix(size, 16).ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_flag_wait_modes_and_clear_modes() {
        assert!(event_flag_matches(0b0110, 0b0010, 0));
        assert!(!event_flag_matches(0b0010, 0b0110, 0));
        assert!(event_flag_matches(0b0010, 0b0110, 1));

        let mut bits = 0b1111;
        apply_event_flag_clear(&mut bits, 0b0101, 0x20);
        assert_eq!(bits, 0b1010);
        apply_event_flag_clear(&mut bits, 0b0010, 0x10);
        assert_eq!(bits, 0);
    }

    #[test]
    fn fatms_state_devctl_returns_inserted_state() {
        assert_eq!(devctl_u32_output("fatms0:", 0x0242_5823), Some(1));
        assert_eq!(devctl_u32_output("FATMS0:", 0x0242_5823), Some(1));
        assert_eq!(devctl_u32_output("disc0:", 0x0242_5823), None);
    }

    #[test]
    fn delivered_byte_hash_is_stable() {
        assert_eq!(fnv1a64(b"ENGLISH.GXT"), 0x37c5_5597_dcf0_b2fc);
    }

    #[test]
    fn async_io_results_are_signed_64_bit_values() {
        assert_eq!(
            signed_io_result(IO_ERROR_FILE_NOT_FOUND),
            i64::from(IO_ERROR_FILE_NOT_FOUND as i32)
        );

        let mut memory = psp_memory::Memory::default();
        memory.map(0x1000, 0x1000, true, false).unwrap();
        write_async_result(&mut memory, 0x1000, -1).unwrap();
        assert_eq!(memory.read_u32(0x1000).unwrap(), u32::MAX);
        assert_eq!(memory.read_u32(0x1004).unwrap(), u32::MAX);
    }

    fn synchronous_read_fixture() -> (HleState, psp_core::Emulator, psp_loader::Import, u32) {
        let mut emulator = psp_core::Emulator::new(0x1000);
        emulator.memory.map(0x1000, 0x10000, true, true).unwrap();
        let mut hle = HleState::new(None, USER_MODULE_BASE, None).unwrap();
        hle.attach_main_thread(&emulator.cpu);
        let mut submitter = psp_cpu::Cpu::new(0x1100);
        submitter.gpr[2] = 0x1234;
        hle.threads.insert(
            2,
            GuestThread {
                entry: submitter.pc,
                priority: 56,
                stack_size: 0,
                stack_base: 0,
                stack_top: 0,
                attributes: 0,
                cpu: Some(submitter),
            },
        );
        hle.ready.push_back(2);
        let fd = hle.insert_file(GuestFile::bytes((0..40_001).map(|i| i as u8).collect(), 0));
        let read = psp_loader::Import {
            syscall: 1,
            library: "IoFileMgrForUser".into(),
            nid: 0x6a63_8d83,
            stub_address: 0,
        };
        (hle, emulator, read, fd)
    }

    #[test]
    fn synchronous_read_yields_to_submitter_before_completion() {
        let (mut hle, mut emulator, read, fd) = synchronous_read_fixture();
        let mut position = 0;
        // include a short read and eof: timing uses the request size while
        // the resumed reader receives the actual byte count in its own v0.
        for (request, count, delay) in [
            (20_000, 20_000, 200),
            (20_000, 20_000, 200),
            (20_000, 1, 200),
            (4, 0, 100),
        ] {
            let now = emulator.scheduler.now();
            emulator.memory.write_u32(0x1000, 0).unwrap();
            emulator.cpu.gpr[4] = fd;
            emulator.cpu.gpr[5] = 0x2000;
            emulator.cpu.gpr[6] = request;
            hle.dispatch(&read, &mut emulator).unwrap();

            assert_eq!(hle.current_uid, 2, "the lower-priority submitter must run");
            assert_eq!(
                emulator.cpu.gpr[2], 0x1234,
                "do not overwrite the next thread's result"
            );
            assert_eq!(
                emulator.scheduler.now(),
                now,
                "I/O must not jump the global clock"
            );
            assert_eq!(hle.timed_waiters[&1], now + delay);
            assert_eq!(
                emulator.memory.read_bytes(0x2000, count).unwrap(),
                (position..position + count)
                    .map(|i| i as u8)
                    .collect::<Vec<_>>()
            );
            position += count;
            assert_eq!(hle.files[&fd].position, position);

            // the submitter publishes the handle before the reader can run
            // its completion handler. this is the lost-completion race.
            emulator.memory.write_u32(0x1000, 42).unwrap();
            emulator.scheduler.advance_micros(delay - 1);
            hle.wake_timed_waiters(&mut emulator);
            assert_eq!(hle.current_uid, 2);
            emulator.scheduler.advance_micros(1);
            hle.wake_timed_waiters(&mut emulator);
            assert_eq!(hle.current_uid, 1);
            assert_eq!(emulator.cpu.pc, 0x1000);
            assert_eq!(emulator.cpu.gpr[2], count as u32);
            assert_eq!(emulator.memory.read_u32(0x1000).unwrap(), 42);
            assert!(!hle.timed_waiters.contains_key(&1));
        }
    }

    #[test]
    fn synchronous_read_keeps_earlier_timer_wakeups() {
        let (mut hle, mut emulator, read, fd) = synchronous_read_fixture();
        hle.ready.clear();
        hle.timed_waiters.insert(2, 50);
        emulator.cpu.gpr[4] = fd;
        emulator.cpu.gpr[5] = 0x2000;
        emulator.cpu.gpr[6] = 20_000;
        hle.dispatch(&read, &mut emulator).unwrap();

        // with no ready threads the scheduler may advance only to the first
        // deadline, letting this thread run during the remaining i/o time.
        assert_eq!(emulator.scheduler.now(), 50);
        assert_eq!(hle.current_uid, 2);
        assert_eq!(hle.timed_waiters[&1], 200);
        assert_eq!(emulator.cpu.gpr[2], 0x1234);
        emulator.scheduler.advance_micros(150);
        hle.wake_timed_waiters(&mut emulator);
        assert_eq!(hle.current_uid, 1);
        assert_eq!(emulator.cpu.gpr[2], 20_000);
    }

    #[test]
    fn psp_iso_stat_has_expected_type_access_and_extent() {
        let file = psp_io_stat(DiscMetadata {
            extent: 0x6660,
            size: 0x2ab0,
            directory: false,
            access: SCE_STM_ISO_ACCESS,
        });
        assert_eq!(u32::from_le_bytes(file[0..4].try_into().unwrap()), 0x216d);
        assert_eq!(u32::from_le_bytes(file[4..8].try_into().unwrap()), 0x20);
        assert_eq!(u64::from_le_bytes(file[8..16].try_into().unwrap()), 0x2ab0);
        assert_eq!(u32::from_le_bytes(file[64..68].try_into().unwrap()), 0x6660);
        assert_eq!(file[16], 0xfe);

        let directory = psp_io_stat(DiscMetadata {
            extent: 0x120,
            size: 0x400,
            directory: true,
            access: SCE_STM_ISO_ACCESS,
        });
        assert_eq!(
            u32::from_le_bytes(directory[0..4].try_into().unwrap()),
            0x116d
        );
        assert_eq!(
            u32::from_le_bytes(directory[4..8].try_into().unwrap()),
            0x10
        );
    }

    #[test]
    fn raw_lbn_paths_decode_sector_and_byte_length() {
        assert_eq!(
            parse_lbn_path("sce_lbn0x6660_size0x2ab0"),
            Some((0x6660, 0x2ab0))
        );
        assert_eq!(parse_lbn_path("sce_lbn0x_size0x2ab0"), None);
        assert_eq!(parse_lbn_path("sce_lbn0x6660_size"), None);
    }

    #[test]
    fn system_parameters_match_english_host_profile() {
        assert_eq!(system_param_int(8), Some(1));
        assert_eq!(system_param_int(2), Some(0));
        assert_eq!(system_param_int(9), Some(0));
        assert_eq!(system_param_int(99), None);
    }

    #[test]
    fn system_clock_conversion_returns_seconds_and_microseconds() {
        assert_eq!(split_system_clock(2_345_678), (2, 345_678));
        assert_eq!(
            split_system_clock(u64::from(u32::MAX) * 1_000_000),
            (u32::MAX, 0)
        );
    }

    #[cfg(feature = "desktop")]
    #[test]
    fn host_scale_accepts_only_supported_integer_factors() {
        assert_eq!(parse_scale("1").unwrap(), 1);
        assert_eq!(parse_scale("0x4").unwrap(), 4);
        assert!(parse_scale("0").is_err());
        assert!(parse_scale("5").is_err());
    }

    #[cfg(feature = "desktop")]
    #[test]
    fn host_keys_map_to_psp_buttons_without_edge_state() {
        assert_eq!(
            host_button_mask(|key| [Scancode::X, Scancode::Z, Scancode::Up].contains(&key)),
            PSP_CTRL_CROSS | PSP_CTRL_CIRCLE | PSP_CTRL_UP
        );
        assert_eq!(
            host_button_mask(|key| [Scancode::J, Scancode::L].contains(&key)),
            0
        );
    }

    #[test]
    fn mpeg_state_tracks_stream_and_es_buffer_lifetimes() {
        let mut state = MpegState::default();
        state.init();
        state.create(0x0989_f4d4);

        assert_eq!(state.register_stream(0x0989_f4d4), Some(1));
        assert_eq!(state.malloc_es_buffer(0x0989_f4d4), Some(1));
        assert_eq!(
            state.au_parameters(0x0989_f4d4, 1),
            Some((MPEG_AVC_ES_SIZE, 0))
        );
        assert_eq!(
            state.au_parameters(0x0989_f4d4, 0x098a_f4c0),
            Some((MPEG_ATRAC_ES_SIZE, u32::MAX))
        );
        assert_eq!(state.free_es_buffer(0x0989_f4d4, 0), Some(false));
        assert_eq!(state.free_es_buffer(0x0989_f4d4, 1), Some(true));
        assert_eq!(state.unregister_stream(0x0989_f4d4, 1), Some(()));

        state.finish();
        assert!(!state.initialized);
        assert_eq!(state.register_stream(0x0989_f4d4), None);
    }

    #[test]
    fn mpeg_guest_structs_follow_psp_layouts() {
        let mut memory = psp_memory::Memory::default();
        memory.map(0x1000, 0x1000, true, false).unwrap();

        initialize_mpeg_ringbuffer(&mut memory, 0x1000, 4, 0x2000, 0x3000, 0x4000).unwrap();
        assert_eq!(memory.read_u32(0x1000 + MPEG_RING_PACKETS).unwrap(), 4);
        assert_eq!(
            memory.read_u32(0x1000 + MPEG_RING_PACKET_SIZE).unwrap(),
            MPEG_PACKET_SIZE
        );
        assert_eq!(
            memory.read_u32(0x1000 + MPEG_RING_CALLBACK).unwrap(),
            0x3000
        );
        assert_eq!(
            memory.read_u32(0x1000 + MPEG_RING_CALLBACK_PARAM).unwrap(),
            0x4000
        );
        assert_eq!(
            memory.read_u32(0x1000 + MPEG_RING_DATA_END).unwrap(),
            0x4000
        );

        write_mpeg_au(&mut memory, 0x1100, 1, u32::MAX, MPEG_ATRAC_ES_SIZE).unwrap();
        assert_eq!(memory.read_u32(0x1100 + MPEG_AU_DTS).unwrap(), u32::MAX);
        assert_eq!(memory.read_u32(0x1100 + MPEG_AU_ES_BUFFER).unwrap(), 1);
        assert_eq!(
            memory.read_u32(0x1100 + MPEG_AU_SIZE).unwrap(),
            MPEG_ATRAC_ES_SIZE
        );

        add_mpeg_packets(&mut memory, 0x1000, 2).unwrap();
        assert_eq!(memory.read_u32(0x1000 + MPEG_RING_PACKETS_READ).unwrap(), 2);
        assert_eq!(
            memory.read_u32(0x1000 + MPEG_RING_PACKETS_WRITTEN).unwrap(),
            2
        );
        assert!(consume_mpeg_packet(&mut memory, 0x1000).unwrap());
        assert_eq!(memory.read_u32(0x1000 + MPEG_RING_PACKETS_READ).unwrap(), 2);
        assert_eq!(
            memory
                .read_u32(0x1000 + MPEG_RING_PACKETS_IN_BUFFER)
                .unwrap(),
            1
        );
    }

    #[test]
    fn audio_channels_follow_psp_reservation_and_validation_rules() {
        let mut audio = AudioState::default();

        assert_eq!(audio.reserve(-1, 0x1c0, 0), 7);
        assert_eq!(audio.reserve(-1, 0x100, 0), 6);
        assert_eq!(audio.reserve(-1, 0x100, 0x10), 5);
        assert_eq!(audio.set_data_len(6, 0x800), 0);
        assert_eq!(audio.change_format(6, 0x10), 0);
        assert_eq!(audio.change_format(6, 1), AUDIO_ERROR_INVALID_FORMAT);
        assert_eq!(audio.change_volume(6, 0x4000, 0x2000), 0);
        assert_eq!(
            audio.output_volumes(6, u32::MAX, u32::MAX),
            Ok((0x4000, 0x2000))
        );
        assert_eq!(
            audio.change_volume(6, 0x1_0000, 0),
            AUDIO_ERROR_INVALID_VOLUME
        );
        assert_eq!(audio.set_data_len(0, 0x800), AUDIO_ERROR_CHANNEL_NOT_INIT);
        assert_eq!(audio.reserve(6, 0x100, 0), AUDIO_ERROR_INVALID_CHANNEL);
        assert_eq!(audio.reserve(-1, 65, 0), AUDIO_ERROR_SAMPLE_SIZE);
        assert_eq!(audio.reserve(-1, 0x100, 1), AUDIO_ERROR_INVALID_FORMAT);

        assert_eq!(audio.release(6), 0);
        assert_eq!(audio.reserve(-1, 0x100, 0), 6);
        assert_eq!(audio.release(6), 0);
        assert_eq!(audio.release(6), AUDIO_ERROR_CHANNEL_NOT_INIT);
    }

    #[test]
    fn blocking_audio_drains_in_psp_hardware_blocks() {
        let mut audio = AudioState::default();
        assert_eq!(audio.reserve(1, 0x100, 0x10), 1);

        assert_eq!(audio.enqueue(1, 0x100, true, true), Ok(0));
        assert_eq!(
            audio.enqueue(1, 0x100, true, false),
            Err(AUDIO_ERROR_CHANNEL_BUSY)
        );

        // blocking output waits for the full previously queued buffer, as
        // on psp hardware, so the audio thread paces to real
        // time instead of spinning on 64-sample mixer blocks.
        let full_buffer = AudioState::wait_duration_us(0x100);
        assert_eq!(full_buffer, 5_805);
        assert_eq!(audio.enqueue(1, 0x100, true, true), Ok(full_buffer));
        audio.advance_to(full_buffer);
        assert_eq!(audio.channel_rest(1), Ok(0x100));
    }

    #[test]
    fn audio_enqueue_refreshes_against_scheduler_time() {
        let mut audio = AudioState::default();
        assert_eq!(audio.reserve(1, 0x100, 0x10), 1);
        assert_eq!(audio.enqueue(1, 0x100, true, true), Ok(0));
        audio.advance_to(6_000);
        assert_eq!(audio.channel_rest(1), Ok(0));
    }

    #[test]
    fn output2_rest_length_is_the_queued_sample_count() {
        let mut audio = AudioState::default();
        assert_eq!(audio.reserve_output2(0x200), 0);
        assert_eq!(audio.enqueue_output2(0x200, true, true), Ok(0));
        assert_eq!(audio.output2_queued_samples, 0x200);
        assert_eq!(audio.output2_sample_count, 0x200);
        assert_eq!(audio.change_output2_length(0x100), 0);
        assert_eq!(
            audio.output2_queued_samples.min(audio.output2_sample_count),
            0x100
        );
    }

    #[test]
    fn blocking_output2_waits_for_the_queued_source_buffer() {
        let mut audio = AudioState::default();
        assert_eq!(audio.reserve_output2(0x800), 0);
        assert_eq!(audio.enqueue_output2(0x800, true, true), Ok(0));

        assert_eq!(
            audio.enqueue_output2(0x800, true, true),
            Ok(AudioState::output2_wait_duration_us(0x800))
        );
        assert_eq!(AudioState::output2_wait_duration_us(0x800), 46_440);

        audio.advance_to(46_440);
        assert_eq!(audio.output2_queued_samples, 0x800);
    }

    #[test]
    fn guest_audio_is_mixed_to_stereo_with_psp_volume_scaling() {
        let mut memory = psp_memory::Memory::default();
        memory.map(0x1000, 0x1000, true, false).unwrap();
        memory
            .write_bytes(0x1000, &[0x00, 0x40, 0x00, 0xc0, 0xff, 0x7f, 0x00, 0x80])
            .unwrap();

        let pcm = read_guest_pcm(&memory, 0x1000, 2, 0, 0x4000, 0x8000).unwrap();
        assert_eq!(pcm, [0x00, 0x20, 0x00, 0xc0, 0xff, 0x3f, 0x00, 0x80]);
    }

    #[test]
    fn atrac_stream_space_respects_circular_buffer_and_loaded_file() {
        let context = AtracContext {
            buffer: 0x1000,
            buffer_size: 16,
            write_offset: 14,
            buffered_bytes: 4,
            next_file_offset: 20,
            format: AtracFormat {
                channels: 2,
                sample_rate: 44_100,
                samples_per_frame: ATRAC_SAMPLES_PER_FRAME,
                total_samples: 8_192,
                data_offset: 0,
                data_bytes: 64,
                block_bytes: 16,
            },
            decoder: None,
            decoded_samples: 0,
            loop_num: 0,
        };

        // only the two bytes before the ring wraps are contiguous at the
        // current write pointer, even though more capacity is free overall.
        assert_eq!(context.writable_bytes(), 2);
    }
}
