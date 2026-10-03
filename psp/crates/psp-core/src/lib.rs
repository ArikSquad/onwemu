//! The small object that wires the emulator's guest-facing subsystems together.
//!
//! `Emulator` intentionally owns the CPU, memory, scheduler, and device HLEs
//! in one place. The frontend can then choose how it loads images and presents
//! output without having to duplicate the guest execution loop.

use psp_audio::AudioEngine;
use psp_cpu::{Cpu, CpuError, Step};
use psp_debugger::Diagnostics;
use psp_gpu::Gpu;
use psp_input::InputState;
use psp_kernel::Scheduler;
use psp_memory::Memory;
use psp_vfs::VirtualFileSystem;
use std::path::PathBuf;
use thiserror::Error;
#[derive(Debug, Error)]
/// Reasons the guest runtime stopped before completing its requested budget.
pub enum RunError {
    #[error(transparent)]
    /// A CPU instruction could not be fetched or executed.
    Cpu(#[from] CpuError),
    #[error("instruction limit reached after {0} instructions")]
    /// The caller's instruction budget expired.
    Limit(u64),
    #[error("guest syscall {0} has no HLE handler")]
    /// The frontend has not registered a handler for this guest syscall.
    UnsupportedSyscall(u32),
}

/// The guest runtime and the services visible to PSP code.
pub struct Emulator {
    /// Guest CPU state and instruction counter.
    pub cpu: Cpu,
    /// Checked guest address space.
    pub memory: Memory,
    /// Guest clock and thread metadata.
    pub scheduler: Scheduler,
    /// GE command state and software renderer.
    pub gpu: Gpu,
    /// Guest PCM queue and optional host sink.
    pub audio: AudioEngine,
    /// Current controller state and sampled input ring.
    pub input: InputState,
    /// Guest path resolver for the configured mounts.
    pub vfs: VirtualFileSystem,
    /// Diagnostics collected during the run.
    pub diagnostics: Diagnostics,
}
impl Emulator {
    /// Create an empty runtime whose CPU will start at `entry`.
    ///
    /// The caller still needs to map guest code and data into `memory` before
    /// calling [`Self::run`].
    pub fn new(entry: u32) -> Self {
        Self {
            cpu: Cpu::new(entry),
            memory: Memory::default(),
            scheduler: Scheduler::default(),
            gpu: Gpu::default(),
            audio: AudioEngine::default(),
            input: InputState::default(),
            vfs: VirtualFileSystem::new(PathBuf::from("memory-stick"), None),
            diagnostics: Diagnostics::default(),
        }
    }

    /// Execute at most `limit` instructions, advancing kernel time after each
    /// successful instruction.
    ///
    /// A guest syscall is returned to the caller so the frontend can dispatch
    /// it through its HLE table. Reaching the limit is also an error: this
    /// keeps accidental infinite loops visible to command-line callers.
    pub fn run(&mut self, limit: u64) -> Result<(), RunError> {
        for _ in 0..limit {
            match self.cpu.step(&mut self.memory)? {
                Step::Continue => {}
                Step::Syscall(code) => return Err(RunError::UnsupportedSyscall(code)),
            }
            self.scheduler.advance(1)
        }
        Err(RunError::Limit(limit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psp_memory::GuestMemory;

    #[test]
    fn runtime_runs_mapped_code_and_advances_guest_time() {
        let mut emulator = Emulator::new(0x1000);
        emulator.memory.map(0x1000, 0x100, true, true).unwrap();
        emulator.memory.write_u32(0x1000, 0x2401_0007).unwrap(); // addiu $at, $zero, 7

        assert!(matches!(emulator.run(1), Err(RunError::Limit(1))));
        assert_eq!(emulator.cpu.gpr[1], 7);
        assert_eq!(
            emulator.scheduler.now(),
            0,
            "one cycle is below one microsecond"
        );
    }

    #[test]
    fn runtime_surfaces_syscalls_before_consuming_the_next_instruction() {
        let mut emulator = Emulator::new(0x1000);
        emulator.memory.map(0x1000, 0x100, true, true).unwrap();
        emulator
            .memory
            .write_u32(0x1000, (0x55u32 << 6) | 0x0c)
            .unwrap();

        assert!(matches!(
            emulator.run(10),
            Err(RunError::UnsupportedSyscall(0x55))
        ));
        assert_eq!(emulator.cpu.instruction_count, 1);
        assert_eq!(emulator.scheduler.now(), 0);
    }

    #[test]
    fn zero_instruction_budget_is_reported_without_touching_guest_state() {
        let mut emulator = Emulator::new(0x4321);
        assert!(matches!(emulator.run(0), Err(RunError::Limit(0))));
        assert_eq!(emulator.cpu.pc, 0x4321);
        assert_eq!(emulator.cpu.instruction_count, 0);
    }
}
