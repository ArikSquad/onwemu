//! Cached basic blocks for the Allegrex CPU.
//!
//! Blocks hold compact operands and resolved opcode handlers shared with the
//! interpreter. Branches include their delay slot; syscalls and faults return
//! to the frontend. Page generations invalidate changed code, including stores
//! to instructions later in the current block. Chaining amortizes scheduler
//! overhead while respecting the caller's instruction budget.

use std::cell::Cell;
use std::collections::HashMap;

use psp_memory::Memory;

use crate::{Cpu, CpuError, Decoded, Handler, Step, handler_for};

/// Maximum operations per compiled block.
///
/// This bounds compile time and keeps invalidation work coarse-grained.
pub const MAX_BLOCK_OPS: usize = 128;
/// Keep executing through hot basic blocks before returning to the frontend.
/// Returning after every branch makes branch-heavy game code pay the full
/// scheduler/host-loop overhead once per block instead of once per syscall or
/// scheduling boundary.
const MAX_CHAIN_OPS: u64 = 4_096;
const HOT_BLOCK_SLOTS: usize = 256;

#[derive(Clone, Copy, Debug)]
struct BlockOp {
    decoded: Decoded,
    handler: Handler,
    writes_memory: bool,
}

#[derive(Clone, Debug)]
struct Block {
    start: u32,
    ops: Vec<BlockOp>,
    /// (`page`, `generation`) snapshot used to detect code modification.
    pages: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpKind {
    Straight,
    Branch,
    Syscall,
}

/// Classify an instruction word for block formation. This must be conservative
/// with respect to the opcode handlers: every operation that can set
/// the pending branch target or adjust the program counter directly is a
/// [`OpKind::Branch`], and the syscall stub terminator is [`OpKind::Syscall`].
/// Anything else executes inline.
fn classify(word: u32) -> OpKind {
    match word >> 26 {
        0x01..=0x07 | 0x14..=0x17 => OpKind::Branch,
        0 => match word & 63 {
            0x08 | 0x09 => OpKind::Branch,
            0x0c => OpKind::Syscall,
            _ => OpKind::Straight,
        },
        // bc1 (any condition/likely form) and vfpu branches.
        0x11 | 0x12 if (word >> 21) & 31 == 8 => OpKind::Branch,
        _ => OpKind::Straight,
    }
}

// stores can change a later instruction in the block we are executing.
fn writes_memory(word: u32) -> bool {
    matches!(word >> 26, 0x28..=0x2b | 0x2e | 0x38..=0x3a | 0x3d | 0x3e)
}

/// The outcome of running compiled guest code. `executed` counts every op
/// the executor completed, including a terminal syscall or the ops before a
/// fault, so the scheduler and instruction budget stay exact.
#[derive(Debug)]
pub enum BlockOutcome {
    /// Ran to the end of the block without leaving compiled code.
    Flowing(u64),
    /// Hit a syscall stub; `pc` is the stub address for diagnostics.
    Syscall {
        /// Emulator-private syscall number.
        code: u32,
        /// Guest address of the syscall stub.
        pc: u32,
        /// Number of operations completed before returning.
        executed: u64,
    },
}

/// A fault plus the operations completed before it, so partially executed blocks
/// still advance guest time exactly.
#[derive(Debug)]
pub struct BlockFault {
    /// The memory, unsupported-instruction, or arithmetic error that stopped
    /// execution.
    pub error: CpuError,
    /// Number of operations completed before the faulting instruction.
    pub executed: u64,
}

#[derive(Debug)]
/// Cached basic blocks and their execution counters.
pub struct Jit {
    /// Compiled blocks. Entries are never removed, only replaced in place, so
    /// indices stay valid for the hot cache below.
    blocks: Vec<Block>,
    index: HashMap<u32, usize>,
    /// Recent blocks indexed by PC; entries still validate their code pages.
    hot: [Cell<Option<usize>>; HOT_BLOCK_SLOTS],
    /// Counters use interior mutability so block execution can run from a
    /// shared cache borrow without copying compiled streams.
    compiled: Cell<u64>,
    hits: Cell<u64>,
    misses: Cell<u64>,
    invalidated: Cell<u64>,
    executed_ops: Cell<u64>,
    executed_blocks: Cell<u64>,
}

impl Default for Jit {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            index: HashMap::new(),
            hot: std::array::from_fn(|_| Cell::new(None)),
            compiled: Cell::new(0),
            hits: Cell::new(0),
            misses: Cell::new(0),
            invalidated: Cell::new(0),
            executed_ops: Cell::new(0),
            executed_blocks: Cell::new(0),
        }
    }
}

impl Jit {
    /// Return the number of block slots currently held by the cache.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Return the number of successful block compilations.
    pub fn compiled(&self) -> u64 {
        self.compiled.get()
    }

    /// Return the number of cache hits, including hot-cache hits.
    pub fn cache_hits(&self) -> u64 {
        self.hits.get()
    }

    /// Return the number of lookups that needed compilation.
    pub fn cache_misses(&self) -> u64 {
        self.misses.get()
    }

    /// Return the number of stale blocks detected through page generations.
    pub fn invalidated(&self) -> u64 {
        self.invalidated.get()
    }

    /// Return the number of guest operations completed through the JIT.
    pub fn executed_ops(&self) -> u64 {
        self.executed_ops.get()
    }

    /// Return the number of block executions, including partial executions.
    pub fn executed_blocks(&self) -> u64 {
        self.executed_blocks.get()
    }

    fn snapshot_pages(ops: &[BlockOp], memory: &Memory) -> Vec<(u32, u32)> {
        let mut pages = Vec::new();
        for op in ops {
            let page = op.decoded.pc >> 12;
            if pages.last().is_none_or(|&(last, _)| last != page) {
                pages.push((page, memory.page_generation(page)));
            }
        }
        pages
    }

    /// Compile the block starting at `pc`.
    ///
    /// Returns `false` when nothing can be cached and the caller must
    /// single-step instead.
    fn compile_at(&mut self, pc: u32, memory: &Memory) -> bool {
        let mut ops = Vec::new();
        let mut addr = pc;
        while ops.len() < MAX_BLOCK_OPS {
            let Ok(word) = memory.fetch_u32(addr) else {
                break;
            };
            let decoded = Decoded::decode(addr, word);
            ops.push(BlockOp {
                decoded,
                handler: handler_for(word),
                writes_memory: writes_memory(word),
            });
            match classify(word) {
                OpKind::Straight => addr = addr.wrapping_add(4),
                // terminal ops end the block; the executor leaves compiled
                // code through the syscall/fault exits.
                OpKind::Syscall => break,
                OpKind::Branch => {
                    // the delay slot is part of the block; without it the
                    // branch cannot execute correctly from cache.
                    let delay_addr = addr.wrapping_add(4);
                    let Ok(delay) = memory.fetch_u32(delay_addr) else {
                        return false;
                    };
                    let delay_decoded = Decoded::decode(delay_addr, delay);
                    ops.push(BlockOp {
                        decoded: delay_decoded,
                        handler: handler_for(delay),
                        writes_memory: writes_memory(delay),
                    });
                    break;
                }
            }
        }
        if ops.is_empty() {
            return false;
        }
        let pages = Self::snapshot_pages(&ops, memory);
        let block = Block {
            start: pc,
            ops,
            pages,
        };
        // block slots stay stable when code is recompiled.
        match self.index.get(&pc).copied() {
            Some(slot) => self.blocks[slot] = block,
            None => {
                self.index.insert(pc, self.blocks.len());
                self.blocks.push(block);
            }
        }
        self.compiled.set(self.compiled.get() + 1);
        true
    }

    fn is_valid_static(block: &Block, memory: &Memory) -> bool {
        block
            .pages
            .iter()
            .all(|&(page, generation)| memory.page_generation(page) == generation)
    }

    /// Look up the block index for `pc`, validating its page snapshot.
    ///
    /// Returns `None` on a miss, stale entry, or hot-cache state that needs the
    /// slow path. Updates the hit and miss counters.
    fn resolve(&self, pc: u32, memory: &Memory) -> Option<usize> {
        let hot = &self.hot[((pc >> 2) ^ (pc >> 10)) as usize & (HOT_BLOCK_SLOTS - 1)];
        if let Some(slot) = hot.get()
            && let Some(block) = self.blocks.get(slot)
            && block.start == pc
            && Self::is_valid_static(block, memory)
        {
            self.hits.set(self.hits.get() + 1);
            return Some(slot);
        }
        let slot = *self.index.get(&pc)?;
        let block = &self.blocks[slot];
        if !Self::is_valid_static(block, memory) {
            return None;
        }
        self.hits.set(self.hits.get() + 1);
        hot.set(Some(slot));
        Some(slot)
    }

    /// Run guest code starting at the CPU's current program counter.
    ///
    /// Executes at most `budget` operations, stopping early at syscalls, faults,
    /// and control-flow exits. Falls back to single-step execution when the
    /// program counter is not compilable, so behavior always matches the
    /// interpreter.
    pub fn step_guest(
        &mut self,
        cpu: &mut Cpu,
        memory: &mut Memory,
        thread_id: u32,
        budget: u64,
    ) -> Result<BlockOutcome, BlockFault> {
        let mut total_executed = 0u64;
        while total_executed < budget && total_executed < MAX_CHAIN_OPS {
            let pc = cpu.pc;
            // resolve (and if needed compile) the block first; the execution
            // loop below only holds a shared borrow while mutating engine
            // counters through interior mutability, so no stream copy is
            // needed.
            let slot = match self.resolve(pc, memory) {
                Some(slot) => slot,
                None => {
                    if self.index.contains_key(&pc) {
                        self.invalidated.set(self.invalidated.get() + 1);
                    } else {
                        self.misses.set(self.misses.get() + 1);
                    }
                    if !self.compile_at(pc, memory) {
                        // not compilable right now (unmapped or truncated
                        // delay slot): execute exactly one interpreter step.
                        return match cpu.step_with_thread_id(memory, thread_id) {
                            Ok(Step::Continue) => Ok(BlockOutcome::Flowing(total_executed + 1)),
                            Ok(Step::Syscall(code)) => Ok(BlockOutcome::Syscall {
                                code,
                                pc,
                                executed: total_executed + 1,
                            }),
                            Err(error) => Err(BlockFault {
                                error,
                                executed: total_executed,
                            }),
                        };
                    }
                    self.resolve(pc, memory).expect("block was just compiled")
                }
            };
            let (outcome, executed) = {
                let block = &self.blocks[slot];
                let mut executed = 0u64;
                let mut index = 0;
                let outcome: Result<BlockOutcome, BlockFault> = loop {
                    if index >= block.ops.len()
                        || total_executed + executed >= budget
                        || total_executed + executed >= MAX_CHAIN_OPS
                    {
                        break Ok(BlockOutcome::Flowing(executed));
                    }
                    let op = block.ops[index];
                    match cpu.execute_predecoded(memory, op.decoded, op.handler, thread_id) {
                        Ok(Step::Continue) => {
                            executed += 1;
                            // delay-slot annulment (branch-likely not taken)
                            // skips the next word inline; any other
                            // redirection means the block's control flow left
                            // the compiled trace. the outer chain resolves
                            // the new pc without returning to the frontend.
                            if cpu.pc != op.decoded.pc.wrapping_add(4)
                                || (op.writes_memory && !Self::is_valid_static(block, memory))
                            {
                                break Ok(BlockOutcome::Flowing(executed));
                            }
                        }
                        Ok(Step::Syscall(code)) => {
                            executed += 1;
                            break Ok(BlockOutcome::Syscall {
                                code,
                                pc: op.decoded.pc,
                                executed,
                            });
                        }
                        Err(error) => break Err(BlockFault { error, executed }),
                    }
                    index += 1;
                };
                (outcome, executed)
            };
            self.executed_ops.set(self.executed_ops.get() + executed);
            self.executed_blocks.set(self.executed_blocks.get() + 1);
            total_executed += executed;
            match outcome {
                Ok(BlockOutcome::Flowing(_)) => {
                    if executed == 0 {
                        return Ok(BlockOutcome::Flowing(total_executed));
                    }
                    // the frontend owns thread returns and interrupt
                    // unwinding signaled by a null pc. do not try to compile
                    // that sentinel as the next chained block.
                    if cpu.pc == 0 {
                        return Ok(BlockOutcome::Flowing(total_executed));
                    }
                }
                Ok(BlockOutcome::Syscall { code, pc, .. }) => {
                    return Ok(BlockOutcome::Syscall {
                        code,
                        pc,
                        executed: total_executed,
                    });
                }
                Err(BlockFault { error, .. }) => {
                    return Err(BlockFault {
                        error,
                        executed: total_executed,
                    });
                }
            }
        }
        Ok(BlockOutcome::Flowing(total_executed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psp_memory::GuestMemory;

    const BASE: u32 = 0x1000;

    fn memory_with(words: &[u32]) -> Memory {
        let mut memory = Memory::default();
        memory.map(BASE, 0x1000, true, true).unwrap();
        memory.map(0x100, 0x100, true, false).unwrap();
        for (index, word) in words.iter().enumerate() {
            memory.write_u32(BASE + index as u32 * 4, *word).unwrap();
        }
        memory
    }

    /// Execute `words` through the interpreter, one step per word.
    fn run_interp(words: &[u32]) -> (Cpu, Memory) {
        let mut memory = memory_with(words);
        let mut cpu = Cpu::new(BASE);
        for _ in words {
            cpu.step(&mut memory).unwrap();
        }
        (cpu, memory)
    }

    /// Execute exactly `words.len()` operations through the JIT.
    fn run_jit(words: &[u32]) -> (Cpu, Memory, Jit) {
        let mut memory = memory_with(words);
        let mut cpu = Cpu::new(BASE);
        let mut jit = Jit::default();
        let mut remaining = words.len() as u64;
        while remaining > 0 {
            match jit.step_guest(&mut cpu, &mut memory, 1, remaining) {
                Ok(BlockOutcome::Flowing(executed)) => {
                    assert!(executed > 0 && executed <= remaining);
                    remaining -= executed;
                }
                outcome => panic!("unexpected outcome {outcome:?}"),
            }
        }
        (cpu, memory, jit)
    }

    fn cpu_state(cpu: &Cpu) -> Vec<u32> {
        let mut state = cpu.gpr.to_vec();
        state.extend_from_slice(&[cpu.pc, cpu.hi, cpu.lo, cpu.vfpu_cc]);
        state.extend_from_slice(&cpu.vfpu);
        state
    }

    #[test]
    fn straight_block_matches_interpreter() {
        // addiu/add/sw/lw chain with no control flow.
        let words = [
            0x2408_0005, // addiu $t0, $zero, 5
            0x2409_0007, // addiu $t1, $zero, 7
            0x0109_4021, // addu $t0, $t0, $t1  ($t0 = 12)
            0xac08_0100, // sw $t0, 0x100($zero)
            0x8c0a_0100, // lw $t2, 0x100($zero)
        ];
        let (expected_cpu, expected_memory) = run_interp(&words);
        let (cpu, memory, jit) = run_jit(&words);
        assert_eq!(cpu_state(&cpu), cpu_state(&expected_cpu));
        assert_eq!(
            memory.read_u32(0x100).unwrap(),
            expected_memory.read_u32(0x100).unwrap()
        );
        assert_eq!(jit.block_count(), 1);
        assert_eq!(jit.compiled(), 1);
    }

    #[test]
    fn taken_branch_skips_shadowed_words() {
        // beq $zero, $zero, +2 (taken) with a delay slot writing $t0;
        // the shadowed addiu must not execute.
        let words = [
            0x1000_0002, // beq $zero, $zero, +2
            0x2408_0001, // delay: addiu $t0, $zero, 1
            0x2408_0002, // shadowed: addiu $t0, $zero, 2
            0x2409_0009, // landing: addiu $t1, $zero, 9
        ];
        let (expected_cpu, _) = run_interp(&words[..2]);
        let (cpu, _, _) = run_jit(&words[..2]);
        assert_eq!(cpu_state(&cpu), cpu_state(&expected_cpu));
        assert_eq!(cpu.gpr[8], 1);
        assert_eq!(cpu.pc, BASE + 12);
    }

    #[test]
    fn not_taken_likely_branch_annuls_delay_slot() {
        // bnel $t0, $t1, +1 not taken: delay slot is annulled (skipped).
        let words = [
            0x2408_0005, // addiu $t0, $zero, 5
            0x2409_0005, // addiu $t1, $zero, 5
            0x5509_0001, // bnel $t0, $t1, +1 (not taken)
            0x240a_0001, // delay (annulled): addiu $t2, $zero, 1
            0x240b_0002, // next: addiu $t3, $zero, 2
        ];
        let (expected_cpu, _) = run_interp(&words);
        // step the same op count through the jit in whole blocks.
        let mut memory = memory_with(&words);
        let mut cpu = Cpu::new(BASE);
        let mut jit = Jit::default();
        let mut remaining = words.len() as u64;
        while remaining > 0 {
            match jit.step_guest(&mut cpu, &mut memory, 1, remaining) {
                Ok(BlockOutcome::Flowing(executed)) => {
                    assert!(executed > 0 && executed <= remaining);
                    remaining -= executed;
                }
                outcome => panic!("unexpected outcome {outcome:?}"),
            }
        }
        assert_eq!(cpu_state(&cpu), cpu_state(&expected_cpu));
        assert_eq!(cpu.gpr[10], 0);
        assert_eq!(cpu.gpr[11], 2);
    }

    #[test]
    fn syscall_exits_with_stub_address() {
        let words = [
            0x2408_0005, // addiu $t0, $zero, 5
            0x0000_000c, // syscall 0
        ];
        let mut memory = memory_with(&words);
        let mut cpu = Cpu::new(BASE);
        let mut jit = Jit::default();
        match jit.step_guest(&mut cpu, &mut memory, 1, 16) {
            Ok(BlockOutcome::Syscall { code, pc, executed }) => {
                assert_eq!(code, 0);
                assert_eq!(pc, BASE + 4);
                assert_eq!(executed, 2);
            }
            outcome => panic!("unexpected outcome {outcome:?}"),
        }
        assert_eq!(cpu.gpr[8], 5);
    }

    #[test]
    fn fault_reports_partial_progress() {
        // lw from an unmapped address faults after the addiu completes.
        let words = [0x2408_0005, 0x8c09_9000];
        let mut memory = memory_with(&words);
        let mut cpu = Cpu::new(BASE);
        let mut jit = Jit::default();
        match jit.step_guest(&mut cpu, &mut memory, 1, 16) {
            Err(BlockFault { executed, .. }) => assert_eq!(executed, 1),
            outcome => panic!("unexpected outcome {outcome:?}"),
        }
        assert_eq!(cpu.gpr[8], 5);
        assert_eq!(cpu.pc, BASE + 8);
    }

    #[test]
    fn self_modifying_code_recompiles() {
        // two addius at base (rest of the page is nops); run exactly two
        // ops, overwrite the second word, and re-run: the new immediate
        // must win via page-generation invalidation.
        let mut memory = memory_with(&[0x2408_0005, 0x2409_0007]);
        let mut jit = Jit::default();
        let mut cpu = Cpu::new(BASE);
        match jit.step_guest(&mut cpu, &mut memory, 1, 2) {
            Ok(BlockOutcome::Flowing(2)) => {}
            outcome => panic!("unexpected outcome {outcome:?}"),
        }
        assert_eq!(cpu.gpr[9], 7);
        assert_eq!(jit.compiled(), 1);
        memory.write_u32(BASE + 4, 0x2409_0009).unwrap();
        let mut cpu = Cpu::new(BASE);
        match jit.step_guest(&mut cpu, &mut memory, 1, 2) {
            Ok(BlockOutcome::Flowing(2)) => {}
            outcome => panic!("unexpected outcome {outcome:?}"),
        }
        assert_eq!(cpu.gpr[9], 9);
        assert_eq!(jit.compiled(), 2);
        assert!(jit.invalidated() >= 1);
    }

    #[test]
    fn stores_redecode_later_instructions_in_the_same_block() {
        // sw t0, 4(t1); addiu t2, zero, 1
        let words = [0xad28_0004, 0x240a_0001];
        let mut memory = memory_with(&words);
        let mut cpu = Cpu::new(BASE);
        cpu.gpr[8] = 0x240a_0009;
        cpu.gpr[9] = BASE;
        let mut jit = Jit::default();
        assert!(matches!(
            jit.step_guest(&mut cpu, &mut memory, 1, 2),
            Ok(BlockOutcome::Flowing(2))
        ));
        assert_eq!(cpu.gpr[10], 9);
        assert_eq!(cpu.instruction_count, 2);
    }

    #[test]
    fn aliased_code_is_invalidated_by_writes_to_physical_ram() {
        let mut memory = Memory::default();
        memory.map(0x0800_0000, 4096, true, true).unwrap();
        memory.write_u32(0x0800_0000, 0x2408_0001).unwrap();
        let mut jit = Jit::default();
        for value in [1, 9] {
            memory.write_u32(0x0800_0000, 0x2408_0000 | value).unwrap();
            let mut cpu = Cpu::new(0x8800_0000);
            jit.step_guest(&mut cpu, &mut memory, 1, 1).unwrap();
            assert_eq!(cpu.gpr[8], value);
        }
        assert_eq!(jit.invalidated(), 1);
    }

    #[test]
    fn budget_caps_execution() {
        let words = [0x2408_0005, 0x2409_0007, 0x240a_0009];
        let mut memory = memory_with(&words);
        let mut cpu = Cpu::new(BASE);
        let mut jit = Jit::default();
        match jit.step_guest(&mut cpu, &mut memory, 1, 2) {
            Ok(BlockOutcome::Flowing(2)) => {}
            outcome => panic!("unexpected outcome {outcome:?}"),
        }
        assert_eq!(cpu.pc, BASE + 8);
        match jit.step_guest(&mut cpu, &mut memory, 1, 8) {
            Ok(BlockOutcome::Flowing(executed)) => assert!(executed >= 1),
            outcome => panic!("unexpected outcome {outcome:?}"),
        }
    }

    /// Deterministic xorshift generator with no external dependencies.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, bound: u64) -> u32 {
            (self.next() % bound) as u32
        }
    }

    /// Random straight-line programs over ALU, memory, and VFPU operations must
    /// execute identically under both engines.
    #[test]
    fn differential_fuzz_matches_interpreter() {
        // scratch data page for loads/stores.
        const OPS: usize = 60;
        for seed in [1u64, 7, 42, 1234, 99991] {
            let mut rng = Rng(seed);
            let mut words = Vec::new();
            for _ in 0..OPS {
                let choice = rng.next() % 10;
                let word: u32 = match choice {
                    // addiu rt, rs, imm with nonzero registers.
                    0 => 0x2400_0000 | ((rng.below(31) + 1) << 16) | (rng.below(0x1_0000)),
                    // addu rd, rs, rt with nonzero registers.
                    1 => {
                        0x21 | ((rng.below(31) + 1) << 21)
                            | ((rng.below(31) + 1) << 16)
                            | ((rng.below(31) + 1) << 11)
                    }
                    // sw/lw within scratch (0x2000 page, 16 words).
                    2 => {
                        let store = rng.next().is_multiple_of(2);
                        let base = 0x2000 + rng.below(16) * 4;
                        let rt = (rng.below(31) + 1) << 16;
                        let op = if store { 0x2bu32 } else { 0x23 };
                        (op << 26) | rt | (base & 0xffff)
                    }
                    // sll/srl by small amounts.
                    3 => {
                        let op = if rng.next().is_multiple_of(2) {
                            0x00u32
                        } else {
                            0x02
                        };
                        op | ((rng.below(31) + 1) << 16)
                            | ((rng.below(31) + 1) << 11)
                            | (rng.below(16) << 6)
                    }
                    // vadd.q / vmul.q over c000/c100/c200.
                    4 => {
                        let vmul = rng.next().is_multiple_of(2);
                        let op = if vmul { 0x19u32 } else { 0x18 };
                        (op << 26)
                            | (1 << 7)
                            | (1 << 15)
                            | (rng.below(3) * 4)
                            | ((rng.below(3) * 4) << 8)
                            | ((rng.below(3) * 4) << 16)
                    }
                    // vdot.q / vhdp.q.
                    5 => {
                        let sub = if rng.next().is_multiple_of(2) {
                            1u32
                        } else {
                            4
                        };
                        (0x19 << 26)
                            | (sub << 23)
                            | (1 << 7)
                            | (1 << 15)
                            | (rng.below(3) * 4)
                            | ((rng.below(3) * 4) << 8)
                            | ((rng.below(3) * 4) << 16)
                    }
                    // andi/ori/xori/lui immediates.
                    6 => {
                        let op = 0x0cu32 + rng.below(4);
                        (op << 26)
                            | ((rng.below(31) + 1) << 21)
                            | ((rng.below(31) + 1) << 16)
                            | rng.below(0x1_0000)
                    }
                    // slti/sltiu.
                    7 => {
                        let op = if rng.next().is_multiple_of(2) {
                            0x0au32
                        } else {
                            0x0b
                        };
                        (op << 26)
                            | ((rng.below(31) + 1) << 21)
                            | ((rng.below(31) + 1) << 16)
                            | rng.below(0x1_0000)
                    }
                    // mult/mflo dance.
                    8 => {
                        if rng.next().is_multiple_of(2) {
                            0x18 | ((rng.below(31) + 1) << 21) | ((rng.below(31) + 1) << 16)
                        } else {
                            0x12 | ((rng.below(31) + 1) << 11)
                        }
                    }
                    // vtfm4 / vmmul 3x3 over low matrices.
                    _ => {
                        (0x3c << 26)
                            | (1 << 15)
                            | (rng.below(3) * 4)
                            | ((rng.below(3) * 4) << 8)
                            | ((rng.below(3) * 4) << 16)
                    }
                };
                words.push(word);
            }
            // interpreter reference with seeded register/vfpu state.
            let mut memory = memory_with(&words);
            memory.map(0x2000, 0x100, true, false).unwrap();
            let mut reference = Cpu::new(BASE);
            let mut rng_state = Rng(seed ^ 0x9e37);
            for reg in 1..32 {
                reference.gpr[reg] = (rng_state.next() & 0xffff_ffff) as u32;
            }
            for lane in 0..128 {
                let bits = (rng_state.next() & 0xffff_ffff) as u32;
                // keep floats mostly finite for stable comparison.
                reference.vfpu[lane] = if bits & 0x7f80_0000 == 0x7f80_0000 {
                    bits & 0x3f80_0000
                } else {
                    bits
                };
            }
            for _ in &words {
                reference.step(&mut memory).unwrap();
            }
            let reference_memory = memory.read_bytes(0x2000, 0x40).unwrap();
            // jit under test with identical start state (reseed the same
            // stream the reference consumed).
            let mut memory = memory_with(&words);
            memory.map(0x2000, 0x100, true, false).unwrap();
            let mut cpu = Cpu::new(BASE);
            let mut rng_state = Rng(seed ^ 0x9e37);
            for reg in 1..32 {
                cpu.gpr[reg] = (rng_state.next() & 0xffff_ffff) as u32;
            }
            for lane in 0..128 {
                let bits = (rng_state.next() & 0xffff_ffff) as u32;
                cpu.vfpu[lane] = if bits & 0x7f80_0000 == 0x7f80_0000 {
                    bits & 0x3f80_0000
                } else {
                    bits
                };
            }
            let mut jit = Jit::default();
            let mut remaining = words.len() as u64;
            while remaining > 0 {
                match jit.step_guest(&mut cpu, &mut memory, 1, remaining) {
                    Ok(BlockOutcome::Flowing(executed)) => {
                        assert!(executed > 0 && executed <= remaining);
                        remaining -= executed;
                    }
                    outcome => panic!("seed {seed}: unexpected outcome {outcome:?}"),
                }
            }
            assert_eq!(cpu_state(&cpu), cpu_state(&reference), "seed {seed}");
            assert_eq!(
                memory.read_bytes(0x2000, 0x40).unwrap(),
                reference_memory,
                "seed {seed}"
            );
        }
    }

    fn seed_state(cpu: &mut Cpu, seed: u64) {
        let mut rng_state = Rng(seed ^ 0x9e37);
        for reg in 1..32 {
            cpu.gpr[reg] = (rng_state.next() & 0xffff_ffff) as u32;
        }
        cpu.hi = (rng_state.next() & 0xffff_ffff) as u32;
        cpu.lo = (rng_state.next() & 0xffff_ffff) as u32;
        cpu.fcr31 = (rng_state.next() & 0x0080_0000) as u32;
        cpu.vfpu_cc = (rng_state.next() & 0x3f) as u32;
        for lane in 0..128 {
            let bits = (rng_state.next() & 0xffff_ffff) as u32;
            cpu.vfpu[lane] = if bits & 0x7f80_0000 == 0x7f80_0000 {
                bits & 0x3f80_0000
            } else {
                bits
            };
        }
        for reg in 0..32 {
            cpu.fpr[reg] = (rng_state.next() & 0xffff_ffff) as u32;
        }
    }

    fn full_state(cpu: &Cpu) -> Vec<u32> {
        let mut state = cpu.gpr.to_vec();
        state.extend_from_slice(&[
            cpu.pc,
            cpu.hi,
            cpu.lo,
            cpu.fcr31,
            cpu.vfpu_cc,
            cpu.vfpu_s_prefix,
            cpu.vfpu_t_prefix,
            cpu.vfpu_d_prefix,
        ]);
        state.extend_from_slice(&cpu.vfpu);
        state.extend_from_slice(&cpu.fpr);
        state
    }

    /// Random programs with control flow (forward/backward branches, jumps,
    /// calls), coprocessor operations, and unaligned memory traffic must execute
    /// identically under both engines.
    #[test]
    fn differential_fuzz_with_branches() {
        const OPS: usize = 80;
        const STEPS: u64 = 400;
        for seed in [3u64, 11, 77, 5555, 424242] {
            let mut rng = Rng(seed);
            let mut words = Vec::new();
            for i in 0..OPS {
                let choice = rng.next() % 14;
                let word: u32 = match choice {
                    // beq/bne with an in-range target (forward or back).
                    0 | 1 => {
                        let op = if choice == 0 { 0x04u32 } else { 0x05 };
                        let target = rng.below((OPS + 8) as u64);
                        let from = i as u32 + 1;
                        let offset = target.wrapping_sub(from) & 0xffff;
                        (op << 26)
                            | ((rng.below(31) + 1) << 21)
                            | ((rng.below(31) + 1) << 16)
                            | offset
                    }
                    // blez/bgtz with an in-range target.
                    2 => {
                        let op = if rng.next().is_multiple_of(2) {
                            0x06u32
                        } else {
                            0x07
                        };
                        let target = rng.below((OPS + 8) as u64);
                        let from = i as u32 + 1;
                        let offset = target.wrapping_sub(from) & 0xffff;
                        (op << 26) | ((rng.below(31) + 1) << 21) | offset
                    }
                    // j / jal to an in-range word.
                    3 => {
                        let op = if rng.next().is_multiple_of(2) {
                            0x02u32
                        } else {
                            0x03
                        };
                        let target = BASE + rng.below((OPS + 8) as u64) * 4;
                        (op << 26) | ((target >> 2) & 0x03ff_ffff)
                    }
                    // lwl/lwr/swl/swr at scratch (any alignment works).
                    4 => {
                        let op = [0x22u32, 0x26, 0x2a, 0x2e][rng.below(4) as usize];
                        let addr = 0x2000 + rng.below(64);
                        (op << 26) | ((rng.below(31) + 1) << 16) | (addr & 0xffff)
                    }
                    // sb/sh/lbu/lhu within scratch (halfwords stay aligned).
                    5 => {
                        let half = rng.next().is_multiple_of(2);
                        let op = if half {
                            [0x29u32, 0x25][rng.below(2) as usize]
                        } else {
                            [0x28u32, 0x24][rng.below(2) as usize]
                        };
                        let mut addr = 0x2000 + rng.below(64);
                        if half {
                            addr &= !1;
                        }
                        (op << 26) | ((rng.below(31) + 1) << 16) | (addr & 0xffff)
                    }
                    // allegrex bit ops and hilo moves (always valid encodings).
                    6 => match rng.below(7) {
                        0 => {
                            // seb rt, rd.
                            ((rng.below(31) + 1) << 16)
                                | ((rng.below(31) + 1) << 11)
                                | (16 << 6)
                                | 0x20
                                | (0x1f << 26)
                        }
                        1 => {
                            // wsbh rd, rt.
                            ((rng.below(31) + 1) << 16)
                                | ((rng.below(31) + 1) << 11)
                                | (2 << 6)
                                | 0x20
                                | (0x1f << 26)
                        }
                        2 => {
                            // ext rt, rs, msb>=lsb by construction.
                            let lsb = rng.below(32);
                            let msb = lsb + rng.below(u64::from(32 - lsb));
                            (0x1f << 26)
                                | ((rng.below(31) + 1) << 21)
                                | ((rng.below(31) + 1) << 16)
                                | (msb << 11)
                                | (lsb << 6)
                        }
                        3 => {
                            // clz rd, rs.
                            0x16 | ((rng.below(31) + 1) << 21) | ((rng.below(31) + 1) << 11)
                        }
                        4 => {
                            // mflo rd.
                            0x12 | ((rng.below(31) + 1) << 11)
                        }
                        5 => {
                            // mult rs, rt.
                            0x18 | ((rng.below(31) + 1) << 21) | ((rng.below(31) + 1) << 16)
                        }
                        _ => {
                            // div rs, rt (never faults by design).
                            0x1a | ((rng.below(31) + 1) << 21) | ((rng.below(31) + 1) << 16)
                        }
                    },
                    // special2 mul / madd-style accumulate.
                    7 => {
                        let funct = [0x02u32, 0x00, 0x01, 0x04, 0x05][rng.below(5) as usize];
                        (0x1c << 26)
                            | ((rng.below(31) + 1) << 21)
                            | ((rng.below(31) + 1) << 16)
                            | ((rng.below(31) + 1) << 11)
                            | funct
                    }
                    // ll/sc on scratch.
                    8 => {
                        let op = if rng.next().is_multiple_of(2) {
                            0x30u32
                        } else {
                            0x38
                        };
                        let addr = 0x2000 + (rng.below(16) * 4);
                        (op << 26) | ((rng.below(31) + 1) << 16) | (addr & 0xffff)
                    }
                    // cop1x madd.s / msub.s over f0-f7.
                    9 => {
                        let funct = if rng.next().is_multiple_of(2) {
                            0x20u32
                        } else {
                            0x21
                        };
                        (0x13 << 26)
                            | (rng.below(8) << 21)
                            | (rng.below(8) << 16)
                            | (rng.below(8) << 11)
                            | (rng.below(8) << 6)
                            | funct
                    }
                    // fpu arith / converts over f0-f7.
                    10 => {
                        let funct = [0u32, 1, 2, 3, 5, 6, 7, 0x24][rng.below(8) as usize];
                        (0x11 << 26)
                            | (16 << 21)
                            | (rng.below(8) << 16)
                            | (rng.below(8) << 11)
                            | (rng.below(8) << 6)
                            | funct
                    }
                    // vfpu variety: vcst, vcmp, vcmov, vrot, vscl, vhdp.
                    11 => {
                        let kind = rng.below(6);
                        let vd = rng.below(3) * 4;
                        let vs = rng.below(3) * 4;
                        let vt = rng.below(3) * 4;
                        match kind {
                            0 => {
                                (0x34 << 26)
                                    | (3 << 21)
                                    | ((rng.below(20)) << 16)
                                    | (1 << 7)
                                    | (1 << 15)
                                    | vd
                            }
                            1 => {
                                (0x1b << 26)
                                    | (1 << 7)
                                    | (1 << 15)
                                    | vd
                                    | (vs << 8)
                                    | (vt << 16)
                                    | rng.below(16)
                            }
                            2 => {
                                (0x34 << 26)
                                    | (21 << 21)
                                    | (rng.below(7) << 16)
                                    | (1 << 7)
                                    | (1 << 15)
                                    | vd
                                    | (vs << 8)
                            }
                            3 => {
                                (0x3c << 26)
                                    | (29 << 21)
                                    | (1 << 7)
                                    | (1 << 15)
                                    | vd
                                    | (vs << 8)
                                    | (rng.below(32) << 16)
                            }
                            4 => {
                                (0x19 << 26)
                                    | (2 << 23)
                                    | (1 << 7)
                                    | (1 << 15)
                                    | vd
                                    | (vs << 8)
                                    | (vt << 16)
                            }
                            _ => {
                                (0x19 << 26)
                                    | (4 << 23)
                                    | (1 << 7)
                                    | (1 << 15)
                                    | vd
                                    | (vs << 8)
                                    | (vt << 16)
                            }
                        }
                    }
                    // vtfm4 / vmmul 2x2 over low matrices.
                    12 => {
                        if rng.next().is_multiple_of(2) {
                            (0x3c << 26)
                                | ((12 + rng.below(4)) << 21)
                                | (1 << 7)
                                | (1 << 15)
                                | (rng.below(3) * 4)
                                | ((rng.below(3) * 4) << 8)
                                | ((rng.below(3) * 4) << 16)
                        } else {
                            (0x3c << 26)
                                | (1 << 15)
                                | (rng.below(3) * 4)
                                | ((rng.below(3) * 4) << 8)
                                | ((rng.below(3) * 4) << 16)
                        }
                    }
                    // addu/subu/or/nor (wrapping, trap-free).
                    _ => {
                        let funct = [0x21u32, 0x23, 0x25, 0x27][rng.below(4) as usize];
                        funct
                            | ((rng.below(31) + 1) << 21)
                            | ((rng.below(31) + 1) << 16)
                            | ((rng.below(31) + 1) << 11)
                    }
                };
                words.push(word);
            }
            // pad with nops so stray branches land on mapped code.
            words.extend(std::iter::repeat_n(0, 16));
            // interpreter reference: fixed op count from identical state.
            let mut memory = memory_with(&words);
            memory.map(0x2000, 0x100, true, false).unwrap();
            let mut reference = Cpu::new(BASE);
            seed_state(&mut reference, seed);
            for _ in 0..STEPS {
                reference.step(&mut memory).unwrap();
            }
            let reference_memory = memory.read_bytes(0x2000, 0x40).unwrap();
            let reference_state = full_state(&reference);
            // jit under test.
            let mut memory = memory_with(&words);
            memory.map(0x2000, 0x100, true, false).unwrap();
            let mut cpu = Cpu::new(BASE);
            seed_state(&mut cpu, seed);
            let mut jit = Jit::default();
            let mut remaining = STEPS;
            while remaining > 0 {
                match jit.step_guest(&mut cpu, &mut memory, 1, remaining) {
                    Ok(BlockOutcome::Flowing(executed)) => {
                        assert!(executed > 0 && executed <= remaining);
                        remaining -= executed;
                    }
                    outcome => panic!("seed {seed}: unexpected outcome {outcome:?}"),
                }
            }
            assert_eq!(full_state(&cpu), reference_state, "seed {seed}");
            assert_eq!(
                memory.read_bytes(0x2000, 0x40).unwrap(),
                reference_memory,
                "seed {seed}"
            );
        }
    }
}
