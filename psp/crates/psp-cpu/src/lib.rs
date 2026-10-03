//! Allegrex CPU state and instruction execution.
//!
//! The interpreter and block JIT share the same decoded opcode handlers. This
//! keeps the fast path honest: the JIT can cache fetch/decode work, but all
//! guest-visible state changes still go through the CPU's instruction semantics.

use psp_memory::{GuestAddress, Memory, MemoryFault};
use thiserror::Error;

mod exec;
pub mod jit;

pub(crate) use exec::{Decoded, Handler, handler_for};
pub use jit::{BlockFault, BlockOutcome, Jit};

#[derive(Clone, Debug)]
/// Architectural state of the Allegrex CPU and PSP VFPU.
pub struct Cpu {
    /// Thirty-two 32-bit general-purpose registers. Register zero is restored
    /// to zero after every successfully executed instruction.
    pub gpr: [u32; 32],
    /// Address of the next instruction to fetch.
    pub pc: u32,
    /// High half of the multiply/divide accumulator.
    pub hi: u32,
    /// Low half of the multiply/divide accumulator.
    pub lo: u32,
    /// Thirty-two scalar floating-point registers, stored as raw IEEE bits.
    pub fpr: [u32; 32],
    /// Scalar FPU control/status register, including the condition bit.
    pub fcr31: u32,
    /// The VFPU's 128 physical 32-bit lanes.
    pub vfpu: [u32; 128],
    /// VFPU condition codes: x/y/z/w in bits 0..3, any in bit 4, and all in
    /// bit 5.
    pub vfpu_cc: u32,
    /// Pending VFPU source prefix consumed by the next vector instruction.
    pub vfpu_s_prefix: u32,
    /// Pending VFPU second-source prefix consumed by the next vector instruction.
    pub vfpu_t_prefix: u32,
    /// Pending VFPU destination prefix consumed by the next vector instruction.
    pub vfpu_d_prefix: u32,
    /// Upper VFPU control registers (indices 4..15), including the revision
    /// value and the eight random-generator registers. Prefixes and condition
    /// codes live in the dedicated fields above so `mtvc`, `vmtvc`, and the
    /// `vrnd*` instructions observe one consistent control file.
    pub vfpu_ctrl_extra: [u32; 12],
    /// Number of guest instructions completed by this CPU.
    pub instruction_count: u64,
    pub(crate) branch_target: Option<u32>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            gpr: [0; 32],
            pc: 0,
            hi: 0,
            lo: 0,
            fpr: [0; 32],
            fcr31: 0,
            vfpu: [0; 128],
            vfpu_cc: 0x3f,
            vfpu_s_prefix: 0xe4,
            vfpu_t_prefix: 0xe4,
            vfpu_d_prefix: 0,
            // default vfpu random seed state.
            vfpu_ctrl_extra: [0, 0, 0, 0, 1, 2, 4, 8, 0, 0, 0, 0],
            instruction_count: 0,
            branch_target: None,
        }
    }
}

#[derive(Debug, Error)]
/// Errors raised while fetching or executing one guest instruction.
pub enum CpuError {
    #[error(transparent)]
    /// A checked guest-memory access failed.
    Memory(#[from] MemoryFault),
    #[error("unsupported instruction 0x{word:08x} at {pc}")]
    /// The instruction decoder has no handler for this word.
    Unsupported {
        /// Guest program counter of the unsupported word.
        pc: GuestAddress,
        /// Raw instruction word that could not be executed.
        word: u32,
    },
    #[error("syscall {code} at {pc}")]
    /// A syscall was raised while using an API that reports it as an error.
    Syscall {
        /// Guest program counter of the syscall instruction.
        pc: GuestAddress,
        /// Emulator-private syscall number.
        code: u32,
    },
    #[error("arithmetic overflow at {0}")]
    /// A host-side counter or fast-path calculation would overflow.
    Overflow(GuestAddress),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// The result of one executed instruction.
pub enum Step {
    /// The instruction completed and execution can continue.
    Continue,
    /// The guest reached an emulator-private syscall stub.
    Syscall(u32),
}

impl Cpu {
    /// Create a reset CPU whose first fetch will use `entry`.
    pub fn new(entry: u32) -> Self {
        Self {
            pc: entry,
            ..Self::default()
        }
    }

    /// Fast-forward the canonical byte-at-a-time memory-fill loop emitted by
    /// PSP C libraries.
    ///
    /// This is deliberately pattern based: it accelerates a general Allegrex
    /// program without depending on a title-specific address or replacing the
    /// normal interpreter for arbitrary code.
    ///
    /// The loop has six instructions: copy the remaining count, store one byte,
    /// decrement the source count, advance the destination, branch back when
    /// the copied count is non-zero, and update the count in the delay slot.
    /// The host fill has the same permission and range checks as individual
    /// `sb` instructions.
    pub fn try_fast_forward_memory_loop(
        &mut self,
        memory: &mut Memory,
        thread_id: u32,
    ) -> Result<Option<u64>, CpuError> {
        if self.branch_target.is_some() {
            return Ok(None);
        }
        let pc = self.pc;
        let Ok(move_word) = memory.fetch_u32(pc) else {
            return Ok(None);
        };

        if move_word >> 26 != 0
            || move_word & 63 != 0x25
            || (move_word >> 16) & 31 != 0
            || (move_word >> 6) & 31 != 0
        {
            return Ok(None);
        }
        let count_reg = ((move_word >> 11) & 31) as usize;
        let decrement_reg = ((move_word >> 21) & 31) as usize;

        let Ok(store_word) = memory.fetch_u32(pc.wrapping_add(4)) else {
            return Ok(None);
        };
        if store_word >> 26 != 0x28 {
            return Ok(None);
        }
        let pointer_reg = ((store_word >> 21) & 31) as usize;
        let value_reg = ((store_word >> 16) & 31) as usize;
        let store_offset = store_word as i16 as i32 as u32;

        let Ok(decrement_word) = memory.fetch_u32(pc.wrapping_add(8)) else {
            return Ok(None);
        };
        if decrement_word >> 26 != 9
            || ((decrement_word >> 16) & 31) as usize != decrement_reg
            || decrement_word as i16 != -1
        {
            return Ok(None);
        }
        let source_reg = ((decrement_word >> 21) & 31) as usize;

        let Ok(advance_word) = memory.fetch_u32(pc.wrapping_add(12)) else {
            return Ok(None);
        };
        if advance_word >> 26 != 9
            || ((advance_word >> 21) & 31) as usize != pointer_reg
            || ((advance_word >> 16) & 31) as usize != pointer_reg
            || advance_word as i16 != 1
        {
            return Ok(None);
        }

        let Ok(branch_word) = memory.fetch_u32(pc.wrapping_add(16)) else {
            return Ok(None);
        };
        if branch_word >> 26 != 5
            || ((branch_word >> 21) & 31) as usize != count_reg
            || ((branch_word >> 16) & 31) != 0
            || (branch_word as i16) != -5
        {
            return Ok(None);
        }
        let branch_target = pc
            .wrapping_add(20)
            .wrapping_add(((branch_word as i16 as i32) << 2) as u32);
        if branch_target != pc {
            return Ok(None);
        }

        let Ok(delay_word) = memory.fetch_u32(pc.wrapping_add(20)) else {
            return Ok(None);
        };
        if delay_word >> 26 != 0
            || delay_word & 63 != 0x25
            || (delay_word >> 16) & 31 != 0
            || (delay_word >> 6) & 31 != 0
            || ((delay_word >> 11) & 31) as usize != source_reg
            || ((delay_word >> 21) & 31) as usize != decrement_reg
        {
            return Ok(None);
        }

        let registers = [count_reg, decrement_reg, source_reg, pointer_reg, value_reg];
        if registers.contains(&0)
            || registers
                .iter()
                .enumerate()
                .any(|(index, &register)| registers[..index].contains(&register))
        {
            return Ok(None);
        }

        // A u32 maximum decrement value would mean 2^32 stores. Leave that
        // pathological case in the interpreter rather than overflowing the
        // guest pointer or the host range length.
        let count = u64::from(self.gpr[decrement_reg]) + 1;
        if count > u64::from(u32::MAX) {
            return Ok(None);
        }
        let count = count as u32;
        let start = self.gpr[pointer_reg].wrapping_add(store_offset);
        if start.checked_add(count).is_none() {
            return Ok(None);
        }
        let final_count = self.gpr[source_reg].wrapping_sub(count);
        let executed = u64::from(count)
            .checked_mul(6)
            .ok_or(CpuError::Overflow(GuestAddress(pc)))?;
        let new_instruction_count = self
            .instruction_count
            .checked_add(executed)
            .ok_or(CpuError::Overflow(GuestAddress(pc)))?;

        memory.set_context(pc, thread_id);
        memory.fill_bytes(start, count as usize, self.gpr[value_reg] as u8)?;
        self.gpr[count_reg] = 0;
        self.gpr[decrement_reg] = final_count;
        self.gpr[source_reg] = final_count;
        self.gpr[pointer_reg] = self.gpr[pointer_reg].wrapping_add(count);
        self.pc = pc.wrapping_add(24);
        self.instruction_count = new_instruction_count;
        self.gpr[0] = 0;
        Ok(Some(executed))
    }

    pub(crate) fn reset_vfpu_prefixes(&mut self) {
        self.vfpu_s_prefix = VFPU_DEFAULT_SOURCE_PREFIX;
        self.vfpu_t_prefix = VFPU_DEFAULT_SOURCE_PREFIX;
        self.vfpu_d_prefix = 0;
    }

    /// Fetch and execute one instruction using the default guest thread id.
    pub fn step(&mut self, memory: &mut Memory) -> Result<Step, CpuError> {
        self.step_with_thread_id(memory, 1)
    }

    /// Fetch and execute one instruction while attaching `thread_id` to faults.
    pub fn step_with_thread_id(
        &mut self,
        memory: &mut Memory,
        thread_id: u32,
    ) -> Result<Step, CpuError> {
        memory.set_context(self.pc, thread_id);
        let pc = self.pc;
        let word = memory.fetch_u32(pc)?;
        self.execute_word(memory, pc, word, thread_id)
    }

    /// Execute one already-fetched instruction word.
    ///
    /// This is the single semantic core shared by the single-step interpreter
    /// and the block JIT: both observe identical guest state transitions,
    /// including delay slots, fault reporting, and `$zero` behavior.
    pub fn execute_word(
        &mut self,
        memory: &mut Memory,
        pc: u32,
        word: u32,
        thread_id: u32,
    ) -> Result<Step, CpuError> {
        let decoded = Decoded::decode(pc, word);
        self.execute_predecoded(memory, decoded, handler_for(word), thread_id)
    }

    /// Execute one pre-decoded instruction with its resolved handler.
    ///
    /// This is identical to [`Self::execute_word`] after fetch, decode, and
    /// dispatch have already been done—the work cached by the block JIT.
    pub(crate) fn execute_predecoded(
        &mut self,
        memory: &mut Memory,
        decoded: Decoded,
        handler: Handler,
        thread_id: u32,
    ) -> Result<Step, CpuError> {
        memory.set_context(decoded.pc, thread_id);
        let pending = self.branch_target.take();
        self.pc = decoded.pc.wrapping_add(4);
        let result = handler(self, memory, decoded)?;
        self.gpr[0] = 0;
        self.instruction_count += 1;
        if let Some(target) = pending {
            self.pc = target;
        }
        Ok(result)
    }
}

pub(crate) const VFPU_DEFAULT_SOURCE_PREFIX: u32 = 0xe4;

pub(crate) fn vfpu_source_values(
    registers: &[u32; 128],
    lanes: [usize; 4],
    length: usize,
    prefix: u32,
) -> [u32; 4] {
    let mut original = [0u32; 4];
    for (index, &lane) in lanes.iter().take(length).enumerate() {
        original[index] = registers[lane];
    }
    if prefix == VFPU_DEFAULT_SOURCE_PREFIX {
        return original;
    }

    // the source prefix is four four-bit controls spread across three
    // bit-fields: two swizzle bits, one constant bit, one absolute bit, and
    // one negate bit for each lane.  constants selected with the absolute
    // bit are the alternate constant bank used by the psp vfpu.
    const CONSTANTS: [f32; 8] = [0.0, 1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 0.25, 1.0 / 6.0];
    let mut values = [0u32; 4];
    for (index, value) in values.iter_mut().take(length).enumerate() {
        let register = ((prefix >> (index * 2)) & 3) as usize;
        let absolute = (prefix >> (8 + index)) & 1 != 0;
        let constant = (prefix >> (12 + index)) & 1 != 0;
        let mut lane_value = if constant {
            CONSTANTS[register + usize::from(absolute) * 4].to_bits()
        } else if register < length {
            original[register]
        } else {
            // hardware swizzles outside the source vector as an invalid
            // lane. for the arithmetic operations implemented here that
            // lane is wired as zero; keeping it explicit avoids reading a
            // neighbouring physical vfpu register.
            0
        };
        if absolute && !constant {
            lane_value &= 0x7fff_ffff;
        }
        if (prefix >> (16 + index)) & 1 != 0 {
            lane_value ^= 0x8000_0000;
        }
        *value = lane_value;
    }
    values
}

pub(crate) fn vfpu_clamp(value: f32, lower: f32, upper: f32) -> f32 {
    if value >= upper {
        upper
    } else if value <= lower {
        lower
    } else {
        value
    }
}

pub(crate) fn vfpu_write_vector(
    registers: &mut [u32; 128],
    lanes: [usize; 4],
    length: usize,
    mut values: [u32; 4],
    prefix: u32,
    apply_saturation: bool,
) {
    if apply_saturation {
        for (index, value) in values.iter_mut().take(length).enumerate() {
            let saturation = (prefix >> (index * 2)) & 3;
            let value_f32 = f32::from_bits(*value);
            *value = match saturation {
                1 => vfpu_clamp(value_f32, 0.0, 1.0).to_bits(),
                3 => vfpu_clamp(value_f32, -1.0, 1.0).to_bits(),
                _ => *value,
            };
        }
    }
    for (index, (&lane, &value)) in lanes.iter().zip(values.iter()).take(length).enumerate() {
        // a set destination-mask bit suppresses the corresponding write.
        if prefix & (1 << (8 + index)) == 0 {
            registers[lane] = value;
        }
    }
}

pub(crate) fn vfpu_vector_lanes(register: u32, length: usize) -> [usize; 4] {
    let matrix = ((register >> 2) & 7) as usize;
    let column = (register & 3) as usize;
    // scalar encodings reuse bit 5 for the row and never select a
    // transposed vector.  treating it as transpose for length one aliases
    // s001 with a different physical lane.
    let transpose = length != 1 && (register >> 5) & 1 != 0;
    let row = match length {
        2 => ((register >> 5) & 2) as usize,
        3 => ((register >> 6) & 1) as usize,
        4 => ((register >> 5) & 2) as usize,
        _ => ((register >> 5) & 3) as usize,
    };
    let mut lanes = [0; 4];
    for (index, lane) in lanes.iter_mut().enumerate().take(length) {
        *lane = if transpose {
            matrix * 16 + column + ((row + index) & 3) * 4
        } else {
            matrix * 16 + column * 4 + ((row + index) & 3)
        };
    }
    lanes
}

pub(crate) fn vfpu_matrix_lanes(register: u32, side: usize) -> [usize; 16] {
    let matrix = ((register >> 2) & 7) as usize;
    let column = (register & 3) as usize;
    let transpose = (register >> 5) & 1 != 0;
    let row = match side {
        2 => ((register >> 5) & 2) as usize,
        3 => ((register >> 6) & 1) as usize,
        4 => ((register >> 5) & 2) as usize,
        _ => ((register >> 5) & 3) as usize,
    };
    let mut lanes = [0; 16];
    for i in 0..side {
        for j in 0..side {
            lanes[j * 4 + i] = if transpose {
                matrix * 16 + ((column + j) & 3) + ((row + i) & 3) * 4
            } else {
                matrix * 16 + ((column + j) & 3) * 4 + ((row + i) & 3)
            };
        }
    }
    lanes
}

/// Write matrix lanes with the destination write mask gating the last column.
/// `values` uses the `rd[j * 4 + i]`
/// (column-major) layout.
pub(crate) fn vfpu_write_matrix(
    registers: &mut [u32; 128],
    register: u32,
    side: usize,
    values: &[u32; 16],
    d_prefix: u32,
) {
    let lanes = vfpu_matrix_lanes(register, side);
    for column in 0..side {
        for row in 0..side {
            if column + 1 != side || d_prefix & (1 << (8 + row)) == 0 {
                registers[lanes[column * 4 + row]] = values[column * 4 + row];
            }
        }
    }
}
pub(crate) fn vfpu_scalar_lane(register: u32) -> usize {
    let matrix = ((register >> 2) & 7) as usize;
    let column = (register & 3) as usize;
    let row = ((register >> 5) & 3) as usize;
    // scalar register encodings use the row bits above the low column bits,
    // while the host layout keeps each vfpu column contiguous.  inverting
    // these terms silently leaves holes when lv.s is followed by sv.q.
    matrix * 16 + column * 4 + row
}

pub(crate) fn expand_half(value: u16) -> u32 {
    let sign = (u32::from(value) & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let mantissa = u32::from(value & 0x03ff);
    match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let shift = mantissa.leading_zeros() - 21;
            let normalized = (mantissa << shift) & 0x03ff;
            let exponent32 = 113u32 - shift;
            sign | (exponent32 << 23) | (normalized << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    }
}

pub(crate) fn shrink_half(value: u32) -> u16 {
    let sign = ((value >> 16) & 0x8000) as u16;
    let exponent = ((value >> 23) & 0xff) as i32;
    let mantissa = value & 0x007f_ffff;
    if exponent == 255 {
        return sign
            | if mantissa == 0 {
                0x7c00
            } else {
                0x7c00 | ((mantissa >> 13) as u16).max(1)
            };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let rounded =
            (significand + (1 << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = mantissa + 0x0000_0fff + ((mantissa >> 13) & 1);
    let carry = rounded >> 23;
    let exponent = half_exponent as u16 + carry as u16;
    if exponent >= 31 {
        sign | 0x7c00
    } else {
        sign | (exponent << 10) | ((rounded >> 13) as u16 & 0x03ff)
    }
}

/// VFPU source-prefix bit constructors.
/// Lanes are ordered x=0, y=1, z=2, w=3.
pub(crate) const fn vfpu_swizzle(x: u32, y: u32, z: u32, w: u32) -> u32 {
    x | (y << 2) | (z << 4) | (w << 6)
}

pub(crate) const fn vfpu_lane_mask(x: u32, y: u32, z: u32, w: u32) -> u32 {
    x | (y << 1) | (z << 2) | (w << 3)
}

pub(crate) const fn vfpu_any_swizzle() -> u32 {
    0x0000_00ff
}

pub(crate) const fn vfpu_abs_bits(x: u32, y: u32, z: u32, w: u32) -> u32 {
    vfpu_lane_mask(x, y, z, w) << 8
}

pub(crate) const fn vfpu_negate_bits(x: u32, y: u32, z: u32, w: u32) -> u32 {
    vfpu_lane_mask(x, y, z, w) << 16
}

/// Constant-bank selectors for [`vfpu_make_constants`]: `-1` leaves the lane
/// untouched; otherwise the value is the 3-bit VFPU constant number (0..7).
pub(crate) fn vfpu_make_constants(x: i8, y: i8, z: i8, w: i8) -> u32 {
    let mut result = 0u32;
    for (index, constant) in [x, y, z, w].into_iter().enumerate() {
        if constant >= 0 {
            let constant = constant as u32;
            result |= ((constant & 3) << (index * 2))
                | (((constant & 4) >> 2) << (8 + index))
                | (1 << (12 + index));
        }
    }
    result
}

/// Compose a forced prefix over the guest prefix: clear `remove` bits, then
/// apply `add` with bitwise OR.
pub(crate) fn vfpu_rewrite_prefix(base: u32, remove: u32, add: u32) -> u32 {
    (base & !remove) | add
}

/// Apply a source prefix at quad width over already-read lane values.
/// Lanes beyond `valid` read as `+0.0`. Horizontal operations
/// (`vdot`, `vhdp`, `vfad`, `vavg`, `vdet`, `vf2h`) zero-fill their
/// scratch vectors before prefixing.
pub(crate) fn vfpu_prefix_quad(values: [u32; 4], valid: usize, prefix: u32) -> [u32; 4] {
    const CONSTANTS: [f32; 8] = [0.0, 1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 0.25, 1.0 / 6.0];
    let mut original = [0u32; 4];
    for (index, value) in original.iter_mut().enumerate().take(valid) {
        *value = values[index];
    }
    if prefix == VFPU_DEFAULT_SOURCE_PREFIX {
        return original;
    }
    let mut prefixed = [0u32; 4];
    for (index, value) in prefixed.iter_mut().enumerate() {
        let register = ((prefix >> (index * 2)) & 3) as usize;
        let absolute = (prefix >> (8 + index)) & 1 != 0;
        let constant = (prefix >> (12 + index)) & 1 != 0;
        let mut lane_value = if constant {
            CONSTANTS[register + usize::from(absolute) * 4].to_bits()
        } else {
            original[register]
        };
        if absolute && !constant {
            lane_value &= 0x7fff_ffff;
        }
        if (prefix >> (16 + index)) & 1 != 0 {
            lane_value ^= 0x8000_0000;
        }
        *value = lane_value;
    }
    prefixed
}

/// Zero destination lanes whose swizzle selects an out-of-range source lane.
/// Only some operations wire invalid lanes
/// through this way; arithmetic ops produce +0.0 via [`vfpu_source_values`].
pub(crate) fn vfpu_retain_invalid_swizzle(
    values: &mut [u32; 4],
    s_prefix: u32,
    t_prefix: u32,
    length: usize,
) {
    for (index, value) in values.iter_mut().enumerate().take(length) {
        let swizzle_s = (s_prefix >> (index * 2)) & 3;
        let swizzle_t = (t_prefix >> (index * 2)) & 3;
        let const_s = (s_prefix >> (12 + index)) & 1;
        let const_t = (t_prefix >> (12 + index)) & 1;
        if (swizzle_s >= length as u32 && const_s == 0)
            || (swizzle_t >= length as u32 && const_t == 0)
        {
            *value = 0;
        }
    }
}

/// Isolate the destination-prefix mask/saturation of lane 0 onto `lane`
/// (last-element handling in `vmmul`, `vtfm`, `vcrsp`/`vqmul`,
/// `vrndX`, and the vv2op transcendental group).
pub(crate) fn vfpu_isolate_prefix_lane(prefix: u32, lane: usize) -> u32 {
    ((prefix & 0x100) >> 8 << (8 + lane)) | ((prefix & 3) << (lane * 2))
}

/// Read a VFPU control register. Indices 0..2 are the prefixes, 3 is the
/// condition code register, and 4..15 are the extra file.
pub(crate) fn vfpu_read_ctrl(cpu: &Cpu, index: u32) -> u32 {
    match index {
        0 => cpu.vfpu_s_prefix,
        1 => cpu.vfpu_t_prefix,
        2 => cpu.vfpu_d_prefix,
        3 => cpu.vfpu_cc,
        4..=15 => cpu.vfpu_ctrl_extra[(index - 4) as usize],
        _ => 0,
    }
}

/// Write a VFPU control register through its hardware write mask. Returns
/// `false` for read-only or unknown registers.
pub(crate) fn vfpu_write_ctrl(cpu: &mut Cpu, index: u32, value: u32) -> bool {
    match index {
        0 | 1 => {
            if index == 0 {
                cpu.vfpu_s_prefix = value & 0x000f_ffff;
            } else {
                cpu.vfpu_t_prefix = value & 0x000f_ffff;
            }
            true
        }
        2 => {
            cpu.vfpu_d_prefix = value & 0x0000_0fff;
            true
        }
        3 => {
            cpu.vfpu_cc = value & 0x3f;
            true
        }
        4 => {
            cpu.vfpu_ctrl_extra[0] = value;
            true
        }
        // rsv5, rsv6, rev are read-only.
        5..=7 => false,
        8..=15 => {
            cpu.vfpu_ctrl_extra[(index - 4) as usize] = value & 0x3fff_ffff;
            true
        }
        _ => false,
    }
}

/// Seed the VFPU random generator.
pub(crate) fn vfpu_rng_seed(seed: u32, rcx: &mut [u32]) {
    for (index, slot) in rcx.iter_mut().enumerate().take(8) {
        *slot = 0x3f80_0000
            | ((seed >> ((index / 4) * 16)) & 0xffff)
            | (((seed >> (4 * index)) & 0xf) << 16);
    }
}

/// Advance the VFPU random generator and return its next word.
pub(crate) fn vfpu_rng_generate(rcx: &mut [u32]) -> u32 {
    let mut a = (rcx[0] & 0xffff) | (rcx[4] << 16);
    let mut b = (rcx[1] & 0xffff) | (rcx[5] << 16);
    let mut c = (rcx[2] & 0xffff) | (rcx[6] << 16);
    let mut d = (rcx[3] & 0xffff) | (rcx[7] << 16);
    let mut e = ((rcx[0] >> 16) & 0xf)
        | (((rcx[1] >> 16) & 0xf) << 4)
        | (((rcx[2] >> 16) & 0xf) << 8)
        | (((rcx[3] >> 16) & 0xf) << 12)
        | (((rcx[4] >> 16) & 0xf) << 16)
        | (((rcx[5] >> 16) & 0xf) << 20)
        | (((rcx[6] >> 16) & 0xf) << 24)
        | (((rcx[7] >> 16) & 0xf) << 28);
    a = 69069u32.wrapping_mul(a).wrapping_add(1);
    b ^= b << 13;
    b ^= b >> 17;
    b ^= b << 5;
    let t = 2u32.wrapping_mul(d).wrapping_add(c).wrapping_add(e);
    e = ((u64::from(c) + u64::from(d >> 1) + u64::from(e)) >> 32) as u32;
    c = d;
    d = t;
    rcx[0] = 0x3f80_0000 | ((e & 0xf) << 16) | (a & 0xffff);
    rcx[1] = 0x3f80_0000 | (((e >> 4) & 0xf) << 16) | (b & 0xffff);
    rcx[2] = 0x3f80_0000 | (((e >> 8) & 0xf) << 16) | (c & 0xffff);
    rcx[3] = 0x3f80_0000 | (((e >> 12) & 0xf) << 16) | (d & 0xffff);
    rcx[4] = 0x3f80_0000 | (((e >> 16) & 0xf) << 16) | (a >> 16);
    rcx[5] = 0x3f80_0000 | (((e >> 20) & 0xf) << 16) | (b >> 16);
    rcx[6] = 0x3f80_0000 | (((e >> 24) & 0xf) << 16) | (c >> 16);
    rcx[7] = 0x3f80_0000 | (((e >> 28) & 0xf) << 16) | (d >> 16);
    a.wrapping_add(b).wrapping_add(d)
}

/// Hardware-faithful reciprocal special cases: overflow flushes to signed zero,
/// denormals map to signed infinity, nan canonicalizes.
pub(crate) fn vfpu_recip_bits(input: u32) -> u32 {
    let sign = input & 0x8000_0000;
    let exponent = input & 0x7f80_0000;
    let mantissa = input & 0x007f_ffff;
    if (input & 0x7fff_ffff) > 0x7e80_0000 {
        if exponent == 0x7f80_0000 && mantissa != 0 {
            return sign ^ 0x7f80_0001;
        }
        return sign;
    }
    if exponent == 0 {
        return sign ^ 0x7f80_0000;
    }
    (1.0f32 / f32::from_bits(input)).to_bits()
}

/// `exp2` special cases:
/// denormals behave as zero input, nan canonicalizes, out-of-range
/// results flush to zero or infinity.
pub(crate) fn vfpu_exp2_value(input: f32) -> f32 {
    let bits = input.to_bits();
    if bits & 0x7fff_ffff <= 0x007f_ffff {
        return 1.0;
    }
    if input.is_nan() {
        return f32::from_bits(0x7f80_0001);
    }
    if input <= -126.0 {
        return 0.0;
    }
    if input >= 128.0 {
        return f32::INFINITY;
    }
    input.exp2()
}

/// Hardware-faithful `log2` special cases: denormals
/// and zero map to -inf, negatives to nan, infinities pass through.
pub(crate) fn vfpu_log2_value(input: f32) -> f32 {
    let bits = input.to_bits();
    if bits & 0x7fff_ffff <= 0x007f_ffff {
        return f32::NEG_INFINITY;
    }
    if bits & 0x8000_0000 != 0 {
        return f32::from_bits(0x7f80_0001);
    }
    if bits >> 23 == 255 {
        return input;
    }
    input.log2()
}

/// Hardware-faithful `asin` domain handling: inputs
/// outside [-1, 1] produce a signed canonical nan.
pub(crate) fn vfpu_asin_value(input: f32) -> f32 {
    if input.to_bits() & 0x7fff_ffff > 0x3f80_0000 {
        return f32::from_bits(0x7f80_0001 | (input.to_bits() & 0x8000_0000));
    }
    input.asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use psp_memory::GuestMemory;
    fn run(words: &[u32]) -> (Cpu, Memory) {
        let mut m = Memory::default();
        m.map(0x1000, 0x1000, true, true).unwrap();
        for (i, w) in words.iter().enumerate() {
            m.write_u32(0x1000 + i as u32 * 4, *w).unwrap()
        }
        let mut c = Cpu::new(0x1000);
        for _ in words {
            c.step(&mut m).unwrap();
        }
        (c, m)
    }
    #[test]
    fn arithmetic_and_zero() {
        let (c, _) = run(&[0x24010005, 0x2422fffe, 0x00221821, 0x2400ffff]);
        assert_eq!(c.gpr[3], 8);
        assert_eq!(c.gpr[0], 0)
    }
    #[test]
    fn branch_has_delay_slot() {
        let (c, _) = run(&[0x10000001, 0x24010007, 0x24020009]);
        assert_eq!(c.gpr[1], 7);
        assert_eq!(c.pc, 0x100c)
    }
    #[test]
    fn load_store() {
        let (c, m) = run(&[0x24011000, 0x3c021234, 0x34425678, 0xac220100, 0x8c230100]);
        assert_eq!(c.gpr[3], 0x12345678);
        assert_eq!(m.read_u32(0x1100).unwrap(), 0x12345678)
    }

    #[test]
    fn fast_forwards_canonical_byte_fill_loop() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        let loop_words = [
            0x00e0_4825, // move t1, a3
            0xa105_0000, // sb a1, 0(t0)
            0x24c7_ffff, // addiu a3, a2, -1
            0x2508_0001, // addiu t0, t0, 1
            0x1520_fffb, // bne t1, zero, loop
            0x00e0_3025, // move a2, a3 (delay slot)
        ];
        for (index, word) in loop_words.into_iter().enumerate() {
            memory.write_u32(0x1000 + index as u32 * 4, word).unwrap();
        }
        let mut cpu = Cpu::new(0x1000);
        cpu.gpr[5] = 0xa5;
        cpu.gpr[6] = 4;
        cpu.gpr[7] = 4;
        cpu.gpr[8] = 0x1100;

        assert_eq!(
            cpu.try_fast_forward_memory_loop(&mut memory, 7).unwrap(),
            Some(30)
        );
        assert_eq!(cpu.pc, 0x1018);
        assert_eq!(cpu.instruction_count, 30);
        assert_eq!(memory.read_bytes(0x1100, 5).unwrap(), [0xa5; 5]);
        assert_eq!(cpu.gpr[8], 0x1105);
        assert_eq!(cpu.gpr[9], 0);
        assert_eq!(cpu.gpr[6], u32::MAX);
        assert_eq!(cpu.gpr[7], u32::MAX);
    }

    #[test]
    fn little_endian_unaligned_word_loads_and_stores() {
        // the pairs emitted by compilers for an unaligned little-endian word
        // must reconstruct/copy the same four consecutive bytes at every
        // possible alignment.
        for offset in 0..4u32 {
            let mut memory = Memory::default();
            memory.map(0x1000, 0x1000, true, true).unwrap();
            memory
                .write_bytes(0x1100, &[0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87])
                .unwrap();
            let mut cpu = Cpu::new(0x1000);
            cpu.gpr[1] = 0x1100 + offset;
            // lwl v0, 3(at); lwr v0, 0(at)
            memory.write_u32(0x1000, 0x8822_0003).unwrap();
            memory.write_u32(0x1004, 0x9822_0000).unwrap();
            cpu.step(&mut memory).unwrap();
            cpu.step(&mut memory).unwrap();
            let bytes = (0..4)
                .map(|index| memory.read_u8(0x1100 + offset + index).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(cpu.gpr[2], u32::from_le_bytes(bytes.try_into().unwrap()));

            cpu.gpr[1] = 0x1180 + offset;
            // swl v0, 3(at); swr v0, 0(at)
            memory.write_u32(0x1008, 0xa822_0003).unwrap();
            memory.write_u32(0x100c, 0xb822_0000).unwrap();
            cpu.step(&mut memory).unwrap();
            cpu.step(&mut memory).unwrap();
            for (index, expected) in cpu.gpr[2].to_le_bytes().into_iter().enumerate() {
                assert_eq!(
                    memory.read_u8(0x1180 + offset + index as u32).unwrap(),
                    expected
                );
            }
        }
    }
    #[test]
    fn allegrex_bitfield_extract_and_insert() {
        let (c, _) = run(&[
            0x3c01_1234, // lui at, 0x1234
            0x3421_5678, // ori at, at, 0x5678
            0x7c22_3c00, // ext v0, at, 16, 8
            0x3c03_aabb, // lui v1, 0xaabb
            0x7c23_fe04, // ins v1, at, 24, 8
        ]);
        assert_eq!(c.gpr[2], 0x34);
        assert_eq!(c.gpr[3], 0x78bb_0000);
    }

    #[test]
    fn mips32_rotates() {
        let (c, _) = run(&[
            0x3c01_1234, // lui at, 0x1234
            0x3421_5678, // ori at, at, 0x5678
            0x0021_1402, // rotr v0, at, 16
            0x2403_0008, // addiu v1, zero, 8
            0x0061_2046, // rotrv a0, at, v1
        ]);
        assert_eq!(c.gpr[2], 0x5678_1234);
        assert_eq!(c.gpr[4], 0x7812_3456);
    }

    #[test]
    fn vfpu_loads_named_constant() {
        let (cpu, _) = run(&[0xd065_0020]);
        assert_eq!(
            cpu.vfpu[vfpu_scalar_lane(0x20)],
            std::f32::consts::FRAC_2_PI.to_bits()
        );
    }

    #[test]
    fn vfpu_scalar_and_vector_views_share_physical_lanes() {
        let vector = vfpu_vector_lanes(0, 4);
        assert_eq!(vfpu_scalar_lane(0), vector[0]);
        assert_eq!(vfpu_scalar_lane(0x20), vector[1]);
        assert_eq!(vfpu_scalar_lane(0x40), vector[2]);
        assert_eq!(vfpu_scalar_lane(0x60), vector[3]);
        assert_eq!(vfpu_scalar_lane(0x01), vfpu_vector_lanes(1, 4)[0]);
    }

    #[test]
    fn vfpu_scalar_loads_feed_quad_store_in_memory_order() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        let source = b"DISC0:/PSP_GAME/USRDIR/RUNDATA\0";
        memory.write_bytes(0x1100, source).unwrap();
        let words = [
            0xc8a0_0000,
            0xc8a0_0005,
            0xc8a0_000a,
            0xc8a0_000f,
            0xc8a1_0010,
            0xc8a1_0015,
            0xc8a1_001a,
            0xc8a1_001f,
            0xf880_ffe0,
            0xf881_fff0,
        ];
        for (index, word) in words.into_iter().enumerate() {
            memory.write_u32(0x1000 + index as u32 * 4, word).unwrap();
        }
        let mut cpu = Cpu::new(0x1000);
        cpu.gpr[5] = 0x1100;
        cpu.gpr[4] = 0x1180 + 32;
        for _ in words {
            cpu.step(&mut memory).unwrap();
        }
        assert_eq!(
            memory.read_bytes(0x1180, 32).unwrap(),
            b"DISC0:/PSP_GAME/USRDIR/RUNDATA\0\0"
        );
    }

    #[test]
    fn vfpu_quad_load_store_preserves_byte_order() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        memory.write_bytes(0x1100, b"DISC0:/PSP_GAME/").unwrap();
        memory.write_u32(0x1000, 0xd820_0000).unwrap(); // lv.q C000, 0(at)
        memory.write_u32(0x1004, 0xf840_0000).unwrap(); // sv.q C000, 0(v0)
        let mut cpu = Cpu::new(0x1000);
        cpu.gpr[1] = 0x1100;
        cpu.gpr[2] = 0x1180;
        cpu.step(&mut memory).unwrap();
        cpu.step(&mut memory).unwrap();
        let copied = (0..16)
            .map(|offset| memory.read_u8(0x1180 + offset).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(copied, b"DISC0:/PSP_GAME/");
    }

    #[test]
    fn vfpu_source_prefix_supports_constants_and_sign_controls() {
        let mut registers = [0u32; 128];
        let lanes = vfpu_vector_lanes(0, 4);
        registers[lanes[0]] = (-2.0f32).to_bits();
        registers[lanes[1]] = 3.0f32.to_bits();
        registers[lanes[2]] = 4.0f32.to_bits();
        registers[lanes[3]] = 5.0f32.to_bits();

        // x = abs(w), y = -z, z = 2, w = 1/2.
        let prefix =
            3 | (2 << 2) | (2 << 4) | (3 << 6) | (1 << 8) | (1 << 14) | (1 << 15) | (1 << 17);
        let values = vfpu_source_values(&registers, lanes, 4, prefix);
        assert_eq!(f32::from_bits(values[0]), 5.0);
        assert_eq!(f32::from_bits(values[1]), -4.0);
        assert_eq!(f32::from_bits(values[2]), 2.0);
        assert_eq!(f32::from_bits(values[3]), 0.5);
    }

    #[test]
    fn vfpu_prefixes_modify_arithmetic_and_are_consumed() {
        let mut memory = Memory::default();
        memory.map(0x1000, 0x1000, true, true).unwrap();
        // vpfxs reverse, vpfxd saturate x/y and mask z, vadd.q c000,c000,c100.
        let words = [0xdc00_001b, 0xde00_040d, 0x6001_8080];
        for (index, word) in words.into_iter().enumerate() {
            memory.write_u32(0x1000 + index as u32 * 4, word).unwrap();
        }
        let mut cpu = Cpu::new(0x1000);
        for (index, value) in [1.0f32, 2.0, 3.0, 4.0].into_iter().enumerate() {
            cpu.vfpu[index] = value.to_bits();
        }
        for (index, value) in [10.0f32, 20.0, 30.0, 40.0].into_iter().enumerate() {
            cpu.vfpu[4 + index] = value.to_bits();
        }
        for _ in words {
            cpu.step(&mut memory).unwrap();
        }

        assert_eq!(f32::from_bits(cpu.vfpu[0]), 1.0);
        assert_eq!(f32::from_bits(cpu.vfpu[1]), 1.0);
        assert_eq!(f32::from_bits(cpu.vfpu[2]), 3.0);
        assert_eq!(f32::from_bits(cpu.vfpu[3]), 41.0);
        assert_eq!(cpu.vfpu_s_prefix, VFPU_DEFAULT_SOURCE_PREFIX);
        assert_eq!(cpu.vfpu_t_prefix, VFPU_DEFAULT_SOURCE_PREFIX);
        assert_eq!(cpu.vfpu_d_prefix, 0);
    }

    #[test]
    fn expands_packed_half_values() {
        assert_eq!(f32::from_bits(expand_half(0x3c00)), 1.0);
        assert_eq!(f32::from_bits(expand_half(0xc000)), -2.0);
        assert_eq!(shrink_half(1.0f32.to_bits()), 0x3c00);
    }

    /// Run words with preset registers, stepping each word once.
    fn run_preset(words: &[u32], gpr: &[(usize, u32)], vfpu: &[(usize, u32)]) -> (Cpu, Memory) {
        let mut m = Memory::default();
        m.map(0x1000, 0x1000, true, true).unwrap();
        for (i, w) in words.iter().enumerate() {
            m.write_u32(0x1000 + i as u32 * 4, *w).unwrap();
        }
        let mut c = Cpu::new(0x1000);
        for &(reg, value) in gpr {
            c.gpr[reg] = value;
        }
        for &(lane, value) in vfpu {
            c.vfpu[lane] = value;
        }
        for _ in words {
            c.step(&mut m).unwrap();
        }
        (c, m)
    }

    fn vfpu_lane_values(cpu: &Cpu, lanes: &[usize]) -> Vec<f32> {
        lanes
            .iter()
            .map(|&lane| f32::from_bits(cpu.vfpu[lane]))
            .collect()
    }

    #[test]
    fn vsbn_replaces_first_lane_exponent() {
        // mtv $8, s000 (raw shift 3); vsbn.s s000, s001, s000 with
        // s001 = 2.0 (mantissa 1.0, exponent 128): the exponent is
        // replaced by 127 + 3, giving 1.0 * 2^3 = 8.0 in s000.
        let (c, _) = run_preset(
            &[0x48e8_0000, 0x6100_0100],
            &[(8, 3)],
            &[(4, 2.0f32.to_bits())],
        );
        assert_eq!(f32::from_bits(c.vfpu[0]), 8.0);
    }

    #[test]
    fn vdet_pair_computes_determinant_with_forced_swizzle() {
        // vdet.p s000, c100, c200 with s=(1,2), t=(3,4): t is force
        // swizzled to yx, so 1*4 - 2*3 = -2.
        let (c, _) = run_preset(
            &[0x6708_8480],
            &[],
            &[
                (16, 1.0f32.to_bits()),
                (17, 2.0f32.to_bits()),
                (32, 3.0f32.to_bits()),
                (33, 4.0f32.to_bits()),
            ],
        );
        assert_eq!(vfpu_lane_values(&c, &[0]), vec![-2.0]);
    }

    #[test]
    fn vcrs_triple_uses_forced_half_swizzles() {
        // vcrs.t c000, c100, c200: s=(1,2,3) as yzx=(2,3,1),
        // t=(4,5,6) as zxy=(6,4,5) -> (12,12,5).
        let (c, _) = run_preset(
            &[0x6688_8400 | 0x8000],
            &[],
            &[
                (16, 1.0f32.to_bits()),
                (17, 2.0f32.to_bits()),
                (18, 3.0f32.to_bits()),
                (32, 4.0f32.to_bits()),
                (33, 5.0f32.to_bits()),
                (34, 6.0f32.to_bits()),
            ],
        );
        assert_eq!(vfpu_lane_values(&c, &[0, 1, 2]), vec![12.0, 12.0, 5.0]);
    }

    #[test]
    fn vcrossquat_triple_matches_hardware_prefix_model() {
        // vcrsp.t c000, c100, c200: raw cross x/y plus a prefixed dot z.
        // d = (2*6-3*5, 3*4-1*6, dot((1,2,3,0),(5,-4,0,0))) = (-3, 6, -3).
        let (c, _) = run_preset(
            &[0xf288_8400 | 0x8000],
            &[],
            &[
                (16, 1.0f32.to_bits()),
                (17, 2.0f32.to_bits()),
                (18, 3.0f32.to_bits()),
                (32, 4.0f32.to_bits()),
                (33, 5.0f32.to_bits()),
                (34, 6.0f32.to_bits()),
            ],
        );
        assert_eq!(vfpu_lane_values(&c, &[0, 1, 2]), vec![-3.0, 6.0, -3.0]);
    }

    #[test]
    fn vrnd_seed_streams_deterministic_words() {
        // vrnds s000 seeds from lane 0; two vrndi.s singles differ.
        let mut m = Memory::default();
        m.map(0x1000, 0x1000, true, true).unwrap();
        for (i, w) in [0xd020_0000u32, 0xd021_0000, 0xd021_0000]
            .into_iter()
            .enumerate()
        {
            m.write_u32(0x1000 + i as u32 * 4, w).unwrap();
        }
        let mut cpu = Cpu::new(0x1000);
        cpu.vfpu[0] = 0x1234_5678;
        cpu.step(&mut m).unwrap();
        cpu.step(&mut m).unwrap();
        let first = cpu.vfpu[0];
        cpu.pc = 0x1004;
        cpu.step(&mut m).unwrap();
        let second = cpu.vfpu[0];
        assert_ne!(first, second);
        assert_ne!(first, 0);
    }

    #[test]
    fn vsbz_and_vlgb_extract_mantissa_and_exponent() {
        // vsbz.s of 2.0 -> 1.0; vlgb.s of 2.0 -> 1.0.
        let (c, _) = run_preset(&[0xd036_0000], &[], &[(0, 2.0f32.to_bits())]);
        assert_eq!(f32::from_bits(c.vfpu[0]), 1.0);
        let (c, _) = run_preset(&[0xd037_0000], &[], &[(0, 2.0f32.to_bits())]);
        assert_eq!(f32::from_bits(c.vfpu[0]), 1.0);
    }

    #[test]
    fn vi2uc_packs_clamped_bytes() {
        // vi2uc.s s000 with lanes (0x00000001, 0x00000100, 0x00010000,
        // 0x01000000): each >> 23 gives (0,0,2,32)... use simple values:
        // lanes (1<<23, 2<<23, 3<<23, 4<<23) -> bytes (1,2,3,4).
        let (c, _) = run_preset(
            &[0xd03c_0000],
            &[],
            &[(0, 1 << 23), (1, 2 << 23), (2, 3 << 23), (3, 4 << 23)],
        );
        assert_eq!(c.vfpu[0], 0x0403_0201);
    }

    #[test]
    fn vt5650_packs_color_nibbles() {
        // vt5650.s s000 of 0xff804020 -> (0x10 << 11) | (0x10 << 5) | 0x04.
        let (c, _) = run_preset(&[0xd05b_0000], &[], &[(0, 0xff80_4020)]);
        assert_eq!(c.vfpu[0] & 0xffff, 0x8204);
    }

    #[test]
    fn vwbn_rebiases_lane_exponent() {
        // vwbn.s s000 of 4.0 with exponent 0x80 -> 2.0.
        let (c, _) = run_preset(&[0xd380_0000], &[], &[(0, 4.0f32.to_bits())]);
        assert_eq!(f32::from_bits(c.vfpu[0]), 2.0);
    }

    #[test]
    fn vmscl_scales_matrix_by_named_scalar() {
        // vmscl m000(2x2)=[1,2;3,4] by s100=2.0 -> [2,4,6,8].
        let (c, _) = run_preset(
            &[0xf204_0080],
            &[],
            &[
                (0, 1.0f32.to_bits()),
                (1, 2.0f32.to_bits()),
                (4, 3.0f32.to_bits()),
                (5, 4.0f32.to_bits()),
                (16, 2.0f32.to_bits()),
            ],
        );
        assert_eq!(
            vfpu_lane_values(&c, &[0, 1, 4, 5]),
            vec![2.0, 4.0, 6.0, 8.0]
        );
    }

    #[test]
    fn vmmov_copies_matrix() {
        // vmmov m000(2x2), m100(2x2): pair size is bit 7, not bit 15.
        let (c, _) = run_preset(
            &[0xf380_0480],
            &[],
            &[
                (16, 5.0f32.to_bits()),
                (17, 6.0f32.to_bits()),
                (20, 7.0f32.to_bits()),
                (21, 8.0f32.to_bits()),
            ],
        );
        assert_eq!(
            vfpu_lane_values(&c, &[0, 1, 4, 5]),
            vec![5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn lvl_loads_unaligned_head_lanes() {
        let mut m = Memory::default();
        m.map(0x1000, 0x100, true, true).unwrap();
        m.map(0x1100, 0x100, true, false).unwrap();
        m.write_u32(0x1100, 10).unwrap();
        m.write_u32(0x1104, 20).unwrap();
        m.write_u32(0x1108, 30).unwrap();
        // lvl.q c000, 8($5) with $5=0x1100: offset 2 -> lanes [0,10,20,30].
        m.write_u32(0x1000, 0xd400_0008 | (5 << 21)).unwrap();
        let mut cpu = Cpu::new(0x1000);
        cpu.gpr[5] = 0x1100;
        cpu.step(&mut m).unwrap();
        assert_eq!(
            [cpu.vfpu[0], cpu.vfpu[1], cpu.vfpu[2], cpu.vfpu[3]],
            [0, 10, 20, 30]
        );
    }

    #[test]
    fn vmtvc_vmfvc_roundtrip_control_file() {
        // mtvc $8, pfxs(128); vmfvc s000, 0.
        let (c, _) = run_preset(&[0x48e8_0080, 0xd050_0000], &[(8, 0x1bef_1234)], &[]);
        assert_eq!(c.vfpu_s_prefix, 0x000f_ffff & 0x1bef_1234);
        assert_eq!(c.vfpu[0], 0x000f_ffff & 0x1bef_1234);
    }

    #[test]
    fn vscmp_reports_lane_signs() {
        // vscmp.p c000, c000, c100 with s=(2,1), t=(1,2) -> (1,-1).
        let (c, _) = run_preset(
            &[0x6e84_0080],
            &[],
            &[
                (0, 2.0f32.to_bits()),
                (1, 1.0f32.to_bits()),
                (16, 1.0f32.to_bits()),
                (17, 2.0f32.to_bits()),
            ],
        );
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![1.0, -1.0]);
    }

    #[test]
    fn vhdp_quad_forces_homogeneous_one() {
        // vhdp.q s000, c000, c100 with s=(1,2,3,9), t=(1,1,1,1) -> 7.
        let (c, _) = run_preset(
            &[0x6604_8080],
            &[],
            &[
                (0, 1.0f32.to_bits()),
                (1, 2.0f32.to_bits()),
                (2, 3.0f32.to_bits()),
                (3, 9.0f32.to_bits()),
                (16, 1.0f32.to_bits()),
                (17, 1.0f32.to_bits()),
                (18, 1.0f32.to_bits()),
                (19, 1.0f32.to_bits()),
            ],
        );
        assert_eq!(vfpu_lane_values(&c, &[0]), vec![7.0]);
    }

    #[test]
    fn vscl_broadcasts_named_scalar_lane() {
        // vscl.p c000, c000, vt=0x20: scalar lane 1 holds 3.0 and the
        // source pair (10.0, 3.0) scales to (30.0, 9.0).
        let (c, _) = run_preset(
            &[0x6520_0080],
            &[],
            &[
                (0, 10.0f32.to_bits()),
                (1, 3.0f32.to_bits()),
                (2, 20.0f32.to_bits()),
            ],
        );
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![30.0, 9.0]);
    }

    #[test]
    fn vrot_single_angle_fills_sine_lanes() {
        // vrot.q c000, s000, 0 with angle 1.0 -> all sin(pi/2)=1.
        let (c, _) = run_preset(&[0x48e8_0000, 0xf3a0_8080], &[(8, 1.0f32.to_bits())], &[]);
        let values = vfpu_lane_values(&c, &[0, 1, 2, 3]);
        for value in values {
            assert!((value - 1.0).abs() < 1e-6, "{value}");
        }
    }

    #[test]
    fn vsgn_pair_reports_signs() {
        // vsgn.p s000, c000 with s=(2,-3) -> (1,-1).
        let (c, _) = run_preset(
            &[0xd04a_0080],
            &[],
            &[(0, 2.0f32.to_bits()), (1, (-3.0f32).to_bits())],
        );
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![1.0, -1.0]);
    }

    #[test]
    fn vfad_quad_sums_all_lanes() {
        // vfad.q s000, c000 with (1,2,3,4) -> 10.
        let (c, _) = run_preset(
            &[0xd046_8080],
            &[],
            &[
                (0, 1.0f32.to_bits()),
                (1, 2.0f32.to_bits()),
                (2, 3.0f32.to_bits()),
                (3, 4.0f32.to_bits()),
            ],
        );
        assert_eq!(vfpu_lane_values(&c, &[0]), vec![10.0]);
    }

    #[test]
    fn vavg_pair_averages_lanes() {
        // vavg.p s000, c000 with (2,4) -> 3.
        let (c, _) = run_preset(
            &[0xd047_0080],
            &[],
            &[(0, 2.0f32.to_bits()), (1, 4.0f32.to_bits())],
        );
        assert_eq!(vfpu_lane_values(&c, &[0]), vec![3.0]);
    }

    #[test]
    fn vbfy1_pair_butterflies_lanes() {
        // vbfy1.p c000, c000 with s=(1,2): (1+2, -2+1) = (3,-1).
        let (c, _) = run_preset(
            &[0xd042_0080],
            &[],
            &[(0, 1.0f32.to_bits()), (1, 2.0f32.to_bits())],
        );
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![3.0, -1.0]);
    }

    #[test]
    fn vsrt1_pair_sorts_lanes() {
        // vsrt1.p c000, c000 with s=(1,3): t=(3,1) -> (min,max)=(1,3).
        let (c, _) = run_preset(
            &[0xd040_0080],
            &[],
            &[(0, 1.0f32.to_bits()), (1, 3.0f32.to_bits())],
        );
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![1.0, 3.0]);
    }

    #[test]
    fn vocp_single_complements() {
        // vocp.s s000 of 0.25 -> 0.75.
        let (c, _) = run_preset(&[0xd044_0000], &[], &[(0, 0.25f32.to_bits())]);
        assert_eq!(vfpu_lane_values(&c, &[0]), vec![0.75]);
    }

    #[test]
    fn vsocp_single_pairs_complement() {
        // vsocp.s s000 of 0.25 -> (0.75, 0.25).
        let (c, _) = run_preset(&[0xd045_0000], &[], &[(0, 0.25f32.to_bits())]);
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![0.75, 0.25]);
    }

    #[test]
    fn vtfm_identity_matrix_preserves_vector() {
        // vtfm4 c100, m100, c200 with m100 = identity, v = (5,6,7,8).
        let (c, _) = run_preset(
            &[0xf188_8484],
            &[],
            &[
                (32, 5.0f32.to_bits()),
                (33, 6.0f32.to_bits()),
                (34, 7.0f32.to_bits()),
                (35, 8.0f32.to_bits()),
                (16, 1.0f32.to_bits()),
                (21, 1.0f32.to_bits()),
                (26, 1.0f32.to_bits()),
                (31, 1.0f32.to_bits()),
            ],
        );
        assert_eq!(
            vfpu_lane_values(&c, &[16, 17, 18, 19]),
            vec![5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn cop1x_madd_fuses_multiply_add() {
        // madd.s $f4, $f3, $f2, $f1 (fd=4, fs=3, ft=2, fr=1):
        // 2.0 * 3.0 + 4.0 = 10.0.
        let mut m = Memory::default();
        m.map(0x1000, 0x1000, true, true).unwrap();
        m.write_u32(0x1000, 0x4c22_1920).unwrap();
        let mut cpu = Cpu::new(0x1000);
        cpu.fpr[1] = 2.0f32.to_bits();
        cpu.fpr[2] = 3.0f32.to_bits();
        cpu.fpr[3] = 4.0f32.to_bits();
        cpu.step(&mut m).unwrap();
        assert_eq!(f32::from_bits(cpu.fpr[4]), 10.0);
    }

    #[test]
    fn special2_mul_multiplies_low_word() {
        // mul $10, $8, $9 with (6, 7) -> 42.
        let (c, _) = run_preset(&[0x7109_5002], &[(8, 6), (9, 7)], &[]);
        assert_eq!(c.gpr[10], 42);
    }

    #[test]
    fn vf2i_rounds_ties_even() {
        // vf2in.q c000, c000, 0 with (1.5, 2.5, 3.0, -0.5) -> (2, 2, 3, 0).
        let (c, _) = run_preset(
            &[0xd200_8080],
            &[],
            &[
                (0, 1.5f32.to_bits()),
                (1, 2.5f32.to_bits()),
                (2, 3.0f32.to_bits()),
                (3, (-0.5f32).to_bits()),
            ],
        );
        assert_eq!([c.vfpu[0], c.vfpu[1], c.vfpu[2], c.vfpu[3]], [2, 2, 3, 0]);
    }

    #[test]
    fn vi2f_halves_integers() {
        // vi2f.p c000, c000, 1 with (3, 4) -> (1.5, 2.0).
        let (c, _) = run_preset(&[0xd281_0080], &[], &[(0, 3), (1, 4)]);
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![1.5, 2.0]);
    }

    #[test]
    fn vh2f_expands_packed_halves() {
        // vh2f.s s000 pair from 0x3c003c00 -> (1.0, 1.0).
        let (c, _) = run_preset(&[0xd033_0000], &[], &[(0, 0x3c00_3c00)]);
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![1.0, 1.0]);
    }

    #[test]
    fn vf2h_packs_float_pair() {
        // vf2h.p s000 from (1.0, 2.0) -> 0x40003c00.
        let (c, _) = run_preset(
            &[0xd032_0080],
            &[],
            &[(0, 1.0f32.to_bits()), (1, 2.0f32.to_bits())],
        );
        assert_eq!(c.vfpu[0], 0x4000_3c00);
    }

    #[test]
    fn vcmov_moves_on_set_cc() {
        // vcmov.t c000, c100, 0 with cc0 set: c000 takes c100's (9, 10).
        let (c, _) = run_preset(
            &[0xd2a0_0480],
            &[],
            &[
                (0, 1.0f32.to_bits()),
                (1, 2.0f32.to_bits()),
                (16, 9.0f32.to_bits()),
                (17, 10.0f32.to_bits()),
            ],
        );
        // cc must be preset; run_preset leaves it 0x3f (all lanes set),
        // so the move happens. a cleared cc must block it:
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![9.0, 10.0]);
        let mut m = Memory::default();
        m.map(0x1000, 0x1000, true, true).unwrap();
        m.write_u32(0x1000, 0xd2a0_0480).unwrap();
        let mut cpu = Cpu::new(0x1000);
        cpu.vfpu_cc = 0;
        cpu.vfpu[0] = 1.0f32.to_bits();
        cpu.vfpu[1] = 2.0f32.to_bits();
        cpu.vfpu[16] = 9.0f32.to_bits();
        cpu.vfpu[17] = 10.0f32.to_bits();
        cpu.step(&mut m).unwrap();
        assert_eq!(vfpu_lane_values(&cpu, &[0, 1]), vec![1.0, 2.0]);
    }

    #[test]
    fn vmin_vmax_pair_follow_ieee() {
        // vmin.p/vmax.p of s=(1,5), t=(3,2) -> (1,2) / (3,5).
        let vfpu = &[
            (0, 1.0f32.to_bits()),
            (1, 5.0f32.to_bits()),
            (16, 3.0f32.to_bits()),
            (17, 2.0f32.to_bits()),
        ];
        let (c, _) = run_preset(&[0x6d04_0080], &[], vfpu);
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![1.0, 2.0]);
        let (c, _) = run_preset(&[0x6d84_0080], &[], vfpu);
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![3.0, 5.0]);
    }

    #[test]
    fn vsge_vslt_pair_compare() {
        // vsge.p/vslt.p of s=(1,5), t=(3,2) -> (0,1) / (1,0).
        let vfpu = &[
            (0, 1.0f32.to_bits()),
            (1, 5.0f32.to_bits()),
            (16, 3.0f32.to_bits()),
            (17, 2.0f32.to_bits()),
        ];
        let (c, _) = run_preset(&[0x6f04_0080], &[], vfpu);
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![0.0, 1.0]);
        let (c, _) = run_preset(&[0x6f84_0080], &[], vfpu);
        assert_eq!(vfpu_lane_values(&c, &[0, 1]), vec![1.0, 0.0]);
    }

    #[test]
    fn svl_svr_roundtrip_unaligned() {
        let mut m = Memory::default();
        m.map(0x1000, 0x100, true, true).unwrap();
        m.map(0x1100, 0x100, true, false).unwrap();
        // svl.q c000, 8($5): offset 2 stores lanes [1..3] to 0x1100..0x1108.
        m.write_u32(0x1000, 0xf4a0_0008).unwrap();
        // svr.q c000, 10($5): offset 2 stores lanes [0..1] to 0x1108..0x110c.
        m.write_u32(0x1004, 0xf4a0_000a).unwrap();
        let mut cpu = Cpu::new(0x1000);
        cpu.gpr[5] = 0x1100;
        cpu.vfpu[0] = 10;
        cpu.vfpu[1] = 20;
        cpu.vfpu[2] = 30;
        cpu.vfpu[3] = 40;
        cpu.step(&mut m).unwrap();
        assert_eq!(m.read_u32(0x1100).unwrap(), 20);
        assert_eq!(m.read_u32(0x1104).unwrap(), 30);
        assert_eq!(m.read_u32(0x1108).unwrap(), 40);
        cpu.step(&mut m).unwrap();
        assert_eq!(m.read_u32(0x1108).unwrap(), 10);
        assert_eq!(m.read_u32(0x110c).unwrap(), 20);
    }

    #[test]
    fn lvq_misaligned_address_faults_and_zeroes() {
        let mut m = Memory::default();
        m.map(0x1000, 0x100, true, true).unwrap();
        m.map(0x1100, 0x100, true, false).unwrap();
        // lv.q c000, 4($5): 4-byte aligned but not 16-byte aligned.
        m.write_u32(0x1000, 0xd8a0_0004).unwrap();
        let mut cpu = Cpu::new(0x1000);
        cpu.gpr[5] = 0x1100;
        cpu.vfpu[0] = 0xdead_beef;
        let result = cpu.step(&mut m);
        assert!(matches!(result, Err(CpuError::Unsupported { .. })));
        assert_eq!([cpu.vfpu[0], cpu.vfpu[1], cpu.vfpu[2], cpu.vfpu[3]], [0; 4]);
    }

    #[test]
    fn viim_loads_signed_immediate() {
        // viim.s s000, 0x1234 -> 4660.0.
        let (c, _) = run_preset(&[0xdf00_1234], &[], &[]);
        assert_eq!(f32::from_bits(c.vfpu[0]), 4660.0);
    }

    #[test]
    fn vcmp_eq_sets_cc_lanes() {
        let mut m = Memory::default();
        m.map(0x1000, 0x1000, true, true).unwrap();
        // vcmp.p eq c000, c000, c100 with s=(1,2), t=(1,0).
        m.write_u32(0x1000, 0x6c04_0081).unwrap();
        let mut cpu = Cpu::new(0x1000);
        cpu.vfpu_cc = 0;
        cpu.vfpu[0] = 1.0f32.to_bits();
        cpu.vfpu[1] = 2.0f32.to_bits();
        cpu.vfpu[16] = 1.0f32.to_bits();
        cpu.vfpu[17] = 0.0f32.to_bits();
        cpu.step(&mut m).unwrap();
        // lanes (1,0), any=1, all=0.
        assert_eq!(cpu.vfpu_cc, 0x11);
    }
}
