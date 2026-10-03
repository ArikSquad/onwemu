//! Opcode handlers shared by the interpreter and block cache.
//!
//! Each handler reads pre-decoded operands from [`Decoded`]. [`handler_for`]
//! resolves overlapping encodings in priority order and routes unknown words
//! to the unsupported-instruction error.

use psp_memory::{GuestAddress, GuestMemory, Memory};

use super::{
    Cpu, CpuError, Step, VFPU_DEFAULT_SOURCE_PREFIX, expand_half, shrink_half, vfpu_abs_bits,
    vfpu_any_swizzle, vfpu_asin_value, vfpu_clamp, vfpu_exp2_value, vfpu_isolate_prefix_lane,
    vfpu_log2_value, vfpu_make_constants, vfpu_matrix_lanes, vfpu_negate_bits, vfpu_prefix_quad,
    vfpu_read_ctrl, vfpu_recip_bits, vfpu_retain_invalid_swizzle, vfpu_rewrite_prefix,
    vfpu_rng_generate, vfpu_rng_seed, vfpu_scalar_lane, vfpu_source_values, vfpu_swizzle,
    vfpu_vector_lanes, vfpu_write_ctrl, vfpu_write_matrix, vfpu_write_vector,
};

/// Pre-decoded instruction fields. The JIT resolves these and the [`Handler`]
/// once at compile time instead of on every execution.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Decoded {
    pub pc: u32,
    pub word: u32,
    pub op: u8,
    pub rs: u8,
    pub rt: u8,
    pub rd: u8,
    pub imm: u16,
}

impl Decoded {
    pub(crate) fn decode(pc: u32, word: u32) -> Self {
        Self {
            pc,
            word,
            op: (word >> 26) as u8,
            rs: ((word >> 21) & 31) as u8,
            rt: ((word >> 16) & 31) as u8,
            rd: ((word >> 11) & 31) as u8,
            imm: word as u16,
        }
    }
}

/// Direct-threaded opcode handler that executes one instruction's semantics.
pub(crate) type Handler = fn(&mut Cpu, &mut Memory, Decoded) -> Result<Step, CpuError>;

/// Resolve the handler for an instruction word. Guard order mirrors the
/// interpreter's historical match priority exactly.
pub(crate) fn handler_for(word: u32) -> Handler {
    let op = word >> 26;
    let rs = ((word >> 21) & 31) as usize;
    match op {
        0 => match word & 63 {
            0x00 => exec_sll,
            0x02 => exec_srl,
            0x03 => exec_sra,
            0x04 => exec_sllv,
            0x06 => exec_srlv,
            0x07 => exec_srav,
            0x08 => exec_jr,
            0x09 => exec_jalr,
            0x0a => exec_movz,
            0x0b => exec_movn,
            0x0c => exec_syscall,
            0x0f => exec_sync,
            0x10 => exec_mfhi,
            0x11 => exec_mthi,
            0x12 => exec_mflo,
            0x13 => exec_mtlo,
            0x16 => exec_clz,
            0x17 => exec_clo,
            0x18 => exec_mult,
            0x19 => exec_multu,
            0x1a => exec_div,
            0x1b => exec_divu,
            0x1c => exec_madd,
            0x1d => exec_maddu,
            0x2e => exec_msub,
            0x2f => exec_msubu,
            0x20 => exec_add,
            0x21 => exec_addu,
            0x22 => exec_sub,
            0x23 => exec_subu,
            0x24 => exec_and,
            0x25 => exec_or,
            0x26 => exec_xor,
            0x27 => exec_nor,
            0x2a => exec_slt,
            0x2b => exec_sltu,
            0x2c => exec_max,
            0x2d => exec_min,
            _ => exec_unsupported,
        },
        0x01 => exec_branch_regimm,
        0x02 => exec_j,
        0x03 => exec_jal,
        0x04 => exec_beq,
        0x05 => exec_bne,
        0x06 => exec_blez,
        0x07 => exec_bgtz,
        0x08 => exec_addi,
        0x09 => exec_addiu,
        0x0a => exec_slti,
        0x0b => exec_sltiu,
        0x0c => exec_andi,
        0x0d => exec_ori,
        0x0e => exec_xori,
        0x0f => exec_lui,
        0x10 => exec_cop0,
        0x11 => exec_fpu,
        0x12 => exec_vfpu_move,
        0x13 => exec_cop1x,
        0x14 => exec_beql,
        0x15 => exec_bnel,
        0x16 => exec_blezl,
        0x17 => exec_bgtzl,
        0x18 if (word >> 23) & 7 == 2 => exec_vsbn,
        0x18 | 0x19 if matches!((op, (word >> 23) & 7), (0x18, 0 | 1 | 7) | (0x19, 0)) => {
            exec_vfpu_arith
        }
        0x19 if matches!((word >> 23) & 7, 1 | 2 | 4 | 5 | 6) => exec_vfpu_horizontal,
        0x1b if (word >> 23) & 7 == 0 => exec_vcmp,
        0x1b if matches!((word >> 23) & 7, 2 | 3 | 5 | 6 | 7) => exec_vminmax_cmp,
        0x1c => exec_special2,
        0x1f => exec_special3,
        0x20 => exec_lb,
        0x21 => exec_lh,
        0x22 => exec_lwl,
        0x23 => exec_lw,
        0x24 => exec_lbu,
        0x25 => exec_lhu,
        0x26 => exec_lwr,
        0x28 => exec_sb,
        0x29 => exec_sh,
        0x2a => exec_swl,
        0x2b => exec_sw,
        0x2e => exec_swr,
        0x2f => exec_cache,
        0x30 => exec_ll,
        0x31 => exec_lwc1,
        0x32 => exec_lvs,
        0x33 => exec_pref,
        0x34 if rs == 3 => exec_vcst,
        0x34 if rs == 21 => exec_vcmov,
        0x34 if rs == 20 => exec_vi2f,
        0x34 if (16..=19).contains(&rs) => exec_vf2i,
        0x34 if rs == 1 && matches!((word >> 16) & 31, 0 | 1 | 2 | 3 | 18 | 19 | 22 | 23) => {
            exec_vfpu7
        }
        0x34 if (24..=31).contains(&rs) => exec_vwbn,
        0x34 if rs == 1 && (28..=31).contains(&((word >> 16) & 31)) => exec_vi2x,
        0x34 if rs == 1 && (24..=27).contains(&((word >> 16) & 31)) => exec_vx2i,
        0x34 if rs == 2
            && matches!(
                (word >> 16) & 31,
                0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 16 | 17 | 25 | 26 | 27
            ) =>
        {
            exec_vfpu9
        }
        0x34 if rs == 0 => exec_vfpu_unary,
        0x35 => exec_lvl,
        0x36 => exec_lvq,
        0x37 if (word >> 23) & 7 <= 5 => exec_vpfx,
        0x37 if matches!((word >> 23) & 7, 6 | 7) => exec_viim,
        0x38 => exec_sc,
        0x39 => exec_swc1,
        0x3a => exec_svs,
        0x3d => exec_svl,
        0x3e => exec_svq,
        0x3f => exec_vflush,
        0x3c if rs < 4 => exec_vmmul,
        0x3c if rs == 29 => exec_vrot,
        0x3c if (4..=15).contains(&rs) => exec_vtfm,
        0x3c if (20..=23).contains(&rs) => exec_crossquat,
        0x3c if (16..=19).contains(&rs) => exec_vmscl,
        0x3c if (word >> 21) & 31 == 28 && (word >> 16) & 15 == 0 => exec_vmmov,
        0x3c if (word >> 21) & 31 == 28 && matches!((word >> 16) & 15, 3 | 6 | 7) => {
            exec_vmatrix_init
        }
        _ => exec_unsupported,
    }
}

fn exec_unsupported(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let _ = cpu;
    Err(CpuError::Unsupported {
        pc: GuestAddress(d.pc),
        word: d.word,
    })
}

fn exec_sll(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rt as usize] << ((d.word >> 6) & 31);
    Ok(Step::Continue)
}

fn exec_srl(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let amount = (d.word >> 6) & 31;
    cpu.gpr[d.rd as usize] = if d.rs == 1 {
        cpu.gpr[d.rt as usize].rotate_right(amount)
    } else {
        cpu.gpr[d.rt as usize] >> amount
    };
    Ok(Step::Continue)
}

fn exec_sra(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = ((cpu.gpr[d.rt as usize] as i32) >> ((d.word >> 6) & 31)) as u32;
    Ok(Step::Continue)
}

fn exec_sllv(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rt as usize] << (cpu.gpr[d.rs as usize] & 31);
    Ok(Step::Continue)
}

fn exec_srlv(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let amount = cpu.gpr[d.rs as usize] & 31;
    cpu.gpr[d.rd as usize] = if (d.word >> 6) & 1 != 0 {
        cpu.gpr[d.rt as usize].rotate_right(amount)
    } else {
        cpu.gpr[d.rt as usize] >> amount
    };
    Ok(Step::Continue)
}

fn exec_srav(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] =
        ((cpu.gpr[d.rt as usize] as i32) >> (cpu.gpr[d.rs as usize] & 31)) as u32;
    Ok(Step::Continue)
}

fn exec_jr(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.branch_target = Some(cpu.gpr[d.rs as usize]);
    Ok(Step::Continue)
}

fn exec_jalr(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = d.pc.wrapping_add(8);
    cpu.branch_target = Some(cpu.gpr[d.rs as usize]);
    Ok(Step::Continue)
}

fn exec_movz(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if cpu.gpr[d.rt as usize] == 0 {
        cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize];
    }
    Ok(Step::Continue)
}

fn exec_movn(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if cpu.gpr[d.rt as usize] != 0 {
        cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize];
    }
    Ok(Step::Continue)
}

fn exec_syscall(_: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    Ok(Step::Syscall((d.word >> 6) & 0xfffff))
}

fn exec_sync(_: &mut Cpu, _: &mut Memory, _: Decoded) -> Result<Step, CpuError> {
    Ok(Step::Continue)
}

fn exec_mfhi(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.hi;
    Ok(Step::Continue)
}

fn exec_mthi(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.hi = cpu.gpr[d.rs as usize];
    Ok(Step::Continue)
}

fn exec_mflo(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.lo;
    Ok(Step::Continue)
}

fn exec_mtlo(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.lo = cpu.gpr[d.rs as usize];
    Ok(Step::Continue)
}

fn exec_clz(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize].leading_zeros();
    Ok(Step::Continue)
}

fn exec_clo(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = (!cpu.gpr[d.rs as usize]).leading_zeros();
    Ok(Step::Continue)
}

fn exec_mult(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let value = (cpu.gpr[d.rs as usize] as i32 as i64) * (cpu.gpr[d.rt as usize] as i32 as i64);
    cpu.lo = value as u32;
    cpu.hi = (value >> 32) as u32;
    Ok(Step::Continue)
}

fn exec_multu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let value = u64::from(cpu.gpr[d.rs as usize]) * u64::from(cpu.gpr[d.rt as usize]);
    cpu.lo = value as u32;
    cpu.hi = (value >> 32) as u32;
    Ok(Step::Continue)
}

fn exec_div(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let (n, divisor) = (cpu.gpr[d.rs as usize] as i32, cpu.gpr[d.rt as usize] as i32);
    if divisor == 0 {
        cpu.lo = if n < 0 { 1 } else { u32::MAX };
        cpu.hi = n as u32;
    } else if n == i32::MIN && divisor == -1 {
        cpu.lo = i32::MIN as u32;
        cpu.hi = 0;
    } else {
        cpu.lo = (n / divisor) as u32;
        cpu.hi = (n % divisor) as u32;
    }
    Ok(Step::Continue)
}

fn exec_divu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let (n, divisor) = (cpu.gpr[d.rs as usize], cpu.gpr[d.rt as usize]);
    if let Some(quotient) = n.checked_div(divisor) {
        cpu.lo = quotient;
        cpu.hi = n % divisor;
    } else {
        cpu.lo = u32::MAX;
        cpu.hi = n;
    }
    Ok(Step::Continue)
}

fn exec_madd(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let product = (cpu.gpr[d.rs as usize] as i32 as i64) * (cpu.gpr[d.rt as usize] as i32 as i64);
    let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
    let result = (acc as i64).wrapping_add(product) as u64;
    cpu.lo = result as u32;
    cpu.hi = (result >> 32) as u32;
    Ok(Step::Continue)
}

fn exec_maddu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let product = u64::from(cpu.gpr[d.rs as usize]) * u64::from(cpu.gpr[d.rt as usize]);
    let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
    let result = acc.wrapping_add(product);
    cpu.lo = result as u32;
    cpu.hi = (result >> 32) as u32;
    Ok(Step::Continue)
}

fn exec_msub(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let product = (cpu.gpr[d.rs as usize] as i32 as i64) * (cpu.gpr[d.rt as usize] as i32 as i64);
    let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
    let result = (acc as i64).wrapping_sub(product) as u64;
    cpu.lo = result as u32;
    cpu.hi = (result >> 32) as u32;
    Ok(Step::Continue)
}

fn exec_msubu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let product = u64::from(cpu.gpr[d.rs as usize]) * u64::from(cpu.gpr[d.rt as usize]);
    let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
    let result = acc.wrapping_sub(product);
    cpu.lo = result as u32;
    cpu.hi = (result >> 32) as u32;
    Ok(Step::Continue)
}

fn exec_add(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = (cpu.gpr[d.rs as usize] as i32)
        .checked_add(cpu.gpr[d.rt as usize] as i32)
        .ok_or(CpuError::Overflow(GuestAddress(d.pc)))? as u32;
    Ok(Step::Continue)
}

fn exec_addu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize].wrapping_add(cpu.gpr[d.rt as usize]);
    Ok(Step::Continue)
}

fn exec_sub(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = (cpu.gpr[d.rs as usize] as i32)
        .checked_sub(cpu.gpr[d.rt as usize] as i32)
        .ok_or(CpuError::Overflow(GuestAddress(d.pc)))? as u32;
    Ok(Step::Continue)
}

fn exec_subu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize].wrapping_sub(cpu.gpr[d.rt as usize]);
    Ok(Step::Continue)
}

fn exec_and(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize] & cpu.gpr[d.rt as usize];
    Ok(Step::Continue)
}

fn exec_or(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize] | cpu.gpr[d.rt as usize];
    Ok(Step::Continue)
}

fn exec_xor(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = cpu.gpr[d.rs as usize] ^ cpu.gpr[d.rt as usize];
    Ok(Step::Continue)
}

fn exec_nor(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = !(cpu.gpr[d.rs as usize] | cpu.gpr[d.rt as usize]);
    Ok(Step::Continue)
}

fn exec_slt(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] =
        ((cpu.gpr[d.rs as usize] as i32) < (cpu.gpr[d.rt as usize] as i32)) as u32;
    Ok(Step::Continue)
}

fn exec_sltu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = (cpu.gpr[d.rs as usize] < cpu.gpr[d.rt as usize]) as u32;
    Ok(Step::Continue)
}

fn exec_max(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = if (cpu.gpr[d.rs as usize] as i32) > (cpu.gpr[d.rt as usize] as i32) {
        cpu.gpr[d.rs as usize]
    } else {
        cpu.gpr[d.rt as usize]
    };
    Ok(Step::Continue)
}

fn exec_min(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rd as usize] = if (cpu.gpr[d.rs as usize] as i32) < (cpu.gpr[d.rt as usize] as i32) {
        cpu.gpr[d.rs as usize]
    } else {
        cpu.gpr[d.rt as usize]
    };
    Ok(Step::Continue)
}

fn exec_branch_regimm(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let taken = match d.rt {
        0 | 2 | 16 | 18 => (cpu.gpr[d.rs as usize] as i32) < 0,
        1 | 3 | 17 | 19 => (cpu.gpr[d.rs as usize] as i32) >= 0,
        _ => {
            return Err(CpuError::Unsupported {
                pc: GuestAddress(d.pc),
                word: d.word,
            });
        }
    };
    if matches!(d.rt, 16..=19) {
        cpu.gpr[31] = d.pc.wrapping_add(8);
    }
    if taken {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    } else if matches!(d.rt, 2 | 3 | 18 | 19) {
        cpu.pc = cpu.pc.wrapping_add(4);
    }
    Ok(Step::Continue)
}

fn exec_j(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.branch_target = Some((cpu.pc & 0xf000_0000) | ((d.word & 0x03ff_ffff) << 2));
    Ok(Step::Continue)
}

fn exec_jal(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[31] = d.pc.wrapping_add(8);
    cpu.branch_target = Some((cpu.pc & 0xf000_0000) | ((d.word & 0x03ff_ffff) << 2));
    Ok(Step::Continue)
}

fn exec_beq(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if cpu.gpr[d.rs as usize] == cpu.gpr[d.rt as usize] {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    }
    Ok(Step::Continue)
}

fn exec_bne(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if cpu.gpr[d.rs as usize] != cpu.gpr[d.rt as usize] {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    }
    Ok(Step::Continue)
}

fn exec_blez(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if (cpu.gpr[d.rs as usize] as i32) <= 0 {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    }
    Ok(Step::Continue)
}

fn exec_bgtz(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if (cpu.gpr[d.rs as usize] as i32) > 0 {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    }
    Ok(Step::Continue)
}

fn exec_beql(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if cpu.gpr[d.rs as usize] == cpu.gpr[d.rt as usize] {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    } else {
        cpu.pc = cpu.pc.wrapping_add(4);
    }
    Ok(Step::Continue)
}

fn exec_bnel(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if cpu.gpr[d.rs as usize] != cpu.gpr[d.rt as usize] {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    } else {
        cpu.pc = cpu.pc.wrapping_add(4);
    }
    Ok(Step::Continue)
}

fn exec_blezl(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if (cpu.gpr[d.rs as usize] as i32) <= 0 {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    } else {
        cpu.pc = cpu.pc.wrapping_add(4);
    }
    Ok(Step::Continue)
}

fn exec_bgtzl(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    if (cpu.gpr[d.rs as usize] as i32) > 0 {
        cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
    } else {
        cpu.pc = cpu.pc.wrapping_add(4);
    }
    Ok(Step::Continue)
}

fn exec_addi(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = (cpu.gpr[d.rs as usize] as i32)
        .checked_add(d.imm as i16 as i32)
        .ok_or(CpuError::Overflow(GuestAddress(d.pc)))? as u32;
    Ok(Step::Continue)
}

fn exec_addiu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    Ok(Step::Continue)
}

fn exec_slti(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = ((cpu.gpr[d.rs as usize] as i32) < (d.imm as i16 as i32)) as u32;
    Ok(Step::Continue)
}

fn exec_sltiu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = (cpu.gpr[d.rs as usize] < (d.imm as i16 as i32 as u32)) as u32;
    Ok(Step::Continue)
}

fn exec_andi(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = cpu.gpr[d.rs as usize] & d.imm as u32;
    Ok(Step::Continue)
}

fn exec_ori(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = cpu.gpr[d.rs as usize] | d.imm as u32;
    Ok(Step::Continue)
}

fn exec_xori(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = cpu.gpr[d.rs as usize] ^ u32::from(d.imm);
    Ok(Step::Continue)
}

fn exec_lui(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    cpu.gpr[d.rt as usize] = (d.imm as u32) << 16;
    Ok(Step::Continue)
}

fn exec_fpu(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let fs = d.rd as usize;
    let fd = ((d.word >> 6) & 31) as usize;
    let ft = d.rt as usize;
    match d.rs {
        0 => cpu.gpr[d.rt as usize] = cpu.fpr[fs],
        2 => cpu.gpr[d.rt as usize] = if fs == 31 { cpu.fcr31 } else { 0 },
        4 => cpu.fpr[fs] = cpu.gpr[d.rt as usize],
        6 => {
            if fs == 31 {
                cpu.fcr31 = cpu.gpr[d.rt as usize]
            }
        }
        8 => {
            let condition = cpu.fcr31 & (1 << 23) != 0;
            let taken = if d.rt & 1 != 0 { condition } else { !condition };
            if taken {
                cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
            } else if d.rt & 2 != 0 {
                cpu.pc = cpu.pc.wrapping_add(4);
            }
        }
        16 => {
            let s = f32::from_bits(cpu.fpr[fs]);
            let t = f32::from_bits(cpu.fpr[ft]);
            match d.word & 63 {
                0 => cpu.fpr[fd] = (s + t).to_bits(),
                1 => cpu.fpr[fd] = (s - t).to_bits(),
                2 => cpu.fpr[fd] = (s * t).to_bits(),
                3 => cpu.fpr[fd] = (s / t).to_bits(),
                4 => cpu.fpr[fd] = s.sqrt().to_bits(),
                5 => cpu.fpr[fd] = s.abs().to_bits(),
                6 => cpu.fpr[fd] = cpu.fpr[fs],
                7 => cpu.fpr[fd] = (-s).to_bits(),
                0x0c => cpu.fpr[fd] = (s.round_ties_even() as i32) as u32,
                0x0d => cpu.fpr[fd] = (s.trunc() as i32) as u32,
                0x0e => cpu.fpr[fd] = (s.ceil() as i32) as u32,
                0x0f => cpu.fpr[fd] = (s.floor() as i32) as u32,
                0x24 => cpu.fpr[fd] = (s as i32) as u32,
                function @ 0x30..=0x3f => {
                    let unordered = s.is_nan() || t.is_nan();
                    let less = !unordered && s < t;
                    let equal = !unordered && s == t;
                    let condition = (function & 1 != 0 && unordered)
                        || (function & 2 != 0 && equal)
                        || (function & 4 != 0 && less);
                    cpu.fcr31 = (cpu.fcr31 & !(1 << 23)) | ((condition as u32) << 23);
                }
                _ => {
                    return Err(CpuError::Unsupported {
                        pc: GuestAddress(d.pc),
                        word: d.word,
                    });
                }
            }
        }
        20 if d.word & 63 == 0x20 => {
            cpu.fpr[fd] = (cpu.fpr[fs] as i32 as f32).to_bits();
        }
        _ => {
            return Err(CpuError::Unsupported {
                pc: GuestAddress(d.pc),
                word: d.word,
            });
        }
    }
    Ok(Step::Continue)
}

fn exec_vfpu_move(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let register = (d.word & 0xff) as usize;
    match d.rs {
        3 if register < 128 => cpu.gpr[d.rt as usize] = cpu.vfpu[vfpu_scalar_lane(register as u32)],
        7 if register < 128 => cpu.vfpu[vfpu_scalar_lane(register as u32)] = cpu.gpr[d.rt as usize],
        // mfvc/mtvc move the vfpu control file (prefixes,
        // condition codes, rng state) through masked writes.
        3 if (128..144).contains(&register) => {
            cpu.gpr[d.rt as usize] = vfpu_read_ctrl(cpu, register as u32 - 128)
        }
        7 if (128..144).contains(&register) => {
            vfpu_write_ctrl(cpu, register as u32 - 128, cpu.gpr[d.rt as usize]);
        }
        // vfpu conditional branches.  bits 18..20 select a lane,
        // `any`, or `all`; bits 16..17 select false/true and likely.
        8 => {
            let condition_index = ((d.word >> 18) & 7) as usize;
            let branch_kind = (d.word >> 16) & 3;
            let condition = condition_index < 6 && cpu.vfpu_cc & (1 << condition_index) != 0;
            let taken = if branch_kind & 1 != 0 {
                condition
            } else {
                !condition
            };
            if taken {
                cpu.branch_target = Some(cpu.pc.wrapping_add(((d.imm as i16 as i32) << 2) as u32));
            } else if branch_kind & 2 != 0 {
                cpu.pc = cpu.pc.wrapping_add(4);
            }
        }
        // control registers are not yet externally observable;
        // accept their transfers so prefix setup remains bounded.
        3 => cpu.gpr[d.rt as usize] = 0,
        7 => {}
        _ => {
            return Err(CpuError::Unsupported {
                pc: GuestAddress(d.pc),
                word: d.word,
            });
        }
    }
    Ok(Step::Continue)
}

fn exec_lb(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    cpu.gpr[d.rt as usize] = memory.read_u8(a)? as i8 as i32 as u32;
    Ok(Step::Continue)
}

fn exec_lh(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    cpu.gpr[d.rt as usize] = memory.read_u16(a)? as i16 as i32 as u32;
    Ok(Step::Continue)
}

fn exec_lwl(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    let shift = (a & 3) * 8;
    let word = memory.read_u32(a & !3)?;
    let keep = 0x00ff_ffffu32.checked_shr(shift).unwrap_or(0);
    cpu.gpr[d.rt as usize] = (cpu.gpr[d.rt as usize] & keep) | word << (24 - shift);
    Ok(Step::Continue)
}

fn exec_lw(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    cpu.gpr[d.rt as usize] = memory.read_u32(a)?;
    Ok(Step::Continue)
}

fn exec_lbu(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    cpu.gpr[d.rt as usize] = u32::from(memory.read_u8(a)?);
    Ok(Step::Continue)
}

fn exec_lhu(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    cpu.gpr[d.rt as usize] = u32::from(memory.read_u16(a)?);
    Ok(Step::Continue)
}

fn exec_lwr(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    let shift = (a & 3) * 8;
    let word = memory.read_u32(a & !3)?;
    let keep = 0xffff_ff00u32.checked_shl(24 - shift).unwrap_or(0);
    cpu.gpr[d.rt as usize] = (cpu.gpr[d.rt as usize] & keep) | word >> shift;
    Ok(Step::Continue)
}

fn exec_sb(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    memory.write_u8(a, cpu.gpr[d.rt as usize] as u8)?;
    Ok(Step::Continue)
}

fn exec_sh(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    memory.write_u16(a, cpu.gpr[d.rt as usize] as u16)?;
    Ok(Step::Continue)
}

fn exec_swl(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    let byte = a & 3;
    let shift = (3 - byte) * 8;
    let old = memory.read_u32(a & !3)?;
    let keep = 0xffff_ff00u32.checked_shl(byte * 8).unwrap_or(0);
    memory.write_u32(a & !3, (old & keep) | cpu.gpr[d.rt as usize] >> shift)?;
    Ok(Step::Continue)
}

fn exec_sw(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    memory.write_u32(a, cpu.gpr[d.rt as usize])?;
    Ok(Step::Continue)
}

fn exec_swr(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    let shift = (a & 3) * 8;
    let old = memory.read_u32(a & !3)?;
    let keep = 0x00ff_ffffu32 >> (24 - shift);
    memory.write_u32(a & !3, (old & keep) | cpu.gpr[d.rt as usize] << shift)?;
    Ok(Step::Continue)
}

fn exec_cache(_: &mut Cpu, _: &mut Memory, _: Decoded) -> Result<Step, CpuError> {
    Ok(Step::Continue)
}

fn exec_ll(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    cpu.gpr[d.rt as usize] = memory.read_u32(a)?;
    Ok(Step::Continue)
}

fn exec_sc(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    memory.write_u32(a, cpu.gpr[d.rt as usize])?;
    cpu.gpr[d.rt as usize] = 1;
    Ok(Step::Continue)
}

fn exec_pref(_: &mut Cpu, _: &mut Memory, _: Decoded) -> Result<Step, CpuError> {
    Ok(Step::Continue)
}

fn exec_vsbn(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let length = ((((d.word >> 7) & 1) | ((d.word >> 14) & 2)) + 1) as usize;
    let destination = vfpu_vector_lanes(d.word & 0x7f, length);
    let source = vfpu_vector_lanes((d.word >> 8) & 0x7f, length);
    let target = vfpu_vector_lanes((d.word >> 16) & 0x7f, length);
    let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
    let target_values = vfpu_source_values(&cpu.vfpu, target, length, cpu.vfpu_t_prefix);
    // the shift amount is the raw integer lane, not a float
    // value: titles upload it with `mtv`.
    let shift = 127i32.wrapping_add(target_values[0] as i32) as u8 as u32;
    let mut values = source_values;
    let previous = values[0] & 0x7f80_0000;
    if previous != 0 && previous != 0x7f80_0000 {
        values[0] = (values[0] & !0x7f80_0000) | (shift << 23);
    }
    vfpu_write_vector(
        &mut cpu.vfpu,
        destination,
        length,
        values,
        cpu.vfpu_d_prefix,
        true,
    );
    cpu.reset_vfpu_prefixes();
    Ok(Step::Continue)
}

fn exec_vfpu_arith(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let sub = (d.word >> 23) & 7;
    let length = ((((d.word >> 7) & 1) | ((d.word >> 14) & 2)) + 1) as usize;
    let destination = vfpu_vector_lanes(d.word & 0x7f, length);
    let source = vfpu_vector_lanes((d.word >> 8) & 0x7f, length);
    let target = vfpu_vector_lanes((d.word >> 16) & 0x7f, length);
    let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
    let target_values = vfpu_source_values(&cpu.vfpu, target, length, cpu.vfpu_t_prefix);
    let mut values = [0u32; 4];
    for index in 0..length {
        let left = f32::from_bits(source_values[index]);
        let right = f32::from_bits(target_values[index]);
        let value = match (d.op, sub) {
            (0x18, 0) => left + right,
            (0x18, 1) => left - right,
            (0x18, 7) => left / right,
            (0x19, 0) => left * right,
            _ => unreachable!("classified by handler_for"),
        };
        values[index] = value.to_bits();
    }
    vfpu_write_vector(
        &mut cpu.vfpu,
        destination,
        length,
        values,
        cpu.vfpu_d_prefix,
        true,
    );
    cpu.reset_vfpu_prefixes();
    Ok(Step::Continue)
}

fn exec_vfpu_horizontal(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let operation = (d.word >> 23) & 7;
    let length = ((((d.word >> 7) & 1) | ((d.word >> 14) & 2)) + 1) as usize;
    let destination = vfpu_vector_lanes(d.word & 0x7f, 1);
    let source_reg = (d.word >> 8) & 0x7f;
    let target_reg = (d.word >> 16) & 0x7f;
    let source_lanes = vfpu_vector_lanes(source_reg, length);
    let mut source_raw = [0u32; 4];
    for (index, lane) in source_lanes.iter().take(length).enumerate() {
        source_raw[index] = cpu.vfpu[*lane];
    }
    let mut target_raw = [0u32; 4];
    if operation != 2 {
        let target_lanes = vfpu_vector_lanes(target_reg, length);
        for (index, lane) in target_lanes.iter().take(length).enumerate() {
            target_raw[index] = cpu.vfpu[*lane];
        }
    }
    // scalar horizontal results share one destination lane with
    // the full destination prefix (saturation and write mask).
    let write_scalar = |cpu: &mut Cpu, value: u32| {
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            1,
            [value, 0, 0, 0],
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
    };
    match operation {
        // vdot: quad-width dot product.
        1 => {
            let left = vfpu_prefix_quad(source_raw, length, cpu.vfpu_s_prefix);
            let right = vfpu_prefix_quad(target_raw, length, cpu.vfpu_t_prefix);
            let mut sum = 0.0f32;
            for index in 0..4 {
                sum += f32::from_bits(left[index]) * f32::from_bits(right[index]);
            }
            let value = sum.to_bits();
            write_scalar(cpu, value);
            Ok(Step::Continue)
        }
        // vscl: the scalar is the single lane named by the target
        // encoding, broadcast with a forced swizzle.
        2 => {
            let source_values =
                vfpu_source_values(&cpu.vfpu, source_lanes, length, cpu.vfpu_s_prefix);
            let lane = ((target_reg >> 5) & 3) as usize;
            let mut scalar_raw = [0u32; 4];
            scalar_raw[lane] = cpu.vfpu[vfpu_scalar_lane(target_reg)];
            let scalar = vfpu_prefix_quad(
                scalar_raw,
                4,
                vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_any_swizzle(),
                    vfpu_swizzle(lane as u32, lane as u32, lane as u32, lane as u32),
                ),
            );
            let mut values = [0u32; 4];
            for index in 0..length {
                values[index] = (f32::from_bits(source_values[index])
                    * f32::from_bits(scalar[index]))
                .to_bits();
            }
            vfpu_write_vector(
                &mut cpu.vfpu,
                vfpu_vector_lanes(d.word & 0x7f, length),
                length,
                values,
                cpu.vfpu_d_prefix,
                true,
            );
            cpu.reset_vfpu_prefixes();
            Ok(Step::Continue)
        }
        // vhdp: like vdot, but the last source lane is forced to
        // the 1.0 constant (negate/abs from the guest prefix
        // still apply to it).
        4 => {
            let remove = match length {
                4 => vfpu_swizzle(0, 0, 0, 3),
                3 => vfpu_swizzle(0, 0, 3, 0),
                2 => vfpu_swizzle(0, 3, 0, 0),
                _ => vfpu_swizzle(3, 0, 0, 0),
            };
            let add = vfpu_make_constants(
                if length == 1 { 1 } else { -1 },
                if length == 2 { 1 } else { -1 },
                if length == 3 { 1 } else { -1 },
                if length == 4 { 1 } else { -1 },
            );
            let left = vfpu_prefix_quad(
                source_raw,
                length,
                vfpu_rewrite_prefix(cpu.vfpu_s_prefix, remove, add),
            );
            let right = vfpu_prefix_quad(target_raw, length, cpu.vfpu_t_prefix);
            let mut sum = 0.0f32;
            for index in 0..4 {
                sum += f32::from_bits(left[index]) * f32::from_bits(right[index]);
            }
            let sum = if sum.is_nan() { sum.abs() } else { sum };
            let value = sum.to_bits();
            write_scalar(cpu, value);
            Ok(Step::Continue)
        }
        // vcrs: half cross product with forced yzx/zxy swizzles.
        5 => {
            let left = vfpu_source_values(
                &cpu.vfpu,
                source_lanes,
                length,
                vfpu_rewrite_prefix(
                    cpu.vfpu_s_prefix,
                    vfpu_swizzle(3, 3, 3, 0),
                    vfpu_swizzle(1, 2, 0, 0),
                ),
            );
            let right = vfpu_source_values(
                &cpu.vfpu,
                vfpu_vector_lanes(target_reg, length),
                length,
                vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_swizzle(3, 3, 3, 0),
                    vfpu_swizzle(2, 0, 1, 0),
                ),
            );
            let mut values = [0u32; 4];
            for index in 0..length {
                values[index] =
                    (f32::from_bits(left[index]) * f32::from_bits(right[index])).to_bits();
            }
            vfpu_write_vector(
                &mut cpu.vfpu,
                vfpu_vector_lanes(d.word & 0x7f, length),
                length,
                values,
                cpu.vfpu_d_prefix,
                true,
            );
            cpu.reset_vfpu_prefixes();
            Ok(Step::Continue)
        }
        // vdet: 2x2 determinant with a forced yx swizzle on t.
        _ => {
            let left = vfpu_prefix_quad(source_raw, length, cpu.vfpu_s_prefix);
            let right = vfpu_prefix_quad(
                target_raw,
                length,
                vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_swizzle(3, 3, 0, 0),
                    vfpu_swizzle(1, 0, 0, 0),
                ),
            );
            let s = [
                f32::from_bits(left[0]),
                f32::from_bits(left[1]),
                f32::from_bits(left[2]),
                f32::from_bits(left[3]),
            ];
            let t = [
                f32::from_bits(right[0]),
                f32::from_bits(right[1]),
                f32::from_bits(right[2]),
                f32::from_bits(right[3]),
            ];
            let value = (s[0] * t[0] - s[1] * t[1] + s[2] * t[2] + s[3] * t[3]).to_bits();
            write_scalar(cpu, value);
            Ok(Step::Continue)
        }
    }
}

fn exec_vcmp(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let condition = d.word & 15;
    let length = ((((d.word >> 7) & 1) | ((d.word >> 14) & 2)) + 1) as usize;
    let source = vfpu_vector_lanes((d.word >> 8) & 0x7f, length);
    let target = vfpu_vector_lanes((d.word >> 16) & 0x7f, length);
    let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
    let target_values = vfpu_source_values(&cpu.vfpu, target, length, cpu.vfpu_t_prefix);
    let mut lane_bits = 0u32;
    let mut any = false;
    let mut all = true;
    for index in 0..length {
        let left = f32::from_bits(source_values[index]);
        let right = f32::from_bits(target_values[index]);
        let result = match condition {
            0 => false,
            1 => left == right,
            2 => left < right,
            3 => left <= right,
            4 => true,
            5 => left != right,
            6 => left >= right,
            7 => left > right,
            8 => left == 0.0,
            9 => left.is_nan(),
            10 => left.is_infinite(),
            11 => !left.is_finite(),
            12 => left != 0.0,
            13 => !left.is_nan(),
            14 => !left.is_infinite(),
            15 => left.is_finite(),
            _ => unreachable!("condition is 4 bits"),
        };
        lane_bits |= (result as u32) << index;
        any |= result;
        all &= result;
    }
    let affected = ((1u32 << length) - 1) | 0x30;
    cpu.vfpu_cc = (cpu.vfpu_cc & !affected) | lane_bits | ((any as u32) << 4) | ((all as u32) << 5);
    cpu.reset_vfpu_prefixes();
    Ok(Step::Continue)
}

fn exec_vminmax_cmp(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let operation = (d.word >> 23) & 7;
    let length = ((((d.word >> 7) & 1) | ((d.word >> 14) & 2)) + 1) as usize;
    let destination = vfpu_vector_lanes(d.word & 0x7f, length);
    let source = vfpu_vector_lanes((d.word >> 8) & 0x7f, length);
    let target = vfpu_vector_lanes((d.word >> 16) & 0x7f, length);
    let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
    let target_values = vfpu_source_values(&cpu.vfpu, target, length, cpu.vfpu_t_prefix);
    let mut values = [0u32; 4];
    for index in 0..length {
        let left_bits = source_values[index];
        let right_bits = target_values[index];
        let left = f32::from_bits(left_bits);
        let right = f32::from_bits(right_bits);
        values[index] = match operation {
            2 | 3 if !left.is_finite() || !right.is_finite() => {
                let (left_signed, right_signed) = (left_bits as i32, right_bits as i32);
                if operation == 2 {
                    if left_signed < 0 && right_signed < 0 {
                        left_signed.max(right_signed) as u32
                    } else {
                        left_signed.min(right_signed) as u32
                    }
                } else if left_signed < 0 && right_signed < 0 {
                    left_signed.min(right_signed) as u32
                } else {
                    left_signed.max(right_signed) as u32
                }
            }
            2 => {
                if left < right {
                    left_bits
                } else {
                    right_bits
                }
            }
            3 => {
                if left > right {
                    left_bits
                } else {
                    right_bits
                }
            }
            5 => {
                let difference = left - right;
                let ordering = if difference.is_nan() {
                    let left_magnitude = (left_bits & 0x7fff_ffff) as i64;
                    let right_magnitude = (right_bits & 0x7fff_ffff) as i64;
                    let signed_left = if left_bits >> 31 != 0 {
                        -left_magnitude
                    } else {
                        left_magnitude
                    };
                    let signed_right = if right_bits >> 31 != 0 {
                        -right_magnitude
                    } else {
                        right_magnitude
                    };
                    signed_left.cmp(&signed_right)
                } else {
                    difference
                        .partial_cmp(&0.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                };
                match ordering {
                    std::cmp::Ordering::Less => (-1.0f32).to_bits(),
                    std::cmp::Ordering::Equal => 0.0f32.to_bits(),
                    std::cmp::Ordering::Greater => 1.0f32.to_bits(),
                }
            }
            6 => if !left.is_nan() && !right.is_nan() && left >= right {
                1.0f32
            } else {
                0.0
            }
            .to_bits(),
            7 => if !left.is_nan() && !right.is_nan() && left < right {
                1.0f32
            } else {
                0.0
            }
            .to_bits(),
            _ => unreachable!("classified by handler_for"),
        };
    }
    // out-of-range swizzles wire their destination lane to zero.
    vfpu_retain_invalid_swizzle(&mut values, cpu.vfpu_s_prefix, cpu.vfpu_t_prefix, length);
    vfpu_write_vector(
        &mut cpu.vfpu,
        destination,
        length,
        values,
        cpu.vfpu_d_prefix,
        true,
    );
    cpu.reset_vfpu_prefixes();
    Ok(Step::Continue)
}

fn exec_special3(cpu: &mut Cpu, _: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    match d.word & 63 {
        // allegrex mips32r2 bitfield extraction.
        0x00 => {
            let position = (d.word >> 6) & 31;
            let size = ((d.word >> 11) & 31) + 1;
            let mask = if size == 32 {
                u32::MAX
            } else {
                (1u32 << size) - 1
            };
            cpu.gpr[d.rt as usize] = (cpu.gpr[d.rs as usize] >> position) & mask;
            Ok(Step::Continue)
        }
        // ins encodes the most-significant bit in rd and the least-
        // significant bit in sa.
        0x04 => {
            let lsb = (d.word >> 6) & 31;
            let msb = (d.word >> 11) & 31;
            if msb < lsb {
                return Err(CpuError::Unsupported {
                    pc: GuestAddress(d.pc),
                    word: d.word,
                });
            }
            let size = msb - lsb + 1;
            let field_mask = if size == 32 {
                u32::MAX
            } else {
                (1u32 << size) - 1
            };
            let mask = field_mask << lsb;
            cpu.gpr[d.rt as usize] =
                (cpu.gpr[d.rt as usize] & !mask) | ((cpu.gpr[d.rs as usize] & field_mask) << lsb);
            Ok(Step::Continue)
        }
        // allegrex byte/bit shuffles (special3/bshfl).
        0x20 => {
            cpu.gpr[d.rd as usize] = match (d.word >> 6) & 31 {
                2 => cpu.gpr[d.rt as usize].swap_bytes().rotate_left(16), // wsbh
                3 => cpu.gpr[d.rt as usize].swap_bytes(),                 // wsbw
                16 => cpu.gpr[d.rt as usize] as i8 as i32 as u32,         // seb
                20 => cpu.gpr[d.rt as usize].reverse_bits(),              // bitrev
                24 => cpu.gpr[d.rt as usize] as i16 as i32 as u32,        // seh
                _ => {
                    return Err(CpuError::Unsupported {
                        pc: GuestAddress(d.pc),
                        word: d.word,
                    });
                }
            };
            Ok(Step::Continue)
        }
        _ => Err(CpuError::Unsupported {
            pc: GuestAddress(d.pc),
            word: d.word,
        }),
    }
}

fn exec_lwc1(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    cpu.fpr[d.rt as usize] = memory.read_u32(a)?;
    Ok(Step::Continue)
}

fn exec_swc1(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let a = cpu.gpr[d.rs as usize].wrapping_add(d.imm as i16 as i32 as u32);
    memory.write_u32(a, cpu.fpr[d.rt as usize])?;
    Ok(Step::Continue)
}

fn exec_lvs(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let register = ((d.word >> 16) & 31) | ((d.word & 3) << 5);
    let a = cpu.gpr[d.rs as usize].wrapping_add((d.word as u16 & !3) as i16 as i32 as u32);
    cpu.vfpu[vfpu_scalar_lane(register)] = memory.read_u32(a)?;
    Ok(Step::Continue)
}
fn exec_vcst(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        const CONSTANTS: [f32; 20] = [
            0.0,
            f32::MAX,
            std::f32::consts::SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_2_SQRT_PI,
            std::f32::consts::FRAC_2_PI,
            std::f32::consts::FRAC_1_PI,
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            std::f32::consts::E,
            std::f32::consts::LOG2_E,
            std::f32::consts::LOG10_E,
            std::f32::consts::LN_2,
            std::f32::consts::LN_10,
            std::f32::consts::TAU,
            std::f32::consts::FRAC_PI_6,
            std::f32::consts::LOG10_2,
            std::f32::consts::LOG2_10,
            0.866_025_4,
        ];
        let constant = ((word >> 16) & 31) as usize;
        let value = CONSTANTS.get(constant).copied().unwrap_or(0.0).to_bits();
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, length);
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            length,
            [value; 4],
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vcmov(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let pc = d.pc;
    let word = d.word;
    let result: Step = {
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, length);
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
        let move_on_set = (word >> 19) & 1 == 0;
        let condition_index = ((word >> 16) & 7) as usize;
        if condition_index > 6 {
            return Err(CpuError::Unsupported {
                pc: GuestAddress(pc),
                word,
            });
        }
        let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
        let mut values = vfpu_source_values(&cpu.vfpu, destination, length, cpu.vfpu_t_prefix);
        for index in 0..length {
            let bit = if condition_index == 6 {
                index
            } else {
                condition_index
            };
            let set = cpu.vfpu_cc & (1 << bit) != 0;
            if set == move_on_set {
                values[index] = source_values[index];
            }
        }
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            length,
            values,
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vi2f(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let scale = 1.0f32 / (1u64 << ((word >> 16) & 31)) as f32;
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, length);
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
        let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
        let mut values = [0u32; 4];
        for index in 0..length {
            values[index] = ((source_values[index] as i32 as f32) * scale).to_bits();
        }
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            length,
            values,
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vf2i(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let rs = d.rs as usize;
    let result: Step = {
        let multiplier = (1u64 << ((word >> 16) & 31)) as f64;
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, length);
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
        let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
        let mut values = [0u32; 4];
        for index in 0..length {
            let input = f32::from_bits(source_values[index]);
            values[index] = if input.is_nan() {
                i32::MAX as u32
            } else {
                let scaled = f64::from(input) * multiplier;
                if scaled > f64::from(i32::MAX) {
                    i32::MAX as u32
                } else if scaled <= f64::from(i32::MIN) {
                    i32::MIN as u32
                } else {
                    let rounded = match rs {
                        16 => scaled.round_ties_even(),
                        17 => scaled.trunc(),
                        18 => scaled.ceil(),
                        19 => scaled.floor(),
                        _ => unreachable!(),
                    };
                    rounded as i32 as u32
                }
            };
        }
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            length,
            values,
            cpu.vfpu_d_prefix,
            false,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vfpu7(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let operation = (word >> 16) & 31;
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let source_reg = (word >> 8) & 0x7f;
        match operation {
            // vrnds: seed the rng from the s-prefixed destination
            // lane; vrndi/vrndf1/vrndf2: emit generator words
            // backwards into the destination lanes.
            0 => {
                let lane = vfpu_scalar_lane(word & 0x7f);
                let seed = vfpu_prefix_quad([cpu.vfpu[lane], 0, 0, 0], 1, cpu.vfpu_s_prefix)[0];
                vfpu_rng_seed(seed, &mut cpu.vfpu_ctrl_extra[4..12]);
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            1..=3 => {
                let destination = vfpu_vector_lanes(word & 0x7f, length);
                let mut values = [0u32; 4];
                for index in (0..length).rev() {
                    let generated = vfpu_rng_generate(&mut cpu.vfpu_ctrl_extra[4..12]);
                    values[index] = match operation {
                        1 => generated,
                        2 => 0x3f80_0000 | (generated & 0x007f_ffff),
                        _ => 0x4000_0000 | (generated & 0x007f_ffff),
                    };
                }
                // the d prefix is broken here: only the last lane
                // sees mask and saturation.
                let isolated = vfpu_isolate_prefix_lane(cpu.vfpu_d_prefix, length - 1);
                vfpu_write_vector(&mut cpu.vfpu, destination, length, values, isolated, true);
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vf2h: pack float lanes to half floats; vh2f: expand.
            18 | 19 => {
                let source = vfpu_vector_lanes(source_reg, length);
                let source_values =
                    vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
                let mut values = [0u32; 4];
                let output_length;
                if operation == 19 {
                    output_length = if length == 1 { 2 } else { 4 };
                    let destination = vfpu_vector_lanes(word & 0x7f, output_length);
                    for index in 0..output_length {
                        let packed = source_values[index / 2];
                        let half = if index & 1 == 0 {
                            packed as u16
                        } else {
                            (packed >> 16) as u16
                        };
                        values[index] = expand_half(half);
                    }
                    vfpu_write_vector(
                        &mut cpu.vfpu,
                        destination,
                        output_length,
                        values,
                        cpu.vfpu_d_prefix,
                        true,
                    );
                } else {
                    // the source prefix applies at quad width, so a
                    // swizzled single still packs both components.
                    let mut raw = [0u32; 4];
                    for (index, lane) in source.iter().take(length).enumerate() {
                        raw[index] = cpu.vfpu[*lane];
                    }
                    let quad = vfpu_prefix_quad(raw, length, cpu.vfpu_s_prefix);
                    let mut quad = quad;
                    vfpu_retain_invalid_swizzle(&mut quad, cpu.vfpu_s_prefix, cpu.vfpu_t_prefix, 4);
                    output_length = if length <= 2 { 1 } else { 2 };
                    let destination = vfpu_vector_lanes(word & 0x7f, output_length);
                    for index in 0..output_length {
                        let low = shrink_half(quad[index * 2]);
                        let high = shrink_half(quad[index * 2 + 1]);
                        values[index] = u32::from(low) | (u32::from(high) << 16);
                    }
                    vfpu_write_vector(
                        &mut cpu.vfpu,
                        destination,
                        output_length,
                        values,
                        cpu.vfpu_d_prefix,
                        true,
                    );
                }
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vsbz: extract the mantissa (exponent becomes 127).
            22 => {
                let source = vfpu_vector_lanes(source_reg, length);
                let source_values =
                    vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
                let mut values = source_values;
                let first = values[0];
                if f32::from_bits(first).is_nan() || first & 0x7f80_0000 == 0 {
                    values[0] = first;
                } else {
                    values[0] = (127 << 23) | (first & 0x007f_ffff);
                }
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, length),
                    length,
                    values,
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vlgb: extract the binary exponent.
            _ => {
                let source = vfpu_vector_lanes(source_reg, length);
                let source_values =
                    vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
                let mut values = source_values;
                let exponent = (values[0] & 0x7f80_0000) >> 23;
                values[0] = if exponent == 0xff {
                    values[0]
                } else if exponent == 0 {
                    f32::NEG_INFINITY.to_bits()
                } else {
                    ((exponent as i32 - 127) as f32).to_bits()
                };
                vfpu_retain_invalid_swizzle(
                    &mut values,
                    cpu.vfpu_s_prefix,
                    cpu.vfpu_t_prefix,
                    length,
                );
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, length),
                    length,
                    values,
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
        }
    };
    Ok(result)
}

fn exec_vwbn(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
        let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
        let mut values = source_values;
        let biased = (word >> 16) & 0xff;
        let first = values[0];
        let previous = (first & 0x7f80_0000) >> 23;
        let mantissa = (first & 0x007f_ffff) | 0x0080_0000;
        if previous != 0xff && previous != 0 {
            let adjusted = if biased > previous {
                mantissa >> ((biased - previous) & 0xf)
            } else {
                mantissa << ((previous - biased) & 0xf)
            };
            values[0] = (first & 0x8000_0000) | (adjusted & 0x007f_ffff) | (biased << 23);
        } else {
            values[0] = first | (biased << 23);
        }
        vfpu_retain_invalid_swizzle(&mut values, cpu.vfpu_s_prefix, cpu.vfpu_t_prefix, length);
        vfpu_write_vector(
            &mut cpu.vfpu,
            vfpu_vector_lanes(word & 0x7f, length),
            length,
            values,
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vi2x(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let operation = (word >> 16) & 3;
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, 4);
        let mut raw = [0u32; 4];
        for (index, lane) in source.iter().enumerate() {
            raw[index] = cpu.vfpu[*lane];
        }
        let prefixed = vfpu_prefix_quad(raw, 4, cpu.vfpu_s_prefix);
        let mut signed = [0i32; 4];
        for (index, value) in signed.iter_mut().enumerate() {
            *value = prefixed[index] as i32;
        }
        let mut packed = [0u32; 2];
        let output_length = match operation {
            0 => {
                for (index, value) in signed.iter().enumerate() {
                    let mut lane = *value;
                    if lane < 0 {
                        lane = 0;
                    }
                    packed[0] |= (((lane >> 23) & 0xff) as u32) << (index * 8);
                }
                1
            }
            1 => {
                for (index, value) in signed.iter().enumerate() {
                    packed[0] |= ((*value as u32) >> 24) << (index * 8);
                }
                1
            }
            2 => {
                let elements = length.div_ceil(2);
                for index in 0..elements {
                    let mut low = signed[index * 2];
                    let mut high = signed[index * 2 + 1];
                    if low < 0 {
                        low = 0;
                    }
                    if high < 0 {
                        high = 0;
                    }
                    packed[index] = ((low >> 15) as u32) | (((high >> 15) as u32) << 16);
                }
                if length >= 3 { 2 } else { 1 }
            }
            _ => {
                let elements = length.div_ceil(2);
                for index in 0..elements {
                    let low = signed[index * 2] as u32;
                    let high = signed[index * 2 + 1] as u32;
                    packed[index] = (low >> 16) | (high >> 16 << 16);
                }
                if length >= 3 { 2 } else { 1 }
            }
        };
        let destination = vfpu_vector_lanes(word & 0x7f, output_length);
        let mut values = [0u32; 4];
        values[..output_length].copy_from_slice(&packed[..output_length]);
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            output_length,
            values,
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vx2i(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let operation = (word >> 16) & 3;
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
        let source_values = vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
        let mut values = [0u32; 4];
        let output_length = if operation <= 1 || length >= 2 { 4 } else { 2 };
        if operation <= 1 {
            let packed = source_values[0];
            for (index, value) in values.iter_mut().enumerate() {
                let byte = (packed >> (index * 8)) & 0xff;
                *value = if operation == 0 {
                    byte.wrapping_mul(0x0101_0101) >> 1
                } else {
                    byte << 24
                };
            }
        } else {
            for index in 0..length.min(2) {
                let packed = source_values[index];
                values[index * 2] = if operation == 2 {
                    (packed & 0xffff) << 15
                } else {
                    (packed & 0xffff) << 16
                };
                values[index * 2 + 1] = if operation == 2 {
                    (packed & 0xffff_0000) >> 1
                } else {
                    packed & 0xffff_0000
                };
            }
        }
        let destination = vfpu_vector_lanes(word & 0x7f, output_length);
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            output_length,
            values,
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vfpu9(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let operation = (word >> 16) & 31;
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        // lane-wise min/max with c++ std::min/std::max nan behavior
        // (a nan operand yields the first operand, unlike rust's
        // propagating `f32::min`/`f32::max`).
        let pp_min = |a: f32, b: f32| if b < a { b } else { a };
        let pp_max = |a: f32, b: f32| if a < b { b } else { a };
        match operation {
            // vsrt1..4: parallel min/max with forced t swizzles.
            0 | 1 | 8 | 9 => {
                let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
                let source_values =
                    vfpu_source_values(&cpu.vfpu, source, length, cpu.vfpu_s_prefix);
                let swizzle = if operation == 0 || operation == 8 {
                    vfpu_swizzle(1, 0, 3, 2)
                } else {
                    vfpu_swizzle(3, 2, 1, 0)
                };
                let target_values = vfpu_source_values(
                    &cpu.vfpu,
                    source,
                    length,
                    vfpu_rewrite_prefix(cpu.vfpu_t_prefix, vfpu_swizzle(3, 3, 3, 3), swizzle),
                );
                let mut s = [0.0f32; 4];
                let mut t = [0.0f32; 4];
                for index in 0..length {
                    s[index] = f32::from_bits(source_values[index]);
                    t[index] = f32::from_bits(target_values[index]);
                }
                let mut values = [0u32; 4];
                for index in 0..length {
                    values[index] = match (operation, index) {
                        (0, 0) | (0, 2) | (1, 0) | (1, 1) | (8, 1) | (8, 3) | (9, 2) | (9, 3) => {
                            pp_min(s[index], t[index]).to_bits()
                        }
                        _ => pp_max(s[index], t[index]).to_bits(),
                    };
                }
                vfpu_retain_invalid_swizzle(
                    &mut values,
                    cpu.vfpu_s_prefix,
                    cpu.vfpu_t_prefix,
                    length,
                );
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, length),
                    length,
                    values,
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vbfy1/2: butterfly sums with forced negate+swizzle.
            2 | 3 => {
                let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
                let (neg, swizzle) = if operation == 2 {
                    (vfpu_negate_bits(0, 1, 0, 1), vfpu_swizzle(1, 0, 3, 2))
                } else {
                    (vfpu_negate_bits(0, 0, 1, 1), vfpu_swizzle(2, 3, 0, 1))
                };
                let t_values = vfpu_source_values(
                    &cpu.vfpu,
                    source,
                    length,
                    vfpu_rewrite_prefix(cpu.vfpu_t_prefix, vfpu_any_swizzle(), swizzle),
                );
                // s gets the forced negate flags on top of the guest
                // prefix (constants/abs keep working).
                let s_neg = vfpu_rewrite_prefix(cpu.vfpu_s_prefix, 0, neg);
                let s_values = vfpu_source_values(&cpu.vfpu, source, length, s_neg);
                let mut values = [0u32; 4];
                for index in 0..length {
                    values[index] = (f32::from_bits(s_values[index])
                        + f32::from_bits(t_values[index]))
                    .to_bits();
                }
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, length),
                    length,
                    values,
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vocp: one's complement (1-x, nan-safe absolute).
            4 => {
                let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
                let forced_s =
                    vfpu_rewrite_prefix(cpu.vfpu_s_prefix, 0, vfpu_negate_bits(1, 1, 1, 1));
                let s_values = vfpu_source_values(&cpu.vfpu, source, length, forced_s);
                let forced_t = vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_any_swizzle(),
                    vfpu_make_constants(1, 1, 1, 1),
                );
                let t_values = vfpu_source_values(&cpu.vfpu, source, length, forced_t);
                let mut values = [0u32; 4];
                for index in 0..length {
                    let s = f32::from_bits(s_values[index]);
                    let t = f32::from_bits(t_values[index]);
                    values[index] = (if s.is_nan() { s.abs() } else { t + s }).to_bits();
                }
                vfpu_retain_invalid_swizzle(
                    &mut values,
                    cpu.vfpu_s_prefix,
                    cpu.vfpu_t_prefix,
                    length,
                );
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, length),
                    length,
                    values,
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vsocp: paired one's complement saturating to [0, 1].
            5 => {
                let output_length = match length {
                    1 => 2,
                    2 | 4 => 4,
                    _ => 4,
                };
                let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
                let mut raw = [0u32; 4];
                for (index, lane) in source.iter().take(length).enumerate() {
                    raw[index] = cpu.vfpu[*lane];
                }
                let s_values = vfpu_prefix_quad(
                    raw,
                    output_length,
                    vfpu_rewrite_prefix(
                        cpu.vfpu_s_prefix,
                        vfpu_any_swizzle() | vfpu_negate_bits(1, 1, 1, 1),
                        vfpu_swizzle(0, 0, 1, 1) | vfpu_negate_bits(1, 0, 1, 0),
                    ),
                );
                let t_values = vfpu_prefix_quad(
                    [0u32; 4],
                    output_length,
                    vfpu_rewrite_prefix(
                        cpu.vfpu_t_prefix,
                        vfpu_any_swizzle(),
                        vfpu_make_constants(1, 0, 1, 0),
                    ),
                );
                let mut values = [0u32; 4];
                for index in 0..output_length {
                    let sum = f32::from_bits(t_values[index]) + f32::from_bits(s_values[index]);
                    values[index] = (if sum.is_nan() {
                        sum
                    } else {
                        sum.clamp(0.0, 1.0)
                    })
                    .to_bits();
                }
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, output_length),
                    output_length,
                    values,
                    cpu.vfpu_d_prefix,
                    false,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vfad/vavg: horizontal sum / average at quad width.
            6 | 7 => {
                let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
                let mut raw = [0u32; 4];
                for (index, lane) in source.iter().take(length).enumerate() {
                    raw[index] = cpu.vfpu[*lane];
                }
                let s_values = vfpu_prefix_quad(raw, length, cpu.vfpu_s_prefix);
                let (remove, add) = if operation == 6 {
                    (vfpu_any_swizzle(), vfpu_make_constants(1, 1, 1, 1))
                } else {
                    let constant = match length {
                        1 => 0,
                        2 => 3,
                        3 => 5,
                        _ => 6,
                    };
                    (
                        vfpu_any_swizzle() | vfpu_abs_bits(1, 1, 1, 1),
                        vfpu_make_constants(constant, constant, constant, constant),
                    )
                };
                let t_values = vfpu_prefix_quad(
                    [0u32; 4],
                    4,
                    vfpu_rewrite_prefix(cpu.vfpu_t_prefix, remove, add),
                );
                let mut sum = 0.0f32;
                for index in 0..4 {
                    sum += f32::from_bits(s_values[index]) * f32::from_bits(t_values[index]);
                }
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, 1),
                    1,
                    [sum.to_bits(), 0, 0, 0],
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vsgn: lane-wise sign against the (possibly
            // abs/negate-modified) zero constant.
            10 => {
                let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
                let mut raw = [0u32; 4];
                for (index, lane) in source.iter().take(length).enumerate() {
                    raw[index] = cpu.vfpu[*lane];
                }
                let s_quad = vfpu_prefix_quad(raw, 4, cpu.vfpu_s_prefix);
                let t_forced = vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_any_swizzle(),
                    vfpu_make_constants(0, 0, 0, 0),
                );
                let t_values = vfpu_source_values(&cpu.vfpu, source, length, t_forced);
                let mut values = [0u32; 4];
                for index in 0..length {
                    let bits =
                        (f32::from_bits(s_quad[index]) - f32::from_bits(t_values[index])).to_bits();
                    let value = if bits == 0 || bits == 0x8000_0000 {
                        0.0f32
                    } else if bits >> 31 == 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    values[index] = value.to_bits();
                }
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, length),
                    length,
                    values,
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
            // vmfvc/vmtvc: vfpu control register moves. prefixes are
            // intentionally left alone.
            16 => {
                let index = (word >> 8) & 0x7f;
                let lane = vfpu_scalar_lane(word & 0x7f);
                cpu.vfpu[lane] = vfpu_read_ctrl(cpu, index);
                Step::Continue
            }
            17 => {
                let index = word & 0x7f;
                let lane = vfpu_scalar_lane((word >> 8) & 0x7f);
                vfpu_write_ctrl(cpu, index, cpu.vfpu[lane]);
                Step::Continue
            }
            // vt4444/vt5551/vt5650: float bytes to packed color.
            _ => {
                let source = vfpu_vector_lanes((word >> 8) & 0x7f, 4);
                let mut raw = [0u32; 4];
                for (index, lane) in source.iter().enumerate() {
                    raw[index] = cpu.vfpu[*lane];
                }
                let s_values = vfpu_prefix_quad(raw, 4, cpu.vfpu_s_prefix);
                let mut colors = [0u16; 4];
                for (index, color) in colors.iter_mut().enumerate() {
                    let input = s_values[index];
                    let (a, b, g, r) = (
                        ((input >> 24) & 0xff) as u16,
                        ((input >> 16) & 0xff) as u16,
                        ((input >> 8) & 0xff) as u16,
                        (input & 0xff) as u16,
                    );
                    *color = match operation {
                        25 => (a >> 4) << 12 | (b >> 4) << 8 | (g >> 4) << 4 | (r >> 4),
                        26 => (a >> 7) << 15 | (b >> 3) << 10 | (g >> 3) << 5 | (r >> 3),
                        _ => (b >> 3) << 11 | (g >> 2) << 5 | (r >> 3),
                    };
                }
                let packed = [
                    u32::from(colors[0]) | (u32::from(colors[1]) << 16),
                    u32::from(colors[2]) | (u32::from(colors[3]) << 16),
                    0,
                    0,
                ];
                let output_length = if length == 1 { 1 } else { 2 };
                vfpu_write_vector(
                    &mut cpu.vfpu,
                    vfpu_vector_lanes(word & 0x7f, output_length),
                    output_length,
                    packed,
                    cpu.vfpu_d_prefix,
                    true,
                );
                cpu.reset_vfpu_prefixes();
                Step::Continue
            }
        }
    };
    Ok(result)
}

fn exec_vpfx(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        match (word >> 23) & 7 {
            0 | 1 => cpu.vfpu_s_prefix = word & 0x000f_ffff,
            2 | 3 => cpu.vfpu_t_prefix = word & 0x000f_ffff,
            4 | 5 => cpu.vfpu_d_prefix = word & 0x0000_0fff,
            _ => unreachable!(),
        }
        Step::Continue
    };
    Ok(result)
}

fn exec_viim(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let imm = d.imm;
    let result: Step = {
        let destination = vfpu_vector_lanes((word >> 16) & 0x7f, 1)[0];
        let value = if (word >> 23) & 7 == 6 {
            (imm as i16 as f32).to_bits()
        } else {
            expand_half(imm)
        };
        let mut values = [0u32; 4];
        values[0] = value;
        vfpu_write_vector(
            &mut cpu.vfpu,
            [destination; 4],
            1,
            values,
            cpu.vfpu_d_prefix,
            true,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vfpu_unary(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let pc = d.pc;
    let word = d.word;
    let result: Step = {
        let operation = ((word >> 16) & 31) as u8;
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, length);
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
        // vabs/vneg force their abs/negate bits on top of the guest
        // prefix; the rest of the prefix keeps
        // working, which a plain `abs()`/`-` would get wrong under a
        // negated guest prefix.
        let s_prefix = match operation {
            1 => vfpu_rewrite_prefix(cpu.vfpu_s_prefix, 0, vfpu_abs_bits(1, 1, 1, 1)),
            2 => vfpu_rewrite_prefix(cpu.vfpu_s_prefix, 0, vfpu_negate_bits(1, 1, 1, 1)),
            _ => cpu.vfpu_s_prefix,
        };
        let source_values = vfpu_source_values(&cpu.vfpu, source, length, s_prefix);
        // transcendental ops prefix only their last lane (with an
        // infinite invalid default); other lanes stay raw.
        let transcendental = matches!(operation, 16..=28);
        let mut last_prefixed = source_values;
        if transcendental {
            let mut raw = [0u32; 4];
            for (index, lane) in source.iter().take(length).enumerate() {
                raw[index] = cpu.vfpu[*lane];
            }
            let invalid = if operation == 24 || operation == 26 {
                f32::NEG_INFINITY.to_bits()
            } else {
                f32::INFINITY.to_bits()
            };
            let remove = if operation == 24 || operation == 26 || operation == 28 {
                vfpu_negate_bits(1, 0, 0, 0)
            } else {
                0
            };
            let rewritten = vfpu_rewrite_prefix(cpu.vfpu_s_prefix, remove, 0);
            let register = (rewritten & 3) as usize;
            let absolute = rewritten & 0x100 != 0;
            let constant = rewritten & 0x1000 != 0;
            const TRANSCENDENTAL_CONSTANTS: [f32; 8] =
                [0.0, 1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 0.25, 1.0 / 6.0];
            let mut lane_value = if constant {
                TRANSCENDENTAL_CONSTANTS[register + usize::from(absolute) * 4].to_bits()
            } else if register == 0 {
                raw[length - 1]
            } else {
                invalid
            };
            if absolute && !constant {
                lane_value &= 0x7fff_ffff;
            }
            if rewritten & 0x10000 != 0 {
                lane_value ^= 0x8000_0000;
            }
            last_prefixed[length - 1] = lane_value;
        }
        // vidt/vzero/vone synthesize forced constants (guest negate
        // still applies); they never read the source value.
        let mut values = [0u32; 4];
        // transcendental lanes before the last see raw register
        // values; only the last lane is prefix-transformed.
        let mut raw_values = [0u32; 4];
        if transcendental {
            for (index, lane) in source.iter().take(length).enumerate() {
                raw_values[index] = cpu.vfpu[*lane];
            }
        }
        for index in 0..length {
            let input = if transcendental && index + 1 != length {
                f32::from_bits(raw_values[index])
            } else {
                f32::from_bits(last_prefixed[index])
            };
            let value = match operation {
                0..=2 => input,
                3 => {
                    // identity position comes from the destination
                    // register.
                    let off_mask = if length >= 3 { 3 } else { 1 };
                    let off = (word & 0x7f) as usize & off_mask;
                    let is_one = [0, 1, 2, 3].map(|lane| lane & off_mask == off);
                    let forced = vfpu_rewrite_prefix(
                        cpu.vfpu_s_prefix,
                        vfpu_any_swizzle(),
                        vfpu_make_constants(
                            if is_one[0] { 1 } else { 0 },
                            if is_one[1] { 1 } else { 0 },
                            if is_one[2] { 1 } else { 0 },
                            if is_one[3] { 1 } else { 0 },
                        ),
                    );
                    let generated = vfpu_prefix_quad([0u32; 4], length, forced);
                    f32::from_bits(generated[index])
                }
                4 => input.clamp(0.0, 1.0),
                5 => input.clamp(-1.0, 1.0),
                6 | 7 => {
                    let forced = vfpu_rewrite_prefix(
                        cpu.vfpu_s_prefix,
                        vfpu_any_swizzle(),
                        vfpu_make_constants(
                            if operation == 7 { 1 } else { 0 },
                            if operation == 7 { 1 } else { 0 },
                            if operation == 7 { 1 } else { 0 },
                            if operation == 7 { 1 } else { 0 },
                        ),
                    );
                    let generated = vfpu_prefix_quad([0u32; 4], length, forced);
                    f32::from_bits(generated[index])
                }
                16 => f32::from_bits(vfpu_recip_bits(input.to_bits())),
                17 => input.sqrt().recip(),
                18 => (input * std::f32::consts::FRAC_PI_2).sin(),
                19 => (input * std::f32::consts::FRAC_PI_2).cos(),
                20 => vfpu_exp2_value(input),
                21 => vfpu_log2_value(input),
                22 => input.sqrt().abs(),
                23 => vfpu_asin_value(input) * std::f32::consts::FRAC_2_PI,
                24 => -f32::from_bits(vfpu_recip_bits(input.to_bits())),
                26 => -(input * std::f32::consts::FRAC_PI_2).sin(),
                28 => vfpu_exp2_value(-input),
                _ => {
                    return Err(CpuError::Unsupported {
                        pc: GuestAddress(pc),
                        word,
                    });
                }
            };
            values[index] = value.to_bits();
        }
        // vsat1 skips saturation (mask only); the transcendental
        // group saturates/masks the last lane only.
        let (out_prefix, saturate) = match operation {
            5 => (cpu.vfpu_d_prefix, false),
            16..=28 => (
                vfpu_isolate_prefix_lane(cpu.vfpu_d_prefix, length - 1),
                true,
            ),
            _ => (cpu.vfpu_d_prefix, true),
        };
        vfpu_write_vector(
            &mut cpu.vfpu,
            destination,
            length,
            values,
            out_prefix,
            saturate,
        );
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_lvq(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let pc = d.pc;
    let word = d.word;
    let rs = d.rs as usize;
    let result: Step = {
        let register = ((word >> 16) & 31) | ((word & 1) << 5);
        let lanes = vfpu_vector_lanes(register, 4);
        let a = cpu.gpr[rs].wrapping_add((word as u16 & !3) as i16 as i32 as u32);
        // hardware rejects quad transfers that are not 16-byte
        // aligned; the destination is zeroed
        // while the fault is reported.
        if a & 15 != 0 {
            // report unmapped memory through the normal fault path.
            memory.read_u32(a & !15)?;
            for lane in lanes {
                cpu.vfpu[lane] = 0;
            }
            return Err(CpuError::Unsupported {
                pc: GuestAddress(pc),
                word,
            });
        }
        for (index, lane) in lanes.into_iter().enumerate() {
            cpu.vfpu[lane] = memory.read_u32(a + index as u32 * 4)?;
        }
        Step::Continue
    };
    Ok(result)
}

fn exec_lvl(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let rs = d.rs as usize;
    let result: Step = {
        let register = ((word >> 16) & 31) | ((word & 1) << 5);
        let lanes = vfpu_vector_lanes(register, 4);
        let a = cpu.gpr[rs].wrapping_add((word as u16 & !3) as i16 as i32 as u32);
        let mut values = [0u32; 4];
        for (index, lane) in lanes.iter().enumerate() {
            values[index] = cpu.vfpu[*lane];
        }
        let offset = ((a >> 2) & 3) as usize;
        if word & 2 == 0 {
            for (index, slot) in values.iter_mut().rev().enumerate().take(offset + 1) {
                *slot = memory.read_u32(a.wrapping_sub(index as u32 * 4))?;
            }
        } else {
            for (index, slot) in values.iter_mut().enumerate().take(4 - offset) {
                *slot = memory.read_u32(a.wrapping_add(index as u32 * 4))?;
            }
        }
        for (index, lane) in lanes.iter().enumerate() {
            cpu.vfpu[*lane] = values[index];
        }
        Step::Continue
    };
    Ok(result)
}

fn exec_svs(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let rs = d.rs as usize;
    let result: Step = {
        let register = ((word >> 16) & 31) | ((word & 3) << 5);
        let a = cpu.gpr[rs].wrapping_add((word as u16 & !3) as i16 as i32 as u32);
        memory.write_u32(a, cpu.vfpu[vfpu_scalar_lane(register)])?;
        Step::Continue
    };
    Ok(result)
}

fn exec_svq(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let pc = d.pc;
    let word = d.word;
    let rs = d.rs as usize;
    let result: Step = {
        let register = ((word >> 16) & 31) | ((word & 1) << 5);
        let lanes = vfpu_vector_lanes(register, 4);
        let a = cpu.gpr[rs].wrapping_add((word as u16 & !3) as i16 as i32 as u32);
        if a & 15 != 0 {
            memory.read_u32(a & !15)?;
            return Err(CpuError::Unsupported {
                pc: GuestAddress(pc),
                word,
            });
        }
        for (index, lane) in lanes.into_iter().enumerate() {
            memory.write_u32(a + index as u32 * 4, cpu.vfpu[lane])?;
        }
        Step::Continue
    };
    Ok(result)
}

fn exec_svl(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let rs = d.rs as usize;
    let result: Step = {
        let register = ((word >> 16) & 31) | ((word & 1) << 5);
        let lanes = vfpu_vector_lanes(register, 4);
        let a = cpu.gpr[rs].wrapping_add((word as u16 & !3) as i16 as i32 as u32);
        let offset = ((a >> 2) & 3) as usize;
        if word & 2 == 0 {
            for index in 0..=offset {
                let lane = lanes[3 - index];
                memory.write_u32(a.wrapping_sub(index as u32 * 4), cpu.vfpu[lane])?;
            }
        } else {
            for (index, &lane) in lanes.iter().enumerate().take(4 - offset) {
                memory.write_u32(a.wrapping_add(index as u32 * 4), cpu.vfpu[lane])?;
            }
        }
        Step::Continue
    };
    Ok(result)
}

fn exec_vflush(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        if word & 0xffff_0000 != 0xffff_0000 {
            cpu.reset_vfpu_prefixes();
        }
        Step::Continue
    };
    Ok(result)
}

fn exec_vmmul(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let side = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_matrix_lanes(word & 0x7f, side);
        let source = vfpu_matrix_lanes((word >> 8) & 0x7f, side);
        let target = vfpu_matrix_lanes((word >> 16) & 0x7f, side);
        // raw matrices in column-major `rd[j * 4 + i]` layout; prefixes
        // apply to the final dot product only.
        let mut s = [0u32; 16];
        let mut t = [0u32; 16];
        for column in 0..side {
            for row in 0..side {
                s[column * 4 + row] = cpu.vfpu[source[column * 4 + row]];
                t[column * 4 + row] = cpu.vfpu[target[column * 4 + row]];
            }
        }
        let mut values = [0u32; 16];
        for a in 0..side {
            for b in 0..side {
                let mut sum = 0.0f32;
                if a + 1 == side && b + 1 == side {
                    let mut scol = [0u32; 4];
                    let mut tcol = [0u32; 4];
                    for c in 0..4 {
                        scol[c] = s[b * 4 + c];
                        tcol[c] = t[a * 4 + c];
                    }
                    let scol = vfpu_prefix_quad(scol, 4, cpu.vfpu_s_prefix);
                    let tcol = vfpu_prefix_quad(tcol, 4, cpu.vfpu_t_prefix);
                    for c in 0..4 {
                        sum += f32::from_bits(scol[c]) * f32::from_bits(tcol[c]);
                    }
                } else {
                    for c in 0..side {
                        sum += f32::from_bits(s[b * 4 + c]) * f32::from_bits(t[a * 4 + c]);
                    }
                }
                values[a * 4 + b] = sum.to_bits();
            }
        }
        // the destination prefix applies to the final element only
        // (saturation included); the write mask gates the last
        // column.
        let isolated = vfpu_isolate_prefix_lane(cpu.vfpu_d_prefix, side - 1);
        for row in 0..4 {
            let saturation = (isolated >> (row * 2)) & 3;
            let slot = (side - 1) * 4 + row;
            let value = f32::from_bits(values[slot]);
            values[slot] = match saturation {
                1 => vfpu_clamp(value, 0.0, 1.0).to_bits(),
                3 => vfpu_clamp(value, -1.0, 1.0).to_bits(),
                _ => values[slot],
            };
        }
        for column in 0..side {
            for row in 0..side {
                if column + 1 != side || isolated & (1 << (8 + row)) == 0 {
                    cpu.vfpu[destination[column * 4 + row]] = values[column * 4 + row];
                }
            }
        }
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vrot(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, length);
        let vs = (word >> 8) & 0x7f;
        let source = vfpu_scalar_lane(vs);
        let angle = f32::from_bits(cpu.vfpu[source]);
        let immediate = (word >> 16) & 31;
        let sine_lane = ((immediate >> 2) & 3) as usize;
        let cosine_lane = (immediate & 3) as usize;
        // cosine ignores all prefixes; sine honors a swizzled
        // source only through lane 0.
        let (sine, cosine) = if cpu.vfpu_s_prefix == VFPU_DEFAULT_SOURCE_PREFIX {
            let radians = angle * std::f32::consts::FRAC_PI_2;
            (radians.sin(), radians.cos())
        } else {
            let rewritten = vfpu_rewrite_prefix(cpu.vfpu_s_prefix, vfpu_negate_bits(1, 0, 0, 0), 0);
            let s = vfpu_prefix_quad([cpu.vfpu[source], 0, 0, 0], 1, rewritten)[0];
            (
                (f32::from_bits(s) * std::f32::consts::FRAC_PI_2).sin(),
                (angle * std::f32::consts::FRAC_PI_2).cos(),
            )
        };
        let mut sine = sine;
        if immediate & 0x10 != 0 {
            sine = -sine;
        }
        let mut values = [0u32; 4];
        if sine_lane == cosine_lane {
            for value in values.iter_mut().take(length) {
                *value = sine.to_bits();
            }
        } else {
            values[sine_lane.min(3)] = sine.to_bits();
        }
        // when source and destination share a matrix, the cosine is
        // recomputed from a matching written sine lane.
        let mut cosine_value = cosine;
        if ((word & 0x7f) >> 2) & 7 == (vs >> 2) & 7 {
            for index in 0..length {
                if source == destination[index] {
                    cosine_value =
                        (f32::from_bits(values[index]) * std::f32::consts::FRAC_PI_2).cos();
                    break;
                }
            }
        }
        if sine_lane != cosine_lane {
            values[cosine_lane.min(3)] = cosine_value.to_bits();
        }
        // saturation and mask skip the cosine lane.
        let cosine_bits = (3 << (cosine_lane.min(3) * 2)) | (1 << (8 + cosine_lane.min(3)));
        let d_prefix = cpu.vfpu_d_prefix & !cosine_bits;
        vfpu_write_vector(&mut cpu.vfpu, destination, length, values, d_prefix, true);
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vtfm(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let side = (((word >> 23) & 3) + 1) as usize;
        let input_length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, side);
        let matrix_lanes = vfpu_matrix_lanes((word >> 8) & 0x7f, side);
        let vector_lanes = vfpu_vector_lanes((word >> 16) & 0x7f, side);
        let ins = side - 1;
        let mut s = [0u32; 16];
        let mut t = [0u32; 4];
        for column in 0..side {
            for row in 0..side {
                s[column * 4 + row] = cpu.vfpu[matrix_lanes[column * 4 + row]];
            }
        }
        for (index, lane) in vector_lanes.iter().take(side).enumerate() {
            t[index] = cpu.vfpu[*lane];
        }
        let plain = input_length.min(side);
        let mut values = [0u32; 4];
        for row in 0..ins {
            let mut sum = 0.0f32;
            for component in 0..plain {
                sum += f32::from_bits(s[row * 4 + component]) * f32::from_bits(t[component]);
            }
            if ins >= input_length {
                sum += f32::from_bits(s[row * 4 + ins]);
            }
            values[row] = sum.to_bits();
        }
        // the final row uses the source prefix over its matrix
        // column and a rewritten target prefix that wires missing
        // lanes to zero (or one for a homogeneous transform).
        let mut scol = [0u32; 4];
        for component in 0..4 {
            scol[component] = s[ins * 4 + component];
        }
        let scol = vfpu_prefix_quad(scol, 4, cpu.vfpu_s_prefix);
        let (remove, add) = {
            let zero_y = input_length < 2;
            let zero_z = input_length < 3;
            let zero_w = input_length < 4;
            let remove = vfpu_swizzle(
                0,
                if zero_y { 3 } else { 0 },
                if zero_z { 3 } else { 0 },
                if zero_w { 3 } else { 0 },
            );
            let mut add = vfpu_make_constants(
                -1,
                if zero_y { 0 } else { -1 },
                if zero_z { 0 } else { -1 },
                if zero_w { 0 } else { -1 },
            );
            if ins >= input_length {
                let one = match ins {
                    1 => vfpu_make_constants(-1, 1, -1, -1),
                    2 => vfpu_make_constants(-1, -1, 1, -1),
                    3 => vfpu_make_constants(-1, -1, -1, 1),
                    _ => 0,
                };
                add |= one;
            }
            (remove, add)
        };
        let tcol = vfpu_prefix_quad(t, 4, vfpu_rewrite_prefix(cpu.vfpu_t_prefix, remove, add));
        let mut sum = 0.0f32;
        for component in 0..4 {
            sum += f32::from_bits(scol[component]) * f32::from_bits(tcol[component]);
        }
        values[ins] = sum.to_bits();
        // the destination prefix applies to the last element only.
        let isolated = vfpu_isolate_prefix_lane(cpu.vfpu_d_prefix, ins);
        vfpu_write_vector(&mut cpu.vfpu, destination, side, values, isolated, true);
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_crossquat(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let length = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let destination = vfpu_vector_lanes(word & 0x7f, length);
        let source = vfpu_vector_lanes((word >> 8) & 0x7f, length);
        let target = vfpu_vector_lanes((word >> 16) & 0x7f, length);
        let mut left = [0.0f32; 4];
        let mut right = [0.0f32; 4];
        for index in 0..length.min(4) {
            left[index] = f32::from_bits(cpu.vfpu[source[index]]);
            right[index] = f32::from_bits(cpu.vfpu[target[index]]);
        }
        let mut values = [0u32; 4];
        match length {
            3 => {
                values[0] = (left[1] * right[2] - left[2] * right[1]).to_bits();
                values[1] = (left[2] * right[0] - left[0] * right[2]).to_bits();
                let t_rewritten = vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_any_swizzle() | vfpu_negate_bits(1, 1, 1, 1),
                    vfpu_swizzle(1, 0, 3, 2) | vfpu_negate_bits(0, 1, 0, 0),
                );
                let mut s_raw = [0u32; 4];
                let mut t_raw = [0u32; 4];
                for index in 0..3 {
                    s_raw[index] = left[index].to_bits();
                    t_raw[index] = right[index].to_bits();
                }
                let s_prefixed = vfpu_prefix_quad(s_raw, 4, cpu.vfpu_s_prefix);
                let t_prefixed = vfpu_prefix_quad(t_raw, 4, t_rewritten);
                let mut sum = 0.0f32;
                for index in 0..4 {
                    sum += f32::from_bits(s_prefixed[index]) * f32::from_bits(t_prefixed[index]);
                }
                values[2] = sum.to_bits();
            }
            4 => {
                values[0] = (left[0] * right[3] + left[1] * right[2] - left[2] * right[1]
                    + left[3] * right[0])
                    .to_bits();
                values[1] = (-left[0] * right[2]
                    + left[1] * right[3]
                    + left[2] * right[0]
                    + left[3] * right[1])
                    .to_bits();
                values[2] = (left[0] * right[1] - left[1] * right[0]
                    + left[2] * right[3]
                    + left[3] * right[2])
                    .to_bits();
                let t_rewritten = vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_any_swizzle() | vfpu_negate_bits(1, 1, 1, 1),
                    vfpu_swizzle(0, 1, 2, 3) | vfpu_negate_bits(1, 1, 1, 0),
                );
                let mut s_raw = [0u32; 4];
                let mut t_raw = [0u32; 4];
                for index in 0..4 {
                    s_raw[index] = left[index].to_bits();
                    t_raw[index] = right[index].to_bits();
                }
                let s_prefixed = vfpu_prefix_quad(s_raw, 4, cpu.vfpu_s_prefix);
                let t_prefixed = vfpu_prefix_quad(t_raw, 4, t_rewritten);
                let mut sum = 0.0f32;
                for index in 0..4 {
                    sum += f32::from_bits(s_prefixed[index]) * f32::from_bits(t_prefixed[index]);
                }
                values[3] = sum.to_bits();
            }
            2 => {
                // a pair swizzles invalid, so the first lane is zero;
                // the second dots a swizzled source lane.
                values[0] = 0.0f32.to_bits();
                let t_rewritten = vfpu_rewrite_prefix(
                    cpu.vfpu_t_prefix,
                    vfpu_any_swizzle() | vfpu_negate_bits(1, 1, 1, 1),
                    vfpu_swizzle(0, 0, 0, 0),
                );
                let mut s_raw = [0u32; 4];
                let mut t_raw = [0u32; 4];
                for index in 0..2 {
                    s_raw[index] = left[index].to_bits();
                    t_raw[index] = right[index].to_bits();
                }
                let s_prefixed = vfpu_prefix_quad(s_raw, 4, cpu.vfpu_s_prefix);
                let t_prefixed = vfpu_prefix_quad(t_raw, 4, t_rewritten);
                values[1] =
                    (f32::from_bits(s_prefixed[2]) * f32::from_bits(t_prefixed[2])).to_bits();
            }
            _ => {
                values[0] = 0.0f32.to_bits();
            }
        }
        // the destination prefix applies to the last element only
        // for pair and larger; single always writes zero.
        if length == 1 {
            let lane = vfpu_scalar_lane(word & 0x7f);
            cpu.vfpu[lane] = 0;
        } else {
            let isolated = vfpu_isolate_prefix_lane(cpu.vfpu_d_prefix, length - 1);
            vfpu_write_vector(&mut cpu.vfpu, destination, length, values, isolated, true);
        }
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vmscl(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let side = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let source = vfpu_matrix_lanes((word >> 8) & 0x7f, side);
        let target_reg = (word >> 16) & 0x7f;
        let mut s = [0u32; 16];
        for column in 0..side {
            for row in 0..side {
                s[column * 4 + row] = cpu.vfpu[source[column * 4 + row]];
            }
        }
        let scalar = f32::from_bits(cpu.vfpu[vfpu_scalar_lane(target_reg)]);
        let mut values = [0u32; 16];
        for a in 0..side.saturating_sub(1) {
            for b in 0..side {
                values[a * 4 + b] = (f32::from_bits(s[a * 4 + b]) * scalar).to_bits();
            }
        }
        let off = side - 1;
        let mut srow = [0u32; 4];
        for b in 0..4 {
            srow[b] = s[off * 4 + b];
        }
        let srow = vfpu_prefix_quad(srow, 4, cpu.vfpu_s_prefix);
        let lane = ((target_reg >> 5) & 3) as usize;
        let mut traw = [0u32; 4];
        traw[lane] = scalar.to_bits();
        let trow = vfpu_prefix_quad(
            traw,
            4,
            vfpu_rewrite_prefix(
                cpu.vfpu_t_prefix,
                vfpu_any_swizzle(),
                vfpu_swizzle(lane as u32, lane as u32, lane as u32, lane as u32),
            ),
        );
        for b in 0..side {
            values[off * 4 + b] = (f32::from_bits(srow[b]) * f32::from_bits(trow[b])).to_bits();
        }
        // saturation honors the rewritten last-row prefix; the
        // write mask gates the last column.
        let mut last = [
            values[off * 4],
            values[off * 4 + 1],
            values[off * 4 + 2],
            values[off * 4 + 3],
        ];
        for (index, value) in last.iter_mut().enumerate() {
            let saturation = (cpu.vfpu_d_prefix >> (index * 2)) & 3;
            let scalar = f32::from_bits(*value);
            *value = match saturation {
                1 => vfpu_clamp(scalar, 0.0, 1.0).to_bits(),
                3 => vfpu_clamp(scalar, -1.0, 1.0).to_bits(),
                _ => *value,
            };
        }
        for b in 0..side {
            values[off * 4 + b] = last[b];
        }
        vfpu_write_matrix(&mut cpu.vfpu, word & 0x7f, side, &values, cpu.vfpu_d_prefix);
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vmmov(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let side = ((((word >> 7) & 1) | ((word >> 14) & 2)) + 1) as usize;
        let source = vfpu_matrix_lanes((word >> 8) & 0x7f, side);
        let mut values = [0u32; 16];
        for column in 0..side {
            for row in 0..side {
                values[column * 4 + row] = cpu.vfpu[source[column * 4 + row]];
            }
        }
        let off = side - 1;
        let mut last = [0u32; 4];
        for b in 0..4 {
            last[b] = values[off * 4 + b];
        }
        let prefixed = vfpu_prefix_quad(last, 4, cpu.vfpu_s_prefix);
        for b in 0..side {
            let saturation = (cpu.vfpu_d_prefix >> (b * 2)) & 3;
            let scalar = f32::from_bits(prefixed[b]);
            values[off * 4 + b] = match saturation {
                1 => vfpu_clamp(scalar, 0.0, 1.0).to_bits(),
                3 => vfpu_clamp(scalar, -1.0, 1.0).to_bits(),
                _ => prefixed[b],
            };
        }
        vfpu_write_matrix(&mut cpu.vfpu, word & 0x7f, side, &values, cpu.vfpu_d_prefix);
        cpu.reset_vfpu_prefixes();
        Step::Continue
    };
    Ok(result)
}

fn exec_vmatrix_init(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let word = d.word;
    let result: Step = {
        let side = (((word >> 7) & 1) | ((word >> 14) & 2) | 1) as usize;
        let lanes = vfpu_matrix_lanes(word & 0x7f, side);
        let kind = (word >> 16) & 15;
        for row in 0..side {
            for column in 0..side {
                let value = match kind {
                    3 if row == column => 1.0f32,
                    7 => 1.0f32,
                    _ => 0.0f32,
                };
                cpu.vfpu[lanes[column * 4 + row]] = value.to_bits();
            }
        }
        Step::Continue
    };
    Ok(result)
}

fn exec_cop0(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let pc = d.pc;
    let word = d.word;
    let rt = d.rt as usize;
    let result: Step = {
        let cop0_op = (word >> 21) & 31;
        match cop0_op {
            0 | 2 => {
                cpu.gpr[rt] = 0;
                Step::Continue
            }
            4 | 6 => Step::Continue,
            16..=31 => Step::Continue,
            _ => {
                return Err(CpuError::Unsupported {
                    pc: GuestAddress(pc),
                    word,
                });
            }
        }
    };
    Ok(result)
}

fn exec_cop1x(cpu: &mut Cpu, memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let pc = d.pc;
    let word = d.word;
    let rs = d.rs as usize;
    let rt = d.rt as usize;
    let rd = d.rd as usize;
    let result: Step = {
        let funct = word & 63;
        match funct {
            // madd.s / msub.s / nmadd.s / nmsub.s
            0x20 | 0x21 | 0x28 | 0x29 => {
                let fr = ((word >> 21) & 31) as usize;
                let ft = rt;
                let fs = rd;
                let fd = ((word >> 6) & 31) as usize;
                let a = f32::from_bits(cpu.fpr[fr]);
                let b = f32::from_bits(cpu.fpr[ft]);
                let c = f32::from_bits(cpu.fpr[fs]);
                let value = match funct {
                    0x20 => a.mul_add(b, c),
                    0x21 => c - a * b,
                    0x28 => -(a.mul_add(b, c)),
                    _ => -(c - a * b),
                };
                // inf*0 produces a positive nan, as with mul.s.
                cpu.fpr[fd] = if value.is_nan()
                    && ((a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0))
                {
                    f32::NAN.to_bits()
                } else {
                    value.to_bits()
                };
                Step::Continue
            }
            // lwxc1 / swxc1 indexed word transfers; prefx is a hint.
            0x00 => {
                let base = cpu.gpr[rs].wrapping_add(cpu.gpr[rt]);
                let fd = ((word >> 6) & 31) as usize;
                cpu.fpr[fd] = memory.read_u32(base)?;
                Step::Continue
            }
            0x08 | 0x09 => {
                let base = cpu.gpr[rs].wrapping_add(cpu.gpr[rt]);
                let fs = rd;
                memory.write_u32(base, cpu.fpr[fs])?;
                Step::Continue
            }
            0x0f | 0x07 | 0x17 | 0x1f | 0x27 | 0x2f | 0x37 => Step::Continue,
            _ => {
                return Err(CpuError::Unsupported {
                    pc: GuestAddress(pc),
                    word,
                });
            }
        }
    };
    Ok(result)
}

fn exec_special2(cpu: &mut Cpu, _memory: &mut Memory, d: Decoded) -> Result<Step, CpuError> {
    let pc = d.pc;
    let word = d.word;
    let rs = d.rs as usize;
    let rt = d.rt as usize;
    let rd = d.rd as usize;
    let result: Step = match d.word & 63 {
        0x02 => {
            cpu.gpr[rd] = (cpu.gpr[rs] as i32).wrapping_mul(cpu.gpr[rt] as i32) as u32;
            Step::Continue
        }
        0x00 => {
            let product = (cpu.gpr[rs] as i32 as i64) * (cpu.gpr[rt] as i32 as i64);
            let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
            let result = (acc as i64).wrapping_add(product) as u64;
            cpu.lo = result as u32;
            cpu.hi = (result >> 32) as u32;
            Step::Continue
        }
        0x01 => {
            let product = u64::from(cpu.gpr[rs]) * u64::from(cpu.gpr[rt]);
            let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
            let result = acc.wrapping_add(product);
            cpu.lo = result as u32;
            cpu.hi = (result >> 32) as u32;
            Step::Continue
        }
        0x04 => {
            let product = (cpu.gpr[rs] as i32 as i64) * (cpu.gpr[rt] as i32 as i64);
            let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
            let result = (acc as i64).wrapping_sub(product) as u64;
            cpu.lo = result as u32;
            cpu.hi = (result >> 32) as u32;
            Step::Continue
        }
        0x05 => {
            let product = u64::from(cpu.gpr[rs]) * u64::from(cpu.gpr[rt]);
            let acc = ((cpu.hi as u64) << 32) | cpu.lo as u64;
            let result = acc.wrapping_sub(product);
            cpu.lo = result as u32;
            cpu.hi = (result >> 32) as u32;
            Step::Continue
        }
        // mfic/mtic interrupt-controller moves: single-threaded nop.
        0x24 | 0x26 => Step::Continue,
        _ => {
            return Err(CpuError::Unsupported {
                pc: GuestAddress(pc),
                word,
            });
        }
    };
    Ok(result)
}
