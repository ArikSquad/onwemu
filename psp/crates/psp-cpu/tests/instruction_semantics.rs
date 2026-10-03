//! Public, instruction-level checks for the Allegrex execution contract.
//!
//! The unit tests in `psp-cpu` cover the VFPU's private register helpers. This
//! file exercises the public CPU surface with readable instruction encoders so
//! scalar MIPS, memory, coprocessor, and control-flow behavior does not depend
//! only on the JIT's generated programs.

use psp_cpu::{Cpu, CpuError, Step};
use psp_memory::{AccessType, GuestMemory, Memory};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x3000;

fn r_type(rs: u8, rt: u8, rd: u8, sa: u8, funct: u8) -> u32 {
    (u32::from(rs) << 21)
        | (u32::from(rt) << 16)
        | (u32::from(rd) << 11)
        | (u32::from(sa) << 6)
        | u32::from(funct)
}

fn i_type(op: u8, rs: u8, rt: u8, immediate: i16) -> u32 {
    (u32::from(op) << 26)
        | (u32::from(rs) << 21)
        | (u32::from(rt) << 16)
        | u32::from(immediate as u16)
}

fn j_type(op: u8, target: u32) -> u32 {
    (u32::from(op) << 26) | ((target >> 2) & 0x03ff_ffff)
}

fn special3(rs: u8, rt: u8, rd: u8, sa: u8, funct: u8) -> u32 {
    (0x1f << 26) | r_type(rs, rt, rd, sa, funct)
}

fn cop0(cop0_op: u8, rt: u8, register: u8) -> u32 {
    (0x10 << 26) | (u32::from(cop0_op) << 21) | (u32::from(rt) << 16) | (u32::from(register) << 11)
}

fn fpu_transfer(operation: u8, rt: u8, fs: u8) -> u32 {
    (0x11 << 26) | (u32::from(operation) << 21) | (u32::from(rt) << 16) | (u32::from(fs) << 11)
}

fn fpu_alu(ft: u8, fs: u8, fd: u8, funct: u8) -> u32 {
    (0x11 << 26)
        | (16 << 21)
        | (u32::from(ft) << 16)
        | (u32::from(fs) << 11)
        | (u32::from(fd) << 6)
        | u32::from(funct)
}

fn fpu_branch(kind: u8, immediate: i16) -> u32 {
    i_type(0x11, 8, kind, immediate)
}

fn cop1x(fr: u8, ft: u8, fs: u8, fd: u8, funct: u8) -> u32 {
    (0x13 << 26)
        | (u32::from(fr) << 21)
        | (u32::from(ft) << 16)
        | (u32::from(fs) << 11)
        | (u32::from(fd) << 6)
        | u32::from(funct)
}

fn mapped_memory(words: &[u32]) -> Memory {
    let mut memory = Memory::default();
    memory.map(CODE, 0x1000, true, true).unwrap();
    memory.map(DATA, 0x1000, true, false).unwrap();
    for (index, word) in words.iter().enumerate() {
        memory.write_u32(CODE + index as u32 * 4, *word).unwrap();
    }
    memory
}

fn run_program(words: &[u32], steps: usize, mut cpu: Cpu) -> (Cpu, Memory) {
    let mut memory = mapped_memory(words);
    for _ in 0..steps {
        cpu.step(&mut memory).unwrap();
    }
    (cpu, memory)
}

fn execute(cpu: &mut Cpu, memory: &mut Memory, word: u32) -> Result<Step, CpuError> {
    cpu.execute_word(memory, cpu.pc, word, 9)
}

#[test]
fn scalar_alu_covers_shifts_logical_ops_and_signedness() {
    let cases = [
        ("sll", r_type(0, 1, 3, 4, 0x00), 0x0000_0003, 0, 0x30),
        ("srl", r_type(0, 1, 3, 4, 0x02), 0x8000_0030, 0, 0x0800_0003),
        ("sra", r_type(0, 1, 3, 4, 0x03), 0x8000_0030, 0, 0xf800_0003),
        ("sllv", r_type(2, 1, 3, 0, 0x04), 0x0000_0003, 4, 0x30),
        (
            "srlv",
            r_type(2, 1, 3, 0, 0x06),
            0x8000_0030,
            4,
            0x0800_0003,
        ),
        (
            "srav",
            r_type(2, 1, 3, 0, 0x07),
            0x8000_0030,
            4,
            0xf800_0003,
        ),
        (
            "and",
            r_type(1, 2, 3, 0, 0x24),
            0xf0f0_1234,
            0x0ff0_00ff,
            0x00f0_0034,
        ),
        (
            "or",
            r_type(1, 2, 3, 0, 0x25),
            0xf0f0_1234,
            0x0ff0_00ff,
            0xfff0_12ff,
        ),
        (
            "xor",
            r_type(1, 2, 3, 0, 0x26),
            0xf0f0_1234,
            0x0ff0_00ff,
            0xff00_12cb,
        ),
        (
            "nor",
            r_type(1, 2, 3, 0, 0x27),
            0xf0f0_1234,
            0x0ff0_00ff,
            0x000f_ed00,
        ),
        ("slt", r_type(1, 2, 3, 0, 0x2a), 0xffff_ffff, 1, 1),
        ("sltu", r_type(1, 2, 3, 0, 0x2b), 0xffff_ffff, 1, 0),
        ("max", r_type(1, 2, 3, 0, 0x2c), 0xffff_fffe, 7, 7),
        ("min", r_type(1, 2, 3, 0, 0x2d), 0xffff_fffe, 7, 0xffff_fffe),
        (
            "rotr",
            r_type(1, 2, 3, 16, 0x02),
            0,
            0x1234_5678,
            0x5678_1234,
        ),
    ];

    for (name, word, first, second, expected) in cases {
        let mut cpu = Cpu::new(CODE);
        let mut memory = Memory::default();
        cpu.gpr[1] = first;
        cpu.gpr[2] = second;
        execute(&mut cpu, &mut memory, word).unwrap();
        assert_eq!(cpu.gpr[3], expected, "{name}");
    }

    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.gpr[1] = 0x1234_5678;
    cpu.gpr[2] = 40;
    execute(&mut cpu, &mut memory, r_type(2, 1, 3, 1, 0x06)).unwrap();
    assert_eq!(
        cpu.gpr[3], 0x7812_3456,
        "variable rotate uses the rotate encoding bit"
    );
}

#[test]
fn immediate_alu_distinguishes_sign_and_zero_extension() {
    let cases = [
        ("addiu", i_type(0x09, 1, 2, -2), 5, 3),
        ("andi", i_type(0x0c, 1, 2, -1), 0x1234_5678, 0x0000_5678),
        ("ori", i_type(0x0d, 1, 2, 0x00f0), 0x1200, 0x12f0),
        ("xori", i_type(0x0e, 1, 2, 0x00f0), 0x12ff, 0x120f),
        ("lui", i_type(0x0f, 0, 2, 0x1234), 0, 0x1234_0000),
        ("slti", i_type(0x0a, 1, 2, -1), 0xffff_fffe, 1),
        ("sltiu", i_type(0x0b, 1, 2, -1), 0, 1),
    ];

    for (name, word, input, expected) in cases {
        let mut cpu = Cpu::new(CODE);
        let mut memory = Memory::default();
        cpu.gpr[1] = input;
        execute(&mut cpu, &mut memory, word).unwrap();
        assert_eq!(cpu.gpr[2], expected, "{name}");
    }

    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.gpr[1] = i32::MAX as u32;
    let error = execute(&mut cpu, &mut memory, i_type(0x08, 1, 2, 1)).unwrap_err();
    assert!(matches!(error, CpuError::Overflow(address) if address.0 == CODE));
}

#[test]
fn conditional_moves_and_zero_register_follow_mips_rules() {
    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.gpr[1] = 0x1111;
    cpu.gpr[2] = 0;
    execute(&mut cpu, &mut memory, r_type(1, 2, 3, 0, 0x0a)).unwrap();
    assert_eq!(cpu.gpr[3], 0x1111);

    cpu.gpr[2] = 1;
    cpu.gpr[3] = 0x2222;
    execute(&mut cpu, &mut memory, r_type(1, 2, 3, 0, 0x0a)).unwrap();
    assert_eq!(cpu.gpr[3], 0x2222, "movz does not move for a nonzero test");

    execute(&mut cpu, &mut memory, r_type(1, 2, 4, 0, 0x0b)).unwrap();
    assert_eq!(cpu.gpr[4], 0x1111);

    cpu.gpr[0] = 99;
    execute(&mut cpu, &mut memory, i_type(0x09, 0, 0, 7)).unwrap();
    assert_eq!(cpu.gpr[0], 0, "$zero is restored after every instruction");
}

#[test]
fn multiply_divide_and_hilo_cover_signed_unsigned_and_edge_cases() {
    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.gpr[1] = (-3i32) as u32;
    cpu.gpr[2] = 7;
    execute(&mut cpu, &mut memory, r_type(1, 2, 0, 0, 0x18)).unwrap();
    assert_eq!(cpu.hi, u32::MAX);
    assert_eq!(cpu.lo, (-21i32) as u32);

    cpu.gpr[1] = u32::MAX;
    cpu.gpr[2] = 2;
    execute(&mut cpu, &mut memory, r_type(1, 2, 0, 0, 0x19)).unwrap();
    assert_eq!((cpu.hi, cpu.lo), (1, u32::MAX - 1));

    cpu.gpr[1] = (-7i32) as u32;
    cpu.gpr[2] = 3;
    execute(&mut cpu, &mut memory, r_type(1, 2, 0, 0, 0x1a)).unwrap();
    assert_eq!(cpu.lo, (-2i32) as u32);
    assert_eq!(cpu.hi, (-1i32) as u32);

    cpu.gpr[1] = 7;
    cpu.gpr[2] = 3;
    execute(&mut cpu, &mut memory, r_type(1, 2, 0, 0, 0x1b)).unwrap();
    assert_eq!((cpu.lo, cpu.hi), (2, 1));

    cpu.gpr[1] = (-5i32) as u32;
    cpu.gpr[2] = 0;
    execute(&mut cpu, &mut memory, r_type(1, 2, 0, 0, 0x1a)).unwrap();
    assert_eq!((cpu.lo, cpu.hi), (1, (-5i32) as u32));
    execute(&mut cpu, &mut memory, r_type(1, 2, 0, 0, 0x1b)).unwrap();
    assert_eq!((cpu.lo, cpu.hi), (u32::MAX, (-5i32) as u32));

    cpu.gpr[1] = 0x8000_0000;
    cpu.gpr[2] = u32::MAX;
    execute(&mut cpu, &mut memory, r_type(1, 2, 3, 0, 0x16)).unwrap();
    assert_eq!(cpu.gpr[3], 0);
    execute(&mut cpu, &mut memory, r_type(2, 2, 4, 0, 0x17)).unwrap();
    assert_eq!(cpu.gpr[4], 32);
}

#[test]
fn multiply_accumulate_variants_update_the_full_hilo_value() {
    let operations = [
        (0x1c, 11u64),
        (0x1d, 11u64),
        (0x2e, u64::MAX),
        (0x2f, u64::MAX),
    ];
    for (funct, expected) in operations {
        let mut cpu = Cpu::new(CODE);
        let mut memory = Memory::default();
        cpu.lo = 5;
        cpu.gpr[1] = 2;
        cpu.gpr[2] = 3;
        execute(&mut cpu, &mut memory, r_type(1, 2, 0, 0, funct)).unwrap();
        let actual = (u64::from(cpu.hi) << 32) | u64::from(cpu.lo);
        assert_eq!(actual, expected, "funct 0x{funct:02x}");
    }
}

#[test]
fn branches_run_delay_slots_and_likely_branches_annul_them() {
    let words = [
        i_type(0x04, 1, 2, 1), // beq: target is the word after the delay slot
        i_type(0x09, 0, 3, 7), // delay slot
        i_type(0x09, 0, 4, 9), // landing point
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = 4;
    cpu.gpr[2] = 4;
    let (cpu, _) = run_program(&words, 3, cpu);
    assert_eq!(cpu.gpr[3], 7);
    assert_eq!(cpu.gpr[4], 9);
    assert_eq!(cpu.pc, CODE + 12);

    let words = [
        i_type(0x14, 1, 2, 1), // beql, not taken
        i_type(0x09, 0, 3, 7), // annulled delay slot
        i_type(0x09, 0, 4, 9),
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = 4;
    cpu.gpr[2] = 5;
    let (cpu, _) = run_program(&words, 3, cpu);
    assert_eq!(cpu.gpr[3], 0);
    assert_eq!(cpu.gpr[4], 9);

    let words = [
        i_type(0x01, 1, 1, 1), // bgez, taken
        i_type(0x09, 0, 3, 7),
        i_type(0x09, 0, 4, 9),
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = 0;
    let (cpu, _) = run_program(&words, 3, cpu);
    assert_eq!((cpu.gpr[3], cpu.gpr[4]), (7, 9));
}

#[test]
fn signed_branch_conditions_and_link_variants_use_the_right_sign() {
    let cases = [
        (0x06, 0, true, "blez zero"),
        (0x06, (-1i32) as u32, true, "blez negative"),
        (0x06, 1, false, "blez positive"),
        (0x07, 1, true, "bgtz positive"),
        (0x07, 0, false, "bgtz zero"),
        (0x07, (-1i32) as u32, false, "bgtz negative"),
    ];
    for (op, value, taken, name) in cases {
        let words = [
            i_type(op, 1, 0, 2),
            i_type(0x09, 0, 3, 7),
            i_type(0x09, 0, 4, 5),
            i_type(0x09, 0, 4, 9),
        ];
        let mut cpu = Cpu::new(CODE);
        cpu.gpr[1] = value;
        let (cpu, _) = run_program(&words, 3, cpu);
        assert_eq!(cpu.gpr[3], 7, "{name} always runs its delay slot");
        assert_eq!(cpu.gpr[4], if taken { 9 } else { 5 }, "{name}");
    }

    let words = [
        i_type(0x01, 1, 16, 1), // bltzal, taken
        0,
        i_type(0x09, 0, 4, 9),
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = (-1i32) as u32;
    let (cpu, _) = run_program(&words, 3, cpu);
    assert_eq!(cpu.gpr[31], CODE + 8);
    assert_eq!(cpu.gpr[4], 9);
}

#[test]
fn jumps_and_register_jumps_preserve_delay_slots_and_return_addresses() {
    let words = [
        j_type(0x03, CODE + 8), // jal
        i_type(0x09, 0, 3, 7),
        i_type(0x09, 0, 4, 9),
    ];
    let (cpu, _) = run_program(&words, 3, Cpu::new(CODE));
    assert_eq!(cpu.gpr[31], CODE + 8);
    assert_eq!((cpu.gpr[3], cpu.gpr[4]), (7, 9));

    let words = [
        r_type(5, 0, 0, 0, 0x08), // jr $a1
        i_type(0x09, 0, 3, 7),
        i_type(0x09, 0, 4, 9),
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[5] = CODE + 8;
    let (cpu, _) = run_program(&words, 3, cpu);
    assert_eq!((cpu.gpr[3], cpu.gpr[4]), (7, 9));

    let words = [
        r_type(5, 0, 31, 0, 0x09), // jalr $ra, $a1
        i_type(0x09, 0, 3, 7),
        i_type(0x09, 0, 4, 9),
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[5] = CODE + 8;
    let (cpu, _) = run_program(&words, 3, cpu);
    assert_eq!(cpu.gpr[31], CODE + 8);
    assert_eq!((cpu.gpr[3], cpu.gpr[4]), (7, 9));
}

#[test]
fn syscall_and_unsupported_words_report_the_instruction_location() {
    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    assert_eq!(
        execute(&mut cpu, &mut memory, (0x12345 << 6) | 0x0c).unwrap(),
        Step::Syscall(0x12345)
    );
    assert_eq!(cpu.pc, CODE + 4);

    let word = 0x0000_0031;
    let error = execute(&mut cpu, &mut memory, word).unwrap_err();
    assert!(
        matches!(error, CpuError::Unsupported { pc, word: actual } if pc.0 == CODE + 4 && actual == word)
    );
}

#[test]
fn byte_halfword_and_word_memory_ops_have_guest_endianness_and_sign_rules() {
    let mut memory = mapped_memory(&[]);
    memory
        .write_bytes(DATA, &[0x80, 0x7f, 0x01, 0x80, 0x34, 0x12, 0xef, 0xcd])
        .unwrap();
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = DATA;

    execute(&mut cpu, &mut memory, i_type(0x20, 1, 2, 0)).unwrap(); // lb
    execute(&mut cpu, &mut memory, i_type(0x24, 1, 3, 0)).unwrap(); // lbu
    execute(&mut cpu, &mut memory, i_type(0x21, 1, 4, 2)).unwrap(); // lh
    execute(&mut cpu, &mut memory, i_type(0x25, 1, 5, 2)).unwrap(); // lhu
    execute(&mut cpu, &mut memory, i_type(0x23, 1, 6, 4)).unwrap(); // lw
    assert_eq!(cpu.gpr[2], 0xffff_ff80);
    assert_eq!(cpu.gpr[3], 0x80);
    assert_eq!(cpu.gpr[4], 0xffff_8001);
    assert_eq!(cpu.gpr[5], 0x8001);
    assert_eq!(cpu.gpr[6], 0xcd_ef_12_34);

    cpu.gpr[7] = 0xa1b2_c3d4;
    execute(&mut cpu, &mut memory, i_type(0x28, 1, 7, 1)).unwrap(); // sb
    execute(&mut cpu, &mut memory, i_type(0x29, 1, 7, 2)).unwrap(); // sh
    execute(&mut cpu, &mut memory, i_type(0x2b, 1, 7, 4)).unwrap(); // sw
    assert_eq!(memory.read_u8(DATA + 1).unwrap(), 0xd4);
    assert_eq!(memory.read_u16(DATA + 2).unwrap(), 0xc3d4);
    assert_eq!(memory.read_u32(DATA + 4).unwrap(), 0xa1b2_c3d4);
}

#[test]
fn ll_and_sc_transfer_memory_and_report_a_successful_store() {
    let mut memory = mapped_memory(&[]);
    memory.write_u32(DATA, 0x1234_5678).unwrap();
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = DATA;
    execute(&mut cpu, &mut memory, i_type(0x30, 1, 2, 0)).unwrap();
    assert_eq!(cpu.gpr[2], 0x1234_5678);

    cpu.gpr[2] = 0x89ab_cdef;
    execute(&mut cpu, &mut memory, i_type(0x38, 1, 2, 0)).unwrap();
    assert_eq!(memory.read_u32(DATA).unwrap(), 0x89ab_cdef);
    assert_eq!(cpu.gpr[2], 1, "sc overwrites rt with its success flag");
}

#[test]
fn memory_faults_keep_the_cpu_context_that_caused_them() {
    let mut memory = mapped_memory(&[]);
    memory.map(0x5000, 4, false, false).unwrap();
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = 0x5000;
    let error = execute(&mut cpu, &mut memory, i_type(0x2b, 1, 2, 0)).unwrap_err();
    assert!(matches!(
        error,
        CpuError::Memory(fault)
            if fault.address.0 == 0x5000
                && fault.pc.0 == CODE
                && fault.thread_id == 9
                && fault.access == AccessType::Write
    ));
}

#[test]
fn fpu_transfers_arithmetic_rounding_and_comparisons_are_observable() {
    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.fpr[1] = 1.5f32.to_bits();
    cpu.fpr[2] = 2.0f32.to_bits();
    execute(&mut cpu, &mut memory, fpu_alu(2, 1, 3, 0)).unwrap(); // add.s
    execute(&mut cpu, &mut memory, fpu_alu(2, 1, 4, 1)).unwrap(); // sub.s
    execute(&mut cpu, &mut memory, fpu_alu(2, 1, 5, 2)).unwrap(); // mul.s
    execute(&mut cpu, &mut memory, fpu_alu(2, 1, 6, 3)).unwrap(); // div.s
    execute(&mut cpu, &mut memory, fpu_alu(0, 2, 7, 4)).unwrap(); // sqrt.s
    execute(&mut cpu, &mut memory, fpu_alu(0, 1, 8, 5)).unwrap(); // abs.s
    execute(&mut cpu, &mut memory, fpu_alu(0, 1, 9, 7)).unwrap(); // neg.s
    assert_eq!(f32::from_bits(cpu.fpr[3]), 3.5);
    assert_eq!(f32::from_bits(cpu.fpr[4]), -0.5);
    assert_eq!(f32::from_bits(cpu.fpr[5]), 3.0);
    assert_eq!(f32::from_bits(cpu.fpr[6]), 0.75);
    assert_eq!(f32::from_bits(cpu.fpr[7]), 2.0f32.sqrt());
    assert_eq!(f32::from_bits(cpu.fpr[8]), 1.5);
    assert_eq!(f32::from_bits(cpu.fpr[9]), -1.5);

    cpu.fpr[1] = 2.5f32.to_bits();
    execute(&mut cpu, &mut memory, fpu_alu(0, 1, 10, 0x0c)).unwrap(); // round.w.s
    execute(&mut cpu, &mut memory, fpu_alu(0, 1, 11, 0x0d)).unwrap(); // trunc.w.s
    execute(&mut cpu, &mut memory, fpu_alu(0, 1, 12, 0x0e)).unwrap(); // ceil.w.s
    execute(&mut cpu, &mut memory, fpu_alu(0, 1, 13, 0x0f)).unwrap(); // floor.w.s
    assert_eq!(cpu.fpr[10], 2);
    assert_eq!(cpu.fpr[11], 2);
    assert_eq!(cpu.fpr[12], 3);
    assert_eq!(cpu.fpr[13], 2);

    cpu.fpr[1] = 3.0f32.to_bits();
    cpu.fpr[2] = 3.0f32.to_bits();
    execute(&mut cpu, &mut memory, fpu_alu(2, 1, 0, 0x32)).unwrap(); // c.eq.s
    assert_ne!(cpu.fcr31 & (1 << 23), 0);
    cpu.fpr[2] = 4.0f32.to_bits();
    execute(&mut cpu, &mut memory, fpu_alu(2, 1, 0, 0x32)).unwrap();
    assert_eq!(cpu.fcr31 & (1 << 23), 0);

    execute(&mut cpu, &mut memory, fpu_transfer(4, 14, 1)).unwrap(); // mtc1
    execute(&mut cpu, &mut memory, fpu_transfer(0, 15, 1)).unwrap(); // mfc1
    assert_eq!(cpu.gpr[15], cpu.fpr[1]);
    cpu.gpr[16] = 0xfeed_beef;
    execute(&mut cpu, &mut memory, fpu_transfer(6, 16, 31)).unwrap(); // ctc1
    execute(&mut cpu, &mut memory, fpu_transfer(2, 17, 31)).unwrap(); // cfc1
    assert_eq!(cpu.gpr[17], 0xfeed_beef);
}

#[test]
fn fpu_condition_branches_have_normal_and_likely_delay_slots() {
    let words = [
        fpu_branch(1, 1),
        i_type(0x09, 0, 3, 7),
        i_type(0x09, 0, 4, 9),
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.fcr31 = 1 << 23;
    let (cpu, _) = run_program(&words, 3, cpu);
    assert_eq!((cpu.gpr[3], cpu.gpr[4]), (7, 9));

    let words = [
        fpu_branch(2, 1), // bc1fl, condition true means not taken and annulled
        i_type(0x09, 0, 3, 7),
        i_type(0x09, 0, 4, 9),
    ];
    let mut cpu = Cpu::new(CODE);
    cpu.fcr31 = 1 << 23;
    let (cpu, _) = run_program(&words, 2, cpu);
    assert_eq!((cpu.gpr[3], cpu.gpr[4]), (0, 9));
}

#[test]
fn fpu_memory_and_indexed_cop1x_transfers_use_guest_addresses() {
    let mut memory = mapped_memory(&[]);
    memory.write_u32(DATA + 4, 0x1122_3344).unwrap();
    memory.write_u32(DATA + 8, 0x5566_7788).unwrap();
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[1] = DATA;
    execute(&mut cpu, &mut memory, i_type(0x31, 1, 2, 4)).unwrap(); // lwc1
    execute(&mut cpu, &mut memory, i_type(0x39, 1, 2, 8)).unwrap(); // swc1
    assert_eq!(memory.read_u32(DATA + 8).unwrap(), 0x1122_3344);

    cpu.gpr[3] = 4;
    execute(&mut cpu, &mut memory, cop1x(1, 3, 0, 4, 0x00)).unwrap(); // lwxc1
    assert_eq!(cpu.fpr[4], 0x1122_3344);
    cpu.fpr[5] = 0xaabb_ccdd;
    execute(&mut cpu, &mut memory, cop1x(1, 3, 5, 0, 0x08)).unwrap(); // swxc1
    assert_eq!(memory.read_u32(DATA + 4).unwrap(), 0xaabb_ccdd);
}

#[test]
fn cop1x_fused_operations_and_special2_mul_are_not_only_jit_paths() {
    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.fpr[1] = 2.0f32.to_bits();
    cpu.fpr[2] = 3.0f32.to_bits();
    cpu.fpr[3] = 4.0f32.to_bits();
    execute(&mut cpu, &mut memory, cop1x(1, 2, 3, 4, 0x20)).unwrap(); // madd.s
    execute(&mut cpu, &mut memory, cop1x(1, 2, 3, 5, 0x21)).unwrap(); // msub.s
    assert_eq!(f32::from_bits(cpu.fpr[4]), 10.0);
    assert_eq!(f32::from_bits(cpu.fpr[5]), -2.0);

    cpu.gpr[8] = 6;
    cpu.gpr[9] = 7;
    execute(
        &mut cpu,
        &mut memory,
        (0x1c << 26) | r_type(8, 9, 10, 0, 0x02),
    )
    .unwrap();
    assert_eq!(cpu.gpr[10], 42);
}

#[test]
fn special3_covers_extract_insert_and_all_allegrex_bit_shuffles() {
    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.gpr[1] = 0xfedc_ba98;
    execute(&mut cpu, &mut memory, special3(1, 2, 3, 4, 0x00)).unwrap();
    assert_eq!(cpu.gpr[2], 0x9);

    cpu.gpr[1] = 0x5a;
    cpu.gpr[2] = 0xffff_0000;
    execute(&mut cpu, &mut memory, special3(1, 2, 15, 8, 0x04)).unwrap();
    assert_eq!(cpu.gpr[2], 0xffff_5a00);

    let shuffles = [
        (2, 0x1234_5678, 0x3412_7856, "wsbh"),
        (3, 0x1234_5678, 0x7856_3412, "wsbw"),
        (16, 0x0000_0080, 0xffff_ff80, "seb"),
        (20, 0x0000_0001, 0x8000_0000, "bitrev"),
        (24, 0x0000_8001, 0xffff_8001, "seh"),
    ];
    for (sa, input, expected, name) in shuffles {
        cpu.gpr[2] = input;
        execute(&mut cpu, &mut memory, special3(0, 2, 3, sa, 0x20)).unwrap();
        assert_eq!(cpu.gpr[3], expected, "{name}");
    }
}

#[test]
fn coprocessor_zero_and_cache_hints_have_explicit_contracts() {
    let mut cpu = Cpu::new(CODE);
    let mut memory = Memory::default();
    cpu.gpr[2] = 0x1234;
    execute(&mut cpu, &mut memory, cop0(0, 3, 12)).unwrap(); // mfc0
    assert_eq!(cpu.gpr[3], 0);
    execute(&mut cpu, &mut memory, cop0(4, 2, 12)).unwrap(); // mtc0
    execute(&mut cpu, &mut memory, cop0(16, 0, 0)).unwrap(); // a supported cop0 no-op
    execute(&mut cpu, &mut memory, i_type(0x2f, 0, 0, 0)).unwrap(); // cache
    execute(&mut cpu, &mut memory, i_type(0x33, 0, 0, 0)).unwrap(); // pref
    execute(&mut cpu, &mut memory, r_type(0, 0, 0, 0, 0x0f)).unwrap(); // sync

    let unsupported = cop0(1, 0, 0);
    let error = execute(&mut cpu, &mut memory, unsupported).unwrap_err();
    assert!(matches!(error, CpuError::Unsupported { word, .. } if word == unsupported));
}

#[test]
fn fast_forward_only_accepts_the_exact_fill_loop_shape() {
    let mut memory = mapped_memory(&[
        0x00e0_4825,
        0xa105_0000,
        0x24c7_ffff,
        0x2508_0001,
        0x1520_fffb,
        0x00e0_3025,
    ]);
    let mut cpu = Cpu::new(CODE);
    cpu.gpr[5] = 0xa5;
    cpu.gpr[6] = 4;
    cpu.gpr[7] = 4;
    cpu.gpr[8] = DATA;
    assert_eq!(
        cpu.try_fast_forward_memory_loop(&mut memory, 3).unwrap(),
        Some(30)
    );

    memory.patch_u32(CODE + 16, 0x1520_fffa).unwrap();
    cpu.pc = CODE;
    cpu.gpr[6] = 1;
    cpu.gpr[7] = 1;
    assert_eq!(
        cpu.try_fast_forward_memory_loop(&mut memory, 3).unwrap(),
        None
    );
}
