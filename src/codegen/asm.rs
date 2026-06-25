use crate::ast::{Expr, Program, Stmt};
use crate::codegen::Target;

pub fn emit_asm(target: Target, prog: &Program, _opt_o1: bool) -> String {
    let exit_code = extract_main_return_imm(prog);
    match target {
        Target::X86_64 => emit_x86_64_main(exit_code),
        Target::AArch64 => emit_aarch64_main(exit_code),
        Target::Riscv64 => emit_riscv64_main(exit_code),
    }
}

fn extract_main_return_imm(prog: &Program) -> i32 {
    let main = prog
        .funcs
        .iter()
        .find(|f| f.name == "main")
        .unwrap_or_else(|| panic!("No main() found"));

    let ret_stmt = main
        .body
        .iter()
        .find_map(|s| match s {
            Stmt::Return(e) => Some(e),
        })
        .unwrap_or_else(|| panic!("main() has no return"));

    match ret_stmt {
        Expr::Int(v) => i32::try_from(*v).unwrap_or_else(|_| panic!("return const out of i32")),
    }
}

fn emit_x86_64_main(imm: i32) -> String {
    // System V ABI: return value in eax.
    format!(
        ".text\n\
        .globl main\n\
        .type main, @function\n\
        main:\n\
        movl ${}, %eax\n\
        ret\n",
        imm
    )
}

fn emit_aarch64_main(imm: i32) -> String {
    // AArch64 AAPCS64: return value in w0.
    //
    // Use MOVZ/MOVK sequence for full 32-bit immediate.
    let v = imm as u32;
    let lo = (v & 0xFFFF) as u16;
    let hi = ((v >> 16) & 0xFFFF) as u16;

    let mut s = String::new();
    s.push_str(".text\n");
    s.push_str(".globl main\n");
    s.push_str(".type main, %function\n");
    s.push_str("main:\n");
    s.push_str(&format!("  movz w0, #{}\n", lo));
    if hi != 0 {
        s.push_str(&format!("  movk w0, #{}, lsl #16\n", hi));
    }
    s.push_str("  ret\n");
    s
}

fn emit_riscv64_main(imm: i32) -> String {
    // RISC-V ABI: return value in a0.
    // Use pseudo 'li' which GCC/GAS expands appropriately.
    format!(
        ".text\n\
        .globl main\n\
        .type main, @function\n\
        main:\n\
        li a0, {}\n\
        ret\n",
        imm
    )
}
