mod common;
pub mod target;

use crate::ast::Program;
use crate::ir::lower::lower_program;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    X86_64,
    AArch64,
    Riscv64,
}

pub fn emit_asm(target: Target, prog: &Program, _opt_o1: bool) -> String {
    match target {
        Target::X86_64 => {
            let module = lower_program(prog).unwrap_or_else(|e| panic!("IR lower failed: {:?}", e));
            target::x86_64::emit_ir_asm(&module)
        }
        Target::AArch64 => {
            let module = lower_program(prog).unwrap_or_else(|e| panic!("IR lower failed: {:?}", e));
            target::aarch64::emit_ir_asm(&module)
        }
        Target::Riscv64 => {
            let module = lower_program(prog).unwrap_or_else(|e| panic!("IR lower failed: {:?}", e));
            target::riscv64::emit_ir_asm(&module)
        }
    }
}
