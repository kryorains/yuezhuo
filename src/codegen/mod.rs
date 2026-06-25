mod common;
pub mod target;

use crate::ir::Module;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    X86_64,
    AArch64,
    Riscv64,
}

pub fn emit_asm(target: Target, module: &Module) -> String {
    match target {
        Target::X86_64 => target::x86_64::emit_ir_asm(module),
        Target::AArch64 => target::aarch64::emit_ir_asm(module),
        Target::Riscv64 => target::riscv64::emit_ir_asm(module),
    }
}
