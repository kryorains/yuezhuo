mod common;
mod cost;
pub mod target;

use crate::ir::Module;
pub use cost::TargetCostModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    X86_64,
    Riscv64,
}

impl Target {
    pub const fn cost_model(self) -> TargetCostModel {
        TargetCostModel::for_target(self)
    }
}

pub fn emit_asm(target: Target, module: &Module) -> String {
    match target {
        Target::X86_64 => target::x86_64::emit_ir_asm(module),
        Target::Riscv64 => target::riscv64::emit_ir_asm(module),
    }
}
