mod abi;
mod call;
mod early_return;
mod emitter;
mod inst;
mod memory;
mod regalloc;

use crate::codegen::common::{IrFuncLayout, IrLocalRegs};
use crate::ir::{Function, Module, ValueId};
use regalloc::{Riscv64FloatRegAlloc, Riscv64RegAlloc};
use std::collections::HashMap;

struct Riscv64IrEmitter<'a> {
    ctx: crate::codegen::common::IrModuleCtx<'a>,
    out: String,
}

struct Riscv64IrFuncEmitter<'a, 'b> {
    parent: &'a mut Riscv64IrEmitter<'b>,
    func: &'b Function,
    layout: IrFuncLayout,
    regalloc: Riscv64RegAlloc,
    float_regalloc: Riscv64FloatRegAlloc,
    saved_area_size: i32,
    local_regs: IrLocalRegs,
    value_use_counts: Vec<usize>,
    folded_memory_geps: HashMap<ValueId, memory::FoldedMemoryGep>,
    body: String,
    return_label: String,
}

pub fn emit_ir_asm(module: &Module) -> String {
    emitter::emit_asm(module)
}

#[cfg(test)]
mod tests;
