mod abi;
mod call;
mod early_return;
mod emitter;
mod inst;
mod memory;
pub(in crate::codegen::target) mod regalloc;

use crate::codegen::common::{IrFuncLayout, IrLocalRegs};
use crate::ir::{Function, Module, ValueId};
use regalloc::{Riscv64FloatRegAlloc, Riscv64RegAlloc};
use std::collections::{HashMap, HashSet};

struct Riscv64IrEmitter<'a> {
    ctx: crate::codegen::common::IrModuleCtx<'a>,
    int_return_ranges: HashMap<String, crate::ir::int_range::IntRange>,
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
    int_ranges: Vec<Option<crate::ir::int_range::IntRange>>,
    folded_memory_geps: HashMap<ValueId, memory::FoldedMemoryGep>,
    elided_values: HashSet<ValueId>,
    body: String,
    return_label: String,
}

pub fn emit_ir_asm(module: &Module) -> String {
    emitter::emit_asm(module)
}
