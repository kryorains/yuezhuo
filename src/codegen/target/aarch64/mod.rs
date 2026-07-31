mod abi;
mod call;
mod early_return;
mod emitter;
mod float_regs;
mod imm;
mod inst;
mod memory;
mod phi_regs;

use crate::codegen::common::{IrFuncLayout, IrLocalRegs};
use crate::ir::{Function, Module};
use float_regs::AArch64FloatRegs;
use phi_regs::AArch64PhiRegs;
use std::collections::HashMap;

struct AArch64IrEmitter<'a> {
    ctx: crate::codegen::common::IrModuleCtx<'a>,
    int_return_ranges: HashMap<String, crate::ir::int_range::IntRange>,
    out: String,
}

struct AArch64IrFuncEmitter<'a, 'b> {
    parent: &'a mut AArch64IrEmitter<'b>,
    func: &'b Function,
    layout: IrFuncLayout,
    phi_regs: AArch64PhiRegs,
    float_regs: AArch64FloatRegs,
    saved_area_size: i32,
    local_regs: IrLocalRegs,
    value_use_counts: Vec<usize>,
    int_ranges: Vec<Option<crate::ir::int_range::IntRange>>,
    folded_memory_geps: HashMap<crate::ir::ValueId, memory::FoldedMemoryGep>,
    frame_accessed: bool,
    body: String,
    return_label: String,
}

pub fn emit_ir_asm(module: &Module) -> String {
    emitter::emit_asm(module)
}

#[cfg(test)]
mod tests;
