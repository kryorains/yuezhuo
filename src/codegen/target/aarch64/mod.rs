mod abi;
mod call;
mod early_return;
mod emitter;
mod imm;
mod inst;
mod memory;
mod phi_regs;
mod thread;

use crate::codegen::common::{IrFuncLayout, IrLocalRegs};
use crate::ir::{AArch64ThreadPlan, Function, Module};
use phi_regs::AArch64PhiRegs;

struct AArch64IrEmitter<'a> {
    ctx: crate::codegen::common::IrModuleCtx<'a>,
    thread_plans: Vec<AArch64ThreadPlan>,
    out: String,
}

struct AArch64IrFuncEmitter<'a, 'b> {
    parent: &'a mut AArch64IrEmitter<'b>,
    func: &'b Function,
    layout: IrFuncLayout,
    phi_regs: AArch64PhiRegs,
    local_regs: IrLocalRegs,
    value_use_counts: Vec<usize>,
    thread_plan: Option<AArch64ThreadPlan>,
    body: String,
    return_label: String,
}

pub fn emit_ir_asm(module: &Module) -> String {
    emitter::emit_asm(module)
}
