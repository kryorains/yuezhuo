mod abi;
mod call;
mod emitter;
mod inst;
mod memory;
mod regalloc;

use crate::codegen::common::IrFuncLayout;
use crate::ir::{Function, Module};
use regalloc::Riscv64RegAlloc;

struct Riscv64IrEmitter<'a> {
    ctx: crate::codegen::common::IrModuleCtx<'a>,
    out: String,
}

struct Riscv64IrFuncEmitter<'a, 'b> {
    parent: &'a mut Riscv64IrEmitter<'b>,
    func: &'b Function,
    layout: IrFuncLayout,
    regalloc: Riscv64RegAlloc,
    body: String,
    return_label: String,
}

pub fn emit_ir_asm(module: &Module) -> String {
    emitter::emit_asm(module)
}
