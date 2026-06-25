mod abi;
mod call;
mod emitter;
mod inst;
mod memory;

use crate::codegen::common::IrFuncLayout;
use crate::ir::{Function, Module};

struct Riscv64IrEmitter<'a> {
    ctx: crate::codegen::common::IrModuleCtx<'a>,
    out: String,
}

struct Riscv64IrFuncEmitter<'a, 'b> {
    parent: &'a mut Riscv64IrEmitter<'b>,
    func: &'b Function,
    layout: IrFuncLayout,
    body: String,
    return_label: String,
}

pub fn emit_ir_asm(module: &Module) -> String {
    emitter::emit_asm(module)
}
