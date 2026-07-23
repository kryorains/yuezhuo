mod abi;
mod call;
mod early_return;
mod emitter;
mod inst;
mod memory;
mod regalloc;

use crate::codegen::common::{IrFuncLayout, IrLocalRegs};
use crate::ir::{Function, Module, ValueId};
use regalloc::Riscv64RegAlloc;
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
mod tests {
    use super::regalloc::Riscv64FloatRegAlloc;
    use crate::ir::{Function, Type, ValueId};

    #[test]
    fn float_regalloc_api_is_available_to_target_consumers() {
        let func = Function::new("float_regalloc_visibility", Type::Void);
        let regs = Riscv64FloatRegAlloc::new(&func);

        assert_eq!(regs.reg(ValueId(0)), None);
        assert!(regs.used_callee_saved().is_empty());
    }
}
