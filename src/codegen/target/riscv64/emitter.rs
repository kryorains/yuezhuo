use super::{Riscv64IrEmitter, Riscv64IrFuncEmitter};
use crate::codegen::common::{emit_ir_data_section, ir_align_to, IrFuncLayout, IrModuleCtx};
use crate::ir::{Function, Module};

pub(super) fn emit_asm(module: &Module) -> String {
    Riscv64IrEmitter::new(module).emit()
}

impl<'a> Riscv64IrEmitter<'a> {
    fn new(module: &'a Module) -> Self {
        Self {
            ctx: IrModuleCtx::new(module),
            out: String::new(),
        }
    }

    fn emit(mut self) -> String {
        self.out
            .push_str(&emit_ir_data_section(self.ctx.module, ".word"));
        self.out.push_str(".text\n");
        for func in &self.ctx.module.funcs {
            Riscv64IrFuncEmitter::new(&mut self, func).emit();
        }
        self.out
    }
}

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    fn new(parent: &'a mut Riscv64IrEmitter<'b>, func: &'b Function) -> Self {
        Self {
            parent,
            func,
            layout: IrFuncLayout::new(func),
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
        }
    }

    fn emit(mut self) {
        self.emit_params();
        for (block_idx, block) in self.func.blocks.iter().enumerate() {
            self.body
                .push_str(&format!("{}:\n", self.block_label(block_idx)));
            for inst in &block.insts {
                self.emit_inst(inst);
            }
            if let Some(terminator) = &block.terminator {
                self.emit_terminator(block_idx, terminator);
            }
        }

        let stack_size = ir_align_to(self.layout.stack_size, 16);
        self.parent.out.push_str(&format!(
            ".globl {0}\n.type {0}, @function\n{0}:\n  addi sp, sp, -16\n  sd ra, 8(sp)\n  sd s0, 0(sp)\n  mv s0, sp\n",
            self.func.name
        ));
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&format!("  li t6, {}\n  sub sp, sp, t6\n", stack_size));
        }
        self.parent.out.push_str(&self.body);
        self.parent
            .out
            .push_str(&format!("{}:\n", self.return_label));
        self.parent
            .out
            .push_str("  mv sp, s0\n  ld ra, 8(sp)\n  ld s0, 0(sp)\n  addi sp, sp, 16\n  ret\n\n");
    }
}
