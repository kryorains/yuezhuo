use super::regalloc::Riscv64RegAlloc;
use super::{Riscv64IrEmitter, Riscv64IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, entry_early_return, ir_align_to, IrFuncLayout, IrModuleCtx,
};
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
            regalloc: Riscv64RegAlloc::new(func),
            local_regs: crate::codegen::common::IrLocalRegs::new(
                func,
                &["t3", "t4", "t5", "t6"],
                true,
            ),
            value_use_counts: crate::codegen::common::ir_value_use_counts(func),
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
        }
    }

    fn emit(mut self) {
        let early_return = entry_early_return(self.func).and_then(|plan| {
            self.pre_prologue_early_return(&plan)
                .map(|prelude| (plan, prelude))
        });
        self.emit_params();
        if let Some((plan, _)) = &early_return {
            self.emit_phi_copies(self.func.entry.0, plan.slow_block.0);
            self.body
                .push_str(&format!("  j {}\n", self.block_label(plan.slow_block.0)));
        }
        let setup_end = self.body.len();
        for (block_idx, block) in self.func.blocks.iter().enumerate() {
            if early_return.as_ref().is_some_and(|(plan, _)| {
                block_idx == self.func.entry.0 || block_idx == plan.fast_block.0
            }) {
                continue;
            }
            self.body
                .push_str(&format!("{}:\n", self.block_label(block_idx)));
            for inst in &block.insts {
                self.emit_inst(inst);
            }
            if let Some(terminator) = &block.terminator {
                self.emit_terminator(block_idx, terminator);
            }
        }
        let setup = self.body[..setup_end].to_string();
        let blocks = self.body[setup_end..].to_string();
        let saved_regs = self.regalloc.used_regs().to_vec();
        let stack_size = ir_align_to(self.layout.stack_size + self.regalloc.saved_area_size(), 16);
        let recursive = self.func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                matches!(&inst.kind, crate::ir::InstKind::Call { name, .. } if name == &self.func.name)
            })
        });
        // Keep large/recursive hot functions on a stable fetch boundary. Small
        // functions use a modest alignment to limit padding growth.
        let function_alignment = if self.func.blocks.len() >= 16 || recursive {
            7
        } else {
            4
        };
        self.parent.out.push_str(&format!(
            ".p2align {1}\n.globl {0}\n.type {0}, @function\n{0}:\n",
            self.func.name, function_alignment
        ));
        if let Some((_, prelude)) = &early_return {
            self.parent.out.push_str(prelude);
        }
        self.parent
            .out
            .push_str("  addi sp, sp, -16\n  sd ra, 8(sp)\n  sd s0, 0(sp)\n  mv s0, sp\n");
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&format!("  li t6, {}\n  sub sp, sp, t6\n", stack_size));
        }
        for (idx, reg) in saved_regs.iter().enumerate() {
            self.parent
                .out
                .push_str(&format!("  sd {}, -{}(s0)\n", reg, (idx + 1) * 8));
        }
        self.parent.out.push_str(&setup);
        self.parent.out.push_str(&blocks);
        self.parent
            .out
            .push_str(&format!("{}:\n", self.return_label));
        for (idx, reg) in saved_regs.iter().enumerate() {
            self.parent
                .out
                .push_str(&format!("  ld {}, -{}(s0)\n", reg, (idx + 1) * 8));
        }
        self.parent
            .out
            .push_str("  mv sp, s0\n  ld ra, 8(sp)\n  ld s0, 0(sp)\n  addi sp, sp, 16\n  ret\n\n");
    }
}
