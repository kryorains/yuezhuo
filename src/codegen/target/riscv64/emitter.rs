use super::regalloc::Riscv64RegAlloc;
use super::{Riscv64IrEmitter, Riscv64IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, emulated_bitwise_op, emulated_shift_op, ir_align_to, EmulatedBitwiseOp,
    EmulatedShiftOp, IrFuncLayout, IrModuleCtx,
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
            local_regs: crate::codegen::common::IrLocalRegs::new(func, &["t3", "t4", "t5", "t6"]),
            value_use_counts: crate::codegen::common::ir_value_use_counts(func),
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

        let saved_regs = self.regalloc.used_regs().to_vec();
        let stack_size = ir_align_to(self.layout.stack_size + self.regalloc.saved_area_size(), 16);
        self.parent.out.push_str(&format!(
            ".globl {0}\n.type {0}, @function\n{0}:\n",
            self.func.name
        ));
        if let Some(op) = emulated_bitwise_op(self.func) {
            let slow_label = self.parent.ctx.fresh_label("bitwise_slow");
            let instruction = match op {
                EmulatedBitwiseOp::And => "and",
                EmulatedBitwiseOp::Or => "or",
                EmulatedBitwiseOp::Xor => "xor",
            };
            self.parent.out.push_str(&format!(
                "  bltz a0, {0}\n  bltz a1, {0}\n  {1} a0, a0, a1\n  ret\n{0}:\n",
                slow_label, instruction
            ));
        } else if let Some(op) = emulated_shift_op(self.func) {
            let slow_label = self.parent.ctx.fresh_label("shift_slow");
            self.parent.out.push_str(&format!(
                "  addi t0, a1, -1\n  li t1, 7\n  bgtu t0, t1, {0}\n",
                slow_label
            ));
            match op {
                EmulatedShiftOp::Left => self
                    .parent
                    .out
                    .push_str(&format!("  sllw a0, a0, a1\n  ret\n{}:\n", slow_label)),
                EmulatedShiftOp::SignedRight => {
                    let signed_label = self.parent.ctx.fresh_label("shift_signed");
                    self.parent.out.push_str(&format!(
                        "  bltz a0, {0}\n  sraw a0, a0, a1\n  ret\n{0}:\n  li t0, 1\n  sllw t0, t0, a1\n  divw a0, a0, t0\n  ret\n{1}:\n",
                        signed_label, slow_label
                    ));
                }
            }
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
        self.parent.out.push_str(&self.body);
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
