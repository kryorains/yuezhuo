use super::imm::mov_x_imm;
use super::phi_regs::AArch64PhiRegs;
use super::{AArch64IrEmitter, AArch64IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, emulated_bitwise_op, emulated_shift_op, ir_align_to, EmulatedBitwiseOp,
    EmulatedShiftOp, IrFuncLayout, IrModuleCtx,
};
use crate::ir::{Function, Module};

pub(super) fn emit_asm(module: &Module) -> String {
    AArch64IrEmitter::new(module).emit()
}

impl<'a> AArch64IrEmitter<'a> {
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
            AArch64IrFuncEmitter::new(&mut self, func).emit();
        }
        self.out
    }
}

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    fn new(parent: &'a mut AArch64IrEmitter<'b>, func: &'b Function) -> Self {
        Self {
            parent,
            func,
            layout: IrFuncLayout::new(func),
            phi_regs: AArch64PhiRegs::new(func),
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

        let saved_regs = self.phi_regs.saved_regs().to_vec();
        let saved_bytes = (saved_regs.len() as i32) * 8;
        let stack_size = ir_align_to(self.layout.stack_size + saved_bytes, 16);
        self.parent.out.push_str(&format!(
            ".globl {0}\n.type {0}, %function\n{0}:\n",
            self.func.name
        ));
        if let Some(op) = emulated_bitwise_op(self.func) {
            let slow_label = self.parent.ctx.fresh_label("bitwise_slow");
            let instruction = match op {
                EmulatedBitwiseOp::And => "and",
                EmulatedBitwiseOp::Or => "orr",
                EmulatedBitwiseOp::Xor => "eor",
            };
            self.parent.out.push_str(&format!(
                "  tbnz w0, #31, {0}\n  tbnz w1, #31, {0}\n  {1} w0, w0, w1\n  ret\n{0}:\n",
                slow_label, instruction
            ));
        } else if let Some(op) = emulated_shift_op(self.func) {
            let slow_label = self.parent.ctx.fresh_label("shift_slow");
            self.parent.out.push_str(&format!(
                "  sub w2, w1, #1\n  cmp w2, #7\n  b.hi {0}\n",
                slow_label
            ));
            match op {
                EmulatedShiftOp::Left => self
                    .parent
                    .out
                    .push_str(&format!("  lsl w0, w0, w1\n  ret\n{}:\n", slow_label)),
                EmulatedShiftOp::SignedRight => {
                    let signed_label = self.parent.ctx.fresh_label("shift_signed");
                    self.parent.out.push_str(&format!(
                        "  tbnz w0, #31, {0}\n  asr w0, w0, w1\n  ret\n{0}:\n  mov w2, #1\n  lsl w2, w2, w1\n  sdiv w0, w0, w2\n  ret\n{1}:\n",
                        signed_label, slow_label
                    ));
                }
            }
        }
        self.parent
            .out
            .push_str("  stp x29, x30, [sp, #-16]!\n  mov x29, sp\n");
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&mov_x_imm("x16", stack_size as i64));
            self.parent.out.push_str("  sub sp, sp, x16\n");
        }
        for (idx, reg) in saved_regs.iter().enumerate() {
            self.parent
                .out
                .push_str(&format!("  str {}, [sp, #{}]\n", reg, idx * 8));
        }
        self.parent.out.push_str(&self.body);
        self.parent
            .out
            .push_str(&format!("{}:\n", self.return_label));
        for (idx, reg) in saved_regs.iter().enumerate() {
            self.parent
                .out
                .push_str(&format!("  ldr {}, [sp, #{}]\n", reg, idx * 8));
        }
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&mov_x_imm("x16", stack_size as i64));
            self.parent.out.push_str("  add sp, sp, x16\n");
        }
        self.parent
            .out
            .push_str("  ldp x29, x30, [sp], #16\n  ret\n\n");
    }
}
