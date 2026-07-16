use super::{X86IrEmitter, X86IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, emulated_bitwise_op, emulated_shift_op, ir_align_to, EmulatedBitwiseOp,
    EmulatedShiftOp, IrFuncLayout, IrModuleCtx,
};
use crate::ir::{Function, Module};

pub(super) fn emit_asm(module: &Module) -> String {
    X86IrEmitter::new(module).emit()
}

impl<'a> X86IrEmitter<'a> {
    fn new(module: &'a Module) -> Self {
        Self {
            ctx: IrModuleCtx::new(module),
            out: String::new(),
        }
    }

    fn emit(mut self) -> String {
        self.out
            .push_str(&emit_ir_data_section(self.ctx.module, ".long"));
        self.out.push_str(".text\n");
        for func in &self.ctx.module.funcs {
            X86IrFuncEmitter::new(&mut self, func).emit();
        }
        self.out
    }
}

impl<'a, 'b> X86IrFuncEmitter<'a, 'b> {
    fn new(parent: &'a mut X86IrEmitter<'b>, func: &'b Function) -> Self {
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
            ".globl {0}\n.type {0}, @function\n{0}:\n",
            self.func.name
        ));
        if let Some(op) = emulated_bitwise_op(self.func) {
            let slow_label = self.parent.ctx.fresh_label("bitwise_slow");
            let instruction = match op {
                EmulatedBitwiseOp::And => "andl",
                EmulatedBitwiseOp::Or => "orl",
                EmulatedBitwiseOp::Xor => "xorl",
            };
            self.parent.out.push_str(&format!(
                "  testl %edi, %edi\n  js {0}\n  testl %esi, %esi\n  js {0}\n  movl %edi, %eax\n  {1} %esi, %eax\n  ret\n{0}:\n",
                slow_label, instruction
            ));
        } else if let Some(op) = emulated_shift_op(self.func) {
            let slow_label = self.parent.ctx.fresh_label("shift_slow");
            self.parent.out.push_str(&format!(
                "  leal -1(%rsi), %ecx\n  cmpl $7, %ecx\n  ja {0}\n  movl %esi, %ecx\n  movl %edi, %eax\n",
                slow_label
            ));
            match op {
                EmulatedShiftOp::Left => self
                    .parent
                    .out
                    .push_str(&format!("  sall %cl, %eax\n  ret\n{}:\n", slow_label)),
                EmulatedShiftOp::SignedRight => {
                    let signed_label = self.parent.ctx.fresh_label("shift_signed");
                    self.parent.out.push_str(&format!(
                        "  testl %eax, %eax\n  js {0}\n  sarl %cl, %eax\n  ret\n{0}:\n  movl $1, %r8d\n  sall %cl, %r8d\n  cltd\n  idivl %r8d\n  ret\n{1}:\n",
                        signed_label, slow_label
                    ));
                }
            }
        }
        self.parent
            .out
            .push_str("  pushq %rbp\n  movq %rsp, %rbp\n");
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&format!("  subq ${}, %rsp\n", stack_size));
        }
        self.parent.out.push_str(&self.body);
        self.parent
            .out
            .push_str(&format!("{}:\n  leave\n  ret\n\n", self.return_label));
    }
}
