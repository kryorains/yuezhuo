use super::X86IrFuncEmitter;
use crate::codegen::common::{assign_arg_locations, IrArgLocation};

impl<'a, 'b> X86IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_params(&mut self) {
        let int_regs = ["%edi", "%esi", "%edx", "%ecx", "%r8d", "%r9d"];
        let ptr_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        let float_regs = [
            "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        ];
        let sig = self.parent.ctx.funcs.get(&self.func.name).cloned().unwrap();
        let locations = assign_arg_locations(&sig.params, int_regs.len(), float_regs.len());
        let mut stack_idx = 0usize;
        for (idx, param) in self.func.params.iter().enumerate() {
            let param_sig = &sig.params[idx];
            let offset = self.layout.offset(*param);
            match locations[idx] {
                IrArgLocation::IntReg(reg_idx) => {
                    if param_sig.is_pointer {
                        self.body
                            .push_str(&format!("  movq {}, {}(%rbp)\n", ptr_regs[reg_idx], offset));
                    } else {
                        self.body
                            .push_str(&format!("  movl {}, {}(%rbp)\n", int_regs[reg_idx], offset));
                    }
                }
                IrArgLocation::FloatReg(reg_idx) => {
                    self.body.push_str(&format!(
                        "  movss {}, {}(%rbp)\n",
                        float_regs[reg_idx], offset
                    ));
                }
                IrArgLocation::Stack => {
                    let src = 16 + (stack_idx as i32) * 8;
                    if param_sig.is_pointer {
                        self.body.push_str(&format!("  movq {}(%rbp), %rax\n", src));
                        self.body
                            .push_str(&format!("  movq %rax, {}(%rbp)\n", offset));
                    } else {
                        self.body.push_str(&format!("  movl {}(%rbp), %eax\n", src));
                        self.body
                            .push_str(&format!("  movl %eax, {}(%rbp)\n", offset));
                    }
                    stack_idx += 1;
                }
            }
        }
    }
}
