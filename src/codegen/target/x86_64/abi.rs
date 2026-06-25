use super::X86IrFuncEmitter;
use crate::codegen::common::IrParamSig;
use crate::ir::Type;

#[derive(Debug, Clone, Copy)]
pub(super) enum IrArgLocation {
    IntReg(usize),
    FloatReg(usize),
    Stack,
}

impl<'a, 'b> X86IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_params(&mut self) {
        let int_regs = ["%edi", "%esi", "%edx", "%ecx", "%r8d", "%r9d"];
        let ptr_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        let float_regs = [
            "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        ];
        let sig = self.parent.ctx.funcs.get(&self.func.name).cloned().unwrap();
        let locations = assign_locations(&sig.params, int_regs.len(), float_regs.len());
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

pub(super) fn assign_locations(
    arg_sigs: &[IrParamSig],
    int_reg_count: usize,
    float_reg_count: usize,
) -> Vec<IrArgLocation> {
    let mut int_idx = 0usize;
    let mut float_idx = 0usize;
    let mut locations = Vec::with_capacity(arg_sigs.len());
    for arg_sig in arg_sigs {
        if arg_sig.is_pointer || arg_sig.ty != Type::F32 {
            if int_idx < int_reg_count {
                locations.push(IrArgLocation::IntReg(int_idx));
                int_idx += 1;
            } else {
                locations.push(IrArgLocation::Stack);
            }
        } else if float_idx < float_reg_count {
            locations.push(IrArgLocation::FloatReg(float_idx));
            float_idx += 1;
        } else {
            locations.push(IrArgLocation::Stack);
        }
    }
    locations
}
