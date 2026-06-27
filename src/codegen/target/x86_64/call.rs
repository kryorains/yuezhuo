use super::X86IrFuncEmitter;
use crate::codegen::common::{assign_arg_locations, resolve_call_sig, IrArgLocation};
use crate::ir::{Type, ValueId};

impl<'a, 'b> X86IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_call(&mut self, name: &str, args: &[ValueId]) -> Type {
        let int_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        let float_regs = [
            "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        ];
        let (sig, arg_sigs) = resolve_call_sig(&self.parent.ctx, self.func, name, args);
        let locations = assign_arg_locations(&arg_sigs, int_regs.len(), float_regs.len());
        let stack_count = locations
            .iter()
            .filter(|location| matches!(location, IrArgLocation::Stack))
            .count();
        let float_reg_count = locations
            .iter()
            .filter(|location| matches!(location, IrArgLocation::FloatReg(_)))
            .count();

        for (arg, sig) in args.iter().zip(arg_sigs.iter()) {
            if sig.ty == Type::F32 && !sig.is_pointer {
                self.load_float_value(*arg, "%xmm0");
                self.body
                    .push_str("  subq $8, %rsp\n  movq $0, (%rsp)\n  movss %xmm0, (%rsp)\n");
            } else {
                self.load_value(*arg);
                if !sig.is_pointer {
                    self.body.push_str("  cltq\n");
                }
                self.body.push_str("  pushq %rax\n");
            }
        }

        let saved_bytes = (args.len() as i32) * 8;
        let call_stack = saved_bytes + (stack_count as i32) * 8;
        let pad = if call_stack % 16 == 0 { 0 } else { 8 };
        if pad != 0 {
            self.body.push_str("  subq $8, %rsp\n");
        }

        let mut pushed_stack = 0usize;
        for (idx, location) in locations.iter().enumerate().rev() {
            if matches!(location, IrArgLocation::Stack) {
                let offset = pad + (pushed_stack as i32) * 8 + ((args.len() - 1 - idx) as i32) * 8;
                self.body
                    .push_str(&format!("  movq {}(%rsp), %rax\n  pushq %rax\n", offset));
                pushed_stack += 1;
            }
        }

        let saved_base = pad + (stack_count as i32) * 8;
        for (idx, location) in locations.iter().enumerate() {
            let offset = saved_base + ((args.len() - 1 - idx) as i32) * 8;
            match location {
                IrArgLocation::IntReg(reg_idx) => {
                    self.body.push_str(&format!(
                        "  movq {}(%rsp), {}\n",
                        offset, int_regs[*reg_idx]
                    ));
                }
                IrArgLocation::FloatReg(reg_idx) => {
                    self.body.push_str(&format!(
                        "  movss {}(%rsp), {}\n",
                        offset, float_regs[*reg_idx]
                    ));
                }
                IrArgLocation::Stack => {}
            }
        }
        self.body
            .push_str(&format!("  movb ${}, %al\n", float_reg_count));
        self.body.push_str(&format!("  call {}\n", name));
        let cleanup = saved_bytes + (stack_count as i32) * 8 + pad;
        if cleanup != 0 {
            self.body.push_str(&format!("  addq ${}, %rsp\n", cleanup));
        }
        sig.ret
    }
}
