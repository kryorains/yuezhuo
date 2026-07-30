use super::Riscv64IrFuncEmitter;
use crate::codegen::common::{resolve_call_sig, IrArgLocation};
use crate::ir::{Type, ValueId};

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_call(
        &mut self,
        name: &str,
        args: &[ValueId],
        result: Option<ValueId>,
    ) -> Type {
        let (sig, arg_sigs) = resolve_call_sig(&self.parent.ctx, self.func, name, args);
        let locations = super::abi::assign_riscv_arg_locations(&arg_sigs);
        let stack_count = locations
            .iter()
            .filter(|location| matches!(location, IrArgLocation::Stack))
            .count();

        let all_integer_register_args = stack_count == 0
            && arg_sigs
                .iter()
                .all(|arg| arg.is_pointer || arg.ty != Type::F32);
        if all_integer_register_args {
            // Call operands are excluded from a2-a7 allocation, so filling ABI
            // registers backwards cannot overwrite a later operand source.
            for idx in (0..args.len()).rev() {
                let IrArgLocation::IntReg(reg_idx) = locations[idx] else {
                    unreachable!();
                };
                self.load_value_into(args[idx], &format!("a{}", reg_idx));
            }
            self.body.push_str(&format!("  call {}\n", name));
            if sig.ret == Type::F32 {
                if let Some(result) = result {
                    self.store_float_result(result, "fa0");
                }
            }
            return sig.ret;
        }

        for (arg, sig) in args.iter().zip(arg_sigs.iter()) {
            if sig.ty == Type::F32 && !sig.is_pointer {
                self.load_float_value(*arg, "fa0");
                self.push_s0();
            } else {
                self.load_value(*arg);
                self.push_x0();
            }
        }

        let saved_bytes = (args.len() as i32) * 16;
        let stack_bytes = (stack_count as i32) * 8;
        let pad = if stack_bytes % 16 == 0 { 0 } else { 8 };
        if stack_bytes + pad != 0 {
            self.adjust_sp(-(stack_bytes + pad));
        }

        let mut pushed_stack = 0usize;
        for (idx, location) in locations.iter().enumerate() {
            if matches!(location, IrArgLocation::Stack) {
                let saved_offset = stack_bytes + pad + ((args.len() - 1 - idx) as i32) * 16;
                self.load_sp_x("t2", saved_offset);
                self.store_sp_x("t2", (pushed_stack as i32) * 8);
                pushed_stack += 1;
            }
        }

        for (idx, location) in locations.iter().enumerate() {
            let saved_offset = stack_bytes + pad + ((args.len() - 1 - idx) as i32) * 16;
            match location {
                IrArgLocation::IntReg(reg_idx) => {
                    self.load_sp_x(&format!("a{}", reg_idx), saved_offset);
                }
                IrArgLocation::FloatReg(reg_idx) => {
                    self.load_sp_s(&format!("fa{}", reg_idx), saved_offset);
                }
                IrArgLocation::Stack => {}
            }
        }

        self.body.push_str(&format!("  call {}\n", name));
        if sig.ret == Type::F32 {
            if let Some(result) = result {
                self.store_float_result(result, "fa0");
            }
        }
        let cleanup = saved_bytes + stack_bytes + pad;
        if cleanup != 0 {
            self.adjust_sp(cleanup);
        }
        sig.ret
    }
}
