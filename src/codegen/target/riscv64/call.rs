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
            if !self.emit_guarded_pow2_digit_call(name, args)
                && !self.emit_guarded_mulmod_call(name)
            {
                self.body.push_str(&format!("  call {}\n", name));
            }
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

    fn emit_guarded_pow2_digit_call(&mut self, name: &str, args: &[ValueId]) -> bool {
        let Some(shift) = self
            .parent
            .ctx
            .module
            .funcs
            .iter()
            .find(|func| func.name == name)
            .and_then(|func| func.guarded_pow2_digit_shift())
        else {
            return false;
        };
        let zero_position = 32u32.div_ceil(shift);
        let mask = (1u32 << shift) - 1;
        if let [_, position] = args {
            if let Some(position) = self.const_i32(*position) {
                if position < 0 {
                    self.body.push_str(&format!("  call {name}\n"));
                    return true;
                }
                if (position as u32) >= zero_position {
                    self.body.push_str("  li a0, 0\n");
                    return true;
                }
                let amount = (position as u32) * shift;
                let fallback = self
                    .parent
                    .ctx
                    .fresh_label("pow2_digit_const_call_fallback");
                let done = self.parent.ctx.fresh_label("pow2_digit_const_call_done");
                self.body.push_str(&format!("  bltz a0, {fallback}\n"));
                if amount != 0 {
                    self.body.push_str(&format!("  sraiw a0, a0, {amount}\n"));
                }
                if mask <= 2047 {
                    self.body.push_str(&format!("  andi a0, a0, {mask}\n"));
                } else {
                    self.body
                        .push_str(&format!("  li t0, {mask}\n  and a0, a0, t0\n"));
                }
                self.body.push_str(&format!(
                    "  j {done}\n{fallback}:\n  call {name}\n{done}:\n"
                ));
                return true;
            }
        }
        let scale_position = if shift.is_power_of_two() {
            format!("  slliw t1, a1, {}\n", shift.trailing_zeros())
        } else {
            format!("  li t0, {shift}\n  mulw t1, a1, t0\n")
        };
        let apply_mask = if mask <= 2047 {
            format!("  andi a0, a0, {mask}\n")
        } else {
            format!("  li t0, {mask}\n  and a0, a0, t0\n")
        };
        let fallback = self.parent.ctx.fresh_label("pow2_digit_call_fallback");
        let zero = self.parent.ctx.fresh_label("pow2_digit_call_zero");
        let done = self.parent.ctx.fresh_label("pow2_digit_call_done");
        self.body.push_str(&format!(
            "  bltz a1, {fallback}\n  li t0, {zero_position}\n  bge a1, t0, {zero}\n  bltz a0, {fallback}\n{scale_position}  sraw a0, a0, t1\n{apply_mask}  j {done}\n{zero}:\n  li a0, 0\n  j {done}\n{fallback}:\n  call {name}\n{done}:\n"
        ));
        true
    }

    fn const_i32(&self, value: ValueId) -> Option<i32> {
        match self.func.value(value).kind {
            crate::ir::ValueKind::Const(crate::ir::Const::Int(value)) => Some(value),
            crate::ir::ValueKind::Const(crate::ir::Const::Zero(Type::I32)) => Some(0),
            _ => None,
        }
    }

    fn emit_guarded_mulmod_call(&mut self, name: &str) -> bool {
        let Some(modulus) = self
            .parent
            .ctx
            .module
            .funcs
            .iter()
            .find(|func| func.name == name)
            .and_then(|func| func.guarded_mulmod_modulus())
        else {
            return false;
        };
        let fallback = self.parent.ctx.fresh_label("mulmod_call_fallback");
        let done = self.parent.ctx.fresh_label("mulmod_call_done");
        self.body.push_str(&format!(
            "  bltz a0, {fallback}\n  bltz a1, {fallback}\n  li t0, {modulus}\n  bge a0, t0, {fallback}\n  bge a1, t0, {fallback}\n  mul t1, a0, a1\n  rem a0, t1, t0\n  j {done}\n{fallback}:\n  call {name}\n{done}:\n"
        ));
        true
    }
}
