use super::AArch64IrFuncEmitter;
use super::abi::{IrArgLocation, assign_locations};
use crate::codegen::common::{IrFuncSig, IrParamSig};
use crate::ir::{Type, ValueId};

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_call(&mut self, name: &str, args: &[ValueId]) -> Type {
        let sig = self
            .parent
            .ctx
            .funcs
            .get(name)
            .cloned()
            .unwrap_or_else(|| IrFuncSig {
                ret: Type::I32,
                params: args
                    .iter()
                    .map(|arg| {
                        let ty = self.func.value(*arg).ty.clone();
                        IrParamSig {
                            is_pointer: matches!(ty, Type::Ptr(_)),
                            ty,
                        }
                    })
                    .collect(),
            });
        let arg_sigs = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| {
                sig.params.get(idx).cloned().unwrap_or_else(|| {
                    let ty = self.func.value(*arg).ty.clone();
                    IrParamSig {
                        is_pointer: matches!(ty, Type::Ptr(_)),
                        ty,
                    }
                })
            })
            .collect::<Vec<_>>();
        let locations = assign_locations(&arg_sigs, 8, 8);
        let stack_count = locations
            .iter()
            .filter(|location| matches!(location, IrArgLocation::Stack))
            .count();

        for (arg, sig) in args.iter().zip(arg_sigs.iter()) {
            if sig.ty == Type::F32 && !sig.is_pointer {
                self.load_float_value(*arg, "s0");
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
                self.load_sp_x("x15", saved_offset);
                self.store_sp_x("x15", (pushed_stack as i32) * 8);
                pushed_stack += 1;
            }
        }

        for (idx, location) in locations.iter().enumerate() {
            let saved_offset = stack_bytes + pad + ((args.len() - 1 - idx) as i32) * 16;
            match location {
                IrArgLocation::IntReg(reg_idx) => {
                    self.load_sp_x(&format!("x{}", reg_idx), saved_offset);
                }
                IrArgLocation::FloatReg(reg_idx) => {
                    self.load_sp_s(&format!("s{}", reg_idx), saved_offset);
                }
                IrArgLocation::Stack => {}
            }
        }

        self.body.push_str(&format!("  bl {}\n", name));
        let cleanup = saved_bytes + stack_bytes + pad;
        if cleanup != 0 {
            self.adjust_sp(cleanup);
        }
        sig.ret
    }
}
