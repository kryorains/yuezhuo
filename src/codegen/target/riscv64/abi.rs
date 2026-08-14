use super::Riscv64IrFuncEmitter;
use crate::codegen::common::{IrArgLocation, IrParamSig};
use crate::ir::Type;

pub(super) fn assign_riscv_arg_locations(arg_sigs: &[IrParamSig]) -> Vec<IrArgLocation> {
    let mut int_idx = 0usize;
    let mut float_idx = 0usize;
    let mut locations = Vec::with_capacity(arg_sigs.len());
    for arg in arg_sigs {
        if !arg.is_pointer && arg.ty == Type::F32 && float_idx < 8 {
            locations.push(IrArgLocation::FloatReg(float_idx));
            float_idx += 1;
        } else if int_idx < 8 {
            // The hard-float RISC-V psABI falls back to the integer calling
            // convention after fa0-fa7 are exhausted.
            locations.push(IrArgLocation::IntReg(int_idx));
            int_idx += 1;
        } else {
            locations.push(IrArgLocation::Stack);
        }
    }
    locations
}

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_params(&mut self) {
        let sig = self.parent.ctx.funcs.get(&self.func.name).cloned().unwrap();
        let locations = assign_riscv_arg_locations(&sig.params);
        let mut stack_idx = 0usize;
        for (idx, param) in self.func.params.iter().enumerate() {
            let param_sig = &sig.params[idx];
            let assigned_reg = self.regalloc.reg(*param);
            let is_stack_arg = matches!(locations[idx], IrArgLocation::Stack);
            if self.value_use_counts[param.0] == 0 {
                // Unused incoming arguments need no register move or spill.
                // Keep the ABI stack cursor in sync for later live arguments.
                stack_idx += usize::from(is_stack_arg);
                continue;
            }
            match locations[idx] {
                IrArgLocation::IntReg(reg_idx) => {
                    let src = format!("a{}", reg_idx);
                    if param_sig.ty == Type::F32 && !param_sig.is_pointer {
                        if let Some(reg) = self.float_regalloc.reg(*param) {
                            self.body.push_str(&format!("  fmv.w.x {}, {}\n", reg, src));
                        } else {
                            self.store_frame_w(&src, self.layout.offset(*param));
                        }
                    } else if let Some(reg) = assigned_reg {
                        self.body.push_str(&format!("  mv {}, {}\n", reg, src));
                    } else if param_sig.is_pointer {
                        self.store_frame_x(&src, self.layout.offset(*param));
                    } else {
                        self.store_frame_w(&src, self.layout.offset(*param));
                    }
                }
                IrArgLocation::FloatReg(reg_idx) => {
                    let src = format!("fa{}", reg_idx);
                    if let Some(reg) = self.float_regalloc.reg(*param) {
                        self.body.push_str(&format!("  fmv.s {}, {}\n", reg, src));
                    } else {
                        self.store_frame_s(&src, self.layout.offset(*param));
                    }
                }
                IrArgLocation::Stack => {
                    let src = 16 + (stack_idx as i32) * 8;
                    if param_sig.is_pointer {
                        self.load_raw_frame_x("a0", src);
                        if let Some(reg) = assigned_reg {
                            self.body.push_str(&format!("  mv {}, a0\n", reg));
                        } else {
                            self.store_frame_x("a0", self.layout.offset(*param));
                        }
                    } else if param_sig.ty == Type::F32 {
                        if let Some(reg) = self.float_regalloc.reg(*param) {
                            self.load_raw_frame_s(reg, src);
                        } else {
                            self.load_raw_frame_s("fa0", src);
                            self.store_frame_s("fa0", self.layout.offset(*param));
                        }
                    } else {
                        self.load_raw_frame_w("a0", src);
                        if let Some(reg) = assigned_reg {
                            self.body.push_str(&format!("  mv {}, a0\n", reg));
                        } else {
                            self.store_frame_w("a0", self.layout.offset(*param));
                        }
                    }
                    stack_idx += 1;
                }
            }
        }
    }
}
