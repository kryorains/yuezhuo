use super::AArch64IrFuncEmitter;
use crate::codegen::common::{assign_arg_locations, IrArgLocation};
use crate::ir::Type;

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_params(&mut self) {
        let sig = self.parent.ctx.funcs.get(&self.func.name).cloned().unwrap();
        let locations = assign_arg_locations(&sig.params, 8, 8);
        let mut stack_idx = 0usize;
        for (idx, param) in self.func.params.iter().enumerate() {
            let param_sig = &sig.params[idx];
            let offset = self.layout.offset(*param);
            match locations[idx] {
                IrArgLocation::IntReg(reg_idx) => {
                    if let Some(target) = self.phi_regs.reg(*param) {
                        if param_sig.is_pointer {
                            self.body
                                .push_str(&format!("  mov {}, x{}\n", target, reg_idx));
                        } else {
                            self.body.push_str(&format!(
                                "  mov {}, w{}\n",
                                target.replacen('x', "w", 1),
                                reg_idx
                            ));
                        }
                    } else if param_sig.is_pointer {
                        self.store_frame_x(&format!("x{}", reg_idx), offset);
                    } else {
                        self.store_frame_w(&format!("w{}", reg_idx), offset);
                    }
                }
                IrArgLocation::FloatReg(reg_idx) => {
                    let source = format!("s{}", reg_idx);
                    if let Some(target) = self.assigned_float_reg(*param) {
                        if target != source {
                            self.body
                                .push_str(&format!("  fmov {}, {}\n", target, source));
                        }
                    }
                    // Keep the canonical parameter slot initialized for fallback
                    // paths and snapshot phi copies.
                    self.store_frame_s(&source, offset);
                }
                IrArgLocation::Stack => {
                    let src = 16 + (stack_idx as i32) * 8;
                    if param_sig.is_pointer {
                        self.load_frame_x("x0", src);
                        if self.phi_regs.reg(*param).is_some() {
                            self.store_result(*param);
                        } else {
                            self.store_frame_x("x0", offset);
                        }
                    } else if param_sig.ty == Type::F32 {
                        let target = self.assigned_float_reg(*param).unwrap_or("s0");
                        self.load_frame_s(target, src);
                        self.store_frame_s(target, offset);
                    } else {
                        self.load_frame_w("w0", src);
                        if self.phi_regs.reg(*param).is_some() {
                            self.store_result(*param);
                        } else {
                            self.store_frame_w("w0", offset);
                        }
                    }
                    stack_idx += 1;
                }
            }
        }
    }
}
