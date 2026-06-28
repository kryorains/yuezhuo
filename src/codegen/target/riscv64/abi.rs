use super::Riscv64IrFuncEmitter;
use crate::codegen::common::IrParamSig;
use crate::ir::Type;

#[derive(Debug, Clone, Copy)]
pub(super) enum IrArgLocation {
    IntReg(usize),
    FloatReg(usize),
    Stack,
}

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_params(&mut self) {
        let sig = self.parent.ctx.funcs.get(&self.func.name).cloned().unwrap();
        let locations = assign_locations(&sig.params, 8, 8);
        let mut stack_idx = 0usize;
        for (idx, param) in self.func.params.iter().enumerate() {
            let param_sig = &sig.params[idx];
            let assigned_reg = self.regalloc.reg(*param);
            let offset = self.layout.offset(*param);
            match locations[idx] {
                IrArgLocation::IntReg(reg_idx) => {
                    let src = format!("a{}", reg_idx);
                    if let Some(reg) = assigned_reg {
                        self.body.push_str(&format!("  mv {}, {}\n", reg, src));
                    } else if param_sig.is_pointer {
                        self.store_frame_x(&src, offset);
                    } else {
                        self.store_frame_w(&src, offset);
                    }
                }
                IrArgLocation::FloatReg(reg_idx) => {
                    self.store_frame_s(&format!("fa{}", reg_idx), offset);
                }
                IrArgLocation::Stack => {
                    let src = 16 + (stack_idx as i32) * 8;
                    if param_sig.is_pointer {
                        self.load_raw_frame_x("a0", src);
                        if let Some(reg) = assigned_reg {
                            self.body.push_str(&format!("  mv {}, a0\n", reg));
                        } else {
                            self.store_frame_x("a0", offset);
                        }
                    } else if param_sig.ty == Type::F32 {
                        self.load_raw_frame_s("fa0", src);
                        self.store_frame_s("fa0", offset);
                    } else {
                        self.load_raw_frame_w("a0", src);
                        if let Some(reg) = assigned_reg {
                            self.body.push_str(&format!("  mv {}, a0\n", reg));
                        } else {
                            self.store_frame_w("a0", offset);
                        }
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
