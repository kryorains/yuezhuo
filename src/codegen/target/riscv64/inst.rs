use super::Riscv64IrFuncEmitter;
use crate::ir::{
    BinaryOp, BlockId, CastOp, CmpOp, Const, Inst, InstKind, Terminator, Type, UnaryOp, ValueId,
    ValueKind,
};

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_inst(&mut self, inst: &Inst) {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { incomings } => {
                let result = inst.result.unwrap();
                for (pred, value) in incomings {
                    self.body
                        .push_str(&format!("# phi {} from {} = {}\n", result, pred, value));
                }
            }
            InstKind::Alloca { ty } => {
                let result = inst.result.unwrap();
                self.frame_addr("a0", self.object_offset(result, ty));
                self.store_result(result);
            }
            InstKind::Load { ptr } => {
                let result = inst.result.unwrap();
                self.load_value(*ptr);
                let ty = self.func.value(result).ty.clone();
                self.load_indirect(&ty);
                self.store_result(result);
            }
            InstKind::Store { ptr, value } => {
                self.load_value(*ptr);
                self.push_x0();
                self.load_value(*value);
                self.pop_x1();
                let ty = self.func.value(*value).ty.clone();
                self.store_indirect(&ty);
            }
            InstKind::MemZero { ptr, bytes } => self.emit_memzero(*ptr, *bytes),
            InstKind::Unary { op, value } => {
                let result = inst.result.unwrap();
                self.emit_unary(*op, *value);
                self.store_result(result);
            }
            InstKind::Binary { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                self.emit_binary(*op, *lhs, *rhs);
                self.store_result(result);
            }
            InstKind::Icmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                self.emit_icmp(*op, *lhs, *rhs);
                self.store_result(result);
            }
            InstKind::Fcmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                self.emit_fcmp(*op, *lhs, *rhs);
                self.store_result(result);
            }
            InstKind::Cast { op, value } => {
                let result = inst.result.unwrap();
                self.emit_cast(*op, *value);
                self.store_result(result);
            }
            InstKind::Gep { base, indices } => {
                let result = inst.result.unwrap();
                self.emit_gep(result, *base, indices);
                self.store_result(result);
            }
            InstKind::Call { name, args } => {
                let ret = self.emit_call(name, args);
                if let Some(result) = inst.result {
                    if ret == Type::F32 {
                        self.store_frame_s("fa0", self.layout.offset(result));
                    } else {
                        self.store_result(result);
                    }
                }
            }
        }
    }

    pub(super) fn emit_terminator(&mut self, block_idx: usize, terminator: &Terminator) {
        match terminator {
            Terminator::Return(value) => {
                if let Some(value) = value {
                    self.load_value(*value);
                    if self.func.value(*value).ty == Type::F32 {
                        self.body.push_str("  fmv.w.x fa0, a0\n");
                    }
                }
                self.body.push_str(&format!("  j {}\n", self.return_label));
            }
            Terminator::Jump(target) => {
                self.emit_phi_copies(block_idx, target.0);
                self.body
                    .push_str(&format!("  j {}\n", self.block_label(target.0)));
            }
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                let else_edge = self.parent.ctx.fresh_label("else_edge");
                self.load_value(*cond);
                self.body.push_str(&format!("  beqz a0, {}\n", else_edge));
                self.emit_phi_copies(block_idx, then_target.0);
                self.body.push_str(&format!(
                    "  j {}\n{}:\n",
                    self.block_label(then_target.0),
                    else_edge
                ));
                self.emit_phi_copies(block_idx, else_target.0);
                self.body
                    .push_str(&format!("  j {}\n", self.block_label(else_target.0)));
            }
        }
    }

    fn emit_phi_copies(&mut self, pred_idx: usize, target_idx: usize) {
        let copies = self
            .func
            .block(BlockId(target_idx))
            .insts
            .iter()
            .filter_map(|inst| {
                let incomings = match &inst.kind {
                    InstKind::Nop => return None,
                    InstKind::Phi { incomings } => incomings,
                    _ => return None,
                };
                let result = inst.result.unwrap();
                incomings
                    .iter()
                    .find(|(pred, _)| pred.0 == pred_idx)
                    .map(|(_, value)| (result, *value))
            })
            .collect::<Vec<_>>();

        for (_, value) in &copies {
            self.load_value(*value);
            self.push_x0();
        }
        for (result, _) in copies.iter().rev() {
            self.pop_x1();
            self.body.push_str("  mv a0, a1\n");
            self.store_result(*result);
        }
    }

    fn emit_unary(&mut self, op: UnaryOp, value: ValueId) {
        self.load_value(value);
        match op {
            UnaryOp::Ineg => self.body.push_str("  negw a0, a0\n"),
            UnaryOp::Fneg => self
                .body
                .push_str("  fmv.w.x fa0, a0\n  fneg.s fa0, fa0\n  fmv.x.w a0, fa0\n"),
            UnaryOp::Not => self.body.push_str("  seqz a0, a0\n"),
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) {
        if self.emit_binary_imm(op, lhs, rhs) {
            return;
        }

        match op {
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => {
                self.load_float_value(lhs, "fa0");
                self.push_s0();
                self.load_float_value(rhs, "fa0");
                self.pop_s1();
                match op {
                    BinaryOp::Fadd => self.body.push_str("  fadd.s fa0, fa1, fa0\n"),
                    BinaryOp::Fsub => self.body.push_str("  fsub.s fa0, fa1, fa0\n"),
                    BinaryOp::Fmul => self.body.push_str("  fmul.s fa0, fa1, fa0\n"),
                    BinaryOp::Fdiv => self.body.push_str("  fdiv.s fa0, fa1, fa0\n"),
                    _ => unreachable!(),
                }
                self.body.push_str("  fmv.x.w a0, fa0\n");
            }
            BinaryOp::And | BinaryOp::Or => {
                self.load_value(lhs);
                self.body.push_str("  snez a0, a0\n");
                self.push_x0();
                self.load_value(rhs);
                self.body.push_str("  snez a0, a0\n");
                self.pop_x1();
                if op == BinaryOp::And {
                    self.body.push_str("  and a0, a1, a0\n");
                } else {
                    self.body.push_str("  or a0, a1, a0\n");
                }
            }
            _ => {
                self.load_value(lhs);
                self.push_x0();
                self.load_value(rhs);
                self.pop_x1();
                match op {
                    BinaryOp::Iadd => self.body.push_str("  addw a0, a1, a0\n"),
                    BinaryOp::Isub => self.body.push_str("  subw a0, a1, a0\n"),
                    BinaryOp::Imul => self.body.push_str("  mulw a0, a1, a0\n"),
                    BinaryOp::Idiv => self.body.push_str("  divw a0, a1, a0\n"),
                    BinaryOp::Imod => self.body.push_str("  remw a0, a1, a0\n"),
                    _ => unreachable!(),
                }
            }
        }
    }

    fn emit_binary_imm(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) -> bool {
        match op {
            BinaryOp::Iadd => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_i12(*imm)) {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  addiw a0, a0, {}\n", imm));
                    return true;
                }
                if let Some(imm) = const_i32(self.func, lhs).filter(|imm| fits_i12(*imm)) {
                    self.load_value(rhs);
                    self.body.push_str(&format!("  addiw a0, a0, {}\n", imm));
                    return true;
                }
            }
            BinaryOp::Isub => {
                if let Some(imm) = const_i32(self.func, rhs)
                    .and_then(|imm| imm.checked_neg())
                    .filter(|imm| fits_i12(*imm))
                {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  addiw a0, a0, {}\n", imm));
                    return true;
                }
            }
            BinaryOp::Imul => {
                if let Some(shift) = pow2_shift(self.func, rhs) {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  slliw a0, a0, {}\n", shift));
                    return true;
                }
                if let Some(shift) = pow2_shift(self.func, lhs) {
                    self.load_value(rhs);
                    self.body.push_str(&format!("  slliw a0, a0, {}\n", shift));
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    fn emit_icmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.load_value(lhs);
        self.push_x0();
        self.load_value(rhs);
        self.pop_x1();
        match op {
            CmpOp::Lt => self.body.push_str("  slt a0, a1, a0\n"),
            CmpOp::Gt => self.body.push_str("  slt a0, a0, a1\n"),
            CmpOp::Le => self.body.push_str("  slt a0, a0, a1\n  seqz a0, a0\n"),
            CmpOp::Ge => self.body.push_str("  slt a0, a1, a0\n  seqz a0, a0\n"),
            CmpOp::Eq => self.body.push_str("  sub a0, a1, a0\n  seqz a0, a0\n"),
            CmpOp::Ne => self.body.push_str("  sub a0, a1, a0\n  snez a0, a0\n"),
        }
    }

    fn emit_fcmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.load_float_value(lhs, "fa0");
        self.push_s0();
        self.load_float_value(rhs, "fa0");
        self.pop_s1();
        match op {
            CmpOp::Lt => self.body.push_str("  flt.s a0, fa1, fa0\n"),
            CmpOp::Gt => self.body.push_str("  flt.s a0, fa0, fa1\n"),
            CmpOp::Le => self.body.push_str("  fle.s a0, fa1, fa0\n"),
            CmpOp::Ge => self.body.push_str("  fle.s a0, fa0, fa1\n"),
            CmpOp::Eq => self.body.push_str("  feq.s a0, fa1, fa0\n"),
            CmpOp::Ne => self.body.push_str("  feq.s a0, fa1, fa0\n  seqz a0, a0\n"),
        }
    }

    fn emit_cast(&mut self, op: CastOp, value: ValueId) {
        match op {
            CastOp::I32ToF32 | CastOp::BoolToI32 | CastOp::I32ToBool => {
                self.load_value(value);
                match op {
                    CastOp::I32ToF32 => self
                        .body
                        .push_str("  fcvt.s.w fa0, a0\n  fmv.x.w a0, fa0\n"),
                    CastOp::BoolToI32 => {}
                    CastOp::I32ToBool => self.body.push_str("  snez a0, a0\n"),
                    _ => unreachable!(),
                }
            }
            CastOp::F32ToI32 => {
                self.load_float_value(value, "fa0");
                self.body.push_str("  fcvt.w.s a0, fa0, rtz\n");
            }
            CastOp::F32ToBool => {
                self.load_float_value(value, "fa0");
                self.body
                    .push_str("  fmv.w.x ft0, zero\n  feq.s a0, fa0, ft0\n  seqz a0, a0\n");
            }
        }
    }
}

fn const_i32(func: &crate::ir::Function, value: ValueId) -> Option<i32> {
    match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        ValueKind::Const(Const::Bool(value)) => Some(*value as i32),
        _ => None,
    }
}

fn pow2_shift(func: &crate::ir::Function, value: ValueId) -> Option<u32> {
    let imm = const_i32(func, value)?;
    (imm > 0 && (imm & (imm - 1)) == 0).then_some(imm.trailing_zeros())
}

fn fits_i12(value: i32) -> bool {
    (-2048..=2047).contains(&value)
}
