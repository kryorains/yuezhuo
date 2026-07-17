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
                if !self.emit_assigned_load(result, *ptr) {
                    self.load_value(*ptr);
                    let ty = self.func.value(result).ty.clone();
                    self.load_indirect(&ty);
                    self.store_result(result);
                }
            }
            InstKind::Store { ptr, value } => {
                if !self.emit_assigned_store(*ptr, *value) {
                    self.load_value_into(*ptr, "a1");
                    self.load_value(*value);
                    let ty = self.func.value(*value).ty.clone();
                    self.store_indirect(&ty);
                }
            }
            InstKind::MemZero { ptr, bytes } => self.emit_memzero(*ptr, *bytes),
            InstKind::Unary { op, value } => {
                let result = inst.result.unwrap();
                self.emit_unary(*op, *value);
                self.store_result(result);
            }
            InstKind::Binary { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                if !self.emit_assigned_binary(result, *op, *lhs, *rhs) {
                    self.emit_binary(*op, *lhs, *rhs);
                    self.store_result(result);
                }
            }
            InstKind::Icmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                if self.direct_branch_icmp(result).is_none() {
                    self.emit_icmp(*op, *lhs, *rhs);
                    self.store_result(result);
                }
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
                if !self.emit_assigned_gep(result, *base, indices) {
                    self.emit_gep(result, *base, indices);
                    self.store_result(result);
                }
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
                if !self.edge_has_phi_copy(block_idx, then_target.0)
                    && !self.edge_has_phi_copy(block_idx, else_target.0)
                {
                    self.emit_branch_if_false(*cond, self.block_label(else_target.0));
                    if then_target.0 != block_idx + 1 {
                        self.body
                            .push_str(&format!("  j {}\n", self.block_label(then_target.0)));
                    }
                    return;
                }

                let else_edge = self.parent.ctx.fresh_label("else_edge");
                self.emit_branch_if_false(*cond, else_edge.clone());
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
                    .filter(|(result, value)| {
                        self.regalloc.reg(*result) != self.regalloc.reg(*value)
                            || self.regalloc.reg(*result).is_none()
                    })
            })
            .collect::<Vec<_>>();

        if let [(result, value)] = copies.as_slice() {
            self.load_value(*value);
            self.store_result(*result);
            return;
        }

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

    fn edge_has_phi_copy(&self, pred_idx: usize, target_idx: usize) -> bool {
        self.func
            .block(BlockId(target_idx))
            .insts
            .iter()
            .any(|inst| {
                matches!(
                    &inst.kind,
                    InstKind::Phi { incomings }
                        if incomings.iter().any(|(pred, _)| pred.0 == pred_idx)
                )
            })
    }

    fn emit_branch_if_false(&mut self, cond: ValueId, target: String) {
        if let Some((op, lhs, rhs)) = self.direct_branch_icmp(cond) {
            if self.emit_zero_icmp_branch_if_false(op, lhs, rhs, &target) {
                return;
            }
            self.load_value_into(lhs, "a1");
            self.load_value(rhs);
            let branch = match op {
                CmpOp::Lt => "bge a1, a0",
                CmpOp::Gt => "bge a0, a1",
                CmpOp::Le => "blt a0, a1",
                CmpOp::Ge => "blt a1, a0",
                CmpOp::Eq => "bne a1, a0",
                CmpOp::Ne => "beq a1, a0",
            };
            self.body.push_str(&format!("  {}, {}\n", branch, target));
        } else {
            self.load_value(cond);
            self.body.push_str(&format!("  beqz a0, {}\n", target));
        }
    }

    fn emit_zero_icmp_branch_if_false(
        &mut self,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
        target: &str,
    ) -> bool {
        let (value, branch) = if const_i32(self.func, rhs) == Some(0) {
            let branch = match op {
                CmpOp::Lt => "bgez",
                CmpOp::Gt => "blez",
                CmpOp::Le => "bgtz",
                CmpOp::Ge => "bltz",
                CmpOp::Eq => "bnez",
                CmpOp::Ne => "beqz",
            };
            (lhs, branch)
        } else if const_i32(self.func, lhs) == Some(0) {
            let branch = match op {
                CmpOp::Lt => "blez",
                CmpOp::Gt => "bgez",
                CmpOp::Le => "bltz",
                CmpOp::Ge => "bgtz",
                CmpOp::Eq => "bnez",
                CmpOp::Ne => "beqz",
            };
            (rhs, branch)
        } else {
            return false;
        };
        let reg = self.load_or_assigned(value, "a0");
        self.body
            .push_str(&format!("  {} {}, {}\n", branch, reg, target));
        true
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

    fn emit_assigned_binary(
        &mut self,
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };

        match op {
            BinaryOp::Iadd => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_i12(*imm)) {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body
                        .push_str(&format!("  addiw {}, {}, {}\n", destination, source, imm));
                    return true;
                }
                if let Some(imm) = const_i32(self.func, lhs).filter(|imm| fits_i12(*imm)) {
                    let source = self.load_or_assigned(rhs, "a0");
                    self.body
                        .push_str(&format!("  addiw {}, {}, {}\n", destination, source, imm));
                    return true;
                }
            }
            BinaryOp::Isub => {
                if let Some(imm) = const_i32(self.func, rhs)
                    .and_then(i32::checked_neg)
                    .filter(|imm| fits_i12(*imm))
                {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body
                        .push_str(&format!("  addiw {}, {}, {}\n", destination, source, imm));
                    return true;
                }
            }
            BinaryOp::Imul => {
                if let Some(shift) = pow2_shift(self.func, rhs) {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body
                        .push_str(&format!("  slliw {}, {}, {}\n", destination, source, shift));
                    return true;
                }
                if let Some(shift) = pow2_shift(self.func, lhs) {
                    let source = self.load_or_assigned(rhs, "a0");
                    self.body
                        .push_str(&format!("  slliw {}, {}, {}\n", destination, source, shift));
                    return true;
                }
            }
            _ => {}
        }

        if matches!(op, BinaryOp::Idiv | BinaryOp::Imod) && const_i32(self.func, rhs).is_some() {
            return false;
        }

        let lhs_reg = self.load_or_assigned(lhs, "a1");
        let rhs_reg = self.load_or_assigned(rhs, "a0");
        match op {
            BinaryOp::Iadd => self.body.push_str(&format!(
                "  addw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Isub => self.body.push_str(&format!(
                "  subw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Imul => self.body.push_str(&format!(
                "  mulw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Idiv => self.body.push_str(&format!(
                "  divw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Imod => self.body.push_str(&format!(
                "  remw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::And | BinaryOp::Or => {
                self.body
                    .push_str(&format!("  snez a1, {}\n  snez a0, {}\n", lhs_reg, rhs_reg));
                let instruction = if op == BinaryOp::And { "and" } else { "or" };
                self.body
                    .push_str(&format!("  {} {}, a1, a0\n", instruction, destination));
            }
            BinaryOp::Iand => self.body.push_str(&format!(
                "  and {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Ior => self
                .body
                .push_str(&format!("  or {}, {}, {}\n", destination, lhs_reg, rhs_reg)),
            BinaryOp::Ixor => self.body.push_str(&format!(
                "  xor {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Ishl => self.body.push_str(&format!(
                "  sllw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Iashr => self.body.push_str(&format!(
                "  sraw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => return false,
        }
        true
    }

    fn load_or_assigned(&mut self, value: ValueId, scratch: &'static str) -> &'static str {
        if let Some(reg) = self.assigned_reg(value) {
            reg
        } else {
            self.load_value_into(value, scratch);
            scratch
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) {
        if self.emit_binary_imm(op, lhs, rhs) {
            return;
        }

        match op {
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => {
                self.load_float_value(lhs, "fa1");
                self.load_float_value(rhs, "fa0");
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
                self.load_value_into(lhs, "a1");
                self.body.push_str("  snez a1, a1\n");
                self.load_value(rhs);
                self.body.push_str("  snez a0, a0\n");
                if op == BinaryOp::And {
                    self.body.push_str("  and a0, a1, a0\n");
                } else {
                    self.body.push_str("  or a0, a1, a0\n");
                }
            }
            _ => {
                self.load_value_into(lhs, "a1");
                self.load_value(rhs);
                match op {
                    BinaryOp::Iadd => self.body.push_str("  addw a0, a1, a0\n"),
                    BinaryOp::Isub => self.body.push_str("  subw a0, a1, a0\n"),
                    BinaryOp::Imul => self.body.push_str("  mulw a0, a1, a0\n"),
                    BinaryOp::Idiv => self.body.push_str("  divw a0, a1, a0\n"),
                    BinaryOp::Imod => self.body.push_str("  remw a0, a1, a0\n"),
                    BinaryOp::Iand => self.body.push_str("  and a0, a1, a0\n"),
                    BinaryOp::Ior => self.body.push_str("  or a0, a1, a0\n"),
                    BinaryOp::Ixor => self.body.push_str("  xor a0, a1, a0\n"),
                    BinaryOp::Ishl => self.body.push_str("  sllw a0, a1, a0\n"),
                    BinaryOp::Iashr => self.body.push_str("  sraw a0, a1, a0\n"),
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
            BinaryOp::Idiv | BinaryOp::Imod => {
                if let Some(divisor) = const_i32(self.func, rhs) {
                    self.emit_signed_divmod_const(lhs, divisor, op == BinaryOp::Imod);
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    fn emit_signed_divmod_const(&mut self, value: ValueId, divisor: i32, remainder: bool) {
        if divisor == 0 {
            if remainder {
                self.load_value(value);
            } else {
                self.body.push_str("  li a0, -1\n");
            }
            return;
        }

        if divisor == 1 || divisor == -1 {
            if remainder {
                self.body.push_str("  li a0, 0\n");
            } else {
                self.load_value(value);
                if divisor < 0 {
                    self.body.push_str("  negw a0, a0\n");
                }
            }
            return;
        }

        self.load_value(value);
        if remainder {
            self.body.push_str("  mv t2, a0\n");
        }

        let abs_divisor = divisor.unsigned_abs();
        // Bias negative dividends before shifting so the quotient truncates toward zero.
        if abs_divisor.is_power_of_two() {
            let shift = abs_divisor.trailing_zeros();
            self.body.push_str(&format!(
                "  sraiw t0, a0, 31\n  srliw t0, t0, {}\n  addw t0, a0, t0\n  sraiw t0, t0, {}\n",
                32 - shift,
                shift
            ));
        } else {
            // Widen to 64 bits: RV64 mulh is a 64x64 high multiply, while this
            // lowering needs the high half of a signed 32x32 product.
            let magic = signed_magic_positive(abs_divisor);
            self.body.push_str(&format!(
                "  li t0, {}\n  mul t0, a0, t0\n  srai t0, t0, 32\n",
                magic.multiplier
            ));
            if magic.add_dividend {
                self.body.push_str("  addw t0, t0, a0\n");
            }
            if magic.shift != 0 {
                self.body
                    .push_str(&format!("  sraiw t0, t0, {}\n", magic.shift));
            }
            self.body
                .push_str("  srliw t1, t0, 31\n  addw t0, t0, t1\n");
        }

        if divisor < 0 {
            self.body.push_str("  negw t0, t0\n");
        }
        if remainder {
            if divisor > 0 && ((divisor as u32) + 1).is_power_of_two() {
                self.body.push_str(&format!(
                    "  slliw t1, t0, {}\n  subw t0, t1, t0\n  subw a0, t2, t0\n",
                    ((divisor as u32) + 1).trailing_zeros()
                ));
            } else {
                self.body.push_str(&format!(
                    "  li t1, {}\n  mulw t0, t0, t1\n  subw a0, t2, t0\n",
                    divisor
                ));
            }
        } else {
            self.body.push_str("  mv a0, t0\n");
        }
    }

    fn direct_branch_icmp(&self, value: ValueId) -> Option<(CmpOp, ValueId, ValueId)> {
        if self.value_use_counts[value.0] != 1 {
            return None;
        }
        let ValueKind::Inst(block, inst_idx) = self.func.value(value).kind else {
            return None;
        };
        let Terminator::Branch { cond, .. } = self.func.block(block).terminator.as_ref()? else {
            return None;
        };
        if *cond != value {
            return None;
        }
        match self.func.block(block).insts.get(inst_idx)?.kind {
            InstKind::Icmp { op, lhs, rhs } => Some((op, lhs, rhs)),
            _ => None,
        }
    }

    fn emit_icmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.load_value_into(lhs, "a1");
        self.load_value(rhs);
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
        self.load_float_value(lhs, "fa1");
        self.load_float_value(rhs, "fa0");
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

#[derive(Debug, Clone, Copy)]
struct SignedMagic {
    multiplier: i32,
    shift: u32,
    add_dividend: bool,
}

// Hacker's Delight, chapter 10: magic multiplier for signed division by a
// positive, non-power-of-two 32-bit constant.
fn signed_magic_positive(divisor: u32) -> SignedMagic {
    debug_assert!((2..=i32::MAX as u32).contains(&divisor));
    debug_assert!(!divisor.is_power_of_two());

    let divisor = u64::from(divisor);
    let two31 = 1u64 << 31;
    let anc = two31 - 1 - two31 % divisor;
    let mut p = 31u32;
    let (mut q1, mut r1) = (two31 / anc, two31 % anc);
    let (mut q2, mut r2) = (two31 / divisor, two31 % divisor);

    loop {
        p += 1;
        q1 <<= 1;
        r1 <<= 1;
        if r1 >= anc {
            q1 += 1;
            r1 -= anc;
        }
        q2 <<= 1;
        r2 <<= 1;
        if r2 >= divisor {
            q2 += 1;
            r2 -= divisor;
        }

        let delta = divisor - r2;
        if q1 > delta || (q1 == delta && r1 != 0) {
            break;
        }
    }

    let multiplier = (q2 + 1) as u32 as i32;
    SignedMagic {
        multiplier,
        shift: p - 32,
        add_dividend: multiplier < 0,
    }
}
