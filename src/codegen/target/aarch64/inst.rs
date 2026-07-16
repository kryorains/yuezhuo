use super::AArch64IrFuncEmitter;
use crate::ir::{
    BinaryOp, BlockId, CastOp, CmpOp, Const, Inst, InstKind, Terminator, Type, UnaryOp, ValueId,
    ValueKind,
};

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_inst(&mut self, inst: &Inst) {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { incomings } => {
                let result = inst.result.unwrap();
                for (pred, value) in incomings {
                    self.body
                        .push_str(&format!("// phi {} from {} = {}\n", result, pred, value));
                }
            }
            InstKind::Alloca { ty } => {
                let result = inst.result.unwrap();
                self.frame_addr("x0", self.object_offset(result, ty));
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
                self.emit_gep(result, *base, indices);
                self.store_result(result);
            }
            InstKind::Call { name, args } => {
                let ret = self.emit_call(name, args);
                if let Some(result) = inst.result {
                    if ret == Type::F32 {
                        self.store_frame_s("s0", self.layout.offset(result));
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
                        self.body.push_str("  fmov s0, w0\n");
                    }
                }
                self.body.push_str(&format!("  b {}\n", self.return_label));
            }
            Terminator::Jump(target) => {
                self.emit_phi_copies(block_idx, target.0);
                self.body
                    .push_str(&format!("  b {}\n", self.block_label(target.0)));
            }
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                let else_edge = self.parent.ctx.fresh_label("else_edge");
                if let Some((op, lhs, rhs)) = self.direct_branch_icmp(*cond) {
                    self.emit_int_compare(lhs, rhs);
                    self.body
                        .push_str(&format!("  b.{} {}\n", inverse_cmp_cc(op), else_edge));
                } else {
                    self.load_value(*cond);
                    self.body
                        .push_str(&format!("  cmp w0, #0\n  beq {}\n", else_edge));
                }
                self.emit_phi_copies(block_idx, then_target.0);
                self.body.push_str(&format!(
                    "  b {}\n{}:\n",
                    self.block_label(then_target.0),
                    else_edge
                ));
                self.emit_phi_copies(block_idx, else_target.0);
                self.body
                    .push_str(&format!("  b {}\n", self.block_label(else_target.0)));
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
            self.body.push_str("  mov x0, x1\n");
            self.store_result(*result);
        }
    }

    fn emit_unary(&mut self, op: UnaryOp, value: ValueId) {
        self.load_value(value);
        match op {
            UnaryOp::Ineg => self.body.push_str("  neg w0, w0\n"),
            UnaryOp::Fneg => self
                .body
                .push_str("  fmov s0, w0\n  fneg s0, s0\n  fmov w0, s0\n"),
            UnaryOp::Not => self.body.push_str("  cmp w0, #0\n  cset w0, eq\n"),
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) {
        if self.emit_binary_imm(op, lhs, rhs) {
            return;
        }

        match op {
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => {
                self.load_float_value(lhs, "s0");
                self.push_s0();
                self.load_float_value(rhs, "s0");
                self.pop_s1();
                match op {
                    BinaryOp::Fadd => self.body.push_str("  fadd s0, s1, s0\n"),
                    BinaryOp::Fsub => self.body.push_str("  fsub s0, s1, s0\n"),
                    BinaryOp::Fmul => self.body.push_str("  fmul s0, s1, s0\n"),
                    BinaryOp::Fdiv => self.body.push_str("  fdiv s0, s1, s0\n"),
                    _ => unreachable!(),
                }
                self.body.push_str("  fmov w0, s0\n");
            }
            BinaryOp::And | BinaryOp::Or => {
                self.load_value(lhs);
                self.body.push_str("  cmp w0, #0\n  cset w0, ne\n");
                self.push_x0();
                self.load_value(rhs);
                self.body.push_str("  cmp w0, #0\n  cset w0, ne\n");
                self.pop_x1();
                if op == BinaryOp::And {
                    self.body.push_str("  and w0, w1, w0\n");
                } else {
                    self.body.push_str("  orr w0, w1, w0\n");
                }
            }
            _ => {
                self.load_value(lhs);
                self.push_x0();
                self.load_value(rhs);
                self.pop_x1();
                match op {
                    BinaryOp::Iadd => self.body.push_str("  add w0, w1, w0\n"),
                    BinaryOp::Isub => self.body.push_str("  sub w0, w1, w0\n"),
                    BinaryOp::Imul => self.body.push_str("  mul w0, w1, w0\n"),
                    BinaryOp::Idiv | BinaryOp::Imod => {
                        self.body.push_str("  sdiv w2, w1, w0\n");
                        if op == BinaryOp::Imod {
                            self.body.push_str("  msub w0, w2, w0, w1\n");
                        } else {
                            self.body.push_str("  mov w0, w2\n");
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn emit_binary_imm(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) -> bool {
        match op {
            BinaryOp::Iadd => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_addsub_imm(*imm)) {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  add w0, w0, #{}\n", imm));
                    return true;
                }
                if let Some(imm) = const_i32(self.func, lhs).filter(|imm| fits_addsub_imm(*imm)) {
                    self.load_value(rhs);
                    self.body.push_str(&format!("  add w0, w0, #{}\n", imm));
                    return true;
                }
            }
            BinaryOp::Isub => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_addsub_imm(*imm)) {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  sub w0, w0, #{}\n", imm));
                    return true;
                }
            }
            BinaryOp::Imul => {
                if let Some(imm) = const_i32(self.func, rhs) {
                    if self.emit_mul_imm(lhs, imm) {
                        return true;
                    }
                }
                if let Some(imm) = const_i32(self.func, lhs) {
                    if self.emit_mul_imm(rhs, imm) {
                        return true;
                    }
                }
            }
            BinaryOp::Idiv | BinaryOp::Imod => {
                if let Some(shift) = pow2_shift(self.func, rhs) {
                    self.emit_signed_divmod_pow2(lhs, shift, op == BinaryOp::Imod);
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    fn emit_mul_imm(&mut self, value: ValueId, imm: i32) -> bool {
        if let Some(shift) = positive_pow2_shift(imm) {
            self.load_value(value);
            self.body.push_str(&format!("  lsl w0, w0, #{}\n", shift));
            return true;
        }

        if let Some(shift) = imm.checked_sub(1).and_then(positive_pow2_shift) {
            self.load_value(value);
            self.body
                .push_str(&format!("  add w0, w0, w0, lsl #{}\n", shift));
            return true;
        }

        if let Some(shift) = imm.checked_add(1).and_then(positive_pow2_shift) {
            self.load_value(value);
            self.body
                .push_str(&format!("  lsl w1, w0, #{}\n  sub w0, w1, w0\n", shift));
            return true;
        }

        false
    }

    fn emit_signed_divmod_pow2(&mut self, value: ValueId, shift: u32, remainder: bool) {
        self.load_value(value);
        if shift == 0 {
            if remainder {
                self.body.push_str("  mov w0, wzr\n");
            }
            return;
        }

        if remainder {
            self.body.push_str("  mov w2, w0\n");
        }
        let mask = (1u32 << shift) - 1;
        self.body.push_str(&format!(
            "  asr w1, w0, #31\n  and w1, w1, #{}\n  add w0, w0, w1\n  asr w0, w0, #{}\n",
            mask, shift
        ));
        if remainder {
            self.body
                .push_str(&format!("  sub w0, w2, w0, lsl #{}\n", shift));
        }
    }

    fn direct_branch_icmp(&self, value: ValueId) -> Option<(CmpOp, ValueId, ValueId)> {
        if value_use_count(self.func, value) != 1 {
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

    fn emit_int_compare(&mut self, lhs: ValueId, rhs: ValueId) {
        if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_addsub_imm(*imm)) {
            self.load_value(lhs);
            self.body.push_str(&format!("  cmp w0, #{}\n", imm));
            return;
        }

        self.load_value(lhs);
        self.push_x0();
        self.load_value(rhs);
        self.pop_x1();
        self.body.push_str("  cmp w1, w0\n");
    }

    fn emit_icmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.emit_int_compare(lhs, rhs);
        let cc = match op {
            CmpOp::Lt => "lt",
            CmpOp::Gt => "gt",
            CmpOp::Le => "le",
            CmpOp::Ge => "ge",
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
        };
        self.body.push_str(&format!("  cset w0, {}\n", cc));
    }

    fn emit_fcmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.load_float_value(lhs, "s0");
        self.push_s0();
        self.load_float_value(rhs, "s0");
        self.pop_s1();
        self.body.push_str("  fcmp s1, s0\n");
        let cc = match op {
            CmpOp::Lt => "lt",
            CmpOp::Gt => "gt",
            CmpOp::Le => "le",
            CmpOp::Ge => "ge",
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
        };
        self.body.push_str(&format!("  cset w0, {}\n", cc));
    }

    fn emit_cast(&mut self, op: CastOp, value: ValueId) {
        match op {
            CastOp::I32ToF32 | CastOp::BoolToI32 | CastOp::I32ToBool => {
                self.load_value(value);
                match op {
                    CastOp::I32ToF32 => self.body.push_str("  scvtf s0, w0\n  fmov w0, s0\n"),
                    CastOp::BoolToI32 => {}
                    CastOp::I32ToBool => self.body.push_str("  cmp w0, #0\n  cset w0, ne\n"),
                    _ => unreachable!(),
                }
            }
            CastOp::F32ToI32 => {
                self.load_float_value(value, "s0");
                self.body.push_str("  fcvtzs w0, s0\n");
            }
            CastOp::F32ToBool => {
                self.load_float_value(value, "s0");
                self.body.push_str("  fcmp s0, #0.0\n  cset w0, ne\n");
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
    positive_pow2_shift(const_i32(func, value)?)
}

fn positive_pow2_shift(value: i32) -> Option<u32> {
    (value > 0 && (value & (value - 1)) == 0).then_some(value.trailing_zeros())
}

fn fits_addsub_imm(value: i32) -> bool {
    (0..=4095).contains(&value)
}

fn inverse_cmp_cc(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "ge",
        CmpOp::Gt => "le",
        CmpOp::Le => "gt",
        CmpOp::Ge => "lt",
        CmpOp::Eq => "ne",
        CmpOp::Ne => "eq",
    }
}

fn value_use_count(func: &crate::ir::Function, value: ValueId) -> usize {
    let mut count = 0;
    for block in &func.blocks {
        for inst in &block.insts {
            count += inst_use_count(&inst.kind, value);
        }
        if let Some(terminator) = &block.terminator {
            count += terminator_use_count(terminator, value);
        }
    }
    count
}

fn inst_use_count(kind: &InstKind, value: ValueId) -> usize {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => 0,
        InstKind::Phi { incomings } => incomings
            .iter()
            .filter(|(_, incoming)| *incoming == value)
            .count(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => (*ptr == value) as usize,
        InstKind::Store { ptr, value: stored } => {
            (*ptr == value) as usize + (*stored == value) as usize
        }
        InstKind::Unary { value: operand, .. } | InstKind::Cast { value: operand, .. } => {
            (*operand == value) as usize
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => (*lhs == value) as usize + (*rhs == value) as usize,
        InstKind::Gep { base, indices } => {
            (*base == value) as usize + indices.iter().filter(|index| **index == value).count()
        }
        InstKind::Call { args, .. } => args.iter().filter(|arg| **arg == value).count(),
    }
}

fn terminator_use_count(terminator: &Terminator, value: ValueId) -> usize {
    match terminator {
        Terminator::Return(Some(returned)) => (*returned == value) as usize,
        Terminator::Branch { cond, .. } => (*cond == value) as usize,
        Terminator::Return(None) | Terminator::Jump(_) => 0,
    }
}
