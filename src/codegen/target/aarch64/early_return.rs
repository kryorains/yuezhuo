use super::imm::mov_w_imm;
use super::AArch64IrFuncEmitter;
use crate::codegen::common::{
    assign_arg_locations, EarlyReturnResult, EntryEarlyReturn, IrArgLocation,
};
use crate::ir::{BinaryOp, CmpOp, Const, InstKind, UnaryOp, ValueId, ValueKind};

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    pub(super) fn pre_prologue_early_return(&self, plan: &EntryEarlyReturn) -> Option<String> {
        let frame_label = format!(".L_frame_{}", self.func.name);
        let mut out = String::new();
        self.emit_guard_branch(plan.condition, !plan.fast_when_true, &frame_label, &mut out)?;
        self.emit_fast_result(plan.result, &mut out)?;
        out.push_str("  ret\n");
        out.push_str(&format!("{}:\n", frame_label));
        Some(out)
    }

    fn emit_guard_branch(
        &self,
        value: ValueId,
        branch_when: bool,
        target: &str,
        out: &mut String,
    ) -> Option<()> {
        match self.func.value(value).kind {
            ValueKind::Const(Const::Bool(value)) => {
                if value == branch_when {
                    out.push_str(&format!("  b {}\n", target));
                }
                Some(())
            }
            ValueKind::Param => {
                let reg = self.preframe_param_reg(value)?;
                let branch = if branch_when { "cbnz" } else { "cbz" };
                out.push_str(&format!("  {} {}, {}\n", branch, reg, target));
                Some(())
            }
            ValueKind::Inst(owner, inst_idx) => {
                let inst = self.func.blocks.get(owner.0)?.insts.get(inst_idx)?;
                if inst.result != Some(value) {
                    return None;
                }
                match inst.kind {
                    InstKind::Unary {
                        op: UnaryOp::Not,
                        value,
                    } => self.emit_guard_branch(value, !branch_when, target, out),
                    InstKind::Binary {
                        op: BinaryOp::And,
                        lhs,
                        rhs,
                    } if !branch_when => {
                        self.emit_guard_branch(lhs, false, target, out)?;
                        self.emit_guard_branch(rhs, false, target, out)
                    }
                    InstKind::Binary {
                        op: BinaryOp::Or,
                        lhs,
                        rhs,
                    } if branch_when => {
                        self.emit_guard_branch(lhs, true, target, out)?;
                        self.emit_guard_branch(rhs, true, target, out)
                    }
                    InstKind::Icmp { op, lhs, rhs } => {
                        self.emit_preframe_icmp_branch(op, lhs, rhs, branch_when, target, out)
                    }
                    _ => None,
                }
            }
            ValueKind::Const(_) | ValueKind::Global(_) => None,
        }
    }

    fn emit_preframe_icmp_branch(
        &self,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
        branch_when: bool,
        target: &str,
        out: &mut String,
    ) -> Option<()> {
        let lhs = self.preframe_operand(lhs)?;
        let rhs = self.preframe_operand(rhs)?;
        if let (PreframeOperand::Imm(lhs), PreframeOperand::Imm(rhs)) = (&lhs, &rhs) {
            if eval_icmp(op, *lhs, *rhs) == branch_when {
                out.push_str(&format!("  b {}\n", target));
            }
            return Some(());
        }

        if let Some((reg, zero_op)) = zero_comparison(op, &lhs, &rhs) {
            if emit_zero_branch(out, reg, zero_op, branch_when, target) {
                return Some(());
            }
        }

        let lhs_reg = self.materialize_operand(lhs, "w9", out);
        let rhs_reg = self.materialize_operand(rhs, "w10", out);
        out.push_str(&format!("  cmp {}, {}\n", lhs_reg, rhs_reg));
        out.push_str(&format!("  {} {}\n", icmp_branch(op, branch_when), target));
        Some(())
    }

    fn emit_fast_result(&self, result: EarlyReturnResult, out: &mut String) -> Option<()> {
        match result {
            EarlyReturnResult::Direct(value) => {
                match self.preframe_operand(value)? {
                    PreframeOperand::Reg(reg) if reg != "w0" => {
                        out.push_str(&format!("  mov w0, {}\n", reg));
                    }
                    PreframeOperand::Reg(_) => {}
                    PreframeOperand::Imm(value) => out.push_str(&mov_w_imm("w0", value)),
                }
                Some(())
            }
            EarlyReturnResult::Binary { op, lhs, rhs } => {
                let lhs = self.preframe_operand(lhs)?;
                let rhs = self.preframe_operand(rhs)?;
                let lhs = self.materialize_operand(lhs, "w9", out);
                let rhs = self.materialize_operand(rhs, "w10", out);
                let instruction = match op {
                    BinaryOp::Iadd => "add",
                    BinaryOp::Isub => "sub",
                    BinaryOp::Imul => "mul",
                    BinaryOp::Iand => "and",
                    BinaryOp::Ior => "orr",
                    BinaryOp::Ixor => "eor",
                    BinaryOp::Ishl => "lsl",
                    BinaryOp::Iashr => "asr",
                    _ => return None,
                };
                out.push_str(&format!("  {} w0, {}, {}\n", instruction, lhs, rhs));
                Some(())
            }
        }
    }

    fn preframe_operand(&self, value: ValueId) -> Option<PreframeOperand> {
        match self.func.value(value).kind {
            ValueKind::Param => Some(PreframeOperand::Reg(self.preframe_param_reg(value)?)),
            ValueKind::Const(Const::Int(value)) => Some(PreframeOperand::Imm(value)),
            ValueKind::Const(Const::Bool(value)) => Some(PreframeOperand::Imm(i32::from(value))),
            ValueKind::Const(_) | ValueKind::Global(_) | ValueKind::Inst(_, _) => None,
        }
    }

    fn preframe_param_reg(&self, value: ValueId) -> Option<String> {
        let param_idx = self.func.params.iter().position(|param| *param == value)?;
        let sig = self.parent.ctx.funcs.get(&self.func.name)?;
        let locations = assign_arg_locations(&sig.params, 8, 8);
        match locations.get(param_idx)? {
            IrArgLocation::IntReg(reg) => Some(format!("w{}", reg)),
            IrArgLocation::FloatReg(_) | IrArgLocation::Stack => None,
        }
    }

    fn materialize_operand(
        &self,
        operand: PreframeOperand,
        scratch: &'static str,
        out: &mut String,
    ) -> String {
        match operand {
            PreframeOperand::Reg(reg) => reg,
            PreframeOperand::Imm(value) => {
                out.push_str(&mov_w_imm(scratch, value));
                scratch.to_string()
            }
        }
    }
}

#[derive(Clone)]
enum PreframeOperand {
    Reg(String),
    Imm(i32),
}

fn zero_comparison<'a>(
    op: CmpOp,
    lhs: &'a PreframeOperand,
    rhs: &'a PreframeOperand,
) -> Option<(&'a str, CmpOp)> {
    match (lhs, rhs) {
        (PreframeOperand::Reg(reg), PreframeOperand::Imm(0)) => Some((reg, op)),
        (PreframeOperand::Imm(0), PreframeOperand::Reg(reg)) => Some((reg, swap_cmp(op))),
        _ => None,
    }
}

fn emit_zero_branch(
    out: &mut String,
    reg: &str,
    op: CmpOp,
    branch_when: bool,
    target: &str,
) -> bool {
    let instruction = match (op, branch_when) {
        (CmpOp::Eq, true) | (CmpOp::Ne, false) => "cbz",
        (CmpOp::Eq, false) | (CmpOp::Ne, true) => "cbnz",
        (CmpOp::Lt, true) | (CmpOp::Ge, false) => "tbnz",
        (CmpOp::Lt, false) | (CmpOp::Ge, true) => "tbz",
        (CmpOp::Le | CmpOp::Gt, _) => return false,
    };
    if instruction == "tbnz" || instruction == "tbz" {
        out.push_str(&format!("  {} {}, #31, {}\n", instruction, reg, target));
    } else {
        out.push_str(&format!("  {} {}, {}\n", instruction, reg, target));
    }
    true
}

fn swap_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}

fn icmp_branch(op: CmpOp, branch_when: bool) -> &'static str {
    match (op, branch_when) {
        (CmpOp::Eq, true) | (CmpOp::Ne, false) => "b.eq",
        (CmpOp::Eq, false) | (CmpOp::Ne, true) => "b.ne",
        (CmpOp::Lt, true) | (CmpOp::Ge, false) => "b.lt",
        (CmpOp::Lt, false) | (CmpOp::Ge, true) => "b.ge",
        (CmpOp::Le, true) | (CmpOp::Gt, false) => "b.le",
        (CmpOp::Le, false) | (CmpOp::Gt, true) => "b.gt",
    }
}

fn eval_icmp(op: CmpOp, lhs: i32, rhs: i32) -> bool {
    match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
    }
}
