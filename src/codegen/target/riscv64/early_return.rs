use super::Riscv64IrFuncEmitter;
use crate::codegen::common::{EarlyReturnResult, EntryEarlyReturn, IrArgLocation};
use crate::ir::{BinaryOp, CmpOp, Const, InstKind, UnaryOp, ValueId, ValueKind};

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
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
                    out.push_str(&format!("  j {}\n", target));
                }
                Some(())
            }
            ValueKind::Param => {
                let reg = self.preframe_param_reg(value)?;
                let branch = if branch_when { "bnez" } else { "beqz" };
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
                out.push_str(&format!("  j {}\n", target));
            }
            return Some(());
        }

        if let Some((reg, zero_op)) = zero_comparison(op, &lhs, &rhs) {
            let branch = zero_branch(zero_op, branch_when);
            out.push_str(&format!("  {} {}, {}\n", branch, reg, target));
            return Some(());
        }

        let lhs_reg = self.materialize_operand(lhs, "t0", out);
        let rhs_reg = self.materialize_operand(rhs, "t1", out);
        let branch = icmp_branch(op, branch_when);
        out.push_str(&format!(
            "  {} {}, {}, {}\n",
            branch, lhs_reg, rhs_reg, target
        ));
        Some(())
    }

    fn emit_fast_result(&self, result: EarlyReturnResult, out: &mut String) -> Option<()> {
        match result {
            EarlyReturnResult::Direct(value) => {
                match self.preframe_operand(value)? {
                    PreframeOperand::Reg(reg) if reg != "a0" => {
                        out.push_str(&format!("  mv a0, {}\n", reg));
                    }
                    PreframeOperand::Reg(_) => {}
                    PreframeOperand::Imm(value) => {
                        out.push_str(&format!("  li a0, {}\n", value));
                    }
                }
                Some(())
            }
            EarlyReturnResult::Binary { op, lhs, rhs } => {
                let lhs = self.preframe_operand(lhs)?;
                let rhs = self.preframe_operand(rhs)?;
                let lhs = self.materialize_operand(lhs, "t0", out);
                let rhs = self.materialize_operand(rhs, "t1", out);
                let instruction = match op {
                    BinaryOp::Iadd => "addw",
                    BinaryOp::Isub => "subw",
                    BinaryOp::Imul => "mulw",
                    BinaryOp::Iand => "and",
                    BinaryOp::Ior => "or",
                    BinaryOp::Ixor => "xor",
                    BinaryOp::Ishl => "sllw",
                    BinaryOp::Iashr => "sraw",
                    _ => return None,
                };
                out.push_str(&format!("  {} a0, {}, {}\n", instruction, lhs, rhs));
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
        let locations = super::abi::assign_riscv_arg_locations(&sig.params);
        match locations.get(param_idx)? {
            IrArgLocation::IntReg(reg) => Some(format!("a{}", reg)),
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
                out.push_str(&format!("  li {}, {}\n", scratch, value));
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

fn zero_branch(op: CmpOp, branch_when: bool) -> &'static str {
    match (op, branch_when) {
        (CmpOp::Eq, true) | (CmpOp::Ne, false) => "beqz",
        (CmpOp::Eq, false) | (CmpOp::Ne, true) => "bnez",
        (CmpOp::Lt, true) | (CmpOp::Ge, false) => "bltz",
        (CmpOp::Lt, false) | (CmpOp::Ge, true) => "bgez",
        (CmpOp::Le, true) | (CmpOp::Gt, false) => "blez",
        (CmpOp::Le, false) | (CmpOp::Gt, true) => "bgtz",
    }
}

fn icmp_branch(op: CmpOp, branch_when: bool) -> &'static str {
    match (op, branch_when) {
        (CmpOp::Eq, true) | (CmpOp::Ne, false) => "beq",
        (CmpOp::Eq, false) | (CmpOp::Ne, true) => "bne",
        (CmpOp::Lt, true) | (CmpOp::Ge, false) => "blt",
        (CmpOp::Lt, false) | (CmpOp::Ge, true) => "bge",
        (CmpOp::Le, true) | (CmpOp::Gt, false) => "ble",
        (CmpOp::Le, false) | (CmpOp::Gt, true) => "bgt",
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
