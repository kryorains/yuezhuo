use super::Riscv64IrFuncEmitter;
use crate::codegen::common::{EarlyReturnResult, EntryEarlyReturn, IrArgLocation};
use crate::ir::{BinaryOp, CmpOp, Const, InstKind, UnaryOp, ValueId, ValueKind};

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn pre_prologue_early_return(&self, plan: &EntryEarlyReturn) -> Option<String> {
        let frame_label = format!(".L_frame_{}", self.func.name);
        let mut out = String::new();
        let phi_inputs = plan
            .entry_phi_values
            .iter()
            .copied()
            .collect::<std::collections::HashMap<_, _>>();
        if let Some(chain) = plan.chained {
            let return_label = format!(".L_preframe_return_{}", self.func.name);
            self.emit_guard_branch(
                plan.condition,
                plan.fast_when_true,
                &return_label,
                &phi_inputs,
                &mut out,
            )?;
            self.emit_guard_branch(
                chain.condition,
                chain.fast_when_true,
                &return_label,
                &phi_inputs,
                &mut out,
            )?;
            out.push_str(&format!("  j {frame_label}\n{return_label}:\n"));
            self.emit_fast_result(&plan.result, &phi_inputs, &mut out)?;
            out.push_str("  ret\n");
            out.push_str(&format!("{frame_label}:\n"));
            return Some(out);
        }
        self.emit_guard_branch(
            plan.condition,
            !plan.fast_when_true,
            &frame_label,
            &phi_inputs,
            &mut out,
        )?;
        self.emit_fast_result(&plan.result, &phi_inputs, &mut out)?;
        out.push_str("  ret\n");
        out.push_str(&format!("{}:\n", frame_label));
        Some(out)
    }

    fn emit_guard_branch(
        &self,
        value: ValueId,
        branch_when: bool,
        target: &str,
        phi_inputs: &std::collections::HashMap<ValueId, ValueId>,
        out: &mut String,
    ) -> Option<()> {
        if let Some(initial) = phi_inputs.get(&value).copied() {
            return self.emit_guard_branch(initial, branch_when, target, phi_inputs, out);
        }
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
                    } => self.emit_guard_branch(value, !branch_when, target, phi_inputs, out),
                    InstKind::Binary {
                        op: BinaryOp::And,
                        lhs,
                        rhs,
                    } if !branch_when => {
                        self.emit_guard_branch(lhs, false, target, phi_inputs, out)?;
                        self.emit_guard_branch(rhs, false, target, phi_inputs, out)
                    }
                    InstKind::Binary {
                        op: BinaryOp::Or,
                        lhs,
                        rhs,
                    } if branch_when => {
                        self.emit_guard_branch(lhs, true, target, phi_inputs, out)?;
                        self.emit_guard_branch(rhs, true, target, phi_inputs, out)
                    }
                    InstKind::Icmp { op, lhs, rhs } => self.emit_preframe_icmp_branch(
                        op,
                        lhs,
                        rhs,
                        branch_when,
                        target,
                        phi_inputs,
                        out,
                    ),
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
        phi_inputs: &std::collections::HashMap<ValueId, ValueId>,
        out: &mut String,
    ) -> Option<()> {
        let lhs = self.preframe_guard_operand(lhs, "t0", phi_inputs, out)?;
        let rhs = self.preframe_guard_operand(rhs, "t1", phi_inputs, out)?;
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

    fn emit_fast_result(
        &self,
        result: &EarlyReturnResult,
        phi_inputs: &std::collections::HashMap<ValueId, ValueId>,
        out: &mut String,
    ) -> Option<()> {
        match result {
            EarlyReturnResult::Void => Some(()),
            EarlyReturnResult::Direct(value) => {
                match self.preframe_mapped_operand(*value, phi_inputs)? {
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
                let lhs = self.preframe_mapped_operand(*lhs, phi_inputs)?;
                let rhs = self.preframe_mapped_operand(*rhs, phi_inputs)?;
                let lhs = self.materialize_operand(lhs, "t0", out);
                let rhs = self.materialize_operand(rhs, "t1", out);
                let instruction = match *op {
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
            EarlyReturnResult::Expression { block, result } => {
                self.emit_fast_expression(*block, *result, phi_inputs, out)
            }
        }
    }

    fn emit_fast_expression(
        &self,
        block: crate::ir::BlockId,
        result: ValueId,
        phi_inputs: &std::collections::HashMap<ValueId, ValueId>,
        out: &mut String,
    ) -> Option<()> {
        let mut regs = std::collections::HashMap::<ValueId, &'static str>::new();
        let scratch = ["t0", "t1", "t2", "t3", "t4", "t5"];
        let active = self.func.blocks[block.0]
            .insts
            .iter()
            .filter(|inst| !matches!(inst.kind, InstKind::Nop));
        for (inst, destination) in active.zip(scratch) {
            let (Some(inst_result), InstKind::Binary { op, lhs, rhs }) = (inst.result, &inst.kind)
            else {
                return None;
            };
            let lhs = self.preframe_expression_operand(*lhs, phi_inputs, &regs)?;
            let rhs = self.preframe_expression_operand(*rhs, phi_inputs, &regs)?;
            if let (PreframeOperand::Imm(lhs), PreframeOperand::Imm(rhs)) = (&lhs, &rhs) {
                let value = eval_binary_i32(*op, *lhs, *rhs)?;
                out.push_str(&format!("  li {destination}, {value}\n"));
                regs.insert(inst_result, destination);
                continue;
            }
            let lhs = self.materialize_operand(lhs, "t6", out);
            let rhs = self.materialize_operand(rhs, "t6", out);
            let instruction = match *op {
                BinaryOp::Iadd => "addw",
                BinaryOp::Isub => "subw",
                BinaryOp::Imul => "mulw",
                BinaryOp::Iand => "and",
                BinaryOp::Ior => "or",
                BinaryOp::Ixor => "xor",
                BinaryOp::Ishl => "sllw",
                BinaryOp::Iashr => "sraw",
                BinaryOp::Idiv => "divw",
                BinaryOp::Imod => "remw",
                BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Fadd
                | BinaryOp::Fsub
                | BinaryOp::Fmul
                | BinaryOp::Fdiv => return None,
            };
            out.push_str(&format!("  {instruction} {destination}, {lhs}, {rhs}\n"));
            regs.insert(inst_result, destination);
        }
        match self.preframe_expression_operand(result, phi_inputs, &regs)? {
            PreframeOperand::Reg(source) if source != "a0" => {
                out.push_str(&format!("  mv a0, {source}\n"));
            }
            PreframeOperand::Reg(_) => {}
            PreframeOperand::Imm(value) => out.push_str(&format!("  li a0, {value}\n")),
        }
        Some(())
    }

    fn preframe_expression_operand(
        &self,
        value: ValueId,
        phi_inputs: &std::collections::HashMap<ValueId, ValueId>,
        regs: &std::collections::HashMap<ValueId, &'static str>,
    ) -> Option<PreframeOperand> {
        if let Some(reg) = regs.get(&value) {
            return Some(PreframeOperand::Reg((*reg).to_string()));
        }
        self.preframe_mapped_operand(value, phi_inputs)
    }

    fn preframe_mapped_operand(
        &self,
        value: ValueId,
        phi_inputs: &std::collections::HashMap<ValueId, ValueId>,
    ) -> Option<PreframeOperand> {
        self.preframe_operand(phi_inputs.get(&value).copied().unwrap_or(value))
    }

    fn preframe_operand(&self, value: ValueId) -> Option<PreframeOperand> {
        match self.func.value(value).kind {
            ValueKind::Param => Some(PreframeOperand::Reg(self.preframe_param_reg(value)?)),
            ValueKind::Const(Const::Int(value)) => Some(PreframeOperand::Imm(value)),
            ValueKind::Const(Const::Bool(value)) => Some(PreframeOperand::Imm(i32::from(value))),
            ValueKind::Const(_) | ValueKind::Global(_) | ValueKind::Inst(_, _) => None,
        }
    }

    fn preframe_guard_operand(
        &self,
        value: ValueId,
        scratch: &'static str,
        phi_inputs: &std::collections::HashMap<ValueId, ValueId>,
        out: &mut String,
    ) -> Option<PreframeOperand> {
        let value = phi_inputs.get(&value).copied().unwrap_or(value);
        if let Some(operand) = self.preframe_operand(value) {
            return Some(operand);
        }
        let ValueKind::Inst(block, inst_index) = self.func.value(value).kind else {
            return None;
        };
        let InstKind::Binary { op, lhs, rhs } =
            self.func.blocks.get(block.0)?.insts.get(inst_index)?.kind
        else {
            return None;
        };
        let PreframeOperand::Imm(rhs) = self.preframe_operand(rhs)? else {
            return None;
        };
        let lhs = self.preframe_mapped_operand(lhs, phi_inputs)?;
        let lhs = self.materialize_operand(lhs, scratch, out);
        match op {
            BinaryOp::Iadd | BinaryOp::Isub => {
                let immediate = if op == BinaryOp::Iadd {
                    rhs
                } else {
                    rhs.checked_neg()?
                };
                if !(-2048..=2047).contains(&immediate) {
                    return None;
                }
                out.push_str(&format!("  addiw {scratch}, {lhs}, {immediate}\n"));
            }
            BinaryOp::Iand | BinaryOp::Ior | BinaryOp::Ixor if (-2048..=2047).contains(&rhs) => {
                let instruction = match op {
                    BinaryOp::Iand => "andi",
                    BinaryOp::Ior => "ori",
                    BinaryOp::Ixor => "xori",
                    _ => unreachable!(),
                };
                out.push_str(&format!("  {instruction} {scratch}, {lhs}, {rhs}\n"));
            }
            _ => return None,
        }
        Some(PreframeOperand::Reg(scratch.to_string()))
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

fn eval_binary_i32(op: BinaryOp, lhs: i32, rhs: i32) -> Option<i32> {
    match op {
        BinaryOp::Iadd => Some(lhs.wrapping_add(rhs)),
        BinaryOp::Isub => Some(lhs.wrapping_sub(rhs)),
        BinaryOp::Imul => Some(lhs.wrapping_mul(rhs)),
        BinaryOp::Idiv => (rhs != 0 && !(lhs == i32::MIN && rhs == -1)).then(|| lhs / rhs),
        BinaryOp::Imod => (rhs != 0 && !(lhs == i32::MIN && rhs == -1)).then(|| lhs % rhs),
        BinaryOp::Iand => Some(lhs & rhs),
        BinaryOp::Ior => Some(lhs | rhs),
        BinaryOp::Ixor => Some(lhs ^ rhs),
        BinaryOp::Ishl => Some(lhs.wrapping_shl(rhs as u32 & 31)),
        BinaryOp::Iashr => Some(lhs.wrapping_shr(rhs as u32 & 31)),
        BinaryOp::And
        | BinaryOp::Or
        | BinaryOp::Fadd
        | BinaryOp::Fsub
        | BinaryOp::Fmul
        | BinaryOp::Fdiv => None,
    }
}
