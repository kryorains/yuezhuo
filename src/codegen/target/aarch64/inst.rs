use super::imm::mov_w_imm;
use super::AArch64IrFuncEmitter;
use crate::codegen::common::signed_magic_positive;
use crate::codegen::Target;
use crate::ir::{
    BinaryOp, BlockId, CastOp, CmpOp, Const, Inst, InstKind, Terminator, Type, UnaryOp, ValueId,
    ValueKind,
};
use std::collections::HashSet;

#[derive(Clone, Copy)]
struct EvenHalvingPlan {
    header: BlockId,
    body: BlockId,
    state: ValueId,
    depth: ValueId,
    division: ValueId,
    add: ValueId,
    known_even_entry: bool,
}

#[derive(Clone, Copy)]
struct BranchlessPhiPlan {
    block: BlockId,
    condition: ValueId,
    compare: (CmpOp, ValueId, ValueId),
    merge: BlockId,
    old_value: ValueId,
    merged_value: ValueId,
    multiply_pointers: [ValueId; 2],
    update_on_true: bool,
}

struct NonnegativeShiftPlan {
    source: ValueId,
    shift: ValueId,
    intermediates: Vec<ValueId>,
}

#[derive(Clone, Copy)]
struct PureBranchlessPhiPlan {
    compare: (CmpOp, ValueId, ValueId),
    update: BlockId,
    merge: BlockId,
    old_value: ValueId,
    updated_value: ValueId,
    merged_value: ValueId,
    update_on_true: bool,
}

fn phi_incoming(func: &crate::ir::Function, phi: ValueId, pred: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst_kind(func, phi)? else {
        return None;
    };
    incomings
        .iter()
        .find_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value))
}

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
                if !self.emit_assigned_load(result, *ptr) {
                    let (base, offset) = self.memory_address(*ptr);
                    self.load_value(base);
                    let ty = self.func.value(result).ty.clone();
                    self.load_indirect_offset(&ty, offset);
                    self.store_result(result);
                }
            }
            InstKind::Store { ptr, value } => {
                if !self.emit_assigned_store(*ptr, *value) {
                    let (base, offset) = self.memory_address(*ptr);
                    self.load_value_into(base, "x1");
                    self.load_value(*value);
                    let ty = self.func.value(*value).ty.clone();
                    self.store_indirect_offset(&ty, offset);
                }
            }
            InstKind::MemZero { ptr, bytes } => self.emit_memzero(*ptr, *bytes),
            InstKind::Unary { op, value } => {
                let result = inst.result.unwrap();
                if !self.emit_assigned_unary(result, *op, *value) {
                    self.emit_unary(*op, *value);
                    self.store_result(result);
                }
            }
            InstKind::Binary { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                if *op == BinaryOp::Iashr && self.emit_fused_nonnegative_adjusted_shift(result) {
                    return;
                }
                if matches!(
                    *op,
                    BinaryOp::Iadd | BinaryOp::Isub | BinaryOp::Iand | BinaryOp::Ishl
                ) && self.fused_nonnegative_shift_user(result)
                {
                    return;
                }
                if matches!(*op, BinaryOp::Ishl | BinaryOp::Imul)
                    && self.fused_shifted_add_user(result)
                {
                    return;
                }
                if *op == BinaryOp::Idiv {
                    if let Some(plan) = self.even_halving_plan(result) {
                        self.emit_even_halving_run(plan);
                        return;
                    }
                }
                if *op == BinaryOp::Iadd && self.is_compressed_halving_add(result) {
                    return;
                }
                if matches!(*op, BinaryOp::Iashr | BinaryOp::Ixor)
                    && self.is_negative_sanitize_intermediate(result)
                {
                    return;
                }
                if *op == BinaryOp::Iand && self.emit_negative_sanitize(result) {
                    return;
                }
                if matches!(*op, BinaryOp::And | BinaryOp::Or)
                    && self.emit_fused_boolean_compare(result, *op, *lhs, *rhs)
                {
                    return;
                }
                if (*op == BinaryOp::Imul && self.fused_madd_user(result).is_some())
                    || (Target::AArch64.cost_model().contract_float_madd()
                        && *op == BinaryOp::Fmul
                        && (self.fused_float_madd_user(result).is_some()
                            || self.fused_float_msub_user(result).is_some()))
                    || (*op == BinaryOp::Iand && self.fused_bit_test_branch_user(result))
                {
                    return;
                }
                if *op == BinaryOp::Iadd {
                    if let Some((mul_lhs, mul_rhs, addend)) =
                        self.fused_madd_operands(result, *lhs, *rhs)
                    {
                        if let (Some(destination), Some(mul_lhs), Some(mul_rhs), Some(addend)) = (
                            self.assigned_w_reg(result),
                            self.assigned_w_reg(mul_lhs),
                            self.assigned_w_reg(mul_rhs),
                            self.assigned_w_reg(addend),
                        ) {
                            self.body.push_str(&format!(
                                "  madd {}, {}, {}, {}\n",
                                destination, mul_lhs, mul_rhs, addend
                            ));
                        } else {
                            self.load_value_into(mul_lhs, "x1");
                            self.load_value(mul_rhs);
                            self.load_value_into(addend, "x2");
                            self.body.push_str("  madd w0, w1, w0, w2\n");
                            self.store_result(result);
                        }
                        return;
                    }
                }
                if Target::AArch64.cost_model().contract_float_madd() && *op == BinaryOp::Fadd {
                    if let Some((mul_lhs, mul_rhs, addend)) =
                        self.fused_float_madd_operands(result, *lhs, *rhs)
                    {
                        let mul_lhs = self.load_or_assigned_float(mul_lhs, "s2");
                        let mul_rhs = self.load_or_assigned_float(mul_rhs, "s1");
                        let addend = self.load_or_assigned_float(addend, "s0");
                        let destination = self.assigned_float_reg(result).unwrap_or("s0");
                        self.body.push_str(&format!(
                            "  fmadd {}, {}, {}, {}\n",
                            destination, mul_lhs, mul_rhs, addend
                        ));
                        if self.assigned_float_reg(result).is_none() {
                            self.store_float_result(result, destination);
                        }
                        return;
                    }
                }
                if Target::AArch64.cost_model().contract_float_madd() && *op == BinaryOp::Fsub {
                    if let Some((instruction, mul_lhs, mul_rhs, addend)) =
                        self.fused_float_msub_operands(result, *lhs, *rhs)
                    {
                        let mul_lhs = self.load_or_assigned_float(mul_lhs, "s2");
                        let mul_rhs = self.load_or_assigned_float(mul_rhs, "s1");
                        let addend = self.load_or_assigned_float(addend, "s0");
                        let destination = self.assigned_float_reg(result).unwrap_or("s0");
                        self.body.push_str(&format!(
                            "  {} {}, {}, {}, {}\n",
                            instruction, destination, mul_lhs, mul_rhs, addend
                        ));
                        if self.assigned_float_reg(result).is_none() {
                            self.store_float_result(result, destination);
                        }
                        return;
                    }
                }
                if self.emit_assigned_binary_imm(result, *op, *lhs, *rhs)
                    || self.emit_assigned_binary(result, *op, *lhs, *rhs)
                {
                    return;
                }
                self.emit_binary(*op, *lhs, *rhs);
                self.store_result(result);
            }
            InstKind::Icmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                if self.direct_branch_icmp(result).is_none()
                    && !self.fused_boolean_compare_user(result)
                {
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
                if !self.emit_assigned_cast(result, *op, *value) {
                    self.emit_cast(*op, *value);
                    self.store_result(result);
                }
            }
            InstKind::Gep { base, indices } => {
                let result = inst.result.unwrap();
                if self.skips_folded_memory_gep(result) {
                    return;
                }
                if !self.emit_assigned_gep(result, *base, indices) {
                    self.emit_gep(result, *base, indices);
                    self.store_result(result);
                }
            }
            InstKind::Call { name, args } => {
                let ret = self.emit_call(name, args, inst.result);
                if let Some(result) = inst.result {
                    if ret != Type::F32 {
                        self.store_result(result);
                    }
                }
            }
        }
    }

    pub(super) fn emit_terminator(
        &mut self,
        block_idx: usize,
        terminator: &Terminator,
        next_block: Option<usize>,
    ) {
        match terminator {
            Terminator::Return(value) => {
                if let Some(value) = value {
                    if self.func.value(*value).ty == Type::F32 {
                        self.load_float_value(*value, "s0");
                    } else {
                        self.load_value(*value);
                    }
                }
                self.body.push_str(&format!("  b {}\n", self.return_label));
            }
            Terminator::Jump(target) => {
                if self.emit_known_even_backedge(block_idx, *target) {
                    return;
                }
                self.emit_phi_copies(block_idx, target.0);
                if next_block != Some(target.0) {
                    self.body
                        .push_str(&format!("  b {}\n", self.block_label(target.0)));
                }
            }
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                if then_target == else_target {
                    self.emit_phi_copies(block_idx, then_target.0);
                    if next_block != Some(then_target.0) {
                        self.body
                            .push_str(&format!("  b {}\n", self.block_label(then_target.0)));
                    }
                    return;
                }

                if self.emit_safe_branchless_phi_update(
                    block_idx,
                    *cond,
                    *then_target,
                    *else_target,
                    next_block,
                ) {
                    return;
                }
                if self.emit_pure_branchless_phi_update(
                    block_idx,
                    *cond,
                    *then_target,
                    *else_target,
                    next_block,
                ) {
                    return;
                }

                if !self.edge_has_phi_copy(block_idx, then_target.0)
                    && !self.edge_has_phi_copy(block_idx, else_target.0)
                {
                    if next_block == Some(then_target.0) {
                        self.emit_branch_if_false(*cond, self.block_label(else_target.0));
                    } else if next_block == Some(else_target.0) {
                        self.emit_branch_if_true(*cond, self.block_label(then_target.0));
                    } else {
                        self.emit_branch_if_false(*cond, self.block_label(else_target.0));
                        self.body
                            .push_str(&format!("  b {}\n", self.block_label(then_target.0)));
                    }
                    return;
                }

                let else_edge = self.parent.ctx.fresh_label("else_edge");
                self.emit_branch_if_false(*cond, else_edge.clone());
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

    pub(super) fn emit_phi_copies(&mut self, pred_idx: usize, target_idx: usize) {
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
                        if self.func.value(*result).ty == Type::F32 {
                            self.assigned_float_reg(*result) != self.assigned_float_reg(*value)
                                || self.assigned_float_reg(*result).is_none()
                        } else {
                            self.phi_regs.reg(*result) != self.phi_regs.reg(*value)
                                || self.phi_regs.reg(*result).is_none()
                        }
                    })
            })
            .collect::<Vec<_>>();

        if let [(result, value)] = copies.as_slice() {
            if self.func.value(*result).ty == Type::F32 {
                if let (Some(destination), Some(source)) = (
                    self.assigned_float_reg(*result),
                    self.assigned_float_reg(*value),
                ) {
                    self.body
                        .push_str(&format!("  fmov {}, {}\n", destination, source));
                } else {
                    self.load_float_value(*value, "s0");
                    self.store_float_result(*result, "s0");
                }
                return;
            }
            if let (Some(destination), Some(source)) =
                (self.assigned_x_reg(*result), self.assigned_x_reg(*value))
            {
                match self.func.value(*result).ty {
                    Type::Ptr(_) => self
                        .body
                        .push_str(&format!("  mov {}, {}\n", destination, source)),
                    _ => self.body.push_str(&format!(
                        "  mov {}, {}\n",
                        w_reg_name(destination),
                        w_reg_name(source)
                    )),
                }
            } else {
                self.load_value(*value);
                self.store_result(*result);
            }
            return;
        }

        if copies.iter().all(|(result, value)| {
            self.func.value(*result).ty != Type::F32
                && self.assigned_x_reg(*result).is_some()
                && (self.assigned_x_reg(*value).is_some()
                    || phi_copy_constant(self.func, *value).is_some())
        }) {
            let mut register_copies = copies
                .iter()
                .map(|(result, value)| {
                    (
                        self.assigned_x_reg(*result)
                            .expect("checked phi destination register"),
                        self.assigned_x_reg(*value),
                        *value,
                        matches!(self.func.value(*result).ty, Type::Ptr(_)),
                    )
                })
                .collect::<Vec<_>>();

            while !register_copies.is_empty() {
                if let Some(ready) = register_copies.iter().position(|(destination, _, _, _)| {
                    !register_copies
                        .iter()
                        .any(|(_, source, _, _)| *source == Some(*destination))
                }) {
                    let (destination, source, value, is_pointer) = register_copies.remove(ready);
                    if let Some(source) = source {
                        if is_pointer {
                            self.body
                                .push_str(&format!("  mov {}, {}\n", destination, source));
                        } else {
                            self.body.push_str(&format!(
                                "  mov {}, {}\n",
                                w_reg_name(destination),
                                w_reg_name(source)
                            ));
                        }
                    } else if is_pointer {
                        self.body.push_str(&format!("  mov {}, xzr\n", destination));
                    } else {
                        self.body.push_str(&mov_w_imm(
                            &w_reg_name(destination),
                            phi_copy_constant(self.func, value)
                                .expect("checked integer phi constant"),
                        ));
                    }
                    continue;
                }

                // Every remaining destination is still needed as a source.
                // Preserve one full physical register in the reserved x16
                // scratch, then schedule the opened cycle without touching the
                // stack.  Saving all 64 bits also covers a mixed i32/pointer
                // register cycle; word consumers simply read w16.
                let preserved = register_copies
                    .iter()
                    .find_map(|(destination, source, _, _)| source.map(|_| *destination))
                    .expect("only register sources can form a phi-copy cycle");
                self.body.push_str(&format!("  mov x16, {}\n", preserved));
                for (_, source, _, _) in &mut register_copies {
                    if *source == Some(preserved) {
                        *source = Some("x16");
                    }
                }
            }
            return;
        }

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

    fn emit_safe_branchless_phi_update(
        &mut self,
        block_idx: usize,
        cond: ValueId,
        then_target: BlockId,
        else_target: BlockId,
        next_block: Option<usize>,
    ) -> bool {
        let Some(plans) = self.branchless_phi_plans(block_idx, cond, then_target, else_target)
        else {
            return false;
        };

        let all_multiply_pointers = plans
            .iter()
            .flat_map(|plan| plan.multiply_pointers)
            .collect::<Vec<_>>();
        let cached_pointer = all_multiply_pointers
            .iter()
            .copied()
            .max_by_key(|pointer| {
                all_multiply_pointers
                    .iter()
                    .filter(|candidate| *candidate == pointer)
                    .count()
            })
            .expect("branchless multiply always has two load pointers");
        self.emit_branchless_i32_load(cached_pointer, "w0");

        for (index, plan) in plans.iter().copied().enumerate() {
            if index != 0 {
                let condition_insts = self.func.block(plan.block).insts.clone();
                for inst in &condition_insts {
                    if !matches!(inst.kind, InstKind::Phi { .. }) {
                        self.emit_inst(inst);
                    }
                }
            }

            if let Some((_, value, mask, _)) = self.direct_branch_bit_test(plan.condition) {
                let source = if let Some(source) = self.assigned_w_reg(value) {
                    source
                } else {
                    self.load_value(value);
                    "w0".to_string()
                };
                self.body
                    .push_str(&format!("  tst {}, #{}\n", source, mask));
            } else {
                self.emit_int_compare(plan.compare.1, plan.compare.2);
            }

            let old_register = self
                .assigned_w_reg(plan.old_value)
                .expect("branchless chain register eligibility checked");
            let mut operand_regs = ["w0", "w0"];
            let mut next_scratch = 1;
            for (operand_idx, pointer) in plan.multiply_pointers.iter().copied().enumerate() {
                if pointer != cached_pointer {
                    let scratch = if next_scratch == 1 { "w1" } else { "w2" };
                    self.emit_branchless_i32_load(pointer, scratch);
                    operand_regs[operand_idx] = scratch;
                    next_scratch += 1;
                }
            }
            self.body.push_str(&format!(
                "  madd w16, {}, {}, {}\n",
                operand_regs[0], operand_regs[1], old_register
            ));

            let condition = if plan.update_on_true {
                cmp_cc(plan.compare.0)
            } else {
                inverse_cmp_cc(plan.compare.0)
            };
            let destination = self
                .assigned_w_reg(plan.merged_value)
                .expect("branchless chain register eligibility checked");
            self.body.push_str(&format!(
                "  csel {}, w16, {}, {}\n",
                destination, old_register, condition
            ));
        }

        let merge = plans.last().expect("non-empty branchless chain").merge;
        if next_block != Some(merge.0) {
            self.body
                .push_str(&format!("  b {}\n", self.block_label(merge.0)));
        }
        true
    }

    fn branchless_phi_plans(
        &self,
        block_idx: usize,
        cond: ValueId,
        then_target: BlockId,
        else_target: BlockId,
    ) -> Option<Vec<BranchlessPhiPlan>> {
        let first_plan = self.branchless_phi_plan(block_idx, cond, then_target, else_target)?;
        let mut plans = vec![first_plan];
        let mut visited = HashSet::from([first_plan.block]);
        while plans.len() < 8 {
            let merge = plans.last()?.merge;
            if !visited.insert(merge) {
                break;
            }
            let Some(Terminator::Branch {
                cond,
                then_target,
                else_target,
            }) = self.func.block(merge).terminator.as_ref()
            else {
                break;
            };
            let Some(next) = self.branchless_phi_plan(merge.0, *cond, *then_target, *else_target)
            else {
                break;
            };
            plans.push(next);
        }

        if plans.iter().any(|plan| {
            self.assigned_w_reg(plan.old_value).is_none()
                || self.assigned_w_reg(plan.merged_value).is_none()
        }) {
            return None;
        }

        // Keep w0 available as a chain-wide cached load. Every instruction in
        // the later condition blocks must therefore either be elided/folded or
        // have an assigned register; otherwise generic lowering could use w0.
        if plans.iter().skip(1).any(|plan| {
            self.func.block(plan.block).insts.iter().any(|inst| {
                inst.result.is_some_and(|result| {
                    result != plan.condition
                        && self.assigned_x_reg(result).is_none()
                        && !self.skips_folded_memory_gep(result)
                })
            })
        }) {
            return None;
        }
        Some(plans)
    }

    pub(super) fn safe_branchless_phi_final_merge(
        &self,
        block_idx: usize,
        cond: ValueId,
        then_target: BlockId,
        else_target: BlockId,
    ) -> Option<BlockId> {
        self.branchless_phi_plans(block_idx, cond, then_target, else_target)
            .and_then(|plans| plans.last().map(|plan| plan.merge))
            .or_else(|| {
                self.pure_branchless_phi_plan(block_idx, cond, then_target, else_target)
                    .map(|plan| plan.merge)
            })
    }

    fn pure_branchless_phi_plan(
        &self,
        block_idx: usize,
        cond: ValueId,
        then_target: BlockId,
        else_target: BlockId,
    ) -> Option<PureBranchlessPhiPlan> {
        let compare = self.direct_branch_icmp(cond)?;
        let (update, merge, update_on_true) = if self.block_jumps_to(then_target, else_target) {
            (then_target, else_target, true)
        } else if self.block_jumps_to(else_target, then_target) {
            (else_target, then_target, false)
        } else {
            return None;
        };
        if self.predecessor_count(update) != 1 || self.predecessor_count(merge) != 2 {
            return None;
        }
        let phis = self
            .func
            .block(merge)
            .insts
            .iter()
            .filter_map(|inst| match (&inst.kind, inst.result) {
                (InstKind::Phi { incomings }, Some(result)) => Some((result, incomings)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [(merged_value, incomings)] = phis.as_slice() else {
            return None;
        };
        let old_value = incomings
            .iter()
            .find_map(|(pred, value)| (*pred == BlockId(block_idx)).then_some(*value))?;
        let updated_value = incomings
            .iter()
            .find_map(|(pred, value)| (*pred == update).then_some(*value))?;
        if incomings.len() != 2
            || [*merged_value, old_value, updated_value]
                .into_iter()
                .any(|value| {
                    self.func.value(value).ty != Type::I32 || self.assigned_w_reg(value).is_none()
                })
            || !matches!(
                self.func.value(updated_value).kind,
                ValueKind::Inst(owner, _) if owner == update
            ) || self.func.block(update).insts.iter().any(|inst| {
            !matches!(inst.kind, InstKind::Nop)
                && !matches!(
                    inst.kind,
                    InstKind::Binary {
                        op: BinaryOp::Iadd
                            | BinaryOp::Isub
                            | BinaryOp::Imul
                            | BinaryOp::Iand
                            | BinaryOp::Ior
                            | BinaryOp::Ixor
                            | BinaryOp::Ishl
                            | BinaryOp::Iashr,
                        ..
                    } if inst.result.is_some_and(|result| self.func.value(result).ty == Type::I32)
                )
        }) {
            return None;
        }
        Some(PureBranchlessPhiPlan {
            compare,
            update,
            merge,
            old_value,
            updated_value,
            merged_value: *merged_value,
            update_on_true,
        })
    }

    fn emit_pure_branchless_phi_update(
        &mut self,
        block_idx: usize,
        cond: ValueId,
        then_target: BlockId,
        else_target: BlockId,
        next_block: Option<usize>,
    ) -> bool {
        let Some(plan) = self.pure_branchless_phi_plan(block_idx, cond, then_target, else_target)
        else {
            return false;
        };
        self.emit_int_compare(plan.compare.1, plan.compare.2);
        let old = self
            .assigned_w_reg(plan.old_value)
            .expect("pure branchless old value register checked");
        let (updated, preserved_old) = if self.emit_pure_shifted_add_into_scratch(plan) {
            ("w16".to_string(), old.clone())
        } else {
            self.body.push_str(&format!("  mov w16, {old}\n"));
            for inst in &self.func.block(plan.update).insts.clone() {
                self.emit_inst(inst);
            }
            (
                self.assigned_w_reg(plan.updated_value)
                    .expect("pure branchless updated value register checked"),
                "w16".to_string(),
            )
        };
        let destination = self
            .assigned_w_reg(plan.merged_value)
            .expect("pure branchless destination register checked");
        let condition = if plan.update_on_true {
            cmp_cc(plan.compare.0)
        } else {
            inverse_cmp_cc(plan.compare.0)
        };
        self.body.push_str(&format!(
            "  csel {destination}, {updated}, {preserved_old}, {condition}\n"
        ));
        if next_block != Some(plan.merge.0) {
            self.body
                .push_str(&format!("  b {}\n", self.block_label(plan.merge.0)));
        }
        true
    }

    fn emit_pure_shifted_add_into_scratch(&mut self, plan: PureBranchlessPhiPlan) -> bool {
        let Some(InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        }) = defining_inst_kind(self.func, plan.updated_value)
        else {
            return false;
        };
        let shifted_add = [(*lhs, *rhs), (*rhs, *lhs)]
            .into_iter()
            .find_map(|(shifted, addend)| {
                let (source, shift) = match defining_inst_kind(self.func, shifted)? {
                    InstKind::Binary {
                        op: BinaryOp::Ishl,
                        lhs,
                        rhs,
                    } => (*lhs, const_i32(self.func, *rhs)?),
                    InstKind::Binary {
                        op: BinaryOp::Imul,
                        lhs,
                        rhs,
                    } => {
                        if let Some(shift) = pow2_shift(self.func, *rhs) {
                            (*lhs, shift as i32)
                        } else {
                            (*rhs, pow2_shift(self.func, *lhs)? as i32)
                        }
                    }
                    _ => return None,
                };
                let shift = (0..=4).contains(&shift).then_some(shift)?;
                (self.value_use_counts[shifted.0] == 1
                    && matches!(
                        self.func.value(shifted).kind,
                        ValueKind::Inst(owner, _) if owner == plan.update
                    ))
                .then_some((shifted, source, addend, shift))
            });
        let Some((shifted, source, addend, shift)) = shifted_add else {
            return false;
        };
        let (Some(source), Some(addend)) =
            (self.assigned_w_reg(source), self.assigned_w_reg(addend))
        else {
            return false;
        };
        for inst in &self.func.block(plan.update).insts.clone() {
            if !matches!(inst.result, Some(result) if result == shifted || result == plan.updated_value)
            {
                self.emit_inst(inst);
            }
        }
        self.body
            .push_str(&format!("  add w16, {addend}, {source}, lsl #{shift}\n"));
        true
    }

    fn emit_branchless_i32_load(&mut self, pointer: ValueId, destination: &str) {
        let (base, offset) = self.memory_address(pointer);
        let base = self
            .assigned_x_reg(base)
            .expect("branchless load address eligibility checked");
        if offset == 0 {
            self.body
                .push_str(&format!("  ldr {}, [{}]\n", destination, base));
        } else {
            debug_assert!((-256..=255).contains(&offset));
            self.body.push_str(&format!(
                "  ldur {}, [{}, #{}]\n",
                destination, base, offset
            ));
        }
    }

    fn branchless_phi_plan(
        &self,
        block_idx: usize,
        cond: ValueId,
        then_target: BlockId,
        else_target: BlockId,
    ) -> Option<BranchlessPhiPlan> {
        let compare = self.direct_branch_icmp(cond)?;
        let (update, merge, update_on_true) = if self.block_jumps_to(then_target, else_target) {
            (then_target, else_target, true)
        } else if self.block_jumps_to(else_target, then_target) {
            (else_target, then_target, false)
        } else {
            return None;
        };
        if self.predecessor_count(update) != 1 || self.predecessor_count(merge) != 2 {
            return None;
        }

        let phis = self
            .func
            .block(merge)
            .insts
            .iter()
            .filter_map(|inst| match (&inst.kind, inst.result) {
                (InstKind::Phi { incomings }, Some(result)) => Some((result, incomings)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [(merged_value, incomings)] = phis.as_slice() else {
            return None;
        };
        if self.func.value(*merged_value).ty != Type::I32 {
            return None;
        }
        let old_value = incomings
            .iter()
            .find_map(|(pred, value)| (*pred == BlockId(block_idx)).then_some(*value))?;
        let updated_value = incomings
            .iter()
            .find_map(|(pred, value)| (*pred == update).then_some(*value))?;
        if incomings.len() != 2
            || self.func.value(old_value).ty != Type::I32
            || self.func.value(updated_value).ty != Type::I32
            || !matches!(
                self.func.value(updated_value).kind,
                ValueKind::Inst(owner, _) if owner == update
            )
        {
            return None;
        }

        let update_insts = &self.func.block(update).insts;
        let mut update_loads = Vec::new();
        for inst in update_insts {
            match inst.kind {
                InstKind::Nop => {}
                InstKind::Load { ptr }
                    if inst
                        .result
                        .is_some_and(|result| self.func.value(result).ty == Type::I32)
                        && {
                            let (base, offset) = self.memory_address(ptr);
                            self.assigned_x_reg(base).is_some() && (-256..=255).contains(&offset)
                        } =>
                {
                    update_loads.push(ptr);
                }
                InstKind::Binary {
                    op: BinaryOp::Imul | BinaryOp::Iadd,
                    ..
                } if inst
                    .result
                    .is_some_and(|result| self.func.value(result).ty == Type::I32) => {}
                _ => return None,
            }
        }
        if update_loads.is_empty() {
            return None;
        }

        let multiply_pointers =
            branchless_madd_load_pointers(self.func, updated_value, old_value, update)?;

        let mut condition_loads = Vec::new();
        collect_condition_load_pointers(self.func, cond, &mut HashSet::new(), &mut condition_loads);
        if condition_loads.is_empty()
            || update_loads.iter().any(|update_ptr| {
                !condition_loads.iter().any(|condition_ptr| {
                    same_typed_global_address_shape(
                        self.func,
                        *update_ptr,
                        *condition_ptr,
                        &mut HashSet::new(),
                    )
                })
            })
        {
            return None;
        }

        Some(BranchlessPhiPlan {
            block: BlockId(block_idx),
            condition: cond,
            compare,
            merge,
            old_value,
            merged_value: *merged_value,
            multiply_pointers,
            update_on_true,
        })
    }

    fn block_jumps_to(&self, block: BlockId, target: BlockId) -> bool {
        matches!(
            self.func.block(block).terminator,
            Some(Terminator::Jump(actual)) if actual == target
        )
    }

    fn predecessor_count(&self, target: BlockId) -> usize {
        self.func
            .blocks
            .iter()
            .filter(|block| match block.terminator {
                Some(Terminator::Jump(successor)) => successor == target,
                Some(Terminator::Branch {
                    then_target,
                    else_target,
                    ..
                }) => then_target == target || else_target == target,
                Some(Terminator::Return(_)) | None => false,
            })
            .count()
    }

    fn edge_has_phi_copy(&self, pred_idx: usize, target_idx: usize) -> bool {
        self.func
            .block(BlockId(target_idx))
            .insts
            .iter()
            .filter_map(|inst| {
                let (Some(result), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
                    return None;
                };
                incomings
                    .iter()
                    .find(|(pred, _)| pred.0 == pred_idx)
                    .map(|(_, incoming)| (result, *incoming))
            })
            .any(|(result, incoming)| {
                self.phi_regs.reg(result) != self.phi_regs.reg(incoming)
                    || self.phi_regs.reg(result).is_none()
            })
    }

    fn emit_branch_if_false(&mut self, cond: ValueId, target: String) {
        if self.emit_bit_test_branch(cond, &target, false) {
            return;
        }
        if let Some((op, lhs, rhs)) = self.direct_branch_icmp(cond) {
            self.emit_int_compare(lhs, rhs);
            self.body
                .push_str(&format!("  b.{} {}\n", inverse_cmp_cc(op), target));
        } else {
            self.load_value(cond);
            self.body
                .push_str(&format!("  cmp w0, #0\n  beq {}\n", target));
        }
    }

    fn emit_branch_if_true(&mut self, cond: ValueId, target: String) {
        if self.emit_bit_test_branch(cond, &target, true) {
            return;
        }
        if let Some((op, lhs, rhs)) = self.direct_branch_icmp(cond) {
            self.emit_int_compare(lhs, rhs);
            self.body
                .push_str(&format!("  b.{} {}\n", cmp_cc(op), target));
        } else {
            self.load_value(cond);
            self.body
                .push_str(&format!("  cmp w0, #0\n  bne {}\n", target));
        }
    }

    fn emit_bit_test_branch(&mut self, cond: ValueId, target: &str, branch_if_true: bool) -> bool {
        let Some((op, value, mask, _)) = self.direct_branch_bit_test(cond) else {
            return false;
        };
        let branch_if_nonzero = matches!(op, CmpOp::Ne) == branch_if_true;
        let source = if let Some(reg) = self.assigned_w_reg(value) {
            reg
        } else {
            self.load_value(value);
            "w0".to_string()
        };
        let short_branch_range = self
            .func
            .blocks
            .iter()
            .map(|block| block.insts.len())
            .sum::<usize>()
            <= 1024;
        if (mask as u32).is_power_of_two() && short_branch_range {
            let instruction = if branch_if_nonzero { "tbnz" } else { "tbz" };
            self.body.push_str(&format!(
                "  {} {}, #{}, {}\n",
                instruction,
                source,
                (mask as u32).trailing_zeros(),
                target
            ));
        } else {
            let condition = if branch_if_nonzero { "ne" } else { "eq" };
            self.body.push_str(&format!(
                "  tst {}, #{}\n  b.{} {}\n",
                source, mask, condition, target
            ));
        }
        true
    }

    fn fused_bit_test_branch_user(&self, masked: ValueId) -> bool {
        self.value_use_counts[masked.0] == 1
            && self.func.blocks.iter().any(|block| {
                let Some(Terminator::Branch { cond, .. }) = block.terminator.as_ref() else {
                    return false;
                };
                self.direct_branch_bit_test(*cond)
                    .is_some_and(|(_, _, _, candidate)| candidate == masked)
            })
    }

    fn direct_branch_bit_test(&self, cond: ValueId) -> Option<(CmpOp, ValueId, i32, ValueId)> {
        let (op, lhs, rhs) = self.direct_branch_icmp(cond)?;
        if !matches!(op, CmpOp::Eq | CmpOp::Ne) {
            return None;
        }
        let masked = if const_i32(self.func, rhs) == Some(0) {
            lhs
        } else if const_i32(self.func, lhs) == Some(0) {
            rhs
        } else {
            return None;
        };
        if self.value_use_counts[masked.0] != 1 {
            return None;
        }
        let InstKind::Binary {
            op: BinaryOp::Iand,
            lhs: mask_lhs,
            rhs: mask_rhs,
        } = defining_inst_kind(self.func, masked)?
        else {
            return None;
        };
        let (value, mask) = if let Some(mask) = const_i32(self.func, *mask_rhs) {
            (*mask_lhs, mask)
        } else if let Some(mask) = const_i32(self.func, *mask_lhs) {
            (*mask_rhs, mask)
        } else {
            return None;
        };
        is_low_bit_mask(mask).then_some((op, value, mask, masked))
    }

    fn fused_boolean_compare_user(&self, compare: ValueId) -> bool {
        const MAX_FUSION_SLOT_DISTANCE: usize = 16;

        if self.value_use_counts[compare.0] != 1 {
            return false;
        }
        let ValueKind::Inst(block, inst_idx) = self.func.value(compare).kind else {
            return false;
        };
        self.func.blocks[block.0].insts[inst_idx + 1
            ..(inst_idx + 1 + MAX_FUSION_SLOT_DISTANCE).min(self.func.blocks[block.0].insts.len())]
            .iter()
            .any(|inst| {
                let (Some(result), InstKind::Binary { op, lhs, rhs }) = (inst.result, &inst.kind)
                else {
                    return false;
                };
                matches!(*op, BinaryOp::And | BinaryOp::Or)
                    && (*lhs == compare || *rhs == compare)
                    && self.boolean_compare_operands(result, *lhs, *rhs).is_some()
            })
    }

    fn boolean_compare_operands(
        &self,
        binary: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Option<[(CmpOp, ValueId, ValueId); 2]> {
        if self.value_use_counts[lhs.0] != 1 || self.value_use_counts[rhs.0] != 1 {
            return None;
        }
        let ValueKind::Inst(binary_block, binary_idx) = self.func.value(binary).kind else {
            return None;
        };
        let ValueKind::Inst(lhs_block, lhs_idx) = self.func.value(lhs).kind else {
            return None;
        };
        let ValueKind::Inst(rhs_block, rhs_idx) = self.func.value(rhs).kind else {
            return None;
        };
        if lhs_block != binary_block
            || rhs_block != binary_block
            || lhs_idx >= binary_idx
            || rhs_idx >= binary_idx
            || binary_idx.saturating_sub(lhs_idx.min(rhs_idx)) > 16
            || self.func.blocks[binary_block.0].insts[lhs_idx.min(rhs_idx)..binary_idx]
                .iter()
                .enumerate()
                .any(|(offset, inst)| {
                    let idx = lhs_idx.min(rhs_idx) + offset;
                    idx != lhs_idx && idx != rhs_idx && !matches!(inst.kind, InstKind::Nop)
                })
        {
            return None;
        }
        let InstKind::Icmp {
            op: lhs_op,
            lhs: lhs_lhs,
            rhs: lhs_rhs,
        } = *defining_inst_kind(self.func, lhs)?
        else {
            return None;
        };
        let InstKind::Icmp {
            op: rhs_op,
            lhs: rhs_lhs,
            rhs: rhs_rhs,
        } = *defining_inst_kind(self.func, rhs)?
        else {
            return None;
        };
        if [lhs_lhs, lhs_rhs, rhs_lhs, rhs_rhs]
            .into_iter()
            .any(|value| self.func.value(value).ty != Type::I32)
        {
            return None;
        }
        Some([(lhs_op, lhs_lhs, lhs_rhs), (rhs_op, rhs_lhs, rhs_rhs)])
    }

    fn emit_fused_boolean_compare(
        &mut self,
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let Some([first, second]) = self.boolean_compare_operands(result, lhs, rhs) else {
            return false;
        };
        self.emit_int_compare(first.1, first.2);
        let (condition, fallback_result) = match op {
            BinaryOp::And => (cmp_cc(first.0), false),
            BinaryOp::Or => (inverse_cmp_cc(first.0), true),
            _ => return false,
        };
        let fallback_nzcv = cmp_nzcv(second.0, fallback_result);
        if !self.emit_conditional_int_compare(second.1, second.2, fallback_nzcv, condition) {
            return false;
        }
        let destination = self
            .assigned_w_reg(result)
            .unwrap_or_else(|| "w0".to_string());
        self.body
            .push_str(&format!("  cset {}, {}\n", destination, cmp_cc(second.0)));
        if self.assigned_w_reg(result).is_none() {
            self.store_result(result);
        }
        true
    }

    fn emit_conditional_int_compare(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        fallback_nzcv: u8,
        condition: &str,
    ) -> bool {
        if let Some(immediate) = const_i32(self.func, rhs).filter(|value| (0..=31).contains(value))
        {
            let lhs = if let Some(lhs) = self.assigned_w_reg(lhs) {
                lhs
            } else {
                self.load_value_into(lhs, "x2");
                "w2".to_string()
            };
            self.body.push_str(&format!(
                "  ccmp {}, #{}, #{}, {}\n",
                lhs, immediate, fallback_nzcv, condition
            ));
            return true;
        }

        let lhs = if let Some(lhs) = self.assigned_w_reg(lhs) {
            lhs
        } else {
            self.load_value_into(lhs, "x2");
            "w2".to_string()
        };
        let rhs = if let Some(rhs) = self.assigned_w_reg(rhs) {
            rhs
        } else {
            self.load_value(rhs);
            "w0".to_string()
        };
        self.body.push_str(&format!(
            "  ccmp {}, {}, #{}, {}\n",
            lhs, rhs, fallback_nzcv, condition
        ));
        true
    }

    fn fused_madd_user(&self, multiply: ValueId) -> Option<ValueId> {
        if self.value_use_counts[multiply.0] != 1 {
            return None;
        }
        let ValueKind::Inst(block, inst_idx) = self.func.value(multiply).kind else {
            return None;
        };
        let next = self.func.block(block).insts.get(inst_idx + 1)?;
        let result = next.result?;
        matches!(
            next.kind,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            } if lhs == multiply || rhs == multiply
        )
        .then_some(result)
    }

    fn fused_madd_operands(
        &self,
        add: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Option<(ValueId, ValueId, ValueId)> {
        let (multiply, addend) = if self.fused_madd_user(lhs) == Some(add) {
            (lhs, rhs)
        } else if self.fused_madd_user(rhs) == Some(add) {
            (rhs, lhs)
        } else {
            return None;
        };
        let ValueKind::Inst(block, inst_idx) = self.func.value(multiply).kind else {
            return None;
        };
        match self.func.block(block).insts.get(inst_idx)?.kind {
            InstKind::Binary {
                op: BinaryOp::Imul,
                lhs,
                rhs,
            } => Some((lhs, rhs, addend)),
            _ => None,
        }
    }

    fn fused_float_madd_user(&self, multiply: ValueId) -> Option<ValueId> {
        if self.value_use_counts[multiply.0] != 1 {
            return None;
        }
        let ValueKind::Inst(block, inst_idx) = self.func.value(multiply).kind else {
            return None;
        };
        let next = self.func.block(block).insts.get(inst_idx + 1)?;
        let result = next.result?;
        matches!(
            next.kind,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs,
                rhs,
            } if lhs == multiply || rhs == multiply
        )
        .then_some(result)
    }

    fn fused_float_madd_operands(
        &self,
        add: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Option<(ValueId, ValueId, ValueId)> {
        let (multiply, addend) = if self.fused_float_madd_user(lhs) == Some(add) {
            (lhs, rhs)
        } else if self.fused_float_madd_user(rhs) == Some(add) {
            (rhs, lhs)
        } else {
            return None;
        };
        let ValueKind::Inst(block, inst_idx) = self.func.value(multiply).kind else {
            return None;
        };
        match self.func.block(block).insts.get(inst_idx)?.kind {
            InstKind::Binary {
                op: BinaryOp::Fmul,
                lhs,
                rhs,
            } => Some((lhs, rhs, addend)),
            _ => None,
        }
    }

    fn fused_float_msub_user(&self, multiply: ValueId) -> Option<ValueId> {
        if self.value_use_counts[multiply.0] != 1 {
            return None;
        }
        let ValueKind::Inst(block, inst_idx) = self.func.value(multiply).kind else {
            return None;
        };
        let next = self.func.block(block).insts.get(inst_idx + 1)?;
        let result = next.result?;
        matches!(
            next.kind,
            InstKind::Binary {
                op: BinaryOp::Fsub,
                lhs,
                rhs,
            } if lhs == multiply || rhs == multiply
        )
        .then_some(result)
    }

    fn fused_float_msub_operands(
        &self,
        subtract: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Option<(&'static str, ValueId, ValueId, ValueId)> {
        let (instruction, multiply, addend) = if self.fused_float_msub_user(rhs) == Some(subtract) {
            // AArch64 FMSUB computes the accumulator minus the product.
            ("fmsub", rhs, lhs)
        } else if self.fused_float_msub_user(lhs) == Some(subtract) {
            // FNMSUB computes the product minus the accumulator.
            ("fnmsub", lhs, rhs)
        } else {
            return None;
        };
        let ValueKind::Inst(block, inst_idx) = self.func.value(multiply).kind else {
            return None;
        };
        match self.func.block(block).insts.get(inst_idx)?.kind {
            InstKind::Binary {
                op: BinaryOp::Fmul,
                lhs,
                rhs,
            } => Some((instruction, lhs, rhs, addend)),
            _ => None,
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

    fn emit_assigned_unary(&mut self, result: ValueId, op: UnaryOp, value: ValueId) -> bool {
        if op != UnaryOp::Fneg {
            return false;
        }
        let Some(destination) = self.assigned_float_reg(result) else {
            return false;
        };
        let source = self.load_or_assigned_float(value, "s0");
        self.body
            .push_str(&format!("  fneg {}, {}\n", destination, source));
        true
    }

    fn emit_assigned_binary_imm(
        &mut self,
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let Some(destination) = self.assigned_w_reg(result) else {
            return false;
        };
        match op {
            BinaryOp::Iadd => {
                if let Some((addend, shifted, shift)) = self.shifted_add_operands(result, lhs, rhs)
                {
                    if let (Some(addend), Some(shifted)) =
                        (self.assigned_w_reg(addend), self.assigned_w_reg(shifted))
                    {
                        self.body.push_str(&format!(
                            "  add {destination}, {addend}, {shifted}, lsl #{shift}\n"
                        ));
                        return true;
                    }
                }
                if let (Some(source), Some(imm)) = (
                    self.assigned_w_reg(lhs),
                    const_i32(self.func, rhs).filter(|imm| fits_addsub_imm(*imm)),
                ) {
                    self.body
                        .push_str(&format!("  add {}, {}, #{}\n", destination, source, imm));
                    return true;
                }
                if let (Some(source), Some(imm)) = (
                    self.assigned_w_reg(rhs),
                    const_i32(self.func, lhs).filter(|imm| fits_addsub_imm(*imm)),
                ) {
                    self.body
                        .push_str(&format!("  add {}, {}, #{}\n", destination, source, imm));
                    return true;
                }
            }
            BinaryOp::Isub => {
                if let (Some(source), Some(imm)) = (
                    self.assigned_w_reg(lhs),
                    const_i32(self.func, rhs).filter(|imm| fits_addsub_imm(*imm)),
                ) {
                    self.body
                        .push_str(&format!("  sub {}, {}, #{}\n", destination, source, imm));
                    return true;
                }
            }
            BinaryOp::Imul => {
                if let (Some(source), Some(shift)) =
                    (self.assigned_w_reg(lhs), pow2_shift(self.func, rhs))
                {
                    self.body
                        .push_str(&format!("  lsl {}, {}, #{}\n", destination, source, shift));
                    return true;
                }
                if let (Some(source), Some(shift)) =
                    (self.assigned_w_reg(rhs), pow2_shift(self.func, lhs))
                {
                    self.body
                        .push_str(&format!("  lsl {}, {}, #{}\n", destination, source, shift));
                    return true;
                }
            }
            BinaryOp::Iand => {
                let lhs = self.transparent_nonnegative_sanitize(lhs).unwrap_or(lhs);
                let rhs = self.transparent_nonnegative_sanitize(rhs).unwrap_or(rhs);
                if let (Some(source), Some(mask)) = (
                    self.assigned_w_reg(lhs),
                    const_i32(self.func, rhs).filter(|mask| is_low_bit_mask(*mask)),
                ) {
                    self.body
                        .push_str(&format!("  and {}, {}, #{}\n", destination, source, mask));
                    return true;
                }
            }
            BinaryOp::Idiv | BinaryOp::Imod => {
                if let Some(divisor) = const_i32(self.func, rhs) {
                    if op == BinaryOp::Imod
                        && self.emit_guarded_additive_remainder(result, lhs, divisor, &destination)
                    {
                        return true;
                    }
                    self.emit_signed_divmod_const_into(
                        lhs,
                        divisor,
                        op == BinaryOp::Imod,
                        &destination,
                    );
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    fn shifted_add_operands(
        &self,
        _result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Option<(ValueId, ValueId, i32)> {
        [lhs, rhs]
            .into_iter()
            .zip([rhs, lhs])
            .find_map(|(shifted_value, addend)| {
                let (shifted, shift) = match defining_inst_kind(self.func, shifted_value)? {
                    InstKind::Binary {
                        op: BinaryOp::Ishl,
                        lhs,
                        rhs,
                    } => (*lhs, const_i32(self.func, *rhs)?),
                    InstKind::Binary {
                        op: BinaryOp::Imul,
                        lhs,
                        rhs,
                    } => {
                        if let Some(shift) = pow2_shift(self.func, *rhs) {
                            (*lhs, shift as i32)
                        } else {
                            (*rhs, pow2_shift(self.func, *lhs)? as i32)
                        }
                    }
                    _ => return None,
                };
                let shift = (0..=4).contains(&shift).then_some(shift)?;
                (self.value_use_counts[shifted_value.0] == 1).then_some((addend, shifted, shift))
            })
    }

    fn fused_shifted_add_user(&self, shifted: ValueId) -> bool {
        let ValueKind::Inst(block, inst_idx) = self.func.value(shifted).kind else {
            return false;
        };
        self.func.block(block).insts[inst_idx + 1..]
            .iter()
            .filter_map(|inst| match (&inst.kind, inst.result) {
                (
                    InstKind::Binary {
                        op: BinaryOp::Iadd,
                        lhs,
                        rhs,
                    },
                    Some(result),
                ) => Some((result, *lhs, *rhs)),
                _ => None,
            })
            .any(|(result, lhs, rhs)| {
                if lhs != shifted && rhs != shifted {
                    return false;
                }
                // Integer MADD has priority when the same add consumes a
                // multiply.  In that case the shifted value remains an
                // ordinary addend and must still be materialized.  Also only
                // suppress the standalone shift when the register-only
                // shifted-add emitter can actually select the combined form.
                if self.fused_madd_operands(result, lhs, rhs).is_some() {
                    return false;
                }
                self.shifted_add_operands(result, lhs, rhs)
                    .is_some_and(|(addend, source, _)| {
                        self.assigned_w_reg(result).is_some()
                            && self.assigned_w_reg(addend).is_some()
                            && self.assigned_w_reg(source).is_some()
                    })
            })
    }

    /// Selects `x & ~((x >> 31))` as AArch64's shifted-register BIC. This is
    /// exactly `x` for non-negative i32 values and zero for negative values.
    /// Both intermediate values must be single-use so their standalone ASR and
    /// XOR instructions can be suppressed safely.
    fn negative_sanitize_operands(
        &self,
        sanitized: ValueId,
    ) -> Option<(ValueId, ValueId, ValueId)> {
        let InstKind::Binary {
            op: BinaryOp::Iand,
            lhs,
            rhs,
        } = defining_inst_kind(self.func, sanitized)?
        else {
            return None;
        };
        for (source, mask) in [(*lhs, *rhs), (*rhs, *lhs)] {
            let InstKind::Binary {
                op: BinaryOp::Ixor,
                lhs: xor_lhs,
                rhs: xor_rhs,
            } = defining_inst_kind(self.func, mask)?
            else {
                continue;
            };
            let sign = if const_i32(self.func, *xor_rhs) == Some(-1) {
                *xor_lhs
            } else if const_i32(self.func, *xor_lhs) == Some(-1) {
                *xor_rhs
            } else {
                continue;
            };
            let InstKind::Binary {
                op: BinaryOp::Iashr,
                lhs: shifted,
                rhs: shift,
            } = defining_inst_kind(self.func, sign)?
            else {
                continue;
            };
            if *shifted == source
                && const_i32(self.func, *shift) == Some(31)
                && ((self.value_use_counts[sign.0] == 1 && self.value_use_counts[mask.0] == 1)
                    || self.is_proven_nonnegative(source))
            {
                return Some((source, sign, mask));
            }
        }
        None
    }

    fn is_negative_sanitize_intermediate(&self, value: ValueId) -> bool {
        let ValueKind::Inst(block, inst_idx) = self.func.value(value).kind else {
            return false;
        };
        self.func.blocks[block.0].insts
            [inst_idx + 1..(inst_idx + 4).min(self.func.blocks[block.0].insts.len())]
            .iter()
            .filter_map(|inst| inst.result)
            .any(|candidate| {
                self.negative_sanitize_operands(candidate)
                    .is_some_and(|(source, sign, mask)| {
                        (value == sign || value == mask)
                            && (self.value_use_counts[value.0] == 1
                                || (self.is_proven_nonnegative(source)
                                    && self.fused_nonnegative_shift_user(value)))
                    })
            })
    }

    fn is_nonnegative_sanitize_intermediate(&self, value: ValueId) -> bool {
        let ValueKind::Inst(block, inst_idx) = self.func.value(value).kind else {
            return false;
        };
        self.func.blocks[block.0].insts
            [inst_idx + 1..(inst_idx + 4).min(self.func.blocks[block.0].insts.len())]
            .iter()
            .filter_map(|inst| inst.result)
            .any(|candidate| {
                self.negative_sanitize_operands(candidate)
                    .is_some_and(|(source, sign, mask)| {
                        (value == sign || value == mask) && self.is_proven_nonnegative(source)
                    })
            })
    }

    fn emit_negative_sanitize(&mut self, result: ValueId) -> bool {
        let Some((source, _, _)) = self.negative_sanitize_operands(result) else {
            return false;
        };
        let source_is_nonnegative = self.is_proven_nonnegative(source);
        if source_is_nonnegative
            && self.value_use_counts[result.0] == 1
            && self.has_masked_transparent_sanitize_user(result)
        {
            return true;
        }
        let assigned_destination = self.assigned_w_reg(result);
        let destination = assigned_destination.clone().unwrap_or_else(|| "w0".into());
        let source = if let Some(source) = self.assigned_w_reg(source) {
            source
        } else {
            self.load_value_into(source, "x1");
            "w1".into()
        };
        if source_is_nonnegative {
            if destination != source {
                self.body
                    .push_str(&format!("  mov {destination}, {source}\n"));
            }
        } else {
            self.body.push_str(&format!(
                "  bic {destination}, {source}, {source}, asr #31\n"
            ));
        }
        if assigned_destination.is_none() {
            self.store_result(result);
        }
        true
    }

    fn transparent_nonnegative_sanitize(&self, value: ValueId) -> Option<ValueId> {
        let (source, _, _) = self.negative_sanitize_operands(value)?;
        self.is_proven_nonnegative(source).then_some(source)
    }

    fn has_masked_transparent_sanitize_user(&self, sanitized: ValueId) -> bool {
        self.func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| match inst.kind {
                InstKind::Binary {
                    op: BinaryOp::Iand,
                    lhs,
                    rhs,
                } if lhs == sanitized => const_i32(self.func, rhs).is_some_and(|mask| mask >= 0),
                InstKind::Binary {
                    op: BinaryOp::Iand,
                    lhs,
                    rhs,
                } if rhs == sanitized => const_i32(self.func, lhs).is_some_and(|mask| mask >= 0),
                _ => false,
            })
        })
    }

    fn is_proven_nonnegative(&self, value: ValueId) -> bool {
        super::emitter::value_is_nonnegative(
            self.func,
            value,
            &self.parent.nonnegative_globals,
            &mut HashSet::new(),
        )
    }

    fn nonnegative_adjusted_shift_plan(&self, result: ValueId) -> Option<NonnegativeShiftPlan> {
        let ValueKind::Inst(owner, _) = self.func.value(result).kind else {
            return None;
        };
        let InstKind::Binary {
            op: BinaryOp::Iashr,
            lhs: adjusted,
            rhs: shift,
        } = defining_inst_kind(self.func, result)?
        else {
            return None;
        };
        let InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        } = defining_inst_kind(self.func, *adjusted)?
        else {
            return None;
        };

        for (source, adjustment) in [(*lhs, *rhs), (*rhs, *lhs)] {
            let Some((sign, mask)) = signed_division_adjustment(self.func, source, adjustment)
            else {
                continue;
            };
            if !self.is_proven_nonnegative(source)
                || self.value_use_counts[adjusted.0] != 1
                || self.value_use_counts[adjustment.0] != 1
            {
                continue;
            }
            let mut intermediates = vec![*adjusted, adjustment];
            if matches!(self.func.value(sign).kind, ValueKind::Inst(block, _) if block == owner)
                && (self.value_use_counts[sign.0] == 1
                    || (self.value_use_counts[sign.0] == 2
                        && self.is_nonnegative_sanitize_intermediate(sign)))
            {
                intermediates.push(sign);
            }
            if let Some(mask) = mask {
                collect_single_use_shift_tree(
                    self.func,
                    &self.value_use_counts,
                    mask,
                    owner,
                    &mut intermediates,
                    &mut HashSet::new(),
                );
            }
            return Some(NonnegativeShiftPlan {
                source,
                shift: *shift,
                intermediates,
            });
        }
        None
    }

    fn fused_nonnegative_shift_user(&self, value: ValueId) -> bool {
        let ValueKind::Inst(block, inst_idx) = self.func.value(value).kind else {
            return false;
        };
        self.func.block(block).insts[inst_idx + 1..]
            .iter()
            .filter_map(|inst| inst.result)
            .any(|candidate| {
                self.nonnegative_adjusted_shift_plan(candidate)
                    .is_some_and(|plan| plan.intermediates.contains(&value))
            })
    }

    fn emit_fused_nonnegative_adjusted_shift(&mut self, result: ValueId) -> bool {
        let Some(plan) = self.nonnegative_adjusted_shift_plan(result) else {
            return false;
        };
        let source = self.assigned_w_reg(plan.source).unwrap_or_else(|| {
            self.load_value_into(plan.source, "x1");
            "w1".to_string()
        });
        let shift = if let Some(shift) =
            const_i32(self.func, plan.shift).filter(|shift| (0..32).contains(shift))
        {
            format!("#{shift}")
        } else {
            self.assigned_w_reg(plan.shift).unwrap_or_else(|| {
                self.load_value_into(plan.shift, "x2");
                "w2".to_string()
            })
        };
        let assigned = self.assigned_w_reg(result);
        let destination = assigned.clone().unwrap_or_else(|| "w0".to_string());
        self.body
            .push_str(&format!("  lsr {destination}, {source}, {shift}\n"));
        if assigned.is_none() {
            self.store_result(result);
        }
        true
    }

    fn even_halving_plan(&self, division: ValueId) -> Option<EvenHalvingPlan> {
        let ValueKind::Inst(body, _) = self.func.value(division).kind else {
            return None;
        };
        let InstKind::Binary {
            op: BinaryOp::Idiv,
            lhs: state,
            rhs: divisor,
        } = defining_inst_kind(self.func, division)?
        else {
            return None;
        };
        if const_i32(self.func, *divisor) != Some(2) {
            return None;
        }
        let Terminator::Jump(header) = self.func.block(body).terminator.as_ref()? else {
            return None;
        };
        let phis = self
            .func
            .block(*header)
            .insts
            .iter()
            .filter_map(|inst| {
                let result = inst.result?;
                matches!(inst.kind, InstKind::Phi { .. }).then_some(result)
            })
            .collect::<Vec<_>>();
        if phis.len() != 2
            || phis.iter().any(|phi| self.func.value(*phi).ty != Type::I32)
            || !phis.contains(state)
        {
            return None;
        }
        let depth = phis.iter().copied().find(|phi| phi != state)?;
        if phi_incoming(self.func, *state, body) != Some(division) {
            return None;
        }
        let add = self.func.block(body).insts.iter().find_map(|inst| {
            let (
                Some(result),
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs,
                    rhs,
                },
            ) = (inst.result, &inst.kind)
            else {
                return None;
            };
            ((*lhs == depth && const_i32(self.func, *rhs) == Some(1))
                || (*rhs == depth && const_i32(self.func, *lhs) == Some(1)))
            .then_some(result)
        })?;
        if phi_incoming(self.func, depth, body) != Some(add)
            || self.func.block(body).insts.iter().any(|inst| {
                !matches!(inst.kind, InstKind::Nop)
                    && !matches!(inst.result, Some(result) if result == division || result == add)
            })
        {
            return None;
        }

        let header_active = self
            .func
            .block(*header)
            .insts
            .iter()
            .filter(|inst| !matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }))
            .collect::<Vec<_>>();
        let [compare] = header_active.as_slice() else {
            return None;
        };
        let (Some(condition), InstKind::Icmp { op, lhs, rhs }) = (compare.result, &compare.kind)
        else {
            return None;
        };
        if !matches!(op, CmpOp::Eq | CmpOp::Ne)
            || !((*lhs == *state && const_i32(self.func, *rhs) == Some(1))
                || (*rhs == *state && const_i32(self.func, *lhs) == Some(1)))
            || !matches!(
                self.func.block(*header).terminator,
                Some(Terminator::Branch { cond, .. }) if cond == condition
            )
        {
            return None;
        }

        if [*state, depth, division, add]
            .into_iter()
            .any(|value| self.assigned_w_reg(value).is_none())
        {
            return None;
        }

        let known_even_entry = self.predecessor_count(body) == 1
            && self.func.blocks.iter().any(|block| {
                let Some(Terminator::Branch {
                    cond,
                    then_target,
                    else_target,
                }) = block.terminator.as_ref()
                else {
                    return false;
                };
                let Some((op, value, mask, _)) = self.direct_branch_bit_test(*cond) else {
                    return false;
                };
                value == *state
                    && mask == 1
                    && ((op == CmpOp::Eq && *then_target == body)
                        || (op == CmpOp::Ne && *else_target == body))
            });

        Some(EvenHalvingPlan {
            header: *header,
            body,
            state: *state,
            depth,
            division,
            add,
            known_even_entry,
        })
    }

    fn is_compressed_halving_add(&self, add: ValueId) -> bool {
        let ValueKind::Inst(block, _) = self.func.value(add).kind else {
            return false;
        };
        self.func.block(block).insts.iter().any(|inst| {
            inst.result
                .and_then(|result| self.even_halving_plan(result))
                .is_some_and(|plan| plan.add == add)
        })
    }

    fn emit_even_halving_run(&mut self, plan: EvenHalvingPlan) {
        let (Some(state), Some(depth), Some(division), Some(add)) = (
            self.assigned_w_reg(plan.state),
            self.assigned_w_reg(plan.depth),
            self.assigned_w_reg(plan.division),
            self.assigned_w_reg(plan.add),
        ) else {
            unreachable!("halving plan requires assigned loop registers");
        };
        if plan.known_even_entry {
            self.body.push_str(&format!(
                "  rbit w2, {state}\n  clz w2, w2\n  asr {division}, {state}, w2\n  add {add}, {depth}, w2\n"
            ));
        } else {
            let odd = self.parent.ctx.fresh_label("halving_odd");
            let done = self.parent.ctx.fresh_label("halving_done");
            self.body.push_str(&format!(
                "  tbnz {state}, #0, {odd}\n  rbit w2, {state}\n  clz w2, w2\n  asr {division}, {state}, w2\n  add {add}, {depth}, w2\n  b {done}\n{odd}:\n  add {division}, {state}, {state}, lsr #31\n  asr {division}, {division}, #1\n  add {add}, {depth}, #1\n{done}:\n"
            ));
        }
    }

    /// Folds an odd affine transition directly into the following pure
    /// halving loop. For example, an odd `n` makes `3*n+1` even; entering the
    /// header only to re-test `n != 1` and its low bit is redundant. The
    /// matcher proves the odd edge, the affine result parity, the exact phi
    /// inputs, and that the skipped header/guard/body contain no other work.
    fn emit_known_even_backedge(&mut self, pred_idx: usize, header: BlockId) -> bool {
        let pred = BlockId(pred_idx);
        let plan = self
            .func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .find_map(|inst| {
                let result = inst.result?;
                self.even_halving_plan(result)
                    .filter(|plan| plan.header == header && plan.body != pred)
            });
        let Some(plan) = plan else {
            return false;
        };
        let (Some(state_incoming), Some(depth_incoming)) = (
            phi_incoming(self.func, plan.state, pred),
            phi_incoming(self.func, plan.depth, pred),
        ) else {
            return false;
        };
        if parity_given_odd(self.func, state_incoming, plan.state, 0) != Some(false) {
            return false;
        }

        let header_condition =
            self.func
                .block(header)
                .insts
                .iter()
                .find_map(|inst| match (inst.result, &inst.kind) {
                    (Some(result), InstKind::Icmp { op, lhs, rhs })
                        if matches!(op, CmpOp::Eq | CmpOp::Ne)
                            && ((*lhs == plan.state && const_i32(self.func, *rhs) == Some(1))
                                || (*rhs == plan.state
                                    && const_i32(self.func, *lhs) == Some(1))) =>
                    {
                        Some((result, *op))
                    }
                    _ => None,
                });
        let Some((header_condition, header_op)) = header_condition else {
            return false;
        };
        let Some(Terminator::Branch {
            cond,
            then_target,
            else_target,
        }) = self.func.block(header).terminator.as_ref()
        else {
            return false;
        };
        if *cond != header_condition {
            return false;
        }
        let (parity_block, header_exit) = if header_op == CmpOp::Ne {
            (*then_target, *else_target)
        } else {
            (*else_target, *then_target)
        };
        let Some(Terminator::Branch {
            cond: parity_condition,
            then_target,
            else_target,
        }) = self.func.block(parity_block).terminator.as_ref()
        else {
            return false;
        };
        let Some((parity_op, parity_value, mask, masked)) =
            self.direct_branch_bit_test(*parity_condition)
        else {
            return false;
        };
        if parity_value != plan.state
            || mask != 1
            || self.func.block(parity_block).insts.iter().any(|inst| {
                !matches!(inst.kind, InstKind::Nop)
                    && !matches!(inst.result, Some(result) if result == masked || result == *parity_condition)
            })
        {
            return false;
        }
        let (even_target, odd_target) = if parity_op == CmpOp::Eq {
            (*then_target, *else_target)
        } else {
            (*else_target, *then_target)
        };
        if even_target != plan.body || !unique_forward_path(self.func, odd_target, pred, 8) {
            return false;
        }

        let (Some(state), Some(depth), Some(state_source), Some(depth_source)) = (
            self.assigned_w_reg(plan.state),
            self.assigned_w_reg(plan.depth),
            self.assigned_w_reg(state_incoming),
            self.assigned_w_reg(depth_incoming),
        ) else {
            return false;
        };
        if state == depth_source {
            return false;
        }

        let exit_edge = self.parent.ctx.fresh_label("halving_exit_edge");
        self.body.push_str(&format!(
            "  rbit w2, {state_source}\n  clz w2, w2\n  asr {state}, {state_source}, w2\n  add {depth}, {depth_source}, w2\n  cmp {state}, #1\n"
        ));
        if !self.edge_has_phi_copy(parity_block.0, odd_target.0)
            && !self.edge_has_phi_copy(header.0, header_exit.0)
        {
            self.body.push_str(&format!(
                "  b.eq {}\n  b {}\n",
                self.block_label(header_exit.0),
                self.block_label(odd_target.0)
            ));
            return true;
        }
        self.body.push_str(&format!("  b.eq {exit_edge}\n"));
        self.emit_phi_copies(parity_block.0, odd_target.0);
        self.body.push_str(&format!(
            "  b {}\n{exit_edge}:\n",
            self.block_label(odd_target.0)
        ));
        self.emit_phi_copies(header.0, header_exit.0);
        self.body
            .push_str(&format!("  b {}\n", self.block_label(header_exit.0)));
        true
    }

    fn emit_guarded_additive_remainder(
        &mut self,
        result: ValueId,
        dividend: ValueId,
        divisor: i32,
        destination: &str,
    ) -> bool {
        let Some(twice_divisor) = divisor
            .checked_mul(2)
            .filter(|twice| divisor >= 1024 && *twice > divisor)
        else {
            return false;
        };
        if !self.is_additive_modular_recurrence(result, dividend) {
            return false;
        }

        let source = if let Some(source) = self.assigned_w_reg(dividend) {
            source
        } else {
            self.load_value(dividend);
            "w0".to_string()
        };
        let fast = self.parent.ctx.fresh_label("mod_fast");
        let done = self.parent.ctx.fresh_label("mod_done");
        let cached_divisor = self
            .cached_i32_constant
            .filter(|(value, _)| *value == divisor)
            .map(|(_, reg)| w_reg_name(reg));
        let cached_twice_divisor = self
            .cached_i32_double_constant
            .filter(|(value, _)| *value == twice_divisor)
            .map(|(_, reg)| w_reg_name(reg));
        if let Some(twice_divisor_reg) = &cached_twice_divisor {
            self.body
                .push_str(&format!("  cmp {source}, {twice_divisor_reg}\n"));
        } else if let Some(divisor_reg) = &cached_divisor {
            self.body
                .push_str(&format!("  add w2, {0}, {0}\n", divisor_reg));
            self.body.push_str(&format!("  cmp {source}, w2\n"));
        } else {
            self.body.push_str(&mov_w_imm("w2", twice_divisor));
            self.body.push_str(&format!("  cmp {source}, w2\n"));
        }
        self.body.push_str(&format!("  b.lo {fast}\n"));
        self.emit_signed_divmod_const_into(dividend, divisor, true, destination);
        self.body.push_str(&format!("  b {done}\n{fast}:\n"));
        let divisor_reg = if let Some(divisor_reg) = &cached_divisor {
            divisor_reg.as_str()
        } else {
            self.body.push_str(&mov_w_imm("w1", divisor));
            "w1"
        };
        self.body.push_str(&format!(
            "  subs w2, {source}, {divisor_reg}\n  csel {destination}, w2, {source}, hs\n"
        ));
        self.body.push_str(&format!("{done}:\n"));
        true
    }

    fn is_additive_modular_recurrence(&self, remainder: ValueId, dividend: ValueId) -> bool {
        let Some(InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        }) = defining_inst_kind(self.func, dividend)
        else {
            return false;
        };
        self.func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                let (Some(phi), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
                    return false;
                };
                (*lhs == phi || *rhs == phi)
                    && incomings.iter().any(|(_, incoming)| *incoming == remainder)
            })
        })
    }

    fn emit_assigned_binary(
        &mut self,
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        if matches!(
            op,
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv
        ) {
            let Some(destination) = self.assigned_float_reg(result) else {
                return false;
            };
            let lhs = self.load_or_assigned_float(lhs, "s1");
            let rhs = self.load_or_assigned_float(rhs, "s0");
            let instruction = match op {
                BinaryOp::Fadd => "fadd",
                BinaryOp::Fsub => "fsub",
                BinaryOp::Fmul => "fmul",
                BinaryOp::Fdiv => "fdiv",
                _ => unreachable!(),
            };
            self.body.push_str(&format!(
                "  {} {}, {}, {}\n",
                instruction, destination, lhs, rhs
            ));
            return true;
        }
        let (Some(destination), Some(lhs), Some(rhs)) = (
            self.assigned_w_reg(result),
            self.assigned_w_reg(lhs),
            self.assigned_w_reg(rhs),
        ) else {
            return false;
        };
        let instruction = match op {
            BinaryOp::Iadd => "add",
            BinaryOp::Isub => "sub",
            BinaryOp::Imul => "mul",
            BinaryOp::Idiv => "sdiv",
            BinaryOp::Iand | BinaryOp::And => "and",
            BinaryOp::Ior | BinaryOp::Or => "orr",
            BinaryOp::Ixor => "eor",
            BinaryOp::Ishl => "lsl",
            BinaryOp::Iashr => "asr",
            BinaryOp::Imod => {
                self.body.push_str(&format!(
                    "  sdiv w2, {}, {}\n  msub {}, w2, {}, {}\n",
                    lhs, rhs, destination, rhs, lhs
                ));
                return true;
            }
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => return false,
        };
        self.body.push_str(&format!(
            "  {} {}, {}, {}\n",
            instruction, destination, lhs, rhs
        ));
        true
    }

    fn load_or_assigned_float(&mut self, value: ValueId, scratch: &'static str) -> &'static str {
        if let Some(reg) = self.assigned_float_reg(value) {
            reg
        } else {
            self.load_float_value(value, scratch);
            scratch
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) {
        if self.emit_binary_imm(op, lhs, rhs) {
            return;
        }

        match op {
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => {
                self.load_float_value(lhs, "s1");
                self.load_float_value(rhs, "s0");
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
                self.load_value_into(lhs, "x1");
                self.load_value(rhs);
                if op == BinaryOp::And {
                    self.body.push_str("  and w0, w1, w0\n");
                } else {
                    self.body.push_str("  orr w0, w1, w0\n");
                }
            }
            _ => {
                self.load_value_into(lhs, "x1");
                self.load_value(rhs);
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
                    BinaryOp::Iand => self.body.push_str("  and w0, w1, w0\n"),
                    BinaryOp::Ior => self.body.push_str("  orr w0, w1, w0\n"),
                    BinaryOp::Ixor => self.body.push_str("  eor w0, w1, w0\n"),
                    BinaryOp::Ishl => self.body.push_str("  lsl w0, w1, w0\n"),
                    BinaryOp::Iashr => self.body.push_str("  asr w0, w1, w0\n"),
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
            BinaryOp::Iand => {
                if let Some(mask) = const_i32(self.func, rhs).filter(|mask| is_low_bit_mask(*mask))
                {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  and w0, w0, #{}\n", mask));
                    return true;
                }
            }
            BinaryOp::Idiv | BinaryOp::Imod => {
                if let Some(divisor) = const_i32(self.func, rhs) {
                    self.emit_signed_divmod_const_into(lhs, divisor, op == BinaryOp::Imod, "w0");
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

    fn emit_assigned_signed_divmod_pow2(
        &mut self,
        destination: &str,
        source: &str,
        shift: u32,
        remainder: bool,
    ) {
        if shift == 0 {
            if remainder {
                self.body.push_str(&format!("  mov {}, wzr\n", destination));
            } else if destination != source {
                self.body
                    .push_str(&format!("  mov {}, {}\n", destination, source));
            }
            return;
        }

        if remainder {
            self.body.push_str(&format!("  cmp {}, #0\n", source));
            if shift == 1 {
                self.body.push_str(&format!(
                    "  and {}, {}, #1\n  cneg {}, {}, lt\n",
                    destination, source, destination, destination
                ));
            } else {
                let mask = (1u32 << shift) - 1;
                self.body.push_str(&format!(
                    "  cneg w2, {}, lt\n  and {}, w2, #{}\n  cneg {}, {}, lt\n",
                    source, destination, mask, destination, destination
                ));
            }
            return;
        }

        if shift == 1 {
            self.body.push_str(&format!(
                "  add {}, {}, {}, lsr #31\n  asr {}, {}, #1\n",
                destination, source, source, destination, destination
            ));
        } else {
            self.body.push_str(&format!(
                "  asr w2, {}, #31\n  add {}, {}, w2, lsr #{}\n  asr {}, {}, #{}\n",
                source,
                destination,
                source,
                32 - shift,
                destination,
                destination,
                shift
            ));
        }
    }

    fn emit_signed_divmod_const_into(
        &mut self,
        value: ValueId,
        divisor: i32,
        remainder: bool,
        destination: &str,
    ) {
        let source = if let Some(source) = self.assigned_w_reg(value) {
            source
        } else {
            self.load_value(value);
            "w0".to_string()
        };

        if divisor == 0 {
            if remainder {
                if destination != source {
                    self.body
                        .push_str(&format!("  mov {}, {}\n", destination, source));
                }
            } else {
                self.body.push_str(&format!("  mov {}, wzr\n", destination));
            }
            return;
        }
        if divisor == 1 || divisor == -1 {
            if remainder {
                self.body.push_str(&format!("  mov {}, wzr\n", destination));
            } else if divisor < 0 {
                self.body
                    .push_str(&format!("  neg {}, {}\n", destination, source));
            } else if destination != source {
                self.body
                    .push_str(&format!("  mov {}, {}\n", destination, source));
            }
            return;
        }

        if remainder && self.emit_narrow_signed_remainder(value, &source, divisor, destination) {
            return;
        }

        let abs_divisor = divisor.unsigned_abs();
        if abs_divisor.is_power_of_two() {
            if divisor > 0 && !remainder && self.is_proven_nonnegative(value) {
                self.body.push_str(&format!(
                    "  lsr {}, {}, #{}\n",
                    destination,
                    source,
                    abs_divisor.trailing_zeros()
                ));
                return;
            }
            self.emit_assigned_signed_divmod_pow2(
                destination,
                &source,
                abs_divisor.trailing_zeros(),
                remainder,
            );
            if divisor < 0 && !remainder {
                self.body
                    .push_str(&format!("  neg {}, {}\n", destination, destination));
            }
            return;
        }

        if remainder {
            self.body.push_str(&format!("  mov w3, {}\n", source));
        }
        let magic = signed_magic_positive(abs_divisor);
        self.body.push_str(&mov_w_imm("w1", magic.multiplier));
        self.body
            .push_str(&format!("  smull x2, {}, w1\n  asr x2, x2, #32\n", source));
        if magic.add_dividend {
            self.body.push_str(&format!("  add w2, w2, {}\n", source));
        }
        if magic.shift != 0 {
            self.body
                .push_str(&format!("  asr w2, w2, #{}\n", magic.shift));
        }
        self.body.push_str("  add w2, w2, w2, lsr #31\n");

        if divisor < 0 && !remainder {
            self.body.push_str("  neg w2, w2\n");
        }
        if remainder {
            if abs_divisor.checked_add(1).is_some_and(u32::is_power_of_two) {
                self.body.push_str(&format!(
                    "  lsl w1, w2, #{}\n  sub w2, w1, w2\n",
                    (abs_divisor + 1).trailing_zeros()
                ));
            } else {
                self.body.push_str(&mov_w_imm("w1", abs_divisor as i32));
                self.body.push_str("  mul w2, w2, w1\n");
            }
            self.body
                .push_str(&format!("  sub {}, w3, w2\n", destination));
        } else if destination != "w2" {
            self.body.push_str(&format!("  mov {}, w2\n", destination));
        }
    }

    fn emit_narrow_signed_remainder(
        &mut self,
        value: ValueId,
        source: &str,
        divisor: i32,
        destination: &str,
    ) -> bool {
        let Some(range) = self.int_ranges[value.0] else {
            return false;
        };
        let bound = i64::from(divisor).abs();
        if bound <= 1 || bound > i64::from(i32::MAX) {
            return false;
        }
        if range.min > -bound && range.max < bound {
            if source != destination {
                self.body
                    .push_str(&format!("  mov {}, {}\n", destination, source));
            }
            return true;
        }
        if (bound as u32).is_power_of_two() || range.min <= -2 * bound || range.max >= 2 * bound {
            return false;
        }

        if range.min > -bound {
            self.body.push_str(&mov_w_imm("w1", bound as i32));
            self.body.push_str(&format!(
                "  cmp {source}, w1\n  sub w2, {source}, w1\n  csel {destination}, w2, {source}, ge\n"
            ));
            return true;
        }
        if range.max < bound {
            self.body.push_str(&mov_w_imm("w1", -(bound as i32)));
            self.body.push_str(&format!(
                "  cmp {source}, w1\n  sub w2, {source}, w1\n  csel {destination}, w2, {source}, le\n"
            ));
            return true;
        }

        self.body.push_str(&mov_w_imm("w1", bound as i32));
        self.body.push_str(&format!(
            "  cmp {source}, w1\n  sub w2, {source}, w1\n  csel {destination}, w2, {source}, ge\n  neg w1, w1\n  cmp {source}, w1\n  sub w2, {source}, w1\n  csel {destination}, w2, {destination}, le\n"
        ));
        true
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

    fn emit_int_compare(&mut self, lhs: ValueId, rhs: ValueId) {
        if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_addsub_imm(*imm)) {
            if let Some(lhs) = self.assigned_w_reg(lhs) {
                self.body.push_str(&format!("  cmp {}, #{}\n", lhs, imm));
            } else {
                self.load_value(lhs);
                self.body.push_str(&format!("  cmp w0, #{}\n", imm));
            }
            return;
        }

        let is_pointer = matches!(self.func.value(lhs).ty, Type::Ptr(_));
        match (self.assigned_x_reg(lhs), self.assigned_x_reg(rhs)) {
            (Some(lhs), Some(rhs)) => {
                let (lhs, rhs) = if is_pointer {
                    (lhs.to_string(), rhs.to_string())
                } else {
                    (w_reg_name(lhs), w_reg_name(rhs))
                };
                self.body.push_str(&format!("  cmp {}, {}\n", lhs, rhs));
            }
            (Some(lhs), None) => {
                let lhs = if is_pointer {
                    lhs.to_string()
                } else {
                    w_reg_name(lhs)
                };
                self.load_value(rhs);
                self.body.push_str(&format!(
                    "  cmp {}, {}0\n",
                    lhs,
                    if is_pointer { "x" } else { "w" }
                ));
            }
            (None, Some(rhs)) => {
                let rhs = if is_pointer {
                    rhs.to_string()
                } else {
                    w_reg_name(rhs)
                };
                self.load_value(lhs);
                self.body.push_str(&format!(
                    "  cmp {}0, {}\n",
                    if is_pointer { "x" } else { "w" },
                    rhs
                ));
            }
            (None, None) => {
                self.load_value_into(lhs, "x1");
                self.load_value(rhs);
                self.body.push_str(if is_pointer {
                    "  cmp x1, x0\n"
                } else {
                    "  cmp w1, w0\n"
                });
            }
        }
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
        self.load_float_value(lhs, "s1");
        self.load_float_value(rhs, "s0");
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

    fn emit_assigned_cast(&mut self, result: ValueId, op: CastOp, value: ValueId) -> bool {
        if op == CastOp::I32ToF32 {
            let Some(destination) = self.assigned_float_reg(result) else {
                return false;
            };
            let source = if let Some(source) = self.assigned_w_reg(value) {
                source
            } else {
                self.load_value(value);
                "w0".to_string()
            };
            self.body
                .push_str(&format!("  scvtf {}, {}\n", destination, source));
            return true;
        }
        if matches!(op, CastOp::F32ToI32 | CastOp::F32ToBool) {
            let Some(destination) = self.assigned_w_reg(result) else {
                return false;
            };
            let source = self.load_or_assigned_float(value, "s0");
            match op {
                CastOp::F32ToI32 => self
                    .body
                    .push_str(&format!("  fcvtzs {}, {}\n", destination, source)),
                CastOp::F32ToBool => self.body.push_str(&format!(
                    "  fcmp {}, #0.0\n  cset {}, ne\n",
                    source, destination
                )),
                _ => unreachable!(),
            }
            return true;
        }
        let (Some(destination), Some(source)) =
            (self.assigned_w_reg(result), self.assigned_w_reg(value))
        else {
            return false;
        };
        match op {
            CastOp::BoolToI32 => {
                if destination != source {
                    self.body
                        .push_str(&format!("  mov {}, {}\n", destination, source));
                }
                true
            }
            CastOp::I32ToBool => {
                self.body.push_str(&format!(
                    "  cmp {}, #0\n  cset {}, ne\n",
                    source, destination
                ));
                true
            }
            _ => false,
        }
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

fn w_reg_name(x_reg: &str) -> String {
    x_reg.replacen('x', "w", 1)
}

fn defining_inst_kind(func: &crate::ir::Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return None;
    };
    func.blocks
        .get(block.0)?
        .insts
        .get(inst_idx)
        .map(|inst| &inst.kind)
}

fn const_i32(func: &crate::ir::Function, value: ValueId) -> Option<i32> {
    match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        ValueKind::Const(Const::Bool(value)) => Some(*value as i32),
        _ => None,
    }
}

fn phi_copy_constant(func: &crate::ir::Function, value: ValueId) -> Option<i32> {
    match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        ValueKind::Const(Const::Bool(value)) => Some(*value as i32),
        ValueKind::Const(Const::Zero(Type::I1 | Type::I32 | Type::Ptr(_))) => Some(0),
        _ => None,
    }
}

fn signed_division_adjustment(
    func: &crate::ir::Function,
    source: ValueId,
    adjustment: ValueId,
) -> Option<(ValueId, Option<ValueId>)> {
    let is_sign = |candidate: ValueId| {
        matches!(
            defining_inst_kind(func, candidate),
            Some(InstKind::Binary {
                op: BinaryOp::Iashr,
                lhs,
                rhs,
            }) if *lhs == source && const_i32(func, *rhs) == Some(31)
        )
    };
    if is_sign(adjustment) {
        return Some((adjustment, None));
    }
    let InstKind::Binary {
        op: BinaryOp::Iand,
        lhs,
        rhs,
    } = defining_inst_kind(func, adjustment)?
    else {
        return None;
    };
    if is_sign(*lhs) {
        Some((*lhs, Some(*rhs)))
    } else if is_sign(*rhs) {
        Some((*rhs, Some(*lhs)))
    } else {
        None
    }
}

fn collect_single_use_shift_tree(
    func: &crate::ir::Function,
    use_counts: &[usize],
    value: ValueId,
    owner: BlockId,
    values: &mut Vec<ValueId>,
    visited: &mut HashSet<ValueId>,
) {
    if !visited.insert(value) || use_counts[value.0] != 1 {
        return;
    }
    let ValueKind::Inst(block, _) = func.value(value).kind else {
        return;
    };
    if block != owner {
        return;
    }
    let Some(InstKind::Binary { op, lhs, rhs }) = defining_inst_kind(func, value) else {
        return;
    };
    if !matches!(
        op,
        BinaryOp::Iadd | BinaryOp::Isub | BinaryOp::Iand | BinaryOp::Ishl
    ) {
        return;
    }
    values.push(value);
    collect_single_use_shift_tree(func, use_counts, *lhs, owner, values, visited);
    collect_single_use_shift_tree(func, use_counts, *rhs, owner, values, visited);
}

fn pow2_shift(func: &crate::ir::Function, value: ValueId) -> Option<u32> {
    positive_pow2_shift(const_i32(func, value)?)
}

fn parity_given_odd(
    func: &crate::ir::Function,
    value: ValueId,
    odd_value: ValueId,
    depth: usize,
) -> Option<bool> {
    if depth > 8 {
        return None;
    }
    if value == odd_value {
        return Some(true);
    }
    if let Some(constant) = const_i32(func, value) {
        return Some(constant & 1 != 0);
    }
    let InstKind::Binary { op, lhs, rhs } = defining_inst_kind(func, value)? else {
        return None;
    };
    let lhs = parity_given_odd(func, *lhs, odd_value, depth + 1)?;
    let rhs = parity_given_odd(func, *rhs, odd_value, depth + 1)?;
    match op {
        BinaryOp::Iadd | BinaryOp::Isub | BinaryOp::Ixor => Some(lhs ^ rhs),
        BinaryOp::Imul | BinaryOp::Iand => Some(lhs & rhs),
        BinaryOp::Ior => Some(lhs | rhs),
        _ => None,
    }
}

fn unique_forward_path(
    func: &crate::ir::Function,
    start: BlockId,
    target: BlockId,
    limit: usize,
) -> bool {
    let mut current = target;
    for _ in 0..=limit {
        if current == start {
            return true;
        }
        let mut predecessors = func
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(block_idx, block)| match block.terminator.as_ref() {
                Some(Terminator::Jump(successor)) if *successor == current => {
                    Some(BlockId(block_idx))
                }
                Some(Terminator::Branch {
                    then_target,
                    else_target,
                    ..
                }) if *then_target == current || *else_target == current => {
                    Some(BlockId(block_idx))
                }
                _ => None,
            });
        let Some(predecessor) = predecessors.next() else {
            return false;
        };
        if predecessors.next().is_some() {
            return false;
        }
        current = predecessor;
    }
    false
}

fn positive_pow2_shift(value: i32) -> Option<u32> {
    (value > 0 && (value & (value - 1)) == 0).then_some(value.trailing_zeros())
}

fn fits_addsub_imm(value: i32) -> bool {
    (0..=4095).contains(&value)
}

fn is_low_bit_mask(value: i32) -> bool {
    value > 0 && ((value as u32) & (value as u32).wrapping_add(1)) == 0
}

fn collect_condition_load_pointers(
    func: &crate::ir::Function,
    value: ValueId,
    visited: &mut HashSet<ValueId>,
    pointers: &mut Vec<ValueId>,
) {
    if !visited.insert(value) {
        return;
    }
    let Some(kind) = defining_inst_kind(func, value) else {
        return;
    };
    match kind {
        InstKind::Load { ptr } => pointers.push(*ptr),
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => {
            collect_condition_load_pointers(func, *value, visited, pointers);
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            collect_condition_load_pointers(func, *lhs, visited, pointers);
            collect_condition_load_pointers(func, *rhs, visited, pointers);
        }
        InstKind::Nop
        | InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Store { .. }
        | InstKind::MemZero { .. }
        | InstKind::Gep { .. }
        | InstKind::Call { .. } => {}
    }
}

fn branchless_madd_load_pointers(
    func: &crate::ir::Function,
    updated: ValueId,
    old: ValueId,
    update_block: BlockId,
) -> Option<[ValueId; 2]> {
    let InstKind::Binary {
        op: BinaryOp::Iadd,
        lhs,
        rhs,
    } = defining_inst_kind(func, updated)?
    else {
        return None;
    };
    let product = if *lhs == old {
        *rhs
    } else if *rhs == old {
        *lhs
    } else {
        return None;
    };
    let InstKind::Binary {
        op: BinaryOp::Imul,
        lhs,
        rhs,
    } = defining_inst_kind(func, product)?
    else {
        return None;
    };

    let load_pointer = |value| {
        if !matches!(func.value(value).kind, ValueKind::Inst(owner, _) if owner == update_block)
            || func.value(value).ty != Type::I32
        {
            return None;
        }
        let InstKind::Load { ptr } = defining_inst_kind(func, value)? else {
            return None;
        };
        Some(*ptr)
    };
    Some([load_pointer(*lhs)?, load_pointer(*rhs)?])
}

/// Proves that two pointers select the same typed element coordinates in
/// equally shaped global objects.  A load through one pointer therefore proves
/// the other pointer dereferenceable whenever the first load executes.  This
/// permits safe if-conversion without speculating an otherwise unchecked
/// address.  Pointer recurrences are compared coinductively: revisiting the
/// same value pair closes only that exact structural cycle.
fn same_typed_global_address_shape(
    func: &crate::ir::Function,
    lhs: ValueId,
    rhs: ValueId,
    visited: &mut HashSet<(ValueId, ValueId)>,
) -> bool {
    if func.value(lhs).ty != func.value(rhs).ty {
        return false;
    }
    if lhs == rhs {
        return true;
    }
    if !visited.insert((lhs, rhs)) {
        return true;
    }

    match (&func.value(lhs).kind, &func.value(rhs).kind) {
        (ValueKind::Global(_), ValueKind::Global(_)) => true,
        (ValueKind::Inst(lhs_block, lhs_idx), ValueKind::Inst(rhs_block, rhs_idx)) => {
            let (Some(lhs_inst), Some(rhs_inst)) = (
                func.blocks
                    .get(lhs_block.0)
                    .and_then(|block| block.insts.get(*lhs_idx)),
                func.blocks
                    .get(rhs_block.0)
                    .and_then(|block| block.insts.get(*rhs_idx)),
            ) else {
                return false;
            };
            match (&lhs_inst.kind, &rhs_inst.kind) {
                (
                    InstKind::Gep {
                        base: lhs_base,
                        indices: lhs_indices,
                    },
                    InstKind::Gep {
                        base: rhs_base,
                        indices: rhs_indices,
                    },
                ) => {
                    lhs_indices == rhs_indices
                        && same_typed_global_address_shape(func, *lhs_base, *rhs_base, visited)
                }
                (
                    InstKind::Phi {
                        incomings: lhs_incomings,
                    },
                    InstKind::Phi {
                        incomings: rhs_incomings,
                    },
                ) if lhs_incomings.len() == rhs_incomings.len() => {
                    lhs_incomings.iter().all(|(lhs_pred, lhs_value)| {
                        rhs_incomings
                            .iter()
                            .find_map(|(rhs_pred, rhs_value)| {
                                (rhs_pred == lhs_pred).then_some(*rhs_value)
                            })
                            .is_some_and(|rhs_value| {
                                same_typed_global_address_shape(
                                    func, *lhs_value, rhs_value, visited,
                                )
                            })
                    })
                }
                _ => false,
            }
        }
        (ValueKind::Param, ValueKind::Param)
        | (ValueKind::Const(_), ValueKind::Const(_))
        | (ValueKind::Global(_), _)
        | (ValueKind::Inst(_, _), _)
        | (_, ValueKind::Global(_))
        | (_, ValueKind::Inst(_, _))
        | (_, ValueKind::Param)
        | (_, ValueKind::Const(_)) => false,
    }
}

fn cmp_cc(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "lt",
        CmpOp::Gt => "gt",
        CmpOp::Le => "le",
        CmpOp::Ge => "ge",
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
    }
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

fn cmp_nzcv(op: CmpOp, result: bool) -> u8 {
    match (op, result) {
        (CmpOp::Eq, true) | (CmpOp::Ne, false) | (CmpOp::Le, true) | (CmpOp::Gt, false) => 0b0100,
        (CmpOp::Lt, true) | (CmpOp::Ge, false) => 0b1000,
        (CmpOp::Eq, false)
        | (CmpOp::Ne, true)
        | (CmpOp::Lt, false)
        | (CmpOp::Le, false)
        | (CmpOp::Gt, true)
        | (CmpOp::Ge, true) => 0,
    }
}
