use crate::codegen::common::{ir_value_use_counts, natural_loop_depths};
use crate::ir::{Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const CALLEE_SAVED_REGS: [&str; 10] = [
    "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
];
const LEAF_REGS: [&str; 7] = ["x9", "x10", "x11", "x12", "x13", "x14", "x15"];

/// Conservatively keeps selected integer and pointer values in registers.
///
/// Loop-depth-weighted phi results are assigned first. Every other globally
/// selected value owns a distinct register for the whole function, so
/// correctness does not depend on an approximate live-interval analysis. Leaf functions may also use the
/// caller-saved x9-x15 set; every x19-x28 assignment is saved normally.
pub(super) struct AArch64PhiRegs {
    regs: HashMap<ValueId, &'static str>,
    saved_regs: Vec<&'static str>,
}

impl AArch64PhiRegs {
    pub(super) fn new(func: &Function) -> Self {
        let is_leaf = !func.blocks.iter().any(|block| {
            block
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
        });
        let phi_registers: &[&'static str] = if is_leaf {
            &LEAF_REGS
        } else {
            &CALLEE_SAVED_REGS
        };
        let mut regs = HashMap::new();
        let mut occupied = HashSet::new();
        let scores = weighted_use_scores(func);
        let mut phi_candidates = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| {
                let InstKind::Phi { .. } = inst.kind else {
                    return None;
                };
                let result = inst.result?;
                is_register_type(&func.value(result).ty).then_some((result, scores[result.0]))
            })
            .collect::<Vec<_>>();
        phi_candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));
        for ((phi, _), reg) in phi_candidates
            .into_iter()
            .zip(phi_registers.iter().copied())
        {
            regs.insert(phi, reg);
            occupied.insert(reg);
        }
        coalesce_phi_incomings(func, &mut regs);

        let mut available = Vec::new();
        if is_leaf {
            available.extend(
                LEAF_REGS
                    .iter()
                    .copied()
                    .filter(|reg| !occupied.contains(reg)),
            );
        }
        available.extend(
            CALLEE_SAVED_REGS
                .iter()
                .copied()
                .filter(|reg| !occupied.contains(reg)),
        );
        let block_local_values = collect_block_local_values(func);
        let direct_branch_conditions = collect_direct_branch_conditions(func);
        let mut candidates = func
            .values
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                let value_id = ValueId(idx);
                if regs.contains_key(&value_id)
                    || !is_register_type(&value.ty)
                    || matches!(value.kind, ValueKind::Const(_) | ValueKind::Global(_))
                    || scores[idx] < 2
                    || block_local_values.contains(&value_id)
                    || direct_branch_conditions.contains(&value_id)
                {
                    return None;
                }
                if let ValueKind::Inst(block, inst_idx) = value.kind {
                    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
                    if inst.result != Some(value_id)
                        || matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. })
                    {
                        return None;
                    }
                }
                Some((value_id, scores[idx]))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));
        for ((value, _), reg) in candidates.into_iter().zip(available) {
            regs.insert(value, reg);
            occupied.insert(reg);
        }

        let saved_regs = CALLEE_SAVED_REGS
            .iter()
            .copied()
            .filter(|reg| occupied.contains(reg))
            .collect();
        Self { regs, saved_regs }
    }

    pub(super) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }

    pub(super) fn saved_regs(&self) -> &[&'static str] {
        &self.saved_regs
    }
}

fn is_register_type(ty: &Type) -> bool {
    matches!(ty, Type::I1 | Type::I32 | Type::Ptr(_))
}

fn coalesce_phi_incomings(func: &Function, regs: &mut HashMap<ValueId, &'static str>) {
    let use_counts = ir_value_use_counts(func);
    let local_last_uses = local_last_uses(func);
    let edge_phi_uses = edge_phi_uses(func);
    let phis = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| {
            let InstKind::Phi { incomings } = &inst.kind else {
                return None;
            };
            let result = inst.result?;
            let reg = regs.get(&result).copied()?;
            Some((result, reg, incomings.clone()))
        })
        .collect::<Vec<_>>();

    for (phi, reg, incomings) in phis {
        for (pred, incoming) in incomings {
            if regs.contains_key(&incoming)
                || use_counts.get(incoming.0) != Some(&1)
                || func.value(incoming).ty != func.value(phi).ty
            {
                continue;
            }
            let ValueKind::Inst(owner, inst_idx) = func.value(incoming).kind else {
                continue;
            };
            if owner != pred
                || !matches!(
                    func.blocks.get(owner.0).and_then(|block| block.insts.get(inst_idx)),
                    Some(crate::ir::Inst {
                        result: Some(result),
                        ..
                    }) if *result == incoming
                )
                || phi_used_after(pred, inst_idx, phi, &local_last_uses, &edge_phi_uses)
            {
                continue;
            }
            regs.insert(incoming, reg);
        }
    }
}

fn local_last_uses(func: &Function) -> HashMap<(usize, usize), usize> {
    let mut last_uses = HashMap::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            if matches!(inst.kind, InstKind::Phi { .. }) {
                continue;
            }
            for operand in inst_operands(&inst.kind) {
                last_uses.insert((block_idx, operand.0), inst_idx);
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_operands(terminator) {
                last_uses.insert((block_idx, operand.0), block.insts.len());
            }
        }
    }
    last_uses
}

fn edge_phi_uses(func: &Function) -> HashSet<(usize, usize)> {
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match &inst.kind {
            InstKind::Phi { incomings } => Some(incomings),
            _ => None,
        })
        .flat_map(|incomings| {
            incomings
                .iter()
                .map(|(pred, incoming)| (pred.0, incoming.0))
        })
        .collect()
}

fn phi_used_after(
    block: crate::ir::BlockId,
    inst_idx: usize,
    phi: ValueId,
    local_last_uses: &HashMap<(usize, usize), usize>,
    edge_phi_uses: &HashSet<(usize, usize)>,
) -> bool {
    local_last_uses
        .get(&(block.0, phi.0))
        .is_some_and(|last_use| *last_use > inst_idx)
        || edge_phi_uses.contains(&(block.0, phi.0))
}

fn collect_direct_branch_conditions(func: &Function) -> HashSet<ValueId> {
    let uses = ir_value_use_counts(func);

    func.blocks
        .iter()
        .filter_map(|block| {
            let Terminator::Branch { cond, .. } = block.terminator.as_ref()? else {
                return None;
            };
            if uses[cond.0] != 1 {
                return None;
            }
            let ValueKind::Inst(owner, inst_idx) = func.value(*cond).kind else {
                return None;
            };
            matches!(
                func.blocks.get(owner.0)?.insts.get(inst_idx)?.kind,
                InstKind::Icmp { .. }
            )
            .then_some(*cond)
        })
        .collect()
}

fn collect_block_local_values(func: &Function) -> HashSet<ValueId> {
    let mut info = func
        .values
        .iter()
        .map(|value| match value.kind {
            ValueKind::Inst(block, _) => Some((block, false, true)),
            _ => None,
        })
        .collect::<Vec<_>>();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let owner = crate::ir::BlockId(block_idx);
        for inst in &block.insts {
            let unsupported = matches!(inst.kind, InstKind::Phi { .. } | InstKind::Call { .. });
            for operand in match &inst.kind {
                InstKind::Phi { incomings } => incomings
                    .iter()
                    .map(|(_, value)| *value)
                    .collect::<Vec<_>>(),
                kind => inst_operands(kind),
            } {
                let Some((def_block, saw_use, valid)) =
                    info.get_mut(operand.0).and_then(Option::as_mut)
                else {
                    continue;
                };
                *saw_use = true;
                if unsupported || *def_block != owner {
                    *valid = false;
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_operands(terminator) {
                if let Some((_, saw_use, valid)) = info.get_mut(operand.0).and_then(Option::as_mut)
                {
                    *saw_use = true;
                    *valid = false;
                }
            }
        }
    }

    info.into_iter()
        .enumerate()
        .filter_map(|(idx, info)| matches!(info, Some((_, true, true))).then_some(ValueId(idx)))
        .collect()
}

fn weighted_use_scores(func: &Function) -> Vec<usize> {
    let loop_depths = natural_loop_depths(func);
    let weight_for = |block_idx: usize| 1usize << loop_depths[block_idx].saturating_mul(4).min(20);
    let mut scores = vec![0usize; func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let weight = weight_for(block_idx);
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings } => {
                    for (pred, value) in incomings {
                        let edge_weight = weight_for(pred.0);
                        scores[value.0] = scores[value.0].saturating_add(edge_weight);
                    }
                }
                kind => {
                    for value in inst_operands(kind) {
                        scores[value.0] = scores[value.0].saturating_add(weight);
                    }
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            for value in terminator_operands(terminator) {
                scores[value.0] = scores[value.0].saturating_add(weight);
            }
        }
    }
    scores
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut operands = Vec::with_capacity(indices.len() + 1);
            operands.push(*base);
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}
