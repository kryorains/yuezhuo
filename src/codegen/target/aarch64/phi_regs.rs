use crate::codegen::common::{ir_value_use_counts, natural_loop_depths};
use crate::ir::{Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const CALLEE_SAVED_REGS: [&str; 10] = [
    "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
];
const LEAF_REGS: [&str; 7] = ["x9", "x10", "x11", "x12", "x13", "x14", "x15"];
const MAX_INTERFERENCE_BLOCKS: usize = 1024;
const MAX_INTERFERENCE_VALUES: usize = 8192;
const MAX_INTERFERENCE_CANDIDATES: usize = 512;
const MAX_LIVENESS_CELLS: usize = 262_144;
const MAX_LIVENESS_ITERATIONS: usize = 128;

/// Keeps selected integer and pointer values in registers.
///
/// Within fixed compile-time budgets, every function uses exact CFG liveness,
/// phi-edge interference coloring and copy affinity. Over-budget functions use
/// a conservative unique-register policy. Leaf functions may use x9-x15 in
/// addition to x19-x28; non-leaf functions use only saved x19-x28.
pub(super) struct AArch64PhiRegs {
    regs: HashMap<ValueId, &'static str>,
    saved_regs: Vec<&'static str>,
}

impl AArch64PhiRegs {
    pub(super) fn new(func: &Function) -> Self {
        if func.blocks.len() > MAX_INTERFERENCE_BLOCKS
            || func.values.len() > MAX_INTERFERENCE_VALUES
        {
            return Self::new_conservative(func);
        }

        let is_leaf = is_leaf(func);
        let scores = weighted_use_scores(func);
        let block_local_values = collect_block_local_values(func);
        let direct_branch_conditions = collect_direct_branch_conditions(func);
        let mut candidates = func
            .values
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                let value_id = ValueId(idx);
                if !is_register_type(&value.ty)
                    || matches!(value.kind, ValueKind::Const(_) | ValueKind::Global(_))
                    || scores[idx] < 2
                    || block_local_values.contains(&value_id)
                    || direct_branch_conditions.contains(&value_id)
                {
                    return None;
                }
                if let ValueKind::Inst(block, inst_idx) = value.kind {
                    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
                    if inst.result != Some(value_id) || matches!(inst.kind, InstKind::Nop) {
                        return None;
                    }
                }
                Some((value_id, scores[idx]))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));
        if candidates.len() > MAX_INTERFERENCE_CANDIDATES
            || func.blocks.len().saturating_mul(candidates.len()) > MAX_LIVENESS_CELLS
        {
            return Self::new_conservative(func);
        }

        // Rank loop phis and other cross-block values in one pool. Reserving all
        // registers for phis can leave a loop bound in the frame even when it is
        // read on every iteration, while a much colder outer phi occupies a
        // register for the whole function.
        let mut available = Vec::new();
        if is_leaf {
            available.extend(LEAF_REGS);
        }
        available.extend(CALLEE_SAVED_REGS);
        let candidate_set = candidates
            .iter()
            .map(|(value, _)| *value)
            .collect::<HashSet<_>>();
        let Some(interference) = interference_graph(func, &candidate_set) else {
            return Self::new_conservative(func);
        };
        let affinities = phi_affinities(func, &candidate_set, &interference);
        let mut regs = HashMap::new();
        for (value, _) in candidates {
            let unavailable = interference[value.0]
                .iter()
                .filter_map(|neighbor| regs.get(neighbor).copied())
                .collect::<HashSet<_>>();
            let preferred = affinities[value.0]
                .iter()
                .find_map(|neighbor| regs.get(neighbor).copied())
                .filter(|reg| !unavailable.contains(reg));
            if let Some(reg) = preferred.or_else(|| {
                available
                    .iter()
                    .copied()
                    .find(|reg| !unavailable.contains(reg))
            }) {
                regs.insert(value, reg);
            }
        }

        let occupied = regs.values().copied().collect::<HashSet<_>>();
        let saved_regs = CALLEE_SAVED_REGS
            .iter()
            .copied()
            .filter(|reg| occupied.contains(reg))
            .collect();
        Self { regs, saved_regs }
    }

    fn new_conservative(func: &Function) -> Self {
        let is_leaf = is_leaf(func);
        let mut available = Vec::new();
        if is_leaf {
            available.extend(LEAF_REGS);
        }
        available.extend(CALLEE_SAVED_REGS);
        let mut regs = HashMap::new();
        let mut occupied = HashSet::new();
        let scores = ir_value_use_counts(func);
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
        for ((phi, _), reg) in phi_candidates.into_iter().zip(available.iter().copied()) {
            regs.insert(phi, reg);
            occupied.insert(reg);
        }

        let available = available
            .into_iter()
            .filter(|reg| !occupied.contains(reg))
            .collect::<Vec<_>>();
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

fn is_leaf(func: &Function) -> bool {
    !func.blocks.iter().any(|block| {
        block
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
    })
}

fn interference_graph(
    func: &Function,
    candidates: &HashSet<ValueId>,
) -> Option<Vec<HashSet<ValueId>>> {
    let block_count = func.blocks.len();
    let mut defs = vec![HashSet::new(); block_count];
    let mut uses = vec![HashSet::new(); block_count];
    let mut phi_defs = vec![HashSet::new(); block_count];
    let mut edge_phi_uses = HashMap::<(usize, usize), Vec<ValueId>>::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.insts {
            if let InstKind::Phi { incomings } = &inst.kind {
                if let Some(result) = inst.result.filter(|result| candidates.contains(result)) {
                    defs[block_idx].insert(result);
                    phi_defs[block_idx].insert(result);
                }
                for (pred, incoming) in incomings
                    .iter()
                    .filter(|(_, incoming)| candidates.contains(incoming))
                {
                    edge_phi_uses
                        .entry((pred.0, block_idx))
                        .or_default()
                        .push(*incoming);
                }
                continue;
            }
            for operand in inst_operands(&inst.kind)
                .into_iter()
                .filter(|operand| candidates.contains(operand))
            {
                if !defs[block_idx].contains(&operand) {
                    uses[block_idx].insert(operand);
                }
            }
            if let Some(result) = inst.result.filter(|result| candidates.contains(result)) {
                defs[block_idx].insert(result);
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_operands(terminator)
                .into_iter()
                .filter(|operand| candidates.contains(operand))
            {
                if !defs[block_idx].contains(&operand) {
                    uses[block_idx].insert(operand);
                }
            }
        }
    }

    let successors = func
        .blocks
        .iter()
        .map(|block| match block.terminator.as_ref() {
            Some(Terminator::Jump(target)) => vec![target.0],
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) if then_target == else_target => vec![then_target.0],
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => vec![then_target.0, else_target.0],
            Some(Terminator::Return(_)) | None => Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut live_in = vec![HashSet::new(); block_count];
    let mut live_out = vec![HashSet::new(); block_count];
    let mut converged = false;
    for _ in 0..MAX_LIVENESS_ITERATIONS {
        let mut changed = false;
        for block_idx in (0..block_count).rev() {
            let mut next_out = HashSet::new();
            for successor in &successors[block_idx] {
                next_out.extend(
                    live_in[*successor]
                        .iter()
                        .filter(|value| !phi_defs[*successor].contains(value))
                        .copied(),
                );
                if let Some(incomings) = edge_phi_uses.get(&(block_idx, *successor)) {
                    next_out.extend(incomings.iter().copied());
                }
            }
            let mut next_in = uses[block_idx].clone();
            next_in.extend(
                next_out
                    .iter()
                    .filter(|value| !defs[block_idx].contains(value))
                    .copied(),
            );
            if live_out[block_idx] != next_out || live_in[block_idx] != next_in {
                live_out[block_idx] = next_out;
                live_in[block_idx] = next_in;
                changed = true;
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }

    let mut graph = vec![HashSet::new(); func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let mut live = live_out[block_idx].clone();
        if let Some(terminator) = &block.terminator {
            live.extend(
                terminator_operands(terminator)
                    .into_iter()
                    .filter(|value| candidates.contains(value)),
            );
        }
        for inst in block.insts.iter().rev() {
            if let Some(result) = inst.result.filter(|result| candidates.contains(result)) {
                for other in live.iter().copied() {
                    if other != result {
                        graph[result.0].insert(other);
                        graph[other.0].insert(result);
                    }
                }
                live.remove(&result);
            }
            if !matches!(inst.kind, InstKind::Phi { .. }) {
                live.extend(
                    inst_operands(&inst.kind)
                        .into_iter()
                        .filter(|value| candidates.contains(value)),
                );
            }
        }
    }
    let params = func
        .params
        .iter()
        .copied()
        .filter(|param| candidates.contains(param))
        .collect::<Vec<_>>();
    for (idx, lhs) in params.iter().copied().enumerate() {
        for rhs in params.iter().copied().skip(idx + 1) {
            graph[lhs.0].insert(rhs);
            graph[rhs.0].insert(lhs);
        }
    }
    Some(graph)
}

fn phi_affinities(
    func: &Function,
    candidates: &HashSet<ValueId>,
    interference: &[HashSet<ValueId>],
) -> Vec<Vec<ValueId>> {
    let mut affinities = vec![Vec::new(); func.values.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let (Some(phi), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
            continue;
        };
        if !candidates.contains(&phi) {
            continue;
        }
        for (_, incoming) in incomings {
            if candidates.contains(incoming)
                && func.value(phi).ty == func.value(*incoming).ty
                && !interference[phi.0].contains(incoming)
            {
                affinities[phi.0].push(*incoming);
                affinities[incoming.0].push(phi);
            }
        }
    }
    affinities
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Function};

    #[test]
    fn interfering_jam_parameters_receive_distinct_registers() {
        let mut func = Function::new("parameter_interference", Type::Void);
        let lhs = func.add_param("lhs", Type::I32);
        let rhs = func.add_param("rhs", Type::I32);
        let body = func.add_block("body");
        func.set_terminator(func.entry, Terminator::Jump(body));
        for _ in 0..2 {
            func.append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs,
                    rhs,
                },
                Some(Type::I32),
            );
        }
        func.set_terminator(body, Terminator::Return(None));
        func.mark_reduction_jammed();

        let regs = AArch64PhiRegs::new(&func);
        assert!(regs.reg(lhs).is_some());
        assert!(regs.reg(rhs).is_some());
        assert_ne!(regs.reg(lhs), regs.reg(rhs));
    }
}
