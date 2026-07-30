use crate::codegen::common::{ir_value_use_counts, weighted_use_scores};
use crate::ir::{Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

mod float;
#[allow(unused_imports)]
pub(super) use float::Riscv64FloatRegAlloc;

const INT_REGS: [&str; 11] = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
];
const CALLER_SAVED_REGS: [&str; 6] = ["a2", "a3", "a4", "a5", "a6", "a7"];
const MAX_INTERFERENCE_BLOCKS: usize = 1024;
const MAX_INTERFERENCE_VALUES: usize = 8192;
const MAX_INTERFERENCE_CANDIDATES: usize = 512;
const MAX_LIVENESS_CELLS: usize = 262_144;
const MAX_LIVENESS_ITERATIONS: usize = 128;
const MAX_CALL_LIVENESS_WORK: usize = 262_144;

/// RISC-V64 的简化整数寄存器分配。
///
/// 这里先不引入机器 IR，而是直接给 IR value 分配整数寄存器：
/// - `s1..s11` 跨 call 保持，所以 non-leaf function 也能安全使用；
/// - 只分配 i1/i32/ptr，不碰 f32；
/// - 在固定编译预算内用 CFG liveness、phi-edge interference 与 copy affinity 安全复用寄存器；
/// - 精确分析时，不跨 call 且不是参数或 call operand 的值还可使用 `a2..a7`；
/// - 超出预算时只让热点 phi 与少量唯一跨块值独占寄存器，不执行不精确的区间复用；
/// - 没有寄存器时回退到原来的栈槽路径。
pub(super) struct Riscv64RegAlloc {
    regs: HashMap<ValueId, &'static str>,
    used_regs: Vec<&'static str>,
}

impl Riscv64RegAlloc {
    pub(super) fn new(func: &Function) -> Self {
        if func.blocks.len() > MAX_INTERFERENCE_BLOCKS
            || func.values.len() > MAX_INTERFERENCE_VALUES
        {
            return Self::new_conservative(func);
        }

        let mut candidate_set = cross_block_candidates(func);
        candidate_set.extend(call_crossing_candidates(func, &HashMap::new()));
        candidate_set
            .retain(|value| matches!(func.value(*value).ty, Type::I1 | Type::I32 | Type::Ptr(_)));
        if candidate_set.len() > MAX_INTERFERENCE_CANDIDATES
            || func.blocks.len().saturating_mul(candidate_set.len()) > MAX_LIVENESS_CELLS
        {
            return Self::new_conservative(func);
        }

        let mut available = Vec::new();
        available.extend(CALLER_SAVED_REGS);
        available.extend(INT_REGS);

        let scores = weighted_use_scores(func);
        let Some(analysis) = interference_graph(func, &candidate_set) else {
            return Self::new_conservative(func);
        };
        let affinities = phi_affinities(func, &candidate_set, &analysis.interference);
        let mut candidates = candidate_set
            .into_iter()
            .map(|value| (value, scores[value.0]))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));

        let mut regs = HashMap::<ValueId, &'static str>::new();
        for (value, _) in candidates {
            let unavailable = analysis.interference[value.0]
                .iter()
                .filter_map(|neighbor| regs.get(neighbor).copied())
                .collect::<HashSet<_>>();
            let allowed = |reg: &'static str| {
                !unavailable.contains(reg)
                    && register_allowed(
                        func,
                        value,
                        reg,
                        &analysis.call_operands,
                        &analysis.live_across_calls,
                    )
            };
            let preferred = affinities[value.0]
                .iter()
                .find_map(|neighbor| regs.get(neighbor).copied().filter(|reg| allowed(reg)));
            if let Some(reg) =
                preferred.or_else(|| available.iter().copied().find(|reg| allowed(reg)))
            {
                regs.insert(value, reg);
            }
        }
        let occupied = regs.values().copied().collect::<HashSet<_>>();
        let used_regs = INT_REGS
            .iter()
            .copied()
            .filter(|reg| occupied.contains(reg))
            .collect();
        Self { regs, used_regs }
    }

    fn new_conservative(func: &Function) -> Self {
        if has_phi(func) {
            let leaf = is_leaf(func);
            let mut available = Vec::new();
            if leaf {
                available.extend(CALLER_SAVED_REGS);
            }
            available.extend(INT_REGS);
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
                    matches!(func.value(result).ty, Type::I1 | Type::I32 | Type::Ptr(_))
                        .then_some((result, scores[result.0]))
                })
                .collect::<Vec<_>>();
            phi_candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));
            let mut regs = HashMap::new();
            for ((phi, _), reg) in phi_candidates.into_iter().zip(available.iter().copied()) {
                regs.insert(phi, reg);
            }
            let occupied = regs.values().copied().collect::<HashSet<_>>();
            let cross_regs = available
                .into_iter()
                .filter(|reg| !occupied.contains(reg))
                .collect::<Vec<_>>();
            let mut global_candidates = call_crossing_candidates(func, &regs);
            let global_candidate_set = global_candidates.iter().copied().collect::<HashSet<_>>();
            global_candidates.extend(
                cross_block_memory_candidates(func, &regs)
                    .into_iter()
                    .filter(|value| !global_candidate_set.contains(value)),
            );
            for (value, reg) in global_candidates.into_iter().zip(cross_regs) {
                regs.insert(value, reg);
            }
            let occupied = regs.values().copied().collect::<HashSet<_>>();
            let used_regs = INT_REGS
                .iter()
                .copied()
                .filter(|reg| occupied.contains(reg))
                .collect();
            return Self { regs, used_regs };
        }

        Self {
            regs: HashMap::new(),
            used_regs: Vec::new(),
        }
    }

    pub(super) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }

    pub(super) fn used_regs(&self) -> &[&'static str] {
        &self.used_regs
    }
}

fn register_allowed(
    func: &Function,
    value: ValueId,
    reg: &'static str,
    call_operands: &HashSet<ValueId>,
    live_across_calls: &HashSet<ValueId>,
) -> bool {
    !CALLER_SAVED_REGS.contains(&reg)
        || (matches!(func.value(value).ty, Type::I1 | Type::I32 | Type::Ptr(_))
            && !matches!(func.value(value).kind, ValueKind::Param)
            && !call_operands.contains(&value)
            && !live_across_calls.contains(&value))
}

fn is_leaf(func: &Function) -> bool {
    !func.blocks.iter().any(|block| {
        block
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
    })
}

fn has_phi(func: &Function) -> bool {
    func.blocks.iter().any(|block| {
        block
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
    })
}

fn inst_uses(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut values = Vec::with_capacity(indices.len() + 1);
            values.push(*base);
            values.extend(indices.iter().copied());
            values
        }
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_uses(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}

fn cross_block_candidates(func: &Function) -> HashSet<ValueId> {
    let mut candidates = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match inst.kind {
            InstKind::Phi { .. } => inst.result,
            _ => None,
        })
        .collect::<HashSet<_>>();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings } => {
                    for (_, value) in incomings {
                        if !matches!(
                            func.value(*value).kind,
                            ValueKind::Const(_) | ValueKind::Global(_)
                        ) {
                            candidates.insert(*value);
                        }
                    }
                }
                kind => {
                    for value in inst_uses(kind) {
                        if value_owner_block(func, value).is_some_and(|owner| owner != block_idx) {
                            candidates.insert(value);
                        }
                    }
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            for value in terminator_uses(terminator) {
                if value_owner_block(func, value).is_some_and(|owner| owner != block_idx) {
                    candidates.insert(value);
                }
            }
        }
    }
    candidates
}

fn value_owner_block(func: &Function, value: ValueId) -> Option<usize> {
    match func.value(value).kind {
        ValueKind::Param => Some(func.entry.0),
        ValueKind::Inst(block, _) => Some(block.0),
        ValueKind::Const(_) | ValueKind::Global(_) => None,
    }
}

pub(in crate::codegen::target) struct InterferenceAnalysis {
    pub(in crate::codegen::target) interference: Vec<HashSet<ValueId>>,
    pub(in crate::codegen::target) call_operands: HashSet<ValueId>,
    pub(in crate::codegen::target) live_across_calls: HashSet<ValueId>,
}

pub(in crate::codegen::target) fn interference_graph(
    func: &Function,
    candidates: &HashSet<ValueId>,
) -> Option<InterferenceAnalysis> {
    let block_count = func.blocks.len();
    let mut defs = vec![HashSet::new(); block_count];
    let mut uses = vec![HashSet::new(); block_count];
    let mut phi_defs = vec![HashSet::new(); block_count];
    let mut edge_phi_uses = HashMap::<(usize, usize), Vec<ValueId>>::new();
    let mut call_operands = HashSet::new();
    let mut call_liveness_work = 0usize;

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
            let operands = inst_uses(&inst.kind);
            if matches!(inst.kind, InstKind::Call { .. }) {
                call_liveness_work = call_liveness_work.saturating_add(operands.len());
                if call_liveness_work > MAX_CALL_LIVENESS_WORK {
                    return None;
                }
                call_operands.extend(
                    operands
                        .iter()
                        .copied()
                        .filter(|operand| candidates.contains(operand)),
                );
            }
            for operand in operands
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
            for operand in terminator_uses(terminator)
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
    let mut live_across_calls = HashSet::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let mut live = live_out[block_idx].clone();
        if let Some(terminator) = &block.terminator {
            live.extend(
                terminator_uses(terminator)
                    .into_iter()
                    .filter(|value| candidates.contains(value)),
            );
        }
        for inst in block.insts.iter().rev() {
            let result = inst.result;
            if let Some(result) = result {
                live.remove(&result);
            }
            if matches!(inst.kind, InstKind::Call { .. }) {
                call_liveness_work = call_liveness_work.saturating_add(live.len());
                if call_liveness_work > MAX_CALL_LIVENESS_WORK {
                    return None;
                }
                // After removing the call result and before adding its
                // arguments, this is exactly the set that must survive the
                // caller-saved clobber on at least one CFG path.
                live_across_calls.extend(live.iter().copied());
            }
            if let Some(result) = result.filter(|result| candidates.contains(result)) {
                for other in live
                    .iter()
                    .copied()
                    .filter(|value| candidates.contains(value))
                {
                    graph[result.0].insert(other);
                    graph[other.0].insert(result);
                }
            }
            if !matches!(inst.kind, InstKind::Phi { .. }) {
                live.extend(
                    inst_uses(&inst.kind)
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
    Some(InterferenceAnalysis {
        interference: graph,
        call_operands,
        live_across_calls,
    })
}

pub(in crate::codegen::target) fn phi_affinities(
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

fn call_crossing_candidates(
    func: &Function,
    assigned: &HashMap<ValueId, &'static str>,
) -> Vec<ValueId> {
    let mut candidates = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let calls = block
            .insts
            .iter()
            .enumerate()
            .filter_map(|(inst_idx, inst)| {
                matches!(inst.kind, InstKind::Call { .. }).then_some(inst_idx)
            })
            .collect::<Vec<_>>();
        if calls.is_empty() {
            continue;
        }

        let mut uses = HashMap::<ValueId, (usize, usize)>::new();
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            for operand in inst_uses(&inst.kind) {
                let entry = uses.entry(operand).or_insert((inst_idx, 0));
                entry.0 = entry.0.max(inst_idx);
                entry.1 += 1;
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_uses(terminator) {
                let entry = uses.entry(operand).or_insert((block.insts.len(), 0));
                entry.0 = block.insts.len();
                entry.1 += 1;
            }
        }

        for (value, (last_use, use_count)) in uses {
            if assigned.contains_key(&value)
                || !matches!(func.value(value).ty, Type::I1 | Type::I32 | Type::Ptr(_))
            {
                continue;
            }
            let ValueKind::Inst(owner, def_idx) = func.value(value).kind else {
                continue;
            };
            let next_call = calls.partition_point(|call_idx| *call_idx <= def_idx);
            if owner.0 != block_idx
                || calls
                    .get(next_call)
                    .is_none_or(|call_idx| *call_idx >= last_use)
            {
                continue;
            }
            candidates.push((value, use_count, last_use - def_idx));
        }
    }
    candidates.sort_by_key(|(value, uses, span)| {
        (std::cmp::Reverse(*uses), std::cmp::Reverse(*span), value.0)
    });
    candidates.into_iter().map(|(value, _, _)| value).collect()
}

fn cross_block_memory_candidates(
    func: &Function,
    assigned: &HashMap<ValueId, &'static str>,
) -> Vec<ValueId> {
    let mut scores = vec![0usize; func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.insts {
            for operand in inst_uses(&inst.kind) {
                if matches!(
                    func.value(operand).kind,
                    ValueKind::Inst(owner, _) if owner.0 != block_idx
                ) {
                    scores[operand.0] += 1;
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_uses(terminator) {
                if matches!(
                    func.value(operand).kind,
                    ValueKind::Inst(owner, _) if owner.0 != block_idx
                ) {
                    scores[operand.0] += 1;
                }
            }
        }
    }

    let mut candidates = func
        .values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let value_id = ValueId(idx);
            if assigned.contains_key(&value_id)
                || scores[idx] == 0
                || !matches!(value.ty, Type::I1 | Type::I32 | Type::Ptr(_))
            {
                return None;
            }
            let ValueKind::Inst(block, inst_idx) = value.kind else {
                return None;
            };
            matches!(
                func.blocks.get(block.0)?.insts.get(inst_idx)?.kind,
                InstKind::Load { .. } | InstKind::Gep { .. }
            )
            .then_some((value_id, scores[idx]))
        })
        .collect::<Vec<_>>();
    candidates
        .sort_by_key(|(value, score)| (std::cmp::Reverse(*score), std::cmp::Reverse(value.0)));
    candidates.into_iter().map(|(value, _)| value).collect()
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
        func.append_inst(
            body,
            InstKind::Phi {
                incomings: vec![(func.entry, lhs)],
            },
            Some(Type::I32),
        );
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

        let regs = Riscv64RegAlloc::new(&func);
        assert!(regs.reg(lhs).is_some());
        assert!(regs.reg(rhs).is_some());
        assert_ne!(regs.reg(lhs), regs.reg(rhs));
    }
}
