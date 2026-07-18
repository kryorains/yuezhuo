use crate::codegen::common::{ir_value_use_counts, natural_loop_depths};
use crate::ir::{Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const INT_REGS: [&str; 11] = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
];
const LEAF_CROSS_REGS: [&str; 6] = ["a2", "a3", "a4", "a5", "a6", "a7"];

/// RISC-V64 的简化整数寄存器分配。
///
/// 这里先不引入机器 IR，而是直接给 IR value 分配 callee-saved `s` 寄存器：
/// - `s1..s11` 跨 call 保持，所以 non-leaf function 也能安全使用；
/// - 只分配 i1/i32/ptr，不碰 f32；
/// - 普通 phi 函数保持热点 phi 独占策略；reduction-jam 函数用 CFG liveness、phi-edge interference 与 copy affinity 安全复用寄存器；
/// - 叶函数还可把 `a2..a7` 用作无需保存的跨块寄存器；
/// - 无 phi 时用线性位置估算 live interval，不重叠的区间可以复用寄存器；
/// - 没有寄存器时回退到原来的栈槽路径。
pub(super) struct Riscv64RegAlloc {
    regs: HashMap<ValueId, &'static str>,
    used_regs: Vec<&'static str>,
}

impl Riscv64RegAlloc {
    pub(super) fn new(func: &Function) -> Self {
        if !func.has_reduction_jam() {
            return Self::new_conservative(func);
        }
        if has_phi(func) {
            let leaf = is_leaf(func);
            let mut available = Vec::new();
            if leaf {
                available.extend(LEAF_CROSS_REGS);
            }
            available.extend(INT_REGS);

            let scores = weighted_use_scores(func);
            let mut candidate_set = cross_block_candidates(func);
            candidate_set.extend(call_crossing_candidates(func, &HashMap::new()));
            let interference = interference_graph(func, &candidate_set);
            let affinities = phi_affinities(func, &candidate_set, &interference);
            let mut candidates = candidate_set
                .into_iter()
                .filter(|value| {
                    matches!(func.value(*value).ty, Type::I1 | Type::I32 | Type::Ptr(_))
                })
                .map(|value| (value, scores[value.0]))
                .collect::<Vec<_>>();
            candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));

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
            let used_regs = INT_REGS
                .iter()
                .copied()
                .filter(|reg| occupied.contains(reg))
                .collect();
            return Self { regs, used_regs };
        }

        let intervals = collect_intervals(func);
        let mut active = Vec::<ActiveInterval>::new();
        let mut free_regs = INT_REGS.to_vec();
        let mut regs = HashMap::new();

        for interval in intervals {
            expire_old_intervals(interval.start, &mut active, &mut free_regs);
            let Some(reg) = free_regs.pop() else {
                continue;
            };
            regs.insert(interval.value, reg);
            active.push(ActiveInterval {
                end: interval.end,
                reg,
            });
            active.sort_by_key(|interval| interval.end);
        }

        let mut used_regs = regs.values().copied().collect::<Vec<_>>();
        used_regs.sort_by_key(|reg| {
            INT_REGS
                .iter()
                .position(|candidate| candidate == reg)
                .unwrap()
        });
        used_regs.dedup();

        Self { regs, used_regs }
    }

    fn new_conservative(func: &Function) -> Self {
        if has_phi(func) {
            let leaf = is_leaf(func);
            let mut available = Vec::new();
            if leaf {
                available.extend(LEAF_CROSS_REGS);
            }
            available.extend(INT_REGS);
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
                    matches!(func.value(result).ty, Type::I1 | Type::I32 | Type::Ptr(_))
                        .then_some((result, scores[result.0]))
                })
                .collect::<Vec<_>>();
            phi_candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));
            let mut regs = HashMap::new();
            for ((phi, _), reg) in phi_candidates.into_iter().zip(available.iter().copied()) {
                regs.insert(phi, reg);
            }
            let use_counts = ir_value_use_counts(func);
            coalesce_phi_incomings(func, &mut regs, &use_counts);
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

        let intervals = collect_intervals(func);
        let mut active = Vec::<ActiveInterval>::new();
        let mut free_regs = INT_REGS.to_vec();
        let mut regs = HashMap::new();
        for interval in intervals {
            expire_old_intervals(interval.start, &mut active, &mut free_regs);
            let Some(reg) = free_regs.pop() else {
                continue;
            };
            regs.insert(interval.value, reg);
            active.push(ActiveInterval {
                end: interval.end,
                reg,
            });
            active.sort_by_key(|interval| interval.end);
        }
        let mut used_regs = regs.values().copied().collect::<Vec<_>>();
        used_regs.sort_by_key(|reg| {
            INT_REGS
                .iter()
                .position(|candidate| candidate == reg)
                .unwrap()
        });
        used_regs.dedup();
        Self { regs, used_regs }
    }

    pub(super) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }

    pub(super) fn used_regs(&self) -> &[&'static str] {
        &self.used_regs
    }

    pub(super) fn saved_area_size(&self) -> i32 {
        align_to((self.used_regs.len() as i32) * 8, 16)
    }
}

#[derive(Debug, Clone, Copy)]
struct LiveInterval {
    value: ValueId,
    start: usize,
    end: usize,
    use_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct ActiveInterval {
    end: usize,
    reg: &'static str,
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

fn collect_intervals(func: &Function) -> Vec<LiveInterval> {
    let block_stride = func
        .blocks
        .iter()
        .map(|block| block.insts.len())
        .max()
        .unwrap_or(0)
        .saturating_add(2)
        .saturating_mul(2);
    let mut starts = vec![0usize; func.values.len()];
    let mut ends = vec![0usize; func.values.len()];
    let mut use_counts = vec![0usize; func.values.len()];

    for (idx, value) in func.values.iter().enumerate() {
        starts[idx] = value_def_pos(value, block_stride);
        ends[idx] = starts[idx];
    }

    for block_idx in 0..func.blocks.len() {
        for inst_idx in 0..func.blocks[block_idx].insts.len() {
            let inst = &func.blocks[block_idx].insts[inst_idx];
            if let InstKind::Phi { incomings } = &inst.kind {
                // Phi 的每个 incoming 实际发生在对应前驱边上，不能按 header 位置计，
                // 否则循环回边上的定义会被误判成很短的 live range。
                for (pred, value) in incomings {
                    use_counts[value.0] += 1;
                    ends[value.0] = ends[value.0].max(term_pos(pred.0, block_stride));
                }
                continue;
            }

            let pos = inst_pos(block_idx, inst_idx, block_stride);
            for value in inst_uses(&inst.kind) {
                use_counts[value.0] += 1;
                ends[value.0] = ends[value.0].max(pos);
            }
        }
        if let Some(terminator) = &func.blocks[block_idx].terminator {
            let pos = term_pos(block_idx, block_stride);
            for value in terminator_uses(terminator) {
                use_counts[value.0] += 1;
                ends[value.0] = ends[value.0].max(pos);
            }
        }
    }

    let mut intervals = func
        .values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let value_id = ValueId(idx);
            let use_count = use_counts[idx];
            (use_count >= 2 && is_int_reg_candidate(func, value, value_id)).then_some(
                LiveInterval {
                    value: value_id,
                    start: starts[idx],
                    end: ends[idx],
                    use_count,
                },
            )
        })
        .collect::<Vec<_>>();

    // 同一起点时先分配更热、生命周期更短的值，提高循环变量/累加器拿到寄存器的概率。
    intervals.sort_by_key(|interval| {
        (
            interval.start,
            std::cmp::Reverse(interval.use_count),
            interval.end - interval.start,
            interval.value.0,
        )
    });
    intervals
}

fn expire_old_intervals(
    current_start: usize,
    active: &mut Vec<ActiveInterval>,
    free_regs: &mut Vec<&'static str>,
) {
    let mut idx = 0;
    while idx < active.len() {
        if active[idx].end < current_start {
            free_regs.push(active[idx].reg);
            active.remove(idx);
        } else {
            idx += 1;
        }
    }
}

fn is_int_reg_candidate(func: &Function, value_info: &crate::ir::Value, value: ValueId) -> bool {
    if matches!(value_info.kind, ValueKind::Const(_) | ValueKind::Global(_)) {
        return false;
    }
    if matches!(value_info.ty, Type::F32 | Type::Void | Type::Array { .. }) {
        return false;
    }
    if let ValueKind::Inst(block, inst_idx) = value_info.kind {
        !matches!(func.block(block).insts[inst_idx].kind, InstKind::Nop)
    } else {
        let _ = value;
        true
    }
}

fn value_def_pos(value: &crate::ir::Value, block_stride: usize) -> usize {
    match value.kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => 0,
        ValueKind::Inst(block, inst_idx) => inst_pos(block.0, inst_idx, block_stride),
    }
}

fn inst_pos(block_idx: usize, inst_idx: usize, block_stride: usize) -> usize {
    block_idx * block_stride + inst_idx * 2 + 1
}

fn term_pos(block_idx: usize, block_stride: usize) -> usize {
    (block_idx + 1) * block_stride - 1
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

fn interference_graph(func: &Function, candidates: &HashSet<ValueId>) -> Vec<HashSet<ValueId>> {
    let block_count = func.blocks.len();
    let mut defs = vec![HashSet::new(); block_count];
    let mut uses = vec![HashSet::new(); block_count];
    let mut phi_defs = vec![HashSet::new(); block_count];
    let mut edge_phi_uses = HashMap::<(usize, usize), Vec<ValueId>>::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.insts {
            if let InstKind::Phi { incomings } = &inst.kind {
                if let Some(result) = inst.result {
                    defs[block_idx].insert(result);
                    phi_defs[block_idx].insert(result);
                }
                for (pred, incoming) in incomings {
                    edge_phi_uses
                        .entry((pred.0, block_idx))
                        .or_default()
                        .push(*incoming);
                }
                continue;
            }
            for operand in inst_uses(&inst.kind) {
                if !defs[block_idx].contains(&operand) {
                    uses[block_idx].insert(operand);
                }
            }
            if let Some(result) = inst.result {
                defs[block_idx].insert(result);
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_uses(terminator) {
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
    loop {
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
            break;
        }
    }

    let mut graph = vec![HashSet::new(); func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let mut live = live_out[block_idx].clone();
        if let Some(terminator) = &block.terminator {
            live.extend(terminator_uses(terminator));
        }
        for inst in block.insts.iter().rev() {
            if let Some(result) = inst.result {
                if candidates.contains(&result) {
                    for other in live
                        .iter()
                        .copied()
                        .filter(|value| candidates.contains(value))
                    {
                        if other != result {
                            graph[result.0].insert(other);
                            graph[other.0].insert(result);
                        }
                    }
                }
                live.remove(&result);
            }
            if !matches!(inst.kind, InstKind::Phi { .. }) {
                live.extend(inst_uses(&inst.kind));
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
    graph
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
                        scores[value.0] = scores[value.0].saturating_add(weight_for(pred.0));
                    }
                }
                kind => {
                    for value in inst_uses(kind) {
                        scores[value.0] = scores[value.0].saturating_add(weight);
                    }
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            for value in terminator_uses(terminator) {
                scores[value.0] = scores[value.0].saturating_add(weight);
            }
        }
    }
    scores
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

fn coalesce_phi_incomings(
    func: &Function,
    regs: &mut HashMap<ValueId, &'static str>,
    use_counts: &[usize],
) {
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
            match func.value(incoming).kind {
                ValueKind::Param => {
                    // A parameter can share the phi register only on an entry edge. On a
                    // later edge, the phi register may already have been overwritten before
                    // the original parameter is consumed (for example by a nested loop).
                    if pred == func.entry && !LEAF_CROSS_REGS.contains(&reg) {
                        regs.insert(incoming, reg);
                    }
                }
                ValueKind::Inst(owner, inst_idx) => {
                    if owner != pred
                        || !matches!(
                            func.blocks
                                .get(owner.0)
                                .and_then(|block| block.insts.get(inst_idx)),
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
                ValueKind::Const(_) | ValueKind::Global(_) => {}
            }
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
            for operand in inst_uses(&inst.kind) {
                last_uses.insert((block_idx, operand.0), inst_idx);
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_uses(terminator) {
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

fn align_to(value: i32, align: i32) -> i32 {
    (value + align - 1) / align * align
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
