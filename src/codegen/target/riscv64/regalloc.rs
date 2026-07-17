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
/// - 有 phi 时按自然循环深度优先给热点 phi 分配互不复用的寄存器，再把剩余寄存器用于跨块 load/GEP；
/// - 叶函数还可把 `a2..a7` 用作无需保存的跨块寄存器；
/// - 无 phi 时用线性位置估算 live interval，不重叠的区间可以复用寄存器；
/// - 没有寄存器时回退到原来的栈槽路径。
pub(super) struct Riscv64RegAlloc {
    regs: HashMap<ValueId, &'static str>,
    used_regs: Vec<&'static str>,
}

impl Riscv64RegAlloc {
    pub(super) fn new(func: &Function) -> Self {
        if has_phi(func) {
            let mut regs = HashMap::new();
            let mut used_regs = Vec::new();
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
            for ((phi, _), reg) in phi_candidates.into_iter().zip(INT_REGS) {
                regs.insert(phi, reg);
                used_regs.push(reg);
            }
            let use_counts = ir_value_use_counts(func);
            coalesce_phi_incomings(func, &mut regs, &use_counts);
            let mut cross_regs = INT_REGS[used_regs.len()..].to_vec();
            if is_leaf(func) {
                cross_regs.extend(LEAF_CROSS_REGS);
            }
            for (value, reg) in cross_block_memory_candidates(func, &regs)
                .into_iter()
                .zip(cross_regs)
            {
                regs.insert(value, reg);
                if INT_REGS.contains(&reg) {
                    used_regs.push(reg);
                }
            }
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
