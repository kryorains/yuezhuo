use crate::ir::{Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::HashMap;

const INT_REGS: [&str; 11] = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
];

/// RISC-V64 的简化整数寄存器分配。
///
/// 这里先不引入机器 IR，而是直接给 IR value 分配 callee-saved `s` 寄存器：
/// - `s1..s11` 跨 call 保持，所以 non-leaf function 也能安全使用；
/// - 只分配 i1/i32/ptr，不碰 f32；
/// - 用线性位置估算 live interval，不重叠的区间可以复用同一个寄存器；
/// - 没有寄存器时回退到原来的栈槽路径。
pub(super) struct Riscv64RegAlloc {
    regs: HashMap<ValueId, &'static str>,
    used_regs: Vec<&'static str>,
}

impl Riscv64RegAlloc {
    pub(super) fn new(func: &Function) -> Self {
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

fn collect_intervals(func: &Function) -> Vec<LiveInterval> {
    let mut starts = vec![0usize; func.values.len()];
    let mut ends = vec![0usize; func.values.len()];
    let mut use_counts = vec![0usize; func.values.len()];

    for (idx, value) in func.values.iter().enumerate() {
        starts[idx] = value_def_pos(value);
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
                    ends[value.0] = ends[value.0].max(term_pos(pred.0));
                }
                continue;
            }

            let pos = inst_pos(block_idx, inst_idx);
            for value in inst_uses(&inst.kind) {
                use_counts[value.0] += 1;
                ends[value.0] = ends[value.0].max(pos);
            }
        }
        if let Some(terminator) = &func.blocks[block_idx].terminator {
            let pos = term_pos(block_idx);
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

fn value_def_pos(value: &crate::ir::Value) -> usize {
    match value.kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => 0,
        ValueKind::Inst(block, inst_idx) => inst_pos(block.0, inst_idx),
    }
}

fn inst_pos(block_idx: usize, inst_idx: usize) -> usize {
    block_idx * 10_000 + inst_idx * 2 + 1
}

fn term_pos(block_idx: usize) -> usize {
    block_idx * 10_000 + 9_999
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

fn align_to(value: i32, align: i32) -> i32 {
    (value + align - 1) / align * align
}
