use crate::codegen::common::{
    gep_elem_type, ir_size, ir_value_use_counts, natural_loop_depths, pointee, weighted_use_scores,
};
use crate::codegen::Target;
use crate::ir::{BinaryOp, CmpOp, Const, Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

mod float;
#[allow(unused_imports)]
pub(super) use float::Riscv64FloatRegAlloc;

const INT_REGS: [&str; 11] = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
];
const CALLER_SAVED_REGS: [&str; 10] = ["t3", "t4", "t5", "t6", "a2", "a3", "a4", "a5", "a6", "a7"];
const LOCAL_REGS: [&str; 4] = ["t3", "t4", "t5", "t6"];
const ARG_REGS: [&str; 6] = ["a2", "a3", "a4", "a5", "a6", "a7"];
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
    local_regs: Vec<&'static str>,
    materialized: Vec<(ValueId, &'static str)>,
    address_offsets: Vec<(i64, &'static str)>,
}

impl Riscv64RegAlloc {
    pub(super) fn new(func: &Function) -> Self {
        if func.blocks.len() > MAX_INTERFERENCE_BLOCKS
            || func.values.len() > MAX_INTERFERENCE_VALUES
        {
            return Self::new_conservative(func);
        }

        let use_counts = ir_value_use_counts(func);
        let scores = weighted_use_scores(func);
        let mut candidate_set = cross_block_candidates(func);
        candidate_set.extend(call_crossing_candidates(func, &HashMap::new()));
        candidate_set
            .retain(|value| matches!(func.value(*value).ty, Type::I1 | Type::I32 | Type::Ptr(_)));
        if candidate_set.len() > MAX_INTERFERENCE_CANDIDATES
            || func.blocks.len().saturating_mul(candidate_set.len()) > MAX_LIVENESS_CELLS
        {
            return Self::new_conservative(func);
        }

        // LLVM and GCC allocate registers from complete live ranges rather
        // than restricting the global allocator to values that cross a basic
        // block. Add the hottest remaining SSA values up to the fixed
        // interference-analysis budget. Values that do not fit remain
        // eligible for the lightweight block-local allocator.
        let capacity = MAX_INTERFERENCE_CANDIDATES.min(
            MAX_LIVENESS_CELLS
                .checked_div(func.blocks.len().max(1))
                .unwrap_or(0),
        );
        let mut local_candidates = func
            .values
            .iter()
            .enumerate()
            .filter_map(|(idx, _)| {
                let value = ValueId(idx);
                if candidate_set.contains(&value)
                    || use_counts[idx] == 0
                    || !matches!(func.value(value).ty, Type::I1 | Type::I32 | Type::Ptr(_))
                {
                    return None;
                }
                let supported = match func.value(value).kind {
                    ValueKind::Param => true,
                    ValueKind::Inst(block, inst_idx) => !matches!(
                        func.blocks[block.0].insts[inst_idx].kind,
                        InstKind::Nop | InstKind::Alloca { .. }
                    ),
                    ValueKind::Const(Const::Int(_) | Const::Bool(_)) | ValueKind::Global(_) => {
                        const_i32_value(func, value) != Some(0)
                            && scores[idx] >= 16
                            && has_non_immediate_use(func, value, &use_counts)
                    }
                    ValueKind::Const(_) => false,
                };
                supported.then_some((value, scores[idx], use_counts[idx]))
            })
            .collect::<Vec<_>>();
        local_candidates.sort_by_key(|(value, score, uses)| {
            (std::cmp::Reverse(*score), std::cmp::Reverse(*uses), value.0)
        });
        candidate_set.extend(
            local_candidates
                .into_iter()
                .take(capacity.saturating_sub(candidate_set.len()))
                .map(|(value, _, _)| value),
        );

        let mut available = Vec::new();
        available.extend(CALLER_SAVED_REGS);
        available.extend(INT_REGS);

        let costs = Target::Riscv64.cost_model();
        let Some(analysis) = interference_graph(func, &candidate_set) else {
            return Self::new_conservative(func);
        };
        let affinities = phi_affinities(func, &candidate_set, &analysis.interference);
        let mut candidates = candidate_set
            .iter()
            .copied()
            .map(|value| (value, scores[value.0]))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));

        let mut regs = HashMap::<ValueId, &'static str>::new();
        for (value, score) in candidates {
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
                        score,
                        costs,
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
        coalesce_phi_inputs(func, &scores, &analysis, costs, &mut regs);
        let mut occupied = regs.values().copied().collect::<HashSet<_>>();
        let address_offsets = select_address_offset_regs(func, &occupied);
        occupied.extend(address_offsets.iter().map(|(_, reg)| *reg));
        let used_regs = INT_REGS
            .iter()
            .copied()
            .filter(|reg| occupied.contains(reg))
            .collect();
        let local_regs = LOCAL_REGS
            .iter()
            .copied()
            .filter(|reg| !occupied.contains(reg))
            .collect();
        let mut materialized = regs
            .iter()
            .filter_map(|(value, reg)| {
                matches!(
                    func.value(*value).kind,
                    ValueKind::Const(_) | ValueKind::Global(_)
                )
                .then_some((*value, *reg))
            })
            .collect::<Vec<_>>();
        materialized.sort_by_key(|(value, _)| value.0);
        Self {
            regs,
            used_regs,
            local_regs,
            materialized,
            address_offsets,
        }
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
            let local_regs = LOCAL_REGS
                .iter()
                .copied()
                .filter(|reg| !occupied.contains(reg))
                .collect();
            return Self {
                regs,
                used_regs,
                local_regs,
                materialized: Vec::new(),
                address_offsets: Vec::new(),
            };
        }

        Self {
            regs: HashMap::new(),
            used_regs: Vec::new(),
            local_regs: LOCAL_REGS.to_vec(),
            materialized: Vec::new(),
            address_offsets: Vec::new(),
        }
    }

    pub(super) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }

    pub(super) fn used_regs(&self) -> &[&'static str] {
        &self.used_regs
    }

    pub(super) fn local_regs(&self) -> &[&'static str] {
        &self.local_regs
    }

    pub(super) fn materialized(&self) -> &[(ValueId, &'static str)] {
        &self.materialized
    }

    pub(super) fn address_offsets(&self) -> &[(i64, &'static str)] {
        &self.address_offsets
    }

    pub(super) fn address_offset_reg(&self, offset: i64) -> Option<&'static str> {
        self.address_offsets
            .iter()
            .find_map(|(candidate, reg)| (*candidate == offset).then_some(*reg))
    }
}

/// Recolors a phi input after greedy coloring when doing so is conflict-free
/// and every phi use of that value wants the same physical register. Ordinary
/// instruction uses are indifferent to the register name, so the recoloring
/// cannot add a copy away from another SSA edge.
fn coalesce_phi_inputs(
    func: &Function,
    scores: &[usize],
    analysis: &InterferenceAnalysis,
    costs: crate::codegen::cost::TargetCostModel,
    regs: &mut HashMap<ValueId, &'static str>,
) {
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let (Some(phi), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
            continue;
        };
        let Some(destination) = regs.get(&phi).copied() else {
            continue;
        };
        for (_, incoming) in incomings {
            if regs.get(incoming).copied() == Some(destination)
                || analysis.interference[phi.0].contains(incoming)
                || !matches!(func.value(*incoming).kind, ValueKind::Inst(_, _))
                || value_is_phi_result(func, *incoming)
                || !all_phi_uses_target_register(func, *incoming, destination, regs)
                || !register_allowed(
                    func,
                    *incoming,
                    destination,
                    &analysis.call_operands,
                    &analysis.live_across_calls,
                    scores[incoming.0],
                    costs,
                )
                || analysis.interference[incoming.0]
                    .iter()
                    .any(|neighbor| regs.get(neighbor).copied() == Some(destination))
            {
                continue;
            }
            regs.insert(*incoming, destination);
        }
    }
}

fn value_is_phi_result(func: &Function, value: ValueId) -> bool {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return false;
    };
    matches!(
        func.blocks
            .get(block.0)
            .and_then(|block| block.insts.get(inst_idx))
            .map(|inst| &inst.kind),
        Some(InstKind::Phi { .. })
    )
}

fn all_phi_uses_target_register(
    func: &Function,
    value: ValueId,
    expected: &'static str,
    regs: &HashMap<ValueId, &'static str>,
) -> bool {
    let mut found = false;
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let (Some(phi), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
            continue;
        };
        if !incomings.iter().any(|(_, incoming)| *incoming == value) {
            continue;
        }
        found = true;
        if regs.get(&phi).copied() != Some(expected) {
            return false;
        }
    }
    found
}

/// Keeps profitable large constant address increments in otherwise-unused
/// callee-saved registers. This is ordinary loop-invariant rematerialization:
/// the static offset must recur in a natural loop often enough to repay one
/// function-entry materialization and the save/restore pair.
fn select_address_offset_regs(
    func: &Function,
    occupied: &HashSet<&'static str>,
) -> Vec<(i64, &'static str)> {
    const MIN_WEIGHTED_SAVINGS: usize = 16;
    const MAX_OFFSETS: usize = 2;

    let loop_depths = natural_loop_depths(func);
    let mut scores = HashMap::<i64, usize>::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let weight = 1usize << loop_depths[block_idx].saturating_mul(4).min(20);
        for inst in &block.insts {
            let InstKind::Gep { base, indices } = &inst.kind else {
                continue;
            };
            let (Some(result), [index]) = (inst.result, indices.as_slice()) else {
                continue;
            };
            let Some(index) = const_i32_value(func, *index) else {
                continue;
            };
            let elem_ty = pointee(&func.value(result).ty)
                .unwrap_or_else(|| gep_elem_type(&func.value(*base).ty));
            let stride = ir_size(&elem_ty).max(1);
            let Some(offset) = i64::from(index).checked_mul(i64::from(stride)) else {
                continue;
            };
            if offset == 0 || i32::try_from(offset).is_err() || fits_i12_value(offset as i32) {
                continue;
            }
            scores
                .entry(offset)
                .and_modify(|score| *score = score.saturating_add(weight))
                .or_insert(weight);
        }
    }

    let mut candidates = scores
        .into_iter()
        .filter(|(_, score)| *score >= MIN_WEIGHTED_SAVINGS)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(offset, score)| (std::cmp::Reverse(*score), *offset));
    let free = INT_REGS
        .iter()
        .copied()
        .filter(|reg| !occupied.contains(reg));
    candidates
        .into_iter()
        .take(MAX_OFFSETS)
        .zip(free)
        .map(|((offset, _), reg)| (offset, reg))
        .collect()
}

fn has_non_immediate_use(func: &Function, value: ValueId, use_counts: &[usize]) -> bool {
    func.blocks.iter().any(|block| {
        block.insts.iter().any(|inst| match &inst.kind {
            InstKind::Nop | InstKind::Alloca { .. } => false,
            InstKind::Phi { incomings } => incomings.iter().any(|(_, incoming)| *incoming == value),
            InstKind::Load { ptr } => *ptr == value,
            InstKind::MemZero { ptr, count, .. } => *ptr == value || *count == Some(value),
            InstKind::MemCopy {
                dst, src, count, ..
            } => *dst == value || *src == value || *count == value,
            InstKind::Store { ptr, value: stored } => *ptr == value || *stored == value,
            InstKind::Unary { value: operand, .. } | InstKind::Cast { value: operand, .. } => {
                *operand == value
            }
            InstKind::Binary { op, lhs, rhs } => {
                (*lhs == value || *rhs == value)
                    && !binary_use_has_immediate_form(func, value, *op, *lhs, *rhs)
            }
            InstKind::Icmp { op, lhs, rhs } => {
                (*lhs == value || *rhs == value)
                    && icmp_use_requires_register(
                        func,
                        value,
                        *op,
                        *lhs,
                        *rhs,
                        inst.result,
                        block.terminator.as_ref(),
                        use_counts,
                    )
            }
            InstKind::Fcmp { .. } => false,
            InstKind::Gep { base, .. } => *base == value,
            InstKind::Call { args, .. } => args.contains(&value),
        }) || matches!(
            block.terminator,
            Some(Terminator::Return(Some(returned))) if returned == value
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn icmp_use_requires_register(
    func: &Function,
    value: ValueId,
    mut op: CmpOp,
    lhs: ValueId,
    rhs: ValueId,
    result: Option<ValueId>,
    terminator: Option<&Terminator>,
    use_counts: &[usize],
) -> bool {
    let Some(immediate) = const_i32_value(func, value) else {
        return true;
    };
    let constant_on_lhs = value == lhs;
    if constant_on_lhs {
        op = reverse_cmp(op);
    } else if value != rhs {
        return true;
    }

    let is_direct_branch = result.is_some_and(|result| {
        use_counts.get(result.0) == Some(&1)
            && matches!(terminator, Some(Terminator::Branch { cond, .. }) if *cond == result)
    });
    if is_direct_branch {
        // Integer branches compare two registers. Only the canonical zero,
        // one, and minus-one boundary forms below become a one-register
        // pseudo-branch; other constants benefit from one hoisted register.
        return if constant_on_lhs {
            immediate != 0
        } else {
            !matches!(
                (op, immediate),
                (
                    CmpOp::Lt | CmpOp::Gt | CmpOp::Le | CmpOp::Ge | CmpOp::Eq | CmpOp::Ne,
                    0
                ) | (CmpOp::Lt | CmpOp::Ge, 1)
                    | (CmpOp::Gt | CmpOp::Le, -1)
            )
        };
    }

    match op {
        CmpOp::Eq | CmpOp::Ne | CmpOp::Lt | CmpOp::Ge => !fits_i12_value(immediate),
        CmpOp::Le | CmpOp::Gt => !immediate.checked_add(1).is_some_and(fits_i12_value),
    }
}

fn reverse_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}

fn binary_use_has_immediate_form(
    func: &Function,
    value: ValueId,
    op: BinaryOp,
    _lhs: ValueId,
    rhs: ValueId,
) -> bool {
    let Some(immediate) = const_i32_value(func, value) else {
        return false;
    };
    match op {
        BinaryOp::Iadd => fits_i12_value(immediate),
        BinaryOp::Isub => value == rhs && immediate.checked_neg().is_some_and(fits_i12_value),
        BinaryOp::Imul => {
            immediate > 0 && (immediate & (immediate - 1)) == 0
                || immediate > 2
                    && (immediate
                        .checked_sub(1)
                        .is_some_and(|v| (v as u32).is_power_of_two())
                        || immediate
                            .checked_add(1)
                            .is_some_and(|v| (v as u32).is_power_of_two()))
        }
        BinaryOp::Iand | BinaryOp::Ior | BinaryOp::Ixor => {
            value == rhs && fits_i12_value(immediate)
        }
        BinaryOp::Ishl | BinaryOp::Iashr => value == rhs && (0..32).contains(&immediate),
        // Constant division expands into several target instructions. Keep a
        // hot divisor eligible for ordinary register allocation so the
        // lowering can reuse it for remainder reconstruction and range-based
        // reductions instead of rematerializing it in every iteration.
        BinaryOp::Idiv => true,
        BinaryOp::Imod => {
            let magnitude = immediate.unsigned_abs();
            magnitude <= 1 || magnitude.is_power_of_two() || has_self_call(func)
        }
        BinaryOp::And
        | BinaryOp::Or
        | BinaryOp::Fadd
        | BinaryOp::Fsub
        | BinaryOp::Fmul
        | BinaryOp::Fdiv => false,
    }
}

fn has_self_call(func: &Function) -> bool {
    func.blocks.iter().any(|block| {
        block
            .insts
            .iter()
            .any(|inst| matches!(&inst.kind, InstKind::Call { name, .. } if name == &func.name))
    })
}

fn const_i32_value(func: &Function, value: ValueId) -> Option<i32> {
    match func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(value),
        ValueKind::Const(Const::Bool(value)) => Some(i32::from(value)),
        _ => None,
    }
}

fn fits_i12_value(value: i32) -> bool {
    (-2048..=2047).contains(&value)
}

fn register_allowed(
    func: &Function,
    value: ValueId,
    reg: &'static str,
    call_operands: &HashSet<ValueId>,
    live_across_calls: &HashSet<ValueId>,
    score: usize,
    costs: crate::codegen::cost::TargetCostModel,
) -> bool {
    if CALLER_SAVED_REGS.contains(&reg) {
        return matches!(func.value(value).ty, Type::I1 | Type::I32 | Type::Ptr(_))
            && !live_across_calls.contains(&value)
            && (!ARG_REGS.contains(&reg)
                || (!matches!(func.value(value).kind, ValueKind::Param)
                    && !call_operands.contains(&value)));
    }
    costs.should_use_callee_saved_register(score, live_across_calls.contains(&value))
}

fn is_leaf(func: &Function) -> bool {
    !func.blocks.iter().any(|block| {
        block
            .insts
            .iter()
            .any(|inst| instruction_may_call(&inst.kind))
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
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
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
    let mut materialized = candidates
        .iter()
        .copied()
        .filter(|value| {
            matches!(
                func.value(*value).kind,
                ValueKind::Const(_) | ValueKind::Global(_)
            )
        })
        .collect::<Vec<_>>();
    materialized.sort_by_key(|value| value.0);
    // These values are emitted once, after parameter setup and immediately
    // before the entry block's ordinary instructions. Model that real
    // definition point so graph coloring can reuse their registers outside
    // the actual live range, just as a normal SSA definition can.
    defs[func.entry.0].extend(materialized.iter().copied());

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
            if instruction_may_call(&inst.kind) {
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
            if instruction_may_call(&inst.kind) {
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
        if block_idx == func.entry.0 {
            for value in materialized.iter().copied() {
                live.remove(&value);
                for other in live
                    .iter()
                    .copied()
                    .filter(|other| candidates.contains(other))
                {
                    graph[value.0].insert(other);
                    graph[other.0].insert(value);
                }
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
            .filter_map(|(inst_idx, inst)| instruction_may_call(&inst.kind).then_some(inst_idx))
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

fn instruction_may_call(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Call { .. } | InstKind::MemZero { count: Some(_), .. } | InstKind::MemCopy { .. }
    )
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
