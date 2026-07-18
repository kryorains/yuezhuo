use super::dominators::{ControlFlowGraph, Dominators};
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Terminator, Type, ValueId, ValueKind,
};
use std::collections::{BTreeMap, HashSet};

/// Function-level natural-loop information derived only from CFG backedges and
/// dominance. Backedges with the same header are represented as one loop.
pub(super) struct LoopInfo {
    loops: Vec<NaturalLoop>,
}

impl LoopInfo {
    pub(super) fn new(cfg: &ControlFlowGraph, dom: &Dominators) -> Self {
        let mut backedges = BTreeMap::<usize, Vec<BlockId>>::new();
        for latch_idx in 0..cfg.succs.len() {
            let latch = BlockId(latch_idx);
            if !dom.is_reachable(latch) {
                continue;
            }
            for header in &cfg.succs[latch_idx] {
                if dom.dominates(*header, latch) {
                    backedges.entry(header.0).or_default().push(latch);
                }
            }
        }

        let loops = backedges
            .into_iter()
            .map(|(header_idx, mut latches)| {
                let header = BlockId(header_idx);
                latches.sort_by_key(|block| block.0);
                latches.dedup();

                let mut blocks = HashSet::from([header]);
                for latch in &latches {
                    collect_backedge_blocks(cfg, dom, header, *latch, &mut blocks);
                }

                let outside_predecessors = cfg.preds[header.0]
                    .iter()
                    .copied()
                    .filter(|pred| !blocks.contains(pred))
                    .collect::<Vec<_>>();
                let unique_entering_pred = match outside_predecessors.as_slice() {
                    [entering_pred] => Some(*entering_pred),
                    _ => None,
                };
                let dedicated_preheader = unique_entering_pred
                    .filter(|entering_pred| cfg.succs[entering_pred.0] == [header]);

                let mut exit_edges = blocks
                    .iter()
                    .flat_map(|from| {
                        cfg.succs[from.0]
                            .iter()
                            .filter(|to| !blocks.contains(to))
                            .map(|to| (*from, *to))
                    })
                    .collect::<Vec<_>>();
                exit_edges.sort_by_key(|(from, to)| (from.0, to.0));
                exit_edges.dedup();

                NaturalLoop {
                    header,
                    unique_entering_pred,
                    dedicated_preheader,
                    latches,
                    blocks,
                    exit_edges,
                }
            })
            .collect();
        Self { loops }
    }

    pub(super) fn loops(&self) -> &[NaturalLoop] {
        &self.loops
    }
}

#[derive(Debug, Clone)]
pub(super) struct NaturalLoop {
    pub(super) header: BlockId,
    pub(super) unique_entering_pred: Option<BlockId>,
    pub(super) dedicated_preheader: Option<BlockId>,
    pub(super) latches: Vec<BlockId>,
    pub(super) blocks: HashSet<BlockId>,
    pub(super) exit_edges: Vec<(BlockId, BlockId)>,
}

impl NaturalLoop {
    pub(super) fn unique_latch(&self) -> Option<BlockId> {
        match self.latches.as_slice() {
            [latch] => Some(*latch),
            _ => None,
        }
    }

    pub(super) fn unique_exit(&self) -> Option<BlockId> {
        let mut exits = self.exit_edges.iter().map(|(_, to)| *to);
        let exit = exits.next()?;
        exits.all(|other| other == exit).then_some(exit)
    }
}

fn collect_backedge_blocks(
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    header: BlockId,
    latch: BlockId,
    blocks: &mut HashSet<BlockId>,
) {
    if blocks.insert(latch) {
        let mut stack = vec![latch];
        while let Some(block) = stack.pop() {
            for pred in &cfg.preds[block.0] {
                if *pred != header
                    && dom.is_reachable(*pred)
                    && dom.dominates(header, *pred)
                    && blocks.insert(*pred)
                {
                    stack.push(*pred);
                }
            }
        }
    }
}

/// A directly affine i32 loop recurrence, `next = phi + step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct InductionVariable {
    pub(super) phi: ValueId,
    pub(super) initial: ValueId,
    pub(super) next: ValueId,
    pub(super) step: i32,
}

/// Recognizes a header phi with one loop-invariant initial incoming and one
/// unique-backedge incoming formed by i32 add/sub with a non-zero constant.
pub(super) fn analyze_i32_induction(
    func: &Function,
    natural_loop: &NaturalLoop,
    phi: ValueId,
) -> Option<InductionVariable> {
    let entering_pred = natural_loop.unique_entering_pred?;
    let latch = natural_loop.unique_latch()?;
    let value = func.values.get(phi.0)?;
    if value.ty != Type::I32 {
        return None;
    }
    let ValueKind::Inst(owner, inst_idx) = value.kind else {
        return None;
    };
    if owner != natural_loop.header {
        return None;
    }
    let inst = func.blocks.get(owner.0)?.insts.get(inst_idx)?;
    if inst.result != Some(phi) {
        return None;
    }
    let InstKind::Phi { incomings } = &inst.kind else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    let initial = unique_incoming(incomings, entering_pred)?;
    let next = unique_incoming(incomings, latch)?;
    if func.values.get(initial.0)?.ty != Type::I32
        || func.values.get(next.0)?.ty != Type::I32
        || !is_loop_invariant(func, natural_loop, initial)
    {
        return None;
    }

    let ValueKind::Inst(next_block, next_idx) = func.values.get(next.0)?.kind else {
        return None;
    };
    if !natural_loop.blocks.contains(&next_block) {
        return None;
    }
    let next_inst = func.blocks.get(next_block.0)?.insts.get(next_idx)?;
    if next_inst.result != Some(next) {
        return None;
    }
    let step = match &next_inst.kind {
        InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        } if *lhs == phi => const_i32(func, *rhs)?,
        InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        } if *rhs == phi => const_i32(func, *lhs)?,
        InstKind::Binary {
            op: BinaryOp::Isub,
            lhs,
            rhs,
        } if *lhs == phi => const_i32(func, *rhs)?.wrapping_neg(),
        _ => return None,
    };
    if step == 0 {
        return None;
    }

    Some(InductionVariable {
        phi,
        initial,
        next,
        step,
    })
}

/// Returns the exact number of header-tested iterations for a constant i32
/// induction and a direct signed `icmp` loop condition.
///
/// The result is available only when the first false comparison is reached
/// without wrapping the induction value. A condition that would rely on i32
/// wrapping to terminate is deliberately rejected.
pub(super) fn analyze_const_i32_trip_count(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: InductionVariable,
) -> Option<u32> {
    let initial = const_i32(func, induction.initial)? as i64;
    let header = func.blocks.get(natural_loop.header.0)?;
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = header.terminator.as_ref()?
    else {
        return None;
    };
    let then_inside = natural_loop.blocks.contains(then_target);
    let else_inside = natural_loop.blocks.contains(else_target);
    if then_inside == else_inside {
        return None;
    }

    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
        return None;
    };
    let mut op = *op;
    let bound = if *lhs == induction.phi {
        const_i32(func, *rhs)?
    } else if *rhs == induction.phi {
        op = reverse_cmp(op);
        const_i32(func, *lhs)?
    } else {
        return None;
    } as i64;
    if !then_inside {
        op = negate_cmp(op);
    }

    exact_i32_trip_count(initial, induction.step as i64, op, bound)
}

fn exact_i32_trip_count(initial: i64, step: i64, op: CmpOp, bound: i64) -> Option<u32> {
    if !eval_cmp(op, initial, bound) {
        return Some(0);
    }

    let iterations = match op {
        CmpOp::Eq => 1,
        CmpOp::Ne => {
            let distance = bound - initial;
            if distance % step != 0 || distance / step <= 0 {
                return None;
            }
            distance / step
        }
        CmpOp::Lt => {
            if step <= 0 {
                return None;
            }
            ceil_div_positive(bound - initial, step)
        }
        CmpOp::Le => {
            if step <= 0 {
                return None;
            }
            (bound - initial) / step + 1
        }
        CmpOp::Gt => {
            if step >= 0 {
                return None;
            }
            ceil_div_positive(initial - bound, -step)
        }
        CmpOp::Ge => {
            if step >= 0 {
                return None;
            }
            (initial - bound) / -step + 1
        }
    };
    let final_value = initial.checked_add(iterations.checked_mul(step)?)?;
    if !(i32::MIN as i64..=i32::MAX as i64).contains(&final_value)
        || eval_cmp(op, final_value, bound)
    {
        return None;
    }
    u32::try_from(iterations).ok()
}

fn ceil_div_positive(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(numerator > 0 && denominator > 0);
    1 + (numerator - 1) / denominator
}

fn eval_cmp(op: CmpOp, lhs: i64, rhs: i64) -> bool {
    match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
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

fn negate_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Ne,
        CmpOp::Ne => CmpOp::Eq,
        CmpOp::Lt => CmpOp::Ge,
        CmpOp::Le => CmpOp::Gt,
        CmpOp::Gt => CmpOp::Le,
        CmpOp::Ge => CmpOp::Lt,
    }
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.values.get(value.0)?.kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

fn unique_incoming(incomings: &[(BlockId, ValueId)], pred: BlockId) -> Option<ValueId> {
    let mut values = incomings
        .iter()
        .filter_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value));
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn is_loop_invariant(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    let Some(value) = func.values.get(value.0) else {
        return false;
    };
    match &value.kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(block, _) => !natural_loop.blocks.contains(block),
    }
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match func.values.get(value.0)? {
        crate::ir::Value {
            ty: Type::I32,
            kind: ValueKind::Const(Const::Int(value)),
            ..
        } => Some(*value),
        _ => None,
    }
}
