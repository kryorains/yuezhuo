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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Function, Terminator};

    #[derive(Clone, Copy)]
    enum StepForm {
        AddRight(i32),
        AddLeft(i32),
        Sub(i32),
    }

    fn induction_loop(form: StepForm, computed_initial: bool) -> Function {
        let mut func = Function::new("induction", Type::Void);
        let entry = func.entry;
        let header = func.add_block("header");
        let latch = func.add_block("latch");
        let exit = func.add_block("exit");
        let parameter = func.add_param("initial", Type::I32);
        let one = func.add_const(Const::Int(1));
        let initial = if computed_initial {
            func.append_inst(
                entry,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: parameter,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap()
        } else {
            parameter
        };
        func.set_terminator(entry, Terminator::Jump(header));

        let phi = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(entry, initial)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let bound = func.add_const(Const::Int(100));
        let cond = func
            .append_inst(
                header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: phi,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            header,
            Terminator::Branch {
                cond,
                then_target: latch,
                else_target: exit,
            },
        );

        let (op, lhs, rhs) = match form {
            StepForm::AddRight(step) => {
                let step = func.add_const(Const::Int(step));
                (BinaryOp::Iadd, phi, step)
            }
            StepForm::AddLeft(step) => {
                let step = func.add_const(Const::Int(step));
                (BinaryOp::Iadd, step, phi)
            }
            StepForm::Sub(step) => {
                let step = func.add_const(Const::Int(step));
                (BinaryOp::Isub, phi, step)
            }
        };
        let next = func
            .append_inst(latch, InstKind::Binary { op, lhs, rhs }, Some(Type::I32))
            .unwrap();
        func.set_terminator(latch, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));
        let InstKind::Phi { incomings } = &mut func.blocks[header.0].insts[0].kind else {
            unreachable!();
        };
        incomings.push((latch, next));
        func
    }

    fn only_loop(func: &Function) -> NaturalLoop {
        let cfg = ControlFlowGraph::new(func);
        let dom = Dominators::new(func, &cfg);
        let loops = LoopInfo::new(&cfg, &dom);
        assert_eq!(loops.loops().len(), 1);
        loops.loops()[0].clone()
    }

    fn trip_count_loop(
        initial: i32,
        step: i32,
        op: CmpOp,
        bound: i32,
        reversed: bool,
        body_on_true: bool,
    ) -> (Function, ValueId) {
        let mut func = Function::new("trip_count", Type::Void);
        let entry = func.entry;
        let header = func.add_block("header");
        let latch = func.add_block("latch");
        let exit = func.add_block("exit");
        let initial = func.add_const(Const::Int(initial));
        let bound = func.add_const(Const::Int(bound));
        let step_value = func.add_const(Const::Int(step));
        func.set_terminator(entry, Terminator::Jump(header));
        let phi = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(entry, initial)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let (lhs, rhs) = if reversed { (bound, phi) } else { (phi, bound) };
        let condition = func
            .append_inst(header, InstKind::Icmp { op, lhs, rhs }, Some(Type::I1))
            .unwrap();
        let (then_target, else_target) = if body_on_true {
            (latch, exit)
        } else {
            (exit, latch)
        };
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target,
                else_target,
            },
        );
        let next = func
            .append_inst(
                latch,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: phi,
                    rhs: step_value,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(latch, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));
        let InstKind::Phi { incomings } = &mut func.blocks[header.0].insts[0].kind else {
            unreachable!();
        };
        incomings.push((latch, next));
        (func, phi)
    }

    #[test]
    fn recognizes_increment_decrement_and_general_i32_steps() {
        for (form, expected_step, computed_initial) in [
            (StepForm::AddRight(1), 1, false),
            (StepForm::Sub(1), -1, false),
            (StepForm::AddLeft(7), 7, true),
            (StepForm::Sub(-4), 4, true),
        ] {
            let func = induction_loop(form, computed_initial);
            assert!(func.verify().is_ok());
            let natural_loop = only_loop(&func);
            let phi = func.blocks[natural_loop.header.0].insts[0].result.unwrap();
            let induction = analyze_i32_induction(&func, &natural_loop, phi).unwrap();
            assert_eq!(induction.phi, phi);
            assert_eq!(induction.step, expected_step);
            assert!(natural_loop.unique_latch().is_some());
            assert!(natural_loop.unique_entering_pred.is_some());
            assert!(natural_loop.dedicated_preheader.is_some());
            assert!(natural_loop.unique_exit().is_some());
        }
    }

    #[test]
    fn computes_exact_trip_counts_for_both_directions_and_reversed_compares() {
        for (initial, step, op, bound, reversed, body_on_true, expected) in [
            (0, 1, CmpOp::Lt, 10, false, true, 10),
            (0, 1, CmpOp::Gt, 10, true, true, 10),
            (0, 1, CmpOp::Ge, 10, false, false, 10),
            (10, -1, CmpOp::Gt, 0, false, true, 10),
            (10, -1, CmpOp::Lt, 0, true, true, 10),
        ] {
            let (func, phi) = trip_count_loop(initial, step, op, bound, reversed, body_on_true);
            assert!(func.verify().is_ok());
            let natural_loop = only_loop(&func);
            let induction = analyze_i32_induction(&func, &natural_loop, phi).unwrap();
            assert_eq!(
                analyze_const_i32_trip_count(&func, &natural_loop, induction),
                Some(expected)
            );
        }
    }

    #[test]
    fn rejects_wrapping_or_nonterminating_i32_trip_counts() {
        for (initial, step, op, bound) in [
            (i32::MAX, 1, CmpOp::Le, i32::MAX),
            (0, -1, CmpOp::Lt, 10),
            (0, 2, CmpOp::Ne, 3),
        ] {
            let (func, phi) = trip_count_loop(initial, step, op, bound, false, true);
            assert!(func.verify().is_ok());
            let natural_loop = only_loop(&func);
            let induction = analyze_i32_induction(&func, &natural_loop, phi).unwrap();
            assert_eq!(
                analyze_const_i32_trip_count(&func, &natural_loop, induction),
                None
            );
        }
    }

    #[test]
    fn does_not_treat_a_conditional_incoming_block_as_a_preheader() {
        let mut func = Function::new("conditional_entry", Type::Void);
        let header = func.add_block("header");
        let latch = func.add_block("latch");
        let exit = func.add_block("exit");
        let condition = func.add_const(Const::Bool(true));

        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond: condition,
                then_target: header,
                else_target: exit,
            },
        );
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: latch,
                else_target: exit,
            },
        );
        func.set_terminator(latch, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));

        let cfg = ControlFlowGraph::new(&func);
        let dom = Dominators::new(&func, &cfg);
        let loops = LoopInfo::new(&cfg, &dom);
        assert_eq!(loops.loops().len(), 1);
        assert_eq!(loops.loops()[0].header, header);
        assert_eq!(loops.loops()[0].unique_entering_pred, Some(func.entry));
        assert_eq!(loops.loops()[0].dedicated_preheader, None);
    }

    #[test]
    fn excludes_unreachable_predecessors_from_natural_loop_blocks() {
        let mut func = Function::new("dead_loop_predecessor", Type::Void);
        let header = func.add_block("header");
        let body = func.add_block("body");
        let exit = func.add_block("exit");
        let dead = func.add_block("dead");
        let condition = func.add_const(Const::Bool(true));

        func.set_terminator(func.entry, Terminator::Jump(header));
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: body,
                else_target: exit,
            },
        );
        func.set_terminator(body, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));
        func.set_terminator(dead, Terminator::Jump(body));

        let natural_loop = only_loop(&func);
        assert_eq!(natural_loop.header, header);
        assert!(natural_loop.blocks.contains(&body));
        assert!(!natural_loop.blocks.contains(&dead));
        assert_eq!(natural_loop.unique_entering_pred, Some(func.entry));
        assert_eq!(natural_loop.dedicated_preheader, Some(func.entry));
    }

    #[test]
    fn finds_multiple_natural_loops_and_their_exits() {
        let mut func = Function::new("two_loops", Type::Void);
        let entry = func.entry;
        let first_header = func.add_block("first.header");
        let first_latch = func.add_block("first.latch");
        let between = func.add_block("between");
        let second_header = func.add_block("second.header");
        let second_latch = func.add_block("second.latch");
        let exit = func.add_block("exit");
        let condition = func.add_const(Const::Bool(true));

        func.set_terminator(entry, Terminator::Jump(first_header));
        func.set_terminator(
            first_header,
            Terminator::Branch {
                cond: condition,
                then_target: first_latch,
                else_target: between,
            },
        );
        func.set_terminator(first_latch, Terminator::Jump(first_header));
        func.set_terminator(between, Terminator::Jump(second_header));
        func.set_terminator(
            second_header,
            Terminator::Branch {
                cond: condition,
                then_target: second_latch,
                else_target: exit,
            },
        );
        func.set_terminator(second_latch, Terminator::Jump(second_header));
        func.set_terminator(exit, Terminator::Return(None));

        let cfg = ControlFlowGraph::new(&func);
        let dom = Dominators::new(&func, &cfg);
        let loops = LoopInfo::new(&cfg, &dom);
        assert_eq!(loops.loops().len(), 2);
        assert_eq!(loops.loops()[0].header, first_header);
        assert_eq!(loops.loops()[0].unique_entering_pred, Some(entry));
        assert_eq!(loops.loops()[0].dedicated_preheader, Some(entry));
        assert_eq!(loops.loops()[0].unique_latch(), Some(first_latch));
        assert_eq!(loops.loops()[0].exit_edges, vec![(first_header, between)]);
        assert_eq!(loops.loops()[0].unique_exit(), Some(between));
        assert_eq!(loops.loops()[1].header, second_header);
        assert_eq!(loops.loops()[1].unique_entering_pred, Some(between));
        assert_eq!(loops.loops()[1].dedicated_preheader, Some(between));
        assert_eq!(loops.loops()[1].unique_latch(), Some(second_latch));
        assert_eq!(loops.loops()[1].unique_exit(), Some(exit));
    }
}
