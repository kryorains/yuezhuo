use super::dominators::{ControlFlowGraph, Dominators};
use crate::ir::{BinaryOp, BlockId, Const, Function, InstKind, Type, ValueId, ValueKind};
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
                    collect_backedge_blocks(cfg, header, *latch, &mut blocks);
                }

                let outside_predecessors = cfg.preds[header.0]
                    .iter()
                    .copied()
                    .filter(|pred| !blocks.contains(pred))
                    .collect::<Vec<_>>();
                let preheader = match outside_predecessors.as_slice() {
                    [preheader] if cfg.succs[preheader.0] == [header] => Some(*preheader),
                    _ => None,
                };

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
                    preheader,
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
    pub(super) preheader: Option<BlockId>,
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
    header: BlockId,
    latch: BlockId,
    blocks: &mut HashSet<BlockId>,
) {
    if blocks.insert(latch) {
        let mut stack = vec![latch];
        while let Some(block) = stack.pop() {
            for pred in &cfg.preds[block.0] {
                if *pred != header && blocks.insert(*pred) {
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
    let preheader = natural_loop.preheader?;
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
    let initial = unique_incoming(incomings, preheader)?;
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
    use crate::ir::{CmpOp, Function, Terminator};

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
            assert!(natural_loop.preheader.is_some());
            assert!(natural_loop.unique_exit().is_some());
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
        assert_eq!(loops.loops()[0].preheader, None);
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
        assert_eq!(loops.loops()[0].preheader, Some(entry));
        assert_eq!(loops.loops()[0].unique_latch(), Some(first_latch));
        assert_eq!(loops.loops()[0].exit_edges, vec![(first_header, between)]);
        assert_eq!(loops.loops()[0].unique_exit(), Some(between));
        assert_eq!(loops.loops()[1].header, second_header);
        assert_eq!(loops.loops()[1].preheader, Some(between));
        assert_eq!(loops.loops()[1].unique_latch(), Some(second_latch));
        assert_eq!(loops.loops()[1].unique_exit(), Some(exit));
    }
}
