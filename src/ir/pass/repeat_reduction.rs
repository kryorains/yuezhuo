use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::HashSet;

/// Collapses a side-effect-free counted loop whose only loop-carried state is
/// `accumulator += invariant_work` into one iteration and a final multiply.
pub(super) struct RepeatReductionPass;

impl RepeatReductionPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RepeatReductionPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            collapse_repeated_reductions(func);
        }
    }
}

fn collapse_repeated_reductions(func: &mut Function) {
    if func.blocks.len() > 1024 || func.values.len() > 8192 {
        return;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let mut loops = loop_info.loops().to_vec();
    // Applying a reduction only changes values, not CFG edges, so one analysis
    // can serve every candidate in the function.
    loops.sort_by_key(|natural_loop| std::cmp::Reverse(natural_loop.blocks.len()));

    let mut changed = false;
    for natural_loop in loops {
        let Some(reduction) = match_reduction(func, &cfg, &natural_loop) else {
            continue;
        };
        apply_reduction(func, reduction);
        changed = true;
    }

    if changed {
        if let Err(errors) = func.verify() {
            panic!(
                "repeat reduction produced invalid IR in {}: {:?}",
                func.name, errors
            );
        }
    }
}

#[derive(Clone, Copy)]
struct Reduction {
    header: BlockId,
    latch: BlockId,
    accumulator: ValueId,
    accumulator_next: ValueId,
    accumulator_initial: ValueId,
    counter: ValueId,
    bound: ValueId,
}

fn match_reduction(
    func: &Function,
    cfg: &ControlFlowGraph,
    natural_loop: &NaturalLoop,
) -> Option<Reduction> {
    let entering_pred = natural_loop.unique_entering_pred?;
    let latch = natural_loop.unique_latch()?;
    if cfg.preds[natural_loop.header.0].len() != 2 {
        return None;
    }
    let header = &func.blocks[natural_loop.header.0];
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = header.terminator.as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target)
        || natural_loop.blocks.contains(else_target)
        || natural_loop.blocks.iter().any(|block| {
            *block != natural_loop.header
                && cfg.succs[block.0]
                    .iter()
                    .any(|succ| !natural_loop.blocks.contains(succ))
        })
    {
        return None;
    }

    let InstKind::Icmp {
        op: CmpOp::Lt,
        lhs: counter,
        rhs: bound,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    if value_defined_in_blocks(func, *bound, &natural_loop.blocks)
        || value_use_count(func, *counter) != 2
        || loop_has_side_effects(func, natural_loop)
    {
        return None;
    }

    let phi_results = header
        .insts
        .iter()
        .filter_map(|inst| {
            matches!(inst.kind, InstKind::Phi { .. })
                .then_some(inst.result)
                .flatten()
        })
        .filter(|value| has_nontrivial_use(func, *value))
        .collect::<Vec<_>>();
    if phi_results.len() != 2 || !phi_results.contains(counter) {
        return None;
    }
    let accumulator = *phi_results.iter().find(|value| *value != counter)?;
    if func.value(accumulator).ty != Type::I32 || func.value(*counter).ty != Type::I32 {
        return None;
    }

    let counter_induction = analyze_i32_induction(func, natural_loop, *counter)?;
    if !is_const_int(func, counter_induction.initial, 0)
        || counter_induction.step != 1
        || !value_defined_in_block(func, counter_induction.next, latch)
        || value_use_count(func, counter_induction.next) != 1
    {
        return None;
    }

    let accumulator_initial = phi_incoming(func, accumulator, entering_pred)?;
    let accumulator_next = phi_incoming(func, accumulator, latch)?;
    if !value_defined_in_blocks(func, accumulator_next, &natural_loop.blocks)
        || !accumulator_is_affine(func, natural_loop, accumulator, accumulator_next)
        || cfg.succs[latch.0] != [natural_loop.header]
    {
        return None;
    }

    Some(Reduction {
        header: natural_loop.header,
        latch,
        accumulator,
        accumulator_next,
        accumulator_initial,
        counter: counter_induction.phi,
        bound: *bound,
    })
}

fn loop_has_side_effects(func: &Function, natural_loop: &NaturalLoop) -> bool {
    natural_loop.blocks.iter().any(|block| {
        func.blocks[block.0].insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::Store { .. } | InstKind::MemZero { .. } | InstKind::Call { .. }
            )
        })
    })
}

fn accumulator_is_affine(
    func: &Function,
    natural_loop: &NaturalLoop,
    accumulator: ValueId,
    accumulator_next: ValueId,
) -> bool {
    let mut dependent = HashSet::from([accumulator]);
    loop {
        let mut changed = false;
        for block in &natural_loop.blocks {
            for inst in &func.blocks[block.0].insts {
                let Some(result) = inst.result else {
                    continue;
                };
                if dependent.contains(&result) {
                    continue;
                }
                if inst_operands(&inst.kind)
                    .iter()
                    .any(|value| dependent.contains(value))
                {
                    dependent.insert(result);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if !dependent.contains(&accumulator_next) {
        return false;
    }

    for block in &natural_loop.blocks {
        if terminator_operands(func.blocks[block.0].terminator.as_ref())
            .iter()
            .any(|value| dependent.contains(value))
        {
            return false;
        }
        for inst in &func.blocks[block.0].insts {
            let Some(result) = inst.result else {
                continue;
            };
            if result == accumulator {
                continue;
            }
            let operands = inst_operands(&inst.kind);
            let dependent_operands = operands
                .iter()
                .filter(|value| dependent.contains(value))
                .count();
            if dependent_operands == 0 {
                continue;
            }
            let valid = match &inst.kind {
                InstKind::Phi { incomings } => {
                    incomings.iter().all(|(_, value)| dependent.contains(value))
                }
                InstKind::Binary {
                    op: BinaryOp::Iadd, ..
                } => dependent_operands == 1,
                InstKind::Binary {
                    op: BinaryOp::Isub,
                    lhs,
                    rhs,
                } => dependent.contains(lhs) && !dependent.contains(rhs),
                _ => false,
            };
            if !valid {
                return false;
            }
        }
    }
    true
}

fn apply_reduction(func: &mut Function, reduction: Reduction) {
    let delta = func
        .append_inst(
            reduction.latch,
            InstKind::Binary {
                op: BinaryOp::Isub,
                lhs: reduction.accumulator_next,
                rhs: reduction.accumulator,
            },
            Some(Type::I32),
        )
        .unwrap();
    let repeated_delta = func
        .append_inst(
            reduction.latch,
            InstKind::Binary {
                op: BinaryOp::Imul,
                lhs: delta,
                rhs: reduction.bound,
            },
            Some(Type::I32),
        )
        .unwrap();
    let collapsed = func
        .append_inst(
            reduction.latch,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: reduction.accumulator_initial,
                rhs: repeated_delta,
            },
            Some(Type::I32),
        )
        .unwrap();

    replace_phi_incoming(func, reduction.accumulator, reduction.latch, collapsed);
    replace_phi_incoming(func, reduction.counter, reduction.latch, reduction.bound);
}

fn replace_phi_incoming(func: &mut Function, phi: ValueId, pred: BlockId, value: ValueId) {
    let ValueKind::Inst(block, inst_idx) = func.value(phi).kind else {
        return;
    };
    let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
        return;
    };
    if let Some((_, incoming)) = incomings
        .iter_mut()
        .find(|(incoming_pred, _)| *incoming_pred == pred)
    {
        *incoming = value;
    }
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return None;
    };
    Some(&func.blocks.get(block.0)?.insts.get(inst_idx)?.kind)
}

fn phi_incoming(func: &Function, phi: ValueId, pred: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    incomings
        .iter()
        .find_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value))
}

fn is_const_int(func: &Function, value: ValueId, expected: i32) -> bool {
    matches!(func.value(value).kind, ValueKind::Const(Const::Int(actual)) if actual == expected)
}

fn value_defined_in_block(func: &Function, value: ValueId, block: BlockId) -> bool {
    matches!(func.value(value).kind, ValueKind::Inst(owner, _) if owner == block)
}

fn value_defined_in_blocks(func: &Function, value: ValueId, blocks: &HashSet<BlockId>) -> bool {
    matches!(func.value(value).kind, ValueKind::Inst(owner, _) if blocks.contains(&owner))
}

fn has_nontrivial_use(func: &Function, value: ValueId) -> bool {
    let mut visited = HashSet::new();
    let mut worklist = vec![value];
    while let Some(current) = worklist.pop() {
        if !visited.insert(current) {
            continue;
        }
        for block in &func.blocks {
            for inst in &block.insts {
                if !inst_operands(&inst.kind).contains(&current) {
                    continue;
                }
                if let InstKind::Phi { .. } = inst.kind {
                    if let Some(result) = inst.result {
                        worklist.push(result);
                        continue;
                    }
                }
                return true;
            }
            if terminator_operands(block.terminator.as_ref()).contains(&current) {
                return true;
            }
        }
    }
    false
}

fn value_use_count(func: &Function, value: ValueId) -> usize {
    let mut count = 0;
    for block in &func.blocks {
        for inst in &block.insts {
            count += inst_operands(&inst.kind)
                .iter()
                .filter(|operand| **operand == value)
                .count();
        }
        count += terminator_operands(block.terminator.as_ref())
            .iter()
            .filter(|operand| **operand == value)
            .count();
    }
    count
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
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

fn terminator_operands(terminator: Option<&Terminator>) -> Vec<ValueId> {
    match terminator {
        Some(Terminator::Return(Some(value))) => vec![*value],
        Some(Terminator::Branch { cond, .. }) => vec![*cond],
        Some(Terminator::Return(None) | Terminator::Jump(_)) | None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_unique_conditional_entering_predecessor_without_preheader() {
        let mut func = Function::new("conditional_reduction_entry", Type::Void);
        let header = func.add_block("header");
        let latch = func.add_block("latch");
        let exit = func.add_block("exit");
        let enter = func.add_const(Const::Bool(true));
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let bound = func.add_const(Const::Int(100));
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond: enter,
                then_target: header,
                else_target: exit,
            },
        );
        let counter = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let accumulator = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let condition = func
            .append_inst(
                header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: counter,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: latch,
                else_target: exit,
            },
        );
        let accumulator_next = func
            .append_inst(
                latch,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: accumulator,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        let counter_next = func
            .append_inst(
                latch,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: counter,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(latch, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));
        for (inst_idx, incoming) in [(0, counter_next), (1, accumulator_next)] {
            let InstKind::Phi { incomings } = &mut func.blocks[header.0].insts[inst_idx].kind
            else {
                unreachable!();
            };
            incomings.push((latch, incoming));
        }
        assert!(func.verify().is_ok());

        let cfg = ControlFlowGraph::new(&func);
        let dom = Dominators::new(&func, &cfg);
        let loops = LoopInfo::new(&cfg, &dom);
        let natural_loop = &loops.loops()[0];
        assert_eq!(natural_loop.unique_entering_pred, Some(func.entry));
        assert_eq!(natural_loop.dedicated_preheader, None);
        assert!(match_reduction(&func, &cfg, natural_loop).is_some());

        collapse_repeated_reductions(&mut func);
        assert!(func.verify().is_ok());
    }
}
