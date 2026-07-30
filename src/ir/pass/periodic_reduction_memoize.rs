use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::util::defining_inst;
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::HashSet;

/// Memoizes a pure inner reduction whose complete varying state alternates
/// between two values selected by a proven periodic reset recurrence.
///
/// Accepted outer state has the form:
///
/// `state = (i % period != 0) ? reset : state + delta`
///
/// with `i = 0, 1, ...`, `period >= 2`, and `state` initially equal to
/// `reset`.  Therefore the two branch results are always exactly `reset` and
/// `reset + delta`.  The inner reduction is executed once for each branch and
/// reused thereafter without changing the outer accumulation order.
pub(super) struct PeriodicReductionMemoizePass;

impl PeriodicReductionMemoizePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for PeriodicReductionMemoizePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            if let Some(candidate) = find_candidate(func) {
                apply_candidate(func, candidate);
            }
        }
    }
}

#[derive(Clone)]
struct Candidate {
    outer_header: BlockId,
    outer_entering: BlockId,
    outer_latch: BlockId,
    inner_header: BlockId,
    inner_preheader: BlockId,
    inner_exit: BlockId,
    accumulator: ValueId,
    branch_condition: ValueId,
}

fn find_candidate(func: &Function) -> Option<Candidate> {
    if func.blocks.len() > 1024 || func.values.len() > 8192 {
        return None;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom).loops().to_vec();
    let mut inner_loops = loops.iter().collect::<Vec<_>>();
    inner_loops.sort_by_key(|natural_loop| natural_loop.blocks.len());

    for inner in inner_loops {
        let (Some(inner_preheader), Some(inner_body), Some(inner_exit)) = (
            inner.dedicated_preheader,
            inner.unique_latch(),
            inner.unique_exit(),
        ) else {
            continue;
        };
        if inner.blocks != HashSet::from([inner.header, inner_body])
            || inner.exit_edges != [(inner.header, inner_exit)]
            || !matches!(
                func.blocks[inner_body.0].terminator,
                Some(Terminator::Jump(target)) if target == inner.header
            )
            || !matches!(
                func.blocks[inner_preheader.0].terminator,
                Some(Terminator::Jump(target)) if target == inner.header
            )
            || !pure_inner_loop(func, inner)
        {
            continue;
        }
        let Some((accumulator, accumulator_initial)) = float_accumulator(func, inner) else {
            continue;
        };
        if !matches!(
            func.value(accumulator_initial).kind,
            ValueKind::Const(Const::Float(0) | Const::Zero(Type::F32))
        ) || !accumulator_external_uses_are_in_exit(func, inner, accumulator, inner_exit)
        {
            continue;
        }

        let Some(outer) = loops
            .iter()
            .filter(|outer| {
                outer.blocks.len() > inner.blocks.len()
                    && inner
                        .blocks
                        .iter()
                        .all(|block| outer.blocks.contains(block))
                    && outer.blocks.contains(&inner_preheader)
                    && outer.blocks.contains(&inner_exit)
            })
            .min_by_key(|outer| outer.blocks.len())
        else {
            continue;
        };
        let (Some(outer_entering), Some(outer_latch)) =
            (outer.unique_entering_pred, outer.unique_latch())
        else {
            continue;
        };
        if outer_latch != inner_exit {
            continue;
        }
        let Some(outer_induction) = header_induction(func, outer) else {
            continue;
        };
        if outer_induction.step != 1
            || !matches!(
                func.value(outer_induction.initial).kind,
                ValueKind::Const(Const::Int(0) | Const::Zero(Type::I32))
            )
        {
            continue;
        }

        let states = inner_external_float_values(func, inner, outer);
        if states.len() < 2 || states.len() > 4 {
            continue;
        }
        let Some((reset_pred, update_pred)) =
            prove_state_recurrences(func, outer, inner_preheader, inner_exit, &states)
        else {
            continue;
        };
        let Some(branch_condition) = prove_periodic_branch(
            func,
            outer,
            outer_induction.phi,
            inner_preheader,
            reset_pred,
            update_pred,
        ) else {
            continue;
        };

        return Some(Candidate {
            outer_header: outer.header,
            outer_entering,
            outer_latch,
            inner_header: inner.header,
            inner_preheader,
            inner_exit,
            accumulator,
            branch_condition,
        });
    }
    None
}

fn pure_inner_loop(func: &Function, inner: &NaturalLoop) -> bool {
    inner.blocks.iter().all(|block| {
        func.blocks[block.0].insts.iter().all(|inst| {
            matches!(
                inst.kind,
                InstKind::Nop
                    | InstKind::Phi { .. }
                    | InstKind::Unary { .. }
                    | InstKind::Binary { .. }
                    | InstKind::Icmp { .. }
                    | InstKind::Fcmp { .. }
                    | InstKind::Cast { .. }
                    | InstKind::Gep { .. }
            )
        })
    })
}

fn float_accumulator(func: &Function, inner: &NaturalLoop) -> Option<(ValueId, ValueId)> {
    let entering = inner.unique_entering_pred?;
    let latch = inner.unique_latch()?;
    let mut accumulator = None;
    for inst in &func.blocks[inner.header.0].insts {
        let (Some(result), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
            continue;
        };
        if func.value(result).ty != Type::F32 {
            continue;
        }
        let initial = unique_incoming(incomings, entering)?;
        let next = unique_incoming(incomings, latch)?;
        if func.value(next).ty != Type::F32 || accumulator.replace((result, initial)).is_some() {
            return None;
        }
    }
    accumulator
}

fn header_induction(
    func: &Function,
    natural_loop: &NaturalLoop,
) -> Option<super::loop_analysis::InductionVariable> {
    func.blocks[natural_loop.header.0]
        .insts
        .iter()
        .filter_map(|inst| matches!(inst.kind, InstKind::Phi { .. }).then_some(inst.result?))
        .find_map(|phi| analyze_i32_induction(func, natural_loop, phi))
}

fn inner_external_float_values(
    func: &Function,
    inner: &NaturalLoop,
    outer: &NaturalLoop,
) -> Vec<ValueId> {
    let mut states = Vec::new();
    for block in &inner.blocks {
        for inst in &func.blocks[block.0].insts {
            for operand in inst_operands(&inst.kind) {
                let Some(value) = func.values.get(operand.0) else {
                    continue;
                };
                if value.ty != Type::F32 {
                    continue;
                }
                let ValueKind::Inst(owner, _) = value.kind else {
                    continue;
                };
                if outer.blocks.contains(&owner)
                    && !inner.blocks.contains(&owner)
                    && !states.contains(&operand)
                {
                    states.push(operand);
                }
            }
        }
    }
    states.sort_by_key(|value| value.0);
    states
}

fn prove_state_recurrences(
    func: &Function,
    outer: &NaturalLoop,
    inner_preheader: BlockId,
    outer_latch: BlockId,
    states: &[ValueId],
) -> Option<(BlockId, BlockId)> {
    let mut common_preds = None;
    for state in states {
        let ValueKind::Inst(owner, _) = func.value(*state).kind else {
            return None;
        };
        if owner != inner_preheader {
            return None;
        }
        let InstKind::Phi { incomings } = defining_inst(func, *state)? else {
            return None;
        };
        if incomings.len() != 2 {
            return None;
        }
        let mut matched = None;
        for (reset_pred, reset) in incomings {
            let ValueKind::Const(reset_constant @ (Const::Float(_) | Const::Zero(Type::F32))) =
                &func.value(*reset).kind
            else {
                continue;
            };
            let (update_pred, updated) = incomings
                .iter()
                .find(|(pred, _)| pred != reset_pred)
                .copied()?;
            let InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs,
                rhs,
            } = defining_inst(func, updated)?
            else {
                continue;
            };
            let (old_state, delta) = if is_float_constant(func, *rhs) {
                (*lhs, *rhs)
            } else if is_float_constant(func, *lhs) {
                (*rhs, *lhs)
            } else {
                continue;
            };
            let ValueKind::Inst(old_owner, _) = func.value(old_state).kind else {
                continue;
            };
            if old_owner != outer.header
                || func.value(old_state).ty != Type::F32
                || func.value(delta).ty != Type::F32
            {
                continue;
            }
            let InstKind::Phi {
                incomings: outer_incomings,
            } = defining_inst(func, old_state)?
            else {
                continue;
            };
            let outer_initial = unique_incoming(outer_incomings, outer.unique_entering_pred?)?;
            let outer_next = unique_incoming(outer_incomings, outer_latch)?;
            if outer_next != *state
                || func.value(outer_initial).kind != ValueKind::Const(reset_constant.clone())
            {
                continue;
            }
            matched = Some((*reset_pred, update_pred));
            break;
        }
        let matched = matched?;
        if common_preds
            .replace(matched)
            .is_some_and(|prior| prior != matched)
        {
            return None;
        }
    }
    common_preds
}

fn prove_periodic_branch(
    func: &Function,
    outer: &NaturalLoop,
    induction: ValueId,
    state_merge: BlockId,
    reset_pred: BlockId,
    update_pred: BlockId,
) -> Option<ValueId> {
    for block in &outer.blocks {
        let Some(Terminator::Branch {
            cond,
            then_target,
            else_target,
        }) = func.blocks[block.0].terminator.as_ref()
        else {
            continue;
        };
        let reset_is_true = if *block == reset_pred
            && *then_target == state_merge
            && *else_target == update_pred
        {
            true
        } else if *block == reset_pred && *else_target == state_merge && *then_target == update_pred
        {
            false
        } else if *then_target == reset_pred && *else_target == update_pred {
            true
        } else if *else_target == reset_pred && *then_target == update_pred {
            false
        } else {
            continue;
        };
        let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
            continue;
        };
        let (remainder, zero) = if const_i32(func, *rhs) == Some(0) {
            (*lhs, *rhs)
        } else if const_i32(func, *lhs) == Some(0) {
            (*rhs, *lhs)
        } else {
            continue;
        };
        let _ = zero;
        let InstKind::Binary {
            op: BinaryOp::Imod,
            lhs: dividend,
            rhs: divisor,
        } = defining_inst(func, remainder)?
        else {
            continue;
        };
        let period = const_i32(func, *divisor)?;
        if *dividend != induction || period < 2 || !matches!(op, CmpOp::Eq | CmpOp::Ne) {
            continue;
        }
        let true_is_nonzero = *op == CmpOp::Ne;
        if true_is_nonzero == reset_is_true {
            return Some(*cond);
        }
    }
    None
}

fn accumulator_external_uses_are_in_exit(
    func: &Function,
    inner: &NaturalLoop,
    accumulator: ValueId,
    exit: BlockId,
) -> bool {
    for (block_index, block) in func.blocks.iter().enumerate() {
        let block_id = BlockId(block_index);
        if inner.blocks.contains(&block_id) || block_id == exit {
            continue;
        }
        if block
            .insts
            .iter()
            .any(|inst| inst_operands(&inst.kind).contains(&accumulator))
            || terminator_operands(block.terminator.as_ref()).contains(&accumulator)
        {
            return false;
        }
    }
    true
}

fn apply_candidate(func: &mut Function, candidate: Candidate) {
    let zero_float = get_or_add_const(func, Const::Float(0));
    let false_value = get_or_add_const(func, Const::Bool(false));
    let true_value = get_or_add_const(func, Const::Bool(true));

    let phi_index = func.blocks[candidate.outer_header.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Phi { .. }))
        .count();
    let cache_true = func
        .insert_inst(
            candidate.outer_header,
            phi_index,
            InstKind::Phi {
                incomings: vec![(candidate.outer_entering, zero_float)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let cache_false = func
        .insert_inst(
            candidate.outer_header,
            phi_index + 1,
            InstKind::Phi {
                incomings: vec![(candidate.outer_entering, zero_float)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let initialized_true = func
        .insert_inst(
            candidate.outer_header,
            phi_index + 2,
            InstKind::Phi {
                incomings: vec![(candidate.outer_entering, false_value)],
            },
            Some(Type::I1),
        )
        .unwrap();
    let initialized_false = func
        .insert_inst(
            candidate.outer_header,
            phi_index + 3,
            InstKind::Phi {
                incomings: vec![(candidate.outer_entering, false_value)],
            },
            Some(Type::I1),
        )
        .unwrap();

    let check_true = func.add_block("periodic.check.true");
    let check_false = func.add_block("periodic.check.false");
    let reuse_true = func.add_block("periodic.reuse.true");
    let reuse_false = func.add_block("periodic.reuse.false");
    let update_true = func.add_block("periodic.update.true");
    let update_false = func.add_block("periodic.update.false");
    let merge = func.add_block("periodic.merge");

    func.blocks[candidate.inner_preheader.0].terminator = Some(Terminator::Branch {
        cond: candidate.branch_condition,
        then_target: check_true,
        else_target: check_false,
    });
    func.set_terminator(
        check_true,
        Terminator::Branch {
            cond: initialized_true,
            then_target: reuse_true,
            else_target: candidate.inner_header,
        },
    );
    func.set_terminator(
        check_false,
        Terminator::Branch {
            cond: initialized_false,
            then_target: reuse_false,
            else_target: candidate.inner_header,
        },
    );
    for inst in &mut func.blocks[candidate.inner_header.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        if let Some(index) = incomings
            .iter()
            .position(|(pred, _)| *pred == candidate.inner_preheader)
        {
            let value = incomings[index].1;
            incomings[index] = (check_true, value);
            incomings.push((check_false, value));
        }
    }

    let suffix = func.blocks[candidate.inner_exit.0].insts.clone();
    for inst in &mut func.blocks[candidate.inner_exit.0].insts {
        inst.result = None;
        inst.kind = InstKind::Nop;
    }
    let old_terminator = func.blocks[candidate.inner_exit.0]
        .terminator
        .take()
        .expect("outer latch must terminate");
    func.blocks[candidate.inner_exit.0].terminator = Some(Terminator::Branch {
        cond: candidate.branch_condition,
        then_target: update_true,
        else_target: update_false,
    });
    func.set_terminator(update_true, Terminator::Jump(merge));
    func.set_terminator(update_false, Terminator::Jump(merge));
    func.set_terminator(reuse_true, Terminator::Jump(merge));
    func.set_terminator(reuse_false, Terminator::Jump(merge));

    let selected = func
        .append_inst(
            merge,
            InstKind::Phi {
                incomings: vec![
                    (update_true, candidate.accumulator),
                    (update_false, candidate.accumulator),
                    (reuse_true, cache_true),
                    (reuse_false, cache_false),
                ],
            },
            Some(Type::F32),
        )
        .unwrap();
    let next_cache_true = func
        .append_inst(
            merge,
            InstKind::Phi {
                incomings: vec![
                    (update_true, candidate.accumulator),
                    (update_false, cache_true),
                    (reuse_true, cache_true),
                    (reuse_false, cache_true),
                ],
            },
            Some(Type::F32),
        )
        .unwrap();
    let next_cache_false = func
        .append_inst(
            merge,
            InstKind::Phi {
                incomings: vec![
                    (update_true, cache_false),
                    (update_false, candidate.accumulator),
                    (reuse_true, cache_false),
                    (reuse_false, cache_false),
                ],
            },
            Some(Type::F32),
        )
        .unwrap();
    let next_initialized_true = func
        .append_inst(
            merge,
            InstKind::Phi {
                incomings: vec![
                    (update_true, true_value),
                    (update_false, initialized_true),
                    (reuse_true, initialized_true),
                    (reuse_false, initialized_true),
                ],
            },
            Some(Type::I1),
        )
        .unwrap();
    let next_initialized_false = func
        .append_inst(
            merge,
            InstKind::Phi {
                incomings: vec![
                    (update_true, initialized_false),
                    (update_false, true_value),
                    (reuse_true, initialized_false),
                    (reuse_false, initialized_false),
                ],
            },
            Some(Type::I1),
        )
        .unwrap();

    let prefix = func.blocks[merge.0].insts.len();
    for (offset, mut inst) in suffix.into_iter().enumerate() {
        rewrite_inst_value(&mut inst, candidate.accumulator, selected);
        if let Some(result) = inst.result {
            func.values[result.0].kind = ValueKind::Inst(merge, prefix + offset);
        }
        func.blocks[merge.0].insts.push(inst);
    }
    let mut new_terminator = old_terminator;
    rewrite_terminator_value(&mut new_terminator, candidate.accumulator, selected);
    retarget_successor_phis(
        func,
        &new_terminator,
        candidate.inner_exit,
        merge,
        candidate.accumulator,
        selected,
    );
    func.blocks[merge.0].terminator = Some(new_terminator);

    for inst in &mut func.blocks[candidate.outer_header.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        for (pred, _) in incomings.iter_mut() {
            if *pred == candidate.outer_latch {
                *pred = merge;
            }
        }
    }
    for (phi, backedge) in [
        (cache_true, next_cache_true),
        (cache_false, next_cache_false),
        (initialized_true, next_initialized_true),
        (initialized_false, next_initialized_false),
    ] {
        let InstKind::Phi { incomings } = defining_inst_mut(func, phi) else {
            unreachable!("cache value must remain an outer phi");
        };
        incomings.push((merge, backedge));
    }

    if let Err(errors) = func.verify() {
        panic!(
            "periodic reduction memoization produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn retarget_successor_phis(
    func: &mut Function,
    terminator: &Terminator,
    old_pred: BlockId,
    new_pred: BlockId,
    old_value: ValueId,
    new_value: ValueId,
) {
    for successor in terminator_targets(terminator) {
        for inst in &mut func.blocks[successor.0].insts {
            let InstKind::Phi { incomings } = &mut inst.kind else {
                continue;
            };
            for (pred, value) in incomings {
                if *pred == old_pred {
                    *pred = new_pred;
                }
                if *value == old_value {
                    *value = new_value;
                }
            }
        }
    }
}

fn defining_inst_mut(func: &mut Function, value: ValueId) -> &mut InstKind {
    let ValueKind::Inst(block, inst_idx) = func.values[value.0].kind else {
        panic!("expected instruction-backed value");
    };
    &mut func.blocks[block.0].insts[inst_idx].kind
}

fn rewrite_inst_value(inst: &mut Inst, old: ValueId, new: ValueId) {
    let replace = |value: &mut ValueId| {
        if *value == old {
            *value = new;
        }
    };
    match &mut inst.kind {
        InstKind::Nop | InstKind::Alloca { .. } => {}
        InstKind::Phi { incomings } => {
            for (_, value) in incomings {
                replace(value);
            }
        }
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => replace(ptr),
        InstKind::Store { ptr, value } => {
            replace(ptr);
            replace(value);
        }
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => replace(value),
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            replace(lhs);
            replace(rhs);
        }
        InstKind::Gep { base, indices } => {
            replace(base);
            for index in indices {
                replace(index);
            }
        }
        InstKind::Call { args, .. } => {
            for arg in args {
                replace(arg);
            }
        }
    }
}

fn rewrite_terminator_value(terminator: &mut Terminator, old: ValueId, new: ValueId) {
    match terminator {
        Terminator::Return(value) => {
            if value == &Some(old) {
                *value = Some(new);
            }
        }
        Terminator::Branch { cond, .. } if *cond == old => *cond = new,
        Terminator::Jump(_) | Terminator::Branch { .. } => {}
    }
}

fn unique_incoming(incomings: &[(BlockId, ValueId)], pred: BlockId) -> Option<ValueId> {
    let mut values = incomings
        .iter()
        .filter_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value));
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn is_float_constant(func: &Function, value: ValueId) -> bool {
    matches!(
        func.value(value).kind,
        ValueKind::Const(Const::Float(_) | Const::Zero(Type::F32))
    )
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(value),
        ValueKind::Const(Const::Zero(Type::I32)) => Some(0),
        _ => None,
    }
}

fn get_or_add_const(func: &mut Function, constant: Const) -> ValueId {
    func.values
        .iter()
        .position(|value| value.kind == ValueKind::Const(constant.clone()))
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(constant))
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
            let mut operands = vec![*base];
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: Option<&Terminator>) -> Vec<ValueId> {
    match terminator {
        Some(Terminator::Return(value)) => value.iter().copied().collect(),
        Some(Terminator::Branch { cond, .. }) => vec![*cond],
        Some(Terminator::Jump(_)) | None => Vec::new(),
    }
}

fn terminator_targets(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Return(_) => Vec::new(),
        Terminator::Jump(target) => vec![*target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } if then_target == else_target => vec![*then_target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => vec![*then_target, *else_target],
    }
}

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, OptLevel, PassOptions};
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    #[test]
    fn memoizes_only_a_proven_periodic_reset_reduction() {
        let optimized = optimized_module(0.0);
        let main = optimized
            .funcs
            .iter()
            .find(|func| func.name == "main")
            .unwrap();
        assert!(
            main.blocks
                .iter()
                .any(|block| block.name.starts_with("periodic.")),
            "blocks: {:?}",
            main.blocks
                .iter()
                .map(|block| &block.name)
                .collect::<Vec<_>>()
        );
        assert!(optimized.funcs.iter().all(|func| func.verify().is_ok()));

        let rejected = optimized_module(2.0);
        let main = rejected
            .funcs
            .iter()
            .find(|func| func.name == "main")
            .unwrap();
        assert!(main
            .blocks
            .iter()
            .all(|block| !block.name.starts_with("periodic.")));
    }

    fn optimized_module(initial_a: f32) -> crate::ir::Module {
        let source = format!(
            r#"
float dot(float x[], float y[], int length) {{
    int i = 0;
    float sum = 0.0;
    while (i < length) {{
        sum = sum + x[i] * y[i];
        i = i + 1;
    }}
    return sum;
}}

int main() {{
    int i = 0;
    float x[4];
    float y[4];
    int len = 4;
    float total = 0.0;
    float a = {initial_a:.1};
    float b = 1.0;
    while (i < 20) {{
        int j = 0;
        if (i % 3) {{
            a = 0.0;
            b = 1.0;
        }} else {{
            a = a + 0.1;
            b = b + 0.2;
        }}
        while (j < len) {{
            x[j] = a + j;
            y[j] = b + j;
            j = j + 1;
        }}
        total = total + dot(x, y, len);
        i = i + 1;
    }}
    return total > 0.0;
}}
"#
        );
        let mut parser = Parser::new(&source);
        let program = parser.parse_program();
        let mut module = lower_program(&program).expect("test source must lower");
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                small_expr_inline_rounds: 1,
                cfg_inline_rounds: 1,
                cfg_inline_global_loads: false,
                enable_constant_address_count_reduction: false,
                enable_recursive_const_specialization: false,
                enable_loop_call_memoize: false,
                enable_loop_invariant_call_memoize: false,
                enable_repeated_overwrite_elision: false,
                enable_guarded_mulmod_idiom: false,
                enable_guarded_pow2_digit_idiom: false,
                enable_regional_global_scalar_promotion: false,
                enable_producer_consumer_fusion: true,
                enable_periodic_reduction_memoize: true,
                enable_write_only_alloca_cleanup_before_inline: true,
            },
        );
        module
    }
}
