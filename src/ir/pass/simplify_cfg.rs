mod structural;

use super::dominators::{ControlFlowGraph, Dominators};
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CastOp, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::HashSet;
use structural::{forward_empty_jump_block, merge_linear_block, remove_unreachable_blocks};

pub(super) struct SimplifyCfgPass;

const MAX_IF_CONVERSIONS: usize = 64;

impl SimplifyCfgPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for SimplifyCfgPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            simplify_function(func);
        }
    }
}

fn simplify_function(func: &mut Function) {
    let mut changed = false;
    loop {
        let mut round_changed = remove_unreachable_blocks(func);
        round_changed |= simplify_branches(func);
        round_changed |= if_convert_short_circuit_booleans(func);
        round_changed |= if_convert_conditional_updates(func);
        round_changed |= thread_boolean_phi_branches(func);
        round_changed |= simplify_trivial_phis(func);
        round_changed |= forward_empty_jump_block(func);
        round_changed |= merge_linear_block(func);
        changed |= round_changed;
        if !round_changed {
            break;
        }
    }
    if changed {
        if let Err(errors) = func.verify() {
            panic!(
                "simplify-cfg produced invalid IR in {}: {:?}",
                func.name, errors
            );
        }
    }
}

fn simplify_branches(func: &mut Function) -> bool {
    // 先替换 terminator，并记录被删除的 CFG 边；最后再统一修 phi incoming。
    let mut removed_edges = Vec::new();
    let mut changed = false;

    for block_idx in 0..func.blocks.len() {
        let Some(terminator) = func.blocks[block_idx].terminator.clone() else {
            continue;
        };

        let (replacement, removed) = match terminator {
            // then/else 相同的条件跳转没有意义，直接改成无条件跳转。
            Terminator::Branch {
                cond: _,
                then_target,
                else_target,
            } if then_target == else_target => (Terminator::Jump(then_target), Vec::new()),
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => match const_bool(func, cond) {
                // 条件已经是常量时，删掉永远不会走到的那条边。
                Some(true) => (
                    Terminator::Jump(then_target),
                    vec![(BlockId(block_idx), else_target)],
                ),
                Some(false) => (
                    Terminator::Jump(else_target),
                    vec![(BlockId(block_idx), then_target)],
                ),
                None => continue,
            },
            _ => continue,
        };

        func.blocks[block_idx].terminator = Some(replacement);
        removed_edges.extend(removed);
        changed = true;
    }

    for (pred, target) in removed_edges {
        remove_phi_incomings(func, pred, target);
    }
    changed
}

fn if_convert_short_circuit_booleans(func: &mut Function) -> bool {
    let mut changed = false;
    let mut rejected = HashSet::new();
    let mut attempts = 0usize;
    while attempts < MAX_IF_CONVERSIONS {
        let predecessors = all_predecessors(func);
        let use_counts = value_use_counts(func);
        let cfg = ControlFlowGraph::new(func);
        let dominators = Dominators::new(func, &cfg);
        let mut candidate = None;

        for (merge_idx, merge) in func.blocks.iter().enumerate() {
            let active = active_insts(func, BlockId(merge_idx));
            let [phi_idx] = active.as_slice() else {
                continue;
            };
            if rejected.contains(&(merge_idx, *phi_idx)) {
                continue;
            }
            let phi = &merge.insts[*phi_idx];
            let (Some(phi_result), InstKind::Phi { incomings }) = (phi.result, &phi.kind) else {
                continue;
            };
            if func.value(phi_result).ty != Type::I1
                || use_counts[phi_result.0] != 1
                || incomings.len() != 2
            {
                continue;
            }
            let merge_block = BlockId(merge_idx);
            for (direct_pred, constant) in incomings {
                let Some(constant) = const_bool(func, *constant) else {
                    continue;
                };
                let Some((rhs_block, rhs_value)) = incomings
                    .iter()
                    .find(|(pred, _)| pred != direct_pred)
                    .copied()
                else {
                    continue;
                };
                if predecessors[merge_idx] != HashSet::from([*direct_pred, rhs_block])
                    || predecessors[rhs_block.0] != HashSet::from([*direct_pred])
                    || !matches!(
                        func.blocks[rhs_block.0].terminator,
                        Some(Terminator::Jump(target)) if target == merge_block
                    )
                {
                    continue;
                }
                let Some(Terminator::Branch {
                    cond,
                    then_target,
                    else_target,
                }) = func.blocks[direct_pred.0].terminator.as_ref()
                else {
                    continue;
                };
                let direct_when_true = if *then_target == merge_block && *else_target == rhs_block {
                    true
                } else if *else_target == merge_block && *then_target == rhs_block {
                    false
                } else {
                    continue;
                };
                let rhs_active = active_insts(func, rhs_block);
                let [rhs_idx] = rhs_active.as_slice() else {
                    continue;
                };
                let rhs_inst = &func.blocks[rhs_block.0].insts[*rhs_idx];
                if rhs_inst.result != Some(rhs_value)
                    || func.value(rhs_value).ty != Type::I1
                    || use_counts[rhs_value.0] != 1
                    || !is_speculatable_bool_inst(&rhs_inst.kind)
                    || bool_inst_operands(&rhs_inst.kind)
                        .into_iter()
                        .any(|operand| {
                            !value_available_at_block(func, &dominators, operand, *direct_pred)
                        })
                {
                    continue;
                }
                candidate = Some((
                    *direct_pred,
                    *cond,
                    rhs_block,
                    *rhs_idx,
                    rhs_inst.kind.clone(),
                    merge_block,
                    *phi_idx,
                    phi_result,
                    direct_when_true,
                    constant,
                ));
                break;
            }
            if candidate.is_some() {
                break;
            }
        }

        let Some((
            direct_pred,
            condition,
            rhs_block,
            rhs_idx,
            rhs_kind,
            merge,
            phi_idx,
            phi_result,
            direct_when_true,
            constant,
        )) = candidate
        else {
            break;
        };
        attempts += 1;
        let before = func.clone();
        let rhs = func
            .append_inst(direct_pred, rhs_kind, Some(Type::I1))
            .expect("speculated boolean instruction must produce a value");
        let (lhs, op) = match (direct_when_true, constant) {
            (true, true) => (condition, BinaryOp::Or),
            (false, false) => (condition, BinaryOp::And),
            (true, false) => {
                let inverted = func
                    .append_inst(
                        direct_pred,
                        InstKind::Unary {
                            op: crate::ir::UnaryOp::Not,
                            value: condition,
                        },
                        Some(Type::I1),
                    )
                    .expect("boolean inversion must produce a value");
                (inverted, BinaryOp::And)
            }
            (false, true) => {
                let inverted = func
                    .append_inst(
                        direct_pred,
                        InstKind::Unary {
                            op: crate::ir::UnaryOp::Not,
                            value: condition,
                        },
                        Some(Type::I1),
                    )
                    .expect("boolean inversion must produce a value");
                (inverted, BinaryOp::Or)
            }
        };
        let combined = func
            .append_inst(
                direct_pred,
                InstKind::Binary { op, lhs, rhs },
                Some(Type::I1),
            )
            .expect("combined boolean must produce a value");
        func.blocks[direct_pred.0].terminator = Some(Terminator::Jump(merge));
        func.blocks[rhs_block.0].insts[rhs_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        func.blocks[merge.0].insts[phi_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        rewrite_function_uses(func, &ValueReplacements::from([(phi_result, combined)]));
        if func.verify().is_err() {
            *func = before;
            rejected.insert((merge.0, phi_idx));
        } else {
            changed = true;
        }
    }
    changed
}

fn is_speculatable_bool_inst(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Unary {
            op: crate::ir::UnaryOp::Not,
            ..
        } | InstKind::Binary {
            op: BinaryOp::And | BinaryOp::Or,
            ..
        } | InstKind::Icmp { .. }
            | InstKind::Cast {
                op: CastOp::I32ToBool,
                ..
            }
    )
}

fn bool_inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. } | InstKind::Icmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        _ => Vec::new(),
    }
}

fn if_convert_conditional_updates(func: &mut Function) -> bool {
    let mut changed = false;
    let mut rejected = HashSet::new();
    let mut attempts = 0usize;
    while attempts < MAX_IF_CONVERSIONS {
        let predecessors = all_predecessors(func);
        let use_counts = value_use_counts(func);
        let cfg = ControlFlowGraph::new(func);
        let dominators = Dominators::new(func, &cfg);
        let mut candidate = None;

        for (block_idx, block) in func.blocks.iter().enumerate() {
            let Terminator::Branch {
                cond,
                then_target,
                else_target,
            } = block
                .terminator
                .as_ref()
                .cloned()
                .unwrap_or(Terminator::Return(None))
            else {
                continue;
            };
            if then_target == else_target || func.value(cond).ty != Type::I1 {
                continue;
            }
            let block = BlockId(block_idx);
            let then_active = active_insts(func, then_target);
            let else_active = active_insts(func, else_target);
            let then_jump = jump_target(func, then_target);
            let else_jump = jump_target(func, else_target);
            let (update_block, idle_predecessor, update, merge, update_when_true) =
                if then_jump == Some(else_target) {
                    let [update] = then_active.as_slice() else {
                        continue;
                    };
                    if predecessors[then_target.0] != HashSet::from([block]) {
                        continue;
                    }
                    (then_target, block, *update, else_target, true)
                } else if else_jump == Some(then_target) {
                    let [update] = else_active.as_slice() else {
                        continue;
                    };
                    if predecessors[else_target.0] != HashSet::from([block]) {
                        continue;
                    }
                    (else_target, block, *update, then_target, false)
                } else {
                    let Some(merge) = common_jump_target(func, then_target, else_target) else {
                        continue;
                    };
                    if predecessors[then_target.0] != HashSet::from([block])
                        || predecessors[else_target.0] != HashSet::from([block])
                    {
                        continue;
                    }
                    match (then_active.as_slice(), else_active.as_slice()) {
                        ([update], []) => (then_target, else_target, *update, merge, true),
                        ([], [update]) => (else_target, then_target, *update, merge, false),
                        _ => continue,
                    }
                };
            let Some(update_result) = func.blocks[update_block.0].insts[update].result else {
                continue;
            };
            let InstKind::Binary {
                op: update_op @ (BinaryOp::Iadd | BinaryOp::Isub),
                lhs: base,
                rhs: delta,
            } = func.blocks[update_block.0].insts[update].kind
            else {
                continue;
            };
            if use_counts[update_result.0] != 1
                || func.value(base).ty != Type::I32
                || func.value(delta).ty != Type::I32
                || !value_available_at_block(func, &dominators, base, BlockId(block_idx))
                || !value_available_at_block(func, &dominators, delta, BlockId(block_idx))
            {
                continue;
            }

            let leading_phis = func.blocks[merge.0]
                .insts
                .iter()
                .enumerate()
                .filter(|(_, inst)| !matches!(inst.kind, InstKind::Nop))
                .take_while(|(_, inst)| matches!(inst.kind, InstKind::Phi { .. }))
                .collect::<Vec<_>>();
            let [(phi_idx, phi)] = leading_phis.as_slice() else {
                continue;
            };
            if rejected.contains(&(block_idx, *phi_idx)) {
                continue;
            }
            let (Some(phi_result), InstKind::Phi { incomings }) = (phi.result, &phi.kind) else {
                continue;
            };
            if incomings.len() != 2
                || phi_incoming(incomings, update_block) != Some(update_result)
                || phi_incoming(incomings, idle_predecessor) != Some(base)
            {
                continue;
            }
            candidate = Some((
                BlockId(block_idx),
                cond,
                update_block,
                update,
                merge,
                *phi_idx,
                phi_result,
                update_op,
                base,
                delta,
                update_when_true,
            ));
            break;
        }

        let Some((
            block,
            cond,
            update_block,
            update_idx,
            merge,
            phi_idx,
            phi_result,
            update_op,
            base,
            delta,
            update_when_true,
        )) = candidate
        else {
            break;
        };
        attempts += 1;
        let before = func.clone();
        let selected_cond = if update_when_true {
            cond
        } else {
            func.append_inst(
                block,
                InstKind::Unary {
                    op: crate::ir::UnaryOp::Not,
                    value: cond,
                },
                Some(Type::I1),
            )
            .expect("conditional inversion must produce a value")
        };
        let as_i32 = func
            .append_inst(
                block,
                InstKind::Cast {
                    op: CastOp::BoolToI32,
                    value: selected_cond,
                },
                Some(Type::I32),
            )
            .expect("boolean cast must produce a value");
        let scaled = func
            .append_inst(
                block,
                InstKind::Binary {
                    op: BinaryOp::Imul,
                    lhs: delta,
                    rhs: as_i32,
                },
                Some(Type::I32),
            )
            .expect("conditional scale must produce a value");
        let selected = func
            .append_inst(
                block,
                InstKind::Binary {
                    op: update_op,
                    lhs: base,
                    rhs: scaled,
                },
                Some(Type::I32),
            )
            .expect("conditional update must produce a value");

        func.blocks[block.0].terminator = Some(Terminator::Jump(merge));
        func.blocks[update_block.0].insts[update_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        func.blocks[merge.0].insts[phi_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        rewrite_function_uses(func, &ValueReplacements::from([(phi_result, selected)]));
        if func.verify().is_err() {
            *func = before;
            rejected.insert((block.0, phi_idx));
        } else {
            changed = true;
        }
    }
    changed
}

fn jump_target(func: &Function, block: BlockId) -> Option<BlockId> {
    match func.blocks[block.0].terminator.as_ref() {
        Some(Terminator::Jump(target)) => Some(*target),
        _ => None,
    }
}

fn common_jump_target(func: &Function, lhs: BlockId, rhs: BlockId) -> Option<BlockId> {
    match (
        func.blocks[lhs.0].terminator.as_ref(),
        func.blocks[rhs.0].terminator.as_ref(),
    ) {
        (Some(Terminator::Jump(lhs)), Some(Terminator::Jump(rhs))) if lhs == rhs => Some(*lhs),
        _ => None,
    }
}

fn active_insts(func: &Function, block: BlockId) -> Vec<usize> {
    func.blocks[block.0]
        .insts
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| (!matches!(inst.kind, InstKind::Nop)).then_some(idx))
        .collect()
}

fn phi_incoming(incomings: &[(BlockId, ValueId)], predecessor: BlockId) -> Option<ValueId> {
    let mut matches = incomings
        .iter()
        .filter_map(|(pred, value)| (*pred == predecessor).then_some(*value));
    let value = matches.next()?;
    matches.next().is_none().then_some(value)
}

fn value_available_at_block(
    func: &Function,
    dominators: &Dominators,
    value: ValueId,
    block: BlockId,
) -> bool {
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(owner, _) => owner == block || dominators.dominates(owner, block),
    }
}

fn simplify_trivial_phis(func: &mut Function) -> bool {
    let mut replacements = ValueReplacements::new();
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            let InstKind::Phi { incomings } = &inst.kind else {
                continue;
            };
            let Some(result) = inst.result else {
                continue;
            };
            let Some(first) = incomings.first().map(|(_, value)| *value) else {
                continue;
            };
            if incomings.iter().all(|(_, value)| *value == first) {
                replacements.insert(result, first);
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
        }
    }
    let changed = !replacements.is_empty();
    rewrite_function_uses(func, &replacements);
    changed
}

fn thread_boolean_phi_branches(func: &mut Function) -> bool {
    let use_counts = value_use_counts(func);
    let predecessor_sets = all_predecessors(func);
    let phi_predecessors = func
        .blocks
        .iter()
        .map(|block| {
            block
                .insts
                .iter()
                .filter_map(|inst| match &inst.kind {
                    InstKind::Phi { incomings } => Some(incomings),
                    _ => None,
                })
                .flat_map(|incomings| incomings.iter().map(|(pred, _)| *pred))
                .collect::<HashSet<_>>()
        })
        .collect::<Vec<_>>();
    let candidates = func
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(block_idx, block)| {
            let Terminator::Branch {
                cond,
                then_target,
                else_target,
            } = block.terminator.as_ref()?
            else {
                return None;
            };
            let mut active = block
                .insts
                .iter()
                .filter(|inst| !matches!(inst.kind, InstKind::Nop));
            let phi = active.next()?;
            if active.next().is_some() || phi.result != Some(*cond) {
                return None;
            }
            let InstKind::Phi { incomings } = &phi.kind else {
                return None;
            };
            (use_counts[cond.0] == 1
                && (block_returns(func, *then_target) || block_returns(func, *else_target))
                && !phi_predecessors[then_target.0].contains(&BlockId(block_idx))
                && !phi_predecessors[else_target.0].contains(&BlockId(block_idx)))
            .then_some((
                BlockId(block_idx),
                *cond,
                *then_target,
                *else_target,
                incomings.clone(),
            ))
        })
        .collect::<Vec<_>>();

    let mut changed = false;
    for (block, condition, then_target, else_target, incomings) in candidates {
        if !matches!(
            func.blocks[block.0].terminator.as_ref(),
            Some(Terminator::Branch {
                cond,
                then_target: current_then,
                else_target: current_else,
            }) if *cond == condition
                && *current_then == then_target
                && *current_else == else_target
        ) {
            continue;
        }

        let cfg_predecessors = predecessor_sets[block.0].clone();
        let incoming_predecessors = incomings
            .iter()
            .map(|(pred, _)| *pred)
            .collect::<HashSet<_>>();
        if cfg_predecessors != incoming_predecessors
            || incomings.iter().any(|(pred, incoming)| {
                if *pred == block {
                    return true;
                }
                match func.blocks[pred.0].terminator.as_ref() {
                    Some(Terminator::Jump(target)) if *target == block => false,
                    Some(Terminator::Branch {
                        then_target,
                        else_target,
                        ..
                    }) if (*then_target == block || *else_target == block) => {
                        const_bool(func, *incoming).is_none()
                    }
                    _ => true,
                }
            })
        {
            continue;
        }

        for (pred, incoming) in incomings {
            let selected =
                const_bool(func, incoming)
                    .map(|condition| if condition { then_target } else { else_target });
            let terminator = func.blocks[pred.0]
                .terminator
                .as_mut()
                .expect("validated predecessor must have a terminator");
            match terminator {
                Terminator::Jump(target) if *target == block => {
                    *terminator = if let Some(target) = selected {
                        Terminator::Jump(target)
                    } else {
                        Terminator::Branch {
                            cond: incoming,
                            then_target,
                            else_target,
                        }
                    };
                }
                Terminator::Branch {
                    then_target: pred_then,
                    else_target: pred_else,
                    ..
                } => {
                    let target = selected.expect("dynamic branch edge was rejected");
                    if *pred_then == block {
                        *pred_then = target;
                    }
                    if *pred_else == block {
                        *pred_else = target;
                    }
                }
                _ => unreachable!("predecessor changed after validation"),
            }
        }
        for inst in &mut func.blocks[block.0].insts {
            if inst.result == Some(condition) {
                inst.result = None;
                inst.kind = InstKind::Nop;
                break;
            }
        }
        func.blocks[block.0].terminator = Some(Terminator::Jump(else_target));
        changed = true;
    }
    changed
}

fn all_predecessors(func: &Function) -> Vec<HashSet<BlockId>> {
    let mut predecessors = vec![HashSet::new(); func.blocks.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let pred = BlockId(block_idx);
        match block.terminator.as_ref() {
            Some(Terminator::Jump(target)) => {
                predecessors[target.0].insert(pred);
            }
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => {
                predecessors[then_target.0].insert(pred);
                predecessors[else_target.0].insert(pred);
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }
    predecessors
}

fn block_returns(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0]
        .insts
        .iter()
        .all(|inst| matches!(inst.kind, InstKind::Nop))
        && matches!(func.blocks[block.0].terminator, Some(Terminator::Return(_)))
}

fn value_use_counts(func: &Function) -> Vec<usize> {
    let mut counts = vec![0usize; func.values.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            let operands = match &inst.kind {
                InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
                InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
                InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
                InstKind::Store { ptr, value } => vec![*ptr, *value],
                InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
                InstKind::Binary { lhs, rhs, .. }
                | InstKind::Icmp { lhs, rhs, .. }
                | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
                InstKind::Gep { base, indices } => {
                    let mut operands = Vec::with_capacity(indices.len() + 1);
                    operands.push(*base);
                    operands.extend(indices.iter().copied());
                    operands
                }
                InstKind::Call { args, .. } => args.clone(),
            };
            for operand in operands {
                counts[operand.0] += 1;
            }
        }
        for operand in match block.terminator.as_ref() {
            Some(Terminator::Return(Some(value))) => vec![*value],
            Some(Terminator::Branch { cond, .. }) => vec![*cond],
            Some(Terminator::Return(None) | Terminator::Jump(_)) | None => Vec::new(),
        } {
            counts[operand.0] += 1;
        }
    }
    counts
}

fn remove_phi_incomings(func: &mut Function, pred: BlockId, target: BlockId) {
    // CFG 边被删后，目标块 phi 中来自这个前驱的值也必须同步删除。
    for inst in &mut func.blocks[target.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            if !matches!(inst.kind, InstKind::Nop) {
                break;
            }
            continue;
        };
        incomings.retain(|(incoming_pred, _)| *incoming_pred != pred);
    }
}

fn const_bool(func: &Function, value: crate::ir::ValueId) -> Option<bool> {
    // 分支条件允许 bool/int/float 常量：非零视为 true。
    match &func.value(value).kind {
        ValueKind::Const(Const::Bool(value)) => Some(*value),
        ValueKind::Const(Const::Int(value)) => Some(*value != 0),
        ValueKind::Const(Const::Float(bits)) => Some(f32::from_bits(*bits) != 0.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
