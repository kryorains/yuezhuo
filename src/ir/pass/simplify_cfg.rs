use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{BlockId, Const, Function, InstKind, Module, Terminator, ValueKind};
use std::collections::HashSet;

pub(super) struct SimplifyCfgPass;

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

fn merge_linear_block(func: &mut Function) -> bool {
    let predecessors = all_predecessors(func);
    for block_idx in 0..func.blocks.len() {
        let block = BlockId(block_idx);
        let Some(Terminator::Jump(successor)) = func.blocks[block_idx].terminator.as_ref() else {
            continue;
        };
        let successor = *successor;
        if successor == block
            || successor == func.entry
            || predecessors[successor.0].len() != 1
            || !predecessors[successor.0].contains(&block)
            || func.blocks[successor.0].terminator.is_none()
        {
            continue;
        }
        if func.blocks[successor.0]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Phi { .. }) && inst.result.is_some())
        {
            continue;
        }

        let moved_insts = std::mem::take(&mut func.blocks[successor.0].insts);
        let moved_terminator = func.blocks[successor.0]
            .terminator
            .take()
            .expect("validated successor must have a terminator");
        let first_moved_idx = func.blocks[block_idx].insts.len();
        for (offset, inst) in moved_insts.iter().enumerate() {
            if let Some(result) = inst.result {
                func.values[result.0].kind = ValueKind::Inst(block, first_moved_idx + offset);
            }
        }
        func.blocks[block_idx].insts.extend(moved_insts);
        func.blocks[block_idx].terminator = Some(moved_terminator);
        for candidate in &mut func.blocks {
            for inst in &mut candidate.insts {
                let InstKind::Phi { incomings } = &mut inst.kind else {
                    if !matches!(inst.kind, InstKind::Nop) {
                        break;
                    }
                    continue;
                };
                for (pred, _) in incomings {
                    if *pred == successor {
                        *pred = block;
                    }
                }
            }
        }
        return true;
    }
    false
}

fn forward_empty_jump_block(func: &mut Function) -> bool {
    let predecessors = all_predecessors(func);
    for (block_idx, block_predecessors) in predecessors.iter().enumerate() {
        let block = BlockId(block_idx);
        if block == func.entry
            || func.blocks[block_idx]
                .insts
                .iter()
                .any(|inst| !matches!(inst.kind, InstKind::Nop))
        {
            continue;
        }
        let Some(Terminator::Jump(target)) = func.blocks[block_idx].terminator.as_ref() else {
            continue;
        };
        let target = *target;
        let mut preds = block_predecessors.iter().copied().collect::<Vec<_>>();
        preds.sort_by_key(|pred| pred.0);
        if target == block || preds.is_empty() {
            continue;
        }
        if preds.iter().any(|pred| {
            terminator_has_target_other_than(func.blocks[pred.0].terminator.as_ref(), block, target)
        }) {
            continue;
        }

        let mut phi_updates = Vec::new();
        let mut valid = true;
        for (inst_idx, inst) in func.blocks[target.0].insts.iter().enumerate() {
            let InstKind::Phi { incomings } = &inst.kind else {
                if !matches!(inst.kind, InstKind::Nop) {
                    break;
                }
                continue;
            };
            let Some((_, incoming)) = incomings.iter().find(|(pred, _)| *pred == block) else {
                valid = false;
                break;
            };
            if preds.iter().any(|pred| {
                incomings
                    .iter()
                    .any(|(incoming_pred, _)| incoming_pred == pred)
            }) {
                valid = false;
                break;
            }
            phi_updates.push((inst_idx, *incoming));
        }
        if !valid {
            continue;
        }

        for pred in &preds {
            redirect_terminator_target(
                func.blocks[pred.0]
                    .terminator
                    .as_mut()
                    .expect("predecessor must have a terminator"),
                block,
                target,
            );
        }
        for (inst_idx, incoming) in phi_updates {
            let InstKind::Phi { incomings } = &mut func.blocks[target.0].insts[inst_idx].kind
            else {
                unreachable!("validated phi changed before update");
            };
            incomings.retain(|(pred, _)| *pred != block);
            incomings.extend(preds.iter().map(|pred| (*pred, incoming)));
        }
        return true;
    }
    false
}

fn terminator_has_target_other_than(
    terminator: Option<&Terminator>,
    excluded: BlockId,
    target: BlockId,
) -> bool {
    match terminator {
        Some(Terminator::Jump(current)) => *current == target && *current != excluded,
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) => {
            (*then_target == target && *then_target != excluded)
                || (*else_target == target && *else_target != excluded)
        }
        Some(Terminator::Return(_)) | None => false,
    }
}

fn redirect_terminator_target(
    terminator: &mut Terminator,
    old_target: BlockId,
    new_target: BlockId,
) {
    match terminator {
        Terminator::Jump(target) => {
            if *target == old_target {
                *target = new_target;
            }
        }
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => {
            if *then_target == old_target {
                *then_target = new_target;
            }
            if *else_target == old_target {
                *else_target = new_target;
            }
        }
        Terminator::Return(_) => {}
    }
}

fn remove_unreachable_blocks(func: &mut Function) -> bool {
    let mut reachable = vec![false; func.blocks.len()];
    let mut worklist = vec![func.entry];
    while let Some(block) = worklist.pop() {
        if reachable[block.0] {
            continue;
        }
        reachable[block.0] = true;
        match func.blocks[block.0].terminator.as_ref() {
            Some(Terminator::Jump(target)) => worklist.push(*target),
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => {
                worklist.push(*then_target);
                worklist.push(*else_target);
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }
    if reachable.iter().all(|is_reachable| *is_reachable) {
        return false;
    }

    let mut block_map = vec![None; func.blocks.len()];
    let mut next_block = 0usize;
    for (old_idx, is_reachable) in reachable.iter().copied().enumerate() {
        if is_reachable {
            block_map[old_idx] = Some(BlockId(next_block));
            next_block += 1;
        }
    }

    for value in &mut func.values {
        let ValueKind::Inst(owner, _) = value.kind else {
            continue;
        };
        if let Some(new_owner) = block_map[owner.0] {
            if let ValueKind::Inst(current_owner, _) = &mut value.kind {
                *current_owner = new_owner;
            }
        } else {
            // ValueIds are stable across passes, so tombstone removed definitions
            // instead of renumbering every value in the function.
            value.kind = ValueKind::Const(Const::Zero(value.ty.clone()));
        }
    }

    let old_blocks = std::mem::take(&mut func.blocks);
    func.blocks = old_blocks
        .into_iter()
        .enumerate()
        .filter_map(|(old_idx, mut block)| {
            reachable[old_idx].then(|| {
                for inst in &mut block.insts {
                    if let InstKind::Phi { incomings } = &mut inst.kind {
                        incomings.retain_mut(|(pred, _)| {
                            let Some(new_pred) = block_map[pred.0] else {
                                return false;
                            };
                            *pred = new_pred;
                            true
                        });
                    }
                }
                if let Some(terminator) = &mut block.terminator {
                    match terminator {
                        Terminator::Jump(target) => {
                            *target = block_map[target.0]
                                .expect("reachable jump target must remain reachable");
                        }
                        Terminator::Branch {
                            then_target,
                            else_target,
                            ..
                        } => {
                            *then_target = block_map[then_target.0]
                                .expect("reachable branch target must remain reachable");
                            *else_target = block_map[else_target.0]
                                .expect("reachable branch target must remain reachable");
                        }
                        Terminator::Return(_) => {}
                    }
                }
                block
            })
        })
        .collect();
    func.entry = block_map[func.entry.0].expect("entry block must remain reachable");
    true
}

fn thread_boolean_phi_branches(func: &mut Function) -> bool {
    let use_counts = value_use_counts(func);
    let predecessor_sets = all_predecessors(func);
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
                && !successor_has_phi_from(func, *then_target, BlockId(block_idx))
                && !successor_has_phi_from(func, *else_target, BlockId(block_idx)))
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

fn successor_has_phi_from(func: &Function, successor: BlockId, pred: BlockId) -> bool {
    func.blocks[successor.0].insts.iter().any(|inst| {
        matches!(
            &inst.kind,
            InstKind::Phi { incomings }
                if incomings.iter().any(|(incoming_pred, _)| *incoming_pred == pred)
        )
    })
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
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Type};

    fn run_pass(func: Function) -> Function {
        let mut module = Module::new();
        module.add_func(func);
        SimplifyCfgPass::new().run(&mut module);
        module.funcs.remove(0)
    }

    #[test]
    fn removes_unreachable_blocks_and_remaps_ids() {
        let mut func = Function::new("unreachable", Type::I32);
        let cond = func.add_param("cond", Type::I1);
        let live = func.add_block("live");
        let dead = func.add_block("dead");
        let exit = func.add_block("exit");
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond,
                then_target: live,
                else_target: exit,
            },
        );
        let live_value = func
            .append_inst(
                live,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: one,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(live, Terminator::Jump(exit));
        let dead_value = func
            .append_inst(
                dead,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: one,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(dead, Terminator::Return(Some(dead_value)));
        let result = func
            .append_inst(
                exit,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero), (live, live_value)],
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(exit, Terminator::Return(Some(result)));
        assert!(func.verify().is_ok());

        let func = run_pass(func);

        assert_eq!(
            func.blocks
                .iter()
                .map(|block| block.name.as_str())
                .collect::<Vec<_>>(),
            vec!["entry", "live", "exit"]
        );
        assert!(matches!(
            func.value(dead_value).kind,
            ValueKind::Const(Const::Zero(Type::I32))
        ));
        assert!(matches!(
            func.blocks[0].terminator,
            Some(Terminator::Branch {
                then_target: BlockId(1),
                else_target: BlockId(2),
                ..
            })
        ));
        assert!(func.verify().is_ok());
    }

    #[test]
    fn forwards_empty_jump_blocks_and_repairs_phis() {
        let mut func = Function::new("empty_jump", Type::I32);
        let cond = func.add_param("cond", Type::I1);
        let input = func.add_param("input", Type::I32);
        let left = func.add_block("left");
        let right = func.add_block("right");
        let empty = func.add_block("empty");
        let target = func.add_block("target");
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond,
                then_target: left,
                else_target: right,
            },
        );
        for block in [left, right] {
            func.append_inst(
                block,
                InstKind::Call {
                    name: "putint".to_string(),
                    args: vec![input],
                },
                None,
            );
            func.set_terminator(block, Terminator::Jump(empty));
        }
        func.set_terminator(empty, Terminator::Jump(target));
        let merged = func
            .append_inst(
                target,
                InstKind::Phi {
                    incomings: vec![(empty, input)],
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(target, Terminator::Return(Some(merged)));
        assert!(func.verify().is_ok());

        let func = run_pass(func);

        assert_eq!(
            func.blocks
                .iter()
                .map(|block| block.name.as_str())
                .collect::<Vec<_>>(),
            vec!["entry", "left", "right", "target"]
        );
        for block_idx in [1, 2] {
            assert_eq!(
                func.blocks[block_idx].terminator,
                Some(Terminator::Jump(BlockId(3)))
            );
        }
        assert!(matches!(func.blocks[3].insts[0].kind, InstKind::Nop));
        assert_eq!(
            func.blocks[3].terminator,
            Some(Terminator::Return(Some(input)))
        );
        assert!(func.verify().is_ok());
    }

    #[test]
    fn merges_linear_blocks_and_rewrites_phi_values() {
        let mut func = Function::new("linear", Type::I32);
        let input = func.add_param("input", Type::I32);
        let body = func.add_block("body");
        let one = func.add_const(Const::Int(1));
        func.set_terminator(func.entry, Terminator::Jump(body));
        let phi = func
            .append_inst(
                body,
                InstKind::Phi {
                    incomings: vec![(func.entry, input)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let sum = func
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: phi,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(body, Terminator::Return(Some(sum)));
        assert!(func.verify().is_ok());

        let func = run_pass(func);

        assert_eq!(func.blocks.len(), 1);
        assert_eq!(
            func.blocks[0].terminator,
            Some(Terminator::Return(Some(sum)))
        );
        assert!(matches!(
            func.value(sum).kind,
            ValueKind::Inst(BlockId(0), _)
        ));
        let ValueKind::Inst(owner, inst_idx) = func.value(sum).kind else {
            unreachable!();
        };
        assert!(matches!(
            func.blocks[owner.0].insts[inst_idx].kind,
            InstKind::Binary {
                lhs,
                op: BinaryOp::Iadd,
                ..
            } if lhs == input
        ));
        assert!(func.verify().is_ok());
    }

    #[test]
    fn removes_unreachable_phi_cycles_before_rewriting_values() {
        let mut func = Function::new("dead_phi_cycle", Type::Void);
        let dead_a = func.add_block("dead_a");
        let dead_b = func.add_block("dead_b");
        let zero = func.add_const(Const::Int(0));
        func.set_terminator(func.entry, Terminator::Return(None));
        let phi_a = func
            .append_inst(
                dead_a,
                InstKind::Phi {
                    incomings: vec![(dead_b, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let phi_b = func
            .append_inst(
                dead_b,
                InstKind::Phi {
                    incomings: vec![(dead_a, phi_a)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let ValueKind::Inst(owner, inst_idx) = func.value(phi_a).kind else {
            unreachable!();
        };
        let InstKind::Phi { incomings } = &mut func.blocks[owner.0].insts[inst_idx].kind else {
            unreachable!();
        };
        incomings[0].1 = phi_b;
        func.append_inst(
            dead_a,
            InstKind::Call {
                name: "putint".to_string(),
                args: vec![phi_a],
            },
            None,
        );
        func.set_terminator(dead_a, Terminator::Jump(dead_b));
        func.set_terminator(dead_b, Terminator::Jump(dead_a));
        assert!(func.verify().is_ok());

        let func = run_pass(func);

        assert_eq!(func.blocks.len(), 1);
        assert!(matches!(
            func.value(phi_a).kind,
            ValueKind::Const(Const::Zero(Type::I32))
        ));
        assert!(matches!(
            func.value(phi_b).kind,
            ValueKind::Const(Const::Zero(Type::I32))
        ));
        assert!(func.verify().is_ok());
    }

    #[test]
    fn preserves_loop_headers_with_multiple_predecessors() {
        let mut func = Function::new("loop", Type::I32);
        let keep_going = func.add_param("keep_going", Type::I1);
        let header = func.add_block("header");
        let body = func.add_block("body");
        let exit = func.add_block("exit");
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        func.set_terminator(func.entry, Terminator::Jump(header));
        let counter = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero), (body, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: keep_going,
                then_target: body,
                else_target: exit,
            },
        );
        let next = func
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: counter,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        let ValueKind::Inst(owner, inst_idx) = func.value(counter).kind else {
            unreachable!();
        };
        let InstKind::Phi { incomings } = &mut func.blocks[owner.0].insts[inst_idx].kind else {
            unreachable!();
        };
        incomings[1].1 = next;
        func.set_terminator(body, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(Some(counter)));
        assert!(func.verify().is_ok());

        let func = run_pass(func);

        let header = func
            .blocks
            .iter()
            .position(|block| block.name == "header")
            .map(BlockId)
            .expect("loop header must remain present");
        assert_eq!(all_predecessors(&func)[header.0].len(), 2);
        assert!(func.blocks[header.0]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Phi { .. })));
        assert!(func.verify().is_ok());
    }

    #[test]
    fn skips_forwarding_when_phi_predecessors_would_conflict() {
        let mut func = Function::new("conflict", Type::I32);
        let cond = func.add_param("cond", Type::I1);
        let direct = func.add_param("direct", Type::I32);
        let indirect = func.add_param("indirect", Type::I32);
        let empty = func.add_block("empty");
        let target = func.add_block("target");
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond,
                then_target: empty,
                else_target: target,
            },
        );
        func.set_terminator(empty, Terminator::Jump(target));
        let result = func
            .append_inst(
                target,
                InstKind::Phi {
                    incomings: vec![(func.entry, direct), (empty, indirect)],
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(target, Terminator::Return(Some(result)));
        assert!(func.verify().is_ok());
        let before = func.clone();

        let func = run_pass(func);

        assert_eq!(func, before);
        assert!(func.verify().is_ok());
    }

    #[test]
    fn repeated_runs_are_idempotent_and_verified() {
        let mut func = Function::new("idempotent", Type::I32);
        let input = func.add_param("input", Type::I32);
        let empty = func.add_block("empty");
        let exit = func.add_block("exit");
        func.set_terminator(func.entry, Terminator::Jump(empty));
        func.set_terminator(empty, Terminator::Jump(exit));
        func.set_terminator(exit, Terminator::Return(Some(input)));

        let once = run_pass(func);
        let twice = run_pass(once.clone());

        assert_eq!(twice, once);
        assert!(twice.verify().is_ok());
    }
}
