use super::all_predecessors;
use crate::ir::{BlockId, Const, Function, InstKind, Terminator, ValueKind};

pub(super) fn merge_linear_block(func: &mut Function) -> bool {
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

pub(super) fn forward_empty_jump_block(func: &mut Function) -> bool {
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

pub(super) fn remove_unreachable_blocks(func: &mut Function) -> bool {
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
