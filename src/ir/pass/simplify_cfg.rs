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
    // 先替换 terminator，并记录被删除的 CFG 边；最后再统一修 phi incoming。
    let mut removed_edges = Vec::new();

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
    }

    for (pred, target) in removed_edges {
        remove_phi_incomings(func, pred, target);
    }
    thread_boolean_phi_branches(func);
}

fn thread_boolean_phi_branches(func: &mut Function) {
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
    }
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
