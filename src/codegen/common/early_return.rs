use crate::ir::{BinaryOp, BlockId, Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::HashSet;

// Keep this analysis intentionally small: it only moves a side-effect-free,
// register-only entry decision ahead of the frame setup. The slow successor
// must need no edge copies, and all omitted blocks must be unreachable from
// anywhere except the original entry edge.
const MAX_GUARD_VALUES: usize = 12;

#[derive(Clone, Copy)]
pub(crate) enum EarlyReturnResult {
    Direct(ValueId),
    Binary {
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct EntryEarlyReturn {
    pub(crate) condition: ValueId,
    pub(crate) fast_when_true: bool,
    pub(crate) fast_block: BlockId,
    pub(crate) slow_block: BlockId,
    pub(crate) result: EarlyReturnResult,
}

/// Finds one guarded, register-only return that can execute before the prologue.
pub(crate) fn entry_early_return(func: &Function) -> Option<EntryEarlyReturn> {
    if func.ret != Type::I32 || has_predecessor(func, func.entry) {
        return None;
    }
    let entry = &func.blocks[func.entry.0];
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = entry.terminator.as_ref()?
    else {
        return None;
    };

    let mut guard_values = HashSet::new();
    if !collect_guard_values(func, func.entry, *cond, &mut guard_values)
        || guard_values.len() > MAX_GUARD_VALUES
        || entry.insts.iter().any(|inst| {
            !matches!(inst.kind, InstKind::Nop)
                && inst
                    .result
                    .is_none_or(|result| !guard_values.contains(&result))
        })
    {
        return None;
    }

    for (fast_when_true, fast_block, slow_block) in [
        (true, *then_target, *else_target),
        (false, *else_target, *then_target),
    ] {
        if fast_block == slow_block
            || !has_only_predecessor(func, fast_block, func.entry)
            || block_has_phi(func, slow_block)
        {
            continue;
        }
        let Some(result) = early_result(func, fast_block) else {
            continue;
        };
        if entry_results_escape(func, func.entry, fast_block) {
            continue;
        }
        return Some(EntryEarlyReturn {
            condition: *cond,
            fast_when_true,
            fast_block,
            slow_block,
            result,
        });
    }
    None
}

fn collect_guard_values(
    func: &Function,
    entry: BlockId,
    value: ValueId,
    values: &mut HashSet<ValueId>,
) -> bool {
    if values.contains(&value) {
        return true;
    }
    if values.len() >= MAX_GUARD_VALUES {
        return false;
    }
    values.insert(value);
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) => func.value(value).ty == Type::I1,
        ValueKind::Inst(owner, inst_idx) if owner == entry => {
            let Some(inst) = func.blocks[owner.0].insts.get(inst_idx) else {
                return false;
            };
            if inst.result != Some(value) || func.value(value).ty != Type::I1 {
                return false;
            }
            match inst.kind {
                InstKind::Icmp { lhs, rhs, .. } => {
                    is_direct_integer(func, lhs) && is_direct_integer(func, rhs)
                }
                InstKind::Unary {
                    op: crate::ir::UnaryOp::Not,
                    value,
                } => collect_guard_values(func, entry, value, values),
                InstKind::Binary {
                    op: BinaryOp::And | BinaryOp::Or,
                    lhs,
                    rhs,
                } => {
                    collect_guard_values(func, entry, lhs, values)
                        && collect_guard_values(func, entry, rhs, values)
                }
                _ => false,
            }
        }
        ValueKind::Inst(_, _) | ValueKind::Global(_) => false,
    }
}

fn early_result(func: &Function, block: BlockId) -> Option<EarlyReturnResult> {
    let owner = &func.blocks[block.0];
    let Terminator::Return(Some(result)) = owner.terminator.as_ref()? else {
        return None;
    };
    let active = owner
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .collect::<Vec<_>>();

    if active.is_empty() && func.value(*result).ty == func.ret && is_direct_integer(func, *result) {
        return Some(EarlyReturnResult::Direct(*result));
    }
    let [inst] = active.as_slice() else {
        return None;
    };
    if inst.result != Some(*result) || func.value(*result).ty != Type::I32 {
        return None;
    }
    let InstKind::Binary { op, lhs, rhs } = inst.kind else {
        return None;
    };
    if !matches!(
        op,
        BinaryOp::Iadd
            | BinaryOp::Isub
            | BinaryOp::Imul
            | BinaryOp::Iand
            | BinaryOp::Ior
            | BinaryOp::Ixor
            | BinaryOp::Ishl
            | BinaryOp::Iashr
    ) || !is_direct_integer(func, lhs)
        || !is_direct_integer(func, rhs)
    {
        return None;
    }
    Some(EarlyReturnResult::Binary { op, lhs, rhs })
}

fn is_direct_integer(func: &Function, value: ValueId) -> bool {
    matches!(func.value(value).ty, Type::I1 | Type::I32)
        && matches!(
            func.value(value).kind,
            ValueKind::Param | ValueKind::Const(_)
        )
}

fn block_has_phi(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0]
        .insts
        .iter()
        .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
}

fn has_predecessor(func: &Function, target: BlockId) -> bool {
    func.blocks
        .iter()
        .any(|block| match block.terminator.as_ref() {
            Some(Terminator::Jump(successor)) => *successor == target,
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => *then_target == target || *else_target == target,
            Some(Terminator::Return(_)) | None => false,
        })
}

fn has_only_predecessor(func: &Function, target: BlockId, expected: BlockId) -> bool {
    let mut predecessors = HashSet::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let reaches_target = match block.terminator.as_ref() {
            Some(Terminator::Jump(successor)) => *successor == target,
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => *then_target == target || *else_target == target,
            Some(Terminator::Return(_)) | None => false,
        };
        if reaches_target {
            predecessors.insert(BlockId(block_idx));
        }
    }
    predecessors == HashSet::from([expected])
}

fn entry_results_escape(func: &Function, entry: BlockId, fast_block: BlockId) -> bool {
    let entry_results = func.blocks[entry.0]
        .insts
        .iter()
        .filter_map(|inst| inst.result)
        .collect::<HashSet<_>>();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        if block_idx == entry.0 || block_idx == fast_block.0 {
            continue;
        }
        if block
            .insts
            .iter()
            .flat_map(|inst| inst_operands(&inst.kind))
            .chain(terminator_operands(block.terminator.as_ref()))
            .any(|operand| entry_results.contains(&operand))
        {
            return true;
        }
    }
    false
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
            let mut operands = Vec::with_capacity(indices.len() + 1);
            operands.push(*base);
            operands.extend(indices.iter().copied());
            operands
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
