use crate::ir::{BinaryOp, BlockId, Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::HashSet;

// Keep this analysis intentionally small: it only moves a side-effect-free,
// register-only entry decision ahead of the frame setup. The framed path emits
// the original entry-to-slow phi copies before bypassing the duplicated guard.
const MAX_GUARD_VALUES: usize = 12;

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum EarlyReturnResult {
    Void,
    Direct(ValueId),
    Binary {
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    Expression {
        block: BlockId,
        result: ValueId,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct ChainedEntryEarlyReturn {
    pub(crate) condition: ValueId,
    pub(crate) fast_when_true: bool,
    pub(crate) guard_block: BlockId,
}

#[derive(Clone)]
pub(crate) struct EntryEarlyReturn {
    pub(crate) guard_block: BlockId,
    pub(crate) condition: ValueId,
    pub(crate) fast_when_true: bool,
    pub(crate) fast_block: BlockId,
    pub(crate) fast_block_exclusive: bool,
    pub(crate) slow_block: BlockId,
    pub(crate) chained: Option<ChainedEntryEarlyReturn>,
    pub(crate) result: EarlyReturnResult,
    pub(crate) entry_phi_values: Vec<(ValueId, ValueId)>,
}

/// Finds one guarded, register-only return that can execute before the prologue.
pub(crate) fn entry_early_return(func: &Function) -> Option<EntryEarlyReturn> {
    if !matches!(func.ret, Type::Void | Type::I32) || has_predecessor(func, func.entry) {
        return None;
    }
    branch_entry_early_return(func).or_else(|| jump_entry_early_return(func))
}

fn branch_entry_early_return(func: &Function) -> Option<EntryEarlyReturn> {
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
            !matches!(inst.kind, InstKind::Nop | InstKind::Alloca { .. })
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
        if fast_block == slow_block {
            continue;
        }
        let Some(result) = early_result(func, fast_block) else {
            continue;
        };
        let chained = chained_early_return(func, slow_block, fast_block, &result);
        let final_slow_block = chained.map_or(slow_block, |chain| {
            let Terminator::Branch {
                then_target,
                else_target,
                ..
            } = func.blocks[chain.guard_block.0]
                .terminator
                .as_ref()
                .expect("chained guard must remain a branch")
            else {
                unreachable!();
            };
            if *then_target == fast_block {
                *else_target
            } else {
                *then_target
            }
        });
        let mut skipped_predecessors = HashSet::from([func.entry]);
        if let Some(chain) = chained {
            skipped_predecessors.insert(chain.guard_block);
        }
        return Some(EntryEarlyReturn {
            guard_block: func.entry,
            condition: *cond,
            fast_when_true,
            fast_block,
            fast_block_exclusive: has_only_predecessors(func, fast_block, &skipped_predecessors),
            slow_block: final_slow_block,
            chained,
            result,
            entry_phi_values: Vec::new(),
        });
    }
    None
}

/// Recognizes an empty entry that feeds initial values into a loop-header
/// guard with a small pure return on one edge. Tail-recursion elimination
/// commonly creates this shape. Duplicating the initial guard and result before
/// the prologue is standard shrink-wrapping; the framed path retains the
/// original header and all loop backedges.
fn jump_entry_early_return(func: &Function) -> Option<EntryEarlyReturn> {
    const MAX_FAST_EXPRESSION_INSTRUCTIONS: usize = 6;

    let entry = &func.blocks[func.entry.0];
    let Terminator::Jump(guard_block) = entry.terminator.as_ref()? else {
        return None;
    };
    if *guard_block == func.entry
        || entry
            .insts
            .iter()
            .any(|inst| !matches!(inst.kind, InstKind::Nop | InstKind::Alloca { .. }))
    {
        return None;
    }
    let guard = &func.blocks[guard_block.0];
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = guard.terminator.as_ref()?
    else {
        return None;
    };

    let mut entry_phi_values = Vec::new();
    for inst in &guard.insts {
        let (Some(phi), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
            continue;
        };
        let mut entering = incomings
            .iter()
            .filter_map(|(pred, value)| (*pred == func.entry).then_some(*value));
        let initial = entering.next()?;
        if entering.next().is_some()
            || !matches!(func.value(initial).ty, Type::I1 | Type::I32)
            || !matches!(
                func.value(initial).kind,
                ValueKind::Param | ValueKind::Const(_)
            )
        {
            return None;
        }
        entry_phi_values.push((phi, initial));
    }
    if entry_phi_values.is_empty() {
        return None;
    }

    let mut guard_values = HashSet::new();
    if !collect_guard_values_with_phi_inputs(
        func,
        *guard_block,
        *cond,
        &entry_phi_values,
        &mut guard_values,
    ) || guard_values.len() > MAX_GUARD_VALUES
        || guard.insts.iter().any(|inst| {
            !matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. })
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
        if fast_block == slow_block || !has_only_predecessor(func, fast_block, *guard_block) {
            continue;
        }
        let Some(result) = expression_result(
            func,
            fast_block,
            &entry_phi_values,
            MAX_FAST_EXPRESSION_INSTRUCTIONS,
        ) else {
            continue;
        };
        return Some(EntryEarlyReturn {
            guard_block: *guard_block,
            condition: *cond,
            fast_when_true,
            fast_block,
            // Retain the original fast block for loop iterations. Only the
            // initial entry edge is shrink-wrapped.
            fast_block_exclusive: false,
            slow_block,
            chained: None,
            result,
            entry_phi_values,
        });
    }
    None
}

fn collect_guard_values_with_phi_inputs(
    func: &Function,
    owner: BlockId,
    value: ValueId,
    phi_inputs: &[(ValueId, ValueId)],
    values: &mut HashSet<ValueId>,
) -> bool {
    if phi_inputs.iter().any(|(phi, _)| *phi == value) {
        return true;
    }
    if values.contains(&value) {
        return true;
    }
    if values.len() >= MAX_GUARD_VALUES {
        return false;
    }
    values.insert(value);
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) => {
            matches!(func.value(value).ty, Type::I1 | Type::I32)
        }
        ValueKind::Inst(block, inst_idx) if block == owner => {
            let Some(inst) = func.blocks[block.0].insts.get(inst_idx) else {
                return false;
            };
            if inst.result != Some(value) {
                return false;
            }
            match inst.kind {
                InstKind::Icmp { lhs, rhs, .. }
                | InstKind::Binary {
                    op:
                        BinaryOp::And
                        | BinaryOp::Or
                        | BinaryOp::Iand
                        | BinaryOp::Ior
                        | BinaryOp::Ixor
                        | BinaryOp::Iadd
                        | BinaryOp::Isub,
                    lhs,
                    rhs,
                } => {
                    collect_guard_values_with_phi_inputs(func, owner, lhs, phi_inputs, values)
                        && collect_guard_values_with_phi_inputs(
                            func, owner, rhs, phi_inputs, values,
                        )
                }
                InstKind::Unary {
                    op: crate::ir::UnaryOp::Not,
                    value,
                } => collect_guard_values_with_phi_inputs(func, owner, value, phi_inputs, values),
                _ => false,
            }
        }
        ValueKind::Inst(_, _) | ValueKind::Global(_) => false,
    }
}

fn expression_result(
    func: &Function,
    block: BlockId,
    phi_inputs: &[(ValueId, ValueId)],
    max_instructions: usize,
) -> Option<EarlyReturnResult> {
    let owner = &func.blocks[block.0];
    let Terminator::Return(Some(result)) = owner.terminator.as_ref()? else {
        return None;
    };
    let active = owner
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .collect::<Vec<_>>();
    if active.is_empty() && expression_operand_available(func, block, *result, phi_inputs) {
        return Some(EarlyReturnResult::Expression {
            block,
            result: *result,
        });
    }
    if active.len() > max_instructions {
        return None;
    }
    let mut available = phi_inputs
        .iter()
        .map(|(phi, _)| *phi)
        .collect::<HashSet<_>>();
    available.extend(func.params.iter().copied());
    available.extend(func.values.iter().enumerate().filter_map(|(idx, value)| {
        matches!(value.kind, ValueKind::Const(_)).then_some(ValueId(idx))
    }));
    for inst in active {
        let (Some(inst_result), InstKind::Binary { op, lhs, rhs }) = (inst.result, &inst.kind)
        else {
            return None;
        };
        if func.value(inst_result).ty != Type::I32
            || !matches!(
                *op,
                BinaryOp::Iadd
                    | BinaryOp::Isub
                    | BinaryOp::Imul
                    | BinaryOp::Idiv
                    | BinaryOp::Imod
                    | BinaryOp::Iand
                    | BinaryOp::Ior
                    | BinaryOp::Ixor
                    | BinaryOp::Ishl
                    | BinaryOp::Iashr
            )
            || !available.contains(lhs)
            || !available.contains(rhs)
        {
            return None;
        }
        available.insert(inst_result);
    }
    available
        .contains(result)
        .then_some(EarlyReturnResult::Expression {
            block,
            result: *result,
        })
}

fn expression_operand_available(
    func: &Function,
    block: BlockId,
    value: ValueId,
    phi_inputs: &[(ValueId, ValueId)],
) -> bool {
    phi_inputs.iter().any(|(phi, _)| *phi == value)
        || matches!(
            func.value(value).kind,
            ValueKind::Param | ValueKind::Const(_)
        )
        || matches!(func.value(value).kind, ValueKind::Inst(owner, _) if owner == block)
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
                    collect_guard_integer(func, entry, lhs, values)
                        && collect_guard_integer(func, entry, rhs, values)
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

fn collect_guard_integer(
    func: &Function,
    owner: BlockId,
    value: ValueId,
    values: &mut HashSet<ValueId>,
) -> bool {
    if !matches!(func.value(value).ty, Type::I1 | Type::I32) {
        return false;
    }
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) => true,
        ValueKind::Inst(block, inst_index) if block == owner => {
            if values.len() >= MAX_GUARD_VALUES {
                return false;
            }
            values.insert(value);
            let Some(inst) = func.blocks[block.0].insts.get(inst_index) else {
                return false;
            };
            matches!(
                inst.kind,
                InstKind::Binary {
                    op: BinaryOp::Iadd | BinaryOp::Isub,
                    ..
                }
            ) && match inst.kind {
                InstKind::Binary { lhs, rhs, .. } => {
                    collect_guard_integer(func, owner, lhs, values)
                        && collect_guard_integer(func, owner, rhs, values)
                }
                _ => false,
            }
        }
        ValueKind::Inst(_, _) | ValueKind::Global(_) => false,
    }
}

fn chained_early_return(
    func: &Function,
    guard_block: BlockId,
    fast_block: BlockId,
    result: &EarlyReturnResult,
) -> Option<ChainedEntryEarlyReturn> {
    if !has_only_predecessor(func, guard_block, func.entry) {
        return None;
    }
    let block = &func.blocks[guard_block.0];
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = block.terminator.as_ref()?
    else {
        return None;
    };
    let fast_when_true = if *then_target == fast_block && *else_target != fast_block {
        true
    } else if *else_target == fast_block && *then_target != fast_block {
        false
    } else {
        return None;
    };
    if &early_result(func, fast_block)? != result {
        return None;
    }
    let mut guard_values = HashSet::new();
    if !collect_guard_values(func, guard_block, *cond, &mut guard_values)
        || guard_values.len() > MAX_GUARD_VALUES
        || block.insts.iter().any(|inst| {
            !matches!(inst.kind, InstKind::Nop)
                && inst
                    .result
                    .is_none_or(|inst_result| !guard_values.contains(&inst_result))
        })
    {
        return None;
    }
    Some(ChainedEntryEarlyReturn {
        condition: *cond,
        fast_when_true,
        guard_block,
    })
}

fn early_result(func: &Function, block: BlockId) -> Option<EarlyReturnResult> {
    let owner = &func.blocks[block.0];
    let result = match owner.terminator.as_ref()? {
        Terminator::Return(None)
            if func.ret == Type::Void
                && owner
                    .insts
                    .iter()
                    .all(|inst| matches!(inst.kind, InstKind::Nop)) =>
        {
            return Some(EarlyReturnResult::Void);
        }
        Terminator::Return(Some(result)) => *result,
        Terminator::Jump(merge) => {
            let merge_block = &func.blocks[merge.0];
            let active = merge_block
                .insts
                .iter()
                .filter(|inst| !matches!(inst.kind, InstKind::Nop))
                .collect::<Vec<_>>();
            let [phi] = active.as_slice() else {
                return None;
            };
            let phi_result = phi.result?;
            if !matches!(
                merge_block.terminator,
                Some(Terminator::Return(Some(value))) if value == phi_result
            ) {
                return None;
            }
            let InstKind::Phi { incomings } = &phi.kind else {
                return None;
            };
            incomings
                .iter()
                .find_map(|(pred, value)| (*pred == block).then_some(*value))?
        }
        Terminator::Return(None) | Terminator::Branch { .. } => return None,
    };
    result_from_block(func, block, result)
}

fn result_from_block(
    func: &Function,
    block: BlockId,
    result: ValueId,
) -> Option<EarlyReturnResult> {
    let active = func.blocks[block.0]
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .collect::<Vec<_>>();
    if active.is_empty() && func.value(result).ty == func.ret && is_direct_integer(func, result) {
        return Some(EarlyReturnResult::Direct(result));
    }
    let [inst] = active.as_slice() else {
        return None;
    };
    if inst.result != Some(result) || func.value(result).ty != Type::I32 {
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
    has_only_predecessors(func, target, &HashSet::from([expected]))
}

fn has_only_predecessors(func: &Function, target: BlockId, expected: &HashSet<BlockId>) -> bool {
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
    &predecessors == expected
}
