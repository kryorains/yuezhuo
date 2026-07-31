use crate::ir::{BlockId, Const, Function, Inst, InstKind, Terminator, ValueId, ValueKind};
use std::collections::HashMap;

pub(super) type ValueReplacements = HashMap<ValueId, ValueId>;

pub(super) fn rewrite_function_uses(func: &mut Function, replacements: &ValueReplacements) -> bool {
    // 统一改写整个函数的操作数；pass 只需要准备 replacement 表即可。
    if replacements.is_empty() {
        return false;
    }

    let mut changed = false;
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            changed |= rewrite_inst_uses(inst, replacements);
        }
        if let Some(terminator) = &mut block.terminator {
            changed |= rewrite_terminator_uses(terminator, replacements);
        }
    }
    changed
}

pub(super) fn rewrite_block_uses_from(
    func: &mut Function,
    block: BlockId,
    inst_start: usize,
    replacements: &ValueReplacements,
) -> bool {
    if replacements.is_empty() {
        return false;
    }
    let Some(block) = func.blocks.get_mut(block.0) else {
        return false;
    };
    let mut changed = false;
    for inst in block.insts.iter_mut().skip(inst_start) {
        changed |= rewrite_inst_uses(inst, replacements);
    }
    if let Some(terminator) = &mut block.terminator {
        changed |= rewrite_terminator_uses(terminator, replacements);
    }
    changed
}

pub(super) fn resolve_replacement(mut value: ValueId, replacements: &ValueReplacements) -> ValueId {
    // 处理链式替换，例如 a -> b、b -> c，最终应该把 a 解析成 c。
    while let Some(next) = replacements.get(&value).copied() {
        if next == value {
            break;
        }
        value = next;
    }
    value
}

/// Moves the original entry body to an appended fallback block, leaving the
/// physical entry block empty so an optimization can install a guarded fast
/// path without changing the backend's block-zero entry convention.
pub(super) fn move_entry_to_fallback(
    func: &mut Function,
    fallback_name: impl Into<String>,
) -> Option<BlockId> {
    let entry = func.entry;
    if entry.0 != 0 || func.blocks.get(entry.0)?.terminator.is_none() {
        return None;
    }
    if func.blocks.iter().any(|block| {
        terminator_targets(block.terminator.as_ref())
            .into_iter()
            .any(|target| target == entry)
    }) {
        return None;
    }

    let fallback = func.add_block(fallback_name);
    let entry_insts = std::mem::take(&mut func.blocks[entry.0].insts);
    let entry_terminator = func.blocks[entry.0].terminator.take();
    func.blocks[fallback.0].insts = entry_insts;
    func.blocks[fallback.0].terminator = entry_terminator;

    for value in &mut func.values {
        if let ValueKind::Inst(owner, _) = &mut value.kind {
            if *owner == entry {
                *owner = fallback;
            }
        }
    }
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            let InstKind::Phi { incomings } = &mut inst.kind else {
                continue;
            };
            for (pred, _) in incomings {
                if *pred == entry {
                    *pred = fallback;
                }
            }
        }
    }
    Some(fallback)
}

pub(super) fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.values.get(value.0)?.kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

pub(super) fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Const(Const::Int(value))) => Some(*value),
        _ => None,
    }
}

pub(super) fn get_or_add_i32_const(func: &mut Function, expected: i32) -> ValueId {
    func.values
        .iter()
        .position(
            |value| matches!(value.kind, ValueKind::Const(Const::Int(value)) if value == expected),
        )
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(Const::Int(expected)))
}

fn terminator_targets(terminator: Option<&Terminator>) -> Vec<BlockId> {
    match terminator {
        Some(Terminator::Jump(target)) => vec![*target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) if then_target == else_target => vec![*then_target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) => vec![*then_target, *else_target],
        Some(Terminator::Return(_)) | None => Vec::new(),
    }
}

fn rewrite_value(value: &mut ValueId, replacements: &ValueReplacements) -> bool {
    let replacement = resolve_replacement(*value, replacements);
    if replacement == *value {
        return false;
    }
    *value = replacement;
    true
}

fn rewrite_inst_uses(inst: &mut Inst, replacements: &ValueReplacements) -> bool {
    // 只改写“使用到的值”，不会改写指令自己的 result。
    match &mut inst.kind {
        InstKind::Nop | InstKind::Alloca { .. } => false,
        InstKind::Phi { incomings } => incomings.iter_mut().fold(false, |changed, (_, value)| {
            rewrite_value(value, replacements) || changed
        }),
        InstKind::Load { ptr } => rewrite_value(ptr, replacements),
        InstKind::Store { ptr, value } => {
            rewrite_value(ptr, replacements) | rewrite_value(value, replacements)
        }
        InstKind::MemZero { ptr, .. } => rewrite_value(ptr, replacements),
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => {
            rewrite_value(value, replacements)
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            rewrite_value(lhs, replacements) | rewrite_value(rhs, replacements)
        }
        InstKind::Gep { base, indices } => {
            let mut changed = rewrite_value(base, replacements);
            for index in indices {
                changed |= rewrite_value(index, replacements);
            }
            changed
        }
        InstKind::Call { args, .. } => {
            let mut changed = false;
            for arg in args {
                changed |= rewrite_value(arg, replacements);
            }
            changed
        }
    }
}

fn rewrite_terminator_uses(terminator: &mut Terminator, replacements: &ValueReplacements) -> bool {
    match terminator {
        Terminator::Return(value) => value
            .as_mut()
            .is_some_and(|value| rewrite_value(value, replacements)),
        Terminator::Jump(_) => false,
        Terminator::Branch { cond, .. } => rewrite_value(cond, replacements),
    }
}
