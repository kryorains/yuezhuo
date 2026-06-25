use crate::ir::{Function, Inst, InstKind, Terminator, ValueId};
use std::collections::HashMap;

pub(super) type ValueReplacements = HashMap<ValueId, ValueId>;

pub(super) fn rewrite_function_uses(func: &mut Function, replacements: &ValueReplacements) -> bool {
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

pub(super) fn resolve_replacement(mut value: ValueId, replacements: &ValueReplacements) -> ValueId {
    while let Some(next) = replacements.get(&value).copied() {
        if next == value {
            break;
        }
        value = next;
    }
    value
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
