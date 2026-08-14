use crate::ir::{Function, InstKind, Terminator, ValueId};

pub(crate) fn ir_value_use_counts(func: &Function) -> Vec<usize> {
    let mut counts = vec![0usize; func.values.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            for operand in inst_operands(&inst.kind) {
                counts[operand.0] += 1;
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_operands(terminator) {
                counts[operand.0] += 1;
            }
        }
    }
    counts
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
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

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}
