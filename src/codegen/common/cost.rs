use super::natural_loop_depths;
use crate::ir::{Function, InstKind, Terminator, ValueId};

/// Static execution-frequency proxy used by target register allocators.
///
/// A use outside a loop has weight 1. Each natural-loop level multiplies the
/// weight by 16, capped at 2^20 so deeply nested or malformed CFGs cannot
/// overflow the score. Phi inputs are charged to their predecessor edge.
pub(crate) fn weighted_use_scores(func: &Function) -> Vec<usize> {
    let loop_depths = natural_loop_depths(func);
    let weight_for = |block_idx: usize| 1usize << loop_depths[block_idx].saturating_mul(4).min(20);
    let mut scores = vec![0usize; func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let weight = weight_for(block_idx);
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings } => {
                    for (pred, value) in incomings {
                        scores[value.0] = scores[value.0].saturating_add(weight_for(pred.0));
                    }
                }
                kind => {
                    for value in inst_operands(kind) {
                        scores[value.0] = scores[value.0].saturating_add(weight);
                    }
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            for value in terminator_operands(terminator) {
                scores[value.0] = scores[value.0].saturating_add(weight);
            }
        }
    }
    scores
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. } => Vec::new(),
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

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Const, Function, InstKind, Terminator, Type};

    #[test]
    fn loop_uses_outweigh_cold_uses() {
        let mut func = Function::new("weighted", Type::I32);
        let cold = func.add_param("cold", Type::I32);
        let hot = func.add_param("hot", Type::I32);
        let header = func.add_block("header");
        let body = func.add_block("body");
        let exit = func.add_block("exit");
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        func.set_terminator(func.entry, Terminator::Jump(header));
        let index = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let cond = func
            .append_inst(
                header,
                InstKind::Icmp {
                    op: crate::ir::CmpOp::Lt,
                    lhs: index,
                    rhs: one,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            header,
            Terminator::Branch {
                cond,
                then_target: body,
                else_target: exit,
            },
        );
        func.append_inst(
            body,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: hot,
                rhs: one,
            },
            Some(Type::I32),
        );
        func.set_terminator(body, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(Some(cold)));

        let scores = weighted_use_scores(&func);

        assert!(scores[hot.0] > scores[cold.0]);
    }
}
