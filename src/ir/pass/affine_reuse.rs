use super::dominators::{ControlFlowGraph, Dominators};
use super::util::{const_i32, defining_inst};
use super::ModulePass;
use crate::ir::{BinaryOp, BlockId, Function, InstKind, Module, Type, ValueId, ValueKind};

/// Reuses an available affine expression when a dominated expression differs
/// by exactly one copy of the same i32 base:
///
///     (k * x + c)  =>  ((k - 1) * x + c) + x
///
/// All operations remain i32 operations, so the identity also holds when the
/// target arithmetic wraps. The previous result must dominate the rewrite.
pub(super) struct AffineReusePass;

impl AffineReusePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for AffineReusePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            reuse_affine_expressions(func);
        }
    }
}

#[derive(Clone, Copy)]
struct AffineExpression {
    result: ValueId,
    block: BlockId,
    inst_index: usize,
    base: ValueId,
    coefficient: i32,
    offset: i32,
}

fn reuse_affine_expressions(func: &mut Function) {
    if func.blocks.len() > 1024 || func.values.len() > 8192 {
        return;
    }
    let cfg = ControlFlowGraph::new(func);
    let dominators = Dominators::new(func, &cfg);
    let expressions = func
        .values
        .iter()
        .enumerate()
        .filter_map(|(index, _)| parse_affine_expression(func, ValueId(index)))
        .collect::<Vec<_>>();
    let mut rewrites = Vec::new();

    for current in &expressions {
        let Some(previous_coefficient) = current.coefficient.checked_sub(1) else {
            continue;
        };
        let previous = expressions.iter().rev().find(|previous| {
            previous.base == current.base
                && previous.coefficient == previous_coefficient
                && previous.offset == current.offset
                && definition_available_before(**previous, *current, &dominators)
        });
        if let Some(previous) = previous {
            rewrites.push((*current, previous.result));
        }
    }

    for (current, previous) in rewrites {
        func.blocks[current.block.0].insts[current.inst_index].kind = InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs: previous,
            rhs: current.base,
        };
    }

    if let Err(errors) = func.verify() {
        panic!(
            "affine reuse produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn parse_affine_expression(func: &Function, result: ValueId) -> Option<AffineExpression> {
    let ValueKind::Inst(block, inst_index) = func.value(result).kind else {
        return None;
    };
    let InstKind::Binary {
        op: BinaryOp::Iadd,
        lhs,
        rhs,
    } = defining_inst(func, result)?
    else {
        return None;
    };
    let (scaled, offset) = if let Some(offset) = const_i32(func, *rhs) {
        (*lhs, offset)
    } else {
        (*rhs, const_i32(func, *lhs)?)
    };
    let InstKind::Binary {
        op: BinaryOp::Imul,
        lhs,
        rhs,
    } = defining_inst(func, scaled)?
    else {
        return None;
    };
    let (base, coefficient) = if let Some(coefficient) = const_i32(func, *rhs) {
        (*lhs, coefficient)
    } else {
        (*rhs, const_i32(func, *lhs)?)
    };
    if coefficient <= 1 || func.value(base).ty != Type::I32 {
        return None;
    }
    Some(AffineExpression {
        result,
        block,
        inst_index,
        base,
        coefficient,
        offset,
    })
}

fn definition_available_before(
    previous: AffineExpression,
    current: AffineExpression,
    dominators: &Dominators,
) -> bool {
    if previous.block == current.block {
        previous.inst_index < current.inst_index
    } else {
        dominators.dominates(previous.block, current.block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Terminator};

    #[test]
    fn reuses_a_dominating_affine_expression() {
        let mut func = Function::new("affine", Type::I32);
        let x = func.add_param("x", Type::I32);
        let three = func.add_const(Const::Int(3));
        let four = func.add_const(Const::Int(4));
        let one = func.add_const(Const::Int(1));
        let entry = func.entry;
        let three_x = append_binary(&mut func, entry, BinaryOp::Imul, x, three);
        let previous = append_binary(&mut func, entry, BinaryOp::Iadd, three_x, one);
        let next = func.add_block("next");
        func.set_terminator(entry, Terminator::Jump(next));
        let four_x = append_binary(&mut func, next, BinaryOp::Imul, x, four);
        let current = append_binary(&mut func, next, BinaryOp::Iadd, four_x, one);
        func.set_terminator(next, Terminator::Return(Some(current)));

        let mut module = Module::new();
        module.add_func(func);
        AffineReusePass::new().run(&mut module);

        assert!(matches!(
            defining_inst(&module.funcs[0], current),
            Some(InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            }) if *lhs == previous && *rhs == x
        ));
        assert!(module.funcs[0].verify().is_ok());
    }

    #[test]
    fn rejects_an_affine_expression_from_a_sibling_branch() {
        let mut func = Function::new("siblings", Type::I32);
        let x = func.add_param("x", Type::I32);
        let cond = func.add_param("cond", Type::I1);
        let three = func.add_const(Const::Int(3));
        let four = func.add_const(Const::Int(4));
        let one = func.add_const(Const::Int(1));
        let left = func.add_block("left");
        let right = func.add_block("right");
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond,
                then_target: left,
                else_target: right,
            },
        );
        let three_x = append_binary(&mut func, left, BinaryOp::Imul, x, three);
        let previous = append_binary(&mut func, left, BinaryOp::Iadd, three_x, one);
        func.set_terminator(left, Terminator::Return(Some(previous)));
        let four_x = append_binary(&mut func, right, BinaryOp::Imul, x, four);
        let current = append_binary(&mut func, right, BinaryOp::Iadd, four_x, one);
        func.set_terminator(right, Terminator::Return(Some(current)));

        let mut module = Module::new();
        module.add_func(func);
        AffineReusePass::new().run(&mut module);

        assert!(matches!(
            defining_inst(&module.funcs[0], current),
            Some(InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            }) if *lhs == four_x && *rhs == one
        ));
    }

    fn append_binary(
        func: &mut Function,
        block: BlockId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        func.append_inst(block, InstKind::Binary { op, lhs, rhs }, Some(Type::I32))
            .unwrap()
    }
}
