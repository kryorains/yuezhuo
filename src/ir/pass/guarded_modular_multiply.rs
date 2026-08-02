use super::util::{const_i32, defining_inst};
use super::ModulePass;
use crate::ir::{BinaryOp, BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId};
use std::collections::{HashSet, VecDeque};

/// Recognizes the exact recursive double-and-add implementation of modular
/// multiplication. Code generation may replace calls with a native widened
/// multiply only after runtime range guards prove that the two are equivalent.
pub(super) struct GuardedModularMultiplyPass;

impl GuardedModularMultiplyPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for GuardedModularMultiplyPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            if let Some(modulus) = match_recursive_modular_multiply(func) {
                func.mark_guarded_mulmod(modulus);
            }
        }
    }
}

fn match_recursive_modular_multiply(func: &Function) -> Option<i32> {
    if func.ret != Type::I32 || func.params.len() != 2 {
        return None;
    }
    let lhs = func.params[0];
    let rhs = func.params[1];
    if func.value(lhs).ty != Type::I32 || func.value(rhs).ty != Type::I32 {
        return None;
    }

    let (zero_block, nonzero_block) = equality_branch(func, func.entry, rhs, 0)?;
    if const_i32(func, returned_value(func, zero_block)?)? != 0 {
        return None;
    }
    let (one_block, recursive_block) = equality_branch(func, nonzero_block, rhs, 1)?;

    let one_result = returned_value(func, one_block)?;
    let (one_lhs, modulus_value) = binary_operands(func, one_result, BinaryOp::Imod)?;
    if one_lhs != lhs {
        return None;
    }
    let modulus = const_i32(func, modulus_value)?;
    // Recursive additions remain within signed i32 for canonical operands.
    if !(2..=1_073_741_824).contains(&modulus) {
        return None;
    }

    let half = find_binary(func, recursive_block, BinaryOp::Idiv, |a, b| {
        a == rhs && const_i32(func, b) == Some(2)
    })?;
    let recursive = find_call(func, recursive_block, &func.name, &[lhs, half])?;
    let doubled = find_binary(func, recursive_block, BinaryOp::Iadd, |a, b| {
        a == recursive && b == recursive
    })
    .or_else(|| {
        find_binary(func, recursive_block, BinaryOp::Imul, |a, b| {
            (a == recursive && const_i32(func, b) == Some(2))
                || (b == recursive && const_i32(func, a) == Some(2))
        })
    })?;
    let reduced = find_binary(func, recursive_block, BinaryOp::Imod, |a, b| {
        a == doubled && b == modulus_value
    })?;
    let parity = find_binary(func, recursive_block, BinaryOp::Imod, |a, b| {
        a == rhs && const_i32(func, b) == Some(2)
    })?;
    let (odd_block, even_block) = equality_branch(func, recursive_block, parity, 1)?;
    if returned_value(func, even_block)? != reduced {
        return None;
    }
    let odd_result = returned_value(func, odd_block)?;
    let (odd_sum, odd_modulus) = binary_operands(func, odd_result, BinaryOp::Imod)?;
    if odd_modulus != modulus_value {
        return None;
    }
    let (sum_lhs, sum_rhs) = binary_operands(func, odd_sum, BinaryOp::Iadd)?;
    if !((sum_lhs == reduced && sum_rhs == lhs) || (sum_lhs == lhs && sum_rhs == reduced)) {
        return None;
    }

    let mut recursive_calls = 0usize;
    for block in reachable_blocks(func) {
        for inst in &func.block(block).insts {
            match &inst.kind {
                InstKind::Nop
                | InstKind::Binary { .. }
                | InstKind::Icmp { .. }
                | InstKind::Unary { .. }
                | InstKind::Cast { .. } => {}
                InstKind::Call { name, .. } if name == &func.name => recursive_calls += 1,
                _ => return None,
            }
        }
    }
    (recursive_calls == 1).then_some(modulus)
}

fn equality_branch(
    func: &Function,
    block: BlockId,
    subject: ValueId,
    expected: i32,
) -> Option<(BlockId, BlockId)> {
    let block = follow_empty_jumps(func, block)?;
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.block(block).terminator.as_ref()?
    else {
        return None;
    };
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
        return None;
    };
    let matches = (*lhs == subject && const_i32(func, *rhs) == Some(expected))
        || (*rhs == subject && const_i32(func, *lhs) == Some(expected));
    if !matches {
        return None;
    }
    match op {
        CmpOp::Eq => Some((*then_target, *else_target)),
        CmpOp::Ne => Some((*else_target, *then_target)),
        _ => None,
    }
}

fn returned_value(func: &Function, block: BlockId) -> Option<ValueId> {
    let block = follow_empty_jumps(func, block)?;
    match func.block(block).terminator.as_ref()? {
        Terminator::Return(Some(value)) => Some(*value),
        _ => None,
    }
}

fn follow_empty_jumps(func: &Function, mut block: BlockId) -> Option<BlockId> {
    let mut seen = HashSet::new();
    while seen.insert(block) {
        let current = func.blocks.get(block.0)?;
        let has_real_inst = current
            .insts
            .iter()
            .any(|inst| !matches!(inst.kind, InstKind::Nop));
        match current.terminator.as_ref()? {
            Terminator::Jump(next) if !has_real_inst => block = *next,
            _ => return Some(block),
        }
    }
    None
}

fn binary_operands(
    func: &Function,
    value: ValueId,
    expected: BinaryOp,
) -> Option<(ValueId, ValueId)> {
    let InstKind::Binary { op, lhs, rhs } = defining_inst(func, value)? else {
        return None;
    };
    (*op == expected).then_some((*lhs, *rhs))
}

fn find_binary(
    func: &Function,
    block: BlockId,
    op: BinaryOp,
    predicate: impl Fn(ValueId, ValueId) -> bool,
) -> Option<ValueId> {
    func.block(block).insts.iter().find_map(|inst| {
        let InstKind::Binary {
            op: actual,
            lhs,
            rhs,
        } = inst.kind
        else {
            return None;
        };
        (actual == op && predicate(lhs, rhs)).then_some(inst.result?)
    })
}

fn find_call(
    func: &Function,
    block: BlockId,
    name: &str,
    expected_args: &[ValueId],
) -> Option<ValueId> {
    func.block(block).insts.iter().find_map(|inst| {
        let InstKind::Call { name: callee, args } = &inst.kind else {
            return None;
        };
        (callee == name && args == expected_args).then_some(inst.result?)
    })
}

fn reachable_blocks(func: &Function) -> Vec<BlockId> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([func.entry]);
    while let Some(block) = queue.pop_front() {
        if !seen.insert(block) {
            continue;
        }
        match func.block(block).terminator.as_ref() {
            Some(Terminator::Jump(target)) => queue.push_back(*target),
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => {
                queue.push_back(*then_target);
                queue.push_back(*else_target);
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }
    seen.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, OptLevel, PassOptions};
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    fn analyze(source: &str) -> Module {
        let program = Parser::new(source).parse_program();
        let mut module = lower_program(&program).unwrap();
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_simple_loop_unroll_in_main: false,
                small_expr_inline_rounds: 0,
                cfg_inline_rounds: 0,
                cfg_inline_global_loads: false,
                cfg_inline_global_stores: false,
                recursive_inline_rounds: 1,
                enable_constant_address_count_reduction: false,
                enable_recursive_const_specialization: false,
                enable_initialized_global_propagation: false,
                enable_uniform_constant_arguments: false,
                enable_loop_call_memoize: false,
                enable_loop_invariant_call_memoize: false,
                enable_regional_global_scalar_promotion: false,
                enable_full_domain_bitwise_digit: false,
                enable_write_only_alloca_cleanup_before_inline: false,
            },
        );
        module
    }

    #[test]
    fn marks_only_the_exact_widenable_modular_recurrence() {
        let module = analyze(
            r#"
            const int modulus = 998244353;
            int arithmetic(int a, int b) {
                if (b == 0) return 0;
                if (b == 1) return a % modulus;
                int cur = arithmetic(a, b / 2);
                cur = (cur + cur) % modulus;
                if (b % 2 == 1) return (cur + a) % modulus;
                return cur;
            }
            "#,
        );
        assert_eq!(module.funcs[0].guarded_mulmod_modulus(), Some(998244353));

        let rejected = analyze(
            r#"
            const int modulus = 998244353;
            int arithmetic(int a, int b) {
                if (b == 0) return 0;
                if (b == 1) return a % modulus;
                int cur = arithmetic(a, b / 2);
                cur = (cur + cur) % modulus;
                if (b % 2 == 1) return (cur - a) % modulus;
                return cur;
            }
            "#,
        );
        assert_eq!(rejected.funcs[0].guarded_mulmod_modulus(), None);
    }
}
