use super::util::{const_i32, defining_inst, rewrite_block_uses_from, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::HashSet;

/// Removes redundant suffixes of a nested integer maximum chain on a guarded
/// non-negative path. The original computation remains as the negative-input
/// fallback, preserving signed-overflow and comparison behavior there.
pub(super) struct GuardedMaxChainPass;

impl GuardedMaxChainPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for GuardedMaxChainPass {
    fn run(&mut self, module: &mut Module) {
        let maximum_functions = module
            .funcs
            .iter()
            .filter(|func| is_integer_maximum(func))
            .map(|func| func.name.clone())
            .collect::<HashSet<_>>();
        if maximum_functions.is_empty() {
            return;
        }
        for func in &mut module.funcs {
            let Some(candidate) = find_candidate(func, &maximum_functions) else {
                continue;
            };
            apply_candidate(func, candidate);
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    parameter: ValueId,
    first_call_index: usize,
    third_call_index: usize,
    first_result: ValueId,
    third_result: ValueId,
}

fn find_candidate(func: &Function, maximum_functions: &HashSet<String>) -> Option<Candidate> {
    if func.ret != Type::I32
        || func.params.len() != 1
        || func.value(func.params[0]).ty != Type::I32
        || func.blocks.len() != 1
        || !matches!(
            func.block(func.entry).terminator,
            Some(Terminator::Return(Some(_)))
        )
    {
        return None;
    }

    let block = func.block(func.entry);
    let calls = block
        .insts
        .iter()
        .enumerate()
        .filter_map(|(index, inst)| {
            let InstKind::Call { name, args } = &inst.kind else {
                return None;
            };
            maximum_functions
                .contains(name)
                .then_some((index, inst.result?, args.as_slice()))
        })
        .collect::<Vec<_>>();
    if block
        .insts
        .iter()
        .filter(|inst| matches!(inst.kind, InstKind::Call { .. }))
        .count()
        != calls.len()
    {
        return None;
    }
    let [(first_index, first_result, first_args), (_, second_result, second_args), (third_index, third_result, third_args)] =
        calls.as_slice()
    else {
        return None;
    };

    let parameter = func.params[0];
    let first_constant = complementary_max_constant(func, first_args, parameter)?;
    let second_constant = complementary_max_constant(func, second_args, *first_result)?;
    let third_constant = complementary_max_constant(func, third_args, *second_result)?;
    if first_constant < 0 || second_constant < 0 || third_constant < 0 {
        return None;
    }

    // For x >= 0, max(x, C-x) is at least ceil(C/2). If twice that
    // lower bound dominates each following complementary operand, every
    // suffix max must return the first result. i64 keeps the proof exact.
    let lower_bound = (i64::from(first_constant) + 1) / 2;
    if 2 * lower_bound < i64::from(second_constant) || 2 * lower_bound < i64::from(third_constant) {
        return None;
    }

    let skipped = &block.insts[first_index + 1..=*third_index];
    if skipped.iter().any(|inst| {
        !matches!(
            inst.kind,
            InstKind::Nop
                | InstKind::Unary { .. }
                | InstKind::Binary { .. }
                | InstKind::Icmp { .. }
                | InstKind::Fcmp { .. }
                | InstKind::Cast { .. }
                | InstKind::Call { .. }
        )
    }) {
        return None;
    }
    let unavailable_on_fast_path = skipped
        .iter()
        .filter_map(|inst| inst.result)
        .filter(|result| *result != *third_result)
        .collect::<HashSet<_>>();
    if block.insts[third_index + 1..]
        .iter()
        .flat_map(|inst| inst_operands(&inst.kind))
        .chain(terminator_operands(block.terminator.as_ref()))
        .any(|operand| unavailable_on_fast_path.contains(&operand))
    {
        return None;
    }

    Some(Candidate {
        parameter,
        first_call_index: *first_index,
        third_call_index: *third_index,
        first_result: *first_result,
        third_result: *third_result,
    })
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
        InstKind::Gep { base, indices } => std::iter::once(*base)
            .chain(indices.iter().copied())
            .collect(),
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

fn complementary_max_constant(func: &Function, args: &[ValueId], previous: ValueId) -> Option<i32> {
    let [lhs, rhs] = args else {
        return None;
    };
    if *lhs == previous {
        subtraction_constant(func, *rhs, previous)
    } else if *rhs == previous {
        subtraction_constant(func, *lhs, previous)
    } else {
        None
    }
}

fn subtraction_constant(func: &Function, value: ValueId, subtrahend: ValueId) -> Option<i32> {
    let InstKind::Binary {
        op: crate::ir::BinaryOp::Isub,
        lhs,
        rhs,
    } = defining_inst(func, value)?
    else {
        return None;
    };
    (*rhs == subtrahend)
        .then(|| const_i32(func, *lhs))
        .flatten()
}

fn apply_candidate(func: &mut Function, candidate: Candidate) {
    let entry = func.entry;
    let original_terminator = func.block_mut(entry).terminator.take().unwrap();
    let mut original_insts = std::mem::take(&mut func.block_mut(entry).insts);
    let suffix = original_insts.split_off(candidate.third_call_index + 1);
    let slow = original_insts.split_off(candidate.first_call_index + 1);
    func.block_mut(entry).insts = original_insts;
    relink_results(func, entry);

    let slow_block = func.add_block("max_chain.fallback");
    let merge_block = func.add_block("max_chain.merge");
    func.block_mut(slow_block).insts = slow;
    relink_results(func, slow_block);
    func.set_terminator(slow_block, Terminator::Jump(merge_block));

    let zero = func.add_const(crate::ir::Const::Int(0));
    let nonnegative = func
        .append_inst(
            entry,
            InstKind::Icmp {
                op: CmpOp::Ge,
                lhs: candidate.parameter,
                rhs: zero,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        entry,
        Terminator::Branch {
            cond: nonnegative,
            then_target: merge_block,
            else_target: slow_block,
        },
    );

    let selected = func
        .append_inst(
            merge_block,
            InstKind::Phi {
                incomings: vec![
                    (entry, candidate.first_result),
                    (slow_block, candidate.third_result),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.block_mut(merge_block).insts.extend(suffix);
    relink_results(func, merge_block);
    func.set_terminator(merge_block, original_terminator);
    rewrite_block_uses_from(
        func,
        merge_block,
        1,
        &ValueReplacements::from([(candidate.third_result, selected)]),
    );
    repair_definition_locations(func);

    if let Err(errors) = func.verify() {
        panic!(
            "guarded maximum-chain simplification produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn repair_definition_locations(func: &mut Function) {
    let mut locations = vec![None; func.values.len()];
    for (block_index, block) in func.blocks.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            if let Some(result) = inst.result {
                locations[result.0] = Some((BlockId(block_index), inst_index));
            }
        }
    }
    for (value_index, value) in func.values.iter_mut().enumerate() {
        if !matches!(value.kind, ValueKind::Inst(..)) {
            continue;
        }
        value.kind = match locations[value_index] {
            Some((block, inst_index)) => ValueKind::Inst(block, inst_index),
            None => ValueKind::Const(Const::Zero(value.ty.clone())),
        };
    }
}

fn relink_results(func: &mut Function, block: BlockId) {
    let results = func
        .block(block)
        .insts
        .iter()
        .enumerate()
        .filter_map(|(index, inst)| inst.result.map(|result| (result, index)))
        .collect::<Vec<_>>();
    for (result, index) in results {
        func.values[result.0].kind = ValueKind::Inst(block, index);
    }
}

fn is_integer_maximum(func: &Function) -> bool {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func
            .params
            .iter()
            .any(|param| func.value(*param).ty != Type::I32)
        || func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .any(|inst| !matches!(inst.kind, InstKind::Nop | InstKind::Icmp { .. }))
    {
        return false;
    }
    let lhs_param = func.params[0];
    let rhs_param = func.params[1];
    let Some((op, compared_lhs, compared_rhs, then_target, else_target)) =
        comparison_branch(func, func.entry)
    else {
        return false;
    };
    let Some(then_value) = returned_value(func, then_target) else {
        return false;
    };
    let Some(else_value) = returned_value(func, else_target) else {
        return false;
    };

    match (op, compared_lhs, compared_rhs) {
        (CmpOp::Lt | CmpOp::Le, lhs, rhs) if lhs == lhs_param && rhs == rhs_param => {
            then_value == rhs_param && else_value == lhs_param
        }
        (CmpOp::Gt | CmpOp::Ge, lhs, rhs) if lhs == lhs_param && rhs == rhs_param => {
            then_value == lhs_param && else_value == rhs_param
        }
        (CmpOp::Lt | CmpOp::Le, lhs, rhs) if lhs == rhs_param && rhs == lhs_param => {
            then_value == lhs_param && else_value == rhs_param
        }
        (CmpOp::Gt | CmpOp::Ge, lhs, rhs) if lhs == rhs_param && rhs == lhs_param => {
            then_value == rhs_param && else_value == lhs_param
        }
        _ => false,
    }
}

fn comparison_branch(
    func: &Function,
    block: BlockId,
) -> Option<(CmpOp, ValueId, ValueId, BlockId, BlockId)> {
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
    Some((*op, *lhs, *rhs, *then_target, *else_target))
}

fn returned_value(func: &Function, mut block: BlockId) -> Option<ValueId> {
    let mut seen = HashSet::new();
    while seen.insert(block) {
        let current = func.block(block);
        let has_effective_inst = current
            .insts
            .iter()
            .any(|inst| !matches!(inst.kind, InstKind::Nop));
        match current.terminator.as_ref()? {
            Terminator::Return(Some(value)) => return Some(*value),
            Terminator::Jump(target) if !has_effective_inst => block = *target,
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::{DcePass, ModulePass, ScalarPromotePass};
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    fn transformed(source: &str) -> Module {
        let program = Parser::new(source).parse_program();
        let mut module = lower_program(&program).unwrap();
        ScalarPromotePass::new().run(&mut module);
        DcePass::new().run(&mut module);
        GuardedMaxChainPass::new().run(&mut module);
        module
    }

    #[test]
    fn versions_a_proven_nested_maximum_chain() {
        let module = transformed(
            r#"
            int choose(int a, int b) { if (a < b) return b; return a; }
            int calculate(int x) {
                int a = choose(x, 101 - x);
                int b = choose(a, 50 - a);
                int c = choose(b, 25 - b);
                return c * 3;
            }
            "#,
        );
        let calculate = module
            .funcs
            .iter()
            .find(|func| func.name == "calculate")
            .unwrap();
        assert!(calculate
            .blocks
            .iter()
            .any(|block| block.name == "max_chain.fallback"));
        assert!(calculate
            .blocks
            .iter()
            .any(|block| block.name == "max_chain.merge"));
        assert!(calculate.verify().is_ok());
    }

    #[test]
    fn rejects_a_chain_without_a_sufficient_first_bound() {
        let module = transformed(
            r#"
            int choose(int a, int b) { if (a < b) return b; return a; }
            int calculate(int x) {
                int a = choose(x, 10 - x);
                int b = choose(a, 100 - a);
                int c = choose(b, 25 - b);
                return c;
            }
            "#,
        );
        let calculate = module
            .funcs
            .iter()
            .find(|func| func.name == "calculate")
            .unwrap();
        assert_eq!(calculate.blocks.len(), 1);
    }

    #[test]
    fn rejects_side_effects_between_maximum_calls() {
        let module = transformed(
            r#"
            int choose(int a, int b) { if (a < b) return b; return a; }
            int calculate(int x) {
                int a = choose(x, 101 - x);
                putint(a);
                int b = choose(a, 50 - a);
                int c = choose(b, 25 - b);
                return c;
            }
            "#,
        );
        let calculate = module
            .funcs
            .iter()
            .find(|func| func.name == "calculate")
            .unwrap();
        assert_eq!(calculate.blocks.len(), 1);
    }

    #[test]
    fn rejects_a_maximum_function_with_side_effects() {
        let module = transformed(
            r#"
            int choose(int a, int b) {
                putint(a);
                if (a < b) return b;
                return a;
            }
            int calculate(int x) {
                int a = choose(x, 101 - x);
                int b = choose(a, 50 - a);
                int c = choose(b, 25 - b);
                return c;
            }
            "#,
        );
        let calculate = module
            .funcs
            .iter()
            .find(|func| func.name == "calculate")
            .unwrap();
        assert_eq!(calculate.blocks.len(), 1);
    }

    #[test]
    fn rejects_uses_of_a_skipped_intermediate_result() {
        let module = transformed(
            r#"
            int choose(int a, int b) { if (a < b) return b; return a; }
            int calculate(int x) {
                int a = choose(x, 101 - x);
                int b = choose(a, 50 - a);
                int c = choose(b, 25 - b);
                return c + b;
            }
            "#,
        );
        let calculate = module
            .funcs
            .iter()
            .find(|func| func.name == "calculate")
            .unwrap();
        assert_eq!(calculate.blocks.len(), 1);
    }
}
