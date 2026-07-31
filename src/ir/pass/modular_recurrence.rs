use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::util::{rewrite_block_uses_from, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::HashSet;

/// Collapses a guarded affine modular recurrence:
///
/// `state = (state + positive_delta) % positive_modulus`
///
/// repeated by a canonical non-negative loop. The fast path is selected only
/// when the initial state is already in the modulus range and the combined
/// multiply-add cannot overflow i32. All other inputs retain the source loop.
pub(super) struct ModularRecurrencePass;

impl ModularRecurrencePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for ModularRecurrencePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            let Some(plan) = find_plan(func) else {
                continue;
            };
            apply_plan(func, plan);
            if let Err(errors) = func.verify() {
                panic!(
                    "modular recurrence collapse produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Plan {
    header: BlockId,
    preheader: BlockId,
    exit: BlockId,
    state: ValueId,
    initial_state: ValueId,
    bound: ValueId,
    delta: i32,
    modulus: i32,
    safe_bound: i32,
}

fn find_plan(func: &Function) -> Option<Plan> {
    if func.blocks.len() > 64 || func.values.len() > 512 {
        return None;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let [natural_loop] = loop_info.loops() else {
        return None;
    };
    let preheader = natural_loop.dedicated_preheader?;
    let latch = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    if natural_loop.blocks.len() > 4
        || func
            .block(exit)
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
    {
        return None;
    }

    let phis = func
        .block(natural_loop.header)
        .insts
        .iter()
        .filter_map(|inst| match inst.kind {
            InstKind::Phi { .. } => inst.result,
            _ => None,
        })
        .collect::<Vec<_>>();
    let [first, second] = phis.as_slice() else {
        return None;
    };
    let (counter, state) = if analyze_i32_induction(func, natural_loop, *first).is_some() {
        (*first, *second)
    } else if analyze_i32_induction(func, natural_loop, *second).is_some() {
        (*second, *first)
    } else {
        return None;
    };
    let induction = analyze_i32_induction(func, natural_loop, counter)?;
    if induction.step != 1 || !is_const_i32(func, induction.initial, 0) {
        return None;
    }

    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.block(natural_loop.header).terminator.as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || natural_loop.blocks.contains(else_target) {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt,
        lhs,
        rhs: bound,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    if *lhs != counter || value_defined_in_loop(func, *bound, natural_loop) {
        return None;
    }

    let initial_state = phi_incoming(func, state, preheader)?;
    let next_state = phi_incoming(func, state, latch)?;
    let InstKind::Binary {
        op: BinaryOp::Imod,
        lhs: added,
        rhs: modulus_value,
    } = defining_inst(func, next_state)?
    else {
        return None;
    };
    let modulus = const_i32(func, *modulus_value)?;
    if modulus <= 1 {
        return None;
    }
    let InstKind::Binary {
        op: BinaryOp::Iadd,
        lhs: add_lhs,
        rhs: add_rhs,
    } = defining_inst(func, *added)?
    else {
        return None;
    };
    let delta = if *add_lhs == state {
        const_i32(func, *add_rhs)?
    } else if *add_rhs == state {
        const_i32(func, *add_lhs)?
    } else {
        return None;
    };
    if delta <= 0 {
        return None;
    }

    let safe_bound = (i64::from(i32::MAX) - i64::from(modulus - 1)) / i64::from(delta);
    if safe_bound < 1 || safe_bound > i64::from(i32::MAX) {
        return None;
    }
    if !loop_is_exact_recurrence(
        func,
        natural_loop,
        counter,
        state,
        *cond,
        *added,
        next_state,
    ) {
        return None;
    }

    Some(Plan {
        header: natural_loop.header,
        preheader,
        exit,
        state,
        initial_state,
        bound: *bound,
        delta,
        modulus,
        safe_bound: safe_bound as i32,
    })
}

fn loop_is_exact_recurrence(
    func: &Function,
    natural_loop: &NaturalLoop,
    counter: ValueId,
    state: ValueId,
    condition: ValueId,
    added: ValueId,
    next_state: ValueId,
) -> bool {
    let mut allowed_results = HashSet::from([counter, state, condition, added, next_state]);
    let Some(induction) = analyze_i32_induction(func, natural_loop, counter) else {
        return false;
    };
    allowed_results.insert(induction.next);

    natural_loop.blocks.iter().all(|block| {
        func.block(*block).insts.iter().all(|inst| {
            matches!(inst.kind, InstKind::Nop)
                || inst
                    .result
                    .is_some_and(|result| allowed_results.contains(&result))
        })
    })
}

fn apply_plan(func: &mut Function, plan: Plan) {
    let fast = func.add_block("modular.recurrence.fast");
    let zero = func.add_const(Const::Int(0));
    let modulus = func.add_const(Const::Int(plan.modulus));
    let delta = func.add_const(Const::Int(plan.delta));
    let safe_bound = func.add_const(Const::Int(plan.safe_bound));

    let state_non_negative = append_cmp(func, plan.preheader, CmpOp::Ge, plan.initial_state, zero);
    let state_below_modulus =
        append_cmp(func, plan.preheader, CmpOp::Lt, plan.initial_state, modulus);
    let bound_non_negative = append_cmp(func, plan.preheader, CmpOp::Ge, plan.bound, zero);
    let bound_is_safe = append_cmp(func, plan.preheader, CmpOp::Le, plan.bound, safe_bound);
    let valid_state = append_bool_and(
        func,
        plan.preheader,
        state_non_negative,
        state_below_modulus,
    );
    let valid_bound = append_bool_and(func, plan.preheader, bound_non_negative, bound_is_safe);
    let use_fast = append_bool_and(func, plan.preheader, valid_state, valid_bound);

    let repeated_delta = func
        .append_inst(
            fast,
            InstKind::Binary {
                op: BinaryOp::Imul,
                lhs: plan.bound,
                rhs: delta,
            },
            Some(Type::I32),
        )
        .unwrap();
    let advanced = func
        .append_inst(
            fast,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: plan.initial_state,
                rhs: repeated_delta,
            },
            Some(Type::I32),
        )
        .unwrap();
    let reduced = func
        .append_inst(
            fast,
            InstKind::Binary {
                op: BinaryOp::Imod,
                lhs: advanced,
                rhs: modulus,
            },
            Some(Type::I32),
        )
        .unwrap();

    func.block_mut(plan.preheader).terminator = Some(Terminator::Branch {
        cond: use_fast,
        then_target: fast,
        else_target: plan.header,
    });
    func.set_terminator(fast, Terminator::Jump(plan.exit));

    let merged = func
        .insert_inst(
            plan.exit,
            0,
            InstKind::Phi {
                incomings: vec![(plan.header, plan.state), (fast, reduced)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let reachable = reachable_from(func, plan.exit);
    let replacements = ValueReplacements::from([(plan.state, merged)]);
    for block in reachable {
        let start = usize::from(block == plan.exit);
        rewrite_block_uses_from(func, block, start, &replacements);
    }
}

fn append_cmp(
    func: &mut Function,
    block: BlockId,
    op: CmpOp,
    lhs: ValueId,
    rhs: ValueId,
) -> ValueId {
    func.append_inst(block, InstKind::Icmp { op, lhs, rhs }, Some(Type::I1))
        .unwrap()
}

fn append_bool_and(func: &mut Function, block: BlockId, lhs: ValueId, rhs: ValueId) -> ValueId {
    func.append_inst(
        block,
        InstKind::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
        },
        Some(Type::I1),
    )
    .unwrap()
}

fn reachable_from(func: &Function, start: BlockId) -> Vec<BlockId> {
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    while let Some(block) = stack.pop() {
        if !seen.insert(block) {
            continue;
        }
        match func.block(block).terminator.as_ref() {
            Some(Terminator::Jump(target)) => stack.push(*target),
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => {
                stack.push(*then_target);
                stack.push(*else_target);
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }
    let mut blocks = seen.into_iter().collect::<Vec<_>>();
    blocks.sort_by_key(|block| block.0);
    blocks
}

fn value_defined_in_loop(func: &Function, value: ValueId, natural_loop: &NaturalLoop) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Inst(block, _)) if natural_loop.blocks.contains(block)
    )
}

fn phi_incoming(func: &Function, phi: ValueId, pred: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    incomings
        .iter()
        .find_map(|(from, value)| (*from == pred).then_some(*value))
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, index) = func.values.get(value.0)?.kind else {
        return None;
    };
    Some(&func.blocks.get(block.0)?.insts.get(index)?.kind)
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Const(Const::Int(value))) => Some(*value),
        Some(ValueKind::Const(Const::Zero(Type::I32))) => Some(0),
        _ => None,
    }
}

fn is_const_i32(func: &Function, value: ValueId, expected: i32) -> bool {
    const_i32(func, value) == Some(expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    #[test]
    fn adds_a_guarded_fast_path_for_a_positive_modular_recurrence() {
        let source = r#"
int advance(int state) {
    int count = state % 8;
    int i = 0;
    while (i < count) {
        state = state + 3;
        state = state % 97;
        i = i + 1;
    }
    return state % 97;
}
"#;
        let mut module = lower_program(&Parser::new(source).parse_program()).unwrap();
        super::super::ScalarPromotePass::new().run(&mut module);
        super::super::DcePass::new().run(&mut module);
        assert!(find_plan(&module.funcs[0]).is_some());

        ModularRecurrencePass::new().run(&mut module);
        let func = &module.funcs[0];
        assert!(func
            .blocks
            .iter()
            .any(|block| block.name == "modular.recurrence.fast"));
        assert!(func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Binary {
                        op: BinaryOp::Imod,
                        ..
                    }
                )
            })
        }));
    }

    #[test]
    fn rejects_a_recurrence_that_can_move_below_zero() {
        let source = r#"
int retreat(int state) {
    int count = state % 8;
    int i = 0;
    while (i < count) {
        state = (state - 3) % 97;
        i = i + 1;
    }
    return state;
}
"#;
        let mut module = lower_program(&Parser::new(source).parse_program()).unwrap();
        super::super::ScalarPromotePass::new().run(&mut module);
        super::super::DcePass::new().run(&mut module);
        assert!(find_plan(&module.funcs[0]).is_none());
    }
}
