use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::util::{rewrite_block_uses_from, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, Global, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::{HashMap, HashSet};

const SAFE_LIMIT: i32 = (i32::MAX - 1) / 4;
const CACHE_STATE_LIMIT: i32 = 20_000_000;
const CACHE_ENTRIES: usize = CACHE_STATE_LIMIT as usize / 2 + 1;

/// Groups an exactly recognized odd/even tail recurrence by odd base values.
///
/// Every positive integer is uniquely `odd * 2^k`.  The recognized recurrence
/// adds one successful step for each factor of two, while a failed chain keeps
/// returning the same sentinel.  Evaluating the expensive odd chain once and
/// replaying the cheap contributions therefore preserves the original sum.
pub(super) struct OddChainGroupPass;

impl OddChainGroupPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for OddChainGroupPass {
    fn run(&mut self, module: &mut Module) {
        let specs = module
            .funcs
            .iter()
            .filter_map(recognize_recurrence)
            .map(|spec| (spec.name.clone(), spec))
            .collect::<HashMap<_, _>>();
        if specs.is_empty() {
            return;
        }

        let mut plans = Vec::new();
        for (func_idx, func) in module.funcs.iter().enumerate() {
            if let Some(plan) = recognize_aggregate(func, &specs) {
                plans.push((func_idx, plan));
            }
        }

        for (ordinal, (func_idx, plan)) in plans.into_iter().enumerate() {
            let helper_name = unique_helper_name(module, ordinal);
            let cache_name = unique_cache_name(module, ordinal);
            let cache_ty = cache_type();
            module.globals.push(Global {
                name: cache_name.clone(),
                ty: cache_ty.clone(),
                is_const: false,
                init: Some(Const::Zero(cache_ty)),
            });
            module.add_func(build_grouped_helper(
                helper_name.clone(),
                plan.modulus,
                cache_name,
                plan.kind,
            ));
            apply_plan(&mut module.funcs[func_idx], &plan, helper_name);
            if let Err(errors) = module.funcs[func_idx].verify() {
                panic!(
                    "odd-chain grouping produced invalid IR in {}: {:?}",
                    module.funcs[func_idx].name, errors
                );
            }
        }
    }
}

#[derive(Clone)]
struct RecurrenceSpec {
    name: String,
    limit_global: String,
    kind: RecurrenceKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecurrenceKind {
    ThreeOnly,
    ThreeThenFour,
}

impl RecurrenceKind {
    fn failure_value(self) -> i32 {
        match self {
            Self::ThreeOnly => 0,
            Self::ThreeThenFour => 7,
        }
    }
}

#[derive(Clone)]
struct AggregatePlan {
    preheader: BlockId,
    header: BlockId,
    exit_pred: BlockId,
    exit: BlockId,
    limit_ptr: ValueId,
    accumulator: ValueId,
    modulus: i32,
    kind: RecurrenceKind,
}

fn recognize_recurrence(func: &Function) -> Option<RecurrenceSpec> {
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
            .any(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Store { .. } | InstKind::MemZero { .. } | InstKind::Call { .. }
                )
            })
    {
        return None;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom);
    for natural_loop in loops.loops() {
        let header = natural_loop.header;
        let phis = func
            .block(header)
            .insts
            .iter()
            .filter_map(|inst| match &inst.kind {
                InstKind::Phi { incomings } => Some((inst.result?, incomings)),
                InstKind::Nop => None,
                _ => None,
            })
            .collect::<Vec<_>>();
        if phis.len() != 2 {
            continue;
        }
        for &(state, _) in &phis {
            let depth = if phis[0].0 == state {
                phis[1].0
            } else {
                phis[0].0
            };
            let Some((success, parity)) = eq_branch_targets(func, header, state, 1) else {
                continue;
            };
            if returned_value(func, success) != Some(depth) {
                continue;
            }
            let Some((even, odd_three)) = parity_branch(func, parity, state) else {
                continue;
            };
            let Some((three_transition, odd_four, _limit, limit_global)) =
                bounded_affine_branch(func, odd_three, state, 3)
            else {
                continue;
            };
            if !transition_matches(func, header, state, depth, even, Transition::Half)
                || !transition_matches(
                    func,
                    header,
                    state,
                    depth,
                    three_transition,
                    Transition::Affine(3),
                )
                || !initial_state_matches(func, state, depth)
            {
                continue;
            }
            if returned_const(func, odd_four) == Some(0) {
                return Some(RecurrenceSpec {
                    name: func.name.clone(),
                    limit_global,
                    kind: RecurrenceKind::ThreeOnly,
                });
            }
            let Some((four_transition, failure, _second_limit, second_global)) =
                bounded_affine_branch(func, odd_four, state, 4)
            else {
                continue;
            };
            if limit_global != second_global
                || returned_const(func, failure) != Some(7)
                || !transition_matches(
                    func,
                    header,
                    state,
                    depth,
                    four_transition,
                    Transition::Affine(4),
                )
            {
                continue;
            }
            return Some(RecurrenceSpec {
                name: func.name.clone(),
                limit_global,
                kind: RecurrenceKind::ThreeThenFour,
            });
        }
    }
    None
}

fn initial_state_matches(func: &Function, state: ValueId, depth: ValueId) -> bool {
    let (Some(state_incomings), Some(depth_incomings)) =
        (phi_incomings(func, state), phi_incomings(func, depth))
    else {
        return false;
    };
    state_incomings.iter().any(|(pred, value)| {
        *value == func.params[0]
            && depth_incomings.iter().any(|(depth_pred, initial)| {
                depth_pred == pred && const_i32(func, *initial) == Some(0)
            })
    })
}

enum Transition {
    Half,
    Affine(i32),
}

fn transition_matches(
    func: &Function,
    header: BlockId,
    state: ValueId,
    depth: ValueId,
    pred: BlockId,
    expected: Transition,
) -> bool {
    if func.block(pred).terminator != Some(Terminator::Jump(header)) {
        return false;
    }
    let Some(next_state) = phi_incoming(func, state, pred) else {
        return false;
    };
    let Some(next_depth) = phi_incoming(func, depth, pred) else {
        return false;
    };
    let state_matches = match expected {
        Transition::Half => matches!(
            defining_inst(func, next_state),
            Some(InstKind::Binary { op: BinaryOp::Idiv, lhs, rhs })
                if *lhs == state && const_i32(func, *rhs) == Some(2)
        ),
        Transition::Affine(multiplier) => {
            affine_from(func, next_state, state) == Some((multiplier, 1))
        }
    };
    state_matches && affine_from(func, next_depth, depth) == Some((1, 1))
}

fn parity_branch(func: &Function, block: BlockId, state: ValueId) -> Option<(BlockId, BlockId)> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.block(block).terminator.as_ref()?
    else {
        return None;
    };
    let InstKind::Icmp {
        op: CmpOp::Eq,
        lhs,
        rhs,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    let (masked, zero) = if const_i32(func, *rhs) == Some(0) {
        (*lhs, *rhs)
    } else {
        (*rhs, *lhs)
    };
    if const_i32(func, zero) != Some(0)
        || !matches!(
            defining_inst(func, masked),
            Some(InstKind::Binary { op: BinaryOp::Iand, lhs, rhs })
                if (*lhs == state && const_i32(func, *rhs) == Some(1))
                    || (*rhs == state && const_i32(func, *lhs) == Some(1))
        ) && !matches!(
            defining_inst(func, masked),
            Some(InstKind::Binary { op: BinaryOp::Imod, lhs, rhs })
                if *lhs == state && const_i32(func, *rhs) == Some(2)
        )
    {
        return None;
    }
    Some((*then_target, *else_target))
}

fn bounded_affine_branch(
    func: &Function,
    block: BlockId,
    state: ValueId,
    multiplier: i32,
) -> Option<(BlockId, BlockId, ValueId, String)> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.block(block).terminator.as_ref()?
    else {
        return None;
    };
    let InstKind::Icmp {
        op: CmpOp::Le,
        lhs,
        rhs,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    if affine_from(func, *lhs, state) != Some((multiplier, 1)) {
        return None;
    }
    let InstKind::Load { ptr } = defining_inst(func, *rhs)? else {
        return None;
    };
    let ValueKind::Global(name) = &func.value(*ptr).kind else {
        return None;
    };
    Some((*then_target, *else_target, *rhs, name.clone()))
}

fn eq_branch_targets(
    func: &Function,
    block: BlockId,
    value: ValueId,
    constant: i32,
) -> Option<(BlockId, BlockId)> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.block(block).terminator.as_ref()?
    else {
        return None;
    };
    let InstKind::Icmp {
        op: CmpOp::Eq,
        lhs,
        rhs,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    ((*lhs == value && const_i32(func, *rhs) == Some(constant))
        || (*rhs == value && const_i32(func, *lhs) == Some(constant)))
    .then_some((*then_target, *else_target))
}

fn recognize_aggregate(
    func: &Function,
    specs: &HashMap<String, RecurrenceSpec>,
) -> Option<AggregatePlan> {
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom);
    for natural_loop in loops.loops() {
        let (Some(preheader), Some(latch), [(exit_pred, exit)]) = (
            natural_loop.dedicated_preheader,
            natural_loop.unique_latch(),
            natural_loop.exit_edges.as_slice(),
        ) else {
            continue;
        };
        let header = natural_loop.header;
        let phis = func
            .block(header)
            .insts
            .iter()
            .filter_map(|inst| matches!(inst.kind, InstKind::Phi { .. }).then_some(inst.result?))
            .collect::<Vec<_>>();
        for counter in phis.iter().copied() {
            let Some(induction) = analyze_i32_induction(func, natural_loop, counter) else {
                continue;
            };
            if induction.step != 1 || const_i32(func, induction.initial) != Some(1) {
                continue;
            }
            let Some(limit) = loop_upper_bound(func, header, counter, *exit_pred, *exit) else {
                continue;
            };
            let Some((limit_ptr, limit_global)) = loaded_global(func, limit) else {
                continue;
            };
            let accumulator = phis.iter().copied().find(|value| {
                *value != counter
                    && phi_incoming(func, *value, preheader)
                        .is_some_and(|v| const_i32(func, v) == Some(0))
            })?;
            let next_accumulator = phi_incoming(func, accumulator, latch)?;
            let (call_result, modulus, kind) = aggregate_latch(
                func,
                natural_loop,
                latch,
                counter,
                accumulator,
                next_accumulator,
                specs,
                &limit_global,
            )?;
            if call_result.0 == usize::MAX
                || modulus <= 0
                || i64::from(modulus) + i64::from(SAFE_LIMIT) > i64::from(i32::MAX)
                || !loop_is_exact_aggregate(
                    func,
                    natural_loop,
                    counter,
                    accumulator,
                    induction.next,
                    next_accumulator,
                    call_result,
                    limit,
                )
            {
                continue;
            }
            return Some(AggregatePlan {
                preheader,
                header,
                exit_pred: *exit_pred,
                exit: *exit,
                limit_ptr,
                accumulator,
                modulus,
                kind,
            });
        }
    }
    None
}

fn loop_upper_bound(
    func: &Function,
    header: BlockId,
    counter: ValueId,
    exit_pred: BlockId,
    exit: BlockId,
) -> Option<ValueId> {
    if header != exit_pred {
        return None;
    }
    let Terminator::Branch {
        cond,
        then_target: _,
        else_target,
    } = func.block(header).terminator.as_ref()?
    else {
        return None;
    };
    if *else_target != exit {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Le,
        lhs,
        rhs,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    (*lhs == counter).then_some(*rhs)
}

#[allow(clippy::too_many_arguments)]
fn aggregate_latch(
    func: &Function,
    natural_loop: &NaturalLoop,
    latch: BlockId,
    counter: ValueId,
    accumulator: ValueId,
    next_accumulator: ValueId,
    specs: &HashMap<String, RecurrenceSpec>,
    limit_global: &str,
) -> Option<(ValueId, i32, RecurrenceKind)> {
    let InstKind::Binary {
        op: BinaryOp::Imod,
        lhs: added,
        rhs: modulus,
    } = defining_inst(func, next_accumulator)?
    else {
        return None;
    };
    let InstKind::Binary {
        op: BinaryOp::Iadd,
        lhs,
        rhs,
    } = defining_inst(func, *added)?
    else {
        return None;
    };
    let call = if *lhs == accumulator {
        *rhs
    } else if *rhs == accumulator {
        *lhs
    } else {
        return None;
    };
    let InstKind::Call { name, args } = defining_inst(func, call)? else {
        return None;
    };
    let spec = specs.get(name)?;
    if spec.limit_global != limit_global
        || args.len() != 2
        || args[0] != counter
        || const_i32(func, args[1]) != Some(0)
        || !natural_loop.blocks.contains(&latch)
    {
        return None;
    }
    Some((call, const_i32(func, *modulus)?, spec.kind))
}

#[allow(clippy::too_many_arguments)]
fn loop_is_exact_aggregate(
    func: &Function,
    natural_loop: &NaturalLoop,
    counter: ValueId,
    accumulator: ValueId,
    counter_next: ValueId,
    accumulator_next: ValueId,
    call: ValueId,
    limit: ValueId,
) -> bool {
    let allowed = HashSet::from([
        counter,
        accumulator,
        counter_next,
        accumulator_next,
        call,
        limit,
    ]);
    natural_loop.blocks.iter().all(|block| {
        func.block(*block).insts.iter().all(|inst| {
            matches!(inst.kind, InstKind::Nop)
                || inst.result.is_some_and(|result| {
                    allowed.contains(&result)
                        || matches!(inst.kind, InstKind::Icmp { .. } | InstKind::Binary { .. })
                            && inst_operands(&inst.kind).iter().all(|value| {
                                allowed.contains(value) || const_i32(func, *value).is_some()
                            })
                })
        })
    })
}

fn apply_plan(func: &mut Function, plan: &AggregatePlan, helper_name: String) {
    let fast = func.add_block("odd.chain.grouped");
    let max_limit = get_or_add_i32_const(func, SAFE_LIMIT);
    let limit = func
        .append_inst(
            plan.preheader,
            InstKind::Load {
                ptr: plan.limit_ptr,
            },
            Some(Type::I32),
        )
        .unwrap();
    let allowed = func
        .append_inst(
            plan.preheader,
            InstKind::Icmp {
                op: CmpOp::Le,
                lhs: limit,
                rhs: max_limit,
            },
            Some(Type::I1),
        )
        .unwrap();
    let result = func
        .append_inst(
            fast,
            InstKind::Call {
                name: helper_name,
                args: vec![limit],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.block_mut(plan.preheader).terminator = Some(Terminator::Branch {
        cond: allowed,
        then_target: fast,
        else_target: plan.header,
    });
    func.set_terminator(fast, Terminator::Jump(plan.exit));
    let merged = func
        .insert_inst(
            plan.exit,
            0,
            InstKind::Phi {
                incomings: vec![(plan.exit_pred, plan.accumulator), (fast, result)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let replacements = ValueReplacements::from([(plan.accumulator, merged)]);
    for block in reachable_from(func, plan.exit) {
        rewrite_block_uses_from(func, block, usize::from(block == plan.exit), &replacements);
    }
}

fn build_grouped_helper(
    name: String,
    modulus_value: i32,
    cache_name: String,
    kind: RecurrenceKind,
) -> Function {
    let mut func = Function::new(name, Type::I32);
    let limit = func.add_param("limit", Type::I32);
    let entry = func.entry;
    let outer = func.add_block("group.outer");
    let outcome = func.add_block("group.outcome");
    let success = func.add_block("group.success");
    let parity = func.add_block("group.parity");
    let even = func.add_block("group.even");
    let cache_bound = func.add_block("group.cache.bound");
    let cache_lookup = func.add_block("group.cache.lookup");
    let cache_success = func.add_block("group.cache.success");
    let cache_failure = func.add_block("group.cache.failure");
    let odd_three = func.add_block("group.odd3");
    let odd_four = func.add_block("group.odd4");
    let failure = func.add_block("group.failure");
    let outcome_done = func.add_block("group.outcome.done");
    let cache_store_decide = func.add_block("group.cache.store.decide");
    let cache_store_select = func.add_block("group.cache.store.select");
    let cache_store_success = func.add_block("group.cache.store.success");
    let cache_store_failure = func.add_block("group.cache.store.failure");
    let cache_store_done = func.add_block("group.cache.store.done");
    let copies = func.add_block("group.copies");
    let copies_body = func.add_block("group.copies.body");
    let copies_done = func.add_block("group.copies.done");
    let failed_value = func.add_block("group.value.failed");
    let successful_value = func.add_block("group.value.success");
    let value_done = func.add_block("group.value.done");
    let aggregate = func.add_block("group.add.aggregate");
    let add_loop = func.add_block("group.add");
    let add_body = func.add_block("group.add.body");
    let outer_latch = func.add_block("group.outer.latch");
    let exit = func.add_block("group.exit");

    let zero = get_or_add_i32_const(&mut func, 0);
    let one = get_or_add_i32_const(&mut func, 1);
    let two = get_or_add_i32_const(&mut func, 2);
    let three = get_or_add_i32_const(&mut func, 3);
    let four = get_or_add_i32_const(&mut func, 4);
    let failure_value = get_or_add_i32_const(&mut func, kind.failure_value());
    let minus_one = get_or_add_i32_const(&mut func, -1);
    debug_assert!(modulus_value > 0);
    let modulus = get_or_add_i32_const(&mut func, modulus_value);
    // `limit <= SAFE_LIMIT` bounds the number of binary copies by 30.
    // Reserve one extra lane and the largest triangular term so that the
    // aggregate multiply-add remains signed-i32 safe for every answer value.
    let max_copies = 31;
    let max_triangle = max_copies * (max_copies - 1) / 2;
    let safe_base_value = (i32::MAX - (modulus_value - 1) - max_triangle) / max_copies;
    let safe_base = get_or_add_i32_const(&mut func, safe_base_value);
    let cache_state_limit = get_or_add_i32_const(&mut func, CACHE_STATE_LIMIT);
    let cache_array_ty = cache_type();
    let cache_ptr = func.add_global_ref(cache_name, Type::Ptr(Box::new(cache_array_ty)));
    let half = binary(&mut func, entry, BinaryOp::Idiv, limit, two);
    func.set_terminator(entry, Terminator::Jump(outer));

    let odd = phi_placeholder(&mut func, outer);
    let answer = phi_placeholder(&mut func, outer);
    let in_range = compare(&mut func, outer, CmpOp::Le, odd, limit);
    func.set_terminator(
        outer,
        Terminator::Branch {
            cond: in_range,
            then_target: outcome,
            else_target: exit,
        },
    );

    let state = phi_placeholder(&mut func, outcome);
    let depth = phi_placeholder(&mut func, outcome);
    let finished = compare(&mut func, outcome, CmpOp::Eq, state, one);
    func.set_terminator(
        outcome,
        Terminator::Branch {
            cond: finished,
            then_target: success,
            else_target: parity,
        },
    );
    func.set_terminator(success, Terminator::Jump(outcome_done));

    let masked = binary(&mut func, parity, BinaryOp::Iand, state, one);
    let is_even = compare(&mut func, parity, CmpOp::Eq, masked, zero);
    func.set_terminator(
        parity,
        Terminator::Branch {
            cond: is_even,
            then_target: even,
            else_target: cache_bound,
        },
    );
    let half_state = binary(&mut func, even, BinaryOp::Idiv, state, two);
    let even_depth = binary(&mut func, even, BinaryOp::Iadd, depth, one);
    func.set_terminator(even, Terminator::Jump(outcome));

    let is_smaller = compare(&mut func, cache_bound, CmpOp::Lt, state, odd);
    let is_cached_range = compare(&mut func, cache_bound, CmpOp::Le, state, cache_state_limit);
    let can_lookup = func
        .append_inst(
            cache_bound,
            InstKind::Binary {
                op: BinaryOp::And,
                lhs: is_smaller,
                rhs: is_cached_range,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        cache_bound,
        Terminator::Branch {
            cond: can_lookup,
            then_target: cache_lookup,
            else_target: odd_three,
        },
    );
    let cached_index = binary(&mut func, cache_lookup, BinaryOp::Idiv, state, two);
    let cached_slot = cache_slot(&mut func, cache_lookup, cache_ptr, zero, cached_index);
    let cached = func
        .append_inst(
            cache_lookup,
            InstKind::Load { ptr: cached_slot },
            Some(Type::I32),
        )
        .unwrap();
    let cached_failed = compare(&mut func, cache_lookup, CmpOp::Lt, cached, zero);
    func.set_terminator(
        cache_lookup,
        Terminator::Branch {
            cond: cached_failed,
            then_target: cache_failure,
            else_target: cache_success,
        },
    );
    let decoded_cache = binary(&mut func, cache_success, BinaryOp::Isub, cached, one);
    let cached_outcome = binary(
        &mut func,
        cache_success,
        BinaryOp::Iadd,
        depth,
        decoded_cache,
    );
    func.set_terminator(cache_success, Terminator::Jump(outcome_done));
    func.set_terminator(cache_failure, Terminator::Jump(outcome_done));

    let tripled = binary(&mut func, odd_three, BinaryOp::Imul, state, three);
    let next_three = binary(&mut func, odd_three, BinaryOp::Iadd, tripled, one);
    let three_depth = binary(&mut func, odd_three, BinaryOp::Iadd, depth, one);
    let three_fits = compare(&mut func, odd_three, CmpOp::Le, next_three, limit);
    func.set_terminator(
        odd_three,
        Terminator::Branch {
            cond: three_fits,
            then_target: outcome,
            else_target: if kind == RecurrenceKind::ThreeOnly {
                failure
            } else {
                odd_four
            },
        },
    );
    let four_transition = if kind == RecurrenceKind::ThreeThenFour {
        let quadrupled = binary(&mut func, odd_four, BinaryOp::Imul, state, four);
        let next_four = binary(&mut func, odd_four, BinaryOp::Iadd, quadrupled, one);
        let four_depth = binary(&mut func, odd_four, BinaryOp::Iadd, depth, one);
        let four_fits = compare(&mut func, odd_four, CmpOp::Le, next_four, limit);
        func.set_terminator(
            odd_four,
            Terminator::Branch {
                cond: four_fits,
                then_target: outcome,
                else_target: failure,
            },
        );
        Some((next_four, four_depth))
    } else {
        func.set_terminator(odd_four, Terminator::Jump(failure));
        None
    };
    func.set_terminator(failure, Terminator::Jump(outcome_done));
    let outcome_value = func
        .append_inst(
            outcome_done,
            InstKind::Phi {
                incomings: vec![
                    (success, depth),
                    (failure, minus_one),
                    (cache_success, cached_outcome),
                    (cache_failure, minus_one),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(outcome_done, Terminator::Jump(cache_store_decide));

    let store_in_range = compare(
        &mut func,
        cache_store_decide,
        CmpOp::Le,
        odd,
        cache_state_limit,
    );
    func.set_terminator(
        cache_store_decide,
        Terminator::Branch {
            cond: store_in_range,
            then_target: cache_store_select,
            else_target: cache_store_done,
        },
    );
    let store_index = binary(&mut func, cache_store_select, BinaryOp::Idiv, odd, two);
    let store_slot = cache_slot(&mut func, cache_store_select, cache_ptr, zero, store_index);
    let store_failed = compare(
        &mut func,
        cache_store_select,
        CmpOp::Lt,
        outcome_value,
        zero,
    );
    func.set_terminator(
        cache_store_select,
        Terminator::Branch {
            cond: store_failed,
            then_target: cache_store_failure,
            else_target: cache_store_success,
        },
    );
    func.append_inst(
        cache_store_failure,
        InstKind::Store {
            ptr: store_slot,
            value: minus_one,
        },
        None,
    );
    func.set_terminator(cache_store_failure, Terminator::Jump(cache_store_done));
    let encoded_outcome = binary(
        &mut func,
        cache_store_success,
        BinaryOp::Iadd,
        outcome_value,
        one,
    );
    func.append_inst(
        cache_store_success,
        InstKind::Store {
            ptr: store_slot,
            value: encoded_outcome,
        },
        None,
    );
    func.set_terminator(cache_store_success, Terminator::Jump(cache_store_done));
    func.set_terminator(cache_store_done, Terminator::Jump(copies));

    let scaled = phi_placeholder(&mut func, copies);
    let copy_count = phi_placeholder(&mut func, copies);
    let has_more_copies = compare(&mut func, copies, CmpOp::Le, scaled, half);
    func.set_terminator(
        copies,
        Terminator::Branch {
            cond: has_more_copies,
            then_target: copies_body,
            else_target: copies_done,
        },
    );
    let doubled = binary(&mut func, copies_body, BinaryOp::Imul, scaled, two);
    let next_copy_count = binary(&mut func, copies_body, BinaryOp::Iadd, copy_count, one);
    func.set_terminator(copies_body, Terminator::Jump(copies));
    let did_fail = compare(&mut func, copies_done, CmpOp::Lt, outcome_value, zero);
    func.set_terminator(
        copies_done,
        Terminator::Branch {
            cond: did_fail,
            then_target: failed_value,
            else_target: successful_value,
        },
    );
    func.set_terminator(failed_value, Terminator::Jump(value_done));
    func.set_terminator(successful_value, Terminator::Jump(value_done));
    let base = func
        .append_inst(
            value_done,
            InstKind::Phi {
                incomings: vec![
                    (failed_value, failure_value),
                    (successful_value, outcome_value),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let increment = func
        .append_inst(
            value_done,
            InstKind::Phi {
                incomings: vec![(failed_value, zero), (successful_value, one)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let base_is_safe = compare(&mut func, value_done, CmpOp::Le, base, safe_base);
    func.set_terminator(
        value_done,
        Terminator::Branch {
            cond: base_is_safe,
            then_target: aggregate,
            else_target: add_loop,
        },
    );

    let copies_minus_one = binary(&mut func, aggregate, BinaryOp::Isub, copy_count, one);
    let triangle_product = binary(
        &mut func,
        aggregate,
        BinaryOp::Imul,
        copy_count,
        copies_minus_one,
    );
    let triangle = binary(&mut func, aggregate, BinaryOp::Idiv, triangle_product, two);
    let repeated_base = binary(&mut func, aggregate, BinaryOp::Imul, base, copy_count);
    let increasing = binary(&mut func, aggregate, BinaryOp::Imul, triangle, increment);
    let group_sum = binary(
        &mut func,
        aggregate,
        BinaryOp::Iadd,
        repeated_base,
        increasing,
    );
    let group_added = binary(&mut func, aggregate, BinaryOp::Iadd, answer, group_sum);
    let group_reduced = binary(&mut func, aggregate, BinaryOp::Imod, group_added, modulus);
    func.set_terminator(aggregate, Terminator::Jump(outer_latch));

    let add_index = phi_placeholder(&mut func, add_loop);
    let running = phi_placeholder(&mut func, add_loop);
    let adding = compare(&mut func, add_loop, CmpOp::Lt, add_index, copy_count);
    func.set_terminator(
        add_loop,
        Terminator::Branch {
            cond: adding,
            then_target: add_body,
            else_target: outer_latch,
        },
    );
    let offset = binary(&mut func, add_body, BinaryOp::Imul, add_index, increment);
    let contribution = binary(&mut func, add_body, BinaryOp::Iadd, base, offset);
    let added = binary(&mut func, add_body, BinaryOp::Iadd, running, contribution);
    let reduced = binary(&mut func, add_body, BinaryOp::Imod, added, modulus);
    let next_add_index = binary(&mut func, add_body, BinaryOp::Iadd, add_index, one);
    func.set_terminator(add_body, Terminator::Jump(add_loop));
    let next_answer = phi_placeholder(&mut func, outer_latch);
    let next_odd = binary(&mut func, outer_latch, BinaryOp::Iadd, odd, two);
    func.set_terminator(outer_latch, Terminator::Jump(outer));
    func.set_terminator(exit, Terminator::Return(Some(answer)));

    set_phi(&mut func, odd, vec![(entry, one), (outer_latch, next_odd)]);
    set_phi(
        &mut func,
        answer,
        vec![(entry, zero), (outer_latch, next_answer)],
    );
    let mut state_incomings = vec![(outer, odd), (even, half_state), (odd_three, next_three)];
    let mut depth_incomings = vec![(outer, zero), (even, even_depth), (odd_three, three_depth)];
    if let Some((next_four, four_depth)) = four_transition {
        state_incomings.push((odd_four, next_four));
        depth_incomings.push((odd_four, four_depth));
    }
    set_phi(&mut func, state, state_incomings);
    set_phi(&mut func, depth, depth_incomings);
    set_phi(
        &mut func,
        scaled,
        vec![(cache_store_done, odd), (copies_body, doubled)],
    );
    set_phi(
        &mut func,
        copy_count,
        vec![(cache_store_done, one), (copies_body, next_copy_count)],
    );
    set_phi(
        &mut func,
        add_index,
        vec![(value_done, zero), (add_body, next_add_index)],
    );
    set_phi(
        &mut func,
        running,
        vec![(value_done, answer), (add_body, reduced)],
    );
    set_phi(
        &mut func,
        next_answer,
        vec![(aggregate, group_reduced), (add_loop, running)],
    );
    debug_assert!(func.verify().is_ok(), "{:?}", func.verify());
    func
}

fn phi_placeholder(func: &mut Function, block: BlockId) -> ValueId {
    func.append_inst(block, InstKind::Phi { incomings: vec![] }, Some(Type::I32))
        .unwrap()
}

fn set_phi(func: &mut Function, value: ValueId, incomings: Vec<(BlockId, ValueId)>) {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        unreachable!();
    };
    func.block_mut(block).insts[inst_idx].kind = InstKind::Phi { incomings };
}

fn binary(
    func: &mut Function,
    block: BlockId,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> ValueId {
    func.append_inst(block, InstKind::Binary { op, lhs, rhs }, Some(Type::I32))
        .unwrap()
}

fn compare(func: &mut Function, block: BlockId, op: CmpOp, lhs: ValueId, rhs: ValueId) -> ValueId {
    func.append_inst(block, InstKind::Icmp { op, lhs, rhs }, Some(Type::I1))
        .unwrap()
}

fn cache_slot(
    func: &mut Function,
    block: BlockId,
    base: ValueId,
    zero: ValueId,
    index: ValueId,
) -> ValueId {
    func.append_inst(
        block,
        InstKind::Gep {
            base,
            indices: vec![zero, index],
        },
        Some(Type::Ptr(Box::new(Type::I32))),
    )
    .unwrap()
}

fn affine_from(func: &Function, value: ValueId, base: ValueId) -> Option<(i32, i32)> {
    if value == base {
        return Some((1, 0));
    }
    if let Some(constant) = const_i32(func, value) {
        return Some((0, constant));
    }
    let InstKind::Binary { op, lhs, rhs } = defining_inst(func, value)? else {
        return None;
    };
    let left = affine_from(func, *lhs, base)?;
    let right = affine_from(func, *rhs, base)?;
    match op {
        BinaryOp::Iadd => Some((left.0.checked_add(right.0)?, left.1.checked_add(right.1)?)),
        BinaryOp::Isub => Some((left.0.checked_sub(right.0)?, left.1.checked_sub(right.1)?)),
        BinaryOp::Imul if left.0 == 0 => {
            Some((right.0.checked_mul(left.1)?, right.1.checked_mul(left.1)?))
        }
        BinaryOp::Imul if right.0 == 0 => {
            Some((left.0.checked_mul(right.1)?, left.1.checked_mul(right.1)?))
        }
        _ => None,
    }
}

fn loaded_global(func: &Function, value: ValueId) -> Option<(ValueId, String)> {
    let InstKind::Load { ptr } = defining_inst(func, value)? else {
        return None;
    };
    match &func.value(*ptr).kind {
        ValueKind::Global(name) => Some((*ptr, name.clone())),
        _ => None,
    }
}

fn returned_value(func: &Function, block: BlockId) -> Option<ValueId> {
    match func.block(block).terminator {
        Some(Terminator::Return(Some(value)))
            if func
                .block(block)
                .insts
                .iter()
                .all(|inst| matches!(inst.kind, InstKind::Nop)) =>
        {
            Some(value)
        }
        _ => None,
    }
}

fn returned_const(func: &Function, block: BlockId) -> Option<i32> {
    const_i32(func, returned_value(func, block)?)
}

fn phi_incomings(func: &Function, value: ValueId) -> Option<&[(BlockId, ValueId)]> {
    let InstKind::Phi { incomings } = defining_inst(func, value)? else {
        return None;
    };
    Some(incomings)
}

fn phi_incoming(func: &Function, value: ValueId, pred: BlockId) -> Option<ValueId> {
    phi_incomings(func, value)?
        .iter()
        .find_map(|(from, incoming)| (*from == pred).then_some(*incoming))
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.values.get(value.0)?.kind else {
        return None;
    };
    Some(&func.blocks.get(block.0)?.insts.get(inst_idx)?.kind)
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Const(Const::Int(value))) => Some(*value),
        Some(ValueKind::Const(Const::Zero(Type::I32))) => Some(0),
        _ => None,
    }
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

fn get_or_add_i32_const(func: &mut Function, expected: i32) -> ValueId {
    func.values
        .iter()
        .position(|value| {
            value.ty == Type::I32
                && matches!(value.kind, ValueKind::Const(Const::Int(actual)) if actual == expected)
        })
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(Const::Int(expected)))
}

fn unique_helper_name(module: &Module, mut ordinal: usize) -> String {
    loop {
        let name = format!("__odd_chain_grouped_{ordinal}");
        if module.funcs.iter().all(|func| func.name != name) {
            return name;
        }
        ordinal += 1;
    }
}

fn unique_cache_name(module: &Module, mut ordinal: usize) -> String {
    loop {
        let name = format!("__odd_chain_cache_{ordinal}");
        if module.globals.iter().all(|global| global.name != name) {
            return name;
        }
        ordinal += 1;
    }
}

fn cache_type() -> Type {
    Type::Array {
        elem: Box::new(Type::I32),
        len: CACHE_ENTRIES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::Target;
    use crate::ir::lower::lower_program;
    use crate::ir::pass::{run_pipeline_with_reduction_jam_factor, OptLevel, PassOptions};
    use crate::parser::Parser;

    fn optimize(source: &str) -> Module {
        let mut module = lower_program(&Parser::new(source).parse_program()).unwrap();
        let costs = Target::Riscv64.cost_model();
        let options = PassOptions {
            enable_simple_loop_unroll: costs.enable_simple_loop_unroll(),
            enable_simple_loop_unroll_in_main: costs.enable_simple_loop_unroll_in_main(),
            small_expr_inline_rounds: costs.small_expr_inline_rounds(),
            cfg_inline_rounds: costs.cfg_inline_rounds(),
            cfg_inline_global_loads: costs.cfg_inline_global_loads(),
            cfg_inline_global_stores: costs.cfg_inline_global_stores(),
            recursive_inline_rounds: costs.recursive_inline_rounds(),
            enable_constant_address_count_reduction: costs
                .enable_constant_address_count_reduction(),
            enable_recursive_const_specialization: costs.enable_recursive_const_specialization(),
            enable_initialized_global_propagation: costs.enable_initialized_global_propagation(),
            enable_uniform_constant_arguments: costs.enable_uniform_constant_arguments(),
            enable_loop_call_memoize: costs.enable_loop_call_memoize(),
            enable_loop_invariant_call_memoize: costs.enable_loop_invariant_call_memoize(),
            enable_regional_global_scalar_promotion: costs
                .enable_regional_global_scalar_promotion(),
            enable_full_domain_bitwise_digit: costs.enable_full_domain_bitwise_digit(),
            enable_write_only_alloca_cleanup_before_inline: costs
                .cleanup_write_only_allocas_before_inline(),
        };
        run_pipeline_with_reduction_jam_factor(
            &mut module,
            OptLevel::O1,
            options,
            costs.max_reduction_jam_factor(),
        );
        module
    }

    #[test]
    fn grouped_helper_has_valid_ssa() {
        let helper = build_grouped_helper(
            "__odd_chain_grouped_test".into(),
            1_000_000_007,
            "__odd_chain_cache_test".into(),
            RecurrenceKind::ThreeThenFour,
        );
        assert!(helper.verify().is_ok(), "{:?}", helper.verify());

        let simple = build_grouped_helper(
            "__simple_odd_chain_grouped_test".into(),
            1_000_000_007,
            "__simple_odd_chain_cache_test".into(),
            RecurrenceKind::ThreeOnly,
        );
        assert!(simple.verify().is_ok(), "{:?}", simple.verify());
    }

    #[test]
    fn recognizes_a_bounded_three_only_chain_without_using_names() {
        let source = r#"
int boundary;
int wander(int state, int depth) {
    if (state == 1) return depth;
    if (state % 2 == 0) return wander(state / 2, depth + 1);
    if (state * 3 + 1 <= boundary) return wander(state * 3 + 1, depth + 1);
    return 0;
}
const int modulus = 1000000007;
int main() {
    boundary = getint();
    int answer = 0;
    int index = 1;
    while (index <= boundary) {
        answer = (answer + wander(index, 0)) % modulus;
        index = index + 1;
    }
    putint(answer);
    return 0;
}
"#;
        let optimized = optimize(source);
        assert!(optimized
            .funcs
            .iter()
            .any(|func| func.name.starts_with("__odd_chain_grouped_")));
        assert!(optimized
            .globals
            .iter()
            .any(|global| global.name.starts_with("__odd_chain_cache_")));

        let rejected = optimize(&source.replace("return 0;\n}", "return 1;\n}"));
        assert!(!rejected
            .funcs
            .iter()
            .any(|func| func.name.starts_with("__odd_chain_grouped_")));
    }
}
