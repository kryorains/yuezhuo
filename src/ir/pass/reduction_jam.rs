use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{
    analyze_const_i32_trip_count, analyze_i32_induction, InductionVariable, LoopInfo, NaturalLoop,
};
use super::util::{const_i32, defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_SETUP_INSTRUCTIONS: usize = 16;
const MAX_INNER_INSTRUCTIONS: usize = 32;
const MAX_FACTOR_FOUR_ACTIVE_ACCUMULATORS: usize = 4;
const MAX_FACTOR_FOUR_POINTER_VALUES: usize = 12;
const MAX_FACTOR_FOUR_MAPPED_VALUES: usize = 96;
const MAX_FACTOR_FOUR_REGISTER_CANDIDATES: usize = 96;
const MAX_FACTOR_FOUR_CODE_GROWTH: usize = 128;
const MAX_FACTOR_FOUR_GROWTH_MULTIPLIER: usize = 5;
const MAX_FACTOR_FOUR_PEAK_LIVE_VALUES: usize = 20;
const MAX_FACTOR_FOUR_CSE_WORK: usize = 4_194_304;
const MAX_CONDITIONAL_INNER_INSTRUCTIONS: usize = 24;
const MAX_CONDITIONAL_POINTER_VALUES: usize = 10;
const MAX_CONDITIONAL_MAPPED_VALUES: usize = 64;
const MAX_CONDITIONAL_REGISTER_CANDIDATES: usize = 64;
const MAX_CONDITIONAL_CODE_GROWTH: usize = 96;
const MAX_CONDITIONAL_GROWTH_MULTIPLIER: usize = 4;
const MAX_CONDITIONAL_PEAK_LIVE_VALUES: usize = 20;
const MAX_CONDITIONAL_FACTOR_FOUR_POINTER_VALUES: usize = 20;
const MAX_CONDITIONAL_FACTOR_FOUR_PEAK_LIVE_VALUES: usize = 32;
const MAX_FUNCTION_BLOCKS: usize = 1024;
const MAX_FUNCTION_VALUES: usize = 8192;
const MAX_MEMORY_GEP_CHAIN_DEPTH: usize = 64;
const MAX_MEMORY_GEP_INDICES: usize = 128;
const MAX_MEMORY_TYPE_NODES: usize = 1024;
const MAX_MEMORY_PROOF_WORK: usize = 4096;
const MAX_FUNCTION_INSTRUCTIONS: usize = 65_536;

/// Processes adjacent outer iterations in one scalar reduction.
///
/// The established path accepts a canonical two-level loop whose inner loop is
/// a single pure reduction block and selects two or four lanes. RISC targets
/// may additionally select a conservative conditional reduction whose
/// original diamond and branch-only work are cloned per lane. In both cases the
/// only outer-iteration side effect is one store and the complete original loop
/// handles every dynamic tail. Memory independence is proved from global-object
/// identity and a checked typed byte-congruence model of complete GEP chains,
/// rather than from source names or fixed dimensions.
pub(super) struct ReductionJamPass {
    max_factor: usize,
}

impl ReductionJamPass {
    pub(super) fn new(max_factor: usize) -> Self {
        debug_assert!(matches!(max_factor, 2 | 4));
        Self { max_factor }
    }
}

impl ModulePass for ReductionJamPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            jam_one_nest_with_factor(func, self.max_factor);
        }
    }
}

#[derive(Clone)]
struct JamCandidate {
    outer_header: BlockId,
    outer_preheader: BlockId,
    outer_latch: BlockId,
    outer_induction: InductionVariable,
    outer_initial: i32,
    outer_bound: ValueId,
    outer_trip_count: Option<u32>,
    inner_header: BlockId,
    inner_preheader: BlockId,
    inner_body: BlockId,
    inner_exit: BlockId,
    inner_induction: InductionVariable,
    inner_bound: ValueId,
    accumulator: ValueId,
    accumulator_initial: ValueId,
    accumulator_next: ValueId,
    store_ptr: ValueId,
}

#[derive(Clone)]
struct ConditionalJamCandidate {
    outer_header: BlockId,
    outer_preheader: BlockId,
    outer_induction: InductionVariable,
    outer_bound: ValueId,
    inner_header: BlockId,
    inner_preheader: BlockId,
    condition: BlockId,
    update: BlockId,
    merge: BlockId,
    inner_exit: BlockId,
    inner_induction: InductionVariable,
    inner_bound: ValueId,
    accumulator: ValueId,
    accumulator_initial: ValueId,
    accumulator_next: ValueId,
    store_ptr: ValueId,
}

fn jam_one_nest_with_factor(func: &mut Function, max_factor: usize) {
    if func.has_reduction_jam()
        || func.blocks.len() > MAX_FUNCTION_BLOCKS
        || func.values.len() > MAX_FUNCTION_VALUES
        || func
            .blocks
            .iter()
            .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
            .is_none_or(|instructions| instructions > MAX_FUNCTION_INSTRUCTIONS)
    {
        return;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom).loops().to_vec();
    // Conditional jamming is enabled only by target cost profiles that accept
    // the wider factor-four pressure budget. Other targets retain factor two.
    if max_factor >= 4 {
        if let Some((candidate, factor)) =
            find_conditional_candidate(func, &loops, &dom, max_factor)
        {
            apply_conditional_candidate(func, &candidate, factor);
            verify_jammed_function(func);
            return;
        }
    }
    let Some(candidate) = find_candidate(func, &loops, &dom) else {
        return;
    };
    let Some(factor) = select_jam_factor(func, &candidate, max_factor) else {
        return;
    };
    apply_candidate(func, &candidate, factor);
    verify_jammed_function(func);
}

fn verify_jammed_function(func: &Function) {
    if let Err(errors) = func.verify() {
        panic!(
            "reduction unroll-and-jam produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn find_candidate(
    func: &Function,
    loops: &[NaturalLoop],
    dom: &Dominators,
) -> Option<JamCandidate> {
    let mut inner_loops = loops.iter().collect::<Vec<_>>();
    inner_loops.sort_by_key(|natural_loop| natural_loop.blocks.len());
    for inner in inner_loops {
        let Some(inner_body) = inner.unique_latch() else {
            continue;
        };
        let (Some(inner_preheader), Some(inner_exit)) =
            (inner.dedicated_preheader, inner.unique_exit())
        else {
            continue;
        };
        if inner.blocks != HashSet::from([inner.header, inner_body])
            || !matches!(
                func.blocks[inner_body.0].terminator,
                Some(Terminator::Jump(target)) if target == inner.header
            )
            || !matches!(
                func.blocks[inner_preheader.0].terminator,
                Some(Terminator::Jump(target)) if target == inner.header
            )
            || executable_instruction_count(func, inner_preheader) > MAX_SETUP_INSTRUCTIONS
            || executable_instruction_count(func, inner_body) > MAX_INNER_INSTRUCTIONS
            || !has_only_phi_and_branch_condition(func, inner.header)
            || !is_pure_setup_block(func, inner_preheader)
            || !is_pure_reduction_block(func, inner_body)
        {
            continue;
        }

        let inner_phis = phi_results(func, inner.header);
        if inner_phis.len() != 2 {
            continue;
        }
        let Some(inner_induction) = inner_phis.iter().find_map(|phi| {
            analyze_i32_induction(func, inner, *phi).filter(|induction| induction.step == 1)
        }) else {
            continue;
        };
        let Some(accumulator) = inner_phis
            .iter()
            .copied()
            .find(|phi| *phi != inner_induction.phi && func.value(*phi).ty == Type::I32)
        else {
            continue;
        };
        let Some((accumulator_initial, accumulator_next)) =
            phi_pair(func, accumulator, inner_preheader, inner_body)
        else {
            continue;
        };
        let Some(inner_bound) = canonical_less_than_bound(func, inner, inner_induction.phi) else {
            continue;
        };

        let Some(outer) = loops
            .iter()
            .filter(|outer| {
                outer.blocks.len() > inner.blocks.len()
                    && outer.blocks.contains(&inner.header)
                    && outer.blocks.contains(&inner_body)
                    && outer.blocks.contains(&inner_preheader)
                    && outer.blocks.contains(&inner_exit)
            })
            .min_by_key(|outer| outer.blocks.len())
        else {
            continue;
        };
        let (Some(outer_preheader), Some(outer_latch)) =
            (outer.dedicated_preheader, outer.unique_latch())
        else {
            continue;
        };
        if outer_latch != inner_exit
            || outer.blocks
                != HashSet::from([
                    outer.header,
                    inner_preheader,
                    inner.header,
                    inner_body,
                    inner_exit,
                ])
            || !matches!(
                func.blocks[outer_preheader.0].terminator,
                Some(Terminator::Jump(target)) if target == outer.header
            )
            || !matches!(
                func.blocks[inner_exit.0].terminator,
                Some(Terminator::Jump(target)) if target == outer.header
            )
            || !has_only_phi_and_branch_condition(func, outer.header)
        {
            continue;
        }
        let outer_phis = phi_results(func, outer.header);
        if outer_phis.len() != 1 {
            continue;
        }
        let Some(outer_induction) = analyze_i32_induction(func, outer, outer_phis[0])
            .filter(|induction| induction.step == 1)
        else {
            continue;
        };
        let Some(outer_initial) = const_i32(func, outer_induction.initial) else {
            continue;
        };
        if !(0..i32::MAX).contains(&outer_initial) || outer_initial & 1 != 0 {
            continue;
        }
        let Some(outer_bound) = canonical_less_than_bound(func, outer, outer_induction.phi) else {
            continue;
        };
        let outer_trip_count = analyze_const_i32_trip_count(func, outer, outer_induction);
        if !value_available_at(func, dom, outer_bound, outer_preheader)
            || !value_available_at(func, dom, inner_bound, outer_preheader)
            || !value_available_at(func, dom, inner_induction.initial, outer_preheader)
            || !value_mappable_from_blocks(
                func,
                dom,
                accumulator_initial,
                outer_preheader,
                outer_induction.phi,
                &[inner_preheader],
            )
        {
            continue;
        }
        let Some(inside_target) = loop_inside_target(func, outer) else {
            continue;
        };
        if inside_target != inner_preheader {
            continue;
        }

        let stores = func.blocks[inner_exit.0]
            .insts
            .iter()
            .filter_map(|inst| match inst.kind {
                InstKind::Store { ptr, value } => Some((ptr, value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [(store_ptr, store_value)] = stores.as_slice() else {
            continue;
        };
        if *store_value != accumulator
            || !has_only_one_store_side_effect(func, inner_exit)
            || !value_mappable_from_blocks(
                func,
                dom,
                *store_ptr,
                outer_preheader,
                outer_induction.phi,
                &[inner_preheader, inner_exit],
            )
            || !proves_lane_independence(
                func,
                *store_ptr,
                *store_value,
                outer_induction.phi,
                inner_body,
                2,
            )
        {
            continue;
        }
        let setup_results = func.blocks[inner_preheader.0]
            .insts
            .iter()
            .filter_map(|inst| inst.result)
            .collect::<HashSet<_>>();
        let setup_mapped = HashSet::from([outer_induction.phi]);
        let mut loop_mapped = setup_results.clone();
        loop_mapped.extend([outer_induction.phi, inner_induction.phi, accumulator]);
        if !block_operands_cloneable(
            func,
            dom,
            inner_preheader,
            outer_preheader,
            &setup_mapped,
            false,
        ) || !block_operands_cloneable(
            func,
            dom,
            inner_body,
            outer_preheader,
            &loop_mapped,
            false,
        ) || !block_operands_cloneable(
            func,
            dom,
            inner_exit,
            outer_preheader,
            &loop_mapped,
            true,
        ) {
            continue;
        }

        return Some(JamCandidate {
            outer_header: outer.header,
            outer_preheader,
            outer_latch,
            outer_induction,
            outer_initial,
            outer_bound,
            outer_trip_count,
            inner_header: inner.header,
            inner_preheader,
            inner_body,
            inner_exit,
            inner_induction,
            inner_bound,
            accumulator,
            accumulator_initial,
            accumulator_next,
            store_ptr: *store_ptr,
        });
    }
    None
}

fn find_conditional_candidate(
    func: &Function,
    loops: &[NaturalLoop],
    dom: &Dominators,
    max_factor: usize,
) -> Option<(ConditionalJamCandidate, usize)> {
    let mut inner_loops = loops.iter().collect::<Vec<_>>();
    inner_loops.sort_by_key(|natural_loop| natural_loop.blocks.len());
    for inner in inner_loops {
        let Some(merge) = inner.unique_latch() else {
            continue;
        };
        let (Some(inner_preheader), Some(inner_exit)) =
            (inner.dedicated_preheader, inner.unique_exit())
        else {
            continue;
        };
        if !matches!(
            func.blocks[merge.0].terminator,
            Some(Terminator::Jump(target)) if target == inner.header
        ) || !matches!(
            func.blocks[inner_preheader.0].terminator,
            Some(Terminator::Jump(target)) if target == inner.header
        ) || !has_only_phi_and_branch_condition(func, inner.header)
            || !is_pure_setup_block(func, inner_preheader)
            || executable_instruction_count(func, inner_preheader) > MAX_SETUP_INSTRUCTIONS
        {
            continue;
        }
        let Some(condition) = loop_inside_target(func, inner) else {
            continue;
        };
        let Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) = func.blocks[condition.0].terminator.as_ref()
        else {
            continue;
        };
        let update = match (*then_target == merge, *else_target == merge) {
            (true, false) => *else_target,
            (false, true) => *then_target,
            _ => continue,
        };
        if inner.blocks != HashSet::from([inner.header, condition, update, merge])
            || !matches!(
                func.blocks[update.0].terminator,
                Some(Terminator::Jump(target)) if target == merge
            )
            || !is_pure_reduction_block(func, condition)
            || !is_pure_reduction_block(func, update)
            || executable_instruction_count(func, condition)
                .checked_add(executable_instruction_count(func, update))
                .is_none_or(|instructions| instructions > MAX_CONDITIONAL_INNER_INSTRUCTIONS)
        {
            continue;
        }
        let inner_phis = phi_results(func, inner.header);
        if inner_phis.len() != 2 {
            continue;
        }
        let Some(inner_induction) = inner_phis.iter().find_map(|phi| {
            analyze_i32_induction(func, inner, *phi).filter(|induction| induction.step == 1)
        }) else {
            continue;
        };
        let Some(accumulator) = inner_phis
            .iter()
            .copied()
            .find(|phi| *phi != inner_induction.phi && func.value(*phi).ty == Type::I32)
        else {
            continue;
        };
        let Some((accumulator_initial, accumulator_next)) =
            phi_pair(func, accumulator, inner_preheader, merge)
        else {
            continue;
        };
        let merge_phis = phi_results(func, merge);
        if merge_phis.as_slice() != [accumulator_next] {
            continue;
        }
        let Some((old_accumulator, updated_accumulator)) =
            phi_pair(func, accumulator_next, condition, update)
        else {
            continue;
        };
        if old_accumulator != accumulator
            || updated_accumulator == accumulator
            || !matches!(
                func.values.get(updated_accumulator.0).map(|value| &value.kind),
                Some(ValueKind::Inst(owner, _)) if *owner == update
            )
            || func
                .values
                .get(updated_accumulator.0)
                .map(|value| &value.ty)
                != Some(&Type::I32)
            || !has_exact_conditional_merge(func, merge, accumulator_next, inner_induction.next)
        {
            continue;
        }
        let Some(inner_bound) = canonical_less_than_bound(func, inner, inner_induction.phi) else {
            continue;
        };
        let Some(outer) = loops
            .iter()
            .filter(|outer| {
                outer.blocks.len() > inner.blocks.len()
                    && inner
                        .blocks
                        .iter()
                        .all(|block| outer.blocks.contains(block))
                    && outer.blocks.contains(&inner_preheader)
                    && outer.blocks.contains(&inner_exit)
            })
            .min_by_key(|outer| outer.blocks.len())
        else {
            continue;
        };
        let (Some(outer_preheader), Some(outer_latch)) =
            (outer.dedicated_preheader, outer.unique_latch())
        else {
            continue;
        };
        if outer_latch != inner_exit
            || outer.blocks
                != HashSet::from([
                    outer.header,
                    inner_preheader,
                    inner.header,
                    condition,
                    update,
                    merge,
                    inner_exit,
                ])
            || !matches!(
                func.blocks[outer_preheader.0].terminator,
                Some(Terminator::Jump(target)) if target == outer.header
            )
            || !matches!(
                func.blocks[inner_exit.0].terminator,
                Some(Terminator::Jump(target)) if target == outer.header
            )
            || !has_only_phi_and_branch_condition(func, outer.header)
        {
            continue;
        }
        let outer_phis = phi_results(func, outer.header);
        if outer_phis.len() != 1 {
            continue;
        }
        let Some(outer_induction) = analyze_i32_induction(func, outer, outer_phis[0])
            .filter(|induction| induction.step == 1)
        else {
            continue;
        };
        let Some(outer_initial) = const_i32(func, outer_induction.initial) else {
            continue;
        };
        if !(0..i32::MAX).contains(&outer_initial) || outer_initial & 1 != 0 {
            continue;
        }
        let Some(outer_bound) = canonical_less_than_bound(func, outer, outer_induction.phi) else {
            continue;
        };
        if analyze_const_i32_trip_count(func, outer, outer_induction).is_some_and(|count| count < 2)
            || !value_available_at(func, dom, outer_bound, outer_preheader)
            || !value_available_at(func, dom, inner_bound, outer_preheader)
            || !value_available_at(func, dom, inner_induction.initial, outer_preheader)
            || !value_mappable_from_blocks(
                func,
                dom,
                accumulator_initial,
                outer_preheader,
                outer_induction.phi,
                &[inner_preheader],
            )
            || loop_inside_target(func, outer) != Some(inner_preheader)
            || !inner_results_have_no_extra_liveouts(func, inner, accumulator, inner_exit)
        {
            continue;
        }

        let stores = func.blocks[inner_exit.0]
            .insts
            .iter()
            .filter_map(|inst| match inst.kind {
                InstKind::Store { ptr, value } => Some((ptr, value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [(store_ptr, store_value)] = stores.as_slice() else {
            continue;
        };
        if *store_value != accumulator
            || !has_only_one_store_side_effect(func, inner_exit)
            || !value_mappable_from_blocks(
                func,
                dom,
                *store_ptr,
                outer_preheader,
                outer_induction.phi,
                &[inner_preheader, inner_exit],
            )
            || !proves_lane_independence_in_blocks(
                func,
                *store_ptr,
                *store_value,
                outer_induction.phi,
                &[condition, update],
                2,
            )
        {
            continue;
        }

        let setup_results = block_results(func, inner_preheader);
        let setup_mapped = HashSet::from([outer_induction.phi]);
        let mut loop_mapped = setup_results.clone();
        loop_mapped.extend([outer_induction.phi, inner_induction.phi, accumulator]);
        let condition_results = block_results(func, condition);
        let mut update_mapped = loop_mapped.clone();
        update_mapped.extend(condition_results.iter().copied());
        if !block_operands_cloneable(
            func,
            dom,
            inner_preheader,
            outer_preheader,
            &setup_mapped,
            false,
        ) || !block_operands_cloneable(
            func,
            dom,
            condition,
            outer_preheader,
            &loop_mapped,
            false,
        ) || !branch_condition_cloneable(func, dom, condition, outer_preheader, &update_mapped)
            || !block_operands_cloneable(func, dom, update, outer_preheader, &update_mapped, false)
            || !block_operands_cloneable(func, dom, inner_exit, outer_preheader, &loop_mapped, true)
        {
            continue;
        }

        let candidate = ConditionalJamCandidate {
            outer_header: outer.header,
            outer_preheader,
            outer_induction,
            outer_bound,
            inner_header: inner.header,
            inner_preheader,
            condition,
            update,
            merge,
            inner_exit,
            inner_induction,
            inner_bound,
            accumulator,
            accumulator_initial,
            accumulator_next,
            store_ptr: *store_ptr,
        };
        let factor = select_conditional_jam_factor(func, &candidate, outer_initial, max_factor)?;
        if conditional_factor_is_profitable(func, &candidate, factor) {
            return Some((candidate, factor));
        }
    }
    None
}

fn select_jam_factor(
    func: &Function,
    candidate: &JamCandidate,
    max_factor: usize,
) -> Option<usize> {
    // A value congruent to zero modulo four is at most i32::MAX - 3. Every
    // fast header can therefore form lanes 1..3 without signed wrap. If the
    // lane-three guard succeeds, `iv + 3 < bound <= i32::MAX` also proves that
    // the taken-path `iv + 4` is representable. The same argument with modulo
    // two is established by candidate matching for the fallback path.
    if candidate.outer_trip_count.is_some_and(|count| count < 2) {
        return None;
    }
    let factor_two_cost = estimate_jam_cost(func, candidate, 2)?;
    if !factor_is_within_hard_budgets(func, &factor_two_cost, 2, 5) {
        return None;
    }
    if max_factor < 4
        || candidate.outer_trip_count.is_some_and(|count| count < 4)
        || candidate.outer_initial & 3 != 0
        || !factor_four_is_profitable(func, candidate)
    {
        return Some(2);
    }
    if !proves_lane_independence(
        func,
        candidate.store_ptr,
        candidate.accumulator,
        candidate.outer_induction.phi,
        candidate.inner_body,
        4,
    ) {
        return Some(2);
    }
    Some(4)
}

#[derive(Debug)]
struct JamCost {
    active_accumulators: usize,
    pointer_values: usize,
    shared_load_streams: usize,
    setup_live_scalars: usize,
    mapped_values: usize,
    register_candidates: usize,
    projected_cse_keys: usize,
    projected_cse_operands: usize,
    peak_live_values: usize,
    code_growth: usize,
    original_region: usize,
}

/// Bounds factor-four pressure using only target-independent SSA properties.
/// Pointer state includes every output lane and distinct load stream, with
/// lane-dependent streams replicated per lane. `mapped_values` bounds all
/// per-lane map entries, while `register_candidates` bounds result-bearing
/// fast-path IR.
fn factor_four_is_profitable(func: &Function, candidate: &JamCandidate) -> bool {
    let Some(cost) = estimate_jam_cost(func, candidate, 4) else {
        return false;
    };
    let Some(relative_growth_budget) = cost
        .original_region
        .checked_mul(MAX_FACTOR_FOUR_GROWTH_MULTIPLIER)
    else {
        return false;
    };
    factor_is_within_hard_budgets(func, &cost, 4, 5)
        && cost.shared_load_streams > 0
        && cost.active_accumulators <= MAX_FACTOR_FOUR_ACTIVE_ACCUMULATORS
        && cost.pointer_values <= MAX_FACTOR_FOUR_POINTER_VALUES
        && cost.mapped_values <= MAX_FACTOR_FOUR_MAPPED_VALUES
        && cost.register_candidates <= MAX_FACTOR_FOUR_REGISTER_CANDIDATES
        && cost.peak_live_values <= MAX_FACTOR_FOUR_PEAK_LIVE_VALUES
        && cse_projection_within_budget(func, &cost)
        && func
            .values
            .len()
            .checked_add(cost.register_candidates)
            .and_then(|values| values.checked_add(4))
            .is_some_and(|values| {
                values <= 512
                    && func
                        .blocks
                        .len()
                        .checked_add(5)
                        .and_then(|blocks| blocks.checked_mul(values))
                        .is_some_and(|work| work <= 262_144)
            })
        && cost.code_growth <= MAX_FACTOR_FOUR_CODE_GROWTH
        && cost.code_growth <= relative_growth_budget
}

fn projected_cse_key_operands(func: &Function, inst: &crate::ir::Inst) -> Option<Option<usize>> {
    let operands = match &inst.kind {
        InstKind::Unary { .. } | InstKind::Cast { .. } => 1usize,
        InstKind::Binary { .. } | InstKind::Icmp { .. } | InstKind::Fcmp { .. } => 2,
        InstKind::Gep { indices, .. } => {
            let result = inst.result?;
            indices
                .len()
                .saturating_add(1)
                .saturating_add(bounded_type_nodes(&func.values.get(result.0)?.ty)?)
        }
        InstKind::Call { args, .. } if inst.result.is_some() => args.len().saturating_add(1),
        _ => return Some(None),
    };
    Some(Some(operands))
}

fn cse_projection_within_budget(func: &Function, cost: &JamCost) -> bool {
    let mut keys = 0usize;
    let mut operands = 0usize;
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let Some(key_operands) = (match &inst.kind {
            InstKind::Unary { .. } | InstKind::Cast { .. } => Some(1usize),
            InstKind::Binary { .. } | InstKind::Icmp { .. } | InstKind::Fcmp { .. } => Some(2),
            InstKind::Gep { indices, .. } => {
                let Some(result_ty) = inst
                    .result
                    .and_then(|result| func.values.get(result.0))
                    .map(|value| &value.ty)
                else {
                    return false;
                };
                let Some(type_nodes) = bounded_type_nodes(result_ty) else {
                    return false;
                };
                Some(indices.len().saturating_add(1).saturating_add(type_nodes))
            }
            InstKind::Call { args, .. } if inst.result.is_some() => {
                Some(args.len().saturating_add(1))
            }
            _ => None,
        }) else {
            continue;
        };
        keys = keys.saturating_add(1);
        operands = operands.saturating_add(key_operands.saturating_add(1));
    }
    keys = keys.saturating_add(cost.projected_cse_keys);
    operands = operands.saturating_add(cost.projected_cse_operands);
    keys <= 512
        && operands <= 4096
        && keys
            .checked_mul(operands)
            .is_some_and(|work| work <= MAX_FACTOR_FOUR_CSE_WORK)
}

fn bounded_type_nodes(mut ty: &Type) -> Option<usize> {
    let mut nodes = 0usize;
    loop {
        nodes = nodes.checked_add(1)?;
        if nodes > 128 {
            return None;
        }
        match ty {
            Type::Ptr(inner) => ty = inner,
            Type::Array { elem, .. } => ty = elem,
            Type::Void | Type::I1 | Type::I32 | Type::F32 => return Some(nodes),
        }
    }
}

fn factor_is_within_hard_budgets(
    func: &Function,
    cost: &JamCost,
    factor: usize,
    added_blocks: usize,
) -> bool {
    let Some(function_instructions) = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
    else {
        return false;
    };
    func.blocks
        .len()
        .checked_add(added_blocks)
        .is_some_and(|blocks| blocks <= MAX_FUNCTION_BLOCKS)
        && func
            .values
            .len()
            .checked_add(cost.register_candidates)
            .and_then(|values| values.checked_add(factor))
            .is_some_and(|values| values <= MAX_FUNCTION_VALUES)
        && function_instructions
            .checked_add(cost.code_growth)
            .is_some_and(|instructions| instructions <= MAX_FUNCTION_INSTRUCTIONS)
}

fn estimate_jam_cost(func: &Function, candidate: &JamCandidate, factor: usize) -> Option<JamCost> {
    let mut cloned_instructions = 0usize;
    let mut cloned_results = 0usize;
    let mut setup_live_scalars = 0usize;
    let mut projected_cse_keys = 0usize;
    let mut projected_cse_operands = 0usize;
    let mut external_invariant_scalars = HashSet::new();
    let candidate_blocks = HashSet::from([
        candidate.outer_header,
        candidate.inner_preheader,
        candidate.inner_header,
        candidate.inner_body,
        candidate.inner_exit,
    ]);
    let mut lane_dependent = HashSet::from([candidate.outer_induction.phi, candidate.accumulator]);
    let mut load_pointers = HashSet::new();
    for (block, is_exit) in [
        (candidate.inner_preheader, false),
        (candidate.inner_body, false),
        (candidate.inner_exit, true),
    ] {
        for inst in &func.blocks.get(block.0)?.insts {
            if block == candidate.inner_body {
                for operand in instruction_operands(&inst.kind) {
                    let value = func.values.get(operand.0)?;
                    let external = matches!(value.kind, ValueKind::Param)
                        || matches!(value.kind, ValueKind::Inst(owner, _) if !candidate_blocks.contains(&owner));
                    if external && matches!(value.ty, Type::I1 | Type::I32 | Type::F32) {
                        external_invariant_scalars.insert(operand);
                    }
                }
            }
            if let Some(key_operands) = projected_cse_key_operands(func, inst)? {
                projected_cse_keys = projected_cse_keys.checked_add(factor)?;
                projected_cse_operands = projected_cse_operands
                    .checked_add(factor.checked_mul(key_operands.saturating_add(1))?)?;
            }
            if let InstKind::Load { ptr } = inst.kind {
                load_pointers.insert(ptr);
            }
            if let Some(result) = inst.result {
                if block == candidate.inner_preheader
                    && !matches!(func.values.get(result.0)?.ty, Type::Ptr(_))
                {
                    setup_live_scalars = setup_live_scalars.checked_add(1)?;
                }
                if instruction_operands(&inst.kind)
                    .iter()
                    .any(|operand| lane_dependent.contains(operand))
                {
                    lane_dependent.insert(result);
                }
            }
            if matches!(inst.kind, InstKind::Nop)
                || (is_exit && matches!(inst.kind, InstKind::Store { .. }))
            {
                continue;
            }
            cloned_instructions = cloned_instructions.checked_add(1)?;
            if inst.result.is_some() {
                cloned_results = cloned_results.checked_add(1)?;
            }
        }
    }
    let mut shared_load_streams = 0usize;
    let load_pointer_values = load_pointers.into_iter().try_fold(0usize, |total, ptr| {
        let copies = if lane_dependent.contains(&ptr) {
            factor
        } else {
            shared_load_streams = shared_load_streams.checked_add(1)?;
            1
        };
        total.checked_add(copies)
    })?;

    let original_blocks = HashSet::from([
        candidate.outer_header,
        candidate.inner_preheader,
        candidate.inner_header,
        candidate.inner_body,
        candidate.outer_latch,
    ]);
    let original_region = original_blocks
        .into_iter()
        .try_fold(0usize, |total, block| {
            total.checked_add(executable_instruction_count(func, block))
        })?;
    let generated_keys = factor.checked_add(2)?;
    projected_cse_keys = projected_cse_keys.checked_add(generated_keys)?;
    projected_cse_operands = projected_cse_operands.checked_add(generated_keys.checked_mul(3)?)?;
    let fixed_results = factor.checked_mul(2)?.checked_add(4)?;
    Some(JamCost {
        active_accumulators: factor,
        // Keep one state value per output lane, one shared object base, and one
        // load pointer stream per lane-independent address (or per lane when
        // the address depends on a lane's outer index or accumulator). This
        // anticipates the
        // later generic GEP-induction pass without querying either backend.
        pointer_values: factor.checked_add(1)?.checked_add(load_pointer_values)?,
        shared_load_streams,
        setup_live_scalars,
        // Each lane map contains the outer induction, the shared inner
        // induction, the accumulator, and every cloned result.
        mapped_values: factor.checked_mul(cloned_results.checked_add(3)?)?,
        register_candidates: factor
            .checked_mul(cloned_results)?
            .checked_add(fixed_results)?,
        projected_cse_keys,
        projected_cse_operands,
        // Four accumulators, pointer state, and the shared outer induction,
        // inner induction, and dynamic bound are simultaneously live around
        // the hot inner loop. Local single-block temporaries remain available
        // to the backend's separate t-register allocator.
        peak_live_values: factor
            .checked_add(factor.checked_add(1)?.checked_add(load_pointer_values)?)?
            .checked_add(factor.checked_mul(setup_live_scalars)?)?
            // outer/inner induction, distinct outer/inner bounds, and each
            // shared load result that remains live while updating all lanes.
            .checked_add(4usize.checked_add(shared_load_streams)?)?
            .checked_add(external_invariant_scalars.len())?,
        // Header indices/guard, inner phis/guard, stores, and the outer next
        // account for `3 * factor + 4` instructions beyond cloned source IR.
        code_growth: factor
            .checked_mul(cloned_instructions)?
            .checked_add(factor.checked_mul(3)?)?
            .checked_add(4)?,
        original_region,
    })
}

fn select_conditional_jam_factor(
    func: &Function,
    candidate: &ConditionalJamCandidate,
    outer_initial: i32,
    max_factor: usize,
) -> Option<usize> {
    let aligned = outer_initial & 3 == 0;
    let independent = proves_lane_independence_in_blocks(
        func,
        candidate.store_ptr,
        candidate.accumulator,
        candidate.outer_induction.phi,
        &[candidate.condition, candidate.update],
        4,
    );
    let profitable = conditional_factor_is_profitable(func, candidate, 4);
    if max_factor >= 4 && aligned && independent && profitable {
        Some(4)
    } else {
        Some(2)
    }
}

fn conditional_factor_is_profitable(
    func: &Function,
    candidate: &ConditionalJamCandidate,
    factor: usize,
) -> bool {
    let Some(cost) = estimate_conditional_jam_cost(func, candidate, factor) else {
        return false;
    };
    let added_blocks = factor.saturating_mul(3).saturating_add(4);
    let (
        max_pointer_values,
        max_mapped_values,
        max_register_candidates,
        max_peak_live_values,
        max_code_growth,
    ) = if factor == 4 {
        (
            MAX_CONDITIONAL_FACTOR_FOUR_POINTER_VALUES,
            MAX_FACTOR_FOUR_MAPPED_VALUES,
            MAX_FACTOR_FOUR_REGISTER_CANDIDATES,
            MAX_CONDITIONAL_FACTOR_FOUR_PEAK_LIVE_VALUES,
            MAX_FACTOR_FOUR_CODE_GROWTH,
        )
    } else {
        (
            MAX_CONDITIONAL_POINTER_VALUES,
            MAX_CONDITIONAL_MAPPED_VALUES,
            MAX_CONDITIONAL_REGISTER_CANDIDATES,
            MAX_CONDITIONAL_PEAK_LIVE_VALUES,
            MAX_CONDITIONAL_CODE_GROWTH,
        )
    };
    let Some(relative_growth_budget) = cost.original_region.checked_mul(if factor == 4 {
        MAX_FACTOR_FOUR_GROWTH_MULTIPLIER
    } else {
        MAX_CONDITIONAL_GROWTH_MULTIPLIER
    }) else {
        return false;
    };
    factor_is_within_hard_budgets(func, &cost, factor, added_blocks)
        && cost.active_accumulators == factor
        && (factor == 2 || cost.shared_load_streams > 0)
        && cost.pointer_values <= max_pointer_values
        && cost.mapped_values <= max_mapped_values
        && cost.register_candidates <= max_register_candidates
        && cost.peak_live_values <= max_peak_live_values
        && cse_projection_within_budget(func, &cost)
        && func
            .values
            .len()
            .checked_add(cost.register_candidates)
            .and_then(|values| values.checked_add(2))
            .is_some_and(|values| {
                values <= 512
                    && func
                        .blocks
                        .len()
                        .checked_add(added_blocks)
                        .and_then(|blocks| blocks.checked_mul(values))
                        .is_some_and(|work| work <= 262_144)
            })
        && cost.code_growth <= max_code_growth
        && cost.code_growth <= relative_growth_budget
}

fn estimate_conditional_jam_cost(
    func: &Function,
    candidate: &ConditionalJamCandidate,
    factor: usize,
) -> Option<JamCost> {
    if !matches!(factor, 2 | 4) {
        return None;
    }
    let candidate_blocks = HashSet::from([
        candidate.outer_header,
        candidate.inner_preheader,
        candidate.inner_header,
        candidate.condition,
        candidate.update,
        candidate.merge,
        candidate.inner_exit,
    ]);
    let mut lane_dependent = HashSet::from([candidate.outer_induction.phi, candidate.accumulator]);
    let mut load_pointers = HashSet::new();
    let mut unconditionally_loaded_pointers = HashSet::new();
    let mut external_invariant_scalars = HashSet::new();
    let mut setup_live_scalars = 0usize;
    let mut condition_cross_edge_results = HashSet::new();
    let mut condition_loads = 0usize;
    let mut cloned_instructions = 0usize;
    let mut cloned_results = 0usize;
    let mut projected_cse_keys = 0usize;
    let mut projected_cse_operands = 0usize;

    for (block, is_setup, is_exit, count_external) in [
        (candidate.inner_preheader, true, false, false),
        (candidate.condition, false, false, true),
        (candidate.update, false, false, true),
        (candidate.inner_exit, false, true, false),
    ] {
        for inst in &func.blocks.get(block.0)?.insts {
            if count_external {
                for operand in instruction_operands(&inst.kind) {
                    let value = func.values.get(operand.0)?;
                    let external = matches!(value.kind, ValueKind::Param)
                        || matches!(value.kind, ValueKind::Inst(owner, _) if !candidate_blocks.contains(&owner));
                    if external && matches!(value.ty, Type::I1 | Type::I32 | Type::F32) {
                        external_invariant_scalars.insert(operand);
                    }
                }
            }
            if let Some(key_operands) = projected_cse_key_operands(func, inst)? {
                projected_cse_keys = projected_cse_keys.checked_add(factor)?;
                projected_cse_operands = projected_cse_operands
                    .checked_add(factor.checked_mul(key_operands.saturating_add(1))?)?;
            }
            if let InstKind::Load { ptr } = inst.kind {
                load_pointers.insert(ptr);
                if block == candidate.condition {
                    condition_loads = condition_loads.checked_add(1)?;
                    // Every taken inner iteration executes the condition block.
                    // Loads found only in an update arm remain lane-local: the
                    // earlier arm does not dominate the later arm, and sharing
                    // them would require forbidden speculation.
                    unconditionally_loaded_pointers.insert(ptr);
                }
            }
            if let Some(result) = inst.result {
                if block == candidate.update {
                    for operand in instruction_operands(&inst.kind) {
                        if matches!(
                            func.values.get(operand.0)?.kind,
                            ValueKind::Inst(owner, _) if owner == candidate.condition
                        ) {
                            condition_cross_edge_results.insert(operand);
                        }
                    }
                }
                if is_setup && !matches!(func.values.get(result.0)?.ty, Type::Ptr(_)) {
                    setup_live_scalars = setup_live_scalars.checked_add(1)?;
                }
                if instruction_operands(&inst.kind)
                    .iter()
                    .any(|operand| lane_dependent.contains(operand))
                {
                    lane_dependent.insert(result);
                }
            }
            if matches!(inst.kind, InstKind::Nop)
                || (is_exit && matches!(inst.kind, InstKind::Store { .. }))
            {
                continue;
            }
            cloned_instructions = cloned_instructions.checked_add(1)?;
            if inst.result.is_some() {
                cloned_results = cloned_results.checked_add(1)?;
            }
        }
    }

    let inner_next = defining_inst(func, candidate.inner_induction.next)?;
    if let Some(key_operands) = projected_cse_key_operands(
        func,
        &crate::ir::Inst {
            result: Some(candidate.inner_induction.next),
            kind: inner_next.clone(),
        },
    )? {
        projected_cse_keys = projected_cse_keys.checked_add(1)?;
        projected_cse_operands =
            projected_cse_operands.checked_add(key_operands.saturating_add(1))?;
    }

    let mut shared_load_streams = 0usize;
    let load_pointer_values = load_pointers.into_iter().try_fold(0usize, |total, ptr| {
        let copies =
            if lane_dependent.contains(&ptr) || !unconditionally_loaded_pointers.contains(&ptr) {
                factor
            } else {
                shared_load_streams = shared_load_streams.checked_add(1)?;
                1
            };
        total.checked_add(copies)
    })?;
    let original_region = candidate_blocks.iter().try_fold(0usize, |total, block| {
        total.checked_add(executable_instruction_count(func, *block))
    })?;
    let generated_keys = factor.checked_add(2)?;
    projected_cse_keys = projected_cse_keys.checked_add(generated_keys)?;
    projected_cse_operands = projected_cse_operands.checked_add(generated_keys.checked_mul(3)?)?;
    let pointer_values = factor.checked_add(1)?.checked_add(load_pointer_values)?;

    Some(JamCost {
        active_accumulators: factor,
        pointer_values,
        shared_load_streams,
        setup_live_scalars,
        // Each map contains cloned source results plus the outer index, shared
        // inner index, lane accumulator, and conditional selector.
        mapped_values: factor
            .checked_mul(cloned_results.checked_add(4)?)?
            // The lane-0 condition reuse table stores each load result and its
            // remapped pointer; the last lane also retains inner.next.
            .checked_add(condition_loads.checked_mul(2)?)?
            .checked_add(1)?,
        // `3 * factor + 5` covers both fast headers, lane selectors, the
        // shared inner next value, and the fast outer next value.
        register_candidates: factor
            .checked_mul(cloned_results)?
            .checked_add(factor.checked_mul(3)?.checked_add(5)?)?,
        projected_cse_keys,
        projected_cse_operands,
        peak_live_values: factor
            .checked_add(pointer_values)?
            .checked_add(factor.checked_mul(setup_live_scalars)?)?
            .checked_add(4usize.checked_add(shared_load_streams)?)?
            .checked_add(external_invariant_scalars.len())?
            .checked_add(factor.checked_mul(condition_cross_edge_results.len())?)?
            // Keep one conservative branch-condition temporary per lane.
            .checked_add(factor)?,
        // Generated headers/selectors/stores/next values contribute
        // `4 * factor + 5` instructions beyond cloned source blocks.
        code_growth: factor
            .checked_mul(cloned_instructions)?
            .checked_add(factor.checked_mul(4)?.checked_add(5)?)?,
        original_region,
    })
}

fn apply_candidate(func: &mut Function, candidate: &JamCandidate, factor: usize) {
    assert!(matches!(factor, 2 | 4));
    let fast_header = func.add_block("reduction.jam.header");
    let fast_setup = func.add_block("reduction.jam.setup");
    let fast_inner_header = func.add_block("reduction.jam.inner");
    let fast_inner_body = func.add_block("reduction.jam.body");
    let fast_exit = func.add_block("reduction.jam.exit");

    let fast_outer = func
        .append_inst(
            fast_header,
            InstKind::Phi {
                incomings: vec![
                    (candidate.outer_preheader, candidate.outer_induction.initial),
                    (fast_exit, candidate.outer_induction.initial),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let one = get_or_add_i32_const(func, 1);
    let factor_value = get_or_add_i32_const(func, factor as i32);
    let mut lane_indices = vec![fast_outer];
    for lane in 1..factor {
        let offset = if lane == 1 {
            one
        } else {
            get_or_add_i32_const(func, lane as i32)
        };
        let index = func
            .append_inst(
                fast_header,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: fast_outer,
                    rhs: offset,
                },
                Some(Type::I32),
            )
            .unwrap();
        lane_indices.push(index);
    }
    let group_condition = func
        .append_inst(
            fast_header,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: *lane_indices.last().unwrap(),
                rhs: candidate.outer_bound,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        fast_header,
        Terminator::Branch {
            cond: group_condition,
            then_target: fast_setup,
            else_target: candidate.outer_header,
        },
    );

    let mut lane_values = lane_indices
        .iter()
        .map(|index| HashMap::from([(candidate.outer_induction.phi, *index)]))
        .collect::<Vec<_>>();
    for values in &mut lane_values {
        clone_pure_block(func, candidate.inner_preheader, fast_setup, values);
    }

    let inner_initial = map_value(candidate.inner_induction.initial, &lane_values[0]);
    assert!(lane_values
        .iter()
        .all(|values| map_value(candidate.inner_induction.initial, values) == inner_initial));
    let fast_inner = func
        .append_inst(
            fast_inner_header,
            InstKind::Phi {
                incomings: vec![
                    (fast_setup, inner_initial),
                    (fast_inner_body, inner_initial),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let mut fast_accumulators = Vec::with_capacity(factor);
    for values in &lane_values {
        let initial = map_value(candidate.accumulator_initial, values);
        let accumulator = func
            .append_inst(
                fast_inner_header,
                InstKind::Phi {
                    incomings: vec![(fast_setup, initial), (fast_inner_body, initial)],
                },
                Some(Type::I32),
            )
            .unwrap();
        fast_accumulators.push(accumulator);
    }
    let inner_condition = func
        .append_inst(
            fast_inner_header,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: fast_inner,
                rhs: candidate.inner_bound,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        fast_inner_header,
        Terminator::Branch {
            cond: inner_condition,
            then_target: fast_inner_body,
            else_target: fast_exit,
        },
    );
    func.set_terminator(fast_setup, Terminator::Jump(fast_inner_header));

    for (values, accumulator) in lane_values.iter_mut().zip(&fast_accumulators) {
        values.insert(candidate.inner_induction.phi, fast_inner);
        values.insert(candidate.accumulator, *accumulator);
        clone_pure_block(func, candidate.inner_body, fast_inner_body, values);
    }
    let fast_inner_next = map_value(candidate.inner_induction.next, &lane_values[0]);
    set_phi_backedge(func, fast_inner, fast_inner_body, fast_inner_next);
    for (values, accumulator) in lane_values.iter().zip(&fast_accumulators) {
        set_phi_backedge(
            func,
            *accumulator,
            fast_inner_body,
            map_value(candidate.accumulator_next, values),
        );
    }
    func.set_terminator(fast_inner_body, Terminator::Jump(fast_inner_header));

    for values in &mut lane_values {
        clone_exit_pure_instructions(func, candidate.inner_exit, fast_exit, values);
    }
    // Preserve original outer-iteration store order. The legality proof covers
    // every earlier store against every load moved before it by lane jamming.
    for (values, accumulator) in lane_values.iter().zip(&fast_accumulators) {
        func.append_inst(
            fast_exit,
            InstKind::Store {
                ptr: map_value(candidate.store_ptr, values),
                value: *accumulator,
            },
            None,
        );
    }
    let fast_outer_next = func
        .append_inst(
            fast_exit,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: fast_outer,
                rhs: factor_value,
            },
            Some(Type::I32),
        )
        .unwrap();
    set_phi_backedge(func, fast_outer, fast_exit, fast_outer_next);
    func.set_terminator(fast_exit, Terminator::Jump(fast_header));

    func.blocks[candidate.outer_preheader.0].terminator = Some(Terminator::Jump(fast_header));
    replace_phi_incoming(
        func,
        candidate.outer_induction.phi,
        candidate.outer_preheader,
        fast_header,
        fast_outer,
    );
    func.mark_reduction_jammed();
}

fn apply_conditional_candidate(
    func: &mut Function,
    candidate: &ConditionalJamCandidate,
    factor: usize,
) {
    assert!(matches!(factor, 2 | 4));
    let fast_header = func.add_block("conditional.reduction.jam.header");
    let fast_setup = func.add_block("conditional.reduction.jam.setup");
    let fast_inner_header = func.add_block("conditional.reduction.jam.inner");
    let lane_blocks = (0..factor)
        .map(|lane| {
            (
                func.add_block(format!("conditional.reduction.jam.condition.{lane}")),
                func.add_block(format!("conditional.reduction.jam.update.{lane}")),
                func.add_block(format!("conditional.reduction.jam.merge.{lane}")),
            )
        })
        .collect::<Vec<_>>();
    let fast_exit = func.add_block("conditional.reduction.jam.exit");
    let last_merge = lane_blocks.last().unwrap().2;

    let fast_outer = func
        .append_inst(
            fast_header,
            InstKind::Phi {
                incomings: vec![
                    (candidate.outer_preheader, candidate.outer_induction.initial),
                    (fast_exit, candidate.outer_induction.initial),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let one = get_or_add_i32_const(func, 1);
    let factor_value = get_or_add_i32_const(func, factor as i32);
    let mut lane_indices = vec![fast_outer];
    for lane in 1..factor {
        let offset = if lane == 1 {
            one
        } else {
            get_or_add_i32_const(func, lane as i32)
        };
        lane_indices.push(
            func.append_inst(
                fast_header,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: fast_outer,
                    rhs: offset,
                },
                Some(Type::I32),
            )
            .unwrap(),
        );
    }
    let group_condition = func
        .append_inst(
            fast_header,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: *lane_indices.last().unwrap(),
                rhs: candidate.outer_bound,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        fast_header,
        Terminator::Branch {
            cond: group_condition,
            then_target: fast_setup,
            else_target: candidate.outer_header,
        },
    );

    let mut lane_values = lane_indices
        .into_iter()
        .map(|index| HashMap::from([(candidate.outer_induction.phi, index)]))
        .collect::<Vec<_>>();
    for values in &mut lane_values {
        clone_pure_block(func, candidate.inner_preheader, fast_setup, values);
    }
    func.set_terminator(fast_setup, Terminator::Jump(fast_inner_header));

    let inner_initial = map_value(candidate.inner_induction.initial, &lane_values[0]);
    assert!(lane_values
        .iter()
        .all(|values| map_value(candidate.inner_induction.initial, values) == inner_initial));
    let fast_inner = func
        .append_inst(
            fast_inner_header,
            InstKind::Phi {
                incomings: vec![(fast_setup, inner_initial), (last_merge, inner_initial)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let mut fast_accumulators = Vec::with_capacity(factor);
    for values in &lane_values {
        let initial = map_value(candidate.accumulator_initial, values);
        let accumulator = func
            .append_inst(
                fast_inner_header,
                InstKind::Phi {
                    incomings: vec![(fast_setup, initial), (last_merge, initial)],
                },
                Some(Type::I32),
            )
            .unwrap();
        fast_accumulators.push(accumulator);
    }
    let inner_condition = func
        .append_inst(
            fast_inner_header,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: fast_inner,
                rhs: candidate.inner_bound,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        fast_inner_header,
        Terminator::Branch {
            cond: inner_condition,
            then_target: lane_blocks[0].0,
            else_target: fast_exit,
        },
    );

    let source_branch = func.blocks[candidate.condition.0]
        .terminator
        .clone()
        .unwrap();
    let selector_incomings = match defining_inst(func, candidate.accumulator_next).unwrap() {
        InstKind::Phi { incomings } => incomings.clone(),
        _ => unreachable!(),
    };
    let inner_next_kind = defining_inst(func, candidate.inner_induction.next)
        .unwrap()
        .clone();
    let mut fast_inner_next = None;
    for lane in 0..factor {
        let (condition_block, update_block, merge_block) = lane_blocks[lane];
        let shared_condition_values = (lane > 0)
            .then(|| conditional_shared_load_values(func, candidate.condition, &lane_values[0]));
        let values = &mut lane_values[lane];
        values.insert(candidate.inner_induction.phi, fast_inner);
        values.insert(candidate.accumulator, fast_accumulators[lane]);
        clone_conditional_condition_block(
            func,
            candidate.condition,
            condition_block,
            values,
            shared_condition_values.as_ref(),
        );

        let Terminator::Branch {
            cond,
            then_target,
            else_target,
        } = &source_branch
        else {
            unreachable!();
        };
        let remap_target = |target: BlockId| {
            if target == candidate.update {
                update_block
            } else if target == candidate.merge {
                merge_block
            } else {
                unreachable!()
            }
        };
        func.set_terminator(
            condition_block,
            Terminator::Branch {
                cond: map_value(*cond, values),
                then_target: remap_target(*then_target),
                else_target: remap_target(*else_target),
            },
        );

        clone_pure_block(func, candidate.update, update_block, values);
        func.set_terminator(update_block, Terminator::Jump(merge_block));
        let incomings = selector_incomings
            .iter()
            .map(|(pred, value)| {
                let pred = if *pred == candidate.condition {
                    condition_block
                } else if *pred == candidate.update {
                    update_block
                } else {
                    unreachable!()
                };
                (pred, map_value(*value, values))
            })
            .collect();
        let selected = func
            .append_inst(merge_block, InstKind::Phi { incomings }, Some(Type::I32))
            .unwrap();
        values.insert(candidate.accumulator_next, selected);

        if lane + 1 == factor {
            let kind = remap_pure_kind(&inner_next_kind, values)
                .expect("validated conditional induction update must be pure");
            let next = func
                .append_inst(merge_block, kind, Some(Type::I32))
                .unwrap();
            values.insert(candidate.inner_induction.next, next);
            fast_inner_next = Some(next);
        }
        let next_target = if lane + 1 == factor {
            fast_inner_header
        } else {
            lane_blocks[lane + 1].0
        };
        func.set_terminator(merge_block, Terminator::Jump(next_target));
    }

    set_phi_backedge(func, fast_inner, last_merge, fast_inner_next.unwrap());
    for (values, accumulator) in lane_values.iter().zip(&fast_accumulators) {
        set_phi_backedge(
            func,
            *accumulator,
            last_merge,
            map_value(candidate.accumulator_next, values),
        );
    }

    for values in &mut lane_values {
        clone_exit_pure_instructions(func, candidate.inner_exit, fast_exit, values);
    }
    // Preserve the source outer-iteration store order. Loads in either arm of
    // each cloned diamond were included in the cross-lane NoAlias proof.
    for (values, accumulator) in lane_values.iter().zip(&fast_accumulators) {
        func.append_inst(
            fast_exit,
            InstKind::Store {
                ptr: map_value(candidate.store_ptr, values),
                value: *accumulator,
            },
            None,
        );
    }
    let fast_outer_next = func
        .append_inst(
            fast_exit,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: fast_outer,
                rhs: factor_value,
            },
            Some(Type::I32),
        )
        .unwrap();
    set_phi_backedge(func, fast_outer, fast_exit, fast_outer_next);
    func.set_terminator(fast_exit, Terminator::Jump(fast_header));

    func.blocks[candidate.outer_preheader.0].terminator = Some(Terminator::Jump(fast_header));
    replace_phi_incoming(
        func,
        candidate.outer_induction.phi,
        candidate.outer_preheader,
        fast_header,
        fast_outer,
    );
    func.mark_reduction_jammed();
}

fn clone_pure_block(
    func: &mut Function,
    source: BlockId,
    target: BlockId,
    values: &mut HashMap<ValueId, ValueId>,
) {
    let instructions = func.blocks[source.0].insts.clone();
    for inst in instructions {
        if matches!(inst.kind, InstKind::Nop) {
            continue;
        }
        let kind = remap_pure_kind(&inst.kind, values)
            .expect("candidate validation must accept every cloned instruction");
        let result_ty = inst.result.map(|result| func.value(result).ty.clone());
        let cloned = func.append_inst(target, kind, result_ty);
        if let (Some(original), Some(cloned)) = (inst.result, cloned) {
            values.insert(original, cloned);
        }
    }
}

fn conditional_shared_load_values(
    func: &Function,
    source: BlockId,
    lane_zero_values: &HashMap<ValueId, ValueId>,
) -> HashMap<ValueId, ValueId> {
    let mut shared = HashMap::new();
    for inst in &func.blocks[source.0].insts {
        let (Some(result), InstKind::Load { ptr }) = (inst.result, &inst.kind) else {
            continue;
        };
        let Some(mapped_result) = lane_zero_values.get(&result).copied() else {
            continue;
        };
        shared.insert(*ptr, map_value(*ptr, lane_zero_values));
        shared.insert(result, mapped_result);
    }
    shared
}

fn clone_conditional_condition_block(
    func: &mut Function,
    source: BlockId,
    target: BlockId,
    values: &mut HashMap<ValueId, ValueId>,
    shared_values: Option<&HashMap<ValueId, ValueId>>,
) {
    let instructions = func.blocks[source.0].insts.clone();
    for inst in instructions {
        if matches!(inst.kind, InstKind::Nop) {
            continue;
        }
        if let (Some(shared_values), Some(result), InstKind::Load { ptr }) =
            (shared_values, inst.result, &inst.kind)
        {
            if map_value(*ptr, values) == map_value(*ptr, shared_values) {
                if let Some(shared_result) = shared_values.get(&result).copied() {
                    values.insert(result, shared_result);
                    continue;
                }
            }
        }
        let kind = remap_pure_kind(&inst.kind, values)
            .expect("conditional candidate validation must accept every cloned instruction");
        let result_ty = inst.result.map(|result| func.value(result).ty.clone());
        let cloned = func.append_inst(target, kind, result_ty);
        if let (Some(original), Some(cloned)) = (inst.result, cloned) {
            values.insert(original, cloned);
        }
    }
}

fn clone_exit_pure_instructions(
    func: &mut Function,
    source: BlockId,
    target: BlockId,
    values: &mut HashMap<ValueId, ValueId>,
) {
    let instructions = func.blocks[source.0].insts.clone();
    for inst in instructions {
        if matches!(inst.kind, InstKind::Nop | InstKind::Store { .. }) {
            continue;
        }
        let kind = remap_pure_kind(&inst.kind, values)
            .expect("candidate exit validation must accept every pure instruction");
        let result_ty = inst.result.map(|result| func.value(result).ty.clone());
        let cloned = func.append_inst(target, kind, result_ty);
        if let (Some(original), Some(cloned)) = (inst.result, cloned) {
            values.insert(original, cloned);
        }
    }
}

fn remap_pure_kind(kind: &InstKind, values: &HashMap<ValueId, ValueId>) -> Option<InstKind> {
    let map = |value| map_value(value, values);
    match kind {
        InstKind::Nop => Some(InstKind::Nop),
        InstKind::Load { ptr } => Some(InstKind::Load { ptr: map(*ptr) }),
        InstKind::Unary { op, value } => Some(InstKind::Unary {
            op: *op,
            value: map(*value),
        }),
        InstKind::Binary { op, lhs, rhs } => Some(InstKind::Binary {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        }),
        InstKind::Icmp { op, lhs, rhs } => Some(InstKind::Icmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        }),
        InstKind::Fcmp { op, lhs, rhs } => Some(InstKind::Fcmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        }),
        InstKind::Cast { op, value } => Some(InstKind::Cast {
            op: *op,
            value: map(*value),
        }),
        InstKind::Gep { base, indices } => Some(InstKind::Gep {
            base: map(*base),
            indices: indices.iter().map(|index| map(*index)).collect(),
        }),
        InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Store { .. }
        | InstKind::MemZero { .. }
        | InstKind::Call { .. } => None,
    }
}

fn map_value(value: ValueId, values: &HashMap<ValueId, ValueId>) -> ValueId {
    values.get(&value).copied().unwrap_or(value)
}

fn set_phi_backedge(func: &mut Function, phi: ValueId, pred: BlockId, value: ValueId) {
    let ValueKind::Inst(block, inst_idx) = func.value(phi).kind else {
        unreachable!();
    };
    let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
        unreachable!();
    };
    incomings
        .iter_mut()
        .find(|(incoming_pred, _)| *incoming_pred == pred)
        .unwrap()
        .1 = value;
}

fn replace_phi_incoming(
    func: &mut Function,
    phi: ValueId,
    old_pred: BlockId,
    new_pred: BlockId,
    value: ValueId,
) {
    let ValueKind::Inst(block, inst_idx) = func.value(phi).kind else {
        unreachable!();
    };
    let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
        unreachable!();
    };
    let incoming = incomings
        .iter_mut()
        .find(|(pred, _)| *pred == old_pred)
        .unwrap();
    *incoming = (new_pred, value);
}

fn phi_results(func: &Function, block: BlockId) -> Vec<ValueId> {
    func.blocks[block.0]
        .insts
        .iter()
        .filter_map(|inst| matches!(inst.kind, InstKind::Phi { .. }).then_some(inst.result?))
        .collect()
}

fn phi_pair(
    func: &Function,
    phi: ValueId,
    entering: BlockId,
    latch: BlockId,
) -> Option<(ValueId, ValueId)> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    Some((
        incomings.iter().find(|(pred, _)| *pred == entering)?.1,
        incomings.iter().find(|(pred, _)| *pred == latch)?.1,
    ))
}

fn canonical_less_than_bound(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: ValueId,
) -> Option<ValueId> {
    if func.values.get(induction.0)?.ty != Type::I32 {
        return None;
    }
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks[natural_loop.header.0].terminator.as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || natural_loop.blocks.contains(else_target) {
        return None;
    }
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
        return None;
    };
    let bound = match (*op, *lhs == induction, *rhs == induction) {
        (CmpOp::Lt, true, false) => *rhs,
        (CmpOp::Gt, false, true) => *lhs,
        _ => return None,
    };
    (func.values.get(bound.0)?.ty == Type::I32).then_some(bound)
}

fn loop_inside_target(func: &Function, natural_loop: &NaturalLoop) -> Option<BlockId> {
    let Terminator::Branch {
        then_target,
        else_target,
        ..
    } = func.blocks[natural_loop.header.0].terminator.as_ref()?
    else {
        return None;
    };
    match (
        natural_loop.blocks.contains(then_target),
        natural_loop.blocks.contains(else_target),
    ) {
        (true, false) => Some(*then_target),
        (false, true) => Some(*else_target),
        _ => None,
    }
}

/// Fast headers are rebuilt rather than cloned. Accept only the phis and the
/// exact branch condition that the transform reconstructs; even a resultless
/// call whose value was folded away must keep the loop on the original path.
fn has_only_phi_and_branch_condition(func: &Function, block: BlockId) -> bool {
    let Some(Terminator::Branch { cond, .. }) = func.blocks[block.0].terminator.as_ref() else {
        return false;
    };
    func.blocks[block.0]
        .insts
        .iter()
        .all(|inst| match &inst.kind {
            InstKind::Nop | InstKind::Phi { .. } => true,
            InstKind::Icmp { .. } => inst.result == Some(*cond),
            _ => false,
        })
}

fn has_exact_conditional_merge(
    func: &Function,
    block: BlockId,
    selector: ValueId,
    induction_next: ValueId,
) -> bool {
    let mut saw_selector = false;
    let mut saw_induction_next = false;
    for inst in &func.blocks[block.0].insts {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { .. } if inst.result == Some(selector) && !saw_selector => {
                saw_selector = true;
            }
            InstKind::Binary { op, .. }
                if inst.result == Some(induction_next)
                    && !saw_induction_next
                    && !matches!(op, BinaryOp::Idiv | BinaryOp::Imod | BinaryOp::Fdiv) =>
            {
                saw_induction_next = true;
            }
            _ => return false,
        }
    }
    saw_selector && saw_induction_next
}

fn inner_results_have_no_extra_liveouts(
    func: &Function,
    inner: &NaturalLoop,
    accumulator: ValueId,
    inner_exit: BlockId,
) -> bool {
    let inner_results = inner
        .blocks
        .iter()
        .flat_map(|block| &func.blocks[block.0].insts)
        .filter_map(|inst| inst.result)
        .collect::<HashSet<_>>();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let block_id = BlockId(block_idx);
        if inner.blocks.contains(&block_id) {
            continue;
        }
        for inst in &block.insts {
            let used_inner = instruction_operands(&inst.kind)
                .into_iter()
                .filter(|operand| inner_results.contains(operand))
                .collect::<Vec<_>>();
            if used_inner.is_empty() {
                continue;
            }
            if block_id == inner_exit
                && matches!(
                    inst.kind,
                    InstKind::Store { value, .. } if value == accumulator
                )
                && used_inner == [accumulator]
            {
                continue;
            }
            return false;
        }
        let terminator_operands = match block.terminator.as_ref() {
            Some(Terminator::Return(value)) => value.iter().copied().collect::<Vec<_>>(),
            Some(Terminator::Branch { cond, .. }) => vec![*cond],
            Some(Terminator::Jump(_)) | None => Vec::new(),
        };
        if terminator_operands
            .iter()
            .any(|operand| inner_results.contains(operand))
        {
            return false;
        }
    }
    true
}

fn is_pure_setup_block(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0]
        .insts
        .iter()
        .all(|inst| is_reorderable_pure_kind(&inst.kind, false))
}

fn is_pure_reduction_block(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0]
        .insts
        .iter()
        .all(|inst| is_reorderable_pure_kind(&inst.kind, true))
}

fn is_reorderable_pure_kind(kind: &InstKind, allow_load: bool) -> bool {
    match kind {
        InstKind::Nop
        | InstKind::Unary { .. }
        | InstKind::Icmp { .. }
        | InstKind::Fcmp { .. }
        | InstKind::Cast { .. }
        | InstKind::Gep { .. } => true,
        // Defined IR memory accesses stay within their complete typed object;
        // invalid/out-of-object accesses are undefined rather than observable
        // exceptions. Under that contract a proven non-aliasing load may move
        // across the lane-zero store. The IR has no volatile/atomic loads.
        InstKind::Load { .. } => allow_load,
        InstKind::Binary { op, .. } => {
            !matches!(op, BinaryOp::Idiv | BinaryOp::Imod | BinaryOp::Fdiv)
        }
        InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Store { .. }
        | InstKind::MemZero { .. }
        | InstKind::Call { .. } => false,
    }
}

fn has_only_one_store_side_effect(func: &Function, block: BlockId) -> bool {
    let mut stores = 0usize;
    for inst in &func.blocks[block.0].insts {
        match inst.kind {
            InstKind::Store { .. } => stores += 1,
            InstKind::Nop
            | InstKind::Unary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. }
            | InstKind::Gep { .. } => {}
            InstKind::Binary { op, .. }
                if !matches!(op, BinaryOp::Idiv | BinaryOp::Imod | BinaryOp::Fdiv) => {}
            InstKind::Phi { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Binary { .. }
            | InstKind::Call { .. }
            | InstKind::MemZero { .. } => return false,
        }
    }
    stores == 1
}

fn proves_lane_independence(
    func: &Function,
    store_ptr: ValueId,
    store_value: ValueId,
    outer_induction: ValueId,
    inner_body: BlockId,
    factor: usize,
) -> bool {
    proves_lane_independence_in_blocks(
        func,
        store_ptr,
        store_value,
        outer_induction,
        &[inner_body],
        factor,
    )
}

fn proves_lane_independence_in_blocks(
    func: &Function,
    store_ptr: ValueId,
    store_value: ValueId,
    outer_induction: ValueId,
    load_blocks: &[BlockId],
    factor: usize,
) -> bool {
    if !matches!(factor, 2 | 4) {
        return false;
    }
    let mut budget = MemoryProofBudget::default();
    let Some(store_ty) = func.values.get(store_value.0).map(|value| &value.ty) else {
        return false;
    };
    let Some(store) = analyze_global_memory_access(func, store_ptr, store_ty, &mut budget) else {
        return false;
    };
    if store.terminal_index != outer_induction {
        return false;
    }

    let mut loads = Vec::new();
    for block in load_blocks {
        for inst in &func.blocks[block.0].insts {
            let InstKind::Load { ptr } = inst.kind else {
                continue;
            };
            let Some(result) = inst.result else {
                return false;
            };
            let Some(load_ty) = func.values.get(result.0).map(|value| &value.ty) else {
                return false;
            };
            let Some(load) = analyze_global_memory_access(func, ptr, load_ty, &mut budget) else {
                return false;
            };
            if load.global == store.global && load.terminal_index != outer_induction {
                return false;
            }
            loads.push(load);
        }
    }

    // Stores stay in original lane order. For every store that originally
    // preceded a later lane, prove NoAlias against every load from each such
    // lane before allowing those loads to move ahead of the store.
    for store_lane in 0..factor - 1 {
        for load_lane in store_lane + 1..factor {
            for load in &loads {
                if load.global != store.global {
                    // Distinct global symbols denote distinct complete objects.
                    // Spelling is ordinary identity, never classification.
                    continue;
                }
                if !same_global_lane_store_load_no_alias(&store, store_lane, load, load_lane)
                    .unwrap_or(false)
                {
                    return false;
                }
            }
        }
    }
    true
}

#[derive(Default)]
struct MemoryProofBudget {
    type_nodes: usize,
    work: usize,
}

impl MemoryProofBudget {
    fn spend_type_node(&mut self) -> Option<()> {
        self.type_nodes = self.type_nodes.saturating_add(1);
        (self.type_nodes <= MAX_MEMORY_TYPE_NODES).then_some(())
    }

    fn spend_work(&mut self, amount: usize) -> Option<()> {
        self.work = self.work.saturating_add(amount);
        (self.work <= MAX_MEMORY_PROOF_WORK).then_some(())
    }
}

struct GlobalMemoryAccess {
    global: String,
    constant_offset: i64,
    prefix_strides: Vec<i64>,
    terminal_index: ValueId,
    terminal_stride: i64,
    width: i64,
}

/// Describes the complete nested GEP chain in the byte-address model used by
/// code generation. The access itself is the terminal one-index GEP; earlier
/// constant indices become a checked byte offset and every other prefix index
/// contributes an independent unknown coefficient.
fn analyze_global_memory_access(
    func: &Function,
    ptr: ValueId,
    access_ty: &Type,
    budget: &mut MemoryProofBudget,
) -> Option<GlobalMemoryAccess> {
    budget.spend_work(1)?;
    let Type::Ptr(pointee) = &func.values.get(ptr.0)?.ty else {
        return None;
    };
    if !types_equal_checked(pointee, access_ty, budget)? {
        return None;
    }
    let width = checked_type_size(access_ty, budget)?;
    if width <= 0 {
        return None;
    }

    let mut current = ptr;
    let mut chain_depth = 0usize;
    let mut index_count = 0usize;
    let mut constant_offset = 0i64;
    let mut prefix_strides = Vec::new();
    let mut terminal = None;

    loop {
        chain_depth = chain_depth.checked_add(1)?;
        if chain_depth > MAX_MEMORY_GEP_CHAIN_DEPTH {
            return None;
        }
        budget.spend_work(1)?;

        let InstKind::Gep { base, indices } = defining_inst(func, current)? else {
            return None;
        };
        if !matches!(func.values.get(current.0)?.ty, Type::Ptr(_))
            || !matches!(func.values.get(base.0)?.ty, Type::Ptr(_))
            || indices.is_empty()
        {
            return None;
        }
        index_count = index_count.checked_add(indices.len())?;
        if index_count > MAX_MEMORY_GEP_INDICES {
            return None;
        }
        budget.spend_work(indices.len())?;
        if !indices.iter().all(|index| {
            matches!(
                func.values.get(index.0).map(|value| &value.ty),
                Some(Type::I32)
            )
        }) {
            return None;
        }
        let strides = typed_gep_byte_strides(func, *base, current, indices.len(), budget)?;

        if terminal.is_none() {
            let [index] = indices.as_slice() else {
                return None;
            };
            terminal = Some((*index, strides[0]));
        } else {
            for (index, stride) in indices.iter().copied().zip(strides) {
                if let Some(value) = constant_i32_index(func, index) {
                    let byte_offset = i64::from(value).checked_mul(stride)?;
                    constant_offset = constant_offset.checked_add(byte_offset)?;
                } else {
                    prefix_strides.push(stride);
                }
            }
        }

        current = *base;
        match &func.values.get(current.0)?.kind {
            ValueKind::Global(name) => {
                budget.spend_work(1)?;
                let (terminal_index, terminal_stride) = terminal?;
                return Some(GlobalMemoryAccess {
                    global: name.clone(),
                    constant_offset,
                    prefix_strides,
                    terminal_index,
                    terminal_stride,
                    width,
                });
            }
            ValueKind::Inst(_, _) => {
                if !matches!(defining_inst(func, current), Some(InstKind::Gep { .. })) {
                    return None;
                }
            }
            ValueKind::Param | ValueKind::Const(_) => return None,
        }
    }
}

fn typed_gep_byte_strides(
    func: &Function,
    base: ValueId,
    result: ValueId,
    index_count: usize,
    budget: &mut MemoryProofBudget,
) -> Option<Vec<i64>> {
    let Type::Ptr(_) = &func.values.get(base.0)?.ty else {
        return None;
    };
    let Type::Ptr(result_pointee) = &func.values.get(result.0)?.ty else {
        return None;
    };
    if index_count == 1 {
        return Some(vec![checked_type_size(result_pointee, budget)?.max(1)]);
    }

    let mut current_ty = &func.values.get(base.0)?.ty;
    let mut strides = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        current_ty = match current_ty {
            Type::Ptr(inner) => inner,
            Type::Array { elem, .. } => elem,
            Type::Void | Type::I1 | Type::I32 | Type::F32 => return None,
        };
        strides.push(checked_type_size(current_ty, budget)?.max(1));
    }
    Some(strides)
}

fn constant_i32_index(func: &Function, value: ValueId) -> Option<i32> {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Const(crate::ir::Const::Int(value))) => Some(*value),
        Some(ValueKind::Const(crate::ir::Const::Zero(Type::I32))) => Some(0),
        _ => None,
    }
}

fn types_equal_checked(
    mut lhs: &Type,
    mut rhs: &Type,
    budget: &mut MemoryProofBudget,
) -> Option<bool> {
    loop {
        budget.spend_type_node()?;
        budget.spend_work(1)?;
        match (lhs, rhs) {
            (Type::Void, Type::Void)
            | (Type::I1, Type::I1)
            | (Type::I32, Type::I32)
            | (Type::F32, Type::F32) => return Some(true),
            (Type::Ptr(lhs_inner), Type::Ptr(rhs_inner)) => {
                lhs = lhs_inner;
                rhs = rhs_inner;
            }
            (
                Type::Array {
                    elem: lhs_elem,
                    len: lhs_len,
                },
                Type::Array {
                    elem: rhs_elem,
                    len: rhs_len,
                },
            ) if lhs_len == rhs_len => {
                lhs = lhs_elem;
                rhs = rhs_elem;
            }
            _ => return Some(false),
        }
    }
}

fn checked_type_size(ty: &Type, budget: &mut MemoryProofBudget) -> Option<i64> {
    let mut current = ty;
    let mut elements = 1i64;
    loop {
        budget.spend_type_node()?;
        budget.spend_work(1)?;
        match current {
            Type::Void => return Some(0),
            Type::I1 | Type::I32 | Type::F32 => {
                return elements
                    .checked_mul(4)
                    .filter(|size| *size <= i64::from(i32::MAX));
            }
            Type::Ptr(_) => {
                return elements
                    .checked_mul(8)
                    .filter(|size| *size <= i64::from(i32::MAX));
            }
            Type::Array { elem, len } => {
                let len = i64::try_from(*len).ok()?;
                elements = elements
                    .checked_mul(len)
                    .filter(|size| *size <= i64::from(i32::MAX))?;
                current = elem;
            }
        }
    }
}

/// For a store in lane `s` and a later load in lane `l` on the same global,
/// their byte-address difference is
///
/// `C + sum(prefix coefficients) + outer * (load stride - store stride)`,
///
/// where `C` includes `l * load_stride - s * store_stride`. Prefix SSA
/// identities are deliberately ignored: every nonconstant prefix index is an
/// independent arbitrary integer, so the possible differences are
/// conservatively represented by one congruence class.
fn same_global_lane_store_load_no_alias(
    store: &GlobalMemoryAccess,
    store_lane: usize,
    load: &GlobalMemoryAccess,
    load_lane: usize,
) -> Option<bool> {
    let store_lane = i64::try_from(store_lane).ok()?;
    let load_lane = i64::try_from(load_lane).ok()?;
    let store_lane_offset = store.terminal_stride.checked_mul(store_lane)?;
    let load_lane_offset = load.terminal_stride.checked_mul(load_lane)?;
    let constant = load
        .constant_offset
        .checked_sub(store.constant_offset)?
        .checked_add(load_lane_offset)?
        .checked_sub(store_lane_offset)?;
    let terminal_difference = load.terminal_stride.checked_sub(store.terminal_stride)?;

    // Machine address arithmetic is modulo 2^64. Include that modulus as a
    // generator so an overlap reachable only through pointer wrap is also
    // treated as possible, rather than relying on an implicit in-bounds rule.
    let mut coefficient_gcd = 1u128 << 64;
    for coefficient in store
        .prefix_strides
        .iter()
        .chain(&load.prefix_strides)
        .copied()
        .chain(std::iter::once(terminal_difference))
    {
        coefficient_gcd =
            nonnegative_gcd_u128(coefficient_gcd, u128::from(coefficient.unsigned_abs()));
    }

    let low = load.width.checked_sub(1)?.checked_neg()?;
    let high = store.width.checked_sub(1)?;
    Some(!interval_contains_congruent_modulo(
        low,
        high,
        constant,
        coefficient_gcd,
    )?)
}

fn nonnegative_gcd_u128(mut lhs: u128, mut rhs: u128) -> u128 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn signed_residue(value: i64, modulus: u128) -> u128 {
    debug_assert!(modulus > 0);
    if value >= 0 {
        (value as u128) % modulus
    } else {
        let magnitude = u128::from(value.unsigned_abs()) % modulus;
        if magnitude == 0 {
            0
        } else {
            modulus - magnitude
        }
    }
}

fn interval_contains_congruent_modulo(
    low: i64,
    high: i64,
    value: i64,
    modulus: u128,
) -> Option<bool> {
    debug_assert!(low <= high);
    debug_assert!(modulus > 0);
    let residue = signed_residue(value, modulus);
    let low_residue = signed_residue(low, modulus);
    let delta = (residue + modulus - low_residue) % modulus;
    let interval_length = u128::from(high.checked_sub(low)?.unsigned_abs());
    Some(delta <= interval_length)
}

fn value_mappable_from_blocks(
    func: &Function,
    dom: &Dominators,
    value: ValueId,
    insertion_preheader: BlockId,
    outer_induction: ValueId,
    cloned_blocks: &[BlockId],
) -> bool {
    if value == outer_induction || value_available_at(func, dom, value, insertion_preheader) {
        return true;
    }
    matches!(
        func.value(value).kind,
        ValueKind::Inst(block, _) if cloned_blocks.contains(&block)
    )
}

fn block_results(func: &Function, block: BlockId) -> HashSet<ValueId> {
    func.blocks[block.0]
        .insts
        .iter()
        .filter_map(|inst| inst.result)
        .collect()
}

fn branch_condition_cloneable(
    func: &Function,
    dom: &Dominators,
    block: BlockId,
    insertion_preheader: BlockId,
    mapped: &HashSet<ValueId>,
) -> bool {
    let Some(Terminator::Branch { cond, .. }) = func.blocks[block.0].terminator.as_ref() else {
        return false;
    };
    mapped.contains(cond) || value_available_at(func, dom, *cond, insertion_preheader)
}

fn block_operands_cloneable(
    func: &Function,
    dom: &Dominators,
    block: BlockId,
    insertion_preheader: BlockId,
    mapped: &HashSet<ValueId>,
    skip_store: bool,
) -> bool {
    let mut local = HashSet::new();
    for inst in &func.blocks[block.0].insts {
        if skip_store && matches!(inst.kind, InstKind::Store { .. }) {
            continue;
        }
        for operand in instruction_operands(&inst.kind) {
            if !mapped.contains(&operand)
                && !local.contains(&operand)
                && !value_available_at(func, dom, operand, insertion_preheader)
            {
                return false;
            }
        }
        if let Some(result) = inst.result {
            local.insert(result);
        }
    }
    true
}

fn instruction_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut operands = vec![*base];
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn value_available_at(func: &Function, dom: &Dominators, value: ValueId, block: BlockId) -> bool {
    match func.value(value).kind {
        ValueKind::Const(_) | ValueKind::Global(_) | ValueKind::Param => true,
        ValueKind::Inst(def_block, _) => def_block == block || dom.dominates(def_block, block),
    }
}

fn executable_instruction_count(func: &Function, block: BlockId) -> usize {
    func.blocks[block.0]
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count()
}
