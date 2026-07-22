use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{
    analyze_const_i32_trip_count, analyze_i32_induction, LoopInfo, NaturalLoop,
};
use super::ModulePass;
use crate::ir::{
    AArch64ThreadCapture, AArch64ThreadPlan, BinaryOp, BlockId, CastOp, CmpOp, Const, Function,
    FunctionId, Inst, InstKind, Module, Terminator, Type, UnaryOp, ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_FUNCTIONS: usize = 256;
const MAX_GLOBALS: usize = 4096;
const MAX_REGION_BLOCKS: usize = 128;
const MAX_REGION_INSTS: usize = 1024;
const MAX_MEMORY_INSTS: usize = 256;
const MAX_GEP_CHAIN: usize = 32;
const MAX_TYPE_DEPTH: usize = 32;
const MAX_DEPENDENCE_VALUES: usize = 2048;
const MAX_CAPTURES: usize = 6;
const MAX_PLANS_PER_MODULE: usize = 16;
const MAX_PROOF_WORK: usize = 262_144;
const MAX_ADDED_VALUES: usize = 2048;
const MAX_ADDED_BLOCKS: usize = 2048;
const MIN_ACTIVE_REGION_COST: usize = 8;
const DEFAULT_PARALLEL_THRESHOLD: u32 = 65_536;

/// Outlines a strictly proven outer owner-computes loop into a verified range
/// helper. The original loop remains semantically unchanged: when necessary a
/// unique entering edge is transactionally split, and AArch64 emission inserts
/// the pthread dispatch at the resulting preheader while retaining the exact
/// scalar edge as the create-failure path.
pub(super) struct AArch64ThreadOutlinePass;

impl AArch64ThreadOutlinePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for AArch64ThreadOutlinePass {
    fn run(&mut self, module: &mut Module) {
        if !module.aarch64_thread_plans.is_empty() {
            return;
        }
        outline_module(module);
    }
}

#[derive(Clone)]
struct Candidate {
    /// The predecessor used by all original-CFG proofs and helper cloning.
    /// When `needs_split` is true this is only a virtual preheader.
    entering_pred: BlockId,
    needs_split: bool,
    header: BlockId,
    /// Kept in the existing plan slot named `body`; for a multi-block region
    /// this is the unique outer latch.
    latch: BlockId,
    exit: BlockId,
    counter: ValueId,
    counter_next: ValueId,
    condition: ValueId,
    bound: ValueId,
    region_blocks: Vec<BlockId>,
    dispatch_setup: Vec<ValueId>,
    captures: Vec<AArch64ThreadCapture>,
    parallel_threshold: i32,
    loop_rank: usize,
}

struct PendingOutline {
    parent: FunctionId,
    candidate: Candidate,
    helper: Function,
    plan_preheader: BlockId,
    parent_after_split: Option<Function>,
    needs_split: bool,
    entering_pred: BlockId,
    context_symbol: String,
    worker_symbol: String,
}

#[derive(Clone, Copy)]
struct UseSite {
    block: BlockId,
    role: UseRole,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UseRole {
    LoadPointer,
    StorePointer,
    GepBase,
    GepIndex,
    Phi {
        predecessor: BlockId,
        target: BlockId,
    },
    Other,
}

struct WorkBudget {
    remaining: usize,
    exhausted: bool,
}

impl WorkBudget {
    fn new() -> Self {
        Self {
            remaining: MAX_PROOF_WORK,
            exhausted: false,
        }
    }

    fn spend(&mut self, amount: usize) -> Option<()> {
        if amount > self.remaining {
            self.remaining = 0;
            self.exhausted = true;
            return None;
        }
        self.remaining -= amount;
        Some(())
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

struct RegionAnalysis {
    active_cost: usize,
    written_roots: HashSet<String>,
}

#[derive(Clone)]
struct PointerPath {
    root: String,
    root_is_const: bool,
    links: Vec<PointerLink>,
    geps: Vec<ValueId>,
}

#[derive(Clone)]
struct PointerLink {
    index: ValueId,
    selected_ty: Type,
}

struct MemoryAccess {
    path: PointerPath,
    is_store: bool,
}

#[derive(Clone, PartialEq, Eq)]
struct SliceSelector {
    depth: usize,
    selected_ty: Type,
}

fn outline_module(module: &mut Module) {
    let original_function_count = module.funcs.len();
    if original_function_count > MAX_FUNCTIONS || module.globals.len() > MAX_GLOBALS {
        return;
    }
    let mut symbol_counts = HashMap::<String, usize>::new();
    for name in module
        .funcs
        .iter()
        .map(|func| &func.name)
        .chain(module.globals.iter().map(|global| &global.name))
    {
        *symbol_counts.entry(name.clone()).or_default() += 1;
    }
    if symbol_counts.contains_key("pthread_create") || symbol_counts.contains_key("pthread_join") {
        return;
    }

    let mut reserved_symbols = symbol_counts.keys().cloned().collect::<HashSet<_>>();
    let mut budget = WorkBudget::new();
    let mut pending = Vec::new();
    let mut added_values = 0usize;
    let mut added_blocks = 0usize;

    for function_idx in 0..original_function_count {
        if pending.len() >= MAX_PLANS_PER_MODULE {
            break;
        }
        let func = &module.funcs[function_idx];
        if func.blocks.is_empty()
            || func.blocks.len() > MAX_BLOCKS
            || func.values.len() > MAX_VALUES
        {
            continue;
        }
        let Some(instruction_count) = func
            .blocks
            .iter()
            .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
        else {
            return;
        };
        let Some(structural_work) = func
            .blocks
            .len()
            .checked_mul(func.blocks.len())
            .and_then(|work| work.checked_add(func.values.len()))
            .and_then(|work| work.checked_add(instruction_count))
        else {
            return;
        };
        if budget.spend(structural_work).is_none() {
            return;
        }
        let Some(verify_work) = function_verify_work(func) else {
            return;
        };
        if budget.spend(verify_work).is_none() {
            return;
        }
        if func.verify().is_err() {
            continue;
        }

        let cfg = ControlFlowGraph::new(func);
        let dom = Dominators::new(func, &cfg);
        let loop_info = LoopInfo::new(&cfg, &dom);
        let Some(uses) = collect_uses(func, &mut budget) else {
            return;
        };
        let mut selected = None::<Candidate>;
        let mut maximum_outer_loop_rank = 0usize;
        for (loop_idx, natural_loop) in loop_info.loops().iter().enumerate() {
            let Some(nesting_work) = loop_info
                .loops()
                .len()
                .checked_mul(natural_loop.blocks.len())
            else {
                return;
            };
            if budget.spend(nesting_work).is_none() {
                return;
            }
            let nested_loops = match complete_nested_loops(&loop_info, loop_idx, &mut budget) {
                Some(nested_loops) => nested_loops,
                None if budget.is_exhausted() => return,
                None => continue,
            };
            let Some(loop_rank) = nested_loops.len().checked_add(1) else {
                return;
            };
            maximum_outer_loop_rank = maximum_outer_loop_rank.max(loop_rank);
            let candidate = match match_candidate(
                module,
                func,
                &cfg,
                &dom,
                natural_loop,
                &nested_loops,
                &uses,
                &mut budget,
            ) {
                Some(candidate) => candidate,
                None if budget.is_exhausted() => return,
                None => continue,
            };
            let is_better = selected.as_ref().is_none_or(|best| {
                candidate.parallel_threshold < best.parallel_threshold
                    || candidate.parallel_threshold == best.parallel_threshold
                        && candidate.region_blocks.len() > best.region_blocks.len()
            });
            if is_better {
                selected = Some(candidate);
            }
        }
        let Some(candidate) = selected else {
            continue;
        };
        if candidate.loop_rank < maximum_outer_loop_rank {
            // Do not pay thread startup for a shallower side loop while a
            // structurally more expensive loop nest in the same function
            // remains serial. This is a target-cost guard, not a legality fact.
            continue;
        }

        let range_symbol = format!("__yuezhuo_parallel_range_{}", function_idx);
        let worker_symbol = format!("__yuezhuo_parallel_worker_{}", function_idx);
        let context_symbol = format!("__yuezhuo_parallel_context_{}", function_idx);
        if [&range_symbol, &worker_symbol, &context_symbol]
            .into_iter()
            .any(|symbol| reserved_symbols.contains(symbol))
        {
            continue;
        }
        let Some(final_function_count) = original_function_count
            .checked_add(pending.len())
            .and_then(|count| count.checked_add(1))
        else {
            return;
        };
        if final_function_count > MAX_FUNCTIONS {
            continue;
        }
        let helper = match build_range_helper(func, &candidate, range_symbol.clone(), &mut budget) {
            Some(helper) => helper,
            None if budget.is_exhausted() => return,
            None => continue,
        };
        let Some(next_added_values) = added_values.checked_add(helper.values.len()) else {
            return;
        };
        let split_block_count = usize::from(candidate.needs_split);
        let Some(parent_block_count) = func.blocks.len().checked_add(split_block_count) else {
            return;
        };
        let Some(candidate_added_blocks) = helper.blocks.len().checked_add(split_block_count)
        else {
            return;
        };
        let Some(next_added_blocks) = added_blocks.checked_add(candidate_added_blocks) else {
            return;
        };
        if next_added_values > MAX_ADDED_VALUES
            || next_added_blocks > MAX_ADDED_BLOCKS
            || parent_block_count > MAX_BLOCKS
            || helper.blocks.len() > MAX_BLOCKS
        {
            continue;
        }

        let (plan_preheader, parent_after_split) = if candidate.needs_split {
            match prepare_parent_split(func, &candidate, &mut budget) {
                Some((parent_after_split, split)) => (split, Some(parent_after_split)),
                None if budget.is_exhausted() => return,
                None => continue,
            }
        } else {
            (candidate.entering_pred, None)
        };

        added_values = next_added_values;
        added_blocks = next_added_blocks;
        reserved_symbols.insert(range_symbol);
        reserved_symbols.insert(worker_symbol.clone());
        reserved_symbols.insert(context_symbol.clone());
        // SysY has no user-created threads. Both lanes execute a call-free
        // helper and the parent waits for completion, so another activation
        // cannot overlap this per-site static context.
        pending.push(PendingOutline {
            parent: FunctionId(function_idx),
            needs_split: candidate.needs_split,
            entering_pred: candidate.entering_pred,
            candidate,
            helper,
            plan_preheader,
            parent_after_split,
            context_symbol,
            worker_symbol,
        });
    }

    for outline in pending {
        let PendingOutline {
            parent,
            candidate,
            helper: helper_function,
            plan_preheader,
            parent_after_split,
            needs_split,
            entering_pred,
            context_symbol,
            worker_symbol,
        } = outline;
        debug_assert_eq!(needs_split, candidate.needs_split);
        debug_assert_eq!(entering_pred, candidate.entering_pred);
        if let Some(parent_after_split) = parent_after_split {
            module.funcs[parent.0] = parent_after_split;
        }
        let helper = module.add_func(helper_function);
        module.aarch64_thread_plans.push(AArch64ThreadPlan {
            parent,
            helper,
            preheader: plan_preheader,
            header: candidate.header,
            body: candidate.latch,
            exit: candidate.exit,
            bound: candidate.bound,
            dispatch_setup: candidate.dispatch_setup,
            captures: candidate.captures,
            parallel_threshold: candidate.parallel_threshold,
            context_symbol,
            worker_symbol,
        });
    }
}

fn complete_nested_loops<'a>(
    loop_info: &'a LoopInfo,
    outer_idx: usize,
    budget: &mut WorkBudget,
) -> Option<Vec<&'a NaturalLoop>> {
    let outer = loop_info.loops().get(outer_idx)?;
    let mut nested = Vec::new();
    for (other_idx, other) in loop_info.loops().iter().enumerate() {
        if other_idx == outer_idx {
            continue;
        }
        budget.spend(other.blocks.len())?;
        if !outer
            .blocks
            .iter()
            .any(|block| other.blocks.contains(block))
        {
            continue;
        }
        // Reject an inner candidate that is contained by an outer loop and any
        // partially overlapping/non-natural nesting. A selected outer region
        // must contain every intersecting natural loop in full.
        if !other.blocks.is_subset(&outer.blocks) || other.blocks.len() >= outer.blocks.len() {
            return None;
        }
        let entering = other.unique_entering_pred?;
        if !outer.blocks.contains(&entering) || other.blocks.contains(&entering) {
            return None;
        }
        nested.push(other);
    }
    Some(nested)
}

#[allow(clippy::too_many_arguments)]
fn match_candidate(
    module: &Module,
    func: &Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    nested_loops: &[&NaturalLoop],
    uses: &[Vec<UseSite>],
    budget: &mut WorkBudget,
) -> Option<Candidate> {
    budget.spend(func.blocks.len())?;
    if !(2..=MAX_REGION_BLOCKS).contains(&natural_loop.blocks.len()) {
        return None;
    }
    let header = natural_loop.header;
    let entering_pred = natural_loop.unique_entering_pred?;
    let needs_split = natural_loop.dedicated_preheader.is_none();
    if natural_loop
        .dedicated_preheader
        .is_some_and(|preheader| preheader != entering_pred)
        || func.blocks.get(header.0).is_none()
        || func.blocks.get(entering_pred.0).is_none()
    {
        return None;
    }
    if needs_split {
        validate_virtual_preheader(func, cfg, header, entering_pred, budget)?;
    } else if func.blocks.get(entering_pred.0)?.terminator != Some(Terminator::Jump(header)) {
        return None;
    }

    let latch = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    if latch == header
        || natural_loop.exit_edges.as_slice() != [(header, exit)]
        || func.blocks.get(latch.0)?.terminator != Some(Terminator::Jump(header))
    {
        return None;
    }

    let header_predecessors = cfg.preds.get(header.0)?;
    if header_predecessors.len() != 2
        || !header_predecessors.contains(&entering_pred)
        || !header_predecessors.contains(&latch)
    {
        return None;
    }
    for block in &natural_loop.blocks {
        budget.spend(1)?;
        if !dom.is_reachable(*block) || !dom.dominates(header, *block) {
            return None;
        }
        if *block != header
            && cfg
                .preds
                .get(block.0)?
                .iter()
                .any(|pred| !natural_loop.blocks.contains(pred))
        {
            return None;
        }
        if cfg.succs.get(block.0)?.iter().any(|target| {
            !(natural_loop.blocks.contains(target) || *block == header && *target == exit)
        }) {
            return None;
        }
    }

    let header_block = func.blocks.get(header.0)?;
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = header_block.terminator.as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || *else_target != exit {
        return None;
    }

    let mut counter = None;
    let mut condition = None;
    let mut setup_values = Vec::new();
    for inst in &header_block.insts {
        budget.spend(1)?;
        match &inst.kind {
            InstKind::Nop => {
                if inst.result.is_some() {
                    return None;
                }
            }
            InstKind::Phi { .. } => {
                if counter.replace(inst.result?).is_some() {
                    return None;
                }
            }
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs,
                rhs,
            } if inst.result == Some(*cond) => {
                if condition.replace((*lhs, *rhs)).is_some() {
                    return None;
                }
            }
            _ => setup_values.push(inst.result?),
        }
    }
    let counter = counter?;
    let (condition_counter, bound) = condition?;
    if condition_counter != counter
        || func.value(counter).ty != Type::I32
        || func.value(bound).ty != Type::I32
        || func.value(*cond).ty != Type::I1
    {
        return None;
    }

    let induction = analyze_i32_induction(func, natural_loop, counter)?;
    if induction.step != 1
        || const_i32(func, induction.initial) != Some(0)
        || !matches!(
            func.value(induction.next).kind,
            ValueKind::Inst(owner, _) if owner == latch
        )
    {
        return None;
    }

    validate_region_liveouts(
        func,
        natural_loop,
        header,
        latch,
        counter,
        induction.next,
        *cond,
        uses,
        budget,
    )?;
    let analysis = analyze_region(
        module,
        func,
        cfg,
        natural_loop,
        header,
        counter,
        induction.next,
        uses,
        budget,
    )?;

    let dispatch_setup = if value_available_at(func, dom, bound, entering_pred) {
        if !setup_values.is_empty() {
            return None;
        }
        Vec::new()
    } else {
        analyze_header_setup(
            module,
            func,
            header,
            bound,
            &setup_values,
            uses,
            &analysis.written_roots,
            budget,
        )?
    };

    let captures = collect_captures(
        func,
        dom,
        natural_loop,
        entering_pred,
        header,
        latch,
        counter,
        induction.next,
        *cond,
        bound,
        budget,
    )?;
    let parallel_threshold = estimate_parallel_threshold(
        func,
        cfg,
        dom,
        natural_loop,
        nested_loops,
        bound,
        analysis.active_cost,
        budget,
    )?;

    let mut region_blocks = natural_loop.blocks.iter().copied().collect::<Vec<_>>();
    region_blocks.sort_by_key(|block| (*block != header, block.0));
    Some(Candidate {
        entering_pred,
        needs_split,
        header,
        latch,
        exit,
        counter,
        counter_next: induction.next,
        condition: *cond,
        bound,
        region_blocks,
        dispatch_setup,
        captures,
        parallel_threshold,
        loop_rank: nested_loops.len().checked_add(1)?,
    })
}

fn validate_virtual_preheader(
    func: &Function,
    cfg: &ControlFlowGraph,
    header: BlockId,
    entering_pred: BlockId,
    budget: &mut WorkBudget,
) -> Option<()> {
    if entering_pred == header {
        return None;
    }
    let successors = cfg.succs.get(entering_pred.0)?;
    let predecessors = cfg.preds.get(header.0)?;
    budget.spend(successors.len().checked_add(predecessors.len())?)?;
    if successors
        .iter()
        .filter(|target| **target == header)
        .count()
        != 1
        || predecessors
            .iter()
            .filter(|pred| **pred == entering_pred)
            .count()
            != 1
    {
        return None;
    }

    let terminator = func.blocks.get(entering_pred.0)?.terminator.as_ref()?;
    let edge_slots = terminator_edge_slots(terminator);
    budget.spend(edge_slots)?;
    if terminator_target_count(terminator, header) == 0 {
        return None;
    }

    for inst in &func.blocks.get(header.0)?.insts {
        budget.spend(1)?;
        let InstKind::Phi { incomings } = &inst.kind else {
            continue;
        };
        budget.spend(incomings.len())?;
        if incomings
            .iter()
            .filter(|(pred, _)| *pred == entering_pred)
            .count()
            != 1
        {
            return None;
        }
    }
    Some(())
}

fn terminator_edge_slots(terminator: &Terminator) -> usize {
    match terminator {
        Terminator::Return(_) => 0,
        Terminator::Jump(_) => 1,
        Terminator::Branch { .. } => 2,
    }
}

fn terminator_target_count(terminator: &Terminator, target: BlockId) -> usize {
    match terminator {
        Terminator::Return(_) => 0,
        Terminator::Jump(destination) => usize::from(*destination == target),
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => usize::from(*then_target == target) + usize::from(*else_target == target),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_region_liveouts(
    func: &Function,
    natural_loop: &NaturalLoop,
    header: BlockId,
    latch: BlockId,
    counter: ValueId,
    counter_next: ValueId,
    condition: ValueId,
    uses: &[Vec<UseSite>],
    budget: &mut WorkBudget,
) -> Option<()> {
    for block in &natural_loop.blocks {
        for inst in &func.blocks.get(block.0)?.insts {
            let Some(result) = inst.result else {
                continue;
            };
            for site in uses.get(result.0)? {
                budget.spend(1)?;
                if !natural_loop.blocks.contains(&site.block) {
                    return None;
                }
            }
        }
    }

    let counter_uses = uses.get(counter.0)?;
    if counter_uses
        .iter()
        .any(|site| !natural_loop.blocks.contains(&site.block))
        || counter_uses
            .iter()
            .filter(|site| site.block == header)
            .count()
            != 1
        || !counter_uses
            .iter()
            .any(|site| site.block == header && site.role == UseRole::Other)
    {
        return None;
    }
    let condition_uses = uses.get(condition.0)?;
    if condition_uses.len() != 1
        || condition_uses[0].block != header
        || condition_uses[0].role != UseRole::Other
    {
        return None;
    }
    let backedge_uses = uses
        .get(counter_next.0)?
        .iter()
        .filter(|site| {
            site.role
                == (UseRole::Phi {
                    predecessor: latch,
                    target: header,
                })
        })
        .count();
    (backedge_uses == 1).then_some(())
}

#[allow(clippy::too_many_arguments)]
fn analyze_region(
    module: &Module,
    func: &Function,
    cfg: &ControlFlowGraph,
    natural_loop: &NaturalLoop,
    header: BlockId,
    counter: ValueId,
    counter_next: ValueId,
    uses: &[Vec<UseSite>],
    budget: &mut WorkBudget,
) -> Option<RegionAnalysis> {
    let region_exit = natural_loop.unique_exit()?;
    let mut instruction_count = 0usize;
    let mut active_cost = 0usize;
    for block in sorted_region_blocks(natural_loop, header) {
        let source_block = func.blocks.get(block.0)?;
        instruction_count = instruction_count.checked_add(source_block.insts.len())?;
        if instruction_count > MAX_REGION_INSTS {
            return None;
        }
        match source_block.terminator.as_ref()? {
            Terminator::Return(_) => return None,
            Terminator::Jump(target) => {
                if !natural_loop.blocks.contains(target) {
                    return None;
                }
            }
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                if func.value(*cond).ty != Type::I1
                    || !(natural_loop.blocks.contains(then_target)
                        || block == header && *then_target == region_exit)
                    || !(natural_loop.blocks.contains(else_target)
                        || block == header && *else_target == region_exit)
                {
                    return None;
                }
            }
        }

        for inst in &source_block.insts {
            budget.spend(1)?;
            validate_region_inst(func, inst)?;
            if let InstKind::Phi { incomings } = &inst.kind {
                if block == header {
                    if inst.result != Some(counter) {
                        return None;
                    }
                } else if incomings.is_empty()
                    || incomings.iter().any(|(pred, _)| {
                        !natural_loop.blocks.contains(pred) || !cfg.preds[block.0].contains(pred)
                    })
                {
                    return None;
                }
            }
            if inst.result != Some(counter)
                && inst.result != Some(counter_next)
                && !matches!(inst.kind, InstKind::Nop)
            {
                active_cost = active_cost.checked_add(instruction_cost(inst))?;
            }
        }
    }
    if active_cost < MIN_ACTIVE_REGION_COST {
        return None;
    }

    let written_roots = analyze_memory(module, func, natural_loop, counter, uses, budget)?;
    Some(RegionAnalysis {
        active_cost,
        written_roots,
    })
}

fn validate_region_inst(func: &Function, inst: &Inst) -> Option<()> {
    match &inst.kind {
        InstKind::Nop => {
            if inst.result.is_some() {
                return None;
            }
        }
        InstKind::Phi { incomings } => {
            let result = inst.result?;
            if incomings.is_empty()
                || incomings
                    .iter()
                    .any(|(_, value)| func.value(*value).ty != func.value(result).ty)
            {
                return None;
            }
        }
        InstKind::Load { ptr } => {
            let result = inst.result?;
            let Type::Ptr(pointee) = &func.value(*ptr).ty else {
                return None;
            };
            if !is_four_byte_scalar(pointee) || **pointee != func.value(result).ty {
                return None;
            }
        }
        InstKind::Store { ptr, value } => {
            if inst.result.is_some() {
                return None;
            }
            let Type::Ptr(pointee) = &func.value(*ptr).ty else {
                return None;
            };
            if !is_four_byte_scalar(pointee) || **pointee != func.value(*value).ty {
                return None;
            }
        }
        InstKind::Unary { op, value } => {
            let result = inst.result?;
            let expected = match op {
                UnaryOp::Ineg => Type::I32,
                UnaryOp::Fneg => Type::F32,
                UnaryOp::Not => Type::I1,
            };
            if func.value(*value).ty != expected || func.value(result).ty != expected {
                return None;
            }
        }
        InstKind::Binary { op, lhs, rhs } => {
            let result = inst.result?;
            let expected = binary_type(*op);
            if func.value(*lhs).ty != expected
                || func.value(*rhs).ty != expected
                || func.value(result).ty != expected
                || *op == BinaryOp::Fdiv
                || (matches!(op, BinaryOp::Idiv | BinaryOp::Imod)
                    && const_i32(func, *rhs).is_none_or(|divisor| divisor == 0))
            {
                return None;
            }
        }
        InstKind::Icmp { lhs, rhs, .. } => {
            let result = inst.result?;
            if func.value(*lhs).ty != Type::I32
                || func.value(*rhs).ty != Type::I32
                || func.value(result).ty != Type::I1
            {
                return None;
            }
        }
        InstKind::Fcmp { lhs, rhs, .. } => {
            let result = inst.result?;
            if func.value(*lhs).ty != Type::F32
                || func.value(*rhs).ty != Type::F32
                || func.value(result).ty != Type::I1
            {
                return None;
            }
        }
        InstKind::Cast { op, value } => {
            let result = inst.result?;
            let (source, destination) = match op {
                CastOp::I32ToF32 => (Type::I32, Type::F32),
                CastOp::F32ToI32 => (Type::F32, Type::I32),
                CastOp::BoolToI32 => (Type::I1, Type::I32),
                CastOp::I32ToBool => (Type::I32, Type::I1),
                CastOp::F32ToBool => (Type::F32, Type::I1),
            };
            if func.value(*value).ty != source || func.value(result).ty != destination {
                return None;
            }
        }
        InstKind::Gep { .. } => {
            if !matches!(
                inst.result.map(|result| &func.value(result).ty),
                Some(Type::Ptr(_))
            ) {
                return None;
            }
        }
        InstKind::Alloca { .. } | InstKind::MemZero { .. } | InstKind::Call { .. } => return None,
    }
    Some(())
}

fn binary_type(op: BinaryOp) -> Type {
    match op {
        BinaryOp::Iadd
        | BinaryOp::Isub
        | BinaryOp::Imul
        | BinaryOp::Idiv
        | BinaryOp::Imod
        | BinaryOp::Iand
        | BinaryOp::Ior
        | BinaryOp::Ixor
        | BinaryOp::Ishl
        | BinaryOp::Iashr => Type::I32,
        BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => Type::F32,
        BinaryOp::And | BinaryOp::Or => Type::I1,
    }
}

fn analyze_memory(
    module: &Module,
    func: &Function,
    natural_loop: &NaturalLoop,
    counter: ValueId,
    uses: &[Vec<UseSite>],
    budget: &mut WorkBudget,
) -> Option<HashSet<String>> {
    let mut accesses = Vec::new();
    let mut all_region_geps = HashSet::new();
    let mut proven_geps = HashSet::new();
    let mut memory_count = 0usize;
    let mut has_store = false;

    for block in sorted_region_blocks(natural_loop, natural_loop.header) {
        for inst in &func.blocks.get(block.0)?.insts {
            if let InstKind::Gep { .. } = inst.kind {
                all_region_geps.insert(inst.result?);
            }
            let (pointer, is_store) = match inst.kind {
                InstKind::Load { ptr } => (ptr, false),
                InstKind::Store { ptr, .. } => (ptr, true),
                _ => continue,
            };
            memory_count = memory_count.checked_add(1)?;
            if memory_count > MAX_MEMORY_INSTS {
                return None;
            }
            has_store |= is_store;
            let path =
                trace_typed_global_pointer(module, func, natural_loop, pointer, uses, budget)?;
            proven_geps.extend(path.geps.iter().copied());
            accesses.push(MemoryAccess { path, is_store });
        }
    }
    if !has_store || all_region_geps != proven_geps {
        return None;
    }

    let mut written_selectors = HashMap::<String, SliceSelector>::new();
    for access in accesses.iter().filter(|access| access.is_store) {
        if access.path.root_is_const {
            return None;
        }
        let selector = partition_selector(func, natural_loop, counter, &access.path, budget)?;
        if written_selectors
            .get(&access.path.root)
            .is_some_and(|existing| *existing != selector)
        {
            return None;
        }
        written_selectors
            .entry(access.path.root.clone())
            .or_insert(selector);
    }
    for access in &accesses {
        let Some(expected) = written_selectors.get(&access.path.root) else {
            // A root with no stores in this region cannot race with either lane;
            // its complete typed in-object address may otherwise be arbitrary.
            continue;
        };
        let selector = partition_selector(func, natural_loop, counter, &access.path, budget)?;
        if &selector != expected {
            return None;
        }
    }
    Some(written_selectors.into_keys().collect())
}

fn trace_typed_global_pointer(
    module: &Module,
    func: &Function,
    natural_loop: &NaturalLoop,
    pointer: ValueId,
    uses: &[Vec<UseSite>],
    budget: &mut WorkBudget,
) -> Option<PointerPath> {
    let mut current = pointer;
    let mut reverse_links = Vec::new();
    let mut reverse_geps = Vec::new();

    loop {
        budget.spend(1)?;
        if reverse_links.len() > MAX_GEP_CHAIN {
            return None;
        }
        match &func.value(current).kind {
            ValueKind::Global(name) => {
                budget.spend(module.globals.len())?;
                let mut globals = module.globals.iter().filter(|global| global.name == *name);
                let global = globals.next()?;
                if globals.next().is_some()
                    || func.value(current).ty != Type::Ptr(Box::new(global.ty.clone()))
                    || checked_object_size(&global.ty, budget).is_none()
                {
                    return None;
                }
                reverse_links.reverse();
                reverse_geps.reverse();
                return Some(PointerPath {
                    root: name.clone(),
                    root_is_const: global.is_const,
                    links: reverse_links,
                    geps: reverse_geps,
                });
            }
            ValueKind::Inst(owner, inst_idx) if natural_loop.blocks.contains(owner) => {
                let inst = func.blocks.get(owner.0)?.insts.get(*inst_idx)?;
                if inst.result != Some(current) {
                    return None;
                }
                let InstKind::Gep { base, indices } = &inst.kind else {
                    return None;
                };
                let [index] = indices.as_slice() else {
                    // The lowering emits one typed array step per GEP. Keeping
                    // that form makes every selected subobject unambiguous.
                    return None;
                };
                let Type::Ptr(container) = &func.value(*base).ty else {
                    return None;
                };
                let Type::Ptr(selected) = &func.value(current).ty else {
                    return None;
                };
                let Type::Array { elem, len } = &**container else {
                    return None;
                };
                if *len == 0
                    || **elem != **selected
                    || func.value(*index).ty != Type::I32
                    || checked_object_size(container, budget).is_none()
                    || checked_object_size(selected, budget).is_none()
                {
                    return None;
                }
                for site in uses.get(current.0)? {
                    budget.spend(1)?;
                    if !natural_loop.blocks.contains(&site.block)
                        || !matches!(
                            site.role,
                            UseRole::LoadPointer | UseRole::StorePointer | UseRole::GepBase
                        )
                    {
                        return None;
                    }
                }
                reverse_links.push(PointerLink {
                    index: *index,
                    selected_ty: (**selected).clone(),
                });
                reverse_geps.push(current);
                current = *base;
            }
            ValueKind::Param | ValueKind::Const(_) | ValueKind::Inst(_, _) => return None,
        }
    }
}

fn checked_object_size(ty: &Type, budget: &mut WorkBudget) -> Option<usize> {
    let mut current = ty;
    let mut elements = 1usize;
    for _ in 0..MAX_TYPE_DEPTH {
        budget.spend(1)?;
        match current {
            Type::Array { elem, len } => {
                if *len == 0 || i32::try_from(*len).is_err() {
                    return None;
                }
                elements = elements.checked_mul(*len)?;
                current = elem;
            }
            Type::I1 | Type::I32 | Type::F32 => {
                return elements
                    .checked_mul(4)
                    .filter(|size| *size <= i32::MAX as usize)
            }
            Type::Ptr(_) => {
                return elements
                    .checked_mul(8)
                    .filter(|size| *size <= i32::MAX as usize)
            }
            Type::Void => return None,
        }
    }
    None
}

fn partition_selector(
    func: &Function,
    natural_loop: &NaturalLoop,
    counter: ValueId,
    path: &PointerPath,
    budget: &mut WorkBudget,
) -> Option<SliceSelector> {
    let mut selector = None;
    for (depth, link) in path.links.iter().enumerate() {
        budget.spend(1)?;
        if link.index == counter {
            if selector
                .replace(SliceSelector {
                    depth,
                    selected_ty: link.selected_ty.clone(),
                })
                .is_some()
            {
                return None;
            }
        } else if value_depends_on_counter(func, natural_loop, link.index, counter, budget)? {
            // This includes i+c, flattened dynamic strides, copied/indirect
            // selectors, and a second counter-dependent subscript.
            return None;
        }
    }
    selector
}

fn value_depends_on_counter(
    func: &Function,
    natural_loop: &NaturalLoop,
    value: ValueId,
    counter: ValueId,
    budget: &mut WorkBudget,
) -> Option<bool> {
    let mut stack = vec![value];
    let mut visited = HashSet::new();
    while let Some(current) = stack.pop() {
        budget.spend(1)?;
        if current == counter {
            return Some(true);
        }
        if !visited.insert(current) {
            continue;
        }
        if visited.len() > MAX_DEPENDENCE_VALUES {
            return None;
        }
        let ValueKind::Inst(owner, inst_idx) = func.value(current).kind else {
            continue;
        };
        if !natural_loop.blocks.contains(&owner) {
            continue;
        }
        let inst = func.blocks.get(owner.0)?.insts.get(inst_idx)?;
        if inst.result != Some(current) {
            return None;
        }
        stack.extend(inst_operands(&inst.kind));
    }
    Some(false)
}

#[allow(clippy::too_many_arguments)]
fn collect_captures(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    preheader: BlockId,
    header: BlockId,
    latch: BlockId,
    counter: ValueId,
    counter_next: ValueId,
    condition: ValueId,
    bound: ValueId,
    budget: &mut WorkBudget,
) -> Option<Vec<AArch64ThreadCapture>> {
    let region_results = natural_loop
        .blocks
        .iter()
        .flat_map(|block| {
            func.blocks[block.0]
                .insts
                .iter()
                .filter_map(|inst| inst.result)
        })
        .collect::<HashSet<_>>();
    let mut captured = HashSet::new();
    let mut captures = Vec::new();

    for block in sorted_region_blocks(natural_loop, header) {
        for inst in &func.blocks.get(block.0)?.insts {
            let operands = if inst.result == Some(counter) {
                let InstKind::Phi { incomings } = &inst.kind else {
                    return None;
                };
                incomings
                    .iter()
                    .filter_map(|(pred, value)| (*pred == latch).then_some(*value))
                    .collect::<Vec<_>>()
            } else if inst.result == Some(condition) {
                // The helper compares its cloned counter with `end`; the source
                // bound operand is not needed for this one use.
                vec![counter]
            } else {
                inst_operands(&inst.kind)
            };
            for operand in operands {
                budget.spend(1)?;
                if operand == counter_next || region_results.contains(&operand) {
                    continue;
                }
                match &func.value(operand).kind {
                    ValueKind::Const(_) | ValueKind::Global(_) => continue,
                    ValueKind::Param | ValueKind::Inst(_, _) => {}
                }
                if !value_available_at(func, dom, operand, preheader)
                    || !matches!(func.value(operand).ty, Type::I32 | Type::Ptr(_))
                {
                    return None;
                }
                if captured.insert(operand) {
                    captures.push(AArch64ThreadCapture {
                        value: operand,
                        ty: func.value(operand).ty.clone(),
                    });
                    if captures.len() > MAX_CAPTURES {
                        return None;
                    }
                }
            }
        }
    }

    // `bound` may also occur outside the rewritten condition. In that case it
    // was encountered above and captured normally; otherwise it deliberately
    // consumes no helper argument.
    let _ = bound;
    Some(captures)
}

#[allow(clippy::too_many_arguments)]
fn analyze_header_setup(
    module: &Module,
    func: &Function,
    header: BlockId,
    bound: ValueId,
    setup_values: &[ValueId],
    uses: &[Vec<UseSite>],
    written_roots: &HashSet<String>,
    budget: &mut WorkBudget,
) -> Option<Vec<ValueId>> {
    let setup_set = setup_values.iter().copied().collect::<HashSet<_>>();
    if setup_set.len() != setup_values.len() || !setup_set.contains(&bound) {
        return None;
    }

    let mut closure = HashSet::new();
    let mut stack = vec![bound];
    while let Some(value) = stack.pop() {
        budget.spend(1)?;
        if !setup_set.contains(&value) || !closure.insert(value) {
            continue;
        }
        let inst = defining_inst(func, value)?;
        stack.extend(inst_operands(&inst.kind));
    }
    if closure != setup_set {
        return None;
    }

    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for inst in &func.blocks.get(header.0)?.insts {
        let Some(result) = inst.result.filter(|result| setup_set.contains(result)) else {
            continue;
        };
        budget.spend(1)?;
        if inst_operands(&inst.kind)
            .into_iter()
            .any(|operand| setup_set.contains(&operand) && !seen.contains(&operand))
        {
            return None;
        }
        match &inst.kind {
            InstKind::Load { ptr } => {
                if func.value(result).ty != Type::I32 {
                    return None;
                }
                let ValueKind::Global(name) = &func.value(*ptr).kind else {
                    return None;
                };
                budget.spend(module.globals.len())?;
                if written_roots.contains(name)
                    || module
                        .globals
                        .iter()
                        .filter(|global| global.name == *name && global.ty == Type::I32)
                        .count()
                        != 1
                {
                    return None;
                }
            }
            InstKind::Unary {
                op: UnaryOp::Ineg | UnaryOp::Not,
                ..
            }
            | InstKind::Binary {
                op:
                    BinaryOp::Iadd
                    | BinaryOp::Isub
                    | BinaryOp::Imul
                    | BinaryOp::Iand
                    | BinaryOp::Ior
                    | BinaryOp::Ixor
                    | BinaryOp::Ishl
                    | BinaryOp::Iashr,
                ..
            }
            | InstKind::Icmp { .. }
            | InstKind::Cast {
                op: CastOp::BoolToI32 | CastOp::I32ToBool,
                ..
            } => {}
            InstKind::Binary {
                op: BinaryOp::Idiv | BinaryOp::Imod,
                rhs,
                ..
            } if const_i32(func, *rhs).is_some_and(|divisor| divisor != 0) => {}
            _ => return None,
        }
        seen.insert(result);
        ordered.push(result);
    }
    if ordered.len() != setup_values.len()
        || setup_values.iter().any(|value| {
            uses.get(value.0)
                .is_none_or(|sites| sites.iter().any(|site| site.block != header))
        })
    {
        return None;
    }
    Some(ordered)
}

fn estimate_parallel_threshold(
    func: &Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    outer: &NaturalLoop,
    nested_loops: &[&NaturalLoop],
    outer_bound: ValueId,
    active_cost: usize,
    budget: &mut WorkBudget,
) -> Option<i32> {
    if active_cost < MIN_ACTIVE_REGION_COST {
        return Some(DEFAULT_PARALLEL_THRESHOLD as i32);
    }
    let outer_latch = outer.unique_latch()?;
    let mut threshold = DEFAULT_PARALLEL_THRESHOLD;
    for inner in nested_loops {
        budget.spend(inner.blocks.len())?;
        let Some(entering) = inner.unique_entering_pred else {
            continue;
        };
        let Some(inner_exit) = inner.unique_exit() else {
            continue;
        };
        if inner.exit_edges.as_slice() != [(inner.header, inner_exit)] {
            // A body/latch break makes the nominal header trip count only an
            // upper bound, not guaranteed work for every outer iteration.
            continue;
        }
        if !outer.blocks.contains(&entering)
            || inner.blocks.contains(&entering)
            || !outer.blocks.contains(&inner_exit)
            || !dom.dominates(inner.header, outer_latch)
            || !dom.dominates(inner_exit, outer_latch)
            || cfg
                .succs
                .get(entering.0)
                .is_none_or(|successors| !successors.contains(&inner.header))
        {
            continue;
        }
        let mut nested_active = 0usize;
        for block in &inner.blocks {
            let instructions = &func.blocks.get(block.0)?.insts;
            budget.spend(instructions.len())?;
            nested_active = nested_active.checked_add(
                instructions
                    .iter()
                    .filter(|inst| !matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }))
                    .count(),
            )?;
        }
        if nested_active < MIN_ACTIVE_REGION_COST {
            continue;
        }
        budget.spend(func.blocks.get(inner.header.0)?.insts.len())?;
        for induction in func.blocks[inner.header.0]
            .insts
            .iter()
            .filter_map(|inst| inst.result)
            .filter_map(|phi| analyze_i32_induction(func, inner, phi))
        {
            if let Some(trip_count) = analyze_const_i32_trip_count(func, inner, induction)
                .filter(|trip_count| *trip_count > 1)
            {
                threshold = threshold.min(DEFAULT_PARALLEL_THRESHOLD.div_ceil(trip_count));
            }
            if induction.step == 1
                && const_i32(func, induction.initial) == Some(0)
                && canonical_lt_bound(func, inner, induction.phi) == Some(outer_bound)
            {
                // Once the outer bound reaches ceil(sqrt(base work)), this
                // guaranteed inner loop contributes at least the same number
                // of iterations to every active outer iteration.
                threshold = threshold.min(ceil_sqrt(DEFAULT_PARALLEL_THRESHOLD));
            }
        }
    }
    Some(threshold.max(2) as i32)
}

fn canonical_lt_bound(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: ValueId,
) -> Option<ValueId> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func
        .blocks
        .get(natural_loop.header.0)?
        .terminator
        .as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || natural_loop.blocks.contains(else_target) {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt,
        lhs,
        rhs,
    } = &defining_inst(func, *cond)?.kind
    else {
        return None;
    };
    (*lhs == induction).then_some(*rhs)
}

fn ceil_sqrt(value: u32) -> u32 {
    let mut low = 0u32;
    let mut high = value;
    while low < high {
        let middle = low + (high - low) / 2;
        if u64::from(middle) * u64::from(middle) >= u64::from(value) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

fn function_ir_work(func: &Function) -> Option<usize> {
    let mut work = func.blocks.len().checked_add(func.values.len())?;
    for block in &func.blocks {
        work = work.checked_add(1)?.checked_add(block.insts.len())?;
        for inst in &block.insts {
            let operands = match &inst.kind {
                InstKind::Nop | InstKind::Alloca { .. } => 0,
                InstKind::Phi { incomings } => incomings.len().checked_mul(2)?,
                InstKind::Load { .. }
                | InstKind::MemZero { .. }
                | InstKind::Unary { .. }
                | InstKind::Cast { .. } => 1,
                InstKind::Store { .. }
                | InstKind::Binary { .. }
                | InstKind::Icmp { .. }
                | InstKind::Fcmp { .. } => 2,
                InstKind::Gep { indices, .. } => indices.len().checked_add(1)?,
                InstKind::Call { args, .. } => args.len(),
            };
            work = work.checked_add(operands)?;
        }
        let terminator_work = match block.terminator.as_ref() {
            Some(Terminator::Return(Some(_))) => 1,
            Some(Terminator::Return(None)) | None => 0,
            Some(Terminator::Jump(_)) => 1,
            Some(Terminator::Branch { .. }) => 3,
        };
        work = work.checked_add(terminator_work)?;
    }
    Some(work)
}

fn function_verify_work(func: &Function) -> Option<usize> {
    function_ir_work(func)?.checked_add(func.blocks.len().checked_mul(func.blocks.len())?)
}

fn prepare_parent_split(
    func: &Function,
    candidate: &Candidate,
    budget: &mut WorkBudget,
) -> Option<(Function, BlockId)> {
    if !candidate.needs_split
        || candidate.entering_pred == candidate.header
        || func.blocks.len().checked_add(1)? > MAX_BLOCKS
    {
        return None;
    }

    let terminator = func
        .blocks
        .get(candidate.entering_pred.0)?
        .terminator
        .as_ref()?;
    budget.spend(terminator_edge_slots(terminator))?;
    if terminator_target_count(terminator, candidate.header) == 0 {
        return None;
    }
    for inst in &func.blocks.get(candidate.header.0)?.insts {
        budget.spend(1)?;
        let InstKind::Phi { incomings } = &inst.kind else {
            continue;
        };
        budget.spend(incomings.len())?;
        if incomings
            .iter()
            .filter(|(pred, _)| *pred == candidate.entering_pred)
            .count()
            != 1
        {
            return None;
        }
    }

    let clone_work = function_ir_work(func)?;
    let split_block_count = func.blocks.len().checked_add(1)?;
    let verify_work = clone_work
        .checked_add(3)?
        .checked_add(split_block_count.checked_mul(split_block_count)?)?;
    budget.spend(clone_work.checked_add(verify_work)?)?;

    let mut transaction = func.clone();
    let split = apply_preheader_split(&mut transaction, candidate.entering_pred, candidate.header)?;
    if transaction.verify().is_err() {
        return None;
    }
    Some((transaction, split))
}

fn apply_preheader_split(
    func: &mut Function,
    entering_pred: BlockId,
    header: BlockId,
) -> Option<BlockId> {
    let expected_split = BlockId(func.blocks.len());
    let split = func.add_block(format!("thread.preheader.{}", expected_split.0));
    if split != expected_split {
        return None;
    }
    func.blocks.get_mut(split.0)?.terminator = Some(Terminator::Jump(header));

    let terminator = func.blocks.get_mut(entering_pred.0)?.terminator.as_mut()?;
    let mut rewritten = 0usize;
    match terminator {
        Terminator::Return(_) => return None,
        Terminator::Jump(target) => {
            if *target == header {
                *target = split;
                rewritten = 1;
            }
        }
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => {
            if *then_target == header {
                *then_target = split;
                rewritten = rewritten.checked_add(1)?;
            }
            if *else_target == header {
                *else_target = split;
                rewritten = rewritten.checked_add(1)?;
            }
        }
    }
    if rewritten == 0 {
        return None;
    }

    for inst in &mut func.blocks.get_mut(header.0)?.insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        let mut repaired = 0usize;
        for (pred, _) in incomings {
            if *pred == entering_pred {
                *pred = split;
                repaired = repaired.checked_add(1)?;
            }
        }
        if repaired != 1 {
            return None;
        }
    }
    Some(split)
}

fn build_range_helper(
    func: &Function,
    candidate: &Candidate,
    name: String,
    budget: &mut WorkBudget,
) -> Option<Function> {
    let instruction_count = candidate
        .region_blocks
        .iter()
        .try_fold(0usize, |total, block| {
            total.checked_add(func.blocks.get(block.0)?.insts.len())
        })?;
    if candidate.region_blocks.len() > MAX_REGION_BLOCKS
        || instruction_count > MAX_REGION_INSTS
        || budget
            .spend(
                candidate
                    .region_blocks
                    .len()
                    .checked_add(instruction_count)?,
            )
            .is_none()
    {
        return None;
    }

    let mut helper = Function::new(name, Type::Void);
    let begin = helper.add_param("begin", Type::I32);
    let end = helper.add_param("end", Type::I32);
    let mut values = HashMap::new();
    for (idx, capture) in candidate.captures.iter().enumerate() {
        let parameter = helper.add_param(format!("capture{}", idx), capture.ty.clone());
        values.insert(capture.value, parameter);
    }

    // Stage one: allocate the complete block map before cloning any edge.
    let mut blocks = HashMap::new();
    for source in &candidate.region_blocks {
        let cloned = helper.add_block(format!("range.{}", func.blocks[source.0].name));
        blocks.insert(*source, cloned);
    }
    let helper_exit = helper.add_block("range.exit");
    helper.set_terminator(
        helper.entry,
        Terminator::Jump(*blocks.get(&candidate.header)?),
    );

    // Stage two: preallocate every instruction/result location. This makes all
    // forward, phi, and backedge values available before any operand remap.
    for source in &candidate.region_blocks {
        let cloned_block = *blocks.get(source)?;
        for inst in &func.blocks.get(source.0)?.insts {
            budget.spend(1)?;
            let result_ty = inst.result.map(|result| func.value(result).ty.clone());
            let cloned_result = helper.append_inst(cloned_block, InstKind::Nop, result_ty);
            if let Some(original) = inst.result {
                values.insert(original, cloned_result?);
            }
            if helper.values.len() > MAX_ADDED_VALUES {
                return None;
            }
        }
    }

    for source in &candidate.region_blocks {
        let cloned_block = *blocks.get(source)?;
        for (inst_idx, inst) in func.blocks.get(source.0)?.insts.iter().enumerate() {
            budget.spend(1)?;
            let kind = clone_inst_kind(
                func,
                &mut helper,
                candidate,
                &blocks,
                &mut values,
                begin,
                end,
                inst,
                budget,
            )?;
            helper
                .blocks
                .get_mut(cloned_block.0)?
                .insts
                .get_mut(inst_idx)?
                .kind = kind;
        }
        let terminator = clone_terminator(
            func,
            func.blocks.get(source.0)?.terminator.as_ref()?,
            *source,
            candidate,
            &blocks,
            &mut helper,
            &mut values,
            helper_exit,
            budget,
        )?;
        helper.set_terminator(cloned_block, terminator);
    }
    helper.set_terminator(helper_exit, Terminator::Return(None));
    if helper.values.len() > MAX_ADDED_VALUES {
        return None;
    }
    budget.spend(function_verify_work(&helper)?)?;
    if helper.verify().is_err() {
        return None;
    }
    Some(helper)
}

#[allow(clippy::too_many_arguments)]
fn clone_inst_kind(
    source: &Function,
    helper: &mut Function,
    candidate: &Candidate,
    blocks: &HashMap<BlockId, BlockId>,
    values: &mut HashMap<ValueId, ValueId>,
    begin: ValueId,
    end: ValueId,
    inst: &Inst,
    budget: &mut WorkBudget,
) -> Option<InstKind> {
    if inst.result == Some(candidate.counter) {
        let InstKind::Phi { incomings } = &inst.kind else {
            return None;
        };
        let mut cloned = Vec::with_capacity(incomings.len());
        for (pred, value) in incomings {
            budget.spend(1)?;
            if *pred == candidate.entering_pred {
                if const_i32(source, *value) != Some(0) {
                    return None;
                }
                cloned.push((helper.entry, begin));
            } else {
                cloned.push((
                    *blocks.get(pred)?,
                    map_value(source, helper, values, *value, budget)?,
                ));
            }
        }
        return Some(InstKind::Phi { incomings: cloned });
    }
    if inst.result == Some(candidate.condition) {
        return Some(InstKind::Icmp {
            op: CmpOp::Lt,
            lhs: *values.get(&candidate.counter)?,
            rhs: end,
        });
    }

    let mut map = |value| map_value(source, helper, values, value, budget);
    Some(match &inst.kind {
        InstKind::Nop => InstKind::Nop,
        InstKind::Phi { incomings } => InstKind::Phi {
            incomings: incomings
                .iter()
                .map(|(pred, value)| Some((*blocks.get(pred)?, map(*value)?)))
                .collect::<Option<Vec<_>>>()?,
        },
        InstKind::Load { ptr } => InstKind::Load { ptr: map(*ptr)? },
        InstKind::Store { ptr, value } => InstKind::Store {
            ptr: map(*ptr)?,
            value: map(*value)?,
        },
        InstKind::Unary { op, value } => InstKind::Unary {
            op: *op,
            value: map(*value)?,
        },
        InstKind::Binary { op, lhs, rhs } => InstKind::Binary {
            op: *op,
            lhs: map(*lhs)?,
            rhs: map(*rhs)?,
        },
        InstKind::Icmp { op, lhs, rhs } => InstKind::Icmp {
            op: *op,
            lhs: map(*lhs)?,
            rhs: map(*rhs)?,
        },
        InstKind::Fcmp { op, lhs, rhs } => InstKind::Fcmp {
            op: *op,
            lhs: map(*lhs)?,
            rhs: map(*rhs)?,
        },
        InstKind::Cast { op, value } => InstKind::Cast {
            op: *op,
            value: map(*value)?,
        },
        InstKind::Gep { base, indices } => InstKind::Gep {
            base: map(*base)?,
            indices: indices
                .iter()
                .map(|index| map(*index))
                .collect::<Option<Vec<_>>>()?,
        },
        InstKind::Alloca { .. } | InstKind::MemZero { .. } | InstKind::Call { .. } => return None,
    })
}

#[allow(clippy::too_many_arguments)]
fn clone_terminator(
    source: &Function,
    terminator: &Terminator,
    source_block: BlockId,
    candidate: &Candidate,
    blocks: &HashMap<BlockId, BlockId>,
    helper: &mut Function,
    values: &mut HashMap<ValueId, ValueId>,
    helper_exit: BlockId,
    budget: &mut WorkBudget,
) -> Option<Terminator> {
    budget.spend(1)?;
    let target = |target: BlockId| {
        if target == candidate.exit && source_block == candidate.header {
            Some(helper_exit)
        } else {
            blocks.get(&target).copied()
        }
    };
    Some(match terminator {
        Terminator::Return(_) => return None,
        Terminator::Jump(destination) => Terminator::Jump(target(*destination)?),
        Terminator::Branch {
            cond,
            then_target,
            else_target,
        } => Terminator::Branch {
            cond: map_value(source, helper, values, *cond, budget)?,
            then_target: target(*then_target)?,
            else_target: target(*else_target)?,
        },
    })
}

fn map_value(
    source: &Function,
    helper: &mut Function,
    values: &mut HashMap<ValueId, ValueId>,
    value: ValueId,
    budget: &mut WorkBudget,
) -> Option<ValueId> {
    budget.spend(1)?;
    if let Some(mapped) = values.get(&value) {
        return Some(*mapped);
    }
    if helper.values.len() >= MAX_ADDED_VALUES {
        return None;
    }
    let mapped = match &source.value(value).kind {
        ValueKind::Const(constant) => helper.add_const(constant.clone()),
        ValueKind::Global(name) => {
            helper.add_global_ref(name.clone(), source.value(value).ty.clone())
        }
        ValueKind::Param | ValueKind::Inst(_, _) => return None,
    };
    values.insert(value, mapped);
    Some(mapped)
}

fn collect_uses(func: &Function, budget: &mut WorkBudget) -> Option<Vec<Vec<UseSite>>> {
    let mut uses = vec![Vec::new(); func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let owner = BlockId(block_idx);
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings } => {
                    for (predecessor, value) in incomings {
                        budget.spend(1)?;
                        uses.get_mut(value.0)?.push(UseSite {
                            block: owner,
                            role: UseRole::Phi {
                                predecessor: *predecessor,
                                target: owner,
                            },
                        });
                    }
                }
                InstKind::Load { ptr } => {
                    budget.spend(1)?;
                    uses.get_mut(ptr.0)?.push(UseSite {
                        block: owner,
                        role: UseRole::LoadPointer,
                    });
                }
                InstKind::Store { ptr, value } => {
                    budget.spend(2)?;
                    uses.get_mut(ptr.0)?.push(UseSite {
                        block: owner,
                        role: UseRole::StorePointer,
                    });
                    uses.get_mut(value.0)?.push(UseSite {
                        block: owner,
                        role: UseRole::Other,
                    });
                }
                InstKind::Gep { base, indices } => {
                    budget.spend(indices.len().checked_add(1)?)?;
                    uses.get_mut(base.0)?.push(UseSite {
                        block: owner,
                        role: UseRole::GepBase,
                    });
                    for index in indices {
                        uses.get_mut(index.0)?.push(UseSite {
                            block: owner,
                            role: UseRole::GepIndex,
                        });
                    }
                }
                kind => {
                    for value in inst_operands(kind) {
                        budget.spend(1)?;
                        uses.get_mut(value.0)?.push(UseSite {
                            block: owner,
                            role: UseRole::Other,
                        });
                    }
                }
            }
        }
        for value in terminator_operands(block.terminator.as_ref()) {
            budget.spend(1)?;
            uses.get_mut(value.0)?.push(UseSite {
                block: owner,
                role: UseRole::Other,
            });
        }
    }
    Some(uses)
}

fn value_available_at(func: &Function, dom: &Dominators, value: ValueId, block: BlockId) -> bool {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(owner, inst_idx)) => {
            func.blocks
                .get(owner.0)
                .and_then(|source| source.insts.get(*inst_idx))
                .is_some_and(|inst| inst.result == Some(value))
                && (*owner == block || dom.dominates(*owner, block))
        }
        None => false,
    }
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&Inst> {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(inst)
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match &func.values.get(value.0)?.kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        _ => None,
    }
}

fn is_four_byte_scalar(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::F32)
}

fn instruction_cost(inst: &Inst) -> usize {
    match inst.kind {
        InstKind::Load { .. } | InstKind::Store { .. } => 3,
        InstKind::Binary {
            op: BinaryOp::Idiv | BinaryOp::Imod,
            ..
        } => 8,
        InstKind::Binary {
            op: BinaryOp::Imul | BinaryOp::Fmul,
            ..
        } => 3,
        InstKind::Gep { .. } => 1,
        InstKind::Nop | InstKind::Phi { .. } => 0,
        _ => 1,
    }
}

fn sorted_region_blocks(natural_loop: &NaturalLoop, header: BlockId) -> Vec<BlockId> {
    let mut blocks = natural_loop.blocks.iter().copied().collect::<Vec<_>>();
    blocks.sort_by_key(|block| (*block != header, block.0));
    blocks
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
        InstKind::Gep { base, indices } => {
            let mut operands = Vec::with_capacity(indices.len() + 1);
            operands.push(*base);
            operands.extend(indices.iter().copied());
            operands
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::pass::{run_pipeline, OptLevel, PassOptions};
    use crate::parser::Parser;

    fn optimize(source: &str, enabled: bool) -> Module {
        let mut parser = Parser::new(source);
        let program = parser.parse_program();
        let mut module = crate::ir::lower::lower_program(&program).unwrap();
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_aarch64_threading: enabled,
            },
        );
        module
    }

    const GLOBAL_MAP: &str = r#"
        int data[100000];
        int transform(int n, int scale) {
            int i = 0;
            while (i < n) {
                int x = data[i];
                data[i] = x * x + x * scale + 1;
                i = i + 1;
            }
            return 0;
        }
    "#;

    const DIFFERENT_ROOT_TRANSPOSE: &str = r#"
        int input[64][64];
        int output[64][64];
        int transform() {
            int i = 0;
            while (i < 64) {
                int j = 0;
                while (j < 64) {
                    output[i][j] = input[j][i] + input[j][i] * 3 + 1;
                    j = j + 1;
                }
                i = i + 1;
            }
            return 0;
        }
    "#;

    const SEQUENTIAL_LOOPS: &str = r#"
        int first[1024];
        int input[1024][1024];
        int output[1024][1024];
        int transform(int bound) {
            int i = 0;
            while (i < bound) {
                int x = first[i];
                first[i] = x * x + x * 3 + 1;
                i = i + 1;
            }
            i = 0;
            while (i < bound) {
                int j = 0;
                while (j < bound) {
                    int x = input[i][j];
                    output[i][j] = x * x + x * 3 + 1;
                    j = j + 1;
                }
                i = i + 1;
            }
            return 0;
        }
    "#;

    const MATRIX_REDUCTION: &str = r#"
        int left[64][64];
        int right[64][64];
        int result[64][64];
        int transform() {
            int i = 0;
            while (i < 64) {
                int j = 0;
                while (j < 64) {
                    int k = 0;
                    int sum = 0;
                    while (k < 64) {
                        sum = sum + left[i][k] * right[k][j];
                        k = k + 1;
                    }
                    result[i][j] = sum;
                    j = j + 1;
                }
                i = i + 1;
            }
            return 0;
        }
    "#;

    #[test]
    fn accepts_pure_header_bound_setup_from_a_distinct_read_only_global() {
        let module = optimize(
            r#"
                int limit;
                int data[100000];
                int square(int value) { return value * value; }
                int transform() {
                    int i = 0;
                    while (i < square(limit)) {
                        int x = data[i];
                        data[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert!(!module.aarch64_thread_plans[0].dispatch_setup.is_empty());
        assert!(module.funcs[module.aarch64_thread_plans[0].helper.0]
            .verify()
            .is_ok());
    }

    #[test]
    fn accepts_different_unique_global_transpose_and_verifies_full_helper() {
        let module = optimize(DIFFERENT_ROOT_TRANSPOSE, true);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        let plan = &module.aarch64_thread_plans[0];
        assert!(module.funcs[plan.parent.0].blocks.len() > 2);
        assert!(module.funcs[plan.helper.0].blocks.len() > 3);
        assert!(module.funcs[plan.helper.0].verify().is_ok());
        assert!(plan.parallel_threshold < DEFAULT_PARALLEL_THRESHOLD as i32);
    }

    #[test]
    fn lowers_work_gate_for_a_guaranteed_inner_loop_sharing_the_outer_bound() {
        let module = optimize(
            r#"
                int input[1024][1024];
                int output[1024][1024];
                int transform(int bound) {
                    int i = 0;
                    while (i < bound) {
                        int j = 0;
                        while (j < bound) {
                            output[i][j] = input[i][j] * 3 + 1;
                            j = j + 1;
                        }
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert_eq!(
            module.aarch64_thread_plans[0].parallel_threshold,
            ceil_sqrt(DEFAULT_PARALLEL_THRESHOLD) as i32
        );
    }

    #[test]
    fn does_not_lower_work_gate_for_an_inner_side_exit() {
        let module = optimize(
            r#"
                int data[100000][1];
                int guard[100000];
                int transform(int bound) {
                    int i = 0;
                    while (i < bound) {
                        int j = 0;
                        while (j < 100000) {
                            if (guard[j] != 0) break;
                            data[i][0] = data[i][0] * data[i][0] + 1;
                            j = j + 1;
                        }
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert_eq!(
            module.aarch64_thread_plans[0].parallel_threshold,
            DEFAULT_PARALLEL_THRESHOLD as i32
        );
    }

    #[test]
    fn accepts_matrix_inner_reduction_with_region_private_phis() {
        let module = optimize(MATRIX_REDUCTION, true);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        let helper = &module.funcs[module.aarch64_thread_plans[0].helper.0];
        assert!(helper.verify().is_ok());
        assert!(
            helper
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .filter(|inst| matches!(inst.kind, InstKind::Phi { .. }))
                .count()
                > 1
        );
    }

    #[test]
    fn splits_only_the_deeper_selected_virtual_preheader_and_repairs_phis() {
        let original = optimize(SEQUENTIAL_LOOPS, false);
        let mut module = optimize(SEQUENTIAL_LOOPS, true);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        let plan = &module.aarch64_thread_plans[0];
        let original_parent = &original.funcs[plan.parent.0];
        let parent = &module.funcs[plan.parent.0];
        assert_eq!(parent.blocks.len(), original_parent.blocks.len() + 1);
        assert_eq!(plan.preheader.0, original_parent.blocks.len());
        assert_eq!(
            parent.blocks[plan.preheader.0].terminator,
            Some(Terminator::Jump(plan.header))
        );
        assert!(module.funcs[plan.helper.0].verify().is_ok());
        assert!(parent.verify().is_ok());

        let original_cfg = ControlFlowGraph::new(original_parent);
        let original_dom = Dominators::new(original_parent, &original_cfg);
        let original_loops = LoopInfo::new(&original_cfg, &original_dom);
        let original_outer = original_loops
            .loops()
            .iter()
            .find(|natural_loop| natural_loop.header == plan.header)
            .unwrap();
        assert!(original_outer.dedicated_preheader.is_none());
        let entering_pred = original_outer.unique_entering_pred.unwrap();
        assert_eq!(
            terminator_target_count(
                original_parent.blocks[entering_pred.0]
                    .terminator
                    .as_ref()
                    .unwrap(),
                plan.header,
            ),
            1
        );
        assert_eq!(
            terminator_target_count(
                parent.blocks[entering_pred.0].terminator.as_ref().unwrap(),
                plan.header,
            ),
            0
        );
        assert_eq!(
            terminator_target_count(
                parent.blocks[entering_pred.0].terminator.as_ref().unwrap(),
                plan.preheader,
            ),
            1
        );

        let original_phi_incomings = original_parent.blocks[plan.header.0]
            .insts
            .iter()
            .filter_map(|inst| {
                let InstKind::Phi { incomings } = &inst.kind else {
                    return None;
                };
                Some((
                    inst.result?,
                    incomings
                        .iter()
                        .find_map(|(pred, value)| (*pred == entering_pred).then_some(*value))?,
                ))
            })
            .collect::<Vec<_>>();
        assert!(!original_phi_incomings.is_empty());
        for (result, original_value) in original_phi_incomings {
            let repaired = parent.blocks[plan.header.0]
                .insts
                .iter()
                .find(|inst| inst.result == Some(result))
                .unwrap();
            let InstKind::Phi { incomings } = &repaired.kind else {
                panic!("original header phi must remain a phi");
            };
            assert!(incomings.contains(&(plan.preheader, original_value)));
            assert!(!incomings.iter().any(|(pred, _)| *pred == entering_pred));
        }

        let cfg = ControlFlowGraph::new(parent);
        let dom = Dominators::new(parent, &cfg);
        let loops = LoopInfo::new(&cfg, &dom);
        let selected = loops
            .loops()
            .iter()
            .find(|natural_loop| natural_loop.header == plan.header)
            .unwrap();
        assert!(loops.loops().iter().any(|natural_loop| {
            natural_loop.header != selected.header
                && natural_loop.blocks.is_subset(&selected.blocks)
                && natural_loop.blocks.len() < selected.blocks.len()
        }));

        let function_count = module.funcs.len();
        let parent_id = plan.parent;
        let parent_block_count = parent.blocks.len();
        let split = plan.preheader;
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_aarch64_threading: true,
            },
        );
        assert_eq!(module.funcs.len(), function_count);
        assert_eq!(module.funcs[parent_id.0].blocks.len(), parent_block_count);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert_eq!(module.aarch64_thread_plans[0].preheader, split);
    }

    #[test]
    fn rewrites_both_branch_slots_without_changing_phi_values() {
        let mut func = Function::new("transaction", Type::I32);
        let zero = func.add_const(Const::Int(0));
        let condition = func.add_const(Const::Bool(true));
        let header = func.add_block("header");
        let exit = func.add_block("exit");
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond: condition,
                then_target: header,
                else_target: header,
            },
        );
        let phi = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(header, Terminator::Jump(exit));
        func.set_terminator(exit, Terminator::Return(Some(phi)));
        assert!(func.verify().is_ok());

        let entry = func.entry;
        let split = apply_preheader_split(&mut func, entry, header).unwrap();
        assert_eq!(
            func.blocks[entry.0].terminator,
            Some(Terminator::Branch {
                cond: condition,
                then_target: split,
                else_target: split,
            })
        );
        let InstKind::Phi { incomings } = &func.blocks[header.0].insts[0].kind else {
            panic!("header instruction must remain a phi");
        };
        assert_eq!(incomings, &vec![(split, zero)]);
        assert_eq!(
            func.blocks[split.0].terminator,
            Some(Terminator::Jump(header))
        );
        assert!(func.verify().is_ok());
    }

    #[test]
    fn accepts_multiblock_diamond_inside_partitioned_outer_region() {
        let module = optimize(
            r#"
                int input[64][64];
                int output[64][64];
                int transform() {
                    int i = 0;
                    while (i < 64) {
                        int j = 0;
                        while (j < 64) {
                            if (input[j][i] > 0) {
                                output[i][j] = input[j][i] * 3 + 1;
                            } else {
                                output[i][j] = 0 - input[j][i] * 2;
                            }
                            j = j + 1;
                        }
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert!(module.funcs[module.aarch64_thread_plans[0].helper.0]
            .verify()
            .is_ok());
    }

    #[test]
    fn rejects_same_root_cross_slice_transpose_load() {
        let module = optimize(
            r#"
                int data[64][64];
                int transform() {
                    int i = 0;
                    while (i < 64) {
                        int j = 0;
                        while (j < 64) {
                            data[i][j] = 0 - data[j][i];
                            j = j + 1;
                        }
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert!(module.aarch64_thread_plans.is_empty());
    }

    #[test]
    fn rejects_flattened_or_indirect_outer_selector() {
        for source in [
            r#"
                int data[4096];
                int transform() {
                    int i = 0;
                    while (i < 64) {
                        int j = 0;
                        while (j < 64) {
                            data[i * 64 + j] = i + j;
                            j = j + 1;
                        }
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            r#"
                int data[4096];
                int transform() {
                    int i = 0;
                    while (i < 64) {
                        int copied = i * 2;
                        data[copied] = i * i + i * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
        ] {
            assert!(optimize(source, true).aarch64_thread_plans.is_empty());
        }
    }

    #[test]
    fn rejects_pointer_parameter_and_alloca_roots() {
        for source in [
            r#"
                int transform(int data[], int n) {
                    int i = 0;
                    while (i < n) {
                        data[i] = data[i] * data[i] + data[i] * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            r#"
                int transform(int n) {
                    int data[100];
                    int i = 0;
                    while (i < n) {
                        data[i] = i * i + i * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
        ] {
            assert!(optimize(source, true).aarch64_thread_plans.is_empty());
        }
    }

    #[test]
    fn rejects_side_exit_call_and_scalar_liveout() {
        for source in [
            r#"
                int data[100000];
                int transform(int n) {
                    int i = 0;
                    while (i < n) {
                        if (data[i] < 0) return i;
                        data[i] = data[i] * data[i] + data[i] * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            r#"
                int data[100000];
                int transform(int n) {
                    int i = 0;
                    while (i < n) {
                        putint(data[i]);
                        data[i] = data[i] * data[i] + data[i] * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            r#"
                int data[100000];
                int transform(int n) {
                    int i = 0;
                    int sum = 0;
                    while (i < n) {
                        sum = sum + data[i];
                        data[i] = data[i] * data[i] + 1;
                        i = i + 1;
                    }
                    return sum;
                }
            "#,
        ] {
            assert!(optimize(source, true).aarch64_thread_plans.is_empty());
        }
    }

    #[test]
    fn rejects_offset_selector_and_unsafe_division() {
        for source in [
            r#"
                int data[100000];
                int transform(int n) {
                    int i = 0;
                    while (i < n) {
                        data[i + 1] = data[i + 1] * data[i + 1] + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            r#"
                int data[100000];
                int transform(int n) {
                    int i = 0;
                    while (i < n) {
                        data[i] = data[i] / n + data[i] * data[i] + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
        ] {
            assert!(optimize(source, true).aarch64_thread_plans.is_empty());
        }
    }

    #[test]
    fn rejected_candidate_does_not_change_any_parent_block() {
        let source = r#"
            int data[64][64];
            int transform() {
                int i = 0;
                while (i < 64) {
                    int j = 0;
                    while (j < 64) {
                        data[i][j] = 0 - data[j][i];
                        j = j + 1;
                    }
                    i = i + 1;
                }
                return 0;
            }
        "#;
        let without_threading = optimize(source, false);
        let with_threading = optimize(source, true);
        assert!(with_threading.aarch64_thread_plans.is_empty());
        assert_eq!(with_threading, without_threading);
    }

    #[test]
    fn proof_budget_rejection_keeps_the_module_byte_identical() {
        let mut module = optimize(GLOBAL_MAP, false);
        let func = &mut module.funcs[0];
        let zero = func.add_const(Const::Int(0));
        while func
            .blocks
            .len()
            .checked_mul(func.blocks.len())
            .is_some_and(|work| work <= MAX_PROOF_WORK)
        {
            let dead = func.add_block("budget.dead");
            func.set_terminator(dead, Terminator::Return(Some(zero)));
        }
        assert!(func.blocks.len() <= MAX_BLOCKS);
        let before = module.clone();
        AArch64ThreadOutlinePass::new().run(&mut module);
        assert_eq!(module, before);
    }

    #[test]
    fn keeps_helper_budget_checked() {
        let mut source = String::from(
            "int data[100000]; int transform(int n) { int i = 0; while (i < n) { int x = data[i];",
        );
        for _ in 0..(MAX_REGION_INSTS + 32) {
            source.push_str("x = x + 1;");
        }
        source.push_str("data[i] = x; i = i + 1; } return 0; }");
        let module = optimize(&source, true);
        assert!(module.aarch64_thread_plans.is_empty());
        assert_eq!(module.funcs.len(), 1);
    }

    #[test]
    fn requires_target_gate_and_is_idempotent() {
        assert!(optimize(GLOBAL_MAP, false).aarch64_thread_plans.is_empty());

        let mut parser = Parser::new(GLOBAL_MAP);
        let mut o0 = crate::ir::lower::lower_program(&parser.parse_program()).unwrap();
        run_pipeline(
            &mut o0,
            OptLevel::O0,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_aarch64_threading: true,
            },
        );
        assert!(o0.aarch64_thread_plans.is_empty());

        let mut module = optimize(GLOBAL_MAP, true);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert_eq!(
            module.aarch64_thread_plans[0].parallel_threshold,
            DEFAULT_PARALLEL_THRESHOLD as i32
        );
        let functions = module.funcs.len();
        let parent = module.aarch64_thread_plans[0].parent;
        let parent_blocks = module.funcs[parent.0].blocks.len();
        let preheader = module.aarch64_thread_plans[0].preheader;
        AArch64ThreadOutlinePass::new().run(&mut module);
        assert_eq!(module.funcs.len(), functions);
        assert_eq!(module.aarch64_thread_plans.len(), 1);

        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_aarch64_threading: true,
            },
        );
        assert_eq!(module.funcs.len(), functions);
        assert_eq!(module.funcs[parent.0].blocks.len(), parent_blocks);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert_eq!(module.aarch64_thread_plans[0].preheader, preheader);
        assert!(module.funcs[module.aarch64_thread_plans[0].helper.0]
            .verify()
            .is_ok());
    }
}
