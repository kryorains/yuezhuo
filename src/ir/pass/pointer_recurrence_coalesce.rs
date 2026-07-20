use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::util::{defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, Const, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTION_BLOCKS: usize = 1024;
const MAX_FUNCTION_VALUES: usize = 8192;
const MAX_FUNCTION_INSTRUCTIONS: usize = 32_768;
const MAX_VALUE_USES: usize = 65_536;
const MAX_RECURRENCES: usize = 128;
const MAX_PAIR_PROOFS: usize = 16_384;
const MAX_GEP_CHAIN_DEPTH: usize = 64;
const MAX_GEP_INDICES: usize = 128;
const MAX_AFFINE_DEPTH: usize = 32;
const MAX_PROOF_WORK: usize = 262_144;
const MAX_TYPE_NODES: usize = 4096;
const MAX_GLOBALS: usize = 8192;
const MAX_GLOBAL_NAME_BYTES: usize = 1_048_576;

/// Reuses one pointer recurrence for streams separated by a proven constant
/// distance. The analysis depends only on typed SSA, natural loops, and checked
/// arithmetic; source names and aggregate dimensions are never profitability
/// keys.
pub(super) struct PointerRecurrenceCoalescePass;

impl PointerRecurrenceCoalescePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for PointerRecurrenceCoalescePass {
    fn run(&mut self, module: &mut Module) {
        let global_name_bytes = module
            .globals
            .iter()
            .try_fold(0usize, |total, global| total.checked_add(global.name.len()));
        let unique_globals = if module.globals.len() <= MAX_GLOBALS
            && global_name_bytes.is_some_and(|bytes| bytes <= MAX_GLOBAL_NAME_BYTES)
        {
            let mut global_counts = HashMap::<String, usize>::new();
            for global in &module.globals {
                *global_counts.entry(global.name.clone()).or_default() += 1;
            }
            global_counts
                .into_iter()
                .enumerate()
                .filter_map(|(identity, (name, count))| (count == 1).then_some((name, identity)))
                .collect::<HashMap<_, _>>()
        } else {
            HashMap::new()
        };
        for func in &mut module.funcs {
            coalesce_function(func, &unique_globals);
        }
    }
}

#[derive(Clone, Copy)]
struct PointerRecurrence {
    phi: ValueId,
    initial: ValueId,
    update: ValueId,
    header: BlockId,
    latch: BlockId,
    step: i32,
}

#[derive(PartialEq, Eq)]
enum AddressRoot {
    Value(ValueId),
    Global(usize),
}

struct AddressExpr {
    root: AddressRoot,
    dynamic_terms: HashMap<ValueId, i64>,
    constant_offset: i64,
}

struct AffineIndex {
    dynamic_term: Option<ValueId>,
    constant: i64,
}

#[derive(Clone, Copy)]
enum UseSite {
    PhiIncoming { phi: Option<ValueId>, pred: BlockId },
    LoadPtr { block: BlockId, inst_idx: usize },
    StorePtr { block: BlockId, inst_idx: usize },
    GepBase { result: Option<ValueId> },
    Other,
}

struct CoalescePlan {
    primary: ValueId,
    secondary: ValueId,
    header: BlockId,
    distance_index: i32,
}

#[derive(Default)]
struct ProofBudget {
    work: usize,
    type_nodes: usize,
    exceeded: bool,
}

impl ProofBudget {
    fn spend(&mut self, amount: usize) -> bool {
        let Some(next) = self.work.checked_add(amount) else {
            self.exceeded = true;
            return false;
        };
        self.work = next;
        if self.work > MAX_PROOF_WORK {
            self.exceeded = true;
            return false;
        }
        true
    }

    fn spend_type(&mut self) -> bool {
        let Some(next) = self.type_nodes.checked_add(1) else {
            self.exceeded = true;
            return false;
        };
        self.type_nodes = next;
        if self.type_nodes > MAX_TYPE_NODES {
            self.exceeded = true;
            return false;
        }
        self.spend(1)
    }
}

fn coalesce_function(func: &mut Function, unique_globals: &HashMap<String, usize>) {
    let Some(plans) = plan_coalescing(func, unique_globals) else {
        return;
    };
    if plans.is_empty() {
        return;
    }

    apply_plans(func, &plans);
    if let Err(errors) = func.verify() {
        panic!(
            "pointer recurrence coalescing produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

/// Returns `None` for any function-wide size or proof-budget failure. No IR is
/// mutated until every selected replacement has been proved and budgeted.
fn plan_coalescing(
    func: &Function,
    unique_globals: &HashMap<String, usize>,
) -> Option<Vec<CoalescePlan>> {
    if func.blocks.len() > MAX_FUNCTION_BLOCKS || func.values.len() > MAX_FUNCTION_VALUES {
        return None;
    }
    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))?;
    if instruction_count > MAX_FUNCTION_INSTRUCTIONS {
        return None;
    }

    let mut budget = ProofBudget::default();
    let uses = collect_uses(func, &mut budget)?;
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let loops_by_header = loop_info
        .loops()
        .iter()
        .map(|natural_loop| (natural_loop.header, natural_loop))
        .collect::<HashMap<_, _>>();
    let induction_ranges = collect_induction_congruence_ranges(func, &loop_info, &mut budget);
    if budget.exceeded {
        return None;
    }

    let mut recurrences = Vec::new();
    for natural_loop in loop_info.loops() {
        let (Some(entering), Some(latch)) = (
            natural_loop.unique_entering_pred,
            natural_loop.unique_latch(),
        ) else {
            continue;
        };
        for inst in &func.blocks.get(natural_loop.header.0)?.insts {
            let Some(phi) = inst.result else {
                continue;
            };
            let Some(recurrence) =
                analyze_pointer_recurrence(func, natural_loop, entering, latch, phi, &mut budget)
            else {
                continue;
            };
            recurrences.push(recurrence);
            if recurrences.len() > MAX_RECURRENCES {
                return None;
            }
        }
    }
    if budget.exceeded || recurrences.len() < 2 {
        return Some(Vec::new());
    }
    recurrences.sort_by_key(|recurrence| recurrence.phi.0);

    let mut addresses = HashMap::new();
    for recurrence in &recurrences {
        if let Some(address) = analyze_initial_address(
            func,
            recurrence.initial,
            &induction_ranges,
            unique_globals,
            &mut budget,
        ) {
            addresses.insert(recurrence.phi, address);
        }
    }
    if budget.exceeded {
        return None;
    }

    let mut plans = Vec::new();
    let mut secondaries = HashSet::new();
    let mut pair_proofs = 0usize;
    for (primary_idx, primary) in recurrences.iter().enumerate() {
        if secondaries.contains(&primary.phi) {
            continue;
        }
        let Some(primary_address) = addresses.get(&primary.phi) else {
            continue;
        };
        let primary_ty = &func.values.get(primary.phi.0)?.ty;
        let Type::Ptr(primary_pointee) = primary_ty else {
            continue;
        };
        let Some(element_size) = checked_type_size(primary_pointee, &mut budget) else {
            continue;
        };
        if element_size <= 0 {
            continue;
        }

        for secondary in recurrences.iter().skip(primary_idx + 1) {
            pair_proofs = pair_proofs.checked_add(1)?;
            if pair_proofs > MAX_PAIR_PROOFS || !budget.spend(1) {
                return None;
            }
            if secondaries.contains(&secondary.phi)
                || primary.header != secondary.header
                || primary.latch != secondary.latch
                || primary.step != secondary.step
                || !types_equal_bounded(
                    primary_ty,
                    &func.values.get(secondary.phi.0)?.ty,
                    &mut budget,
                )
                || !secondary_uses_are_foldable(
                    func,
                    secondary,
                    loops_by_header.get(&secondary.header).copied()?,
                    &uses,
                    &mut budget,
                )
            {
                continue;
            }
            let Some(secondary_address) = addresses.get(&secondary.phi) else {
                continue;
            };
            let term_comparison_work = primary_address
                .dynamic_terms
                .len()
                .checked_add(secondary_address.dynamic_terms.len())?;
            if !budget.spend(term_comparison_work) {
                return None;
            }
            if primary_address.root != secondary_address.root
                || primary_address.dynamic_terms != secondary_address.dynamic_terms
            {
                continue;
            }
            let Some(distance_bytes) = secondary_address
                .constant_offset
                .checked_sub(primary_address.constant_offset)
            else {
                continue;
            };
            if distance_bytes % element_size != 0 {
                continue;
            }
            let Ok(distance_index) = i32::try_from(distance_bytes / element_size) else {
                continue;
            };
            plans.push(CoalescePlan {
                primary: primary.phi,
                secondary: secondary.phi,
                header: primary.header,
                distance_index,
            });
            secondaries.insert(secondary.phi);
        }
    }
    if budget.exceeded || plans.len() > MAX_RECURRENCES {
        return None;
    }
    Some(plans)
}

fn analyze_pointer_recurrence(
    func: &Function,
    natural_loop: &NaturalLoop,
    entering: BlockId,
    latch: BlockId,
    phi: ValueId,
    budget: &mut ProofBudget,
) -> Option<PointerRecurrence> {
    if !budget.spend(1) {
        return None;
    }
    let value = func.values.get(phi.0)?;
    let Type::Ptr(_) = &value.ty else {
        return None;
    };
    let ValueKind::Inst(owner, inst_idx) = value.kind else {
        return None;
    };
    if owner != natural_loop.header {
        return None;
    }
    let inst = func.blocks.get(owner.0)?.insts.get(inst_idx)?;
    if inst.result != Some(phi) {
        return None;
    }
    let InstKind::Phi { incomings } = &inst.kind else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    let initial = unique_incoming(incomings, entering)?;
    let update = unique_incoming(incomings, latch)?;
    if !types_equal_bounded(&value.ty, &func.values.get(initial.0)?.ty, budget)
        || !types_equal_bounded(&value.ty, &func.values.get(update.0)?.ty, budget)
    {
        return None;
    }

    let ValueKind::Inst(update_block, update_idx) = func.values.get(update.0)?.kind else {
        return None;
    };
    if update_block != latch {
        return None;
    }
    let update_inst = func.blocks.get(update_block.0)?.insts.get(update_idx)?;
    if update_inst.result != Some(update) {
        return None;
    }
    let InstKind::Gep { base, indices } = &update_inst.kind else {
        return None;
    };
    let [index] = indices.as_slice() else {
        return None;
    };
    if *base != phi || func.values.get(index.0)?.ty != Type::I32 {
        return None;
    }
    typed_gep_byte_strides(func, *base, update, 1, budget)?;
    let step = constant_i32(func, *index)?;
    if step == 0 {
        return None;
    }

    Some(PointerRecurrence {
        phi,
        initial,
        update,
        header: natural_loop.header,
        latch,
        step,
    })
}

fn analyze_initial_address(
    func: &Function,
    initial: ValueId,
    induction_ranges: &HashMap<ValueId, (i64, i64)>,
    unique_globals: &HashMap<String, usize>,
    budget: &mut ProofBudget,
) -> Option<AddressExpr> {
    let mut current = initial;
    let mut dynamic_terms = HashMap::<ValueId, i64>::new();
    let mut constant_offset = 0i64;
    let mut chain_depth = 0usize;
    let mut index_count = 0usize;

    loop {
        if !budget.spend(1) {
            return None;
        }
        let Some(InstKind::Gep { base, indices }) = defining_inst(func, current) else {
            break;
        };
        chain_depth = chain_depth.checked_add(1)?;
        index_count = index_count.checked_add(indices.len())?;
        if chain_depth > MAX_GEP_CHAIN_DEPTH || index_count > MAX_GEP_INDICES || indices.is_empty()
        {
            return None;
        }
        let strides = typed_gep_byte_strides(func, *base, current, indices.len(), budget)?;
        for (index, stride) in indices.iter().copied().zip(strides) {
            let affine = analyze_affine_index(func, index, induction_ranges, budget, 0)?;
            if let Some(term) = affine.dynamic_term {
                let coefficient = dynamic_terms.entry(term).or_default();
                *coefficient = coefficient.checked_add(stride)?;
                if *coefficient == 0 {
                    dynamic_terms.remove(&term);
                }
            }
            constant_offset = constant_offset.checked_add(affine.constant.checked_mul(stride)?)?;
        }
        current = *base;
    }

    let root_value = func.values.get(current.0)?;
    if !matches!(root_value.ty, Type::Ptr(_)) {
        return None;
    }
    let root = match &root_value.kind {
        ValueKind::Global(name) if unique_globals.contains_key(name) => {
            AddressRoot::Global(*unique_globals.get(name)?)
        }
        _ => AddressRoot::Value(current),
    };
    Some(AddressExpr {
        root,
        dynamic_terms,
        constant_offset,
    })
}

/// Extracts only `dynamic + constant` forms whose signed i32 additions or
/// subtractions cannot wrap. On a failed nowrap proof the complete SSA value is
/// retained as one opaque dynamic term, so equal expressions remain usable.
fn analyze_affine_index(
    func: &Function,
    value: ValueId,
    induction_ranges: &HashMap<ValueId, (i64, i64)>,
    budget: &mut ProofBudget,
    depth: usize,
) -> Option<AffineIndex> {
    if depth > MAX_AFFINE_DEPTH || !budget.spend(1) || func.values.get(value.0)?.ty != Type::I32 {
        return None;
    }
    if let Some(constant) = constant_i32(func, value) {
        return Some(AffineIndex {
            dynamic_term: None,
            constant: i64::from(constant),
        });
    }

    let extracted = match defining_inst(func, value) {
        Some(InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        }) => constant_i32(func, *rhs)
            .map(|constant| (*lhs, i64::from(constant)))
            .or_else(|| constant_i32(func, *lhs).map(|constant| (*rhs, i64::from(constant)))),
        Some(InstKind::Binary {
            op: BinaryOp::Isub,
            lhs,
            rhs,
        }) => constant_i32(func, *rhs).map(|constant| (*lhs, -i64::from(constant))),
        _ => None,
    };
    if let Some((base, added_constant)) = extracted {
        if let Some(mut affine) =
            analyze_affine_index(func, base, induction_ranges, budget, depth + 1)
        {
            if let Some(constant) = affine.constant.checked_add(added_constant) {
                affine.constant = constant;
                if affine_is_i32_nowrap(&affine, induction_ranges) {
                    return Some(affine);
                }
            }
        }
    }

    Some(AffineIndex {
        dynamic_term: Some(value),
        constant: 0,
    })
}

fn affine_is_i32_nowrap(
    affine: &AffineIndex,
    induction_ranges: &HashMap<ValueId, (i64, i64)>,
) -> bool {
    let (minimum, maximum) = match affine.dynamic_term {
        None => (0, 0),
        Some(term) => match induction_ranges.get(&term) {
            Some(range) => *range,
            None => return affine.constant == 0,
        },
    };
    minimum
        .checked_add(affine.constant)
        .zip(maximum.checked_add(affine.constant))
        .is_some_and(|(minimum, maximum)| {
            minimum >= i64::from(i32::MIN) && maximum <= i64::from(i32::MAX)
        })
}

/// Every wrapping i32 addrec remains in one congruence class modulo
/// `gcd(abs(step), 2^32)`. The signed extrema of that class are enough to prove
/// small lane offsets such as `iv + 1` without assuming a loop dimension.
fn collect_induction_congruence_ranges(
    func: &Function,
    loop_info: &LoopInfo,
    budget: &mut ProofBudget,
) -> HashMap<ValueId, (i64, i64)> {
    let mut ranges = HashMap::new();
    for natural_loop in loop_info.loops() {
        for inst in &func.blocks[natural_loop.header.0].insts {
            if !budget.spend(1) {
                return ranges;
            }
            let Some(phi) = inst.result else {
                continue;
            };
            let Some(induction) = analyze_i32_induction(func, natural_loop, phi) else {
                continue;
            };
            let Some(initial) = constant_i32(func, induction.initial) else {
                continue;
            };
            let modulus = gcd_u64(induction.step.unsigned_abs() as u64, 1u64 << 32) as i64;
            if modulus == 0 {
                continue;
            }
            let initial = i64::from(initial);
            let signed_minimum = i64::from(i32::MIN);
            let signed_maximum = i64::from(i32::MAX);
            let minimum = signed_minimum + (initial - signed_minimum).rem_euclid(modulus);
            let maximum = signed_maximum - (signed_maximum - initial).rem_euclid(modulus);
            ranges.insert(phi, (minimum, maximum));
        }
    }
    ranges
}

fn gcd_u64(mut lhs: u64, mut rhs: u64) -> u64 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn secondary_uses_are_foldable(
    func: &Function,
    recurrence: &PointerRecurrence,
    natural_loop: &NaturalLoop,
    uses: &[Vec<UseSite>],
    budget: &mut ProofBudget,
) -> bool {
    let Some(phi_uses) = uses.get(recurrence.phi.0) else {
        return false;
    };
    let mut update_base_uses = 0usize;
    let mut memory_uses = 0usize;
    for site in phi_uses {
        if !budget.spend(1) {
            return false;
        }
        match *site {
            UseSite::GepBase { result } if result == Some(recurrence.update) => {
                update_base_uses += 1;
            }
            UseSite::LoadPtr { block, inst_idx }
                if natural_loop.blocks.contains(&block)
                    && typed_memory_use(func, block, inst_idx, recurrence.phi) =>
            {
                memory_uses += 1;
            }
            UseSite::StorePtr { block, inst_idx }
                if natural_loop.blocks.contains(&block)
                    && typed_memory_use(func, block, inst_idx, recurrence.phi) =>
            {
                memory_uses += 1;
            }
            _ => return false,
        }
    }
    if update_base_uses != 1 || memory_uses == 0 {
        return false;
    }

    let Some(update_uses) = uses.get(recurrence.update.0) else {
        return false;
    };
    matches!(
        update_uses.as_slice(),
        [UseSite::PhiIncoming { phi: Some(phi), pred }]
            if *phi == recurrence.phi && *pred == recurrence.latch
    )
}

fn typed_memory_use(func: &Function, block: BlockId, inst_idx: usize, ptr: ValueId) -> bool {
    let Some(Type::Ptr(pointee)) = func.values.get(ptr.0).map(|value| &value.ty) else {
        return false;
    };
    let Some(inst) = func
        .blocks
        .get(block.0)
        .and_then(|block| block.insts.get(inst_idx))
    else {
        return false;
    };
    match &inst.kind {
        InstKind::Load { ptr: actual } if *actual == ptr => inst
            .result
            .and_then(|result| func.values.get(result.0))
            .is_some_and(|result| &result.ty == pointee.as_ref()),
        InstKind::Store { ptr: actual, value } if *actual == ptr => func
            .values
            .get(value.0)
            .is_some_and(|value| &value.ty == pointee.as_ref()),
        _ => false,
    }
}

fn collect_uses(func: &Function, budget: &mut ProofBudget) -> Option<Vec<Vec<UseSite>>> {
    let mut uses = vec![Vec::new(); func.values.len()];
    let mut use_count = 0usize;
    let mut record = |value: ValueId, site: UseSite| -> Option<()> {
        use_count = use_count.checked_add(1)?;
        if use_count > MAX_VALUE_USES || !budget.spend(1) {
            return None;
        }
        uses.get_mut(value.0)?.push(site);
        Some(())
    };

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let owner = BlockId(block_idx);
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            match &inst.kind {
                InstKind::Nop | InstKind::Alloca { .. } => {}
                InstKind::Phi { incomings } => {
                    for (pred, incoming) in incomings {
                        record(
                            *incoming,
                            UseSite::PhiIncoming {
                                phi: inst.result,
                                pred: *pred,
                            },
                        )?;
                    }
                }
                InstKind::Load { ptr } => record(
                    *ptr,
                    UseSite::LoadPtr {
                        block: owner,
                        inst_idx,
                    },
                )?,
                InstKind::Store { ptr, value } => {
                    record(
                        *ptr,
                        UseSite::StorePtr {
                            block: owner,
                            inst_idx,
                        },
                    )?;
                    record(*value, UseSite::Other)?;
                }
                InstKind::MemZero { ptr, .. }
                | InstKind::Unary { value: ptr, .. }
                | InstKind::Cast { value: ptr, .. } => record(*ptr, UseSite::Other)?,
                InstKind::Binary { lhs, rhs, .. }
                | InstKind::Icmp { lhs, rhs, .. }
                | InstKind::Fcmp { lhs, rhs, .. } => {
                    record(*lhs, UseSite::Other)?;
                    record(*rhs, UseSite::Other)?;
                }
                InstKind::Gep { base, indices } => {
                    record(
                        *base,
                        UseSite::GepBase {
                            result: inst.result,
                        },
                    )?;
                    for index in indices {
                        record(*index, UseSite::Other)?;
                    }
                }
                InstKind::Call { args, .. } => {
                    for arg in args {
                        record(*arg, UseSite::Other)?;
                    }
                }
            }
        }
        match block.terminator.as_ref() {
            Some(Terminator::Return(Some(value))) => record(*value, UseSite::Other)?,
            Some(Terminator::Branch { cond, .. }) => record(*cond, UseSite::Other)?,
            Some(Terminator::Return(None) | Terminator::Jump(_)) | None => {}
        }
    }
    Some(uses)
}

fn apply_plans(func: &mut Function, plans: &[CoalescePlan]) {
    let mut replacements = HashMap::<ValueId, ValueId>::new();
    let mut inserted_per_header = HashMap::<BlockId, usize>::new();
    for plan in plans {
        let replacement = if plan.distance_index == 0 {
            plan.primary
        } else {
            let index = get_or_add_i32_const(func, plan.distance_index);
            let inserted = inserted_per_header.entry(plan.header).or_default();
            let first_non_phi = func.blocks[plan.header.0]
                .insts
                .iter()
                .position(|inst| !matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }))
                .unwrap_or(func.blocks[plan.header.0].insts.len());
            let result_ty = func.values[plan.primary.0].ty.clone();
            let pointer = func
                .insert_inst(
                    plan.header,
                    first_non_phi + *inserted,
                    InstKind::Gep {
                        base: plan.primary,
                        indices: vec![index],
                    },
                    Some(result_ty),
                )
                .expect("a coalesced pointer offset must produce a value");
            *inserted += 1;
            pointer
        };
        replacements.insert(plan.secondary, replacement);
    }

    // Replace only the proved memory-pointer observations. Keeping each old
    // phi/update edge intact leaves a dead SCC for mark-and-sweep DCE.
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            match &mut inst.kind {
                InstKind::Load { ptr } | InstKind::Store { ptr, .. } => {
                    if let Some(replacement) = replacements.get(ptr) {
                        *ptr = *replacement;
                    }
                }
                _ => {}
            }
        }
    }
}

fn typed_gep_byte_strides(
    func: &Function,
    base: ValueId,
    result: ValueId,
    index_count: usize,
    budget: &mut ProofBudget,
) -> Option<Vec<i64>> {
    let Type::Ptr(base_pointee) = &func.values.get(base.0)?.ty else {
        return None;
    };
    let Type::Ptr(result_pointee) = &func.values.get(result.0)?.ty else {
        return None;
    };
    if index_count == 0 {
        return None;
    }
    if index_count == 1 {
        let same_pointee = types_equal_bounded(base_pointee, result_pointee, budget);
        let array_element = match base_pointee.as_ref() {
            Type::Array { elem, .. } => types_equal_bounded(elem, result_pointee, budget),
            _ => false,
        };
        if !same_pointee && !array_element {
            return None;
        }
        return Some(vec![checked_type_size(result_pointee, budget)?.max(1)]);
    }

    let mut current_ty = &func.values.get(base.0)?.ty;
    let mut strides = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        if !budget.spend(1) {
            return None;
        }
        current_ty = match current_ty {
            Type::Ptr(inner) => inner,
            Type::Array { elem, .. } => elem,
            Type::Void | Type::I1 | Type::I32 | Type::F32 => return None,
        };
        strides.push(checked_type_size(current_ty, budget)?.max(1));
    }
    if !types_equal_bounded(current_ty, result_pointee, budget) {
        return None;
    }
    Some(strides)
}

fn checked_type_size(ty: &Type, budget: &mut ProofBudget) -> Option<i64> {
    let mut current = ty;
    let mut elements = 1i64;
    loop {
        if !budget.spend_type() {
            return None;
        }
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
                elements = elements.checked_mul(i64::try_from(*len).ok()?)?;
                if elements > i64::from(i32::MAX) {
                    return None;
                }
                current = elem;
            }
        }
    }
}

fn types_equal_bounded(lhs: &Type, rhs: &Type, budget: &mut ProofBudget) -> bool {
    let mut worklist = vec![(lhs, rhs)];
    while let Some((lhs, rhs)) = worklist.pop() {
        if !budget.spend_type() {
            return false;
        }
        match (lhs, rhs) {
            (Type::Void, Type::Void)
            | (Type::I1, Type::I1)
            | (Type::I32, Type::I32)
            | (Type::F32, Type::F32) => {}
            (Type::Ptr(lhs), Type::Ptr(rhs)) => worklist.push((lhs, rhs)),
            (
                Type::Array {
                    elem: lhs,
                    len: lhs_len,
                },
                Type::Array {
                    elem: rhs,
                    len: rhs_len,
                },
            ) if lhs_len == rhs_len => worklist.push((lhs, rhs)),
            _ => return false,
        }
    }
    true
}

fn constant_i32(func: &Function, value: ValueId) -> Option<i32> {
    match func.values.get(value.0)? {
        crate::ir::Value {
            ty: Type::I32,
            kind: ValueKind::Const(Const::Int(value)),
            ..
        } => Some(*value),
        crate::ir::Value {
            ty: Type::I32,
            kind: ValueKind::Const(Const::Zero(Type::I32)),
            ..
        } => Some(0),
        _ => None,
    }
}

fn unique_incoming(incomings: &[(BlockId, ValueId)], pred: BlockId) -> Option<ValueId> {
    let mut values = incomings
        .iter()
        .filter_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value));
    let value = values.next()?;
    values.next().is_none().then_some(value)
}
