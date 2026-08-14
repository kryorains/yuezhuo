use super::util::{
    const_i32, defining_inst, get_or_add_i32_const, rewrite_function_uses, ValueReplacements,
};
use super::ModulePass;
use crate::ir::{BinaryOp, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId};
use std::collections::{HashMap, HashSet};

/// Canonicalizes and combines local integer instructions.
///
/// Every rule is based only on an instruction and the definitions of its
/// operands. In particular, this pass does not inspect source names, block
/// names, or whole-function control-flow shapes.
pub(super) struct InstCombinePass {
    divisibility_only: bool,
}

impl InstCombinePass {
    pub(super) fn new() -> Self {
        Self {
            divisibility_only: false,
        }
    }

    pub(super) fn divisibility_only() -> Self {
        Self {
            divisibility_only: true,
        }
    }
}

impl ModulePass for InstCombinePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            if self.divisibility_only {
                combine_divisibility_remainders(func);
                if let Err(errors) = func.verify() {
                    panic!(
                        "divisibility combining produced invalid IR in {}: {:?}",
                        func.name, errors
                    );
                }
            } else {
                combine_function(func);
            }
        }
    }
}

fn combine_function(func: &mut Function) {
    combine_divisibility_remainders(func);
    combine_adjacent_equality_sets(func);
    combine_linear_integer_expressions(func);

    // Reassociation can expose another constant-bearing definition, so keep
    // scanning until every local expression reaches its canonical form.
    loop {
        let mut changed = false;
        for block_idx in 0..func.blocks.len() {
            for inst_idx in 0..func.blocks[block_idx].insts.len() {
                let old_kind = func.blocks[block_idx].insts[inst_idx].kind.clone();
                let new_kind = combine_inst(func, &old_kind);
                if new_kind != old_kind {
                    func.blocks[block_idx].insts[inst_idx].kind = new_kind;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    if let Err(errors) = func.verify() {
        panic!(
            "instruction combining produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

/// Reassociates wrapping i32 additions, subtractions, negations, and products
/// by constants into a canonical linear expression.
///
/// This is the small integer subset of the Reassociate/InstCombine algebra
/// implemented by mature compilers.  It deliberately treats loads, calls,
/// phis, comparisons, and non-linear products as opaque leaves.  A fixed
/// function-wide work budget bounds expansion of shared SSA DAGs, and a form
/// is rebuilt only when the instructions made dead by the replacement
/// outnumber the new arithmetic instructions.
fn combine_linear_integer_expressions(func: &mut Function) {
    const MAX_VALUES: usize = 16_384;
    const MAX_INSTRUCTIONS: usize = 65_536;
    const MAX_EXPRESSION_OCCURRENCES: usize = 128;
    const MAX_EXPRESSION_TERMS: usize = 64;
    const MAX_TOTAL_WORK: usize = 262_144;

    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    if func.values.len() > MAX_VALUES
        || instruction_count.is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return;
    }

    let consumers = collect_value_consumers(func);
    let mut roots = Vec::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let Some(result) = inst.result else {
                continue;
            };
            if func.value(result).ty != Type::I32 || !is_linear_integer_inst(func, &inst.kind) {
                continue;
            }
            let has_non_linear_use = consumers[result.0].iter().any(|consumer| {
                consumer.is_none_or(|consumer| {
                    defining_inst(func, consumer)
                        .is_none_or(|kind| !is_linear_integer_inst(func, kind))
                })
            });
            if has_non_linear_use {
                roots.push(result);
            }
        }
    }

    // Rewriting a root can change the profitability of a shared expression.
    // Process in a stable reverse order and recompute the small local facts for
    // each root from the current IR.
    roots.sort_by_key(|value| value.0);
    let mut total_work = MAX_TOTAL_WORK;
    for root in roots.into_iter().rev() {
        if total_work == 0 || !is_current_linear_result(func, root) {
            continue;
        }
        let mut form = LinearForm::default();
        let mut expression_work = MAX_EXPRESSION_OCCURRENCES.min(total_work);
        let mut summaries = HashMap::new();
        let mut visiting = HashSet::new();
        if !collect_linear_form(
            func,
            root,
            1,
            &mut form,
            &mut expression_work,
            MAX_EXPRESSION_TERMS,
            &mut summaries,
            &mut visiting,
        ) {
            total_work = total_work.saturating_sub(MAX_EXPRESSION_OCCURRENCES - expression_work);
            continue;
        }
        total_work = total_work.saturating_sub(MAX_EXPRESSION_OCCURRENCES - expression_work);
        form.terms.retain(|_, coefficient| *coefficient != 0);

        let removable = removable_linear_nodes(func, root, &form.nodes);
        let replacement_cost = linear_form_cost(&form);
        if replacement_cost >= removable.len() {
            continue;
        }

        let Some(replacement) = build_linear_form_before_root(func, root, &form) else {
            continue;
        };
        rewrite_function_uses(func, &ValueReplacements::from([(root, replacement)]));
        if let crate::ir::ValueKind::Inst(block, inst_idx) = func.value(root).kind {
            let inst = &mut func.blocks[block.0].insts[inst_idx];
            inst.result = None;
            inst.kind = InstKind::Nop;
        }
    }
}

#[derive(Clone, Default)]
struct LinearForm {
    terms: HashMap<ValueId, i32>,
    constant: i32,
    nodes: HashSet<ValueId>,
}

fn collect_linear_form(
    func: &Function,
    value: ValueId,
    coefficient: i32,
    form: &mut LinearForm,
    work: &mut usize,
    max_terms: usize,
    summaries: &mut HashMap<ValueId, Option<LinearForm>>,
    visiting: &mut HashSet<ValueId>,
) -> bool {
    if coefficient == 0 {
        return true;
    }
    let Some(summary) = summarize_linear_value(func, value, work, max_terms, summaries, visiting)
    else {
        return false;
    };
    merge_scaled_linear_form(form, &summary, coefficient, max_terms)
}

/// Computes each SSA sub-DAG summary once.  Reassociation commonly sees
/// diamond-shaped expression DAGs after inlining; recursively expanding every
/// occurrence can otherwise consume exponential work even when the number of
/// distinct instructions is small.
fn summarize_linear_value(
    func: &Function,
    value: ValueId,
    work: &mut usize,
    max_terms: usize,
    summaries: &mut HashMap<ValueId, Option<LinearForm>>,
    visiting: &mut HashSet<ValueId>,
) -> Option<LinearForm> {
    if let Some(summary) = summaries.get(&value) {
        return summary.clone();
    }
    let Some(next_work) = work.checked_sub(1) else {
        return None;
    };
    *work = next_work;

    if let Some(constant) = const_i32(func, value) {
        return Some(LinearForm {
            constant,
            ..LinearForm::default()
        });
    }
    if func.value(value).ty != Type::I32 {
        let mut form = LinearForm::default();
        add_linear_term(&mut form, value, 1, max_terms).then_some(form)
    } else if !visiting.insert(value) {
        None
    } else {
        let mut form = LinearForm::default();
        let success = match defining_inst(func, value) {
            Some(InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            }) => {
                form.nodes.insert(value);
                collect_linear_form(
                    func, *lhs, 1, &mut form, work, max_terms, summaries, visiting,
                ) && collect_linear_form(
                    func, *rhs, 1, &mut form, work, max_terms, summaries, visiting,
                )
            }
            Some(InstKind::Binary {
                op: BinaryOp::Isub,
                lhs,
                rhs,
            }) => {
                form.nodes.insert(value);
                collect_linear_form(
                    func, *lhs, 1, &mut form, work, max_terms, summaries, visiting,
                ) && collect_linear_form(
                    func, *rhs, -1, &mut form, work, max_terms, summaries, visiting,
                )
            }
            Some(InstKind::Unary {
                op: crate::ir::UnaryOp::Ineg,
                value: operand,
            }) => {
                form.nodes.insert(value);
                collect_linear_form(
                    func, *operand, -1, &mut form, work, max_terms, summaries, visiting,
                )
            }
            Some(InstKind::Binary {
                op: BinaryOp::Imul,
                lhs,
                rhs,
            }) => {
                let scaled = match (const_i32(func, *lhs), const_i32(func, *rhs)) {
                    (Some(scale), None) => Some((*rhs, scale)),
                    (None, Some(scale)) => Some((*lhs, scale)),
                    (Some(lhs), Some(rhs)) => {
                        form.nodes.insert(value);
                        form.constant = lhs.wrapping_mul(rhs);
                        None
                    }
                    (None, None) => {
                        add_linear_term(&mut form, value, 1, max_terms);
                        None
                    }
                };
                if let Some((operand, scale)) = scaled {
                    form.nodes.insert(value);
                    collect_linear_form(
                        func, operand, scale, &mut form, work, max_terms, summaries, visiting,
                    )
                } else {
                    true
                }
            }
            _ => add_linear_term(&mut form, value, 1, max_terms),
        };
        visiting.remove(&value);
        let summary = success.then_some(form);
        summaries.insert(value, summary.clone());
        summary
    }
}

fn merge_scaled_linear_form(
    destination: &mut LinearForm,
    source: &LinearForm,
    coefficient: i32,
    max_terms: usize,
) -> bool {
    let new_terms = source
        .terms
        .keys()
        .filter(|value| !destination.terms.contains_key(value))
        .count();
    if destination.terms.len().saturating_add(new_terms) > max_terms {
        return false;
    }
    destination.constant = destination
        .constant
        .wrapping_add(source.constant.wrapping_mul(coefficient));
    destination.nodes.extend(source.nodes.iter().copied());
    for (value, source_coefficient) in &source.terms {
        let combined = destination
            .terms
            .get(value)
            .copied()
            .unwrap_or(0)
            .wrapping_add(source_coefficient.wrapping_mul(coefficient));
        destination.terms.insert(*value, combined);
    }
    true
}

fn add_linear_term(
    form: &mut LinearForm,
    value: ValueId,
    coefficient: i32,
    max_terms: usize,
) -> bool {
    if !form.terms.contains_key(&value) && form.terms.len() >= max_terms {
        return false;
    }
    let combined = form
        .terms
        .get(&value)
        .copied()
        .unwrap_or(0)
        .wrapping_add(coefficient);
    form.terms.insert(value, combined);
    true
}

fn is_linear_integer_inst(func: &Function, kind: &InstKind) -> bool {
    match kind {
        InstKind::Binary {
            op: BinaryOp::Iadd | BinaryOp::Isub,
            ..
        }
        | InstKind::Unary {
            op: crate::ir::UnaryOp::Ineg,
            ..
        } => true,
        InstKind::Binary {
            op: BinaryOp::Imul,
            lhs,
            rhs,
        } => const_i32(func, *lhs).is_some() || const_i32(func, *rhs).is_some(),
        _ => false,
    }
}

fn is_current_linear_result(func: &Function, value: ValueId) -> bool {
    func.value(value).ty == Type::I32
        && defining_inst(func, value).is_some_and(|kind| is_linear_integer_inst(func, kind))
}

fn collect_value_consumers(func: &Function) -> Vec<Vec<Option<ValueId>>> {
    let mut consumers = vec![Vec::new(); func.values.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            for operand in inst_operands(&inst.kind) {
                consumers[operand.0].push(inst.result);
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_operands(terminator) {
                consumers[operand.0].push(None);
            }
        }
    }
    consumers
}

fn removable_linear_nodes(
    func: &Function,
    root: ValueId,
    nodes: &HashSet<ValueId>,
) -> HashSet<ValueId> {
    let consumers = collect_value_consumers(func);
    let mut required = HashSet::new();
    let mut worklist = Vec::new();
    for node in nodes.iter().copied().filter(|node| *node != root) {
        if consumers[node.0]
            .iter()
            .any(|consumer| consumer.is_none_or(|consumer| !nodes.contains(&consumer)))
        {
            required.insert(node);
            worklist.push(node);
        }
    }
    while let Some(node) = worklist.pop() {
        let Some(kind) = defining_inst(func, node) else {
            continue;
        };
        for operand in inst_operands(kind) {
            if nodes.contains(&operand) && required.insert(operand) {
                worklist.push(operand);
            }
        }
    }
    nodes
        .iter()
        .copied()
        .filter(|node| !required.contains(node))
        .collect()
}

fn linear_form_cost(form: &LinearForm) -> usize {
    let term_count = form.terms.len();
    if term_count == 0 {
        return 0;
    }
    let scaled_terms = form
        .terms
        .values()
        .filter(|coefficient| **coefficient != 1 && **coefficient != -1)
        .count();
    let component_count = term_count + usize::from(form.constant != 0);
    let has_starting_component =
        form.constant != 0 || form.terms.values().any(|coefficient| *coefficient != -1);
    scaled_terms + component_count.saturating_sub(1) + usize::from(!has_starting_component)
}

fn build_linear_form_before_root(
    func: &mut Function,
    root: ValueId,
    form: &LinearForm,
) -> Option<ValueId> {
    let crate::ir::ValueKind::Inst(block, mut cursor) = func.value(root).kind else {
        return None;
    };
    let mut terms = form
        .terms
        .iter()
        .map(|(value, coefficient)| (*value, *coefficient))
        .collect::<Vec<_>>();
    terms.sort_by_key(|(value, _)| value.0);

    let mut accumulator = (form.constant != 0).then(|| get_or_add_i32_const(func, form.constant));

    // Start from an unscaled or already-scaled positive component when
    // possible.  Unit negative terms can then use subtraction directly.
    for (value, coefficient) in terms.iter().copied().filter(|(_, c)| *c != -1) {
        let component = if coefficient == 1 {
            value
        } else {
            let constant = get_or_add_i32_const(func, coefficient);
            let result = func.insert_inst(
                block,
                cursor,
                InstKind::Binary {
                    op: BinaryOp::Imul,
                    lhs: value,
                    rhs: constant,
                },
                Some(Type::I32),
            )?;
            cursor += 1;
            result
        };
        accumulator = Some(if let Some(lhs) = accumulator {
            let result = func.insert_inst(
                block,
                cursor,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs,
                    rhs: component,
                },
                Some(Type::I32),
            )?;
            cursor += 1;
            result
        } else {
            component
        });
    }

    for (value, _) in terms
        .into_iter()
        .filter(|(_, coefficient)| *coefficient == -1)
    {
        let lhs = accumulator.unwrap_or_else(|| get_or_add_i32_const(func, 0));
        let result = func.insert_inst(
            block,
            cursor,
            InstKind::Binary {
                op: BinaryOp::Isub,
                lhs,
                rhs: value,
            },
            Some(Type::I32),
        )?;
        cursor += 1;
        accumulator = Some(result);
    }

    Some(accumulator.unwrap_or_else(|| get_or_add_i32_const(func, 0)))
}

/// Folds `(x == 2k) || (x == 2k + 1)` into a masked equality.  This is the
/// integer-set form of the usual `icmp ult (x - 2k), 2` InstCombine rule, but
/// uses only signed-agnostic bit operations available in this IR.
fn combine_adjacent_equality_sets(func: &mut Function) {
    const MAX_VALUES: usize = 16_384;
    const MAX_INSTRUCTIONS: usize = 65_536;
    if func.values.len() > MAX_VALUES
        || func
            .blocks
            .iter()
            .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
            .is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return;
    }

    let mut use_counts = vec![0usize; func.values.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            for operand in inst_operands(&inst.kind) {
                use_counts[operand.0] = use_counts[operand.0].saturating_add(1);
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_operands(terminator) {
                use_counts[operand.0] = use_counts[operand.0].saturating_add(1);
            }
        }
    }

    let mut candidates = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            let (
                Some(result),
                InstKind::Binary {
                    op: BinaryOp::Or,
                    lhs,
                    rhs,
                },
            ) = (inst.result, &inst.kind)
            else {
                continue;
            };
            let (Some((lhs_value, lhs_constant)), Some((rhs_value, rhs_constant))) = (
                equality_with_constant(func, *lhs),
                equality_with_constant(func, *rhs),
            ) else {
                continue;
            };
            if lhs_value != rhs_value
                || use_counts.get(lhs.0) != Some(&1)
                || use_counts.get(rhs.0) != Some(&1)
            {
                continue;
            }
            let base = lhs_constant.min(rhs_constant);
            if base & 1 != 0
                || lhs_constant.max(rhs_constant) != base.wrapping_add(1)
                || func.value(lhs_value).ty != crate::ir::Type::I32
            {
                continue;
            }
            candidates.push((block_idx, inst_idx, result, lhs_value, base));
        }
    }

    candidates.sort_by_key(|(block, inst, ..)| (*block, *inst));
    for (block_idx, inst_idx, result, value, base) in candidates.into_iter().rev() {
        if !matches!(
            func.blocks[block_idx].insts.get(inst_idx),
            Some(crate::ir::Inst {
                result: Some(current),
                kind: InstKind::Binary { op: BinaryOp::Or, .. },
            }) if *current == result
        ) {
            continue;
        }
        let block = crate::ir::BlockId(block_idx);
        let mut cursor = inst_idx;
        let normalized = if base == 0 {
            value
        } else {
            let base_value = get_or_add_i32_const(func, base);
            let normalized = func
                .insert_inst(
                    block,
                    cursor,
                    InstKind::Binary {
                        op: BinaryOp::Ixor,
                        lhs: value,
                        rhs: base_value,
                    },
                    Some(crate::ir::Type::I32),
                )
                .expect("an adjacent equality normalization must produce i32");
            cursor += 1;
            normalized
        };
        let mask = get_or_add_i32_const(func, -2);
        let masked = func
            .insert_inst(
                block,
                cursor,
                InstKind::Binary {
                    op: BinaryOp::Iand,
                    lhs: normalized,
                    rhs: mask,
                },
                Some(crate::ir::Type::I32),
            )
            .expect("an adjacent equality mask must produce i32");
        cursor += 1;
        let zero = get_or_add_i32_const(func, 0);
        func.blocks[block_idx].insts[cursor].kind = InstKind::Icmp {
            op: CmpOp::Eq,
            lhs: masked,
            rhs: zero,
        };
    }
}

fn equality_with_constant(func: &Function, value: ValueId) -> Option<(ValueId, i32)> {
    let InstKind::Icmp {
        op: CmpOp::Eq,
        lhs,
        rhs,
    } = defining_inst(func, value)?
    else {
        return None;
    };
    match (const_i32(func, *lhs), const_i32(func, *rhs)) {
        (Some(constant), None) => Some((*rhs, constant)),
        (None, Some(constant)) => Some((*lhs, constant)),
        _ => None,
    }
}

/// Rewrites a signed remainder to a mask only when every observation asks
/// whether that remainder is zero. For a power-of-two divisor this preserves
/// divisibility for positive and negative dividends without changing the
/// remainder value in contexts where its sign would matter.
fn combine_divisibility_remainders(func: &mut Function) {
    const MAX_VALUES: usize = 16_384;
    const MAX_INSTRUCTIONS: usize = 65_536;
    const MAX_OPERAND_WORK: usize = 262_144;

    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    if func.values.len() > MAX_VALUES
        || instruction_count.is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return;
    }

    #[derive(Clone, Copy)]
    struct Candidate {
        block: usize,
        inst: usize,
        dividend: ValueId,
        mask: i32,
        found_use: bool,
        valid: bool,
    }

    let mut candidates = vec![None; func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            let (
                Some(result),
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs,
                    rhs,
                },
            ) = (inst.result, &inst.kind)
            else {
                continue;
            };
            let Some(divisor) = const_i32(func, *rhs) else {
                continue;
            };
            if divisor == 0 {
                continue;
            }
            let magnitude = divisor.wrapping_abs() as u32;
            if magnitude.is_power_of_two() {
                candidates[result.0] = Some(Candidate {
                    block: block_idx,
                    inst: inst_idx,
                    dividend: *lhs,
                    mask: magnitude.wrapping_sub(1) as i32,
                    found_use: false,
                    valid: true,
                });
            }
        }
    }

    let mut operand_work = 0usize;
    for block in &func.blocks {
        for inst in &block.insts {
            let operands = inst_operands(&inst.kind);
            operand_work = operand_work.saturating_add(operands.len());
            if operand_work > MAX_OPERAND_WORK {
                return;
            }
            for remainder in operands {
                let Some(candidate) = candidates.get_mut(remainder.0).and_then(Option::as_mut)
                else {
                    continue;
                };
                candidate.found_use = true;
                let valid = matches!(
                    &inst.kind,
                    InstKind::Icmp { op, lhs, rhs }
                        if matches!(*op, CmpOp::Eq | CmpOp::Ne)
                            && ((*lhs == remainder && const_i32(func, *rhs) == Some(0))
                                || (*rhs == remainder && const_i32(func, *lhs) == Some(0)))
                );
                candidate.valid &= valid;
            }
        }
        if let Some(terminator) = &block.terminator {
            let operands = terminator_operands(terminator);
            operand_work = operand_work.saturating_add(operands.len());
            if operand_work > MAX_OPERAND_WORK {
                return;
            }
            for remainder in operands {
                if let Some(candidate) = candidates.get_mut(remainder.0).and_then(Option::as_mut) {
                    candidate.found_use = true;
                    candidate.valid = false;
                }
            }
        }
    }

    let mut mask_values = HashMap::<i32, ValueId>::new();
    for candidate in candidates.into_iter().flatten() {
        if !candidate.found_use || !candidate.valid {
            continue;
        }
        let mask = if let Some(mask) = mask_values.get(&candidate.mask).copied() {
            mask
        } else {
            let mask = get_or_add_i32_const(func, candidate.mask);
            mask_values.insert(candidate.mask, mask);
            mask
        };
        func.blocks[candidate.block].insts[candidate.inst].kind = InstKind::Binary {
            op: BinaryOp::Iand,
            lhs: candidate.dividend,
            rhs: mask,
        };
    }
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } => vec![*ptr],
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
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}

fn combine_inst(func: &mut Function, kind: &InstKind) -> InstKind {
    match *kind {
        InstKind::Binary { op, lhs, rhs } => combine_binary(func, op, lhs, rhs),
        InstKind::Icmp { op, lhs, rhs }
            if const_i32(func, lhs).is_some() && const_i32(func, rhs).is_none() =>
        {
            InstKind::Icmp {
                op: reverse_cmp(op),
                lhs: rhs,
                rhs: lhs,
            }
        }
        _ => kind.clone(),
    }
}

fn combine_binary(
    func: &mut Function,
    mut op: BinaryOp,
    mut lhs: ValueId,
    mut rhs: ValueId,
) -> InstKind {
    (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);

    // i32 arithmetic wraps, so doubling is exactly multiplication by two.
    if op == BinaryOp::Iadd && lhs == rhs {
        op = BinaryOp::Imul;
        rhs = get_or_add_i32_const(func, 2);
        (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);
    }

    if op == BinaryOp::Isub {
        if lhs == rhs {
            return InstKind::Binary {
                op: BinaryOp::Iand,
                lhs,
                rhs: get_or_add_i32_const(func, 0),
            };
        }
        if let Some(remaining) = cancel_sub_from_sum(func, lhs, rhs) {
            return InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: remaining,
                rhs: get_or_add_i32_const(func, 0),
            };
        }
        if let Some(divisor) = match_remainder_divisor(func, lhs, rhs) {
            return InstKind::Binary {
                op: BinaryOp::Imod,
                lhs,
                rhs: divisor,
            };
        }
    }

    if op == BinaryOp::Idiv {
        if let Some(combined) = combine_positive_constant_divisions(func, lhs, rhs) {
            return combined;
        }
    }

    if matches!(op, BinaryOp::Iadd | BinaryOp::Imul) {
        if let Some((base, constant)) = reassociate_constants(func, op, lhs, rhs) {
            lhs = base;
            rhs = constant;
            (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);
        }
    }

    if op == BinaryOp::Iadd {
        if let Some(remaining) = cancel_add_with_sub(func, lhs, rhs) {
            return InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: remaining,
                rhs: get_or_add_i32_const(func, 0),
            };
        }
    }

    InstKind::Binary { op, lhs, rhs }
}

fn cancel_sub_from_sum(func: &Function, sum: ValueId, subtrahend: ValueId) -> Option<ValueId> {
    let InstKind::Binary {
        op: BinaryOp::Iadd,
        lhs,
        rhs,
    } = defining_inst(func, sum)?
    else {
        return None;
    };
    if *lhs == subtrahend {
        Some(*rhs)
    } else if *rhs == subtrahend {
        Some(*lhs)
    } else {
        None
    }
}

fn cancel_add_with_sub(func: &Function, lhs: ValueId, rhs: ValueId) -> Option<ValueId> {
    for (difference, added) in [(lhs, rhs), (rhs, lhs)] {
        let Some(InstKind::Binary {
            op: BinaryOp::Isub,
            lhs: minuend,
            rhs: subtrahend,
        }) = defining_inst(func, difference)
        else {
            continue;
        };
        if *subtrahend == added {
            return Some(*minuend);
        }
    }
    None
}

/// Reassociates `(x / a) / b` to `x / (a * b)` for positive constants.
///
/// Signed integer division truncates toward zero, so this is exact when both
/// divisors are positive. If their product is larger than every possible i32
/// magnitude, the quotient is identically zero. The boundary product `2^31`
/// is deliberately left alone because `i32::MIN / 2^31 == -1`.
fn combine_positive_constant_divisions(
    func: &mut Function,
    quotient: ValueId,
    outer_divisor: ValueId,
) -> Option<InstKind> {
    let outer = const_i32(func, outer_divisor)?;
    if outer <= 0 {
        return None;
    }
    let InstKind::Binary {
        op: BinaryOp::Idiv,
        lhs: dividend,
        rhs: inner_divisor,
    } = defining_inst(func, quotient)?.clone()
    else {
        return None;
    };
    let inner = const_i32(func, inner_divisor)?;
    if inner <= 0 {
        return None;
    }

    let product = i64::from(inner) * i64::from(outer);
    if product <= i64::from(i32::MAX) {
        return Some(InstKind::Binary {
            op: BinaryOp::Idiv,
            lhs: dividend,
            rhs: get_or_add_i32_const(func, product as i32),
        });
    }
    if product > i64::from(i32::MAX) + 1 {
        return Some(InstKind::Binary {
            op: BinaryOp::Iand,
            lhs: dividend,
            rhs: get_or_add_i32_const(func, 0),
        });
    }
    None
}

/// Keeps constants on the right and otherwise orders operands by ValueId.
/// Floating-point and boolean operations are deliberately excluded.
fn canonicalize_integer_operands(
    func: &Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> (ValueId, ValueId) {
    if !is_commutative_integer_op(op) {
        return (lhs, rhs);
    }

    let lhs_is_const = const_i32(func, lhs).is_some();
    let rhs_is_const = const_i32(func, rhs).is_some();
    let should_swap = match (lhs_is_const, rhs_is_const) {
        (true, false) => true,
        (false, true) => false,
        _ => rhs.0 < lhs.0,
    };
    if should_swap {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn is_commutative_integer_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Iadd | BinaryOp::Imul | BinaryOp::Iand | BinaryOp::Ior | BinaryOp::Ixor
    )
}

fn reassociate_constants(
    func: &mut Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<(ValueId, ValueId)> {
    let outer_constant = const_i32(func, rhs)?;
    let InstKind::Binary {
        op: inner_op,
        lhs: inner_lhs,
        rhs: inner_rhs,
    } = defining_inst(func, lhs)?.clone()
    else {
        return None;
    };
    if inner_op != op {
        return None;
    }

    let (base, inner_constant) = one_constant_operand(func, inner_lhs, inner_rhs)?;
    if base == lhs {
        return None;
    }
    let combined = match op {
        BinaryOp::Iadd => inner_constant.wrapping_add(outer_constant),
        BinaryOp::Imul => inner_constant.wrapping_mul(outer_constant),
        _ => return None,
    };
    Some((base, get_or_add_i32_const(func, combined)))
}

fn one_constant_operand(func: &Function, lhs: ValueId, rhs: ValueId) -> Option<(ValueId, i32)> {
    match (const_i32(func, lhs), const_i32(func, rhs)) {
        (Some(constant), None) => Some((rhs, constant)),
        (None, Some(constant)) => Some((lhs, constant)),
        _ => None,
    }
}

/// Matches `x - (x / d) * d` through direct use-def edges.
///
/// Both occurrences of `d` must be known i32 constants with the same nonzero
/// value, and the division must use the exact subtraction dividend.
fn match_remainder_divisor(
    func: &Function,
    dividend: ValueId,
    product: ValueId,
) -> Option<ValueId> {
    let InstKind::Binary {
        op: BinaryOp::Imul,
        lhs: product_lhs,
        rhs: product_rhs,
    } = defining_inst(func, product)?
    else {
        return None;
    };

    for (quotient, factor) in [(*product_lhs, *product_rhs), (*product_rhs, *product_lhs)] {
        let Some(factor_value) = const_i32(func, factor) else {
            continue;
        };
        if factor_value == 0 {
            continue;
        }
        let Some(InstKind::Binary {
            op: BinaryOp::Idiv,
            lhs: division_dividend,
            rhs: divisor,
        }) = defining_inst(func, quotient)
        else {
            continue;
        };
        if *division_dividend == dividend && const_i32(func, *divisor) == Some(factor_value) {
            return Some(*divisor);
        }
    }
    None
}

fn reverse_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}
