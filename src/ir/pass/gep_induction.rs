use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{
    analyze_const_i32_trip_count, analyze_i32_induction, InductionVariable, LoopInfo, NaturalLoop,
};
use super::util::{
    const_i32, defining_inst, get_or_add_i32_const, rewrite_function_uses, ValueReplacements,
};
use super::ModulePass;
use crate::ir::{BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTION_BLOCKS: usize = 1024;
const MAX_FUNCTION_VALUES: usize = 8192;
const MAX_FUNCTION_INSTRUCTIONS: usize = 32_768;
const MAX_CANDIDATES: usize = 128;
const MAX_NEW_VALUES: usize = 1024;
const MAX_GEP_CHAIN_DEPTH: usize = 64;
const MAX_GEP_INDICES: usize = 128;

/// Replaces an affine loop-varying address with a pointer recurrence.
///
/// The proof deliberately starts from natural-loop structure and typed GEP
/// strides. It does not depend on source names, block numbering, or a
/// particular array shape.
pub(super) struct GepInductionPass;

impl GepInductionPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for GepInductionPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            strength_reduce_function(func);
        }
    }
}

#[derive(Clone)]
struct GepTemplate {
    indices: Vec<ValueId>,
    result_ty: Type,
}

#[derive(Clone)]
struct Candidate {
    target: ValueId,
    root: ValueId,
    chain: Vec<GepTemplate>,
    induction: InductionVariable,
    preheader: BlockId,
    header: BlockId,
    latch: BlockId,
    exit_condition: ValueId,
    continue_on_true: bool,
    body_entry: BlockId,
    exit_block: BlockId,
    exact_trip_count: Option<u32>,
    exit_bound: Option<ValueId>,
    pointer_ty: Type,
    byte_step: i64,
    step_index: i32,
    affine_indices: HashMap<ValueId, AffineOffset>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum AffineOffset {
    Constant(i32),
    Invariant(ValueId),
}

fn strength_reduce_function(func: &mut Function) {
    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    if func.blocks.len() > MAX_FUNCTION_BLOCKS
        || func.values.len() > MAX_FUNCTION_VALUES
        || instruction_count.is_none_or(|count| count > MAX_FUNCTION_INSTRUCTIONS)
    {
        return;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom).loops().to_vec();
    let write_only_allocas = super::dce::write_only_alloca_values(func);
    let mut candidates = Vec::new();
    let mut claimed = HashSet::new();

    for natural_loop in &loops {
        let Some(preheader) = natural_loop.dedicated_preheader else {
            continue;
        };
        if natural_loop.unique_latch().is_none() {
            continue;
        }
        if natural_loop.unique_entering_pred != Some(preheader)
            || natural_loop.blocks.iter().any(|block| {
                func.blocks[block.0]
                    .insts
                    .iter()
                    .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
            })
        {
            // A pointer carried across a call consumes a callee-saved register
            // (or adds spill traffic) while address recomputation is usually a
            // small part of such a loop. Keep this as a target-independent
            // profitability boundary; call-free inner loops are still reduced.
            continue;
        }

        let header_phis = func.blocks[natural_loop.header.0]
            .insts
            .iter()
            .filter_map(|inst| {
                matches!(inst.kind, InstKind::Phi { .. })
                    .then_some(inst.result)
                    .flatten()
            })
            .collect::<Vec<_>>();
        for phi in header_phis {
            let Some(induction) = analyze_i32_induction(func, natural_loop, phi) else {
                continue;
            };
            if !induction_does_not_wrap(func, &dom, natural_loop, induction) {
                continue;
            }

            let mut affine = Vec::new();
            for block in sorted_loop_blocks(natural_loop) {
                for inst in &func.blocks[block.0].insts {
                    let Some(target) = inst.result else {
                        continue;
                    };
                    if claimed.contains(&target) || !matches!(inst.kind, InstKind::Gep { .. }) {
                        continue;
                    }
                    if !uses_are_internal_and_dominated(func, &dom, natural_loop, target) {
                        continue;
                    }
                    if let Some(candidate) =
                        analyze_candidate(func, &dom, natural_loop, &loops, induction, target)
                    {
                        if candidates.len().saturating_add(affine.len()) >= MAX_CANDIDATES {
                            return;
                        }
                        affine.push(candidate);
                    }
                }
            }

            // A chain's intermediate GEP becomes dead when every use is the
            // base of another selected GEP. Avoid creating a dead pointer cycle
            // that the intentionally simple DCE pass could not peel later.
            let affine_targets = affine
                .iter()
                .map(|candidate| candidate.target)
                .collect::<HashSet<_>>();
            affine.retain(|candidate| {
                has_use_not_covered_by_candidate(func, candidate.target, &affine_targets)
            });
            for candidate in affine {
                claimed.insert(candidate.target);
                candidates.push(candidate);
            }
        }
    }

    if candidates.is_empty() {
        return;
    }
    let projection =
        candidates
            .iter()
            .try_fold((0usize, 0usize), |(values, instructions), candidate| {
                let affine_starts = candidate
                    .affine_indices
                    .values()
                    .copied()
                    .filter(|offset| *offset != AffineOffset::Constant(0))
                    .collect::<HashSet<_>>()
                    .len();
                let new_instructions = candidate
                    .chain
                    .len()
                    .checked_add(2)?
                    .checked_add(affine_starts)?;
                let new_values = new_instructions
                    // Conservatively charge one step constant and one constant
                    // for every rebuilt affine start even when already interned.
                    .checked_add(1)?
                    .checked_add(affine_starts)?;
                Some((
                    values.checked_add(new_values)?,
                    instructions.checked_add(new_instructions)?,
                ))
            });
    if projection.is_none_or(|(values, instructions)| {
        values > MAX_NEW_VALUES
            || func
                .values
                .len()
                .checked_add(values)
                .is_none_or(|total| total > MAX_FUNCTION_VALUES)
            || instruction_count
                .and_then(|count| count.checked_add(instructions))
                .is_none_or(|total| total > MAX_FUNCTION_INSTRUCTIONS)
    }) {
        return;
    }

    let mut replacements = ValueReplacements::new();
    let mut built_pointers = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let pointer = build_pointer_recurrence(func, candidate);
        replacements.insert(candidate.target, pointer);
        built_pointers.push(pointer);
    }

    rewrite_function_uses(func, &replacements);
    for candidate in &candidates {
        remove_replaced_gep(func, candidate.target);
    }
    if let Err(errors) = func.verify() {
        panic!(
            "GEP induction recurrence construction produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }

    // Building and installing every recurrence first keeps all candidate
    // dominance proofs on the same CFG snapshot. Apply exit rewrites as
    // verifier-guarded transactions: rotating one nested loop can change the
    // dominance assumptions of another otherwise independent recurrence.
    let mut rewritten_exit_tests = HashSet::new();
    for (candidate, pointer) in candidates.iter().zip(built_pointers) {
        let key = (candidate.header, candidate.induction.phi);
        if !rewritten_exit_tests.insert(key) {
            continue;
        }
        let before = func.clone();
        if !rewrite_exact_exit_test_with_pointer(func, candidate, pointer, &write_only_allocas)
            || func.verify().is_err()
        {
            *func = before;
            rewritten_exit_tests.remove(&key);
        }
    }

    if let Err(errors) = func.verify() {
        panic!(
            "GEP induction strength reduction produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn analyze_candidate(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    all_loops: &[NaturalLoop],
    induction: InductionVariable,
    target: ValueId,
) -> Option<Candidate> {
    let preheader = natural_loop.dedicated_preheader?;
    let latch = natural_loop.unique_latch()?;
    let Terminator::Branch {
        cond: exit_condition,
        then_target,
        else_target,
    } = func.blocks[natural_loop.header.0].terminator.as_ref()?
    else {
        return None;
    };
    let then_inside = natural_loop.blocks.contains(then_target);
    let else_inside = natural_loop.blocks.contains(else_target);
    if then_inside == else_inside {
        return None;
    }
    let pointer_ty = func.values.get(target.0)?.ty.clone();
    let final_elem_size = match &pointer_ty {
        Type::Ptr(pointee) => checked_type_size(pointee)?.max(1),
        _ => return None,
    };

    let mut current = target;
    let mut reverse_chain = Vec::new();
    let mut coefficient = 0i64;
    let mut affine_indices = HashMap::new();
    let mut chain_depth = 0usize;
    let mut index_count = 0usize;
    let root = loop {
        let InstKind::Gep { base, indices } = defining_inst(func, current)? else {
            return None;
        };
        chain_depth = chain_depth.checked_add(1)?;
        index_count = index_count.checked_add(indices.len())?;
        if chain_depth > MAX_GEP_CHAIN_DEPTH
            || index_count > MAX_GEP_INDICES
            || indices.is_empty()
            || !matches!(func.values.get(base.0)?.ty, Type::Ptr(_))
        {
            return None;
        }
        let result_ty = func.values.get(current.0)?.ty.clone();
        let strides = gep_index_strides(func, *base, indices.len(), &result_ty)?;
        for (index, stride) in indices.iter().zip(strides) {
            if func.values.get(index.0)?.ty != Type::I32 {
                return None;
            }
            if let Some(offset) =
                affine_induction_offset(func, *index, induction.phi).filter(|offset| {
                    affine_offset_available(func, dom, natural_loop, preheader, *offset)
                        && derived_index_does_not_wrap(
                            func,
                            dom,
                            natural_loop,
                            all_loops,
                            induction,
                            *offset,
                            target,
                        )
                })
            {
                coefficient = coefficient.checked_add(stride)?;
                affine_indices.insert(*index, offset);
            } else if !is_loop_invariant(func, natural_loop, *index)
                || !value_available_at(func, dom, *index, preheader)
            {
                return None;
            }
        }
        reverse_chain.push(GepTemplate {
            indices: indices.clone(),
            result_ty,
        });

        if is_loop_invariant(func, natural_loop, *base) {
            if !value_available_at(func, dom, *base, preheader) {
                return None;
            }
            break *base;
        }
        if !matches!(defining_inst(func, *base), Some(InstKind::Gep { .. })) {
            return None;
        }
        current = *base;
    };

    if coefficient == 0 {
        return None;
    }
    let byte_step = coefficient.checked_mul(i64::from(induction.step))?;
    if byte_step % final_elem_size != 0 {
        return None;
    }
    let step_index = i32::try_from(byte_step / final_elem_size).ok()?;
    if step_index == 0 {
        return None;
    }

    reverse_chain.reverse();
    Some(Candidate {
        target,
        root,
        chain: reverse_chain,
        induction,
        preheader,
        header: natural_loop.header,
        latch,
        exit_condition: *exit_condition,
        continue_on_true: then_inside,
        body_entry: if then_inside {
            *then_target
        } else {
            *else_target
        },
        exit_block: if then_inside {
            *else_target
        } else {
            *then_target
        },
        exact_trip_count: analyze_const_i32_trip_count(func, natural_loop, induction),
        exit_bound: active_header_comparison(func, natural_loop, induction.phi)
            .and_then(|(op, bound, _)| (op == CmpOp::Lt && induction.step == 1).then_some(bound)),
        pointer_ty,
        byte_step,
        step_index,
        affine_indices,
    })
}

fn affine_induction_offset(
    func: &Function,
    value: ValueId,
    induction: ValueId,
) -> Option<AffineOffset> {
    if value == induction {
        return Some(AffineOffset::Constant(0));
    }
    match defining_inst(func, value)? {
        InstKind::Binary {
            op: crate::ir::BinaryOp::Iadd,
            lhs,
            rhs,
        } if *lhs == induction => Some(
            const_i32(func, *rhs)
                .map(AffineOffset::Constant)
                .unwrap_or(AffineOffset::Invariant(*rhs)),
        ),
        InstKind::Binary {
            op: crate::ir::BinaryOp::Iadd,
            lhs,
            rhs,
        } if *rhs == induction => Some(
            const_i32(func, *lhs)
                .map(AffineOffset::Constant)
                .unwrap_or(AffineOffset::Invariant(*lhs)),
        ),
        InstKind::Binary {
            op: crate::ir::BinaryOp::Isub,
            lhs,
            rhs,
        } if *lhs == induction => const_i32(func, *rhs)?
            .checked_neg()
            .map(AffineOffset::Constant),
        _ => None,
    }
}

fn affine_offset_available(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    preheader: BlockId,
    offset: AffineOffset,
) -> bool {
    match offset {
        AffineOffset::Constant(_) => true,
        AffineOffset::Invariant(value) => {
            is_loop_invariant(func, natural_loop, value)
                && value_available_at(func, dom, value, preheader)
        }
    }
}

/// Proves that sign-extending `iv + offset` is equivalent to adding the
/// mathematical offset. Exact finite trip counts use checked endpoints;
/// otherwise a constant-start addrec must avoid signed overflow over its
/// complete modulo-2^32 congruence class.
fn derived_index_does_not_wrap(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    all_loops: &[NaturalLoop],
    induction: InductionVariable,
    offset: AffineOffset,
    target: ValueId,
) -> bool {
    if offset == AffineOffset::Constant(0) {
        return true;
    }
    if let (Some(induction_range), Some(offset_range)) = (
        const_trip_induction_range(func, natural_loop, induction),
        affine_offset_range(func, all_loops, offset),
    ) {
        return induction_range
            .0
            .checked_add(offset_range.0)
            .zip(induction_range.1.checked_add(offset_range.1))
            .is_some_and(|(minimum, maximum)| {
                minimum >= i64::from(i32::MIN) && maximum <= i64::from(i32::MAX)
            });
    }

    let AffineOffset::Constant(offset) = offset else {
        return false;
    };
    if active_header_guard_proves_unit_offset(func, dom, natural_loop, induction, offset, target) {
        return true;
    }
    let Some(initial) = const_i32(func, induction.initial).map(i64::from) else {
        return false;
    };

    // Once the AddRec itself is known not to wrap, an offset directed away
    // from the moving endpoint only needs a proof at the initial value.  This
    // is the standard SCEV range proof for `i - k` in an increasing loop (and
    // symmetrically `i + k` in a decreasing loop).
    if (induction.step > 0 && offset < 0) || (induction.step < 0 && offset > 0) {
        return initial
            .checked_add(i64::from(offset))
            .is_some_and(|value| value >= i64::from(i32::MIN) && value <= i64::from(i32::MAX));
    }

    let modulus = gcd_u64(induction.step.unsigned_abs() as u64, 1u64 << 32) as i64;
    if modulus == 0 {
        return false;
    }
    let signed_minimum = i64::from(i32::MIN);
    let signed_maximum = i64::from(i32::MAX);
    let minimum = signed_minimum + (initial - signed_minimum).rem_euclid(modulus);
    let maximum = signed_maximum - (signed_maximum - initial).rem_euclid(modulus);
    minimum
        .checked_add(i64::from(offset))
        .zip(maximum.checked_add(i64::from(offset)))
        .is_some_and(|(minimum, maximum)| minimum >= signed_minimum && maximum <= signed_maximum)
}

/// A strict header guard proves a unit derived index safe on the taken path:
/// `iv < bound` over i32 implies `iv <= INT_MAX - 1`, and symmetrically
/// `iv > bound` implies `iv >= INT_MIN + 1`. The address must be formed in a
/// block dominated by that taken successor; an expression in the header would
/// also execute on the final failed test and cannot use this proof.
fn active_header_guard_proves_unit_offset(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    induction: InductionVariable,
    offset: i32,
    target: ValueId,
) -> bool {
    let Some(ValueKind::Inst(target_block, _)) = func.values.get(target.0).map(|value| &value.kind)
    else {
        return false;
    };
    let Some((op, _, taken_successor)) =
        active_header_comparison(func, natural_loop, induction.phi)
    else {
        return false;
    };
    if *target_block == natural_loop.header || !dom.dominates(taken_successor, *target_block) {
        return false;
    }
    matches!(
        (induction.step, op, offset),
        (1, CmpOp::Lt | CmpOp::Le, 1) | (-1, CmpOp::Gt | CmpOp::Ge, -1)
    )
}

fn affine_offset_range(
    func: &Function,
    all_loops: &[NaturalLoop],
    offset: AffineOffset,
) -> Option<(i64, i64)> {
    match offset {
        AffineOffset::Constant(offset) => {
            let offset = i64::from(offset);
            Some((offset, offset))
        }
        AffineOffset::Invariant(value) => all_loops.iter().find_map(|natural_loop| {
            let induction = analyze_i32_induction(func, natural_loop, value)?;
            const_trip_induction_range(func, natural_loop, induction)
        }),
    }
}

fn const_trip_induction_range(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: InductionVariable,
) -> Option<(i64, i64)> {
    let initial = i64::from(const_i32(func, induction.initial)?);
    let trip_count = i64::from(analyze_const_i32_trip_count(func, natural_loop, induction)?);
    let final_value = i64::from(induction.step)
        .checked_mul(trip_count)?
        .checked_add(initial)?;
    let minimum = initial.min(final_value);
    let maximum = initial.max(final_value);
    (minimum >= i64::from(i32::MIN) && maximum <= i64::from(i32::MAX)).then_some((minimum, maximum))
}

fn gcd_u64(mut lhs: u64, mut rhs: u64) -> u64 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

/// A pointer recurrence is valid only while the sign-extended i32 induction
/// changes by the mathematical step. If the i32 wrapped, recomputing the GEP
/// and incrementing the pointer would differ by a multiple of 2^32 bytes.
fn induction_does_not_wrap(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    induction: InductionVariable,
) -> bool {
    if analyze_const_i32_trip_count(func, natural_loop, induction).is_some() {
        return true;
    }

    // Front-end i32 arithmetic carries the source language's signed-overflow
    // undefined behavior.  Consequently a unit source recurrence is an nsw
    // AddRec independently of whether a changing/aliasing loop bound can be
    // hoisted.  LLVM SCEV and GCC ivopts rely on the same fact; a pointer
    // recurrence only differs after the original `iv +/- 1` has overflowed.
    if matches!(induction.step, -1 | 1) {
        return true;
    }

    let Some((op, bound, _)) = active_header_comparison(func, natural_loop, induction.phi) else {
        return false;
    };
    let Some(preheader) = natural_loop.dedicated_preheader else {
        return false;
    };
    if func.values.get(bound.0).map(|value| &value.ty) != Some(&Type::I32)
        || !is_loop_invariant(func, natural_loop, bound)
        || !value_available_at(func, dom, bound, preheader)
    {
        return false;
    }

    let step = i64::from(induction.step);
    let constant_bound = const_i32(func, bound).map(i64::from);
    match (step.is_positive(), op) {
        (true, CmpOp::Lt) => {
            step == 1 || constant_bound.is_some_and(|bound| bound <= i64::from(i32::MAX) - step + 1)
        }
        (true, CmpOp::Le) => {
            // A unit signed induction that executes past INT_MAX has already
            // invoked source-level signed-overflow undefined behavior.  GCC's
            // ivopts and LLVM's nsw AddRec reasoning use the same fact.  For
            // larger steps we still require an explicit endpoint proof.
            step == 1 || constant_bound.is_some_and(|bound| bound <= i64::from(i32::MAX) - step)
        }
        (false, CmpOp::Gt) => {
            step == -1
                || constant_bound.is_some_and(|bound| bound >= i64::from(i32::MIN) - step - 1)
        }
        (false, CmpOp::Ge) => {
            step == -1 || constant_bound.is_some_and(|bound| bound >= i64::from(i32::MIN) - step)
        }
        _ => false,
    }
}

/// Returns the comparison as observed on the loop-entering header edge.
fn active_header_comparison(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: ValueId,
) -> Option<(CmpOp, ValueId, BlockId)> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks[natural_loop.header.0].terminator.as_ref()?
    else {
        return None;
    };
    let then_inside = natural_loop.blocks.contains(then_target);
    let else_inside = natural_loop.blocks.contains(else_target);
    if then_inside == else_inside {
        return None;
    }
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
        return None;
    };
    let (mut op, bound) = if *lhs == induction {
        (*op, *rhs)
    } else if *rhs == induction {
        (reverse_cmp(*op), *lhs)
    } else {
        return None;
    };
    if !then_inside {
        op = negate_cmp(op);
    }
    Some((
        op,
        bound,
        if then_inside {
            *then_target
        } else {
            *else_target
        },
    ))
}

fn build_pointer_recurrence(func: &mut Function, candidate: &Candidate) -> ValueId {
    let mut affine_initials = HashMap::<AffineOffset, ValueId>::new();
    affine_initials.insert(AffineOffset::Constant(0), candidate.induction.initial);
    let mut initial_pointer = candidate.root;
    for node in &candidate.chain {
        let indices = node
            .indices
            .iter()
            .map(|index| {
                let Some(offset) = candidate.affine_indices.get(index).copied() else {
                    return *index;
                };
                if let Some(initial) = affine_initials.get(&offset).copied() {
                    return initial;
                }
                let rhs = match offset {
                    AffineOffset::Constant(offset) => get_or_add_i32_const(func, offset),
                    AffineOffset::Invariant(value) => value,
                };
                let initial = if const_i32(func, candidate.induction.initial) == Some(0) {
                    rhs
                } else {
                    func.append_inst(
                        candidate.preheader,
                        InstKind::Binary {
                            op: crate::ir::BinaryOp::Iadd,
                            lhs: candidate.induction.initial,
                            rhs,
                        },
                        Some(Type::I32),
                    )
                    .expect("an affine recurrence start must produce i32")
                };
                affine_initials.insert(offset, initial);
                initial
            })
            .collect();
        initial_pointer = func
            .append_inst(
                candidate.preheader,
                InstKind::Gep {
                    base: initial_pointer,
                    indices,
                },
                Some(node.result_ty.clone()),
            )
            .expect("a cloned GEP must produce a pointer");
    }

    let first_non_phi = func.blocks[candidate.header.0]
        .insts
        .iter()
        .position(|inst| !matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }))
        .unwrap_or(func.blocks[candidate.header.0].insts.len());
    let pointer = func
        .insert_inst(
            candidate.header,
            first_non_phi,
            InstKind::Phi {
                incomings: vec![
                    (candidate.preheader, initial_pointer),
                    (candidate.latch, initial_pointer),
                ],
            },
            Some(candidate.pointer_ty.clone()),
        )
        .expect("a pointer phi must produce a value");

    let step = get_or_add_i32_const(func, candidate.step_index);
    let next_pointer = func
        .append_inst(
            candidate.latch,
            InstKind::Gep {
                base: pointer,
                indices: vec![step],
            },
            Some(candidate.pointer_ty.clone()),
        )
        .expect("a recurrence GEP must produce a pointer");
    let ValueKind::Inst(phi_block, phi_idx) = func.values[pointer.0].kind else {
        unreachable!("new pointer value must be instruction-backed");
    };
    let InstKind::Phi { incomings } = &mut func.blocks[phi_block.0].insts[phi_idx].kind else {
        unreachable!("new pointer value must be a phi");
    };
    incomings[1].1 = next_pointer;
    pointer
}

/// Rewrites an exact integer induction exit test to compare the corresponding
/// pointer recurrence against its end pointer. This is the same generic
/// induction-variable strength reduction performed by LLVM LSR and GCC
/// ivopts: the proof uses only the AddRec, exact trip count and pointer stride.
fn rewrite_exact_exit_test_with_pointer(
    func: &mut Function,
    candidate: &Candidate,
    pointer: ValueId,
    write_only_allocas: &HashSet<ValueId>,
) -> bool {
    if write_only_allocas.contains(&candidate.root) {
        return false;
    }
    let Some(trip_count) = candidate.exact_trip_count else {
        return rotate_dynamic_exit_test_with_pointer(func, candidate, pointer);
    };
    let Some(initial) = const_i32(func, candidate.induction.initial).map(i64::from) else {
        return false;
    };
    let Some(exit_value) = i64::from(candidate.induction.step)
        .checked_mul(i64::from(trip_count))
        .and_then(|distance| initial.checked_add(distance))
        .and_then(|value| i32::try_from(value).ok())
    else {
        return false;
    };
    let exit_index = get_or_add_i32_const(func, exit_value);

    // A modular machine pointer must not reach the end address before the
    // integer recurrence reaches its exact exit value.
    let stride = candidate.byte_step.unsigned_abs() as u128;
    let pointer_modulus = 1u128 << 64;
    let period = pointer_modulus / gcd_u128(stride, pointer_modulus);
    if u128::from(trip_count) >= period {
        return false;
    }

    let Some(Terminator::Branch { cond, .. }) = func.blocks[candidate.header.0].terminator.as_ref()
    else {
        return false;
    };
    if *cond != candidate.exit_condition {
        return false;
    }
    let condition = candidate.exit_condition;
    let Some(InstKind::Icmp { lhs, rhs, .. }) = defining_inst(func, condition) else {
        return false;
    };
    if *lhs != candidate.induction.phi && *rhs != candidate.induction.phi {
        return false;
    }

    let Some(end_pointer) = build_pointer_at_induction(func, candidate, exit_index) else {
        return false;
    };
    let Some(pointer_condition) = func.append_inst(
        candidate.header,
        InstKind::Icmp {
            op: if candidate.continue_on_true {
                CmpOp::Ne
            } else {
                CmpOp::Eq
            },
            lhs: pointer,
            rhs: end_pointer,
        },
        Some(Type::I1),
    ) else {
        return false;
    };
    let Some(Terminator::Branch { cond, .. }) = func.blocks[candidate.header.0].terminator.as_mut()
    else {
        return false;
    };
    *cond = pointer_condition;
    true
}

/// Rotates a strict unit-step loop so the ordered zero-trip test remains in
/// the preheader while subsequent iterations use pointer equality at the
/// latch. This is the canonical dynamic-trip IV exit transform used by GCC
/// ivopts and LLVM loop rotation/LSR.
fn rotate_dynamic_exit_test_with_pointer(
    func: &mut Function,
    candidate: &Candidate,
    pointer: ValueId,
) -> bool {
    let Some(bound) = candidate.exit_bound else {
        return false;
    };
    if candidate
        .affine_indices
        .values()
        .any(|offset| *offset != AffineOffset::Constant(0))
    {
        // Building a dynamic end address for `iv + offset` would require a
        // separate signed no-wrap proof at the failed-loop endpoint.
        return false;
    }
    if func.blocks[candidate.exit_block.0]
        .insts
        .iter()
        .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
        || func.blocks[candidate.header.0].insts.iter().any(|inst| {
            !matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. })
                && inst.result != Some(candidate.exit_condition)
        })
        || !matches!(
            func.blocks[candidate.preheader.0].terminator,
            Some(Terminator::Jump(target)) if target == candidate.header
        )
        || !matches!(
            func.blocks[candidate.latch.0].terminator,
            Some(Terminator::Jump(target)) if target == candidate.header
        )
    {
        return false;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    if !value_available_at(func, &dom, bound, candidate.preheader) {
        return false;
    }

    let stride = candidate.byte_step.unsigned_abs() as u128;
    let pointer_modulus = 1u128 << 64;
    let period = pointer_modulus / gcd_u128(stride, pointer_modulus);
    if period <= (1u128 << 32) {
        // The initial ordered guard plus source signed-overflow UB bounds a
        // strict unit-step loop below 2^32 iterations.
        return false;
    }

    let Some(next_pointer) = pointer_latch_incoming(func, pointer, candidate.latch) else {
        return false;
    };
    let Some(end_pointer) = build_pointer_at_induction(func, candidate, bound) else {
        return false;
    };
    let Some(initial_condition) = func.append_inst(
        candidate.preheader,
        InstKind::Icmp {
            op: CmpOp::Lt,
            lhs: candidate.induction.initial,
            rhs: bound,
        },
        Some(Type::I1),
    ) else {
        return false;
    };
    let Some(latch_condition) = func.append_inst(
        candidate.latch,
        InstKind::Icmp {
            op: CmpOp::Ne,
            lhs: next_pointer,
            rhs: end_pointer,
        },
        Some(Type::I1),
    ) else {
        return false;
    };
    func.blocks[candidate.preheader.0].terminator = Some(Terminator::Branch {
        cond: initial_condition,
        then_target: candidate.header,
        else_target: candidate.exit_block,
    });
    func.blocks[candidate.header.0].terminator = Some(Terminator::Jump(candidate.body_entry));
    func.blocks[candidate.latch.0].terminator = Some(Terminator::Branch {
        cond: latch_condition,
        then_target: candidate.header,
        else_target: candidate.exit_block,
    });
    true
}

fn pointer_latch_incoming(func: &Function, pointer: ValueId, latch: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, pointer)? else {
        return None;
    };
    incomings
        .iter()
        .find_map(|(predecessor, value)| (*predecessor == latch).then_some(*value))
}

fn build_pointer_at_induction(
    func: &mut Function,
    candidate: &Candidate,
    induction: ValueId,
) -> Option<ValueId> {
    let mut affine_values = HashMap::<AffineOffset, ValueId>::new();
    affine_values.insert(AffineOffset::Constant(0), induction);
    let mut pointer = candidate.root;
    for node in &candidate.chain {
        let mut indices = Vec::with_capacity(node.indices.len());
        for index in &node.indices {
            let Some(offset) = candidate.affine_indices.get(index).copied() else {
                indices.push(*index);
                continue;
            };
            if let Some(value) = affine_values.get(&offset).copied() {
                indices.push(value);
                continue;
            }
            let rhs = match offset {
                AffineOffset::Constant(offset) => get_or_add_i32_const(func, offset),
                AffineOffset::Invariant(value) => value,
            };
            let value = if const_i32(func, induction) == Some(0) {
                rhs
            } else {
                func.append_inst(
                    candidate.preheader,
                    InstKind::Binary {
                        op: crate::ir::BinaryOp::Iadd,
                        lhs: induction,
                        rhs,
                    },
                    Some(Type::I32),
                )?
            };
            affine_values.insert(offset, value);
            indices.push(value);
        }
        pointer = func.append_inst(
            candidate.preheader,
            InstKind::Gep {
                base: pointer,
                indices,
            },
            Some(node.result_ty.clone()),
        )?;
    }
    Some(pointer)
}

fn gcd_u128(mut lhs: u128, mut rhs: u128) -> u128 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn remove_replaced_gep(func: &mut Function, value: ValueId) {
    let Some(ValueKind::Inst(block, inst_idx)) =
        func.values.get(value.0).map(|value| value.kind.clone())
    else {
        return;
    };
    let inst = &mut func.blocks[block.0].insts[inst_idx];
    inst.result = None;
    inst.kind = InstKind::Nop;
}

fn uses_are_internal_and_dominated(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    value: ValueId,
) -> bool {
    let mut has_use = false;
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let use_block = BlockId(block_idx);
        for inst in &block.insts {
            if let InstKind::Phi { incomings } = &inst.kind {
                for (pred, incoming) in incomings {
                    if *incoming != value {
                        continue;
                    }
                    has_use = true;
                    if !natural_loop.blocks.contains(&use_block)
                        || !natural_loop.blocks.contains(pred)
                        || !dom.dominates(natural_loop.header, *pred)
                    {
                        return false;
                    }
                }
            } else if inst_operands(&inst.kind).contains(&value) {
                has_use = true;
                if !natural_loop.blocks.contains(&use_block)
                    || !dom.dominates(natural_loop.header, use_block)
                {
                    return false;
                }
            }
        }
        if terminator_operands(block.terminator.as_ref()).contains(&value) {
            has_use = true;
            if !natural_loop.blocks.contains(&use_block)
                || !dom.dominates(natural_loop.header, use_block)
            {
                return false;
            }
        }
    }
    has_use
}

fn has_use_not_covered_by_candidate(
    func: &Function,
    value: ValueId,
    candidate_targets: &HashSet<ValueId>,
) -> bool {
    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings }
                    if incomings.iter().any(|(_, incoming)| *incoming == value) =>
                {
                    return true;
                }
                InstKind::Gep { base, indices }
                    if *base == value
                        && !indices.contains(&value)
                        && inst
                            .result
                            .is_some_and(|result| candidate_targets.contains(&result)) =>
                {
                    // This use disappears when the child GEP is replaced.
                }
                kind if inst_operands(kind).contains(&value) => return true,
                _ => {}
            }
        }
        if terminator_operands(block.terminator.as_ref()).contains(&value) {
            return true;
        }
    }
    false
}

fn value_available_at(
    func: &Function,
    dom: &Dominators,
    value: ValueId,
    destination: BlockId,
) -> bool {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(block, inst_idx)) => {
            func.blocks
                .get(block.0)
                .and_then(|owner| owner.insts.get(*inst_idx))
                .is_some_and(|inst| inst.result == Some(value))
                && (*block == destination || dom.dominates(*block, destination))
        }
        None => false,
    }
}

fn is_loop_invariant(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(block, _)) => !natural_loop.blocks.contains(block),
        None => false,
    }
}

fn gep_index_strides(
    func: &Function,
    base: ValueId,
    index_count: usize,
    result_ty: &Type,
) -> Option<Vec<i64>> {
    if index_count == 1 {
        let Type::Ptr(pointee) = result_ty else {
            return None;
        };
        return Some(vec![checked_type_size(pointee)?.max(1)]);
    }

    let mut current_ty = func.values.get(base.0)?.ty.clone();
    let mut strides = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        let elem_ty = match current_ty {
            Type::Ptr(inner) => *inner,
            Type::Array { elem, .. } => *elem,
            _ => return None,
        };
        strides.push(checked_type_size(&elem_ty)?.max(1));
        current_ty = elem_ty;
    }
    Some(strides)
}

fn checked_type_size(ty: &Type) -> Option<i64> {
    match ty {
        Type::Void => Some(0),
        Type::I1 | Type::I32 | Type::F32 => Some(4),
        Type::Ptr(_) => Some(8),
        Type::Array { elem, len } => {
            let len = i32::try_from(*len).ok()?;
            checked_type_size(elem)?
                .checked_mul(i64::from(len))
                .filter(|size| *size <= i64::from(i32::MAX))
        }
    }
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
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

fn sorted_loop_blocks(natural_loop: &NaturalLoop) -> Vec<BlockId> {
    let mut blocks = natural_loop.blocks.iter().copied().collect::<Vec<_>>();
    blocks.sort_by_key(|block| (*block != natural_loop.header, block.0));
    blocks
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

fn negate_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Ne,
        CmpOp::Ne => CmpOp::Eq,
        CmpOp::Lt => CmpOp::Ge,
        CmpOp::Le => CmpOp::Gt,
        CmpOp::Gt => CmpOp::Le,
        CmpOp::Ge => CmpOp::Lt,
    }
}
