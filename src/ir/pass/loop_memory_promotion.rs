use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, InductionVariable, LoopInfo, NaturalLoop};
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::HashSet;

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 16_384;
const MAX_PROMOTIONS_PER_FUNCTION: usize = 8;

/// Promotes one invariant array element across a canonical loop when every
/// competing memory access is proven not to alias it.
///
/// This is the narrow loop load/store promotion performed by mature LICM
/// implementations.  The pass deliberately accepts only typed GEP paths and
/// two exact induction proofs; unknown pointer arithmetic remains in memory.
pub(super) struct LoopMemoryPromotionPass;

impl LoopMemoryPromotionPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LoopMemoryPromotionPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            forward_preheader_stores_into_loops(func);
            for _ in 0..MAX_PROMOTIONS_PER_FUNCTION {
                let Some(candidate) = find_candidate(func) else {
                    break;
                };
                promote_candidate(func, candidate);
            }
        }
    }
}

/// Forwards a value stored in a natural-loop preheader to repeated loads of
/// the same invariant typed address when every write in the loop is proven
/// disjoint.  This is the MemorySSA store-to-load form of loop load promotion:
/// no access is speculated and zero-trip behavior is unchanged.
fn forward_preheader_stores_into_loops(func: &mut Function) {
    if func.blocks.len() > MAX_BLOCKS || func.values.len() > MAX_VALUES {
        return;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let all_loops = loop_info.loops();
    let mut replacements = ValueReplacements::new();
    let mut dead_loads = Vec::new();
    let mut forwarded = 0usize;

    for natural_loop in all_loops {
        if forwarded >= MAX_PROMOTIONS_PER_FUNCTION {
            break;
        }
        let (Some(preheader), Some(induction)) = (
            natural_loop.dedicated_preheader,
            header_induction(func, natural_loop),
        ) else {
            continue;
        };
        if natural_loop.unique_latch().is_none()
            || natural_loop.blocks.iter().any(|block| {
                func.blocks[block.0].insts.iter().any(|inst| {
                    matches!(
                        inst.kind,
                        InstKind::Call { .. } | InstKind::MemZero { .. } | InstKind::MemCopy { .. }
                    )
                })
            })
        {
            continue;
        }

        let preheader_insts = &func.blocks[preheader.0].insts;
        for (store_idx, store) in preheader_insts.iter().enumerate().rev() {
            if forwarded >= MAX_PROMOTIONS_PER_FUNCTION {
                break;
            }
            let InstKind::Store { ptr, value } = store.kind else {
                continue;
            };
            let Some(path) = access_path(func, ptr) else {
                continue;
            };
            if !is_loop_invariant(func, natural_loop, ptr)
                || !value_available_at(func, &dom, ptr, preheader)
                || !value_available_at(func, &dom, value, natural_loop.header)
                || preheader_has_later_clobber(func, preheader, store_idx, &path)
            {
                continue;
            }

            let writes_are_disjoint = natural_loop.blocks.iter().all(|block| {
                func.blocks[block.0]
                    .insts
                    .iter()
                    .all(|inst| match inst.kind {
                        InstKind::Store { ptr: other, .. } => {
                            access_path(func, other).is_some_and(|other| {
                                other != path
                                    && paths_proven_disjoint(
                                        func,
                                        &dom,
                                        natural_loop,
                                        all_loops,
                                        induction,
                                        &path,
                                        &other,
                                        *block,
                                    )
                            })
                        }
                        InstKind::Call { .. }
                        | InstKind::MemZero { .. }
                        | InstKind::MemCopy { .. } => false,
                        _ => true,
                    })
            });
            if !writes_are_disjoint {
                continue;
            }

            let mut loads = Vec::new();
            for block in &natural_loop.blocks {
                for (inst_idx, inst) in func.blocks[block.0].insts.iter().enumerate() {
                    let (Some(result), InstKind::Load { ptr: loaded }) = (inst.result, &inst.kind)
                    else {
                        continue;
                    };
                    if access_path(func, *loaded).as_ref() == Some(&path)
                        && func.value(result).ty == func.value(value).ty
                    {
                        loads.push((*block, inst_idx, result));
                    }
                }
            }
            if loads.is_empty() {
                continue;
            }
            for (block, inst_idx, result) in loads {
                replacements.insert(result, value);
                dead_loads.push((block, inst_idx, result));
            }
            forwarded += 1;
            break;
        }
    }

    if replacements.is_empty() {
        return;
    }
    rewrite_function_uses(func, &replacements);
    for (block, inst_idx, result) in dead_loads {
        let inst = &mut func.blocks[block.0].insts[inst_idx];
        if inst.result == Some(result) && matches!(inst.kind, InstKind::Load { .. }) {
            inst.result = None;
            inst.kind = InstKind::Nop;
        }
    }
    if let Err(errors) = func.verify() {
        panic!(
            "loop store-to-load forwarding produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn preheader_has_later_clobber(
    func: &Function,
    preheader: BlockId,
    store_idx: usize,
    _target: &AccessPath,
) -> bool {
    func.blocks[preheader.0]
        .insts
        .iter()
        .skip(store_idx + 1)
        .any(|inst| match inst.kind {
            InstKind::Store { .. } => true,
            InstKind::Call { .. } | InstKind::MemZero { .. } | InstKind::MemCopy { .. } => true,
            _ => false,
        })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AccessPath {
    root: ValueId,
    indices: Vec<ValueId>,
}

#[derive(Clone, Debug)]
struct Candidate {
    pointer: ValueId,
    path: AccessPath,
    value_ty: Type,
    preheader: BlockId,
    header: BlockId,
    exit_from: BlockId,
    exit: BlockId,
    initial: ValueId,
    guard_op: CmpOp,
    guard_bound: ValueId,
    loop_blocks: HashSet<BlockId>,
}

fn find_candidate(func: &Function) -> Option<Candidate> {
    if func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        // A call can invalidate the simple dereferenceability argument used
        // for the speculative preheader load.  SysY kernels normally keep
        // their hot arithmetic loops in call-free functions.
        || func.blocks.iter().any(|block| {
            block
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
        })
    {
        return None;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let all_loops = loop_info.loops();

    for natural_loop in all_loops {
        let Some(preheader) = natural_loop.dedicated_preheader else {
            continue;
        };
        if natural_loop.unique_latch().is_none() {
            continue;
        }
        let [(exit_from, exit)] = natural_loop.exit_edges.as_slice() else {
            continue;
        };
        if *exit_from != natural_loop.header
            || !matches!(
                func.blocks[preheader.0].terminator,
                Some(Terminator::Jump(target)) if target == natural_loop.header
            )
            || func.blocks[natural_loop.header.0].insts.iter().any(|inst| {
                !matches!(
                    inst.kind,
                    InstKind::Nop | InstKind::Phi { .. } | InstKind::Icmp { .. }
                )
            })
            || func.blocks[exit.0]
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
            || natural_loop.blocks.iter().any(|block| {
                func.blocks[block.0].insts.iter().any(|inst| {
                    matches!(
                        inst.kind,
                        InstKind::Call { .. } | InstKind::MemZero { .. } | InstKind::MemCopy { .. }
                    )
                })
            })
        {
            continue;
        }

        let Some(induction) = header_induction(func, natural_loop) else {
            continue;
        };
        let Some((guard_op, guard_bound, taken)) =
            active_header_comparison(func, natural_loop, induction.phi)
        else {
            continue;
        };
        if !value_available_at(func, &dom, induction.initial, preheader)
            || !value_available_at(func, &dom, guard_bound, preheader)
            || !loop_definitions_do_not_escape(func, natural_loop)
        {
            continue;
        }
        let mut pointers = Vec::new();
        for block in sorted_loop_blocks(natural_loop) {
            for inst in &func.blocks[block.0].insts {
                let pointer = match inst.kind {
                    InstKind::Load { ptr } | InstKind::Store { ptr, .. } => ptr,
                    _ => continue,
                };
                if !pointers.contains(&pointer) {
                    pointers.push(pointer);
                }
            }
        }

        for pointer in pointers {
            let invariant = is_loop_invariant(func, natural_loop, pointer);
            let available = value_available_at(func, &dom, pointer, preheader);
            if !invariant || !available {
                continue;
            }
            let Some(path) = access_path(func, pointer) else {
                continue;
            };
            if path.indices.len() < 2 || is_alloca_root(func, path.root) {
                continue;
            }
            let Some(value_ty) = scalar_pointee(func, pointer) else {
                continue;
            };

            let (loads, stores) = count_equivalent_accesses(func, natural_loop, &path);
            if loads == 0 || stores == 0 {
                continue;
            }
            if !has_must_execute_target_load(func, &dom, natural_loop, taken, &path)
                || !loop_memory_is_disjoint(func, &dom, natural_loop, all_loops, induction, &path)
            {
                continue;
            }

            return Some(Candidate {
                pointer,
                path,
                value_ty,
                preheader,
                header: natural_loop.header,
                exit_from: *exit_from,
                exit: *exit,
                initial: induction.initial,
                guard_op,
                guard_bound,
                loop_blocks: natural_loop.blocks.clone(),
            });
        }
    }
    None
}

fn header_induction(func: &Function, natural_loop: &NaturalLoop) -> Option<InductionVariable> {
    func.blocks[natural_loop.header.0]
        .insts
        .iter()
        .filter_map(|inst| inst.result)
        .find_map(|phi| analyze_i32_induction(func, natural_loop, phi))
}

fn sorted_loop_blocks(natural_loop: &NaturalLoop) -> Vec<BlockId> {
    let mut blocks = natural_loop.blocks.iter().copied().collect::<Vec<_>>();
    blocks.sort_by_key(|block| block.0);
    blocks
}

fn scalar_pointee(func: &Function, pointer: ValueId) -> Option<Type> {
    let Type::Ptr(pointee) = &func.values.get(pointer.0)?.ty else {
        return None;
    };
    matches!(pointee.as_ref(), Type::I1 | Type::I32 | Type::F32).then(|| pointee.as_ref().clone())
}

fn access_path(func: &Function, pointer: ValueId) -> Option<AccessPath> {
    let mut current = pointer;
    let mut reverse_indices = Vec::new();
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let value = func.values.get(current.0)?;
        match value.kind {
            ValueKind::Inst(block, inst_idx) => {
                let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
                if inst.result != Some(current) {
                    return None;
                }
                match &inst.kind {
                    InstKind::Gep { base, indices } if !indices.is_empty() => {
                        reverse_indices.extend(indices.iter().rev().copied());
                        current = *base;
                    }
                    InstKind::Alloca { .. } if reverse_indices.is_empty() => return None,
                    InstKind::Alloca { .. } => break,
                    _ => break,
                }
            }
            ValueKind::Param | ValueKind::Global(_) => break,
            ValueKind::Const(_) => return None,
        }
    }
    reverse_indices.reverse();
    (!reverse_indices.is_empty()).then_some(AccessPath {
        root: current,
        indices: reverse_indices,
    })
}

fn count_equivalent_accesses(
    func: &Function,
    natural_loop: &NaturalLoop,
    target: &AccessPath,
) -> (usize, usize) {
    let mut loads = 0;
    let mut stores = 0;
    for block in &natural_loop.blocks {
        for inst in &func.blocks[block.0].insts {
            match inst.kind {
                InstKind::Load { ptr } if access_path(func, ptr).as_ref() == Some(target) => {
                    loads += 1;
                }
                InstKind::Store { ptr, .. } if access_path(func, ptr).as_ref() == Some(target) => {
                    stores += 1;
                }
                _ => {}
            }
        }
    }
    (loads, stores)
}

/// Once the first header test succeeds, an original load of the target must be
/// unavoidable before the backedge.  The transform guards its initialization
/// with an equivalent first-iteration test, so zero-trip paths never load it.
fn has_must_execute_target_load(
    func: &Function,
    _dom: &Dominators,
    _natural_loop: &NaturalLoop,
    taken: BlockId,
    target: &AccessPath,
) -> bool {
    func.blocks[taken.0].insts.iter().any(|inst| {
        matches!(
            inst.kind,
            InstKind::Load { ptr }
                if access_path(func, ptr).as_ref() == Some(target)
        )
    })
}

fn loop_definitions_do_not_escape(func: &Function, natural_loop: &NaturalLoop) -> bool {
    let definitions = natural_loop
        .blocks
        .iter()
        .flat_map(|block| {
            func.blocks[block.0]
                .insts
                .iter()
                .filter_map(|inst| inst.result)
        })
        .collect::<HashSet<_>>();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        if natural_loop.blocks.contains(&BlockId(block_idx)) {
            continue;
        }
        if block
            .insts
            .iter()
            .flat_map(|inst| operands(&inst.kind))
            .any(|operand| definitions.contains(&operand))
            || block
                .terminator
                .as_ref()
                .into_iter()
                .flat_map(terminator_operands)
                .any(|operand| definitions.contains(&operand))
        {
            return false;
        }
    }
    true
}

fn operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut operands = vec![*base];
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Call { args, .. } => args.clone(),
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) | Terminator::Branch { cond: value, .. } => vec![*value],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}

fn loop_memory_is_disjoint(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    all_loops: &[NaturalLoop],
    induction: InductionVariable,
    target: &AccessPath,
) -> bool {
    natural_loop.blocks.iter().all(|block| {
        func.blocks[block.0].insts.iter().all(|inst| {
            let pointer = match inst.kind {
                InstKind::Load { ptr } | InstKind::Store { ptr, .. } => ptr,
                InstKind::MemZero { .. } | InstKind::MemCopy { .. } | InstKind::Call { .. } => {
                    return false
                }
                _ => return true,
            };
            let Some(other) = access_path(func, pointer) else {
                return false;
            };
            other == *target
                || paths_proven_disjoint(
                    func,
                    dom,
                    natural_loop,
                    all_loops,
                    induction,
                    target,
                    &other,
                    *block,
                )
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn paths_proven_disjoint(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    all_loops: &[NaturalLoop],
    induction: InductionVariable,
    target: &AccessPath,
    other: &AccessPath,
    access_block: BlockId,
) -> bool {
    if target.root != other.root {
        return roots_proven_disjoint(func, target.root, other.root);
    }
    if target.indices.len() != other.indices.len() {
        return false;
    }
    target
        .indices
        .iter()
        .zip(&other.indices)
        .any(|(target_index, other_index)| {
            *target_index != *other_index
                && indices_proven_distinct(
                    func,
                    dom,
                    natural_loop,
                    all_loops,
                    induction,
                    *target_index,
                    *other_index,
                    access_block,
                )
        })
}

fn roots_proven_disjoint(func: &Function, lhs: ValueId, rhs: ValueId) -> bool {
    match (&func.values[lhs.0].kind, &func.values[rhs.0].kind) {
        (ValueKind::Global(lhs), ValueKind::Global(rhs)) => lhs != rhs,
        (ValueKind::Inst(_, _), _) if is_alloca_root(func, lhs) => true,
        (_, ValueKind::Inst(_, _)) if is_alloca_root(func, rhs) => true,
        _ => false,
    }
}

fn is_alloca_root(func: &Function, value: ValueId) -> bool {
    matches!(defining_inst(func, value), Some(InstKind::Alloca { .. }))
}

fn indices_proven_distinct(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    _all_loops: &[NaturalLoop],
    induction: InductionVariable,
    target: ValueId,
    other: ValueId,
    access_block: BlockId,
) -> bool {
    let Some((op, bound, taken)) = active_header_comparison(func, natural_loop, induction.phi)
    else {
        return false;
    };
    if !dom.dominates(taken, access_block) {
        return false;
    }

    // On the taken edge a strict loop guard directly separates the induction
    // value and its invariant bound, e.g. `k < j` proves `[i][k] != [i][j]`.
    if ((target == bound && other == induction.phi) || (other == bound && target == induction.phi))
        && matches!(op, CmpOp::Lt | CmpOp::Gt)
    {
        return true;
    }

    // A no-wrap positive recurrence that begins strictly above an outer
    // induction stays above it.  This proves forms such as
    // `k = i + 1; ... table[k + 1][j]` disjoint from `table[i][j]`.
    if op != CmpOp::Lt || induction.step != 1 {
        return false;
    }
    let Some(initial_delta) = affine_delta(func, induction.initial, target) else {
        return false;
    };
    let Some(other_delta) = affine_delta(func, other, induction.phi) else {
        return false;
    };
    if initial_delta <= 0 || !(0..=1).contains(&other_delta) {
        return false;
    }
    // The source-language `outer + positive_delta` and the unit inner
    // recurrence both have signed-overflow UB.  On every defined execution
    // the inner induction therefore begins strictly above the outer value and
    // remains so; no constant trip count is required for this nsw proof.
    true
}

fn affine_delta(func: &Function, value: ValueId, base: ValueId) -> Option<i32> {
    if value == base {
        return Some(0);
    }
    match defining_inst(func, value)? {
        InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        } if *lhs == base => const_i32(func, *rhs),
        InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        } if *rhs == base => const_i32(func, *lhs),
        InstKind::Binary {
            op: BinaryOp::Isub,
            lhs,
            rhs,
        } if *lhs == base => const_i32(func, *rhs)?.checked_neg(),
        _ => None,
    }
}

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
    let mut normalized = *op;
    let bound = if *lhs == induction {
        *rhs
    } else if *rhs == induction {
        normalized = reverse_cmp(normalized);
        *lhs
    } else {
        return None;
    };
    if !then_inside {
        normalized = negate_cmp(normalized);
    }
    Some((
        normalized,
        bound,
        if then_inside {
            *then_target
        } else {
            *else_target
        },
    ))
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

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.values.get(value.0)?.kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
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

fn is_loop_invariant(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    match &func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(block, _)) => !natural_loop.blocks.contains(block),
        None => false,
    }
}

fn value_available_at(func: &Function, dom: &Dominators, value: ValueId, block: BlockId) -> bool {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(def_block, _)) => {
            *def_block == block || dom.dominates(*def_block, block)
        }
        None => false,
    }
}

fn promote_candidate(func: &mut Function, candidate: Candidate) {
    let slot = func
        .append_inst(
            candidate.preheader,
            InstKind::Alloca {
                ty: candidate.value_ty.clone(),
            },
            Some(Type::Ptr(Box::new(candidate.value_ty.clone()))),
        )
        .expect("a promoted loop element needs a scalar slot");
    let guard = func
        .append_inst(
            candidate.preheader,
            InstKind::Icmp {
                op: candidate.guard_op,
                lhs: candidate.initial,
                rhs: candidate.guard_bound,
            },
            Some(Type::I1),
        )
        .expect("a guarded loop promotion needs a first-iteration test");

    let initialize = func.add_block("loop.memory.initialize");
    let initial = func
        .append_inst(
            initialize,
            InstKind::Load {
                ptr: candidate.pointer,
            },
            Some(candidate.value_ty.clone()),
        )
        .expect("a promoted loop element needs an initial value");
    func.append_inst(
        initialize,
        InstKind::Store {
            ptr: slot,
            value: initial,
        },
        None,
    );
    func.set_terminator(initialize, Terminator::Jump(candidate.header));

    func.blocks[candidate.preheader.0].terminator = Some(Terminator::Branch {
        cond: guard,
        then_target: initialize,
        else_target: candidate.exit,
    });
    for inst in &mut func.blocks[candidate.header.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        for (predecessor, _) in incomings {
            if *predecessor == candidate.preheader {
                *predecessor = initialize;
            }
        }
    }

    for block in &candidate.loop_blocks {
        for inst_idx in 0..func.blocks[block.0].insts.len() {
            let replacement = match func.blocks[block.0].insts[inst_idx].kind {
                InstKind::Load { ptr }
                    if access_path(func, ptr).as_ref() == Some(&candidate.path) =>
                {
                    Some(InstKind::Load { ptr: slot })
                }
                InstKind::Store { ptr, value }
                    if access_path(func, ptr).as_ref() == Some(&candidate.path) =>
                {
                    Some(InstKind::Store { ptr: slot, value })
                }
                _ => None,
            };
            if let Some(replacement) = replacement {
                func.blocks[block.0].insts[inst_idx].kind = replacement;
            }
        }
    }

    let flush = func.add_block("loop.memory.flush");
    let final_value = func
        .append_inst(
            flush,
            InstKind::Load { ptr: slot },
            Some(candidate.value_ty),
        )
        .expect("a promoted loop element needs a final value");
    func.append_inst(
        flush,
        InstKind::Store {
            ptr: candidate.pointer,
            value: final_value,
        },
        None,
    );
    func.set_terminator(flush, Terminator::Jump(candidate.exit));

    redirect_edge(func, candidate.exit_from, candidate.exit, flush);
    for inst in &mut func.blocks[candidate.exit.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        for (predecessor, _) in incomings {
            if *predecessor == candidate.exit_from {
                *predecessor = flush;
            }
        }
    }

    if let Err(errors) = func.verify() {
        panic!(
            "loop memory promotion produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn redirect_edge(func: &mut Function, from: BlockId, old: BlockId, new: BlockId) {
    let terminator = func.blocks[from.0]
        .terminator
        .as_mut()
        .expect("a loop exit edge needs a terminator");
    match terminator {
        Terminator::Jump(target) => {
            debug_assert_eq!(*target, old);
            *target = new;
        }
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => {
            if *then_target == old {
                *then_target = new;
            }
            if *else_target == old {
                *else_target = new;
            }
        }
        Terminator::Return(_) => unreachable!("a return cannot be a loop exit edge"),
    }
}
