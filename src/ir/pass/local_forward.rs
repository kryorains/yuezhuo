use super::dominators::{ControlFlowGraph, Dominators};
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{Function, InstKind, Module, Type, ValueId, ValueKind};
use std::collections::HashMap;

const MAX_BLOCKS: usize = 2048;
const MAX_VALUES: usize = 16_384;
const MAX_INSTRUCTIONS: usize = 65_536;
const MAX_FIXED_POINT_ITERATIONS: usize = 128;
const MAX_POINTER_CHAIN_DEPTH: usize = 64;
const MAX_TYPE_NODES: usize = 65_536;
const MAX_SINGLE_TYPE_DEPTH: usize = 128;
const MAX_POINTER_PROOF_WORK: usize = 1_048_576;
const MAX_OPERAND_EDGES: usize = 262_144;
const MAX_CONST_NODES: usize = 262_144;
const MAX_CONST_STRING_BYTES: usize = 1_048_576;
const MAX_EDGE_LOADS: usize = 32;
const MAX_LOAD_DATAFLOW_WORK: usize = 1_048_576;

pub(super) struct LocalForwardPass;

impl LocalForwardPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LocalForwardPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            forward_function(func);
        }
    }
}

fn forward_function(func: &mut Function) {
    // DSE can expose forwarding and forwarding can canonicalize a DSE match.
    // Keep the pre-existing forwarding enabled for over-budget functions; only
    // the new pointer-chain analysis is size-gated.
    let dse_enabled = within_budget(func);
    let original = dse_enabled.then(|| func.clone());
    let iteration_limit = if dse_enabled {
        MAX_FIXED_POINT_ITERATIONS
    } else {
        usize::MAX
    };
    let mut changed = false;
    let mut converged = false;
    for _ in 0..iteration_limit {
        let dse_changed = dse_enabled && eliminate_redundant_writebacks(func);
        let mut replacements = ValueReplacements::new();
        let pointer_roots = dse_enabled.then(|| {
            func.values
                .iter()
                .enumerate()
                .map(|(idx, _)| typed_scalar_pointer_root(func, ValueId(idx)))
                .collect::<Vec<_>>()
        });
        let load_entries = pointer_roots
            .as_ref()
            .and_then(|pointer_roots| {
                let cfg = ControlFlowGraph::new(func);
                let dom = Dominators::new(func, &cfg);
                available_load_entries(func, &cfg, &dom, pointer_roots)
            })
            .unwrap_or_else(|| vec![HashMap::new(); func.blocks.len()]);

        for (block_idx, block) in func.blocks.iter().enumerate() {
            // Stores-to-alloca remain block-local. Exact typed loads may enter
            // from every predecessor only when available-expression
            // intersection proves the same dominating load on all paths.
            let mut known_memory = HashMap::<ValueId, ValueId>::new();
            let mut known_loads = load_entries.get(block_idx).cloned().unwrap_or_default();
            for inst in &block.insts {
                match &inst.kind {
                    InstKind::Nop | InstKind::Alloca { .. } => {}
                    InstKind::Store { ptr, value } => {
                        let ptr = resolve(*ptr, &replacements);
                        if known_loads.len() <= MAX_EDGE_LOADS {
                            known_loads.retain(|load_ptr, _| {
                                pointers_proven_disjoint(
                                    func,
                                    pointer_roots.as_deref().unwrap_or(&[]),
                                    *load_ptr,
                                    ptr,
                                )
                            });
                        } else {
                            known_loads.clear();
                        }
                        let value = resolve(*value, &replacements);
                        if tracked_pointer(func, ptr) {
                            known_memory.insert(ptr, value);
                        } else {
                            known_memory.clear();
                        }
                    }
                    InstKind::Load { ptr } => {
                        // 如果前面已知 ptr 里就是某个 value，就把这次 load 直接替换成 value。
                        let Some(result) = inst.result else {
                            continue;
                        };
                        let ptr = resolve(*ptr, &replacements);
                        if let Some(value) = known_memory
                            .get(&ptr)
                            .or_else(|| known_loads.get(&ptr))
                            .copied()
                        {
                            if func.value(value).ty == func.value(result).ty {
                                replacements.insert(result, value);
                            }
                        } else {
                            known_loads.insert(ptr, result);
                        }
                    }
                    InstKind::Call { .. } | InstKind::MemZero { .. } => {
                        // 调用和批量清零都可能改写内存，保守地丢弃本块内已知信息。
                        known_memory.clear();
                        known_loads.clear();
                    }
                    InstKind::Gep { .. } => {}
                    _ => {}
                }
            }
        }

        let forward_changed = rewrite_function_uses(func, &replacements);
        changed |= dse_changed || forward_changed;
        if !dse_changed && !forward_changed {
            converged = true;
            break;
        }
    }

    if dse_enabled && !converged {
        *func = original.expect("budgeted local forwarding must have a snapshot");
        return;
    }

    if changed {
        if let Err(errors) = func.verify() {
            panic!(
                "local forwarding produced invalid IR in {}: {:?}",
                func.name, errors
            );
        }
    }
}

fn available_load_entries(
    func: &Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    pointer_roots: &[Option<TypedScalarRoot>],
) -> Option<Vec<HashMap<ValueId, ValueId>>> {
    let mut entries = vec![HashMap::new(); func.blocks.len()];
    let mut exits = vec![HashMap::new(); func.blocks.len()];
    let mut work = 0usize;
    for _ in 0..MAX_FIXED_POINT_ITERATIONS {
        let mut changed = false;
        for block_idx in 0..func.blocks.len() {
            if !dom.is_reachable(crate::ir::BlockId(block_idx)) {
                continue;
            }
            let mut entry = if block_idx == func.entry.0 {
                HashMap::new()
            } else {
                intersect_load_states(cfg.preds.get(block_idx)?, &exits)
            };
            if entry.len() > MAX_EDGE_LOADS {
                entry.clear();
            }
            let mut exit = entry.clone();
            transfer_load_state(func, block_idx, &mut exit, pointer_roots, &mut work)?;
            if exit.len() > MAX_EDGE_LOADS {
                exit.clear();
            }
            if entries[block_idx] != entry || exits[block_idx] != exit {
                entries[block_idx] = entry;
                exits[block_idx] = exit;
                changed = true;
            }
        }
        if !changed {
            return Some(entries);
        }
    }
    None
}

fn intersect_load_states(
    predecessors: &[crate::ir::BlockId],
    exits: &[HashMap<ValueId, ValueId>],
) -> HashMap<ValueId, ValueId> {
    let Some((first, rest)) = predecessors.split_first() else {
        return HashMap::new();
    };
    let mut intersection = exits.get(first.0).cloned().unwrap_or_default();
    for predecessor in rest {
        let Some(state) = exits.get(predecessor.0) else {
            return HashMap::new();
        };
        intersection.retain(|ptr, value| state.get(ptr) == Some(value));
    }
    intersection
}

fn transfer_load_state(
    func: &Function,
    block_idx: usize,
    state: &mut HashMap<ValueId, ValueId>,
    pointer_roots: &[Option<TypedScalarRoot>],
    work: &mut usize,
) -> Option<()> {
    for inst in &func.blocks.get(block_idx)?.insts {
        *work = work.checked_add(1)?;
        if *work > MAX_LOAD_DATAFLOW_WORK {
            return None;
        }
        match &inst.kind {
            InstKind::Load { ptr } => {
                let result = inst.result?;
                match state.get(ptr).copied() {
                    Some(previous) if func.value(previous).ty == func.value(result).ty => {}
                    _ => {
                        state.insert(*ptr, result);
                    }
                }
            }
            InstKind::Store { ptr, .. } => {
                *work = work.checked_add(state.len())?;
                if *work > MAX_LOAD_DATAFLOW_WORK || state.len() > MAX_EDGE_LOADS {
                    state.clear();
                } else {
                    state.retain(|load_ptr, _| {
                        pointers_proven_disjoint(func, pointer_roots, *load_ptr, *ptr)
                    });
                }
            }
            InstKind::Call { .. } | InstKind::MemZero { .. } => state.clear(),
            _ => {}
        }
    }
    Some(())
}

/// Removes `store p, v` after `v = load p` in the same block when every
/// intervening store also writes `v`. Exact pointer identity is required only
/// for the store being removed; an intervening `store q, v` is harmless even
/// when `q` aliases `p`.
fn eliminate_redundant_writebacks(func: &mut Function) -> bool {
    let pointer_roots = func
        .values
        .iter()
        .enumerate()
        .map(|(idx, _)| typed_scalar_pointer_root(func, ValueId(idx)))
        .collect::<Vec<_>>();
    let value_types = func
        .values
        .iter()
        .map(|value| scalar_memory_type(&value.ty))
        .collect::<Vec<_>>();

    let mut changed = false;
    for block in &mut func.blocks {
        let mut candidates = HashMap::<ValueId, ValueId>::new();
        for inst in &mut block.insts {
            if let InstKind::Load { ptr } = &inst.kind {
                if let Some(result) = inst.result.filter(|result| {
                    pointer_roots
                        .get(ptr.0)
                        .and_then(|root| *root)
                        .zip(value_types.get(result.0).copied().flatten())
                        .is_some_and(|(root, value_ty)| root.ty == value_ty)
                }) {
                    candidates.insert(result, *ptr);
                }
                continue;
            }

            if let InstKind::Store { ptr, value } = &inst.kind {
                let (ptr, value) = (*ptr, *value);
                let candidate_ptr = candidates.get(&value).copied();
                if candidate_ptr == Some(ptr) {
                    inst.kind = InstKind::Nop;
                    changed = true;
                    continue;
                }

                // A same-value store through another pointer is harmless only
                // when typed GEP provenance proves both accesses are naturally
                // aligned within the same object. Equal-width aligned accesses
                // are either identical or disjoint, never partially overlapping.
                let preserves_candidate = candidate_ptr.is_some_and(|candidate_ptr| {
                    pointer_roots.get(candidate_ptr.0).and_then(|root| *root)
                        == pointer_roots.get(ptr.0).and_then(|root| *root)
                        && pointer_roots
                            .get(ptr.0)
                            .and_then(|root| *root)
                            .zip(value_types.get(value.0).copied().flatten())
                            .is_some_and(|(root, value_ty)| root.ty == value_ty)
                });
                candidates.clear();
                if preserves_candidate {
                    candidates.insert(value, candidate_ptr.unwrap());
                }
                continue;
            }

            if matches!(inst.kind, InstKind::Call { .. } | InstKind::MemZero { .. }) {
                candidates.clear();
            }
        }
    }
    changed
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarMemoryType {
    I1,
    I32,
    F32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct TypedScalarRoot {
    root: ValueId,
    ty: ScalarMemoryType,
}

fn scalar_memory_type(ty: &Type) -> Option<ScalarMemoryType> {
    match ty {
        Type::I1 => Some(ScalarMemoryType::I1),
        Type::I32 => Some(ScalarMemoryType::I32),
        Type::F32 => Some(ScalarMemoryType::F32),
        Type::Void | Type::Ptr(_) | Type::Array { .. } => None,
    }
}

/// Traces a scalar pointer through checked typed GEPs. All scalar memory types
/// are four bytes in this IR, and every valid typed GEP stride is a multiple of
/// four, so two accesses with the same root/type cannot partially overlap.
fn typed_scalar_pointer_root(func: &Function, mut ptr: ValueId) -> Option<TypedScalarRoot> {
    let Type::Ptr(pointee) = &func.values.get(ptr.0)?.ty else {
        return None;
    };
    let access_ty = scalar_memory_type(pointee)?;

    for _ in 0..MAX_POINTER_CHAIN_DEPTH {
        let value = func.values.get(ptr.0)?;
        match &value.kind {
            ValueKind::Param | ValueKind::Global(_) => {
                return Some(TypedScalarRoot {
                    root: ptr,
                    ty: access_ty,
                });
            }
            ValueKind::Const(_) => return None,
            ValueKind::Inst(block, inst_idx) => {
                let inst = func.blocks.get(block.0)?.insts.get(*inst_idx)?;
                if inst.result != Some(ptr) {
                    return None;
                }
                match &inst.kind {
                    InstKind::Alloca { .. } => {
                        return Some(TypedScalarRoot {
                            root: ptr,
                            ty: access_ty,
                        });
                    }
                    InstKind::Gep { base, indices }
                        if matches!(
                            &value.ty,
                            Type::Ptr(pointee)
                                if scalar_memory_type(pointee) == Some(access_ty)
                        ) && matches!(func.values.get(base.0)?.ty, Type::Ptr(_))
                            && indices.len() == 1
                            && func.values.get(indices[0].0).map(|value| &value.ty)
                                == Some(&Type::I32) =>
                    {
                        // A one-index GEP whose result pointee is the same
                        // four-byte scalar advances by exactly four bytes in
                        // every backend. Reject multi-index or differently
                        // typed intermediate GEPs instead of reconstructing
                        // their more complex layout here.
                        ptr = *base;
                    }
                    _ => return None,
                }
            }
        }
    }
    None
}

fn pointers_proven_disjoint(
    func: &Function,
    roots: &[Option<TypedScalarRoot>],
    lhs: ValueId,
    rhs: ValueId,
) -> bool {
    let (Some(lhs), Some(rhs)) = (
        roots.get(lhs.0).copied().flatten(),
        roots.get(rhs.0).copied().flatten(),
    ) else {
        return false;
    };
    if lhs.root == rhs.root {
        return false;
    }
    is_alloca_root(func, lhs.root) || is_alloca_root(func, rhs.root)
}

fn is_alloca_root(func: &Function, root: ValueId) -> bool {
    let Some(value) = func.values.get(root.0) else {
        return false;
    };
    let ValueKind::Inst(block, inst_idx) = value.kind else {
        return false;
    };
    func.blocks
        .get(block.0)
        .and_then(|owner| owner.insts.get(inst_idx))
        .is_some_and(|inst| {
            inst.result == Some(root) && matches!(inst.kind, InstKind::Alloca { .. })
        })
}

fn within_budget(func: &Function) -> bool {
    if func.blocks.len() > MAX_BLOCKS || func.values.len() > MAX_VALUES {
        return false;
    }
    let mut instruction_count = 0usize;
    let mut type_nodes = 0usize;
    let mut proof_work = func.params.len();
    let mut operand_edges = func.params.len();
    let mut const_nodes = 0usize;
    let mut const_string_bytes = 0usize;
    if !spend_type_budget(&func.ret, &mut type_nodes, &mut proof_work, MAX_TYPE_NODES) {
        return false;
    }
    for value in &func.values {
        if !spend_type_budget(&value.ty, &mut type_nodes, &mut proof_work, MAX_TYPE_NODES) {
            return false;
        }
        if let ValueKind::Const(value) = &value.kind {
            if !spend_const_budget(
                value,
                &mut type_nodes,
                &mut proof_work,
                &mut const_nodes,
                &mut const_string_bytes,
            ) {
                return false;
            }
        }
    }
    for block in &func.blocks {
        let Some(count) = instruction_count.checked_add(block.insts.len()) else {
            return false;
        };
        if count > MAX_INSTRUCTIONS {
            return false;
        }
        instruction_count = count;
        for inst in &block.insts {
            proof_work = proof_work.saturating_add(1);
            match &inst.kind {
                InstKind::Phi { incomings } => {
                    operand_edges = operand_edges.saturating_add(incomings.len());
                }
                InstKind::Gep { indices, .. } => {
                    operand_edges = operand_edges.saturating_add(indices.len().saturating_add(1));
                    proof_work = proof_work.saturating_add(indices.len());
                }
                InstKind::Call { args, .. } => {
                    operand_edges = operand_edges.saturating_add(args.len());
                }
                InstKind::Alloca { ty } => {
                    if !spend_type_budget(ty, &mut type_nodes, &mut proof_work, MAX_TYPE_NODES) {
                        return false;
                    }
                }
                _ => {}
            }
            if proof_work > MAX_POINTER_PROOF_WORK || operand_edges > MAX_OPERAND_EDGES {
                return false;
            }
        }
    }
    proof_work =
        proof_work.saturating_add(func.values.len().saturating_mul(MAX_POINTER_CHAIN_DEPTH));
    proof_work <= MAX_POINTER_PROOF_WORK
}

fn spend_const_budget(
    root: &crate::ir::Const,
    type_nodes: &mut usize,
    proof_work: &mut usize,
    const_nodes: &mut usize,
    string_bytes: &mut usize,
) -> bool {
    let mut stack = vec![(root, 1usize)];
    while let Some((value, depth)) = stack.pop() {
        if depth > MAX_SINGLE_TYPE_DEPTH {
            return false;
        }
        *const_nodes = const_nodes.saturating_add(1);
        *proof_work = proof_work.saturating_add(1);
        if *const_nodes > MAX_CONST_NODES || *proof_work > MAX_POINTER_PROOF_WORK {
            return false;
        }
        match value {
            crate::ir::Const::Zero(ty) => {
                if !spend_type_budget(ty, type_nodes, proof_work, MAX_TYPE_NODES) {
                    return false;
                }
            }
            crate::ir::Const::String(value) => {
                *string_bytes = string_bytes.saturating_add(value.len());
                if *string_bytes > MAX_CONST_STRING_BYTES {
                    return false;
                }
            }
            crate::ir::Const::Array(values) => {
                let pending = const_nodes.saturating_add(stack.len());
                if values.len() > MAX_CONST_NODES.saturating_sub(pending) {
                    return false;
                }
                stack.extend(values.iter().map(|value| (value, depth.saturating_add(1))));
            }
            crate::ir::Const::Int(_) | crate::ir::Const::Bool(_) | crate::ir::Const::Float(_) => {}
        }
    }
    true
}

fn spend_type_budget(
    ty: &Type,
    total_nodes: &mut usize,
    total_work: &mut usize,
    max_nodes: usize,
) -> bool {
    let Some(nodes) = type_node_count(ty, max_nodes.saturating_sub(*total_nodes)) else {
        return false;
    };
    *total_nodes = total_nodes.saturating_add(nodes);
    *total_work = total_work.saturating_add(nodes);
    *total_nodes <= max_nodes && *total_work <= MAX_POINTER_PROOF_WORK
}

fn type_node_count(mut ty: &Type, remaining: usize) -> Option<usize> {
    let mut nodes = 0usize;
    loop {
        nodes = nodes.checked_add(1)?;
        if nodes > remaining || nodes > MAX_SINGLE_TYPE_DEPTH {
            return None;
        }
        match ty {
            Type::Ptr(inner) => ty = inner,
            Type::Array { elem, .. } => ty = elem,
            Type::Void | Type::I1 | Type::I32 | Type::F32 => return Some(nodes),
        }
    }
}

fn tracked_pointer(func: &Function, value: ValueId) -> bool {
    // 目前只跟踪非数组 alloca，避免数组/复杂别名导致错误转发。
    let Type::Ptr(inner) = &func.value(value).ty else {
        return false;
    };
    if matches!(**inner, Type::Array { .. }) {
        return false;
    }
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return false;
    };
    matches!(
        func.block(block).insts[inst_idx].kind,
        InstKind::Alloca { .. }
    )
}

fn resolve(value: ValueId, replacements: &ValueReplacements) -> ValueId {
    super::util::resolve_replacement(value, replacements)
}
