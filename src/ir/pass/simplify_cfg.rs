mod structural;

use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::LoopInfo;
use super::util::{get_or_add_i32_const, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CastOp, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::{HashMap, HashSet};
use structural::{
    forward_empty_jump_block, forward_empty_jump_block_except, merge_linear_block,
    remove_unreachable_blocks,
};

pub(super) struct SimplifyCfgPass {
    preserve_loop_preheaders: bool,
}

const MAX_IF_CONVERSIONS: usize = 64;
const MAX_PREHEADER_ANALYSIS_BLOCKS: usize = 256;

impl SimplifyCfgPass {
    pub(super) fn new() -> Self {
        Self {
            preserve_loop_preheaders: false,
        }
    }

    /// Keeps empty dedicated loop preheaders until shape-sensitive loop
    /// transforms have consumed them. A later full CFG simplification still
    /// forwards the remaining empty blocks.
    pub(super) fn preserving_loop_preheaders() -> Self {
        Self {
            preserve_loop_preheaders: true,
        }
    }
}

impl ModulePass for SimplifyCfgPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            simplify_function(func, self.preserve_loop_preheaders);
        }
    }
}

fn simplify_function(func: &mut Function, preserve_loop_preheaders: bool) {
    let mut changed = false;
    let thread_source_limit = func.blocks.len();
    let mut attempted_thread_sources = HashSet::<BlockId>::new();
    let mut thread_attempt_budget = 64usize;
    loop {
        let mut round_changed = remove_unreachable_blocks(func);
        round_changed |= simplify_branches(func);
        round_changed |= if_convert_short_circuit_booleans(func);
        round_changed |= if_convert_conditional_updates(func);
        round_changed |= if_convert_boolean_stores(func);
        round_changed |= thread_boolean_phi_branches(func);
        if !preserve_loop_preheaders {
            round_changed |= thread_known_integer_edges(
                func,
                thread_source_limit,
                &mut attempted_thread_sources,
                &mut thread_attempt_budget,
            );
        }
        round_changed |= simplify_trivial_phis(func);
        if !preserve_loop_preheaders {
            round_changed |= merge_equivalent_phi_edge_updates(func);
        }
        if preserve_loop_preheaders && func.blocks.len() <= MAX_PREHEADER_ANALYSIS_BLOCKS {
            let cfg = ControlFlowGraph::new(func);
            let dom = Dominators::new(func, &cfg);
            let protected = LoopInfo::new(&cfg, &dom)
                .loops()
                .iter()
                .filter_map(|natural_loop| natural_loop.dedicated_preheader)
                .collect();
            round_changed |= forward_empty_jump_block_except(func, &protected);
        } else if !preserve_loop_preheaders {
            round_changed |= forward_empty_jump_block(func);
        }
        round_changed |= merge_linear_block(func);
        changed |= round_changed;
        if !round_changed {
            break;
        }
    }
    if changed {
        if let Err(errors) = func.verify() {
            panic!(
                "simplify-cfg produced invalid IR in {}: {:?}",
                func.name, errors
            );
        }
    }
}

/// Merges duplicated pure updates at two incoming edges of the same phi into
/// a shared tail block. This is ordinary CFG tail merging and, in particular,
/// canonicalizes loops with `continue` into a unique latch for later LSR.
fn merge_equivalent_phi_edge_updates(func: &mut Function) -> bool {
    const MAX_TAIL_MERGE_BLOCKS: usize = 1024;
    const MAX_TAIL_MERGE_PAIRS: usize = 32_768;

    if func.blocks.len() > MAX_TAIL_MERGE_BLOCKS {
        return false;
    }
    let use_counts = value_use_counts(func);
    let predecessors = all_predecessors(func);
    let mut proof_work = 0usize;

    for header_idx in 0..func.blocks.len() {
        let header = BlockId(header_idx);
        let phis = func.blocks[header_idx]
            .insts
            .iter()
            .enumerate()
            .filter(|(_, inst)| !matches!(inst.kind, InstKind::Nop))
            .take_while(|(_, inst)| matches!(inst.kind, InstKind::Phi { .. }))
            .filter_map(|(idx, inst)| {
                let (Some(result), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
                    return None;
                };
                Some((idx, result, incomings.clone()))
            })
            .collect::<Vec<_>>();
        if phis.is_empty() {
            continue;
        }
        let mut incoming_edges = predecessors[header_idx]
            .iter()
            .copied()
            .filter(|pred| {
                matches!(func.blocks[pred.0].terminator, Some(Terminator::Jump(target)) if target == header)
            })
            .collect::<Vec<_>>();
        incoming_edges.sort_by_key(|block| block.0);

        for lhs_idx in 0..incoming_edges.len() {
            for rhs_idx in lhs_idx + 1..incoming_edges.len() {
                proof_work = proof_work.saturating_add(1);
                if proof_work > MAX_TAIL_MERGE_PAIRS {
                    return false;
                }
                let lhs_pred = incoming_edges[lhs_idx];
                let rhs_pred = incoming_edges[rhs_idx];
                let mut differing = None;
                let mut merged_incomings = Vec::with_capacity(phis.len());
                let mut valid = true;

                for (phi_idx, _, incomings) in &phis {
                    let Some(lhs) = phi_incoming(incomings, lhs_pred) else {
                        valid = false;
                        break;
                    };
                    let Some(rhs) = phi_incoming(incomings, rhs_pred) else {
                        valid = false;
                        break;
                    };
                    if lhs == rhs {
                        merged_incomings.push((*phi_idx, lhs));
                        continue;
                    }
                    if differing.is_some()
                        || use_counts.get(lhs.0) != Some(&1)
                        || use_counts.get(rhs.0) != Some(&1)
                    {
                        valid = false;
                        break;
                    }
                    let Some((lhs_inst_idx, lhs_kind)) = edge_tail_definition(func, lhs_pred, lhs)
                    else {
                        valid = false;
                        break;
                    };
                    let Some((rhs_inst_idx, rhs_kind)) = edge_tail_definition(func, rhs_pred, rhs)
                    else {
                        valid = false;
                        break;
                    };
                    if !equivalent_tail_inst(func, &lhs_kind, &rhs_kind)
                        || func.value(lhs).ty != func.value(rhs).ty
                        || !tail_mergeable_inst(&lhs_kind)
                    {
                        valid = false;
                        break;
                    }
                    differing = Some((
                        *phi_idx,
                        lhs,
                        rhs,
                        lhs_inst_idx,
                        rhs_inst_idx,
                        lhs_kind,
                        func.value(lhs).ty.clone(),
                    ));
                    merged_incomings.push((*phi_idx, lhs));
                }
                let Some((
                    differing_phi,
                    lhs_value,
                    rhs_value,
                    lhs_inst_idx,
                    rhs_inst_idx,
                    kind,
                    ty,
                )) = differing
                else {
                    continue;
                };
                if !valid {
                    continue;
                }

                let tail = func.add_block("tail.merge");
                let merged = func
                    .append_inst(tail, kind, Some(ty))
                    .expect("a cloned tail update must produce a value");
                func.blocks[tail.0].terminator = Some(Terminator::Jump(header));
                func.blocks[lhs_pred.0].terminator = Some(Terminator::Jump(tail));
                func.blocks[rhs_pred.0].terminator = Some(Terminator::Jump(tail));
                func.blocks[lhs_pred.0].insts[lhs_inst_idx] = Inst {
                    result: None,
                    kind: InstKind::Nop,
                };
                func.blocks[rhs_pred.0].insts[rhs_inst_idx] = Inst {
                    result: None,
                    kind: InstKind::Nop,
                };

                for (phi_idx, old_value) in merged_incomings {
                    let InstKind::Phi { incomings } =
                        &mut func.blocks[header_idx].insts[phi_idx].kind
                    else {
                        unreachable!("tail merge candidate must remain a phi")
                    };
                    incomings.retain(|(pred, _)| *pred != lhs_pred && *pred != rhs_pred);
                    incomings.push((
                        tail,
                        if phi_idx == differing_phi {
                            merged
                        } else {
                            old_value
                        },
                    ));
                }
                debug_assert_ne!(lhs_value, rhs_value);
                if let Err(errors) = func.verify() {
                    panic!(
                        "tail merging produced invalid IR in {}: {:?}",
                        func.name, errors
                    );
                }
                return true;
            }
        }
    }
    false
}

fn edge_tail_definition(
    func: &Function,
    block: BlockId,
    value: ValueId,
) -> Option<(usize, InstKind)> {
    let ValueKind::Inst(owner, inst_idx) = func.value(value).kind else {
        return None;
    };
    if owner != block {
        return None;
    }
    let last_active = func.blocks[block.0]
        .insts
        .iter()
        .rposition(|inst| !matches!(inst.kind, InstKind::Nop))?;
    let inst = func.blocks[block.0].insts.get(inst_idx)?;
    (inst_idx == last_active && inst.result == Some(value)).then(|| (inst_idx, inst.kind.clone()))
}

fn tail_mergeable_inst(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Unary { .. }
            | InstKind::Binary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. }
            | InstKind::Gep { .. }
    )
}

fn equivalent_tail_inst(func: &Function, lhs: &InstKind, rhs: &InstKind) -> bool {
    let value_eq = |lhs: ValueId, rhs: ValueId| {
        lhs == rhs
            || matches!(
                (&func.value(lhs).kind, &func.value(rhs).kind),
                (ValueKind::Const(lhs), ValueKind::Const(rhs)) if lhs == rhs
            )
    };
    match (lhs, rhs) {
        (
            InstKind::Unary {
                op: lhs_op,
                value: lhs,
            },
            InstKind::Unary {
                op: rhs_op,
                value: rhs,
            },
        ) => lhs_op == rhs_op && value_eq(*lhs, *rhs),
        (
            InstKind::Binary {
                op: lhs_op,
                lhs: lhs_lhs,
                rhs: lhs_rhs,
            },
            InstKind::Binary {
                op: rhs_op,
                lhs: rhs_lhs,
                rhs: rhs_rhs,
            },
        ) => lhs_op == rhs_op && value_eq(*lhs_lhs, *rhs_lhs) && value_eq(*lhs_rhs, *rhs_rhs),
        (
            InstKind::Icmp {
                op: lhs_op,
                lhs: lhs_lhs,
                rhs: lhs_rhs,
            },
            InstKind::Icmp {
                op: rhs_op,
                lhs: rhs_lhs,
                rhs: rhs_rhs,
            },
        ) => lhs_op == rhs_op && value_eq(*lhs_lhs, *rhs_lhs) && value_eq(*lhs_rhs, *rhs_rhs),
        (
            InstKind::Fcmp {
                op: lhs_op,
                lhs: lhs_lhs,
                rhs: lhs_rhs,
            },
            InstKind::Fcmp {
                op: rhs_op,
                lhs: rhs_lhs,
                rhs: rhs_rhs,
            },
        ) => lhs_op == rhs_op && value_eq(*lhs_lhs, *rhs_lhs) && value_eq(*lhs_rhs, *rhs_rhs),
        (
            InstKind::Cast {
                op: lhs_op,
                value: lhs,
            },
            InstKind::Cast {
                op: rhs_op,
                value: rhs,
            },
        ) => lhs_op == rhs_op && value_eq(*lhs, *rhs),
        (
            InstKind::Gep {
                base: lhs_base,
                indices: lhs_indices,
            },
            InstKind::Gep {
                base: rhs_base,
                indices: rhs_indices,
            },
        ) => {
            value_eq(*lhs_base, *rhs_base)
                && lhs_indices.len() == rhs_indices.len()
                && lhs_indices
                    .iter()
                    .zip(rhs_indices)
                    .all(|(lhs, rhs)| value_eq(*lhs, *rhs))
        }
        _ => false,
    }
}

#[derive(Clone)]
struct BooleanStoreArm {
    block: BlockId,
    pointer: ValueId,
    value: BooleanStoreValue,
    gep: Option<(usize, InstKind)>,
    value_insts: Vec<(usize, ValueId, InstKind)>,
    value_cast_idx: Option<usize>,
    store_idx: usize,
    merge: BlockId,
}

#[derive(Clone)]
enum BooleanStoreValue {
    Constant(bool),
    Dynamic(ValueId),
}

/// If-converts a diamond whose two arms store boolean 0/1 to the same exact
/// address.  This is the ordinary SimplifyCFG/if-conversion transform: the
/// control condition becomes an i32 value and only one store remains.
fn if_convert_boolean_stores(func: &mut Function) -> bool {
    let predecessors = all_predecessors(func);
    let use_counts = value_use_counts(func);
    let cfg = ControlFlowGraph::new(func);
    let dominators = Dominators::new(func, &cfg);
    let mut candidate = None;

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let Some(Terminator::Branch {
            cond,
            then_target,
            else_target,
        }) = block.terminator.as_ref()
        else {
            continue;
        };
        if then_target == else_target
            || predecessors[then_target.0] != HashSet::from([BlockId(block_idx)])
            || predecessors[else_target.0] != HashSet::from([BlockId(block_idx)])
        {
            continue;
        }
        let owner = BlockId(block_idx);
        let Some(then_arm) = boolean_store_arm(func, *then_target, owner, &use_counts, &dominators)
        else {
            continue;
        };
        let Some(else_arm) = boolean_store_arm(func, *else_target, owner, &use_counts, &dominators)
        else {
            continue;
        };
        if then_arm.merge != else_arm.merge
            || merge_has_phis(func, then_arm.merge)
            || !equivalent_store_pointers(func, &then_arm, &else_arm)
            || !boolean_store_values_are_selectable(&then_arm.value, &else_arm.value)
        {
            continue;
        }
        let pointer_operands_available = then_arm
            .gep
            .as_ref()
            .map_or_else(|| vec![then_arm.pointer], |(_, kind)| inst_operands(kind))
            .into_iter()
            .all(|value| value_available_at_block(func, &dominators, value, owner));
        if !pointer_operands_available {
            continue;
        }
        candidate = Some((owner, *cond, then_arm, else_arm));
        break;
    }

    let Some((block, cond, then_arm, else_arm)) = candidate else {
        return false;
    };
    let before = func.clone();
    let Some(then_value) = materialize_boolean_store_value(func, block, &then_arm) else {
        return false;
    };
    let Some(else_value) = materialize_boolean_store_value(func, block, &else_arm) else {
        *func = before;
        return false;
    };
    let Some(selected_bool) =
        combine_boolean_store_values(func, block, cond, then_value, else_value)
    else {
        *func = before;
        return false;
    };
    let selected = match selected_bool {
        BooleanStoreValue::Constant(value) => get_or_add_i32_const(func, i32::from(value)),
        BooleanStoreValue::Dynamic(value) => func
            .append_inst(
                block,
                InstKind::Cast {
                    op: CastOp::BoolToI32,
                    value,
                },
                Some(Type::I32),
            )
            .expect("a boolean store selection must produce i32"),
    };
    let pointer = if let Some((_, kind)) = &then_arm.gep {
        func.append_inst(
            block,
            kind.clone(),
            Some(func.value(then_arm.pointer).ty.clone()),
        )
        .expect("an if-converted store address must produce a pointer")
    } else {
        then_arm.pointer
    };
    func.append_inst(
        block,
        InstKind::Store {
            ptr: pointer,
            value: selected,
        },
        None,
    );
    func.blocks[block.0].terminator = Some(Terminator::Jump(then_arm.merge));
    clear_boolean_store_arm(func, &then_arm);
    clear_boolean_store_arm(func, &else_arm);
    if func.verify().is_err() {
        *func = before;
        false
    } else {
        true
    }
}

fn boolean_store_arm(
    func: &Function,
    block: BlockId,
    owner: BlockId,
    use_counts: &[usize],
    dominators: &Dominators,
) -> Option<BooleanStoreArm> {
    let Terminator::Jump(merge) = func.blocks.get(block.0)?.terminator.as_ref()? else {
        return None;
    };
    let active = active_insts(func, block);
    let store_idx = *active.last()?;
    let InstKind::Store { ptr, value } = func.blocks[block.0].insts[store_idx].kind else {
        return None;
    };
    let gep = defining_inst_in_block(func, block, ptr).and_then(|(idx, inst)| {
        (matches!(inst.kind, InstKind::Gep { .. }) && use_counts.get(ptr.0) == Some(&1))
            .then_some((idx, inst.kind.clone()))
    });
    if matches!(func.value(ptr).kind, ValueKind::Inst(def_block, _) if def_block == block)
        && gep.is_none()
    {
        return None;
    }
    let (stored_value, value_insts, value_cast_idx) =
        analyze_boolean_store_value(func, block, owner, value, use_counts, dominators)?;
    let consumed = std::iter::once(store_idx)
        .chain(gep.iter().map(|(idx, _)| *idx))
        .chain(value_insts.iter().map(|(idx, _, _)| *idx))
        .chain(value_cast_idx)
        .collect::<HashSet<_>>();
    if active.iter().any(|idx| !consumed.contains(idx)) {
        return None;
    }
    Some(BooleanStoreArm {
        block,
        pointer: ptr,
        value: stored_value,
        gep,
        value_insts,
        value_cast_idx,
        store_idx,
        merge: *merge,
    })
}

fn analyze_boolean_store_value(
    func: &Function,
    block: BlockId,
    owner: BlockId,
    stored: ValueId,
    use_counts: &[usize],
    dominators: &Dominators,
) -> Option<(
    BooleanStoreValue,
    Vec<(usize, ValueId, InstKind)>,
    Option<usize>,
)> {
    if let Some(value) = const_i32_value(func, stored) {
        return matches!(value, 0 | 1).then_some((
            BooleanStoreValue::Constant(value != 0),
            Vec::new(),
            None,
        ));
    }
    let (cast_idx, cast) = defining_inst_in_block(func, block, stored)?;
    let InstKind::Cast {
        op: CastOp::BoolToI32,
        value: source,
    } = cast.kind
    else {
        return None;
    };
    if use_counts.get(stored.0) != Some(&1) || func.value(source).ty != Type::I1 {
        return None;
    }
    let mut insts = Vec::new();
    let mut visiting = HashSet::new();
    collect_speculatable_boolean_value(
        func,
        block,
        owner,
        source,
        use_counts,
        dominators,
        &mut visiting,
        &mut insts,
    )?;
    Some((BooleanStoreValue::Dynamic(source), insts, Some(cast_idx)))
}

fn collect_speculatable_boolean_value(
    func: &Function,
    block: BlockId,
    owner: BlockId,
    value: ValueId,
    use_counts: &[usize],
    dominators: &Dominators,
    visiting: &mut HashSet<ValueId>,
    insts: &mut Vec<(usize, ValueId, InstKind)>,
) -> Option<()> {
    let ValueKind::Inst(def_block, inst_idx) = func.value(value).kind else {
        return value_available_at_block(func, dominators, value, owner).then_some(());
    };
    if def_block != block {
        return value_available_at_block(func, dominators, value, owner).then_some(());
    }
    if insts.iter().any(|(_, result, _)| *result == value) {
        return Some(());
    }
    if !visiting.insert(value) {
        return None;
    }
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    if inst.result != Some(value)
        || func.value(value).ty != Type::I1
        || !is_speculatable_bool_inst(&inst.kind)
        || use_counts.get(value.0) != Some(&1)
    {
        return None;
    }
    for operand in bool_inst_operands(&inst.kind) {
        collect_speculatable_boolean_value(
            func, block, owner, operand, use_counts, dominators, visiting, insts,
        )?;
    }
    visiting.remove(&value);
    insts.push((inst_idx, value, inst.kind.clone()));
    Some(())
}

fn defining_inst_in_block(
    func: &Function,
    block: BlockId,
    value: ValueId,
) -> Option<(usize, &Inst)> {
    let ValueKind::Inst(owner, inst_idx) = func.value(value).kind else {
        return None;
    };
    if owner != block {
        return None;
    }
    func.blocks
        .get(block.0)?
        .insts
        .get(inst_idx)
        .map(|inst| (inst_idx, inst))
}

fn boolean_store_values_are_selectable(
    then_value: &BooleanStoreValue,
    else_value: &BooleanStoreValue,
) -> bool {
    matches!(then_value, BooleanStoreValue::Constant(_))
        || matches!(else_value, BooleanStoreValue::Constant(_))
        || matches!(
            (then_value, else_value),
            (BooleanStoreValue::Dynamic(lhs), BooleanStoreValue::Dynamic(rhs)) if lhs == rhs
        )
}

fn materialize_boolean_store_value(
    func: &mut Function,
    block: BlockId,
    arm: &BooleanStoreArm,
) -> Option<BooleanStoreValue> {
    let BooleanStoreValue::Dynamic(source) = &arm.value else {
        return Some(arm.value.clone());
    };
    let mut replacements = HashMap::new();
    for (_, old_result, old_kind) in &arm.value_insts {
        let kind = remap_boolean_inst(old_kind, &replacements)?;
        let result = func.append_inst(block, kind, Some(Type::I1))?;
        replacements.insert(*old_result, result);
    }
    Some(BooleanStoreValue::Dynamic(
        *replacements.get(source).unwrap_or(source),
    ))
}

fn remap_boolean_inst(
    kind: &InstKind,
    replacements: &HashMap<ValueId, ValueId>,
) -> Option<InstKind> {
    let map = |value: &ValueId| replacements.get(value).copied().unwrap_or(*value);
    match kind {
        InstKind::Unary { op, value } => Some(InstKind::Unary {
            op: *op,
            value: map(value),
        }),
        InstKind::Binary { op, lhs, rhs } => Some(InstKind::Binary {
            op: *op,
            lhs: map(lhs),
            rhs: map(rhs),
        }),
        InstKind::Icmp { op, lhs, rhs } => Some(InstKind::Icmp {
            op: *op,
            lhs: map(lhs),
            rhs: map(rhs),
        }),
        InstKind::Cast {
            op: CastOp::I32ToBool,
            value,
        } => Some(InstKind::Cast {
            op: CastOp::I32ToBool,
            value: map(value),
        }),
        _ => None,
    }
}

fn combine_boolean_store_values(
    func: &mut Function,
    block: BlockId,
    condition: ValueId,
    then_value: BooleanStoreValue,
    else_value: BooleanStoreValue,
) -> Option<BooleanStoreValue> {
    match (then_value, else_value) {
        (BooleanStoreValue::Constant(lhs), BooleanStoreValue::Constant(rhs)) if lhs == rhs => {
            Some(BooleanStoreValue::Constant(lhs))
        }
        (BooleanStoreValue::Constant(true), BooleanStoreValue::Constant(false)) => {
            Some(BooleanStoreValue::Dynamic(condition))
        }
        (BooleanStoreValue::Constant(false), BooleanStoreValue::Constant(true)) => func
            .append_inst(
                block,
                InstKind::Unary {
                    op: crate::ir::UnaryOp::Not,
                    value: condition,
                },
                Some(Type::I1),
            )
            .map(BooleanStoreValue::Dynamic),
        (BooleanStoreValue::Constant(true), BooleanStoreValue::Dynamic(rhs)) => func
            .append_inst(
                block,
                InstKind::Binary {
                    op: BinaryOp::Or,
                    lhs: condition,
                    rhs,
                },
                Some(Type::I1),
            )
            .map(BooleanStoreValue::Dynamic),
        (BooleanStoreValue::Constant(false), BooleanStoreValue::Dynamic(rhs)) => {
            let inverted = func.append_inst(
                block,
                InstKind::Unary {
                    op: crate::ir::UnaryOp::Not,
                    value: condition,
                },
                Some(Type::I1),
            )?;
            func.append_inst(
                block,
                InstKind::Binary {
                    op: BinaryOp::And,
                    lhs: inverted,
                    rhs,
                },
                Some(Type::I1),
            )
            .map(BooleanStoreValue::Dynamic)
        }
        (BooleanStoreValue::Dynamic(lhs), BooleanStoreValue::Constant(true)) => {
            let inverted = func.append_inst(
                block,
                InstKind::Unary {
                    op: crate::ir::UnaryOp::Not,
                    value: condition,
                },
                Some(Type::I1),
            )?;
            func.append_inst(
                block,
                InstKind::Binary {
                    op: BinaryOp::Or,
                    lhs: inverted,
                    rhs: lhs,
                },
                Some(Type::I1),
            )
            .map(BooleanStoreValue::Dynamic)
        }
        (BooleanStoreValue::Dynamic(lhs), BooleanStoreValue::Constant(false)) => func
            .append_inst(
                block,
                InstKind::Binary {
                    op: BinaryOp::And,
                    lhs: condition,
                    rhs: lhs,
                },
                Some(Type::I1),
            )
            .map(BooleanStoreValue::Dynamic),
        (BooleanStoreValue::Dynamic(lhs), BooleanStoreValue::Dynamic(rhs)) if lhs == rhs => {
            Some(BooleanStoreValue::Dynamic(lhs))
        }
        _ => None,
    }
}

fn equivalent_store_pointers(
    func: &Function,
    lhs: &BooleanStoreArm,
    rhs: &BooleanStoreArm,
) -> bool {
    if lhs.pointer == rhs.pointer {
        return true;
    }
    matches!(
        (&lhs.gep, &rhs.gep),
        (Some((_, lhs_kind)), Some((_, rhs_kind)))
            if lhs_kind == rhs_kind && func.value(lhs.pointer).ty == func.value(rhs.pointer).ty
    )
}

fn clear_boolean_store_arm(func: &mut Function, arm: &BooleanStoreArm) {
    if let Some((gep_idx, _)) = arm.gep {
        func.blocks[arm.block.0].insts[gep_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
    }
    for (inst_idx, _, _) in &arm.value_insts {
        func.blocks[arm.block.0].insts[*inst_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
    }
    if let Some(inst_idx) = arm.value_cast_idx {
        func.blocks[arm.block.0].insts[inst_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
    }
    func.blocks[arm.block.0].insts[arm.store_idx] = Inst {
        result: None,
        kind: InstKind::Nop,
    };
}

fn merge_has_phis(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0]
        .insts
        .iter()
        .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
}

fn const_i32_value(func: &Function, value: ValueId) -> Option<i32> {
    match func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(value),
        ValueKind::Const(Const::Bool(value)) => Some(i32::from(value)),
        ValueKind::Const(Const::Zero(Type::I32 | Type::I1)) => Some(0),
        _ => None,
    }
}

fn simplify_branches(func: &mut Function) -> bool {
    // 先替换 terminator，并记录被删除的 CFG 边；最后再统一修 phi incoming。
    let mut removed_edges = Vec::new();
    let mut changed = false;

    for block_idx in 0..func.blocks.len() {
        let Some(terminator) = func.blocks[block_idx].terminator.clone() else {
            continue;
        };

        let (replacement, removed) = match terminator {
            // then/else 相同的条件跳转没有意义，直接改成无条件跳转。
            Terminator::Branch {
                cond: _,
                then_target,
                else_target,
            } if then_target == else_target => (Terminator::Jump(then_target), Vec::new()),
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => match const_bool(func, cond) {
                // 条件已经是常量时，删掉永远不会走到的那条边。
                Some(true) => (
                    Terminator::Jump(then_target),
                    vec![(BlockId(block_idx), else_target)],
                ),
                Some(false) => (
                    Terminator::Jump(else_target),
                    vec![(BlockId(block_idx), then_target)],
                ),
                None => continue,
            },
            _ => continue,
        };

        func.blocks[block_idx].terminator = Some(replacement);
        removed_edges.extend(removed);
        changed = true;
    }

    for (pred, target) in removed_edges {
        remove_phi_incomings(func, pred, target);
    }
    changed
}

fn if_convert_short_circuit_booleans(func: &mut Function) -> bool {
    let mut changed = false;
    let mut rejected = HashSet::new();
    let mut attempts = 0usize;
    while attempts < MAX_IF_CONVERSIONS {
        let predecessors = all_predecessors(func);
        let use_counts = value_use_counts(func);
        let cfg = ControlFlowGraph::new(func);
        let dominators = Dominators::new(func, &cfg);
        let mut candidate = None;

        for (merge_idx, merge) in func.blocks.iter().enumerate() {
            let active = active_insts(func, BlockId(merge_idx));
            let [phi_idx] = active.as_slice() else {
                continue;
            };
            if rejected.contains(&(merge_idx, *phi_idx)) {
                continue;
            }
            let phi = &merge.insts[*phi_idx];
            let (Some(phi_result), InstKind::Phi { incomings }) = (phi.result, &phi.kind) else {
                continue;
            };
            if func.value(phi_result).ty != Type::I1
                || use_counts[phi_result.0] != 1
                || incomings.len() != 2
            {
                continue;
            }
            let merge_block = BlockId(merge_idx);
            for (direct_pred, constant) in incomings {
                let Some(constant) = const_bool(func, *constant) else {
                    continue;
                };
                let Some((rhs_block, rhs_value)) = incomings
                    .iter()
                    .find(|(pred, _)| pred != direct_pred)
                    .copied()
                else {
                    continue;
                };
                if predecessors[merge_idx] != HashSet::from([*direct_pred, rhs_block])
                    || predecessors[rhs_block.0] != HashSet::from([*direct_pred])
                    || !matches!(
                        func.blocks[rhs_block.0].terminator,
                        Some(Terminator::Jump(target)) if target == merge_block
                    )
                {
                    continue;
                }
                let Some(Terminator::Branch {
                    cond,
                    then_target,
                    else_target,
                }) = func.blocks[direct_pred.0].terminator.as_ref()
                else {
                    continue;
                };
                let direct_when_true = if *then_target == merge_block && *else_target == rhs_block {
                    true
                } else if *else_target == merge_block && *then_target == rhs_block {
                    false
                } else {
                    continue;
                };
                let rhs_active = active_insts(func, rhs_block);
                let [rhs_idx] = rhs_active.as_slice() else {
                    continue;
                };
                let rhs_inst = &func.blocks[rhs_block.0].insts[*rhs_idx];
                if rhs_inst.result != Some(rhs_value)
                    || func.value(rhs_value).ty != Type::I1
                    || use_counts[rhs_value.0] != 1
                    || !is_speculatable_bool_inst(&rhs_inst.kind)
                    || bool_inst_operands(&rhs_inst.kind)
                        .into_iter()
                        .any(|operand| {
                            !value_available_at_block(func, &dominators, operand, *direct_pred)
                        })
                {
                    continue;
                }
                candidate = Some((
                    *direct_pred,
                    *cond,
                    rhs_block,
                    *rhs_idx,
                    rhs_inst.kind.clone(),
                    merge_block,
                    *phi_idx,
                    phi_result,
                    direct_when_true,
                    constant,
                ));
                break;
            }
            if candidate.is_some() {
                break;
            }
        }

        let Some((
            direct_pred,
            condition,
            rhs_block,
            rhs_idx,
            rhs_kind,
            merge,
            phi_idx,
            phi_result,
            direct_when_true,
            constant,
        )) = candidate
        else {
            break;
        };
        attempts += 1;
        let before = func.clone();
        let rhs = func
            .append_inst(direct_pred, rhs_kind, Some(Type::I1))
            .expect("speculated boolean instruction must produce a value");
        let (lhs, op) = match (direct_when_true, constant) {
            (true, true) => (condition, BinaryOp::Or),
            (false, false) => (condition, BinaryOp::And),
            (true, false) => {
                let inverted = func
                    .append_inst(
                        direct_pred,
                        InstKind::Unary {
                            op: crate::ir::UnaryOp::Not,
                            value: condition,
                        },
                        Some(Type::I1),
                    )
                    .expect("boolean inversion must produce a value");
                (inverted, BinaryOp::And)
            }
            (false, true) => {
                let inverted = func
                    .append_inst(
                        direct_pred,
                        InstKind::Unary {
                            op: crate::ir::UnaryOp::Not,
                            value: condition,
                        },
                        Some(Type::I1),
                    )
                    .expect("boolean inversion must produce a value");
                (inverted, BinaryOp::Or)
            }
        };
        let combined = func
            .append_inst(
                direct_pred,
                InstKind::Binary { op, lhs, rhs },
                Some(Type::I1),
            )
            .expect("combined boolean must produce a value");
        func.blocks[direct_pred.0].terminator = Some(Terminator::Jump(merge));
        func.blocks[rhs_block.0].insts[rhs_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        func.blocks[merge.0].insts[phi_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        rewrite_function_uses(func, &ValueReplacements::from([(phi_result, combined)]));
        if func.verify().is_err() {
            *func = before;
            rejected.insert((merge.0, phi_idx));
        } else {
            changed = true;
        }
    }
    changed
}

fn is_speculatable_bool_inst(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Unary {
            op: crate::ir::UnaryOp::Not,
            ..
        } | InstKind::Binary {
            op: BinaryOp::And | BinaryOp::Or,
            ..
        } | InstKind::Icmp { .. }
            | InstKind::Cast {
                op: CastOp::I32ToBool,
                ..
            }
    )
}

fn bool_inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. } | InstKind::Icmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        _ => Vec::new(),
    }
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
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

fn if_convert_conditional_updates(func: &mut Function) -> bool {
    let mut changed = false;
    let mut rejected = HashSet::new();
    let mut attempts = 0usize;
    while attempts < MAX_IF_CONVERSIONS {
        let predecessors = all_predecessors(func);
        let use_counts = value_use_counts(func);
        let cfg = ControlFlowGraph::new(func);
        let dominators = Dominators::new(func, &cfg);
        let mut candidate = None;

        for (block_idx, block) in func.blocks.iter().enumerate() {
            let Terminator::Branch {
                cond,
                then_target,
                else_target,
            } = block
                .terminator
                .as_ref()
                .cloned()
                .unwrap_or(Terminator::Return(None))
            else {
                continue;
            };
            if then_target == else_target || func.value(cond).ty != Type::I1 {
                continue;
            }
            let block = BlockId(block_idx);
            let then_active = active_insts(func, then_target);
            let else_active = active_insts(func, else_target);
            let then_jump = jump_target(func, then_target);
            let else_jump = jump_target(func, else_target);
            let (update_block, idle_predecessor, update, merge, update_when_true) =
                if then_jump == Some(else_target) {
                    let [update] = then_active.as_slice() else {
                        continue;
                    };
                    if predecessors[then_target.0] != HashSet::from([block]) {
                        continue;
                    }
                    (then_target, block, *update, else_target, true)
                } else if else_jump == Some(then_target) {
                    let [update] = else_active.as_slice() else {
                        continue;
                    };
                    if predecessors[else_target.0] != HashSet::from([block]) {
                        continue;
                    }
                    (else_target, block, *update, then_target, false)
                } else {
                    let Some(merge) = common_jump_target(func, then_target, else_target) else {
                        continue;
                    };
                    if predecessors[then_target.0] != HashSet::from([block])
                        || predecessors[else_target.0] != HashSet::from([block])
                    {
                        continue;
                    }
                    match (then_active.as_slice(), else_active.as_slice()) {
                        ([update], []) => (then_target, else_target, *update, merge, true),
                        ([], [update]) => (else_target, then_target, *update, merge, false),
                        _ => continue,
                    }
                };
            let Some(update_result) = func.blocks[update_block.0].insts[update].result else {
                continue;
            };
            let InstKind::Binary {
                op: update_op @ (BinaryOp::Iadd | BinaryOp::Isub),
                lhs: base,
                rhs: delta,
            } = func.blocks[update_block.0].insts[update].kind
            else {
                continue;
            };
            if use_counts[update_result.0] != 1
                || func.value(base).ty != Type::I32
                || func.value(delta).ty != Type::I32
                || !value_available_at_block(func, &dominators, base, BlockId(block_idx))
                || !value_available_at_block(func, &dominators, delta, BlockId(block_idx))
            {
                continue;
            }

            let leading_phis = func.blocks[merge.0]
                .insts
                .iter()
                .enumerate()
                .filter(|(_, inst)| !matches!(inst.kind, InstKind::Nop))
                .take_while(|(_, inst)| matches!(inst.kind, InstKind::Phi { .. }))
                .collect::<Vec<_>>();
            let [(phi_idx, phi)] = leading_phis.as_slice() else {
                continue;
            };
            if rejected.contains(&(block_idx, *phi_idx)) {
                continue;
            }
            let (Some(phi_result), InstKind::Phi { incomings }) = (phi.result, &phi.kind) else {
                continue;
            };
            if incomings.len() != 2
                || phi_incoming(incomings, update_block) != Some(update_result)
                || phi_incoming(incomings, idle_predecessor) != Some(base)
            {
                continue;
            }
            candidate = Some((
                BlockId(block_idx),
                cond,
                update_block,
                update,
                merge,
                *phi_idx,
                phi_result,
                update_op,
                base,
                delta,
                update_when_true,
            ));
            break;
        }

        let Some((
            block,
            cond,
            update_block,
            update_idx,
            merge,
            phi_idx,
            phi_result,
            update_op,
            base,
            delta,
            update_when_true,
        )) = candidate
        else {
            break;
        };
        attempts += 1;
        let before = func.clone();
        let selected_cond = if update_when_true {
            cond
        } else {
            func.append_inst(
                block,
                InstKind::Unary {
                    op: crate::ir::UnaryOp::Not,
                    value: cond,
                },
                Some(Type::I1),
            )
            .expect("conditional inversion must produce a value")
        };
        let as_i32 = func
            .append_inst(
                block,
                InstKind::Cast {
                    op: CastOp::BoolToI32,
                    value: selected_cond,
                },
                Some(Type::I32),
            )
            .expect("boolean cast must produce a value");
        let scaled = func
            .append_inst(
                block,
                InstKind::Binary {
                    op: BinaryOp::Imul,
                    lhs: delta,
                    rhs: as_i32,
                },
                Some(Type::I32),
            )
            .expect("conditional scale must produce a value");
        let selected = func
            .append_inst(
                block,
                InstKind::Binary {
                    op: update_op,
                    lhs: base,
                    rhs: scaled,
                },
                Some(Type::I32),
            )
            .expect("conditional update must produce a value");

        func.blocks[block.0].terminator = Some(Terminator::Jump(merge));
        func.blocks[update_block.0].insts[update_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        func.blocks[merge.0].insts[phi_idx] = Inst {
            result: None,
            kind: InstKind::Nop,
        };
        rewrite_function_uses(func, &ValueReplacements::from([(phi_result, selected)]));
        if func.verify().is_err() {
            *func = before;
            rejected.insert((block.0, phi_idx));
        } else {
            changed = true;
        }
    }
    changed
}

fn jump_target(func: &Function, block: BlockId) -> Option<BlockId> {
    match func.blocks[block.0].terminator.as_ref() {
        Some(Terminator::Jump(target)) => Some(*target),
        _ => None,
    }
}

fn common_jump_target(func: &Function, lhs: BlockId, rhs: BlockId) -> Option<BlockId> {
    match (
        func.blocks[lhs.0].terminator.as_ref(),
        func.blocks[rhs.0].terminator.as_ref(),
    ) {
        (Some(Terminator::Jump(lhs)), Some(Terminator::Jump(rhs))) if lhs == rhs => Some(*lhs),
        _ => None,
    }
}

fn active_insts(func: &Function, block: BlockId) -> Vec<usize> {
    func.blocks[block.0]
        .insts
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| (!matches!(inst.kind, InstKind::Nop)).then_some(idx))
        .collect()
}

fn phi_incoming(incomings: &[(BlockId, ValueId)], predecessor: BlockId) -> Option<ValueId> {
    let mut matches = incomings
        .iter()
        .filter_map(|(pred, value)| (*pred == predecessor).then_some(*value));
    let value = matches.next()?;
    matches.next().is_none().then_some(value)
}

fn value_available_at_block(
    func: &Function,
    dominators: &Dominators,
    value: ValueId,
    block: BlockId,
) -> bool {
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(owner, _) => owner == block || dominators.dominates(owner, block),
    }
}

fn simplify_trivial_phis(func: &mut Function) -> bool {
    let mut replacements = ValueReplacements::new();
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            let InstKind::Phi { incomings } = &inst.kind else {
                continue;
            };
            let Some(result) = inst.result else {
                continue;
            };
            let Some(first) = incomings.first().map(|(_, value)| *value) else {
                continue;
            };
            if incomings.iter().all(|(_, value)| *value == first) {
                replacements.insert(result, first);
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
        }
    }
    let changed = !replacements.is_empty();
    rewrite_function_uses(func, &replacements);
    changed
}

fn thread_boolean_phi_branches(func: &mut Function) -> bool {
    let use_counts = value_use_counts(func);
    let predecessor_sets = all_predecessors(func);
    let phi_predecessors = func
        .blocks
        .iter()
        .map(|block| {
            block
                .insts
                .iter()
                .filter_map(|inst| match &inst.kind {
                    InstKind::Phi { incomings } => Some(incomings),
                    _ => None,
                })
                .flat_map(|incomings| incomings.iter().map(|(pred, _)| *pred))
                .collect::<HashSet<_>>()
        })
        .collect::<Vec<_>>();
    let candidates = func
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(block_idx, block)| {
            let Terminator::Branch {
                cond,
                then_target,
                else_target,
            } = block.terminator.as_ref()?
            else {
                return None;
            };
            let mut active = block
                .insts
                .iter()
                .filter(|inst| !matches!(inst.kind, InstKind::Nop));
            let phi = active.next()?;
            if active.next().is_some() || phi.result != Some(*cond) {
                return None;
            }
            let InstKind::Phi { incomings } = &phi.kind else {
                return None;
            };
            (use_counts[cond.0] == 1
                && (block_returns(func, *then_target) || block_returns(func, *else_target))
                && !phi_predecessors[then_target.0].contains(&BlockId(block_idx))
                && !phi_predecessors[else_target.0].contains(&BlockId(block_idx)))
            .then_some((
                BlockId(block_idx),
                *cond,
                *then_target,
                *else_target,
                incomings.clone(),
            ))
        })
        .collect::<Vec<_>>();

    let mut changed = false;
    for (block, condition, then_target, else_target, incomings) in candidates {
        if !matches!(
            func.blocks[block.0].terminator.as_ref(),
            Some(Terminator::Branch {
                cond,
                then_target: current_then,
                else_target: current_else,
            }) if *cond == condition
                && *current_then == then_target
                && *current_else == else_target
        ) {
            continue;
        }

        let cfg_predecessors = predecessor_sets[block.0].clone();
        let incoming_predecessors = incomings
            .iter()
            .map(|(pred, _)| *pred)
            .collect::<HashSet<_>>();
        if cfg_predecessors != incoming_predecessors
            || incomings.iter().any(|(pred, incoming)| {
                if *pred == block {
                    return true;
                }
                match func.blocks[pred.0].terminator.as_ref() {
                    Some(Terminator::Jump(target)) if *target == block => false,
                    Some(Terminator::Branch {
                        then_target,
                        else_target,
                        ..
                    }) if (*then_target == block || *else_target == block) => {
                        const_bool(func, *incoming).is_none()
                    }
                    _ => true,
                }
            })
        {
            continue;
        }

        for (pred, incoming) in incomings {
            let selected =
                const_bool(func, incoming)
                    .map(|condition| if condition { then_target } else { else_target });
            let terminator = func.blocks[pred.0]
                .terminator
                .as_mut()
                .expect("validated predecessor must have a terminator");
            match terminator {
                Terminator::Jump(target) if *target == block => {
                    *terminator = if let Some(target) = selected {
                        Terminator::Jump(target)
                    } else {
                        Terminator::Branch {
                            cond: incoming,
                            then_target,
                            else_target,
                        }
                    };
                }
                Terminator::Branch {
                    then_target: pred_then,
                    else_target: pred_else,
                    ..
                } => {
                    let target = selected.expect("dynamic branch edge was rejected");
                    if *pred_then == block {
                        *pred_then = target;
                    }
                    if *pred_else == block {
                        *pred_else = target;
                    }
                }
                _ => unreachable!("predecessor changed after validation"),
            }
        }
        for inst in &mut func.blocks[block.0].insts {
            if inst.result == Some(condition) {
                inst.result = None;
                inst.kind = InstKind::Nop;
                break;
            }
        }
        func.blocks[block.0].terminator = Some(Terminator::Jump(else_target));
        changed = true;
    }
    changed
}

#[derive(Clone, Copy, Default)]
struct KnownBits {
    zero: u32,
    one: u32,
}

#[derive(Clone, Copy)]
struct KnownBitFact {
    region: BlockId,
    value: ValueId,
    bits: KnownBits,
}

/// Threads a predecessor edge through short chains of branches whose outcomes
/// become known after substituting that edge's phi inputs. This is the classic
/// SSA jump-threading transform used by GCC and LLVM: path-sensitive known bits
/// prove the branch, while bounded tail duplication preserves the other paths.
/// It does not recognize source functions or larger computation shapes.
fn thread_known_integer_edges(
    func: &mut Function,
    source_limit: usize,
    attempted_sources: &mut HashSet<BlockId>,
    attempt_budget: &mut usize,
) -> bool {
    const MAX_BLOCKS: usize = 1024;

    if func.blocks.len() > MAX_BLOCKS || *attempt_budget == 0 {
        return false;
    }
    let cfg = ControlFlowGraph::new(func);
    let dominators = Dominators::new(func, &cfg);
    let facts = collect_known_bit_facts(func);
    let candidates = func
        .blocks
        .iter()
        .enumerate()
        .take(source_limit)
        .filter_map(|(pred, block)| match block.terminator.as_ref() {
            Some(Terminator::Jump(target))
                if target.0 != pred && !attempted_sources.contains(&BlockId(pred)) =>
            {
                Some((BlockId(pred), *target))
            }
            _ => None,
        })
        .take(*attempt_budget)
        .collect::<Vec<_>>();

    for (pred, target) in candidates {
        attempted_sources.insert(pred);
        *attempt_budget -= 1;
        if try_thread_known_integer_edge(func, pred, target, &dominators, &facts) {
            return true;
        }
        if *attempt_budget == 0 {
            break;
        }
    }
    false
}

fn try_thread_known_integer_edge(
    func: &mut Function,
    pred: BlockId,
    header: BlockId,
    dominators: &Dominators,
    facts: &[KnownBitFact],
) -> bool {
    const MAX_THREADED_BLOCKS: usize = 6;
    const MAX_DUPLICATED_INSTS: usize = 12;
    const MAX_PROVEN_BRANCHES: usize = 4;

    if !matches!(func.blocks[pred.0].terminator, Some(Terminator::Jump(target)) if target == header)
        || !block_has_leading_phi(func, header)
    {
        return false;
    }

    let before = func.clone();
    let thread = func.add_block("jump.thread");
    let mut replacements = HashMap::<ValueId, ValueId>::new();
    let mut visited = HashSet::new();
    let mut current = header;
    let mut incoming_pred = pred;
    let mut duplicated = 0usize;
    let mut proven_branches = 0usize;
    let mut final_terminator = None;

    while visited.len() < MAX_THREADED_BLOCKS && visited.insert(current) {
        if !map_threaded_phis(func, current, incoming_pred, &mut replacements) {
            *func = before;
            return false;
        }
        for inst in func.blocks[current.0].insts.clone() {
            if matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }) {
                continue;
            }
            let (Some(old_result), Some(kind)) = (
                inst.result,
                remap_threadable_inst(&inst.kind, &replacements),
            ) else {
                *func = before;
                return false;
            };
            duplicated += 1;
            if duplicated > MAX_DUPLICATED_INSTS {
                *func = before;
                return false;
            }
            let result = func
                .append_inst(thread, kind, Some(func.value(old_result).ty.clone()))
                .expect("a duplicated pure instruction must produce a value");
            replacements.insert(old_result, result);
        }

        let Some(terminator) = func.blocks[current.0].terminator.clone() else {
            *func = before;
            return false;
        };
        match terminator {
            Terminator::Return(value) => {
                final_terminator = Some(Terminator::Return(
                    value.map(|value| resolve_thread_value(value, &replacements)),
                ));
                break;
            }
            Terminator::Jump(target) => {
                if visited.contains(&target) {
                    if !add_threaded_phi_edge(func, thread, current, target, &replacements) {
                        *func = before;
                        return false;
                    }
                    final_terminator = Some(Terminator::Jump(target));
                    break;
                }
                incoming_pred = current;
                current = target;
            }
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                let cond = resolve_thread_value(cond, &replacements);
                let mut proof_work = 256usize;
                let Some(taken_then) =
                    known_bool_at(func, cond, pred, dominators, facts, 0, &mut proof_work)
                else {
                    if proven_branches == 0
                        || !add_threaded_phi_edge(func, thread, current, then_target, &replacements)
                        || !add_threaded_phi_edge(func, thread, current, else_target, &replacements)
                    {
                        *func = before;
                        return false;
                    }
                    final_terminator = Some(Terminator::Branch {
                        cond,
                        then_target,
                        else_target,
                    });
                    break;
                };
                proven_branches += 1;
                if proven_branches > MAX_PROVEN_BRANCHES {
                    *func = before;
                    return false;
                }
                let target = if taken_then { then_target } else { else_target };
                if visited.contains(&target) {
                    if !add_threaded_phi_edge(func, thread, current, target, &replacements) {
                        *func = before;
                        return false;
                    }
                    final_terminator = Some(Terminator::Jump(target));
                    break;
                }
                incoming_pred = current;
                current = target;
            }
        }
    }

    let Some(final_terminator) = final_terminator else {
        *func = before;
        return false;
    };
    if proven_branches == 0 {
        *func = before;
        return false;
    }
    remove_phi_incomings(func, pred, header);
    func.blocks[pred.0].terminator = Some(Terminator::Jump(thread));
    func.blocks[thread.0].terminator = Some(final_terminator);
    if func.verify().is_err() {
        *func = before;
        false
    } else {
        true
    }
}

fn block_has_leading_phi(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0]
        .insts
        .iter()
        .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
}

fn map_threaded_phis(
    func: &Function,
    block: BlockId,
    incoming_pred: BlockId,
    replacements: &mut HashMap<ValueId, ValueId>,
) -> bool {
    let mut saw_non_phi = false;
    for inst in &func.blocks[block.0].insts {
        match (&inst.result, &inst.kind) {
            (_, InstKind::Nop) => {}
            (Some(result), InstKind::Phi { incomings }) if !saw_non_phi => {
                let Some(incoming) = phi_incoming(incomings, incoming_pred) else {
                    return false;
                };
                replacements.insert(*result, resolve_thread_value(incoming, replacements));
            }
            (_, InstKind::Phi { .. }) => return false,
            _ => saw_non_phi = true,
        }
    }
    true
}

fn add_threaded_phi_edge(
    func: &mut Function,
    thread: BlockId,
    old_pred: BlockId,
    target: BlockId,
    replacements: &HashMap<ValueId, ValueId>,
) -> bool {
    let mut additions = Vec::new();
    for (inst_idx, inst) in func.blocks[target.0].insts.iter().enumerate() {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { incomings } => {
                let Some(incoming) = phi_incoming(incomings, old_pred) else {
                    return false;
                };
                additions.push((inst_idx, resolve_thread_value(incoming, replacements)));
            }
            _ => break,
        }
    }
    for (inst_idx, incoming) in additions {
        let InstKind::Phi { incomings } = &mut func.blocks[target.0].insts[inst_idx].kind else {
            unreachable!("validated threaded successor phi changed")
        };
        if incomings.iter().any(|(pred, _)| *pred == thread) {
            return false;
        }
        incomings.push((thread, incoming));
    }
    true
}

fn resolve_thread_value(mut value: ValueId, replacements: &HashMap<ValueId, ValueId>) -> ValueId {
    for _ in 0..32 {
        let Some(next) = replacements.get(&value).copied() else {
            break;
        };
        if next == value {
            break;
        }
        value = next;
    }
    value
}

fn remap_threadable_inst(
    kind: &InstKind,
    replacements: &HashMap<ValueId, ValueId>,
) -> Option<InstKind> {
    let remap = |value| resolve_thread_value(value, replacements);
    match kind {
        InstKind::Unary { op, value }
            if matches!(op, crate::ir::UnaryOp::Ineg | crate::ir::UnaryOp::Not) =>
        {
            Some(InstKind::Unary {
                op: *op,
                value: remap(*value),
            })
        }
        InstKind::Binary { op, lhs, rhs }
            if !matches!(
                op,
                BinaryOp::Idiv
                    | BinaryOp::Imod
                    | BinaryOp::Fadd
                    | BinaryOp::Fsub
                    | BinaryOp::Fmul
                    | BinaryOp::Fdiv
            ) =>
        {
            Some(InstKind::Binary {
                op: *op,
                lhs: remap(*lhs),
                rhs: remap(*rhs),
            })
        }
        InstKind::Icmp { op, lhs, rhs } => Some(InstKind::Icmp {
            op: *op,
            lhs: remap(*lhs),
            rhs: remap(*rhs),
        }),
        InstKind::Cast { op, value } if matches!(op, CastOp::BoolToI32 | CastOp::I32ToBool) => {
            Some(InstKind::Cast {
                op: *op,
                value: remap(*value),
            })
        }
        _ => None,
    }
}

fn collect_known_bit_facts(func: &Function) -> Vec<KnownBitFact> {
    const MAX_FACTS: usize = 4096;
    let mut facts = Vec::new();
    let predecessors = all_predecessors(func);
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let Some(Terminator::Branch {
            cond,
            then_target,
            else_target,
        }) = block.terminator.as_ref()
        else {
            continue;
        };
        let Some((value, mask, zero_on_true)) = masked_zero_condition(func, *cond) else {
            continue;
        };
        let zero_region = if zero_on_true {
            *then_target
        } else {
            *else_target
        };
        let nonzero_region = if zero_on_true {
            *else_target
        } else {
            *then_target
        };
        let source = BlockId(block_idx);
        if zero_region != source && predecessors[zero_region.0] == HashSet::from([source]) {
            facts.push(KnownBitFact {
                region: zero_region,
                value,
                bits: KnownBits { zero: mask, one: 0 },
            });
        }
        if nonzero_region != source
            && mask.is_power_of_two()
            && predecessors[nonzero_region.0] == HashSet::from([source])
        {
            facts.push(KnownBitFact {
                region: nonzero_region,
                value,
                bits: KnownBits { zero: 0, one: mask },
            });
        }
        if facts.len() > MAX_FACTS {
            return Vec::new();
        }
    }
    facts
}

fn masked_zero_condition(func: &Function, condition: ValueId) -> Option<(ValueId, u32, bool)> {
    let InstKind::Icmp { op, lhs, rhs } = defining_inst_kind(func, condition)? else {
        return None;
    };
    let masked = if const_i32_value(func, *rhs) == Some(0) {
        *lhs
    } else if const_i32_value(func, *lhs) == Some(0) {
        *rhs
    } else {
        return None;
    };
    let zero_on_true = match op {
        crate::ir::CmpOp::Eq => true,
        crate::ir::CmpOp::Ne => false,
        _ => return None,
    };
    let InstKind::Binary {
        op: BinaryOp::Iand,
        lhs,
        rhs,
    } = defining_inst_kind(func, masked)?
    else {
        return None;
    };
    if let Some(mask) = const_i32_value(func, *rhs) {
        Some((*lhs, mask as u32, zero_on_true))
    } else {
        const_i32_value(func, *lhs).map(|mask| (*rhs, mask as u32, zero_on_true))
    }
}

fn known_bool_at(
    func: &Function,
    value: ValueId,
    at: BlockId,
    dominators: &Dominators,
    facts: &[KnownBitFact],
    depth: usize,
    work: &mut usize,
) -> Option<bool> {
    if depth > 32 || *work == 0 {
        return None;
    }
    *work -= 1;
    if let Some(value) = const_bool(func, value) {
        return Some(value);
    }
    match defining_inst_kind(func, value)? {
        InstKind::Unary {
            op: crate::ir::UnaryOp::Not,
            value,
        } => {
            known_bool_at(func, *value, at, dominators, facts, depth + 1, work).map(|value| !value)
        }
        InstKind::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
        } => match (
            known_bool_at(func, *lhs, at, dominators, facts, depth + 1, work),
            known_bool_at(func, *rhs, at, dominators, facts, depth + 1, work),
        ) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        },
        InstKind::Binary {
            op: BinaryOp::Or,
            lhs,
            rhs,
        } => match (
            known_bool_at(func, *lhs, at, dominators, facts, depth + 1, work),
            known_bool_at(func, *rhs, at, dominators, facts, depth + 1, work),
        ) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        },
        InstKind::Icmp { op, lhs, rhs } => {
            let lhs = known_bits_at(func, *lhs, at, dominators, facts, depth + 1, work);
            let rhs = known_bits_at(func, *rhs, at, dominators, facts, depth + 1, work);
            evaluate_known_compare(*op, lhs, rhs)
        }
        InstKind::Cast {
            op: CastOp::I32ToBool,
            value,
        } => {
            let bits = known_bits_at(func, *value, at, dominators, facts, depth + 1, work);
            if bits.one != 0 {
                Some(true)
            } else if bits.zero == u32::MAX {
                Some(false)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn known_bits_at(
    func: &Function,
    value: ValueId,
    at: BlockId,
    dominators: &Dominators,
    facts: &[KnownBitFact],
    depth: usize,
    work: &mut usize,
) -> KnownBits {
    if depth > 32 || *work == 0 {
        return KnownBits::default();
    }
    *work -= 1;
    let mut known = match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => exact_known_bits(*value as u32),
        ValueKind::Const(Const::Bool(value)) => exact_known_bits(u32::from(*value)),
        ValueKind::Const(Const::Zero(Type::I32 | Type::I1)) => exact_known_bits(0),
        ValueKind::Inst(_, _) => match defining_inst_kind(func, value) {
            Some(InstKind::Unary {
                op: crate::ir::UnaryOp::Ineg,
                value,
            }) => parity_unary(known_bits_at(
                func,
                *value,
                at,
                dominators,
                facts,
                depth + 1,
                work,
            )),
            Some(InstKind::Binary { op, lhs, rhs }) => {
                let lhs_bits = known_bits_at(func, *lhs, at, dominators, facts, depth + 1, work);
                let rhs_bits = known_bits_at(func, *rhs, at, dominators, facts, depth + 1, work);
                known_bits_binary(func, *op, *lhs, *rhs, lhs_bits, rhs_bits)
            }
            _ => KnownBits::default(),
        },
        _ => KnownBits::default(),
    };
    for fact in facts {
        if fact.value == value && dominators.dominates(fact.region, at) {
            known.zero |= fact.bits.zero;
            known.one |= fact.bits.one;
        }
    }
    let conflict = known.zero & known.one;
    known.zero &= !conflict;
    known.one &= !conflict;
    known
}

fn defining_inst_kind(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

fn exact_known_bits(value: u32) -> KnownBits {
    KnownBits {
        zero: !value,
        one: value,
    }
}

fn parity_unary(value: KnownBits) -> KnownBits {
    KnownBits {
        zero: value.zero & 1,
        one: value.one & 1,
    }
}

fn known_bits_binary(
    func: &Function,
    op: BinaryOp,
    _lhs_value: ValueId,
    rhs_value: ValueId,
    lhs: KnownBits,
    rhs: KnownBits,
) -> KnownBits {
    match op {
        BinaryOp::Iand => KnownBits {
            zero: lhs.zero | rhs.zero,
            one: lhs.one & rhs.one,
        },
        BinaryOp::Ior => KnownBits {
            zero: lhs.zero & rhs.zero,
            one: lhs.one | rhs.one,
        },
        BinaryOp::Ixor => KnownBits {
            zero: (lhs.zero & rhs.zero) | (lhs.one & rhs.one),
            one: (lhs.zero & rhs.one) | (lhs.one & rhs.zero),
        },
        BinaryOp::Iadd | BinaryOp::Isub => {
            if let (Some(lhs), Some(rhs)) = (exact_known_value(lhs), exact_known_value(rhs)) {
                return exact_known_bits(if op == BinaryOp::Iadd {
                    lhs.wrapping_add(rhs)
                } else {
                    lhs.wrapping_sub(rhs)
                });
            }
            known_low_bit_xor(lhs, rhs)
        }
        BinaryOp::Imul => {
            if let (Some(lhs), Some(rhs)) = (exact_known_value(lhs), exact_known_value(rhs)) {
                return exact_known_bits(lhs.wrapping_mul(rhs));
            }
            let lhs_zero = lhs.zero & 1 != 0;
            let rhs_zero = rhs.zero & 1 != 0;
            let lhs_one = lhs.one & 1 != 0;
            let rhs_one = rhs.one & 1 != 0;
            KnownBits {
                zero: u32::from(lhs_zero || rhs_zero),
                one: u32::from(lhs_one && rhs_one),
            }
        }
        BinaryOp::Ishl => {
            let Some(shift) = const_i32_value(func, rhs_value).map(|value| (value as u32) & 31)
            else {
                return KnownBits::default();
            };
            let introduced_zero = if shift == 0 { 0 } else { (1u32 << shift) - 1 };
            KnownBits {
                zero: (lhs.zero << shift) | introduced_zero,
                one: lhs.one << shift,
            }
        }
        BinaryOp::Iashr => {
            let Some(shift) = const_i32_value(func, rhs_value).map(|value| (value as u32) & 31)
            else {
                return KnownBits::default();
            };
            if shift == 0 {
                return lhs;
            }
            let high_mask = u32::MAX << (32 - shift);
            KnownBits {
                zero: (lhs.zero >> shift)
                    | if lhs.zero & (1 << 31) != 0 {
                        high_mask
                    } else {
                        0
                    },
                one: (lhs.one >> shift)
                    | if lhs.one & (1 << 31) != 0 {
                        high_mask
                    } else {
                        0
                    },
            }
        }
        BinaryOp::And => KnownBits {
            zero: (lhs.zero | rhs.zero) & 1,
            one: (lhs.one & rhs.one) & 1,
        },
        BinaryOp::Or => KnownBits {
            zero: (lhs.zero & rhs.zero) & 1,
            one: (lhs.one | rhs.one) & 1,
        },
        BinaryOp::Idiv
        | BinaryOp::Imod
        | BinaryOp::Fadd
        | BinaryOp::Fsub
        | BinaryOp::Fmul
        | BinaryOp::Fdiv => KnownBits::default(),
    }
}

fn known_low_bit_xor(lhs: KnownBits, rhs: KnownBits) -> KnownBits {
    KnownBits {
        zero: ((lhs.zero & rhs.zero) | (lhs.one & rhs.one)) & 1,
        one: ((lhs.zero & rhs.one) | (lhs.one & rhs.zero)) & 1,
    }
}

fn exact_known_value(bits: KnownBits) -> Option<u32> {
    (bits.zero | bits.one == u32::MAX && bits.zero & bits.one == 0).then_some(bits.one)
}

fn evaluate_known_compare(op: crate::ir::CmpOp, lhs: KnownBits, rhs: KnownBits) -> Option<bool> {
    let contradiction = (lhs.one & rhs.zero) | (lhs.zero & rhs.one);
    match op {
        crate::ir::CmpOp::Eq if contradiction != 0 => Some(false),
        crate::ir::CmpOp::Ne if contradiction != 0 => Some(true),
        crate::ir::CmpOp::Eq | crate::ir::CmpOp::Ne => {
            let equal = exact_known_value(lhs)? == exact_known_value(rhs)?;
            Some(if op == crate::ir::CmpOp::Eq {
                equal
            } else {
                !equal
            })
        }
        crate::ir::CmpOp::Lt
        | crate::ir::CmpOp::Le
        | crate::ir::CmpOp::Gt
        | crate::ir::CmpOp::Ge => {
            let lhs = exact_known_value(lhs)? as i32;
            let rhs = exact_known_value(rhs)? as i32;
            Some(match op {
                crate::ir::CmpOp::Lt => lhs < rhs,
                crate::ir::CmpOp::Le => lhs <= rhs,
                crate::ir::CmpOp::Gt => lhs > rhs,
                crate::ir::CmpOp::Ge => lhs >= rhs,
                crate::ir::CmpOp::Eq | crate::ir::CmpOp::Ne => unreachable!(),
            })
        }
    }
}

fn all_predecessors(func: &Function) -> Vec<HashSet<BlockId>> {
    let mut predecessors = vec![HashSet::new(); func.blocks.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let pred = BlockId(block_idx);
        match block.terminator.as_ref() {
            Some(Terminator::Jump(target)) => {
                predecessors[target.0].insert(pred);
            }
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => {
                predecessors[then_target.0].insert(pred);
                predecessors[else_target.0].insert(pred);
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }
    predecessors
}

fn block_returns(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0]
        .insts
        .iter()
        .all(|inst| matches!(inst.kind, InstKind::Nop))
        && matches!(func.blocks[block.0].terminator, Some(Terminator::Return(_)))
}

fn value_use_counts(func: &Function) -> Vec<usize> {
    let mut counts = vec![0usize; func.values.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            let operands = match &inst.kind {
                InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
                InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
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
            };
            for operand in operands {
                counts[operand.0] += 1;
            }
        }
        for operand in match block.terminator.as_ref() {
            Some(Terminator::Return(Some(value))) => vec![*value],
            Some(Terminator::Branch { cond, .. }) => vec![*cond],
            Some(Terminator::Return(None) | Terminator::Jump(_)) | None => Vec::new(),
        } {
            counts[operand.0] += 1;
        }
    }
    counts
}

fn remove_phi_incomings(func: &mut Function, pred: BlockId, target: BlockId) {
    // CFG 边被删后，目标块 phi 中来自这个前驱的值也必须同步删除。
    for inst in &mut func.blocks[target.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            if !matches!(inst.kind, InstKind::Nop) {
                break;
            }
            continue;
        };
        incomings.retain(|(incoming_pred, _)| *incoming_pred != pred);
    }
}

fn const_bool(func: &Function, value: crate::ir::ValueId) -> Option<bool> {
    // 分支条件允许 bool/int/float 常量：非零视为 true。
    match &func.value(value).kind {
        ValueKind::Const(Const::Bool(value)) => Some(*value),
        ValueKind::Const(Const::Int(value)) => Some(*value != 0),
        ValueKind::Const(Const::Float(bits)) => Some(f32::from_bits(*bits) != 0.0),
        _ => None,
    }
}
