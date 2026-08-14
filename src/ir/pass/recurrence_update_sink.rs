use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::LoopInfo;
use super::util::{const_i32, defining_inst};
use super::ModulePass;
use crate::ir::{BlockId, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_INSTRUCTIONS: usize = 32_768;
const MAX_CANDIDATES: usize = 128;

/// Sinks a pointer recurrence update from the loop header to its backedges.
///
/// LSR commonly leaves one `next = gep(phi, constant)` value serving both as
/// an adjacent memory address and as the next phi input. Cloning that pure
/// update onto each backedge makes the header copy memory-only, so ordinary
/// target address-mode folding can use `offset(phi)` and the backedge can
/// update the phi register in place. This is standard code sinking plus
/// addressing-mode selection and depends only on SSA/CFG facts.
pub(super) struct RecurrenceUpdateSinkPass;

impl RecurrenceUpdateSinkPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RecurrenceUpdateSinkPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            sink_function(func);
        }
    }
}

#[derive(Clone)]
struct Candidate {
    phi: ValueId,
    update: ValueId,
    update_kind: InstKind,
    update_ty: Type,
    backedges: Vec<BlockId>,
}

fn sink_function(func: &mut Function) {
    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    if func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || instruction_count.is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let reordered = schedule_header_pointer_updates(func, &loop_info);
    let mut candidates = Vec::new();
    let mut claimed_updates = HashSet::new();

    for natural_loop in loop_info.loops() {
        for inst in &func.blocks[natural_loop.header.0].insts {
            let (Some(phi), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
                continue;
            };
            if !matches!(func.value(phi).ty, Type::Ptr(_)) {
                continue;
            }
            let mut loop_incomings = incomings
                .iter()
                .filter(|(pred, _)| natural_loop.blocks.contains(pred));
            let Some((_, update)) = loop_incomings.next().copied() else {
                continue;
            };
            if loop_incomings.any(|(_, incoming)| *incoming != update)
                || !claimed_updates.insert(update)
            {
                continue;
            }
            let Some(InstKind::Gep { base, indices }) = defining_inst(func, update) else {
                continue;
            };
            let ValueKind::Inst(update_block, _) = func.value(update).kind else {
                continue;
            };
            if update_block != natural_loop.header
                || *base != phi
                || !constant_offset_is_addressable(func, update, indices)
                || !update_uses_are_sinkable(func, phi, update)
            {
                continue;
            }
            let mut backedges = incomings
                .iter()
                .filter_map(|(pred, incoming)| {
                    (natural_loop.blocks.contains(pred) && *incoming == update).then_some(*pred)
                })
                .collect::<Vec<_>>();
            backedges.sort_by_key(|block| block.0);
            backedges.dedup();
            if backedges.is_empty()
                || backedges.iter().any(|pred| {
                    indices
                        .iter()
                        .any(|index| !value_available_on_edge(func, &dom, phi, *index, *pred))
                })
            {
                continue;
            }
            candidates.push(Candidate {
                phi,
                update,
                update_kind: InstKind::Gep {
                    base: *base,
                    indices: indices.clone(),
                },
                update_ty: func.value(update).ty.clone(),
                backedges,
            });
            if candidates.len() == MAX_CANDIDATES {
                break;
            }
        }
        if candidates.len() == MAX_CANDIDATES {
            break;
        }
    }

    if candidates.is_empty() {
        if reordered {
            if let Err(errors) = func.verify() {
                panic!(
                    "recurrence update scheduling produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
        return;
    }
    let mut dedicated_latches = HashMap::<BlockId, BlockId>::new();
    for candidate in candidates {
        let header = candidate_header(func, candidate.phi);
        let insertion_block =
            recurrence_update_block(func, &candidate.backedges, header, &mut dedicated_latches);
        let clone = func
            .append_inst(
                insertion_block,
                candidate.update_kind.clone(),
                Some(candidate.update_ty.clone()),
            )
            .expect("a cloned recurrence update must produce a pointer");
        let ValueKind::Inst(phi_block, phi_idx) = func.value(candidate.phi).kind else {
            unreachable!("a recurrence phi must be instruction-backed");
        };
        let InstKind::Phi { incomings } = &mut func.blocks[phi_block.0].insts[phi_idx].kind else {
            unreachable!("candidate must remain a phi");
        };
        for (incoming_pred, incoming) in incomings {
            if *incoming_pred == insertion_block && *incoming == candidate.update {
                *incoming = clone;
            }
        }
    }

    if let Err(errors) = func.verify() {
        panic!(
            "recurrence update sinking produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

/// Moves a pure pointer recurrence update below the final same-block use of
/// the old pointer.  This shortens the two live ranges enough for ordinary
/// phi-copy coalescing to assign them the same physical register.  The move is
/// legal only when no use of the update would be crossed.
fn schedule_header_pointer_updates(func: &mut Function, loop_info: &LoopInfo) -> bool {
    let mut candidates = Vec::<(BlockId, ValueId, ValueId)>::new();
    let mut seen = HashSet::new();
    for natural_loop in loop_info.loops() {
        for inst in &func.blocks[natural_loop.header.0].insts {
            let (Some(phi), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
                continue;
            };
            if !matches!(func.value(phi).ty, Type::Ptr(_)) {
                continue;
            }
            for (_, update) in incomings
                .iter()
                .filter(|(pred, _)| natural_loop.blocks.contains(pred))
            {
                if !seen.insert(*update) {
                    continue;
                }
                let Some(update_kind) = defining_inst(func, *update) else {
                    continue;
                };
                if recurrence_update_base(update_kind) == Some(phi)
                    && matches!(func.value(*update).kind, ValueKind::Inst(owner, _) if owner == natural_loop.header)
                {
                    candidates.push((natural_loop.header, phi, *update));
                }
            }
        }
    }

    let mut changed = false;
    for (header, phi, update) in candidates {
        changed |= schedule_one_header_update(func, header, phi, update);
    }
    changed
}

fn schedule_one_header_update(
    func: &mut Function,
    header: BlockId,
    phi: ValueId,
    update: ValueId,
) -> bool {
    let ValueKind::Inst(owner, update_idx) = func.value(update).kind else {
        return false;
    };
    if owner != header
        || !func.blocks[header.0]
            .insts
            .get(update_idx)
            .is_some_and(|inst| {
                inst.result == Some(update) && recurrence_update_base(&inst.kind) == Some(phi)
            })
    {
        return false;
    }

    let mut last_phi_use = None;
    let mut first_update_use = None;
    for (inst_idx, inst) in func.blocks[header.0].insts.iter().enumerate() {
        if inst_idx == update_idx {
            continue;
        }
        let operands = inst_operands(&inst.kind);
        if operands.contains(&phi) {
            last_phi_use = Some(inst_idx);
        }
        // Phi operands execute on predecessor edges, not at the phi's textual
        // position at the beginning of this block.
        if !matches!(inst.kind, InstKind::Phi { .. }) && operands.contains(&update) {
            first_update_use.get_or_insert(inst_idx);
        }
    }
    let Some(last_phi_use) = last_phi_use.filter(|last_use| *last_use > update_idx) else {
        return false;
    };
    if first_update_use.is_some_and(|first_use| first_use <= last_phi_use) {
        return false;
    }

    // Removing an earlier instruction shifts the old last-use instruction one
    // slot left, so insertion at its old index places the update immediately
    // after that use.
    let update_inst = func.blocks[header.0].insts.remove(update_idx);
    func.blocks[header.0]
        .insts
        .insert(last_phi_use, update_inst);
    let result_locations = func.blocks[header.0]
        .insts
        .iter()
        .enumerate()
        .filter_map(|(inst_idx, inst)| inst.result.map(|result| (result, inst_idx)))
        .collect::<Vec<_>>();
    for (result, inst_idx) in result_locations {
        func.values[result.0].kind = ValueKind::Inst(header, inst_idx);
    }
    true
}

fn recurrence_update_base(kind: &InstKind) -> Option<ValueId> {
    match kind {
        InstKind::Gep { base, .. } => Some(*base),
        InstKind::Binary {
            op: crate::ir::BinaryOp::Iadd,
            lhs,
            ..
        }
        | InstKind::Binary {
            op: crate::ir::BinaryOp::Isub,
            lhs,
            ..
        } => Some(*lhs),
        _ => None,
    }
}

fn candidate_header(func: &Function, phi: ValueId) -> BlockId {
    match func.value(phi).kind {
        ValueKind::Inst(block, _) => block,
        _ => unreachable!("a recurrence phi must be instruction-backed"),
    }
}

fn recurrence_update_block(
    func: &mut Function,
    backedges: &[BlockId],
    header: BlockId,
    dedicated_latches: &mut HashMap<BlockId, BlockId>,
) -> BlockId {
    if let Some(block) = dedicated_latches.get(&header).copied() {
        return block;
    }
    if let [pred] = backedges {
        let successor_count = terminator_targets(func.blocks[pred.0].terminator.as_ref()).len();
        if successor_count == 1 {
            dedicated_latches.insert(header, *pred);
            return *pred;
        }
    }

    let latch = func.add_block("recurrence.update.latch");
    func.set_terminator(latch, Terminator::Jump(header));
    let backedge_set = backedges.iter().copied().collect::<HashSet<_>>();
    let header_phis = func.blocks[header.0]
        .insts
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| {
            matches!(inst.kind, InstKind::Phi { .. }).then_some((idx, inst.result))
        })
        .collect::<Vec<_>>();
    for (inst_idx, result) in header_phis {
        let InstKind::Phi { incomings } = &func.blocks[header.0].insts[inst_idx].kind else {
            unreachable!();
        };
        let loop_incomings = incomings
            .iter()
            .filter(|(pred, _)| backedge_set.contains(pred))
            .copied()
            .collect::<Vec<_>>();
        if loop_incomings.is_empty() {
            continue;
        }
        let first = loop_incomings[0].1;
        let merged = if loop_incomings
            .iter()
            .all(|(_, incoming)| *incoming == first)
        {
            first
        } else {
            let ty = func
                .value(result.expect("a phi must produce a value"))
                .ty
                .clone();
            func.append_inst(
                latch,
                InstKind::Phi {
                    incomings: loop_incomings.clone(),
                },
                Some(ty),
            )
            .expect("a dedicated-latch phi must produce a value")
        };
        let InstKind::Phi { incomings } = &mut func.blocks[header.0].insts[inst_idx].kind else {
            unreachable!();
        };
        incomings.retain(|(pred, _)| !backedge_set.contains(pred));
        incomings.push((latch, merged));
    }
    for pred in backedges {
        replace_successor(
            func.blocks[pred.0]
                .terminator
                .as_mut()
                .expect("a loop backedge must have a terminator"),
            header,
            latch,
        );
    }
    dedicated_latches.insert(header, latch);
    latch
}

fn replace_successor(terminator: &mut Terminator, old: BlockId, new: BlockId) {
    match terminator {
        Terminator::Jump(target) => {
            if *target == old {
                *target = new;
            }
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
        Terminator::Return(_) => {}
    }
}

fn constant_offset_is_addressable(func: &Function, result: ValueId, indices: &[ValueId]) -> bool {
    let [index] = indices else {
        return false;
    };
    let Some(index) = const_i32(func, *index).map(i64::from) else {
        return false;
    };
    let Type::Ptr(pointee) = &func.value(result).ty else {
        return false;
    };
    checked_type_size(pointee)
        .and_then(|size| index.checked_mul(size))
        .is_some_and(|offset| (-2048..=2047).contains(&offset))
}

fn checked_type_size(ty: &Type) -> Option<i64> {
    match ty {
        Type::Void => Some(0),
        Type::I1 | Type::I32 | Type::F32 => Some(4),
        Type::Ptr(_) => Some(8),
        Type::Array { elem, len } => {
            checked_type_size(elem)?.checked_mul(i64::try_from(*len).ok()?)
        }
    }
}

fn update_uses_are_sinkable(func: &Function, phi: ValueId, update: ValueId) -> bool {
    let mut has_memory_use = false;
    for block in &func.blocks {
        for inst in &block.insts {
            let valid = match &inst.kind {
                InstKind::Phi { incomings } => {
                    !incomings.iter().any(|(_, incoming)| *incoming == update)
                        || inst.result == Some(phi)
                }
                InstKind::Load { ptr } if *ptr == update => {
                    has_memory_use = true;
                    true
                }
                InstKind::Store { ptr, value } if *ptr == update && *value != update => {
                    has_memory_use = true;
                    true
                }
                kind => !inst_operands(kind).contains(&update),
            };
            if !valid {
                return false;
            }
        }
        if terminator_operands(block.terminator.as_ref()).contains(&update) {
            return false;
        }
    }
    has_memory_use
}

fn value_available_on_edge(
    func: &Function,
    dom: &Dominators,
    phi: ValueId,
    value: ValueId,
    pred: BlockId,
) -> bool {
    if value == phi {
        return true;
    }
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(owner, _)) => dom.dominates(*owner, pred),
        None => false,
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

fn terminator_operands(terminator: Option<&Terminator>) -> Vec<ValueId> {
    match terminator {
        Some(Terminator::Return(Some(value))) => vec![*value],
        Some(Terminator::Branch { cond, .. }) => vec![*cond],
        Some(Terminator::Return(None) | Terminator::Jump(_)) | None => Vec::new(),
    }
}

fn terminator_targets(terminator: Option<&Terminator>) -> Vec<BlockId> {
    match terminator {
        Some(Terminator::Jump(target)) => vec![*target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) if then_target == else_target => vec![*then_target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) => vec![*then_target, *else_target],
        Some(Terminator::Return(_)) | None => Vec::new(),
    }
}
