use super::ModulePass;
use crate::ir::{BlockId, Function, Inst, InstKind, Module, Terminator, Type, ValueId, ValueKind};
use std::collections::HashSet;

pub(super) struct TailRecursionPass;

impl TailRecursionPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for TailRecursionPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            eliminate_tail_recursion(func);
        }
    }
}

#[derive(Debug, Clone)]
struct TailCallSite {
    block: BlockId,
    call_idx: usize,
    args: Vec<ValueId>,
}

fn eliminate_tail_recursion(func: &mut Function) {
    if collect_tail_self_calls(func).is_empty() {
        return;
    }

    let Some(param_slots) = collect_param_slots(func) else {
        return;
    };
    let loop_entry = split_entry(func, &param_slots);
    let mut sites = collect_tail_self_calls(func);
    if sites.is_empty() {
        return;
    }

    sites.sort_by_key(|site| (site.block.0, site.call_idx));
    for site in sites.into_iter().rev() {
        rewrite_tail_call(func, &param_slots, loop_entry, site);
    }

    if let Err(errors) = func.verify() {
        panic!(
            "tail recursion pass produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn collect_param_slots(func: &Function) -> Option<Vec<ValueId>> {
    let entry = func.block(func.entry);
    let mut slots = Vec::with_capacity(func.params.len());
    let mut cursor = 0usize;

    for param in &func.params {
        cursor = skip_nops(&entry.insts, cursor);
        let slot = match entry.insts.get(cursor) {
            Some(Inst {
                result: Some(result),
                kind: InstKind::Alloca { .. },
            }) => *result,
            _ => return None,
        };
        cursor += 1;

        cursor = skip_nops(&entry.insts, cursor);
        match entry.insts.get(cursor) {
            Some(Inst {
                result: None,
                kind: InstKind::Store { ptr, value },
            }) if *ptr == slot && *value == *param => {}
            _ => return None,
        }
        cursor += 1;

        slots.push(slot);
    }

    Some(slots)
}

fn skip_nops(insts: &[Inst], mut cursor: usize) -> usize {
    while matches!(
        insts.get(cursor).map(|inst| &inst.kind),
        Some(InstKind::Nop)
    ) {
        cursor += 1;
    }
    cursor
}

fn split_entry(func: &mut Function, param_slots: &[ValueId]) -> BlockId {
    let entry = func.entry;
    let params = func.params.clone();
    let old_terminator = func.blocks[entry.0].terminator.take();
    let old_insts = std::mem::take(&mut func.blocks[entry.0].insts);
    let mut prelude_insts = Vec::new();
    let mut body_insts = Vec::new();
    let mut new_locations = Vec::with_capacity(old_insts.len());

    let loop_entry = func.add_block("tail.entry");
    for inst in old_insts {
        if is_entry_prelude_inst(&inst, param_slots, &params) {
            new_locations.push((entry, prelude_insts.len()));
            prelude_insts.push(inst);
        } else {
            new_locations.push((loop_entry, body_insts.len()));
            body_insts.push(inst);
        }
    }

    retarget_successor_phi_incomings(func, entry, loop_entry, old_terminator.as_ref());
    func.blocks[entry.0].insts = prelude_insts;
    func.blocks[entry.0].terminator = Some(Terminator::Jump(loop_entry));
    func.blocks[loop_entry.0].insts = body_insts;
    func.blocks[loop_entry.0].terminator = old_terminator;
    remap_moved_entry_values(func, entry, &new_locations);
    loop_entry
}

fn retarget_successor_phi_incomings(
    func: &mut Function,
    old_pred: BlockId,
    new_pred: BlockId,
    terminator: Option<&Terminator>,
) {
    let Some(terminator) = terminator else {
        return;
    };
    for succ in terminator_successors(terminator) {
        for inst in &mut func.blocks[succ.0].insts {
            match &mut inst.kind {
                InstKind::Nop => continue,
                InstKind::Phi { incomings } => {
                    for (pred, _) in incomings {
                        if *pred == old_pred {
                            *pred = new_pred;
                        }
                    }
                }
                _ => break,
            }
        }
    }
}

fn terminator_successors(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Return(_) => Vec::new(),
        Terminator::Jump(target) => vec![*target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } if then_target == else_target => vec![*then_target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => vec![*then_target, *else_target],
    }
}

fn is_entry_prelude_inst(inst: &Inst, param_slots: &[ValueId], params: &[ValueId]) -> bool {
    match &inst.kind {
        InstKind::Alloca { .. } => true,
        InstKind::Store { ptr, value } => param_slots
            .iter()
            .zip(params.iter())
            .any(|(slot, param)| ptr == slot && value == param),
        _ => false,
    }
}

fn remap_moved_entry_values(
    func: &mut Function,
    entry: BlockId,
    new_locations: &[(BlockId, usize)],
) {
    for value in &mut func.values {
        let ValueKind::Inst(block, inst_idx) = value.kind else {
            continue;
        };
        if block != entry {
            continue;
        }
        if let Some((new_block, new_inst_idx)) = new_locations.get(inst_idx).copied() {
            value.kind = ValueKind::Inst(new_block, new_inst_idx);
        }
    }
}

fn collect_tail_self_calls(func: &Function) -> Vec<TailCallSite> {
    (0..func.blocks.len())
        .filter_map(|block_idx| tail_self_call(func, BlockId(block_idx)))
        .filter(|site| !has_frame_local_pointer_arg(func, &site.args))
        .collect()
}

fn has_frame_local_pointer_arg(func: &Function, args: &[ValueId]) -> bool {
    args.iter().copied().any(|arg| {
        matches!(&func.value(arg).ty, Type::Ptr(_))
            && pointer_derived_from_alloca(func, arg, &mut HashSet::new())
    })
}

fn pointer_derived_from_alloca(
    func: &Function,
    value: ValueId,
    visited: &mut HashSet<ValueId>,
) -> bool {
    if !visited.insert(value) {
        return false;
    }

    let ValueKind::Inst(block, inst_idx) = &func.value(value).kind else {
        return false;
    };
    match &func.block(*block).insts[*inst_idx].kind {
        InstKind::Alloca { .. } => true,
        InstKind::Gep { base, .. } => pointer_derived_from_alloca(func, *base, visited),
        InstKind::Phi { incomings } => incomings
            .iter()
            .any(|(_, incoming)| pointer_derived_from_alloca(func, *incoming, visited)),
        _ => false,
    }
}

fn tail_self_call(func: &Function, block: BlockId) -> Option<TailCallSite> {
    match func.block(block).terminator.as_ref()? {
        Terminator::Return(Some(value)) => tail_self_call_returning_value(func, block, *value),
        Terminator::Return(None) if func.ret == Type::Void => {
            tail_self_call_returning_void(func, block)
        }
        _ => None,
    }
}

fn tail_self_call_returning_value(
    func: &Function,
    block: BlockId,
    value: ValueId,
) -> Option<TailCallSite> {
    let ValueKind::Inst(call_block, call_idx) = func.value(value).kind else {
        return None;
    };
    if call_block != block || !only_nops_after(func, block, call_idx) {
        return None;
    }
    if value_use_count(func, value) != 1 {
        return None;
    }

    let inst = func.block(block).insts.get(call_idx)?;
    if inst.result != Some(value) {
        return None;
    }
    match &inst.kind {
        InstKind::Call { name, args } if name == &func.name && args.len() == func.params.len() => {
            Some(TailCallSite {
                block,
                call_idx,
                args: args.clone(),
            })
        }
        _ => None,
    }
}

fn tail_self_call_returning_void(func: &Function, block: BlockId) -> Option<TailCallSite> {
    let call_idx = func
        .block(block)
        .insts
        .iter()
        .rposition(|inst| !matches!(inst.kind, InstKind::Nop))?;
    let inst = &func.block(block).insts[call_idx];
    match &inst.kind {
        InstKind::Call { name, args }
            if name == &func.name && args.len() == func.params.len() && inst.result.is_none() =>
        {
            Some(TailCallSite {
                block,
                call_idx,
                args: args.clone(),
            })
        }
        _ => None,
    }
}

fn only_nops_after(func: &Function, block: BlockId, inst_idx: usize) -> bool {
    func.block(block).insts[inst_idx + 1..]
        .iter()
        .all(|inst| matches!(inst.kind, InstKind::Nop))
}

fn value_use_count(func: &Function, value: ValueId) -> usize {
    let mut count = 0usize;
    for block in &func.blocks {
        for inst in &block.insts {
            count += inst_value_use_count(&inst.kind, value);
        }
        if let Some(terminator) = &block.terminator {
            count += terminator_value_use_count(terminator, value);
        }
    }
    count
}

fn inst_value_use_count(kind: &InstKind, needle: ValueId) -> usize {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => 0,
        InstKind::Phi { incomings } => incomings
            .iter()
            .filter(|(_, value)| *value == needle)
            .count(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => (*ptr == needle) as usize,
        InstKind::Store { ptr, value } => (*ptr == needle) as usize + (*value == needle) as usize,
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => (*value == needle) as usize,
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => (*lhs == needle) as usize + (*rhs == needle) as usize,
        InstKind::Gep { base, indices } => {
            (*base == needle) as usize + indices.iter().filter(|index| **index == needle).count()
        }
        InstKind::Call { args, .. } => args.iter().filter(|arg| **arg == needle).count(),
    }
}

fn terminator_value_use_count(terminator: &Terminator, needle: ValueId) -> usize {
    match terminator {
        Terminator::Return(Some(value)) => (*value == needle) as usize,
        Terminator::Branch { cond, .. } => (*cond == needle) as usize,
        Terminator::Return(None) | Terminator::Jump(_) => 0,
    }
}

fn rewrite_tail_call(
    func: &mut Function,
    param_slots: &[ValueId],
    loop_entry: BlockId,
    site: TailCallSite,
) {
    let stores = param_slots
        .iter()
        .copied()
        .zip(site.args)
        .map(|(ptr, value)| InstKind::Store { ptr, value })
        .collect::<Vec<_>>();

    let insert_pos = site.call_idx + 1;
    func.blocks[site.block.0].insts[site.call_idx].result = None;
    func.blocks[site.block.0].insts[site.call_idx].kind = InstKind::Nop;
    for (offset, store) in stores.into_iter().enumerate() {
        func.insert_inst(site.block, insert_pos + offset, store, None);
    }
    func.blocks[site.block.0].terminator = Some(Terminator::Jump(loop_entry));
}
