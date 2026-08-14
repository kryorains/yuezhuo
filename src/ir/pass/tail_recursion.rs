use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
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
    accumulation: Option<Accumulation>,
}

#[derive(Debug, Clone, Copy)]
struct Accumulation {
    combine_idx: usize,
    contribution: ValueId,
}

fn eliminate_tail_recursion(func: &mut Function) {
    let initial_sites = collect_tail_self_calls(func);
    if initial_sites.is_empty() {
        return;
    }

    let Some(param_slots) = collect_param_slots(func) else {
        return;
    };
    let accumulator =
        if initial_sites.iter().any(|site| site.accumulation.is_some()) && func.ret == Type::I32 {
            let zero = func.add_const(Const::Int(0));
            let slot = func
                .append_inst(
                    func.entry,
                    InstKind::Alloca { ty: Type::I32 },
                    Some(Type::Ptr(Box::new(Type::I32))),
                )
                .expect("an accumulator alloca must produce a pointer");
            func.append_inst(
                func.entry,
                InstKind::Store {
                    ptr: slot,
                    value: zero,
                },
                None,
            );
            Some((slot, zero))
        } else {
            None
        };
    let loop_entry = split_entry(func, &param_slots, accumulator);
    let mut sites = collect_tail_self_calls(func);
    if sites.is_empty() {
        return;
    }

    if let Some((accumulator, _)) = accumulator {
        rewrite_base_returns_with_accumulator(func, accumulator, &sites);
    }

    sites.sort_by_key(|site| (site.block.0, site.call_idx));
    for site in sites.into_iter().rev() {
        rewrite_tail_call(
            func,
            &param_slots,
            accumulator.map(|(slot, _)| slot),
            loop_entry,
            site,
        );
    }
    if accumulator.is_some() {
        func.mark_accumulator_tail_recursion_eliminated();
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

fn split_entry(
    func: &mut Function,
    param_slots: &[ValueId],
    accumulator: Option<(ValueId, ValueId)>,
) -> BlockId {
    let entry = func.entry;
    let params = func.params.clone();
    let old_terminator = func.blocks[entry.0].terminator.take();
    let old_insts = std::mem::take(&mut func.blocks[entry.0].insts);
    let mut prelude_insts = Vec::new();
    let mut body_insts = Vec::new();
    let mut new_locations = Vec::with_capacity(old_insts.len());

    let loop_entry = func.add_block("tail.entry");
    for inst in old_insts {
        if is_entry_prelude_inst(&inst, param_slots, &params, accumulator) {
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

fn is_entry_prelude_inst(
    inst: &Inst,
    param_slots: &[ValueId],
    params: &[ValueId],
    accumulator: Option<(ValueId, ValueId)>,
) -> bool {
    match &inst.kind {
        InstKind::Alloca { .. } => true,
        InstKind::Store { ptr, value } => {
            param_slots
                .iter()
                .zip(params.iter())
                .any(|(slot, param)| ptr == slot && value == param)
                || accumulator == Some((*ptr, *value))
        }
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
    if let ValueKind::Inst(call_block, call_idx) = func.value(value).kind {
        let inst = func.block(block).insts.get(call_idx)?;
        if call_block == block
            && only_nops_after(func, block, call_idx)
            && value_use_count(func, value) == 1
            && inst.result == Some(value)
        {
            if let InstKind::Call { name, args } = &inst.kind {
                if name == &func.name && args.len() == func.params.len() {
                    return Some(TailCallSite {
                        block,
                        call_idx,
                        args: args.clone(),
                        accumulation: None,
                    });
                }
            }
        }
    }

    let ValueKind::Inst(combine_block, combine_idx) = func.value(value).kind else {
        return None;
    };
    if combine_block != block
        || !only_nops_after(func, block, combine_idx)
        || value_use_count(func, value) != 1
    {
        return None;
    }
    let combine = func.block(block).insts.get(combine_idx)?;
    let InstKind::Binary {
        op: BinaryOp::Iadd,
        lhs,
        rhs,
    } = combine.kind
    else {
        return None;
    };
    // Preserve source evaluation order: only the last self call can become the
    // backedge.  The accumulated contribution must already be available when
    // that call would execute.
    let (call_result, contribution, call_idx) = [(lhs, rhs), (rhs, lhs)]
        .into_iter()
        .filter_map(|(call_result, contribution)| {
            let ValueKind::Inst(call_block, call_idx) = func.value(call_result).kind else {
                return None;
            };
            (call_block == block
                && is_self_call_result(func, block, call_result)
                && value_available_before(func, block, contribution, call_idx))
            .then_some((call_result, contribution, call_idx))
        })
        .max_by_key(|(_, _, call_idx)| *call_idx)?;
    let call_block = block;
    if call_block != block
        || call_idx >= combine_idx
        || !func.block(block).insts[call_idx + 1..combine_idx]
            .iter()
            .all(|inst| matches!(inst.kind, InstKind::Nop))
        || value_use_count(func, call_result) != 1
        || func.value(contribution).ty != Type::I32
    {
        return None;
    }
    let inst = func.block(block).insts.get(call_idx)?;
    match &inst.kind {
        InstKind::Call { name, args } if name == &func.name && args.len() == func.params.len() => {
            Some(TailCallSite {
                block,
                call_idx,
                args: args.clone(),
                accumulation: Some(Accumulation {
                    combine_idx,
                    contribution,
                }),
            })
        }
        _ => None,
    }
}

fn value_available_before(
    func: &Function,
    block: BlockId,
    value: ValueId,
    inst_idx: usize,
) -> bool {
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(owner, owner_idx) => owner != block || owner_idx < inst_idx,
    }
}

fn is_self_call_result(func: &Function, block: BlockId, value: ValueId) -> bool {
    let ValueKind::Inst(owner, inst_idx) = func.value(value).kind else {
        return false;
    };
    owner == block
        && matches!(
            func.block(owner).insts.get(inst_idx),
            Some(Inst {
                result: Some(result),
                kind: InstKind::Call { name, args },
            }) if *result == value && name == &func.name && args.len() == func.params.len()
        )
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
                accumulation: None,
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
        InstKind::MemCopy {
            dst, src, count, ..
        } => (*dst == needle) as usize + (*src == needle) as usize + (*count == needle) as usize,
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
    accumulator: Option<ValueId>,
    loop_entry: BlockId,
    site: TailCallSite,
) {
    let stores = param_slots
        .iter()
        .copied()
        .zip(site.args)
        .map(|(ptr, value)| InstKind::Store { ptr, value })
        .collect::<Vec<_>>();

    func.blocks[site.block.0].insts[site.call_idx].result = None;
    func.blocks[site.block.0].insts[site.call_idx].kind = InstKind::Nop;
    if let Some(accumulation) = site.accumulation {
        func.blocks[site.block.0].insts[accumulation.combine_idx].result = None;
        func.blocks[site.block.0].insts[accumulation.combine_idx].kind = InstKind::Nop;
        let accumulator = accumulator.expect("accumulating tail calls require an accumulator");
        let current = func
            .append_inst(
                site.block,
                InstKind::Load { ptr: accumulator },
                Some(Type::I32),
            )
            .expect("an accumulator load must produce i32");
        let updated = func
            .append_inst(
                site.block,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: current,
                    rhs: accumulation.contribution,
                },
                Some(Type::I32),
            )
            .expect("an accumulator update must produce i32");
        func.append_inst(
            site.block,
            InstKind::Store {
                ptr: accumulator,
                value: updated,
            },
            None,
        );
    }
    for store in stores {
        func.append_inst(site.block, store, None);
    }
    func.blocks[site.block.0].terminator = Some(Terminator::Jump(loop_entry));
}

fn rewrite_base_returns_with_accumulator(
    func: &mut Function,
    accumulator: ValueId,
    sites: &[TailCallSite],
) {
    let tail_blocks = sites.iter().map(|site| site.block).collect::<HashSet<_>>();
    for block_idx in 0..func.blocks.len() {
        let block = BlockId(block_idx);
        if tail_blocks.contains(&block) {
            continue;
        }
        let Some(Terminator::Return(Some(value))) = func.blocks[block_idx].terminator.clone()
        else {
            continue;
        };
        let current = func
            .append_inst(block, InstKind::Load { ptr: accumulator }, Some(Type::I32))
            .expect("an accumulator load must produce i32");
        let result = func
            .append_inst(
                block,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: current,
                    rhs: value,
                },
                Some(Type::I32),
            )
            .expect("an accumulated return must produce i32");
        func.blocks[block_idx].terminator = Some(Terminator::Return(Some(result)));
    }
}
