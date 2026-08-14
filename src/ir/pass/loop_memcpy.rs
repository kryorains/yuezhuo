use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{
    BlockId, CmpOp, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::HashSet;

/// Replaces a canonical unit-stride copy loop over proven-disjoint objects
/// with an explicit memory copy. This is the standard loop-idiom transform
/// performed by GCC and LLVM; no source name, array dimension, or trip-count
/// threshold participates in recognition.
pub(super) struct LoopMemcpyPass;

impl LoopMemcpyPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LoopMemcpyPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            while let Some(plan) = find_plan(func) {
                apply_plan(func, plan);
                if let Err(errors) = func.verify() {
                    panic!(
                        "loop memcpy recognition produced invalid IR in {}: {:?}",
                        func.name, errors
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Plan {
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    condition: ValueId,
    initial_induction: ValueId,
    bound: ValueId,
    initial_dst: ValueId,
    initial_src: ValueId,
    element_bytes: usize,
}

fn find_plan(func: &Function) -> Option<Plan> {
    const MAX_BLOCKS: usize = 1024;
    const MAX_VALUES: usize = 16_384;
    if func.blocks.len() > MAX_BLOCKS || func.values.len() > MAX_VALUES {
        return None;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom);
    loops
        .loops()
        .iter()
        .find_map(|natural_loop| analyze_loop(func, natural_loop))
}

fn analyze_loop(func: &Function, natural_loop: &NaturalLoop) -> Option<Plan> {
    let preheader = natural_loop.dedicated_preheader?;
    let header = natural_loop.header;
    let body = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    if header == body
        || natural_loop.blocks.len() != 2
        || natural_loop.exit_edges.as_slice() != [(header, exit)]
        || !matches!(
            func.blocks.get(preheader.0)?.terminator,
            Some(Terminator::Jump(target)) if target == header
        )
        || !matches!(
            func.blocks.get(body.0)?.terminator,
            Some(Terminator::Jump(target)) if target == header
        )
    {
        return None;
    }

    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks.get(header.0)?.terminator.as_ref()?
    else {
        return None;
    };
    if *then_target != body || *else_target != exit {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt,
        lhs,
        rhs: bound,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    if !is_loop_invariant(func, natural_loop, *bound) {
        return None;
    }
    let induction = analyze_i32_induction(func, natural_loop, *lhs)?;
    if induction.phi != *lhs || induction.step != 1 || const_i32(func, induction.initial) != Some(0)
    {
        return None;
    }

    let header_active = active_instructions(&func.blocks[header.0]);
    let phi_results = header_active
        .iter()
        .filter_map(|(_, inst)| matches!(inst.kind, InstKind::Phi { .. }).then_some(inst.result?))
        .collect::<Vec<_>>();
    if header_active.len() != phi_results.len() + 1
        || phi_results.len() != 3
        || !phi_results.contains(&induction.phi)
        || !header_active
            .iter()
            .any(|(_, inst)| inst.result == Some(*cond))
    {
        return None;
    }

    let body_active = active_instructions(&func.blocks[body.0]);
    if body_active.len() != 5 {
        return None;
    }
    let (loaded, src_pointer) = body_active.iter().find_map(|(_, inst)| {
        let (Some(result), InstKind::Load { ptr }) = (inst.result, &inst.kind) else {
            return None;
        };
        Some((result, *ptr))
    })?;
    let dst_pointer = body_active.iter().find_map(|(_, inst)| {
        let InstKind::Store { ptr, value } = inst.kind else {
            return None;
        };
        (value == loaded).then_some(ptr)
    })?;
    if src_pointer == dst_pointer
        || !phi_results.contains(&src_pointer)
        || !phi_results.contains(&dst_pointer)
    {
        return None;
    }

    let initial_src = phi_incoming(func, src_pointer, preheader)?;
    let next_src = phi_incoming(func, src_pointer, body)?;
    let initial_dst = phi_incoming(func, dst_pointer, preheader)?;
    let next_dst = phi_incoming(func, dst_pointer, body)?;
    if !is_unit_pointer_step(func, next_src, src_pointer)
        || !is_unit_pointer_step(func, next_dst, dst_pointer)
        || !body_active
            .iter()
            .any(|(_, inst)| inst.result == Some(next_src))
        || !body_active
            .iter()
            .any(|(_, inst)| inst.result == Some(next_dst))
        || !body_active
            .iter()
            .any(|(_, inst)| inst.result == Some(induction.next))
    {
        return None;
    }

    let element_ty = pointee_scalar(&func.value(src_pointer).ty)?;
    if pointee_scalar(&func.value(dst_pointer).ty)? != element_ty
        || func.value(loaded).ty != *element_ty
    {
        return None;
    }
    let element_bytes = match element_ty {
        Type::I1 | Type::I32 | Type::F32 => 4,
        _ => return None,
    };
    if !proven_disjoint_objects(func, initial_src, initial_dst) {
        return None;
    }

    let loop_values = phi_results
        .iter()
        .copied()
        .chain(body_active.iter().filter_map(|(_, inst)| inst.result))
        .collect::<HashSet<_>>();
    if loop_values
        .iter()
        .any(|value| used_outside_loop(func, natural_loop, *value))
    {
        return None;
    }

    Some(Plan {
        header,
        body,
        exit,
        condition: *cond,
        initial_induction: induction.initial,
        bound: *bound,
        initial_dst,
        initial_src,
        element_bytes,
    })
}

fn apply_plan(func: &mut Function, plan: Plan) {
    func.append_inst(
        plan.body,
        InstKind::MemCopy {
            dst: plan.initial_dst,
            src: plan.initial_src,
            element_bytes: plan.element_bytes,
            count: plan.bound,
        },
        None,
    );
    for inst in &mut func.blocks[plan.header.0].insts {
        if matches!(inst.kind, InstKind::Phi { .. }) {
            inst.result = None;
            inst.kind = InstKind::Nop;
        }
    }
    let Some(Inst {
        kind: InstKind::Icmp { lhs, .. },
        ..
    }) = func.blocks[plan.header.0]
        .insts
        .iter_mut()
        .find(|inst| inst.result == Some(plan.condition))
    else {
        unreachable!("the accepted copy-loop condition must remain present")
    };
    *lhs = plan.initial_induction;
    let copy_index = func.blocks[plan.body.0].insts.len() - 1;
    for inst in &mut func.blocks[plan.body.0].insts[..copy_index] {
        inst.result = None;
        inst.kind = InstKind::Nop;
    }
    func.blocks[plan.body.0].terminator = Some(Terminator::Jump(plan.exit));
}

fn active_instructions(block: &crate::ir::Block) -> Vec<(usize, &Inst)> {
    block
        .insts
        .iter()
        .enumerate()
        .filter(|(_, inst)| !matches!(inst.kind, InstKind::Nop))
        .collect()
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, index) = func.values.get(value.0)?.kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(index)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

fn phi_incoming(func: &Function, phi: ValueId, pred: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    let mut values = incomings
        .iter()
        .filter_map(|(candidate, value)| (*candidate == pred).then_some(*value));
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn is_unit_pointer_step(func: &Function, next: ValueId, pointer: ValueId) -> bool {
    matches!(
        defining_inst(func, next),
        Some(InstKind::Gep { base, indices })
            if *base == pointer
                && indices.len() == 1
                && const_i32(func, indices[0]) == Some(1)
    )
}

fn proven_disjoint_objects(func: &Function, lhs: ValueId, rhs: ValueId) -> bool {
    match (pointer_object(func, lhs), pointer_object(func, rhs)) {
        (Some(PointerObject::Global(lhs)), Some(PointerObject::Global(rhs))) => lhs != rhs,
        (Some(PointerObject::Alloca(lhs)), Some(PointerObject::Alloca(rhs))) => lhs != rhs,
        (Some(PointerObject::Global(_)), Some(PointerObject::Alloca(_)))
        | (Some(PointerObject::Alloca(_)), Some(PointerObject::Global(_))) => true,
        _ => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PointerObject<'a> {
    Global(&'a str),
    Alloca(ValueId),
}

fn pointer_object(func: &Function, value: ValueId) -> Option<PointerObject<'_>> {
    let mut worklist = vec![value];
    let mut visited = HashSet::new();
    let mut object = None;
    while let Some(current) = worklist.pop() {
        if !visited.insert(current) {
            continue;
        }
        let candidate = match &func.values.get(current.0)?.kind {
            ValueKind::Global(name) => Some(PointerObject::Global(name)),
            ValueKind::Inst(_, _) => match defining_inst(func, current)? {
                InstKind::Alloca { .. } => Some(PointerObject::Alloca(current)),
                InstKind::Gep { base, .. } => {
                    worklist.push(*base);
                    None
                }
                InstKind::Phi { incomings } if !incomings.is_empty() => {
                    worklist.extend(incomings.iter().map(|(_, value)| *value));
                    None
                }
                _ => return None,
            },
            ValueKind::Param | ValueKind::Const(_) => return None,
        };
        if let Some(candidate) = candidate {
            match object {
                Some(previous) if previous != candidate => return None,
                Some(_) => {}
                None => object = Some(candidate),
            }
        }
    }
    object
}

fn used_outside_loop(func: &Function, natural_loop: &NaturalLoop, needle: ValueId) -> bool {
    for (block_idx, block) in func.blocks.iter().enumerate() {
        if natural_loop.blocks.contains(&BlockId(block_idx)) {
            continue;
        }
        if block
            .insts
            .iter()
            .flat_map(inst_operands)
            .any(|operand| operand == needle)
            || terminator_operands(block.terminator.as_ref()).any(|operand| operand == needle)
        {
            return true;
        }
    }
    false
}

fn is_loop_invariant(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Inst(block, _)) => !natural_loop.blocks.contains(block),
        Some(ValueKind::Param | ValueKind::Global(_) | ValueKind::Const(_)) => true,
        None => false,
    }
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match func.values.get(value.0)?.kind {
        ValueKind::Const(Const::Int(value)) => Some(value),
        _ => None,
    }
}

fn pointee_scalar(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Ptr(pointee) => Some(pointee),
        _ => None,
    }
}

fn inst_operands(inst: &Inst) -> Vec<ValueId> {
    match &inst.kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => std::iter::once(*base)
            .chain(indices.iter().copied())
            .collect(),
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: Option<&Terminator>) -> impl Iterator<Item = ValueId> {
    match terminator {
        Some(Terminator::Return(Some(value))) => Some(*value),
        Some(Terminator::Branch { cond, .. }) => Some(*cond),
        Some(Terminator::Return(None) | Terminator::Jump(_)) | None => None,
    }
    .into_iter()
}
