use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{
    BlockId, CmpOp, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};

/// Recognizes a canonical unit-stride loop whose only observable effect is
/// storing zero to consecutive scalar elements. This is the ordinary loop
/// idiom recognition performed by GCC and LLVM before lowering to `memset`.
pub(super) struct LoopMemsetPass;

impl LoopMemsetPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LoopMemsetPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            // Recompute natural loops after each rewrite because removing a
            // backedge changes predecessor and dominance information.
            while let Some(plan) = find_plan(func) {
                apply_plan(func, plan);
                if let Err(errors) = func.verify() {
                    panic!(
                        "loop memset recognition produced invalid IR in {}: {:?}",
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
    induction: ValueId,
    initial: ValueId,
    next: ValueId,
    condition: ValueId,
    bound: ValueId,
    address: ValueId,
    base: ValueId,
    store_index: usize,
    element_bytes: i32,
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
    let header = natural_loop.header;
    let body = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    natural_loop.dedicated_preheader?;
    if header == body
        || natural_loop.blocks.len() != 2
        || natural_loop.exit_edges.as_slice() != [(header, exit)]
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
    if header_active.len() != 2
        || !header_active.iter().any(|(_, inst)| {
            inst.result == Some(induction.phi) && matches!(inst.kind, InstKind::Phi { .. })
        })
        || !header_active.iter().any(|(_, inst)| {
            inst.result == Some(*cond) && matches!(inst.kind, InstKind::Icmp { .. })
        })
    {
        return None;
    }

    let body_active = active_instructions(&func.blocks[body.0]);
    if body_active.len() != 3 {
        return None;
    }
    let (store_index, address, stored) = body_active.iter().find_map(|(index, inst)| {
        let InstKind::Store { ptr, value } = inst.kind else {
            return None;
        };
        Some((*index, ptr, value))
    })?;
    if !is_zero(func, stored) {
        return None;
    }
    let InstKind::Gep { base, indices } = defining_inst(func, address)? else {
        return None;
    };
    if indices.as_slice() != [induction.phi] || !is_loop_invariant(func, natural_loop, *base) {
        return None;
    }
    let element_bytes = match pointee_scalar(&func.value(address).ty)? {
        Type::I1 | Type::I32 | Type::F32 => 4,
        _ => return None,
    };
    if !body_active
        .iter()
        .any(|(_, inst)| inst.result == Some(address) && matches!(inst.kind, InstKind::Gep { .. }))
        || !body_active.iter().any(|(_, inst)| {
            inst.result == Some(induction.next) && matches!(inst.kind, InstKind::Binary { .. })
        })
    {
        return None;
    }

    // The eliminated recurrence and address must not escape the loop. In
    // particular, an exit phi would make the final induction value observable.
    for (block_idx, block) in func.blocks.iter().enumerate() {
        if natural_loop.blocks.contains(&BlockId(block_idx)) {
            continue;
        }
        if block
            .insts
            .iter()
            .flat_map(inst_operands)
            .any(|operand| {
                matches!(operand, value if value == induction.phi || value == induction.next || value == address)
            })
            || terminator_operands(block.terminator.as_ref()).any(|operand| {
                operand == induction.phi || operand == induction.next || operand == address
            })
        {
            return None;
        }
    }

    Some(Plan {
        header,
        body,
        exit,
        induction: induction.phi,
        initial: induction.initial,
        next: induction.next,
        condition: *cond,
        bound: *bound,
        address,
        base: *base,
        store_index,
        element_bytes,
    })
}

fn apply_plan(func: &mut Function, plan: Plan) {
    let initial_address = func
        .append_inst(
            plan.body,
            InstKind::Gep {
                base: plan.base,
                indices: vec![plan.initial],
            },
            Some(func.value(plan.address).ty.clone()),
        )
        .expect("a memset address must produce a pointer");
    func.append_inst(
        plan.body,
        InstKind::MemZero {
            ptr: initial_address,
            bytes: plan.element_bytes as usize,
            count: Some(plan.bound),
        },
        None,
    );

    for inst in &mut func.blocks[plan.header.0].insts {
        if inst.result == Some(plan.induction) {
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
        unreachable!("the accepted loop condition must remain present")
    };
    *lhs = plan.initial;

    for inst in &mut func.blocks[plan.body.0].insts[..=plan.store_index] {
        if inst.result == Some(plan.address) || matches!(inst.kind, InstKind::Store { .. }) {
            inst.result = None;
            inst.kind = InstKind::Nop;
        }
    }
    for inst in &mut func.blocks[plan.body.0].insts {
        if inst.result == Some(plan.next) {
            inst.result = None;
            inst.kind = InstKind::Nop;
        }
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

fn is_zero(func: &Function, value: ValueId) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Const(
            Const::Int(0) | Const::Bool(false) | Const::Float(0) | Const::Zero(_)
        ))
    )
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
