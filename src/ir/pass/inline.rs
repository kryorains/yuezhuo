use super::util::{resolve_replacement, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CastOp, Function, Inst, InstKind, Module, Terminator, Type, UnaryOp, Value,
    ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_INLINE_INSTS: usize = 8;
const MAX_INLINE_SLOTS: usize = 64;
const MAX_INLINED_INSTS_PER_FUNCTION: usize = 512;

pub(super) struct InlineSmallExprPass;

impl InlineSmallExprPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for InlineSmallExprPass {
    fn run(&mut self, module: &mut Module) {
        let candidates = module
            .funcs
            .iter()
            .filter(|func| is_inline_candidate(func))
            .map(|func| (func.name.clone(), func.clone()))
            .collect::<HashMap<_, _>>();

        for func in &mut module.funcs {
            inline_calls(func, &candidates);
        }
    }
}

fn is_inline_candidate(func: &Function) -> bool {
    if func.blocks.len() != 1 || !is_inline_scalar(&func.ret) {
        return false;
    }
    let mut unique_params = HashSet::new();
    if func.params.iter().any(|param| {
        !unique_params.insert(*param)
            || !matches!(
                func.values.get(param.0),
                Some(Value {
                    ty,
                    kind: ValueKind::Param,
                    ..
                }) if is_inline_scalar(ty)
            )
    }) {
        return false;
    }

    let block = &func.blocks[0];
    let Some(Terminator::Return(Some(returned))) = block.terminator.as_ref() else {
        return false;
    };
    if block.insts.len() > MAX_INLINE_SLOTS
        || !matches!(func.values.get(returned.0), Some(value) if value.ty == func.ret)
    {
        return false;
    }
    let dead_slots = discardable_allocas(func);
    let active_inst_count = block
        .insts
        .iter()
        .filter(|inst| !is_discardable_inline_inst(inst, &dead_slots))
        .count();
    if active_inst_count > MAX_INLINE_INSTS {
        return false;
    }

    let mut available = func.params.iter().copied().collect::<HashSet<_>>();
    available.extend(func.values.iter().enumerate().filter_map(|(idx, value)| {
        matches!(&value.kind, ValueKind::Const(constant) if value.ty == constant.ty())
            .then_some(ValueId(idx))
    }));

    for inst in &block.insts {
        if is_discardable_inline_inst(inst, &dead_slots) {
            continue;
        }
        if !is_inline_inst(&inst.kind)
            || inst_operands(&inst.kind)
                .into_iter()
                .any(|operand| !available.contains(&operand))
            || !inline_inst_types_match(func, inst)
        {
            return false;
        }
        let Some(result) = inst.result else {
            return false;
        };
        available.insert(result);
    }

    available.contains(returned) && func.verify().is_ok()
}

fn is_inline_scalar(ty: &Type) -> bool {
    matches!(ty, Type::I1 | Type::I32 | Type::F32)
}

fn discardable_allocas(func: &Function) -> HashSet<ValueId> {
    func.blocks[0]
        .insts
        .iter()
        .filter_map(|inst| match (&inst.kind, inst.result) {
            (InstKind::Alloca { ty }, Some(slot))
                if is_inline_scalar(ty) && alloca_is_write_only(func, slot) =>
            {
                Some(slot)
            }
            _ => None,
        })
        .collect()
}

fn alloca_is_write_only(func: &Function, slot: ValueId) -> bool {
    func.blocks.iter().all(|block| {
        block.insts.iter().all(|inst| match &inst.kind {
            InstKind::Store { ptr, value } if *ptr == slot => *value != slot,
            kind => !inst_uses_value(kind, slot),
        }) && !terminator_uses_value(block.terminator.as_ref(), slot)
    })
}

fn is_discardable_inline_inst(inst: &Inst, dead_slots: &HashSet<ValueId>) -> bool {
    matches!(inst.kind, InstKind::Nop)
        || inst
            .result
            .is_some_and(|result| dead_slots.contains(&result))
        || matches!(inst.kind, InstKind::Store { ptr, .. } if dead_slots.contains(&ptr))
}

fn inst_uses_value(kind: &InstKind, value: ValueId) -> bool {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => false,
        InstKind::Phi { incomings } => incomings.iter().any(|(_, incoming)| *incoming == value),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => *ptr == value,
        InstKind::Store { ptr, value: stored } => *ptr == value || *stored == value,
        InstKind::Unary { value: operand, .. } | InstKind::Cast { value: operand, .. } => {
            *operand == value
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => *lhs == value || *rhs == value,
        InstKind::Gep { base, indices } => *base == value || indices.contains(&value),
        InstKind::Call { args, .. } => args.contains(&value),
    }
}

fn terminator_uses_value(terminator: Option<&Terminator>, value: ValueId) -> bool {
    match terminator {
        Some(Terminator::Return(returned)) => returned == &Some(value),
        Some(Terminator::Branch { cond, .. }) => *cond == value,
        Some(Terminator::Jump(_)) | None => false,
    }
}

fn is_inline_inst(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Unary { .. }
            | InstKind::Binary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. }
    )
}

fn inline_inst_types_match(func: &Function, inst: &Inst) -> bool {
    let Some(result_ty) = inst
        .result
        .and_then(|result| func.values.get(result.0))
        .map(|value| &value.ty)
    else {
        return false;
    };
    let ty = |value: ValueId| func.values.get(value.0).map(|value| &value.ty);

    match &inst.kind {
        InstKind::Unary { op, value } => match op {
            UnaryOp::Ineg => ty(*value) == Some(&Type::I32) && result_ty == &Type::I32,
            UnaryOp::Fneg => ty(*value) == Some(&Type::F32) && result_ty == &Type::F32,
            UnaryOp::Not => ty(*value) == Some(&Type::I1) && result_ty == &Type::I1,
        },
        InstKind::Binary { op, lhs, rhs } => {
            let expected = match op {
                BinaryOp::Iadd
                | BinaryOp::Isub
                | BinaryOp::Imul
                | BinaryOp::Idiv
                | BinaryOp::Imod
                | BinaryOp::Iand
                | BinaryOp::Ior
                | BinaryOp::Ixor
                | BinaryOp::Ishl
                | BinaryOp::Iashr => Type::I32,
                BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => Type::F32,
                BinaryOp::And | BinaryOp::Or => Type::I1,
            };
            ty(*lhs) == Some(&expected) && ty(*rhs) == Some(&expected) && result_ty == &expected
        }
        InstKind::Icmp { lhs, rhs, .. } => {
            ty(*lhs) == Some(&Type::I32) && ty(*rhs) == Some(&Type::I32) && result_ty == &Type::I1
        }
        InstKind::Fcmp { lhs, rhs, .. } => {
            ty(*lhs) == Some(&Type::F32) && ty(*rhs) == Some(&Type::F32) && result_ty == &Type::I1
        }
        InstKind::Cast { op, value } => {
            let (source, target) = match op {
                CastOp::I32ToF32 => (Type::I32, Type::F32),
                CastOp::F32ToI32 => (Type::F32, Type::I32),
                CastOp::BoolToI32 => (Type::I1, Type::I32),
                CastOp::I32ToBool => (Type::I32, Type::I1),
                CastOp::F32ToBool => (Type::F32, Type::I1),
            };
            ty(*value) == Some(&source) && result_ty == &target
        }
        _ => false,
    }
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        _ => Vec::new(),
    }
}

fn inline_calls(func: &mut Function, candidates: &HashMap<String, Function>) {
    let mut replacements = ValueReplacements::new();
    let mut inlined_inst_count = 0usize;

    for block_idx in 0..func.blocks.len() {
        let block = BlockId(block_idx);
        let mut inst_idx = 0usize;
        while inst_idx < func.blocks[block_idx].insts.len() {
            let inst = func.blocks[block_idx].insts[inst_idx].clone();
            let InstKind::Call { name, args } = inst.kind else {
                inst_idx += 1;
                continue;
            };
            let Some(result) = inst.result else {
                inst_idx += 1;
                continue;
            };
            let Some(callee) = candidates.get(&name) else {
                inst_idx += 1;
                continue;
            };
            let dead_slots = discardable_allocas(callee);
            let inline_cost = callee.blocks[0]
                .insts
                .iter()
                .filter(|inst| !is_discardable_inline_inst(inst, &dead_slots))
                .count();
            if inlined_inst_count + inline_cost > MAX_INLINED_INSTS_PER_FUNCTION {
                inst_idx += 1;
                continue;
            }

            let args = args
                .into_iter()
                .map(|arg| resolve_replacement(arg, &replacements))
                .collect::<Vec<_>>();
            if !call_types_match(func, result, callee, &args) {
                inst_idx += 1;
                continue;
            }

            let (returned, inserted) = inline_call_at(func, block, inst_idx, callee, &args);
            let call_idx = inst_idx + inserted;
            func.blocks[block_idx].insts[call_idx] = Inst {
                result: None,
                kind: InstKind::Nop,
            };
            replacements.insert(result, returned);
            inlined_inst_count += inserted;
            inst_idx = call_idx + 1;
        }
    }

    if rewrite_function_uses(func, &replacements) {
        if let Err(errors) = func.verify() {
            panic!(
                "small-expression inlining produced invalid IR in {}: {:?}",
                func.name, errors
            );
        }
    }
}

fn call_types_match(func: &Function, result: ValueId, callee: &Function, args: &[ValueId]) -> bool {
    args.len() == callee.params.len()
        && func.value(result).ty == callee.ret
        && args
            .iter()
            .zip(&callee.params)
            .all(|(arg, param)| func.value(*arg).ty == callee.value(*param).ty)
}

fn inline_call_at(
    func: &mut Function,
    block: BlockId,
    inst_idx: usize,
    callee: &Function,
    args: &[ValueId],
) -> (ValueId, usize) {
    let mut values = callee
        .params
        .iter()
        .copied()
        .zip(args.iter().copied())
        .collect::<HashMap<_, _>>();
    let dead_slots = discardable_allocas(callee);
    let mut inserted = 0usize;

    for inst in &callee.blocks[0].insts {
        if is_discardable_inline_inst(inst, &dead_slots) {
            continue;
        }
        let result = inst.result.expect("validated inline instruction result");
        let kind = clone_inline_inst(func, callee, &inst.kind, &mut values);
        let new_result = func
            .insert_inst(
                block,
                inst_idx + inserted,
                kind,
                Some(callee.value(result).ty.clone()),
            )
            .unwrap();
        values.insert(result, new_result);
        inserted += 1;
    }

    let Some(Terminator::Return(Some(returned))) = &callee.blocks[0].terminator else {
        unreachable!("validated inline return");
    };
    let returned = map_inline_value(func, callee, *returned, &mut values);
    (returned, inserted)
}

fn clone_inline_inst(
    func: &mut Function,
    callee: &Function,
    kind: &InstKind,
    values: &mut HashMap<ValueId, ValueId>,
) -> InstKind {
    match kind {
        InstKind::Unary { op, value } => InstKind::Unary {
            op: *op,
            value: map_inline_value(func, callee, *value, values),
        },
        InstKind::Binary { op, lhs, rhs } => InstKind::Binary {
            op: *op,
            lhs: map_inline_value(func, callee, *lhs, values),
            rhs: map_inline_value(func, callee, *rhs, values),
        },
        InstKind::Icmp { op, lhs, rhs } => InstKind::Icmp {
            op: *op,
            lhs: map_inline_value(func, callee, *lhs, values),
            rhs: map_inline_value(func, callee, *rhs, values),
        },
        InstKind::Fcmp { op, lhs, rhs } => InstKind::Fcmp {
            op: *op,
            lhs: map_inline_value(func, callee, *lhs, values),
            rhs: map_inline_value(func, callee, *rhs, values),
        },
        InstKind::Cast { op, value } => InstKind::Cast {
            op: *op,
            value: map_inline_value(func, callee, *value, values),
        },
        _ => unreachable!("validated inline instruction kind"),
    }
}

fn map_inline_value(
    func: &mut Function,
    callee: &Function,
    value: ValueId,
    values: &mut HashMap<ValueId, ValueId>,
) -> ValueId {
    if let Some(mapped) = values.get(&value).copied() {
        return mapped;
    }
    let ValueKind::Const(constant) = &callee.value(value).kind else {
        unreachable!("validated inline operand");
    };
    let mapped = get_or_add_const(func, constant.clone());
    values.insert(value, mapped);
    mapped
}

fn get_or_add_const(func: &mut Function, constant: crate::ir::Const) -> ValueId {
    let expected_ty = constant.ty();
    func.values
        .iter()
        .position(|value| {
            value.ty == expected_ty
                && matches!(&value.kind, ValueKind::Const(existing) if *existing == constant)
        })
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(constant))
}
