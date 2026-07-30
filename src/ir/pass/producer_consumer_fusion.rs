use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::util::{defining_inst, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, Const, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::HashMap;

/// Fuses a canonical pair of local-array producers with an immediately
/// following floating-point dot-product consumer.
///
/// The producer's stored SSA values are the values the consumer would load.
/// Accumulating them in the producer loop makes both temporary arrays
/// write-only, allowing the normal DCE pass to erase their stack traffic.
pub(super) struct ProducerConsumerFusionPass;

impl ProducerConsumerFusionPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for ProducerConsumerFusionPass {
    fn run(&mut self, module: &mut Module) {
        let consumers = module
            .funcs
            .iter()
            .filter_map(analyze_dot_consumer)
            .map(|consumer| (consumer.name.clone(), consumer))
            .collect::<HashMap<_, _>>();
        if consumers.is_empty() {
            return;
        }
        for func in &mut module.funcs {
            fuse_calls(func, &consumers);
        }
    }
}

#[derive(Clone)]
struct DotConsumer {
    name: String,
    lhs_param: usize,
    rhs_param: usize,
    length_param: usize,
    initial: Const,
}

fn analyze_dot_consumer(func: &Function) -> Option<DotConsumer> {
    if func.ret != Type::F32 || func.params.len() != 3 || func.verify().is_err() {
        return None;
    }
    let (lhs_param, rhs_param, length_param) = match (
        &func.value(func.params[0]).ty,
        &func.value(func.params[1]).ty,
        &func.value(func.params[2]).ty,
    ) {
        (Type::Ptr(lhs), Type::Ptr(rhs), Type::I32)
            if lhs.as_ref() == &Type::F32 && rhs.as_ref() == &Type::F32 =>
        {
            (0, 1, 2)
        }
        _ => return None,
    };

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom);
    let natural_loop = loops.loops().iter().find(|natural_loop| {
        natural_loop.dedicated_preheader == Some(func.entry)
            && natural_loop.blocks.len() == 2
            && natural_loop.exit_edges.len() == 1
    })?;
    let latch = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    if natural_loop.exit_edges != [(natural_loop.header, exit)]
        || !matches!(
            func.blocks[latch.0].terminator,
            Some(Terminator::Jump(target)) if target == natural_loop.header
        )
        || !matches!(
            func.blocks[exit.0].terminator,
            Some(Terminator::Return(Some(_)))
        )
    {
        return None;
    }

    let phis = func.blocks[natural_loop.header.0]
        .insts
        .iter()
        .filter_map(|inst| matches!(inst.kind, InstKind::Phi { .. }).then_some(inst.result?))
        .collect::<Vec<_>>();
    if phis.len() != 2 {
        return None;
    }
    let induction = phis
        .iter()
        .find_map(|phi| analyze_i32_induction(func, natural_loop, *phi))?;
    if induction.step != 1
        || !matches!(
            func.value(induction.initial).kind,
            ValueKind::Const(Const::Int(0) | Const::Zero(Type::I32))
        )
    {
        return None;
    }
    let accumulator = phis
        .iter()
        .copied()
        .find(|phi| *phi != induction.phi && func.value(*phi).ty == Type::F32)?;
    let (accumulator_initial, accumulator_next) =
        phi_incomings(func, accumulator, func.entry, latch)?;
    let initial = match &func.value(accumulator_initial).kind {
        ValueKind::Const(constant @ (Const::Float(_) | Const::Zero(Type::F32))) => constant.clone(),
        _ => return None,
    };
    let returned = match &func.blocks[exit.0].terminator {
        Some(Terminator::Return(Some(returned))) => *returned,
        _ => return None,
    };
    if returned != accumulator
        || canonical_bound(func, natural_loop, induction.phi)? != func.params[length_param]
    {
        return None;
    }

    let InstKind::Binary {
        op: BinaryOp::Fadd,
        lhs: add_lhs,
        rhs: product,
    } = defining_inst(func, accumulator_next)?
    else {
        return None;
    };
    if *add_lhs != accumulator {
        return None;
    }
    let InstKind::Binary {
        op: BinaryOp::Fmul,
        lhs: loaded_lhs,
        rhs: loaded_rhs,
    } = defining_inst(func, *product)?
    else {
        return None;
    };
    if !is_indexed_load(func, *loaded_lhs, func.params[lhs_param], induction.phi)
        || !is_indexed_load(func, *loaded_rhs, func.params[rhs_param], induction.phi)
        || !consumer_has_only_expected_effects(func, natural_loop)
    {
        return None;
    }

    Some(DotConsumer {
        name: func.name.clone(),
        lhs_param,
        rhs_param,
        length_param,
        initial,
    })
}

fn canonical_bound(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: ValueId,
) -> Option<ValueId> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks[natural_loop.header.0].terminator.as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || natural_loop.blocks.contains(else_target) {
        return None;
    }
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
        return None;
    };
    (*op == crate::ir::CmpOp::Lt && *lhs == induction).then_some(*rhs)
}

fn phi_incomings(
    func: &Function,
    phi: ValueId,
    initial_pred: BlockId,
    latch: BlockId,
) -> Option<(ValueId, ValueId)> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    Some((
        incomings
            .iter()
            .find_map(|(pred, value)| (*pred == initial_pred).then_some(*value))?,
        incomings
            .iter()
            .find_map(|(pred, value)| (*pred == latch).then_some(*value))?,
    ))
}

fn is_indexed_load(
    func: &Function,
    loaded: ValueId,
    expected_base: ValueId,
    induction: ValueId,
) -> bool {
    let Some(InstKind::Load { ptr }) = defining_inst(func, loaded) else {
        return false;
    };
    matches!(
        defining_inst(func, *ptr),
        Some(InstKind::Gep { base, indices })
            if *base == expected_base && indices.as_slice() == [induction]
    )
}

fn consumer_has_only_expected_effects(func: &Function, natural_loop: &NaturalLoop) -> bool {
    natural_loop.blocks.iter().all(|block| {
        func.blocks[block.0].insts.iter().all(|inst| {
            matches!(
                inst.kind,
                InstKind::Nop
                    | InstKind::Phi { .. }
                    | InstKind::Load { .. }
                    | InstKind::Gep { .. }
                    | InstKind::Binary {
                        op: BinaryOp::Iadd | BinaryOp::Fmul | BinaryOp::Fadd,
                        ..
                    }
                    | InstKind::Icmp { .. }
            )
        })
    })
}

fn fuse_calls(func: &mut Function, consumers: &HashMap<String, DotConsumer>) {
    loop {
        let Some(plan) = find_fusion(func, consumers) else {
            return;
        };
        apply_fusion(func, plan);
    }
}

#[derive(Clone)]
struct FusionPlan {
    call_block: BlockId,
    call_index: usize,
    call_result: ValueId,
    producer_header: BlockId,
    producer_preheader: BlockId,
    producer_latch: BlockId,
    lhs_value: ValueId,
    rhs_value: ValueId,
    lhs_store: (BlockId, usize),
    rhs_store: (BlockId, usize),
    initial: Const,
}

fn find_fusion(func: &Function, consumers: &HashMap<String, DotConsumer>) -> Option<FusionPlan> {
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom);

    for (block_index, block) in func.blocks.iter().enumerate() {
        for (call_index, inst) in block.insts.iter().enumerate() {
            let (Some(call_result), InstKind::Call { name, args }) = (inst.result, &inst.kind)
            else {
                continue;
            };
            let Some(consumer) = consumers.get(name) else {
                continue;
            };
            if args.len() != 3 {
                continue;
            }
            let lhs_array = args[consumer.lhs_param];
            let rhs_array = args[consumer.rhs_param];
            if !distinct_local_arrays(func, lhs_array, rhs_array) {
                continue;
            }

            let call_block = BlockId(block_index);
            let Some(natural_loop) = loops.loops().iter().find(|natural_loop| {
                natural_loop.exit_edges == [(natural_loop.header, call_block)]
            }) else {
                continue;
            };
            let (Some(preheader), Some(latch)) = (
                natural_loop.dedicated_preheader,
                natural_loop.unique_latch(),
            ) else {
                continue;
            };
            if natural_loop.blocks.len() != 2
                || !matches!(
                    func.blocks[latch.0].terminator,
                    Some(Terminator::Jump(target)) if target == natural_loop.header
                )
            {
                continue;
            }
            let Some(induction) = header_induction(func, natural_loop) else {
                continue;
            };
            if induction.step != 1
                || !matches!(
                    func.value(induction.initial).kind,
                    ValueKind::Const(Const::Int(0) | Const::Zero(Type::I32))
                )
                || canonical_bound(func, natural_loop, induction.phi)?
                    != args[consumer.length_param]
            {
                continue;
            }
            let Some((lhs_value, lhs_store)) =
                unique_indexed_store(func, natural_loop, lhs_array, induction.phi)
            else {
                continue;
            };
            let Some((rhs_value, rhs_store)) =
                unique_indexed_store(func, natural_loop, rhs_array, induction.phi)
            else {
                continue;
            };
            if !producer_has_only_local_stores(
                func,
                natural_loop,
                lhs_array,
                rhs_array,
                induction.phi,
            ) || !array_is_confined_to_producer_and_call(
                func,
                natural_loop,
                lhs_array,
                induction.phi,
                call_block,
                call_index,
            ) || !array_is_confined_to_producer_and_call(
                func,
                natural_loop,
                rhs_array,
                induction.phi,
                call_block,
                call_index,
            ) {
                continue;
            }
            return Some(FusionPlan {
                call_block,
                call_index,
                call_result,
                producer_header: natural_loop.header,
                producer_preheader: preheader,
                producer_latch: latch,
                lhs_value,
                rhs_value,
                lhs_store,
                rhs_store,
                initial: consumer.initial.clone(),
            });
        }
    }
    None
}

fn distinct_local_arrays(func: &Function, lhs: ValueId, rhs: ValueId) -> bool {
    lhs != rhs
        && matches!(
            defining_inst(func, lhs),
            Some(InstKind::Alloca {
                ty: Type::Array { elem, .. }
            }) if elem.as_ref() == &Type::F32
        )
        && matches!(
            defining_inst(func, rhs),
            Some(InstKind::Alloca {
                ty: Type::Array { elem, .. }
            }) if elem.as_ref() == &Type::F32
        )
}

fn header_induction(
    func: &Function,
    natural_loop: &NaturalLoop,
) -> Option<super::loop_analysis::InductionVariable> {
    func.blocks[natural_loop.header.0]
        .insts
        .iter()
        .filter_map(|inst| matches!(inst.kind, InstKind::Phi { .. }).then_some(inst.result?))
        .find_map(|phi| analyze_i32_induction(func, natural_loop, phi))
}

fn unique_indexed_store(
    func: &Function,
    natural_loop: &NaturalLoop,
    base: ValueId,
    induction: ValueId,
) -> Option<(ValueId, (BlockId, usize))> {
    let mut values = natural_loop.blocks.iter().flat_map(|block| {
        func.blocks[block.0]
            .insts
            .iter()
            .enumerate()
            .filter_map(|(inst_index, inst)| {
                let InstKind::Store { ptr, value } = &inst.kind else {
                    return None;
                };
                matches!(
                    defining_inst(func, *ptr),
                    Some(InstKind::Gep { base: gep_base, indices })
                        if *gep_base == base && indices.as_slice() == [induction]
                )
                .then_some((*value, (*block, inst_index)))
            })
    });
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn array_is_confined_to_producer_and_call(
    func: &Function,
    natural_loop: &NaturalLoop,
    array: ValueId,
    induction: ValueId,
    call_block: BlockId,
    call_index: usize,
) -> bool {
    let mut geps = Vec::new();
    for (block_index, block) in func.blocks.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            match &inst.kind {
                InstKind::Gep { base, indices } if *base == array => {
                    if !natural_loop.blocks.contains(&BlockId(block_index))
                        || indices.as_slice() != [induction]
                    {
                        return false;
                    }
                    let Some(result) = inst.result else {
                        return false;
                    };
                    geps.push(result);
                }
                InstKind::Call { args, .. } if args.contains(&array) => {
                    if BlockId(block_index) != call_block || inst_index != call_index {
                        return false;
                    }
                }
                kind if inst_operands(kind).contains(&array) => return false,
                _ => {}
            }
        }
        if terminator_operand(block.terminator.as_ref()) == Some(array) {
            return false;
        }
    }
    !geps.is_empty()
        && geps.into_iter().all(|gep| {
            func.blocks.iter().enumerate().all(|(block_index, block)| {
                block.insts.iter().all(|inst| match &inst.kind {
                    InstKind::Store { ptr, value } if *ptr == gep => {
                        natural_loop.blocks.contains(&BlockId(block_index)) && *value != gep
                    }
                    kind => !inst_operands(kind).contains(&gep),
                }) && terminator_operand(block.terminator.as_ref()) != Some(gep)
            })
        })
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
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

fn terminator_operand(terminator: Option<&Terminator>) -> Option<ValueId> {
    match terminator {
        Some(Terminator::Return(value)) => *value,
        Some(Terminator::Branch { cond, .. }) => Some(*cond),
        Some(Terminator::Jump(_)) | None => None,
    }
}

fn producer_has_only_local_stores(
    func: &Function,
    natural_loop: &NaturalLoop,
    lhs_array: ValueId,
    rhs_array: ValueId,
    induction: ValueId,
) -> bool {
    natural_loop.blocks.iter().all(|block| {
        func.blocks[block.0]
            .insts
            .iter()
            .all(|inst| match &inst.kind {
                InstKind::Call { .. } | InstKind::Load { .. } | InstKind::MemZero { .. } => false,
                InstKind::Store { ptr, .. } => matches!(
                    defining_inst(func, *ptr),
                    Some(InstKind::Gep { base, indices })
                        if (*base == lhs_array || *base == rhs_array)
                            && indices.as_slice() == [induction]
                ),
                _ => true,
            })
    })
}

fn get_or_add_const(func: &mut Function, constant: Const) -> ValueId {
    func.values
        .iter()
        .position(|value| value.kind == ValueKind::Const(constant.clone()))
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(constant))
}

fn apply_fusion(func: &mut Function, plan: FusionPlan) {
    let initial = get_or_add_const(func, plan.initial);
    let phi_index = func.blocks[plan.producer_header.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Phi { .. }))
        .count();
    let accumulator = func
        .insert_inst(
            plan.producer_header,
            phi_index,
            InstKind::Phi {
                incomings: vec![(plan.producer_preheader, initial)],
            },
            Some(Type::F32),
        )
        .expect("fused accumulator phi must produce a value");
    let product = func
        .append_inst(
            plan.producer_latch,
            InstKind::Binary {
                op: BinaryOp::Fmul,
                lhs: plan.lhs_value,
                rhs: plan.rhs_value,
            },
            Some(Type::F32),
        )
        .expect("fused product must produce a value");
    let next = func
        .append_inst(
            plan.producer_latch,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: accumulator,
                rhs: product,
            },
            Some(Type::F32),
        )
        .expect("fused accumulator update must produce a value");
    let InstKind::Phi { incomings } =
        &mut func.blocks[plan.producer_header.0].insts[phi_index].kind
    else {
        unreachable!("inserted accumulator must remain a phi");
    };
    incomings.push((plan.producer_latch, next));

    let call = &mut func.blocks[plan.call_block.0].insts[plan.call_index];
    debug_assert_eq!(call.result, Some(plan.call_result));
    call.result = None;
    call.kind = InstKind::Nop;
    for (block, index) in [plan.lhs_store, plan.rhs_store] {
        let store = &mut func.blocks[block.0].insts[index];
        debug_assert!(matches!(store.kind, InstKind::Store { .. }));
        store.kind = InstKind::Nop;
    }
    rewrite_function_uses(
        func,
        &ValueReplacements::from([(plan.call_result, accumulator)]),
    );

    if let Err(errors) = func.verify() {
        panic!(
            "producer-consumer fusion produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}
