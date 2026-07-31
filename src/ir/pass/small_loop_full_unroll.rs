use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::simplify_cfg::SimplifyCfgPass;
use super::util::{const_i32, resolve_replacement, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BlockId, CmpOp, Function, Inst, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::HashSet;

const MAX_TRIP_COUNT: i32 = 16;
const MAX_BODY_INSTRUCTIONS: usize = 16;
const MAX_CLONED_INSTRUCTIONS_PER_FUNCTION: usize = 256;
const MAX_FUNCTION_BLOCKS: usize = 1024;
const MAX_FUNCTION_VALUES: usize = 8192;
const MAX_FUNCTION_INSTRUCTIONS: usize = 32_768;

/// Fully unrolls canonical single-block loops with a small constant trip count.
///
/// This is ordinary partial evaluation: every original iteration is cloned in
/// source order into the preheader, and the original loop becomes unreachable.
/// It does not inspect function names, called functions, constants used by the
/// body, or the source-level purpose of the loop.
pub(super) struct SmallLoopFullUnrollPass;

impl SmallLoopFullUnrollPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for SmallLoopFullUnrollPass {
    fn run(&mut self, module: &mut Module) {
        let mut changed = false;
        for func in &mut module.funcs {
            changed |= fully_unroll_small_loop(func);
        }
        if changed {
            SimplifyCfgPass::preserving_loop_preheaders().run(module);
            for func in &module.funcs {
                if let Err(errors) = func.verify() {
                    panic!(
                        "small constant-trip loop unrolling produced invalid IR in {}: {:?}",
                        func.name, errors
                    );
                }
            }
        }
    }
}

#[derive(Clone)]
struct LoopPhi {
    value: ValueId,
    initial: ValueId,
    next: ValueId,
    inst_index: usize,
}

#[derive(Clone)]
struct Candidate {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    trip_count: usize,
    phis: Vec<LoopPhi>,
    body_insts: Vec<Inst>,
}

fn fully_unroll_small_loop(func: &mut Function) -> bool {
    let Some(instruction_count) = instruction_count(func) else {
        return false;
    };
    if func.blocks.len() > MAX_FUNCTION_BLOCKS
        || func.values.len() > MAX_FUNCTION_VALUES
        || instruction_count > MAX_FUNCTION_INSTRUCTIONS
    {
        return false;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom);
    let Some(candidate) = loops
        .loops()
        .iter()
        .filter_map(|natural_loop| match_candidate(func, natural_loop))
        .min_by_key(|candidate| candidate.trip_count)
    else {
        return false;
    };
    let active_body_instructions = candidate
        .body_insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count();
    let Some(growth) = active_body_instructions.checked_mul(candidate.trip_count) else {
        return false;
    };
    if growth > MAX_CLONED_INSTRUCTIONS_PER_FUNCTION
        || func
            .values
            .len()
            .checked_add(growth)
            .is_none_or(|count| count > MAX_FUNCTION_VALUES)
        || instruction_count
            .checked_add(growth)
            .is_none_or(|count| count > MAX_FUNCTION_INSTRUCTIONS)
    {
        return false;
    }

    apply_candidate(func, &candidate);
    true
}

fn instruction_count(func: &Function) -> Option<usize> {
    func.blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
}

fn match_candidate(func: &Function, natural_loop: &NaturalLoop) -> Option<Candidate> {
    let preheader = natural_loop.dedicated_preheader?;
    let body = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    let header = natural_loop.header;
    if natural_loop.blocks != HashSet::from([header, body])
        || natural_loop.exit_edges != [(header, exit)]
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

    let header_block = func.blocks.get(header.0)?;
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = header_block.terminator.as_ref()?
    else {
        return None;
    };
    if *then_target != body || *else_target != exit {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt,
        lhs: counter,
        rhs: bound,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    let induction = analyze_i32_induction(func, natural_loop, *counter)?;
    let initial = const_i32(func, induction.initial)?;
    let trip_count = const_i32(func, *bound)?.checked_sub(initial)?;
    if induction.phi != *counter
        || induction.step != 1
        || !(0..=MAX_TRIP_COUNT).contains(&trip_count)
    {
        return None;
    }

    let mut phis = Vec::new();
    let mut comparison_count = 0usize;
    for (inst_index, inst) in header_block.insts.iter().enumerate() {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { incomings } => {
                let value = inst.result?;
                if incomings.len() != 2
                    || !matches!(
                        func.value(value).ty,
                        Type::I1 | Type::I32 | Type::F32 | Type::Ptr(_)
                    )
                {
                    return None;
                }
                let initial = incomings
                    .iter()
                    .find_map(|(pred, value)| (*pred == preheader).then_some(*value))?;
                let next = incomings
                    .iter()
                    .find_map(|(pred, value)| (*pred == body).then_some(*value))?;
                if func.value(initial).ty != func.value(value).ty
                    || func.value(next).ty != func.value(value).ty
                {
                    return None;
                }
                phis.push(LoopPhi {
                    value,
                    initial,
                    next,
                    inst_index,
                });
            }
            InstKind::Icmp { .. } if inst.result == Some(*cond) => {
                comparison_count += 1;
            }
            _ => return None,
        }
    }
    if comparison_count != 1 || !phis.iter().any(|phi| phi.value == *counter) {
        return None;
    }

    let body_insts = func.blocks.get(body.0)?.insts.clone();
    let active_body_instructions = body_insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count();
    if active_body_instructions == 0
        || active_body_instructions > MAX_BODY_INSTRUCTIONS
        || (trip_count > 8
            && body_insts.iter().any(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Call { .. }
                        | InstKind::Binary {
                            op: crate::ir::BinaryOp::Idiv
                                | crate::ir::BinaryOp::Imod
                                | crate::ir::BinaryOp::Fdiv,
                            ..
                        }
                )
            }))
        || body_insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::Phi { .. } | InstKind::Alloca { .. } | InstKind::MemZero { .. }
            )
        })
    {
        return None;
    }

    let phi_values = phis.iter().map(|phi| phi.value).collect::<HashSet<_>>();
    let mut defined = HashSet::new();
    for inst in &body_insts {
        if inst_operands(&inst.kind).into_iter().any(|operand| {
            matches!(
                func.value(operand).kind,
                ValueKind::Inst(owner, _)
                    if owner == body && !defined.contains(&operand)
                        || owner == header && !phi_values.contains(&operand)
            )
        }) {
            return None;
        }
        if let Some(result) = inst.result {
            defined.insert(result);
        }
    }
    if phis
        .iter()
        .any(|phi| !defined.contains(&phi.next) && !phi_values.contains(&phi.next))
    {
        return None;
    }
    if !exit_phi_values_are_mappable(func, exit, header, &phi_values) {
        return None;
    }

    Some(Candidate {
        preheader,
        header,
        body,
        exit,
        trip_count: trip_count as usize,
        phis,
        body_insts,
    })
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_index) = func.value(value).kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_index)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

fn exit_phi_values_are_mappable(
    func: &Function,
    exit: BlockId,
    header: BlockId,
    phi_values: &HashSet<ValueId>,
) -> bool {
    func.blocks[exit.0].insts.iter().all(|inst| {
        let InstKind::Phi { incomings } = &inst.kind else {
            return true;
        };
        incomings.iter().all(|(pred, value)| {
            if *pred != header {
                return true;
            }
            phi_values.contains(value)
                || !matches!(
                    func.value(*value).kind,
                    ValueKind::Inst(owner, _) if owner == header
                )
        })
    })
}

fn apply_candidate(func: &mut Function, candidate: &Candidate) {
    let mut current = candidate
        .phis
        .iter()
        .map(|phi| (phi.value, phi.initial))
        .collect::<ValueReplacements>();

    for _ in 0..candidate.trip_count {
        let mut iteration = current.clone();
        for inst in &candidate.body_insts {
            if matches!(inst.kind, InstKind::Nop) {
                continue;
            }
            let kind = remap_inst(&inst.kind, &iteration);
            let result_ty = inst.result.map(|result| func.value(result).ty.clone());
            let cloned = func.append_inst(candidate.preheader, kind, result_ty);
            if let Some(original) = inst.result {
                iteration.insert(
                    original,
                    cloned.expect("validated value-producing instruction lost its result"),
                );
            }
        }
        current = candidate
            .phis
            .iter()
            .map(|phi| (phi.value, resolve_replacement(phi.next, &iteration)))
            .collect();
    }

    add_preheader_exit_phi_incomings(func, candidate, &current);
    rewrite_uses_outside_original_loop(func, candidate, &current);
    for phi in &candidate.phis {
        let InstKind::Phi { incomings } =
            &mut func.blocks[candidate.header.0].insts[phi.inst_index].kind
        else {
            unreachable!("matched header phi changed before full unroll");
        };
        incomings.retain(|(pred, _)| *pred != candidate.preheader);
    }
    func.blocks[candidate.preheader.0].terminator = Some(Terminator::Jump(candidate.exit));
}

fn add_preheader_exit_phi_incomings(
    func: &mut Function,
    candidate: &Candidate,
    replacements: &ValueReplacements,
) {
    for inst in &mut func.blocks[candidate.exit.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        let Some((_, value)) = incomings
            .iter()
            .find(|(pred, _)| *pred == candidate.header)
            .copied()
        else {
            continue;
        };
        incomings.push((
            candidate.preheader,
            resolve_replacement(value, replacements),
        ));
    }
}

fn rewrite_uses_outside_original_loop(
    func: &mut Function,
    candidate: &Candidate,
    replacements: &ValueReplacements,
) {
    let original_loop = HashSet::from([candidate.header, candidate.body]);
    for (block_index, block) in func.blocks.iter_mut().enumerate() {
        let block_id = BlockId(block_index);
        if original_loop.contains(&block_id) {
            continue;
        }
        for inst in &mut block.insts {
            match &mut inst.kind {
                InstKind::Phi { incomings } => {
                    for (pred, value) in incomings {
                        if !original_loop.contains(pred) {
                            rewrite_value(value, replacements);
                        }
                    }
                }
                kind => rewrite_inst_operands(kind, replacements),
            }
        }
        if let Some(terminator) = &mut block.terminator {
            rewrite_terminator_operands(terminator, replacements);
        }
    }
}

fn rewrite_value(value: &mut ValueId, replacements: &ValueReplacements) {
    *value = resolve_replacement(*value, replacements);
}

fn rewrite_inst_operands(kind: &mut InstKind, replacements: &ValueReplacements) {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => {}
        InstKind::Phi { .. } => unreachable!("phis are rewritten with edge context"),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => {
            rewrite_value(ptr, replacements);
        }
        InstKind::Store { ptr, value } => {
            rewrite_value(ptr, replacements);
            rewrite_value(value, replacements);
        }
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => {
            rewrite_value(value, replacements);
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            rewrite_value(lhs, replacements);
            rewrite_value(rhs, replacements);
        }
        InstKind::Gep { base, indices } => {
            rewrite_value(base, replacements);
            for index in indices {
                rewrite_value(index, replacements);
            }
        }
        InstKind::Call { args, .. } => {
            for arg in args {
                rewrite_value(arg, replacements);
            }
        }
    }
}

fn rewrite_terminator_operands(terminator: &mut Terminator, replacements: &ValueReplacements) {
    match terminator {
        Terminator::Return(Some(value)) | Terminator::Branch { cond: value, .. } => {
            rewrite_value(value, replacements);
        }
        Terminator::Return(None) | Terminator::Jump(_) => {}
    }
}

fn remap_inst(kind: &InstKind, replacements: &ValueReplacements) -> InstKind {
    let map = |value| resolve_replacement(value, replacements);
    match kind {
        InstKind::Nop => InstKind::Nop,
        InstKind::Load { ptr } => InstKind::Load { ptr: map(*ptr) },
        InstKind::Store { ptr, value } => InstKind::Store {
            ptr: map(*ptr),
            value: map(*value),
        },
        InstKind::Unary { op, value } => InstKind::Unary {
            op: *op,
            value: map(*value),
        },
        InstKind::Binary { op, lhs, rhs } => InstKind::Binary {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Icmp { op, lhs, rhs } => InstKind::Icmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Fcmp { op, lhs, rhs } => InstKind::Fcmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Cast { op, value } => InstKind::Cast {
            op: *op,
            value: map(*value),
        },
        InstKind::Gep { base, indices } => InstKind::Gep {
            base: map(*base),
            indices: indices.iter().map(|index| map(*index)).collect(),
        },
        InstKind::Call { name, args } => InstKind::Call {
            name: name.clone(),
            args: args.iter().map(|arg| map(*arg)).collect(),
        },
        InstKind::Phi { .. } | InstKind::Alloca { .. } | InstKind::MemZero { .. } => {
            unreachable!("unsupported instruction passed full-unroll matching")
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Const};

    #[test]
    fn unrolls_small_constant_loops_independently_of_names_and_trip_counts() {
        let mut module = Module::new();
        module.add_func(canonical_sum_loop("alpha", 3));
        module.add_func(canonical_sum_loop("unrelated_worker", 5));

        SmallLoopFullUnrollPass::new().run(&mut module);

        for func in &module.funcs {
            assert!(func.verify().is_ok());
            assert!(func.blocks.iter().all(|block| {
                block
                    .insts
                    .iter()
                    .all(|inst| !matches!(inst.kind, InstKind::Phi { .. }))
            }));
            assert!(func
                .blocks
                .iter()
                .all(|block| { !matches!(block.terminator, Some(Terminator::Branch { .. })) }));
        }
    }

    #[test]
    fn unrolls_a_small_constant_loop_with_a_nonzero_initial_value() {
        let mut func = canonical_sum_loop("offset_loop", 16);
        let three = func.add_const(Const::Int(3));
        let header = BlockId(1);
        let InstKind::Phi { incomings } = &mut func.blocks[header.0].insts[0].kind else {
            unreachable!();
        };
        incomings[0].1 = three;
        let mut module = Module::new();
        module.add_func(func);

        SmallLoopFullUnrollPass::new().run(&mut module);

        let unrolled = &module.funcs[0];
        assert!(unrolled.verify().is_ok());
        assert!(unrolled
            .blocks
            .iter()
            .all(|block| !matches!(block.terminator, Some(Terminator::Branch { .. }))));
    }

    #[test]
    fn keeps_long_small_loop_with_expensive_division_instruction() {
        let mut func = canonical_sum_loop("costly", 14);
        let three = func.add_const(Const::Int(3));
        func.blocks[BlockId(2).0].insts[0].kind = InstKind::Binary {
            op: BinaryOp::Imod,
            lhs: func.blocks[BlockId(1).0].insts[1].result.unwrap(),
            rhs: three,
        };
        let mut module = Module::new();
        module.add_func(func);

        SmallLoopFullUnrollPass::new().run(&mut module);

        assert!(module.funcs[0]
            .blocks
            .iter()
            .any(|block| matches!(block.terminator, Some(Terminator::Branch { .. }))));
        assert!(module.funcs[0].verify().is_ok());
    }

    fn canonical_sum_loop(name: &str, trip_count: i32) -> Function {
        let mut func = Function::new(name, Type::I32);
        let header = func.add_block("loop.header");
        let body = func.add_block("loop.body");
        let exit = func.add_block("loop.exit");
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let bound = func.add_const(Const::Int(trip_count));

        func.set_terminator(func.entry, Terminator::Jump(header));
        let counter = func
            .append_inst(header, InstKind::Phi { incomings: vec![] }, Some(Type::I32))
            .unwrap();
        let sum = func
            .append_inst(header, InstKind::Phi { incomings: vec![] }, Some(Type::I32))
            .unwrap();
        let condition = func
            .append_inst(
                header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: counter,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: body,
                else_target: exit,
            },
        );

        let next_sum = func
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: sum,
                    rhs: counter,
                },
                Some(Type::I32),
            )
            .unwrap();
        let next_counter = func
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: counter,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(body, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(Some(sum)));

        func.blocks[header.0].insts[0].kind = InstKind::Phi {
            incomings: vec![(func.entry, zero), (body, next_counter)],
        };
        func.blocks[header.0].insts[1].kind = InstKind::Phi {
            incomings: vec![(func.entry, zero), (body, next_sum)],
        };
        assert!(func.verify().is_ok());
        func
    }
}
