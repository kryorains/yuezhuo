use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_ACTIVE_BODY_INSTS: usize = 16;
const MAX_FACTOR_FOUR_BODY_INSTS: usize = 8;
const MAX_CLONED_INSTS_PER_FUNCTION: usize = 256;
const MAX_FUNCTION_INSTRUCTIONS: usize = 32_768;

/// Unrolls a deliberately narrow class of single-block counted loops.
///
/// The original loop remains as a scalar remainder. A target-configured two-
/// or four-lane fast loop is inserted in front of it, so dynamic tails retain
/// the source program's exact iteration order.
pub(super) struct SimpleLoopUnrollPass {
    factor: usize,
}

impl SimpleLoopUnrollPass {
    pub(super) fn new(factor: usize) -> Self {
        assert!(matches!(factor, 2 | 4));
        Self { factor }
    }
}

impl ModulePass for SimpleLoopUnrollPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            if func.simple_loop_unroll_decided() {
                continue;
            }
            unroll_simple_loops(func, self.factor);
            func.mark_simple_loop_unroll_decided();
        }
    }
}

#[derive(Clone)]
struct UnrollCandidate {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    counter: ValueId,
    counter_initial: ValueId,
    counter_next: ValueId,
    bound: ValueId,
    header_phi_inst: usize,
    accumulator: Option<Accumulator>,
    body_insts: Vec<Inst>,
    active_body_insts: usize,
}

#[derive(Clone)]
struct Accumulator {
    value: ValueId,
    initial: ValueId,
    next: ValueId,
    header_phi_inst: usize,
}

fn unroll_simple_loops(func: &mut Function, factor: usize) {
    let Some(mut instruction_count) = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
    else {
        return;
    };
    if func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || instruction_count > MAX_FUNCTION_INSTRUCTIONS
    {
        return;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let candidates = loop_info
        .loops()
        .iter()
        .filter_map(|natural_loop| match_candidate(func, &cfg, &dom, natural_loop))
        .collect::<Vec<_>>();

    let mut cloned_insts = 0usize;
    let mut projected_values = func.values.len();
    let mut projected_blocks = func.blocks.len();
    let mut changed = false;
    for candidate in candidates {
        if factor == 4
            && (candidate.active_body_insts > MAX_FACTOR_FOUR_BODY_INSTS
                || candidate.body_insts.iter().any(|inst| {
                    matches!(
                        inst.kind,
                        InstKind::Binary {
                            op: BinaryOp::Idiv | BinaryOp::Imod | BinaryOp::Fdiv,
                            ..
                        }
                    )
                }))
        {
            continue;
        }
        let Some(growth) = candidate.active_body_insts.checked_mul(factor) else {
            continue;
        };
        let Some(next_cloned) = cloned_insts.checked_add(growth) else {
            continue;
        };
        // Four control/setup results and at most two newly interned constants
        // accompany every transformed loop. Charging every cloned instruction
        // as value-producing is conservative for stores.
        let Some(value_growth) = growth.checked_add(6) else {
            continue;
        };
        let Some(next_values) = projected_values.checked_add(value_growth) else {
            continue;
        };
        let Some(next_blocks) = projected_blocks.checked_add(2) else {
            continue;
        };
        let Some(next_instructions) = instruction_count
            .checked_add(growth)
            .and_then(|count| count.checked_add(4))
        else {
            continue;
        };
        if next_cloned > MAX_CLONED_INSTS_PER_FUNCTION
            || next_values > MAX_VALUES
            || next_blocks > MAX_BLOCKS
            || next_instructions > MAX_FUNCTION_INSTRUCTIONS
        {
            continue;
        }
        apply_unroll(func, &candidate, factor);
        cloned_insts = next_cloned;
        projected_values = next_values;
        projected_blocks = next_blocks;
        instruction_count = next_instructions;
        changed = true;
    }

    if changed {
        if let Err(errors) = func.verify() {
            panic!(
                "simple loop unrolling produced invalid IR in {}: {:?}",
                func.name, errors
            );
        }
    }
}

fn match_candidate(
    func: &Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
) -> Option<UnrollCandidate> {
    let header_id = natural_loop.header;
    let preheader_id = natural_loop.dedicated_preheader?;
    let latch = natural_loop.unique_latch()?;
    let header = func.blocks.get(header_id.0)?;
    let Terminator::Branch {
        cond,
        then_target: body_id,
        else_target: exit_id,
    } = header.terminator.as_ref()?
    else {
        return None;
    };
    if *body_id != latch
        || body_id == &header_id
        || exit_id == &header_id
        || body_id == exit_id
        || natural_loop.unique_exit() != Some(*exit_id)
    {
        return None;
    }

    let body = func.blocks.get(body_id.0)?;
    if body.terminator != Some(Terminator::Jump(header_id))
        || cfg.preds.get(body_id.0)?.as_slice() != [header_id]
    {
        return None;
    }

    let header_preds = cfg.preds.get(header_id.0)?;
    if header_preds.len() != 2
        || !header_preds.contains(body_id)
        || !header_preds.contains(&preheader_id)
        || func.blocks.get(preheader_id.0)?.terminator != Some(Terminator::Jump(header_id))
    {
        return None;
    }

    let mut phis = Vec::new();
    let mut condition = None;
    for (inst_idx, inst) in header.insts.iter().enumerate() {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { incomings } => {
                phis.push((inst_idx, inst.result?, incomings));
            }
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs,
                rhs,
            } if inst.result == Some(*cond) => {
                if condition.replace((*lhs, *rhs)).is_some() {
                    return None;
                }
            }
            _ => return None,
        }
    }

    let (condition_counter, bound) = condition?;
    let (header_phi_inst, counter, _) = phis
        .iter()
        .find(|(_, result, _)| *result == condition_counter)
        .copied()?;
    if counter != condition_counter
        || func.values.get(counter.0)?.ty != Type::I32
        || func.values.get(bound.0)?.ty != Type::I32
        || func.values.get(cond.0)?.ty != Type::I1
    {
        return None;
    }
    let induction = analyze_i32_induction(func, natural_loop, counter)?;
    if !is_const_i32(func, induction.initial, 0)
        || induction.step != 1
        || !matches!(
            func.values.get(induction.next.0)?.kind,
            ValueKind::Inst(owner, _) if owner == *body_id
        )
        || !value_available_at_preheader(func, dom, bound, preheader_id)
    {
        return None;
    }
    let counter_initial = induction.initial;
    let counter_next = induction.next;
    let mut other_phis = phis.into_iter().filter(|(_, result, _)| *result != counter);
    let accumulator = if let Some((header_phi_inst, value, incomings)) = other_phis.next() {
        if func.values.get(value.0)?.ty != Type::F32 || incomings.len() != 2 {
            return None;
        }
        let initial = incomings
            .iter()
            .find_map(|(pred, value)| (*pred == preheader_id).then_some(*value))?;
        let next = incomings
            .iter()
            .find_map(|(pred, value)| (*pred == *body_id).then_some(*value))?;
        if func.values.get(initial.0)?.ty != Type::F32
            || func.values.get(next.0)?.ty != Type::F32
            || !matches!(
                func.values.get(next.0)?.kind,
                ValueKind::Inst(owner, _) if owner == *body_id
            )
            || !value_available_at_preheader(func, dom, initial, preheader_id)
        {
            return None;
        }
        Some(Accumulator {
            value,
            initial,
            next,
            header_phi_inst,
        })
    } else {
        None
    };
    if other_phis.next().is_some() {
        return None;
    }

    let active_body_insts = body
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count();
    if active_body_insts == 0 || active_body_insts > MAX_ACTIVE_BODY_INSTS {
        return None;
    }
    let mut has_store = false;
    for inst in &body.insts {
        match &inst.kind {
            InstKind::Nop
            | InstKind::Load { .. }
            | InstKind::Unary { .. }
            | InstKind::Binary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. }
            | InstKind::Gep { .. } => {}
            InstKind::Store { .. } => has_store = true,
            InstKind::Phi { .. }
            | InstKind::Alloca { .. }
            | InstKind::MemZero { .. }
            | InstKind::Call { .. } => return None,
        }
        if !matches!(inst.kind, InstKind::Nop | InstKind::Store { .. }) && inst.result.is_none() {
            return None;
        }
    }
    if (!has_store && accumulator.is_none())
        || !body_instructions_are_ordered(
            func,
            *body_id,
            header_id,
            counter,
            accumulator.as_ref().map(|accumulator| accumulator.value),
        )
        || body_values_escape(
            func,
            *body_id,
            header_id,
            counter_next,
            accumulator.as_ref().map(|accumulator| accumulator.next),
        )
    {
        return None;
    }

    Some(UnrollCandidate {
        preheader: preheader_id,
        header: header_id,
        body: *body_id,
        counter,
        counter_initial,
        counter_next,
        bound,
        header_phi_inst,
        accumulator,
        body_insts: body.insts.clone(),
        active_body_insts,
    })
}

fn is_const_i32(func: &Function, value: ValueId, expected: i32) -> bool {
    matches!(
        func.values.get(value.0),
        Some(crate::ir::Value {
            ty: Type::I32,
            kind: ValueKind::Const(Const::Int(actual)),
            ..
        }) if *actual == expected
    )
}

fn value_available_at_preheader(
    func: &Function,
    dom: &Dominators,
    value: ValueId,
    preheader: BlockId,
) -> bool {
    let Some(value) = func.values.get(value.0) else {
        return false;
    };
    match &value.kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(block, inst_idx) => {
            func.blocks
                .get(block.0)
                .and_then(|owner| owner.insts.get(*inst_idx))
                .is_some()
                && dom.dominates(*block, preheader)
        }
    }
}

fn body_instructions_are_ordered(
    func: &Function,
    body: BlockId,
    header: BlockId,
    counter: ValueId,
    accumulator: Option<ValueId>,
) -> bool {
    let mut defined = HashSet::new();
    for inst in &func.blocks[body.0].insts {
        if inst_operands(&inst.kind).into_iter().any(|operand| {
            matches!(
                func.values.get(operand.0).map(|value| &value.kind),
                Some(ValueKind::Inst(owner, _))
                    if (*owner == body && !defined.contains(&operand))
                        || (*owner == header
                            && operand != counter
                            && Some(operand) != accumulator)
            )
        }) {
            return false;
        }
        if let Some(result) = inst.result {
            defined.insert(result);
        }
    }
    true
}

fn body_values_escape(
    func: &Function,
    body: BlockId,
    header: BlockId,
    counter_next: ValueId,
    accumulator_next: Option<ValueId>,
) -> bool {
    let body_values = func.blocks[body.0]
        .insts
        .iter()
        .filter_map(|inst| inst.result)
        .collect::<HashSet<_>>();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        if block_idx == body.0 {
            continue;
        }
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings } if block_idx == header.0 => {
                    for (pred, value) in incomings {
                        if body_values.contains(value)
                            && !(*pred == body
                                && (*value == counter_next || Some(*value) == accumulator_next))
                        {
                            return true;
                        }
                    }
                }
                kind => {
                    if inst_operands(kind)
                        .iter()
                        .any(|value| body_values.contains(value))
                    {
                        return true;
                    }
                }
            }
        }
        if terminator_operands(block.terminator.as_ref())
            .iter()
            .any(|value| body_values.contains(value))
        {
            return true;
        }
    }
    false
}

fn apply_unroll(func: &mut Function, candidate: &UnrollCandidate, factor: usize) {
    let cutoff_delta = get_or_add_i32_const(func, (factor - 1) as i32);
    let minimum_bound = get_or_add_i32_const(func, factor as i32);
    let cutoff = func
        .append_inst(
            candidate.preheader,
            InstKind::Binary {
                op: BinaryOp::Isub,
                lhs: candidate.bound,
                rhs: cutoff_delta,
            },
            Some(Type::I32),
        )
        .unwrap();
    let use_fast_loop = func
        .append_inst(
            candidate.preheader,
            InstKind::Icmp {
                op: CmpOp::Ge,
                lhs: candidate.bound,
                rhs: minimum_bound,
            },
            Some(Type::I1),
        )
        .unwrap();

    let fast_header = func.add_block(format!("unroll{}.header.{}", factor, candidate.header.0));
    let fast_body = func.add_block(format!("unroll{}.body.{}", factor, candidate.header.0));
    let fast_counter = func
        .append_inst(
            fast_header,
            InstKind::Phi {
                incomings: vec![
                    (candidate.preheader, candidate.counter_initial),
                    (fast_body, candidate.counter_initial),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let fast_accumulator = candidate.accumulator.as_ref().map(|accumulator| {
        func.append_inst(
            fast_header,
            InstKind::Phi {
                incomings: vec![
                    (candidate.preheader, accumulator.initial),
                    (fast_body, accumulator.initial),
                ],
            },
            Some(Type::F32),
        )
        .expect("fast-loop accumulator must produce a value")
    });
    let run_batch = func
        .append_inst(
            fast_header,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: fast_counter,
                rhs: cutoff,
            },
            Some(Type::I1),
        )
        .unwrap();

    let mut current_counter = fast_counter;
    let mut current_accumulator = fast_accumulator;
    for _ in 0..factor {
        (current_counter, current_accumulator) = clone_body_lane(
            func,
            fast_body,
            candidate,
            current_counter,
            current_accumulator,
        );
    }

    func.blocks[candidate.preheader.0].terminator = Some(Terminator::Branch {
        cond: use_fast_loop,
        then_target: fast_header,
        else_target: candidate.header,
    });
    func.set_terminator(
        fast_header,
        Terminator::Branch {
            cond: run_batch,
            then_target: fast_body,
            else_target: candidate.header,
        },
    );
    func.set_terminator(fast_body, Terminator::Jump(fast_header));

    let InstKind::Phi { incomings } = &mut func.blocks[fast_header.0].insts[0].kind else {
        unreachable!("new fast-loop counter must be a phi");
    };
    incomings[1].1 = current_counter;
    if let (Some(accumulator), Some(current_accumulator)) =
        (&candidate.accumulator, current_accumulator)
    {
        let InstKind::Phi { incomings } = &mut func.blocks[fast_header.0].insts[1].kind else {
            unreachable!("new fast-loop accumulator must be a phi");
        };
        incomings[1].1 = current_accumulator;

        let InstKind::Phi { incomings } =
            &mut func.blocks[candidate.header.0].insts[accumulator.header_phi_inst].kind
        else {
            unreachable!("matched loop accumulator must remain a phi");
        };
        incomings.push((
            fast_header,
            fast_accumulator.expect("candidate accumulator needs a fast-loop phi"),
        ));
    }

    let InstKind::Phi { incomings } =
        &mut func.blocks[candidate.header.0].insts[candidate.header_phi_inst].kind
    else {
        unreachable!("matched loop counter must remain a phi");
    };
    incomings.push((fast_header, fast_counter));

    debug_assert!(matches!(
        func.blocks[candidate.body.0].terminator.as_ref(),
        Some(Terminator::Jump(target)) if *target == candidate.header
    ));
}

fn clone_body_lane(
    func: &mut Function,
    destination: BlockId,
    candidate: &UnrollCandidate,
    current_counter: ValueId,
    current_accumulator: Option<ValueId>,
) -> (ValueId, Option<ValueId>) {
    let mut values = HashMap::from([(candidate.counter, current_counter)]);
    if let (Some(accumulator), Some(current)) = (&candidate.accumulator, current_accumulator) {
        values.insert(accumulator.value, current);
    }
    for inst in &candidate.body_insts {
        if matches!(inst.kind, InstKind::Nop) {
            continue;
        }
        let kind = remap_inst(&inst.kind, &values);
        let result_ty = inst.result.map(|result| func.values[result.0].ty.clone());
        let cloned_result = func.append_inst(destination, kind, result_ty);
        if let Some(original) = inst.result {
            values.insert(
                original,
                cloned_result.expect("validated value-producing instruction lost its result"),
            );
        }
    }
    (
        values[&candidate.counter_next],
        candidate
            .accumulator
            .as_ref()
            .map(|accumulator| values[&accumulator.next]),
    )
}

fn remap_inst(kind: &InstKind, values: &HashMap<ValueId, ValueId>) -> InstKind {
    let map = |value: ValueId| values.get(&value).copied().unwrap_or(value);
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
            indices: indices.iter().map(|value| map(*value)).collect(),
        },
        InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::MemZero { .. }
        | InstKind::Call { .. } => unreachable!("unsupported instruction passed unroll matching"),
    }
}

fn get_or_add_i32_const(func: &mut Function, expected: i32) -> ValueId {
    func.values
        .iter()
        .position(|value| {
            value.ty == Type::I32
                && matches!(&value.kind, ValueKind::Const(Const::Int(actual)) if *actual == expected)
        })
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(Const::Int(expected)))
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

fn terminator_operands(terminator: Option<&Terminator>) -> Vec<ValueId> {
    match terminator {
        Some(Terminator::Return(Some(value))) => vec![*value],
        Some(Terminator::Branch { cond, .. }) => vec![*cond],
        Some(Terminator::Return(None) | Terminator::Jump(_)) | None => Vec::new(),
    }
}
