use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{
    analyze_const_i32_trip_count, analyze_i32_induction, LoopInfo, NaturalLoop,
};
use super::util::{const_i32, defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CastOp, CmpOp, Const, Function, InstKind, Module, Terminator, Type, UnaryOp,
    ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_TRANSFORMS_PER_FUNCTION: usize = 64;
const I32_BITS: u32 = i32::BITS;

/// Versions one proven natural-loop region with a native integer-bitwise fast
/// path. The original loop remains as the fallback for negative inputs, whose
/// signed division recurrence is not equivalent to native bitwise operations.
pub(super) struct LoopIdiomPass;

impl LoopIdiomPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LoopIdiomPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            transform_loop_idioms(func);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    preheader: BlockId,
    exiting: BlockId,
    exit: BlockId,
    lhs: ValueId,
    rhs: ValueId,
    accumulator: ValueId,
    iterations: u32,
    truth_table: u8,
}

#[derive(Clone, Copy)]
struct Recurrences {
    body_entry: BlockId,
    lhs: ValueId,
    rhs: ValueId,
    lhs_bit: ValueId,
    rhs_bit: ValueId,
    power: ValueId,
    accumulator: ValueId,
    accumulator_next: ValueId,
    iterations: u32,
}

fn transform_loop_idioms(func: &mut Function) {
    if func.blocks.len() > MAX_BLOCKS || func.values.len() > MAX_VALUES {
        return;
    }

    // Keep the compile-time budget deterministic and pass-idempotent: if a
    // function exceeds the regional transform budget, leave all of its loops
    // untouched instead of transforming another batch on every invocation.
    let candidate_count = {
        let cfg = ControlFlowGraph::new(func);
        let dom = Dominators::new(func, &cfg);
        let loop_info = LoopInfo::new(&cfg, &dom);
        loop_info
            .loops()
            .iter()
            .filter(|natural_loop| match_candidate(func, &cfg, &dom, natural_loop).is_some())
            .take(MAX_TRANSFORMS_PER_FUNCTION + 1)
            .count()
    };
    if candidate_count > MAX_TRANSFORMS_PER_FUNCTION {
        return;
    }

    let mut transformed = 0;
    while transformed < candidate_count {
        let candidate = {
            let cfg = ControlFlowGraph::new(func);
            let dom = Dominators::new(func, &cfg);
            let loop_info = LoopInfo::new(&cfg, &dom);
            loop_info
                .loops()
                .iter()
                .find_map(|natural_loop| match_candidate(func, &cfg, &dom, natural_loop))
        };
        let Some(candidate) = candidate else {
            break;
        };
        install_fast_path(func, candidate);
        transformed += 1;
    }

    if transformed != 0 {
        if let Err(errors) = func.verify() {
            panic!(
                "loop idiom transformation produced invalid IR in {}: {:?}",
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
) -> Option<Candidate> {
    let preheader = natural_loop.dedicated_preheader?;
    let latch = natural_loop.unique_latch()?;
    let [(exiting, exit)] = natural_loop.exit_edges.as_slice() else {
        return None;
    };
    if *exiting != natural_loop.header
        || natural_loop.unique_exit() != Some(*exit)
        || cfg.succs.get(latch.0)?.as_slice() != [natural_loop.header]
        || cfg.preds.get(natural_loop.header.0)?.len() != 2
        || !cfg.preds[natural_loop.header.0].contains(&preheader)
        || !cfg.preds[natural_loop.header.0].contains(&latch)
        || cfg.preds.get(exit.0)?.as_slice() != [*exiting]
        || func.blocks.get(preheader.0)?.terminator != Some(Terminator::Jump(natural_loop.header))
        || !loop_is_pure_scalar(func, natural_loop)
    {
        return None;
    }

    let Terminator::Branch {
        then_target,
        else_target,
        ..
    } = func
        .blocks
        .get(natural_loop.header.0)?
        .terminator
        .as_ref()?
    else {
        return None;
    };
    let body_entry = match (
        natural_loop.blocks.contains(then_target),
        natural_loop.blocks.contains(else_target),
    ) {
        (true, false) if *else_target == *exit => *then_target,
        (false, true) if *then_target == *exit => *else_target,
        _ => return None,
    };

    let recurrences = match_recurrences(func, natural_loop, preheader, latch, body_entry)?;
    if func.values.get(recurrences.lhs.0)?.ty != Type::I32
        || func.values.get(recurrences.rhs.0)?.ty != Type::I32
        || !value_available_at_preheader(func, dom, recurrences.lhs, preheader)
        || !value_available_at_preheader(func, dom, recurrences.rhs, preheader)
        || !has_only_accumulator_liveout(func, natural_loop, recurrences.accumulator)
        || !exit_phis_are_safe(func, dom, *exit, *exiting, preheader)
    {
        return None;
    }
    let truth_table = derive_truth_table(func, natural_loop, recurrences)?;

    Some(Candidate {
        preheader,
        exiting: *exiting,
        exit: *exit,
        lhs: recurrences.lhs,
        rhs: recurrences.rhs,
        accumulator: recurrences.accumulator,
        iterations: recurrences.iterations,
        truth_table,
    })
}

fn loop_is_pure_scalar(func: &Function, natural_loop: &NaturalLoop) -> bool {
    natural_loop.blocks.iter().all(|block| {
        let block = &func.blocks[block.0];
        !matches!(block.terminator, Some(Terminator::Return(_)))
            && block.insts.iter().all(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Nop
                        | InstKind::Phi { .. }
                        | InstKind::Unary { .. }
                        | InstKind::Binary { .. }
                        | InstKind::Icmp { .. }
                        | InstKind::Cast { .. }
                )
            })
    })
}

fn match_recurrences(
    func: &Function,
    natural_loop: &NaturalLoop,
    preheader: BlockId,
    latch: BlockId,
    body_entry: BlockId,
) -> Option<Recurrences> {
    let header_phis = func.blocks[natural_loop.header.0]
        .insts
        .iter()
        .filter_map(|inst| {
            matches!(inst.kind, InstKind::Phi { .. })
                .then_some(inst.result)
                .flatten()
        })
        .collect::<Vec<_>>();
    if header_phis.len() != 5 {
        return None;
    }

    let mut count = None;
    let mut power = None;
    let mut accumulator = None;
    let mut inputs = Vec::new();
    for phi in header_phis {
        let (initial, next) = phi_recurrence(func, phi, preheader, latch)?;

        if let Some(induction) = analyze_i32_induction(func, natural_loop, phi) {
            if let Some(iterations) = analyze_const_i32_trip_count(func, natural_loop, induction)
                .filter(|iterations| (1..=I32_BITS).contains(iterations))
            {
                if count.replace(iterations).is_some() {
                    return None;
                }
                continue;
            }
        }
        if is_const_i32(func, initial, 1) && is_multiply_two(func, next, phi) {
            if power.replace(phi).is_some() {
                return None;
            }
            continue;
        }
        if is_divide_two(func, next, phi) {
            if func.values.get(initial.0)?.ty != Type::I32 {
                return None;
            }
            inputs.push((phi, initial));
            continue;
        }
        if is_const_i32(func, initial, 0) {
            if accumulator.replace((phi, next)).is_some() {
                return None;
            }
            continue;
        }
        return None;
    }

    let iterations = count?;
    let power = power?;
    let (accumulator, accumulator_next) = accumulator?;
    let [(lhs_phi, lhs), (rhs_phi, rhs)] = inputs.as_slice() else {
        return None;
    };
    if lhs_phi == rhs_phi || func.values.get(accumulator.0)?.ty != Type::I32 {
        return None;
    }
    let lhs_bit = find_modulo_two(func, &natural_loop.blocks, *lhs_phi)?;
    let rhs_bit = find_modulo_two(func, &natural_loop.blocks, *rhs_phi)?;

    Some(Recurrences {
        body_entry,
        lhs: *lhs,
        rhs: *rhs,
        lhs_bit,
        rhs_bit,
        power,
        accumulator,
        accumulator_next,
        iterations,
    })
}

fn phi_recurrence(
    func: &Function,
    phi: ValueId,
    preheader: BlockId,
    latch: BlockId,
) -> Option<(ValueId, ValueId)> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    Some((
        unique_phi_incoming(incomings, preheader)?,
        unique_phi_incoming(incomings, latch)?,
    ))
}

fn unique_phi_incoming(incomings: &[(BlockId, ValueId)], pred: BlockId) -> Option<ValueId> {
    let mut values = incomings
        .iter()
        .filter_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value));
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn find_modulo_two(func: &Function, blocks: &HashSet<BlockId>, input: ValueId) -> Option<ValueId> {
    let mut matches = blocks
        .iter()
        .flat_map(|block| &func.blocks[block.0].insts)
        .filter_map(|inst| match (&inst.kind, inst.result) {
            (
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs,
                    rhs,
                },
                Some(result),
            ) if *lhs == input && is_const_i32(func, *rhs, 2) => Some(result),
            _ => None,
        });
    let result = matches.next()?;
    matches.next().is_none().then_some(result)
}

fn value_available_at_preheader(
    func: &Function,
    dom: &Dominators,
    value: ValueId,
    preheader: BlockId,
) -> bool {
    let Some(value_data) = func.values.get(value.0) else {
        return false;
    };
    match &value_data.kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(block, inst_idx) => {
            func.blocks
                .get(block.0)
                .and_then(|owner| owner.insts.get(*inst_idx))
                .is_some_and(|inst| inst.result == Some(value))
                && dom.dominates(*block, preheader)
        }
    }
}

fn has_only_accumulator_liveout(
    func: &Function,
    natural_loop: &NaturalLoop,
    accumulator: ValueId,
) -> bool {
    let mut accumulator_is_live = false;
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let block_id = BlockId(block_idx);
        if natural_loop.blocks.contains(&block_id) {
            continue;
        }
        for inst in &block.insts {
            if let InstKind::Phi { incomings } = &inst.kind {
                if incomings
                    .iter()
                    .any(|(_, value)| value_defined_in_loop(func, natural_loop, *value))
                {
                    return false;
                }
                continue;
            }
            for operand in inst_operands(&inst.kind) {
                if !value_defined_in_loop(func, natural_loop, operand) {
                    continue;
                }
                if operand != accumulator {
                    return false;
                }
                accumulator_is_live = true;
            }
        }
        for operand in terminator_operands(block.terminator.as_ref()) {
            if !value_defined_in_loop(func, natural_loop, operand) {
                continue;
            }
            if operand != accumulator {
                return false;
            }
            accumulator_is_live = true;
        }
    }
    accumulator_is_live
}

fn value_defined_in_loop(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Inst(block, _)) if natural_loop.blocks.contains(block)
    )
}

fn exit_phis_are_safe(
    func: &Function,
    dom: &Dominators,
    exit: BlockId,
    exiting: BlockId,
    preheader: BlockId,
) -> bool {
    func.blocks[exit.0].insts.iter().all(|inst| {
        let InstKind::Phi { incomings } = &inst.kind else {
            return true;
        };
        incomings.len() == 1
            && unique_phi_incoming(incomings, exiting)
                .is_some_and(|value| value_available_at_preheader(func, dom, value, preheader))
    })
}

fn derive_truth_table(
    func: &Function,
    natural_loop: &NaturalLoop,
    recurrences: Recurrences,
) -> Option<u8> {
    let mut table = 0u8;
    for lhs_bit in 0..=1i32 {
        for rhs_bit in 0..=1i32 {
            let expression = evaluate_iteration(func, natural_loop, recurrences, lhs_bit, rhs_bit)?;
            let output = match expression {
                Linear {
                    accumulator: 1,
                    power: 0,
                    constant: 0,
                } => false,
                Linear {
                    accumulator: 1,
                    power: 1,
                    constant: 0,
                } => true,
                _ => return None,
            };
            if output {
                table |= 1 << ((lhs_bit << 1) | rhs_bit);
            }
        }
    }
    Some(table)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Linear {
    accumulator: i32,
    power: i32,
    constant: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Symbolic {
    Int(i32),
    Bool(bool),
    Linear(Linear),
    Unknown,
}

fn evaluate_iteration(
    func: &Function,
    natural_loop: &NaturalLoop,
    recurrences: Recurrences,
    lhs_bit: i32,
    rhs_bit: i32,
) -> Option<Linear> {
    let mut values = HashMap::from([
        (
            recurrences.accumulator,
            Symbolic::Linear(Linear {
                accumulator: 1,
                power: 0,
                constant: 0,
            }),
        ),
        (
            recurrences.power,
            Symbolic::Linear(Linear {
                accumulator: 0,
                power: 1,
                constant: 0,
            }),
        ),
        (recurrences.lhs_bit, Symbolic::Int(lhs_bit)),
        (recurrences.rhs_bit, Symbolic::Int(rhs_bit)),
    ]);
    let mut current = recurrences.body_entry;
    let mut predecessor = natural_loop.header;
    let mut visited = HashSet::new();

    loop {
        if !natural_loop.blocks.contains(&current)
            || current == natural_loop.header
            || !visited.insert(current)
        {
            return None;
        }
        let block = &func.blocks[current.0];
        for inst in &block.insts {
            let Some(result) = inst.result else {
                continue;
            };
            if result == recurrences.lhs_bit || result == recurrences.rhs_bit {
                continue;
            }
            let symbolic = match &inst.kind {
                InstKind::Nop => continue,
                InstKind::Phi { incomings } => {
                    let incoming = incomings
                        .iter()
                        .find_map(|(pred, value)| (*pred == predecessor).then_some(*value))?;
                    symbolic_value(func, incoming, &values)
                }
                InstKind::Unary { op, value } => {
                    eval_unary(*op, symbolic_value(func, *value, &values))
                }
                InstKind::Binary { op, lhs, rhs } => eval_binary(
                    *op,
                    symbolic_value(func, *lhs, &values),
                    symbolic_value(func, *rhs, &values),
                ),
                InstKind::Icmp { op, lhs, rhs } => eval_icmp(
                    *op,
                    symbolic_value(func, *lhs, &values),
                    symbolic_value(func, *rhs, &values),
                ),
                InstKind::Cast { op, value } => {
                    eval_cast(*op, symbolic_value(func, *value, &values))
                }
                _ => Symbolic::Unknown,
            };
            values.insert(result, symbolic);
        }

        let next = match block.terminator.as_ref()? {
            Terminator::Jump(target) => *target,
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => match truthy(symbolic_value(func, *cond, &values))? {
                true => *then_target,
                false => *else_target,
            },
            Terminator::Return(_) => return None,
        };
        if next == natural_loop.header {
            return match symbolic_value(func, recurrences.accumulator_next, &values) {
                Symbolic::Linear(expression) => Some(expression),
                _ => None,
            };
        }
        predecessor = current;
        current = next;
    }
}

fn symbolic_value(
    func: &Function,
    value: ValueId,
    values: &HashMap<ValueId, Symbolic>,
) -> Symbolic {
    values.get(&value).copied().unwrap_or_else(|| {
        match func.values.get(value.0).map(|value| &value.kind) {
            Some(ValueKind::Const(Const::Int(value))) => Symbolic::Int(*value),
            Some(ValueKind::Const(Const::Bool(value))) => Symbolic::Bool(*value),
            _ => Symbolic::Unknown,
        }
    })
}

fn eval_unary(op: UnaryOp, value: Symbolic) -> Symbolic {
    match (op, value) {
        (UnaryOp::Ineg, Symbolic::Int(value)) => Symbolic::Int(value.wrapping_neg()),
        (UnaryOp::Ineg, Symbolic::Linear(value)) => Symbolic::Linear(scale_linear(value, -1)),
        (UnaryOp::Not, value) => truthy(value)
            .map(|value| Symbolic::Bool(!value))
            .unwrap_or(Symbolic::Unknown),
        _ => Symbolic::Unknown,
    }
}

fn eval_binary(op: BinaryOp, lhs: Symbolic, rhs: Symbolic) -> Symbolic {
    match (op, lhs, rhs) {
        (BinaryOp::Iadd, Symbolic::Int(lhs), Symbolic::Int(rhs)) => {
            Symbolic::Int(lhs.wrapping_add(rhs))
        }
        (BinaryOp::Iadd, lhs, rhs) => combine_linear(lhs, rhs, false),
        (BinaryOp::Isub, Symbolic::Int(lhs), Symbolic::Int(rhs)) => {
            Symbolic::Int(lhs.wrapping_sub(rhs))
        }
        (BinaryOp::Isub, lhs, rhs) => combine_linear(lhs, rhs, true),
        (BinaryOp::Imul, Symbolic::Int(lhs), Symbolic::Int(rhs)) => {
            Symbolic::Int(lhs.wrapping_mul(rhs))
        }
        (BinaryOp::Imul, Symbolic::Linear(value), Symbolic::Int(scale))
        | (BinaryOp::Imul, Symbolic::Int(scale), Symbolic::Linear(value)) => {
            Symbolic::Linear(scale_linear(value, scale))
        }
        (BinaryOp::And, lhs, rhs) => match (truthy(lhs), truthy(rhs)) {
            (Some(lhs), Some(rhs)) => Symbolic::Bool(lhs && rhs),
            _ => Symbolic::Unknown,
        },
        (BinaryOp::Or, lhs, rhs) => match (truthy(lhs), truthy(rhs)) {
            (Some(lhs), Some(rhs)) => Symbolic::Bool(lhs || rhs),
            _ => Symbolic::Unknown,
        },
        _ => Symbolic::Unknown,
    }
}

fn combine_linear(lhs: Symbolic, rhs: Symbolic, subtract: bool) -> Symbolic {
    let Some(lhs) = as_linear(lhs) else {
        return Symbolic::Unknown;
    };
    let Some(mut rhs) = as_linear(rhs) else {
        return Symbolic::Unknown;
    };
    if subtract {
        rhs = scale_linear(rhs, -1);
    }
    Symbolic::Linear(Linear {
        accumulator: lhs.accumulator.wrapping_add(rhs.accumulator),
        power: lhs.power.wrapping_add(rhs.power),
        constant: lhs.constant.wrapping_add(rhs.constant),
    })
}

fn as_linear(value: Symbolic) -> Option<Linear> {
    match value {
        Symbolic::Linear(value) => Some(value),
        Symbolic::Int(value) => Some(Linear {
            accumulator: 0,
            power: 0,
            constant: value,
        }),
        _ => None,
    }
}

fn scale_linear(value: Linear, scale: i32) -> Linear {
    Linear {
        accumulator: value.accumulator.wrapping_mul(scale),
        power: value.power.wrapping_mul(scale),
        constant: value.constant.wrapping_mul(scale),
    }
}

fn eval_icmp(op: CmpOp, lhs: Symbolic, rhs: Symbolic) -> Symbolic {
    let (Some(lhs), Some(rhs)) = (concrete_i32(lhs), concrete_i32(rhs)) else {
        return Symbolic::Unknown;
    };
    Symbolic::Bool(match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
    })
}

fn eval_cast(op: CastOp, value: Symbolic) -> Symbolic {
    match op {
        CastOp::BoolToI32 => truthy(value)
            .map(|value| Symbolic::Int(value as i32))
            .unwrap_or(Symbolic::Unknown),
        CastOp::I32ToBool => truthy(value)
            .map(Symbolic::Bool)
            .unwrap_or(Symbolic::Unknown),
        _ => Symbolic::Unknown,
    }
}

fn concrete_i32(value: Symbolic) -> Option<i32> {
    match value {
        Symbolic::Int(value) => Some(value),
        Symbolic::Bool(value) => Some(value as i32),
        _ => None,
    }
}

fn truthy(value: Symbolic) -> Option<bool> {
    concrete_i32(value).map(|value| value != 0)
}

fn install_fast_path(func: &mut Function, candidate: Candidate) {
    let fast = func.add_block(format!("loop.idiom.fast.{}", candidate.exiting.0));
    let zero = get_or_add_i32_const(func, 0);
    let lhs_nonnegative = func
        .append_inst(
            candidate.preheader,
            InstKind::Icmp {
                op: CmpOp::Ge,
                lhs: candidate.lhs,
                rhs: zero,
            },
            Some(Type::I1),
        )
        .unwrap();
    let rhs_nonnegative = func
        .append_inst(
            candidate.preheader,
            InstKind::Icmp {
                op: CmpOp::Ge,
                lhs: candidate.rhs,
                rhs: zero,
            },
            Some(Type::I1),
        )
        .unwrap();
    let use_fast_path = func
        .append_inst(
            candidate.preheader,
            InstKind::Binary {
                op: BinaryOp::And,
                lhs: lhs_nonnegative,
                rhs: rhs_nonnegative,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.blocks[candidate.preheader.0].terminator = Some(Terminator::Branch {
        cond: use_fast_path,
        then_target: fast,
        else_target: candidate.exiting,
    });

    let mut fast_result = synthesize_truth_table(
        func,
        fast,
        candidate.lhs,
        candidate.rhs,
        candidate.truth_table,
    );
    if candidate.iterations < I32_BITS {
        let mask = ((1u64 << candidate.iterations) - 1) as u32 as i32;
        let mask = get_or_add_i32_const(func, mask);
        fast_result = append_binary(func, fast, BinaryOp::Iand, fast_result, mask);
    }
    func.set_terminator(fast, Terminator::Jump(candidate.exit));

    for inst in &mut func.blocks[candidate.exit.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        let incoming = unique_phi_incoming(incomings, candidate.exiting)
            .expect("candidate exit phi was proven safe");
        incomings.push((fast, incoming));
    }
    let phi_pos = func.blocks[candidate.exit.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }))
        .count();
    let merged = func
        .insert_inst(
            candidate.exit,
            phi_pos,
            InstKind::Phi {
                incomings: vec![
                    (candidate.exiting, candidate.accumulator),
                    (fast, fast_result),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    rewrite_accumulator_liveout(func, candidate.accumulator, merged, candidate.exiting);
}

fn synthesize_truth_table(
    func: &mut Function,
    block: BlockId,
    lhs: ValueId,
    rhs: ValueId,
    table: u8,
) -> ValueId {
    if table == 0 {
        return get_or_add_i32_const(func, 0);
    }
    if table == 0b1111 {
        return get_or_add_i32_const(func, -1);
    }
    if table == 0b1000 {
        return append_binary(func, block, BinaryOp::Iand, lhs, rhs);
    }
    if table == 0b1110 {
        return append_binary(func, block, BinaryOp::Ior, lhs, rhs);
    }
    if table == 0b0110 {
        return append_binary(func, block, BinaryOp::Ixor, lhs, rhs);
    }

    let all_ones = get_or_add_i32_const(func, -1);
    let not_lhs = append_binary(func, block, BinaryOp::Ixor, lhs, all_ones);
    let not_rhs = append_binary(func, block, BinaryOp::Ixor, rhs, all_ones);
    let mut terms = Vec::new();
    for lhs_bit in 0..=1u8 {
        for rhs_bit in 0..=1u8 {
            let bit = 1 << ((lhs_bit << 1) | rhs_bit);
            if table & bit == 0 {
                continue;
            }
            let lhs_value = if lhs_bit == 0 { not_lhs } else { lhs };
            let rhs_value = if rhs_bit == 0 { not_rhs } else { rhs };
            terms.push(append_binary(
                func,
                block,
                BinaryOp::Iand,
                lhs_value,
                rhs_value,
            ));
        }
    }
    terms
        .into_iter()
        .reduce(|lhs, rhs| append_binary(func, block, BinaryOp::Ior, lhs, rhs))
        .unwrap()
}

fn append_binary(
    func: &mut Function,
    block: BlockId,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> ValueId {
    func.append_inst(block, InstKind::Binary { op, lhs, rhs }, Some(Type::I32))
        .unwrap()
}

fn rewrite_accumulator_liveout(
    func: &mut Function,
    accumulator: ValueId,
    merged: ValueId,
    loop_header: BlockId,
) {
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let loop_blocks = loop_info
        .loops()
        .iter()
        .find(|natural_loop| natural_loop.header == loop_header)
        .map(|natural_loop| natural_loop.blocks.clone())
        .expect("fallback loop must remain natural");

    for (block_idx, block) in func.blocks.iter_mut().enumerate() {
        if loop_blocks.contains(&BlockId(block_idx)) {
            continue;
        }
        for inst in &mut block.insts {
            if matches!(inst.kind, InstKind::Phi { .. }) {
                continue;
            }
            rewrite_inst_operand(&mut inst.kind, accumulator, merged);
        }
        if let Some(terminator) = &mut block.terminator {
            rewrite_terminator_operand(terminator, accumulator, merged);
        }
    }
}

fn rewrite_inst_operand(kind: &mut InstKind, old: ValueId, new: ValueId) {
    let rewrite = |value: &mut ValueId| {
        if *value == old {
            *value = new;
        }
    };
    match kind {
        InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. } => {}
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => rewrite(ptr),
        InstKind::Store { ptr, value } => {
            rewrite(ptr);
            rewrite(value);
        }
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => rewrite(value),
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            rewrite(lhs);
            rewrite(rhs);
        }
        InstKind::Gep { base, indices } => {
            rewrite(base);
            for index in indices {
                rewrite(index);
            }
        }
        InstKind::Call { args, .. } => {
            for arg in args {
                rewrite(arg);
            }
        }
    }
}

fn rewrite_terminator_operand(terminator: &mut Terminator, old: ValueId, new: ValueId) {
    let operand = match terminator {
        Terminator::Return(Some(value)) => Some(value),
        Terminator::Branch { cond, .. } => Some(cond),
        Terminator::Return(None) | Terminator::Jump(_) => None,
    };
    if let Some(operand) = operand {
        if *operand == old {
            *operand = new;
        }
    }
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut values = Vec::with_capacity(indices.len() + 1);
            values.push(*base);
            values.extend(indices.iter().copied());
            values
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

fn is_multiply_two(func: &Function, value: ValueId, input: ValueId) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Binary { op: BinaryOp::Imul, lhs, rhs })
            if (*lhs == input && is_const_i32(func, *rhs, 2))
                || (*rhs == input && is_const_i32(func, *lhs, 2))
    )
}

fn is_divide_two(func: &Function, value: ValueId, input: ValueId) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Binary { op: BinaryOp::Idiv, lhs, rhs })
            if *lhs == input && is_const_i32(func, *rhs, 2)
    )
}

fn is_const_i32(func: &Function, value: ValueId, expected: i32) -> bool {
    const_i32(func, value) == Some(expected)
}

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, ModulePass, OptLevel, PassOptions};
    use super::*;
    use crate::ir::{self, Function, Module, ValueKind};
    use crate::parser::Parser;

    fn bit_source(
        count_up: bool,
        truth_condition: &str,
        left_bit: &str,
        right_bit: &str,
        weight_update: &str,
        left_initial: &str,
        right_initial: &str,
    ) -> String {
        let (count_initial, loop_condition, count_update) = if count_up {
            ("0", "count < 32", "count = count + 1;")
        } else {
            ("32", "count > 0", "count = count - 1;")
        };
        format!(
            r#"
                int target(int extra, int first, int second, int unused) {{
                    int before = extra + 7;
                    int left = {left_initial};
                    int right = {right_initial};
                    int output = 0;
                    int count = {count_initial};
                    int weight = 1;
                    int left_bit;
                    int right_bit;
                    while ({loop_condition}) {{
                        left_bit = {left_bit};
                        right_bit = {right_bit};
                        left = left / 2;
                        right = right / 2;
                        if ({truth_condition}) output = output + weight;
                        {count_update}
                        weight = {weight_update};
                    }}
                    return output + before;
                }}
            "#
        )
    }

    fn default_source(count_up: bool) -> String {
        bit_source(
            count_up,
            "left_bit == 1 && right_bit == 1",
            "left % 2",
            "right % 2",
            "weight * 2",
            "first",
            "second",
        )
    }

    fn lower(source: &str) -> Module {
        let mut parser = Parser::new(source);
        ir::lower::lower_program(&parser.parse_program()).expect("test source must lower")
    }

    fn optimize(source: &str, opt_level: OptLevel) -> Module {
        let mut module = lower(source);
        run_pipeline(
            &mut module,
            opt_level,
            PassOptions {
                enable_simple_loop_unroll: false,
            },
        );
        module
    }

    fn function<'a>(module: &'a Module, name: &str) -> &'a Function {
        module
            .funcs
            .iter()
            .find(|func| func.name == name)
            .expect("test function must exist")
    }

    fn fast_block_count(func: &Function) -> usize {
        func.blocks
            .iter()
            .filter(|block| block.name.starts_with("loop.idiom.fast."))
            .count()
    }

    fn count_binary(func: &Function, expected: BinaryOp) -> usize {
        func.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter(|inst| matches!(inst.kind, InstKind::Binary { op, .. } if op == expected))
            .count()
    }

    #[test]
    fn transforms_one_region_with_context_and_preserves_another_loop() {
        let source = r#"
            void work(int extra, int first, int second, int limit) {
                int before = extra + 7;
                int left = first;
                int right = second;
                int output = 0;
                int count = 0;
                int weight = 1;
                int left_bit;
                int right_bit;
                while (count < 32) {
                    left_bit = left % 2;
                    right_bit = right % 2;
                    left = left / 2;
                    right = right / 2;
                    if (left_bit == 1 && right_bit == 1) output = output + weight;
                    count = count + 1;
                    weight = weight * 2;
                }
                putint(output + before);
                int i = 0;
                while (i < limit) {
                    putint(i);
                    i = i + 1;
                }
            }
        "#;
        let module = optimize(source, OptLevel::O1);
        let func = function(&module, "work");
        assert_eq!(func.ret, Type::Void);
        assert_eq!(func.params.len(), 4);
        assert_eq!(fast_block_count(func), 1);
        assert!(func.blocks.iter().any(|block| {
            block
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
        }));
        let cfg = ControlFlowGraph::new(func);
        let dom = Dominators::new(func, &cfg);
        assert_eq!(LoopInfo::new(&cfg, &dom).loops().len(), 2);
        assert!(func.verify().is_ok());
    }

    #[test]
    fn supports_count_up_and_count_down_loops() {
        for count_up in [true, false] {
            let module = optimize(&default_source(count_up), OptLevel::O1);
            let func = function(&module, "target");
            assert_eq!(fast_block_count(func), 1);
            assert_eq!(count_binary(func, BinaryOp::Iand), 1);
            assert!(func.verify().is_ok());
        }
    }

    #[test]
    fn supports_every_bit_width_from_one_through_thirty_two() {
        for iterations in 1..=32 {
            let source = default_source(false)
                .replace("int count = 32;", &format!("int count = {iterations};"));
            let module = optimize(&source, OptLevel::O1);
            let func = function(&module, "target");
            assert_eq!(fast_block_count(func), 1, "width {iterations}");
            assert_eq!(
                count_binary(func, BinaryOp::Iand),
                if iterations == 32 { 1 } else { 2 },
                "width {iterations} should have the expected fast-path mask"
            );
            assert!(func.verify().is_ok());
        }
    }

    #[test]
    fn relies_on_generic_integer_canonicalization() {
        let variants = [
            bit_source(
                false,
                "left_bit == 1 && right_bit == 1",
                "left % 2",
                "right % 2",
                "weight + weight",
                "first",
                "second",
            ),
            bit_source(
                false,
                "left_bit == 1 && right_bit == 1",
                "left - (left / 2) * 2",
                "right - (right / 2) * 2",
                "weight * 2",
                "first",
                "second",
            ),
            bit_source(
                false,
                "left_bit == 1 && right_bit == 1",
                "left % 2",
                "right % 2",
                "(weight * -1) * -2",
                "first",
                "second",
            ),
        ];
        for source in variants {
            let module = optimize(&source, OptLevel::O1);
            assert_eq!(fast_block_count(function(&module, "target")), 1);
        }
    }

    #[test]
    fn guards_computed_inputs_and_merges_only_the_accumulator_liveout() {
        let source = bit_source(
            false,
            "left_bit == 1 && right_bit == 1",
            "left % 2",
            "right % 2",
            "weight * 2",
            "first + extra",
            "second * 2",
        );
        let module = optimize(&source, OptLevel::O1);
        let func = function(&module, "target");
        let fast = func
            .blocks
            .iter()
            .position(|block| block.name.starts_with("loop.idiom.fast."))
            .map(BlockId)
            .unwrap();
        let (preheader, fallback, guard) = func
            .blocks
            .iter()
            .enumerate()
            .find_map(|(idx, block)| match block.terminator {
                Some(Terminator::Branch {
                    cond,
                    then_target,
                    else_target,
                }) if then_target == fast => Some((BlockId(idx), else_target, cond)),
                _ => None,
            })
            .unwrap();
        let InstKind::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
        } = defining_inst(func, guard).unwrap()
        else {
            panic!("fast guard must combine both nonnegative checks");
        };
        for comparison in [*lhs, *rhs] {
            assert!(matches!(
                defining_inst(func, comparison),
                Some(InstKind::Icmp {
                    op: CmpOp::Ge,
                    lhs,
                    rhs,
                }) if matches!(func.value(*lhs).kind, ValueKind::Inst(_, _))
                    && const_i32(func, *rhs) == Some(0)
            ));
        }
        assert!(func.blocks[fallback.0]
            .insts
            .iter()
            .any(|inst| { matches!(inst.kind, InstKind::Phi { .. }) }));
        assert!(func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                matches!(
                    &inst.kind,
                    InstKind::Phi { incomings }
                        if incomings.iter().any(|(pred, _)| *pred == fast)
                            && incomings.iter().any(|(pred, _)| *pred == fallback)
                )
            })
        }));
        assert!(matches!(
            func.blocks[preheader.0].terminator,
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) if then_target == fast && else_target == fallback
        ));
        assert!(func.verify().is_ok());
    }

    #[test]
    fn rejects_effects_side_exits_extra_liveouts_and_thirty_three_iterations() {
        let side_effect = default_source(false).replace(
            "left_bit = left % 2;",
            "left_bit = left % 2; putint(left_bit);",
        );
        let side_exit = default_source(false).replace(
            "left_bit = left % 2;",
            "left_bit = left % 2; if (left_bit < 0) break;",
        );
        let extra_liveout = default_source(false)
            .replace("return output + before;", "return output + before + left;");
        let too_many = default_source(false).replace("int count = 32;", "int count = 33;");

        for source in [side_effect, side_exit, extra_liveout, too_many] {
            let module = optimize(&source, OptLevel::O1);
            assert_eq!(fast_block_count(function(&module, "target")), 0);
            assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
        }
    }

    #[test]
    fn rejects_a_loop_input_that_is_not_available_at_the_preheader() {
        let mut module = optimize(&default_source(false), OptLevel::O0);
        let func = module.funcs.first_mut().unwrap();
        let (natural_loop, candidate) = {
            let cfg = ControlFlowGraph::new(func);
            let dom = Dominators::new(func, &cfg);
            let loop_info = LoopInfo::new(&cfg, &dom);
            loop_info
                .loops()
                .iter()
                .find_map(|natural_loop| {
                    match_candidate(func, &cfg, &dom, natural_loop)
                        .map(|candidate| (natural_loop.clone(), candidate))
                })
                .expect("canonical loop should match before corruption")
        };
        let preheader = natural_loop.dedicated_preheader.unwrap();
        let dead = func.add_block("dead.input");
        let one = get_or_add_i32_const(func, 1);
        let unavailable = func
            .append_inst(
                dead,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: one,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(dead, Terminator::Return(Some(unavailable)));
        let input_phi = func.blocks[natural_loop.header.0]
            .insts
            .iter_mut()
            .find(|inst| {
                matches!(
                    &inst.kind,
                    InstKind::Phi { incomings }
                        if incomings.contains(&(preheader, candidate.lhs))
                )
            })
            .unwrap();
        let InstKind::Phi { incomings } = &mut input_phi.kind else {
            unreachable!();
        };
        incomings
            .iter_mut()
            .find(|(pred, _)| *pred == preheader)
            .unwrap()
            .1 = unavailable;

        let cfg = ControlFlowGraph::new(func);
        let dom = Dominators::new(func, &cfg);
        let loop_info = LoopInfo::new(&cfg, &dom);
        assert!(loop_info.loops().iter().all(|natural_loop| match_candidate(
            func,
            &cfg,
            &dom,
            natural_loop
        )
        .is_none()));
    }

    #[test]
    fn extends_safe_existing_exit_phis_for_the_fast_edge() {
        let mut module = optimize(&default_source(false), OptLevel::O0);
        let (exit, exiting, invariant) = {
            let func = module.funcs.first().unwrap();
            let cfg = ControlFlowGraph::new(func);
            let dom = Dominators::new(func, &cfg);
            let loop_info = LoopInfo::new(&cfg, &dom);
            loop_info
                .loops()
                .iter()
                .find_map(|natural_loop| {
                    match_candidate(func, &cfg, &dom, natural_loop)
                        .map(|candidate| (candidate.exit, candidate.exiting, candidate.lhs))
                })
                .unwrap()
        };
        let existing_phi = module.funcs[0]
            .insert_inst(
                exit,
                0,
                InstKind::Phi {
                    incomings: vec![(exiting, invariant)],
                },
                Some(Type::I32),
            )
            .unwrap();

        LoopIdiomPass::new().run(&mut module);
        let func = &module.funcs[0];
        let InstKind::Phi { incomings } = defining_inst(func, existing_phi).unwrap() else {
            unreachable!();
        };
        assert_eq!(incomings.len(), 2);
        assert!(incomings.iter().all(|(_, value)| *value == invariant));
        assert_eq!(fast_block_count(func), 1);
        assert!(func.verify().is_ok());
    }

    #[test]
    fn supports_all_two_input_boolean_truth_tables() {
        for table in 0u8..16 {
            let condition = if table == 0 {
                "left_bit == 0 && right_bit == 0 && left_bit == 1".to_string()
            } else {
                let mut terms = Vec::new();
                for left in 0..=1u8 {
                    for right in 0..=1u8 {
                        if table & (1 << ((left << 1) | right)) != 0 {
                            terms.push(format!("(left_bit == {left} && right_bit == {right})"));
                        }
                    }
                }
                terms.join(" || ")
            };
            let source = bit_source(
                false,
                &condition,
                "left % 2",
                "right % 2",
                "weight * 2",
                "first",
                "second",
            );
            let module = optimize(&source, OptLevel::O1);
            let func = function(&module, "target");
            assert_eq!(
                fast_block_count(func),
                1,
                "truth table {table:04b} did not transform"
            );
            assert!(func.verify().is_ok());
        }
    }

    #[test]
    fn synthesized_fast_expressions_implement_every_truth_table() {
        for table in 0u8..16 {
            let mut func = Function::new("truth", Type::I32);
            let lhs = func.add_param("lhs", Type::I32);
            let rhs = func.add_param("rhs", Type::I32);
            let entry = func.entry;
            let result = synthesize_truth_table(&mut func, entry, lhs, rhs, table);
            func.set_terminator(func.entry, Terminator::Return(Some(result)));
            assert!(func.verify().is_ok());

            for (lhs_value, rhs_value) in [
                (0, 0),
                (0, -1),
                (-1, 0),
                (-1, -1),
                (0x5555_5555, 0x3333_3333),
                (0x1234_5678, 0x7654_3210),
            ] {
                assert_eq!(
                    evaluate_synthesized(&func, lhs, rhs, lhs_value, rhs_value),
                    evaluate_truth_table(table, lhs_value, rhs_value),
                    "table {table:04b}, lhs={lhs_value:#x}, rhs={rhs_value:#x}"
                );
            }
        }
    }

    fn evaluate_synthesized(
        func: &Function,
        lhs_param: ValueId,
        rhs_param: ValueId,
        lhs_value: i32,
        rhs_value: i32,
    ) -> i32 {
        let mut values = vec![None; func.values.len()];
        for (idx, value) in func.values.iter().enumerate() {
            if let ValueKind::Const(Const::Int(value)) = value.kind {
                values[idx] = Some(value);
            }
        }
        values[lhs_param.0] = Some(lhs_value);
        values[rhs_param.0] = Some(rhs_value);
        for inst in &func.blocks[func.entry.0].insts {
            let Some(result) = inst.result else {
                continue;
            };
            let InstKind::Binary { op, lhs, rhs } = inst.kind else {
                panic!("unexpected synthesized instruction: {:?}", inst.kind);
            };
            let lhs = values[lhs.0].unwrap();
            let rhs = values[rhs.0].unwrap();
            values[result.0] = Some(match op {
                BinaryOp::Iand => lhs & rhs,
                BinaryOp::Ior => lhs | rhs,
                BinaryOp::Ixor => lhs ^ rhs,
                _ => panic!("unexpected synthesized operation: {op:?}"),
            });
        }
        let Terminator::Return(Some(result)) =
            func.blocks[func.entry.0].terminator.as_ref().unwrap()
        else {
            unreachable!();
        };
        values[result.0].unwrap()
    }

    fn evaluate_truth_table(table: u8, lhs: i32, rhs: i32) -> i32 {
        let mut result = 0u32;
        for shift in 0..32 {
            let lhs_bit = ((lhs as u32 >> shift) & 1) as u8;
            let rhs_bit = ((rhs as u32 >> shift) & 1) as u8;
            if table & (1 << ((lhs_bit << 1) | rhs_bit)) != 0 {
                result |= 1 << shift;
            }
        }
        result as i32
    }

    #[test]
    fn is_o1_only_idempotent_and_verifier_clean() {
        let source = default_source(false);
        let o0 = optimize(&source, OptLevel::O0);
        assert_eq!(fast_block_count(function(&o0, "target")), 0);

        let mut o1 = optimize(&source, OptLevel::O1);
        assert_eq!(fast_block_count(function(&o1, "target")), 1);
        let transformed = o1.clone();
        LoopIdiomPass::new().run(&mut o1);
        assert_eq!(o1, transformed);
        assert!(o1.funcs.iter().all(|func| func.verify().is_ok()));
    }
}
