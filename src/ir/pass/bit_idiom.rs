use super::dominators::{ControlFlowGraph, Dominators};
use super::util::{
    const_i32, defining_inst, get_or_add_i32_const as get_or_add_const, move_entry_to_fallback,
};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CastOp, CmpOp, Const, Function, InstKind, Module, Terminator, Type, UnaryOp,
    ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

// Keep whole-function semantic matching linear and bounded on generated stress cases.
const MAX_BLOCKS: usize = 128;
const MAX_VALUES: usize = 2048;
const I32_BITS: i32 = i32::BITS as i32;

/// Recognizes pure bit-sliced recurrences by their SSA semantics and versions
/// them with a native integer-bitwise fast path. The original function remains
/// as the fallback for negative operands, whose signed division/remainder
/// recurrence is not equivalent to a native bitwise operation.
pub(super) struct BitwiseIdiomPass;

impl BitwiseIdiomPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for BitwiseIdiomPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            let Some(candidate) = match_bitwise_idiom(func) else {
                continue;
            };
            if !install_fast_path(func, candidate) {
                continue;
            }
            if let Err(errors) = func.verify() {
                panic!(
                    "bitwise idiom recognition produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    lhs: ValueId,
    rhs: ValueId,
    iterations: u32,
    truth_table: u8,
}

#[derive(Clone)]
struct NaturalLoop {
    header: BlockId,
    preheader: BlockId,
    latch: BlockId,
    body_entry: BlockId,
    exit: BlockId,
    blocks: HashSet<BlockId>,
}

#[derive(Clone, Copy)]
struct Recurrences {
    lhs: ValueId,
    rhs: ValueId,
    lhs_bit: ValueId,
    rhs_bit: ValueId,
    power: ValueId,
    accumulator: ValueId,
    accumulator_next: ValueId,
    iterations: u32,
}

fn match_bitwise_idiom(func: &Function) -> Option<Candidate> {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || func
            .params
            .iter()
            .any(|param| func.value(*param).ty != Type::I32)
        || function_has_integer_bitwise(func)
        || !function_is_pure_scalar(func)
    {
        return None;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let mut matched = None;
    for natural_loop in find_natural_loops(func, &cfg, &dom) {
        let Some(recurrences) = match_recurrences(func, &cfg, &natural_loop) else {
            continue;
        };
        let Some(truth_table) = derive_truth_table(func, &natural_loop, recurrences) else {
            continue;
        };
        let candidate = Candidate {
            lhs: recurrences.lhs,
            rhs: recurrences.rhs,
            iterations: recurrences.iterations,
            truth_table,
        };
        if matched.replace(candidate).is_some() {
            return None;
        }
    }
    matched
}

fn function_has_integer_bitwise(func: &Function) -> bool {
    func.blocks.iter().any(|block| {
        block.insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::Binary {
                    op: BinaryOp::Iand
                        | BinaryOp::Ior
                        | BinaryOp::Ixor
                        | BinaryOp::Ishl
                        | BinaryOp::Iashr,
                    ..
                }
            )
        })
    })
}

fn function_is_pure_scalar(func: &Function) -> bool {
    func.blocks.iter().all(|block| {
        block.insts.iter().all(|inst| {
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

fn find_natural_loops(
    func: &Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
) -> Vec<NaturalLoop> {
    let mut loops = Vec::new();
    for latch_idx in 0..func.blocks.len() {
        let latch = BlockId(latch_idx);
        for header in &cfg.succs[latch_idx] {
            if !dom.dominates(*header, latch) {
                continue;
            }
            let blocks = collect_loop_blocks(cfg, *header, latch);
            let outside_preds = cfg.preds[header.0]
                .iter()
                .copied()
                .filter(|pred| !blocks.contains(pred))
                .collect::<Vec<_>>();
            let [preheader] = outside_preds.as_slice() else {
                continue;
            };
            if *preheader != func.entry
                || !cfg.preds[func.entry.0].is_empty()
                || cfg.preds[header.0].len() != 2
                || cfg.preds[header.0]
                    .iter()
                    .filter(|pred| blocks.contains(pred))
                    .count()
                    != 1
            {
                continue;
            }

            let Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) = func.blocks[header.0].terminator.as_ref()
            else {
                continue;
            };
            let (body_entry, exit) =
                match (blocks.contains(then_target), blocks.contains(else_target)) {
                    (true, false) => (*then_target, *else_target),
                    _ => continue,
                };
            if blocks.iter().any(|block| {
                *block != *header && cfg.succs[block.0].iter().any(|succ| !blocks.contains(succ))
            }) {
                continue;
            }

            let mut allowed_blocks = blocks.clone();
            allowed_blocks.insert(*preheader);
            allowed_blocks.insert(exit);
            if allowed_blocks.len() != func.blocks.len()
                || !func.blocks[preheader.0]
                    .insts
                    .iter()
                    .all(|inst| matches!(inst.kind, InstKind::Nop))
                || !func.blocks[exit.0]
                    .insts
                    .iter()
                    .all(|inst| matches!(inst.kind, InstKind::Nop))
            {
                continue;
            }
            loops.push(NaturalLoop {
                header: *header,
                preheader: *preheader,
                latch,
                body_entry,
                exit,
                blocks,
            });
        }
    }
    loops
}

fn collect_loop_blocks(
    cfg: &ControlFlowGraph,
    header: BlockId,
    latch: BlockId,
) -> HashSet<BlockId> {
    let mut blocks = HashSet::from([header, latch]);
    let mut stack = vec![latch];
    while let Some(block) = stack.pop() {
        for pred in &cfg.preds[block.0] {
            if *pred != header && blocks.insert(*pred) {
                stack.push(*pred);
            }
        }
    }
    blocks
}

fn match_recurrences(
    func: &Function,
    cfg: &ControlFlowGraph,
    natural_loop: &NaturalLoop,
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
    let mut count = None;
    let mut power = None;
    let mut accumulator = None;
    let mut inputs = Vec::new();
    for phi in &header_phis {
        let initial = phi_incoming(func, *phi, natural_loop.preheader)?;
        let next = phi_incoming(func, *phi, natural_loop.latch)?;
        if let Some(iterations) = const_i32(func, initial)
            .filter(|value| (1..=I32_BITS).contains(value))
            .filter(|_| is_decrement_one(func, next, *phi))
        {
            if count.replace((*phi, iterations as u32)).is_some() {
                return None;
            }
        } else if is_const_i32(func, initial, 1) && is_multiply_two(func, next, *phi) {
            if power.replace(*phi).is_some() {
                return None;
            }
        } else if is_const_i32(func, initial, 0) {
            if accumulator.replace((*phi, next)).is_some() {
                return None;
            }
        } else if func.params.contains(&initial) && is_divide_two(func, next, *phi) {
            inputs.push((*phi, initial));
        } else {
            return None;
        }
    }
    let (count, iterations) = count?;
    let power = power?;
    let (accumulator, accumulator_next) = accumulator?;
    if inputs.len() != 2
        || inputs[0].1 == inputs[1].1
        || !matches_loop_condition(func, natural_loop, count)
        || cfg.succs[natural_loop.latch.0] != [natural_loop.header]
        || !matches!(
            func.blocks[natural_loop.exit.0].terminator,
            Some(Terminator::Return(Some(value))) if value == accumulator
        )
        || func
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| matches!(block.terminator, Some(Terminator::Return(_))))
            .count()
            != 1
    {
        return None;
    }

    let lhs_bit = find_modulo_two(func, &natural_loop.blocks, inputs[0].0)?;
    let rhs_bit = find_modulo_two(func, &natural_loop.blocks, inputs[1].0)?;
    Some(Recurrences {
        lhs: inputs[0].1,
        rhs: inputs[1].1,
        lhs_bit,
        rhs_bit,
        power,
        accumulator,
        accumulator_next,
        iterations,
    })
}

fn matches_loop_condition(func: &Function, natural_loop: &NaturalLoop, count: ValueId) -> bool {
    let Some(Terminator::Branch {
        cond,
        then_target,
        else_target,
    }) = func.blocks[natural_loop.header.0].terminator.as_ref()
    else {
        return false;
    };
    if *then_target != natural_loop.body_entry || *else_target != natural_loop.exit {
        return false;
    }
    matches!(
        defining_inst(func, *cond),
        Some(InstKind::Icmp { op: CmpOp::Ne, lhs, rhs })
            if (*lhs == count && is_const_i32(func, *rhs, 0))
                || (*rhs == count && is_const_i32(func, *lhs, 0))
    ) || matches!(
        defining_inst(func, *cond),
        Some(InstKind::Icmp { op: CmpOp::Gt, lhs, rhs })
            if *lhs == count && is_const_i32(func, *rhs, 0)
    ) || matches!(
        defining_inst(func, *cond),
        Some(InstKind::Icmp { op: CmpOp::Lt, lhs, rhs })
            if is_const_i32(func, *lhs, 0) && *rhs == count
    )
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
    let mut current = natural_loop.body_entry;
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

fn install_fast_path(func: &mut Function, candidate: Candidate) -> bool {
    let Some(fallback) = move_entry_to_fallback(func, "idiom.bitwise.fallback") else {
        return false;
    };
    let fast = func.add_block("idiom.bitwise.fast");
    let zero = get_or_add_const(func, 0);
    let lhs_nonnegative = func
        .append_inst(
            func.entry,
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
            func.entry,
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
            func.entry,
            InstKind::Binary {
                op: BinaryOp::And,
                lhs: lhs_nonnegative,
                rhs: rhs_nonnegative,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        func.entry,
        Terminator::Branch {
            cond: use_fast_path,
            then_target: fast,
            else_target: fallback,
        },
    );

    let mut result = synthesize_truth_table(
        func,
        fast,
        candidate.lhs,
        candidate.rhs,
        candidate.truth_table,
    );
    if candidate.iterations < i32::BITS {
        let mask = ((1u64 << candidate.iterations) - 1) as u32 as i32;
        let mask = get_or_add_const(func, mask);
        result = append_binary(func, fast, BinaryOp::Iand, result, mask);
    }
    func.set_terminator(fast, Terminator::Return(Some(result)));
    true
}

fn synthesize_truth_table(
    func: &mut Function,
    block: BlockId,
    lhs: ValueId,
    rhs: ValueId,
    table: u8,
) -> ValueId {
    if table == 0 {
        return get_or_add_const(func, 0);
    }
    if table == 0b1111 {
        return get_or_add_const(func, -1);
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

    let all_ones = get_or_add_const(func, -1);
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

fn phi_incoming(func: &Function, phi: ValueId, pred: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    incomings
        .iter()
        .find_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value))
}

fn is_decrement_one(func: &Function, value: ValueId, input: ValueId) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Binary { op: BinaryOp::Isub, lhs, rhs })
            if *lhs == input && is_const_i32(func, *rhs, 1)
    ) || matches!(
        defining_inst(func, value),
        Some(InstKind::Binary { op: BinaryOp::Iadd, lhs, rhs })
            if (*lhs == input && is_const_i32(func, *rhs, -1))
                || (*rhs == input && is_const_i32(func, *lhs, -1))
    )
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
