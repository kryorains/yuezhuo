use crate::ir::{
    BinaryOp, CastOp, Const, Function, InstKind, Module, Terminator, Type, UnaryOp, ValueId,
    ValueKind,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IntRange {
    pub(crate) min: i64,
    pub(crate) max: i64,
}

impl IntRange {
    fn exact(value: i32) -> Self {
        let value = i64::from(value);
        Self {
            min: value,
            max: value,
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn checked_i32(min: i64, max: i64) -> Option<Self> {
        (min >= i64::from(i32::MIN) && max <= i64::from(i32::MAX)).then_some(Self { min, max })
    }
}

pub(crate) fn collect_function_return_ranges(module: &Module) -> HashMap<String, IntRange> {
    let mut summaries = HashMap::new();
    for _ in 0..=module.funcs.len() {
        let mut changed = false;
        for func in &module.funcs {
            if func.ret != Type::I32 || summaries.contains_key(&func.name) {
                continue;
            }
            let ranges = collect_value_ranges(func, &summaries);
            let mut returned = None;
            let mut has_return = false;
            let mut complete = true;
            for block in &func.blocks {
                let Some(Terminator::Return(Some(value))) = block.terminator.as_ref() else {
                    continue;
                };
                has_return = true;
                let Some(range) = ranges[value.0] else {
                    complete = false;
                    break;
                };
                returned = Some(returned.map_or(range, |current: IntRange| current.union(range)));
            }
            if has_return && complete {
                summaries.insert(func.name.clone(), returned.unwrap());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    summaries
}

pub(crate) fn collect_value_ranges(
    func: &Function,
    return_ranges: &HashMap<String, IntRange>,
) -> Vec<Option<IntRange>> {
    let mut ranges = vec![None; func.values.len()];
    let mut visiting = vec![false; func.values.len()];
    for index in 0..func.values.len() {
        value_range(
            func,
            ValueId(index),
            return_ranges,
            &mut ranges,
            &mut visiting,
        );
    }
    ranges
}

fn value_range(
    func: &Function,
    value: ValueId,
    return_ranges: &HashMap<String, IntRange>,
    ranges: &mut [Option<IntRange>],
    visiting: &mut [bool],
) -> Option<IntRange> {
    if ranges[value.0].is_some() || visiting[value.0] {
        return ranges[value.0];
    }
    visiting[value.0] = true;
    let range = match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(IntRange::exact(*value)),
        ValueKind::Const(Const::Bool(value)) => Some(IntRange::exact(if *value { 1 } else { 0 })),
        ValueKind::Const(Const::Zero(Type::I32 | Type::I1)) => Some(IntRange::exact(0)),
        ValueKind::Inst(block, index) => match func
            .blocks
            .get(block.0)
            .and_then(|block| block.insts.get(*index))
            .filter(|inst| inst.result == Some(value))
            .map(|inst| &inst.kind)
        {
            None => None,
            Some(kind) => match kind {
                InstKind::Phi { incomings } => {
                    union_ranges(incomings.iter().map(|(_, incoming)| {
                        value_range(func, *incoming, return_ranges, ranges, visiting)
                    }))
                }
                InstKind::Unary { op, value } => {
                    let input = value_range(func, *value, return_ranges, ranges, visiting)?;
                    match op {
                        UnaryOp::Ineg => IntRange::checked_i32(-input.max, -input.min),
                        UnaryOp::Not => Some(IntRange { min: 0, max: 1 }),
                        UnaryOp::Fneg => None,
                    }
                }
                InstKind::Binary { op, lhs, rhs } => {
                    binary_range(func, *op, *lhs, *rhs, return_ranges, ranges, visiting)
                }
                InstKind::Icmp { .. } => Some(IntRange { min: 0, max: 1 }),
                InstKind::Cast { op, value } => match op {
                    CastOp::BoolToI32 | CastOp::I32ToBool | CastOp::F32ToBool => {
                        Some(IntRange { min: 0, max: 1 })
                    }
                    CastOp::F32ToI32 => None,
                    CastOp::I32ToF32 => {
                        let _ = value;
                        None
                    }
                },
                InstKind::Call { name, .. } => return_ranges.get(name).copied(),
                InstKind::Nop
                | InstKind::Alloca { .. }
                | InstKind::Load { .. }
                | InstKind::Store { .. }
                | InstKind::MemZero { .. }
                | InstKind::MemCopy { .. }
                | InstKind::Fcmp { .. }
                | InstKind::Gep { .. } => None,
            },
        },
        ValueKind::Param
        | ValueKind::Global(_)
        | ValueKind::Const(Const::Float(_) | Const::Zero(_) | Const::String(_) | Const::Array(_)) => {
            None
        }
    };
    visiting[value.0] = false;
    ranges[value.0] = range;
    range
}

fn binary_range(
    func: &Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
    return_ranges: &HashMap<String, IntRange>,
    ranges: &mut [Option<IntRange>],
    visiting: &mut [bool],
) -> Option<IntRange> {
    if op == BinaryOp::Imod {
        let divisor = const_i32(func, rhs)?;
        if divisor == 0 {
            return None;
        }
        let bound = i64::from(divisor).abs();
        return Some(
            match value_range(func, lhs, return_ranges, ranges, visiting) {
                Some(range) if range.min > -bound && range.max < bound => range,
                Some(range) if range.min >= 0 => IntRange {
                    min: 0,
                    max: bound - 1,
                },
                Some(range) if range.max <= 0 => IntRange {
                    min: -bound + 1,
                    max: 0,
                },
                Some(_) | None => IntRange {
                    min: -bound + 1,
                    max: bound - 1,
                },
            },
        );
    }

    let lhs = value_range(func, lhs, return_ranges, ranges, visiting)?;
    let rhs = value_range(func, rhs, return_ranges, ranges, visiting)?;
    match op {
        BinaryOp::Iadd => IntRange::checked_i32(lhs.min + rhs.min, lhs.max + rhs.max),
        BinaryOp::Isub => IntRange::checked_i32(lhs.min - rhs.max, lhs.max - rhs.min),
        BinaryOp::Imul => {
            let products = [
                lhs.min * rhs.min,
                lhs.min * rhs.max,
                lhs.max * rhs.min,
                lhs.max * rhs.max,
            ];
            IntRange::checked_i32(
                *products.iter().min().unwrap(),
                *products.iter().max().unwrap(),
            )
        }
        BinaryOp::And | BinaryOp::Or => Some(IntRange { min: 0, max: 1 }),
        BinaryOp::Idiv
        | BinaryOp::Imod
        | BinaryOp::Iand
        | BinaryOp::Ior
        | BinaryOp::Ixor
        | BinaryOp::Ishl
        | BinaryOp::Iashr
        | BinaryOp::Fadd
        | BinaryOp::Fsub
        | BinaryOp::Fmul
        | BinaryOp::Fdiv => None,
    }
}

fn union_ranges(ranges: impl Iterator<Item = Option<IntRange>>) -> Option<IntRange> {
    let mut union = None;
    for range in ranges {
        let range = range?;
        union = Some(union.map_or(range, |current: IntRange| current.union(range)));
    }
    union
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        ValueKind::Const(Const::Zero(Type::I32)) => Some(0),
        _ => None,
    }
}
