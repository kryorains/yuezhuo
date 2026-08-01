use super::util::{const_i32, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{BinaryOp, BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId};
use std::collections::HashSet;

const MIN_CASES: i32 = 3;
const MAX_CASES: i32 = 30;

/// Collapses a scalar dispatch over consecutive powers of two into a guarded
/// variable shift.
///
/// Frontends without source-level shifts often produce chains such as
/// `n == 1 ? x * 2 : n == 2 ? x * 4 : ... : x`. The left-shift form is exact
/// modulo i32. Signed division needs a sign-dependent bias before arithmetic
/// shift to preserve truncation toward zero. Values outside the recognized
/// selector interval retain the original fallback result.
pub(super) struct GuardedShiftDispatchPass;

impl GuardedShiftDispatchPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for GuardedShiftDispatchPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            let Some(dispatch) = recognize_dispatch(func) else {
                continue;
            };
            apply_dispatch(func, dispatch);
            if let Err(errors) = func.verify() {
                panic!(
                    "guarded shift dispatch produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShiftDispatch {
    data: ValueId,
    selector: ValueId,
    operation: BinaryOp,
    cases: i32,
}

fn recognize_dispatch(func: &Function) -> Option<ShiftDispatch> {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func
            .params
            .iter()
            .any(|param| func.value(*param).ty != Type::I32)
        || func.blocks.len() > 2 * (MAX_CASES as usize) + 1
        || func.values.len() > 512
    {
        return None;
    }

    let mut visited = HashSet::new();
    let mut current = func.entry;
    let mut selector = None;
    let mut data = None;
    let mut operation = None;
    let mut expected_case = 1i32;

    loop {
        if !visited.insert(current) {
            return None;
        }
        let block = func.block(current);
        let Some((condition, compared, case)) = dispatch_condition(func, current) else {
            break;
        };
        if case != expected_case || case > MAX_CASES {
            return None;
        }
        let selected = selector.get_or_insert(compared);
        if *selected != compared || !func.params.contains(selected) {
            return None;
        }
        let Terminator::Branch {
            cond,
            then_target,
            else_target,
        } = block.terminator.as_ref()?
        else {
            return None;
        };
        if *cond != condition {
            return None;
        }

        let selected_data = func
            .params
            .iter()
            .copied()
            .find(|param| *param != compared)?;
        let expected_power = 1i32.checked_shl(case as u32)?;
        let (case_target, next_target, case_operation) = if let Some(case_operation) =
            case_return(func, *then_target, selected_data, expected_power)
        {
            (*then_target, *else_target, case_operation)
        } else if let Some(case_operation) =
            case_return(func, *else_target, selected_data, expected_power)
        {
            (*else_target, *then_target, case_operation)
        } else {
            return None;
        };
        if !visited.insert(case_target) {
            return None;
        }
        if data
            .replace(selected_data)
            .is_some_and(|old| old != selected_data)
            || operation
                .replace(case_operation)
                .is_some_and(|old| old != case_operation)
        {
            return None;
        }

        current = next_target;
        expected_case += 1;
    }

    let cases = expected_case - 1;
    if cases < MIN_CASES
        || !fallback_return(func, current, data?)
        || visited.len() != func.blocks.len()
    {
        return None;
    }
    Some(ShiftDispatch {
        data: data?,
        selector: selector?,
        operation: operation?,
        cases,
    })
}

fn dispatch_condition(func: &Function, block: BlockId) -> Option<(ValueId, ValueId, i32)> {
    let active = func
        .block(block)
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .collect::<Vec<_>>();
    let [compare] = active.as_slice() else {
        return None;
    };
    let (
        Some(result),
        InstKind::Icmp {
            op: CmpOp::Eq,
            lhs,
            rhs,
        },
    ) = (compare.result, &compare.kind)
    else {
        return None;
    };
    if let Some(case) = const_i32(func, *rhs) {
        Some((result, *lhs, case))
    } else {
        Some((result, *rhs, const_i32(func, *lhs)?))
    }
}

fn case_return(
    func: &Function,
    block: BlockId,
    data: ValueId,
    expected_power: i32,
) -> Option<BinaryOp> {
    let active = func
        .block(block)
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .collect::<Vec<_>>();
    let [binary] = active.as_slice() else {
        return None;
    };
    let (Some(result), InstKind::Binary { op, lhs, rhs }) = (binary.result, &binary.kind) else {
        return None;
    };
    if func.block(block).terminator != Some(Terminator::Return(Some(result))) {
        return None;
    }
    match op {
        BinaryOp::Imul
            if (*lhs == data && const_i32(func, *rhs) == Some(expected_power))
                || (*rhs == data && const_i32(func, *lhs) == Some(expected_power)) =>
        {
            Some(BinaryOp::Imul)
        }
        BinaryOp::Idiv if *lhs == data && const_i32(func, *rhs) == Some(expected_power) => {
            Some(BinaryOp::Idiv)
        }
        _ => None,
    }
}

fn fallback_return(func: &Function, block: BlockId, data: ValueId) -> bool {
    func.block(block)
        .insts
        .iter()
        .all(|inst| matches!(inst.kind, InstKind::Nop))
        && func.block(block).terminator == Some(Terminator::Return(Some(data)))
}

fn apply_dispatch(func: &mut Function, dispatch: ShiftDispatch) {
    let entry = func.entry;
    for inst in &mut func.blocks[entry.0].insts {
        inst.result = None;
        inst.kind = InstKind::Nop;
    }

    let upper_check = func.add_block("shift.dispatch.upper");
    let fast = func.add_block("shift.dispatch.fast");
    let fallback = func.add_block("shift.dispatch.fallback");
    let one = get_or_add_i32_const(func, 1);
    let cases = get_or_add_i32_const(func, dispatch.cases);
    let below_range = func
        .append_inst(
            entry,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: dispatch.selector,
                rhs: one,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.blocks[entry.0].terminator = Some(Terminator::Branch {
        cond: below_range,
        then_target: fallback,
        else_target: upper_check,
    });
    let in_range = func
        .append_inst(
            upper_check,
            InstKind::Icmp {
                op: CmpOp::Le,
                lhs: dispatch.selector,
                rhs: cases,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        upper_check,
        Terminator::Branch {
            cond: in_range,
            then_target: fast,
            else_target: fallback,
        },
    );

    let result = match dispatch.operation {
        BinaryOp::Imul => func
            .append_inst(
                fast,
                InstKind::Binary {
                    op: BinaryOp::Ishl,
                    lhs: dispatch.data,
                    rhs: dispatch.selector,
                },
                Some(Type::I32),
            )
            .unwrap(),
        BinaryOp::Idiv => {
            let thirty_one = get_or_add_i32_const(func, 31);
            let sign = func
                .append_inst(
                    fast,
                    InstKind::Binary {
                        op: BinaryOp::Iashr,
                        lhs: dispatch.data,
                        rhs: thirty_one,
                    },
                    Some(Type::I32),
                )
                .unwrap();
            let power = func
                .append_inst(
                    fast,
                    InstKind::Binary {
                        op: BinaryOp::Ishl,
                        lhs: one,
                        rhs: dispatch.selector,
                    },
                    Some(Type::I32),
                )
                .unwrap();
            let mask = func
                .append_inst(
                    fast,
                    InstKind::Binary {
                        op: BinaryOp::Isub,
                        lhs: power,
                        rhs: one,
                    },
                    Some(Type::I32),
                )
                .unwrap();
            let bias = func
                .append_inst(
                    fast,
                    InstKind::Binary {
                        op: BinaryOp::Iand,
                        lhs: sign,
                        rhs: mask,
                    },
                    Some(Type::I32),
                )
                .unwrap();
            let biased = func
                .append_inst(
                    fast,
                    InstKind::Binary {
                        op: BinaryOp::Iadd,
                        lhs: dispatch.data,
                        rhs: bias,
                    },
                    Some(Type::I32),
                )
                .unwrap();
            func.append_inst(
                fast,
                InstKind::Binary {
                    op: BinaryOp::Iashr,
                    lhs: biased,
                    rhs: dispatch.selector,
                },
                Some(Type::I32),
            )
            .unwrap()
        }
        _ => unreachable!(),
    };
    func.set_terminator(fast, Terminator::Return(Some(result)));
    func.set_terminator(fallback, Terminator::Return(Some(dispatch.data)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Function};

    fn dispatch_function(operation: BinaryOp, cases: i32) -> Function {
        let mut func = Function::new("dispatch", Type::I32);
        let data = func.add_param("data", Type::I32);
        let selector = func.add_param("selector", Type::I32);
        let mut check = func.entry;
        for case in 1..=cases {
            let leaf = func.add_block(format!("case.{case}"));
            let next = func.add_block(format!("next.{case}"));
            let case_value = func.add_const(Const::Int(case));
            let condition = func
                .append_inst(
                    check,
                    InstKind::Icmp {
                        op: CmpOp::Eq,
                        lhs: selector,
                        rhs: case_value,
                    },
                    Some(Type::I1),
                )
                .unwrap();
            func.set_terminator(
                check,
                Terminator::Branch {
                    cond: condition,
                    then_target: leaf,
                    else_target: next,
                },
            );
            let power = func.add_const(Const::Int(1i32 << case));
            let result = func
                .append_inst(
                    leaf,
                    InstKind::Binary {
                        op: operation,
                        lhs: data,
                        rhs: power,
                    },
                    Some(Type::I32),
                )
                .unwrap();
            func.set_terminator(leaf, Terminator::Return(Some(result)));
            check = next;
        }
        func.set_terminator(check, Terminator::Return(Some(data)));
        func
    }

    #[test]
    fn recognizes_consecutive_multiply_and_signed_divide_dispatches() {
        for operation in [BinaryOp::Imul, BinaryOp::Idiv] {
            let mut func = dispatch_function(operation, 8);
            let dispatch = recognize_dispatch(&func).unwrap();
            assert_eq!(dispatch.operation, operation);
            assert_eq!(dispatch.cases, 8);
            apply_dispatch(&mut func, dispatch);
            assert!(func.verify().is_ok());
            assert!(func.blocks.iter().any(|block| {
                block.insts.iter().any(|inst| {
                    matches!(
                        inst.kind,
                        InstKind::Binary {
                            op: BinaryOp::Ishl,
                            ..
                        }
                    )
                })
            }));
        }
    }

    #[test]
    fn rejects_holes_and_non_identity_fallbacks() {
        let mut hole = dispatch_function(BinaryOp::Imul, 8);
        let selector = hole.params[1];
        let four = hole.add_const(Const::Int(4));
        let InstKind::Icmp { rhs, .. } = &mut hole.blocks[4].insts[0].kind else {
            panic!();
        };
        *rhs = four;
        assert!(recognize_dispatch(&hole).is_none());

        let mut fallback = dispatch_function(BinaryOp::Idiv, 8);
        let last = fallback.blocks.len() - 1;
        fallback.blocks[last].terminator = Some(Terminator::Return(Some(selector)));
        assert!(recognize_dispatch(&fallback).is_none());
    }

    #[test]
    fn signed_bias_formula_matches_truncating_division() {
        let values = [
            i32::MIN,
            i32::MIN + 1,
            -65537,
            -257,
            -3,
            -1,
            0,
            1,
            3,
            257,
            i32::MAX,
        ];
        for shift in 1..=30u32 {
            let divisor = 1i32 << shift;
            for value in values {
                let sign = value >> 31;
                let mask = divisor - 1;
                let shifted = value.wrapping_add(sign & mask) >> shift;
                assert_eq!(shifted, value / divisor);
            }
        }
    }
}
