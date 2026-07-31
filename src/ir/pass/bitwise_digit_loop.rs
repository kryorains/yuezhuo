use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_const_i32_trip_count, analyze_i32_induction, LoopInfo};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};

/// Replaces a non-negative, fixed-width digit-by-digit bitwise computation
/// with the native integer operation while retaining the original loop as a
/// fallback for signed inputs.
///
/// SysY has no source-level integer bitwise operators, so programs commonly
/// spell AND/OR/XOR as 32 iterations of `% 2`, `/ 2`, and a doubling power.
/// For non-negative i32 inputs that computation is exactly the corresponding
/// native operation. Negative inputs use truncating division and have different
/// semantics, hence the explicit runtime guard and untouched fallback CFG.
pub(super) struct BitwiseDigitLoopPass;

impl BitwiseDigitLoopPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for BitwiseDigitLoopPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            let Some(op) = recognize_bitwise_digit_loop(func) else {
                continue;
            };
            add_guarded_fast_path(func, op);
            if let Err(errors) = func.verify() {
                panic!(
                    "guarded bitwise digit loop produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

fn recognize_bitwise_digit_loop(func: &Function) -> Option<BinaryOp> {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func
            .params
            .iter()
            .any(|param| func.value(*param).ty != Type::I32)
        || func.blocks.len() > 32
        || func.values.len() > 256
    {
        return None;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let [natural_loop] = loop_info.loops() else {
        return None;
    };
    let entering_pred = natural_loop.unique_entering_pred?;
    let latch = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    if natural_loop.blocks.len() > 16
        || !matches!(func.block(exit).terminator, Some(Terminator::Return(_)))
        || loop_has_unsupported_effects(func, &natural_loop.blocks)
    {
        return None;
    }

    let phis = func
        .block(natural_loop.header)
        .insts
        .iter()
        .filter_map(|inst| match inst.kind {
            InstKind::Phi { .. } => inst.result,
            _ => None,
        })
        .collect::<Vec<_>>();
    if phis.len() != 5 {
        return None;
    }

    let counter = phis.iter().find_map(|phi| {
        let induction = analyze_i32_induction(func, natural_loop, *phi)?;
        (analyze_const_i32_trip_count(func, natural_loop, induction) == Some(32)).then_some(*phi)
    })?;

    let mut digit_phis = Vec::new();
    for param in &func.params {
        let candidates = phis
            .iter()
            .copied()
            .filter(|phi| {
                phi_incoming(func, *phi, entering_pred) == Some(*param)
                    && phi_incoming(func, *phi, latch)
                        .is_some_and(|next| is_divide_by_two(func, next, *phi))
            })
            .collect::<Vec<_>>();
        let [candidate] = candidates.as_slice() else {
            return None;
        };
        digit_phis.push(*candidate);
    }
    if digit_phis[0] == digit_phis[1] {
        return None;
    }

    let power = phis.iter().copied().find(|phi| {
        phi_incoming(func, *phi, entering_pred).is_some_and(|value| is_const_i32(func, value, 1))
            && phi_incoming(func, *phi, latch)
                .is_some_and(|next| is_multiply_by_two(func, next, *phi))
    })?;

    let returned = match func.block(exit).terminator.as_ref()? {
        Terminator::Return(Some(value)) => *value,
        _ => return None,
    };
    let result = phis.iter().copied().find(|phi| {
        *phi == returned
            && phi_incoming(func, *phi, entering_pred)
                .is_some_and(|value| is_const_i32(func, value, 0))
    })?;

    let expected = [counter, digit_phis[0], digit_phis[1], power, result];
    if phis.iter().any(|phi| !expected.contains(phi)) {
        return None;
    }

    let mut truth = [false; 4];
    for lhs in 0..=1 {
        for rhs in 0..=1 {
            let next = evaluate_one_iteration(
                func,
                natural_loop.header,
                latch,
                &[
                    (digit_phis[0], lhs),
                    (digit_phis[1], rhs),
                    (counter, 1),
                    (power, 1),
                    (result, 0),
                    (func.params[0], lhs),
                    (func.params[1], rhs),
                ],
                result,
            )?;
            if !matches!(next, 0 | 1) {
                return None;
            }
            truth[(lhs * 2 + rhs) as usize] = next == 1;
        }
    }

    match truth {
        [false, false, false, true] => Some(BinaryOp::Iand),
        [false, true, true, true] => Some(BinaryOp::Ior),
        [false, true, true, false] => Some(BinaryOp::Ixor),
        _ => None,
    }
}

fn loop_has_unsupported_effects(
    func: &Function,
    blocks: &std::collections::HashSet<BlockId>,
) -> bool {
    blocks.iter().any(|block| {
        func.block(*block).insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::Load { .. }
                    | InstKind::Store { .. }
                    | InstKind::MemZero { .. }
                    | InstKind::Call { .. }
                    | InstKind::Alloca { .. }
                    | InstKind::Gep { .. }
                    | InstKind::Unary { .. }
                    | InstKind::Cast { .. }
                    | InstKind::Fcmp { .. }
            )
        })
    })
}

fn evaluate_one_iteration(
    func: &Function,
    header: BlockId,
    latch: BlockId,
    seeded: &[(ValueId, i32)],
    result_phi: ValueId,
) -> Option<i32> {
    let mut values = vec![None; func.values.len()];
    for (index, value) in func.values.iter().enumerate() {
        values[index] = match &value.kind {
            ValueKind::Const(Const::Int(value)) => Some(*value),
            ValueKind::Const(Const::Bool(value)) => Some(i32::from(*value)),
            ValueKind::Const(Const::Zero(Type::I32 | Type::I1)) => Some(0),
            _ => None,
        };
    }
    for (value, seeded) in seeded {
        values[value.0] = Some(*seeded);
    }

    let mut block = header;
    let mut predecessor = None;
    for _ in 0..64 {
        for inst in &func.block(block).insts {
            let Some(result) = inst.result else {
                if !matches!(inst.kind, InstKind::Nop) {
                    return None;
                }
                continue;
            };
            if block == header && matches!(inst.kind, InstKind::Phi { .. }) {
                values[result.0]?;
                continue;
            }
            values[result.0] = Some(match &inst.kind {
                InstKind::Nop => continue,
                InstKind::Phi { incomings } => {
                    let pred = predecessor?;
                    let incoming = incomings
                        .iter()
                        .find_map(|(from, value)| (*from == pred).then_some(*value))?;
                    value_of(func, &values, incoming)?
                }
                InstKind::Binary { op, lhs, rhs } => eval_binary(
                    *op,
                    value_of(func, &values, *lhs)?,
                    value_of(func, &values, *rhs)?,
                )?,
                InstKind::Icmp { op, lhs, rhs } => i32::from(eval_cmp(
                    *op,
                    value_of(func, &values, *lhs)?,
                    value_of(func, &values, *rhs)?,
                )),
                _ => return None,
            });
        }

        let next = match func.block(block).terminator.as_ref()? {
            Terminator::Jump(target) => *target,
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                if value_of(func, &values, *cond)? != 0 {
                    *then_target
                } else {
                    *else_target
                }
            }
            Terminator::Return(_) => return None,
        };
        if next == header {
            if block != latch {
                return None;
            }
            let incoming = phi_incoming(func, result_phi, latch)?;
            return value_of(func, &values, incoming);
        }
        predecessor = Some(block);
        block = next;
    }
    None
}

fn eval_binary(op: BinaryOp, lhs: i32, rhs: i32) -> Option<i32> {
    Some(match op {
        BinaryOp::Iadd => lhs.wrapping_add(rhs),
        BinaryOp::Isub => lhs.wrapping_sub(rhs),
        BinaryOp::Imul => lhs.wrapping_mul(rhs),
        BinaryOp::Idiv => lhs.overflowing_div(rhs).0,
        BinaryOp::Imod => lhs.overflowing_rem(rhs).0,
        BinaryOp::Iand => lhs & rhs,
        BinaryOp::Ior => lhs | rhs,
        BinaryOp::Ixor => lhs ^ rhs,
        BinaryOp::Ishl => lhs.wrapping_shl((rhs as u32) & 31),
        BinaryOp::Iashr => lhs >> ((rhs as u32) & 31),
        BinaryOp::And => i32::from(lhs != 0 && rhs != 0),
        BinaryOp::Or => i32::from(lhs != 0 || rhs != 0),
        BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => return None,
    })
}

fn eval_cmp(op: CmpOp, lhs: i32, rhs: i32) -> bool {
    match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
    }
}

fn add_guarded_fast_path(func: &mut Function, op: BinaryOp) {
    let fallback = func.entry;
    let guard = func.add_block("bitwise.fast.guard");
    let fast = func.add_block("bitwise.fast");
    let zero = func.add_const(Const::Int(0));
    let lhs_non_negative = func
        .append_inst(
            guard,
            InstKind::Icmp {
                op: CmpOp::Ge,
                lhs: func.params[0],
                rhs: zero,
            },
            Some(Type::I1),
        )
        .unwrap();
    let rhs_non_negative = func
        .append_inst(
            guard,
            InstKind::Icmp {
                op: CmpOp::Ge,
                lhs: func.params[1],
                rhs: zero,
            },
            Some(Type::I1),
        )
        .unwrap();
    let both_non_negative = func
        .append_inst(
            guard,
            InstKind::Binary {
                op: BinaryOp::And,
                lhs: lhs_non_negative,
                rhs: rhs_non_negative,
            },
            Some(Type::I1),
        )
        .unwrap();
    let result = func
        .append_inst(
            fast,
            InstKind::Binary {
                op,
                lhs: func.params[0],
                rhs: func.params[1],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(
        guard,
        Terminator::Branch {
            cond: both_non_negative,
            then_target: fast,
            else_target: fallback,
        },
    );
    func.set_terminator(fast, Terminator::Return(Some(result)));
    func.entry = guard;
}

fn is_divide_by_two(func: &Function, value: ValueId, input: ValueId) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Binary {
            op: BinaryOp::Idiv,
            lhs,
            rhs,
        }) if *lhs == input && is_const_i32(func, *rhs, 2)
    )
}

fn is_multiply_by_two(func: &Function, value: ValueId, input: ValueId) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Binary {
            op: BinaryOp::Imul,
            lhs,
            rhs,
        }) if (*lhs == input && is_const_i32(func, *rhs, 2))
            || (*rhs == input && is_const_i32(func, *lhs, 2))
    )
}

fn phi_incoming(func: &Function, phi: ValueId, pred: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    incomings
        .iter()
        .find_map(|(from, value)| (*from == pred).then_some(*value))
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, index) = func.values.get(value.0)?.kind else {
        return None;
    };
    Some(&func.blocks.get(block.0)?.insts.get(index)?.kind)
}

fn is_const_i32(func: &Function, value: ValueId, expected: i32) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Const(Const::Int(value))) if *value == expected
    ) || (expected == 0
        && matches!(
            func.values.get(value.0).map(|value| &value.kind),
            Some(ValueKind::Const(Const::Zero(Type::I32)))
        ))
}

fn value_of(func: &Function, values: &[Option<i32>], value: ValueId) -> Option<i32> {
    values.get(value.0).copied().flatten().or_else(|| {
        match func.values.get(value.0).map(|value| &value.kind) {
            Some(ValueKind::Const(Const::Int(value))) => Some(*value),
            Some(ValueKind::Const(Const::Bool(value))) => Some(i32::from(*value)),
            Some(ValueKind::Const(Const::Zero(Type::I32 | Type::I1))) => Some(0),
            _ => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    #[test]
    fn recognizes_all_three_digit_by_digit_operations_without_using_names() {
        for (condition, expected) in [
            ("bit_a == 1 && bit_b == 1", BinaryOp::Iand),
            ("bit_a == 1 || bit_b == 1", BinaryOp::Ior),
            ("bit_a != bit_b", BinaryOp::Ixor),
        ] {
            let source = format!(
                r#"
int arbitrary(int left, int right) {{
    int len = 32;
    int result = 0;
    int power = 1;
    while (len) {{
        int bit_a = left % 2;
        int bit_b = right % 2;
        left = left / 2;
        right = right / 2;
        if ({condition}) result = result + power;
        power = power * 2;
        len = len - 1;
    }}
    return result;
}}
int main() {{ return arbitrary(6, 3); }}
"#
            );
            let mut module = lower_program(&Parser::new(&source).parse_program()).unwrap();
            super::super::ScalarPromotePass::new().run(&mut module);
            super::super::DcePass::new().run(&mut module);
            let func = &module.funcs[0];
            assert_eq!(recognize_bitwise_digit_loop(func), Some(expected));

            BitwiseDigitLoopPass::new().run(&mut module);
            let func = &module.funcs[0];
            assert!(func.blocks.iter().any(|block| {
                block
                    .insts
                    .iter()
                    .any(|inst| matches!(inst.kind, InstKind::Binary { op, .. } if op == expected))
            }));
            assert!(func.blocks.iter().any(|block| {
                block.insts.iter().any(|inst| {
                    matches!(
                        inst.kind,
                        InstKind::Binary {
                            op: BinaryOp::Idiv,
                            ..
                        }
                    )
                })
            }));
        }
    }

    #[test]
    fn rejects_a_non_word_width_loop() {
        let source = r#"
int combine(int a, int b) {
    int len = 31;
    int result = 0;
    int power = 1;
    while (len) {
        int bit_a = a % 2;
        int bit_b = b % 2;
        a = a / 2;
        b = b / 2;
        if (bit_a != bit_b) result = result + power;
        power = power * 2;
        len = len - 1;
    }
    return result;
}
"#;
        let mut module = lower_program(&Parser::new(source).parse_program()).unwrap();
        super::super::ScalarPromotePass::new().run(&mut module);
        super::super::DcePass::new().run(&mut module);
        assert_eq!(recognize_bitwise_digit_loop(&module.funcs[0]), None);
    }
}
