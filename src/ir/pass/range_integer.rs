use super::util::{const_i32, get_or_add_i32_const, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::int_range::{collect_function_return_ranges, collect_value_ranges};
use crate::ir::{BinaryOp, InstKind, Module};

pub(super) struct RangeIntegerSimplifyPass;

impl RangeIntegerSimplifyPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RangeIntegerSimplifyPass {
    fn run(&mut self, module: &mut Module) {
        let return_ranges = collect_function_return_ranges(module);
        for func in &mut module.funcs {
            let ranges = collect_value_ranges(func, &return_ranges);
            let mut replacements = ValueReplacements::new();
            let mut removed = Vec::new();
            let mut zero_results = Vec::new();

            for (block_idx, block) in func.blocks.iter().enumerate() {
                for (inst_idx, inst) in block.insts.iter().enumerate() {
                    let (Some(result), InstKind::Binary { op, lhs, rhs }) =
                        (inst.result, &inst.kind)
                    else {
                        continue;
                    };
                    if !matches!(op, BinaryOp::Idiv | BinaryOp::Imod) {
                        continue;
                    }
                    let Some(divisor) = const_i32(func, *rhs).filter(|divisor| *divisor != 0)
                    else {
                        continue;
                    };
                    let Some(range) = ranges[lhs.0] else {
                        continue;
                    };
                    let bound = i64::from(divisor).abs();
                    if range.min > -bound && range.max < bound {
                        if *op == BinaryOp::Imod {
                            replacements.insert(result, *lhs);
                        } else {
                            zero_results.push(result);
                        }
                        removed.push((block_idx, inst_idx));
                    }
                }
            }

            if !zero_results.is_empty() {
                let zero = get_or_add_i32_const(func, 0);
                replacements.extend(zero_results.into_iter().map(|result| (result, zero)));
            }
            if removed.is_empty() {
                continue;
            }
            rewrite_function_uses(func, &replacements);
            for (block_idx, inst_idx) in removed {
                let inst = &mut func.blocks[block_idx].insts[inst_idx];
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
            if let Err(errors) = func.verify() {
                panic!(
                    "range integer simplification produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Function, Terminator, Type};

    #[test]
    fn removes_remainder_of_a_bounded_call_result() {
        let mut reduce = Function::new("reduce", Type::I32);
        let input = reduce.add_param("input", Type::I32);
        let divisor = reduce.add_const(Const::Int(97));
        let reduced = reduce
            .append_inst(
                reduce.entry,
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs: input,
                    rhs: divisor,
                },
                Some(Type::I32),
            )
            .unwrap();
        reduce.set_terminator(reduce.entry, Terminator::Return(Some(reduced)));

        let mut caller = Function::new("caller", Type::I32);
        let input = caller.add_param("input", Type::I32);
        let call = caller
            .append_inst(
                caller.entry,
                InstKind::Call {
                    name: "reduce".into(),
                    args: vec![input],
                },
                Some(Type::I32),
            )
            .unwrap();
        let divisor = caller.add_const(Const::Int(97));
        let redundant = caller
            .append_inst(
                caller.entry,
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs: call,
                    rhs: divisor,
                },
                Some(Type::I32),
            )
            .unwrap();
        caller.set_terminator(caller.entry, Terminator::Return(Some(redundant)));

        let mut module = Module::new();
        module.add_func(reduce);
        module.add_func(caller);
        RangeIntegerSimplifyPass::new().run(&mut module);

        assert!(matches!(
            module.funcs[1].blocks[0].insts[1].kind,
            InstKind::Nop
        ));
        assert_eq!(
            module.funcs[1].blocks[0].terminator,
            Some(Terminator::Return(Some(call)))
        );
    }

    #[test]
    fn keeps_remainder_when_the_dividend_is_outside_the_modulus() {
        let mut function = Function::new("outside", Type::I32);
        let dividend = function.add_const(Const::Int(100));
        let divisor = function.add_const(Const::Int(97));
        let remainder = function
            .append_inst(
                function.entry,
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs: dividend,
                    rhs: divisor,
                },
                Some(Type::I32),
            )
            .unwrap();
        function.set_terminator(function.entry, Terminator::Return(Some(remainder)));
        let mut module = Module::new();
        module.add_func(function);

        RangeIntegerSimplifyPass::new().run(&mut module);

        assert!(matches!(
            module.funcs[0].blocks[0].insts[0].kind,
            InstKind::Binary {
                op: BinaryOp::Imod,
                ..
            }
        ));
    }

    #[test]
    fn replaces_bounded_division_with_zero() {
        let mut reduce = Function::new("reduce_for_division", Type::I32);
        let input = reduce.add_param("input", Type::I32);
        let modulus = reduce.add_const(Const::Int(97));
        let reduced = reduce
            .append_inst(
                reduce.entry,
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs: input,
                    rhs: modulus,
                },
                Some(Type::I32),
            )
            .unwrap();
        reduce.set_terminator(reduce.entry, Terminator::Return(Some(reduced)));

        let mut caller = Function::new("divide_bounded", Type::I32);
        let input = caller.add_param("input", Type::I32);
        let call = caller
            .append_inst(
                caller.entry,
                InstKind::Call {
                    name: "reduce_for_division".into(),
                    args: vec![input],
                },
                Some(Type::I32),
            )
            .unwrap();
        let divisor = caller.add_const(Const::Int(97));
        let quotient = caller
            .append_inst(
                caller.entry,
                InstKind::Binary {
                    op: BinaryOp::Idiv,
                    lhs: call,
                    rhs: divisor,
                },
                Some(Type::I32),
            )
            .unwrap();
        caller.set_terminator(caller.entry, Terminator::Return(Some(quotient)));

        let mut module = Module::new();
        module.add_func(reduce);
        module.add_func(caller);
        RangeIntegerSimplifyPass::new().run(&mut module);

        assert!(matches!(
            module.funcs[1].blocks[0].insts[1].kind,
            InstKind::Nop
        ));
        let Some(Terminator::Return(Some(returned))) =
            module.funcs[1].blocks[0].terminator.as_ref()
        else {
            panic!("expected an integer return");
        };
        assert_eq!(
            module.funcs[1].values[returned.0].kind,
            crate::ir::ValueKind::Const(Const::Int(0))
        );
    }
}
