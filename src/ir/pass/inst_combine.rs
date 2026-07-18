use super::util::{const_i32, defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{BinaryOp, CmpOp, Function, InstKind, Module, ValueId};

/// Canonicalizes and combines local integer instructions.
///
/// Every rule is based only on an instruction and the definitions of its
/// operands. In particular, this pass does not inspect source names, block
/// names, or whole-function control-flow shapes.
pub(super) struct InstCombinePass;

impl InstCombinePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for InstCombinePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            combine_function(func);
        }
    }
}

fn combine_function(func: &mut Function) {
    // Reassociation can expose another constant-bearing definition, so keep
    // scanning until every local expression reaches its canonical form.
    loop {
        let mut changed = false;
        for block_idx in 0..func.blocks.len() {
            for inst_idx in 0..func.blocks[block_idx].insts.len() {
                let old_kind = func.blocks[block_idx].insts[inst_idx].kind.clone();
                let new_kind = combine_inst(func, &old_kind);
                if new_kind != old_kind {
                    func.blocks[block_idx].insts[inst_idx].kind = new_kind;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    if let Err(errors) = func.verify() {
        panic!(
            "instruction combining produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn combine_inst(func: &mut Function, kind: &InstKind) -> InstKind {
    match *kind {
        InstKind::Binary { op, lhs, rhs } => combine_binary(func, op, lhs, rhs),
        InstKind::Icmp { op, lhs, rhs }
            if const_i32(func, lhs).is_some() && const_i32(func, rhs).is_none() =>
        {
            InstKind::Icmp {
                op: reverse_cmp(op),
                lhs: rhs,
                rhs: lhs,
            }
        }
        _ => kind.clone(),
    }
}

fn combine_binary(
    func: &mut Function,
    mut op: BinaryOp,
    mut lhs: ValueId,
    mut rhs: ValueId,
) -> InstKind {
    (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);

    // i32 arithmetic wraps, so doubling is exactly multiplication by two.
    if op == BinaryOp::Iadd && lhs == rhs {
        op = BinaryOp::Imul;
        rhs = get_or_add_i32_const(func, 2);
        (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);
    }

    if op == BinaryOp::Isub {
        if let Some(divisor) = match_remainder_divisor(func, lhs, rhs) {
            return InstKind::Binary {
                op: BinaryOp::Imod,
                lhs,
                rhs: divisor,
            };
        }
    }

    if matches!(op, BinaryOp::Iadd | BinaryOp::Imul) {
        if let Some((base, constant)) = reassociate_constants(func, op, lhs, rhs) {
            lhs = base;
            rhs = constant;
            (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);
        }
    }

    InstKind::Binary { op, lhs, rhs }
}

/// Keeps constants on the right and otherwise orders operands by ValueId.
/// Floating-point and boolean operations are deliberately excluded.
fn canonicalize_integer_operands(
    func: &Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> (ValueId, ValueId) {
    if !is_commutative_integer_op(op) {
        return (lhs, rhs);
    }

    let lhs_is_const = const_i32(func, lhs).is_some();
    let rhs_is_const = const_i32(func, rhs).is_some();
    let should_swap = match (lhs_is_const, rhs_is_const) {
        (true, false) => true,
        (false, true) => false,
        _ => rhs.0 < lhs.0,
    };
    if should_swap {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn is_commutative_integer_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Iadd | BinaryOp::Imul | BinaryOp::Iand | BinaryOp::Ior | BinaryOp::Ixor
    )
}

fn reassociate_constants(
    func: &mut Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<(ValueId, ValueId)> {
    let outer_constant = const_i32(func, rhs)?;
    let InstKind::Binary {
        op: inner_op,
        lhs: inner_lhs,
        rhs: inner_rhs,
    } = defining_inst(func, lhs)?.clone()
    else {
        return None;
    };
    if inner_op != op {
        return None;
    }

    let (base, inner_constant) = one_constant_operand(func, inner_lhs, inner_rhs)?;
    if base == lhs {
        return None;
    }
    let combined = match op {
        BinaryOp::Iadd => inner_constant.wrapping_add(outer_constant),
        BinaryOp::Imul => inner_constant.wrapping_mul(outer_constant),
        _ => return None,
    };
    Some((base, get_or_add_i32_const(func, combined)))
}

fn one_constant_operand(func: &Function, lhs: ValueId, rhs: ValueId) -> Option<(ValueId, i32)> {
    match (const_i32(func, lhs), const_i32(func, rhs)) {
        (Some(constant), None) => Some((rhs, constant)),
        (None, Some(constant)) => Some((lhs, constant)),
        _ => None,
    }
}

/// Matches `x - (x / d) * d` through direct use-def edges.
///
/// Both occurrences of `d` must be known i32 constants with the same nonzero
/// value, and the division must use the exact subtraction dividend.
fn match_remainder_divisor(
    func: &Function,
    dividend: ValueId,
    product: ValueId,
) -> Option<ValueId> {
    let InstKind::Binary {
        op: BinaryOp::Imul,
        lhs: product_lhs,
        rhs: product_rhs,
    } = defining_inst(func, product)?
    else {
        return None;
    };

    for (quotient, factor) in [(*product_lhs, *product_rhs), (*product_rhs, *product_lhs)] {
        let Some(factor_value) = const_i32(func, factor) else {
            continue;
        };
        if factor_value == 0 {
            continue;
        }
        let Some(InstKind::Binary {
            op: BinaryOp::Idiv,
            lhs: division_dividend,
            rhs: divisor,
        }) = defining_inst(func, quotient)
        else {
            continue;
        };
        if *division_dividend == dividend && const_i32(func, *divisor) == Some(factor_value) {
            return Some(*divisor);
        }
    }
    None
}

fn reverse_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, ModulePass, OptLevel, PassOptions};
    use super::{const_i32, defining_inst, InstCombinePass};
    use crate::ir::{
        self, BinaryOp, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId,
        ValueKind,
    };
    use crate::parser::Parser;

    fn append_binary(func: &mut Function, op: BinaryOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        func.append_inst(
            func.entry,
            InstKind::Binary { op, lhs, rhs },
            Some(Type::I32),
        )
        .unwrap()
    }

    fn run_inst_combine(func: Function) -> Function {
        let mut module = Module::new();
        module.add_func(func);
        InstCombinePass::new().run(&mut module);
        module.funcs.pop().unwrap()
    }

    fn assert_binary(
        func: &Function,
        value: ValueId,
        expected_op: BinaryOp,
        expected_lhs: ValueId,
        expected_rhs: i32,
    ) {
        assert!(matches!(
            defining_inst(func, value),
            Some(InstKind::Binary { op, lhs, rhs })
                if *op == expected_op
                    && *lhs == expected_lhs
                    && const_i32(func, *rhs) == Some(expected_rhs)
        ));
    }

    #[test]
    fn combines_local_integer_instructions_by_structure() {
        let mut func = Function::new("arbitrary", Type::I32);
        let x = func.add_param("first", Type::I32);
        let y = func.add_param("second", Type::I32);
        let floating = func.add_param("floating", Type::F32);
        let two = func.add_const(Const::Int(2));
        let three = func.add_const(Const::Int(3));
        let four = func.add_const(Const::Int(4));
        let float_two = func.add_const(Const::Float(2.0f32.to_bits()));

        let inner_mul = append_binary(&mut func, BinaryOp::Imul, two, x);
        let outer_mul = append_binary(&mut func, BinaryOp::Imul, two, inner_mul);
        let inner_add = append_binary(&mut func, BinaryOp::Iadd, three, y);
        let outer_add = append_binary(&mut func, BinaryOp::Iadd, inner_add, four);
        let doubled = append_binary(&mut func, BinaryOp::Iadd, x, x);
        let xor = append_binary(&mut func, BinaryOp::Ixor, three, x);
        let quotient = append_binary(&mut func, BinaryOp::Idiv, x, three);
        let product = append_binary(&mut func, BinaryOp::Imul, three, quotient);
        let remainder = append_binary(&mut func, BinaryOp::Isub, x, product);
        let compare = func
            .append_inst(
                func.entry,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: three,
                    rhs: y,
                },
                Some(Type::I1),
            )
            .unwrap();
        let float_add = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs: float_two,
                    rhs: floating,
                },
                Some(Type::F32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(remainder)));

        let func = run_inst_combine(func);

        assert_binary(&func, inner_mul, BinaryOp::Imul, x, 2);
        assert_binary(&func, outer_mul, BinaryOp::Imul, x, 4);
        assert_binary(&func, inner_add, BinaryOp::Iadd, y, 3);
        assert_binary(&func, outer_add, BinaryOp::Iadd, y, 7);
        assert_binary(&func, doubled, BinaryOp::Imul, x, 2);
        assert_binary(&func, xor, BinaryOp::Ixor, x, 3);
        assert_binary(&func, remainder, BinaryOp::Imod, x, 3);
        assert!(matches!(
            defining_inst(&func, compare),
            Some(InstKind::Icmp {
                op: CmpOp::Gt,
                lhs,
                rhs,
            }) if *lhs == y && *rhs == three
        ));
        assert!(matches!(
            defining_inst(&func, float_add),
            Some(InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs,
                rhs,
            }) if *lhs == float_two && *rhs == floating
        ));
    }

    #[test]
    fn rejects_zero_divisors_and_different_dividends() {
        let mut func = Function::new("negative_cases", Type::I32);
        let x = func.add_param("x", Type::I32);
        let y = func.add_param("y", Type::I32);
        let zero = func.add_const(Const::Int(0));
        let three = func.add_const(Const::Int(3));

        let zero_quotient = append_binary(&mut func, BinaryOp::Idiv, x, zero);
        let zero_product = append_binary(&mut func, BinaryOp::Imul, zero_quotient, zero);
        let zero_case = append_binary(&mut func, BinaryOp::Isub, x, zero_product);
        let other_quotient = append_binary(&mut func, BinaryOp::Idiv, y, three);
        let other_product = append_binary(&mut func, BinaryOp::Imul, other_quotient, three);
        let other_case = append_binary(&mut func, BinaryOp::Isub, x, other_product);
        func.set_terminator(func.entry, Terminator::Return(Some(other_case)));

        let func = run_inst_combine(func);

        assert!(matches!(
            defining_inst(&func, zero_case),
            Some(InstKind::Binary {
                op: BinaryOp::Isub,
                ..
            })
        ));
        assert!(matches!(
            defining_inst(&func, other_case),
            Some(InstKind::Binary {
                op: BinaryOp::Isub,
                ..
            })
        ));
    }

    #[test]
    fn preserves_wrapping_execution_semantics() {
        let mut func = Function::new("execute", Type::I32);
        let x = func.add_param("input", Type::I32);
        let minus_one = func.add_const(Const::Int(-1));
        let two = func.add_const(Const::Int(2));
        let three = func.add_const(Const::Int(3));
        let four = func.add_const(Const::Int(4));

        let inner_mul = append_binary(&mut func, BinaryOp::Imul, two, x);
        let outer_mul = append_binary(&mut func, BinaryOp::Imul, inner_mul, two);
        let inner_add = append_binary(&mut func, BinaryOp::Iadd, three, x);
        let outer_add = append_binary(&mut func, BinaryOp::Iadd, inner_add, four);
        let quotient = append_binary(&mut func, BinaryOp::Idiv, x, three);
        let product = append_binary(&mut func, BinaryOp::Imul, quotient, three);
        let remainder = append_binary(&mut func, BinaryOp::Isub, x, product);
        let overflow_quotient = append_binary(&mut func, BinaryOp::Idiv, x, minus_one);
        let overflow_product =
            append_binary(&mut func, BinaryOp::Imul, overflow_quotient, minus_one);
        let overflow_remainder = append_binary(&mut func, BinaryOp::Isub, x, overflow_product);
        let scaled_sum = append_binary(&mut func, BinaryOp::Iadd, outer_mul, outer_add);
        let remainder_sum = append_binary(&mut func, BinaryOp::Iadd, remainder, overflow_remainder);
        let result = append_binary(&mut func, BinaryOp::Iadd, scaled_sum, remainder_sum);
        func.set_terminator(func.entry, Terminator::Return(Some(result)));

        let original = func.clone();
        let combined = run_inst_combine(func);
        for input in [i32::MIN, -17, -1, 0, 1, 17, i32::MAX] {
            assert_eq!(eval_i32(&original, input), eval_i32(&combined, input));
        }
    }

    #[test]
    fn o1_pipeline_combines_and_cleans_up_but_o0_does_not() {
        let source = r#"
            int nested(int value) { return (value * 2) * 2; }
            int doubled(int value) { return value + value; }
            int remainder(int value) { return value - (value / 3) * 3; }
            int compare(int value) { return 5 < value; }
        "#;
        let o0 = optimize(source, OptLevel::O0);
        let o1 = optimize(source, OptLevel::O1);

        let o0_nested = function(&o0, "nested");
        assert_eq!(count_binary(o0_nested, BinaryOp::Imul), 2);
        assert_eq!(count_binary(function(&o0, "doubled"), BinaryOp::Iadd), 1);
        assert_eq!(count_binary(function(&o0, "remainder"), BinaryOp::Imod), 0);
        assert!(matches!(
            first_icmp(function(&o0, "compare")),
            Some(InstKind::Icmp { lhs, .. }) if const_i32(function(&o0, "compare"), *lhs) == Some(5)
        ));

        let o1_nested = function(&o1, "nested");
        assert_eq!(count_binary(o1_nested, BinaryOp::Imul), 1);
        let nested_result = returned_value(o1_nested);
        assert_binary(
            o1_nested,
            nested_result,
            BinaryOp::Imul,
            o1_nested.params[0],
            4,
        );
        let doubled = function(&o1, "doubled");
        assert_binary(
            doubled,
            returned_value(doubled),
            BinaryOp::Imul,
            doubled.params[0],
            2,
        );
        let remainder = function(&o1, "remainder");
        assert_binary(
            remainder,
            returned_value(remainder),
            BinaryOp::Imod,
            remainder.params[0],
            3,
        );
        assert!(matches!(
            first_icmp(function(&o1, "compare")),
            Some(InstKind::Icmp {
                op: CmpOp::Gt,
                lhs,
                rhs,
            }) if *lhs == function(&o1, "compare").params[0]
                && const_i32(function(&o1, "compare"), *rhs) == Some(5)
        ));
        assert!(o1.funcs.iter().all(|func| func.verify().is_ok()));
    }

    fn eval_i32(func: &Function, input: i32) -> i32 {
        let mut values = vec![None; func.values.len()];
        for (idx, value) in func.values.iter().enumerate() {
            if let ValueKind::Const(Const::Int(value)) = value.kind {
                values[idx] = Some(value);
            }
        }
        values[func.params[0].0] = Some(input);

        for inst in &func.blocks[func.entry.0].insts {
            let Some(result) = inst.result else {
                continue;
            };
            let InstKind::Binary { op, lhs, rhs } = inst.kind else {
                panic!("unexpected instruction in execution test: {:?}", inst.kind);
            };
            let lhs = values[lhs.0].expect("lhs must be defined");
            let rhs = values[rhs.0].expect("rhs must be defined");
            values[result.0] = Some(match op {
                BinaryOp::Iadd => lhs.wrapping_add(rhs),
                BinaryOp::Isub => lhs.wrapping_sub(rhs),
                BinaryOp::Imul => lhs.wrapping_mul(rhs),
                BinaryOp::Idiv => lhs.wrapping_div(rhs),
                BinaryOp::Imod => lhs.wrapping_rem(rhs),
                _ => panic!("unexpected binary op in execution test: {:?}", op),
            });
        }

        let Terminator::Return(Some(result)) =
            func.blocks[func.entry.0].terminator.as_ref().unwrap()
        else {
            panic!("execution test must return i32");
        };
        values[result.0].expect("return value must be defined")
    }

    fn optimize(source: &str, opt_level: OptLevel) -> Module {
        let mut parser = Parser::new(source);
        let mut module =
            ir::lower::lower_program(&parser.parse_program()).expect("test source must lower");
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

    fn count_binary(func: &Function, expected: BinaryOp) -> usize {
        func.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter(|inst| matches!(inst.kind, InstKind::Binary { op, .. } if op == expected))
            .count()
    }

    fn returned_value(func: &Function) -> ValueId {
        func.blocks
            .iter()
            .find_map(|block| match block.terminator {
                Some(Terminator::Return(Some(value))) => Some(value),
                _ => None,
            })
            .expect("test function must return a value")
    }

    fn first_icmp(func: &Function) -> Option<&InstKind> {
        func.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .find_map(|inst| matches!(inst.kind, InstKind::Icmp { .. }).then_some(&inst.kind))
    }
}
