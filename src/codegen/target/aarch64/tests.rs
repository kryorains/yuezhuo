use super::emit_ir_asm;
use super::float_regs::AArch64FloatRegs;
use crate::ir::{BinaryOp, CastOp, Const, Function, InstKind, Module, Terminator, Type};

fn function_asm<'a>(asm: &'a str, name: &str) -> &'a str {
    let start = asm.find(&format!("{name}:\n")).unwrap();
    let rest = &asm[start..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    &rest[..end]
}

fn block_asm<'a>(function_asm: &'a str, name: &str, block: usize) -> &'a str {
    let start = function_asm
        .find(&format!(".L_{name}_bb{block}:\n"))
        .unwrap();
    let body = &function_asm[start..];
    let end = body[1..]
        .find("\n.L_")
        .map_or(body.len(), |offset| offset + 1);
    &body[..end]
}

#[test]
fn reduces_a_twice_bounded_signed_remainder_with_conditional_selects() {
    let mut reduce = Function::new("bounded_reduce", Type::I32);
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

    let mut caller = Function::new("double_bounded", Type::I32);
    let input = caller.add_param("input", Type::I32);
    let reduced = caller
        .append_inst(
            caller.entry,
            InstKind::Call {
                name: "bounded_reduce".into(),
                args: vec![input],
            },
            Some(Type::I32),
        )
        .unwrap();
    let doubled = caller
        .append_inst(
            caller.entry,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: reduced,
                rhs: reduced,
            },
            Some(Type::I32),
        )
        .unwrap();
    let divisor = caller.add_const(Const::Int(97));
    let result = caller
        .append_inst(
            caller.entry,
            InstKind::Binary {
                op: BinaryOp::Imod,
                lhs: doubled,
                rhs: divisor,
            },
            Some(Type::I32),
        )
        .unwrap();
    caller.set_terminator(caller.entry, Terminator::Return(Some(result)));

    let mut module = Module::new();
    module.add_func(reduce);
    module.add_func(caller);
    let asm = emit_ir_asm(&module);
    let caller_asm = function_asm(&asm, "double_bounded");

    assert_eq!(caller_asm.matches("  csel ").count(), 2);
    assert!(caller_asm.contains("  neg w1, w1\n"));
    assert!(!caller_asm.contains("  smull "));
}

#[test]
fn keeps_straight_line_float_intermediates_in_registers() {
    let mut function = Function::new("float_expr", Type::F32);
    let lhs = function.add_param("lhs", Type::F32);
    let rhs = function.add_param("rhs", Type::F32);
    let addend = function.add_param("addend", Type::F32);
    let product = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Fmul,
                lhs,
                rhs,
            },
            Some(Type::F32),
        )
        .unwrap();
    let sum = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: product,
                rhs: addend,
            },
            Some(Type::F32),
        )
        .unwrap();
    function.set_terminator(function.entry, Terminator::Return(Some(sum)));

    let regs = AArch64FloatRegs::new(&function);
    let product_reg = regs.reg(product).unwrap();
    let sum_reg = regs.reg(sum).unwrap();
    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let body = block_asm(function_asm(&asm, "float_expr"), "float_expr", 0);

    assert!(body.contains(&format!("fmul {product_reg},")));
    assert!(body.contains(&format!("fadd {sum_reg},")));
    assert!(!body.contains("fmadd"));
    assert!(!body.contains(&format!("fmov {product_reg}, s0")));
    assert!(!body.contains(&format!("fmov {sum_reg}, s0")));
    assert!(!function_asm(&asm, "float_expr").contains("stp x29, x30"));
}

#[test]
fn preserves_float_subtract_rounding() {
    let mut function = Function::new("float_sub_product", Type::F32);
    let lhs = function.add_param("lhs", Type::F32);
    let rhs = function.add_param("rhs", Type::F32);
    let accumulator = function.add_param("accumulator", Type::F32);
    let product = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Fmul,
                lhs,
                rhs,
            },
            Some(Type::F32),
        )
        .unwrap();
    let result = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Fsub,
                lhs: accumulator,
                rhs: product,
            },
            Some(Type::F32),
        )
        .unwrap();
    function.set_terminator(function.entry, Terminator::Return(Some(result)));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let body = function_asm(&asm, "float_sub_product");

    assert!(body.contains("fmul"));
    assert!(body.contains("fsub"));
    assert!(!body.contains("fmsub"));
    assert!(!body.contains("fnmsub"));
}

#[test]
fn lowers_signed_constant_division_and_remainder_with_magic_multiplication() {
    let mut function = Function::new("constant_divmod", Type::I32);
    let value = function.add_param("value", Type::I32);
    let seven = function.add_const(crate::ir::Const::Int(7));
    let modulus = function.add_const(crate::ir::Const::Int(1_000_000_007));
    let quotient = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Idiv,
                lhs: value,
                rhs: seven,
            },
            Some(Type::I32),
        )
        .unwrap();
    let remainder = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Imod,
                lhs: value,
                rhs: modulus,
            },
            Some(Type::I32),
        )
        .unwrap();
    let result = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: quotient,
                rhs: remainder,
            },
            Some(Type::I32),
        )
        .unwrap();
    function.set_terminator(function.entry, Terminator::Return(Some(result)));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let body = function_asm(&asm, "constant_divmod");

    assert!(body.matches("smull").count() >= 2);
    assert!(!body.contains("sdiv"));
}

#[test]
fn saves_a_call_crossing_float_in_aapcs64_callee_saved_storage() {
    let mut callee = Function::new("side_effect", Type::Void);
    callee.set_terminator(callee.entry, Terminator::Return(None));

    let mut caller = Function::new("float_caller", Type::F32);
    let value = caller.add_param("value", Type::F32);
    caller.append_inst(
        caller.entry,
        InstKind::Call {
            name: "side_effect".into(),
            args: vec![],
        },
        None,
    );
    let one = caller.add_const(crate::ir::Const::Float(1.0f32.to_bits()));
    let result = caller
        .append_inst(
            caller.entry,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: value,
                rhs: one,
            },
            Some(Type::F32),
        )
        .unwrap();
    caller.set_terminator(caller.entry, Terminator::Return(Some(result)));

    let regs = AArch64FloatRegs::new(&caller);
    assert!(regs.reg(value).is_some_and(|reg| reg.starts_with("s8")));
    let mut module = Module::new();
    module.add_func(callee);
    module.add_func(caller);
    let asm = emit_ir_asm(&module);
    let caller_asm = function_asm(&asm, "float_caller");

    assert!(caller_asm.contains("str d8, [sp"));
    assert!(caller_asm.contains("ldr d8, [sp"));
    assert!(caller_asm.contains("bl side_effect"));
}

#[test]
fn emits_assigned_float_loads_and_stores_directly() {
    let mut function = Function::new("float_memory", Type::I32);
    let ptr = function.add_param("ptr", Type::Ptr(Box::new(Type::F32)));
    let value = function.add_param("value", Type::F32);
    let body_block = function.add_block("body");
    function.set_terminator(function.entry, Terminator::Jump(body_block));
    function.append_inst(body_block, InstKind::Store { ptr, value }, None);
    let loaded = function
        .append_inst(body_block, InstKind::Load { ptr }, Some(Type::F32))
        .unwrap();
    let result = function
        .append_inst(
            body_block,
            InstKind::Cast {
                op: CastOp::F32ToI32,
                value: loaded,
            },
            Some(Type::I32),
        )
        .unwrap();
    function.set_terminator(body_block, Terminator::Return(Some(result)));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let body = block_asm(
        function_asm(&asm, "float_memory"),
        "float_memory",
        body_block.0,
    );

    assert!(body
        .lines()
        .any(|line| line.starts_with("  str s") && line.ends_with("]")));
    assert!(body
        .lines()
        .any(|line| line.starts_with("  ldr s") && line.ends_with("]")));
    assert!(!body.contains("fmov w0"));
}

#[test]
fn folds_nonzero_float_geps_into_memory_operands() {
    let mut function = Function::new("folded_float_memory", Type::I32);
    let base = function.add_param("base", Type::Ptr(Box::new(Type::F32)));
    let value = function.add_param("value", Type::F32);
    let index = function.add_const(Const::Int(2));
    let body_block = function.add_block("body");
    function.set_terminator(function.entry, Terminator::Jump(body_block));
    let ptr = function
        .append_inst(
            body_block,
            InstKind::Gep {
                base,
                indices: vec![index],
            },
            Some(Type::Ptr(Box::new(Type::F32))),
        )
        .unwrap();
    function.append_inst(body_block, InstKind::Store { ptr, value }, None);
    let loaded = function
        .append_inst(body_block, InstKind::Load { ptr }, Some(Type::F32))
        .unwrap();
    let result = function
        .append_inst(
            body_block,
            InstKind::Cast {
                op: CastOp::F32ToI32,
                value: loaded,
            },
            Some(Type::I32),
        )
        .unwrap();
    function.set_terminator(body_block, Terminator::Return(Some(result)));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let body = block_asm(
        function_asm(&asm, "folded_float_memory"),
        "folded_float_memory",
        body_block.0,
    );

    assert!(body
        .lines()
        .any(|line| line.starts_with("  stur s") && line.ends_with(", #8]")));
    assert!(body
        .lines()
        .any(|line| line.starts_with("  ldur s") && line.ends_with(", #8]")));
    assert!(!body.lines().any(|line| line.starts_with("  add x")));
}
