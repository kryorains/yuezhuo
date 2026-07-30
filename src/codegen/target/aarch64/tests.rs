use super::emit_ir_asm;
use super::float_regs::AArch64FloatRegs;
use crate::ir::{BinaryOp, CastOp, Function, InstKind, Module, Terminator, Type};

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
fn keeps_straight_line_float_intermediates_in_registers() {
    let mut function = Function::new("float_expr", Type::I32);
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
    let result = function
        .append_inst(
            function.entry,
            InstKind::Cast {
                op: CastOp::F32ToI32,
                value: sum,
            },
            Some(Type::I32),
        )
        .unwrap();
    function.set_terminator(function.entry, Terminator::Return(Some(result)));

    let regs = AArch64FloatRegs::new(&function);
    let product_reg = regs.reg(product).unwrap();
    let sum_reg = regs.reg(sum).unwrap();
    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let body = block_asm(function_asm(&asm, "float_expr"), "float_expr", 0);

    assert!(body.contains(&format!("fmul {product_reg},")));
    assert!(body.contains(&format!("fadd {sum_reg},")));
    assert!(!body.contains(&format!("fmov {product_reg}, s0")));
    assert!(!body.contains(&format!("fmov {sum_reg}, s0")));
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
