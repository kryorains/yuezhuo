use super::emit_ir_asm;
use super::regalloc::Riscv64FloatRegAlloc;
use crate::ir::{BinaryOp, Const, Function, InstKind, Module, Terminator, Type, ValueId};

fn function_asm<'a>(asm: &'a str, name: &str) -> &'a str {
    let start = asm.find(&format!("{name}:\n")).unwrap();
    let rest = &asm[start..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn float_regalloc_api_is_available_to_target_consumers() {
    let func = Function::new("float_regalloc_visibility", Type::Void);
    let regs = Riscv64FloatRegAlloc::new(&func);

    assert_eq!(regs.reg(ValueId(0)), None);
    assert!(regs.used_callee_saved().is_empty());
}

#[test]
fn preserves_call_crossing_float_parameter() {
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
    let one = caller.add_const(Const::Float(1.0f32.to_bits()));
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

    let mut module = Module::new();
    module.add_func(callee);
    module.add_func(caller);
    let asm = emit_ir_asm(&module);
    let caller_asm = function_asm(&asm, "float_caller");

    assert!(caller_asm.contains("fmv.s fs0, fa0"));
    let frame_store = caller_asm.find("fsw fa0, -24(s0)").unwrap();
    let frame_load = caller_asm.find("lw a0, -24(s0)").unwrap();
    assert!(frame_store < frame_load);
    assert!(caller_asm.contains("fsw fs0, -8(s0)"));
    assert!(caller_asm.contains("flw fs0, -8(s0)"));
}

#[test]
fn stores_assigned_float_parameter_for_direct_return() {
    let mut function = Function::new("return_float_parameter", Type::F32);
    let value = function.add_param("value", Type::F32);
    function.set_terminator(function.entry, Terminator::Return(Some(value)));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "return_float_parameter");

    assert!(function_asm.contains("fmv.s ft1, fa0"));
    assert!(function_asm.contains("fsw fa0, -8(s0)"));
    assert!(function_asm.contains("lw a0, -8(s0)"));
}

#[test]
fn stores_assigned_stack_float_parameter_in_frame_slot() {
    let mut function = Function::new("return_ninth_float_parameter", Type::F32);
    let mut value = None;
    for index in 0..9 {
        value = Some(function.add_param(format!("value_{index}"), Type::F32));
    }
    function.set_terminator(function.entry, Terminator::Return(value));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "return_ninth_float_parameter");

    assert!(function_asm.contains("flw ft1, 16(s0)"));
    assert!(function_asm.contains("fsw ft1, -72(s0)"));
    assert!(function_asm.contains("lw a0, -72(s0)"));
}

#[test]
fn keeps_stack_slots_below_float_callee_save_area() {
    let mut callee = Function::new("side_effect", Type::Void);
    callee.set_terminator(callee.entry, Terminator::Return(None));

    let mut caller = Function::new("float_caller_with_stack_slot", Type::F32);
    let value = caller.add_param("value", Type::F32);
    caller.add_param("stack_backed", Type::I32);
    caller.append_inst(
        caller.entry,
        InstKind::Call {
            name: "side_effect".into(),
            args: vec![],
        },
        None,
    );
    let one = caller.add_const(Const::Float(1.0f32.to_bits()));
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

    let mut module = Module::new();
    module.add_func(callee);
    module.add_func(caller);
    let asm = emit_ir_asm(&module);
    let caller_asm = function_asm(&asm, "float_caller_with_stack_slot");

    assert!(caller_asm.contains("fsw fs0, -8(s0)"));
    assert!(caller_asm.contains("sw a0, -32(s0)"));
}
