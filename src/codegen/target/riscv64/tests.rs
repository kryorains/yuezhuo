use super::emit_ir_asm;
use super::regalloc::Riscv64FloatRegAlloc;
use crate::codegen::common::{IrArgLocation, IrParamSig};
use crate::ir::{
    BinaryOp, CastOp, Const, Function, InstKind, Module, Terminator, Type, UnaryOp, ValueId,
};

fn function_asm<'a>(asm: &'a str, name: &str) -> &'a str {
    let start = asm.find(&format!("{name}:\n")).unwrap();
    let rest = &asm[start..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    &rest[..end]
}

fn entry_block_asm<'a>(function_asm: &'a str, name: &str) -> &'a str {
    block_asm(function_asm, name, 0)
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
fn float_regalloc_api_is_available_to_target_consumers() {
    let func = Function::new("float_regalloc_visibility", Type::Void);
    let regs = Riscv64FloatRegAlloc::new(&func);

    assert_eq!(regs.reg(ValueId(0)), None);
    assert!(regs.used_callee_saved().is_empty());
}

#[test]
fn assigns_float_overflow_to_integer_registers_before_the_stack() {
    let params = (0..17)
        .map(|_| IrParamSig {
            ty: Type::F32,
            is_pointer: false,
        })
        .collect::<Vec<_>>();

    let locations = super::abi::assign_riscv_arg_locations(&params);

    for (index, location) in locations.iter().take(8).enumerate() {
        assert!(matches!(location, IrArgLocation::FloatReg(reg) if *reg == index));
    }
    for (index, location) in locations.iter().skip(8).take(8).enumerate() {
        assert!(matches!(location, IrArgLocation::IntReg(reg) if *reg == index));
    }
    assert!(matches!(locations[16], IrArgLocation::Stack));
}

#[test]
fn does_not_add_float_frame_to_integer_only_function() {
    let mut function = Function::new("integer_only", Type::I32);
    let lhs = function.add_param("lhs", Type::I32);
    let rhs = function.add_param("rhs", Type::I32);
    let sum = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            },
            Some(Type::I32),
        )
        .unwrap();
    function.set_terminator(function.entry, Terminator::Return(Some(sum)));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "integer_only");

    assert!(function_asm.contains("addw"));
    assert!(!function_asm.contains("fsw fs"));
    assert!(!function_asm.contains("flw fs"));
}

#[test]
fn clears_aligned_local_arrays_with_wide_stores() {
    let mut function = Function::new("clear_local", Type::Void);
    let array = function
        .append_inst(
            function.entry,
            InstKind::Alloca {
                ty: Type::Array {
                    elem: Box::new(Type::I32),
                    len: 16,
                },
            },
            Some(Type::Ptr(Box::new(Type::Array {
                elem: Box::new(Type::I32),
                len: 16,
            }))),
        )
        .unwrap();
    function.append_inst(
        function.entry,
        InstKind::MemZero {
            ptr: array,
            bytes: 64,
        },
        None,
    );
    function.set_terminator(function.entry, Terminator::Return(None));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "clear_local");

    assert!(function_asm.contains("  sd zero, 0(a0)\n"));
    assert!(!function_asm.contains("  sb zero,"));
}

#[test]
fn returns_from_chained_void_guards_before_building_the_frame() {
    let mut function = Function::new("guarded_void", Type::Void);
    let depth = function.add_param("depth", Type::I32);
    let left = function.add_param("left", Type::I32);
    let right = function.add_param("right", Type::I32);
    let minus_one = function.add_const(Const::Int(-1));
    let one = function.add_const(Const::Int(1));
    let fast = function.add_block("fast");
    let second_guard = function.add_block("second_guard");
    let work = function.add_block("work");
    let depth_done = function
        .append_inst(
            function.entry,
            InstKind::Icmp {
                op: crate::ir::CmpOp::Eq,
                lhs: depth,
                rhs: minus_one,
            },
            Some(Type::I1),
        )
        .unwrap();
    function.set_terminator(
        function.entry,
        Terminator::Branch {
            cond: depth_done,
            then_target: fast,
            else_target: second_guard,
        },
    );
    let next_left = function
        .append_inst(
            second_guard,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: left,
                rhs: one,
            },
            Some(Type::I32),
        )
        .unwrap();
    let range_done = function
        .append_inst(
            second_guard,
            InstKind::Icmp {
                op: crate::ir::CmpOp::Ge,
                lhs: next_left,
                rhs: right,
            },
            Some(Type::I1),
        )
        .unwrap();
    function.set_terminator(
        second_guard,
        Terminator::Branch {
            cond: range_done,
            then_target: fast,
            else_target: work,
        },
    );
    function.set_terminator(fast, Terminator::Return(None));
    function.set_terminator(work, Terminator::Return(None));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "guarded_void");
    let frame = function_asm.find("  addi sp, sp, -16\n").unwrap();
    let preframe = &function_asm[..frame];

    assert!(preframe.contains("  beq a0, t1, .L_preframe_return_guarded_void\n"));
    assert!(preframe.contains("  addiw t0, a1, 1\n"));
    assert!(preframe.contains("  bge t0, a2, .L_preframe_return_guarded_void\n"));
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

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "float_expr");
    let body = entry_block_asm(function_asm, "float_expr");

    assert!(!body.contains("fmadd.s"));
    assert!(body.lines().any(|line| line.starts_with("  fmul.s ft")));
    assert!(body.lines().any(|line| line.starts_with("  fadd.s ft")));
    assert!(!body.lines().any(|line| line.starts_with("  fmv.s ft")));
    assert!(!body.contains("fsw fa0"));
}

#[test]
fn emits_float_results_directly_into_assigned_registers() {
    let mut function = Function::new("direct_float_results", Type::F32);
    let integer = function.add_param("integer", Type::I32);
    let addend = function.add_param("addend", Type::F32);
    let converted = function
        .append_inst(
            function.entry,
            InstKind::Cast {
                op: CastOp::I32ToF32,
                value: integer,
            },
            Some(Type::F32),
        )
        .unwrap();
    let negated = function
        .append_inst(
            function.entry,
            InstKind::Unary {
                op: UnaryOp::Fneg,
                value: converted,
            },
            Some(Type::F32),
        )
        .unwrap();
    let sum = function
        .append_inst(
            function.entry,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: negated,
                rhs: addend,
            },
            Some(Type::F32),
        )
        .unwrap();
    function.set_terminator(function.entry, Terminator::Return(Some(sum)));

    let regs = Riscv64FloatRegAlloc::new(&function);
    let converted_reg = regs.reg(converted).unwrap();
    let negated_reg = regs.reg(negated).unwrap();
    let sum_reg = regs.reg(sum).unwrap();
    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "direct_float_results");
    let body = entry_block_asm(function_asm, "direct_float_results");

    assert!(body.contains(&format!("fcvt.s.w {converted_reg},")));
    assert!(body.contains(&format!("fneg.s {negated_reg},")));
    assert!(body.contains(&format!("fadd.s {sum_reg},")));
    assert!(!body.contains(&format!("fmv.s {converted_reg}, fa0")));
    assert!(!body.contains(&format!("fmv.s {negated_reg}, fa0")));
    assert!(!body.contains(&format!("fmv.s {sum_reg}, fa0")));
}

#[test]
fn emits_assigned_float_loads_and_stores_directly() {
    let mut function = Function::new("assigned_float_memory", Type::I32);
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
    let function_asm = function_asm(&asm, "assigned_float_memory");
    let body = block_asm(function_asm, "assigned_float_memory", body_block.0);

    assert!(body
        .lines()
        .any(|line| line.starts_with("  fsw ft") && line.contains(", 0(")));
    assert!(body
        .lines()
        .any(|line| line.starts_with("  flw ft") && line.contains(", 0(")));
    assert!(!body.contains("fmv.w.x"));
    assert!(!body.contains("flw fa0, 0(a0)"));
}

#[test]
fn falls_back_to_stack_for_unassigned_float_loads_and_stores() {
    let mut function = Function::new("fallback_float_memory", Type::Void);
    let ptr = function.add_param("ptr", Type::Ptr(Box::new(Type::F32)));
    let value = function.add_const(Const::Float(1.5f32.to_bits()));
    function.append_inst(function.entry, InstKind::Store { ptr, value }, None);
    function.append_inst(function.entry, InstKind::Load { ptr }, Some(Type::F32));
    function.set_terminator(function.entry, Terminator::Return(None));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "fallback_float_memory");
    let body = entry_block_asm(function_asm, "fallback_float_memory");

    assert!(body.contains("fsw fa0, 0(a1)"));
    assert!(body.contains("flw fa0, 0(a0)"));
    assert!(body
        .lines()
        .any(|line| line.starts_with("  fsw fa0, -") && line.ends_with("(s0)")));
    assert!(!body.contains("fmv.w.x"));
}

#[test]
fn folds_float_loads_and_stores_into_their_base_address() {
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
    let function_asm = function_asm(&asm, "folded_float_memory");
    let body = block_asm(function_asm, "folded_float_memory", body_block.0);

    assert!(body
        .lines()
        .any(|line| line.starts_with("  fsw ft") && line.contains(", 8(")));
    assert!(body
        .lines()
        .any(|line| line.starts_with("  flw ft") && line.contains(", 8(")));
    assert!(!body.contains("fmv.w.x"));
}

#[test]
fn routes_float_call_arguments_and_results_through_assigned_registers() {
    let mut callee = Function::new("float_identity", Type::F32);
    let callee_value = callee.add_param("value", Type::F32);
    callee.set_terminator(callee.entry, Terminator::Return(Some(callee_value)));

    let mut caller = Function::new("float_call_pipeline", Type::F32);
    let integer = caller.add_param("integer", Type::I32);
    let converted = caller
        .append_inst(
            caller.entry,
            InstKind::Cast {
                op: CastOp::I32ToF32,
                value: integer,
            },
            Some(Type::F32),
        )
        .unwrap();
    let call_result = caller
        .append_inst(
            caller.entry,
            InstKind::Call {
                name: "float_identity".into(),
                args: vec![converted],
            },
            Some(Type::F32),
        )
        .unwrap();
    let negated = caller
        .append_inst(
            caller.entry,
            InstKind::Unary {
                op: UnaryOp::Fneg,
                value: call_result,
            },
            Some(Type::F32),
        )
        .unwrap();
    caller.set_terminator(caller.entry, Terminator::Return(Some(negated)));

    let mut module = Module::new();
    module.add_func(callee);
    module.add_func(caller);
    let asm = emit_ir_asm(&module);
    let caller_asm = function_asm(&asm, "float_call_pipeline");
    let body = entry_block_asm(caller_asm, "float_call_pipeline");

    assert!(body.lines().any(|line| line.starts_with("  fcvt.s.w ft")));
    assert!(body.contains("fmv.s fa0, ft"));
    assert!(body.contains("call float_identity\n  fmv.s ft"));
    assert!(body.lines().any(|line| line.starts_with("  fneg.s ft")));
    assert!(!body.contains("fneg.s fa0"));
    assert!(!body.contains("fmv.x.w"));
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
    assert!(!caller_asm.contains("fmv.s fa1, fs0"));
    assert!(caller_asm.contains("fadd.s ft1, fs0, fa0"));
    assert!(caller_asm.contains("fmv.s fa0, ft1"));
    assert!(!caller_asm.contains("lw a0, -24(s0)"));
    assert!(caller_asm.contains("fsd fs0, -8(s0)"));
    assert!(caller_asm.contains("fld fs0, -8(s0)"));
    assert!(!caller_asm.contains("fsw fs0, -8(s0)"));
    assert!(!caller_asm.contains("flw fs0, -8(s0)"));
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
    assert!(function_asm.contains("fmv.s fa0, ft1"));
    assert!(!function_asm.contains("lw a0, -8(s0)"));
}

#[test]
fn receives_ninth_float_parameter_from_integer_fallback_register() {
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

    assert!(function_asm.contains("fmv.w.x ft1, a0"));
    assert!(function_asm.contains("fsw ft1, -72(s0)"));
    assert!(function_asm.contains("fmv.s fa0, ft1"));
    assert!(!function_asm.contains("flw ft1, 16(s0)"));
    assert!(!function_asm.contains("lw a0, -72(s0)"));
}

#[test]
fn receives_seventeenth_float_parameter_from_stack() {
    let mut function = Function::new("return_seventeenth_float_parameter", Type::F32);
    let mut value = None;
    for index in 0..17 {
        value = Some(function.add_param(format!("value_{index}"), Type::F32));
    }
    function.set_terminator(function.entry, Terminator::Return(value));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "return_seventeenth_float_parameter");

    assert!(function_asm.contains("flw ft1, 16(s0)"));
    assert!(function_asm.contains("fsw ft1, -136(s0)"));
    assert!(function_asm.contains("fmv.s fa0, ft1"));
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

    assert!(caller_asm.contains("fsd fs0, -8(s0)"));
    assert!(caller_asm.contains("sw a0, -32(s0)"));
}

#[test]
fn keeps_float_accumulator_phi_off_the_stack() {
    let mut function = Function::new("float_loop", Type::F32);
    let initial = function.add_param("initial", Type::F32);
    let count = function.add_param("count", Type::I32);
    let zero_i = function.add_const(Const::Int(0));
    let one_i = function.add_const(Const::Int(1));
    let one_f = function.add_const(Const::Float(1.0f32.to_bits()));
    let header = function.add_block("header");
    let exit = function.add_block("exit");
    function.set_terminator(function.entry, Terminator::Jump(header));

    let acc = function
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(function.entry, initial)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let index = function
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(function.entry, zero_i)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let next_acc = function
        .append_inst(
            header,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: acc,
                rhs: one_f,
            },
            Some(Type::F32),
        )
        .unwrap();
    let next_index = function
        .append_inst(
            header,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: index,
                rhs: one_i,
            },
            Some(Type::I32),
        )
        .unwrap();
    let cond = function
        .append_inst(
            header,
            InstKind::Icmp {
                op: crate::ir::CmpOp::Lt,
                lhs: next_index,
                rhs: count,
            },
            Some(Type::I1),
        )
        .unwrap();
    let InstKind::Phi { incomings } = &mut function.block_mut(header).insts[0].kind else {
        unreachable!();
    };
    incomings.push((header, next_acc));
    let InstKind::Phi { incomings } = &mut function.block_mut(header).insts[1].kind else {
        unreachable!();
    };
    incomings.push((header, next_index));
    function.set_terminator(
        header,
        Terminator::Branch {
            cond,
            then_target: header,
            else_target: exit,
        },
    );
    function.set_terminator(exit, Terminator::Return(Some(next_acc)));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "float_loop");
    let emitted_blocks = &function_asm[function_asm.find("\n.L_float_loop_bb").unwrap()..];

    assert!(function_asm.contains("fadd.s"));
    assert!(function_asm.contains("fmv.s ft"));
    assert!(!emitted_blocks.contains("fsw fa0"));
    assert!(
        !emitted_blocks
            .lines()
            .any(|line| line.trim_start().starts_with("sw ")),
        "{emitted_blocks}"
    );
}

#[test]
fn breaks_float_phi_copy_cycle_with_ft0() {
    let mut function = Function::new("float_swap", Type::Void);
    let lhs = function.add_param("lhs", Type::F32);
    let rhs = function.add_param("rhs", Type::F32);
    let header = function.add_block("header");
    function.set_terminator(function.entry, Terminator::Jump(header));

    let left_phi = function
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(function.entry, lhs)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let right_phi = function
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(function.entry, rhs)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let InstKind::Phi { incomings } = &mut function.block_mut(header).insts[0].kind else {
        unreachable!();
    };
    incomings.push((header, right_phi));
    let InstKind::Phi { incomings } = &mut function.block_mut(header).insts[1].kind else {
        unreachable!();
    };
    incomings.push((header, left_phi));
    function.set_terminator(header, Terminator::Jump(header));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "float_swap");

    assert!(function_asm.contains("fmv.s ft0,"));
    assert!(function_asm.lines().any(|line| line.ends_with(", ft0")));
}

#[test]
fn snapshots_large_mixed_phi_copy_sets_with_typed_stack_records() {
    const COPY_COUNT: usize = 129;
    const FLOAT_COPY_COUNT: usize = 65;
    const INT_COPY_COUNT: usize = 64;

    let mut function = Function::new("mixed_phi_snapshot", Type::Void);
    let integer = function.add_const(Const::Int(37));
    let float_bits = 2.5f32.to_bits();
    let float = function.add_const(Const::Float(float_bits));
    let target = function.add_block("target");
    function.set_terminator(function.entry, Terminator::Jump(target));

    for index in 0..COPY_COUNT {
        let (incoming, ty) = if index % 2 == 0 {
            (float, Type::F32)
        } else {
            (integer, Type::I32)
        };
        function.append_inst(
            target,
            InstKind::Phi {
                incomings: vec![(function.entry, incoming)],
            },
            Some(ty),
        );
    }
    function.set_terminator(target, Terminator::Return(None));

    let mut module = Module::new();
    module.add_func(function);
    let asm = emit_ir_asm(&module);
    let function_asm = function_asm(&asm, "mixed_phi_snapshot");
    let edge_asm = entry_block_asm(function_asm, "mixed_phi_snapshot");

    assert!(asm.contains(&format!("  .word {float_bits}")));
    assert!(edge_asm.contains("  la t0, .L_float_"));
    assert!(edge_asm.contains("  flw fa0, 0(t0)\n"));
    assert_eq!(edge_asm.matches("  sd a0, 0(sp)\n").count(), INT_COPY_COUNT);
    assert_eq!(
        edge_asm.matches("  sd zero, 0(sp)\n").count(),
        FLOAT_COPY_COUNT
    );
    assert_eq!(
        edge_asm.matches("  fsw fa0, 0(sp)\n").count(),
        FLOAT_COPY_COUNT
    );
    assert_eq!(edge_asm.matches("  addi sp, sp, -16\n").count(), COPY_COUNT);
    assert_eq!(edge_asm.matches("  addi sp, sp, 16\n").count(), COPY_COUNT);
    assert_eq!(
        edge_asm.matches("  flw fa0, 0(sp)\n").count(),
        FLOAT_COPY_COUNT
    );
    assert_eq!(edge_asm.matches("  ld a1, 0(sp)\n").count(), INT_COPY_COUNT);

    let first_restore = edge_asm.find("  flw fa0, 0(sp)\n").unwrap();
    let restores = &edge_asm[first_restore..];
    assert!(restores.starts_with("  flw fa0, 0(sp)\n  addi sp, sp, 16\n"));
    let first_int_restore = restores.find("  ld a1, 0(sp)\n").unwrap();
    assert!(restores[..first_int_restore].contains("  fsw fa0, "));
    let next_float_restore = restores[first_int_restore..]
        .find("  flw fa0, 0(sp)\n")
        .map(|offset| offset + first_int_restore)
        .unwrap();
    let int_restore = &restores[first_int_restore..next_float_restore];
    assert!(int_restore.starts_with("  ld a1, 0(sp)\n  addi sp, sp, 16\n  mv a0, a1\n"));

    assert!(!edge_asm.contains("fmv.w.x"));
    assert!(!edge_asm.contains("fmv.x.w"));
    assert!(!edge_asm.lines().any(|line| line.contains("mv a0, ft")));
}
