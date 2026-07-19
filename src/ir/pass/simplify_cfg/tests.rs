use super::*;
use crate::ir::{BinaryOp, Type};

fn run_pass(func: Function) -> Function {
    let mut module = Module::new();
    module.add_func(func);
    SimplifyCfgPass::new().run(&mut module);
    module.funcs.remove(0)
}

#[test]
fn removes_unreachable_blocks_and_remaps_ids() {
    let mut func = Function::new("unreachable", Type::I32);
    let cond = func.add_param("cond", Type::I1);
    let live = func.add_block("live");
    let dead = func.add_block("dead");
    let exit = func.add_block("exit");
    let zero = func.add_const(Const::Int(0));
    let one = func.add_const(Const::Int(1));
    func.set_terminator(
        func.entry,
        Terminator::Branch {
            cond,
            then_target: live,
            else_target: exit,
        },
    );
    let live_value = func
        .append_inst(
            live,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: one,
                rhs: one,
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(live, Terminator::Jump(exit));
    let dead_value = func
        .append_inst(
            dead,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: one,
                rhs: one,
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(dead, Terminator::Return(Some(dead_value)));
    let result = func
        .append_inst(
            exit,
            InstKind::Phi {
                incomings: vec![(func.entry, zero), (live, live_value)],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(exit, Terminator::Return(Some(result)));
    assert!(func.verify().is_ok());

    let func = run_pass(func);

    assert_eq!(
        func.blocks
            .iter()
            .map(|block| block.name.as_str())
            .collect::<Vec<_>>(),
        vec!["entry", "live", "exit"]
    );
    assert!(matches!(
        func.value(dead_value).kind,
        ValueKind::Const(Const::Zero(Type::I32))
    ));
    assert!(matches!(
        func.blocks[0].terminator,
        Some(Terminator::Branch {
            then_target: BlockId(1),
            else_target: BlockId(2),
            ..
        })
    ));
    assert!(func.verify().is_ok());
}

#[test]
fn forwards_empty_jump_blocks_and_repairs_phis() {
    let mut func = Function::new("empty_jump", Type::I32);
    let cond = func.add_param("cond", Type::I1);
    let input = func.add_param("input", Type::I32);
    let left = func.add_block("left");
    let right = func.add_block("right");
    let empty = func.add_block("empty");
    let target = func.add_block("target");
    func.set_terminator(
        func.entry,
        Terminator::Branch {
            cond,
            then_target: left,
            else_target: right,
        },
    );
    for block in [left, right] {
        func.append_inst(
            block,
            InstKind::Call {
                name: "putint".to_string(),
                args: vec![input],
            },
            None,
        );
        func.set_terminator(block, Terminator::Jump(empty));
    }
    func.set_terminator(empty, Terminator::Jump(target));
    let merged = func
        .append_inst(
            target,
            InstKind::Phi {
                incomings: vec![(empty, input)],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(target, Terminator::Return(Some(merged)));
    assert!(func.verify().is_ok());

    let func = run_pass(func);

    assert_eq!(
        func.blocks
            .iter()
            .map(|block| block.name.as_str())
            .collect::<Vec<_>>(),
        vec!["entry", "left", "right", "target"]
    );
    for block_idx in [1, 2] {
        assert_eq!(
            func.blocks[block_idx].terminator,
            Some(Terminator::Jump(BlockId(3)))
        );
    }
    assert!(matches!(func.blocks[3].insts[0].kind, InstKind::Nop));
    assert_eq!(
        func.blocks[3].terminator,
        Some(Terminator::Return(Some(input)))
    );
    assert!(func.verify().is_ok());
}

#[test]
fn merges_linear_blocks_and_rewrites_phi_values() {
    let mut func = Function::new("linear", Type::I32);
    let input = func.add_param("input", Type::I32);
    let body = func.add_block("body");
    let one = func.add_const(Const::Int(1));
    func.set_terminator(func.entry, Terminator::Jump(body));
    let phi = func
        .append_inst(
            body,
            InstKind::Phi {
                incomings: vec![(func.entry, input)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let sum = func
        .append_inst(
            body,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: phi,
                rhs: one,
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(body, Terminator::Return(Some(sum)));
    assert!(func.verify().is_ok());

    let func = run_pass(func);

    assert_eq!(func.blocks.len(), 1);
    assert_eq!(
        func.blocks[0].terminator,
        Some(Terminator::Return(Some(sum)))
    );
    assert!(matches!(
        func.value(sum).kind,
        ValueKind::Inst(BlockId(0), _)
    ));
    let ValueKind::Inst(owner, inst_idx) = func.value(sum).kind else {
        unreachable!();
    };
    assert!(matches!(
        func.blocks[owner.0].insts[inst_idx].kind,
        InstKind::Binary {
            lhs,
            op: BinaryOp::Iadd,
            ..
        } if lhs == input
    ));
    assert!(func.verify().is_ok());
}

#[test]
fn removes_unreachable_phi_cycles_before_rewriting_values() {
    let mut func = Function::new("dead_phi_cycle", Type::Void);
    let dead_a = func.add_block("dead_a");
    let dead_b = func.add_block("dead_b");
    let zero = func.add_const(Const::Int(0));
    func.set_terminator(func.entry, Terminator::Return(None));
    let phi_a = func
        .append_inst(
            dead_a,
            InstKind::Phi {
                incomings: vec![(dead_b, zero)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let phi_b = func
        .append_inst(
            dead_b,
            InstKind::Phi {
                incomings: vec![(dead_a, phi_a)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let ValueKind::Inst(owner, inst_idx) = func.value(phi_a).kind else {
        unreachable!();
    };
    let InstKind::Phi { incomings } = &mut func.blocks[owner.0].insts[inst_idx].kind else {
        unreachable!();
    };
    incomings[0].1 = phi_b;
    func.append_inst(
        dead_a,
        InstKind::Call {
            name: "putint".to_string(),
            args: vec![phi_a],
        },
        None,
    );
    func.set_terminator(dead_a, Terminator::Jump(dead_b));
    func.set_terminator(dead_b, Terminator::Jump(dead_a));
    assert!(func.verify().is_ok());

    let func = run_pass(func);

    assert_eq!(func.blocks.len(), 1);
    assert!(matches!(
        func.value(phi_a).kind,
        ValueKind::Const(Const::Zero(Type::I32))
    ));
    assert!(matches!(
        func.value(phi_b).kind,
        ValueKind::Const(Const::Zero(Type::I32))
    ));
    assert!(func.verify().is_ok());
}

#[test]
fn preserves_loop_headers_with_multiple_predecessors() {
    let mut func = Function::new("loop", Type::I32);
    let keep_going = func.add_param("keep_going", Type::I1);
    let header = func.add_block("header");
    let body = func.add_block("body");
    let exit = func.add_block("exit");
    let zero = func.add_const(Const::Int(0));
    let one = func.add_const(Const::Int(1));
    func.set_terminator(func.entry, Terminator::Jump(header));
    let counter = func
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(func.entry, zero), (body, zero)],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(
        header,
        Terminator::Branch {
            cond: keep_going,
            then_target: body,
            else_target: exit,
        },
    );
    let next = func
        .append_inst(
            body,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: counter,
                rhs: one,
            },
            Some(Type::I32),
        )
        .unwrap();
    let ValueKind::Inst(owner, inst_idx) = func.value(counter).kind else {
        unreachable!();
    };
    let InstKind::Phi { incomings } = &mut func.blocks[owner.0].insts[inst_idx].kind else {
        unreachable!();
    };
    incomings[1].1 = next;
    func.set_terminator(body, Terminator::Jump(header));
    func.set_terminator(exit, Terminator::Return(Some(counter)));
    assert!(func.verify().is_ok());

    let func = run_pass(func);

    let header = func
        .blocks
        .iter()
        .position(|block| block.name == "header")
        .map(BlockId)
        .expect("loop header must remain present");
    assert_eq!(all_predecessors(&func)[header.0].len(), 2);
    assert!(func.blocks[header.0]
        .insts
        .iter()
        .any(|inst| matches!(inst.kind, InstKind::Phi { .. })));
    assert!(func.verify().is_ok());
}

#[test]
fn skips_forwarding_when_phi_predecessors_would_conflict() {
    let mut func = Function::new("conflict", Type::I32);
    let cond = func.add_param("cond", Type::I1);
    let direct = func.add_param("direct", Type::I32);
    let indirect = func.add_param("indirect", Type::I32);
    let empty = func.add_block("empty");
    let target = func.add_block("target");
    func.set_terminator(
        func.entry,
        Terminator::Branch {
            cond,
            then_target: empty,
            else_target: target,
        },
    );
    func.set_terminator(empty, Terminator::Jump(target));
    let result = func
        .append_inst(
            target,
            InstKind::Phi {
                incomings: vec![(func.entry, direct), (empty, indirect)],
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(target, Terminator::Return(Some(result)));
    assert!(func.verify().is_ok());
    let before = func.clone();

    let func = run_pass(func);

    assert_eq!(func, before);
    assert!(func.verify().is_ok());
}

#[test]
fn repeated_runs_are_idempotent_and_verified() {
    let mut func = Function::new("idempotent", Type::I32);
    let input = func.add_param("input", Type::I32);
    let empty = func.add_block("empty");
    let exit = func.add_block("exit");
    func.set_terminator(func.entry, Terminator::Jump(empty));
    func.set_terminator(empty, Terminator::Jump(exit));
    func.set_terminator(exit, Terminator::Return(Some(input)));

    let once = run_pass(func);
    let twice = run_pass(once.clone());

    assert_eq!(twice, once);
    assert!(twice.verify().is_ok());
}
