use super::ModulePass;
use crate::ir::{Function, Inst, InstKind, Module, Terminator, ValueId, ValueKind};
use std::collections::HashSet;

pub(super) struct DcePass;

impl DcePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for DcePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            eliminate_dead_code(func);
        }
    }
}

fn eliminate_dead_code(func: &mut Function) {
    let live = collect_live_values(func);
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            let Some(result) = inst.result else {
                continue;
            };
            if live.contains(&result) || !is_removable(inst) {
                continue;
            }

            // Keep instruction indices stable for ValueKind::Inst references.
            inst.result = None;
            inst.kind = InstKind::Nop;
        }
    }
}

/// Marks values reachable from observable side effects and terminators.
///
/// A plain use-count DCE cannot remove a dead SCC such as `phi -> add -> phi`:
/// every member appears used by another dead member. Walking backwards only
/// from observable roots removes those cycles while preserving the complete
/// dependency chain of every live instruction.
fn collect_live_values(func: &Function) -> HashSet<ValueId> {
    let mut roots = HashSet::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if !is_removable(inst) {
                collect_inst_operands(inst, &mut roots);
            }
        }
        if let Some(terminator) = &block.terminator {
            collect_terminator_operands(terminator, &mut roots);
        }
    }

    let mut live = HashSet::new();
    let mut worklist = roots.into_iter().collect::<Vec<_>>();
    while let Some(value) = worklist.pop() {
        if !live.insert(value) {
            continue;
        }
        let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
            continue;
        };
        let Some(inst) = func
            .blocks
            .get(block.0)
            .and_then(|block| block.insts.get(inst_idx))
        else {
            continue;
        };
        let mut operands = HashSet::new();
        collect_inst_operands(inst, &mut operands);
        worklist.extend(operands);
    }
    live
}

fn collect_inst_operands(inst: &Inst, used: &mut HashSet<ValueId>) {
    match &inst.kind {
        InstKind::Nop | InstKind::Alloca { .. } => {}
        InstKind::Phi { incomings } => {
            for (_, value) in incomings {
                used.insert(*value);
            }
        }
        InstKind::Load { ptr } => {
            used.insert(*ptr);
        }
        InstKind::Store { ptr, value } => {
            used.insert(*ptr);
            used.insert(*value);
        }
        InstKind::MemZero { ptr, .. } => {
            used.insert(*ptr);
        }
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => {
            used.insert(*value);
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            used.insert(*lhs);
            used.insert(*rhs);
        }
        InstKind::Gep { base, indices } => {
            used.insert(*base);
            used.extend(indices.iter().copied());
        }
        InstKind::Call { args, .. } => {
            used.extend(args.iter().copied());
        }
    }
}

fn collect_terminator_operands(terminator: &Terminator, used: &mut HashSet<ValueId>) {
    match terminator {
        Terminator::Return(value) => {
            if let Some(value) = value {
                used.insert(*value);
            }
        }
        Terminator::Jump(_) => {}
        Terminator::Branch { cond, .. } => {
            used.insert(*cond);
        }
    }
}

fn is_removable(inst: &Inst) -> bool {
    // store/call/memzero 这类可能有副作用的指令不能因为结果没被用就删掉。
    matches!(
        inst.kind,
        InstKind::Nop
            | InstKind::Alloca { .. }
            | InstKind::Phi { .. }
            | InstKind::Load { .. }
            | InstKind::Unary { .. }
            | InstKind::Binary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. }
            | InstKind::Gep { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Const, Function, Type};

    #[test]
    fn removes_an_unobservable_phi_cycle() {
        let mut func = Function::new("dead_cycle", Type::Void);
        let keep_looping = func.add_param("continue", Type::I1);
        let header = func.add_block("header");
        let body = func.add_block("body");
        let exit = func.add_block("exit");
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        func.set_terminator(func.entry, Terminator::Jump(header));
        let phi = func
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
                cond: keep_looping,
                then_target: body,
                else_target: exit,
            },
        );
        let next = func
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
        let ValueKind::Inst(phi_block, phi_idx) = func.value(phi).kind else {
            unreachable!();
        };
        let InstKind::Phi { incomings } = &mut func.blocks[phi_block.0].insts[phi_idx].kind else {
            unreachable!();
        };
        incomings[1].1 = next;
        func.set_terminator(body, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));

        assert!(func.verify().is_ok());
        eliminate_dead_code(&mut func);
        for value in [phi, next] {
            let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
                unreachable!();
            };
            assert!(matches!(
                func.blocks[block.0].insts[inst_idx].kind,
                InstKind::Nop
            ));
        }
        assert!(func.verify().is_ok());
    }
}
