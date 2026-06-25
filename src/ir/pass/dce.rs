use super::ModulePass;
use crate::ir::{Function, Inst, InstKind, Module, Terminator, ValueId};
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
    loop {
        let used = collect_used_values(func);
        let mut changed = false;

        for block in &mut func.blocks {
            for inst in &mut block.insts {
                let Some(result) = inst.result else {
                    continue;
                };
                if used.contains(&result) || !is_removable(inst) {
                    continue;
                }

                inst.result = None;
                inst.kind = InstKind::Nop;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

fn collect_used_values(func: &Function) -> HashSet<ValueId> {
    let mut used = HashSet::new();

    for block in &func.blocks {
        for inst in &block.insts {
            collect_inst_operands(inst, &mut used);
        }
        if let Some(terminator) = &block.terminator {
            collect_terminator_operands(terminator, &mut used);
        }
    }

    used
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
    matches!(
        inst.kind,
        InstKind::Nop
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
