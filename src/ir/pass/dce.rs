use super::ModulePass;
use crate::ir::{Function, Inst, InstKind, Module, Terminator, ValueId, ValueKind};
use std::collections::HashSet;

pub(super) struct DcePass {
    remove_write_only_allocas: bool,
}

impl DcePass {
    pub(super) fn new() -> Self {
        Self {
            remove_write_only_allocas: true,
        }
    }

    pub(super) fn preserving_write_only_allocas() -> Self {
        Self {
            remove_write_only_allocas: false,
        }
    }
}

impl ModulePass for DcePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            eliminate_dead_code_with_alloca_cleanup(func, self.remove_write_only_allocas);
        }
    }
}

fn eliminate_dead_code(func: &mut Function) {
    eliminate_dead_code_with_alloca_cleanup(func, true);
}

fn eliminate_dead_code_with_alloca_cleanup(func: &mut Function, cleanup_write_only_allocas: bool) {
    if cleanup_write_only_allocas {
        remove_write_only_allocas(func);
    }
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

/// Removes writes to local stack objects whose address never escapes and whose
/// value is never read. A store through such an alloca has no observable
/// effect, so treating it as an unconditional DCE root needlessly keeps both
/// the store and the stack object alive.
fn remove_write_only_allocas(func: &mut Function) {
    let write_only = write_only_alloca_values(func);
    if write_only.is_empty() {
        return;
    }

    for block in &mut func.blocks {
        for inst in &mut block.insts {
            if matches!(
                inst.kind,
                InstKind::Store { ptr, .. } | InstKind::MemZero { ptr, .. }
                    if write_only.contains(&ptr)
            ) {
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
        }
    }
}

pub(super) fn write_only_alloca_values(func: &Function) -> HashSet<ValueId> {
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match inst {
            Inst {
                result: Some(slot),
                kind: InstKind::Alloca { .. },
            } => write_only_alloca_pointers(func, *slot),
            _ => None,
        })
        .flatten()
        .collect()
}

pub(super) fn write_only_alloca_pointers(
    func: &Function,
    slot: ValueId,
) -> Option<HashSet<ValueId>> {
    let mut pointers = HashSet::from([slot]);
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.insts {
                let (Some(result), InstKind::Gep { base, .. }) = (inst.result, &inst.kind) else {
                    continue;
                };
                if pointers.contains(base) {
                    changed |= pointers.insert(result);
                }
            }
        }
        if !changed {
            break;
        }
    }

    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Alloca { .. } if inst.result == Some(slot) => continue,
                InstKind::Gep { base, indices }
                    if pointers.contains(base)
                        && indices.iter().all(|index| !pointers.contains(index)) =>
                {
                    continue;
                }
                InstKind::Store { ptr, value }
                    if pointers.contains(ptr) && !pointers.contains(value) =>
                {
                    continue;
                }
                InstKind::MemZero { ptr, .. } if pointers.contains(ptr) => continue,
                _ => {}
            }
            let mut operands = HashSet::new();
            collect_inst_operands(inst, &mut operands);
            if operands.iter().any(|operand| pointers.contains(operand)) {
                return None;
            }
        }
        let mut operands = HashSet::new();
        if let Some(terminator) = &block.terminator {
            collect_terminator_operands(terminator, &mut operands);
        }
        if operands.iter().any(|operand| pointers.contains(operand)) {
            return None;
        }
    }
    Some(pointers)
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
        InstKind::MemZero { ptr, count, .. } => {
            used.insert(*ptr);
            used.extend(count.iter().copied());
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => used.extend([*dst, *src, *count]),
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
