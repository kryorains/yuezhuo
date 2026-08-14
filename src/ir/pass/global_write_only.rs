use super::ModulePass;
use crate::ir::{Function, InstKind, Module, Terminator, Type, ValueId, ValueKind};
use std::collections::HashSet;

/// Removes writes to whole-program globals whose contents are never observed.
///
/// A global remains observable when any load reads it or when its address
/// escapes through a call, a store, or a return. Merely deriving an address
/// with GEP and writing through it does not make the old or new contents
/// observable.
pub(super) struct GlobalWriteOnlyPass;

impl GlobalWriteOnlyPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for GlobalWriteOnlyPass {
    fn run(&mut self, module: &mut Module) {
        let declared = module
            .globals
            .iter()
            .map(|global| global.name.clone())
            .collect::<HashSet<_>>();
        let mut observed = HashSet::new();
        for func in &module.funcs {
            collect_observed_globals(func, &mut observed);
        }
        let write_only = declared
            .difference(&observed)
            .cloned()
            .collect::<HashSet<_>>();
        if write_only.is_empty() {
            return;
        }

        for func in &mut module.funcs {
            for block_idx in 0..func.blocks.len() {
                for inst_idx in 0..func.blocks[block_idx].insts.len() {
                    let pointer = match func.blocks[block_idx].insts[inst_idx].kind {
                        InstKind::Store { ptr, .. } | InstKind::MemZero { ptr, .. } => ptr,
                        _ => continue,
                    };
                    let roots = pointer_global_roots(func, pointer);
                    if roots.is_empty() || !roots.iter().all(|root| write_only.contains(root)) {
                        continue;
                    }
                    let inst = &mut func.blocks[block_idx].insts[inst_idx];
                    inst.result = None;
                    inst.kind = InstKind::Nop;
                }
            }
            if let Err(errors) = func.verify() {
                panic!(
                    "write-only global elimination produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

fn collect_observed_globals(func: &Function, observed: &mut HashSet<String>) {
    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Load { ptr } => observe_pointer(func, *ptr, observed),
                InstKind::Call { args, .. } => {
                    for arg in args {
                        if matches!(func.value(*arg).ty, Type::Ptr(_)) {
                            observe_pointer(func, *arg, observed);
                        }
                    }
                }
                InstKind::Store { value, .. } => {
                    if matches!(func.value(*value).ty, Type::Ptr(_)) {
                        observe_pointer(func, *value, observed);
                    }
                }
                InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => {
                    if matches!(func.value(*value).ty, Type::Ptr(_)) {
                        observe_pointer(func, *value, observed);
                    }
                }
                InstKind::Binary { lhs, rhs, .. }
                | InstKind::Icmp { lhs, rhs, .. }
                | InstKind::Fcmp { lhs, rhs, .. } => {
                    for value in [*lhs, *rhs] {
                        if matches!(func.value(value).ty, Type::Ptr(_)) {
                            observe_pointer(func, value, observed);
                        }
                    }
                }
                InstKind::Nop
                | InstKind::Phi { .. }
                | InstKind::Alloca { .. }
                | InstKind::MemZero { .. }
                | InstKind::Gep { .. } => {}
                InstKind::MemCopy { dst, src, .. } => {
                    observe_pointer(func, *dst, observed);
                    observe_pointer(func, *src, observed);
                }
            }
        }
        if let Some(Terminator::Return(Some(value))) = &block.terminator {
            if matches!(func.value(*value).ty, Type::Ptr(_)) {
                observe_pointer(func, *value, observed);
            }
        }
    }
}

fn observe_pointer(func: &Function, value: ValueId, observed: &mut HashSet<String>) {
    observed.extend(pointer_global_roots(func, value));
}

fn pointer_global_roots(func: &Function, value: ValueId) -> HashSet<String> {
    let mut roots = HashSet::new();
    let mut visited = HashSet::new();
    let mut worklist = vec![value];
    while let Some(value) = worklist.pop() {
        if !visited.insert(value) {
            continue;
        }
        match &func.value(value).kind {
            ValueKind::Global(name) => {
                roots.insert(name.clone());
            }
            ValueKind::Inst(block, inst_idx) => match &func.blocks[block.0].insts[*inst_idx].kind {
                InstKind::Gep { base, .. } => worklist.push(*base),
                InstKind::Phi { incomings } => {
                    worklist.extend(incomings.iter().map(|(_, incoming)| *incoming));
                }
                _ => {}
            },
            ValueKind::Param | ValueKind::Const(_) => {}
        }
    }
    roots
}
