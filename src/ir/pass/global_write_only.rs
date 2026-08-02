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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Function, Global, InstKind, Module, Terminator};

    #[test]
    fn removes_stores_to_an_unobserved_global_array() {
        let mut module = Module::new();
        for name in ["dead", "live"] {
            module.globals.push(Global {
                name: name.into(),
                ty: Type::Array {
                    elem: Box::new(Type::I32),
                    len: 8,
                },
                is_const: false,
                init: Some(Const::Zero(Type::Array {
                    elem: Box::new(Type::I32),
                    len: 8,
                })),
            });
        }
        let mut func = Function::new("main", Type::I32);
        let index = func.add_param("index", Type::I32);
        let value = func.add_param("value", Type::I32);
        let dead = func.add_global_ref(
            "dead",
            Type::Ptr(Box::new(Type::Array {
                elem: Box::new(Type::I32),
                len: 8,
            })),
        );
        let live = func.add_global_ref(
            "live",
            Type::Ptr(Box::new(Type::Array {
                elem: Box::new(Type::I32),
                len: 8,
            })),
        );
        let dead_address = func
            .append_inst(
                func.entry,
                InstKind::Gep {
                    base: dead,
                    indices: vec![index],
                },
                Some(Type::Ptr(Box::new(Type::I32))),
            )
            .unwrap();
        let live_address = func
            .append_inst(
                func.entry,
                InstKind::Gep {
                    base: live,
                    indices: vec![index],
                },
                Some(Type::Ptr(Box::new(Type::I32))),
            )
            .unwrap();
        func.append_inst(
            func.entry,
            InstKind::Store {
                ptr: dead_address,
                value,
            },
            None,
        );
        func.append_inst(
            func.entry,
            InstKind::Store {
                ptr: live_address,
                value,
            },
            None,
        );
        let loaded = func
            .append_inst(
                func.entry,
                InstKind::Load { ptr: live_address },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(loaded)));
        module.add_func(func);

        GlobalWriteOnlyPass::new().run(&mut module);

        assert!(matches!(
            module.funcs[0].blocks[0].insts[2].kind,
            InstKind::Nop
        ));
        assert!(matches!(
            module.funcs[0].blocks[0].insts[3].kind,
            InstKind::Store { .. }
        ));
        assert!(module.funcs[0].verify().is_ok());
    }

    #[test]
    fn keeps_a_global_array_whose_address_escapes_to_a_call() {
        let mut module = Module::new();
        module.globals.push(Global {
            name: "output".into(),
            ty: Type::Array {
                elem: Box::new(Type::I32),
                len: 8,
            },
            is_const: false,
            init: None,
        });
        let mut func = Function::new("main", Type::I32);
        let output = func.add_global_ref(
            "output",
            Type::Ptr(Box::new(Type::Array {
                elem: Box::new(Type::I32),
                len: 8,
            })),
        );
        let zero = func.add_const(Const::Int(0));
        func.append_inst(
            func.entry,
            InstKind::Store {
                ptr: output,
                value: zero,
            },
            None,
        );
        func.append_inst(
            func.entry,
            InstKind::Call {
                name: "putarray".into(),
                args: vec![zero, output],
            },
            Some(Type::I32),
        );
        func.set_terminator(func.entry, Terminator::Return(Some(zero)));
        module.add_func(func);

        GlobalWriteOnlyPass::new().run(&mut module);

        assert!(matches!(
            module.funcs[0].blocks[0].insts[0].kind,
            InstKind::Store { .. }
        ));
    }
}
