use super::ModulePass;
use super::util::{ValueReplacements, rewrite_function_uses};
use crate::ir::{Function, InstKind, Module, Type, ValueId, ValueKind};
use std::collections::HashMap;

pub(super) struct LocalForwardPass;

impl LocalForwardPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LocalForwardPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            forward_function(func);
        }
    }
}

fn forward_function(func: &mut Function) {
    loop {
        let mut replacements = ValueReplacements::new();

        for block in &func.blocks {
            let mut known_memory = HashMap::<ValueId, ValueId>::new();
            for inst in &block.insts {
                match &inst.kind {
                    InstKind::Nop | InstKind::Alloca { .. } => {}
                    InstKind::Store { ptr, value } => {
                        let ptr = resolve(*ptr, &replacements);
                        let value = resolve(*value, &replacements);
                        if tracked_pointer(func, ptr) {
                            known_memory.insert(ptr, value);
                        } else {
                            known_memory.clear();
                        }
                    }
                    InstKind::Load { ptr } => {
                        let Some(result) = inst.result else {
                            continue;
                        };
                        let ptr = resolve(*ptr, &replacements);
                        if let Some(value) = known_memory.get(&ptr).copied() {
                            if func.value(value).ty == func.value(result).ty {
                                replacements.insert(result, value);
                            }
                        }
                    }
                    InstKind::Call { .. } | InstKind::MemZero { .. } => {
                        known_memory.clear();
                    }
                    InstKind::Gep { .. } => {}
                    _ => {}
                }
            }
        }

        if !rewrite_function_uses(func, &replacements) {
            break;
        }
    }
}

fn tracked_pointer(func: &Function, value: ValueId) -> bool {
    let Type::Ptr(inner) = &func.value(value).ty else {
        return false;
    };
    if matches!(**inner, Type::Array { .. }) {
        return false;
    }
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return false;
    };
    matches!(
        func.block(block).insts[inst_idx].kind,
        InstKind::Alloca { .. }
    )
}

fn resolve(value: ValueId, replacements: &ValueReplacements) -> ValueId {
    super::util::resolve_replacement(value, replacements)
}
