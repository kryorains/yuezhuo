use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
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
    // 局部转发可能形成链式替换，循环到本轮没有任何使用点被改写。
    loop {
        let mut replacements = ValueReplacements::new();

        for block in &func.blocks {
            // 只在单个基本块内追踪内存状态，跨块交给更强的 SSA/phi 逻辑处理。
            let mut known_memory = HashMap::<ValueId, ValueId>::new();
            let mut known_loads = HashMap::<ValueId, ValueId>::new();
            for inst in &block.insts {
                match &inst.kind {
                    InstKind::Nop | InstKind::Alloca { .. } => {}
                    InstKind::Store { ptr, value } => {
                        // Any store may alias a previous general load. Scalar
                        // alloca forwarding remains precise, while redundant
                        // load forwarding restarts after the clobber.
                        known_loads.clear();
                        let ptr = resolve(*ptr, &replacements);
                        let value = resolve(*value, &replacements);
                        if tracked_pointer(func, ptr) {
                            known_memory.insert(ptr, value);
                        } else {
                            known_memory.clear();
                        }
                    }
                    InstKind::Load { ptr } => {
                        // 如果前面已知 ptr 里就是某个 value，就把这次 load 直接替换成 value。
                        let Some(result) = inst.result else {
                            continue;
                        };
                        let ptr = resolve(*ptr, &replacements);
                        if let Some(value) = known_memory
                            .get(&ptr)
                            .or_else(|| known_loads.get(&ptr))
                            .copied()
                        {
                            if func.value(value).ty == func.value(result).ty {
                                replacements.insert(result, value);
                            }
                        } else {
                            known_loads.insert(ptr, result);
                        }
                    }
                    InstKind::Call { .. } | InstKind::MemZero { .. } => {
                        // 调用和批量清零都可能改写内存，保守地丢弃本块内已知信息。
                        known_memory.clear();
                        known_loads.clear();
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
    // 目前只跟踪非数组 alloca，避免数组/复杂别名导致错误转发。
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
