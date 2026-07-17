use crate::ir::{Function, InstKind, Module, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTIONS: usize = 1024;
const MAX_GLOBALS: usize = 8192;
const MAX_PARAMS: usize = 64;
const MAX_VALUES_PER_FUNCTION: usize = 16_384;
const MAX_POINTER_CHAIN: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MemoryRoot {
    Global(usize),
    Param(usize),
    Unknown,
}

/// Closed-world facts that are valid only for this exact module snapshot.
///
/// A parameter pair is retained only if the function has a callsite and every
/// direct call passes pointers rooted in two distinct, complete global objects.
pub(super) struct WholeProgramAliasInfo {
    global_ids: HashMap<String, usize>,
    disjoint_param_pairs: Vec<HashSet<(usize, usize)>>,
}

impl WholeProgramAliasInfo {
    pub(super) fn analyze(module: &Module) -> Self {
        let mut info = Self {
            global_ids: unique_global_ids(module),
            disjoint_param_pairs: vec![HashSet::new(); module.funcs.len()],
        };
        if module.funcs.len() > MAX_FUNCTIONS || module.globals.len() > MAX_GLOBALS {
            return info;
        }

        let user_functions = unique_function_ids(module);
        let mut saw_callsite = vec![false; module.funcs.len()];
        for (func_idx, func) in module.funcs.iter().enumerate() {
            if func.params.len() > MAX_PARAMS || func.values.len() > MAX_VALUES_PER_FUNCTION {
                continue;
            }
            let pointer_params = func
                .params
                .iter()
                .enumerate()
                .filter_map(|(index, value)| {
                    matches!(func.value(*value).ty, Type::Ptr(_)).then_some(index)
                })
                .collect::<Vec<_>>();
            for (position, lhs) in pointer_params.iter().enumerate() {
                for rhs in &pointer_params[position + 1..] {
                    info.disjoint_param_pairs[func_idx].insert((*lhs, *rhs));
                }
            }
        }

        for caller in &module.funcs {
            for block in &caller.blocks {
                for inst in &block.insts {
                    let InstKind::Call { name, args } = &inst.kind else {
                        continue;
                    };
                    let Some(Some(callee_idx)) = user_functions.get(name).copied() else {
                        continue;
                    };
                    saw_callsite[callee_idx] = true;
                    let retained = info.disjoint_param_pairs[callee_idx]
                        .iter()
                        .copied()
                        .filter(|(lhs, rhs)| {
                            let (Some(lhs), Some(rhs)) = (args.get(*lhs), args.get(*rhs)) else {
                                return false;
                            };
                            matches!(
                                (info.root(caller, *lhs), info.root(caller, *rhs)),
                                (MemoryRoot::Global(a), MemoryRoot::Global(b)) if a != b
                            )
                        })
                        .collect();
                    info.disjoint_param_pairs[callee_idx] = retained;
                }
            }
        }

        for (func_idx, saw_callsite) in saw_callsite.into_iter().enumerate() {
            if !saw_callsite {
                info.disjoint_param_pairs[func_idx].clear();
            }
        }
        info
    }

    pub(super) fn root(&self, func: &Function, mut value: ValueId) -> MemoryRoot {
        for _ in 0..MAX_POINTER_CHAIN {
            let Some(definition) = func.values.get(value.0) else {
                return MemoryRoot::Unknown;
            };
            match &definition.kind {
                ValueKind::Global(name) => {
                    return self
                        .global_ids
                        .get(name)
                        .copied()
                        .map(MemoryRoot::Global)
                        .unwrap_or(MemoryRoot::Unknown);
                }
                ValueKind::Param => {
                    return func
                        .params
                        .iter()
                        .position(|param| *param == value)
                        .map(MemoryRoot::Param)
                        .unwrap_or(MemoryRoot::Unknown);
                }
                ValueKind::Inst(block, inst_idx) => {
                    let Some(inst) = func
                        .blocks
                        .get(block.0)
                        .and_then(|block| block.insts.get(*inst_idx))
                    else {
                        return MemoryRoot::Unknown;
                    };
                    if let InstKind::Gep { base, .. } = inst.kind {
                        value = base;
                    } else {
                        return MemoryRoot::Unknown;
                    }
                }
                ValueKind::Const(_) => return MemoryRoot::Unknown,
            }
        }
        MemoryRoot::Unknown
    }

    pub(super) fn are_disjoint(&self, func_idx: usize, lhs: MemoryRoot, rhs: MemoryRoot) -> bool {
        match (lhs, rhs) {
            (MemoryRoot::Global(lhs), MemoryRoot::Global(rhs)) => lhs != rhs,
            (MemoryRoot::Param(lhs), MemoryRoot::Param(rhs)) if lhs != rhs => {
                let pair = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
                self.disjoint_param_pairs
                    .get(func_idx)
                    .is_some_and(|pairs| pairs.contains(&pair))
            }
            _ => false,
        }
    }
}

fn unique_global_ids(module: &Module) -> HashMap<String, usize> {
    let mut ids = HashMap::<String, Option<usize>>::new();
    for (index, global) in module.globals.iter().enumerate() {
        ids.entry(global.name.clone())
            .and_modify(|id| *id = None)
            .or_insert(Some(index));
    }
    ids.into_iter()
        .filter_map(|(name, id)| id.map(|id| (name, id)))
        .collect()
}

fn unique_function_ids(module: &Module) -> HashMap<String, Option<usize>> {
    let mut ids = HashMap::new();
    for (index, func) in module.funcs.iter().enumerate() {
        ids.entry(func.name.clone())
            .and_modify(|id| *id = None)
            .or_insert(Some(index));
    }
    ids
}
