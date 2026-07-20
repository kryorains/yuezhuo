use crate::ir::{Function, InstKind, Module, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTIONS: usize = 1024;
const MAX_GLOBALS: usize = 8192;
const MAX_PARAMS: usize = 64;
const MAX_VALUES_PER_FUNCTION: usize = 16_384;
const MAX_POINTER_CHAIN: usize = 128;
const MAX_CALL_SITES: usize = 65_536;
const MAX_PARAM_GLOBAL_ROOTS: usize = 65_536;
const MAX_TOTAL_PARAMS: usize = 65_536;
const MAX_TOTAL_VALUES: usize = 1_048_576;
const MAX_CALL_ARGUMENTS: usize = 262_144;
const MAX_PARAM_PAIR_WORK: usize = 262_144;

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
    /// Complete global objects observed for each pointer parameter at every
    /// closed-world direct callsite. `None` means at least one call passed an
    /// unknown root or analysis exceeded its budget.
    param_global_roots: Vec<Vec<Option<HashSet<usize>>>>,
}

impl WholeProgramAliasInfo {
    pub(super) fn analyze(module: &Module) -> Self {
        let total_params = module
            .funcs
            .iter()
            .try_fold(0usize, |total, func| total.checked_add(func.params.len()));
        let total_values = module
            .funcs
            .iter()
            .try_fold(0usize, |total, func| total.checked_add(func.values.len()));
        let within_module_budget = module.funcs.len() <= MAX_FUNCTIONS
            && module.globals.len() <= MAX_GLOBALS
            && total_params.is_some_and(|total| total <= MAX_TOTAL_PARAMS)
            && total_values.is_some_and(|total| total <= MAX_TOTAL_VALUES);
        let mut info = Self {
            global_ids: if within_module_budget {
                unique_global_ids(module)
            } else {
                HashMap::new()
            },
            disjoint_param_pairs: vec![HashSet::new(); module.funcs.len()],
            param_global_roots: module
                .funcs
                .iter()
                .map(|func| {
                    if !within_module_budget
                        || func.params.len() > MAX_PARAMS
                        || func.values.len() > MAX_VALUES_PER_FUNCTION
                    {
                        return Vec::new();
                    }
                    func.params
                        .iter()
                        .map(|param| {
                            matches!(func.value(*param).ty, Type::Ptr(_)).then(HashSet::new)
                        })
                        .collect()
                })
                .collect(),
        };
        if !within_module_budget {
            return info;
        }

        let user_functions = unique_function_ids(module);
        let mut saw_callsite = vec![false; module.funcs.len()];
        let mut call_sites = 0usize;
        let mut param_global_roots = 0usize;
        let mut call_arguments = 0usize;
        let mut param_pair_work = 0usize;
        let mut analysis_budget_exceeded = false;
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

        'callers: for caller in &module.funcs {
            for block in &caller.blocks {
                for inst in &block.insts {
                    let InstKind::Call { name, args } = &inst.kind else {
                        continue;
                    };
                    let Some(Some(callee_idx)) = user_functions.get(name).copied() else {
                        continue;
                    };
                    call_sites = call_sites.saturating_add(1);
                    call_arguments = call_arguments.saturating_add(args.len());
                    if call_sites > MAX_CALL_SITES || call_arguments > MAX_CALL_ARGUMENTS {
                        analysis_budget_exceeded = true;
                        break 'callers;
                    }
                    saw_callsite[callee_idx] = true;
                    let argument_roots = args
                        .iter()
                        .map(|arg| info.root(caller, *arg))
                        .collect::<Vec<_>>();
                    let existing_pairs = info.disjoint_param_pairs[callee_idx]
                        .iter()
                        .copied()
                        .collect::<Vec<_>>();
                    let mut retained = HashSet::new();
                    for (lhs, rhs) in existing_pairs {
                        param_pair_work = param_pair_work.saturating_add(1);
                        if param_pair_work > MAX_PARAM_PAIR_WORK {
                            analysis_budget_exceeded = true;
                            break 'callers;
                        }
                        if matches!(
                            (argument_roots.get(lhs), argument_roots.get(rhs)),
                            (Some(MemoryRoot::Global(a)), Some(MemoryRoot::Global(b))) if a != b
                        ) {
                            retained.insert((lhs, rhs));
                        }
                    }
                    info.disjoint_param_pairs[callee_idx] = retained;

                    for (param_idx, roots) in
                        info.param_global_roots[callee_idx].iter_mut().enumerate()
                    {
                        let Some(known_roots) = roots else {
                            continue;
                        };
                        match argument_roots.get(param_idx) {
                            Some(MemoryRoot::Global(global)) => {
                                if known_roots.insert(*global) {
                                    param_global_roots = param_global_roots.saturating_add(1);
                                    if param_global_roots > MAX_PARAM_GLOBAL_ROOTS {
                                        analysis_budget_exceeded = true;
                                        break 'callers;
                                    }
                                }
                            }
                            _ => *roots = None,
                        }
                    }
                }
            }
        }

        if analysis_budget_exceeded {
            for pairs in &mut info.disjoint_param_pairs {
                pairs.clear();
            }
            for roots in &mut info.param_global_roots {
                roots.clear();
            }
            return info;
        }

        for (func_idx, saw_callsite) in saw_callsite.into_iter().enumerate() {
            if !saw_callsite {
                info.disjoint_param_pairs[func_idx].clear();
                for roots in &mut info.param_global_roots[func_idx] {
                    *roots = None;
                }
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
            (MemoryRoot::Global(global), MemoryRoot::Param(param))
            | (MemoryRoot::Param(param), MemoryRoot::Global(global)) => self
                .param_global_roots
                .get(func_idx)
                .and_then(|roots| roots.get(param))
                .and_then(Option::as_ref)
                .is_some_and(|roots| !roots.is_empty() && !roots.contains(&global)),
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
