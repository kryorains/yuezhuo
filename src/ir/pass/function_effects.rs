use crate::ir::{Function, FunctionId, InstKind, Module, Type, ValueId};
use std::collections::{HashMap, VecDeque};

// These limits bound both the symbol table and reverse call graph. Exceeding
// either limit disables every proof for the module rather than keeping a
// partial, order-dependent result.
const MAX_FUNCTIONS: usize = 4096;
const MAX_CALL_EDGES: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FunctionEffect {
    NoMemory,
    MayMemory,
}

/// Closed-world function effects for one immutable module snapshot.
///
/// Direct calls are resolved only when their exact symbol denotes one module
/// function. Names are not otherwise inspected or used as an optimization
/// heuristic.
pub(super) struct FunctionEffects {
    summaries: Vec<FunctionEffect>,
    unique_targets: HashMap<String, Option<FunctionId>>,
    signatures: Vec<Option<FunctionSignature>>,
}

impl FunctionEffects {
    pub(super) fn analyze(module: &Module) -> Self {
        if module.funcs.len() > MAX_FUNCTIONS {
            return Self::fully_conservative(
                module.funcs.len(),
                HashMap::new(),
                std::iter::repeat_with(|| None)
                    .take(module.funcs.len())
                    .collect(),
            );
        }

        let unique_targets = collect_unique_targets(module);
        let signatures = collect_signatures(module);
        let mut no_memory = vec![true; module.funcs.len()];
        let mut callers = vec![Vec::<FunctionId>::new(); module.funcs.len()];
        let mut call_edges = 0usize;

        for (func_idx, func) in module.funcs.iter().enumerate() {
            let function_id = FunctionId(func_idx);
            if resolve_unique(&unique_targets, &func.name) != Some(function_id) {
                // A duplicate definition has no unique callable identity.
                no_memory[func_idx] = false;
            }

            for inst in func.blocks.iter().flat_map(|block| &block.insts) {
                match &inst.kind {
                    InstKind::Load { .. }
                    | InstKind::Store { .. }
                    | InstKind::MemZero { .. }
                    | InstKind::MemCopy { .. } => no_memory[func_idx] = false,
                    InstKind::Call { name, args } => {
                        call_edges = call_edges.saturating_add(1);
                        if call_edges > MAX_CALL_EDGES {
                            return Self::fully_conservative(
                                module.funcs.len(),
                                unique_targets,
                                signatures,
                            );
                        }
                        if let Some(callee) = resolve_unique(&unique_targets, name) {
                            callers[callee.0].push(function_id);
                            if !signatures
                                .get(callee.0)
                                .and_then(Option::as_ref)
                                .is_some_and(|signature| {
                                    call_matches_signature(func, inst.result, args, signature)
                                })
                            {
                                no_memory[func_idx] = false;
                            }
                        } else {
                            // Unknown external and ambiguous calls may access memory.
                            no_memory[func_idx] = false;
                        }
                    }
                    InstKind::Nop
                    | InstKind::Phi { .. }
                    | InstKind::Alloca { .. }
                    | InstKind::Unary { .. }
                    | InstKind::Binary { .. }
                    | InstKind::Icmp { .. }
                    | InstKind::Fcmp { .. }
                    | InstKind::Cast { .. }
                    | InstKind::Gep { .. } => {}
                }
            }
        }

        // Greatest fixed point: begin by assuming every locally admissible
        // function is NoMemory, then propagate each disproven callee to all of
        // its callers. A recursive SCC with no rejected member remains proven.
        let mut worklist = no_memory
            .iter()
            .enumerate()
            .filter_map(|(idx, candidate)| (!candidate).then_some(FunctionId(idx)))
            .collect::<VecDeque<_>>();
        while let Some(callee) = worklist.pop_front() {
            for caller in &callers[callee.0] {
                if no_memory[caller.0] {
                    no_memory[caller.0] = false;
                    worklist.push_back(*caller);
                }
            }
        }

        let summaries = no_memory
            .into_iter()
            .map(|proven| {
                if proven {
                    FunctionEffect::NoMemory
                } else {
                    FunctionEffect::MayMemory
                }
            })
            .collect();
        Self {
            summaries,
            unique_targets,
            signatures,
        }
    }

    /// Resolves a call to its unique identity only when the callee is proven
    /// NoMemory and the result/argument types exactly match its signature.
    pub(super) fn resolve_no_memory_call(
        &self,
        caller: &Function,
        name: &str,
        result: ValueId,
        args: &[ValueId],
    ) -> Option<FunctionId> {
        let callee = resolve_unique(&self.unique_targets, name)?;
        if self.summaries.get(callee.0) != Some(&FunctionEffect::NoMemory) {
            return None;
        }
        let signature = self.signatures.get(callee.0)?.as_ref()?;
        (signature.ret != Type::Void
            && call_matches_signature(caller, Some(result), args, signature))
        .then_some(callee)
    }

    fn fully_conservative(
        function_count: usize,
        unique_targets: HashMap<String, Option<FunctionId>>,
        signatures: Vec<Option<FunctionSignature>>,
    ) -> Self {
        Self {
            summaries: vec![FunctionEffect::MayMemory; function_count],
            unique_targets,
            signatures,
        }
    }
}

struct FunctionSignature {
    ret: Type,
    params: Vec<Type>,
}

fn call_matches_signature(
    caller: &Function,
    result: Option<ValueId>,
    args: &[ValueId],
    signature: &FunctionSignature,
) -> bool {
    let result_matches = match (&signature.ret, result) {
        (Type::Void, None) => true,
        (Type::Void, Some(_)) | (_, None) => false,
        (expected, Some(result)) => caller
            .values
            .get(result.0)
            .is_some_and(|value| &value.ty == expected),
    };
    result_matches
        && args.len() == signature.params.len()
        && args.iter().zip(&signature.params).all(|(arg, expected)| {
            caller
                .values
                .get(arg.0)
                .is_some_and(|value| &value.ty == expected)
        })
}

fn collect_signatures(module: &Module) -> Vec<Option<FunctionSignature>> {
    module
        .funcs
        .iter()
        .map(|func| {
            let params = func
                .params
                .iter()
                .map(|param| func.values.get(param.0).map(|value| value.ty.clone()))
                .collect::<Option<Vec<_>>>()?;
            Some(FunctionSignature {
                ret: func.ret.clone(),
                params,
            })
        })
        .collect()
}

fn collect_unique_targets(module: &Module) -> HashMap<String, Option<FunctionId>> {
    let mut targets = HashMap::new();
    for (func_idx, func) in module.funcs.iter().enumerate() {
        targets
            .entry(func.name.clone())
            .and_modify(|target| *target = None)
            .or_insert(Some(FunctionId(func_idx)));
    }
    targets
}

fn resolve_unique(targets: &HashMap<String, Option<FunctionId>>, name: &str) -> Option<FunctionId> {
    targets.get(name).copied().flatten()
}
