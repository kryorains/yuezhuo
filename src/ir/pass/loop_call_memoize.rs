use super::dominators::{ControlFlowGraph, Dominators};
use super::function_effects::FunctionEffects;
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::util::{const_i32, defining_inst, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BlockId, CmpOp, Const, Function, FunctionId, Inst, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTIONS: usize = 4096;
const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_INSTRUCTIONS: usize = 32_768;
const MAX_POINTER_DEPTH: usize = 64;

/// Memoizes a read-only call in a counted outer loop after proving that every
/// object it reads is initialized by a persistent inner induction exactly
/// once. This covers source loops whose initialization counter is deliberately
/// not reset by the outer loop.
pub(super) struct LoopCallMemoizePass {
    persistent_initialization: bool,
    invariant_calls: bool,
}

impl LoopCallMemoizePass {
    pub(super) fn new() -> Self {
        Self {
            persistent_initialization: true,
            invariant_calls: false,
        }
    }

    pub(super) fn new_invariant_calls() -> Self {
        Self {
            persistent_initialization: false,
            invariant_calls: true,
        }
    }
}

impl ModulePass for LoopCallMemoizePass {
    fn run(&mut self, module: &mut Module) {
        if module.funcs.len() > MAX_FUNCTIONS {
            return;
        }
        let snapshots = module.funcs.clone();
        let effects = FunctionEffects::analyze(module);
        let targets = unique_targets(&snapshots);
        let readonly = snapshots
            .iter()
            .map(analyze_readonly_arg_callee)
            .collect::<Vec<_>>();

        for func_idx in 0..module.funcs.len() {
            let candidate = self
                .persistent_initialization
                .then(|| find_candidate(&snapshots[func_idx], &snapshots, &targets, &readonly))
                .flatten()
                .or_else(|| {
                    self.invariant_calls
                        .then(|| find_invariant_call_candidate(&snapshots[func_idx], &effects))?
                });
            let Some(candidate) = candidate else {
                continue;
            };
            apply_candidate(&mut module.funcs[func_idx], candidate);
            if let Err(errors) = module.funcs[func_idx].verify() {
                panic!(
                    "loop call memoization produced invalid IR in {}: {:?}",
                    module.funcs[func_idx].name, errors
                );
            }
        }
    }
}

fn find_invariant_call_candidate(func: &Function, effects: &FunctionEffects) -> Option<Candidate> {
    if func.blocks.is_empty() || func.blocks.len() > MAX_BLOCKS || func.values.len() > MAX_VALUES {
        return None;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    for natural_loop in loop_info.loops() {
        let (Some(outer_entering), Some(outer_latch)) = (
            natural_loop.unique_entering_pred,
            natural_loop.unique_latch(),
        ) else {
            continue;
        };
        let Some((outer_counter, outer_initial)) = canonical_outer_counter(func, natural_loop)
        else {
            continue;
        };
        for (call_idx, inst) in func.blocks[outer_latch.0].insts.iter().enumerate() {
            let (Some(call_result), InstKind::Call { name, args }) = (inst.result, &inst.kind)
            else {
                continue;
            };
            if !matches!(func.value(call_result).ty, Type::I1 | Type::I32 | Type::F32)
                || effects
                    .resolve_no_memory_call(func, name, call_result, args)
                    .is_none()
                || args
                    .iter()
                    .any(|arg| !value_is_loop_invariant(func, natural_loop, *arg, effects))
            {
                continue;
            }
            return Some(Candidate {
                outer_header: natural_loop.header,
                outer_entering,
                outer_latch,
                outer_counter,
                outer_initial,
                call_idx,
                call_result,
            });
        }
    }
    None
}

fn value_is_loop_invariant(
    func: &Function,
    natural_loop: &NaturalLoop,
    value: ValueId,
    effects: &FunctionEffects,
) -> bool {
    let Some(value_data) = func.values.get(value.0) else {
        return false;
    };
    let ValueKind::Inst(block, inst_idx) = value_data.kind else {
        return true;
    };
    if !natural_loop.blocks.contains(&block) {
        return true;
    }
    let Some(InstKind::Load { ptr }) = func
        .blocks
        .get(block.0)
        .and_then(|block| block.insts.get(inst_idx))
        .map(|inst| &inst.kind)
    else {
        return false;
    };
    let Some(global) = direct_global_name(func, *ptr) else {
        return false;
    };
    natural_loop.blocks.iter().all(|block| {
        func.blocks[block.0]
            .insts
            .iter()
            .all(|inst| match &inst.kind {
                InstKind::Store { ptr, .. } | InstKind::MemZero { ptr, .. } => {
                    pointer_root_global(func, *ptr).is_some_and(|root| root != global)
                }
                InstKind::Call { name, args } => inst.result.is_some_and(|result| {
                    effects
                        .resolve_no_memory_call(func, name, result, args)
                        .is_some()
                }),
                _ => true,
            })
    })
}

fn direct_global_name(func: &Function, value: ValueId) -> Option<&str> {
    match &func.values.get(value.0)?.kind {
        ValueKind::Global(name) => Some(name),
        _ => None,
    }
}

fn pointer_root_global(func: &Function, mut value: ValueId) -> Option<&str> {
    for _ in 0..MAX_POINTER_DEPTH {
        match &func.values.get(value.0)?.kind {
            ValueKind::Global(name) => return Some(name),
            ValueKind::Inst(block, inst_idx) => {
                let InstKind::Gep { base, .. } =
                    &func.blocks.get(block.0)?.insts.get(*inst_idx)?.kind
                else {
                    return None;
                };
                value = *base;
            }
            ValueKind::Param | ValueKind::Const(_) => return None,
        }
    }
    None
}

#[derive(Clone)]
struct ReadonlyArgCallee {
    read_pointer_params: HashSet<usize>,
}

#[derive(Clone, Copy)]
struct Candidate {
    outer_header: BlockId,
    outer_entering: BlockId,
    outer_latch: BlockId,
    outer_counter: ValueId,
    outer_initial: ValueId,
    call_idx: usize,
    call_result: ValueId,
}

fn find_candidate(
    func: &Function,
    funcs: &[Function],
    targets: &HashMap<String, Option<FunctionId>>,
    readonly: &[Option<ReadonlyArgCallee>],
) -> Option<Candidate> {
    if func.blocks.is_empty()
        || func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || func
            .blocks
            .iter()
            .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
            .is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return None;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    for outer in loop_info.loops() {
        let (Some(outer_entering), Some(outer_latch)) =
            (outer.unique_entering_pred, outer.unique_latch())
        else {
            continue;
        };
        let Some((outer_counter, outer_initial)) = canonical_outer_counter(func, outer) else {
            continue;
        };
        for (call_idx, inst) in func.blocks[outer_latch.0].insts.iter().enumerate() {
            let (Some(call_result), InstKind::Call { name, args }) = (inst.result, &inst.kind)
            else {
                continue;
            };
            if !matches!(func.value(call_result).ty, Type::I1 | Type::I32 | Type::F32) {
                continue;
            }
            let Some(callee_id) = targets.get(name).copied().flatten() else {
                continue;
            };
            let Some(summary) = readonly.get(callee_id.0).and_then(Option::as_ref) else {
                continue;
            };
            if !call_matches(func, call_result, args, &funcs[callee_id.0])
                || args
                    .iter()
                    .any(|arg| value_defined_in_loop(func, outer, *arg))
            {
                continue;
            }
            let Some(read_roots) = caller_read_roots(func, args, summary) else {
                continue;
            };
            let Some(init_loop) = find_persistent_init_loop(
                func,
                outer,
                outer_latch,
                loop_info.loops(),
                &read_roots,
                &dom,
            ) else {
                continue;
            };
            if !init_loop
                .blocks
                .iter()
                .all(|block| outer.blocks.contains(block))
                || !outer_effects_fit(func, outer, init_loop, outer_latch, call_idx, &read_roots)
            {
                continue;
            }
            return Some(Candidate {
                outer_header: outer.header,
                outer_entering,
                outer_latch,
                outer_counter,
                outer_initial,
                call_idx,
                call_result,
            });
        }
    }
    None
}

fn canonical_outer_counter(func: &Function, outer: &NaturalLoop) -> Option<(ValueId, ValueId)> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks.get(outer.header.0)?.terminator.as_ref()?
    else {
        return None;
    };
    let then_inside = outer.blocks.contains(then_target);
    let else_inside = outer.blocks.contains(else_target);
    if then_inside == else_inside {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt, lhs, ..
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    let induction = analyze_i32_induction(func, outer, *lhs)?;
    (induction.step == 1 && const_i32(func, induction.initial) == Some(0))
        .then_some((induction.phi, induction.initial))
}

fn find_persistent_init_loop<'a>(
    func: &Function,
    outer: &NaturalLoop,
    call_block: BlockId,
    loops: &'a [NaturalLoop],
    read_roots: &HashSet<ValueId>,
    dom: &Dominators,
) -> Option<&'a NaturalLoop> {
    let outer_entering = outer.unique_entering_pred?;
    let outer_latch = outer.unique_latch()?;
    loops
        .iter()
        .filter(|inner| {
            inner.header != outer.header
                && inner.blocks.len() < outer.blocks.len()
                && inner
                    .blocks
                    .iter()
                    .all(|block| outer.blocks.contains(block))
                && inner.unique_exit() == Some(call_block)
        })
        .find(|inner| {
            let (Some(inner_entering), Some(_)) =
                (inner.unique_entering_pred, inner.unique_latch())
            else {
                return false;
            };
            let Some((induction, bound)) = canonical_inner_counter(func, inner) else {
                return false;
            };
            let Some(start) = outer_phi_pair(
                func,
                outer.header,
                induction.initial,
                outer_entering,
                outer_latch,
            ) else {
                return false;
            };
            let Some(start) = const_i32(func, start) else {
                return false;
            };
            let Some(bound) = const_i32(func, bound) else {
                return false;
            };
            if induction.step != 1
                || start < 0
                || bound <= start
                || !dom.dominates(inner.header, call_block)
                || func.blocks.get(inner_entering.0).is_none()
            {
                return false;
            }
            outer_phi_backedge(func, induction.initial, outer_latch) == Some(induction.phi)
                && inner_has_relevant_store(func, inner, read_roots)
        })
}

fn canonical_inner_counter(
    func: &Function,
    inner: &NaturalLoop,
) -> Option<(super::loop_analysis::InductionVariable, ValueId)> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks.get(inner.header.0)?.terminator.as_ref()?
    else {
        return None;
    };
    if !inner.blocks.contains(then_target) || inner.blocks.contains(else_target) {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt,
        lhs,
        rhs,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    let induction = analyze_i32_induction(func, inner, *lhs)?;
    Some((induction, *rhs))
}

fn outer_phi_pair(
    func: &Function,
    outer_header: BlockId,
    phi: ValueId,
    entering: BlockId,
    latch: BlockId,
) -> Option<ValueId> {
    let ValueKind::Inst(owner, inst_idx) = func.values.get(phi.0)?.kind else {
        return None;
    };
    if owner != outer_header {
        return None;
    }
    let InstKind::Phi { incomings } = &func.blocks.get(owner.0)?.insts.get(inst_idx)?.kind else {
        return None;
    };
    let initial = unique_incoming(incomings, entering)?;
    unique_incoming(incomings, latch)?;
    Some(initial)
}

fn outer_phi_backedge(func: &Function, phi: ValueId, latch: BlockId) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    unique_incoming(incomings, latch)
}

fn inner_has_relevant_store(
    func: &Function,
    inner: &NaturalLoop,
    read_roots: &HashSet<ValueId>,
) -> bool {
    inner.blocks.iter().any(|block| {
        func.blocks[block.0].insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::Store { ptr, .. }
                    if pointer_root(func, ptr).is_some_and(|root| read_roots.contains(&root))
            )
        })
    })
}

fn outer_effects_fit(
    func: &Function,
    outer: &NaturalLoop,
    init_loop: &NaturalLoop,
    call_block: BlockId,
    call_idx: usize,
    read_roots: &HashSet<ValueId>,
) -> bool {
    for block in &outer.blocks {
        for (inst_idx, inst) in func.blocks[block.0].insts.iter().enumerate() {
            match &inst.kind {
                InstKind::Store { ptr, .. } => {
                    if !init_loop.blocks.contains(block)
                        || pointer_root(func, *ptr).is_none_or(|root| !read_roots.contains(&root))
                    {
                        return false;
                    }
                }
                InstKind::MemZero { .. } => return false,
                InstKind::Call { .. } if *block != call_block || inst_idx != call_idx => {
                    return false;
                }
                _ => {}
            }
        }
    }
    true
}

fn analyze_readonly_arg_callee(func: &Function) -> Option<ReadonlyArgCallee> {
    if func.blocks.is_empty()
        || func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || !matches!(func.ret, Type::I1 | Type::I32 | Type::F32)
    {
        return None;
    }
    let params = func
        .params
        .iter()
        .enumerate()
        .map(|(idx, param)| (*param, idx))
        .collect::<HashMap<_, _>>();
    let mut read_pointer_params = HashSet::new();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        match &inst.kind {
            InstKind::Load { ptr } => {
                let root = pointer_root(func, *ptr)?;
                let param = *params.get(&root)?;
                if !matches!(func.value(root).ty, Type::Ptr(_)) {
                    return None;
                }
                read_pointer_params.insert(param);
            }
            InstKind::Store { .. } | InstKind::MemZero { .. } | InstKind::Call { .. } => {
                return None;
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
    (!read_pointer_params.is_empty()).then_some(ReadonlyArgCallee {
        read_pointer_params,
    })
}

fn caller_read_roots(
    caller: &Function,
    args: &[ValueId],
    summary: &ReadonlyArgCallee,
) -> Option<HashSet<ValueId>> {
    summary
        .read_pointer_params
        .iter()
        .map(|idx| pointer_root(caller, *args.get(*idx)?))
        .collect()
}

fn pointer_root(func: &Function, mut value: ValueId) -> Option<ValueId> {
    for _ in 0..MAX_POINTER_DEPTH {
        let Some(InstKind::Gep { base, .. }) = defining_inst(func, value) else {
            return matches!(func.values.get(value.0)?.ty, Type::Ptr(_)).then_some(value);
        };
        value = *base;
    }
    None
}

fn value_defined_in_loop(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Inst(block, _)) if natural_loop.blocks.contains(block)
    )
}

fn call_matches(caller: &Function, result: ValueId, args: &[ValueId], callee: &Function) -> bool {
    caller.value(result).ty == callee.ret
        && args.len() == callee.params.len()
        && args.iter().zip(&callee.params).all(|(arg, param)| {
            call_arg_type_matches(&caller.value(*arg).ty, &callee.value(*param).ty)
        })
}

fn call_arg_type_matches(actual: &Type, expected: &Type) -> bool {
    actual == expected
        || matches!(
            (actual, expected),
            (
                Type::Ptr(actual),
                Type::Ptr(expected)
            ) if matches!(
                actual.as_ref(),
                Type::Array { elem, .. } if elem.as_ref() == expected.as_ref()
            )
        )
}

fn apply_candidate(func: &mut Function, candidate: Candidate) {
    let call = func.blocks[candidate.outer_latch.0].insts[candidate.call_idx].clone();
    let result_ty = func.value(candidate.call_result).ty.clone();
    let suffix = func.blocks[candidate.outer_latch.0]
        .insts
        .split_off(candidate.call_idx + 1);
    let old_terminator = func.blocks[candidate.outer_latch.0]
        .terminator
        .take()
        .expect("verified loop latch must terminate");
    func.blocks[candidate.outer_latch.0].insts[candidate.call_idx] = Inst {
        result: None,
        kind: InstKind::Nop,
    };

    let execute = func.add_block("memoize.execute");
    let reuse = func.add_block("memoize.reuse");
    let merge = func.add_block("memoize.merge");
    func.blocks[execute.0].insts.push(call);
    func.values[candidate.call_result.0].kind = ValueKind::Inst(execute, 0);

    let zero = get_or_add_zero(func, &result_ty);
    let phi_end = func.blocks[candidate.outer_header.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }))
        .count();
    let cached = func
        .insert_inst(
            candidate.outer_header,
            phi_end,
            InstKind::Phi {
                incomings: vec![
                    (candidate.outer_entering, zero),
                    (merge, candidate.call_result),
                ],
            },
            Some(result_ty.clone()),
        )
        .expect("memoized value needs a loop phi");

    let memoized = func
        .append_inst(
            merge,
            InstKind::Phi {
                incomings: vec![(execute, candidate.call_result), (reuse, cached)],
            },
            Some(result_ty),
        )
        .expect("memoized branches need a merge phi");
    let merge_prefix = func.blocks[merge.0].insts.len();
    func.blocks[merge.0].insts.extend(suffix);
    for value in &mut func.values {
        let ValueKind::Inst(owner, inst_idx) = &mut value.kind else {
            continue;
        };
        if *owner == candidate.outer_latch && *inst_idx > candidate.call_idx {
            *owner = merge;
            *inst_idx = merge_prefix + (*inst_idx - candidate.call_idx - 1);
        }
    }

    rewrite_function_uses(
        func,
        &ValueReplacements::from([(candidate.call_result, memoized)]),
    );
    let InstKind::Phi { incomings } = &mut func.blocks[merge.0].insts[0].kind else {
        unreachable!("memoized result must remain a phi");
    };
    incomings[0].1 = candidate.call_result;

    retarget_successor_phis(func, &old_terminator, candidate.outer_latch, merge);
    let is_first = func
        .append_inst(
            candidate.outer_latch,
            InstKind::Icmp {
                op: CmpOp::Eq,
                lhs: candidate.outer_counter,
                rhs: candidate.outer_initial,
            },
            Some(Type::I1),
        )
        .expect("first-iteration check must produce i1");
    func.blocks[candidate.outer_latch.0].terminator = Some(Terminator::Branch {
        cond: is_first,
        then_target: execute,
        else_target: reuse,
    });
    func.set_terminator(execute, Terminator::Jump(merge));
    func.set_terminator(reuse, Terminator::Jump(merge));
    func.blocks[merge.0].terminator = Some(old_terminator);
}

fn retarget_successor_phis(
    func: &mut Function,
    terminator: &Terminator,
    old_pred: BlockId,
    new_pred: BlockId,
) {
    let successors = match terminator {
        Terminator::Return(_) => Vec::new(),
        Terminator::Jump(target) => vec![*target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } if then_target == else_target => vec![*then_target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => vec![*then_target, *else_target],
    };
    for successor in successors {
        for inst in &mut func.blocks[successor.0].insts {
            if let InstKind::Phi { incomings } = &mut inst.kind {
                for (pred, _) in incomings {
                    if *pred == old_pred {
                        *pred = new_pred;
                    }
                }
            }
        }
    }
}

fn get_or_add_zero(func: &mut Function, ty: &Type) -> ValueId {
    let constant = match ty {
        Type::I1 => Const::Bool(false),
        Type::I32 => Const::Int(0),
        Type::F32 => Const::Float(0),
        _ => unreachable!("memoized call result must be scalar"),
    };
    func.values
        .iter()
        .position(|value| value.ty == *ty && value.kind == ValueKind::Const(constant.clone()))
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(constant))
}

fn unique_incoming(incomings: &[(BlockId, ValueId)], pred: BlockId) -> Option<ValueId> {
    let mut matches = incomings
        .iter()
        .filter_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value));
    let value = matches.next()?;
    matches.next().is_none().then_some(value)
}

fn unique_targets(funcs: &[Function]) -> HashMap<String, Option<FunctionId>> {
    let mut targets = HashMap::new();
    for (idx, func) in funcs.iter().enumerate() {
        targets
            .entry(func.name.clone())
            .and_modify(|target| *target = None)
            .or_insert(Some(FunctionId(idx)));
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, OptLevel, PassOptions};
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    #[test]
    fn memoizes_a_readonly_call_after_persistent_initialization() {
        let mut module = optimized_module(
            r#"
float total(float values[], int count) {
    int i = 0;
    float sum = 0;
    while (i < count) {
        sum = sum + values[i];
        i = i + 1;
    }
    return sum;
}

int main() {
    float values[4];
    int initialized = 0;
    int iteration = 0;
    float answer = 0;
    while (iteration < 3) {
        while (initialized < 4) {
            values[initialized] = initialized;
            initialized = initialized + 1;
        }
        answer = answer + total(values, 4);
        iteration = iteration + 1;
    }
    return answer > 0;
}
"#,
            true,
        );
        let main = function(&module, "main");
        assert_eq!(call_count(main, "total"), 1);
        assert!(main
            .blocks
            .iter()
            .any(|block| block.name == "memoize.execute"));
        assert!(main.verify().is_ok());

        let once = module.clone();
        LoopCallMemoizePass::new().run(&mut module);
        assert_eq!(module, once);
    }

    #[test]
    fn rejects_a_call_when_the_read_object_is_mutated_each_iteration() {
        let module = optimized_module(
            r#"
float total(float values[], int count) {
    int i = 0;
    float sum = 0;
    while (i < count) {
        sum = sum + values[i];
        i = i + 1;
    }
    return sum;
}

int main() {
    float values[4];
    int initialized = 0;
    int iteration = 0;
    float answer = 0;
    while (iteration < 3) {
        while (initialized < 4) {
            values[initialized] = initialized;
            initialized = initialized + 1;
        }
        values[0] = iteration;
        answer = answer + total(values, 4);
        iteration = iteration + 1;
    }
    return answer > 0;
}
"#,
            true,
        );
        assert!(!function(&module, "main")
            .blocks
            .iter()
            .any(|block| block.name.starts_with("memoize.")));
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    fn optimized_module(source: &str, enable_loop_call_memoize: bool) -> Module {
        let program = Parser::new(source).parse_program();
        let mut module = lower_program(&program).expect("test source must lower");
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                small_expr_inline_rounds: 1,
                cfg_inline_rounds: 0,
                cfg_inline_global_loads: false,
                enable_constant_address_count_reduction: false,
                enable_recursive_const_specialization: false,
                enable_loop_call_memoize,
                enable_loop_invariant_call_memoize: false,
                enable_repeated_overwrite_elision: false,
                enable_guarded_mulmod_idiom: false,
                enable_guarded_pow2_digit_idiom: false,
                enable_regional_global_scalar_promotion: false,
                enable_producer_consumer_fusion: false,
                enable_periodic_reduction_memoize: false,
                enable_write_only_alloca_cleanup_before_inline: false,
            },
        );
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
        module
    }

    fn function<'a>(module: &'a Module, name: &str) -> &'a Function {
        module
            .funcs
            .iter()
            .find(|func| func.name == name)
            .expect("test function must exist")
    }

    fn call_count(func: &Function, name: &str) -> usize {
        func.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter(|inst| {
                matches!(
                    &inst.kind,
                    InstKind::Call {
                        name: target,
                        ..
                    } if target == name
                )
            })
            .count()
    }
}
