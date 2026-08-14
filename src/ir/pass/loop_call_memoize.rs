use super::dominators::{ControlFlowGraph, Dominators};
use super::function_effects::FunctionEffects;
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::util::{const_i32, defining_inst, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BlockId, CmpOp, Const, Function, Inst, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};

const MAX_FUNCTIONS: usize = 4096;
const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_POINTER_DEPTH: usize = 64;

/// Memoizes a no-memory call whose arguments are invariant in a counted loop.
pub(super) struct LoopCallMemoizePass;

impl LoopCallMemoizePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LoopCallMemoizePass {
    fn run(&mut self, module: &mut Module) {
        if module.funcs.len() > MAX_FUNCTIONS {
            return;
        }
        let snapshots = module.funcs.clone();
        let effects = FunctionEffects::analyze(module);

        for (func_idx, snapshot) in snapshots.iter().enumerate() {
            let candidate = find_invariant_call_candidate(snapshot, &effects);
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
                InstKind::MemCopy { .. } => false,
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
