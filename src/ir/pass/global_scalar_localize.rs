use super::function_effects::FunctionEffects;
use super::scalar_promote::{MAX_PROMOTION_BLOCKS, MAX_PROMOTION_VALUES};
use super::ModulePass;
use crate::ir::{Function, Inst, InstKind, Module, Terminator, Type, Value, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

pub(super) struct GlobalScalarLocalizePass {
    across_no_memory_calls: bool,
}

const MAX_LOCALIZED_GLOBALS: usize = 64;

impl GlobalScalarLocalizePass {
    pub(super) fn new() -> Self {
        Self {
            across_no_memory_calls: false,
        }
    }

    pub(super) fn new_across_no_memory_calls() -> Self {
        Self {
            across_no_memory_calls: true,
        }
    }
}

impl ModulePass for GlobalScalarLocalizePass {
    fn run(&mut self, module: &mut Module) {
        let effects = FunctionEffects::analyze(module);
        for func in &mut module.funcs {
            localize_globals(func, &effects, self.across_no_memory_calls);
        }
    }
}

#[derive(Clone)]
struct Candidate {
    global: ValueId,
    value_ty: Type,
    has_store: bool,
    valid: bool,
}

fn localize_globals(func: &mut Function, effects: &FunctionEffects, across_no_memory_calls: bool) {
    if func.blocks.len() > MAX_PROMOTION_BLOCKS || func.values.len() >= MAX_PROMOTION_VALUES {
        return;
    }
    let calls = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match &inst.kind {
            InstKind::Call { name, args } => Some((inst.result, name, args)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if calls.is_empty() == across_no_memory_calls
        || calls.iter().any(|(result, name, args)| {
            let Some(result) = result else {
                return true;
            };
            effects
                .resolve_no_memory_call(func, name, *result, args)
                .is_none()
        })
    {
        return;
    }

    let mut candidates = HashMap::<String, Candidate>::new();
    for (value_idx, value) in func.values.iter().enumerate() {
        let ValueKind::Global(name) = &value.kind else {
            continue;
        };
        let Type::Ptr(pointee) = &value.ty else {
            continue;
        };
        if !matches!(pointee.as_ref(), Type::I1 | Type::I32 | Type::F32) {
            continue;
        }
        candidates.entry(name.clone()).or_insert(Candidate {
            global: ValueId(value_idx),
            value_ty: pointee.as_ref().clone(),
            has_store: false,
            valid: true,
        });
    }
    if candidates.is_empty() {
        return;
    }

    for block in &func.blocks {
        for inst in &block.insts {
            let direct_pointer = match &inst.kind {
                InstKind::Load { ptr } | InstKind::Store { ptr, .. } => Some(*ptr),
                _ => None,
            };
            for operand in inst_operands(&inst.kind) {
                let Some(name) = global_name(func, operand) else {
                    continue;
                };
                let Some(candidate) = candidates.get_mut(name) else {
                    continue;
                };
                if direct_pointer != Some(operand) {
                    candidate.valid = false;
                } else if matches!(inst.kind, InstKind::Store { ptr, .. } if ptr == operand) {
                    candidate.has_store = true;
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            for operand in terminator_operands(terminator) {
                if let Some(name) = global_name(func, operand) {
                    if let Some(candidate) = candidates.get_mut(name) {
                        candidate.valid = false;
                    }
                }
            }
        }
    }

    let mut ordered = candidates
        .into_iter()
        .filter(|(_, candidate)| candidate.valid && candidate.has_store)
        .collect::<Vec<_>>();
    ordered.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    let return_count = func
        .blocks
        .iter()
        .filter(|block| matches!(block.terminator, Some(Terminator::Return(_))))
        .count();
    let values_per_global = 2 + return_count;
    let promotion_capacity = (MAX_PROMOTION_VALUES - func.values.len()) / values_per_global;
    ordered.truncate(MAX_LOCALIZED_GLOBALS.min(promotion_capacity));
    for (name, candidate) in ordered {
        localize_global(func, &name, &candidate);
    }
}

fn localize_global(func: &mut Function, name: &str, candidate: &Candidate) {
    let local = ValueId(func.values.len());
    func.values.push(Value {
        name: Some(format!("localized.{}", name)),
        ty: Type::Ptr(Box::new(candidate.value_ty.clone())),
        kind: ValueKind::Inst(func.entry, 0),
    });

    for block in &mut func.blocks {
        for inst in &mut block.insts {
            match &mut inst.kind {
                InstKind::Load { ptr } | InstKind::Store { ptr, .. }
                    if global_name_by_values(&func.values, *ptr) == Some(name) =>
                {
                    *ptr = local;
                }
                _ => {}
            }
        }
    }

    let initial = ValueId(func.values.len());
    func.values.push(Value {
        name: None,
        ty: candidate.value_ty.clone(),
        kind: ValueKind::Inst(func.entry, 0),
    });
    let insertion = func.blocks[func.entry.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Phi { .. }))
        .count();
    func.blocks[func.entry.0].insts.splice(
        insertion..insertion,
        [
            Inst {
                result: Some(local),
                kind: InstKind::Alloca {
                    ty: candidate.value_ty.clone(),
                },
            },
            Inst {
                result: Some(initial),
                kind: InstKind::Load {
                    ptr: candidate.global,
                },
            },
            Inst {
                result: None,
                kind: InstKind::Store {
                    ptr: local,
                    value: initial,
                },
            },
        ],
    );
    reindex_block(func, func.entry);

    let return_blocks = func
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(block_idx, block)| {
            matches!(block.terminator, Some(Terminator::Return(_)))
                .then_some(crate::ir::BlockId(block_idx))
        })
        .collect::<Vec<_>>();
    for block in return_blocks {
        let final_value = func
            .append_inst(
                block,
                InstKind::Load { ptr: local },
                Some(candidate.value_ty.clone()),
            )
            .expect("localized global load must produce a value");
        func.append_inst(
            block,
            InstKind::Store {
                ptr: candidate.global,
                value: final_value,
            },
            None,
        );
    }
}

fn reindex_block(func: &mut Function, block: crate::ir::BlockId) {
    for (inst_idx, inst) in func.blocks[block.0].insts.iter().enumerate() {
        let Some(result) = inst.result else {
            continue;
        };
        if let ValueKind::Inst(owner, old_idx) = &mut func.values[result.0].kind {
            debug_assert_eq!(*owner, block);
            *old_idx = inst_idx;
        }
    }
}

fn global_name(func: &Function, value: ValueId) -> Option<&str> {
    global_name_by_values(&func.values, value)
}

fn global_name_by_values(values: &[Value], value: ValueId) -> Option<&str> {
    match &values.get(value.0)?.kind {
        ValueKind::Global(name) => Some(name),
        _ => None,
    }
}

fn inst_operands(kind: &InstKind) -> HashSet<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => HashSet::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } => HashSet::from([*ptr]),
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => [*dst, *src, *count].into_iter().collect(),
        InstKind::Store { ptr, value } => HashSet::from([*ptr, *value]),
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => HashSet::from([*value]),
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => HashSet::from([*lhs, *rhs]),
        InstKind::Gep { base, indices } => {
            let mut operands = HashSet::from([*base]);
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Call { args, .. } => args.iter().copied().collect(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}
