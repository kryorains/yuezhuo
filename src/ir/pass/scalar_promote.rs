use super::ModulePass;
use super::util::{ValueReplacements, rewrite_function_uses};
use crate::ir::{BlockId, Function, Inst, InstKind, Module, Terminator, Type, ValueId};
use std::collections::{HashMap, HashSet};

pub(super) struct ScalarPromotePass;

impl ScalarPromotePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for ScalarPromotePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            promote_function(func);
        }
    }
}

fn promote_function(func: &mut Function) {
    let mut candidates = collect_scalar_allocas(func);
    if candidates.is_empty() {
        return;
    }

    let mut use_blocks = HashMap::<ValueId, BlockId>::new();
    let mut rejected = HashSet::<ValueId>::new();
    collect_uses(func, &candidates, &mut use_blocks, &mut rejected);
    candidates.retain(|value| use_blocks.contains_key(value) && !rejected.contains(value));
    candidates.retain(|value| has_ordered_stores_before_loads(func, *value, use_blocks[value]));
    if candidates.is_empty() {
        return;
    }

    let mut replacements = ValueReplacements::new();
    let mut current_values = HashMap::<ValueId, ValueId>::new();

    for block in &mut func.blocks {
        for inst in &mut block.insts {
            match &inst.kind {
                InstKind::Alloca { .. } if inst.result.is_some_and(|r| candidates.contains(&r)) => {
                    inst.result = None;
                    inst.kind = InstKind::Nop;
                }
                InstKind::Store { ptr, value } if candidates.contains(ptr) => {
                    current_values.insert(*ptr, *value);
                    inst.kind = InstKind::Nop;
                }
                InstKind::Load { ptr } if candidates.contains(ptr) => {
                    let result = inst.result.expect("load must have a result");
                    let value = current_values[ptr];
                    replacements.insert(result, value);
                    inst.result = None;
                    inst.kind = InstKind::Nop;
                }
                _ => {}
            }
        }
    }

    rewrite_function_uses(func, &replacements);
}

fn collect_scalar_allocas(func: &Function) -> HashSet<ValueId> {
    let mut allocas = HashSet::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let Some(result) = inst.result else {
                continue;
            };
            let InstKind::Alloca { ty } = &inst.kind else {
                continue;
            };
            if !matches!(ty, Type::Array { .. }) {
                allocas.insert(result);
            }
        }
    }
    allocas
}

fn collect_uses(
    func: &Function,
    candidates: &HashSet<ValueId>,
    use_blocks: &mut HashMap<ValueId, BlockId>,
    rejected: &mut HashSet<ValueId>,
) {
    for block_idx in 0..func.blocks.len() {
        let block = BlockId(block_idx);
        for inst in &func.blocks[block_idx].insts {
            collect_inst_uses(inst, block, candidates, use_blocks, rejected);
        }
        if let Some(terminator) = &func.blocks[block_idx].terminator {
            collect_terminator_uses(terminator, candidates, rejected);
        }
    }
}

fn collect_inst_uses(
    inst: &Inst,
    block: BlockId,
    candidates: &HashSet<ValueId>,
    use_blocks: &mut HashMap<ValueId, BlockId>,
    rejected: &mut HashSet<ValueId>,
) {
    match &inst.kind {
        InstKind::Nop | InstKind::Alloca { .. } => {}
        InstKind::Load { ptr } if candidates.contains(ptr) => record_use(*ptr, block, use_blocks),
        InstKind::Store { ptr, value } if candidates.contains(ptr) => {
            record_use(*ptr, block, use_blocks);
            reject_if_candidate(*value, candidates, rejected);
        }
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => {
            reject_if_candidate(*ptr, candidates, rejected);
        }
        InstKind::Store { ptr, value } => {
            reject_if_candidate(*ptr, candidates, rejected);
            reject_if_candidate(*value, candidates, rejected);
        }
        InstKind::Phi { incomings } => {
            for (_, value) in incomings {
                reject_if_candidate(*value, candidates, rejected);
            }
        }
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => {
            reject_if_candidate(*value, candidates, rejected);
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            reject_if_candidate(*lhs, candidates, rejected);
            reject_if_candidate(*rhs, candidates, rejected);
        }
        InstKind::Gep { base, indices } => {
            reject_if_candidate(*base, candidates, rejected);
            for index in indices {
                reject_if_candidate(*index, candidates, rejected);
            }
        }
        InstKind::Call { args, .. } => {
            for arg in args {
                reject_if_candidate(*arg, candidates, rejected);
            }
        }
    }
}

fn collect_terminator_uses(
    terminator: &Terminator,
    candidates: &HashSet<ValueId>,
    rejected: &mut HashSet<ValueId>,
) {
    match terminator {
        Terminator::Return(Some(value)) | Terminator::Branch { cond: value, .. } => {
            reject_if_candidate(*value, candidates, rejected);
        }
        Terminator::Return(None) | Terminator::Jump(_) => {}
    }
}

fn record_use(value: ValueId, block: BlockId, use_blocks: &mut HashMap<ValueId, BlockId>) {
    use_blocks
        .entry(value)
        .and_modify(|existing| {
            if *existing != block {
                *existing = BlockId(usize::MAX);
            }
        })
        .or_insert(block);
}

fn reject_if_candidate(
    value: ValueId,
    candidates: &HashSet<ValueId>,
    rejected: &mut HashSet<ValueId>,
) {
    if candidates.contains(&value) {
        rejected.insert(value);
    }
}

fn has_ordered_stores_before_loads(func: &Function, alloca: ValueId, block: BlockId) -> bool {
    if block.0 == usize::MAX {
        return false;
    }

    let mut has_value = false;
    for inst in &func.block(block).insts {
        match &inst.kind {
            InstKind::Store { ptr, .. } if *ptr == alloca => has_value = true,
            InstKind::Load { ptr } if *ptr == alloca && !has_value => return false,
            _ => {}
        }
    }
    true
}
