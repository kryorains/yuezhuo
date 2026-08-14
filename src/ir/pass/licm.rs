use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{BinaryOp, BlockId, Function, Inst, InstKind, Module, ValueId, ValueKind};
use std::collections::{HashMap, VecDeque};

pub(super) struct LicmPass;

impl LicmPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LicmPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            licm_function(func);
        }
    }
}

fn licm_function(func: &mut Function) {
    if func.blocks.len() > 1024 || func.values.len() > 8192 {
        return;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let loops = loop_info.loops().to_vec();
    if loops.is_empty() {
        return;
    }

    let mut changed = false;
    for natural_loop in loops {
        changed |= hoist_loop(func, &dom, &natural_loop);
    }

    if changed {
        if let Err(errors) = func.verify() {
            panic!("licm produced invalid IR in {}: {:?}", func.name, errors);
        }
    }
}

fn hoist_loop(func: &mut Function, dom: &Dominators, natural_loop: &NaturalLoop) -> bool {
    let Some(preheader) = natural_loop.dedicated_preheader else {
        // Creating a dedicated preheader requires a separate CFG transformation.
        return false;
    };
    let order = collect_hoist_order(func, dom, natural_loop, preheader);
    for (block, inst_idx) in &order {
        let inst = func.blocks[block.0].insts[*inst_idx].clone();
        move_inst_to_preheader(func, *block, *inst_idx, preheader, inst);
    }
    !order.is_empty()
}

struct HoistCandidate {
    block: BlockId,
    inst_idx: usize,
    pending_dependencies: usize,
    blocked: bool,
}

fn collect_hoist_order(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    preheader: BlockId,
) -> Vec<(BlockId, usize)> {
    let mut candidates = Vec::<HoistCandidate>::new();
    let mut candidate_for_value = HashMap::<ValueId, usize>::new();
    for block in sorted_loop_blocks(natural_loop) {
        for (inst_idx, inst) in func.blocks[block.0].insts.iter().enumerate() {
            let Some(result) = inst.result else {
                continue;
            };
            if !is_safe_to_hoist(&inst.kind) {
                continue;
            }
            candidate_for_value.insert(result, candidates.len());
            candidates.push(HoistCandidate {
                block,
                inst_idx,
                pending_dependencies: 0,
                blocked: false,
            });
        }
    }

    let mut dependents = vec![Vec::<usize>::new(); candidates.len()];
    let candidate_locations = candidates
        .iter()
        .map(|candidate| (candidate.block, candidate.inst_idx))
        .collect::<Vec<_>>();
    for (candidate_idx, (block, inst_idx)) in candidate_locations.into_iter().enumerate() {
        let kind = &func.blocks[block.0].insts[inst_idx].kind;
        let mut dependencies = Vec::new();
        for operand in operands(kind) {
            match &func.value(operand).kind {
                ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => {}
                ValueKind::Inst(block, _) if natural_loop.blocks.contains(block) => {
                    let Some(dependency) = candidate_for_value.get(&operand).copied() else {
                        candidates[candidate_idx].blocked = true;
                        continue;
                    };
                    if !dependencies.contains(&dependency) {
                        dependencies.push(dependency);
                    }
                }
                ValueKind::Inst(block, _) => {
                    if *block != preheader && !dom.dominates(*block, preheader) {
                        candidates[candidate_idx].blocked = true;
                    }
                }
            }
        }
        candidates[candidate_idx].pending_dependencies = dependencies.len();
        for dependency in dependencies {
            dependents[dependency].push(candidate_idx);
        }
    }

    // Kahn-style dependency scheduling makes the move order independent of
    // BlockId allocation and appends every definition before its users in one
    // pass. Cyclic or unsafe dependency groups simply never become ready.
    let mut ready = (0..candidates.len())
        .filter(|candidate| {
            !candidates[*candidate].blocked && candidates[*candidate].pending_dependencies == 0
        })
        .collect::<VecDeque<_>>();
    let mut order = Vec::new();
    while let Some(candidate_idx) = ready.pop_front() {
        let candidate = &candidates[candidate_idx];
        order.push((candidate.block, candidate.inst_idx));
        for dependent in &dependents[candidate_idx] {
            candidates[*dependent].pending_dependencies -= 1;
            if !candidates[*dependent].blocked && candidates[*dependent].pending_dependencies == 0 {
                ready.push_back(*dependent);
            }
        }
    }
    order
}

fn move_inst_to_preheader(
    func: &mut Function,
    block: BlockId,
    inst_idx: usize,
    preheader: BlockId,
    inst: Inst,
) {
    let Some(result) = inst.result else {
        return;
    };

    func.blocks[block.0].insts[inst_idx].result = None;
    func.blocks[block.0].insts[inst_idx].kind = InstKind::Nop;

    let new_idx = func.blocks[preheader.0].insts.len();
    func.blocks[preheader.0].insts.push(inst);
    func.values[result.0].kind = ValueKind::Inst(preheader, new_idx);
}

fn operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut values = Vec::with_capacity(indices.len() + 1);
            values.push(*base);
            values.extend(indices.iter().copied());
            values
        }
        InstKind::Nop
        | InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::MemZero { .. }
        | InstKind::MemCopy { .. }
        | InstKind::Call { .. } => Vec::new(),
    }
}

fn is_safe_to_hoist(kind: &InstKind) -> bool {
    match kind {
        InstKind::Unary { .. }
        | InstKind::Cast { .. }
        | InstKind::Icmp { .. }
        | InstKind::Fcmp { .. }
        | InstKind::Gep { .. } => true,
        InstKind::Binary { op, .. } => {
            !matches!(op, BinaryOp::Idiv | BinaryOp::Imod | BinaryOp::Fdiv)
        }
        InstKind::Nop
        | InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::MemZero { .. }
        | InstKind::MemCopy { .. }
        | InstKind::Call { .. } => false,
    }
}

fn sorted_loop_blocks(natural_loop: &NaturalLoop) -> Vec<BlockId> {
    let mut blocks = natural_loop.blocks.iter().copied().collect::<Vec<_>>();
    blocks.sort_by_key(|block| (*block != natural_loop.header, block.0));
    blocks
}
