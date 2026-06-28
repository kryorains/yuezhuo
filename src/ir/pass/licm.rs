use super::dominators::{ControlFlowGraph, Dominators};
use super::ModulePass;
use crate::ir::{BinaryOp, BlockId, Function, Inst, InstKind, Module, ValueId, ValueKind};
use std::collections::HashSet;

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
    let loops = find_natural_loops(func, &cfg, &dom);
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

#[derive(Debug, Clone)]
struct NaturalLoop {
    header: BlockId,
    preheader: BlockId,
    blocks: HashSet<BlockId>,
}

fn find_natural_loops(
    func: &Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
) -> Vec<NaturalLoop> {
    let mut loops = Vec::new();

    for tail_idx in 0..func.blocks.len() {
        let tail = BlockId(tail_idx);
        for header in &cfg.succs[tail_idx] {
            if !dom.dominates(*header, tail) {
                continue;
            }
            let blocks = collect_loop_blocks(cfg, *header, tail);
            let outside_preds = cfg.preds[header.0]
                .iter()
                .copied()
                .filter(|pred| !blocks.contains(pred))
                .collect::<Vec<_>>();
            // 先只处理有唯一 preheader 的循环；没有 preheader 时需要改 CFG，留给后续 pass。
            let [preheader] = outside_preds.as_slice() else {
                continue;
            };
            loops.push(NaturalLoop {
                header: *header,
                preheader: *preheader,
                blocks,
            });
        }
    }

    loops
}

fn collect_loop_blocks(cfg: &ControlFlowGraph, header: BlockId, tail: BlockId) -> HashSet<BlockId> {
    let mut blocks = HashSet::from([header, tail]);
    let mut stack = vec![tail];

    while let Some(block) = stack.pop() {
        for pred in &cfg.preds[block.0] {
            if blocks.insert(*pred) {
                stack.push(*pred);
            }
        }
    }

    blocks
}

fn hoist_loop(func: &mut Function, dom: &Dominators, natural_loop: &NaturalLoop) -> bool {
    let mut changed = false;

    loop {
        let invariant_values = collect_invariant_values(func, dom, natural_loop);
        let candidates = collect_hoist_candidates(func, natural_loop, &invariant_values);
        if candidates.is_empty() {
            break;
        }

        for (block, inst_idx) in candidates {
            let inst = func.blocks[block.0].insts[inst_idx].clone();
            move_inst_to_preheader(func, block, inst_idx, natural_loop.preheader, inst);
            changed = true;
        }
    }

    changed
}

fn collect_invariant_values(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
) -> HashSet<ValueId> {
    let mut invariant = HashSet::new();

    loop {
        let mut changed = false;
        for block in sorted_loop_blocks(natural_loop) {
            for inst in &func.blocks[block.0].insts {
                let Some(result) = inst.result else {
                    continue;
                };
                if invariant.contains(&result) || !is_safe_to_hoist(&inst.kind) {
                    continue;
                }
                if operands_are_invariant(func, dom, natural_loop, &inst.kind, &invariant) {
                    invariant.insert(result);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    invariant
}

fn collect_hoist_candidates(
    func: &Function,
    natural_loop: &NaturalLoop,
    invariant_values: &HashSet<ValueId>,
) -> Vec<(BlockId, usize)> {
    let mut candidates = Vec::new();
    for block in sorted_loop_blocks(natural_loop) {
        // Phi 必须留在循环头；其它 Nop/非纯指令也不会出现在 invariant_values 里。
        for (inst_idx, inst) in func.blocks[block.0].insts.iter().enumerate() {
            if inst
                .result
                .is_some_and(|result| invariant_values.contains(&result))
            {
                candidates.push((block, inst_idx));
            }
        }
    }
    candidates
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

    reindex_block(func, block);
    if block != preheader {
        reindex_block(func, preheader);
    }
}

fn reindex_block(func: &mut Function, block: BlockId) {
    for inst_idx in 0..func.blocks[block.0].insts.len() {
        let Some(result) = func.blocks[block.0].insts[inst_idx].result else {
            continue;
        };
        func.values[result.0].kind = ValueKind::Inst(block, inst_idx);
    }
}

fn operands_are_invariant(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    kind: &InstKind,
    invariant: &HashSet<ValueId>,
) -> bool {
    operands(kind)
        .into_iter()
        .all(|value| value_available_in_preheader(func, dom, natural_loop, value, invariant))
}

fn value_available_in_preheader(
    func: &Function,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    value: ValueId,
    invariant: &HashSet<ValueId>,
) -> bool {
    match &func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(block, _) if natural_loop.blocks.contains(block) => {
            invariant.contains(&value)
        }
        ValueKind::Inst(block, _) => {
            *block == natural_loop.preheader || dom.dominates(*block, natural_loop.preheader)
        }
    }
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
        | InstKind::Call { .. } => false,
    }
}

fn sorted_loop_blocks(natural_loop: &NaturalLoop) -> Vec<BlockId> {
    let mut blocks = natural_loop.blocks.iter().copied().collect::<Vec<_>>();
    blocks.sort_by_key(|block| (*block != natural_loop.header, block.0));
    blocks
}
