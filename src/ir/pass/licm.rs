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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Terminator, Type};

    #[test]
    fn hoists_dependencies_before_uses_when_block_storage_order_differs() {
        let mut func = Function::new("licm_order", Type::Void);
        let header = func.add_block("header");
        let use_block = func.add_block("use");
        let def_block = func.add_block("def");
        let latch = func.add_block("latch");
        let exit = func.add_block("exit");
        let condition = func.add_const(Const::Bool(true));
        let one = func.add_const(Const::Int(1));

        func.set_terminator(func.entry, Terminator::Jump(header));
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: def_block,
                else_target: exit,
            },
        );
        let definition = func
            .append_inst(
                def_block,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: one,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(def_block, Terminator::Jump(use_block));
        let usage = func
            .append_inst(
                use_block,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: definition,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(use_block, Terminator::Jump(latch));
        func.set_terminator(latch, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));
        assert!(func.verify().is_ok());

        licm_function(&mut func);

        assert!(func.verify().is_ok());
        assert_eq!(
            func.values[definition.0].kind,
            ValueKind::Inst(func.entry, 0)
        );
        assert_eq!(func.values[usage.0].kind, ValueKind::Inst(func.entry, 1));
    }

    #[test]
    fn does_not_hoist_from_unreachable_loop_predecessor() {
        let mut func = Function::new("licm_dead_predecessor", Type::Void);
        let header = func.add_block("header");
        let body = func.add_block("body");
        let exit = func.add_block("exit");
        let dead = func.add_block("dead");
        let condition = func.add_const(Const::Bool(true));
        let one = func.add_const(Const::Int(1));

        func.set_terminator(func.entry, Terminator::Jump(header));
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: body,
                else_target: exit,
            },
        );
        func.set_terminator(body, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));
        let dead_value = func
            .append_inst(
                dead,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: one,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(dead, Terminator::Jump(body));
        assert!(func.verify().is_ok());

        licm_function(&mut func);

        assert!(func.verify().is_ok());
        assert_eq!(func.values[dead_value.0].kind, ValueKind::Inst(dead, 0));
        assert!(func.blocks[func.entry.0].insts.is_empty());
    }

    #[test]
    fn hoists_long_reverse_block_id_dependency_chain_in_one_order() {
        const CHAIN_LEN: usize = 256;

        let mut func = Function::new("licm_reverse_chain", Type::Void);
        let header = func.add_block("header");
        // Allocate the chain in use-to-definition order. CFG dominance and
        // value dependencies therefore run opposite to BlockId order.
        let reverse_blocks = (0..CHAIN_LEN)
            .map(|index| func.add_block(format!("chain.{index}")))
            .collect::<Vec<_>>();
        let latch = func.add_block("latch");
        let exit = func.add_block("exit");
        let condition = func.add_const(Const::Bool(true));
        let one = func.add_const(Const::Int(1));

        func.set_terminator(func.entry, Terminator::Jump(header));
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: reverse_blocks[CHAIN_LEN - 1],
                else_target: exit,
            },
        );
        let mut dependency = one;
        let mut values = Vec::new();
        for block in reverse_blocks.iter().rev() {
            dependency = func
                .append_inst(
                    *block,
                    InstKind::Binary {
                        op: BinaryOp::Iadd,
                        lhs: dependency,
                        rhs: one,
                    },
                    Some(Type::I32),
                )
                .unwrap();
            values.push(dependency);
        }
        for index in (1..CHAIN_LEN).rev() {
            func.set_terminator(
                reverse_blocks[index],
                Terminator::Jump(reverse_blocks[index - 1]),
            );
        }
        func.set_terminator(reverse_blocks[0], Terminator::Jump(latch));
        func.set_terminator(latch, Terminator::Jump(header));
        func.set_terminator(exit, Terminator::Return(None));
        assert!(func.verify().is_ok());

        licm_function(&mut func);

        assert!(func.verify().is_ok());
        assert_eq!(func.blocks[func.entry.0].insts.len(), CHAIN_LEN);
        for (inst_idx, value) in values.into_iter().enumerate() {
            assert_eq!(
                func.values[value.0].kind,
                ValueKind::Inst(func.entry, inst_idx)
            );
        }
    }
}
