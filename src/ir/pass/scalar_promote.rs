use super::dominators::{ControlFlowGraph, Dominators};
use super::util::{resolve_replacement, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BlockId, Const, Function, Inst, InstKind, Module, Terminator, Type, Value, ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet, VecDeque};

pub(super) struct ScalarPromotePass;

pub(super) const MAX_PROMOTION_BLOCKS: usize = 512;
pub(super) const MAX_PROMOTION_VALUES: usize = 4096;

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
    // 当前实现使用较朴素的数据结构；大函数先跳过，避免编译时间爆炸。
    if func.blocks.len() > MAX_PROMOTION_BLOCKS || func.values.len() > MAX_PROMOTION_VALUES {
        return;
    }

    // mem2reg 的核心流程：找候选 alloca -> 插 phi -> 沿支配树重命名。
    let cfg = ControlFlowGraph::new(func);
    let reachable = reachable_blocks(func, &cfg);
    let dom = Dominators::new(func, &cfg);
    let candidates = collect_candidates(func, &reachable, &dom);
    if candidates.is_empty() {
        return;
    }

    let phi_values = insert_phis(func, &candidates, &dom);
    let phi_allocas = phi_values
        .iter()
        .map(|((block, alloca), phi)| (*phi, (*block, *alloca)))
        .collect::<HashMap<_, _>>();
    // The unpruned phi placement can create a dead phi before the first real
    // store. Seed every stack so those phis still receive a type-correct value
    // on every predecessor edge. `loads_are_defined` guarantees the seed can
    // never replace an observable load.
    let mut stacks = HashMap::new();
    for alloca in &candidates {
        let Some(ty) = promoted_type(func, *alloca) else {
            continue;
        };
        let seed = func.add_const(Const::Zero(ty));
        stacks.insert(*alloca, vec![seed]);
    }
    let mut replacements = ValueReplacements::new();

    rename_block(
        func,
        func.entry,
        &cfg,
        &dom,
        &candidates,
        &phi_values,
        &phi_allocas,
        &mut stacks,
        &mut replacements,
    );
    fill_unreachable_phi_incomings(func, &cfg, &dom, &phi_values, &stacks);

    rewrite_function_uses(func, &replacements);
    if let Err(errors) = func.verify() {
        panic!(
            "scalar promotion produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn reachable_blocks(func: &Function, cfg: &ControlFlowGraph) -> HashSet<BlockId> {
    let mut reachable = HashSet::new();
    let mut worklist = VecDeque::from([func.entry]);
    while let Some(block) = worklist.pop_front() {
        if !reachable.insert(block) {
            continue;
        }
        for succ in &cfg.succs[block.0] {
            worklist.push_back(*succ);
        }
    }
    reachable
}

fn collect_candidates(
    func: &Function,
    reachable: &HashSet<BlockId>,
    dom: &Dominators,
) -> Vec<ValueId> {
    // 候选必须是标量 alloca，且地址不能逃逸，只能被 load/store 以受控方式访问。
    let mut candidates = Vec::new();
    let mut ordered_allocas = collect_scalar_allocas(func);
    ordered_allocas.sort_by_key(|value| value.0);

    for alloca in ordered_allocas {
        let mut info = CandidateInfo::new();
        collect_candidate_info(func, alloca, reachable, &mut info);
        if info.rejected || info.loads.is_empty() || info.stores.is_empty() {
            continue;
        }
        if !loads_are_defined(func, alloca, &info, dom) {
            continue;
        }
        candidates.push(alloca);
    }

    candidates
}

fn collect_scalar_allocas(func: &Function) -> Vec<ValueId> {
    let mut allocas = Vec::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let Some(result) = inst.result else {
                continue;
            };
            let InstKind::Alloca { ty } = &inst.kind else {
                continue;
            };
            if !matches!(ty, Type::Array { .. }) {
                allocas.push(result);
            }
        }
    }
    allocas
}

struct CandidateInfo {
    loads: Vec<(BlockId, usize)>,
    stores: Vec<(BlockId, usize)>,
    store_blocks: HashSet<BlockId>,
    rejected: bool,
}

impl CandidateInfo {
    fn new() -> Self {
        Self {
            loads: Vec::new(),
            stores: Vec::new(),
            store_blocks: HashSet::new(),
            rejected: false,
        }
    }
}

fn collect_candidate_info(
    func: &Function,
    alloca: ValueId,
    reachable: &HashSet<BlockId>,
    info: &mut CandidateInfo,
) {
    for block_idx in 0..func.blocks.len() {
        let block = BlockId(block_idx);
        for inst_idx in 0..func.blocks[block_idx].insts.len() {
            let inst = &func.blocks[block_idx].insts[inst_idx];
            if !reachable.contains(&block) {
                if inst_uses_value_as_escape(&inst.kind, alloca) {
                    info.rejected = true;
                }
                continue;
            }
            match &inst.kind {
                InstKind::Load { ptr } if *ptr == alloca => info.loads.push((block, inst_idx)),
                InstKind::Store { ptr, value } if *ptr == alloca => {
                    if *value == alloca {
                        info.rejected = true;
                    }
                    info.stores.push((block, inst_idx));
                    info.store_blocks.insert(block);
                }
                _ => {
                    if inst_uses_value_as_escape(&inst.kind, alloca) {
                        info.rejected = true;
                    }
                }
            }
        }
        if let Some(terminator) = &func.blocks[block_idx].terminator {
            if terminator_uses_value(terminator, alloca) {
                info.rejected = true;
            }
        }
    }
}

fn inst_uses_value_as_escape(kind: &InstKind, value: ValueId) -> bool {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => false,
        InstKind::Load { ptr } => *ptr == value,
        InstKind::Store { ptr, value: stored } => *ptr == value || *stored == value,
        InstKind::MemZero { ptr, .. } => *ptr == value,
        InstKind::Phi { incomings } => incomings.iter().any(|(_, incoming)| *incoming == value),
        InstKind::Unary { value: operand, .. } | InstKind::Cast { value: operand, .. } => {
            *operand == value
        }
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => *lhs == value || *rhs == value,
        InstKind::Gep { base, indices } => *base == value || indices.contains(&value),
        InstKind::Call { args, .. } => args.contains(&value),
    }
}

fn terminator_uses_value(terminator: &Terminator, value: ValueId) -> bool {
    match terminator {
        Terminator::Return(Some(ret)) | Terminator::Branch { cond: ret, .. } => *ret == value,
        Terminator::Return(None) | Terminator::Jump(_) => false,
    }
}

fn loads_are_defined(
    func: &Function,
    alloca: ValueId,
    info: &CandidateInfo,
    dom: &Dominators,
) -> bool {
    // 保证每个 load 在所有可见路径上至少能看到一个先前 store，避免读未定义值。
    for (load_block, load_idx) in &info.loads {
        let mut defined = false;
        for (store_block, store_idx) in &info.stores {
            if store_block == load_block {
                if store_idx < load_idx {
                    defined = true;
                    break;
                }
            } else if dom.dominates(*store_block, *load_block) {
                defined = true;
                break;
            }
        }
        if !defined {
            return false;
        }
    }

    promoted_type(func, alloca).is_some()
}

fn insert_phis(
    func: &mut Function,
    candidates: &[ValueId],
    dom: &Dominators,
) -> HashMap<(BlockId, ValueId), ValueId> {
    // 从所有 store 所在块出发，在 dominance frontier 上放置必要的 phi。
    let mut phi_values = HashMap::new();

    for alloca in candidates {
        let Some(ty) = promoted_type(func, *alloca) else {
            continue;
        };
        let mut store_blocks = store_blocks(func, *alloca).into_iter().collect::<Vec<_>>();
        store_blocks.sort_by_key(|block| block.0);
        let mut worklist = store_blocks.into_iter().collect::<VecDeque<_>>();
        let mut placed = HashSet::<BlockId>::new();

        while let Some(block) = worklist.pop_front() {
            for frontier in &dom.frontier[block.0] {
                if !placed.insert(*frontier) {
                    continue;
                }
                let phi = insert_phi(func, *frontier, ty.clone());
                phi_values.insert((*frontier, *alloca), phi);
                worklist.push_back(*frontier);
            }
        }
    }

    phi_values
}

fn store_blocks(func: &Function, alloca: ValueId) -> HashSet<BlockId> {
    let mut blocks = HashSet::new();
    for block_idx in 0..func.blocks.len() {
        for inst in &func.blocks[block_idx].insts {
            if matches!(inst.kind, InstKind::Store { ptr, .. } if ptr == alloca) {
                blocks.insert(BlockId(block_idx));
            }
        }
    }
    blocks
}

fn insert_phi(func: &mut Function, block: BlockId, ty: Type) -> ValueId {
    // phi 必须插在块开头，跳过已有 Nop/phi，保持块内 phi 区域连续。
    let pos = func.blocks[block.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Nop | InstKind::Phi { .. }))
        .count();
    let result = ValueId(func.values.len());
    func.values.push(Value {
        name: None,
        ty,
        kind: ValueKind::Inst(block, pos),
    });
    func.blocks[block.0].insts.insert(
        pos,
        Inst {
            result: Some(result),
            kind: InstKind::Phi {
                incomings: Vec::new(),
            },
        },
    );
    reindex_block(func, block);
    result
}

fn reindex_block(func: &mut Function, block: BlockId) {
    for inst_idx in 0..func.blocks[block.0].insts.len() {
        let Some(result) = func.blocks[block.0].insts[inst_idx].result else {
            continue;
        };
        func.values[result.0].kind = ValueKind::Inst(block, inst_idx);
    }
}

fn rename_block(
    func: &mut Function,
    block: BlockId,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    candidates: &[ValueId],
    phi_values: &HashMap<(BlockId, ValueId), ValueId>,
    phi_allocas: &HashMap<ValueId, (BlockId, ValueId)>,
    stacks: &mut HashMap<ValueId, Vec<ValueId>>,
    replacements: &mut ValueReplacements,
) {
    // 沿支配树 DFS。每个 alloca 有一条值栈，表示当前路径上最新的 SSA 定义。
    let mut pushed = Vec::<ValueId>::new();

    for inst in &func.blocks[block.0].insts {
        let Some(result) = inst.result else {
            continue;
        };
        if let Some((_, alloca)) = phi_allocas.get(&result) {
            stacks.get_mut(alloca).unwrap().push(result);
            pushed.push(*alloca);
        }
    }

    for inst_idx in 0..func.blocks[block.0].insts.len() {
        // alloca/load/store 被提升后都变成 Nop；真正的数据流由值栈和替换表表达。
        let inst = func.blocks[block.0].insts[inst_idx].clone();
        match inst.kind {
            InstKind::Alloca { .. } if inst.result.is_some_and(|r| candidates.contains(&r)) => {
                func.blocks[block.0].insts[inst_idx].result = None;
                func.blocks[block.0].insts[inst_idx].kind = InstKind::Nop;
            }
            InstKind::Store { ptr, value } if candidates.contains(&ptr) => {
                let value = resolve_replacement(value, replacements);
                stacks.get_mut(&ptr).unwrap().push(value);
                pushed.push(ptr);
                func.blocks[block.0].insts[inst_idx].kind = InstKind::Nop;
            }
            InstKind::Load { ptr } if candidates.contains(&ptr) => {
                let Some(result) = inst.result else {
                    continue;
                };
                let stack = stacks.get(&ptr).unwrap();
                let Some(value) = stack.last().copied() else {
                    continue;
                };
                replacements.insert(result, value);
                func.blocks[block.0].insts[inst_idx].result = None;
                func.blocks[block.0].insts[inst_idx].kind = InstKind::Nop;
            }
            _ => {}
        }
    }

    for succ in &cfg.succs[block.0] {
        // 当前块处理完后，把每个候选变量的当前值填到后继块对应 phi 的 incoming 里。
        for alloca in candidates {
            let Some(phi) = phi_values.get(&(*succ, *alloca)).copied() else {
                continue;
            };
            let Some(value) = stacks.get(alloca).and_then(|stack| stack.last()).copied() else {
                continue;
            };
            add_phi_incoming(func, phi, block, resolve_replacement(value, replacements));
        }
    }

    for child in &dom.children[block.0] {
        rename_block(
            func,
            *child,
            cfg,
            dom,
            candidates,
            phi_values,
            phi_allocas,
            stacks,
            replacements,
        );
    }

    for alloca in pushed.into_iter().rev() {
        // 离开当前支配树节点时回滚值栈，恢复父路径的“当前值”。
        stacks.get_mut(&alloca).unwrap().pop();
    }
}

fn fill_unreachable_phi_incomings(
    func: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    phi_values: &HashMap<(BlockId, ValueId), ValueId>,
    stacks: &HashMap<ValueId, Vec<ValueId>>,
) {
    // rename_block only visits the entry-reachable dominator tree. Dead CFG
    // predecessors still need a structurally complete phi, but their value can
    // never be observed, so use the type-correct seed for those edges.
    for ((block, alloca), phi) in phi_values {
        let Some(seed) = stacks.get(alloca).and_then(|stack| stack.first()).copied() else {
            continue;
        };
        let ValueKind::Inst(owner, inst_idx) = func.value(*phi).kind else {
            continue;
        };
        let InstKind::Phi { incomings } = &func.blocks[owner.0].insts[inst_idx].kind else {
            continue;
        };
        let present = incomings
            .iter()
            .map(|(pred, _)| *pred)
            .collect::<HashSet<_>>();
        let missing = cfg.preds[block.0]
            .iter()
            .copied()
            .filter(|pred| !dom.is_reachable(*pred) && !present.contains(pred))
            .collect::<Vec<_>>();
        for pred in missing {
            add_phi_incoming(func, *phi, pred, seed);
        }
    }
}

fn add_phi_incoming(func: &mut Function, phi: ValueId, pred: BlockId, value: ValueId) {
    let ValueKind::Inst(block, inst_idx) = func.value(phi).kind else {
        panic!("phi value must be an instruction");
    };
    let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
        panic!("phi value must point to a phi instruction");
    };
    incomings.push((pred, value));
}

fn promoted_type(func: &Function, alloca: ValueId) -> Option<Type> {
    let ValueKind::Inst(block, inst_idx) = func.value(alloca).kind else {
        return None;
    };
    let InstKind::Alloca { ty } = &func.block(block).insts[inst_idx].kind else {
        return None;
    };
    Some(ty.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completes_promoted_phis_for_unreachable_predecessors() {
        let mut func = Function::new("dead_phi_pred", Type::I32);
        let left = func.add_block("left");
        let right = func.add_block("right");
        let merge = func.add_block("merge");
        let dead = func.add_block("dead");
        let condition = func.add_const(Const::Bool(true));
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let two = func.add_const(Const::Int(2));
        let slot = func
            .append_inst(
                func.entry,
                InstKind::Alloca { ty: Type::I32 },
                Some(Type::Ptr(Box::new(Type::I32))),
            )
            .unwrap();
        func.append_inst(
            func.entry,
            InstKind::Store {
                ptr: slot,
                value: zero,
            },
            None,
        );
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond: condition,
                then_target: left,
                else_target: right,
            },
        );
        func.append_inst(
            left,
            InstKind::Store {
                ptr: slot,
                value: one,
            },
            None,
        );
        func.set_terminator(left, Terminator::Jump(merge));
        func.append_inst(
            right,
            InstKind::Store {
                ptr: slot,
                value: two,
            },
            None,
        );
        func.set_terminator(right, Terminator::Jump(merge));
        func.set_terminator(dead, Terminator::Jump(merge));
        let loaded = func
            .append_inst(merge, InstKind::Load { ptr: slot }, Some(Type::I32))
            .unwrap();
        func.set_terminator(merge, Terminator::Return(Some(loaded)));
        assert!(func.verify().is_ok());

        promote_function(&mut func);

        assert!(func.verify().is_ok());
        let phi_incomings = func.blocks[merge.0]
            .insts
            .iter()
            .find_map(|inst| match &inst.kind {
                InstKind::Phi { incomings } => Some(incomings),
                _ => None,
            })
            .expect("promotion should insert a merge phi");
        assert!(phi_incomings.iter().any(|(pred, _)| *pred == dead));
    }
}
