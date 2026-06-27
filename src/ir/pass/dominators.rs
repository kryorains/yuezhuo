use crate::ir::{BlockId, Function, Terminator};
use std::collections::{HashSet, VecDeque};

pub(super) struct ControlFlowGraph {
    pub(super) preds: Vec<Vec<BlockId>>,
    pub(super) succs: Vec<Vec<BlockId>>,
}

impl ControlFlowGraph {
    pub(super) fn new(func: &Function) -> Self {
        let mut preds = vec![Vec::new(); func.blocks.len()];
        let mut succs = vec![Vec::new(); func.blocks.len()];

        for block_idx in 0..func.blocks.len() {
            let pred = BlockId(block_idx);
            let Some(terminator) = &func.blocks[block_idx].terminator else {
                continue;
            };
            match terminator {
                Terminator::Return(_) => {}
                Terminator::Jump(target) => add_edge(pred, *target, &mut preds, &mut succs),
                Terminator::Branch {
                    then_target,
                    else_target,
                    ..
                } => {
                    add_edge(pred, *then_target, &mut preds, &mut succs);
                    if then_target != else_target {
                        add_edge(pred, *else_target, &mut preds, &mut succs);
                    }
                }
            }
        }

        Self { preds, succs }
    }
}

fn add_edge(pred: BlockId, succ: BlockId, preds: &mut [Vec<BlockId>], succs: &mut [Vec<BlockId>]) {
    succs[pred.0].push(succ);
    preds[succ.0].push(pred);
}

pub(super) struct Dominators {
    pub(super) dom_sets: Vec<HashSet<usize>>,
    pub(super) children: Vec<Vec<BlockId>>,
    pub(super) frontier: Vec<Vec<BlockId>>,
}

impl Dominators {
    pub(super) fn new(func: &Function, cfg: &ControlFlowGraph) -> Self {
        let reachable = reachable_blocks(func, cfg);
        let dom_sets = compute_dom_sets(func, cfg, &reachable);
        let idom = compute_idoms(func, &dom_sets, &reachable);
        let children = compute_children(func, &idom);
        let frontier = compute_frontier(func, cfg, &idom, &reachable);
        Self {
            dom_sets,
            children,
            frontier,
        }
    }

    pub(super) fn dominates(&self, dominator: BlockId, block: BlockId) -> bool {
        self.dom_sets[block.0].contains(&dominator.0)
    }
}

fn reachable_blocks(func: &Function, cfg: &ControlFlowGraph) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    let mut worklist = VecDeque::from([func.entry]);
    while let Some(block) = worklist.pop_front() {
        if !reachable.insert(block.0) {
            continue;
        }
        for succ in &cfg.succs[block.0] {
            worklist.push_back(*succ);
        }
    }
    reachable
}

fn compute_dom_sets(
    func: &Function,
    cfg: &ControlFlowGraph,
    reachable: &HashSet<usize>,
) -> Vec<HashSet<usize>> {
    let n = func.blocks.len();
    let all = reachable.iter().copied().collect::<HashSet<_>>();
    let mut doms = (0..n)
        .map(|block| {
            if reachable.contains(&block) {
                all.clone()
            } else {
                HashSet::from([block])
            }
        })
        .collect::<Vec<_>>();
    doms[func.entry.0] = HashSet::from([func.entry.0]);

    loop {
        let mut changed = false;
        for block_idx in 0..n {
            if block_idx == func.entry.0 || !reachable.contains(&block_idx) {
                continue;
            }

            let reachable_preds = cfg.preds[block_idx]
                .iter()
                .filter(|pred| reachable.contains(&pred.0))
                .collect::<Vec<_>>();
            let mut next = if reachable_preds.is_empty() {
                HashSet::new()
            } else {
                reachable_preds
                    .iter()
                    .map(|pred| doms[pred.0].clone())
                    .reduce(|acc, pred_doms| acc.intersection(&pred_doms).copied().collect())
                    .unwrap_or_default()
            };
            next.insert(block_idx);

            if next != doms[block_idx] {
                doms[block_idx] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    doms
}

fn compute_idoms(
    func: &Function,
    dom_sets: &[HashSet<usize>],
    reachable: &HashSet<usize>,
) -> Vec<Option<BlockId>> {
    let mut idom = vec![None; func.blocks.len()];

    for block_idx in 0..func.blocks.len() {
        if block_idx == func.entry.0 || !reachable.contains(&block_idx) {
            continue;
        }

        let strict_doms = dom_sets[block_idx]
            .iter()
            .copied()
            .filter(|dom| *dom != block_idx)
            .collect::<Vec<_>>();

        let immediate = strict_doms
            .iter()
            .copied()
            .max_by_key(|candidate| dom_sets[*candidate].len());
        idom[block_idx] = immediate.map(BlockId);
    }

    idom
}

fn compute_children(func: &Function, idom: &[Option<BlockId>]) -> Vec<Vec<BlockId>> {
    let mut children = vec![Vec::new(); func.blocks.len()];
    for (block_idx, parent) in idom.iter().enumerate() {
        if let Some(parent) = parent {
            children[parent.0].push(BlockId(block_idx));
        }
    }
    children
}

fn compute_frontier(
    func: &Function,
    cfg: &ControlFlowGraph,
    idom: &[Option<BlockId>],
    reachable: &HashSet<usize>,
) -> Vec<Vec<BlockId>> {
    let mut frontier = vec![Vec::new(); func.blocks.len()];

    for block_idx in 0..func.blocks.len() {
        if !reachable.contains(&block_idx) {
            continue;
        }
        let reachable_preds = cfg.preds[block_idx]
            .iter()
            .filter(|pred| reachable.contains(&pred.0))
            .copied()
            .collect::<Vec<_>>();
        if reachable_preds.len() < 2 {
            continue;
        }

        let Some(stop) = idom[block_idx] else {
            continue;
        };
        for pred in &reachable_preds {
            let mut runner = *pred;
            while runner != stop {
                push_unique(&mut frontier[runner.0], BlockId(block_idx));
                let Some(next) = idom[runner.0] else {
                    break;
                };
                runner = next;
            }
        }
    }

    frontier
}

fn push_unique(values: &mut Vec<BlockId>, value: BlockId) {
    if !values.contains(&value) {
        values.push(value);
    }
}
