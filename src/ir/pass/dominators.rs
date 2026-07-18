use crate::ir::{BlockId, Function, Terminator};

pub(crate) struct ControlFlowGraph {
    pub(crate) preds: Vec<Vec<BlockId>>,
    pub(crate) succs: Vec<Vec<BlockId>>,
}

impl ControlFlowGraph {
    pub(crate) fn new(func: &Function) -> Self {
        // 根据每个基本块的 terminator 建出前驱/后继表。
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

pub(crate) struct Dominators {
    pub(crate) children: Vec<Vec<BlockId>>,
    pub(crate) frontier: Vec<Vec<BlockId>>,
    entry: BlockId,
    reachable: Vec<bool>,
    tree_preorder: Vec<usize>,
    tree_subtree_end: Vec<usize>,
    unreachable_preorder: Vec<usize>,
    unreachable_subtree_end: Vec<usize>,
}

impl Dominators {
    pub(crate) fn new(func: &Function, cfg: &ControlFlowGraph) -> Self {
        // Cooper-Harvey-Kennedy：只在入口可达 CFG 的 RPO 上求 immediate dominator。
        let (rpo, reachable) = compute_reverse_postorder(func, cfg);
        let idom = compute_idoms(func, cfg, &rpo);
        let children = compute_children(func, &idom);
        let (tree_preorder, tree_subtree_end) =
            compute_tree_intervals(func.entry, &children, func.blocks.len());
        let frontier = compute_frontier(func, cfg, &idom, &reachable);
        let (unreachable_preorder, unreachable_subtree_end) =
            compute_unreachable_intervals(cfg, &reachable);
        Self {
            children,
            frontier,
            entry: func.entry,
            reachable,
            tree_preorder,
            tree_subtree_end,
            unreachable_preorder,
            unreachable_subtree_end,
        }
    }

    pub(crate) fn dominates(&self, dominator: BlockId, block: BlockId) -> bool {
        // 不可达块保持原有保守语义：只认为块支配自身，不在不同不可达根之间
        // 发明支配关系。可达块则通过支配树 DFS 的半开区间 O(1) 查询。
        if !self.is_reachable(block) {
            return dominator == block;
        }
        if !self.is_reachable(dominator) {
            return false;
        }
        let dominator_preorder = self.tree_preorder[dominator.0];
        let block_preorder = self.tree_preorder[block.0];
        dominator_preorder <= block_preorder && block_preorder < self.tree_subtree_end[dominator.0]
    }

    /// Returns whether a definition is available on every structural path to
    /// `block`. For dead CFG regions this uses a synthetic-root dominance
    /// forest instead of either accepting or rejecting every cross-block use.
    pub(crate) fn dominates_for_availability(&self, dominator: BlockId, block: BlockId) -> bool {
        if self.is_reachable(block) {
            return self.dominates(dominator, block);
        }
        if self.is_reachable(dominator) {
            // Entry instructions execute before every possible function path.
            // Treat the synthetic dead-region root as following the physical
            // entry so cleanup passes may leave entry-defined values in code
            // that later became disconnected. No other reachable definition
            // is conservatively available in an unrelated dead component.
            return dominator == self.entry;
        }
        let dominator_preorder = self.unreachable_preorder[dominator.0];
        let block_preorder = self.unreachable_preorder[block.0];
        dominator_preorder != usize::MAX
            && dominator_preorder <= block_preorder
            && block_preorder < self.unreachable_subtree_end[dominator.0]
    }

    pub(crate) fn is_reachable(&self, block: BlockId) -> bool {
        self.reachable.get(block.0).copied().unwrap_or(false)
    }
}

fn compute_reverse_postorder(func: &Function, cfg: &ControlFlowGraph) -> (Vec<BlockId>, Vec<bool>) {
    let mut reachable = vec![false; func.blocks.len()];
    let mut postorder = Vec::new();
    let mut stack = vec![(func.entry, 0usize)];
    reachable[func.entry.0] = true;

    // 用显式栈避免大型直线 CFG 使宿主 Rust 栈溢出。
    while let Some((block, next_succ)) = stack.pop() {
        if next_succ < cfg.succs[block.0].len() {
            stack.push((block, next_succ + 1));
            let succ = cfg.succs[block.0][next_succ];
            if !reachable[succ.0] {
                reachable[succ.0] = true;
                stack.push((succ, 0));
            }
        } else {
            postorder.push(block);
        }
    }

    postorder.reverse();
    (postorder, reachable)
}

fn compute_idoms(func: &Function, cfg: &ControlFlowGraph, rpo: &[BlockId]) -> Vec<Option<BlockId>> {
    let mut rpo_index = vec![usize::MAX; func.blocks.len()];
    for (index, block) in rpo.iter().enumerate() {
        rpo_index[block.0] = index;
    }

    // 在求解期间让 entry 的 idom 指向自身，作为 intersect 的根哨兵。
    let mut idom = vec![None; func.blocks.len()];
    idom[func.entry.0] = Some(func.entry);

    loop {
        let mut changed = false;
        for block in rpo.iter().copied().skip(1) {
            let mut processed_preds = cfg.preds[block.0]
                .iter()
                .copied()
                .filter(|pred| idom[pred.0].is_some());
            let Some(mut new_idom) = processed_preds.next() else {
                continue;
            };
            for pred in processed_preds {
                new_idom = intersect(pred, new_idom, &idom, &rpo_index);
            }
            if idom[block.0] != Some(new_idom) {
                idom[block.0] = Some(new_idom);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    idom[func.entry.0] = None;
    idom
}

fn intersect(
    mut first: BlockId,
    mut second: BlockId,
    idom: &[Option<BlockId>],
    rpo_index: &[usize],
) -> BlockId {
    while first != second {
        while rpo_index[first.0] > rpo_index[second.0] {
            first = idom[first.0].expect("processed RPO predecessor must have an idom");
        }
        while rpo_index[second.0] > rpo_index[first.0] {
            second = idom[second.0].expect("processed RPO predecessor must have an idom");
        }
    }
    first
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

fn compute_tree_intervals(
    entry: BlockId,
    children: &[Vec<BlockId>],
    block_count: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut preorder = vec![usize::MAX; block_count];
    let mut subtree_end = vec![usize::MAX; block_count];
    let mut next_preorder = 0usize;
    let mut stack = vec![(entry, 0usize)];
    preorder[entry.0] = next_preorder;
    next_preorder += 1;

    while let Some((block, next_child)) = stack.pop() {
        if next_child < children[block.0].len() {
            stack.push((block, next_child + 1));
            let child = children[block.0][next_child];
            preorder[child.0] = next_preorder;
            next_preorder += 1;
            stack.push((child, 0));
        } else {
            subtree_end[block.0] = next_preorder;
        }
    }

    (preorder, subtree_end)
}

fn compute_unreachable_intervals(
    cfg: &ControlFlowGraph,
    reachable: &[bool],
) -> (Vec<usize>, Vec<usize>) {
    let block_count = cfg.succs.len();
    let virtual_root = block_count;
    let mut succs = vec![Vec::<usize>::new(); block_count + 1];
    let mut preds = vec![Vec::<usize>::new(); block_count + 1];

    for block in 0..block_count {
        if reachable[block] {
            continue;
        }
        for succ in &cfg.succs[block] {
            if !reachable[succ.0] {
                succs[block].push(succ.0);
                preds[succ.0].push(block);
            }
        }
    }

    // A dead region may have several ordinary roots or consist only of
    // cycles. Connect a virtual root to every member of each source SCC. The
    // latter avoids inventing an arbitrary first block inside a rootless dead
    // cycle, while still proving dominance in well-formed dead.def->dead.use
    // chains and downstream regions.
    let roots =
        unreachable_source_scc_members(&succs[..block_count], &preds[..block_count], reachable);
    for root in roots {
        succs[virtual_root].push(root);
        preds[root].push(virtual_root);
    }

    let rpo = compute_index_reverse_postorder(virtual_root, &succs);
    let idom = compute_index_idoms(virtual_root, &preds, &rpo);
    let mut children = vec![Vec::new(); block_count + 1];
    for (node, parent) in idom.iter().enumerate() {
        if let Some(parent) = parent {
            children[*parent].push(node);
        }
    }

    let mut preorder = vec![usize::MAX; block_count + 1];
    let mut subtree_end = vec![usize::MAX; block_count + 1];
    let mut next_preorder = 0usize;
    let mut stack = vec![(virtual_root, 0usize)];
    preorder[virtual_root] = next_preorder;
    next_preorder += 1;
    while let Some((node, next_child)) = stack.pop() {
        if next_child < children[node].len() {
            stack.push((node, next_child + 1));
            let child = children[node][next_child];
            preorder[child] = next_preorder;
            next_preorder += 1;
            stack.push((child, 0));
        } else {
            subtree_end[node] = next_preorder;
        }
    }

    preorder.truncate(block_count);
    subtree_end.truncate(block_count);
    (preorder, subtree_end)
}

fn unreachable_source_scc_members(
    succs: &[Vec<usize>],
    preds: &[Vec<usize>],
    reachable: &[bool],
) -> Vec<usize> {
    let mut visited = reachable.to_vec();
    let mut postorder = Vec::new();
    for start in 0..succs.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next_succ)) = stack.pop() {
            if next_succ < succs[node].len() {
                stack.push((node, next_succ + 1));
                let succ = succs[node][next_succ];
                if !visited[succ] {
                    visited[succ] = true;
                    stack.push((succ, 0));
                }
            } else {
                postorder.push(node);
            }
        }
    }

    let mut component = vec![usize::MAX; succs.len()];
    let mut component_count = 0usize;
    for start in postorder.into_iter().rev() {
        if component[start] != usize::MAX {
            continue;
        }
        component[start] = component_count;
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            for pred in &preds[node] {
                if component[*pred] == usize::MAX {
                    component[*pred] = component_count;
                    stack.push(*pred);
                }
            }
        }
        component_count += 1;
    }

    let mut is_source = vec![true; component_count];
    for from in 0..succs.len() {
        if reachable[from] {
            continue;
        }
        for to in &succs[from] {
            if component[from] != component[*to] {
                is_source[component[*to]] = false;
            }
        }
    }

    (0..succs.len())
        .filter(|node| !reachable[*node] && is_source[component[*node]])
        .collect()
}

fn compute_index_reverse_postorder(root: usize, succs: &[Vec<usize>]) -> Vec<usize> {
    let mut visited = vec![false; succs.len()];
    let mut postorder = Vec::new();
    let mut stack = vec![(root, 0usize)];
    visited[root] = true;
    while let Some((node, next_succ)) = stack.pop() {
        if next_succ < succs[node].len() {
            stack.push((node, next_succ + 1));
            let succ = succs[node][next_succ];
            if !visited[succ] {
                visited[succ] = true;
                stack.push((succ, 0));
            }
        } else {
            postorder.push(node);
        }
    }
    postorder.reverse();
    postorder
}

fn compute_index_idoms(root: usize, preds: &[Vec<usize>], rpo: &[usize]) -> Vec<Option<usize>> {
    let mut rpo_index = vec![usize::MAX; preds.len()];
    for (index, node) in rpo.iter().enumerate() {
        rpo_index[*node] = index;
    }
    let mut idom = vec![None; preds.len()];
    idom[root] = Some(root);

    loop {
        let mut changed = false;
        for node in rpo.iter().copied().skip(1) {
            let mut processed_preds = preds[node]
                .iter()
                .copied()
                .filter(|pred| idom[*pred].is_some());
            let Some(mut new_idom) = processed_preds.next() else {
                continue;
            };
            for pred in processed_preds {
                new_idom = intersect_index(pred, new_idom, &idom, &rpo_index);
            }
            if idom[node] != Some(new_idom) {
                idom[node] = Some(new_idom);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    idom[root] = None;
    idom
}

fn intersect_index(
    mut first: usize,
    mut second: usize,
    idom: &[Option<usize>],
    rpo_index: &[usize],
) -> usize {
    while first != second {
        while rpo_index[first] > rpo_index[second] {
            first = idom[first].expect("processed RPO predecessor must have an idom");
        }
        while rpo_index[second] > rpo_index[first] {
            second = idom[second].expect("processed RPO predecessor must have an idom");
        }
    }
    first
}

fn compute_frontier(
    func: &Function,
    cfg: &ControlFlowGraph,
    idom: &[Option<BlockId>],
    reachable: &[bool],
) -> Vec<Vec<BlockId>> {
    // dominance frontier 用来回答“某个定义从哪些汇合点开始不再支配所有路径”。
    let mut frontier = vec![Vec::new(); func.blocks.len()];
    let mut last_inserted = vec![usize::MAX; func.blocks.len()];

    for block_idx in 0..func.blocks.len() {
        if !reachable[block_idx] {
            continue;
        }
        let reachable_preds = cfg.preds[block_idx]
            .iter()
            .filter(|pred| reachable[pred.0])
            .copied()
            .collect::<Vec<_>>();
        if reachable_preds.len() < 2 {
            continue;
        }

        let Some(stop) = idom[block_idx] else {
            continue;
        };
        for pred in &reachable_preds {
            // 从每个前驱沿 idom 链向上走到当前块的 idom，沿途都需要把当前块放进 frontier。
            let mut runner = *pred;
            while runner != stop {
                if last_inserted[runner.0] != block_idx {
                    frontier[runner.0].push(BlockId(block_idx));
                    last_inserted[runner.0] = block_idx;
                }
                let Some(next) = idom[runner.0] else {
                    break;
                };
                runner = next;
            }
        }
    }

    frontier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Type};

    #[test]
    fn computes_tree_intervals_and_frontiers_in_cfg_order() {
        let mut func = Function::new("dominators", Type::Void);
        // Deliberately allocate the merge before its predecessors so block
        // storage order differs from CFG reverse postorder.
        let merge = func.add_block("merge");
        let left = func.add_block("left");
        let right = func.add_block("right");
        let latch = func.add_block("latch");
        let exit = func.add_block("exit");
        let dead_def = func.add_block("dead.def");
        let dead_use = func.add_block("dead.use");
        let condition = func.add_const(Const::Bool(true));

        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond: condition,
                then_target: left,
                else_target: right,
            },
        );
        func.set_terminator(left, Terminator::Jump(merge));
        func.set_terminator(right, Terminator::Jump(merge));
        func.set_terminator(
            merge,
            Terminator::Branch {
                cond: condition,
                then_target: latch,
                else_target: exit,
            },
        );
        func.set_terminator(latch, Terminator::Jump(merge));
        func.set_terminator(exit, Terminator::Return(None));
        func.set_terminator(dead_def, Terminator::Jump(dead_use));
        func.set_terminator(dead_use, Terminator::Return(None));

        let cfg = ControlFlowGraph::new(&func);
        let dom = Dominators::new(&func, &cfg);

        assert!(dom.dominates(func.entry, merge));
        assert!(dom.dominates(merge, latch));
        assert!(dom.dominates(merge, exit));
        assert!(!dom.dominates(left, merge));
        assert!(!dom.dominates(merge, left));
        assert_eq!(dom.children[merge.0], vec![latch, exit]);
        assert_eq!(dom.frontier[left.0], vec![merge]);
        assert_eq!(dom.frontier[right.0], vec![merge]);
        assert_eq!(dom.frontier[latch.0], vec![merge]);
        assert_eq!(dom.frontier[merge.0], vec![merge]);

        assert!(!dom.is_reachable(dead_def));
        assert!(!dom.is_reachable(dead_use));
        assert!(dom.dominates(dead_def, dead_def));
        assert!(dom.dominates(dead_use, dead_use));
        assert!(!dom.dominates(dead_def, dead_use));
        assert!(!dom.dominates(func.entry, dead_use));
        assert!(dom.dominates_for_availability(dead_def, dead_use));
        assert!(!dom.dominates_for_availability(dead_use, dead_def));
        assert!(dom.dominates_for_availability(func.entry, dead_use));
    }

    #[test]
    fn handles_large_unreachable_cfg_without_rejecting_linear_availability() {
        const DEAD_BLOCKS: usize = 4096;

        let mut func = Function::new("long_dead_cfg", Type::Void);
        func.set_terminator(func.entry, Terminator::Return(None));
        let first = func.add_block("dead.0");
        let mut previous = first;
        for index in 1..DEAD_BLOCKS {
            let next = func.add_block(format!("dead.{index}"));
            func.set_terminator(previous, Terminator::Jump(next));
            previous = next;
        }
        func.set_terminator(previous, Terminator::Return(None));

        let cfg = ControlFlowGraph::new(&func);
        let dom = Dominators::new(&func, &cfg);
        assert!(dom.dominates_for_availability(first, previous));
        assert!(!dom.dominates_for_availability(previous, first));
    }

    #[test]
    fn handles_large_linear_cfg_without_recursive_dfs() {
        const BLOCKS: usize = 4096;

        let mut func = Function::new("long_cfg", Type::Void);
        let mut previous = func.entry;
        for index in 1..BLOCKS {
            let next = func.add_block(format!("block.{index}"));
            func.set_terminator(previous, Terminator::Jump(next));
            previous = next;
        }
        func.set_terminator(previous, Terminator::Return(None));

        let cfg = ControlFlowGraph::new(&func);
        let dom = Dominators::new(&func, &cfg);
        let middle = BlockId(BLOCKS / 2);
        let last = BlockId(BLOCKS - 1);
        assert!(dom.dominates(func.entry, last));
        assert!(dom.dominates(middle, last));
        assert!(!dom.dominates(last, middle));
        assert_eq!(dom.children[middle.0], vec![BlockId(middle.0 + 1)]);
        assert!(dom.frontier.iter().all(Vec::is_empty));
    }
}
