use crate::ir::{BlockId, Function, Terminator};

const MAX_NATURAL_LOOP_VISITS_PER_BLOCK: usize = 64;

pub(crate) fn natural_loop_depths(func: &Function) -> Vec<usize> {
    let block_count = func.blocks.len();
    if block_count == 0 || func.entry.0 >= block_count {
        return vec![0; block_count];
    }

    let (predecessors, successors) = control_flow_graph(func);
    let reverse_postorder = reverse_postorder(&successors, func.entry.0);
    let immediate_dominators =
        immediate_dominators(&predecessors, &reverse_postorder, func.entry.0, block_count);
    let (dom_in, dom_out) = dominator_tree_intervals(
        &immediate_dominators,
        &reverse_postorder,
        func.entry.0,
        block_count,
    );

    let mut latches = vec![Vec::new(); block_count];
    for (pred, block_successors) in successors.iter().enumerate() {
        for successor in block_successors {
            if dominates(*successor, pred, &dom_in, &dom_out) {
                latches[*successor].push(pred);
            }
        }
    }

    let visit_budget = block_count
        .saturating_mul(MAX_NATURAL_LOOP_VISITS_PER_BLOCK)
        .max(1024);
    let mut total_visits = 0usize;
    let mut marks = vec![0usize; block_count];
    let mut generation = 0usize;
    let mut depths = vec![0usize; block_count];

    for (header, loop_latches) in latches.into_iter().enumerate() {
        if loop_latches.is_empty() {
            continue;
        }
        generation += 1;
        let mut members = Vec::new();
        let mut worklist = Vec::new();
        mark_member(
            header,
            generation,
            &mut marks,
            &mut members,
            &mut worklist,
            false,
        );
        for latch in loop_latches {
            mark_member(
                latch,
                generation,
                &mut marks,
                &mut members,
                &mut worklist,
                latch != header,
            );
        }
        while let Some(block) = worklist.pop() {
            for predecessor in &predecessors[block] {
                if dominates(header, *predecessor, &dom_in, &dom_out) {
                    mark_member(
                        *predecessor,
                        generation,
                        &mut marks,
                        &mut members,
                        &mut worklist,
                        *predecessor != header,
                    );
                }
            }
        }

        if total_visits.saturating_add(members.len()) > visit_budget {
            break;
        }
        total_visits += members.len();
        for block in members {
            depths[block] = depths[block].saturating_add(1);
        }
    }
    depths
}

fn control_flow_graph(func: &Function) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut predecessors = vec![Vec::new(); func.blocks.len()];
    let mut successors_by_block = vec![Vec::new(); func.blocks.len()];
    for (pred, block) in func.blocks.iter().enumerate() {
        for successor in successors(block.terminator.as_ref()) {
            if successor.0 >= func.blocks.len() {
                continue;
            }
            predecessors[successor.0].push(pred);
            successors_by_block[pred].push(successor.0);
        }
    }
    (predecessors, successors_by_block)
}

fn reverse_postorder(successors: &[Vec<usize>], entry: usize) -> Vec<usize> {
    let mut seen = vec![false; successors.len()];
    let mut postorder = Vec::new();
    let mut stack = vec![(entry, 0usize)];
    seen[entry] = true;
    while let Some((block, next_successor)) = stack.last_mut() {
        if *next_successor < successors[*block].len() {
            let successor = successors[*block][*next_successor];
            *next_successor += 1;
            if !seen[successor] {
                seen[successor] = true;
                stack.push((successor, 0));
            }
        } else {
            postorder.push(*block);
            stack.pop();
        }
    }
    postorder.reverse();
    postorder
}

fn immediate_dominators(
    predecessors: &[Vec<usize>],
    reverse_postorder: &[usize],
    entry: usize,
    block_count: usize,
) -> Vec<Option<usize>> {
    let mut rank = vec![usize::MAX; block_count];
    for (index, block) in reverse_postorder.iter().enumerate() {
        rank[*block] = index;
    }
    let mut idom = vec![None; block_count];
    idom[entry] = Some(entry);

    loop {
        let mut changed = false;
        for block in reverse_postorder.iter().copied().skip(1) {
            let mut defined_predecessors = predecessors[block]
                .iter()
                .copied()
                .filter(|pred| idom[*pred].is_some());
            let Some(mut next_idom) = defined_predecessors.next() else {
                continue;
            };
            for predecessor in defined_predecessors {
                next_idom = intersect(predecessor, next_idom, &idom, &rank);
            }
            if idom[block] != Some(next_idom) {
                idom[block] = Some(next_idom);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    idom
}

fn intersect(mut lhs: usize, mut rhs: usize, idom: &[Option<usize>], rank: &[usize]) -> usize {
    while lhs != rhs {
        while rank[lhs] > rank[rhs] {
            lhs = idom[lhs].expect("reachable dominator must be defined");
        }
        while rank[rhs] > rank[lhs] {
            rhs = idom[rhs].expect("reachable dominator must be defined");
        }
    }
    lhs
}

fn dominator_tree_intervals(
    idom: &[Option<usize>],
    reachable: &[usize],
    entry: usize,
    block_count: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut children = vec![Vec::new(); block_count];
    for block in reachable.iter().copied() {
        if block == entry {
            continue;
        }
        if let Some(parent) = idom[block] {
            children[parent].push(block);
        }
    }

    let mut dom_in = vec![usize::MAX; block_count];
    let mut dom_out = vec![usize::MAX; block_count];
    let mut timer = 0usize;
    let mut stack = vec![(entry, 0usize)];
    dom_in[entry] = timer;
    timer += 1;
    while let Some((block, next_child)) = stack.last_mut() {
        if *next_child < children[*block].len() {
            let child = children[*block][*next_child];
            *next_child += 1;
            dom_in[child] = timer;
            timer += 1;
            stack.push((child, 0));
        } else {
            dom_out[*block] = timer;
            timer += 1;
            stack.pop();
        }
    }
    (dom_in, dom_out)
}

fn dominates(block: usize, other: usize, dom_in: &[usize], dom_out: &[usize]) -> bool {
    dom_in[block] != usize::MAX
        && dom_in[block] <= dom_in[other]
        && dom_out[other] <= dom_out[block]
}

fn mark_member(
    block: usize,
    generation: usize,
    marks: &mut [usize],
    members: &mut Vec<usize>,
    worklist: &mut Vec<usize>,
    traverse_predecessors: bool,
) {
    if marks[block] == generation {
        return;
    }
    marks[block] = generation;
    members.push(block);
    if traverse_predecessors {
        worklist.push(block);
    }
}

fn successors(terminator: Option<&Terminator>) -> Vec<BlockId> {
    match terminator {
        Some(Terminator::Jump(target)) => vec![*target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) if then_target == else_target => vec![*then_target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) => vec![*then_target, *else_target],
        Some(Terminator::Return(_)) | None => Vec::new(),
    }
}
