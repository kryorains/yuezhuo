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

/// Places blocks using a profile-free CFG trace and orders canonical natural
/// loops as `body .. latch, header, exit`.
///
/// The preheader still jumps to the header for the zero-trip check. On the hot
/// path, however, the latch falls through to the header and the header branches
/// back to the body, removing one unconditional branch per iteration. A reverse
/// postorder trace is used only when it preserves every fallthrough backedge and
/// statically reduces control transfers. Block identities and CFG edges are
/// unchanged; this is only an assembly layout.
pub(crate) fn loop_rotated_block_order(func: &Function) -> Vec<usize> {
    const MAX_LAYOUT_BLOCKS: usize = 1024;
    const MAX_ROTATED_LOOPS: usize = 64;

    let block_count = func.blocks.len();
    if block_count == 0 || block_count > MAX_LAYOUT_BLOCKS || func.entry.0 >= block_count {
        return (0..block_count).collect();
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

    let mut candidates = Vec::new();
    let mut latch_exit_candidates = Vec::new();
    for (header, loop_latches) in latches.iter().enumerate() {
        let [latch] = loop_latches.as_slice() else {
            continue;
        };
        if *latch == header {
            continue;
        }

        let members = natural_loop_members(
            header,
            *latch,
            &predecessors,
            &dom_in,
            &dom_out,
            block_count,
        );
        if predecessors[header]
            .iter()
            .filter(|pred| !members[**pred])
            .count()
            != 1
        {
            continue;
        }

        let member_count = members.iter().filter(|member| **member).count();
        if let Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) = func.blocks[*latch].terminator.as_ref()
        {
            let exit = if then_target.0 == header && !members[else_target.0] {
                Some(else_target.0)
            } else if else_target.0 == header && !members[then_target.0] {
                Some(then_target.0)
            } else {
                None
            };
            if let Some(exit) =
                exit.filter(|exit| *exit != header && *exit != *latch && *exit != func.entry.0)
            {
                // A post-tested loop is cheapest as `header .. latch, exit`:
                // the latch branches back and the cold exit falls through.
                latch_exit_candidates.push((member_count, header, *latch, exit));
                continue;
            }
        }

        let Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) = func.blocks[header].terminator.as_ref()
        else {
            continue;
        };
        let then_inside = members[then_target.0];
        let else_inside = members[else_target.0];
        if then_inside == else_inside {
            continue;
        }
        let exit = if then_inside {
            else_target.0
        } else {
            then_target.0
        };
        let latch_falls_back = matches!(
            func.blocks[*latch].terminator,
            Some(Terminator::Jump(target)) if target.0 == header
        );
        let latch_has_equivalent_branch_edges = matches!(
            func.blocks[*latch].terminator,
            Some(Terminator::Branch { then_target, else_target, .. })
                if then_target.0 == header && else_target.0 == header
        );
        if !latch_falls_back && !latch_has_equivalent_branch_edges {
            continue;
        }
        if exit == header || exit == *latch || exit == func.entry.0 {
            continue;
        }
        candidates.push((member_count, header, *latch, exit));
    }

    if candidates.len().saturating_add(latch_exit_candidates.len()) > MAX_ROTATED_LOOPS {
        return (0..block_count).collect();
    }

    // Rotate inner loops before their containing loops. Every edit looks up the
    // current positions, so nested rotations remain stable.
    candidates.sort_by_key(|(member_count, _, _, _)| *member_count);
    latch_exit_candidates.sort_by_key(|(member_count, _, _, _)| *member_count);
    let mut reachable = vec![false; block_count];
    for block in &reverse_postorder {
        reachable[*block] = true;
    }
    let mut source_order = (0..block_count).collect::<Vec<_>>();
    source_order.retain(|block| reachable[*block]);
    rotate_loop_layouts(&mut source_order, &candidates);

    let mut trace_order = reverse_postorder;
    rotate_loop_layouts(&mut trace_order, &candidates);

    // Loop backedges are the only profile-free edges known to be hot. Never
    // trade one of their fallthroughs for an acyclic trace improvement. When
    // both layouts preserve the same number, select RPO only if it eliminates
    // at least one statically emitted jump/branch.
    let source_backedges = fallthrough_backedges(&source_order, &successors, &dom_in, &dom_out);
    let trace_backedges = fallthrough_backedges(&trace_order, &successors, &dom_in, &dom_out);
    let source_transfers = estimated_control_transfers(func, &source_order);
    let trace_transfers = estimated_control_transfers(func, &trace_order);
    let mut chosen = if trace_backedges >= source_backedges && trace_transfers < source_transfers {
        trace_order
    } else {
        source_order
    };
    // Latch-exit placement is a local post-layout improvement.  Keeping it
    // out of the source-vs-RPO choice prevents one newly adjacent exit from
    // changing unrelated diamond fallthroughs elsewhere in the function.
    place_latch_exits(func, &mut chosen, &latch_exit_candidates);
    chosen
}

fn place_latch_exits(
    func: &Function,
    order: &mut Vec<usize>,
    candidates: &[(usize, usize, usize, usize)],
) {
    for (_, header, latch, exit) in candidates.iter().copied() {
        let Some(latch_position) = order.iter().position(|block| *block == latch) else {
            continue;
        };
        // If the backedge already falls through, switching the fallthrough to
        // the exit does not reduce a control transfer and may expose a phi-copy
        // stub on every continuing iteration.
        if order.get(latch_position + 1) == Some(&header) {
            continue;
        }
        let old_transfers = estimated_control_transfers(func, order);
        let mut candidate = order.clone();
        candidate.retain(|block| *block != exit);
        let Some(latch_position) = candidate.iter().position(|block| *block == latch) else {
            continue;
        };
        candidate.insert(latch_position + 1, exit);
        if estimated_control_transfers(func, &candidate) < old_transfers {
            *order = candidate;
        }
    }
}

fn rotate_loop_layouts(order: &mut Vec<usize>, candidates: &[(usize, usize, usize, usize)]) {
    for (_, header, latch, exit) in candidates.iter().copied() {
        order.retain(|block| *block != header && *block != exit);
        let Some(latch_position) = order.iter().position(|block| *block == latch) else {
            continue;
        };
        order.insert(latch_position + 1, header);
        order.insert(latch_position + 2, exit);
    }
}

fn fallthrough_backedges(
    order: &[usize],
    successors: &[Vec<usize>],
    dom_in: &[usize],
    dom_out: &[usize],
) -> usize {
    order
        .windows(2)
        .filter(|pair| {
            let source = pair[0];
            let target = pair[1];
            successors[source].contains(&target) && dominates(target, source, dom_in, dom_out)
        })
        .count()
}

fn estimated_control_transfers(func: &Function, order: &[usize]) -> usize {
    order
        .iter()
        .copied()
        .enumerate()
        .map(|(position, block)| {
            let next = order.get(position + 1).copied();
            match func.blocks[block].terminator.as_ref() {
                Some(Terminator::Jump(target)) => usize::from(next != Some(target.0)),
                Some(Terminator::Branch {
                    then_target,
                    else_target,
                    ..
                }) if then_target == else_target => usize::from(next != Some(then_target.0)),
                Some(Terminator::Branch {
                    then_target,
                    else_target,
                    ..
                }) => {
                    if next == Some(then_target.0) || next == Some(else_target.0) {
                        1
                    } else {
                        2
                    }
                }
                Some(Terminator::Return(_)) => usize::from(next.is_some()),
                None => 0,
            }
        })
        .sum()
}

fn natural_loop_members(
    header: usize,
    latch: usize,
    predecessors: &[Vec<usize>],
    dom_in: &[usize],
    dom_out: &[usize],
    block_count: usize,
) -> Vec<bool> {
    let mut members = vec![false; block_count];
    members[header] = true;
    members[latch] = true;
    let mut worklist = vec![latch];
    while let Some(block) = worklist.pop() {
        for predecessor in &predecessors[block] {
            if !members[*predecessor] && dominates(header, *predecessor, dom_in, dom_out) {
                members[*predecessor] = true;
                if *predecessor != header {
                    worklist.push(*predecessor);
                }
            }
        }
    }
    members
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
