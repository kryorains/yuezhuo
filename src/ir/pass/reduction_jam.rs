use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, InductionVariable, LoopInfo, NaturalLoop};
use super::util::{const_i32, defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_SETUP_INSTRUCTIONS: usize = 16;
const MAX_INNER_INSTRUCTIONS: usize = 32;

/// Processes two adjacent outer-loop iterations in one shared scalar reduction.
///
/// This is deliberately narrower than a general loop interchange: it accepts a
/// canonical two-level loop whose inner loop is a single pure reduction block
/// and whose only outer-iteration side effect is one store. The fast pair loop
/// is followed by the complete original loop as a zero/one-iteration tail.
/// Memory independence is proved from global-object identity and the final GEP
/// index, rather than from source names or fixed dimensions.
pub(super) struct ReductionJamPass;

impl ReductionJamPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for ReductionJamPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            jam_one_nest(func);
        }
    }
}

#[derive(Clone)]
struct JamCandidate {
    outer_header: BlockId,
    outer_preheader: BlockId,
    outer_latch: BlockId,
    outer_induction: InductionVariable,
    outer_bound: ValueId,
    inner_header: BlockId,
    inner_preheader: BlockId,
    inner_body: BlockId,
    inner_exit: BlockId,
    inner_induction: InductionVariable,
    inner_bound: ValueId,
    accumulator: ValueId,
    accumulator_initial: ValueId,
    accumulator_next: ValueId,
    store_ptr: ValueId,
}

fn jam_one_nest(func: &mut Function) {
    if func.has_reduction_jam() || func.blocks.len() > 1024 || func.values.len() > 8192 {
        return;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loops = LoopInfo::new(&cfg, &dom).loops().to_vec();
    let Some(candidate) = find_candidate(func, &loops, &dom) else {
        return;
    };
    apply_candidate(func, &candidate);
    if let Err(errors) = func.verify() {
        panic!(
            "reduction unroll-and-jam produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn find_candidate(
    func: &Function,
    loops: &[NaturalLoop],
    dom: &Dominators,
) -> Option<JamCandidate> {
    let mut inner_loops = loops.iter().collect::<Vec<_>>();
    inner_loops.sort_by_key(|natural_loop| natural_loop.blocks.len());
    for inner in inner_loops {
        let Some(inner_body) = inner.unique_latch() else {
            continue;
        };
        let (Some(inner_preheader), Some(inner_exit)) =
            (inner.dedicated_preheader, inner.unique_exit())
        else {
            continue;
        };
        if inner.blocks != HashSet::from([inner.header, inner_body])
            || !matches!(
                func.blocks[inner_body.0].terminator,
                Some(Terminator::Jump(target)) if target == inner.header
            )
            || !matches!(
                func.blocks[inner_preheader.0].terminator,
                Some(Terminator::Jump(target)) if target == inner.header
            )
            || executable_instruction_count(func, inner_preheader) > MAX_SETUP_INSTRUCTIONS
            || executable_instruction_count(func, inner_body) > MAX_INNER_INSTRUCTIONS
            || !is_pure_setup_block(func, inner_preheader)
            || !is_pure_reduction_block(func, inner_body)
        {
            continue;
        }

        let inner_phis = phi_results(func, inner.header);
        if inner_phis.len() != 2 {
            continue;
        }
        let Some(inner_induction) = inner_phis.iter().find_map(|phi| {
            analyze_i32_induction(func, inner, *phi).filter(|induction| induction.step == 1)
        }) else {
            continue;
        };
        let Some(accumulator) = inner_phis
            .iter()
            .copied()
            .find(|phi| *phi != inner_induction.phi && func.value(*phi).ty == Type::I32)
        else {
            continue;
        };
        let Some((accumulator_initial, accumulator_next)) =
            phi_pair(func, accumulator, inner_preheader, inner_body)
        else {
            continue;
        };
        let Some(inner_bound) = canonical_less_than_bound(func, inner, inner_induction.phi) else {
            continue;
        };

        let Some(outer) = loops
            .iter()
            .filter(|outer| {
                outer.blocks.len() > inner.blocks.len()
                    && outer.blocks.contains(&inner.header)
                    && outer.blocks.contains(&inner_body)
                    && outer.blocks.contains(&inner_preheader)
                    && outer.blocks.contains(&inner_exit)
            })
            .min_by_key(|outer| outer.blocks.len())
        else {
            continue;
        };
        let (Some(outer_preheader), Some(outer_latch)) =
            (outer.dedicated_preheader, outer.unique_latch())
        else {
            continue;
        };
        if outer_latch != inner_exit
            || outer.blocks
                != HashSet::from([
                    outer.header,
                    inner_preheader,
                    inner.header,
                    inner_body,
                    inner_exit,
                ])
            || !matches!(
                func.blocks[outer_preheader.0].terminator,
                Some(Terminator::Jump(target)) if target == outer.header
            )
            || !matches!(
                func.blocks[inner_exit.0].terminator,
                Some(Terminator::Jump(target)) if target == outer.header
            )
        {
            continue;
        }
        let outer_phis = phi_results(func, outer.header);
        if outer_phis.len() != 1 {
            continue;
        }
        let Some(outer_induction) = analyze_i32_induction(func, outer, outer_phis[0])
            .filter(|induction| induction.step == 1)
        else {
            continue;
        };
        let Some(outer_initial) = const_i32(func, outer_induction.initial) else {
            continue;
        };
        if !(0..i32::MAX).contains(&outer_initial) || outer_initial & 1 != 0 {
            continue;
        }
        let Some(outer_bound) = canonical_less_than_bound(func, outer, outer_induction.phi) else {
            continue;
        };
        if !value_available_at(func, dom, outer_bound, outer_preheader)
            || !value_available_at(func, dom, inner_bound, outer_preheader)
            || !value_available_at(func, dom, inner_induction.initial, outer_preheader)
            || !value_mappable_from_blocks(
                func,
                dom,
                accumulator_initial,
                outer_preheader,
                outer_induction.phi,
                &[inner_preheader],
            )
        {
            continue;
        }
        let Some(inside_target) = loop_inside_target(func, outer) else {
            continue;
        };
        if inside_target != inner_preheader {
            continue;
        }

        let stores = func.blocks[inner_exit.0]
            .insts
            .iter()
            .filter_map(|inst| match inst.kind {
                InstKind::Store { ptr, value } => Some((ptr, value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [(store_ptr, store_value)] = stores.as_slice() else {
            continue;
        };
        if *store_value != accumulator
            || !has_only_one_store_side_effect(func, inner_exit)
            || !value_mappable_from_blocks(
                func,
                dom,
                *store_ptr,
                outer_preheader,
                outer_induction.phi,
                &[inner_preheader, inner_exit],
            )
            || !proves_lane_independence(func, *store_ptr, outer_induction.phi, inner_body)
        {
            continue;
        }
        let setup_results = func.blocks[inner_preheader.0]
            .insts
            .iter()
            .filter_map(|inst| inst.result)
            .collect::<HashSet<_>>();
        let setup_mapped = HashSet::from([outer_induction.phi]);
        let mut loop_mapped = setup_results.clone();
        loop_mapped.extend([outer_induction.phi, inner_induction.phi, accumulator]);
        if !block_operands_cloneable(
            func,
            dom,
            inner_preheader,
            outer_preheader,
            &setup_mapped,
            false,
        ) || !block_operands_cloneable(
            func,
            dom,
            inner_body,
            outer_preheader,
            &loop_mapped,
            false,
        ) || !block_operands_cloneable(
            func,
            dom,
            inner_exit,
            outer_preheader,
            &loop_mapped,
            true,
        ) {
            continue;
        }

        return Some(JamCandidate {
            outer_header: outer.header,
            outer_preheader,
            outer_latch,
            outer_induction,
            outer_bound,
            inner_header: inner.header,
            inner_preheader,
            inner_body,
            inner_exit,
            inner_induction,
            inner_bound,
            accumulator,
            accumulator_initial,
            accumulator_next,
            store_ptr: *store_ptr,
        });
    }
    None
}

fn apply_candidate(func: &mut Function, candidate: &JamCandidate) {
    let fast_header = func.add_block("reduction.jam.header");
    let fast_setup = func.add_block("reduction.jam.setup");
    let fast_inner_header = func.add_block("reduction.jam.inner");
    let fast_inner_body = func.add_block("reduction.jam.body");
    let fast_exit = func.add_block("reduction.jam.exit");

    let fast_outer = func
        .append_inst(
            fast_header,
            InstKind::Phi {
                incomings: vec![
                    (candidate.outer_preheader, candidate.outer_induction.initial),
                    (fast_exit, candidate.outer_induction.initial),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let one = get_or_add_i32_const(func, 1);
    let two = get_or_add_i32_const(func, 2);
    let lane_one = func
        .append_inst(
            fast_header,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: fast_outer,
                rhs: one,
            },
            Some(Type::I32),
        )
        .unwrap();
    let pair_condition = func
        .append_inst(
            fast_header,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: lane_one,
                rhs: candidate.outer_bound,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        fast_header,
        Terminator::Branch {
            cond: pair_condition,
            then_target: fast_setup,
            else_target: candidate.outer_header,
        },
    );

    let mut lane_zero = HashMap::from([(candidate.outer_induction.phi, fast_outer)]);
    let mut lane_one_map = HashMap::from([(candidate.outer_induction.phi, lane_one)]);
    clone_pure_block(func, candidate.inner_preheader, fast_setup, &mut lane_zero);
    clone_pure_block(
        func,
        candidate.inner_preheader,
        fast_setup,
        &mut lane_one_map,
    );

    let inner_initial_zero = map_value(candidate.inner_induction.initial, &lane_zero);
    let inner_initial_one = map_value(candidate.inner_induction.initial, &lane_one_map);
    assert_eq!(inner_initial_zero, inner_initial_one);
    let fast_inner = func
        .append_inst(
            fast_inner_header,
            InstKind::Phi {
                incomings: vec![
                    (fast_setup, inner_initial_zero),
                    (fast_inner_body, inner_initial_zero),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let fast_accumulator_zero = func
        .append_inst(
            fast_inner_header,
            InstKind::Phi {
                incomings: vec![
                    (
                        fast_setup,
                        map_value(candidate.accumulator_initial, &lane_zero),
                    ),
                    (
                        fast_inner_body,
                        map_value(candidate.accumulator_initial, &lane_zero),
                    ),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let fast_accumulator_one = func
        .append_inst(
            fast_inner_header,
            InstKind::Phi {
                incomings: vec![
                    (
                        fast_setup,
                        map_value(candidate.accumulator_initial, &lane_one_map),
                    ),
                    (
                        fast_inner_body,
                        map_value(candidate.accumulator_initial, &lane_one_map),
                    ),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    let inner_condition = func
        .append_inst(
            fast_inner_header,
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs: fast_inner,
                rhs: candidate.inner_bound,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.set_terminator(
        fast_inner_header,
        Terminator::Branch {
            cond: inner_condition,
            then_target: fast_inner_body,
            else_target: fast_exit,
        },
    );
    func.set_terminator(fast_setup, Terminator::Jump(fast_inner_header));

    lane_zero.insert(candidate.inner_induction.phi, fast_inner);
    lane_zero.insert(candidate.accumulator, fast_accumulator_zero);
    lane_one_map.insert(candidate.inner_induction.phi, fast_inner);
    lane_one_map.insert(candidate.accumulator, fast_accumulator_one);
    clone_pure_block(func, candidate.inner_body, fast_inner_body, &mut lane_zero);
    clone_pure_block(
        func,
        candidate.inner_body,
        fast_inner_body,
        &mut lane_one_map,
    );
    let fast_inner_next = map_value(candidate.inner_induction.next, &lane_zero);
    let fast_accumulator_zero_next = map_value(candidate.accumulator_next, &lane_zero);
    let fast_accumulator_one_next = map_value(candidate.accumulator_next, &lane_one_map);
    set_phi_backedge(func, fast_inner, fast_inner_body, fast_inner_next);
    set_phi_backedge(
        func,
        fast_accumulator_zero,
        fast_inner_body,
        fast_accumulator_zero_next,
    );
    set_phi_backedge(
        func,
        fast_accumulator_one,
        fast_inner_body,
        fast_accumulator_one_next,
    );
    func.set_terminator(fast_inner_body, Terminator::Jump(fast_inner_header));

    clone_exit_pure_instructions(func, candidate.inner_exit, fast_exit, &mut lane_zero);
    clone_exit_pure_instructions(func, candidate.inner_exit, fast_exit, &mut lane_one_map);
    func.append_inst(
        fast_exit,
        InstKind::Store {
            ptr: map_value(candidate.store_ptr, &lane_zero),
            value: fast_accumulator_zero,
        },
        None,
    );
    func.append_inst(
        fast_exit,
        InstKind::Store {
            ptr: map_value(candidate.store_ptr, &lane_one_map),
            value: fast_accumulator_one,
        },
        None,
    );
    let fast_outer_next = func
        .append_inst(
            fast_exit,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: fast_outer,
                rhs: two,
            },
            Some(Type::I32),
        )
        .unwrap();
    set_phi_backedge(func, fast_outer, fast_exit, fast_outer_next);
    func.set_terminator(fast_exit, Terminator::Jump(fast_header));

    func.blocks[candidate.outer_preheader.0].terminator = Some(Terminator::Jump(fast_header));
    replace_phi_incoming(
        func,
        candidate.outer_induction.phi,
        candidate.outer_preheader,
        fast_header,
        fast_outer,
    );
    func.mark_reduction_jammed();
}

fn clone_pure_block(
    func: &mut Function,
    source: BlockId,
    target: BlockId,
    values: &mut HashMap<ValueId, ValueId>,
) {
    let instructions = func.blocks[source.0].insts.clone();
    for inst in instructions {
        if matches!(inst.kind, InstKind::Nop) {
            continue;
        }
        let kind = remap_pure_kind(&inst.kind, values)
            .expect("candidate validation must accept every cloned instruction");
        let result_ty = inst.result.map(|result| func.value(result).ty.clone());
        let cloned = func.append_inst(target, kind, result_ty);
        if let (Some(original), Some(cloned)) = (inst.result, cloned) {
            values.insert(original, cloned);
        }
    }
}

fn clone_exit_pure_instructions(
    func: &mut Function,
    source: BlockId,
    target: BlockId,
    values: &mut HashMap<ValueId, ValueId>,
) {
    let instructions = func.blocks[source.0].insts.clone();
    for inst in instructions {
        if matches!(inst.kind, InstKind::Nop | InstKind::Store { .. }) {
            continue;
        }
        let kind = remap_pure_kind(&inst.kind, values)
            .expect("candidate exit validation must accept every pure instruction");
        let result_ty = inst.result.map(|result| func.value(result).ty.clone());
        let cloned = func.append_inst(target, kind, result_ty);
        if let (Some(original), Some(cloned)) = (inst.result, cloned) {
            values.insert(original, cloned);
        }
    }
}

fn remap_pure_kind(kind: &InstKind, values: &HashMap<ValueId, ValueId>) -> Option<InstKind> {
    let map = |value| map_value(value, values);
    match kind {
        InstKind::Nop => Some(InstKind::Nop),
        InstKind::Load { ptr } => Some(InstKind::Load { ptr: map(*ptr) }),
        InstKind::Unary { op, value } => Some(InstKind::Unary {
            op: *op,
            value: map(*value),
        }),
        InstKind::Binary { op, lhs, rhs } => Some(InstKind::Binary {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        }),
        InstKind::Icmp { op, lhs, rhs } => Some(InstKind::Icmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        }),
        InstKind::Fcmp { op, lhs, rhs } => Some(InstKind::Fcmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        }),
        InstKind::Cast { op, value } => Some(InstKind::Cast {
            op: *op,
            value: map(*value),
        }),
        InstKind::Gep { base, indices } => Some(InstKind::Gep {
            base: map(*base),
            indices: indices.iter().map(|index| map(*index)).collect(),
        }),
        InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Store { .. }
        | InstKind::MemZero { .. }
        | InstKind::Call { .. } => None,
    }
}

fn map_value(value: ValueId, values: &HashMap<ValueId, ValueId>) -> ValueId {
    values.get(&value).copied().unwrap_or(value)
}

fn set_phi_backedge(func: &mut Function, phi: ValueId, pred: BlockId, value: ValueId) {
    let ValueKind::Inst(block, inst_idx) = func.value(phi).kind else {
        unreachable!();
    };
    let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
        unreachable!();
    };
    incomings
        .iter_mut()
        .find(|(incoming_pred, _)| *incoming_pred == pred)
        .unwrap()
        .1 = value;
}

fn replace_phi_incoming(
    func: &mut Function,
    phi: ValueId,
    old_pred: BlockId,
    new_pred: BlockId,
    value: ValueId,
) {
    let ValueKind::Inst(block, inst_idx) = func.value(phi).kind else {
        unreachable!();
    };
    let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
        unreachable!();
    };
    let incoming = incomings
        .iter_mut()
        .find(|(pred, _)| *pred == old_pred)
        .unwrap();
    *incoming = (new_pred, value);
}

fn phi_results(func: &Function, block: BlockId) -> Vec<ValueId> {
    func.blocks[block.0]
        .insts
        .iter()
        .filter_map(|inst| matches!(inst.kind, InstKind::Phi { .. }).then_some(inst.result?))
        .collect()
}

fn phi_pair(
    func: &Function,
    phi: ValueId,
    entering: BlockId,
    latch: BlockId,
) -> Option<(ValueId, ValueId)> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    Some((
        incomings.iter().find(|(pred, _)| *pred == entering)?.1,
        incomings.iter().find(|(pred, _)| *pred == latch)?.1,
    ))
}

fn canonical_less_than_bound(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: ValueId,
) -> Option<ValueId> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks[natural_loop.header.0].terminator.as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || natural_loop.blocks.contains(else_target) {
        return None;
    }
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
        return None;
    };
    match (*op, *lhs == induction, *rhs == induction) {
        (CmpOp::Lt, true, false) => Some(*rhs),
        (CmpOp::Gt, false, true) => Some(*lhs),
        _ => None,
    }
}

fn loop_inside_target(func: &Function, natural_loop: &NaturalLoop) -> Option<BlockId> {
    let Terminator::Branch {
        then_target,
        else_target,
        ..
    } = func.blocks[natural_loop.header.0].terminator.as_ref()?
    else {
        return None;
    };
    match (
        natural_loop.blocks.contains(then_target),
        natural_loop.blocks.contains(else_target),
    ) {
        (true, false) => Some(*then_target),
        (false, true) => Some(*else_target),
        _ => None,
    }
}

fn is_pure_setup_block(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0].insts.iter().all(|inst| {
        matches!(
            inst.kind,
            InstKind::Nop
                | InstKind::Unary { .. }
                | InstKind::Binary { .. }
                | InstKind::Icmp { .. }
                | InstKind::Fcmp { .. }
                | InstKind::Cast { .. }
                | InstKind::Gep { .. }
        )
    })
}

fn is_pure_reduction_block(func: &Function, block: BlockId) -> bool {
    func.blocks[block.0].insts.iter().all(|inst| {
        matches!(
            inst.kind,
            InstKind::Nop
                | InstKind::Load { .. }
                | InstKind::Unary { .. }
                | InstKind::Binary { .. }
                | InstKind::Icmp { .. }
                | InstKind::Fcmp { .. }
                | InstKind::Cast { .. }
                | InstKind::Gep { .. }
        )
    })
}

fn has_only_one_store_side_effect(func: &Function, block: BlockId) -> bool {
    let mut stores = 0usize;
    for inst in &func.blocks[block.0].insts {
        match inst.kind {
            InstKind::Store { .. } => stores += 1,
            InstKind::Call { .. } | InstKind::MemZero { .. } | InstKind::Load { .. } => {
                return false;
            }
            _ => {}
        }
    }
    stores == 1
}

fn proves_lane_independence(
    func: &Function,
    store_ptr: ValueId,
    outer_induction: ValueId,
    inner_body: BlockId,
) -> bool {
    let Some((store_global, store_final_index)) = global_and_final_index(func, store_ptr) else {
        return false;
    };
    if store_final_index != outer_induction {
        return false;
    }
    for inst in &func.blocks[inner_body.0].insts {
        let InstKind::Load { ptr } = inst.kind else {
            continue;
        };
        let Some((load_global, load_final_index)) = global_and_final_index(func, ptr) else {
            return false;
        };
        if load_global == store_global && load_final_index != outer_induction {
            return false;
        }
    }
    true
}

fn global_and_final_index(func: &Function, ptr: ValueId) -> Option<(String, ValueId)> {
    let InstKind::Gep { base, indices } = defining_inst(func, ptr)? else {
        return None;
    };
    let [final_index] = indices.as_slice() else {
        return None;
    };
    let mut root = *base;
    loop {
        match &func.value(root).kind {
            ValueKind::Global(name) => return Some((name.clone(), *final_index)),
            ValueKind::Inst(_, _) => {
                let InstKind::Gep { base, .. } = defining_inst(func, root)? else {
                    return None;
                };
                root = *base;
            }
            ValueKind::Param | ValueKind::Const(_) => return None,
        }
    }
}

fn value_mappable_from_blocks(
    func: &Function,
    dom: &Dominators,
    value: ValueId,
    insertion_preheader: BlockId,
    outer_induction: ValueId,
    cloned_blocks: &[BlockId],
) -> bool {
    if value == outer_induction || value_available_at(func, dom, value, insertion_preheader) {
        return true;
    }
    matches!(
        func.value(value).kind,
        ValueKind::Inst(block, _) if cloned_blocks.contains(&block)
    )
}

fn block_operands_cloneable(
    func: &Function,
    dom: &Dominators,
    block: BlockId,
    insertion_preheader: BlockId,
    mapped: &HashSet<ValueId>,
    skip_store: bool,
) -> bool {
    let mut local = HashSet::new();
    for inst in &func.blocks[block.0].insts {
        if skip_store && matches!(inst.kind, InstKind::Store { .. }) {
            continue;
        }
        for operand in instruction_operands(&inst.kind) {
            if !mapped.contains(&operand)
                && !local.contains(&operand)
                && !value_available_at(func, dom, operand, insertion_preheader)
            {
                return false;
            }
        }
        if let Some(result) = inst.result {
            local.insert(result);
        }
    }
    true
}

fn instruction_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut operands = vec![*base];
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn value_available_at(func: &Function, dom: &Dominators, value: ValueId, block: BlockId) -> bool {
    match func.value(value).kind {
        ValueKind::Const(_) | ValueKind::Global(_) | ValueKind::Param => true,
        ValueKind::Inst(def_block, _) => def_block == block || dom.dominates(def_block, block),
    }
}

fn executable_instruction_count(func: &Function, block: BlockId) -> usize {
    func.blocks[block.0]
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Const;

    #[test]
    fn jams_independent_adjacent_reductions_and_is_idempotent() {
        let mut func = reduction_nest(false, 0);
        let original_blocks = func.blocks.len();
        assert!(func.verify().is_ok());

        jam_one_nest(&mut func);
        assert_eq!(func.blocks.len(), original_blocks + 5);
        assert_eq!(store_count(&func), 3);
        assert!(func.verify().is_ok());

        let once = func.clone();
        jam_one_nest(&mut func);
        assert_eq!(func, once);
    }

    #[test]
    fn rejects_a_cross_lane_store_load_dependence() {
        let mut func = reduction_nest(true, 0);
        let original = func.clone();
        assert!(func.verify().is_ok());
        jam_one_nest(&mut func);
        assert_eq!(func, original);
    }

    #[test]
    fn rejects_an_odd_initial_value_that_could_reach_i32_max() {
        let mut func = reduction_nest(false, 1);
        let original = func.clone();
        jam_one_nest(&mut func);
        assert_eq!(func, original);
    }

    fn reduction_nest(load_uses_inner_index: bool, outer_initial: i32) -> Function {
        let mut func = Function::new("generic_reduction", Type::Void);
        let bound = func.add_param("bound", Type::I32);
        let row = func.add_param("row", Type::I32);
        let matrix_ty = Type::Array {
            elem: Box::new(Type::Array {
                elem: Box::new(Type::I32),
                len: 16,
            }),
            len: 16,
        };
        let row_ty = Type::Ptr(Box::new(Type::Array {
            elem: Box::new(Type::I32),
            len: 16,
        }));
        let elem_ty = Type::Ptr(Box::new(Type::I32));
        let output = func.add_global_ref("output", Type::Ptr(Box::new(matrix_ty.clone())));
        let input = if load_uses_inner_index {
            // Lowering creates a fresh ValueId for every reference to the same
            // global symbol; the dependence proof must compare symbol identity.
            func.add_global_ref("output", Type::Ptr(Box::new(matrix_ty.clone())))
        } else {
            func.add_global_ref("input", Type::Ptr(Box::new(matrix_ty)))
        };
        let initial = func.add_const(Const::Int(outer_initial));
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));

        let outer_header = func.add_block("outer.header");
        let inner_preheader = func.add_block("inner.preheader");
        let inner_header = func.add_block("inner.header");
        let inner_body = func.add_block("inner.body");
        let inner_exit = func.add_block("inner.exit");
        let outer_exit = func.add_block("outer.exit");
        func.set_terminator(func.entry, Terminator::Jump(outer_header));

        let outer_index = func
            .append_inst(
                outer_header,
                InstKind::Phi {
                    incomings: vec![(func.entry, initial), (inner_exit, initial)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let outer_condition = func
            .append_inst(
                outer_header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: outer_index,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            outer_header,
            Terminator::Branch {
                cond: outer_condition,
                then_target: inner_preheader,
                else_target: outer_exit,
            },
        );

        let output_row = func
            .append_inst(
                inner_preheader,
                InstKind::Gep {
                    base: output,
                    indices: vec![row],
                },
                Some(row_ty.clone()),
            )
            .unwrap();
        let output_element = func
            .append_inst(
                inner_preheader,
                InstKind::Gep {
                    base: output_row,
                    indices: vec![outer_index],
                },
                Some(elem_ty.clone()),
            )
            .unwrap();
        let input_row = func
            .append_inst(
                inner_preheader,
                InstKind::Gep {
                    base: input,
                    indices: vec![row],
                },
                Some(row_ty.clone()),
            )
            .unwrap();
        func.set_terminator(inner_preheader, Terminator::Jump(inner_header));

        let inner_index = func
            .append_inst(
                inner_header,
                InstKind::Phi {
                    incomings: vec![(inner_preheader, zero), (inner_body, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let accumulator = func
            .append_inst(
                inner_header,
                InstKind::Phi {
                    incomings: vec![(inner_preheader, zero), (inner_body, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let inner_condition = func
            .append_inst(
                inner_header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: inner_index,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            inner_header,
            Terminator::Branch {
                cond: inner_condition,
                then_target: inner_body,
                else_target: inner_exit,
            },
        );

        let selected_index = if load_uses_inner_index {
            inner_index
        } else {
            outer_index
        };
        let input_element = func
            .append_inst(
                inner_body,
                InstKind::Gep {
                    base: input_row,
                    indices: vec![selected_index],
                },
                Some(elem_ty),
            )
            .unwrap();
        let loaded = func
            .append_inst(
                inner_body,
                InstKind::Load { ptr: input_element },
                Some(Type::I32),
            )
            .unwrap();
        let accumulator_next = func
            .append_inst(
                inner_body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: accumulator,
                    rhs: loaded,
                },
                Some(Type::I32),
            )
            .unwrap();
        let inner_next = func
            .append_inst(
                inner_body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: inner_index,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(inner_body, Terminator::Jump(inner_header));
        set_phi_backedge(&mut func, inner_index, inner_body, inner_next);
        set_phi_backedge(&mut func, accumulator, inner_body, accumulator_next);

        func.append_inst(
            inner_exit,
            InstKind::Store {
                ptr: output_element,
                value: accumulator,
            },
            None,
        );
        let outer_next = func
            .append_inst(
                inner_exit,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: outer_index,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        set_phi_backedge(&mut func, outer_index, inner_exit, outer_next);
        func.set_terminator(inner_exit, Terminator::Jump(outer_header));
        func.set_terminator(outer_exit, Terminator::Return(None));
        func
    }

    fn store_count(func: &Function) -> usize {
        func.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter(|inst| matches!(inst.kind, InstKind::Store { .. }))
            .count()
    }
}
