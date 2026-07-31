use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::simplify_cfg::SimplifyCfgPass;
use super::util::{defining_inst, resolve_replacement, ValueReplacements};
use super::ModulePass;
use crate::ir::{BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 16_384;
const MAX_INSTRUCTIONS: usize = 65_536;
const MAX_FUSED_BODY_INSTRUCTIONS: usize = 64;

/// Fuses adjacent canonical loops with the same iteration domain.
///
/// The dependence proof is intentionally strict: the first loop may only
/// produce elements of confined local arrays at index `i`, and the second may
/// only read those same elements at index `i`. This is standard loop fusion
/// plus scalar forwarding; it does not inspect function names, constants,
/// arithmetic performed on the elements, or the source-level algorithm.
pub(super) struct AdjacentLoopFusionPass;

impl AdjacentLoopFusionPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for AdjacentLoopFusionPass {
    fn run(&mut self, module: &mut Module) {
        let mut changed = false;
        for func in &mut module.funcs {
            changed |= fuse_one_pair(func);
        }
        if changed {
            SimplifyCfgPass::preserving_loop_preheaders().run(module);
            for func in &module.funcs {
                if let Err(errors) = func.verify() {
                    panic!(
                        "adjacent loop fusion produced invalid IR in {}: {:?}",
                        func.name, errors
                    );
                }
            }
        }
    }
}

#[derive(Clone)]
struct CanonicalLoop {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    induction: ValueId,
    initial: ValueId,
    bound: ValueId,
}

#[derive(Clone)]
struct CarriedPhi {
    original: ValueId,
    initial: ValueId,
    next: ValueId,
}

#[derive(Clone)]
struct FusionPlan {
    first: CanonicalLoop,
    second: CanonicalLoop,
    carried: Vec<CarriedPhi>,
}

fn fuse_one_pair(func: &mut Function) -> bool {
    let Some(instruction_count) = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
    else {
        return false;
    };
    if func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || instruction_count > MAX_INSTRUCTIONS
    {
        return false;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let loops = loop_info
        .loops()
        .iter()
        .filter_map(|natural_loop| canonical_loop(func, natural_loop))
        .collect::<Vec<_>>();
    let plan = loops.iter().find_map(|first| {
        loops
            .iter()
            .filter(|second| first.header != second.header)
            .find_map(|second| analyze_pair(func, first, second))
    });
    let Some(plan) = plan else {
        return false;
    };
    apply_plan(func, plan);
    true
}

fn canonical_loop(func: &Function, natural_loop: &NaturalLoop) -> Option<CanonicalLoop> {
    let preheader = natural_loop.dedicated_preheader?;
    let body = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    let header = natural_loop.header;
    if natural_loop.blocks != HashSet::from([header, body])
        || natural_loop.exit_edges != [(header, exit)]
        || !matches!(
            func.blocks.get(preheader.0)?.terminator,
            Some(Terminator::Jump(target)) if target == header
        )
        || !matches!(
            func.blocks.get(body.0)?.terminator,
            Some(Terminator::Jump(target)) if target == header
        )
    {
        return None;
    }
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.blocks.get(header.0)?.terminator.as_ref()?
    else {
        return None;
    };
    if *then_target != body || *else_target != exit {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt,
        lhs,
        rhs,
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    let induction = analyze_i32_induction(func, natural_loop, *lhs)?;
    if induction.phi != *lhs || induction.step != 1 || !is_zero_i32(func, induction.initial) {
        return None;
    }
    Some(CanonicalLoop {
        preheader,
        header,
        body,
        exit,
        induction: induction.phi,
        initial: induction.initial,
        bound: *rhs,
    })
}

fn analyze_pair(
    func: &Function,
    first: &CanonicalLoop,
    second: &CanonicalLoop,
) -> Option<FusionPlan> {
    if first.exit != second.preheader
        || first.bound != second.bound
        || !same_i32_constant(func, first.initial, second.initial)
        || func.blocks[first.exit.0]
            .insts
            .iter()
            .any(|inst| !matches!(inst.kind, InstKind::Nop))
        || !matches!(
            func.blocks[first.exit.0].terminator,
            Some(Terminator::Jump(target)) if target == second.header
        )
    {
        return None;
    }

    let first_active = active_instruction_count(&func.blocks[first.body.0].insts);
    let second_active = active_instruction_count(&func.blocks[second.body.0].insts);
    if first_active
        .checked_add(second_active)
        .is_none_or(|count| count > MAX_FUSED_BODY_INSTRUCTIONS)
        || !memory_dependences_allow_fusion(func, first, second)
    {
        return None;
    }

    let mut carried = Vec::new();
    for inst in &func.blocks[second.header.0].insts {
        let (Some(result), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
            continue;
        };
        if result == second.induction {
            continue;
        }
        if !matches!(
            func.value(result).ty,
            Type::I1 | Type::I32 | Type::F32 | Type::Ptr(_)
        ) || incomings.len() != 2
        {
            return None;
        }
        let initial = unique_incoming(incomings, second.preheader)?;
        let next = unique_incoming(incomings, second.body)?;
        if value_defined_in_blocks(
            func,
            initial,
            &[second.header, second.body, second.preheader],
        ) {
            return None;
        }
        carried.push(CarriedPhi {
            original: result,
            initial,
            next,
        });
    }
    (!carried.is_empty()).then_some(FusionPlan {
        first: first.clone(),
        second: second.clone(),
        carried,
    })
}

fn memory_dependences_allow_fusion(
    func: &Function,
    first: &CanonicalLoop,
    second: &CanonicalLoop,
) -> bool {
    let mut produced_roots = HashSet::new();
    let mut accesses = HashMap::<ValueId, AccessRole>::new();

    for inst in &func.blocks[first.body.0].insts {
        match &inst.kind {
            InstKind::Nop
            | InstKind::Unary { .. }
            | InstKind::Binary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. } => {}
            InstKind::Gep { .. } => {
                let Some(result) = inst.result else {
                    return false;
                };
                let Some(root) = indexed_local_root(func, result, first.induction) else {
                    return false;
                };
                accesses.insert(result, AccessRole::Producer(root));
            }
            InstKind::Store { ptr, .. } => {
                let Some(root) = indexed_local_root(func, *ptr, first.induction) else {
                    return false;
                };
                produced_roots.insert(root);
            }
            InstKind::Phi { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Call { .. }
            | InstKind::MemZero { .. } => return false,
        }
    }

    let mut consumed_roots = HashSet::new();
    for inst in &func.blocks[second.body.0].insts {
        match &inst.kind {
            InstKind::Nop
            | InstKind::Unary { .. }
            | InstKind::Binary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. } => {}
            InstKind::Gep { .. } => {
                let Some(result) = inst.result else {
                    return false;
                };
                let Some(root) = indexed_local_root(func, result, second.induction) else {
                    return false;
                };
                accesses.insert(result, AccessRole::Consumer(root));
            }
            InstKind::Load { ptr } => {
                let Some(root) = indexed_local_root(func, *ptr, second.induction) else {
                    return false;
                };
                consumed_roots.insert(root);
            }
            InstKind::Phi { .. }
            | InstKind::Alloca { .. }
            | InstKind::Store { .. }
            | InstKind::Call { .. }
            | InstKind::MemZero { .. } => return false,
        }
    }
    if consumed_roots.is_empty() || !consumed_roots.is_subset(&produced_roots) {
        return false;
    }

    for root in &produced_roots {
        if !local_root_is_confined(func, *root, first, second, &accesses) {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy)]
enum AccessRole {
    Producer(ValueId),
    Consumer(ValueId),
}

fn indexed_local_root(func: &Function, address: ValueId, induction: ValueId) -> Option<ValueId> {
    let InstKind::Gep { base, indices } = defining_inst(func, address)? else {
        return None;
    };
    if indices.as_slice() != [induction] {
        return None;
    }
    matches!(
        defining_inst(func, *base),
        Some(InstKind::Alloca {
            ty: Type::Array { .. }
        })
    )
    .then_some(*base)
}

fn local_root_is_confined(
    func: &Function,
    root: ValueId,
    first: &CanonicalLoop,
    second: &CanonicalLoop,
    accesses: &HashMap<ValueId, AccessRole>,
) -> bool {
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let block_id = BlockId(block_idx);
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Gep { base, indices } if *base == root => {
                    let Some(result) = inst.result else {
                        return false;
                    };
                    let valid = (block_id == first.body
                        && indices.as_slice() == [first.induction]
                        && matches!(
                            accesses.get(&result),
                            Some(AccessRole::Producer(access_root)) if *access_root == root
                        ))
                        || (block_id == second.body
                            && indices.as_slice() == [second.induction]
                            && matches!(
                                accesses.get(&result),
                                Some(AccessRole::Consumer(access_root)) if *access_root == root
                            ));
                    if !valid {
                        return false;
                    }
                }
                kind if inst_operands(kind).contains(&root) => return false,
                _ => {}
            }
        }
        if terminator_operands(block.terminator.as_ref()).contains(&root) {
            return false;
        }
    }

    accesses.iter().all(|(address, role)| {
        let expected = match role {
            AccessRole::Producer(access_root) if *access_root == root => Some((first.body, true)),
            AccessRole::Consumer(access_root) if *access_root == root => Some((second.body, false)),
            _ => None,
        };
        let Some((owner, producer)) = expected else {
            return true;
        };
        func.blocks.iter().enumerate().all(|(block_idx, block)| {
            block.insts.iter().all(|inst| {
                if !inst_operands(&inst.kind).contains(address) {
                    return true;
                }
                BlockId(block_idx) == owner
                    && if producer {
                        matches!(inst.kind, InstKind::Store { ptr, .. } if ptr == *address)
                    } else {
                        matches!(inst.kind, InstKind::Load { ptr } if ptr == *address)
                    }
            }) && !terminator_operands(block.terminator.as_ref()).contains(address)
        })
    })
}

fn apply_plan(func: &mut Function, plan: FusionPlan) {
    let mut clone_map = ValueReplacements::from([(plan.second.induction, plan.first.induction)]);
    let mut fused_phis = Vec::new();
    let insertion = func.blocks[plan.first.header.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Phi { .. }))
        .count();

    for (offset, carried) in plan.carried.iter().enumerate() {
        let ty = func.value(carried.original).ty.clone();
        let fused = func
            .insert_inst(
                plan.first.header,
                insertion + offset,
                InstKind::Phi {
                    incomings: vec![(plan.first.preheader, carried.initial)],
                },
                Some(ty),
            )
            .expect("fused loop-carried value must produce a phi");
        clone_map.insert(carried.original, fused);
        fused_phis.push((fused, carried.next));
    }

    for inst in func.blocks[plan.second.body.0].insts.clone() {
        if matches!(inst.kind, InstKind::Nop) {
            continue;
        }
        let kind = clone_kind(&inst.kind, &clone_map);
        let ty = inst.result.map(|result| func.value(result).ty.clone());
        let cloned = func.append_inst(plan.first.body, kind, ty);
        if let (Some(original), Some(cloned)) = (inst.result, cloned) {
            clone_map.insert(original, cloned);
        }
    }

    for (fused, original_next) in fused_phis {
        let next = resolve_replacement(original_next, &clone_map);
        let ValueKind::Inst(block, inst_idx) = func.value(fused).kind else {
            unreachable!("inserted fused phi lost its instruction location");
        };
        let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
            unreachable!("inserted fused value stopped being a phi");
        };
        incomings.push((plan.first.body, next));
    }

    let outside_replacements = plan
        .carried
        .iter()
        .map(|carried| {
            (
                carried.original,
                *clone_map
                    .get(&carried.original)
                    .expect("carried phi must have a fused replacement"),
            )
        })
        .collect::<ValueReplacements>();
    rewrite_uses_outside_blocks(
        func,
        &[plan.second.header, plan.second.body],
        &outside_replacements,
    );

    let Terminator::Branch { else_target, .. } = func.blocks[plan.first.header.0]
        .terminator
        .as_mut()
        .expect("canonical first loop must remain terminated")
    else {
        unreachable!("canonical first loop terminator changed before fusion");
    };
    *else_target = plan.second.exit;
    for inst in &mut func.blocks[plan.second.exit.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        for (pred, _) in incomings {
            if *pred == plan.second.header {
                *pred = plan.first.header;
            }
        }
    }
}

fn clone_kind(kind: &InstKind, replacements: &ValueReplacements) -> InstKind {
    let map = |value| resolve_replacement(value, replacements);
    match kind {
        InstKind::Nop => InstKind::Nop,
        InstKind::Load { ptr } => InstKind::Load { ptr: map(*ptr) },
        InstKind::Unary { op, value } => InstKind::Unary {
            op: *op,
            value: map(*value),
        },
        InstKind::Binary { op, lhs, rhs } => InstKind::Binary {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Icmp { op, lhs, rhs } => InstKind::Icmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Fcmp { op, lhs, rhs } => InstKind::Fcmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Cast { op, value } => InstKind::Cast {
            op: *op,
            value: map(*value),
        },
        InstKind::Gep { base, indices } => InstKind::Gep {
            base: map(*base),
            indices: indices.iter().map(|index| map(*index)).collect(),
        },
        InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Store { .. }
        | InstKind::Call { .. }
        | InstKind::MemZero { .. } => {
            unreachable!("unsupported instruction reached adjacent loop fusion")
        }
    }
}

fn rewrite_uses_outside_blocks(
    func: &mut Function,
    excluded: &[BlockId],
    replacements: &ValueReplacements,
) {
    let excluded = excluded.iter().copied().collect::<HashSet<_>>();
    for (block_idx, block) in func.blocks.iter_mut().enumerate() {
        if excluded.contains(&BlockId(block_idx)) {
            continue;
        }
        for inst in &mut block.insts {
            rewrite_kind(&mut inst.kind, replacements);
        }
        if let Some(terminator) = &mut block.terminator {
            rewrite_terminator(terminator, replacements);
        }
    }
}

fn rewrite_kind(kind: &mut InstKind, replacements: &ValueReplacements) {
    let rewrite = |value: &mut ValueId| *value = resolve_replacement(*value, replacements);
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => {}
        InstKind::Phi { incomings } => {
            for (_, value) in incomings {
                rewrite(value);
            }
        }
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => rewrite(ptr),
        InstKind::Store { ptr, value } => {
            rewrite(ptr);
            rewrite(value);
        }
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => rewrite(value),
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => {
            rewrite(lhs);
            rewrite(rhs);
        }
        InstKind::Gep { base, indices } => {
            rewrite(base);
            for index in indices {
                rewrite(index);
            }
        }
        InstKind::Call { args, .. } => {
            for arg in args {
                rewrite(arg);
            }
        }
    }
}

fn rewrite_terminator(terminator: &mut Terminator, replacements: &ValueReplacements) {
    match terminator {
        Terminator::Return(Some(value)) | Terminator::Branch { cond: value, .. } => {
            *value = resolve_replacement(*value, replacements);
        }
        Terminator::Return(None) | Terminator::Jump(_) => {}
    }
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut values = vec![*base];
            values.extend(indices.iter().copied());
            values
        }
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: Option<&Terminator>) -> Vec<ValueId> {
    match terminator {
        Some(Terminator::Return(Some(value))) | Some(Terminator::Branch { cond: value, .. }) => {
            vec![*value]
        }
        Some(Terminator::Return(None) | Terminator::Jump(_)) | None => Vec::new(),
    }
}

fn active_instruction_count(insts: &[crate::ir::Inst]) -> usize {
    insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count()
}

fn unique_incoming(incomings: &[(BlockId, ValueId)], pred: BlockId) -> Option<ValueId> {
    let mut matching = incomings
        .iter()
        .filter_map(|(incoming_pred, value)| (*incoming_pred == pred).then_some(*value));
    let value = matching.next()?;
    matching.next().is_none().then_some(value)
}

fn value_defined_in_blocks(func: &Function, value: ValueId, blocks: &[BlockId]) -> bool {
    matches!(
        func.value(value).kind,
        ValueKind::Inst(owner, _) if blocks.contains(&owner)
    )
}

fn is_zero_i32(func: &Function, value: ValueId) -> bool {
    matches!(
        func.value(value).kind,
        ValueKind::Const(crate::ir::Const::Int(0) | crate::ir::Const::Zero(Type::I32))
    )
}

fn same_i32_constant(func: &Function, lhs: ValueId, rhs: ValueId) -> bool {
    is_zero_i32(func, lhs) && is_zero_i32(func, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Const};

    #[test]
    fn fuses_adjacent_elementwise_producer_and_reduction() {
        let mut func = Function::new("fuse", Type::I32);
        let bound = func.add_param("n", Type::I32);
        let first_header = func.add_block("first.header");
        let first_body = func.add_block("first.body");
        let bridge = func.add_block("bridge");
        let second_header = func.add_block("second.header");
        let second_body = func.add_block("second.body");
        let exit = func.add_block("exit");
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let array_ty = Type::Array {
            elem: Box::new(Type::I32),
            len: 32,
        };
        let array = func
            .append_inst(
                func.entry,
                InstKind::Alloca {
                    ty: array_ty.clone(),
                },
                Some(Type::Ptr(Box::new(array_ty))),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Jump(first_header));

        let first_i = func
            .append_inst(
                first_header,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero), (first_body, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let first_cond = func
            .append_inst(
                first_header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: first_i,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            first_header,
            Terminator::Branch {
                cond: first_cond,
                then_target: first_body,
                else_target: bridge,
            },
        );
        let produced = func
            .append_inst(
                first_body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: first_i,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        let first_address = func
            .append_inst(
                first_body,
                InstKind::Gep {
                    base: array,
                    indices: vec![first_i],
                },
                Some(Type::Ptr(Box::new(Type::I32))),
            )
            .unwrap();
        func.append_inst(
            first_body,
            InstKind::Store {
                ptr: first_address,
                value: produced,
            },
            None,
        );
        let first_next = func
            .append_inst(
                first_body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: first_i,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        set_backedge(&mut func, first_i, first_next);
        func.set_terminator(first_body, Terminator::Jump(first_header));
        func.set_terminator(bridge, Terminator::Jump(second_header));

        let second_i = func
            .append_inst(
                second_header,
                InstKind::Phi {
                    incomings: vec![(bridge, zero), (second_body, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let sum = func
            .append_inst(
                second_header,
                InstKind::Phi {
                    incomings: vec![(bridge, zero), (second_body, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let second_cond = func
            .append_inst(
                second_header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: second_i,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            second_header,
            Terminator::Branch {
                cond: second_cond,
                then_target: second_body,
                else_target: exit,
            },
        );
        let second_address = func
            .append_inst(
                second_body,
                InstKind::Gep {
                    base: array,
                    indices: vec![second_i],
                },
                Some(Type::Ptr(Box::new(Type::I32))),
            )
            .unwrap();
        let loaded = func
            .append_inst(
                second_body,
                InstKind::Load {
                    ptr: second_address,
                },
                Some(Type::I32),
            )
            .unwrap();
        let sum_next = func
            .append_inst(
                second_body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: sum,
                    rhs: loaded,
                },
                Some(Type::I32),
            )
            .unwrap();
        let second_next = func
            .append_inst(
                second_body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: second_i,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        set_backedge(&mut func, second_i, second_next);
        set_backedge(&mut func, sum, sum_next);
        func.set_terminator(second_body, Terminator::Jump(second_header));
        func.set_terminator(exit, Terminator::Return(Some(sum)));
        assert!(func.verify().is_ok());

        let mut module = Module::new();
        module.add_func(func);
        AdjacentLoopFusionPass::new().run(&mut module);
        let fused = &module.funcs[0];

        assert!(fused.verify().is_ok());
        let fused_exit = fused
            .blocks
            .iter()
            .position(|block| block.name == "exit")
            .map(BlockId)
            .unwrap();
        assert!(matches!(
            fused.blocks[first_header.0].terminator,
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) if then_target == first_body && else_target == fused_exit
        ));
        assert!(fused.blocks[first_body.0]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Load { .. })));
        assert!(!fused
            .blocks
            .iter()
            .any(|block| block.name == "second.header" || block.name == "second.body"));
    }

    fn set_backedge(func: &mut Function, phi: ValueId, value: ValueId) {
        let ValueKind::Inst(block, inst_idx) = func.value(phi).kind else {
            unreachable!();
        };
        let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
            unreachable!();
        };
        incomings[1].1 = value;
    }
}
