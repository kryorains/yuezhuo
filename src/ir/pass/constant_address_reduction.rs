use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};

const MAX_BLOCKS: usize = 2048;
const MAX_VALUES: usize = 16_384;
const MAX_INSTRUCTIONS: usize = 65_536;

/// Collapses `n` unit increments of one loop-invariant scalar address into one
/// wrapping add by `n`.
///
/// The matched loop retains its original header test. The first taken
/// iteration performs the aggregate update and feeds the bound to the counter
/// phi, so the next header test exits through the original CFG edge.
pub(super) struct ConstantAddressReductionPass;

impl ConstantAddressReductionPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for ConstantAddressReductionPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            reduce_function(func);
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    header: BlockId,
    body: BlockId,
    header_phi_inst: usize,
    update_inst: usize,
    loaded: ValueId,
    bound: ValueId,
}

fn reduce_function(func: &mut Function) {
    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    if func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || instruction_count.is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let candidates = loop_info
        .loops()
        .iter()
        .filter_map(|natural_loop| match_candidate(func, natural_loop))
        .collect::<Vec<_>>();

    for candidate in candidates {
        func.blocks[candidate.body.0].insts[candidate.update_inst].kind = InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs: candidate.loaded,
            rhs: candidate.bound,
        };
        let InstKind::Phi { incomings } =
            &mut func.blocks[candidate.header.0].insts[candidate.header_phi_inst].kind
        else {
            unreachable!("matched counter phi must remain a phi");
        };
        let incoming = incomings
            .iter_mut()
            .find(|(pred, _)| *pred == candidate.body)
            .expect("matched counter phi must retain its backedge");
        incoming.1 = candidate.bound;
    }

    if let Err(errors) = func.verify() {
        panic!(
            "constant-address reduction produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn match_candidate(func: &Function, natural_loop: &NaturalLoop) -> Option<Candidate> {
    if natural_loop.blocks.len() != 2 {
        return None;
    }
    let header = natural_loop.header;
    let body = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    if body == header
        || func.block(body).terminator != Some(Terminator::Jump(header))
        || natural_loop.dedicated_preheader.is_none()
    {
        return None;
    }

    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.block(header).terminator.as_ref()?
    else {
        return None;
    };
    if *then_target != body || *else_target != exit {
        return None;
    }

    let mut phi = None;
    let mut condition = None;
    for (inst_idx, inst) in func.block(header).insts.iter().enumerate() {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { .. } if phi.is_none() => {
                phi = Some((inst_idx, inst.result?));
            }
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs,
                rhs,
            } if inst.result == Some(*cond) && condition.is_none() => {
                condition = Some((*lhs, *rhs));
            }
            _ => return None,
        }
    }
    let (header_phi_inst, counter) = phi?;
    let (condition_counter, bound) = condition?;
    if condition_counter != counter
        || func.value(counter).ty != Type::I32
        || func.value(bound).ty != Type::I32
        || !loop_invariant_value(func, natural_loop, bound, 0)
    {
        return None;
    }
    let induction = analyze_i32_induction(func, natural_loop, counter)?;
    if induction.step != 1 || const_i32(func, induction.initial) != Some(0) {
        return None;
    }

    let mut load = None;
    let mut update = None;
    let mut store = None;
    for (inst_idx, inst) in func.block(body).insts.iter().enumerate() {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Gep { .. } if inst.result.is_some() => {}
            InstKind::Load { ptr } if load.is_none() => {
                let result = inst.result?;
                if !loop_invariant_value(func, natural_loop, *ptr, 0) {
                    return None;
                }
                load = Some((*ptr, result, inst_idx));
            }
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            } if inst.result == Some(induction.next)
                && ((*lhs == counter && const_i32(func, *rhs) == Some(1))
                    || (*rhs == counter && const_i32(func, *lhs) == Some(1))) => {}
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            } if update.is_none() => {
                let (_, loaded, load_idx) = load?;
                if inst_idx <= load_idx
                    || !((*lhs == loaded && const_i32(func, *rhs) == Some(1))
                        || (*rhs == loaded && const_i32(func, *lhs) == Some(1)))
                {
                    return None;
                }
                update = Some((inst.result?, inst_idx, loaded));
            }
            InstKind::Store { ptr, value } if store.is_none() => {
                let (loaded_ptr, _, load_idx) = load?;
                let (updated, update_idx, _) = update?;
                if *ptr != loaded_ptr
                    || *value != updated
                    || inst_idx <= load_idx
                    || inst_idx <= update_idx
                {
                    return None;
                }
                store = Some(inst_idx);
            }
            _ => return None,
        }
    }
    let (_, update_inst, loaded) = update?;
    store?;

    Some(Candidate {
        header,
        body,
        header_phi_inst,
        update_inst,
        loaded,
        bound,
    })
}

fn loop_invariant_value(
    func: &Function,
    natural_loop: &NaturalLoop,
    value: ValueId,
    depth: usize,
) -> bool {
    if depth >= 16 {
        return false;
    }
    match &func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(owner, inst_idx) if !natural_loop.blocks.contains(owner) => func
            .blocks
            .get(owner.0)
            .and_then(|block| block.insts.get(*inst_idx))
            .is_some(),
        ValueKind::Inst(owner, inst_idx) => {
            let Some(inst) = func
                .blocks
                .get(owner.0)
                .and_then(|block| block.insts.get(*inst_idx))
            else {
                return false;
            };
            matches!(
                &inst.kind,
                InstKind::Gep { base, indices }
                    if loop_invariant_value(func, natural_loop, *base, depth + 1)
                        && indices.iter().all(|index| {
                            loop_invariant_value(func, natural_loop, *index, depth + 1)
                        })
            )
        }
    }
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        ValueKind::Const(Const::Zero(Type::I32)) => Some(0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_a_unit_increment_of_one_invariant_address() {
        let mut func = Function::new("count", Type::I32);
        let bound = func.add_param("bound", Type::I32);
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let slot = func
            .append_inst(
                func.entry,
                InstKind::Alloca { ty: Type::I32 },
                Some(Type::Ptr(Box::new(Type::I32))),
            )
            .unwrap();
        let ptr = func
            .append_inst(
                func.entry,
                InstKind::Gep {
                    base: slot,
                    indices: vec![zero],
                },
                Some(Type::Ptr(Box::new(Type::I32))),
            )
            .unwrap();
        func.append_inst(func.entry, InstKind::Store { ptr, value: zero }, None);
        let header = func.add_block("header");
        let body = func.add_block("body");
        let exit = func.add_block("exit");
        func.set_terminator(func.entry, Terminator::Jump(header));
        let counter = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(func.entry, zero), (body, zero)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let condition = func
            .append_inst(
                header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: counter,
                    rhs: bound,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: body,
                else_target: exit,
            },
        );
        let loaded = func
            .append_inst(body, InstKind::Load { ptr }, Some(Type::I32))
            .unwrap();
        let updated = func
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: loaded,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.append_inst(
            body,
            InstKind::Store {
                ptr,
                value: updated,
            },
            None,
        );
        let next = func
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: counter,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        let InstKind::Phi { incomings } = &mut func.blocks[header.0].insts[0].kind else {
            unreachable!();
        };
        incomings[1].1 = next;
        func.set_terminator(body, Terminator::Jump(header));
        let returned = func
            .append_inst(exit, InstKind::Load { ptr }, Some(Type::I32))
            .unwrap();
        func.set_terminator(exit, Terminator::Return(Some(returned)));

        reduce_function(&mut func);

        assert!(matches!(
            func.blocks[body.0].insts[1].kind,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            } if lhs == loaded && rhs == bound
        ));
        let InstKind::Phi { incomings } = &func.blocks[header.0].insts[0].kind else {
            unreachable!();
        };
        assert!(incomings.contains(&(body, bound)));
        assert!(func.verify().is_ok());
    }
}
