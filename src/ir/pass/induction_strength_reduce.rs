use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo};
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::HashSet;

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_INSTRUCTIONS: usize = 32_768;
const MAX_CANDIDATES: usize = 128;

/// Replaces repeated `induction * loop_invariant` expressions with an i32
/// recurrence. Both the original multiplication and the recurrence use
/// wrapping i32 arithmetic, so the transform is exact even when values wrap.
pub(super) struct InductionStrengthReducePass;

impl InductionStrengthReducePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for InductionStrengthReducePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            strength_reduce_function(func);
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    target: ValueId,
    initial_induction: ValueId,
    step: i32,
    factor: ValueId,
    offset: Option<ValueId>,
    preheader: BlockId,
    header: BlockId,
    latch: BlockId,
}

fn strength_reduce_function(func: &mut Function) {
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
    let loops = LoopInfo::new(&cfg, &dom);
    let mut candidates = Vec::new();
    let mut claimed = HashSet::new();

    for natural_loop in loops.loops() {
        let (Some(preheader), Some(latch)) = (
            natural_loop.unique_entering_pred,
            natural_loop.unique_latch(),
        ) else {
            continue;
        };
        // A dedicated preheader is unnecessary here: the inserted i32
        // arithmetic is pure and non-trapping, so it may execute on another
        // successor of the unique entering block when the loop is skipped.
        for inst in &func.blocks[natural_loop.header.0].insts {
            let Some(phi) = inst
                .result
                .filter(|_| matches!(inst.kind, InstKind::Phi { .. }))
            else {
                continue;
            };
            let Some(induction) = analyze_i32_induction(func, natural_loop, phi) else {
                continue;
            };
            for block in &natural_loop.blocks {
                for inst in &func.blocks[block.0].insts {
                    let (
                        Some(product),
                        InstKind::Binary {
                            op: BinaryOp::Imul,
                            lhs,
                            rhs,
                        },
                    ) = (inst.result, &inst.kind)
                    else {
                        continue;
                    };
                    // The replacement adds one recurrence update on every
                    // backedge. Require the original product to execute on
                    // every iteration as well, so the transform cannot turn
                    // a conditional multiply into unconditional work.
                    if !dom.dominates(*block, latch) {
                        continue;
                    }
                    let factor = if *lhs == induction.phi {
                        *rhs
                    } else if *rhs == induction.phi {
                        *lhs
                    } else {
                        continue;
                    };
                    if claimed.contains(&product)
                        || !is_loop_invariant(func, natural_loop, factor)
                        || !value_available_at(func, &dom, factor, preheader)
                        || !value_available_at(func, &dom, induction.initial, preheader)
                        || used_outside_loop(func, natural_loop, product)
                    {
                        continue;
                    }
                    let (target, offset) =
                        affine_extension(func, natural_loop, product).unwrap_or((product, None));
                    if claimed.contains(&target) || used_outside_loop(func, natural_loop, target) {
                        continue;
                    }
                    claimed.insert(product);
                    claimed.insert(target);
                    candidates.push(Candidate {
                        target,
                        initial_induction: induction.initial,
                        step: induction.step,
                        factor,
                        offset,
                        preheader,
                        header: natural_loop.header,
                        latch,
                    });
                    if candidates.len() == MAX_CANDIDATES {
                        break;
                    }
                }
                if candidates.len() == MAX_CANDIDATES {
                    break;
                }
            }
            if candidates.len() == MAX_CANDIDATES {
                break;
            }
        }
        if candidates.len() == MAX_CANDIDATES {
            break;
        }
    }

    let projected_values = candidates.len().saturating_mul(5);
    if projected_values == 0
        || func
            .values
            .len()
            .checked_add(projected_values)
            .is_none_or(|total| total > MAX_VALUES)
        || instruction_count
            .and_then(|count| count.checked_add(projected_values))
            .is_none_or(|total| total > MAX_INSTRUCTIONS)
    {
        return;
    }

    let mut replacements = ValueReplacements::new();
    for candidate in candidates {
        let initial = func
            .append_inst(
                candidate.preheader,
                InstKind::Binary {
                    op: BinaryOp::Imul,
                    lhs: candidate.initial_induction,
                    rhs: candidate.factor,
                },
                Some(Type::I32),
            )
            .expect("an initial product produces i32");
        let initial = if let Some(offset) = candidate.offset {
            func.append_inst(
                candidate.preheader,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: initial,
                    rhs: offset,
                },
                Some(Type::I32),
            )
            .expect("an affine initial value produces i32")
        } else {
            initial
        };
        let step_constant = func.add_const(crate::ir::Const::Int(candidate.step));
        let delta = func
            .append_inst(
                candidate.preheader,
                InstKind::Binary {
                    op: BinaryOp::Imul,
                    lhs: candidate.factor,
                    rhs: step_constant,
                },
                Some(Type::I32),
            )
            .expect("a recurrence delta produces i32");
        let phi_index = func.blocks[candidate.header.0]
            .insts
            .iter()
            .take_while(|inst| matches!(inst.kind, InstKind::Phi { .. }))
            .count();
        let recurrence = func
            .insert_inst(
                candidate.header,
                phi_index,
                InstKind::Phi {
                    incomings: vec![(candidate.preheader, initial)],
                },
                Some(Type::I32),
            )
            .expect("a recurrence phi produces i32");
        let next = func
            .append_inst(
                candidate.latch,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: recurrence,
                    rhs: delta,
                },
                Some(Type::I32),
            )
            .expect("a recurrence update produces i32");
        let ValueKind::Inst(header, index) = func.value(recurrence).kind else {
            unreachable!();
        };
        let InstKind::Phi { incomings } = &mut func.blocks[header.0].insts[index].kind else {
            unreachable!();
        };
        incomings.push((candidate.latch, next));
        replacements.insert(candidate.target, recurrence);
    }
    rewrite_function_uses(func, &replacements);
    for product in claimed {
        let ValueKind::Inst(block, index) = func.value(product).kind else {
            continue;
        };
        func.blocks[block.0].insts[index].result = None;
        func.blocks[block.0].insts[index].kind = InstKind::Nop;
    }
    if let Err(errors) = func.verify() {
        panic!(
            "integer induction strength reduction produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn affine_extension(
    func: &Function,
    natural_loop: &super::loop_analysis::NaturalLoop,
    product: ValueId,
) -> Option<(ValueId, Option<ValueId>)> {
    let mut found = None;
    for block in &natural_loop.blocks {
        for inst in &func.blocks[block.0].insts {
            let (
                Some(result),
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs,
                    rhs,
                },
            ) = (inst.result, &inst.kind)
            else {
                continue;
            };
            let offset = if *lhs == product {
                *rhs
            } else if *rhs == product {
                *lhs
            } else {
                continue;
            };
            if !is_loop_invariant(func, natural_loop, offset) || found.is_some() {
                return None;
            }
            found = Some((result, Some(offset)));
        }
    }
    let (target, offset) = found?;
    let mut uses = 0usize;
    for block in &func.blocks {
        for inst in &block.insts {
            uses += inst_operands(&inst.kind)
                .into_iter()
                .filter(|operand| *operand == product)
                .count();
        }
        if let Some(terminator) = &block.terminator {
            uses += terminator_operands(terminator)
                .into_iter()
                .filter(|operand| *operand == product)
                .count();
        }
    }
    (uses == 1).then_some((target, offset))
}

fn value_available_at(
    func: &Function,
    dom: &Dominators,
    value: ValueId,
    destination: BlockId,
) -> bool {
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(block, _) => block == destination || dom.dominates(block, destination),
    }
}

fn is_loop_invariant(
    func: &Function,
    natural_loop: &super::loop_analysis::NaturalLoop,
    value: ValueId,
) -> bool {
    match func.value(value).kind {
        ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_) => true,
        ValueKind::Inst(block, _) => !natural_loop.blocks.contains(&block),
    }
}

fn used_outside_loop(
    func: &Function,
    natural_loop: &super::loop_analysis::NaturalLoop,
    value: ValueId,
) -> bool {
    for (block_idx, block) in func.blocks.iter().enumerate() {
        if natural_loop.blocks.contains(&BlockId(block_idx)) {
            continue;
        }
        for inst in &block.insts {
            if inst_operands(&inst.kind).contains(&value) {
                return true;
            }
        }
        if let Some(terminator) = &block.terminator {
            if terminator_operands(terminator).contains(&value) {
                return true;
            }
        }
    }
    false
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => std::iter::once(*base)
            .chain(indices.iter().copied())
            .collect(),
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) | Terminator::Branch { cond: value, .. } => vec![*value],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}
