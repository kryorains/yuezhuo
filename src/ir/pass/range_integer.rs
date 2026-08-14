use super::dominators::{ControlFlowGraph, Dominators};
use super::util::{
    const_i32, defining_inst, get_or_add_i32_const, rewrite_function_uses, ValueReplacements,
};
use super::ModulePass;
use crate::ir::int_range::{collect_function_return_ranges, collect_value_ranges};
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, ValueId, ValueKind,
};

pub(super) struct RangeIntegerSimplifyPass;

impl RangeIntegerSimplifyPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RangeIntegerSimplifyPass {
    fn run(&mut self, module: &mut Module) {
        let return_ranges = collect_function_return_ranges(module);
        for func in &mut module.funcs {
            fold_dominated_integer_conditions(func);
            reduce_path_exact_power_of_two_divisions(func);
            let ranges = collect_value_ranges(func, &return_ranges);
            let mut replacements = ValueReplacements::new();
            let mut removed = Vec::new();
            let mut zero_results = Vec::new();

            for (block_idx, block) in func.blocks.iter().enumerate() {
                for (inst_idx, inst) in block.insts.iter().enumerate() {
                    let (Some(result), InstKind::Binary { op, lhs, rhs }) =
                        (inst.result, &inst.kind)
                    else {
                        continue;
                    };
                    if !matches!(op, BinaryOp::Idiv | BinaryOp::Imod) {
                        continue;
                    }
                    let Some(divisor) = const_i32(func, *rhs).filter(|divisor| *divisor != 0)
                    else {
                        continue;
                    };
                    let Some(range) = ranges[lhs.0] else {
                        continue;
                    };
                    let bound = i64::from(divisor).abs();
                    if range.min > -bound && range.max < bound {
                        if *op == BinaryOp::Imod {
                            replacements.insert(result, *lhs);
                        } else {
                            zero_results.push(result);
                        }
                        removed.push((block_idx, inst_idx));
                    }
                }
            }

            if !zero_results.is_empty() {
                let zero = get_or_add_i32_const(func, 0);
                replacements.extend(zero_results.into_iter().map(|result| (result, zero)));
            }
            if removed.is_empty() {
                continue;
            }
            rewrite_function_uses(func, &replacements);
            for (block_idx, inst_idx) in removed {
                let inst = &mut func.blocks[block_idx].insts[inst_idx];
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
            if let Err(errors) = func.verify() {
                panic!(
                    "range integer simplification produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct DominatingComparison {
    region: BlockId,
    op: CmpOp,
    lhs: ValueId,
    rhs: ValueId,
    truth: bool,
}

/// Propagates the truth value of an integer comparison along a branch edge
/// that dominates a later comparison. This is the basic control-flow fact
/// used by GCC's value-range propagation and LLVM's LazyValueInfo: it is
/// independent of source names and only reasons about SSA identity,
/// constants, comparison equivalence, and dominance.
fn fold_dominated_integer_conditions(func: &mut Function) {
    const MAX_BLOCKS: usize = 4096;
    const MAX_FACTS: usize = 8192;
    const MAX_PROOF_WORK: usize = 262_144;

    if func.blocks.len() > MAX_BLOCKS {
        return;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let mut facts = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let Some(Terminator::Branch {
            cond,
            then_target,
            else_target,
        }) = block.terminator.as_ref()
        else {
            continue;
        };
        if then_target == else_target {
            continue;
        }
        let Some((op, lhs, rhs)) = integer_comparison(func, *cond) else {
            continue;
        };
        let source = BlockId(block_idx);
        // A successor block can stand in for a branch edge only when that edge
        // is its sole entry. In particular, a loop header may dominate the
        // branch that jumps back to it, but the backedge's truth value does not
        // hold for all entries to the header.
        if *then_target != source && cfg.preds[then_target.0] == vec![source] {
            facts.push(DominatingComparison {
                region: *then_target,
                op,
                lhs,
                rhs,
                truth: true,
            });
        }
        if *else_target != source && cfg.preds[else_target.0] == vec![source] {
            facts.push(DominatingComparison {
                region: *else_target,
                op,
                lhs,
                rhs,
                truth: false,
            });
        }
        if facts.len() > MAX_FACTS {
            return;
        }
    }
    if facts.len().saturating_mul(func.blocks.len()) > MAX_PROOF_WORK {
        return;
    }

    let mut folds = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let Some(Terminator::Branch { cond, .. }) = block.terminator.as_ref() else {
            continue;
        };
        let Some((op, lhs, rhs)) = integer_comparison(func, *cond) else {
            continue;
        };
        let owner = BlockId(block_idx);
        if let Some(truth) = facts.iter().find_map(|fact| {
            (dom.dominates(fact.region, owner)
                && equivalent_comparison(func, fact.op, fact.lhs, fact.rhs, op, lhs, rhs))
            .then_some(fact.truth)
        }) {
            folds.push((owner, truth));
        }
    }

    if folds.is_empty() {
        return;
    }
    let false_value = folds
        .iter()
        .any(|(_, truth)| !truth)
        .then(|| get_or_add_bool_const(func, false));
    let true_value = folds
        .iter()
        .any(|(_, truth)| *truth)
        .then(|| get_or_add_bool_const(func, true));
    for (block, truth) in folds {
        let Some(Terminator::Branch { cond, .. }) = &mut func.blocks[block.0].terminator else {
            continue;
        };
        *cond = if truth {
            true_value.expect("true comparison constant was created")
        } else {
            false_value.expect("false comparison constant was created")
        };
    }
}

fn integer_comparison(func: &Function, condition: ValueId) -> Option<(CmpOp, ValueId, ValueId)> {
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, condition)? else {
        return None;
    };
    Some((*op, *lhs, *rhs))
}

fn equivalent_comparison(
    func: &Function,
    fact_op: CmpOp,
    fact_lhs: ValueId,
    fact_rhs: ValueId,
    candidate_op: CmpOp,
    candidate_lhs: ValueId,
    candidate_rhs: ValueId,
) -> bool {
    (fact_op == candidate_op
        && equivalent_operand(func, fact_lhs, candidate_lhs)
        && equivalent_operand(func, fact_rhs, candidate_rhs))
        || (reverse_comparison(fact_op) == candidate_op
            && equivalent_operand(func, fact_lhs, candidate_rhs)
            && equivalent_operand(func, fact_rhs, candidate_lhs))
}

fn equivalent_operand(func: &Function, lhs: ValueId, rhs: ValueId) -> bool {
    lhs == rhs || const_i32(func, lhs).is_some_and(|value| const_i32(func, rhs) == Some(value))
}

fn reverse_comparison(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}

fn get_or_add_bool_const(func: &mut Function, expected: bool) -> ValueId {
    func.values
        .iter()
        .position(
            |value| matches!(value.kind, ValueKind::Const(Const::Bool(value)) if value == expected),
        )
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(Const::Bool(expected)))
}

/// Lowers a signed division to an arithmetic shift when a dominating branch
/// proves the dividend is an exact multiple of the positive power-of-two
/// divisor.  This is the path-sensitive known-bits reasoning used by GCC's
/// value-range passes and LLVM's value tracking/InstCombine; it depends only
/// on local SSA values and dominance, never on a source-level loop shape.
fn reduce_path_exact_power_of_two_divisions(func: &mut Function) {
    const MAX_BLOCKS: usize = 4096;
    const MAX_PROOF_WORK: usize = 262_144;

    if func.blocks.len() > MAX_BLOCKS {
        return;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let mut proofs = Vec::<(BlockId, ValueId, u32)>::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let Some(Terminator::Branch {
            cond,
            then_target,
            else_target,
        }) = block.terminator.as_ref()
        else {
            continue;
        };
        let Some((value, zero_on_true)) = zero_test(func, *cond) else {
            continue;
        };
        let Some((dividend, known_zero_mask)) = low_zero_mask(func, value) else {
            continue;
        };
        let taken = if zero_on_true {
            *then_target
        } else {
            *else_target
        };
        if taken.0 == block_idx {
            continue;
        }
        proofs.push((taken, dividend, known_zero_mask));
    }
    if proofs.len().saturating_mul(func.blocks.len()) > MAX_PROOF_WORK {
        return;
    }

    let mut rewrites = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            let InstKind::Binary {
                op: BinaryOp::Idiv,
                lhs,
                rhs,
            } = inst.kind
            else {
                continue;
            };
            let Some(divisor) = const_i32(func, rhs)
                .filter(|divisor| *divisor > 1 && (*divisor as u32).is_power_of_two())
            else {
                continue;
            };
            let required_mask = (divisor as u32) - 1;
            if proofs.iter().any(|(taken, dividend, known_zero_mask)| {
                *dividend == lhs
                    && (*known_zero_mask & required_mask) == required_mask
                    && dom.dominates(*taken, BlockId(block_idx))
            }) {
                rewrites.push((block_idx, inst_idx, lhs, divisor.trailing_zeros() as i32));
            }
        }
    }
    for (block_idx, inst_idx, lhs, shift) in rewrites {
        let rhs = get_or_add_i32_const(func, shift);
        func.blocks[block_idx].insts[inst_idx].kind = InstKind::Binary {
            op: BinaryOp::Iashr,
            lhs,
            rhs,
        };
    }
}

fn zero_test(func: &Function, condition: ValueId) -> Option<(ValueId, bool)> {
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, condition)? else {
        return None;
    };
    let (value, zero_on_true) = if const_i32(func, *rhs) == Some(0) {
        (*lhs, *op == CmpOp::Eq)
    } else if const_i32(func, *lhs) == Some(0) {
        (*rhs, *op == CmpOp::Eq)
    } else {
        return None;
    };
    matches!(op, CmpOp::Eq | CmpOp::Ne).then_some((value, zero_on_true))
}

fn low_zero_mask(func: &Function, value: ValueId) -> Option<(ValueId, u32)> {
    match defining_inst(func, value)? {
        InstKind::Binary {
            op: BinaryOp::Iand,
            lhs,
            rhs,
        } => {
            if let Some(mask) = const_i32(func, *rhs) {
                Some((*lhs, mask as u32))
            } else {
                const_i32(func, *lhs).map(|mask| (*rhs, mask as u32))
            }
        }
        InstKind::Binary {
            op: BinaryOp::Imod,
            lhs,
            rhs,
        } => {
            let divisor = const_i32(func, *rhs)?;
            (divisor > 1 && (divisor as u32).is_power_of_two())
                .then_some((*lhs, (divisor as u32) - 1))
        }
        _ => None,
    }
}
