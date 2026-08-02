use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::{HashMap, HashSet};

const SPECIALIZED_DIVISOR: i32 = 3;
const MAX_LOOP_BLOCKS: usize = 12;
const MAX_LOOP_INSTRUCTIONS: usize = 192;

/// Versions a compact loop around a loop-invariant signed divisor.
///
/// The original loop remains the fallback.  The cloned path is entered only
/// when the divisor is exactly three, allowing ordinary constant-division
/// lowering to replace a hardware divide in every iteration.
pub(super) struct LoopDivisionSpecializePass;

impl LoopDivisionSpecializePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LoopDivisionSpecializePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            let Some(candidate) = find_candidate(func) else {
                continue;
            };
            apply_candidate(func, candidate);
            if let Err(errors) = func.verify() {
                panic!(
                    "loop division specialization produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

#[derive(Clone)]
struct Candidate {
    preheader: BlockId,
    header: BlockId,
    exit_pred: BlockId,
    exit: BlockId,
    blocks: HashSet<BlockId>,
    divisor: ValueId,
}

fn find_candidate(func: &Function) -> Option<Candidate> {
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);
    let mut loops = loop_info.loops().iter().collect::<Vec<_>>();
    loops.sort_by_key(|natural_loop| natural_loop.blocks.len());

    for natural_loop in loops {
        let (Some(preheader), [(exit_pred, exit)]) = (
            natural_loop.dedicated_preheader,
            natural_loop.exit_edges.as_slice(),
        ) else {
            continue;
        };
        if natural_loop.blocks.len() > MAX_LOOP_BLOCKS
            || natural_loop
                .blocks
                .iter()
                .map(|block| func.block(*block).insts.len())
                .sum::<usize>()
                > MAX_LOOP_INSTRUCTIONS
            || func.block(preheader).terminator != Some(Terminator::Jump(natural_loop.header))
            || has_unsupported_live_out(func, natural_loop, *exit_pred, *exit)
        {
            continue;
        }

        let mut divisors = natural_loop
            .blocks
            .iter()
            .flat_map(|block| &func.block(*block).insts)
            .filter_map(|inst| match inst.kind {
                InstKind::Binary {
                    op: BinaryOp::Idiv,
                    rhs,
                    ..
                } if is_loop_invariant(func, natural_loop, rhs)
                    && !matches!(func.value(rhs).kind, ValueKind::Const(_)) =>
                {
                    Some(rhs)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        divisors.sort_by_key(|value| value.0);
        divisors.dedup();
        let [divisor] = divisors.as_slice() else {
            continue;
        };
        return Some(Candidate {
            preheader,
            header: natural_loop.header,
            exit_pred: *exit_pred,
            exit: *exit,
            blocks: natural_loop.blocks.clone(),
            divisor: *divisor,
        });
    }
    None
}

fn is_loop_invariant(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    !matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Inst(block, _)) if natural_loop.blocks.contains(block)
    )
}

fn has_unsupported_live_out(
    func: &Function,
    natural_loop: &NaturalLoop,
    exit_pred: BlockId,
    exit: BlockId,
) -> bool {
    let loop_values = natural_loop
        .blocks
        .iter()
        .flat_map(|block| &func.block(*block).insts)
        .filter_map(|inst| inst.result)
        .collect::<HashSet<_>>();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let block_id = BlockId(block_idx);
        if natural_loop.blocks.contains(&block_id) {
            continue;
        }
        for inst in &block.insts {
            if block_id == exit {
                if let InstKind::Phi { incomings } = &inst.kind {
                    if incomings
                        .iter()
                        .all(|(pred, value)| *pred != exit_pred || loop_values.contains(value))
                    {
                        continue;
                    }
                }
            }
            if inst_operands(&inst.kind)
                .iter()
                .any(|value| loop_values.contains(value))
            {
                return true;
            }
        }
        if block
            .terminator
            .as_ref()
            .into_iter()
            .flat_map(terminator_operands)
            .any(|value| loop_values.contains(&value))
        {
            return true;
        }
    }
    false
}

fn apply_candidate(func: &mut Function, candidate: Candidate) {
    let mut original_blocks = candidate.blocks.iter().copied().collect::<Vec<_>>();
    original_blocks.sort_by_key(|block| block.0);
    let snapshots = original_blocks
        .iter()
        .map(|block| (*block, func.block(*block).clone()))
        .collect::<HashMap<_, _>>();
    let block_map = original_blocks
        .iter()
        .map(|block| (*block, func.add_block("div.specialized")))
        .collect::<HashMap<_, _>>();
    let mut value_map = HashMap::<ValueId, ValueId>::new();
    let mut locations = Vec::new();

    for source_block in &original_blocks {
        let cloned_block = block_map[source_block];
        for (inst_idx, inst) in snapshots[source_block].insts.iter().enumerate() {
            let result_ty = inst.result.map(|result| func.value(result).ty.clone());
            let cloned_result = func.append_inst(cloned_block, InstKind::Nop, result_ty);
            if let (Some(source), Some(cloned)) = (inst.result, cloned_result) {
                value_map.insert(source, cloned);
            }
            locations.push((*source_block, inst_idx, cloned_block));
        }
    }

    let three = get_or_add_i32_const(func, SPECIALIZED_DIVISOR);
    for (source_block, inst_idx, cloned_block) in locations {
        let source = &snapshots[&source_block].insts[inst_idx];
        let mut kind = remap_inst(&source.kind, &block_map, &value_map);
        if let InstKind::Binary {
            op: BinaryOp::Idiv,
            rhs,
            ..
        } = &mut kind
        {
            if *rhs == candidate.divisor {
                *rhs = three;
            }
        }
        func.block_mut(cloned_block).insts[inst_idx].kind = kind;
    }

    for source_block in &original_blocks {
        let terminator = remap_terminator(
            snapshots[source_block]
                .terminator
                .as_ref()
                .expect("natural-loop block must be terminated"),
            &block_map,
            &value_map,
        );
        func.set_terminator(block_map[source_block], terminator);
    }

    let cloned_exit_pred = block_map[&candidate.exit_pred];
    for inst in &mut func.block_mut(candidate.exit).insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            break;
        };
        let additions = incomings
            .iter()
            .filter(|(pred, _)| *pred == candidate.exit_pred)
            .map(|(_, value)| {
                (
                    cloned_exit_pred,
                    value_map.get(value).copied().unwrap_or(*value),
                )
            })
            .collect::<Vec<_>>();
        incomings.extend(additions);
    }

    let condition = func
        .append_inst(
            candidate.preheader,
            InstKind::Icmp {
                op: CmpOp::Eq,
                lhs: candidate.divisor,
                rhs: three,
            },
            Some(Type::I1),
        )
        .expect("division specialization guard must return a boolean");
    func.block_mut(candidate.preheader).terminator = Some(Terminator::Branch {
        cond: condition,
        then_target: block_map[&candidate.header],
        else_target: candidate.header,
    });
}

fn remap_inst(
    kind: &InstKind,
    blocks: &HashMap<BlockId, BlockId>,
    values: &HashMap<ValueId, ValueId>,
) -> InstKind {
    let value = |value: ValueId| values.get(&value).copied().unwrap_or(value);
    let block = |block: BlockId| blocks.get(&block).copied().unwrap_or(block);
    match kind {
        InstKind::Nop => InstKind::Nop,
        InstKind::Phi { incomings } => InstKind::Phi {
            incomings: incomings
                .iter()
                .map(|(pred, incoming)| (block(*pred), value(*incoming)))
                .collect(),
        },
        InstKind::Alloca { ty } => InstKind::Alloca { ty: ty.clone() },
        InstKind::Load { ptr } => InstKind::Load { ptr: value(*ptr) },
        InstKind::Store { ptr, value: stored } => InstKind::Store {
            ptr: value(*ptr),
            value: value(*stored),
        },
        InstKind::MemZero { ptr, bytes } => InstKind::MemZero {
            ptr: value(*ptr),
            bytes: *bytes,
        },
        InstKind::Unary { op, value: operand } => InstKind::Unary {
            op: *op,
            value: value(*operand),
        },
        InstKind::Binary { op, lhs, rhs } => InstKind::Binary {
            op: *op,
            lhs: value(*lhs),
            rhs: value(*rhs),
        },
        InstKind::Icmp { op, lhs, rhs } => InstKind::Icmp {
            op: *op,
            lhs: value(*lhs),
            rhs: value(*rhs),
        },
        InstKind::Fcmp { op, lhs, rhs } => InstKind::Fcmp {
            op: *op,
            lhs: value(*lhs),
            rhs: value(*rhs),
        },
        InstKind::Cast { op, value: operand } => InstKind::Cast {
            op: *op,
            value: value(*operand),
        },
        InstKind::Gep { base, indices } => InstKind::Gep {
            base: value(*base),
            indices: indices.iter().map(|index| value(*index)).collect(),
        },
        InstKind::Call { name, args } => InstKind::Call {
            name: name.clone(),
            args: args.iter().map(|arg| value(*arg)).collect(),
        },
    }
}

fn remap_terminator(
    terminator: &Terminator,
    blocks: &HashMap<BlockId, BlockId>,
    values: &HashMap<ValueId, ValueId>,
) -> Terminator {
    let block = |block: BlockId| blocks.get(&block).copied().unwrap_or(block);
    let value = |value: ValueId| values.get(&value).copied().unwrap_or(value);
    match terminator {
        Terminator::Return(result) => Terminator::Return(result.map(value)),
        Terminator::Jump(target) => Terminator::Jump(block(*target)),
        Terminator::Branch {
            cond,
            then_target,
            else_target,
        } => Terminator::Branch {
            cond: value(*cond),
            then_target: block(*then_target),
            else_target: block(*else_target),
        },
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
        InstKind::Gep { base, indices } => std::iter::once(*base)
            .chain(indices.iter().copied())
            .collect(),
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}

fn get_or_add_i32_const(func: &mut Function, expected: i32) -> ValueId {
    func.values
        .iter()
        .position(|value| {
            value.ty == Type::I32
                && matches!(value.kind, ValueKind::Const(Const::Int(actual)) if actual == expected)
        })
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(Const::Int(expected)))
}

#[cfg(test)]
mod tests {
    use super::super::{DcePass, ScalarPromotePass};
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    #[test]
    fn versions_an_invariant_division_without_removing_the_fallback() {
        let source = r#"
            int values[128];
            void fill_divided(int n, int divisor) {
                int i = 0;
                while (i < n) {
                    values[i] = i / divisor;
                    i = i + 1;
                }
                return;
            }
        "#;
        let mut module = lower_program(&Parser::new(source).parse_program()).unwrap();
        ScalarPromotePass::new().run(&mut module);
        DcePass::new().run(&mut module);
        LoopDivisionSpecializePass::new().run(&mut module);
        let func = &module.funcs[0];
        assert!(func
            .blocks
            .iter()
            .any(|block| block.name == "div.specialized"));
        assert!(func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .any(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Binary {
                        op: BinaryOp::Idiv,
                        rhs,
                        ..
                    } if matches!(func.value(rhs).kind, ValueKind::Const(Const::Int(3)))
                )
            }));
        assert!(func.verify().is_ok());
    }
}
