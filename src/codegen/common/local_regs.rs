use crate::ir::{BlockId, Function, InstKind, Terminator, Type, ValueId, ValueKind};
use std::collections::HashMap;

/// Assigns caller-saved scratch registers to values whose complete live range
/// is contained in one basic block and does not cross a call.
///
/// This deliberately avoids global liveness: each mapping is safe under any
/// CFG because the value is defined before every use in the same block.
pub(crate) struct IrLocalRegs {
    regs: HashMap<ValueId, &'static str>,
}

impl IrLocalRegs {
    pub(crate) fn new(
        func: &Function,
        available_regs: &'static [&'static str],
        allow_call_uses: bool,
    ) -> Self {
        if available_regs.is_empty() || func.values.len() > 16_384 {
            return Self {
                regs: HashMap::new(),
            };
        }

        let mut candidates = HashMap::<ValueId, Candidate>::new();
        for (value_idx, value) in func.values.iter().enumerate() {
            if !matches!(value.ty, Type::I1 | Type::I32 | Type::Ptr(_)) {
                continue;
            }
            let ValueKind::Inst(block, inst_idx) = value.kind else {
                continue;
            };
            let Some(inst) = func
                .blocks
                .get(block.0)
                .and_then(|owner| owner.insts.get(inst_idx))
            else {
                continue;
            };
            if inst.result != Some(ValueId(value_idx))
                || matches!(
                    inst.kind,
                    InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. }
                )
                || (!allow_call_uses && matches!(inst.kind, InstKind::Call { .. }))
            {
                continue;
            }
            candidates.insert(
                ValueId(value_idx),
                Candidate {
                    value: ValueId(value_idx),
                    block,
                    start: inst_idx,
                    end: inst_idx,
                    uses: 0,
                    valid: true,
                },
            );
        }

        for (block_idx, block) in func.blocks.iter().enumerate() {
            let owner = BlockId(block_idx);
            for (inst_idx, inst) in block.insts.iter().enumerate() {
                let unsupported_use = matches!(inst.kind, InstKind::Phi { .. })
                    || (!allow_call_uses && matches!(inst.kind, InstKind::Call { .. }));
                for operand in inst_operands(&inst.kind) {
                    record_use(&mut candidates, operand, owner, inst_idx, unsupported_use);
                }
            }
            if let Some(terminator) = &block.terminator {
                for operand in terminator_operands(terminator) {
                    if let Some(candidate) = candidates.get_mut(&operand) {
                        candidate.valid = false;
                    }
                }
            }
        }

        let calls = func
            .blocks
            .iter()
            .map(|block| {
                block
                    .insts
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, inst)| {
                        matches!(inst.kind, InstKind::Call { .. }).then_some(idx)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut by_block = vec![Vec::<Candidate>::new(); func.blocks.len()];
        for candidate in candidates.into_values() {
            if !candidate.valid || candidate.uses == 0 {
                continue;
            }
            let block_calls = &calls[candidate.block.0];
            let next_call = block_calls.partition_point(|call| *call <= candidate.start);
            if block_calls.get(next_call).is_some_and(|call| {
                *call < candidate.end || (!allow_call_uses && *call <= candidate.end)
            }) {
                continue;
            }
            by_block[candidate.block.0].push(candidate);
        }

        let mut regs = HashMap::new();
        for candidates in &mut by_block {
            candidates.sort_by_key(|candidate| (candidate.start, candidate.end, candidate.value.0));
            let mut free = available_regs.to_vec();
            let mut active = Vec::<Active>::new();
            for candidate in candidates {
                let mut idx = 0;
                while idx < active.len() {
                    // A definition is emitted after the current instruction's
                    // operands are consumed, so a register can be reused when
                    // the previous value's last use is this instruction.
                    if active[idx].end <= candidate.start {
                        free.push(active.remove(idx).reg);
                    } else {
                        idx += 1;
                    }
                }
                let Some(reg) = free.pop() else {
                    continue;
                };
                regs.insert(candidate.value, reg);
                active.push(Active {
                    end: candidate.end,
                    reg,
                });
                active.sort_by_key(|interval| interval.end);
            }
        }

        Self { regs }
    }

    pub(crate) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    value: ValueId,
    block: BlockId,
    start: usize,
    end: usize,
    uses: usize,
    valid: bool,
}

#[derive(Clone, Copy)]
struct Active {
    end: usize,
    reg: &'static str,
}

fn record_use(
    candidates: &mut HashMap<ValueId, Candidate>,
    value: ValueId,
    block: BlockId,
    inst_idx: usize,
    unsupported: bool,
) {
    let Some(candidate) = candidates.get_mut(&value) else {
        return;
    };
    if unsupported || candidate.block != block || inst_idx <= candidate.start {
        candidate.valid = false;
        return;
    }
    candidate.uses += 1;
    candidate.end = candidate.end.max(inst_idx);
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
            let mut operands = Vec::with_capacity(indices.len() + 1);
            operands.push(*base);
            operands.extend(indices.iter().copied());
            operands
        }
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
