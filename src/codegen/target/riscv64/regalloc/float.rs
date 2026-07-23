use super::{interference_graph, phi_affinities, weighted_use_scores};
use crate::codegen::common::ir_value_use_counts;
use crate::ir::{Function, InstKind, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const CALLER_SAVED_REGS: [&str; 11] = [
    "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11",
];
const CALLEE_SAVED_REGS: [&str; 12] = [
    "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
];
const MIN_CALLEE_SAVED_SCORE: usize = 16;

pub(in crate::codegen::target::riscv64) struct Riscv64FloatRegAlloc {
    regs: HashMap<ValueId, &'static str>,
    used_callee_saved: Vec<&'static str>,
}

impl Riscv64FloatRegAlloc {
    pub(super) fn new(func: &Function) -> Self {
        let use_counts = ir_value_use_counts(func);
        let candidate_set = func
            .values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                let value_id = ValueId(index);
                let supported_kind = match value.kind {
                    ValueKind::Param => true,
                    ValueKind::Inst(block, inst_idx) => !matches!(
                        func.blocks[block.0].insts[inst_idx].kind,
                        InstKind::Nop | InstKind::Alloca { .. }
                    ),
                    ValueKind::Const(_) | ValueKind::Global(_) => false,
                };
                (value.ty == Type::F32 && use_counts[index] != 0 && supported_kind)
                    .then_some(value_id)
            })
            .collect::<HashSet<_>>();
        if candidate_set.is_empty() {
            return Self::empty();
        }

        let scores = weighted_use_scores(func);
        let Some(analysis) = interference_graph(func, &candidate_set) else {
            return Self::empty();
        };
        let affinities = phi_affinities(func, &candidate_set, &analysis.interference);
        let mut candidates = candidate_set
            .into_iter()
            .map(|value| (value, scores[value.0]))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));

        let mut regs = HashMap::<ValueId, &'static str>::new();
        for (value, score) in candidates {
            let unavailable = analysis.interference[value.0]
                .iter()
                .filter_map(|neighbor| regs.get(neighbor).copied())
                .collect::<HashSet<_>>();
            let allowed = |reg: &'static str| {
                !unavailable.contains(reg)
                    && (CALLER_SAVED_REGS.contains(&reg)
                        && !analysis.live_across_calls.contains(&value)
                        || CALLEE_SAVED_REGS.contains(&reg)
                            && (analysis.live_across_calls.contains(&value)
                                || score >= MIN_CALLEE_SAVED_SCORE))
            };
            let preferred = affinities[value.0]
                .iter()
                .find_map(|neighbor| regs.get(neighbor).copied().filter(|reg| allowed(*reg)));
            let selected = preferred
                .or_else(|| CALLER_SAVED_REGS.iter().copied().find(|reg| allowed(*reg)))
                .or_else(|| CALLEE_SAVED_REGS.iter().copied().find(|reg| allowed(*reg)));
            if let Some(reg) = selected {
                regs.insert(value, reg);
            }
        }

        let occupied = regs.values().copied().collect::<HashSet<_>>();
        let used_callee_saved = CALLEE_SAVED_REGS
            .iter()
            .copied()
            .filter(|reg| occupied.contains(reg))
            .collect();
        Self {
            regs,
            used_callee_saved,
        }
    }

    fn empty() -> Self {
        Self {
            regs: HashMap::new(),
            used_callee_saved: Vec::new(),
        }
    }

    pub(super) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }

    pub(super) fn used_callee_saved(&self) -> &[&'static str] {
        &self.used_callee_saved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Const, Function, InstKind, Terminator, Type};

    #[test]
    fn assigns_block_local_float_to_caller_saved_register() {
        let mut func = Function::new("block_local_float", Type::F32);
        let param = func.add_param("value", Type::F32);
        let one = func.add_const(Const::Float(1.0f32.to_bits()));
        let result = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs: param,
                    rhs: one,
                },
                Some(Type::F32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(result)));

        let regs = Riscv64FloatRegAlloc::new(&func);

        assert!(regs.reg(result).is_some_and(|reg| reg.starts_with("ft")));
    }

    #[test]
    fn keeps_call_crossing_float_in_callee_saved_register() {
        let mut func = Function::new("call_crossing_float", Type::F32);
        let param = func.add_param("value", Type::F32);
        let one = func.add_const(Const::Float(1.0f32.to_bits()));
        func.append_inst(
            func.entry,
            InstKind::Call {
                name: "side_effect".into(),
                args: Vec::new(),
            },
            None,
        );
        let result = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs: param,
                    rhs: one,
                },
                Some(Type::F32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(result)));

        let regs = Riscv64FloatRegAlloc::new(&func);

        assert!(regs.reg(param).is_some_and(|reg| reg.starts_with("fs")));
        assert_eq!(regs.used_callee_saved(), &["fs0"]);
    }

    #[test]
    fn coalesces_non_interfering_float_phi_incoming() {
        let mut func = Function::new("float_phi", Type::F32);
        let initial = func.add_const(Const::Float(0.0f32.to_bits()));
        let one = func.add_const(Const::Float(1.0f32.to_bits()));
        let header = func.add_block("header");
        func.set_terminator(func.entry, Terminator::Jump(header));
        let phi = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(func.entry, initial)],
                },
                Some(Type::F32),
            )
            .unwrap();
        let next = func
            .append_inst(
                header,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs: phi,
                    rhs: one,
                },
                Some(Type::F32),
            )
            .unwrap();
        let InstKind::Phi { incomings } = &mut func.block_mut(header).insts[0].kind else {
            unreachable!();
        };
        incomings.push((header, next));
        func.set_terminator(header, Terminator::Jump(header));

        let regs = Riscv64FloatRegAlloc::new(&func);

        let phi_reg = regs.reg(phi).expect("phi should receive a register");
        assert_eq!(regs.reg(next), Some(phi_reg));
    }
}
