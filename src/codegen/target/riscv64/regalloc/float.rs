use super::{interference_graph, phi_affinities};
use crate::codegen::common::{ir_value_use_counts, weighted_use_scores};
use crate::codegen::Target;
use crate::ir::{Const, Function, InstKind, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const CALLER_SAVED_REGS: [&str; 11] = [
    "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11",
];
const CALLEE_SAVED_REGS: [&str; 12] = [
    "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
];
const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_CANDIDATES: usize = 512;
const MAX_LIVENESS_CELLS: usize = 262_144;

pub(in crate::codegen::target::riscv64) struct Riscv64FloatRegAlloc {
    regs: HashMap<ValueId, &'static str>,
    used_callee_saved: Vec<&'static str>,
    materialized: Vec<(ValueId, &'static str)>,
}

impl Riscv64FloatRegAlloc {
    pub(in crate::codegen::target::riscv64) fn new(func: &Function) -> Self {
        if func.blocks.len() > MAX_BLOCKS || func.values.len() > MAX_VALUES {
            return Self::empty();
        }

        let use_counts = ir_value_use_counts(func);
        let scores = weighted_use_scores(func);
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
                    ValueKind::Const(Const::Float(_) | Const::Zero(Type::F32)) => {
                        scores[index] >= 16
                    }
                    ValueKind::Const(_) | ValueKind::Global(_) => false,
                };
                (value.ty == Type::F32 && use_counts[index] != 0 && supported_kind)
                    .then_some(value_id)
            })
            .collect::<HashSet<_>>();
        if candidate_set.is_empty() {
            return Self::empty();
        }
        if candidate_set.len() > MAX_CANDIDATES
            || func.blocks.len().saturating_mul(candidate_set.len()) > MAX_LIVENESS_CELLS
        {
            return Self::empty();
        }

        let costs = Target::Riscv64.cost_model();
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
                            && costs.should_use_callee_saved_register(
                                score,
                                analysis.live_across_calls.contains(&value),
                            ))
            };
            let preferred = affinities[value.0]
                .iter()
                .find_map(|neighbor| regs.get(neighbor).copied().filter(|reg| allowed(reg)));
            let selected = preferred
                .or_else(|| CALLER_SAVED_REGS.iter().copied().find(|reg| allowed(reg)))
                .or_else(|| CALLEE_SAVED_REGS.iter().copied().find(|reg| allowed(reg)));
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
        let mut materialized = regs
            .iter()
            .filter_map(|(value, reg)| {
                matches!(func.value(*value).kind, ValueKind::Const(_)).then_some((*value, *reg))
            })
            .collect::<Vec<_>>();
        materialized.sort_by_key(|(value, _)| value.0);
        Self {
            regs,
            used_callee_saved,
            materialized,
        }
    }

    fn empty() -> Self {
        Self {
            regs: HashMap::new(),
            used_callee_saved: Vec::new(),
            materialized: Vec::new(),
        }
    }

    pub(in crate::codegen::target::riscv64) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }

    pub(in crate::codegen::target::riscv64) fn used_callee_saved(&self) -> &[&'static str] {
        &self.used_callee_saved
    }

    pub(in crate::codegen::target::riscv64) fn materialized(&self) -> &[(ValueId, &'static str)] {
        &self.materialized
    }
}
