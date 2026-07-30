use crate::codegen::common::ir_value_use_counts;
use crate::codegen::target::riscv64::regalloc::{
    interference_graph, phi_affinities, weighted_use_scores,
};
use crate::ir::{Function, InstKind, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

// AAPCS64 makes v16-v31 caller-saved. The low 64 bits of v8-v15 are
// callee-saved, which is sufficient for the scalar f32 values allocated here.
const CALLER_SAVED_REGS: [&str; 16] = [
    "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28",
    "s29", "s30", "s31",
];
const CALLEE_SAVED_REGS: [&str; 8] = ["s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15"];
const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_CANDIDATES: usize = 512;
const MAX_LIVENESS_CELLS: usize = 262_144;
const CALLEE_SAVED_SAVE_RESTORE_COST: usize = 16;

pub(super) struct AArch64FloatRegs {
    regs: HashMap<ValueId, &'static str>,
    used_callee_saved: Vec<&'static str>,
}

impl AArch64FloatRegs {
    pub(super) fn new(func: &Function) -> Self {
        if func.blocks.len() > MAX_BLOCKS || func.values.len() > MAX_VALUES {
            return Self::empty();
        }

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
        if candidate_set.len() > MAX_CANDIDATES
            || func.blocks.len().saturating_mul(candidate_set.len()) > MAX_LIVENESS_CELLS
        {
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
                                || score >= CALLEE_SAVED_SAVE_RESTORE_COST))
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
