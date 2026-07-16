use crate::ir::{Function, InstKind, Type, ValueId};
use std::collections::HashMap;

const CALLEE_SAVED_PHI_REGS: [&str; 10] = [
    "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
];
const LEAF_PHI_REGS: [&str; 7] = ["x9", "x10", "x11", "x12", "x13", "x14", "x15"];

/// Conservatively keeps selected integer and pointer phi values in registers.
///
/// Every mapped value owns one callee-saved register for the whole function, so
/// values never interfere even across backedges or irreducible control flow.
pub(super) struct AArch64PhiRegs {
    regs: HashMap<ValueId, &'static str>,
    saved_regs: Vec<&'static str>,
}

impl AArch64PhiRegs {
    pub(super) fn new(func: &Function) -> Self {
        let is_leaf = !func.blocks.iter().any(|block| {
            block
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
        });
        let available_regs: &[&'static str] = if is_leaf {
            &LEAF_PHI_REGS
        } else {
            &CALLEE_SAVED_PHI_REGS
        };
        let mut regs = HashMap::new();
        let mut assigned_regs = Vec::new();

        for block in &func.blocks {
            for inst in &block.insts {
                if assigned_regs.len() == available_regs.len() {
                    break;
                }
                if !matches!(inst.kind, InstKind::Phi { .. }) {
                    continue;
                }
                let Some(result) = inst.result else {
                    continue;
                };
                if !matches!(func.value(result).ty, Type::I1 | Type::I32 | Type::Ptr(_)) {
                    continue;
                }

                let reg = available_regs[assigned_regs.len()];
                regs.insert(result, reg);
                assigned_regs.push(reg);
            }
        }

        let saved_regs = if is_leaf { Vec::new() } else { assigned_regs };
        Self { regs, saved_regs }
    }

    pub(super) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }

    pub(super) fn saved_regs(&self) -> &[&'static str] {
        &self.saved_regs
    }
}
