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

#[cfg(test)]
mod tests {
    use super::AArch64PhiRegs;
    use crate::ir::{Function, InstKind, Type};

    #[test]
    fn uses_caller_saved_registers_for_leaf_functions() {
        let mut func = Function::new("leaf", Type::I32);
        let param = func.add_param("value", Type::I32);
        let loop_block = func.add_block("loop");
        let phi = func
            .append_inst(
                loop_block,
                InstKind::Phi {
                    incomings: vec![(func.entry, param)],
                },
                Some(Type::I32),
            )
            .unwrap();

        let allocation = AArch64PhiRegs::new(&func);

        assert_eq!(allocation.reg(phi), Some("x9"));
        assert!(allocation.saved_regs().is_empty());
    }

    #[test]
    fn uses_and_saves_callee_saved_registers_across_calls() {
        let mut func = Function::new("non_leaf", Type::I32);
        func.append_inst(
            func.entry,
            InstKind::Call {
                name: "callee".into(),
                args: Vec::new(),
            },
            None,
        );
        let param = func.add_param("value", Type::I32);
        let loop_block = func.add_block("loop");
        let phi = func
            .append_inst(
                loop_block,
                InstKind::Phi {
                    incomings: vec![(func.entry, param)],
                },
                Some(Type::I32),
            )
            .unwrap();

        let allocation = AArch64PhiRegs::new(&func);

        assert_eq!(allocation.reg(phi), Some("x19"));
        assert_eq!(allocation.saved_regs(), &["x19"]);
    }

    #[test]
    fn falls_back_to_stack_after_registers_are_exhausted() {
        let mut func = Function::new("many_phis", Type::I32);
        func.append_inst(
            func.entry,
            InstKind::Call {
                name: "callee".into(),
                args: Vec::new(),
            },
            None,
        );
        let param = func.add_param("value", Type::I32);
        let loop_block = func.add_block("loop");
        let phis = (0..11)
            .map(|_| {
                func.append_inst(
                    loop_block,
                    InstKind::Phi {
                        incomings: vec![(func.entry, param)],
                    },
                    Some(Type::I32),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let allocation = AArch64PhiRegs::new(&func);

        assert_eq!(allocation.reg(phis[0]), Some("x19"));
        assert_eq!(allocation.reg(phis[9]), Some("x28"));
        assert_eq!(allocation.reg(phis[10]), None);
        assert_eq!(allocation.saved_regs().len(), 10);
    }
}
