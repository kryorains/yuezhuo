use super::ModulePass;
use crate::ir::{BlockId, Const, Function, InstKind, Module, Terminator, ValueKind};

pub(super) struct SimplifyCfgPass;

impl SimplifyCfgPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for SimplifyCfgPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            simplify_function(func);
        }
    }
}

fn simplify_function(func: &mut Function) {
    // 先替换 terminator，并记录被删除的 CFG 边；最后再统一修 phi incoming。
    let mut removed_edges = Vec::new();

    for block_idx in 0..func.blocks.len() {
        let Some(terminator) = func.blocks[block_idx].terminator.clone() else {
            continue;
        };

        let (replacement, removed) = match terminator {
            // then/else 相同的条件跳转没有意义，直接改成无条件跳转。
            Terminator::Branch {
                cond: _,
                then_target,
                else_target,
            } if then_target == else_target => (Terminator::Jump(then_target), Vec::new()),
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => match const_bool(func, cond) {
                // 条件已经是常量时，删掉永远不会走到的那条边。
                Some(true) => (
                    Terminator::Jump(then_target),
                    vec![(BlockId(block_idx), else_target)],
                ),
                Some(false) => (
                    Terminator::Jump(else_target),
                    vec![(BlockId(block_idx), then_target)],
                ),
                None => continue,
            },
            _ => continue,
        };

        func.blocks[block_idx].terminator = Some(replacement);
        removed_edges.extend(removed);
    }

    for (pred, target) in removed_edges {
        remove_phi_incomings(func, pred, target);
    }
}

fn remove_phi_incomings(func: &mut Function, pred: BlockId, target: BlockId) {
    // CFG 边被删后，目标块 phi 中来自这个前驱的值也必须同步删除。
    for inst in &mut func.blocks[target.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            if !matches!(inst.kind, InstKind::Nop) {
                break;
            }
            continue;
        };
        incomings.retain(|(incoming_pred, _)| *incoming_pred != pred);
    }
}

fn const_bool(func: &Function, value: crate::ir::ValueId) -> Option<bool> {
    // 分支条件允许 bool/int/float 常量：非零视为 true。
    match &func.value(value).kind {
        ValueKind::Const(Const::Bool(value)) => Some(*value),
        ValueKind::Const(Const::Int(value)) => Some(*value != 0),
        ValueKind::Const(Const::Float(bits)) => Some(f32::from_bits(*bits) != 0.0),
        _ => None,
    }
}
