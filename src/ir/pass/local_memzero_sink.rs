use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::LoopInfo;
use super::ModulePass;
use crate::ir::{BlockId, Function, InstKind, Module, Terminator, ValueId, ValueKind};
use std::collections::HashSet;

/// Defers initialization of non-escaping local arrays until a chain of
/// side-effect-free early-return guards has accepted the work.
///
/// Frontends naturally emit aggregate initialization in the entry block.  In
/// recursive functions that test an empty range immediately afterwards, that
/// makes every leaf clear its local scratch arrays before returning.  This
/// pass splits the accepted edge and moves only those `MemZero`s whose complete
/// alias set is proven unused before, and dominated by, that edge.
pub(super) struct LocalMemzeroSinkPass;

impl LocalMemzeroSinkPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for LocalMemzeroSinkPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            sink_entry_memzeros(func);
        }
    }
}

#[derive(Clone, Copy)]
struct Memzero {
    inst_index: usize,
    ptr: ValueId,
    bytes: usize,
}

fn sink_entry_memzeros(func: &mut Function) {
    if func.blocks.len() > 4096 || func.values.len() > 32_768 {
        return;
    }
    let entry = func.entry;
    let memzeros = func.blocks[entry.0]
        .insts
        .iter()
        .enumerate()
        .filter_map(|(inst_index, inst)| {
            let InstKind::MemZero { ptr, bytes } = inst.kind else {
                return None;
            };
            is_direct_local_alloca(func, ptr).then_some(Memzero {
                inst_index,
                ptr,
                bytes,
            })
        })
        .collect::<Vec<_>>();
    if memzeros.is_empty() {
        return;
    }

    let Some((edge_from, accepted)) = accepted_edge_after_early_returns(func) else {
        return;
    };
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let movable = memzeros
        .into_iter()
        .filter(|memzero| {
            let aliases = pointer_aliases(func, memzero.ptr);
            actual_use_blocks(func, &aliases, entry, memzero.inst_index).is_some_and(|uses| {
                !uses.is_empty()
                    && uses.into_iter().all(|use_block| {
                        !dom.is_reachable(use_block) || dom.dominates(accepted, use_block)
                    })
            })
        })
        .collect::<Vec<_>>();
    if movable.is_empty() {
        return;
    }

    let accepted_is_loop_header = LoopInfo::new(&cfg, &dom)
        .loops()
        .iter()
        .any(|natural_loop| natural_loop.header == accepted);
    if !accepted_is_loop_header {
        let insertion_index = func.blocks[accepted.0]
            .insts
            .iter()
            .take_while(|inst| matches!(inst.kind, InstKind::Phi { .. }))
            .count();
        for (offset, memzero) in movable.iter().enumerate() {
            func.insert_inst(
                accepted,
                insertion_index + offset,
                InstKind::MemZero {
                    ptr: memzero.ptr,
                    bytes: memzero.bytes,
                },
                None,
            );
            func.blocks[entry.0].insts[memzero.inst_index].kind = InstKind::Nop;
        }
        if let Err(errors) = func.verify() {
            panic!(
                "local memzero sinking produced invalid IR in {}: {:?}",
                func.name, errors
            );
        }
        return;
    }

    let deferred = func.add_block("deferred.memzero");
    for memzero in &movable {
        func.append_inst(
            deferred,
            InstKind::MemZero {
                ptr: memzero.ptr,
                bytes: memzero.bytes,
            },
            None,
        );
        func.blocks[entry.0].insts[memzero.inst_index].kind = InstKind::Nop;
    }
    func.set_terminator(deferred, Terminator::Jump(accepted));
    replace_successor(
        func.blocks[edge_from.0]
            .terminator
            .as_mut()
            .expect("guard block must retain its terminator"),
        accepted,
        deferred,
    );
    for inst in &mut func.blocks[accepted.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            break;
        };
        for (pred, _) in incomings {
            if *pred == edge_from {
                *pred = deferred;
            }
        }
    }

    if let Err(errors) = func.verify() {
        panic!(
            "local memzero sinking produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn accepted_edge_after_early_returns(func: &Function) -> Option<(BlockId, BlockId)> {
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_headers = LoopInfo::new(&cfg, &dom)
        .loops()
        .iter()
        .map(|natural_loop| natural_loop.header)
        .collect::<HashSet<_>>();
    let mut current = func.entry;
    let mut seen = HashSet::new();
    let mut accepted_edge = None;
    while seen.insert(current) {
        let Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) = func.blocks[current.0].terminator.as_ref()
        else {
            return accepted_edge;
        };
        let then_returns = is_direct_return(func, *then_target);
        let else_returns = is_direct_return(func, *else_target);
        let next = match (then_returns, else_returns) {
            (true, false) => *else_target,
            (false, true) => *then_target,
            _ => return accepted_edge,
        };
        accepted_edge = Some((current, next));
        if loop_headers.contains(&next) {
            return accepted_edge;
        }
        current = next;
    }
    accepted_edge
}

fn is_direct_return(func: &Function, block: BlockId) -> bool {
    matches!(func.blocks[block.0].terminator, Some(Terminator::Return(_)))
}

fn is_direct_local_alloca(func: &Function, value: ValueId) -> bool {
    let ValueKind::Inst(block, inst_index) = func.value(value).kind else {
        return false;
    };
    matches!(
        func.blocks[block.0].insts[inst_index].kind,
        InstKind::Alloca { .. }
    )
}

fn pointer_aliases(func: &Function, root: ValueId) -> HashSet<ValueId> {
    let mut aliases = HashSet::from([root]);
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for inst in &block.insts {
                let Some(result) = inst.result else {
                    continue;
                };
                let derives_alias = match &inst.kind {
                    InstKind::Gep { base, .. } => aliases.contains(base),
                    InstKind::Phi { incomings } => {
                        incomings.iter().any(|(_, value)| aliases.contains(value))
                    }
                    _ => false,
                };
                if derives_alias {
                    changed |= aliases.insert(result);
                }
            }
        }
        if !changed {
            return aliases;
        }
    }
}

fn actual_use_blocks(
    func: &Function,
    aliases: &HashSet<ValueId>,
    memzero_block: BlockId,
    memzero_index: usize,
) -> Option<Vec<BlockId>> {
    let mut uses = Vec::new();
    for (block_index, block) in func.blocks.iter().enumerate() {
        for (inst_index, inst) in block.insts.iter().enumerate() {
            if BlockId(block_index) == memzero_block && inst_index == memzero_index {
                continue;
            }
            let derives_alias = matches!(
                &inst.kind,
                InstKind::Gep { base, .. } if aliases.contains(base)
            ) || matches!(
                &inst.kind,
                InstKind::Phi { incomings }
                    if incomings.iter().any(|(_, value)| aliases.contains(value))
            );
            if derives_alias {
                continue;
            }
            if inst_operands(&inst.kind)
                .into_iter()
                .any(|operand| aliases.contains(&operand))
            {
                uses.push(BlockId(block_index));
            }
        }
        if terminator_operand(block.terminator.as_ref())
            .is_some_and(|operand| aliases.contains(&operand))
        {
            uses.push(BlockId(block_index));
        }
    }
    Some(uses)
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

fn terminator_operand(terminator: Option<&Terminator>) -> Option<ValueId> {
    match terminator {
        Some(Terminator::Return(value)) => *value,
        Some(Terminator::Branch { cond, .. }) => Some(*cond),
        Some(Terminator::Jump(_)) | None => None,
    }
}

fn replace_successor(terminator: &mut Terminator, old: BlockId, new: BlockId) {
    match terminator {
        Terminator::Jump(target) => {
            if *target == old {
                *target = new;
            }
        }
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => {
            if *then_target == old {
                *then_target = new;
            }
            if *else_target == old {
                *else_target = new;
            }
        }
        Terminator::Return(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    #[test]
    fn defers_local_array_zeroing_past_early_return_guards() {
        let mut module = lower(
            r#"
int work(int n) {
    int scratch[16] = {};
    if (n < 0) return 0;
    if (n < 2) return 1;
    int i = 0;
    while (i < n) {
        scratch[i % 16] = i;
        i = i + 1;
    }
    return scratch[0];
}
"#,
        );
        LocalMemzeroSinkPass::new().run(&mut module);
        let work = module
            .funcs
            .iter()
            .find(|func| func.name == "work")
            .unwrap();
        assert!(work.blocks[work.entry.0]
            .insts
            .iter()
            .all(|inst| !matches!(inst.kind, InstKind::MemZero { .. })));
        assert!(work.blocks.iter().enumerate().any(|(block_index, block)| {
            BlockId(block_index) != work.entry
                && block
                    .insts
                    .iter()
                    .any(|inst| matches!(inst.kind, InstKind::MemZero { bytes: 64, .. }))
        }));
        assert!(work.verify().is_ok());
    }

    #[test]
    fn keeps_zeroing_before_an_escaping_use() {
        let mut module = lower(
            r#"
int work(int n) {
    int scratch[16] = {};
    getarray(scratch);
    if (n < 0) return 0;
    return scratch[0];
}
"#,
        );
        LocalMemzeroSinkPass::new().run(&mut module);
        let work = module
            .funcs
            .iter()
            .find(|func| func.name == "work")
            .unwrap();
        assert!(work.blocks[work.entry.0]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::MemZero { .. })));
        assert!(work
            .blocks
            .iter()
            .all(|block| block.name != "deferred.memzero"));
    }

    fn lower(source: &str) -> Module {
        let mut parser = Parser::new(source);
        let program = parser.parse_program();
        lower_program(&program).expect("test source must lower")
    }
}
