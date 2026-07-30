use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::LoopInfo;
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{BlockId, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind};
use std::collections::{BTreeMap, HashSet};

/// Keeps a directly accessed scalar global in SSA form for one canonical loop.
///
/// Whole-function global localization must conservatively stop at calls.  A
/// hot loop can still be isolated from those calls when it has a dedicated
/// preheader, one latch and one header exit.  This pass handles that narrower
/// case and flushes the scalar on the exit edge.
pub(super) struct RegionalGlobalScalarPass;

impl RegionalGlobalScalarPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RegionalGlobalScalarPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            while let Some(candidate) = find_candidate(func) {
                promote_candidate(func, candidate);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    name: String,
    value_ty: Type,
    global: ValueId,
    preheader: BlockId,
    header: BlockId,
    latch: BlockId,
    exit: BlockId,
    load_index: usize,
    load_result: ValueId,
    store_index: usize,
    stored_value: ValueId,
}

#[derive(Default)]
struct Accesses {
    loads: Vec<(BlockId, usize, ValueId, ValueId, Type)>,
    stores: Vec<(BlockId, usize, ValueId, ValueId, Type)>,
}

fn find_candidate(func: &Function) -> Option<Candidate> {
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);

    for natural_loop in loop_info.loops() {
        let Some(preheader) = natural_loop.dedicated_preheader else {
            continue;
        };
        let Some(latch) = natural_loop.unique_latch() else {
            continue;
        };
        if latch == natural_loop.header {
            continue;
        }
        let [(exit_from, exit)] = natural_loop.exit_edges.as_slice() else {
            continue;
        };
        if *exit_from != natural_loop.header {
            continue;
        }
        if natural_loop.blocks.iter().any(|block| {
            func.blocks[block.0]
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, InstKind::Call { .. } | InstKind::MemZero { .. }))
        }) {
            continue;
        }

        let mut accesses = BTreeMap::<String, Accesses>::new();
        for block in &natural_loop.blocks {
            for (inst_index, inst) in func.blocks[block.0].insts.iter().enumerate() {
                match &inst.kind {
                    InstKind::Load { ptr } => {
                        let Some((name, value_ty)) = scalar_global(func, *ptr) else {
                            continue;
                        };
                        let Some(result) = inst.result else {
                            continue;
                        };
                        accesses
                            .entry(name.to_owned())
                            .or_default()
                            .loads
                            .push((*block, inst_index, result, *ptr, value_ty));
                    }
                    InstKind::Store { ptr, value } => {
                        let Some((name, value_ty)) = scalar_global(func, *ptr) else {
                            continue;
                        };
                        accesses
                            .entry(name.to_owned())
                            .or_default()
                            .stores
                            .push((*block, inst_index, *value, *ptr, value_ty));
                    }
                    _ => {}
                }
            }
        }

        for (name, accesses) in accesses {
            let [(load_block, load_index, load_result, load_global, load_ty)] =
                accesses.loads.as_slice()
            else {
                continue;
            };
            let [(store_block, store_index, stored_value, _store_global, store_ty)] =
                accesses.stores.as_slice()
            else {
                continue;
            };
            if *load_block != latch
                || *store_block != latch
                || load_index >= store_index
                || load_ty != store_ty
                || !loop_memory_is_disjoint(func, natural_loop, &name)
            {
                continue;
            }

            return Some(Candidate {
                name,
                value_ty: load_ty.clone(),
                global: *load_global,
                preheader,
                header: natural_loop.header,
                latch,
                exit: *exit,
                load_index: *load_index,
                load_result: *load_result,
                store_index: *store_index,
                stored_value: *stored_value,
            });
        }
    }
    None
}

fn scalar_global(func: &Function, value: ValueId) -> Option<(&str, Type)> {
    let value = func.values.get(value.0)?;
    let ValueKind::Global(name) = &value.kind else {
        return None;
    };
    let Type::Ptr(pointee) = &value.ty else {
        return None;
    };
    matches!(pointee.as_ref(), Type::I1 | Type::I32 | Type::F32)
        .then(|| (name.as_str(), pointee.as_ref().clone()))
}

fn loop_memory_is_disjoint(
    func: &Function,
    natural_loop: &super::loop_analysis::NaturalLoop,
    candidate: &str,
) -> bool {
    natural_loop.blocks.iter().all(|block| {
        func.blocks[block.0].insts.iter().all(|inst| {
            let ptr = match &inst.kind {
                InstKind::Load { ptr } | InstKind::Store { ptr, .. } => *ptr,
                InstKind::MemZero { .. } | InstKind::Call { .. } => return false,
                _ => return true,
            };
            match scalar_global(func, ptr) {
                Some((name, _)) if name == candidate => true,
                _ => pointer_is_proven_disjoint(func, ptr, candidate),
            }
        })
    })
}

fn pointer_is_proven_disjoint(func: &Function, pointer: ValueId, candidate: &str) -> bool {
    let mut worklist = vec![pointer];
    let mut visited = HashSet::new();
    let mut has_distinct_root = false;

    while let Some(value_id) = worklist.pop() {
        if !visited.insert(value_id) {
            continue;
        }
        let Some(value) = func.values.get(value_id.0) else {
            return false;
        };
        match &value.kind {
            ValueKind::Global(name) => {
                if name == candidate {
                    return false;
                }
                has_distinct_root = true;
            }
            ValueKind::Inst(block, inst_index) => {
                let Some(inst) = func
                    .blocks
                    .get(block.0)
                    .and_then(|block| block.insts.get(*inst_index))
                else {
                    return false;
                };
                match &inst.kind {
                    InstKind::Alloca { .. } => has_distinct_root = true,
                    InstKind::Gep { base, .. } => worklist.push(*base),
                    InstKind::Phi { incomings } => {
                        worklist.extend(incomings.iter().map(|(_, value)| *value));
                    }
                    _ => return false,
                }
            }
            ValueKind::Param | ValueKind::Const(_) => return false,
        }
    }
    has_distinct_root
}

fn promote_candidate(func: &mut Function, candidate: Candidate) {
    let initial = func
        .append_inst(
            candidate.preheader,
            InstKind::Load {
                ptr: candidate.global,
            },
            Some(candidate.value_ty.clone()),
        )
        .expect("regional global load must produce a value");
    let phi_index = func.blocks[candidate.header.0]
        .insts
        .iter()
        .take_while(|inst| matches!(inst.kind, InstKind::Phi { .. }))
        .count();
    let carried = func
        .insert_inst(
            candidate.header,
            phi_index,
            InstKind::Phi {
                incomings: vec![
                    (candidate.preheader, initial),
                    (candidate.latch, candidate.stored_value),
                ],
            },
            Some(candidate.value_ty),
        )
        .expect("regional global phi must produce a value");

    let replacements = ValueReplacements::from([(candidate.load_result, carried)]);
    rewrite_function_uses(func, &replacements);
    let load = &mut func.blocks[candidate.latch.0].insts[candidate.load_index];
    debug_assert_eq!(load.result, Some(candidate.load_result));
    load.result = None;
    load.kind = InstKind::Nop;
    let store = &mut func.blocks[candidate.latch.0].insts[candidate.store_index];
    debug_assert!(matches!(store.kind, InstKind::Store { .. }));
    store.kind = InstKind::Nop;

    let flush = func.add_block(format!("regional.{}.flush", candidate.name));
    func.append_inst(
        flush,
        InstKind::Store {
            ptr: candidate.global,
            value: carried,
        },
        None,
    );
    func.set_terminator(flush, Terminator::Jump(candidate.exit));

    let terminator = func.blocks[candidate.header.0]
        .terminator
        .as_mut()
        .expect("loop header must have an exit terminator");
    match terminator {
        Terminator::Jump(target) => {
            debug_assert_eq!(*target, candidate.exit);
            *target = flush;
        }
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => {
            if *then_target == candidate.exit {
                *then_target = flush;
            }
            if *else_target == candidate.exit {
                *else_target = flush;
            }
        }
        Terminator::Return(_) => unreachable!("loop exit edge cannot be a return"),
    }
    for inst in &mut func.blocks[candidate.exit.0].insts {
        let InstKind::Phi { incomings } = &mut inst.kind else {
            continue;
        };
        for (predecessor, _) in incomings {
            if *predecessor == candidate.header {
                *predecessor = flush;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Const, Function, Global, InstKind, Module, Terminator};

    #[test]
    fn promotes_scalar_global_only_across_canonical_loop() {
        let mut module = Module::new();
        module.globals.push(Global {
            name: "sum".into(),
            ty: Type::I32,
            is_const: false,
            init: Some(Const::Int(0)),
        });
        let mut func = Function::new("main", Type::I32);
        let preheader = func.entry;
        let header = func.add_block("loop.header");
        let latch = func.add_block("loop.body");
        let exit = func.add_block("exit");
        let global = func.add_global_ref("sum", Type::Ptr(Box::new(Type::I32)));
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let keep_going = func.add_const(Const::Bool(true));
        func.set_terminator(preheader, Terminator::Jump(header));
        let _induction = func
            .append_inst(
                header,
                InstKind::Phi {
                    incomings: vec![(preheader, zero), (latch, one)],
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(
            header,
            Terminator::Branch {
                cond: keep_going,
                then_target: latch,
                else_target: exit,
            },
        );
        let old = func
            .append_inst(latch, InstKind::Load { ptr: global }, Some(Type::I32))
            .unwrap();
        let next = func
            .append_inst(
                latch,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: old,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.append_inst(
            latch,
            InstKind::Store {
                ptr: global,
                value: next,
            },
            None,
        );
        func.set_terminator(latch, Terminator::Jump(header));
        let final_value = func
            .append_inst(exit, InstKind::Load { ptr: global }, Some(Type::I32))
            .unwrap();
        func.set_terminator(exit, Terminator::Return(Some(final_value)));
        module.funcs.push(func);

        RegionalGlobalScalarPass::new().run(&mut module);

        let func = &module.funcs[0];
        assert!(func.verify().is_ok());
        assert_eq!(func.blocks.len(), 5);
        assert!(func.blocks[latch.0]
            .insts
            .iter()
            .all(|inst| !matches!(inst.kind, InstKind::Load { ptr } if ptr == global)));
        assert!(func.blocks[latch.0]
            .insts
            .iter()
            .all(|inst| !matches!(inst.kind, InstKind::Store { ptr, .. } if ptr == global)));
        assert!(func.blocks[4]
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Store { ptr, .. } if ptr == global)));
    }
}
