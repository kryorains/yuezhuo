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

/// Hoists repeated reads of one directly addressed scalar global out of a
/// call-free loop when every loop write is proven to target another object.
pub(super) struct RegionalInvariantGlobalLoadPass;

impl RegionalGlobalScalarPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl RegionalInvariantGlobalLoadPass {
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

impl ModulePass for RegionalInvariantGlobalLoadPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            for _ in 0..MAX_READONLY_HOISTS_PER_FUNCTION {
                let Some(candidate) = find_readonly_candidate(func) else {
                    break;
                };
                hoist_readonly_candidate(func, candidate);
            }
        }
    }
}

const MAX_READONLY_HOISTS_PER_FUNCTION: usize = 16;

#[derive(Debug, Clone)]
struct ReadonlyCandidate {
    name: String,
    value_ty: Type,
    global: ValueId,
    preheader: BlockId,
    loop_blocks: HashSet<BlockId>,
}

fn find_readonly_candidate(func: &Function) -> Option<ReadonlyCandidate> {
    if func.blocks.len() > 1024 || func.values.len() > 16_384 {
        return None;
    }
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let loop_info = LoopInfo::new(&cfg, &dom);

    for natural_loop in loop_info.loops() {
        let Some(preheader) = natural_loop.dedicated_preheader else {
            continue;
        };
        if natural_loop.blocks.iter().any(|block| {
            func.blocks[block.0].insts.iter().any(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Call { .. } | InstKind::MemZero { .. } | InstKind::MemCopy { .. }
                )
            })
        }) {
            continue;
        }

        let mut loads = BTreeMap::<String, (ValueId, Type, usize)>::new();
        let mut stored_globals = HashSet::new();
        for block in &natural_loop.blocks {
            for inst in &func.blocks[block.0].insts {
                match inst.kind {
                    InstKind::Load { ptr } => {
                        let Some((name, value_ty)) = scalar_global(func, ptr) else {
                            continue;
                        };
                        let entry = loads.entry(name.to_owned()).or_insert((ptr, value_ty, 0));
                        entry.2 += 1;
                    }
                    InstKind::Store { ptr, .. } => {
                        if let Some((name, _)) = scalar_global(func, ptr) {
                            stored_globals.insert(name.to_owned());
                        }
                    }
                    _ => {}
                }
            }
        }

        for (name, (global, value_ty, load_count)) in loads {
            if load_count < 2
                || stored_globals.contains(&name)
                || !loop_memory_is_disjoint(func, natural_loop, &name)
            {
                continue;
            }
            return Some(ReadonlyCandidate {
                name,
                value_ty,
                global,
                preheader,
                loop_blocks: natural_loop.blocks.clone(),
            });
        }
    }
    None
}

fn hoist_readonly_candidate(func: &mut Function, candidate: ReadonlyCandidate) {
    let invariant = func
        .append_inst(
            candidate.preheader,
            InstKind::Load {
                ptr: candidate.global,
            },
            Some(candidate.value_ty),
        )
        .expect("a hoisted scalar-global load must produce a value");
    let mut replacements = ValueReplacements::new();
    for block in &candidate.loop_blocks {
        for inst in &mut func.blocks[block.0].insts {
            let (Some(result), InstKind::Load { ptr }) = (inst.result, &inst.kind) else {
                continue;
            };
            if scalar_global_name_by_values(&func.values, *ptr) == Some(candidate.name.as_str()) {
                replacements.insert(result, invariant);
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
        }
    }
    rewrite_function_uses(func, &replacements);
    if let Err(errors) = func.verify() {
        panic!(
            "regional invariant global-load hoisting produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn scalar_global_name_by_values(values: &[crate::ir::Value], value: ValueId) -> Option<&str> {
    let ValueKind::Global(name) = &values.get(value.0)?.kind else {
        return None;
    };
    Some(name)
}

#[derive(Debug, Clone)]
struct Candidate {
    name: String,
    value_ty: Type,
    global: ValueId,
    preheader: BlockId,
    header: BlockId,
    exit: BlockId,
    loop_blocks: HashSet<BlockId>,
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
            func.blocks[block.0].insts.iter().any(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Call { .. } | InstKind::MemZero { .. } | InstKind::MemCopy { .. }
                )
            })
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
            let Some((_, _, _, load_global, load_ty)) = accesses.loads.first() else {
                continue;
            };
            if accesses.stores.is_empty() {
                continue;
            }
            if accesses.loads.iter().any(|(_, _, _, _, ty)| ty != load_ty)
                || accesses.stores.iter().any(|(_, _, _, _, ty)| ty != load_ty)
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
                exit: *exit,
                loop_blocks: natural_loop.blocks.clone(),
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
                InstKind::MemZero { .. } | InstKind::MemCopy { .. } | InstKind::Call { .. } => {
                    return false
                }
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
    let local = func
        .append_inst(
            candidate.preheader,
            InstKind::Alloca {
                ty: candidate.value_ty.clone(),
            },
            Some(Type::Ptr(Box::new(candidate.value_ty.clone()))),
        )
        .expect("regional scalar needs a local stack slot");
    let initial = func
        .append_inst(
            candidate.preheader,
            InstKind::Load {
                ptr: candidate.global,
            },
            Some(candidate.value_ty.clone()),
        )
        .expect("regional global load must produce a value");
    func.append_inst(
        candidate.preheader,
        InstKind::Store {
            ptr: local,
            value: initial,
        },
        None,
    );
    for block in &candidate.loop_blocks {
        for inst in &mut func.blocks[block.0].insts {
            match &mut inst.kind {
                InstKind::Load { ptr } | InstKind::Store { ptr, .. }
                    if scalar_global_name_by_values(&func.values, *ptr)
                        == Some(candidate.name.as_str()) =>
                {
                    *ptr = local;
                }
                _ => {}
            }
        }
    }

    let flush = func.add_block(format!("regional.{}.flush", candidate.name));
    let final_value = func
        .append_inst(
            flush,
            InstKind::Load { ptr: local },
            Some(candidate.value_ty),
        )
        .expect("regional scalar flush must load the final value");
    func.append_inst(
        flush,
        InstKind::Store {
            ptr: candidate.global,
            value: final_value,
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
