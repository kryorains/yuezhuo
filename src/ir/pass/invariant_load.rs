mod analysis;

use self::analysis::{MemoryRoot, WholeProgramAliasInfo};
use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::LoopInfo;
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{BlockId, Function, InstKind, Module, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_BLOCKS: usize = 2048;
const MAX_VALUES: usize = 16_384;
const MAX_INSTRUCTIONS: usize = 65_536;

/// Forwards loads from memory that cannot be changed by the current function.
///
/// Loads stay at their original control-flow points: only a dominating earlier
/// load of the exact same (normally GEP-CSE'd) pointer value can be reused.
pub(super) struct InvariantLoadForwardPass;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum LoadLocation {
    Exact(ValueId),
    DirectGlobal(usize),
}

impl InvariantLoadForwardPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for InvariantLoadForwardPass {
    fn run(&mut self, module: &mut Module) {
        let alias_info = WholeProgramAliasInfo::analyze(module);
        for func_idx in 0..module.funcs.len() {
            forward_invariant_loads(&mut module.funcs[func_idx], func_idx, &alias_info);
        }
    }
}

fn forward_invariant_loads(
    func: &mut Function,
    func_idx: usize,
    alias_info: &WholeProgramAliasInfo,
) {
    let instruction_count: usize = func.blocks.iter().map(|block| block.insts.len()).sum();
    if func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || instruction_count > MAX_INSTRUCTIONS
    {
        return;
    }

    let mut writes = Vec::new();
    let mut load_ptrs = HashSet::new();
    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.kind {
                // Calls are unknown clobbers. Reject all read-only inference in
                // the function rather than relying on call ordering.
                InstKind::Call { .. } => return,
                InstKind::Store { ptr, .. } | InstKind::MemZero { ptr, .. } => {
                    writes.push(alias_info.root(func, *ptr));
                }
                InstKind::MemCopy { .. } => return,
                InstKind::Load { ptr } => {
                    load_ptrs.insert(*ptr);
                }
                _ => {}
            }
        }
    }

    let readonly_ptrs = load_ptrs
        .into_iter()
        .filter(|ptr| {
            let load_root = alias_info.root(func, *ptr);
            writes
                .iter()
                .all(|write_root| alias_info.are_disjoint(func_idx, load_root, *write_root))
        })
        .collect::<HashSet<_>>();
    if readonly_ptrs.is_empty() {
        return;
    }

    // A direct global object is always a valid address.  Once whole-program
    // alias analysis has proved that the function cannot modify it, its load
    // is safe to speculate into a natural-loop preheader.  Process inner
    // loops first so the same load can subsequently migrate out of enclosing
    // loops, just as mature LICM implementations do.
    hoist_readonly_global_loads(func, alias_info, &readonly_ptrs);

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let mut available = HashMap::<LoadLocation, ValueId>::new();
    let mut replacements = ValueReplacements::new();
    visit_dom_tree(
        func,
        func.entry,
        &dom,
        alias_info,
        &readonly_ptrs,
        &mut available,
        &mut replacements,
    );

    if replacements.is_empty() {
        return;
    }
    rewrite_function_uses(func, &replacements);
    if let Err(errors) = func.verify() {
        panic!(
            "invariant load forwarding produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn hoist_readonly_global_loads(
    func: &mut Function,
    alias_info: &WholeProgramAliasInfo,
    readonly_ptrs: &HashSet<ValueId>,
) {
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let mut loops = LoopInfo::new(&cfg, &dom).loops().to_vec();
    loops.sort_by_key(|natural_loop| natural_loop.blocks.len());

    for natural_loop in loops {
        let Some(preheader) = natural_loop.dedicated_preheader else {
            continue;
        };
        let mut locations = HashSet::new();
        let mut candidates = Vec::new();
        let mut blocks = natural_loop.blocks.iter().copied().collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.0);
        for block in blocks {
            for (inst_idx, inst) in func.blocks[block.0].insts.iter().enumerate() {
                let (Some(result), InstKind::Load { ptr }) = (inst.result, &inst.kind) else {
                    continue;
                };
                if !readonly_ptrs.contains(ptr)
                    || !matches!(func.value(*ptr).kind, ValueKind::Global(_))
                {
                    continue;
                }
                let MemoryRoot::Global(global) = alias_info.root(func, *ptr) else {
                    continue;
                };
                let location = LoadLocation::DirectGlobal(global);
                if locations.insert(location) {
                    candidates.push((block, inst_idx, result, *ptr));
                }
            }
        }

        for (block, inst_idx, result, ptr) in candidates {
            if !natural_loop.blocks.contains(&block)
                || !matches!(func.blocks[block.0].insts[inst_idx].kind, InstKind::Load { ptr: current } if current == ptr)
            {
                continue;
            }
            func.blocks[block.0].insts[inst_idx].result = None;
            func.blocks[block.0].insts[inst_idx].kind = InstKind::Nop;
            let new_idx = func.blocks[preheader.0].insts.len();
            func.blocks[preheader.0].insts.push(crate::ir::Inst {
                result: Some(result),
                kind: InstKind::Load { ptr },
            });
            func.values[result.0].kind = ValueKind::Inst(preheader, new_idx);
        }
    }
}

fn visit_dom_tree(
    func: &mut Function,
    block: BlockId,
    dom: &Dominators,
    alias_info: &WholeProgramAliasInfo,
    readonly_ptrs: &HashSet<ValueId>,
    available: &mut HashMap<LoadLocation, ValueId>,
    replacements: &mut ValueReplacements,
) {
    let mut introduced = Vec::new();
    for inst_idx in 0..func.blocks[block.0].insts.len() {
        let inst = &func.blocks[block.0].insts[inst_idx];
        let (Some(result), InstKind::Load { ptr }) = (inst.result, &inst.kind) else {
            continue;
        };
        if !readonly_ptrs.contains(ptr) {
            continue;
        }
        let location = match func.value(*ptr).kind {
            ValueKind::Global(_) => match alias_info.root(func, *ptr) {
                analysis::MemoryRoot::Global(global) => LoadLocation::DirectGlobal(global),
                _ => LoadLocation::Exact(*ptr),
            },
            _ => LoadLocation::Exact(*ptr),
        };

        if let Some(previous) = available.get(&location).copied() {
            // The pointer identity is exact and `previous` dominates this load.
            // Keep the earlier load where it was, and remove only the duplicate.
            if func.value(previous).ty == func.value(result).ty {
                replacements.insert(result, previous);
                let duplicate = &mut func.blocks[block.0].insts[inst_idx];
                duplicate.result = None;
                duplicate.kind = InstKind::Nop;
            }
        } else {
            available.insert(location, result);
            introduced.push(location);
        }
    }

    for child in &dom.children[block.0] {
        visit_dom_tree(
            func,
            *child,
            dom,
            alias_info,
            readonly_ptrs,
            available,
            replacements,
        );
    }

    for location in introduced {
        available.remove(&location);
    }
}
