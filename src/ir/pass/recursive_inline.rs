use super::dominators::ControlFlowGraph;
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CastOp, Function, FunctionId, Inst, InstKind, Module, Terminator, Type,
    UnaryOp, ValueId, ValueKind,
};
use std::collections::HashMap;

// The source-function limits keep analysis and each cloned CFG small. Growth is
// charged independently at the call site, caller, and module levels; a block
// also carries a cost because its terminator and phi-edge copies affect codegen.
const MAX_SOURCE_BLOCKS: usize = 16;
const MAX_SOURCE_INSTS: usize = 80;
const MAX_GENERAL_SOURCE_BLOCKS: usize = 32;
const MAX_GENERAL_SOURCE_INST_SLOTS: usize = 512;
const MAX_GENERAL_FUNCTION_GROWTH_INSTS: usize = 512;
const MAX_GENERAL_FUNCTION_GROWTH_BLOCKS: usize = 128;
const BLOCK_COST: usize = 2;
const MAX_CALLSITE_COST: usize = 120;
const MAX_FUNCTION_GROWTH_INSTS: usize = 192;
const MAX_FUNCTION_GROWTH_BLOCKS: usize = 40;
const MAX_MODULE_GROWTH_INSTS: usize = 1024;
const MAX_MODULE_GROWTH_BLOCKS: usize = 256;

pub(super) struct RecursiveInlinePass {
    expansion_rounds: usize,
}
pub(super) struct CfgInlinePass {
    allow_global_loads: bool,
}

impl RecursiveInlinePass {
    pub(super) fn new() -> Self {
        Self {
            expansion_rounds: 1,
        }
    }

    pub(super) fn with_rounds(expansion_rounds: usize) -> Self {
        assert!(expansion_rounds > 0);
        Self { expansion_rounds }
    }
}

impl CfgInlinePass {
    pub(super) fn new(allow_global_loads: bool) -> Self {
        Self { allow_global_loads }
    }
}

impl ModulePass for RecursiveInlinePass {
    fn run(&mut self, module: &mut Module) {
        let initial = module.funcs.clone();
        let initial_targets = CallGraphTargets::new(&initial);
        let eligible = initial
            .iter()
            .enumerate()
            .map(|(func_idx, source)| {
                !module.funcs[func_idx].has_recursive_cfg_inline_decision()
                    && Candidate::analyze(source, FunctionId(func_idx), &initial_targets).is_some()
            })
            .collect::<Vec<_>>();
        let mut module_growth = Growth::default();
        let mut function_growth = vec![Growth::default(); module.funcs.len()];

        for _ in 0..self.expansion_rounds {
            let snapshots = module.funcs.clone();
            let targets = CallGraphTargets::new(&snapshots);
            let mut changed = false;
            for (func_idx, source) in snapshots.iter().enumerate() {
                if !eligible[func_idx] {
                    continue;
                }
                let Some(candidate) = Candidate::analyze(source, FunctionId(func_idx), &targets)
                else {
                    continue;
                };

                for site in candidate.sites.iter().rev() {
                    let growth = candidate.growth;
                    if !candidate.callsite_fits()
                        || !function_growth[func_idx].fits_with(
                            growth,
                            MAX_FUNCTION_GROWTH_INSTS,
                            MAX_FUNCTION_GROWTH_BLOCKS,
                        )
                        || !module_growth.fits_with(
                            growth,
                            MAX_MODULE_GROWTH_INSTS,
                            MAX_MODULE_GROWTH_BLOCKS,
                        )
                    {
                        continue;
                    }
                    if !inline_call_site(
                        &mut module.funcs[func_idx],
                        source,
                        &candidate.reachable,
                        site,
                    ) {
                        continue;
                    }

                    function_growth[func_idx].add(growth);
                    module_growth.add(growth);
                    changed = true;
                    if let Err(errors) = module.funcs[func_idx].verify() {
                        panic!(
                            "recursive CFG inlining produced invalid IR in {}: {:?}",
                            module.funcs[func_idx].name, errors
                        );
                    }
                }
            }
            if !changed {
                break;
            }
        }

        for (func_idx, eligible) in eligible.into_iter().enumerate() {
            if !eligible {
                continue;
            }
            // Record the decision even when the module-wide budget refused all
            // sites. Otherwise a second pipeline run could spend a fresh module
            // budget on deferred functions and cease to be idempotent.
            // This is bookkeeping, not an eligibility heuristic.
            module.funcs[func_idx].mark_recursive_cfg_inline_decision();
        }
    }
}

impl ModulePass for CfgInlinePass {
    fn run(&mut self, module: &mut Module) {
        let snapshots = module.funcs.clone();
        let targets = CallGraphTargets::new(&snapshots);
        let candidates = snapshots
            .iter()
            .map(|func| GeneralCallee::analyze(func, self.allow_global_loads))
            .collect::<Vec<_>>();
        let mut plans = vec![Vec::new(); snapshots.len()];
        let mut module_growth = Growth::default();

        for (caller_idx, caller) in snapshots.iter().enumerate() {
            let reachable = reachable_blocks(caller);
            let has_unreachable_blocks = reachable.iter().any(|block| !block);
            let mut growth = Growth::default();
            let mut sites = Vec::new();
            for (block_idx, block) in caller.blocks.iter().enumerate() {
                if !reachable[block_idx] {
                    continue;
                }
                for (inst_idx, inst) in block.insts.iter().enumerate() {
                    let (Some(result), InstKind::Call { name, args }) = (inst.result, &inst.kind)
                    else {
                        continue;
                    };
                    let Some(callee_id) = targets.resolve(name) else {
                        continue;
                    };
                    if callee_id.0 == caller_idx {
                        continue;
                    }
                    let Some(candidate) = candidates[callee_id.0].as_ref() else {
                        continue;
                    };
                    let callee = &snapshots[callee_id.0];
                    if !call_types_match(caller, result, callee, args)
                        || (block_idx == caller.entry.0 && has_unreachable_blocks)
                        || !candidate.callsite_fits()
                    {
                        continue;
                    }
                    growth.add(candidate.growth);
                    sites.push(GeneralCallSite {
                        site: CallSite {
                            block: BlockId(block_idx),
                            inst_idx,
                            result,
                        },
                        callee: callee_id,
                    });
                }
            }
            if sites.is_empty()
                || !Growth::default().fits_with(
                    growth,
                    MAX_GENERAL_FUNCTION_GROWTH_INSTS,
                    MAX_GENERAL_FUNCTION_GROWTH_BLOCKS,
                )
            {
                continue;
            }
            module_growth.add(growth);
            plans[caller_idx] = sites;
        }

        if !Growth::default().fits_with(
            module_growth,
            MAX_MODULE_GROWTH_INSTS,
            MAX_MODULE_GROWTH_BLOCKS,
        ) {
            return;
        }

        for (caller_idx, sites) in plans.iter_mut().enumerate() {
            sites.sort_by_key(|site| (site.site.block.0, site.site.inst_idx));
            for planned in sites.iter().rev() {
                let source = &snapshots[planned.callee.0];
                let candidate = candidates[planned.callee.0]
                    .as_ref()
                    .expect("planned callee must remain eligible");
                if !inline_call_site(
                    &mut module.funcs[caller_idx],
                    source,
                    &candidate.reachable,
                    &planned.site,
                ) {
                    continue;
                }
                if let Err(errors) = module.funcs[caller_idx].verify() {
                    panic!(
                        "CFG inlining produced invalid IR in {}: {:?}",
                        module.funcs[caller_idx].name, errors
                    );
                }
            }
        }
    }
}

struct GeneralCallee {
    reachable: Vec<bool>,
    growth: Growth,
}

impl GeneralCallee {
    fn analyze(func: &Function, allow_global_loads: bool) -> Option<Self> {
        if !matches!(func.ret, Type::I1 | Type::I32 | Type::F32)
            || func.blocks.is_empty()
            || func.blocks.len() > MAX_GENERAL_SOURCE_BLOCKS
            || func.verify().is_err()
            || func
                .params
                .iter()
                .any(|param| !is_general_inline_parameter(&func.value(*param).ty))
        {
            return None;
        }
        let reachable = reachable_blocks(func);
        let reachable_block_count = reachable.iter().filter(|block| **block).count();
        let reachable_insts = func
            .blocks
            .iter()
            .enumerate()
            .filter(|(block_idx, _)| reachable[*block_idx])
            .flat_map(|(_, block)| &block.insts)
            .collect::<Vec<_>>();
        let active_insts = reachable_insts
            .iter()
            .filter(|inst| !matches!(inst.kind, InstKind::Nop))
            .count();
        let has_return = func.blocks.iter().enumerate().any(|(block_idx, block)| {
            reachable[block_idx] && matches!(block.terminator, Some(Terminator::Return(Some(_))))
        });
        if !has_return
            || active_insts > MAX_SOURCE_INSTS
            || reachable_insts.len() > MAX_GENERAL_SOURCE_INST_SLOTS
            || !func.blocks.iter().enumerate().all(|(block_idx, block)| {
                !reachable[block_idx]
                    || (block
                        .insts
                        .iter()
                        .all(|inst| is_readonly_inline_inst(func, inst, allow_global_loads))
                        && !matches!(block.terminator, Some(Terminator::Return(None))))
            })
        {
            return None;
        }
        let growth = Growth {
            insts: active_insts.checked_add(1)?,
            blocks: reachable_block_count.checked_add(1)?,
        };
        Some(Self { reachable, growth })
    }

    fn callsite_fits(&self) -> bool {
        self.growth
            .blocks
            .checked_mul(BLOCK_COST)
            .and_then(|block_cost| self.growth.insts.checked_add(block_cost))
            .is_some_and(|cost| cost <= MAX_CALLSITE_COST)
    }
}

fn is_readonly_inline_inst(func: &Function, inst: &Inst, allow_global_loads: bool) -> bool {
    matches!(
        inst.kind,
        InstKind::Nop
            | InstKind::Phi { .. }
            | InstKind::Unary {
                op: UnaryOp::Ineg | UnaryOp::Fneg | UnaryOp::Not,
                ..
            }
            | InstKind::Binary {
                op: BinaryOp::Iadd
                    | BinaryOp::Isub
                    | BinaryOp::Imul
                    | BinaryOp::Idiv
                    | BinaryOp::Imod
                    | BinaryOp::Iand
                    | BinaryOp::Ior
                    | BinaryOp::Ixor
                    | BinaryOp::Ishl
                    | BinaryOp::Iashr
                    | BinaryOp::And
                    | BinaryOp::Or
                    | BinaryOp::Fadd
                    | BinaryOp::Fsub
                    | BinaryOp::Fmul
                    | BinaryOp::Fdiv,
                ..
            }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast {
                op: CastOp::BoolToI32
                    | CastOp::I32ToBool
                    | CastOp::I32ToF32
                    | CastOp::F32ToI32
                    | CastOp::F32ToBool,
                ..
            }
            | InstKind::Gep { .. }
    ) || matches!(
        inst.kind,
        InstKind::Load { ptr }
            if inline_load_root(func, ptr).is_some_and(|root| {
                matches!(func.value(root).kind, ValueKind::Param)
                    || (allow_global_loads
                        && matches!(func.value(root).kind, ValueKind::Global(_)))
            })
    ) || matches!(
        inst.kind,
        InstKind::Store { ptr, .. }
            if allow_global_loads
                && matches!(func.value(ptr).kind, ValueKind::Global(_))
    )
}

fn is_general_inline_parameter(ty: &Type) -> bool {
    matches!(ty, Type::I1 | Type::I32 | Type::F32 | Type::Ptr(_))
}

fn inline_load_root(func: &Function, mut value: ValueId) -> Option<ValueId> {
    for _ in 0..64 {
        match func.value(value).kind {
            ValueKind::Param | ValueKind::Global(_) => return Some(value),
            ValueKind::Inst(block, inst_idx) => {
                let InstKind::Gep { base, .. } =
                    func.blocks.get(block.0)?.insts.get(inst_idx)?.kind
                else {
                    return None;
                };
                value = base;
            }
            ValueKind::Const(_) => return None,
        }
    }
    None
}

#[derive(Debug, Clone)]
struct GeneralCallSite {
    site: CallSite,
    callee: FunctionId,
}

#[derive(Debug, Clone, Copy, Default)]
struct Growth {
    insts: usize,
    blocks: usize,
}

impl Growth {
    fn fits_with(self, extra: Self, max_insts: usize, max_blocks: usize) -> bool {
        self.insts
            .checked_add(extra.insts)
            .is_some_and(|insts| insts <= max_insts)
            && self
                .blocks
                .checked_add(extra.blocks)
                .is_some_and(|blocks| blocks <= max_blocks)
    }

    fn add(&mut self, extra: Self) {
        self.insts += extra.insts;
        self.blocks += extra.blocks;
    }
}

struct CallGraphTargets {
    unique: HashMap<String, Option<FunctionId>>,
}

impl CallGraphTargets {
    fn new(funcs: &[Function]) -> Self {
        let mut unique = HashMap::new();
        for (idx, func) in funcs.iter().enumerate() {
            unique
                .entry(func.name.clone())
                .and_modify(|target| *target = None)
                .or_insert(Some(FunctionId(idx)));
        }
        Self { unique }
    }

    fn resolve(&self, target: &str) -> Option<FunctionId> {
        self.unique.get(target).copied().flatten()
    }
}

#[derive(Debug, Clone)]
struct CallSite {
    block: BlockId,
    inst_idx: usize,
    result: ValueId,
}

struct Candidate {
    sites: Vec<CallSite>,
    reachable: Vec<bool>,
    growth: Growth,
}

impl Candidate {
    fn analyze(func: &Function, self_id: FunctionId, targets: &CallGraphTargets) -> Option<Self> {
        if !matches!(func.ret, Type::I1 | Type::I32)
            || func.blocks.is_empty()
            || func.blocks.len() > MAX_SOURCE_BLOCKS
            || func.verify().is_err()
            || func.values.iter().any(|value| type_contains_f32(&value.ty))
        {
            return None;
        }

        let reachable = reachable_blocks(func);
        let has_unreachable_blocks = reachable.iter().any(|block| !block);
        let reachable_block_count = reachable.iter().filter(|block| **block).count();
        let active_insts = func
            .blocks
            .iter()
            .enumerate()
            .filter(|(block_idx, _)| reachable[*block_idx])
            .flat_map(|(_, block)| &block.insts)
            .filter(|inst| !matches!(inst.kind, InstKind::Nop))
            .count();
        let return_count = func
            .blocks
            .iter()
            .enumerate()
            .filter(|(block_idx, _)| reachable[*block_idx])
            .filter(|(_, block)| matches!(block.terminator, Some(Terminator::Return(Some(_)))))
            .count();
        if active_insts > MAX_SOURCE_INSTS
            || return_count == 0
            || !instructions_are_cloneable(func, &reachable)
        {
            return None;
        }

        let mut sites = Vec::new();
        for (block_idx, block) in func.blocks.iter().enumerate() {
            if !reachable[block_idx] {
                continue;
            }
            for (inst_idx, inst) in block.insts.iter().enumerate() {
                let InstKind::Call { name, args } = &inst.kind else {
                    continue;
                };
                if targets.resolve(name) != Some(self_id) {
                    continue;
                }
                let result = inst.result?;
                if !call_types_match(func, result, func, args) {
                    return None;
                }
                if !is_tail_position(func, BlockId(block_idx), inst_idx, result) {
                    // The verifier deliberately permits dead CFG to use values
                    // defined in the physical entry. Splitting entry would move
                    // the call and its suffix behind the cloned CFG, invalidating
                    // any such dead use. Conservatively leave entry calls alone
                    // whenever detached blocks exist.
                    if block_idx == func.entry.0 && has_unreachable_blocks {
                        continue;
                    }
                    sites.push(CallSite {
                        block: BlockId(block_idx),
                        inst_idx,
                        result,
                    });
                }
            }
        }
        if sites.is_empty() {
            return None;
        }
        sites.sort_by_key(|site| (site.block.0, site.inst_idx));

        // One continuation phi is charged even for a single return. This is
        // conservative and guarantees that every cloned return has one uniform
        // value boundary into the caller continuation.
        let growth = Growth {
            insts: active_insts.checked_add(1)?,
            blocks: reachable_block_count.checked_add(1)?,
        };
        Some(Self {
            sites,
            reachable,
            growth,
        })
    }

    fn callsite_fits(&self) -> bool {
        self.growth
            .blocks
            .checked_mul(BLOCK_COST)
            .and_then(|block_cost| self.growth.insts.checked_add(block_cost))
            .is_some_and(|cost| cost <= MAX_CALLSITE_COST)
    }
}

fn reachable_blocks(func: &Function) -> Vec<bool> {
    let cfg = ControlFlowGraph::new(func);
    let mut reachable = vec![false; func.blocks.len()];
    let mut stack = vec![func.entry];
    reachable[func.entry.0] = true;
    while let Some(block) = stack.pop() {
        for succ in &cfg.succs[block.0] {
            if !reachable[succ.0] {
                reachable[succ.0] = true;
                stack.push(*succ);
            }
        }
    }
    reachable
}

fn type_contains_f32(ty: &Type) -> bool {
    match ty {
        Type::F32 => true,
        Type::Ptr(pointee) => type_contains_f32(pointee),
        Type::Array { elem, .. } => type_contains_f32(elem),
        Type::Void | Type::I1 | Type::I32 => false,
    }
}

fn instructions_are_cloneable(func: &Function, reachable: &[bool]) -> bool {
    func.blocks.iter().enumerate().all(|(block_idx, block)| {
        if !reachable[block_idx] {
            return true;
        }
        block.insts.iter().all(|inst| match &inst.kind {
            // Stack-object lifetime and bulk initialization need a separate
            // frame/escape proof. Reject them rather than approximating it.
            InstKind::Alloca { .. } | InstKind::MemZero { .. } | InstKind::Fcmp { .. } => false,
            InstKind::Unary {
                op: UnaryOp::Fneg, ..
            }
            | InstKind::Binary {
                op: BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv,
                ..
            }
            | InstKind::Cast {
                op: CastOp::I32ToF32 | CastOp::F32ToI32 | CastOp::F32ToBool,
                ..
            } => false,
            // A result-less call crosses a void ABI boundary. Ordinary scalar
            // loads/stores and result-producing calls keep their exact order.
            InstKind::Call { .. } if inst.result.is_none() => false,
            _ => true,
        })
    })
}

fn call_types_match(
    caller: &Function,
    result: ValueId,
    callee: &Function,
    args: &[ValueId],
) -> bool {
    args.len() == callee.params.len()
        && caller
            .values
            .get(result.0)
            .is_some_and(|value| value.ty == callee.ret)
        && args.iter().zip(&callee.params).all(|(arg, param)| {
            caller
                .values
                .get(arg.0)
                .zip(callee.values.get(param.0))
                .is_some_and(|(arg, param)| inline_types_compatible(&arg.ty, &param.ty))
        })
}

fn inline_types_compatible(argument: &Type, parameter: &Type) -> bool {
    argument == parameter || matches!((argument, parameter), (Type::Ptr(_), Type::Ptr(_)))
}

fn is_tail_position(func: &Function, block: BlockId, inst_idx: usize, result: ValueId) -> bool {
    func.block(block).insts[inst_idx + 1..]
        .iter()
        .all(|inst| matches!(inst.kind, InstKind::Nop))
        && matches!(
            func.block(block).terminator,
            Some(Terminator::Return(Some(returned))) if returned == result
        )
}

fn inline_call_site(
    func: &mut Function,
    source: &Function,
    reachable: &[bool],
    site: &CallSite,
) -> bool {
    let Some(Inst {
        result: Some(result),
        kind: InstKind::Call { args, .. },
    }) = func
        .blocks
        .get(site.block.0)
        .and_then(|block| block.insts.get(site.inst_idx))
        .cloned()
    else {
        return false;
    };
    if result != site.result || !call_types_match(func, result, source, &args) {
        return false;
    }

    let continuation = split_call_block(func, site.block, site.inst_idx);
    let cloned_blocks = reachable
        .iter()
        .map(|is_reachable| is_reachable.then(|| func.add_block("recursive.inline")))
        .collect::<Vec<_>>();
    let mut values = source
        .params
        .iter()
        .copied()
        .zip(args)
        .collect::<HashMap<_, _>>();
    for (value_idx, value) in source.values.iter().enumerate() {
        let mapped = match &value.kind {
            ValueKind::Const(constant) => Some(get_or_add_const(func, constant)),
            ValueKind::Global(name) => Some(get_or_add_global(func, name, &value.ty)),
            _ => None,
        };
        if let Some(mapped) = mapped {
            values.insert(ValueId(value_idx), mapped);
        }
    }
    let mut cloned_insts = Vec::new();

    // Allocate all result IDs before cloning operands so loop-carried phi
    // values and forward block references can be remapped without ordering
    // assumptions about Function::blocks.
    for (source_block_idx, source_block) in source.blocks.iter().enumerate() {
        let Some(cloned_block) = cloned_blocks[source_block_idx] else {
            continue;
        };
        for (source_inst_idx, inst) in source_block.insts.iter().enumerate() {
            if matches!(inst.kind, InstKind::Nop) {
                continue;
            }
            let result_ty = inst.result.map(|value| source.value(value).ty.clone());
            let cloned_result = func.append_inst(cloned_block, InstKind::Nop, result_ty);
            if let (Some(source_result), Some(cloned_result)) = (inst.result, cloned_result) {
                values.insert(source_result, cloned_result);
            }
            cloned_insts.push((
                BlockId(source_block_idx),
                source_inst_idx,
                cloned_block,
                func.block(cloned_block).insts.len() - 1,
            ));
        }
    }

    for (source_block, source_inst_idx, cloned_block, cloned_inst_idx) in cloned_insts {
        let kind = clone_inst_kind(
            source,
            &source.block(source_block).insts[source_inst_idx].kind,
            &cloned_blocks,
            &values,
        );
        func.block_mut(cloned_block).insts[cloned_inst_idx].kind = kind;
    }

    let mut returns = Vec::new();
    for (source_block_idx, source_block) in source.blocks.iter().enumerate() {
        let Some(cloned_block) = cloned_blocks[source_block_idx] else {
            continue;
        };
        let terminator = match source_block
            .terminator
            .as_ref()
            .expect("verified source block terminator")
        {
            Terminator::Return(Some(value)) => {
                returns.push((cloned_block, map_value(source, *value, &values)));
                Terminator::Jump(continuation)
            }
            Terminator::Return(None) => unreachable!("non-void verified source return"),
            Terminator::Jump(target) => Terminator::Jump(
                cloned_blocks[target.0].expect("reachable successor must be cloned"),
            ),
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => Terminator::Branch {
                cond: map_value(source, *cond, &values),
                then_target: cloned_blocks[then_target.0]
                    .expect("reachable successor must be cloned"),
                else_target: cloned_blocks[else_target.0]
                    .expect("reachable successor must be cloned"),
            },
        };
        func.set_terminator(cloned_block, terminator);
    }
    debug_assert!(!returns.is_empty(), "candidate has a reachable return");

    func.block_mut(site.block).terminator = Some(Terminator::Jump(
        cloned_blocks[source.entry.0].expect("source entry must be cloned"),
    ));
    let merged = func
        .insert_inst(
            continuation,
            0,
            InstKind::Phi { incomings: returns },
            Some(source.ret.clone()),
        )
        .expect("non-void recursive inline return phi");
    rewrite_function_uses(func, &ValueReplacements::from([(site.result, merged)]));
    true
}

fn split_call_block(func: &mut Function, block: BlockId, call_idx: usize) -> BlockId {
    let continuation = func.add_block("recursive.continue");
    let suffix = func.block_mut(block).insts.split_off(call_idx + 1);
    let old_terminator = func.block_mut(block).terminator.take();

    func.block_mut(block).insts[call_idx] = Inst {
        result: None,
        kind: InstKind::Nop,
    };
    func.block_mut(continuation).insts = suffix;
    func.block_mut(continuation).terminator = old_terminator;

    for value in &mut func.values {
        let ValueKind::Inst(owner, inst_idx) = &mut value.kind else {
            continue;
        };
        if *owner == block && *inst_idx > call_idx {
            *owner = continuation;
            *inst_idx -= call_idx + 1;
        }
    }
    retarget_successor_phis(func, block, continuation);
    continuation
}

fn retarget_successor_phis(func: &mut Function, old_pred: BlockId, new_pred: BlockId) {
    let Some(terminator) = func.block(new_pred).terminator.as_ref() else {
        return;
    };
    let successors = match terminator {
        Terminator::Return(_) => Vec::new(),
        Terminator::Jump(target) => vec![*target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } if then_target == else_target => vec![*then_target],
        Terminator::Branch {
            then_target,
            else_target,
            ..
        } => vec![*then_target, *else_target],
    };
    for successor in successors {
        for inst in &mut func.block_mut(successor).insts {
            if let InstKind::Phi { incomings } = &mut inst.kind {
                for (pred, _) in incomings {
                    if *pred == old_pred {
                        *pred = new_pred;
                    }
                }
            }
        }
    }
}

fn clone_inst_kind(
    source: &Function,
    kind: &InstKind,
    blocks: &[Option<BlockId>],
    values: &HashMap<ValueId, ValueId>,
) -> InstKind {
    let map = |value| map_value(source, value, values);
    match kind {
        InstKind::Nop => InstKind::Nop,
        InstKind::Phi { incomings } => InstKind::Phi {
            incomings: incomings
                .iter()
                .filter_map(|(block, value)| blocks[block.0].map(|block| (block, map(*value))))
                .collect(),
        },
        InstKind::Alloca { ty } => InstKind::Alloca { ty: ty.clone() },
        InstKind::Load { ptr } => InstKind::Load { ptr: map(*ptr) },
        InstKind::Store { ptr, value } => InstKind::Store {
            ptr: map(*ptr),
            value: map(*value),
        },
        InstKind::MemZero { ptr, bytes } => InstKind::MemZero {
            ptr: map(*ptr),
            bytes: *bytes,
        },
        InstKind::Unary { op, value } => InstKind::Unary {
            op: *op,
            value: map(*value),
        },
        InstKind::Binary { op, lhs, rhs } => InstKind::Binary {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Icmp { op, lhs, rhs } => InstKind::Icmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Fcmp { op, lhs, rhs } => InstKind::Fcmp {
            op: *op,
            lhs: map(*lhs),
            rhs: map(*rhs),
        },
        InstKind::Cast { op, value } => InstKind::Cast {
            op: *op,
            value: map(*value),
        },
        InstKind::Gep { base, indices } => InstKind::Gep {
            base: map(*base),
            indices: indices.iter().map(|index| map(*index)).collect(),
        },
        InstKind::Call { name, args } => InstKind::Call {
            name: name.clone(),
            args: args.iter().map(|arg| map(*arg)).collect(),
        },
    }
}

fn get_or_add_global(func: &mut Function, name: &str, ty: &Type) -> ValueId {
    func.values
        .iter()
        .position(|value| {
            value.ty == *ty
                && matches!(&value.kind, ValueKind::Global(existing) if existing == name)
        })
        .map(ValueId)
        .unwrap_or_else(|| func.add_global_ref(name, ty.clone()))
}

fn get_or_add_const(func: &mut Function, constant: &crate::ir::Const) -> ValueId {
    func.values
        .iter()
        .position(|value| {
            value.ty == constant.ty()
                && matches!(&value.kind, ValueKind::Const(existing) if existing == constant)
        })
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(constant.clone()))
}

fn map_value(_source: &Function, value: ValueId, values: &HashMap<ValueId, ValueId>) -> ValueId {
    if let Some(mapped) = values.get(&value) {
        return *mapped;
    }
    panic!("verified clone operand must have a value mapping");
}

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, OptLevel, PassOptions};
    use super::*;
    use crate::ir::{CmpOp, Const};

    #[test]
    fn cfg_inlines_readonly_float_pointer_loops() {
        let mut reduce = Function::new("reduce_values", Type::F32);
        let values = reduce.add_param("values", Type::Ptr(Box::new(Type::F32)));
        let count = reduce.add_param("count", Type::I32);
        let zero_i32 = reduce.add_const(Const::Int(0));
        let zero_f32 = reduce.add_const(Const::Float(0));
        let one = reduce.add_const(Const::Int(1));
        let header = reduce.add_block("header");
        let body = reduce.add_block("body");
        let exit = reduce.add_block("exit");
        reduce.set_terminator(reduce.entry, Terminator::Jump(header));
        let index = reduce
            .append_inst(header, InstKind::Phi { incomings: vec![] }, Some(Type::I32))
            .unwrap();
        let sum = reduce
            .append_inst(header, InstKind::Phi { incomings: vec![] }, Some(Type::F32))
            .unwrap();
        let condition = reduce
            .append_inst(
                header,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: index,
                    rhs: count,
                },
                Some(Type::I1),
            )
            .unwrap();
        reduce.set_terminator(
            header,
            Terminator::Branch {
                cond: condition,
                then_target: body,
                else_target: exit,
            },
        );
        let address = reduce
            .append_inst(
                body,
                InstKind::Gep {
                    base: values,
                    indices: vec![index],
                },
                Some(Type::Ptr(Box::new(Type::F32))),
            )
            .unwrap();
        let loaded = reduce
            .append_inst(body, InstKind::Load { ptr: address }, Some(Type::F32))
            .unwrap();
        let next_sum = reduce
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs: sum,
                    rhs: loaded,
                },
                Some(Type::F32),
            )
            .unwrap();
        let next_index = reduce
            .append_inst(
                body,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: index,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        reduce.set_terminator(body, Terminator::Jump(header));
        reduce.set_terminator(exit, Terminator::Return(Some(sum)));
        reduce.blocks[header.0].insts[0].kind = InstKind::Phi {
            incomings: vec![(reduce.entry, zero_i32), (body, next_index)],
        };
        reduce.blocks[header.0].insts[1].kind = InstKind::Phi {
            incomings: vec![(reduce.entry, zero_f32), (body, next_sum)],
        };

        let mut caller = Function::new("caller", Type::F32);
        let caller_values = caller.add_param(
            "values",
            Type::Ptr(Box::new(Type::Array {
                elem: Box::new(Type::F32),
                len: 16,
            })),
        );
        let caller_count = caller.add_param("count", Type::I32);
        let result = caller
            .append_inst(
                caller.entry,
                InstKind::Call {
                    name: "reduce_values".to_string(),
                    args: vec![caller_values, caller_count],
                },
                Some(Type::F32),
            )
            .unwrap();
        caller.set_terminator(caller.entry, Terminator::Return(Some(result)));

        let mut module = Module::new();
        module.add_func(reduce);
        module.add_func(caller);
        CfgInlinePass::new(false).run(&mut module);

        assert!(module.funcs[1].blocks.iter().all(|block| {
            block
                .insts
                .iter()
                .all(|inst| !matches!(inst.kind, InstKind::Call { .. }))
        }));
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    #[test]
    fn inlines_different_names_and_cfgs_from_one_snapshot() {
        let first = branching_recursion("spruce");
        let second = phi_recursion("willow");
        let original_blocks = [first.blocks.len(), second.blocks.len()];
        let mut module = Module::new();
        module.add_func(first);
        module.add_func(second);

        RecursiveInlinePass::new().run(&mut module);

        for (idx, func) in module.funcs.iter().enumerate() {
            assert!(func.has_recursive_cfg_inline_decision());
            assert_eq!(
                self_call_count(func),
                1,
                "cloned calls must not expand again"
            );
            assert_eq!(
                func.blocks.len(),
                original_blocks[idx] * 2 + 1,
                "one continuation and one source CFG must be added"
            );
            assert!(func.verify().is_ok());
        }
    }

    #[test]
    fn merges_multiple_returns_and_remaps_existing_phis() {
        let func = phi_recursion("maple");
        let call_result = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .find_map(|inst| matches!(inst.kind, InstKind::Call { .. }).then_some(inst.result?))
            .unwrap();
        let consumer = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .find_map(|inst| match inst.kind {
                InstKind::Binary {
                    lhs,
                    op: BinaryOp::Iadd,
                    ..
                } if lhs == call_result => inst.result,
                _ => None,
            })
            .unwrap();
        let original_phi_count = phi_count(&func);
        let mut module = Module::new();
        module.add_func(func);

        RecursiveInlinePass::new().run(&mut module);
        let func = &module.funcs[0];
        let ValueKind::Inst(consumer_block, consumer_idx) = func.value(consumer).kind else {
            panic!("consumer must remain instruction-backed");
        };
        let InstKind::Binary { lhs: merged, .. } =
            func.block(consumer_block).insts[consumer_idx].kind
        else {
            panic!("consumer must remain an add");
        };
        let ValueKind::Inst(phi_block, phi_idx) = func.value(merged).kind else {
            panic!("recursive result must be merged by a phi");
        };
        let InstKind::Phi { incomings } = &func.block(phi_block).insts[phi_idx].kind else {
            panic!("recursive result must be merged by a phi");
        };

        assert_eq!(incomings.len(), 2, "both cloned returns must reach the phi");
        assert_eq!(phi_count(func), original_phi_count * 2 + 1);
        assert!(func.verify().is_ok());
    }

    #[test]
    fn rejects_a_source_over_the_instruction_budget() {
        let func = oversized_recursion("ash");
        let mut module = Module::new();
        module.add_func(func);
        let before = module.clone();

        RecursiveInlinePass::new().run(&mut module);

        assert_eq!(module, before);
        assert!(module.funcs[0].verify().is_ok());
    }

    #[test]
    fn leaves_non_recursive_functions_unchanged() {
        let mut func = Function::new("elm", Type::I32);
        let value = func.add_param("input", Type::I32);
        let one = func.add_const(Const::Int(1));
        let sum = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: value,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(sum)));
        let mut module = Module::new();
        module.add_func(func);
        let before = module.clone();

        RecursiveInlinePass::new().run(&mut module);

        assert_eq!(module, before);
        assert!(module.funcs[0].verify().is_ok());
    }

    #[test]
    fn repeated_runs_are_idempotent_and_verified() {
        let mut module = Module::new();
        module.add_func(branching_recursion("birch"));
        RecursiveInlinePass::new().run(&mut module);
        let once = module.clone();

        RecursiveInlinePass::new().run(&mut module);

        assert_eq!(module, once);
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    #[test]
    fn module_budget_refusal_is_idempotent() {
        let mut module = Module::new();
        for idx in 0..65 {
            module.add_func(branching_recursion(&format!("tree_{idx}")));
        }

        RecursiveInlinePass::new().run(&mut module);
        let transformed = module
            .funcs
            .iter()
            .filter(|func| func.blocks.len() > 3)
            .count();
        let once = module.clone();
        RecursiveInlinePass::new().run(&mut module);

        assert_eq!(transformed, MAX_MODULE_GROWTH_BLOCKS / 4);
        assert!(module
            .funcs
            .iter()
            .all(Function::has_recursive_cfg_inline_decision));
        assert_eq!(module, once);
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    #[test]
    fn pipeline_enables_recursive_inlining_only_at_o1() {
        let func = branching_recursion("larch");
        let original_blocks = func.blocks.len();
        let mut o0 = Module::new();
        o0.add_func(func.clone());
        let mut o1 = Module::new();
        o1.add_func(func);
        let options = PassOptions {
            enable_simple_loop_unroll: false,
            small_expr_inline_rounds: 1,
            cfg_inline_rounds: 1,
            cfg_inline_global_loads: false,
            recursive_inline_rounds: 1,
            enable_constant_address_count_reduction: false,
            enable_recursive_const_specialization: false,
            enable_initialized_global_propagation: false,
            enable_uniform_constant_arguments: false,
            enable_loop_call_memoize: false,
            enable_loop_invariant_call_memoize: false,
            enable_regional_global_scalar_promotion: false,
            enable_full_domain_bitwise_digit: false,
            enable_write_only_alloca_cleanup_before_inline: true,
        };

        run_pipeline(&mut o0, OptLevel::O0, options);
        run_pipeline(&mut o1, OptLevel::O1, options);

        assert!(!o0.funcs[0].has_recursive_cfg_inline_decision());
        assert_eq!(o0.funcs[0].blocks.len(), original_blocks);
        assert!(o1.funcs[0].has_recursive_cfg_inline_decision());
        assert!(o1.funcs[0].blocks.len() > original_blocks);
        assert!(o0.funcs[0].verify().is_ok());
        assert!(o1.funcs[0].verify().is_ok());
    }

    #[test]
    fn omits_unreachable_blocks_from_the_cloned_cfg() {
        let mut func = branching_recursion("hemlock");
        let original_blocks = func.blocks.len();
        let dead = func.add_block("detached");
        let zero = func.add_const(Const::Int(0));
        func.set_terminator(dead, Terminator::Return(Some(zero)));
        let mut module = Module::new();
        module.add_func(func);

        RecursiveInlinePass::new().run(&mut module);

        let func = &module.funcs[0];
        assert!(func.has_recursive_cfg_inline_decision());
        assert_eq!(func.blocks.len(), original_blocks + 1 + original_blocks + 1);
        assert!(func.verify().is_ok());
    }

    #[test]
    fn conservatively_rejects_active_stack_allocation() {
        let mut func = branching_recursion("fir");
        func.insert_inst(
            func.entry,
            0,
            InstKind::Alloca { ty: Type::I32 },
            Some(Type::Ptr(Box::new(Type::I32))),
        );
        assert!(func.verify().is_ok());
        let mut module = Module::new();
        module.add_func(func);
        let before = module.clone();

        RecursiveInlinePass::new().run(&mut module);

        assert_eq!(module, before);
    }

    fn branching_recursion(name: &str) -> Function {
        let mut func = Function::new(name, Type::I32);
        let input = func.add_param("seed", Type::I32);
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let base = func.add_block("stop");
        let recurse = func.add_block("advance");
        let condition = func
            .append_inst(
                func.entry,
                InstKind::Icmp {
                    op: CmpOp::Le,
                    lhs: input,
                    rhs: zero,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond: condition,
                then_target: base,
                else_target: recurse,
            },
        );
        func.set_terminator(base, Terminator::Return(Some(input)));
        let next = func
            .append_inst(
                recurse,
                InstKind::Binary {
                    op: BinaryOp::Isub,
                    lhs: input,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        let call = func
            .append_inst(
                recurse,
                InstKind::Call {
                    name: name.to_string(),
                    args: vec![next],
                },
                Some(Type::I32),
            )
            .unwrap();
        let result = func
            .append_inst(
                recurse,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: call,
                    rhs: input,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(recurse, Terminator::Return(Some(result)));
        assert!(func.verify().is_ok());
        func
    }

    fn phi_recursion(name: &str) -> Function {
        let mut func = Function::new(name, Type::I32);
        let input = func.add_param("amount", Type::I32);
        let zero = func.add_const(Const::Int(0));
        let one = func.add_const(Const::Int(1));
        let base = func.add_block("base.case");
        let dispatch = func.add_block("dispatch");
        let left = func.add_block("left.path");
        let right = func.add_block("right.path");
        let join = func.add_block("merge.path");
        let done = func
            .append_inst(
                func.entry,
                InstKind::Icmp {
                    op: CmpOp::Le,
                    lhs: input,
                    rhs: zero,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            func.entry,
            Terminator::Branch {
                cond: done,
                then_target: base,
                else_target: dispatch,
            },
        );
        func.set_terminator(base, Terminator::Return(Some(input)));
        let choose = func
            .append_inst(
                dispatch,
                InstKind::Icmp {
                    op: CmpOp::Eq,
                    lhs: input,
                    rhs: one,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(
            dispatch,
            Terminator::Branch {
                cond: choose,
                then_target: left,
                else_target: right,
            },
        );
        func.set_terminator(left, Terminator::Jump(join));
        func.set_terminator(right, Terminator::Jump(join));
        let selected = func
            .append_inst(
                join,
                InstKind::Phi {
                    incomings: vec![(left, input), (right, one)],
                },
                Some(Type::I32),
            )
            .unwrap();
        let next = func
            .append_inst(
                join,
                InstKind::Binary {
                    op: BinaryOp::Isub,
                    lhs: input,
                    rhs: selected,
                },
                Some(Type::I32),
            )
            .unwrap();
        let call = func
            .append_inst(
                join,
                InstKind::Call {
                    name: name.to_string(),
                    args: vec![next],
                },
                Some(Type::I32),
            )
            .unwrap();
        let result = func
            .append_inst(
                join,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: call,
                    rhs: selected,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(join, Terminator::Return(Some(result)));
        assert!(func.verify().is_ok());
        func
    }

    fn oversized_recursion(name: &str) -> Function {
        let mut func = branching_recursion(name);
        let recurse = func
            .blocks
            .iter()
            .position(|block| {
                block
                    .insts
                    .iter()
                    .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
            })
            .map(BlockId)
            .unwrap();
        let one = func.add_const(Const::Int(1));
        for _ in 0..=MAX_SOURCE_INSTS {
            func.insert_inst(
                recurse,
                1,
                InstKind::Binary {
                    op: BinaryOp::Iadd,
                    lhs: one,
                    rhs: one,
                },
                Some(Type::I32),
            );
        }
        assert!(func.verify().is_ok());
        func
    }

    fn self_call_count(func: &Function) -> usize {
        func.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter(|inst| matches!(&inst.kind, InstKind::Call { name, .. } if name == &func.name))
            .count()
    }

    fn phi_count(func: &Function) -> usize {
        func.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter(|inst| matches!(inst.kind, InstKind::Phi { .. }))
            .count()
    }
}
