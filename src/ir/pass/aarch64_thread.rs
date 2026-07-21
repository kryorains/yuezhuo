use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{analyze_i32_induction, LoopInfo, NaturalLoop};
use super::ModulePass;
use crate::ir::{
    AArch64ThreadCapture, AArch64ThreadPlan, BinaryOp, BlockId, CastOp, CmpOp, Const, Function,
    FunctionId, Inst, InstKind, Module, Terminator, Type, UnaryOp, ValueId, ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_FUNCTIONS: usize = 256;
const MAX_ACTIVE_BODY_INSTS: usize = 64;
const MAX_MEMORY_INSTS: usize = 32;
const MAX_CAPTURES: usize = 6;
const MAX_PLANS_PER_MODULE: usize = 16;
const MAX_PROOF_WORK: usize = 65_536;
const MAX_ADDED_VALUES: usize = 2048;
const MIN_ACTIVE_BODY_COST: usize = 8;

/// Outlines an intentionally narrow owner-computes loop into a verified range
/// helper. The original loop is not rewritten: AArch64 emission adds a guarded
/// pthread dispatch and retains the exact scalar edge as its failure path.
pub(super) struct AArch64ThreadOutlinePass;

impl AArch64ThreadOutlinePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for AArch64ThreadOutlinePass {
    fn run(&mut self, module: &mut Module) {
        if !module.aarch64_thread_plans.is_empty() {
            return;
        }
        outline_module(module);
    }
}

#[derive(Clone)]
struct Candidate {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    counter: ValueId,
    counter_next: ValueId,
    bound: ValueId,
    dispatch_setup: Vec<ValueId>,
    captures: Vec<AArch64ThreadCapture>,
}

struct PendingOutline {
    parent: FunctionId,
    candidate: Candidate,
    helper: Function,
    context_symbol: String,
    worker_symbol: String,
}

#[derive(Clone, Copy)]
struct UseSite {
    block: BlockId,
    role: UseRole,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UseRole {
    LoadPointer,
    StorePointer,
    Phi {
        predecessor: BlockId,
        target: BlockId,
    },
    Other,
}

struct WorkBudget {
    remaining: usize,
}

impl WorkBudget {
    fn new() -> Self {
        Self {
            remaining: MAX_PROOF_WORK,
        }
    }

    fn spend(&mut self, amount: usize) -> Option<()> {
        if amount > self.remaining {
            self.remaining = 0;
            return None;
        }
        self.remaining -= amount;
        Some(())
    }
}

fn outline_module(module: &mut Module) {
    let original_function_count = module.funcs.len();
    if original_function_count > MAX_FUNCTIONS {
        return;
    }
    let mut symbol_counts = HashMap::<String, usize>::new();
    for name in module
        .funcs
        .iter()
        .map(|func| &func.name)
        .chain(module.globals.iter().map(|global| &global.name))
    {
        *symbol_counts.entry(name.clone()).or_default() += 1;
    }
    if symbol_counts.contains_key("pthread_create") || symbol_counts.contains_key("pthread_join") {
        return;
    }

    let mut reserved_symbols = symbol_counts.keys().cloned().collect::<HashSet<_>>();
    let mut budget = WorkBudget::new();
    let mut pending = Vec::new();
    let mut added_values = 0usize;

    for function_idx in 0..original_function_count {
        if pending.len() >= MAX_PLANS_PER_MODULE {
            break;
        }
        let func = &module.funcs[function_idx];
        if func.blocks.is_empty()
            || func.blocks.len() > MAX_BLOCKS
            || func.values.len() > MAX_VALUES
        {
            continue;
        }
        let Some(instruction_count) = func
            .blocks
            .iter()
            .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
        else {
            return;
        };
        let Some(structural_work) = func
            .blocks
            .len()
            .checked_mul(func.blocks.len())
            .and_then(|work| work.checked_add(func.values.len()))
            .and_then(|work| work.checked_add(instruction_count))
        else {
            return;
        };
        if budget.spend(structural_work).is_none() {
            return;
        }
        if func.verify().is_err() {
            continue;
        }

        let cfg = ControlFlowGraph::new(func);
        let dom = Dominators::new(func, &cfg);
        let loop_info = LoopInfo::new(&cfg, &dom);
        for (loop_idx, natural_loop) in loop_info.loops().iter().enumerate() {
            let Some(nesting_work) = loop_info
                .loops()
                .len()
                .checked_mul(natural_loop.blocks.len())
            else {
                return;
            };
            if budget.spend(nesting_work).is_none() {
                return;
            }
            if loop_info
                .loops()
                .iter()
                .enumerate()
                .any(|(other_idx, other)| {
                    other_idx != loop_idx
                        && natural_loop
                            .blocks
                            .iter()
                            .any(|block| other.blocks.contains(block))
                })
            {
                continue;
            }
            let Some(candidate) =
                match_candidate(module, func, &cfg, &dom, natural_loop, &mut budget)
            else {
                continue;
            };

            let range_symbol = format!("__yuezhuo_parallel_range_{}", function_idx);
            let worker_symbol = format!("__yuezhuo_parallel_worker_{}", function_idx);
            let context_symbol = format!("__yuezhuo_parallel_context_{}", function_idx);
            if [&range_symbol, &worker_symbol, &context_symbol]
                .into_iter()
                .any(|symbol| reserved_symbols.contains(symbol))
            {
                continue;
            }
            let Some(helper) = build_range_helper(func, &candidate, range_symbol.clone()) else {
                continue;
            };
            let Some(next_added_values) = added_values.checked_add(helper.values.len()) else {
                return;
            };
            if next_added_values > MAX_ADDED_VALUES {
                continue;
            }
            added_values = next_added_values;
            reserved_symbols.insert(range_symbol);
            reserved_symbols.insert(worker_symbol.clone());
            reserved_symbols.insert(context_symbol.clone());
            // SysY has no user-created threads. While this per-site context is
            // active, both lanes execute this call-free helper and the parent
            // cannot continue (or recurse) until join/completion. Consequently
            // another activation cannot overlap this static context.
            pending.push(PendingOutline {
                parent: FunctionId(function_idx),
                candidate,
                helper,
                context_symbol,
                worker_symbol,
            });
            // At most one candidate per source function.
            break;
        }
    }

    if module.funcs.len().checked_add(pending.len()).is_none() {
        return;
    }
    for outline in pending {
        let helper = module.add_func(outline.helper);
        let candidate = outline.candidate;
        module.aarch64_thread_plans.push(AArch64ThreadPlan {
            parent: outline.parent,
            helper,
            preheader: candidate.preheader,
            header: candidate.header,
            body: candidate.body,
            exit: candidate.exit,
            bound: candidate.bound,
            dispatch_setup: candidate.dispatch_setup,
            captures: candidate.captures,
            context_symbol: outline.context_symbol,
            worker_symbol: outline.worker_symbol,
        });
    }
}

fn match_candidate(
    module: &Module,
    func: &Function,
    cfg: &ControlFlowGraph,
    dom: &Dominators,
    natural_loop: &NaturalLoop,
    budget: &mut WorkBudget,
) -> Option<Candidate> {
    budget.spend(func.blocks.len())?;
    if natural_loop.blocks.len() != 2 {
        return None;
    }
    let preheader = natural_loop.dedicated_preheader?;
    let header = natural_loop.header;
    let body = natural_loop.unique_latch()?;
    let exit = natural_loop.unique_exit()?;
    if body == header
        || natural_loop.exit_edges.as_slice() != [(header, exit)]
        || func.blocks.get(preheader.0)?.terminator != Some(Terminator::Jump(header))
        || func.blocks.get(body.0)?.terminator != Some(Terminator::Jump(header))
        || cfg.preds.get(body.0)?.as_slice() != [header]
    {
        return None;
    }
    let header_predecessors = cfg.preds.get(header.0)?;
    if header_predecessors.len() != 2
        || !header_predecessors.contains(&preheader)
        || !header_predecessors.contains(&body)
        || func
            .blocks
            .get(exit.0)?
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Phi { .. }))
    {
        return None;
    }

    let header_block = func.blocks.get(header.0)?;
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = header_block.terminator.as_ref()?
    else {
        return None;
    };
    if *then_target != body || *else_target != exit {
        return None;
    }

    let mut counter = None;
    let mut condition = None;
    let mut non_condition_values = Vec::new();
    for inst in &header_block.insts {
        budget.spend(1)?;
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { .. } => {
                if counter.replace(inst.result?).is_some() {
                    return None;
                }
            }
            InstKind::Icmp {
                op: CmpOp::Lt,
                lhs,
                rhs,
            } if inst.result == Some(*cond) => {
                if condition.replace((*lhs, *rhs)).is_some() {
                    return None;
                }
            }
            _ => non_condition_values.push(inst.result?),
        }
    }
    let counter = counter?;
    let (condition_counter, bound) = condition?;
    if condition_counter != counter
        || func.value(counter).ty != Type::I32
        || func.value(bound).ty != Type::I32
        || func.value(*cond).ty != Type::I1
    {
        return None;
    }
    let induction = analyze_i32_induction(func, natural_loop, counter)?;
    if induction.step != 1
        || const_i32(func, induction.initial) != Some(0)
        || !matches!(
            func.value(induction.next).kind,
            ValueKind::Inst(owner, _) if owner == body
        )
    {
        return None;
    }

    let uses = collect_uses(func, budget)?;
    let counter_uses = uses.get(counter.0)?;
    if counter_uses
        .iter()
        .any(|site| site.block != header && site.block != body)
        || counter_uses
            .iter()
            .filter(|site| site.block == header)
            .count()
            != 1
    {
        // The successful path skips the scalar header/body entirely. Until a
        // final-counter merge is represented explicitly, no outside/header
        // use beyond the loop condition may observe that skipped phi state.
        return None;
    }
    let (captures, memory_root) = analyze_body(
        module,
        func,
        dom,
        body,
        header,
        preheader,
        counter,
        induction.next,
        &uses,
        budget,
    )?;

    let dispatch_setup = if value_available_at(func, dom, bound, preheader) {
        if !non_condition_values.is_empty() {
            return None;
        }
        Vec::new()
    } else {
        analyze_header_setup(
            module,
            func,
            header,
            bound,
            &non_condition_values,
            &uses,
            memory_root,
            budget,
        )?
    };

    Some(Candidate {
        preheader,
        header,
        body,
        exit,
        counter,
        counter_next: induction.next,
        bound,
        dispatch_setup,
        captures,
    })
}

#[allow(clippy::too_many_arguments)]
fn analyze_body(
    module: &Module,
    func: &Function,
    dom: &Dominators,
    body: BlockId,
    header: BlockId,
    preheader: BlockId,
    counter: ValueId,
    counter_next: ValueId,
    uses: &[Vec<UseSite>],
    budget: &mut WorkBudget,
) -> Option<(Vec<AArch64ThreadCapture>, ValueId)> {
    let block = func.blocks.get(body.0)?;
    let active = block
        .insts
        .iter()
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count();
    if active == 0 || active > MAX_ACTIVE_BODY_INSTS {
        return None;
    }

    let body_results = block
        .insts
        .iter()
        .filter_map(|inst| inst.result)
        .collect::<HashSet<_>>();
    let mut defined = HashSet::new();
    let mut memory_count = 0usize;
    let mut has_store = false;
    let mut memory_root = None;
    let mut body_cost = 0usize;

    for (inst_idx, inst) in block.insts.iter().enumerate() {
        budget.spend(1)?;
        if matches!(inst.kind, InstKind::Nop) {
            continue;
        }
        for operand in inst_operands(&inst.kind) {
            budget.spend(1)?;
            if matches!(func.value(operand).kind, ValueKind::Inst(owner, owner_idx)
                if owner == body && (owner_idx >= inst_idx || !defined.contains(&operand)))
            {
                return None;
            }
        }
        validate_body_inst(func, inst)?;
        if let Some(result) = inst.result {
            defined.insert(result);
        }
        body_cost = body_cost.checked_add(instruction_cost(inst, counter_next))?;

        match &inst.kind {
            InstKind::Load { ptr } => {
                memory_count = memory_count.checked_add(1)?;
                check_memory_pointer(func, body, counter, *ptr, uses, &mut memory_root)?;
                let Type::Ptr(pointee) = &func.value(*ptr).ty else {
                    return None;
                };
                if **pointee != func.value(inst.result?).ty || !is_four_byte_scalar(pointee) {
                    return None;
                }
            }
            InstKind::Store { ptr, value } => {
                memory_count = memory_count.checked_add(1)?;
                has_store = true;
                check_memory_pointer(func, body, counter, *ptr, uses, &mut memory_root)?;
                let Type::Ptr(pointee) = &func.value(*ptr).ty else {
                    return None;
                };
                if **pointee != func.value(*value).ty || !is_four_byte_scalar(pointee) {
                    return None;
                }
            }
            InstKind::Gep { .. } => {
                let result = inst.result?;
                if uses.get(result.0)?.is_empty()
                    || uses[result.0].iter().any(|site| {
                        site.block != body
                            || !matches!(site.role, UseRole::LoadPointer | UseRole::StorePointer)
                    })
                {
                    return None;
                }
            }
            _ => {}
        }
    }
    if !has_store || memory_count > MAX_MEMORY_INSTS || body_cost < MIN_ACTIVE_BODY_COST {
        return None;
    }

    for result in &body_results {
        for site in uses.get(result.0)? {
            budget.spend(1)?;
            if site.block == body {
                continue;
            }
            if *result != counter_next
                || site.role
                    != (UseRole::Phi {
                        predecessor: body,
                        target: header,
                    })
            {
                return None;
            }
        }
    }
    let next_uses = uses.get(counter_next.0)?;
    if next_uses.iter().filter(|site| site.block != body).count() != 1 {
        return None;
    }

    let root = memory_root?;
    validate_memory_root(module, func, dom, root, preheader)?;
    let mut captures = Vec::new();
    let mut captured = HashSet::new();
    for inst in &block.insts {
        for operand in inst_operands(&inst.kind) {
            if operand == counter || body_results.contains(&operand) {
                continue;
            }
            match &func.value(operand).kind {
                ValueKind::Const(_) | ValueKind::Global(_) => continue,
                ValueKind::Inst(owner, _) if *owner == header => return None,
                _ => {}
            }
            if !value_available_at(func, dom, operand, preheader)
                || !matches!(func.value(operand).ty, Type::I32 | Type::Ptr(_))
            {
                return None;
            }
            if captured.insert(operand) {
                captures.push(AArch64ThreadCapture {
                    value: operand,
                    ty: func.value(operand).ty.clone(),
                });
                if captures.len() > MAX_CAPTURES {
                    return None;
                }
            }
        }
    }
    Some((captures, root))
}

fn validate_body_inst(func: &Function, inst: &Inst) -> Option<()> {
    match &inst.kind {
        InstKind::Nop => {}
        InstKind::Load { .. }
        | InstKind::Unary { .. }
        | InstKind::Icmp { .. }
        | InstKind::Fcmp { .. }
        | InstKind::Cast { .. }
        | InstKind::Gep { .. } => {
            inst.result?;
        }
        InstKind::Store { .. } => {
            if inst.result.is_some() {
                return None;
            }
        }
        InstKind::Binary { op, rhs, .. } => {
            inst.result?;
            if *op == BinaryOp::Fdiv
                || (matches!(op, BinaryOp::Idiv | BinaryOp::Imod)
                    && const_i32(func, *rhs).is_none_or(|divisor| divisor == 0))
            {
                return None;
            }
        }
        InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::MemZero { .. }
        | InstKind::Call { .. } => return None,
    }
    Some(())
}

fn check_memory_pointer(
    func: &Function,
    body: BlockId,
    counter: ValueId,
    pointer: ValueId,
    uses: &[Vec<UseSite>],
    memory_root: &mut Option<ValueId>,
) -> Option<()> {
    let ValueKind::Inst(owner, inst_idx) = func.value(pointer).kind else {
        return None;
    };
    if owner != body {
        return None;
    }
    let inst = func.blocks.get(owner.0)?.insts.get(inst_idx)?;
    let InstKind::Gep { base, indices } = &inst.kind else {
        return None;
    };
    if inst.result != Some(pointer)
        || indices.as_slice() != [counter]
        || !matches!(func.value(pointer).ty, Type::Ptr(ref pointee) if is_four_byte_scalar(pointee))
        || uses.get(pointer.0)?.iter().any(|site| {
            site.block != body || !matches!(site.role, UseRole::LoadPointer | UseRole::StorePointer)
        })
    {
        return None;
    }
    if memory_root.is_some_and(|root| !roots_are_identical(func, root, *base)) {
        return None;
    }
    if memory_root.is_none() {
        *memory_root = Some(*base);
    }
    Some(())
}

fn roots_are_identical(func: &Function, lhs: ValueId, rhs: ValueId) -> bool {
    lhs == rhs
        || matches!(
            (&func.value(lhs).kind, &func.value(rhs).kind),
            (ValueKind::Global(lhs_name), ValueKind::Global(rhs_name)) if lhs_name == rhs_name
        )
}

fn validate_memory_root(
    module: &Module,
    func: &Function,
    dom: &Dominators,
    root: ValueId,
    preheader: BlockId,
) -> Option<()> {
    if !matches!(func.value(root).ty, Type::Ptr(_))
        || !value_available_at(func, dom, root, preheader)
    {
        return None;
    }
    match &func.value(root).kind {
        ValueKind::Param => Some(()),
        ValueKind::Inst(_, _) => Some(()),
        ValueKind::Global(name) => (module
            .globals
            .iter()
            .filter(|global| global.name == *name)
            .count()
            == 1)
            .then_some(()),
        ValueKind::Const(_) => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze_header_setup(
    module: &Module,
    func: &Function,
    header: BlockId,
    bound: ValueId,
    setup_values: &[ValueId],
    uses: &[Vec<UseSite>],
    memory_root: ValueId,
    budget: &mut WorkBudget,
) -> Option<Vec<ValueId>> {
    let setup_set = setup_values.iter().copied().collect::<HashSet<_>>();
    if setup_set.len() != setup_values.len() || !setup_set.contains(&bound) {
        return None;
    }

    let mut closure = HashSet::new();
    let mut stack = vec![bound];
    while let Some(value) = stack.pop() {
        budget.spend(1)?;
        if !setup_set.contains(&value) || !closure.insert(value) {
            continue;
        }
        let inst = defining_inst(func, value)?;
        stack.extend(inst_operands(&inst.kind));
    }
    if closure != setup_set {
        return None;
    }

    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    let mut loaded_global = None::<String>;
    for inst in &func.blocks.get(header.0)?.insts {
        let Some(result) = inst.result.filter(|result| setup_set.contains(result)) else {
            continue;
        };
        budget.spend(1)?;
        if inst_operands(&inst.kind)
            .into_iter()
            .any(|operand| setup_set.contains(&operand) && !seen.contains(&operand))
        {
            return None;
        }
        match &inst.kind {
            InstKind::Load { ptr } => {
                if loaded_global.is_some() || func.value(result).ty != Type::I32 {
                    return None;
                }
                let ValueKind::Global(name) = &func.value(*ptr).kind else {
                    return None;
                };
                if module
                    .globals
                    .iter()
                    .filter(|global| global.name == *name && global.ty == Type::I32)
                    .count()
                    != 1
                {
                    return None;
                }
                loaded_global = Some(name.clone());
            }
            InstKind::Unary {
                op: UnaryOp::Ineg | UnaryOp::Not,
                ..
            }
            | InstKind::Binary {
                op:
                    BinaryOp::Iadd
                    | BinaryOp::Isub
                    | BinaryOp::Imul
                    | BinaryOp::Iand
                    | BinaryOp::Ior
                    | BinaryOp::Ixor
                    | BinaryOp::Ishl
                    | BinaryOp::Iashr,
                ..
            }
            | InstKind::Icmp { .. }
            | InstKind::Cast {
                op: CastOp::BoolToI32 | CastOp::I32ToBool,
                ..
            } => {}
            InstKind::Binary {
                op: BinaryOp::Idiv | BinaryOp::Imod,
                rhs,
                ..
            } if const_i32(func, *rhs).is_some_and(|divisor| divisor != 0) => {}
            _ => return None,
        }
        seen.insert(result);
        ordered.push(result);
    }
    let loaded_global = loaded_global?;
    if ordered.len() != setup_values.len()
        || setup_values.iter().any(|value| {
            uses.get(value.0)
                .is_none_or(|sites| sites.iter().any(|site| site.block != header))
        })
    {
        return None;
    }

    // A copied bound load may run concurrently with the worker. For this MVP,
    // prove separation by requiring the loop's only memory root to be a
    // distinct, uniquely defined global object. Unknown pointer/global aliasing
    // is deliberately not inferred from source-language conventions.
    let ValueKind::Global(root_name) = &func.value(memory_root).kind else {
        return None;
    };
    if *root_name == loaded_global
        || module
            .globals
            .iter()
            .filter(|global| global.name == *root_name)
            .count()
            != 1
    {
        return None;
    }
    Some(ordered)
}

fn build_range_helper(func: &Function, candidate: &Candidate, name: String) -> Option<Function> {
    let mut helper = Function::new(name, Type::Void);
    let begin = helper.add_param("begin", Type::I32);
    let end = helper.add_param("end", Type::I32);
    let mut values = HashMap::from([(candidate.counter, begin)]);
    for (idx, capture) in candidate.captures.iter().enumerate() {
        let parameter = helper.add_param(format!("capture{}", idx), capture.ty.clone());
        values.insert(capture.value, parameter);
    }

    let header = helper.add_block("range.header");
    let body = helper.add_block("range.body");
    let exit = helper.add_block("range.exit");
    helper.set_terminator(helper.entry, Terminator::Jump(header));
    let counter = helper.append_inst(
        header,
        InstKind::Phi {
            incomings: vec![(helper.entry, begin), (body, begin)],
        },
        Some(Type::I32),
    )?;
    values.insert(candidate.counter, counter);
    let condition = helper.append_inst(
        header,
        InstKind::Icmp {
            op: CmpOp::Lt,
            lhs: counter,
            rhs: end,
        },
        Some(Type::I1),
    )?;
    helper.set_terminator(
        header,
        Terminator::Branch {
            cond: condition,
            then_target: body,
            else_target: exit,
        },
    );

    for inst in &func.blocks.get(candidate.body.0)?.insts {
        if matches!(inst.kind, InstKind::Nop) {
            continue;
        }
        let kind = remap_inst(func, &mut helper, &mut values, &inst.kind)?;
        let result_ty = inst.result.map(|result| func.value(result).ty.clone());
        let cloned = helper.append_inst(body, kind, result_ty);
        if let Some(original) = inst.result {
            values.insert(original, cloned?);
        }
    }
    let next = *values.get(&candidate.counter_next)?;
    helper.set_terminator(body, Terminator::Jump(header));
    helper.set_terminator(exit, Terminator::Return(None));
    let InstKind::Phi { incomings } = &mut helper.blocks[header.0].insts[0].kind else {
        return None;
    };
    incomings[1].1 = next;
    helper.verify().ok()?;
    Some(helper)
}

fn remap_inst(
    source: &Function,
    helper: &mut Function,
    values: &mut HashMap<ValueId, ValueId>,
    kind: &InstKind,
) -> Option<InstKind> {
    let mut map = |value| map_value(source, helper, values, value);
    Some(match kind {
        InstKind::Load { ptr } => InstKind::Load { ptr: map(*ptr)? },
        InstKind::Store { ptr, value } => InstKind::Store {
            ptr: map(*ptr)?,
            value: map(*value)?,
        },
        InstKind::Unary { op, value } => InstKind::Unary {
            op: *op,
            value: map(*value)?,
        },
        InstKind::Binary { op, lhs, rhs } => InstKind::Binary {
            op: *op,
            lhs: map(*lhs)?,
            rhs: map(*rhs)?,
        },
        InstKind::Icmp { op, lhs, rhs } => InstKind::Icmp {
            op: *op,
            lhs: map(*lhs)?,
            rhs: map(*rhs)?,
        },
        InstKind::Fcmp { op, lhs, rhs } => InstKind::Fcmp {
            op: *op,
            lhs: map(*lhs)?,
            rhs: map(*rhs)?,
        },
        InstKind::Cast { op, value } => InstKind::Cast {
            op: *op,
            value: map(*value)?,
        },
        InstKind::Gep { base, indices } => InstKind::Gep {
            base: map(*base)?,
            indices: indices
                .iter()
                .map(|index| map(*index))
                .collect::<Option<Vec<_>>>()?,
        },
        InstKind::Nop
        | InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::MemZero { .. }
        | InstKind::Call { .. } => return None,
    })
}

fn map_value(
    source: &Function,
    helper: &mut Function,
    values: &mut HashMap<ValueId, ValueId>,
    value: ValueId,
) -> Option<ValueId> {
    if let Some(mapped) = values.get(&value) {
        return Some(*mapped);
    }
    let mapped = match &source.value(value).kind {
        ValueKind::Const(constant) => helper.add_const(constant.clone()),
        ValueKind::Global(name) => {
            helper.add_global_ref(name.clone(), source.value(value).ty.clone())
        }
        ValueKind::Param | ValueKind::Inst(_, _) => return None,
    };
    values.insert(value, mapped);
    Some(mapped)
}

fn collect_uses(func: &Function, budget: &mut WorkBudget) -> Option<Vec<Vec<UseSite>>> {
    let mut uses = vec![Vec::new(); func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let owner = BlockId(block_idx);
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings } => {
                    for (predecessor, value) in incomings {
                        budget.spend(1)?;
                        uses.get_mut(value.0)?.push(UseSite {
                            block: owner,
                            role: UseRole::Phi {
                                predecessor: *predecessor,
                                target: owner,
                            },
                        });
                    }
                }
                InstKind::Load { ptr } => {
                    budget.spend(1)?;
                    uses.get_mut(ptr.0)?.push(UseSite {
                        block: owner,
                        role: UseRole::LoadPointer,
                    });
                }
                InstKind::Store { ptr, value } => {
                    budget.spend(2)?;
                    uses.get_mut(ptr.0)?.push(UseSite {
                        block: owner,
                        role: UseRole::StorePointer,
                    });
                    uses.get_mut(value.0)?.push(UseSite {
                        block: owner,
                        role: UseRole::Other,
                    });
                }
                kind => {
                    for value in inst_operands(kind) {
                        budget.spend(1)?;
                        uses.get_mut(value.0)?.push(UseSite {
                            block: owner,
                            role: UseRole::Other,
                        });
                    }
                }
            }
        }
        for value in terminator_operands(block.terminator.as_ref()) {
            budget.spend(1)?;
            uses.get_mut(value.0)?.push(UseSite {
                block: owner,
                role: UseRole::Other,
            });
        }
    }
    Some(uses)
}

fn value_available_at(func: &Function, dom: &Dominators, value: ValueId, block: BlockId) -> bool {
    match func.values.get(value.0).map(|value| &value.kind) {
        Some(ValueKind::Param | ValueKind::Const(_) | ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(owner, inst_idx)) => {
            func.blocks
                .get(owner.0)
                .and_then(|block| block.insts.get(*inst_idx))
                .is_some_and(|inst| inst.result == Some(value))
                && (*owner == block || dom.dominates(*owner, block))
        }
        None => false,
    }
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&Inst> {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(inst)
}

fn const_i32(func: &Function, value: ValueId) -> Option<i32> {
    match &func.values.get(value.0)?.kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        _ => None,
    }
}

fn is_four_byte_scalar(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::F32)
}

fn instruction_cost(inst: &Inst, counter_next: ValueId) -> usize {
    if inst.result == Some(counter_next) {
        return 0;
    }
    match inst.kind {
        InstKind::Load { .. } | InstKind::Store { .. } => 3,
        InstKind::Binary {
            op: BinaryOp::Idiv | BinaryOp::Imod,
            ..
        } => 8,
        InstKind::Binary {
            op: BinaryOp::Imul | BinaryOp::Fmul,
            ..
        } => 3,
        InstKind::Gep { .. } => 1,
        InstKind::Nop => 0,
        _ => 1,
    }
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

fn terminator_operands(terminator: Option<&Terminator>) -> Vec<ValueId> {
    match terminator {
        Some(Terminator::Return(Some(value))) => vec![*value],
        Some(Terminator::Branch { cond, .. }) => vec![*cond],
        Some(Terminator::Return(None) | Terminator::Jump(_)) | None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::pass::{run_pipeline, OptLevel, PassOptions};
    use crate::parser::Parser;

    fn optimize(source: &str, enabled: bool) -> Module {
        let mut parser = Parser::new(source);
        let program = parser.parse_program();
        let mut module = crate::ir::lower::lower_program(&program).unwrap();
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_aarch64_threading: enabled,
            },
        );
        module
    }

    const SAFE_MAP: &str = r#"
        int transform(int a[], int n) {
            int i = 0;
            while (i < n) {
                int x = a[i];
                a[i] = x * x + x * 3 + 1;
                i = i + 1;
            }
            return 0;
        }
    "#;

    #[test]
    fn accepts_exact_same_root_element_mapping_and_verifies_helper() {
        let module = optimize(SAFE_MAP, true);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        let plan = &module.aarch64_thread_plans[0];
        assert_eq!(plan.captures.len(), 1);
        assert_eq!(plan.captures[0].ty, Type::Ptr(Box::new(Type::I32)));
        assert!(module.funcs[plan.helper.0].verify().is_ok());
    }

    #[test]
    fn accepts_bounded_header_global_setup_for_a_distinct_global_root() {
        let module = optimize(
            r#"
                int side;
                int data[1000000];
                int square(int x) { return x * x; }
                int transform() {
                    int i = 0;
                    while (i < square(side)) {
                        int x = data[i];
                        data[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert_eq!(module.aarch64_thread_plans[0].dispatch_setup.len(), 2);
        assert!(module.funcs[module.aarch64_thread_plans[0].helper.0]
            .verify()
            .is_ok());
    }

    #[test]
    fn accepts_repeated_refs_to_one_unique_global_root() {
        let module = optimize(
            r#"
                int data[10000];
                int transform(int n) {
                    int i = 0;
                    while (i < n) {
                        int x = data[i];
                        data[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert_eq!(module.aarch64_thread_plans.len(), 1);
        assert!(module.aarch64_thread_plans[0].captures.is_empty());
    }

    #[test]
    fn rejects_generated_symbol_collisions() {
        let module = optimize(
            r#"
                int __yuezhuo_parallel_context_0;
                int transform(int a[], int n) {
                    int i = 0;
                    while (i < n) {
                        int x = a[i];
                        a[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert!(module.aarch64_thread_plans.is_empty());
    }

    #[test]
    fn rejects_unknown_alias_between_input_and_output_roots() {
        let module = optimize(
            r#"
                int transform(int a[], int b[], int n) {
                    int i = 0;
                    while (i < n) {
                        int x = b[i];
                        a[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert!(module.aarch64_thread_plans.is_empty());
    }

    #[test]
    fn rejects_offset_index() {
        let module = optimize(
            r#"
                int transform(int a[], int n) {
                    int i = 1;
                    while (i < n) {
                        int x = a[i - 1];
                        a[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            true,
        );
        assert!(module.aarch64_thread_plans.is_empty());
    }

    #[test]
    fn rejects_division_without_a_safe_integer_literal() {
        for source in [
            r#"
                int transform(int a[], int n) {
                    int i = 0;
                    while (i < n) {
                        int x = a[i];
                        a[i] = x / n + x * x + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            r#"
                int transform(float a[], int n) {
                    int i = 0;
                    while (i < n) {
                        float x = a[i];
                        a[i] = x / 3.0 + x * x + 1.0;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
        ] {
            assert!(optimize(source, true).aarch64_thread_plans.is_empty());
        }
    }

    #[test]
    fn rejects_calls_reductions_and_liveouts() {
        for source in [
            r#"
                int transform(int a[], int n) {
                    int i = 0;
                    while (i < n) {
                        int x = a[i];
                        putint(x);
                        a[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return 0;
                }
            "#,
            r#"
                int transform(int a[], int n) {
                    int i = 0;
                    int sum = 0;
                    while (i < n) {
                        sum = sum + a[i];
                        a[i] = a[i] * a[i] + 1;
                        i = i + 1;
                    }
                    return sum;
                }
            "#,
            r#"
                int transform(int a[], int n) {
                    int i = 0;
                    int last = 0;
                    while (i < n) {
                        int x = a[i];
                        last = x * x + x * 3 + 1;
                        a[i] = last;
                        i = i + 1;
                    }
                    return last;
                }
            "#,
            r#"
                int transform(int a[], int n) {
                    int i = 0;
                    while (i < n) {
                        int x = a[i];
                        a[i] = x * x + x * 3 + 1;
                        i = i + 1;
                    }
                    return i;
                }
            "#,
        ] {
            assert!(optimize(source, true).aarch64_thread_plans.is_empty());
        }
    }

    #[test]
    fn requires_aarch64_o1_gate_and_is_idempotent() {
        assert!(optimize(SAFE_MAP, false).aarch64_thread_plans.is_empty());

        let mut parser = Parser::new(SAFE_MAP);
        let mut o0 = crate::ir::lower::lower_program(&parser.parse_program()).unwrap();
        run_pipeline(
            &mut o0,
            OptLevel::O0,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_aarch64_threading: true,
            },
        );
        assert!(o0.aarch64_thread_plans.is_empty());

        let mut module = optimize(SAFE_MAP, true);
        let functions = module.funcs.len();
        AArch64ThreadOutlinePass::new().run(&mut module);
        assert_eq!(module.funcs.len(), functions);
        assert_eq!(module.aarch64_thread_plans.len(), 1);
    }
}
