use super::dominators::{ControlFlowGraph, Dominators};
use super::loop_analysis::{
    analyze_const_i32_trip_count, analyze_i32_induction, InductionVariable, LoopInfo, NaturalLoop,
};
use super::util::{const_i32, defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Function, FunctionId, InstKind, Module, Terminator, Type, ValueId,
    ValueKind,
};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTIONS: usize = 4096;
const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_INSTRUCTIONS: usize = 32_768;
const MAX_POINTER_DEPTH: usize = 64;

/// Executes only the final iteration of a repeated deterministic overwrite.
///
/// The accepted shape first copies a complete two-dimensional rectangle from
/// a stable source object into a distinct destination, then calls one
/// closed-world function. The callee may read and update only that destination
/// rectangle and may otherwise only read distinct pointer arguments. Because
/// every iteration recreates the same complete input state, all but the final
/// iteration are unobservable.
pub(super) struct RepeatedOverwritePass;

impl RepeatedOverwritePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RepeatedOverwritePass {
    fn run(&mut self, module: &mut Module) {
        if module.funcs.len() > MAX_FUNCTIONS {
            return;
        }
        let snapshot = module.funcs.clone();
        let targets = unique_targets(&snapshot);
        for func_idx in 0..module.funcs.len() {
            let Some(candidate) = find_candidate(&snapshot[func_idx], &snapshot, &targets) else {
                continue;
            };
            apply_candidate(&mut module.funcs[func_idx], candidate);
            if let Err(errors) = module.funcs[func_idx].verify() {
                panic!(
                    "repeated overwrite elision produced invalid IR in {}: {:?}",
                    module.funcs[func_idx].name, errors
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    counter: ValueId,
    entering: BlockId,
    final_initial: i32,
}

#[derive(Clone, Copy)]
struct MatrixCopy {
    destination_root: ValueId,
    source_root: ValueId,
    bound: ValueId,
}

#[derive(Clone)]
struct CallSite {
    block: BlockId,
    name: String,
    args: Vec<ValueId>,
}

fn find_candidate(
    caller: &Function,
    funcs: &[Function],
    targets: &HashMap<String, Option<FunctionId>>,
) -> Option<Candidate> {
    if exceeds_budget(caller) {
        return None;
    }
    let cfg = ControlFlowGraph::new(caller);
    let dom = Dominators::new(caller, &cfg);
    let loops = LoopInfo::new(&cfg, &dom).loops().to_vec();

    let mut outer_loops = loops.iter().collect::<Vec<_>>();
    outer_loops.sort_by_key(|natural_loop| usize::MAX - natural_loop.blocks.len());
    for outer in outer_loops {
        let (Some(entering), Some(latch)) = (outer.unique_entering_pred, outer.unique_latch())
        else {
            continue;
        };
        let Some((counter, condition, induction, trip_count)) = counted_outer_loop(caller, outer)
        else {
            continue;
        };
        if trip_count < 2
            || header_phis(caller, outer.header) != [counter]
            || !counter_is_control_only(caller, counter, condition, induction.next)
        {
            continue;
        }

        let calls = calls_in_loop(caller, outer);
        let [call] = calls.as_slice() else {
            continue;
        };
        if call.block != latch {
            continue;
        }
        let Some(callee_id) = targets.get(&call.name).copied().flatten() else {
            continue;
        };
        let Some(callee) = funcs.get(callee_id.0) else {
            continue;
        };
        let Some(matrix_copy) = complete_matrix_copy(caller, outer, &loops, call, &dom) else {
            continue;
        };
        let Some(destination_param) = call.args.iter().position(|arg| {
            pointer_root(caller, *arg)
                .is_some_and(|root| same_memory_root(caller, root, matrix_copy.destination_root))
        }) else {
            continue;
        };
        let Some(bound_param) = call.args.iter().position(|arg| *arg == matrix_copy.bound) else {
            continue;
        };
        if !stable_call_arguments(
            caller,
            outer,
            call,
            matrix_copy.destination_root,
            matrix_copy.source_root,
        ) || !callee_fits_complete_overwrite(callee, destination_param, bound_param)
        {
            continue;
        }

        let initial = i64::from(const_i32(caller, induction.initial)?);
        let distance = i64::from(induction.step).checked_mul(i64::from(trip_count - 1))?;
        let final_initial = i32::try_from(initial.checked_add(distance)?).ok()?;
        return Some(Candidate {
            counter,
            entering,
            final_initial,
        });
    }
    None
}

fn counted_outer_loop(
    func: &Function,
    natural_loop: &NaturalLoop,
) -> Option<(ValueId, ValueId, InductionVariable, u32)> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func
        .blocks
        .get(natural_loop.header.0)?
        .terminator
        .as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || natural_loop.blocks.contains(else_target) {
        return None;
    }
    if natural_loop.exit_edges.as_slice() != [(natural_loop.header, *else_target)] {
        return None;
    }
    let InstKind::Icmp {
        op: CmpOp::Lt, lhs, ..
    } = defining_inst(func, *cond)?
    else {
        return None;
    };
    let induction = analyze_i32_induction(func, natural_loop, *lhs)?;
    let trip_count = analyze_const_i32_trip_count(func, natural_loop, induction)?;
    Some((induction.phi, *cond, induction, trip_count))
}

fn complete_matrix_copy(
    func: &Function,
    outer: &NaturalLoop,
    loops: &[NaturalLoop],
    call: &CallSite,
    dom: &Dominators,
) -> Option<MatrixCopy> {
    let mut stores = Vec::new();
    for block in &outer.blocks {
        for inst in &func.blocks[block.0].insts {
            match &inst.kind {
                InstKind::Store { ptr, value } => stores.push((*block, *ptr, *value)),
                InstKind::MemZero { .. } => return None,
                _ => {}
            }
        }
    }
    let [(store_block, destination, value)] = stores.as_slice() else {
        return None;
    };
    let InstKind::Load { ptr: source } = defining_inst(func, *value)? else {
        return None;
    };
    let (destination_root, destination_indices) = pointer_path(func, *destination)?;
    let (source_root, source_indices) = pointer_path(func, *source)?;
    if same_memory_root(func, destination_root, source_root)
        || destination_indices.len() != 2
        || source_indices != destination_indices
        || !stable_distinct_memory_root(func, destination_root, outer)
        || !stable_distinct_memory_root(func, source_root, outer)
    {
        return None;
    }

    let row = zero_based_bounded_loop(func, loops, destination_indices[0])?;
    let column = zero_based_bounded_loop(func, loops, destination_indices[1])?;
    let row_latch = row.natural_loop.unique_latch()?;
    if row.bound != column.bound
        || !outer.blocks.is_superset(&row.natural_loop.blocks)
        || !row
            .natural_loop
            .blocks
            .is_superset(&column.natural_loop.blocks)
        || !column.natural_loop.blocks.contains(store_block)
        || column.natural_loop.unique_latch() != Some(*store_block)
        || row.natural_loop.unique_exit() != Some(call.block)
        || column.natural_loop.unique_exit() != Some(row_latch)
        || row.natural_loop.exit_edges.as_slice() != [(row.natural_loop.header, call.block)]
        || column.natural_loop.exit_edges.as_slice() != [(column.natural_loop.header, row_latch)]
        || !dom.dominates(row.natural_loop.header, column.natural_loop.header)
        || !dom.dominates(column.natural_loop.header, row_latch)
        || !dom.dominates(row.natural_loop.header, call.block)
    {
        return None;
    }

    for block in &outer.blocks {
        for inst in &func.blocks[block.0].insts {
            if let InstKind::Load { ptr } = inst.kind {
                if pointer_root(func, ptr)
                    .is_none_or(|root| !same_memory_root(func, root, source_root))
                {
                    return None;
                }
            }
        }
    }
    Some(MatrixCopy {
        destination_root,
        source_root,
        bound: row.bound,
    })
}

struct BoundedLoop<'a> {
    natural_loop: &'a NaturalLoop,
    bound: ValueId,
}

fn zero_based_bounded_loop<'a>(
    func: &Function,
    loops: &'a [NaturalLoop],
    phi: ValueId,
) -> Option<BoundedLoop<'a>> {
    for natural_loop in loops {
        let Some(induction) = analyze_i32_induction(func, natural_loop, phi) else {
            continue;
        };
        if induction.step != 1 || const_i32(func, induction.initial) != Some(0) {
            continue;
        }
        let Some(bound) = direct_less_than_bound(func, natural_loop, phi) else {
            continue;
        };
        return Some(BoundedLoop {
            natural_loop,
            bound,
        });
    }
    None
}

fn direct_less_than_bound(
    func: &Function,
    natural_loop: &NaturalLoop,
    induction: ValueId,
) -> Option<ValueId> {
    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func
        .blocks
        .get(natural_loop.header.0)?
        .terminator
        .as_ref()?
    else {
        return None;
    };
    if !natural_loop.blocks.contains(then_target) || natural_loop.blocks.contains(else_target) {
        return None;
    }
    match defining_inst(func, *cond)? {
        InstKind::Icmp {
            op: CmpOp::Lt,
            lhs,
            rhs,
        } if *lhs == induction => Some(*rhs),
        _ => None,
    }
}

fn stable_call_arguments(
    func: &Function,
    outer: &NaturalLoop,
    call: &CallSite,
    destination_root: ValueId,
    source_root: ValueId,
) -> bool {
    if call
        .args
        .iter()
        .any(|arg| value_defined_in_loop(func, outer, *arg))
    {
        return false;
    }
    let mut destination_arguments = 0usize;
    for arg in &call.args {
        if !matches!(
            func.values.get(arg.0).map(|value| &value.ty),
            Some(Type::Ptr(_))
        ) {
            continue;
        }
        let Some(root) = pointer_root(func, *arg) else {
            return false;
        };
        if same_memory_root(func, root, destination_root) {
            destination_arguments += 1;
        } else if !stable_distinct_memory_root(func, root, outer) {
            return false;
        }
    }
    destination_arguments == 1
        && !same_memory_root(func, source_root, destination_root)
        && stable_distinct_memory_root(func, destination_root, outer)
        && stable_distinct_memory_root(func, source_root, outer)
}

fn callee_fits_complete_overwrite(
    callee: &Function,
    destination_param: usize,
    bound_param: usize,
) -> bool {
    if exceeds_budget(callee)
        || callee.ret != Type::Void
        || destination_param >= callee.params.len()
        || bound_param >= callee.params.len()
    {
        return false;
    }
    let destination = callee.params[destination_param];
    let bound = callee.params[bound_param];
    if !matches!(callee.value(destination).ty, Type::Ptr(_))
        || callee.value(bound).ty != Type::I32
        || callee
            .values
            .iter()
            .any(|value| matches!(value.kind, ValueKind::Global(_)))
    {
        return false;
    }

    let cfg = ControlFlowGraph::new(callee);
    let dom = Dominators::new(callee, &cfg);
    let loops = LoopInfo::new(&cfg, &dom).loops().to_vec();
    let mut destination_accesses = 0usize;
    let mut destination_stores = 0usize;
    for (block_idx, block) in callee.blocks.iter().enumerate() {
        let block_id = BlockId(block_idx);
        for inst in &block.insts {
            let pointer = match &inst.kind {
                InstKind::Load { ptr } => Some((*ptr, false)),
                InstKind::Store { ptr, .. } => Some((*ptr, true)),
                InstKind::Call { .. } | InstKind::MemZero { .. } => return false,
                _ => None,
            };
            let Some((pointer, is_store)) = pointer else {
                continue;
            };
            let Some((root, indices)) = pointer_path(callee, pointer) else {
                return false;
            };
            if !callee.params.contains(&root) {
                return false;
            }
            if is_store && root != destination {
                return false;
            }
            if root == destination {
                if indices.len() != 2
                    || indices
                        .iter()
                        .any(|index| !index_within_bound(callee, &loops, *index, bound, block_id))
                {
                    return false;
                }
                destination_accesses += 1;
                destination_stores += usize::from(is_store);
            }
        }
    }
    destination_accesses != 0 && destination_stores != 0
}

fn index_within_bound(
    func: &Function,
    loops: &[NaturalLoop],
    index: ValueId,
    bound: ValueId,
    access_block: BlockId,
) -> bool {
    let mut visited = HashSet::new();
    index_within_bound_inner(func, loops, index, bound, access_block, &mut visited)
}

fn index_within_bound_inner(
    func: &Function,
    loops: &[NaturalLoop],
    index: ValueId,
    bound: ValueId,
    access_block: BlockId,
    visited: &mut HashSet<ValueId>,
) -> bool {
    if !visited.insert(index) {
        return false;
    }
    for natural_loop in loops {
        let Some(induction) = analyze_i32_induction(func, natural_loop, index) else {
            continue;
        };
        if induction.step != 1
            || direct_less_than_bound(func, natural_loop, index) != Some(bound)
            || !natural_loop.blocks.contains(&access_block)
        {
            continue;
        }
        if const_i32(func, induction.initial).is_some_and(|initial| initial >= 0) {
            return true;
        }
        let Some(parent) = plus_one_source(func, induction.initial) else {
            continue;
        };
        if loops.iter().any(|parent_loop| {
            parent_loop.blocks.is_superset(&natural_loop.blocks)
                && parent_loop.blocks.len() > natural_loop.blocks.len()
                && index_within_bound_inner(
                    func,
                    loops,
                    parent,
                    bound,
                    natural_loop.header,
                    visited,
                )
        }) {
            return true;
        }
    }
    false
}

fn plus_one_source(func: &Function, value: ValueId) -> Option<ValueId> {
    match defining_inst(func, value)? {
        InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        } if const_i32(func, *rhs) == Some(1) => Some(*lhs),
        InstKind::Binary {
            op: BinaryOp::Iadd,
            lhs,
            rhs,
        } if const_i32(func, *lhs) == Some(1) => Some(*rhs),
        _ => None,
    }
}

fn counter_is_control_only(
    func: &Function,
    counter: ValueId,
    condition: ValueId,
    next: ValueId,
) -> bool {
    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Phi { incomings } => {
                    if incomings.iter().any(|(_, incoming)| *incoming == counter) {
                        return false;
                    }
                }
                kind if inst_operands(kind).contains(&counter) => {
                    if inst.result != Some(condition) && inst.result != Some(next) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        if matches!(
            block.terminator,
            Some(Terminator::Return(Some(value))) if value == counter
        ) {
            return false;
        }
    }
    true
}

fn header_phis(func: &Function, header: BlockId) -> Vec<ValueId> {
    func.blocks[header.0]
        .insts
        .iter()
        .filter_map(|inst| {
            matches!(inst.kind, InstKind::Phi { .. })
                .then_some(inst.result)
                .flatten()
        })
        .collect()
}

fn calls_in_loop(func: &Function, natural_loop: &NaturalLoop) -> Vec<CallSite> {
    let mut calls = Vec::new();
    for block in &natural_loop.blocks {
        for inst in &func.blocks[block.0].insts {
            if let InstKind::Call { name, args } = &inst.kind {
                if inst.result.is_some() {
                    return Vec::new();
                }
                calls.push(CallSite {
                    block: *block,
                    name: name.clone(),
                    args: args.clone(),
                });
            }
        }
    }
    calls
}

fn pointer_path(func: &Function, mut value: ValueId) -> Option<(ValueId, Vec<ValueId>)> {
    let mut reversed = Vec::new();
    for _ in 0..MAX_POINTER_DEPTH {
        let Some(InstKind::Gep { base, indices }) = defining_inst(func, value) else {
            return matches!(func.values.get(value.0)?.ty, Type::Ptr(_)).then(|| {
                reversed.reverse();
                (value, reversed.into_iter().flatten().collect())
            });
        };
        reversed.push(indices.clone());
        value = *base;
    }
    None
}

fn pointer_root(func: &Function, value: ValueId) -> Option<ValueId> {
    pointer_path(func, value).map(|(root, _)| root)
}

fn same_memory_root(func: &Function, lhs: ValueId, rhs: ValueId) -> bool {
    if lhs == rhs {
        return true;
    }
    matches!(
        (
            func.values.get(lhs.0).map(|value| &value.kind),
            func.values.get(rhs.0).map(|value| &value.kind),
        ),
        (Some(ValueKind::Global(lhs)), Some(ValueKind::Global(rhs))) if lhs == rhs
    )
}

fn stable_distinct_memory_root(func: &Function, root: ValueId, outer: &NaturalLoop) -> bool {
    match func.values.get(root.0).map(|value| &value.kind) {
        Some(ValueKind::Global(_)) => true,
        Some(ValueKind::Inst(block, inst_idx)) if !outer.blocks.contains(block) => func
            .blocks
            .get(block.0)
            .and_then(|owner| owner.insts.get(*inst_idx))
            .is_some_and(|inst| matches!(inst.kind, InstKind::Alloca { .. })),
        _ => false,
    }
}

fn value_defined_in_loop(func: &Function, natural_loop: &NaturalLoop, value: ValueId) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Inst(block, _)) if natural_loop.blocks.contains(block)
    )
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. } => Vec::new(),
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

fn exceeds_budget(func: &Function) -> bool {
    func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || func
            .blocks
            .iter()
            .try_fold(0usize, |total, block| total.checked_add(block.insts.len()))
            .is_none_or(|count| count > MAX_INSTRUCTIONS)
}

fn apply_candidate(func: &mut Function, candidate: Candidate) {
    let final_initial = get_or_add_i32_const(func, candidate.final_initial);
    let ValueKind::Inst(block, inst_idx) = func.value(candidate.counter).kind else {
        return;
    };
    let InstKind::Phi { incomings } = &mut func.blocks[block.0].insts[inst_idx].kind else {
        return;
    };
    for (pred, value) in incomings {
        if *pred == candidate.entering {
            *value = final_initial;
        }
    }
}

fn unique_targets(funcs: &[Function]) -> HashMap<String, Option<FunctionId>> {
    let mut targets = HashMap::new();
    for (idx, func) in funcs.iter().enumerate() {
        targets
            .entry(func.name.clone())
            .and_modify(|target| *target = None)
            .or_insert(Some(FunctionId(idx)));
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, OptLevel, PassOptions};
    use super::*;
    use crate::ir::lower::lower_program;
    use crate::parser::Parser;

    const CALLEE: &str = r#"
void update(int n, float left[][4], float right[][4]) {
    int column = 0;
    while (column < n) {
        int row = 0;
        while (row < n) {
            right[row][column] = right[row][column] + left[row][column];
            row = row + 1;
        }
        column = column + 1;
    }
}
"#;

    #[test]
    fn keeps_only_the_last_complete_overwrite_iteration() {
        let module = optimized(&format!(
            r#"
{CALLEE}
float left[4][4];
float right[4][4];
float source[4][4];

int main() {{
    int n = getint();
    int repeat = 0;
    while (repeat < 3) {{
        int row = 0;
        while (row < n) {{
            int column = 0;
            while (column < n) {{
                right[row][column] = source[row][column];
                column = column + 1;
            }}
            row = row + 1;
        }}
        update(n, left, right);
        repeat = repeat + 1;
    }}
    return right[0][0] > 0;
}}
"#
        ));

        assert_eq!(repeated_call_loop_initial(&module, "main", "update"), 2);
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    #[test]
    fn rejects_a_partial_overwrite_before_the_call() {
        let module = optimized(&format!(
            r#"
{CALLEE}
float left[4][4];
float right[4][4];
float source[4][4];

int main() {{
    int n = getint();
    int repeat = 0;
    while (repeat < 3) {{
        int row = 0;
        while (row < n) {{
            right[row][0] = source[row][0];
            row = row + 1;
        }}
        update(n, left, right);
        repeat = repeat + 1;
    }}
    return right[0][0] > 0;
}}
"#
        ));

        assert_eq!(repeated_call_loop_initial(&module, "main", "update"), 0);
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    fn optimized(source: &str) -> Module {
        let program = Parser::new(source).parse_program();
        let mut module = lower_program(&program).expect("test source must lower");
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                cfg_inline_rounds: 0,
                cfg_inline_global_loads: false,
                enable_loop_call_memoize: false,
                enable_repeated_overwrite_elision: true,
                enable_guarded_mulmod_idiom: false,
                enable_guarded_pow2_digit_idiom: false,
                enable_write_only_alloca_cleanup_before_inline: false,
            },
        );
        module
    }

    fn repeated_call_loop_initial(module: &Module, caller: &str, callee: &str) -> i32 {
        let func = module
            .funcs
            .iter()
            .find(|func| func.name == caller)
            .expect("caller must exist");
        let cfg = ControlFlowGraph::new(func);
        let dom = Dominators::new(func, &cfg);
        let loops = LoopInfo::new(&cfg, &dom);
        for natural_loop in loops.loops() {
            let contains_call = natural_loop.blocks.iter().any(|block| {
                func.blocks[block.0].insts.iter().any(|inst| {
                    matches!(
                        &inst.kind,
                        InstKind::Call { name, .. } if name == callee
                    )
                })
            });
            if !contains_call {
                continue;
            }
            for phi in header_phis(func, natural_loop.header) {
                if let Some(induction) = analyze_i32_induction(func, natural_loop, phi) {
                    if induction.step == 1 {
                        return const_i32(func, induction.initial)
                            .expect("repeat loop must have a constant initial value");
                    }
                }
            }
        }
        panic!("repeat loop containing the call must exist in {func:#?}");
    }
}
