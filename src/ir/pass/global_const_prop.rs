use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{Const, Function, Global, InstKind, Module, Terminator, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

pub(super) struct GlobalConstPropPass {
    include_initialized_mutable: bool,
}

impl GlobalConstPropPass {
    pub(super) fn new(include_initialized_mutable: bool) -> Self {
        Self {
            include_initialized_mutable,
        }
    }
}

impl ModulePass for GlobalConstPropPass {
    fn run(&mut self, module: &mut Module) {
        let mut candidates = collect_candidates(&module.globals, self.include_initialized_mutable);
        reject_written_or_escaping_globals(&module.funcs, &mut candidates);
        if candidates.symbols.is_empty() {
            return;
        }

        for func in &mut module.funcs {
            propagate_function(func, &candidates);
        }
    }
}

#[derive(Clone)]
struct Candidate {
    ty: Type,
    init: Const,
}

struct Candidates {
    symbols: HashMap<String, usize>,
    objects: Vec<Option<Candidate>>,
}

fn collect_candidates(globals: &[Global], include_initialized_mutable: bool) -> Candidates {
    let mut seen = HashSet::new();
    let mut symbols = HashMap::new();
    let mut objects = vec![None; globals.len()];

    for (object, global) in globals.iter().enumerate() {
        if !seen.insert(global.name.clone()) {
            // A symbol must resolve to one global object before its identity is
            // usable as a propagation proof.
            symbols.remove(&global.name);
            continue;
        }
        if let Some(candidate) = candidate(global, include_initialized_mutable) {
            objects[object] = Some(candidate);
            symbols.insert(global.name.clone(), object);
        }
    }
    Candidates { symbols, objects }
}

fn candidate(global: &Global, include_initialized_mutable: bool) -> Option<Candidate> {
    if (!global.is_const && !include_initialized_mutable)
        || !matches!(global.ty, Type::I1 | Type::I32 | Type::F32)
    {
        return None;
    }
    let init = global.init.as_ref()?;
    if init.ty() != global.ty {
        return None;
    }
    Some(Candidate {
        ty: global.ty.clone(),
        init: init.clone(),
    })
}

fn reject_written_or_escaping_globals(funcs: &[Function], candidates: &mut Candidates) {
    let mut rejected = HashSet::new();
    for func in funcs {
        for block in &func.blocks {
            for inst in &block.insts {
                match &inst.kind {
                    // A direct load of the complete object is the only address
                    // use that preserves the immutable-object proof.
                    InstKind::Load { .. } | InstKind::Nop | InstKind::Alloca { .. } => {}
                    kind => {
                        for operand in instruction_operands(kind) {
                            reject_global_operand(func, operand, candidates, &mut rejected);
                        }
                    }
                }
            }
            if let Some(terminator) = &block.terminator {
                for operand in terminator_operands(terminator) {
                    reject_global_operand(func, operand, candidates, &mut rejected);
                }
            }
        }
    }
    for object in rejected {
        candidates.objects[object] = None;
    }
    candidates
        .symbols
        .retain(|_, object| candidates.objects[*object].is_some());
}

fn reject_global_operand(
    func: &Function,
    operand: ValueId,
    candidates: &Candidates,
    rejected: &mut HashSet<usize>,
) {
    let Some(ValueKind::Global(symbol)) = func.values.get(operand.0).map(|value| &value.kind)
    else {
        return;
    };
    if let Some(object) = candidates.symbols.get(symbol) {
        rejected.insert(*object);
    }
}

fn instruction_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
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

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(value) => value.iter().copied().collect(),
        Terminator::Jump(_) => Vec::new(),
        Terminator::Branch { cond, .. } => vec![*cond],
    }
}

fn propagate_function(func: &mut Function, candidates: &Candidates) {
    let mut loads = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            let (Some(result), InstKind::Load { ptr }) = (inst.result, &inst.kind) else {
                continue;
            };
            let Some(pointer) = func.values.get(ptr.0) else {
                continue;
            };
            let ValueKind::Global(symbol) = &pointer.kind else {
                continue;
            };
            let Some(object) = candidates.symbols.get(symbol).copied() else {
                continue;
            };
            let candidate = candidates.objects[object]
                .as_ref()
                .expect("candidate symbol must resolve to its global object");
            if pointer.ty != Type::Ptr(Box::new(candidate.ty.clone()))
                || func.values.get(result.0).map(|value| &value.ty) != Some(&candidate.ty)
            {
                continue;
            }
            loads.push((block_idx, inst_idx, result, object));
        }
    }
    if loads.is_empty() {
        return;
    }

    let mut constants = HashMap::<usize, ValueId>::new();
    let mut replacements = ValueReplacements::new();
    for (_, _, load, object) in &loads {
        let constant = *constants.entry(*object).or_insert_with(|| {
            func.add_const(
                candidates.objects[*object]
                    .as_ref()
                    .expect("candidate object must exist")
                    .init
                    .clone(),
            )
        });
        replacements.insert(*load, constant);
    }

    rewrite_function_uses(func, &replacements);
    for (block_idx, inst_idx, _, _) in loads {
        let inst = &mut func.blocks[block_idx].insts[inst_idx];
        inst.result = None;
        inst.kind = InstKind::Nop;
    }
    if let Err(errors) = func.verify() {
        panic!(
            "global constant propagation produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}
