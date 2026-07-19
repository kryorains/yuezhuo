use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{Const, Function, FunctionId, InstKind, Module, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTION_BLOCKS: usize = 32;
const MAX_FUNCTION_INSTS: usize = 96;
const MAX_FUNCTION_INST_SLOTS: usize = 512;
const MAX_VARIANTS_PER_FUNCTION: usize = 4;
const MAX_ADDED_FUNCTIONS: usize = 32;
const MAX_ADDED_BLOCKS: usize = 256;
const MAX_ADDED_INST_SLOTS: usize = 4096;
const MAX_SPECIALIZATION_ROUNDS: usize = 3;

pub(super) struct ConstSpecializePass;

impl ConstSpecializePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for ConstSpecializePass {
    fn run(&mut self, module: &mut Module) {
        let mut added_functions = 0usize;
        let mut added_blocks = 0usize;
        let mut added_inst_slots = 0usize;
        let mut variants_per_function = HashMap::<FunctionId, usize>::new();

        for _ in 0..MAX_SPECIALIZATION_ROUNDS {
            let snapshots = module.funcs.clone();
            let targets = unique_targets(&snapshots);
            let used_params = snapshots
                .iter()
                .map(parameter_use_counts)
                .collect::<Vec<_>>();
            let eligible = snapshots.iter().map(specializable).collect::<Vec<_>>();
            let mut variants = Vec::<Variant>::new();
            let mut sites = Vec::<Site>::new();

            for (caller_idx, caller) in snapshots.iter().enumerate() {
                for (block_idx, block) in caller.blocks.iter().enumerate() {
                    for (inst_idx, inst) in block.insts.iter().enumerate() {
                        let InstKind::Call { name, args } = &inst.kind else {
                            continue;
                        };
                        let Some(callee) = targets.get(name).copied().flatten() else {
                            continue;
                        };
                        if callee.0 == caller_idx || !eligible[callee.0] {
                            continue;
                        }
                        let constants = args
                            .iter()
                            .zip(&snapshots[callee.0].params)
                            .map(|(arg, param)| {
                                used_params[callee.0]
                                    .get(param)
                                    .is_some_and(|uses| *uses >= 2)
                                    .then(|| scalar_const(caller, *arg))
                                    .flatten()
                            })
                            .collect::<Vec<_>>();
                        if constants.iter().all(Option::is_none) {
                            continue;
                        }
                        let variant_idx = variants
                            .iter()
                            .position(|variant| {
                                variant.callee == callee && variant.constants == constants
                            })
                            .unwrap_or_else(|| {
                                variants.push(Variant {
                                    callee,
                                    constants: constants.clone(),
                                });
                                variants.len() - 1
                            });
                        sites.push(Site {
                            caller: FunctionId(caller_idx),
                            block_idx,
                            inst_idx,
                            variant_idx,
                        });
                    }
                }
            }

            if variants.is_empty() {
                break;
            }
            let mut requested = HashMap::<FunctionId, usize>::new();
            for variant in &variants {
                *requested.entry(variant.callee).or_default() += 1;
            }
            let mut accepted = vec![None; variants.len()];
            for (variant_idx, variant) in variants.iter().enumerate() {
                let source = &snapshots[variant.callee.0];
                let inst_slots = inst_slot_count(source);
                let count = variants_per_function.entry(variant.callee).or_default();
                if requested[&variant.callee] > MAX_VARIANTS_PER_FUNCTION.saturating_sub(*count)
                    || *count >= MAX_VARIANTS_PER_FUNCTION
                    || added_functions >= MAX_ADDED_FUNCTIONS
                    || added_blocks.saturating_add(source.blocks.len()) > MAX_ADDED_BLOCKS
                    || added_inst_slots.saturating_add(inst_slots) > MAX_ADDED_INST_SLOTS
                {
                    continue;
                }

                let mut clone = source.clone();
                clone.name = fresh_variant_name(module, variant.callee, *count);
                let mut replacements = ValueReplacements::new();
                for (param, constant) in clone.params.clone().into_iter().zip(&variant.constants) {
                    if let Some(constant) = constant {
                        let value = get_or_add_const(&mut clone, constant);
                        replacements.insert(param, value);
                    }
                }
                rewrite_function_uses(&mut clone, &replacements);
                if let Err(errors) = clone.verify() {
                    panic!("constant specialization produced invalid IR: {:?}", errors);
                }
                let name = clone.name.clone();
                module.add_func(clone);
                accepted[variant_idx] = Some(name);
                *count += 1;
                added_functions += 1;
                added_blocks += source.blocks.len();
                added_inst_slots += inst_slots;
            }

            let mut rewrote = false;
            for site in sites {
                let Some(name) = accepted[site.variant_idx].as_ref() else {
                    continue;
                };
                let Some(InstKind::Call {
                    name: current_name, ..
                }) = module.funcs[site.caller.0]
                    .blocks
                    .get_mut(site.block_idx)
                    .and_then(|block| block.insts.get_mut(site.inst_idx))
                    .map(|inst| &mut inst.kind)
                else {
                    continue;
                };
                *current_name = name.clone();
                rewrote = true;
            }
            if !rewrote {
                break;
            }
        }
    }
}

#[derive(Clone)]
struct Variant {
    callee: FunctionId,
    constants: Vec<Option<Const>>,
}

struct Site {
    caller: FunctionId,
    block_idx: usize,
    inst_idx: usize,
    variant_idx: usize,
}

fn specializable(func: &Function) -> bool {
    !func.blocks.is_empty()
        && func.blocks.len() <= MAX_FUNCTION_BLOCKS
        && active_inst_count(func) <= MAX_FUNCTION_INSTS
        && inst_slot_count(func) <= MAX_FUNCTION_INST_SLOTS
        && !func.blocks.iter().any(|block| {
            block
                .insts
                .iter()
                .any(|inst| matches!(&inst.kind, InstKind::Call { name, .. } if name == &func.name))
        })
        && func.verify().is_ok()
}

fn active_inst_count(func: &Function) -> usize {
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter(|inst| !matches!(inst.kind, InstKind::Nop))
        .count()
}

fn inst_slot_count(func: &Function) -> usize {
    func.blocks.iter().map(|block| block.insts.len()).sum()
}

fn parameter_use_counts(func: &Function) -> HashMap<ValueId, usize> {
    let params = func.params.iter().copied().collect::<HashSet<_>>();
    let mut used = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            for operand in inst_operands(&inst.kind) {
                if params.contains(&operand) {
                    *used.entry(operand).or_default() += 1;
                }
            }
        }
        if let Some(
            crate::ir::Terminator::Return(Some(value))
            | crate::ir::Terminator::Branch { cond: value, .. },
        ) = block.terminator.as_ref()
        {
            if params.contains(value) {
                *used.entry(*value).or_default() += 1;
            }
        }
    }
    used
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
            let mut values = Vec::with_capacity(indices.len() + 1);
            values.push(*base);
            values.extend(indices.iter().copied());
            values
        }
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn scalar_const(func: &Function, value: ValueId) -> Option<Const> {
    match &func.value(value).kind {
        ValueKind::Const(constant @ (Const::Int(_) | Const::Bool(_) | Const::Float(_))) => {
            Some(constant.clone())
        }
        _ => None,
    }
}

fn get_or_add_const(func: &mut Function, constant: &Const) -> ValueId {
    func.values
        .iter()
        .position(|value| {
            value.ty == constant.ty()
                && matches!(&value.kind, ValueKind::Const(existing) if existing == constant)
        })
        .map(ValueId)
        .unwrap_or_else(|| func.add_const(constant.clone()))
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

fn fresh_variant_name(module: &Module, callee: FunctionId, variant: usize) -> String {
    let base = format!(
        "{}.specialized.{}.{}",
        module.funcs[callee.0].name, callee.0, variant
    );
    if !module.funcs.iter().any(|func| func.name == base) {
        return base;
    }
    let mut suffix = 0usize;
    loop {
        let candidate = format!("{}.{}", base, suffix);
        if !module.funcs.iter().any(|func| func.name == candidate) {
            return candidate;
        }
        suffix += 1;
    }
}
