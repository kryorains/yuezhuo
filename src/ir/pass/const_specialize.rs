use super::const_fold::ConstFoldPass;
use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{Const, Function, FunctionId, InstKind, Module, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_FUNCTION_BLOCKS: usize = 32;
const MAX_FUNCTION_INSTS: usize = 256;
const MAX_FUNCTION_INST_SLOTS: usize = 1024;
const MAX_VARIANTS_PER_FUNCTION: usize = 9;
const MAX_ADDED_FUNCTIONS: usize = 32;
const MAX_ADDED_BLOCKS: usize = 256;
const MAX_ADDED_INST_SLOTS: usize = 4096;
const MAX_SPECIALIZATION_ROUNDS: usize = 9;

pub(super) struct ConstSpecializePass {
    allow_recursive: bool,
    propagate_uniform_arguments: bool,
}

impl ConstSpecializePass {
    pub(super) fn new(allow_recursive: bool, propagate_uniform_arguments: bool) -> Self {
        Self {
            allow_recursive,
            propagate_uniform_arguments,
        }
    }
}

impl ModulePass for ConstSpecializePass {
    fn run(&mut self, module: &mut Module) {
        if self.propagate_uniform_arguments {
            propagate_uniform_constant_arguments(module);
            ConstFoldPass::new().run(module);
        }

        let mut added_functions = 0usize;
        let mut added_blocks = 0usize;
        let mut added_inst_slots = 0usize;
        let mut variants_per_function = HashMap::<FunctionId, usize>::new();

        for _ in 0..MAX_SPECIALIZATION_ROUNDS {
            let snapshots = module.funcs.clone();
            let targets = unique_targets(&snapshots);
            let parameter_scores = snapshots
                .iter()
                .map(parameter_specialization_scores)
                .collect::<Vec<_>>();
            let eligible = snapshots
                .iter()
                .map(|func| specializable(func, self.allow_recursive))
                .collect::<Vec<_>>();
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
                                parameter_scores[callee.0]
                                    .get(param)
                                    .is_some_and(|score| *score >= 2)
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
            // Expose constants in recursive call arguments before taking the
            // next snapshot. Cloned variants keep self-calls directed at the
            // original function, so each round specializes exactly one
            // additional constant recursion state without recursive cloning.
            ConstFoldPass::new().run(module);
        }
    }
}

#[derive(Clone)]
enum UniformConstant {
    Unseen,
    Known(Const),
    Varying,
}

fn propagate_uniform_constant_arguments(module: &mut Module) {
    let snapshots = module.funcs.clone();
    let targets = unique_targets(&snapshots);
    let mut arguments = snapshots
        .iter()
        .map(|func| vec![UniformConstant::Unseen; func.params.len()])
        .collect::<Vec<_>>();

    for caller in &snapshots {
        for block in &caller.blocks {
            for inst in &block.insts {
                let InstKind::Call { name, args } = &inst.kind else {
                    continue;
                };
                let Some(callee) = targets.get(name).copied().flatten() else {
                    continue;
                };
                if args.len() != arguments[callee.0].len() {
                    continue;
                }
                for (state, argument) in arguments[callee.0].iter_mut().zip(args) {
                    let Some(constant) = scalar_const(caller, *argument) else {
                        *state = UniformConstant::Varying;
                        continue;
                    };
                    match state {
                        UniformConstant::Unseen => *state = UniformConstant::Known(constant),
                        UniformConstant::Known(existing) if existing == &constant => {}
                        UniformConstant::Known(_) | UniformConstant::Varying => {
                            *state = UniformConstant::Varying;
                        }
                    }
                }
            }
        }
    }

    for (func, states) in module.funcs.iter_mut().zip(arguments) {
        let mut replacements = ValueReplacements::new();
        for (param, state) in func.params.clone().into_iter().zip(states) {
            let UniformConstant::Known(constant) = state else {
                continue;
            };
            replacements.insert(param, get_or_add_const(func, &constant));
        }
        if replacements.is_empty() {
            continue;
        }
        rewrite_function_uses(func, &replacements);
        if let Err(errors) = func.verify() {
            panic!(
                "uniform constant argument propagation produced invalid IR: {:?}",
                errors
            );
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

fn specializable(func: &Function, allow_recursive: bool) -> bool {
    !func.blocks.is_empty()
        && func.blocks.len() <= MAX_FUNCTION_BLOCKS
        && active_inst_count(func) <= MAX_FUNCTION_INSTS
        && inst_slot_count(func) <= MAX_FUNCTION_INST_SLOTS
        && (allow_recursive
            || !func.blocks.iter().any(|block| {
                block.insts.iter().any(
                    |inst| matches!(&inst.kind, InstKind::Call { name, .. } if name == &func.name),
                )
            }))
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

fn parameter_specialization_scores(func: &Function) -> HashMap<ValueId, usize> {
    let params = func.params.iter().copied().collect::<HashSet<_>>();
    let mut scores = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            for (operand, score) in specialization_operands(&inst.kind) {
                if params.contains(&operand) {
                    *scores.entry(operand).or_default() += score;
                }
            }
        }
        match block.terminator.as_ref() {
            Some(crate::ir::Terminator::Branch { cond, .. }) if params.contains(cond) => {
                *scores.entry(*cond).or_default() += 4;
            }
            Some(crate::ir::Terminator::Return(Some(value))) if params.contains(value) => {
                *scores.entry(*value).or_default() += 1;
            }
            _ => {}
        }
    }
    scores
}

fn specialization_operands(kind: &InstKind) -> Vec<(ValueId, usize)> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| (*value, 1)).collect(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![(*ptr, 1)],
        InstKind::Store { ptr, value } => vec![(*ptr, 1), (*value, 1)],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![(*value, 1)],
        InstKind::Binary {
            op:
                crate::ir::BinaryOp::Idiv
                | crate::ir::BinaryOp::Imod
                | crate::ir::BinaryOp::Ishl
                | crate::ir::BinaryOp::Iashr,
            lhs,
            rhs,
        } => vec![(*lhs, 1), (*rhs, 4)],
        InstKind::Binary { lhs, rhs, .. } => vec![(*lhs, 1), (*rhs, 1)],
        InstKind::Icmp { lhs, rhs, .. } | InstKind::Fcmp { lhs, rhs, .. } => {
            vec![(*lhs, 4), (*rhs, 4)]
        }
        InstKind::Gep { base, indices } => {
            let mut values = Vec::with_capacity(indices.len() + 1);
            values.push((*base, 1));
            values.extend(indices.iter().map(|index| (*index, 3)));
            values
        }
        InstKind::Call { args, .. } => args.iter().map(|arg| (*arg, 1)).collect(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, CmpOp, Function, InstKind, Module, Terminator, Type};

    #[test]
    fn prioritizes_control_operands_over_incidental_single_uses() {
        let mut func = Function::new("arbitrary_name", Type::I1);
        let control = func.add_param("control", Type::I32);
        let data = func.add_param("data", Type::I32);
        let zero = func.add_const(Const::Int(0));
        func.append_inst(
            func.entry,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: data,
                rhs: zero,
            },
            Some(Type::I32),
        )
        .unwrap();
        let condition = func
            .append_inst(
                func.entry,
                InstKind::Icmp {
                    op: CmpOp::Lt,
                    lhs: control,
                    rhs: zero,
                },
                Some(Type::I1),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(condition)));

        let scores = parameter_specialization_scores(&func);
        assert_eq!(scores[&control], 4);
        assert_eq!(scores[&data], 1);
    }

    #[test]
    fn propagates_only_uniform_whole_program_arguments() {
        let mut uniform = Function::new("uniform", Type::I32);
        let uniform_param = uniform.add_param("value", Type::I32);
        uniform.set_terminator(uniform.entry, Terminator::Return(Some(uniform_param)));

        let mut varying = Function::new("varying", Type::I32);
        let varying_param = varying.add_param("value", Type::I32);
        varying.set_terminator(varying.entry, Terminator::Return(Some(varying_param)));

        let mut main = Function::new("main", Type::I32);
        let seven = main.add_const(Const::Int(7));
        let eight = main.add_const(Const::Int(8));
        main.append_inst(
            main.entry,
            InstKind::Call {
                name: "uniform".into(),
                args: vec![seven],
            },
            Some(Type::I32),
        );
        main.append_inst(
            main.entry,
            InstKind::Call {
                name: "varying".into(),
                args: vec![seven],
            },
            Some(Type::I32),
        );
        let answer = main
            .append_inst(
                main.entry,
                InstKind::Call {
                    name: "varying".into(),
                    args: vec![eight],
                },
                Some(Type::I32),
            )
            .unwrap();
        main.set_terminator(main.entry, Terminator::Return(Some(answer)));

        let mut module = Module::new();
        module.add_func(uniform);
        module.add_func(varying);
        module.add_func(main);
        propagate_uniform_constant_arguments(&mut module);

        let Some(Terminator::Return(Some(uniform_return))) =
            module.funcs[0].blocks[0].terminator.as_ref()
        else {
            panic!("uniform function must return a value");
        };
        assert!(matches!(
            module.funcs[0].values[uniform_return.0].kind,
            ValueKind::Const(Const::Int(7))
        ));
        assert_eq!(
            module.funcs[1].blocks[0].terminator,
            Some(Terminator::Return(Some(varying_param)))
        );
    }

    #[test]
    fn recursively_specializes_a_constant_state_chain_only_when_enabled() {
        let mut enabled = recursive_module();
        ConstSpecializePass::new(true, false).run(&mut enabled);

        let variants = enabled
            .funcs
            .iter()
            .filter(|func| func.name.starts_with("countdown.specialized."))
            .collect::<Vec<_>>();
        assert_eq!(variants.len(), MAX_VARIANTS_PER_FUNCTION);
        assert!(variants[0].blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                matches!(
                    &inst.kind,
                    InstKind::Call { name, .. }
                        if name.starts_with("countdown.specialized.") && name != &variants[0].name
                )
            })
        }));
        assert!(enabled.funcs.iter().all(|func| func.verify().is_ok()));

        let mut disabled = recursive_module();
        ConstSpecializePass::new(false, false).run(&mut disabled);
        assert!(disabled
            .funcs
            .iter()
            .all(|func| !func.name.starts_with("countdown.specialized.")));
    }

    fn recursive_module() -> Module {
        let mut countdown = Function::new("countdown", Type::I32);
        let depth = countdown.add_param("depth", Type::I32);
        let one = countdown.add_const(Const::Int(1));
        let next = countdown
            .append_inst(
                countdown.entry,
                InstKind::Binary {
                    op: BinaryOp::Isub,
                    lhs: depth,
                    rhs: one,
                },
                Some(Type::I32),
            )
            .unwrap();
        countdown.append_inst(
            countdown.entry,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: depth,
                rhs: one,
            },
            Some(Type::I32),
        );
        let recursive = countdown
            .append_inst(
                countdown.entry,
                InstKind::Call {
                    name: "countdown".to_string(),
                    args: vec![next],
                },
                Some(Type::I32),
            )
            .unwrap();
        countdown.set_terminator(countdown.entry, Terminator::Return(Some(recursive)));

        let mut main = Function::new("main", Type::I32);
        let three = main.add_const(Const::Int(3));
        let result = main
            .append_inst(
                main.entry,
                InstKind::Call {
                    name: "countdown".to_string(),
                    args: vec![three],
                },
                Some(Type::I32),
            )
            .unwrap();
        main.set_terminator(main.entry, Terminator::Return(Some(result)));

        let mut module = Module::new();
        module.add_func(countdown);
        module.add_func(main);
        module
    }
}
