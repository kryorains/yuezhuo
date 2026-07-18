use super::util::{
    const_i32, defining_inst, get_or_add_i32_const as get_or_add_const, move_entry_to_fallback,
};
use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId, ValueKind,
};
use std::collections::{BTreeSet, HashSet};

// Bound repeated decision-tree interpretation on generated stress cases.
const MAX_BLOCKS: usize = 128;
const MAX_VALUES: usize = 2048;

/// Collapses pure equality-decision functions whose contiguous selector range
/// maps `x` to `x * 2^selector` or `x / 2^selector`. Matching interprets the
/// decision tree and return expressions; it does not depend on block order,
/// names, source variable names, or a fixed selector range.
pub(super) struct PiecewiseExprPass;

impl PiecewiseExprPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for PiecewiseExprPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            let Some(candidate) = match_piecewise_expr(func) else {
                continue;
            };
            if !install_fast_path(func, candidate) {
                continue;
            }
            if let Err(errors) = func.verify() {
                panic!(
                    "piecewise expression folding produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScaleKind {
    Left,
    SignedDivide,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    input: ValueId,
    selector: ValueId,
    minimum: i32,
    maximum: i32,
    kind: ScaleKind,
}

#[derive(Clone, Copy)]
enum SelectorCase {
    Exact(i32),
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReturnExpr {
    Input,
    Scale { kind: ScaleKind, shift: u32 },
}

fn match_piecewise_expr(func: &Function) -> Option<Candidate> {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func.blocks.len() > MAX_BLOCKS
        || func.values.len() > MAX_VALUES
        || func
            .params
            .iter()
            .any(|param| func.value(*param).ty != Type::I32)
        || !function_is_pure_scalar(func)
        || function_has_dynamic_shift(func)
    {
        return None;
    }

    let roles = [
        (func.params[0], func.params[1]),
        (func.params[1], func.params[0]),
    ];
    let mut matched = None;
    for (input, selector) in roles {
        let Some(candidate) = match_roles(func, input, selector) else {
            continue;
        };
        if matched.replace(candidate).is_some() {
            return None;
        }
    }
    matched
}

fn function_is_pure_scalar(func: &Function) -> bool {
    func.blocks.iter().all(|block| {
        block.insts.iter().all(|inst| {
            matches!(
                inst.kind,
                InstKind::Nop
                    | InstKind::Unary { .. }
                    | InstKind::Binary { .. }
                    | InstKind::Icmp { .. }
                    | InstKind::Cast { .. }
            )
        })
    })
}

fn function_has_dynamic_shift(func: &Function) -> bool {
    func.blocks.iter().any(|block| {
        block.insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::Binary {
                    op: BinaryOp::Ishl | BinaryOp::Iashr,
                    ..
                }
            )
        })
    })
}

fn match_roles(func: &Function, input: ValueId, selector: ValueId) -> Option<Candidate> {
    let mut keys = BTreeSet::new();
    for block in &func.blocks {
        let Some(Terminator::Branch { cond, .. }) = block.terminator.as_ref() else {
            continue;
        };
        let (_, key) = selector_test(func, *cond, selector)?;
        if !(0..i32::BITS as i32).contains(&key) {
            return None;
        }
        keys.insert(key);
    }
    if keys.len() < 2 {
        return None;
    }
    let minimum = *keys.first()?;
    let maximum = *keys.last()?;
    if keys.len() != (maximum - minimum + 1) as usize {
        return None;
    }

    let default = interpret_return(func, input, selector, SelectorCase::Other)?;
    if default != ReturnExpr::Input {
        return None;
    }

    let mut kind = None;
    for key in keys {
        let expression = interpret_return(func, input, selector, SelectorCase::Exact(key))?;
        if key == 0 && expression == ReturnExpr::Input {
            continue;
        }
        let ReturnExpr::Scale {
            kind: actual_kind,
            shift,
        } = expression
        else {
            return None;
        };
        if shift != key as u32 {
            return None;
        }
        if let Some(expected) = kind {
            if expected != actual_kind {
                return None;
            }
        } else {
            kind = Some(actual_kind);
        }
    }

    Some(Candidate {
        input,
        selector,
        minimum,
        maximum,
        kind: kind?,
    })
}

fn selector_test(func: &Function, value: ValueId, selector: ValueId) -> Option<(CmpOp, i32)> {
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, value)? else {
        return None;
    };
    if !matches!(op, CmpOp::Eq | CmpOp::Ne) {
        return None;
    }
    if *lhs == selector {
        Some((*op, const_i32(func, *rhs)?))
    } else if *rhs == selector {
        Some((*op, const_i32(func, *lhs)?))
    } else {
        None
    }
}

fn interpret_return(
    func: &Function,
    input: ValueId,
    selector: ValueId,
    selector_case: SelectorCase,
) -> Option<ReturnExpr> {
    let mut current = func.entry;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let block = func.blocks.get(current.0)?;
        match block.terminator.as_ref()? {
            Terminator::Jump(target) => current = *target,
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                let (op, key) = selector_test(func, *cond, selector)?;
                let equal = match selector_case {
                    SelectorCase::Exact(value) => value == key,
                    SelectorCase::Other => false,
                };
                let condition = if op == CmpOp::Eq { equal } else { !equal };
                current = if condition {
                    *then_target
                } else {
                    *else_target
                };
            }
            Terminator::Return(Some(value)) => {
                return classify_return(func, *value, input, &visited)
            }
            Terminator::Return(None) => return None,
        }
    }
}

fn classify_return(
    func: &Function,
    value: ValueId,
    input: ValueId,
    visited: &HashSet<BlockId>,
) -> Option<ReturnExpr> {
    if value == input {
        return Some(ReturnExpr::Input);
    }
    let ValueKind::Inst(owner, _) = func.values.get(value.0)?.kind else {
        return None;
    };
    if !visited.contains(&owner) {
        return None;
    }
    let InstKind::Binary { op, lhs, rhs } = defining_inst(func, value)? else {
        return None;
    };
    match op {
        BinaryOp::Imul => {
            let scale = if *lhs == input {
                const_i32(func, *rhs)?
            } else if *rhs == input {
                const_i32(func, *lhs)?
            } else {
                return None;
            };
            Some(ReturnExpr::Scale {
                kind: ScaleKind::Left,
                shift: positive_pow2_shift(scale)?,
            })
        }
        BinaryOp::Idiv if *lhs == input => Some(ReturnExpr::Scale {
            kind: ScaleKind::SignedDivide,
            shift: positive_pow2_shift(const_i32(func, *rhs)?)?,
        }),
        _ => None,
    }
}

fn install_fast_path(func: &mut Function, candidate: Candidate) -> bool {
    let Some(fallback) = move_entry_to_fallback(func, "idiom.piecewise.fallback") else {
        return false;
    };
    let fast = func.add_block("idiom.piecewise.fast");
    let minimum = get_or_add_const(func, candidate.minimum);
    let maximum = get_or_add_const(func, candidate.maximum);
    let above_minimum = func
        .append_inst(
            func.entry,
            InstKind::Icmp {
                op: CmpOp::Ge,
                lhs: candidate.selector,
                rhs: minimum,
            },
            Some(Type::I1),
        )
        .unwrap();
    let below_maximum = func
        .append_inst(
            func.entry,
            InstKind::Icmp {
                op: CmpOp::Le,
                lhs: candidate.selector,
                rhs: maximum,
            },
            Some(Type::I1),
        )
        .unwrap();
    let mut use_fast_path = append_bool_and(func, above_minimum, below_maximum);
    if candidate.kind == ScaleKind::SignedDivide {
        let zero = get_or_add_const(func, 0);
        let input_nonnegative = func
            .append_inst(
                func.entry,
                InstKind::Icmp {
                    op: CmpOp::Ge,
                    lhs: candidate.input,
                    rhs: zero,
                },
                Some(Type::I1),
            )
            .unwrap();
        use_fast_path = append_bool_and(func, use_fast_path, input_nonnegative);
    }
    func.set_terminator(
        func.entry,
        Terminator::Branch {
            cond: use_fast_path,
            then_target: fast,
            else_target: fallback,
        },
    );

    let op = match candidate.kind {
        ScaleKind::Left => BinaryOp::Ishl,
        ScaleKind::SignedDivide => BinaryOp::Iashr,
    };
    let result = func
        .append_inst(
            fast,
            InstKind::Binary {
                op,
                lhs: candidate.input,
                rhs: candidate.selector,
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(fast, Terminator::Return(Some(result)));
    true
}

fn append_bool_and(func: &mut Function, lhs: ValueId, rhs: ValueId) -> ValueId {
    func.append_inst(
        func.entry,
        InstKind::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
        },
        Some(Type::I1),
    )
    .unwrap()
}

fn positive_pow2_shift(value: i32) -> Option<u32> {
    (value > 0 && (value & (value - 1)) == 0).then_some(value.trailing_zeros())
}
