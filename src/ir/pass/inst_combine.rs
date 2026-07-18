use super::util::{const_i32, defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{BinaryOp, CmpOp, Function, InstKind, Module, ValueId};

/// Canonicalizes and combines local integer instructions.
///
/// Every rule is based only on an instruction and the definitions of its
/// operands. In particular, this pass does not inspect source names, block
/// names, or whole-function control-flow shapes.
pub(super) struct InstCombinePass;

impl InstCombinePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for InstCombinePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            combine_function(func);
        }
    }
}

fn combine_function(func: &mut Function) {
    // Reassociation can expose another constant-bearing definition, so keep
    // scanning until every local expression reaches its canonical form.
    loop {
        let mut changed = false;
        for block_idx in 0..func.blocks.len() {
            for inst_idx in 0..func.blocks[block_idx].insts.len() {
                let old_kind = func.blocks[block_idx].insts[inst_idx].kind.clone();
                let new_kind = combine_inst(func, &old_kind);
                if new_kind != old_kind {
                    func.blocks[block_idx].insts[inst_idx].kind = new_kind;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    if let Err(errors) = func.verify() {
        panic!(
            "instruction combining produced invalid IR in {}: {:?}",
            func.name, errors
        );
    }
}

fn combine_inst(func: &mut Function, kind: &InstKind) -> InstKind {
    match *kind {
        InstKind::Binary { op, lhs, rhs } => combine_binary(func, op, lhs, rhs),
        InstKind::Icmp { op, lhs, rhs }
            if const_i32(func, lhs).is_some() && const_i32(func, rhs).is_none() =>
        {
            InstKind::Icmp {
                op: reverse_cmp(op),
                lhs: rhs,
                rhs: lhs,
            }
        }
        _ => kind.clone(),
    }
}

fn combine_binary(
    func: &mut Function,
    mut op: BinaryOp,
    mut lhs: ValueId,
    mut rhs: ValueId,
) -> InstKind {
    (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);

    // i32 arithmetic wraps, so doubling is exactly multiplication by two.
    if op == BinaryOp::Iadd && lhs == rhs {
        op = BinaryOp::Imul;
        rhs = get_or_add_i32_const(func, 2);
        (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);
    }

    if op == BinaryOp::Isub {
        if let Some(divisor) = match_remainder_divisor(func, lhs, rhs) {
            return InstKind::Binary {
                op: BinaryOp::Imod,
                lhs,
                rhs: divisor,
            };
        }
    }

    if matches!(op, BinaryOp::Iadd | BinaryOp::Imul) {
        if let Some((base, constant)) = reassociate_constants(func, op, lhs, rhs) {
            lhs = base;
            rhs = constant;
            (lhs, rhs) = canonicalize_integer_operands(func, op, lhs, rhs);
        }
    }

    InstKind::Binary { op, lhs, rhs }
}

/// Keeps constants on the right and otherwise orders operands by ValueId.
/// Floating-point and boolean operations are deliberately excluded.
fn canonicalize_integer_operands(
    func: &Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> (ValueId, ValueId) {
    if !is_commutative_integer_op(op) {
        return (lhs, rhs);
    }

    let lhs_is_const = const_i32(func, lhs).is_some();
    let rhs_is_const = const_i32(func, rhs).is_some();
    let should_swap = match (lhs_is_const, rhs_is_const) {
        (true, false) => true,
        (false, true) => false,
        _ => rhs.0 < lhs.0,
    };
    if should_swap {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn is_commutative_integer_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Iadd | BinaryOp::Imul | BinaryOp::Iand | BinaryOp::Ior | BinaryOp::Ixor
    )
}

fn reassociate_constants(
    func: &mut Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<(ValueId, ValueId)> {
    let outer_constant = const_i32(func, rhs)?;
    let InstKind::Binary {
        op: inner_op,
        lhs: inner_lhs,
        rhs: inner_rhs,
    } = defining_inst(func, lhs)?.clone()
    else {
        return None;
    };
    if inner_op != op {
        return None;
    }

    let (base, inner_constant) = one_constant_operand(func, inner_lhs, inner_rhs)?;
    if base == lhs {
        return None;
    }
    let combined = match op {
        BinaryOp::Iadd => inner_constant.wrapping_add(outer_constant),
        BinaryOp::Imul => inner_constant.wrapping_mul(outer_constant),
        _ => return None,
    };
    Some((base, get_or_add_i32_const(func, combined)))
}

fn one_constant_operand(func: &Function, lhs: ValueId, rhs: ValueId) -> Option<(ValueId, i32)> {
    match (const_i32(func, lhs), const_i32(func, rhs)) {
        (Some(constant), None) => Some((rhs, constant)),
        (None, Some(constant)) => Some((lhs, constant)),
        _ => None,
    }
}

/// Matches `x - (x / d) * d` through direct use-def edges.
///
/// Both occurrences of `d` must be known i32 constants with the same nonzero
/// value, and the division must use the exact subtraction dividend.
fn match_remainder_divisor(
    func: &Function,
    dividend: ValueId,
    product: ValueId,
) -> Option<ValueId> {
    let InstKind::Binary {
        op: BinaryOp::Imul,
        lhs: product_lhs,
        rhs: product_rhs,
    } = defining_inst(func, product)?
    else {
        return None;
    };

    for (quotient, factor) in [(*product_lhs, *product_rhs), (*product_rhs, *product_lhs)] {
        let Some(factor_value) = const_i32(func, factor) else {
            continue;
        };
        if factor_value == 0 {
            continue;
        }
        let Some(InstKind::Binary {
            op: BinaryOp::Idiv,
            lhs: division_dividend,
            rhs: divisor,
        }) = defining_inst(func, quotient)
        else {
            continue;
        };
        if *division_dividend == dividend && const_i32(func, *divisor) == Some(factor_value) {
            return Some(*divisor);
        }
    }
    None
}

fn reverse_cmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}
