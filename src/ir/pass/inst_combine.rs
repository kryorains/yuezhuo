use super::util::{const_i32, defining_inst, get_or_add_i32_const};
use super::ModulePass;
use crate::ir::{BinaryOp, CmpOp, Function, InstKind, Module, Terminator, ValueId};
use std::collections::HashMap;

/// Canonicalizes and combines local integer instructions.
///
/// Every rule is based only on an instruction and the definitions of its
/// operands. In particular, this pass does not inspect source names, block
/// names, or whole-function control-flow shapes.
pub(super) struct InstCombinePass {
    divisibility_only: bool,
}

impl InstCombinePass {
    pub(super) fn new() -> Self {
        Self {
            divisibility_only: false,
        }
    }

    pub(super) fn divisibility_only() -> Self {
        Self {
            divisibility_only: true,
        }
    }
}

impl ModulePass for InstCombinePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            if self.divisibility_only {
                combine_divisibility_remainders(func);
                combine_parity_products(func);
                if let Err(errors) = func.verify() {
                    panic!(
                        "divisibility combining produced invalid IR in {}: {:?}",
                        func.name, errors
                    );
                }
            } else {
                combine_function(func);
            }
        }
    }
}

fn combine_function(func: &mut Function) {
    combine_divisibility_remainders(func);
    combine_parity_products(func);

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

/// Replaces a product with a bitwise AND when every observation retains only
/// bit zero. In two's-complement modular arithmetic the low bit of `a * b` is
/// exactly the low bit of `a & b`; keeping the users' `& 1` preserves their
/// complete i32 result while avoiding a multiply on parity-only paths.
fn combine_parity_products(func: &mut Function) {
    const MAX_VALUES: usize = 16_384;
    const MAX_INSTRUCTIONS: usize = 65_536;
    const MAX_OPERAND_WORK: usize = 262_144;

    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    if func.values.len() > MAX_VALUES
        || instruction_count.is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return;
    }

    #[derive(Clone, Copy)]
    struct Candidate {
        block: usize,
        inst: usize,
        lhs: ValueId,
        rhs: ValueId,
        found_use: bool,
        valid: bool,
    }

    let mut candidates = vec![None; func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            let (
                Some(result),
                InstKind::Binary {
                    op: BinaryOp::Imul,
                    lhs,
                    rhs,
                },
            ) = (inst.result, &inst.kind)
            else {
                continue;
            };
            candidates[result.0] = Some(Candidate {
                block: block_idx,
                inst: inst_idx,
                lhs: *lhs,
                rhs: *rhs,
                found_use: false,
                valid: true,
            });
        }
    }

    let mut operand_work = 0usize;
    for block in &func.blocks {
        for inst in &block.insts {
            let operands = inst_operands(&inst.kind);
            operand_work = operand_work.saturating_add(operands.len());
            if operand_work > MAX_OPERAND_WORK {
                return;
            }
            for product in operands {
                let Some(candidate) = candidates.get_mut(product.0).and_then(Option::as_mut) else {
                    continue;
                };
                candidate.found_use = true;
                candidate.valid &= matches!(
                    &inst.kind,
                    InstKind::Binary {
                        op: BinaryOp::Iand,
                        lhs,
                        rhs,
                    } if (*lhs == product && const_i32(func, *rhs) == Some(1))
                        || (*rhs == product && const_i32(func, *lhs) == Some(1))
                );
            }
        }
        if let Some(terminator) = &block.terminator {
            for product in terminator_operands(terminator) {
                operand_work = operand_work.saturating_add(1);
                if operand_work > MAX_OPERAND_WORK {
                    return;
                }
                if let Some(candidate) = candidates.get_mut(product.0).and_then(Option::as_mut) {
                    candidate.found_use = true;
                    candidate.valid = false;
                }
            }
        }
    }

    for candidate in candidates.into_iter().flatten() {
        if candidate.found_use && candidate.valid {
            func.blocks[candidate.block].insts[candidate.inst].kind = InstKind::Binary {
                op: BinaryOp::Iand,
                lhs: candidate.lhs,
                rhs: candidate.rhs,
            };
        }
    }
}

/// Rewrites a signed remainder to a mask only when every observation asks
/// whether that remainder is zero. For a power-of-two divisor this preserves
/// divisibility for positive and negative dividends without changing the
/// remainder value in contexts where its sign would matter.
fn combine_divisibility_remainders(func: &mut Function) {
    const MAX_VALUES: usize = 16_384;
    const MAX_INSTRUCTIONS: usize = 65_536;
    const MAX_OPERAND_WORK: usize = 262_144;

    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    if func.values.len() > MAX_VALUES
        || instruction_count.is_none_or(|count| count > MAX_INSTRUCTIONS)
    {
        return;
    }

    #[derive(Clone, Copy)]
    struct Candidate {
        block: usize,
        inst: usize,
        dividend: ValueId,
        mask: i32,
        found_use: bool,
        valid: bool,
    }

    let mut candidates = vec![None; func.values.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.insts.iter().enumerate() {
            let (
                Some(result),
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs,
                    rhs,
                },
            ) = (inst.result, &inst.kind)
            else {
                continue;
            };
            let Some(divisor) = const_i32(func, *rhs) else {
                continue;
            };
            if divisor == 0 {
                continue;
            }
            let magnitude = divisor.wrapping_abs() as u32;
            if magnitude.is_power_of_two() {
                candidates[result.0] = Some(Candidate {
                    block: block_idx,
                    inst: inst_idx,
                    dividend: *lhs,
                    mask: magnitude.wrapping_sub(1) as i32,
                    found_use: false,
                    valid: true,
                });
            }
        }
    }

    let mut operand_work = 0usize;
    for block in &func.blocks {
        for inst in &block.insts {
            let operands = inst_operands(&inst.kind);
            operand_work = operand_work.saturating_add(operands.len());
            if operand_work > MAX_OPERAND_WORK {
                return;
            }
            for remainder in operands {
                let Some(candidate) = candidates.get_mut(remainder.0).and_then(Option::as_mut)
                else {
                    continue;
                };
                candidate.found_use = true;
                let valid = matches!(
                    &inst.kind,
                    InstKind::Icmp { op, lhs, rhs }
                        if matches!(*op, CmpOp::Eq | CmpOp::Ne)
                            && ((*lhs == remainder && const_i32(func, *rhs) == Some(0))
                                || (*rhs == remainder && const_i32(func, *lhs) == Some(0)))
                );
                candidate.valid &= valid;
            }
        }
        if let Some(terminator) = &block.terminator {
            let operands = terminator_operands(terminator);
            operand_work = operand_work.saturating_add(operands.len());
            if operand_work > MAX_OPERAND_WORK {
                return;
            }
            for remainder in operands {
                if let Some(candidate) = candidates.get_mut(remainder.0).and_then(Option::as_mut) {
                    candidate.found_use = true;
                    candidate.valid = false;
                }
            }
        }
    }

    let mut mask_values = HashMap::<i32, ValueId>::new();
    for candidate in candidates.into_iter().flatten() {
        if !candidate.found_use || !candidate.valid {
            continue;
        }
        let mask = if let Some(mask) = mask_values.get(&candidate.mask).copied() {
            mask
        } else {
            let mask = get_or_add_i32_const(func, candidate.mask);
            mask_values.insert(candidate.mask, mask);
            mask
        };
        func.blocks[candidate.block].insts[candidate.inst].kind = InstKind::Binary {
            op: BinaryOp::Iand,
            lhs: candidate.dividend,
            rhs: mask,
        };
    }
}

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut operands = vec![*base];
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Call { args, .. } => args.clone(),
        InstKind::MemZero { ptr, .. } => vec![*ptr],
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
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

    if op == BinaryOp::Idiv {
        if let Some(combined) = combine_positive_constant_divisions(func, lhs, rhs) {
            return combined;
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

/// Reassociates `(x / a) / b` to `x / (a * b)` for positive constants.
///
/// Signed integer division truncates toward zero, so this is exact when both
/// divisors are positive. If their product is larger than every possible i32
/// magnitude, the quotient is identically zero. The boundary product `2^31`
/// is deliberately left alone because `i32::MIN / 2^31 == -1`.
fn combine_positive_constant_divisions(
    func: &mut Function,
    quotient: ValueId,
    outer_divisor: ValueId,
) -> Option<InstKind> {
    let outer = const_i32(func, outer_divisor)?;
    if outer <= 0 {
        return None;
    }
    let InstKind::Binary {
        op: BinaryOp::Idiv,
        lhs: dividend,
        rhs: inner_divisor,
    } = defining_inst(func, quotient)?.clone()
    else {
        return None;
    };
    let inner = const_i32(func, inner_divisor)?;
    if inner <= 0 {
        return None;
    }

    let product = i64::from(inner) * i64::from(outer);
    if product <= i64::from(i32::MAX) {
        return Some(InstKind::Binary {
            op: BinaryOp::Idiv,
            lhs: dividend,
            rhs: get_or_add_i32_const(func, product as i32),
        });
    }
    if product > i64::from(i32::MAX) + 1 {
        return Some(InstKind::Binary {
            op: BinaryOp::Iand,
            lhs: dividend,
            rhs: get_or_add_i32_const(func, 0),
        });
    }
    None
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
