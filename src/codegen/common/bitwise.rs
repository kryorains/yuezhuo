use crate::ir::{BinaryOp, CmpOp, Const, Function, InstKind, Terminator, Type, ValueId, ValueKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmulatedBitwiseOp {
    And,
    Or,
    Xor,
}

/// Recognizes the canonical 32-iteration bitwise helpers used by SysY programs
/// that cannot spell integer bit operations directly. The fast path is only
/// valid for non-negative operands; target emitters retain the original body as
/// a fallback for negative inputs.
pub(crate) fn emulated_bitwise_op(func: &Function) -> Option<EmulatedBitwiseOp> {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func.params.iter().any(|param| {
            func.values
                .get(param.0)
                .is_none_or(|value| value.ty != Type::I32)
        })
    {
        return None;
    }

    [
        EmulatedBitwiseOp::And,
        EmulatedBitwiseOp::Or,
        EmulatedBitwiseOp::Xor,
    ]
    .into_iter()
    .find(|op| matches_helper_cfg(func, *op) && matches_helper_instructions(func, *op))
}

fn matches_helper_cfg(func: &Function, op: EmulatedBitwiseOp) -> bool {
    let expected_names: &[&str] = match op {
        EmulatedBitwiseOp::And => &[
            "entry",
            "while.cond",
            "while.body",
            "while.end",
            "if.then",
            "if.else",
            "if.end",
            "land.rhs",
            "land.end",
        ],
        EmulatedBitwiseOp::Or => &[
            "entry",
            "while.cond",
            "while.body",
            "while.end",
            "if.then",
            "if.else",
            "if.end",
            "lor.rhs",
            "lor.end",
        ],
        EmulatedBitwiseOp::Xor => &[
            "entry",
            "while.cond",
            "while.body",
            "while.end",
            "if.then",
            "if.else",
            "if.end",
        ],
    };
    if func
        .blocks
        .iter()
        .map(|block| block.name.as_str())
        .ne(expected_names.iter().copied())
        || !is_jump(func, 0, 1)
        || !is_branch(func, 1, 2, 3)
        || !matches!(func.blocks[3].terminator, Some(Terminator::Return(Some(_))))
    {
        return false;
    }

    match op {
        EmulatedBitwiseOp::And => {
            is_branch(func, 2, 7, 8)
                && is_jump(func, 7, 8)
                && is_branch(func, 8, 4, 5)
                && is_jump(func, 4, 6)
                && is_jump(func, 5, 6)
                && is_jump(func, 6, 1)
        }
        EmulatedBitwiseOp::Or => {
            is_branch(func, 2, 8, 7)
                && is_jump(func, 7, 8)
                && is_branch(func, 8, 4, 5)
                && is_jump(func, 4, 6)
                && is_jump(func, 5, 6)
                && is_jump(func, 6, 1)
        }
        EmulatedBitwiseOp::Xor => {
            is_branch(func, 2, 4, 5)
                && is_jump(func, 4, 6)
                && is_jump(func, 5, 6)
                && is_jump(func, 6, 1)
        }
    }
}

fn matches_helper_instructions(func: &Function, op: EmulatedBitwiseOp) -> bool {
    let local_allocas = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match (&inst.kind, inst.result) {
            (InstKind::Alloca { ty: Type::I32 }, Some(result)) => Some(result),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut phi_count = 0;
    let mut alloca_count = 0;
    let mut load_count = 0;
    let mut store_count = 0;
    let mut iadd_count = 0;
    let mut isub_one_count = 0;
    let mut imul_two_count = 0;
    let mut idiv_two_count = 0;
    let mut imod_two_count = 0;
    let mut eq_count = 0;
    let mut ne_count = 0;

    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Nop => {}
                InstKind::Phi { .. } => phi_count += 1,
                InstKind::Alloca { ty: Type::I32 } if inst.result.is_some() => alloca_count += 1,
                InstKind::Load { ptr } if local_allocas.contains(ptr) => load_count += 1,
                InstKind::Store { ptr, .. } if local_allocas.contains(ptr) => store_count += 1,
                InstKind::Binary { op, lhs, rhs } => match op {
                    BinaryOp::Iadd => iadd_count += 1,
                    BinaryOp::Isub if is_const_int(func, *rhs, 1) => isub_one_count += 1,
                    BinaryOp::Imul
                        if is_const_int(func, *lhs, 2) || is_const_int(func, *rhs, 2) =>
                    {
                        imul_two_count += 1
                    }
                    BinaryOp::Idiv if is_const_int(func, *rhs, 2) => idiv_two_count += 1,
                    BinaryOp::Imod if is_const_int(func, *rhs, 2) => imod_two_count += 1,
                    _ => return false,
                },
                InstKind::Icmp { op, .. } => match op {
                    CmpOp::Eq => eq_count += 1,
                    CmpOp::Ne => ne_count += 1,
                    _ => return false,
                },
                _ => return false,
            }
        }
    }

    let (optimized_phis, unoptimized_phis, expected_eq, expected_ne) = match op {
        EmulatedBitwiseOp::And | EmulatedBitwiseOp::Or => (7, 1, 2, 1),
        EmulatedBitwiseOp::Xor => (6, 0, 0, 2),
    };
    let optimized_shape =
        phi_count == optimized_phis && alloca_count == 0 && load_count == 0 && store_count == 0;
    let unoptimized_shape =
        phi_count == unoptimized_phis && alloca_count == 7 && load_count == 12 && store_count == 12;
    let operations_match = iadd_count == 1
        && isub_one_count == 1
        && imul_two_count == 1
        && idiv_two_count == 2
        && imod_two_count == 2
        && eq_count == expected_eq
        && ne_count == expected_ne;
    operations_match
        && ((optimized_shape && matches_promoted_dataflow(func, op))
            || (unoptimized_shape && matches_stack_dataflow(func, op)))
}

fn matches_promoted_dataflow(func: &Function, op: EmulatedBitwiseOp) -> bool {
    let Some((a, next_a)) = find_loop_phi(func, func.params[0], |func, phi, next| {
        value_defined_in(func, next, 2) && is_binary_const(func, next, BinaryOp::Idiv, phi, 2)
    }) else {
        return false;
    };
    let Some((b, next_b)) = find_loop_phi(func, func.params[1], |func, phi, next| {
        value_defined_in(func, next, 2) && is_binary_const(func, next, BinaryOp::Idiv, phi, 2)
    }) else {
        return false;
    };
    let Some((len, next_len)) = find_loop_phi_by_const(func, 32, |func, phi, next| {
        value_defined_in(func, next, 6) && is_binary_const(func, next, BinaryOp::Isub, phi, 1)
    }) else {
        return false;
    };
    let Some((power, next_power)) = find_loop_phi_by_const(func, 1, |func, phi, next| {
        value_defined_in(func, next, 6)
            && is_commutative_binary_const(func, next, BinaryOp::Imul, phi, 2)
    }) else {
        return false;
    };
    let Some((result, next_result)) = find_loop_phi_by_const(func, 0, |func, _, next| {
        value_defined_in(func, next, 6)
            && matches!(defining_inst(func, next), Some(InstKind::Phi { .. }))
    }) else {
        return false;
    };

    let Some(bit_a) = find_binary_result(func, 2, BinaryOp::Imod, a, 2) else {
        return false;
    };
    let Some(bit_b) = find_binary_result(func, 2, BinaryOp::Imod, b, 2) else {
        return false;
    };
    let Some(header_cond) = branch_cond(func, 1) else {
        return false;
    };
    if !is_cmp_const(func, header_cond, CmpOp::Ne, len, 0)
        || return_value(func, 3) != Some(result)
        || phi_incoming(func, a, 6) != Some(next_a)
        || phi_incoming(func, b, 6) != Some(next_b)
        || phi_incoming(func, len, 6) != Some(next_len)
        || phi_incoming(func, power, 6) != Some(next_power)
        || phi_incoming(func, result, 6) != Some(next_result)
    {
        return false;
    }

    let Some(add_result) = find_commutative_binary_result(func, 4, BinaryOp::Iadd, result, power)
    else {
        return false;
    };
    if !phi_has_exact_incomings(func, next_result, &[(4, add_result), (5, result)]) {
        return false;
    }

    match op {
        EmulatedBitwiseOp::Xor => branch_cond(func, 2)
            .is_some_and(|cond| is_cmp_values(func, cond, CmpOp::Ne, bit_a, bit_b)),
        EmulatedBitwiseOp::And | EmulatedBitwiseOp::Or => {
            let Some(first_cond) = branch_cond(func, 2) else {
                return false;
            };
            let Some(second_cond) = block_result_matching(func, 7, |kind| {
                matches!(kind, InstKind::Icmp { op: CmpOp::Eq, lhs, rhs }
                    if (*lhs == bit_b && is_const_int(func, *rhs, 1))
                        || (*rhs == bit_b && is_const_int(func, *lhs, 1)))
            }) else {
                return false;
            };
            let Some(combined_cond) = branch_cond(func, 8) else {
                return false;
            };
            let short_circuit = op == EmulatedBitwiseOp::Or;
            is_cmp_const(func, first_cond, CmpOp::Eq, bit_a, 1)
                && phi_has_exact_incomings_with_const(
                    func,
                    combined_cond,
                    2,
                    short_circuit,
                    7,
                    second_cond,
                )
        }
    }
}

fn matches_stack_dataflow(func: &Function, op: EmulatedBitwiseOp) -> bool {
    let Some(ptr_a) = named_i32_alloca(func, "a") else {
        return false;
    };
    let Some(ptr_b) = named_i32_alloca(func, "b") else {
        return false;
    };
    let Some(ptr_bit_a) = named_i32_alloca(func, "bit_a") else {
        return false;
    };
    let Some(ptr_bit_b) = named_i32_alloca(func, "bit_b") else {
        return false;
    };
    let Some(ptr_len) = named_i32_alloca(func, "len") else {
        return false;
    };
    let Some(ptr_result) = named_i32_alloca(func, "result") else {
        return false;
    };
    let Some(ptr_power) = named_i32_alloca(func, "power") else {
        return false;
    };

    if !block_has_store(func, 0, ptr_a, func.params[0])
        || !block_has_store(func, 0, ptr_b, func.params[1])
        || !block_has_store_const(func, 0, ptr_len, 32)
        || !block_has_store_const(func, 0, ptr_result, 0)
        || !block_has_store_const(func, 0, ptr_power, 1)
    {
        return false;
    }

    let Some(header_cond) = branch_cond(func, 1) else {
        return false;
    };
    let Some(returned) = return_value(func, 3) else {
        return false;
    };
    if !is_cmp_load_const(func, header_cond, CmpOp::Ne, ptr_len, 0)
        || !is_load_from(func, returned, ptr_result)
    {
        return false;
    }

    let Some(bit_a) = find_binary_from_load(func, 2, BinaryOp::Imod, ptr_a, 2) else {
        return false;
    };
    let Some(bit_b) = find_binary_from_load(func, 2, BinaryOp::Imod, ptr_b, 2) else {
        return false;
    };
    let Some(next_a) = find_binary_from_load(func, 2, BinaryOp::Idiv, ptr_a, 2) else {
        return false;
    };
    let Some(next_b) = find_binary_from_load(func, 2, BinaryOp::Idiv, ptr_b, 2) else {
        return false;
    };
    if !block_has_store(func, 2, ptr_bit_a, bit_a)
        || !block_has_store(func, 2, ptr_bit_b, bit_b)
        || !block_has_store(func, 2, ptr_a, next_a)
        || !block_has_store(func, 2, ptr_b, next_b)
        || !load_precedes_update(func, 2, bit_a, ptr_a, next_a)
        || !load_precedes_update(func, 2, bit_b, ptr_b, next_b)
    {
        return false;
    }

    let Some(add_result) =
        find_binary_from_two_loads(func, 4, BinaryOp::Iadd, ptr_result, ptr_power)
    else {
        return false;
    };
    let Some(next_power) = find_binary_from_load(func, 6, BinaryOp::Imul, ptr_power, 2) else {
        return false;
    };
    let Some(next_len) = find_binary_from_load(func, 6, BinaryOp::Isub, ptr_len, 1) else {
        return false;
    };
    if !block_has_store(func, 4, ptr_result, add_result)
        || !block_has_store(func, 6, ptr_power, next_power)
        || !block_has_store(func, 6, ptr_len, next_len)
    {
        return false;
    }

    match op {
        EmulatedBitwiseOp::Xor => branch_cond(func, 2)
            .is_some_and(|cond| is_cmp_two_loads(func, cond, CmpOp::Ne, ptr_bit_a, ptr_bit_b)),
        EmulatedBitwiseOp::And | EmulatedBitwiseOp::Or => {
            let Some(first_cond) = branch_cond(func, 2) else {
                return false;
            };
            let Some(second_cond) = block_result_matching(func, 7, |kind| {
                matches!(kind, InstKind::Icmp { op: CmpOp::Eq, lhs, rhs }
                    if (is_load_from(func, *lhs, ptr_bit_b) && is_const_int(func, *rhs, 1))
                        || (is_load_from(func, *rhs, ptr_bit_b)
                            && is_const_int(func, *lhs, 1)))
            }) else {
                return false;
            };
            let Some(combined_cond) = branch_cond(func, 8) else {
                return false;
            };
            let short_circuit = op == EmulatedBitwiseOp::Or;
            is_cmp_load_const(func, first_cond, CmpOp::Eq, ptr_bit_a, 1)
                && phi_has_exact_incomings_with_const(
                    func,
                    combined_cond,
                    2,
                    short_circuit,
                    7,
                    second_cond,
                )
        }
    }
}

fn find_loop_phi(
    func: &Function,
    initial: ValueId,
    back_matches: impl Fn(&Function, ValueId, ValueId) -> bool,
) -> Option<(ValueId, ValueId)> {
    block_results(func, 1).find_map(|phi| {
        let first = phi_incoming(func, phi, 0)?;
        let next = phi_incoming(func, phi, 6)?;
        (first == initial && back_matches(func, phi, next)).then_some((phi, next))
    })
}

fn find_loop_phi_by_const(
    func: &Function,
    initial: i32,
    back_matches: impl Fn(&Function, ValueId, ValueId) -> bool,
) -> Option<(ValueId, ValueId)> {
    block_results(func, 1).find_map(|phi| {
        let first = phi_incoming(func, phi, 0)?;
        let next = phi_incoming(func, phi, 6)?;
        (is_const_int(func, first, initial) && back_matches(func, phi, next)).then_some((phi, next))
    })
}

fn phi_incoming(func: &Function, phi: ValueId, pred: usize) -> Option<ValueId> {
    let InstKind::Phi { incomings } = defining_inst(func, phi)? else {
        return None;
    };
    if incomings.len() != 2 {
        return None;
    }
    incomings
        .iter()
        .find_map(|(incoming_pred, value)| (incoming_pred.0 == pred).then_some(*value))
}

fn phi_has_exact_incomings(func: &Function, phi: ValueId, expected: &[(usize, ValueId)]) -> bool {
    let Some(InstKind::Phi { incomings }) = defining_inst(func, phi) else {
        return false;
    };
    incomings.len() == expected.len()
        && expected.iter().all(|(pred, value)| {
            incomings
                .iter()
                .any(|(actual_pred, actual_value)| actual_pred.0 == *pred && actual_value == value)
        })
}

fn phi_has_exact_incomings_with_const(
    func: &Function,
    phi: ValueId,
    const_pred: usize,
    const_value: bool,
    value_pred: usize,
    value: ValueId,
) -> bool {
    let Some(InstKind::Phi { incomings }) = defining_inst(func, phi) else {
        return false;
    };
    incomings.len() == 2
        && incomings.iter().any(|(pred, incoming)| {
            pred.0 == const_pred && is_const_bool(func, *incoming, const_value)
        })
        && incomings
            .iter()
            .any(|(pred, incoming)| pred.0 == value_pred && *incoming == value)
}

fn find_binary_result(
    func: &Function,
    block: usize,
    op: BinaryOp,
    lhs: ValueId,
    rhs_const: i32,
) -> Option<ValueId> {
    block_result_matching(func, block, |kind| {
        matches!(kind, InstKind::Binary { op: actual, lhs: actual_lhs, rhs }
            if *actual == op && *actual_lhs == lhs && is_const_int(func, *rhs, rhs_const))
    })
}

fn find_commutative_binary_result(
    func: &Function,
    block: usize,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<ValueId> {
    block_result_matching(func, block, |kind| {
        matches!(kind, InstKind::Binary { op: actual, lhs: actual_lhs, rhs: actual_rhs }
            if *actual == op
                && ((*actual_lhs == lhs && *actual_rhs == rhs)
                    || (*actual_lhs == rhs && *actual_rhs == lhs)))
    })
}

fn find_binary_from_load(
    func: &Function,
    block: usize,
    op: BinaryOp,
    ptr: ValueId,
    rhs_const: i32,
) -> Option<ValueId> {
    block_result_matching(func, block, |kind| {
        matches!(kind, InstKind::Binary { op: actual, lhs, rhs }
            if *actual == op
                && is_load_from(func, *lhs, ptr)
                && is_const_int(func, *rhs, rhs_const))
    })
}

fn find_binary_from_two_loads(
    func: &Function,
    block: usize,
    op: BinaryOp,
    lhs_ptr: ValueId,
    rhs_ptr: ValueId,
) -> Option<ValueId> {
    block_result_matching(func, block, |kind| {
        matches!(kind, InstKind::Binary { op: actual, lhs, rhs }
            if *actual == op
                && ((is_load_from(func, *lhs, lhs_ptr) && is_load_from(func, *rhs, rhs_ptr))
                    || (is_load_from(func, *lhs, rhs_ptr)
                        && is_load_from(func, *rhs, lhs_ptr))))
    })
}

fn is_binary_const(
    func: &Function,
    value: ValueId,
    op: BinaryOp,
    lhs: ValueId,
    rhs_const: i32,
) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Binary { op: actual, lhs: actual_lhs, rhs })
            if *actual == op && *actual_lhs == lhs && is_const_int(func, *rhs, rhs_const)
    )
}

fn is_commutative_binary_const(
    func: &Function,
    value: ValueId,
    op: BinaryOp,
    operand: ValueId,
    constant: i32,
) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Binary { op: actual, lhs, rhs })
            if *actual == op
                && ((*lhs == operand && is_const_int(func, *rhs, constant))
                    || (*rhs == operand && is_const_int(func, *lhs, constant)))
    )
}

fn is_cmp_const(
    func: &Function,
    value: ValueId,
    op: CmpOp,
    operand: ValueId,
    constant: i32,
) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Icmp { op: actual, lhs, rhs })
            if *actual == op
                && ((*lhs == operand && is_const_int(func, *rhs, constant))
                    || (*rhs == operand && is_const_int(func, *lhs, constant)))
    )
}

fn is_cmp_values(func: &Function, value: ValueId, op: CmpOp, lhs: ValueId, rhs: ValueId) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Icmp { op: actual, lhs: actual_lhs, rhs: actual_rhs })
            if *actual == op
                && ((*actual_lhs == lhs && *actual_rhs == rhs)
                    || (*actual_lhs == rhs && *actual_rhs == lhs))
    )
}

fn is_cmp_load_const(
    func: &Function,
    value: ValueId,
    op: CmpOp,
    ptr: ValueId,
    constant: i32,
) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Icmp { op: actual, lhs, rhs })
            if *actual == op
                && ((is_load_from(func, *lhs, ptr) && is_const_int(func, *rhs, constant))
                    || (is_load_from(func, *rhs, ptr)
                        && is_const_int(func, *lhs, constant)))
    )
}

fn is_cmp_two_loads(
    func: &Function,
    value: ValueId,
    op: CmpOp,
    lhs_ptr: ValueId,
    rhs_ptr: ValueId,
) -> bool {
    matches!(
        defining_inst(func, value),
        Some(InstKind::Icmp { op: actual, lhs, rhs })
            if *actual == op
                && ((is_load_from(func, *lhs, lhs_ptr) && is_load_from(func, *rhs, rhs_ptr))
                    || (is_load_from(func, *lhs, rhs_ptr)
                        && is_load_from(func, *rhs, lhs_ptr)))
    )
}

fn named_i32_alloca(func: &Function, name: &str) -> Option<ValueId> {
    func.values.iter().enumerate().find_map(|(idx, value)| {
        let id = ValueId(idx);
        (value.name.as_deref() == Some(name)
            && value.ty == Type::Ptr(Box::new(Type::I32))
            && matches!(
                defining_inst(func, id),
                Some(InstKind::Alloca { ty: Type::I32 })
            ))
        .then_some(id)
    })
}

fn block_has_store(func: &Function, block: usize, ptr: ValueId, value: ValueId) -> bool {
    func.blocks[block].insts.iter().any(|inst| {
        matches!(inst.kind, InstKind::Store { ptr: actual_ptr, value: actual_value }
            if actual_ptr == ptr && actual_value == value)
    })
}

fn block_has_store_const(func: &Function, block: usize, ptr: ValueId, value: i32) -> bool {
    func.blocks[block].insts.iter().any(|inst| {
        matches!(inst.kind, InstKind::Store { ptr: actual_ptr, value: actual_value }
            if actual_ptr == ptr && is_const_int(func, actual_value, value))
    })
}

fn is_load_from(func: &Function, value: ValueId, ptr: ValueId) -> bool {
    matches!(defining_inst(func, value), Some(InstKind::Load { ptr: actual }) if *actual == ptr)
}

fn block_results(func: &Function, block: usize) -> impl Iterator<Item = ValueId> + '_ {
    func.blocks[block]
        .insts
        .iter()
        .filter_map(|inst| inst.result)
}

fn block_result_matching(
    func: &Function,
    block: usize,
    matches_kind: impl Fn(&InstKind) -> bool,
) -> Option<ValueId> {
    func.blocks[block]
        .insts
        .iter()
        .find_map(|inst| matches_kind(&inst.kind).then_some(inst.result).flatten())
}

fn load_precedes_update(
    func: &Function,
    block: usize,
    loaded_value: ValueId,
    updated_ptr: ValueId,
    updated_value: ValueId,
) -> bool {
    let Some(ValueKind::Inst(owner, load_pos)) =
        func.values.get(loaded_value.0).map(|value| &value.kind)
    else {
        return false;
    };
    if owner.0 != block {
        return false;
    }
    func.blocks[block]
        .insts
        .iter()
        .position(|inst| {
            matches!(inst.kind, InstKind::Store { ptr, value }
                if ptr == updated_ptr && value == updated_value)
        })
        .is_some_and(|store_pos| *load_pos < store_pos)
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.values.get(value.0)?.kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

fn value_defined_in(func: &Function, value: ValueId, block: usize) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Inst(owner, _)) if owner.0 == block
    )
}

fn branch_cond(func: &Function, block: usize) -> Option<ValueId> {
    match func.blocks[block].terminator {
        Some(Terminator::Branch { cond, .. }) => Some(cond),
        _ => None,
    }
}

fn return_value(func: &Function, block: usize) -> Option<ValueId> {
    match func.blocks[block].terminator {
        Some(Terminator::Return(Some(value))) => Some(value),
        _ => None,
    }
}

fn is_const_bool(func: &Function, value: ValueId, expected: bool) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Const(Const::Bool(actual))) if *actual == expected
    )
}

fn is_jump(func: &Function, block: usize, target: usize) -> bool {
    matches!(
        func.blocks[block].terminator,
        Some(Terminator::Jump(actual)) if actual.0 == target
    )
}

fn is_branch(func: &Function, block: usize, then_target: usize, else_target: usize) -> bool {
    matches!(
        func.blocks[block].terminator,
        Some(Terminator::Branch {
            then_target: actual_then,
            else_target: actual_else,
            ..
        }) if actual_then.0 == then_target && actual_else.0 == else_target
    )
}

fn is_const_int(func: &Function, value: ValueId, expected: i32) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Const(Const::Int(actual))) if *actual == expected
    )
}
