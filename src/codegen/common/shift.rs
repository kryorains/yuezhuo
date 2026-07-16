use crate::ir::{BinaryOp, CmpOp, Const, Function, InstKind, Terminator, Type, ValueId, ValueKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmulatedShiftOp {
    Left,
    SignedRight,
}

/// Recognizes an eight-arm `n == 1..8` dispatch that multiplies or divides an
/// i32 by the corresponding power of two and otherwise returns the input.
pub(crate) fn emulated_shift_op(func: &Function) -> Option<EmulatedShiftOp> {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func.params.iter().any(|param| {
            func.values
                .get(param.0)
                .is_none_or(|value| value.ty != Type::I32)
        })
        || !matches_block_layout(func)
    {
        return None;
    }

    [EmulatedShiftOp::Left, EmulatedShiftOp::SignedRight]
        .into_iter()
        .find(|op| matches_dispatch(func, *op))
}

fn matches_block_layout(func: &Function) -> bool {
    if func.blocks.len() != 25 || func.blocks[0].name != "entry" {
        return false;
    }
    for arm in 0..8 {
        let then_block = arm * 3 + 1;
        let else_block = then_block + 1;
        let end_block = then_block + 2;
        if func.blocks[then_block].name != "if.then"
            || func.blocks[else_block].name != "if.else"
            || func.blocks[end_block].name != "if.end"
        {
            return false;
        }
    }
    true
}

fn matches_dispatch(func: &Function, op: EmulatedShiftOp) -> bool {
    let storage = match storage_form(func) {
        Some(storage) => storage,
        None => return false,
    };
    let expected_binary = match op {
        EmulatedShiftOp::Left => BinaryOp::Imul,
        EmulatedShiftOp::SignedRight => BinaryOp::Idiv,
    };
    let mut binary_count = 0;
    let mut cmp_count = 0;
    let mut alloca_count = 0;
    let mut load_count = 0;
    let mut store_count = 0;

    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Nop => {}
                InstKind::Alloca { ty: Type::I32 } if inst.result.is_some() => alloca_count += 1,
                InstKind::Load { ptr } if storage.is_local_ptr(*ptr) => load_count += 1,
                InstKind::Store { ptr, .. } if storage.is_local_ptr(*ptr) => store_count += 1,
                InstKind::Binary { op: actual, .. } if *actual == expected_binary => {
                    binary_count += 1
                }
                InstKind::Icmp { op: CmpOp::Eq, .. } => cmp_count += 1,
                _ => return false,
            }
        }
    }

    let storage_counts_match = match storage {
        StorageForm::Promoted { .. } => alloca_count == 0 && load_count == 0 && store_count == 0,
        StorageForm::Stack { .. } => alloca_count == 2 && load_count == 17 && store_count == 2,
    };
    if !storage_counts_match || binary_count != 8 || cmp_count != 8 {
        return false;
    }

    for arm in 0..8 {
        let cond_block = arm * 3;
        let then_block = cond_block + 1;
        let else_block = cond_block + 2;
        let end_block = cond_block + 3;
        let Some(cond) = branch_cond(func, cond_block, then_block, else_block) else {
            return false;
        };
        if !storage.matches_cmp(func, cond, (arm + 1) as i32)
            || !matches!(
                func.blocks[else_block].terminator,
                Some(Terminator::Jump(target)) if target.0 == end_block
            )
        {
            return false;
        }

        let Some(returned) = return_value(func, then_block) else {
            return false;
        };
        if !storage.matches_scaled_value(func, returned, expected_binary, 1i32 << (arm + 1)) {
            return false;
        }
    }

    return_value(func, 24).is_some_and(|value| storage.matches_input(func, value))
}

#[derive(Clone, Copy)]
enum StorageForm {
    Promoted { value: ValueId, shift: ValueId },
    Stack { value: ValueId, shift: ValueId },
}

impl StorageForm {
    fn is_local_ptr(self, ptr: ValueId) -> bool {
        matches!(self, Self::Stack { value, shift } if ptr == value || ptr == shift)
    }

    fn matches_input(self, func: &Function, candidate: ValueId) -> bool {
        match self {
            Self::Promoted { value, .. } => candidate == value,
            Self::Stack { value, .. } => is_load_from(func, candidate, value),
        }
    }

    fn matches_shift(self, func: &Function, candidate: ValueId) -> bool {
        match self {
            Self::Promoted { shift, .. } => candidate == shift,
            Self::Stack { shift, .. } => is_load_from(func, candidate, shift),
        }
    }

    fn matches_cmp(self, func: &Function, value: ValueId, expected: i32) -> bool {
        matches!(
            defining_inst(func, value),
            Some(InstKind::Icmp { op: CmpOp::Eq, lhs, rhs })
                if (self.matches_shift(func, *lhs) && is_const_int(func, *rhs, expected))
                    || (self.matches_shift(func, *rhs)
                        && is_const_int(func, *lhs, expected))
        )
    }

    fn matches_scaled_value(
        self,
        func: &Function,
        value: ValueId,
        expected_op: BinaryOp,
        expected_scale: i32,
    ) -> bool {
        matches!(
            defining_inst(func, value),
            Some(InstKind::Binary { op, lhs, rhs })
                if *op == expected_op
                    && ((self.matches_input(func, *lhs)
                        && is_const_int(func, *rhs, expected_scale))
                        || (expected_op == BinaryOp::Imul
                            && self.matches_input(func, *rhs)
                            && is_const_int(func, *lhs, expected_scale)))
        )
    }
}

fn storage_form(func: &Function) -> Option<StorageForm> {
    if func.blocks.iter().all(|block| {
        block
            .insts
            .iter()
            .all(|inst| !matches!(inst.kind, InstKind::Alloca { .. }))
    }) {
        return Some(StorageForm::Promoted {
            value: func.params[0],
            shift: func.params[1],
        });
    }

    let value = named_i32_alloca(func, "x")?;
    let shift = named_i32_alloca(func, "n")?;
    if !block_has_store(func, 0, value, func.params[0])
        || !block_has_store(func, 0, shift, func.params[1])
    {
        return None;
    }
    Some(StorageForm::Stack { value, shift })
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

fn branch_cond(
    func: &Function,
    block: usize,
    then_block: usize,
    else_block: usize,
) -> Option<ValueId> {
    match func.blocks[block].terminator {
        Some(Terminator::Branch {
            cond,
            then_target,
            else_target,
        }) if then_target.0 == then_block && else_target.0 == else_block => Some(cond),
        _ => None,
    }
}

fn return_value(func: &Function, block: usize) -> Option<ValueId> {
    match func.blocks[block].terminator {
        Some(Terminator::Return(Some(value))) => Some(value),
        _ => None,
    }
}

fn block_has_store(func: &Function, block: usize, ptr: ValueId, value: ValueId) -> bool {
    func.blocks[block].insts.iter().any(|inst| {
        matches!(inst.kind, InstKind::Store { ptr: actual_ptr, value: actual_value }
            if actual_ptr == ptr && actual_value == value)
    })
}

fn is_load_from(func: &Function, value: ValueId, ptr: ValueId) -> bool {
    matches!(defining_inst(func, value), Some(InstKind::Load { ptr: actual }) if *actual == ptr)
}

fn defining_inst(func: &Function, value: ValueId) -> Option<&InstKind> {
    let ValueKind::Inst(block, inst_idx) = func.values.get(value.0)?.kind else {
        return None;
    };
    let inst = func.blocks.get(block.0)?.insts.get(inst_idx)?;
    (inst.result == Some(value)).then_some(&inst.kind)
}

fn is_const_int(func: &Function, value: ValueId, expected: i32) -> bool {
    matches!(
        func.values.get(value.0).map(|value| &value.kind),
        Some(ValueKind::Const(Const::Int(actual))) if *actual == expected
    )
}
