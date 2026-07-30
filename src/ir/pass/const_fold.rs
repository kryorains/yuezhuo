use super::util::{rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, CastOp, CmpOp, Const, Function, InstKind, Module, Type, UnaryOp, ValueId, ValueKind,
};

pub(super) struct ConstFoldPass;

impl ConstFoldPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for ConstFoldPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            fold_function(func);
        }
    }
}

fn fold_function(func: &mut Function) {
    // 反复扫描直到收敛：一次改写使用点后，可能让更多指令变成可折叠常量。
    loop {
        let mut replacements = ValueReplacements::new();
        let mut changed = false;

        for block_idx in 0..func.blocks.len() {
            let inst_len = func.blocks[block_idx].insts.len();
            for inst_idx in 0..inst_len {
                let inst = func.blocks[block_idx].insts[inst_idx].clone();
                let Some(result) = inst.result else {
                    continue;
                };

                if let Some(value) = fold_inst(func, &inst.kind) {
                    // 真正的常量折叠：结果值变成 Const，原指令本体改成 Nop。
                    func.values[result.0].kind = ValueKind::Const(value);
                    func.blocks[block_idx].insts[inst_idx].result = None;
                    func.blocks[block_idx].insts[inst_idx].kind = InstKind::Nop;
                    changed = true;
                    continue;
                }

                if let Some(replacement) = simplify_inst(func, &inst.kind) {
                    // 代数化简不一定产生新常量，先记录替换，最后统一改写所有使用点。
                    replacements.insert(result, replacement);
                }
            }
        }

        changed |= rewrite_function_uses(func, &replacements);
        if !changed {
            break;
        }
    }
}

fn fold_inst(func: &Function, kind: &InstKind) -> Option<Const> {
    // 只有所有输入都能读成 Const 时，才会在编译期直接求值。
    match kind {
        InstKind::Unary { op, value } => fold_unary(*op, const_value(func, *value)?),
        InstKind::Binary { op, lhs, rhs } => {
            fold_binary(*op, const_value(func, *lhs)?, const_value(func, *rhs)?)
        }
        InstKind::Icmp { op, lhs, rhs } => {
            fold_icmp(*op, const_value(func, *lhs)?, const_value(func, *rhs)?)
        }
        InstKind::Fcmp { op, lhs, rhs } => {
            fold_fcmp(*op, const_value(func, *lhs)?, const_value(func, *rhs)?)
        }
        InstKind::Cast { op, value } => fold_cast(*op, const_value(func, *value)?),
        _ => None,
    }
}

fn simplify_inst(func: &mut Function, kind: &InstKind) -> Option<ValueId> {
    // 这里处理 x + 0、x * 1、无意义 cast、退化 phi 等不需要完整求值的规则。
    match kind {
        InstKind::Unary {
            op: UnaryOp::Not,
            value,
        } => match const_value(func, *value) {
            Some(Const::Bool(false)) => Some(get_or_add_const(func, Const::Bool(true))),
            Some(Const::Bool(true)) => Some(get_or_add_const(func, Const::Bool(false))),
            _ => None,
        },
        InstKind::Cast { op, value } if is_noop_cast(*op, &func.value(*value).ty) => Some(*value),
        InstKind::Phi { incomings } => simplify_phi(incomings),
        InstKind::Binary { op, lhs, rhs } => simplify_binary(func, *op, *lhs, *rhs),
        InstKind::Icmp {
            op: CmpOp::Eq,
            lhs,
            rhs,
        } if lhs == rhs => Some(get_or_add_const(func, Const::Bool(true))),
        InstKind::Icmp {
            op: CmpOp::Ne,
            lhs,
            rhs,
        } if lhs == rhs => Some(get_or_add_const(func, Const::Bool(false))),
        _ => None,
    }
}

fn simplify_phi(incomings: &[(crate::ir::BlockId, ValueId)]) -> Option<ValueId> {
    let first = incomings.first()?.1;
    incomings
        .iter()
        .all(|(_, value)| *value == first)
        .then_some(first)
}

fn simplify_binary(
    func: &mut Function,
    op: BinaryOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<ValueId> {
    match op {
        BinaryOp::Iadd => match (const_int(func, lhs), const_int(func, rhs)) {
            (Some(0), _) => Some(rhs),
            (_, Some(0)) => Some(lhs),
            _ => None,
        },
        BinaryOp::Isub => match const_int(func, rhs) {
            Some(0) => Some(lhs),
            _ if lhs == rhs => Some(get_or_add_const(func, Const::Int(0))),
            _ => None,
        },
        BinaryOp::Imul => match (const_int(func, lhs), const_int(func, rhs)) {
            (Some(0), _) | (_, Some(0)) => Some(get_or_add_const(func, Const::Int(0))),
            (Some(1), _) => Some(rhs),
            (_, Some(1)) => Some(lhs),
            _ => None,
        },
        BinaryOp::Idiv => match const_int(func, rhs) {
            Some(1) => Some(lhs),
            _ if lhs == rhs => Some(get_or_add_const(func, Const::Int(1))),
            _ => None,
        },
        BinaryOp::Imod => match const_int(func, rhs) {
            Some(1) => Some(get_or_add_const(func, Const::Int(0))),
            _ if lhs == rhs => Some(get_or_add_const(func, Const::Int(0))),
            _ => None,
        },
        BinaryOp::Iand => match (const_int(func, lhs), const_int(func, rhs)) {
            (Some(0), _) | (_, Some(0)) => Some(get_or_add_const(func, Const::Int(0))),
            (Some(-1), _) => Some(rhs),
            (_, Some(-1)) => Some(lhs),
            _ if lhs == rhs => Some(lhs),
            _ => None,
        },
        BinaryOp::Ior => match (const_int(func, lhs), const_int(func, rhs)) {
            (Some(0), _) => Some(rhs),
            (_, Some(0)) => Some(lhs),
            (Some(-1), _) | (_, Some(-1)) => Some(get_or_add_const(func, Const::Int(-1))),
            _ if lhs == rhs => Some(lhs),
            _ => None,
        },
        BinaryOp::Ixor => match (const_int(func, lhs), const_int(func, rhs)) {
            (Some(0), _) => Some(rhs),
            (_, Some(0)) => Some(lhs),
            _ if lhs == rhs => Some(get_or_add_const(func, Const::Int(0))),
            _ => None,
        },
        BinaryOp::Ishl | BinaryOp::Iashr => match (const_int(func, lhs), const_int(func, rhs)) {
            (Some(0), _) => Some(get_or_add_const(func, Const::Int(0))),
            (_, Some(0)) => Some(lhs),
            _ => None,
        },
        BinaryOp::Fadd => None,
        BinaryOp::Fsub => match const_float(func, rhs) {
            Some(0.0) => Some(lhs),
            _ => None,
        },
        BinaryOp::Fmul => match (const_float(func, lhs), const_float(func, rhs)) {
            (Some(1.0), _) => Some(rhs),
            (_, Some(1.0)) => Some(lhs),
            _ => None,
        },
        BinaryOp::Fdiv => match const_float(func, rhs) {
            Some(1.0) => Some(lhs),
            _ => None,
        },
        BinaryOp::And => match (const_bool(func, lhs), const_bool(func, rhs)) {
            (Some(false), _) | (_, Some(false)) => Some(get_or_add_const(func, Const::Bool(false))),
            (Some(true), _) => Some(rhs),
            (_, Some(true)) => Some(lhs),
            _ => None,
        },
        BinaryOp::Or => match (const_bool(func, lhs), const_bool(func, rhs)) {
            (Some(true), _) | (_, Some(true)) => Some(get_or_add_const(func, Const::Bool(true))),
            (Some(false), _) => Some(rhs),
            (_, Some(false)) => Some(lhs),
            _ => None,
        },
    }
}

fn fold_unary(op: UnaryOp, value: &Const) -> Option<Const> {
    match (op, value) {
        (UnaryOp::Ineg, Const::Int(value)) => Some(Const::Int(value.wrapping_neg())),
        (UnaryOp::Fneg, Const::Float(bits)) => {
            Some(Const::Float((-f32::from_bits(*bits)).to_bits()))
        }
        (UnaryOp::Not, Const::Bool(value)) => Some(Const::Bool(!value)),
        (UnaryOp::Not, Const::Int(value)) => Some(Const::Bool(*value == 0)),
        _ => None,
    }
}

fn fold_binary(op: BinaryOp, lhs: &Const, rhs: &Const) -> Option<Const> {
    match (op, lhs, rhs) {
        (BinaryOp::Iadd, Const::Int(lhs), Const::Int(rhs)) => {
            Some(Const::Int(lhs.wrapping_add(*rhs)))
        }
        (BinaryOp::Isub, Const::Int(lhs), Const::Int(rhs)) => {
            Some(Const::Int(lhs.wrapping_sub(*rhs)))
        }
        (BinaryOp::Imul, Const::Int(lhs), Const::Int(rhs)) => {
            Some(Const::Int(lhs.wrapping_mul(*rhs)))
        }
        (BinaryOp::Idiv, Const::Int(_), Const::Int(0))
        | (BinaryOp::Imod, Const::Int(_), Const::Int(0)) => None,
        (BinaryOp::Idiv, Const::Int(lhs), Const::Int(rhs)) => {
            Some(Const::Int(lhs.wrapping_div(*rhs)))
        }
        (BinaryOp::Imod, Const::Int(lhs), Const::Int(rhs)) => {
            Some(Const::Int(lhs.wrapping_rem(*rhs)))
        }
        (BinaryOp::Iand, Const::Int(lhs), Const::Int(rhs)) => Some(Const::Int(*lhs & *rhs)),
        (BinaryOp::Ior, Const::Int(lhs), Const::Int(rhs)) => Some(Const::Int(*lhs | *rhs)),
        (BinaryOp::Ixor, Const::Int(lhs), Const::Int(rhs)) => Some(Const::Int(*lhs ^ *rhs)),
        (BinaryOp::Ishl, Const::Int(lhs), Const::Int(rhs)) => {
            Some(Const::Int(lhs.wrapping_shl((*rhs as u32) & 31)))
        }
        (BinaryOp::Iashr, Const::Int(lhs), Const::Int(rhs)) => {
            Some(Const::Int(lhs.wrapping_shr((*rhs as u32) & 31)))
        }
        (BinaryOp::Fadd, Const::Float(lhs), Const::Float(rhs)) => Some(Const::Float(
            (f32::from_bits(*lhs) + f32::from_bits(*rhs)).to_bits(),
        )),
        (BinaryOp::Fsub, Const::Float(lhs), Const::Float(rhs)) => Some(Const::Float(
            (f32::from_bits(*lhs) - f32::from_bits(*rhs)).to_bits(),
        )),
        (BinaryOp::Fmul, Const::Float(lhs), Const::Float(rhs)) => Some(Const::Float(
            (f32::from_bits(*lhs) * f32::from_bits(*rhs)).to_bits(),
        )),
        (BinaryOp::Fdiv, Const::Float(lhs), Const::Float(rhs)) => Some(Const::Float(
            (f32::from_bits(*lhs) / f32::from_bits(*rhs)).to_bits(),
        )),
        (BinaryOp::And, lhs, rhs) => Some(Const::Bool(const_truthy(lhs)? && const_truthy(rhs)?)),
        (BinaryOp::Or, lhs, rhs) => Some(Const::Bool(const_truthy(lhs)? || const_truthy(rhs)?)),
        _ => None,
    }
}

fn fold_icmp(op: CmpOp, lhs: &Const, rhs: &Const) -> Option<Const> {
    let lhs = const_i32(lhs)?;
    let rhs = const_i32(rhs)?;
    Some(Const::Bool(match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
    }))
}

fn fold_fcmp(op: CmpOp, lhs: &Const, rhs: &Const) -> Option<Const> {
    let lhs = const_f32(lhs)?;
    let rhs = const_f32(rhs)?;
    Some(Const::Bool(match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
    }))
}

fn fold_cast(op: CastOp, value: &Const) -> Option<Const> {
    match op {
        CastOp::I32ToF32 => Some(Const::Float((const_i32(value)? as f32).to_bits())),
        CastOp::F32ToI32 => Some(Const::Int(const_f32(value)? as i32)),
        CastOp::BoolToI32 => Some(Const::Int(const_truthy(value)? as i32)),
        CastOp::I32ToBool | CastOp::F32ToBool => Some(Const::Bool(const_truthy(value)?)),
    }
}

fn is_noop_cast(op: CastOp, source: &Type) -> bool {
    matches!(
        (op, source),
        (CastOp::BoolToI32, Type::I32) | (CastOp::I32ToBool, Type::I1)
    )
}

fn const_value(func: &Function, value: ValueId) -> Option<&Const> {
    match &func.value(value).kind {
        ValueKind::Const(value) => Some(value),
        _ => None,
    }
}

fn get_or_add_const(func: &mut Function, value: Const) -> ValueId {
    // 复用函数里已有的同值常量，避免因为化简制造一堆重复 Const。
    if let Some(value) = func
        .values
        .iter()
        .position(
            |candidate| matches!(&candidate.kind, ValueKind::Const(existing) if *existing == value),
        )
        .map(ValueId)
    {
        value
    } else {
        func.add_const(value)
    }
}

fn const_int(func: &Function, value: ValueId) -> Option<i32> {
    const_value(func, value).and_then(const_i32)
}

fn const_float(func: &Function, value: ValueId) -> Option<f32> {
    const_value(func, value).and_then(const_f32)
}

fn const_bool(func: &Function, value: ValueId) -> Option<bool> {
    const_value(func, value).and_then(const_truthy)
}

fn const_i32(value: &Const) -> Option<i32> {
    match value {
        Const::Int(value) => Some(*value),
        Const::Bool(value) => Some(*value as i32),
        _ => None,
    }
}

fn const_f32(value: &Const) -> Option<f32> {
    match value {
        Const::Float(bits) => Some(f32::from_bits(*bits)),
        Const::Int(value) => Some(*value as f32),
        Const::Bool(value) => Some((*value as i32) as f32),
        _ => None,
    }
}

fn const_truthy(value: &Const) -> Option<bool> {
    match value {
        Const::Bool(value) => Some(*value),
        Const::Int(value) => Some(*value != 0),
        Const::Float(bits) => Some(f32::from_bits(*bits) != 0.0),
        _ => None,
    }
}
