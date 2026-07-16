use super::types::array_elem_type;
use super::LowerError;
use crate::ast::{BinaryOp as AstBinaryOp, Expr, Init, UnaryOp as AstUnaryOp};
use crate::ir::{Const, Type};
use std::collections::HashMap;

pub(super) fn const_expr_usize(
    expr: &Expr,
    consts: &HashMap<String, Const>,
) -> Result<usize, LowerError> {
    match eval_const_int(expr, consts)? {
        value if value >= 0 => Ok(value as usize),
        value => Err(LowerError::new(format!(
            "negative array dimension {}",
            value
        ))),
    }
}

pub(super) fn lower_const_init(
    init: &Init,
    ty: &Type,
    consts: &HashMap<String, Const>,
) -> Result<Const, LowerError> {
    match init {
        Init::Expr(expr) => lower_const_expr(expr, ty, consts),
        Init::List(values) => {
            if values.is_empty() {
                return Ok(Const::Zero(ty.clone()));
            }
            let elem_ty = array_elem_type(ty)?;
            let mut elems = Vec::new();
            for value in values {
                elems.push(lower_const_init(value, &elem_ty, consts)?);
            }
            Ok(Const::Array(elems))
        }
    }
}

fn lower_const_expr(
    expr: &Expr,
    ty: &Type,
    consts: &HashMap<String, Const>,
) -> Result<Const, LowerError> {
    match ty {
        Type::F32 => Ok(Const::Float(eval_const_float(expr, consts)?.to_bits())),
        _ => Ok(Const::Int(eval_const_int(expr, consts)?)),
    }
}

fn eval_const_int(expr: &Expr, consts: &HashMap<String, Const>) -> Result<i32, LowerError> {
    match expr {
        Expr::Int(value) => i32::try_from(*value)
            .map_err(|_| LowerError::new(format!("integer literal out of i32 range: {}", value))),
        Expr::Float(value) => Ok(*value as i32),
        Expr::LValue(lvalue) if lvalue.indices.is_empty() => match consts.get(&lvalue.name) {
            Some(Const::Int(value)) => Ok(*value),
            Some(Const::Float(bits)) => Ok(f32::from_bits(*bits) as i32),
            Some(other) => Err(LowerError::new(format!(
                "cannot use {:?} as const int",
                other
            ))),
            None => Err(LowerError::new(format!("unknown const '{}'", lvalue.name))),
        },
        Expr::Unary { op, expr } => {
            let value = eval_const_int(expr, consts)?;
            match op {
                AstUnaryOp::Pos => Ok(value),
                AstUnaryOp::Neg => Ok(-value),
                AstUnaryOp::Not => Ok((value == 0) as i32),
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let lhs = eval_const_int(lhs, consts)?;
            let rhs = eval_const_int(rhs, consts)?;
            match op {
                AstBinaryOp::Add => Ok(lhs + rhs),
                AstBinaryOp::Sub => Ok(lhs - rhs),
                AstBinaryOp::Mul => Ok(lhs * rhs),
                AstBinaryOp::Div => Ok(lhs / rhs),
                AstBinaryOp::Mod => Ok(lhs % rhs),
                AstBinaryOp::Lt => Ok((lhs < rhs) as i32),
                AstBinaryOp::Gt => Ok((lhs > rhs) as i32),
                AstBinaryOp::Le => Ok((lhs <= rhs) as i32),
                AstBinaryOp::Ge => Ok((lhs >= rhs) as i32),
                AstBinaryOp::Eq => Ok((lhs == rhs) as i32),
                AstBinaryOp::Ne => Ok((lhs != rhs) as i32),
                AstBinaryOp::And => Ok((lhs != 0 && rhs != 0) as i32),
                AstBinaryOp::Or => Ok((lhs != 0 || rhs != 0) as i32),
            }
        }
        _ => Err(LowerError::new("unsupported constant integer expression")),
    }
}

fn eval_const_float(expr: &Expr, consts: &HashMap<String, Const>) -> Result<f32, LowerError> {
    match expr {
        Expr::Float(value) => Ok(*value),
        Expr::Int(value) => Ok(*value as f32),
        Expr::LValue(lvalue) if lvalue.indices.is_empty() => match consts.get(&lvalue.name) {
            Some(Const::Float(bits)) => Ok(f32::from_bits(*bits)),
            Some(Const::Int(value)) => Ok(*value as f32),
            Some(other) => Err(LowerError::new(format!(
                "cannot use {:?} as const float",
                other
            ))),
            None => Err(LowerError::new(format!("unknown const '{}'", lvalue.name))),
        },
        Expr::Unary { op, expr } => {
            let value = eval_const_float(expr, consts)?;
            match op {
                AstUnaryOp::Pos => Ok(value),
                AstUnaryOp::Neg => Ok(-value),
                AstUnaryOp::Not => Ok((value == 0.0) as i32 as f32),
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let lhs = eval_const_float(lhs, consts)?;
            let rhs = eval_const_float(rhs, consts)?;
            match op {
                AstBinaryOp::Add => Ok(lhs + rhs),
                AstBinaryOp::Sub => Ok(lhs - rhs),
                AstBinaryOp::Mul => Ok(lhs * rhs),
                AstBinaryOp::Div => Ok(lhs / rhs),
                AstBinaryOp::Lt => Ok((lhs < rhs) as i32 as f32),
                AstBinaryOp::Gt => Ok((lhs > rhs) as i32 as f32),
                AstBinaryOp::Le => Ok((lhs <= rhs) as i32 as f32),
                AstBinaryOp::Ge => Ok((lhs >= rhs) as i32 as f32),
                AstBinaryOp::Eq => Ok((lhs == rhs) as i32 as f32),
                AstBinaryOp::Ne => Ok((lhs != rhs) as i32 as f32),
                AstBinaryOp::And => Ok((lhs != 0.0 && rhs != 0.0) as i32 as f32),
                AstBinaryOp::Or => Ok((lhs != 0.0 || rhs != 0.0) as i32 as f32),
                AstBinaryOp::Mod => Err(LowerError::new("float const modulo is not supported")),
            }
        }
        _ => Err(LowerError::new("unsupported constant float expression")),
    }
}
