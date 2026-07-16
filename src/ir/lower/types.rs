use super::const_eval::const_expr_usize;
use super::LowerError;
use crate::ast::{self, Expr};
use crate::ir::{Const, Type};
use std::collections::HashMap;

pub(super) fn lower_type(ty: &ast::Type) -> Type {
    match ty {
        ast::Type::Int => Type::I32,
        ast::Type::Void => Type::Void,
        ast::Type::Float => Type::F32,
    }
}

pub(super) fn lower_param_type(
    param: &ast::FuncParam,
    consts: &HashMap<String, Const>,
) -> Result<Type, LowerError> {
    let base = lower_type(&param.ty);
    if param.dims.is_empty() {
        Ok(base)
    } else {
        let mut ty = base;
        for dim in param.dims.iter().skip(1).rev() {
            let Some(dim) = dim else {
                return Err(LowerError::new(
                    "only first array parameter dimension may be empty",
                ));
            };
            ty = Type::Array {
                elem: Box::new(ty),
                len: const_expr_usize(dim, consts)?,
            };
        }
        Ok(Type::Ptr(Box::new(ty)))
    }
}

pub(super) fn array_type(
    base: Type,
    dims: &[Expr],
    consts: &HashMap<String, Const>,
) -> Result<Type, LowerError> {
    let mut ty = base;
    for dim in dims.iter().rev() {
        ty = Type::Array {
            elem: Box::new(ty),
            len: const_expr_usize(dim, consts)?,
        };
    }
    Ok(ty)
}

pub(super) fn array_elem_type(ty: &Type) -> Result<Type, LowerError> {
    match ty {
        Type::Array { elem, .. } => Ok((**elem).clone()),
        Type::Ptr(inner) => Ok((**inner).clone()),
        _ => Err(LowerError::new(format!(
            "expected array element type, got {:?}",
            ty
        ))),
    }
}

pub(super) fn lvalue_type_after_indices(
    mut ty: Type,
    indices: &[Expr],
) -> Result<Type, LowerError> {
    for _ in indices {
        ty = array_elem_type(&ty)?;
    }
    Ok(ty)
}

pub(super) fn array_scalar_type(ty: &Type) -> Result<Type, LowerError> {
    match ty {
        Type::Array { elem, .. } => array_scalar_type(elem),
        Type::Ptr(inner) => array_scalar_type(inner),
        ty => Ok(ty.clone()),
    }
}

pub(super) fn array_shape(ty: &Type) -> Result<Vec<usize>, LowerError> {
    match ty {
        Type::Array { elem, len } => {
            let mut shape = vec![*len];
            shape.extend(array_shape(elem)?);
            Ok(shape)
        }
        _ => Ok(Vec::new()),
    }
}

pub(super) fn ir_byte_size(ty: &Type) -> Result<usize, LowerError> {
    match ty {
        Type::Void => Ok(0),
        Type::I1 | Type::I32 | Type::F32 => Ok(4),
        Type::Ptr(_) => Ok(8),
        Type::Array { elem, len } => ir_byte_size(elem)?
            .checked_mul(*len)
            .ok_or_else(|| LowerError::new("array byte size overflow")),
    }
}

pub(super) fn pointee(ty: &Type) -> Option<Type> {
    match ty {
        Type::Ptr(inner) => Some((**inner).clone()),
        _ => None,
    }
}
