use super::types::{array_elem_type, array_scalar_type, array_shape, ir_byte_size};
use super::{FunctionLowerer, LowerError};
use crate::ast::{Expr, Init};
use crate::ir::{Type, ValueId};

impl<'a> FunctionLowerer<'a> {
    pub(super) fn lower_init(
        &mut self,
        ptr: ValueId,
        ty: &Type,
        init: &Init,
    ) -> Result<(), LowerError> {
        match init {
            Init::Expr(expr) => {
                let value = self.lower_expr(expr)?;
                let value = self.cast_to(value, ty.clone())?;
                self.store(ptr, value);
            }
            Init::List(values) => {
                if matches!(ty, Type::Array { .. }) {
                    self.lower_array_init_flat(ptr, ty, init)?;
                } else if let Some(first) = values.first() {
                    self.lower_init(ptr, ty, first)?;
                }
            }
        }
        Ok(())
    }

    fn lower_array_init_flat(
        &mut self,
        ptr: ValueId,
        ty: &Type,
        init: &Init,
    ) -> Result<(), LowerError> {
        let shape = array_shape(ty)?;
        let scalar_ty = array_scalar_type(ty)?;
        let exprs = flatten_init_exprs(init, &shape);

        self.memzero(ptr, ir_byte_size(ty)?);
        let total_len = shape.iter().product::<usize>();
        for (flat_idx, expr) in exprs.into_iter().take(total_len).enumerate() {
            let Some(expr) = expr else {
                continue;
            };
            let elem_ptr = self.lower_array_element_ptr(ptr, ty, &shape, flat_idx)?;
            let value = self.lower_expr(expr)?;
            let value = self.cast_to(value, scalar_ty.clone())?;
            self.store(elem_ptr, value);
        }
        Ok(())
    }

    fn lower_array_element_ptr(
        &mut self,
        mut ptr: ValueId,
        ty: &Type,
        shape: &[usize],
        flat_idx: usize,
    ) -> Result<ValueId, LowerError> {
        let mut remainder = flat_idx;
        let mut cur_ty = ty.clone();
        for dim_idx in 0..shape.len() {
            let stride = shape[dim_idx + 1..].iter().product::<usize>().max(1);
            let index = remainder / stride;
            remainder %= stride;
            let elem_ty = array_elem_type(&cur_ty)?;
            let index = self.const_int(index as i32);
            ptr = self.gep(ptr, vec![index], Type::Ptr(Box::new(elem_ty.clone())));
            cur_ty = elem_ty;
        }
        Ok(ptr)
    }
}

fn flatten_init_exprs<'a>(init: &'a Init, dims: &[usize]) -> Vec<Option<&'a Expr>> {
    let mut out = vec![None; element_count(dims)];
    fill_init(init, dims, 0, &mut out);
    out
}

fn fill_init<'a>(
    init: &'a Init,
    dims: &[usize],
    start: usize,
    out: &mut [Option<&'a Expr>],
) -> usize {
    if start >= out.len() {
        return start;
    }

    match init {
        Init::Expr(expr) => {
            out[start] = Some(expr);
            start + 1
        }
        Init::List(values) => {
            if dims.is_empty() {
                if let Some(first) = values.first() {
                    return fill_init(first, dims, start, out);
                }
                return start;
            }

            let sub_size = if dims.len() > 1 {
                element_count(&dims[1..])
            } else {
                1
            };
            let mut pos = start;
            for value in values {
                match value {
                    Init::List(_) if dims.len() > 1 => {
                        fill_init(value, &dims[1..], pos, out);
                        pos += sub_size;
                    }
                    _ => {
                        pos = fill_init(value, &dims[1..], pos, out);
                    }
                }
                if pos >= out.len() {
                    break;
                }
            }
            pos
        }
    }
}

fn element_count(dims: &[usize]) -> usize {
    if dims.is_empty() {
        1
    } else {
        dims.iter().product()
    }
}
