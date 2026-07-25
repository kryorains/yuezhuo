use super::types::{array_elem_type, lvalue_type_after_indices, pointee};
use super::{FunctionLowerer, LowerError};
use crate::ast::{BinaryOp as AstBinaryOp, Expr, LValue, UnaryOp as AstUnaryOp};
use crate::ir::{
    BinaryOp as IrBinaryOp, CastOp, CmpOp, Const, InstKind, Type, UnaryOp as IrUnaryOp, ValueId,
};

impl<'a> FunctionLowerer<'a> {
    pub(super) fn lower_expr(&mut self, expr: &Expr) -> Result<ValueId, LowerError> {
        match expr {
            Expr::Int(v) => Ok(self.const_int(i32::try_from(*v).map_err(|_| {
                LowerError::new(format!("integer literal out of i32 range: {}", v))
            })?)),
            Expr::Float(v) => Ok(self.func.add_const(Const::Float(v.to_bits()))),
            Expr::String(s) => Ok(self.func.add_const(Const::String(s.clone()))),
            Expr::LValue(lvalue) => {
                let ptr = self.lower_lvalue_addr(lvalue)?;
                if self.lvalue_yields_pointer(lvalue)? {
                    Ok(ptr)
                } else {
                    Ok(self.load(ptr, self.pointee_type(ptr)?))
                }
            }
            Expr::Call { name, args } => self.lower_call(name, args),
            Expr::Unary { op, expr } => self.lower_unary(*op, expr),
            Expr::Binary { op, lhs, rhs } => self.lower_binary(*op, lhs, rhs),
        }
    }

    fn lower_unary(&mut self, op: AstUnaryOp, expr: &Expr) -> Result<ValueId, LowerError> {
        let value = self.lower_expr(expr)?;
        match op {
            AstUnaryOp::Pos => Ok(value),
            AstUnaryOp::Neg => match self.value_type(value) {
                Type::F32 => Ok(self.unary(IrUnaryOp::Fneg, value, Type::F32)),
                _ => {
                    let value = self.cast_to(value, Type::I32)?;
                    Ok(self.unary(IrUnaryOp::Ineg, value, Type::I32))
                }
            },
            AstUnaryOp::Not => {
                let value = self.lower_to_bool(value)?;
                Ok(self.unary(IrUnaryOp::Not, value, Type::I1))
            }
        }
    }

    fn lower_binary(
        &mut self,
        op: AstBinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<ValueId, LowerError> {
        match op {
            AstBinaryOp::And => self.lower_short_circuit_and(lhs, rhs),
            AstBinaryOp::Or => self.lower_short_circuit_or(lhs, rhs),
            AstBinaryOp::Eq
            | AstBinaryOp::Ne
            | AstBinaryOp::Lt
            | AstBinaryOp::Le
            | AstBinaryOp::Gt
            | AstBinaryOp::Ge => self.lower_compare(op, lhs, rhs),
            AstBinaryOp::Add
            | AstBinaryOp::Sub
            | AstBinaryOp::Mul
            | AstBinaryOp::Div
            | AstBinaryOp::Mod => self.lower_arith(op, lhs, rhs),
        }
    }

    fn lower_arith(
        &mut self,
        op: AstBinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<ValueId, LowerError> {
        let lhs = self.lower_expr(lhs)?;
        let rhs = self.lower_expr(rhs)?;
        let use_float = self.value_type(lhs) == Type::F32 || self.value_type(rhs) == Type::F32;
        if use_float {
            if op == AstBinaryOp::Mod {
                return Err(LowerError::new("float modulo is not supported"));
            }
            let lhs = self.cast_to(lhs, Type::F32)?;
            let rhs = self.cast_to(rhs, Type::F32)?;
            let op = match op {
                AstBinaryOp::Add => IrBinaryOp::Fadd,
                AstBinaryOp::Sub => IrBinaryOp::Fsub,
                AstBinaryOp::Mul => IrBinaryOp::Fmul,
                AstBinaryOp::Div => IrBinaryOp::Fdiv,
                _ => unreachable!(),
            };
            Ok(self.binary(op, lhs, rhs, Type::F32))
        } else {
            let lhs = self.cast_to(lhs, Type::I32)?;
            let rhs = self.cast_to(rhs, Type::I32)?;
            let op = match op {
                AstBinaryOp::Add => IrBinaryOp::Iadd,
                AstBinaryOp::Sub => IrBinaryOp::Isub,
                AstBinaryOp::Mul => IrBinaryOp::Imul,
                AstBinaryOp::Div => IrBinaryOp::Idiv,
                AstBinaryOp::Mod => IrBinaryOp::Imod,
                _ => unreachable!(),
            };
            Ok(self.binary(op, lhs, rhs, Type::I32))
        }
    }

    fn lower_compare(
        &mut self,
        op: AstBinaryOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<ValueId, LowerError> {
        let lhs = self.lower_expr(lhs)?;
        let rhs = self.lower_expr(rhs)?;
        let op = match op {
            AstBinaryOp::Eq => CmpOp::Eq,
            AstBinaryOp::Ne => CmpOp::Ne,
            AstBinaryOp::Lt => CmpOp::Lt,
            AstBinaryOp::Le => CmpOp::Le,
            AstBinaryOp::Gt => CmpOp::Gt,
            AstBinaryOp::Ge => CmpOp::Ge,
            _ => unreachable!(),
        };
        if self.value_type(lhs) == Type::F32 || self.value_type(rhs) == Type::F32 {
            let lhs = self.cast_to(lhs, Type::F32)?;
            let rhs = self.cast_to(rhs, Type::F32)?;
            Ok(self.fcmp(op, lhs, rhs))
        } else {
            let lhs = self.cast_to(lhs, Type::I32)?;
            let rhs = self.cast_to(rhs, Type::I32)?;
            Ok(self.icmp(op, lhs, rhs))
        }
    }

    fn lower_short_circuit_and(&mut self, lhs: &Expr, rhs: &Expr) -> Result<ValueId, LowerError> {
        let rhs_block = self.func.add_block("land.rhs");
        let end_block = self.func.add_block("land.end");
        let lhs = self.lower_bool_expr(lhs)?;
        let lhs_block = self.current;
        self.terminate(crate::ir::Terminator::Branch {
            cond: lhs,
            then_target: rhs_block,
            else_target: end_block,
        });

        self.current = rhs_block;
        let rhs = self.lower_bool_expr(rhs)?;
        let rhs_pred = self.current;
        self.terminate(crate::ir::Terminator::Jump(end_block));

        self.current = end_block;
        let false_value = self.func.add_const(Const::Bool(false));
        Ok(self.phi(vec![(lhs_block, false_value), (rhs_pred, rhs)], Type::I1))
    }

    fn lower_short_circuit_or(&mut self, lhs: &Expr, rhs: &Expr) -> Result<ValueId, LowerError> {
        let rhs_block = self.func.add_block("lor.rhs");
        let end_block = self.func.add_block("lor.end");
        let lhs = self.lower_bool_expr(lhs)?;
        let lhs_block = self.current;
        self.terminate(crate::ir::Terminator::Branch {
            cond: lhs,
            then_target: end_block,
            else_target: rhs_block,
        });

        self.current = rhs_block;
        let rhs = self.lower_bool_expr(rhs)?;
        let rhs_pred = self.current;
        self.terminate(crate::ir::Terminator::Jump(end_block));

        self.current = end_block;
        let true_value = self.func.add_const(Const::Bool(true));
        Ok(self.phi(vec![(lhs_block, true_value), (rhs_pred, rhs)], Type::I1))
    }

    fn lower_call(&mut self, name: &str, args: &[Expr]) -> Result<ValueId, LowerError> {
        let (name, implicit_args) = match (name, args.is_empty()) {
            ("starttime", true) => ("_sysy_starttime", vec![self.const_int(0)]),
            ("stoptime", true) => ("_sysy_stoptime", vec![self.const_int(0)]),
            _ => (name, Vec::new()),
        };
        let sig = self.sigs.get(name).cloned();
        let mut lowered_args = implicit_args;
        for (idx, arg) in args.iter().enumerate() {
            let value = self.lower_expr(arg)?;
            let value = match sig.as_ref().and_then(|sig| sig.params.get(idx)) {
                Some(Type::Ptr(_)) | None => value,
                Some(target) => self.cast_to(value, target.clone())?,
            };
            lowered_args.push(value);
        }
        let ret = sig.map(|sig| sig.ret).unwrap_or(Type::I32);
        let result_ty = if ret == Type::Void {
            None
        } else {
            Some(ret.clone())
        };
        let result = self.func.append_inst(
            self.current,
            InstKind::Call {
                name: name.to_string(),
                args: lowered_args,
            },
            result_ty,
        );
        Ok(result.unwrap_or_else(|| self.const_int(0)))
    }

    pub(super) fn lower_lvalue_addr(&mut self, lvalue: &LValue) -> Result<ValueId, LowerError> {
        let (mut ptr, mut ty) = if let Some(symbol) = self.lookup(&lvalue.name).cloned() {
            if matches!(symbol.ty, Type::Ptr(_)) {
                let ptr = self.load(symbol.ptr, symbol.ty.clone());
                (ptr, symbol.ty)
            } else {
                (symbol.ptr, symbol.ty)
            }
        } else if let Some(global_ty) = self.globals.get(&lvalue.name) {
            (
                self.func.add_global_ref(&lvalue.name, global_ty.clone()),
                pointee(global_ty).unwrap_or_else(|| global_ty.clone()),
            )
        } else {
            return Err(LowerError::new(format!(
                "undefined symbol '{}'",
                lvalue.name
            )));
        };

        for index in &lvalue.indices {
            let index = self.lower_expr(index)?;
            let index = self.cast_to(index, Type::I32)?;
            ty = array_elem_type(&ty)?;
            ptr = self.gep(ptr, vec![index], Type::Ptr(Box::new(ty.clone())));
        }
        Ok(ptr)
    }

    fn lvalue_yields_pointer(&self, lvalue: &LValue) -> Result<bool, LowerError> {
        if let Some(symbol) = self.lookup(&lvalue.name) {
            if matches!(symbol.ty, Type::Ptr(_)) && lvalue.indices.is_empty() {
                return Ok(true);
            }
            let ty = lvalue_type_after_indices(symbol.ty.clone(), &lvalue.indices)?;
            Ok(matches!(ty, Type::Array { .. }))
        } else if let Some(global_ty) = self.globals.get(&lvalue.name) {
            let ty = pointee(global_ty).unwrap_or_else(|| global_ty.clone());
            let ty = lvalue_type_after_indices(ty, &lvalue.indices)?;
            Ok(matches!(ty, Type::Array { .. }))
        } else {
            Err(LowerError::new(format!(
                "undefined symbol '{}'",
                lvalue.name
            )))
        }
    }

    pub(super) fn lower_bool_expr(&mut self, expr: &Expr) -> Result<ValueId, LowerError> {
        let value = self.lower_expr(expr)?;
        self.lower_to_bool(value)
    }

    fn lower_to_bool(&mut self, value: ValueId) -> Result<ValueId, LowerError> {
        match self.value_type(value) {
            Type::I1 => Ok(value),
            Type::I32 => {
                let zero = self.const_int(0);
                Ok(self.icmp(CmpOp::Ne, value, zero))
            }
            Type::F32 => {
                let zero = self.func.add_const(Const::Float(0.0f32.to_bits()));
                Ok(self.fcmp(CmpOp::Ne, value, zero))
            }
            ty => Err(LowerError::new(format!("cannot convert {:?} to bool", ty))),
        }
    }

    pub(super) fn cast_to(&mut self, value: ValueId, target: Type) -> Result<ValueId, LowerError> {
        let source = self.value_type(value);
        if source == target {
            return Ok(value);
        }
        if source == Type::I1 && target == Type::F32 {
            let value = self.cast(CastOp::BoolToI32, value, Type::I32);
            return Ok(self.cast(CastOp::I32ToF32, value, Type::F32));
        }
        let op = match (&source, &target) {
            (Type::I32, Type::F32) => CastOp::I32ToF32,
            (Type::F32, Type::I32) => CastOp::F32ToI32,
            (Type::I1, Type::I32) => CastOp::BoolToI32,
            (Type::I32, Type::I1) => CastOp::I32ToBool,
            (Type::F32, Type::I1) => CastOp::F32ToBool,
            _ => {
                return Err(LowerError::new(format!(
                    "cannot cast {:?} to {:?}",
                    source, target
                )));
            }
        };
        Ok(self.cast(op, value, target))
    }
}
