use super::{
    BinaryOp as IrBinaryOp, CastOp, CmpOp, Const, Function, Global, InstKind, Module, Terminator,
    Type, UnaryOp as IrUnaryOp, ValueId,
};
use crate::ast::{
    self, BinaryOp as AstBinaryOp, BlockItem, Expr, Init, Item, LValue, Stmt, UnaryOp as AstUnaryOp,
};
use std::collections::HashMap;

pub fn lower_program(program: &ast::Program) -> Result<Module, LowerError> {
    Lowerer::new(program).lower()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerError {
    pub message: String,
}

impl LowerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct FunctionSig {
    ret: Type,
}

struct Lowerer<'a> {
    program: &'a ast::Program,
    module: Module,
    funcs: HashMap<String, FunctionSig>,
    globals: HashMap<String, Type>,
    consts: HashMap<String, Const>,
}

impl<'a> Lowerer<'a> {
    fn new(program: &'a ast::Program) -> Self {
        let mut lowerer = Self {
            program,
            module: Module::new(),
            funcs: HashMap::new(),
            globals: HashMap::new(),
            consts: HashMap::new(),
        };
        lowerer.add_runtime_sigs();
        lowerer
    }

    fn lower(mut self) -> Result<Module, LowerError> {
        self.collect_func_sigs();

        for item in &self.program.items {
            match item {
                Item::Decl(decl) => self.lower_global_decl(decl)?,
                Item::Func(func) => {
                    let func =
                        FunctionLowerer::new(func, &self.funcs, &self.globals, &self.consts)?
                            .lower()?;
                    if let Err(errors) = func.verify() {
                        return Err(LowerError::new(format!("IR verifier failed: {:?}", errors)));
                    }
                    self.module.add_func(func);
                }
            }
        }

        Ok(self.module)
    }

    fn collect_func_sigs(&mut self) {
        for func in self.program.funcs() {
            self.funcs.insert(
                func.name.clone(),
                FunctionSig {
                    ret: lower_type(&func.ret),
                },
            );
        }
    }

    fn add_runtime_sigs(&mut self) {
        for (name, ret) in [
            ("getint", Type::I32),
            ("getch", Type::I32),
            ("getfloat", Type::F32),
            ("getarray", Type::I32),
            ("getfarray", Type::I32),
            ("putint", Type::Void),
            ("putch", Type::Void),
            ("putfloat", Type::Void),
            ("putarray", Type::Void),
            ("putfarray", Type::Void),
            ("putf", Type::Void),
            ("starttime", Type::Void),
            ("stoptime", Type::Void),
        ] {
            self.funcs.insert(name.to_string(), FunctionSig { ret });
        }
    }

    fn lower_global_decl(&mut self, decl: &ast::Decl) -> Result<(), LowerError> {
        let base = lower_type(&decl.ty);
        for def in &decl.defs {
            let ty = array_type(base.clone(), &def.dims, &self.consts)?;
            let init = match &def.init {
                Some(init) => Some(lower_const_init(init, &ty, &self.consts)?),
                None => Some(Const::Zero(ty.clone())),
            };
            if decl.is_const {
                if let Some(init) = &init {
                    self.consts.insert(def.name.clone(), init.clone());
                }
            }
            self.globals
                .insert(def.name.clone(), Type::Ptr(Box::new(ty.clone())));
            self.module.globals.push(Global {
                name: def.name.clone(),
                ty,
                is_const: decl.is_const,
                init,
            });
        }
        Ok(())
    }
}

struct FunctionLowerer<'a> {
    ast_func: &'a ast::Func,
    sigs: &'a HashMap<String, FunctionSig>,
    globals: &'a HashMap<String, Type>,
    consts: &'a HashMap<String, Const>,
    func: Function,
    scopes: Vec<HashMap<String, Symbol>>,
    current: super::BlockId,
    loop_stack: Vec<LoopTargets>,
}

impl<'a> FunctionLowerer<'a> {
    fn new(
        ast_func: &'a ast::Func,
        sigs: &'a HashMap<String, FunctionSig>,
        globals: &'a HashMap<String, Type>,
        consts: &'a HashMap<String, Const>,
    ) -> Result<Self, LowerError> {
        let func = Function::new(&ast_func.name, lower_type(&ast_func.ret));
        let current = func.entry;
        Ok(Self {
            ast_func,
            sigs,
            globals,
            consts,
            func,
            scopes: vec![HashMap::new()],
            current,
            loop_stack: Vec::new(),
        })
    }

    fn lower(mut self) -> Result<Function, LowerError> {
        for param in &self.ast_func.params {
            let ty = lower_param_type(param, self.consts)?;
            let value = self.func.add_param(&param.name, ty.clone());
            let ptr = self.alloca(param.name.clone(), ty.clone());
            self.store(ptr, value);
            self.define(param.name.clone(), Symbol { ptr, ty })?;
        }

        self.lower_block_items(&self.ast_func.body.items)?;
        if !self.is_terminated(self.current) {
            match self.func.ret {
                Type::Void => self.terminate(Terminator::Return(None)),
                _ => {
                    let zero = self.const_zero(self.func.ret.clone());
                    self.terminate(Terminator::Return(Some(zero)));
                }
            }
        }
        Ok(self.func)
    }

    fn lower_block_items(&mut self, items: &[BlockItem]) -> Result<(), LowerError> {
        self.push_scope();
        for item in items {
            if self.is_terminated(self.current) {
                break;
            }
            match item {
                BlockItem::Decl(decl) => self.lower_decl(decl)?,
                BlockItem::Stmt(stmt) => self.lower_stmt(stmt)?,
            }
        }
        self.pop_scope();
        Ok(())
    }

    fn lower_decl(&mut self, decl: &ast::Decl) -> Result<(), LowerError> {
        let base = lower_type(&decl.ty);
        for def in &decl.defs {
            let ty = array_type(base.clone(), &def.dims, self.consts)?;
            let ptr = self.alloca(def.name.clone(), ty.clone());
            self.define(
                def.name.clone(),
                Symbol {
                    ptr,
                    ty: ty.clone(),
                },
            )?;
            if let Some(init) = &def.init {
                self.lower_init(ptr, &ty, init)?;
            }
        }
        Ok(())
    }

    fn lower_init(&mut self, ptr: ValueId, ty: &Type, init: &Init) -> Result<(), LowerError> {
        match init {
            Init::Expr(expr) => {
                let value = self.lower_expr(expr)?;
                let value = self.cast_to(value, ty.clone())?;
                self.store(ptr, value);
            }
            Init::List(values) => {
                if matches!(ty, Type::Array { .. }) {
                    self.lower_array_init_flat(ptr, ty, values)?;
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
        values: &[Init],
    ) -> Result<(), LowerError> {
        let shape = array_shape(ty)?;
        let scalar_ty = array_scalar_type(ty)?;
        let mut exprs = Vec::new();
        flatten_init_exprs(values, &mut exprs);

        let total_len = shape.iter().product::<usize>();
        for (flat_idx, expr) in exprs.into_iter().take(total_len).enumerate() {
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

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), LowerError> {
        match stmt {
            Stmt::Assign { target, value } => {
                let ptr = self.lower_lvalue_addr(target)?;
                let ty = self.pointee_type(ptr)?;
                let value = self.lower_expr(value)?;
                let value = self.cast_to(value, ty)?;
                self.store(ptr, value);
            }
            Stmt::Expr(expr) => {
                if let Some(expr) = expr {
                    self.lower_expr(expr)?;
                }
            }
            Stmt::Block(block) => self.lower_block_items(&block.items)?,
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => self.lower_if(cond, then_branch, else_branch.as_deref())?,
            Stmt::While { cond, body } => self.lower_while(cond, body)?,
            Stmt::Break => {
                let targets = self
                    .loop_stack
                    .last()
                    .ok_or_else(|| LowerError::new("break outside loop"))?;
                self.terminate(Terminator::Jump(targets.break_target));
            }
            Stmt::Continue => {
                let targets = self
                    .loop_stack
                    .last()
                    .ok_or_else(|| LowerError::new("continue outside loop"))?;
                self.terminate(Terminator::Jump(targets.continue_target));
            }
            Stmt::Return(expr) => {
                let value = match expr {
                    Some(expr) => {
                        let value = self.lower_expr(expr)?;
                        Some(self.cast_to(value, self.func.ret.clone())?)
                    }
                    None => None,
                };
                self.terminate(Terminator::Return(value));
            }
        }
        Ok(())
    }

    fn lower_if(
        &mut self,
        cond: &Expr,
        then_branch: &Stmt,
        else_branch: Option<&Stmt>,
    ) -> Result<(), LowerError> {
        let then_block = self.func.add_block("if.then");
        let else_block = self.func.add_block("if.else");
        let cont_block = self.func.add_block("if.end");
        let cond = self.lower_bool_expr(cond)?;
        self.terminate(Terminator::Branch {
            cond,
            then_target: then_block,
            else_target: else_block,
        });

        self.current = then_block;
        self.lower_stmt(then_branch)?;
        if !self.is_terminated(self.current) {
            self.terminate(Terminator::Jump(cont_block));
        }

        self.current = else_block;
        if let Some(else_branch) = else_branch {
            self.lower_stmt(else_branch)?;
        }
        if !self.is_terminated(self.current) {
            self.terminate(Terminator::Jump(cont_block));
        }

        self.current = cont_block;
        Ok(())
    }

    fn lower_while(&mut self, cond: &Expr, body: &Stmt) -> Result<(), LowerError> {
        let cond_block = self.func.add_block("while.cond");
        let body_block = self.func.add_block("while.body");
        let end_block = self.func.add_block("while.end");
        self.terminate(Terminator::Jump(cond_block));

        self.current = cond_block;
        let cond_value = self.lower_bool_expr(cond)?;
        self.terminate(Terminator::Branch {
            cond: cond_value,
            then_target: body_block,
            else_target: end_block,
        });

        self.current = body_block;
        self.loop_stack.push(LoopTargets {
            break_target: end_block,
            continue_target: cond_block,
        });
        self.lower_stmt(body)?;
        self.loop_stack.pop();
        if !self.is_terminated(self.current) {
            self.terminate(Terminator::Jump(cond_block));
        }

        self.current = end_block;
        Ok(())
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<ValueId, LowerError> {
        match expr {
            Expr::Int(v) => Ok(self.const_int(i32::try_from(*v).map_err(|_| {
                LowerError::new(format!("integer literal out of i32 range: {}", v))
            })?)),
            Expr::Float(v) => Ok(self.func.add_const(Const::Float(v.to_bits()))),
            Expr::String(s) => Ok(self.func.add_const(Const::String(s.clone()))),
            Expr::LValue(lvalue) => {
                let ptr = self.lower_lvalue_addr(lvalue)?;
                Ok(self.load(ptr, self.pointee_type(ptr)?))
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
                let value = self.to_bool(value)?;
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
        let lhs_block = self.current;
        let rhs_block = self.func.add_block("land.rhs");
        let end_block = self.func.add_block("land.end");
        let lhs = self.lower_bool_expr(lhs)?;
        self.terminate(Terminator::Branch {
            cond: lhs,
            then_target: rhs_block,
            else_target: end_block,
        });

        self.current = rhs_block;
        let rhs = self.lower_bool_expr(rhs)?;
        let rhs_pred = self.current;
        self.terminate(Terminator::Jump(end_block));

        self.current = end_block;
        let false_value = self.func.add_const(Const::Bool(false));
        Ok(self.phi(vec![(lhs_block, false_value), (rhs_pred, rhs)], Type::I1))
    }

    fn lower_short_circuit_or(&mut self, lhs: &Expr, rhs: &Expr) -> Result<ValueId, LowerError> {
        let lhs_block = self.current;
        let rhs_block = self.func.add_block("lor.rhs");
        let end_block = self.func.add_block("lor.end");
        let lhs = self.lower_bool_expr(lhs)?;
        self.terminate(Terminator::Branch {
            cond: lhs,
            then_target: end_block,
            else_target: rhs_block,
        });

        self.current = rhs_block;
        let rhs = self.lower_bool_expr(rhs)?;
        let rhs_pred = self.current;
        self.terminate(Terminator::Jump(end_block));

        self.current = end_block;
        let true_value = self.func.add_const(Const::Bool(true));
        Ok(self.phi(vec![(lhs_block, true_value), (rhs_pred, rhs)], Type::I1))
    }

    fn lower_call(&mut self, name: &str, args: &[Expr]) -> Result<ValueId, LowerError> {
        let mut lowered_args = Vec::new();
        for arg in args {
            lowered_args.push(self.lower_expr(arg)?);
        }
        let ret = self
            .sigs
            .get(name)
            .map(|sig| sig.ret.clone())
            .unwrap_or(Type::I32);
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

    fn lower_lvalue_addr(&mut self, lvalue: &LValue) -> Result<ValueId, LowerError> {
        let (mut ptr, mut ty) = if let Some(symbol) = self.lookup(&lvalue.name).cloned() {
            (symbol.ptr, symbol.ty)
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

    fn lower_bool_expr(&mut self, expr: &Expr) -> Result<ValueId, LowerError> {
        let value = self.lower_expr(expr)?;
        self.to_bool(value)
    }

    fn to_bool(&mut self, value: ValueId) -> Result<ValueId, LowerError> {
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

    fn cast_to(&mut self, value: ValueId, target: Type) -> Result<ValueId, LowerError> {
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

    fn alloca(&mut self, name: String, ty: Type) -> ValueId {
        self.func
            .append_inst(
                self.func.entry,
                InstKind::Alloca { ty: ty.clone() },
                Some(Type::Ptr(Box::new(ty))),
            )
            .unwrap()
            .tap_name(&mut self.func, name)
    }

    fn load(&mut self, ptr: ValueId, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Load { ptr }, Some(ty))
            .unwrap()
    }

    fn store(&mut self, ptr: ValueId, value: ValueId) {
        self.func
            .append_inst(self.current, InstKind::Store { ptr, value }, None);
    }

    fn unary(&mut self, op: IrUnaryOp, value: ValueId, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Unary { op, value }, Some(ty))
            .unwrap()
    }

    fn binary(&mut self, op: IrBinaryOp, lhs: ValueId, rhs: ValueId, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Binary { op, lhs, rhs }, Some(ty))
            .unwrap()
    }

    fn icmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.func
            .append_inst(
                self.current,
                InstKind::Icmp { op, lhs, rhs },
                Some(Type::I1),
            )
            .unwrap()
    }

    fn fcmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.func
            .append_inst(
                self.current,
                InstKind::Fcmp { op, lhs, rhs },
                Some(Type::I1),
            )
            .unwrap()
    }

    fn cast(&mut self, op: CastOp, value: ValueId, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Cast { op, value }, Some(ty))
            .unwrap()
    }

    fn gep(&mut self, base: ValueId, indices: Vec<ValueId>, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Gep { base, indices }, Some(ty))
            .unwrap()
    }

    fn phi(&mut self, incomings: Vec<(super::BlockId, ValueId)>, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Phi { incomings }, Some(ty))
            .unwrap()
    }

    fn const_int(&mut self, value: i32) -> ValueId {
        self.func.add_const(Const::Int(value))
    }

    fn const_zero(&mut self, ty: Type) -> ValueId {
        self.func.add_const(Const::Zero(ty))
    }

    fn terminate(&mut self, terminator: Terminator) {
        if !self.is_terminated(self.current) {
            self.func.set_terminator(self.current, terminator);
        }
    }

    fn is_terminated(&self, block: super::BlockId) -> bool {
        self.func.block(block).terminator.is_some()
    }

    fn value_type(&self, value: ValueId) -> Type {
        self.func.value(value).ty.clone()
    }

    fn pointee_type(&self, ptr: ValueId) -> Result<Type, LowerError> {
        pointee(&self.value_type(ptr)).ok_or_else(|| {
            LowerError::new(format!("expected pointer, got {:?}", self.value_type(ptr)))
        })
    }

    fn define(&mut self, name: String, symbol: Symbol) -> Result<(), LowerError> {
        let scope = self.scopes.last_mut().unwrap();
        if scope.contains_key(&name) {
            return Err(LowerError::new(format!("redefined symbol '{}'", name)));
        }
        scope.insert(name, symbol);
        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

#[derive(Debug, Clone)]
struct Symbol {
    ptr: ValueId,
    ty: Type,
}

#[derive(Debug, Clone, Copy)]
struct LoopTargets {
    break_target: super::BlockId,
    continue_target: super::BlockId,
}

trait NameValue {
    fn tap_name(self, func: &mut Function, name: String) -> Self;
}

impl NameValue for ValueId {
    fn tap_name(self, func: &mut Function, name: String) -> Self {
        func.values[self.0].name = Some(name);
        self
    }
}

fn lower_type(ty: &ast::Type) -> Type {
    match ty {
        ast::Type::Int => Type::I32,
        ast::Type::Void => Type::Void,
        ast::Type::Float => Type::F32,
    }
}

fn lower_param_type(
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

fn array_type(
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

fn array_elem_type(ty: &Type) -> Result<Type, LowerError> {
    match ty {
        Type::Array { elem, .. } => Ok((**elem).clone()),
        Type::Ptr(inner) => Ok((**inner).clone()),
        _ => Err(LowerError::new(format!(
            "expected array element type, got {:?}",
            ty
        ))),
    }
}

fn array_scalar_type(ty: &Type) -> Result<Type, LowerError> {
    match ty {
        Type::Array { elem, .. } => array_scalar_type(elem),
        Type::Ptr(inner) => array_scalar_type(inner),
        ty => Ok(ty.clone()),
    }
}

fn array_shape(ty: &Type) -> Result<Vec<usize>, LowerError> {
    match ty {
        Type::Array { elem, len } => {
            let mut shape = vec![*len];
            shape.extend(array_shape(elem)?);
            Ok(shape)
        }
        _ => Ok(Vec::new()),
    }
}

fn flatten_init_exprs<'a>(values: &'a [Init], out: &mut Vec<&'a Expr>) {
    for value in values {
        match value {
            Init::Expr(expr) => out.push(expr),
            Init::List(values) => flatten_init_exprs(values, out),
        }
    }
}

fn pointee(ty: &Type) -> Option<Type> {
    match ty {
        Type::Ptr(inner) => Some((**inner).clone()),
        _ => None,
    }
}

fn const_expr_usize(expr: &Expr, consts: &HashMap<String, Const>) -> Result<usize, LowerError> {
    match eval_const_int(expr, consts)? {
        value if value >= 0 => Ok(value as usize),
        value => Err(LowerError::new(format!(
            "negative array dimension {}",
            value
        ))),
    }
}

fn lower_const_init(
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
