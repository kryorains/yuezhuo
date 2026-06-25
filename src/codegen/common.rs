use crate::ast::{BinaryOp, Expr, Init, Item, LValue, Program, Type, UnaryOp};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct LocalInfo {
    pub(crate) ty: Type,
    pub(crate) offset: i32,
    pub(crate) dims: Vec<i32>,
    pub(crate) is_array_param: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct GlobalInfo {
    pub(crate) ty: Type,
    pub(crate) dims: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParamSig {
    pub(crate) ty: Type,
    pub(crate) is_array: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct FuncSig {
    pub(crate) ret: Type,
    pub(crate) params: Vec<ParamSig>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ConstValue {
    Int(i32),
    Float(f32),
}

impl ConstValue {
    pub(crate) fn to_bits(self) -> u32 {
        match self {
            ConstValue::Int(value) => value as u32,
            ConstValue::Float(value) => value.to_bits(),
        }
    }

    pub(crate) fn as_i32(self) -> i32 {
        match self {
            ConstValue::Int(value) => value,
            ConstValue::Float(value) => value as i32,
        }
    }

    pub(crate) fn as_f32(self) -> f32 {
        match self {
            ConstValue::Int(value) => value as f32,
            ConstValue::Float(value) => value,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ArgLocation {
    IntReg(usize),
    FloatReg(usize),
    Stack,
}

pub(crate) fn assign_arg_locations(
    arg_sigs: &[ParamSig],
    int_reg_count: usize,
    float_reg_count: usize,
) -> Vec<ArgLocation> {
    let mut int_idx = 0usize;
    let mut float_idx = 0usize;
    let mut locations = Vec::with_capacity(arg_sigs.len());
    for arg_sig in arg_sigs {
        if arg_sig.is_array || arg_sig.ty != Type::Float {
            if int_idx < int_reg_count {
                locations.push(ArgLocation::IntReg(int_idx));
                int_idx += 1;
            } else {
                locations.push(ArgLocation::Stack);
            }
        } else if float_idx < float_reg_count {
            locations.push(ArgLocation::FloatReg(float_idx));
            float_idx += 1;
        } else {
            locations.push(ArgLocation::Stack);
        }
    }
    locations
}

pub(crate) struct ModuleCtx<'a> {
    pub(crate) prog: &'a Program,
    pub(crate) globals: HashMap<String, GlobalInfo>,
    pub(crate) consts: HashMap<String, ConstValue>,
    pub(crate) funcs: HashMap<String, FuncSig>,
    label_id: usize,
}

impl<'a> ModuleCtx<'a> {
    pub(crate) fn new(prog: &'a Program) -> Self {
        let mut ctx = Self {
            prog,
            globals: HashMap::new(),
            consts: HashMap::new(),
            funcs: HashMap::new(),
            label_id: 0,
        };
        ctx.collect_func_sigs();
        ctx.collect_globals();
        ctx
    }

    pub(crate) fn fresh_label(&mut self, prefix: &str) -> String {
        let label = format!(".L_{}_{}", prefix, self.label_id);
        self.label_id += 1;
        label
    }

    fn collect_func_sigs(&mut self) {
        self.add_runtime_sigs();
        for func in self.prog.funcs() {
            let params = func
                .params
                .iter()
                .map(|param| ParamSig {
                    ty: param.ty.clone(),
                    is_array: !param.dims.is_empty(),
                })
                .collect();
            self.funcs.insert(
                func.name.clone(),
                FuncSig {
                    ret: func.ret.clone(),
                    params,
                },
            );
        }
    }

    fn add_runtime_sigs(&mut self) {
        for (name, ret, params) in [
            ("getint", Type::Int, vec![]),
            ("getch", Type::Int, vec![]),
            ("getfloat", Type::Float, vec![]),
            (
                "getarray",
                Type::Int,
                vec![ParamSig {
                    ty: Type::Int,
                    is_array: true,
                }],
            ),
            (
                "getfarray",
                Type::Int,
                vec![ParamSig {
                    ty: Type::Float,
                    is_array: true,
                }],
            ),
            (
                "putint",
                Type::Void,
                vec![ParamSig {
                    ty: Type::Int,
                    is_array: false,
                }],
            ),
            (
                "putch",
                Type::Void,
                vec![ParamSig {
                    ty: Type::Int,
                    is_array: false,
                }],
            ),
            (
                "putfloat",
                Type::Void,
                vec![ParamSig {
                    ty: Type::Float,
                    is_array: false,
                }],
            ),
            (
                "putarray",
                Type::Void,
                vec![
                    ParamSig {
                        ty: Type::Int,
                        is_array: false,
                    },
                    ParamSig {
                        ty: Type::Int,
                        is_array: true,
                    },
                ],
            ),
            (
                "putfarray",
                Type::Void,
                vec![
                    ParamSig {
                        ty: Type::Int,
                        is_array: false,
                    },
                    ParamSig {
                        ty: Type::Float,
                        is_array: true,
                    },
                ],
            ),
            ("starttime", Type::Void, vec![]),
            ("stoptime", Type::Void, vec![]),
        ] {
            self.funcs.insert(name.to_string(), FuncSig { ret, params });
        }
    }

    fn collect_globals(&mut self) {
        for item in &self.prog.items {
            let Item::Decl(decl) = item else { continue };
            for def in &decl.defs {
                let dims = const_dims(&def.dims, &self.consts);
                self.globals.insert(
                    def.name.clone(),
                    GlobalInfo {
                        ty: decl.ty.clone(),
                        dims,
                    },
                );
                if decl.is_const && def.dims.is_empty() {
                    if let Some(Init::Expr(expr)) = &def.init {
                        let value = eval_const_value(expr, &decl.ty, &self.consts);
                        self.consts.insert(def.name.clone(), value);
                    }
                }
            }
        }
    }
}

pub(crate) trait FunctionSymbols {
    fn module_ctx(&self) -> &ModuleCtx<'_>;
    fn scopes(&self) -> &[HashMap<String, LocalInfo>];
    fn const_scopes(&self) -> &[HashMap<String, ConstValue>];

    fn lookup(&self, name: &str) -> Option<&LocalInfo> {
        self.scopes().iter().rev().find_map(|scope| scope.get(name))
    }

    fn expr_yields_pointer(&self, expr: &Expr) -> bool {
        match expr {
            Expr::LValue(lvalue) => self.lvalue_yields_pointer(lvalue),
            _ => false,
        }
    }

    fn lvalue_yields_pointer(&self, lvalue: &LValue) -> bool {
        if let Some(local) = self.lookup(&lvalue.name) {
            if local.is_array_param {
                lvalue.indices.len() <= local.dims.len()
            } else {
                lvalue.indices.len() < local.dims.len()
            }
        } else if let Some(global) = self.module_ctx().globals.get(&lvalue.name) {
            lvalue.indices.len() < global.dims.len()
        } else {
            false
        }
    }

    fn visible_consts(&self) -> HashMap<String, ConstValue> {
        let mut out = self.module_ctx().consts.clone();
        for scope in self.const_scopes() {
            out.extend(scope.iter().map(|(name, value)| (name.clone(), *value)));
        }
        out
    }

    fn const_dims(&self, dims: &[Expr]) -> Vec<i32> {
        dims.iter().map(|dim| self.eval_const_int(dim)).collect()
    }

    fn eval_const_int(&self, expr: &Expr) -> i32 {
        self.eval_const_value(expr, &Type::Int).as_i32()
    }

    fn eval_const_value(&self, expr: &Expr, target: &Type) -> ConstValue {
        match target {
            Type::Float => ConstValue::Float(eval_const_float(expr, &self.visible_consts())),
            _ => ConstValue::Int(eval_const_int(expr, &self.visible_consts())),
        }
    }

    fn lvalue_base_type(&self, lvalue: &LValue) -> Type {
        self.lookup(&lvalue.name)
            .map(|local| local.ty.clone())
            .or_else(|| {
                self.module_ctx()
                    .globals
                    .get(&lvalue.name)
                    .map(|global| global.ty.clone())
            })
            .unwrap_or(Type::Int)
    }

    fn expr_type(&self, expr: &Expr) -> Type {
        match expr {
            Expr::Float(_) => Type::Float,
            Expr::Int(_) | Expr::String(_) => Type::Int,
            Expr::LValue(lvalue) => {
                if self.lvalue_yields_pointer(lvalue) {
                    Type::Int
                } else {
                    self.lvalue_base_type(lvalue)
                }
            }
            Expr::Call { name, .. } => self
                .module_ctx()
                .funcs
                .get(name)
                .map(|sig| sig.ret.clone())
                .unwrap_or(Type::Int),
            Expr::Unary { op, expr } => match op {
                UnaryOp::Not => Type::Int,
                UnaryOp::Pos | UnaryOp::Neg => self.expr_type(expr),
            },
            Expr::Binary { op, lhs, rhs } => match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                    if self.expr_type(lhs) == Type::Float || self.expr_type(rhs) == Type::Float {
                        Type::Float
                    } else {
                        Type::Int
                    }
                }
                BinaryOp::Mod
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::Le
                | BinaryOp::Ge
                | BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::And
                | BinaryOp::Or => Type::Int,
            },
        }
    }

    fn expr_may_be_float(&self, expr: &Expr) -> bool {
        let mut stack = vec![expr];
        while let Some(expr) = stack.pop() {
            match expr {
                Expr::Float(_) => return true,
                Expr::LValue(lvalue) => {
                    if !self.lvalue_yields_pointer(lvalue)
                        && self.lvalue_base_type(lvalue) == Type::Float
                    {
                        return true;
                    }
                }
                Expr::Call { name, .. } => {
                    if self
                        .module_ctx()
                        .funcs
                        .get(name)
                        .is_some_and(|sig| sig.ret == Type::Float)
                    {
                        return true;
                    }
                }
                Expr::Unary { expr, .. } => stack.push(expr),
                Expr::Binary { op, lhs, rhs } => {
                    if matches!(
                        op,
                        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                    ) {
                        stack.push(rhs);
                        stack.push(lhs);
                    }
                }
                Expr::Int(_) | Expr::String(_) => {}
            }
        }
        false
    }
}

pub(crate) fn emit_data_section(ctx: &ModuleCtx<'_>, word_directive: &str) -> String {
    let mut out = String::new();
    let mut any = false;
    for item in &ctx.prog.items {
        let Item::Decl(decl) = item else { continue };
        if !any {
            out.push_str(".data\n");
            any = true;
        }
        for def in &decl.defs {
            out.push_str(&format!(".globl {}\n{}:\n", def.name, def.name));
            let dims = const_dims(&def.dims, &ctx.consts);
            let values = flatten_const_init(def.init.as_ref(), &decl.ty, &dims, &ctx.consts);
            for value in values {
                out.push_str(&format!("  {} {}\n", word_directive, value.to_bits()));
            }
        }
    }
    out
}

pub(crate) fn const_dims(dims: &[Expr], consts: &HashMap<String, ConstValue>) -> Vec<i32> {
    dims.iter().map(|dim| eval_const_int(dim, consts)).collect()
}

pub(crate) fn value_size(dims: &[i32]) -> i32 {
    element_count(dims) as i32 * 4
}

pub(crate) fn element_count(dims: &[i32]) -> usize {
    if dims.is_empty() {
        1
    } else {
        dims.iter().product::<i32>() as usize
    }
}

pub(crate) fn flatten_const_init(
    init: Option<&Init>,
    ty: &Type,
    dims: &[i32],
    consts: &HashMap<String, ConstValue>,
) -> Vec<ConstValue> {
    let count = element_count(dims);
    let mut values = init
        .map(|init| {
            flatten_init_exprs(init, dims)
                .into_iter()
                .map(|expr| {
                    expr.map(|expr| eval_const_value(expr, ty, consts))
                        .unwrap_or_else(|| zero_const_value(ty))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    values.resize(count, zero_const_value(ty));
    values
}

pub(crate) fn flatten_init_exprs<'a>(init: &'a Init, dims: &[i32]) -> Vec<Option<&'a Expr>> {
    let mut out = vec![None; element_count(dims)];
    fill_init(init, dims, 0, &mut out);
    out
}

fn fill_init<'a>(
    init: &'a Init,
    dims: &[i32],
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

pub(crate) fn eval_const_value(
    expr: &Expr,
    target: &Type,
    consts: &HashMap<String, ConstValue>,
) -> ConstValue {
    if *target == Type::Float {
        ConstValue::Float(eval_const_float(expr, consts))
    } else {
        ConstValue::Int(eval_const_int(expr, consts))
    }
}

pub(crate) fn zero_const_value(target: &Type) -> ConstValue {
    if *target == Type::Float {
        ConstValue::Float(0.0)
    } else {
        ConstValue::Int(0)
    }
}

pub(crate) fn eval_const_int(expr: &Expr, consts: &HashMap<String, ConstValue>) -> i32 {
    match expr {
        Expr::Int(v) => *v as i32,
        Expr::Float(v) => *v as i32,
        Expr::LValue(lvalue) if lvalue.indices.is_empty() => consts
            .get(&lvalue.name)
            .map(|value| value.as_i32())
            .unwrap_or(0),
        Expr::Unary { op, expr } => {
            let v = eval_const_int(expr, consts);
            match op {
                UnaryOp::Pos => v,
                UnaryOp::Neg => -v,
                UnaryOp::Not => (v == 0) as i32,
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let l = eval_const_int(lhs, consts);
            let r = eval_const_int(rhs, consts);
            eval_const_binary(*op, l, r)
        }
        _ => 0,
    }
}

pub(crate) fn eval_const_float(expr: &Expr, consts: &HashMap<String, ConstValue>) -> f32 {
    match expr {
        Expr::Float(v) => *v,
        Expr::Int(v) => *v as f32,
        Expr::LValue(lvalue) if lvalue.indices.is_empty() => consts
            .get(&lvalue.name)
            .map(|value| value.as_f32())
            .unwrap_or(0.0),
        Expr::Unary { op, expr } => {
            let value = eval_const_float(expr, consts);
            match op {
                UnaryOp::Pos => value,
                UnaryOp::Neg => -value,
                UnaryOp::Not => (value == 0.0) as i32 as f32,
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let lhs_is_float = const_expr_is_float(lhs, consts);
            let rhs_is_float = const_expr_is_float(rhs, consts);
            if !lhs_is_float && !rhs_is_float {
                return eval_const_int(expr, consts) as f32;
            }
            let l = eval_const_float(lhs, consts);
            let r = eval_const_float(rhs, consts);
            match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => l / r,
                BinaryOp::Mod => (l as i32 % r as i32) as f32,
                BinaryOp::Lt => (l < r) as i32 as f32,
                BinaryOp::Gt => (l > r) as i32 as f32,
                BinaryOp::Le => (l <= r) as i32 as f32,
                BinaryOp::Ge => (l >= r) as i32 as f32,
                BinaryOp::Eq => (l == r) as i32 as f32,
                BinaryOp::Ne => (l != r) as i32 as f32,
                BinaryOp::And => (l != 0.0 && r != 0.0) as i32 as f32,
                BinaryOp::Or => (l != 0.0 || r != 0.0) as i32 as f32,
            }
        }
        _ => 0.0,
    }
}

fn const_expr_is_float(expr: &Expr, consts: &HashMap<String, ConstValue>) -> bool {
    match expr {
        Expr::Float(_) => true,
        Expr::LValue(lvalue) if lvalue.indices.is_empty() => {
            matches!(consts.get(&lvalue.name), Some(ConstValue::Float(_)))
        }
        Expr::Unary { expr, .. } => const_expr_is_float(expr, consts),
        Expr::Binary { lhs, rhs, .. } => {
            const_expr_is_float(lhs, consts) || const_expr_is_float(rhs, consts)
        }
        _ => false,
    }
}

fn eval_const_binary(op: BinaryOp, l: i32, r: i32) -> i32 {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l / r,
        BinaryOp::Mod => l % r,
        BinaryOp::Lt => (l < r) as i32,
        BinaryOp::Gt => (l > r) as i32,
        BinaryOp::Le => (l <= r) as i32,
        BinaryOp::Ge => (l >= r) as i32,
        BinaryOp::Eq => (l == r) as i32,
        BinaryOp::Ne => (l != r) as i32,
        BinaryOp::And => (l != 0 && r != 0) as i32,
        BinaryOp::Or => (l != 0 || r != 0) as i32,
    }
}

pub(crate) fn align_to(value: i32, align: i32) -> i32 {
    (value + align - 1) / align * align
}
