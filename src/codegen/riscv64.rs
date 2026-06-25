use crate::ast::{
    BinaryOp, BlockItem, Decl, Def, Expr, Func, Init, Item, LValue, Program, Stmt, Type, UnaryOp,
};
use crate::codegen::common::{
    ArgLocation, ConstValue, FuncSig, FunctionSymbols, LocalInfo, ModuleCtx, ParamSig, align_to,
    assign_arg_locations, emit_data_section, flatten_init_exprs, value_size,
};
use std::collections::HashMap;

pub fn emit_asm(prog: &Program) -> String {
    Riscv64Emitter::new(prog).emit()
}

struct Riscv64Emitter<'a> {
    ctx: ModuleCtx<'a>,
    out: String,
}

impl<'a> Riscv64Emitter<'a> {
    fn new(prog: &'a Program) -> Self {
        Self {
            ctx: ModuleCtx::new(prog),
            out: String::new(),
        }
    }

    fn emit(mut self) -> String {
        self.out.push_str(&emit_data_section(&self.ctx, ".word"));
        self.out.push_str(".text\n");
        for item in &self.ctx.prog.items {
            if let Item::Func(func) = item {
                FuncEmitter::new(&mut self, func).emit();
            }
        }
        self.out
    }
}

struct FuncEmitter<'a, 'b> {
    parent: &'a mut Riscv64Emitter<'b>,
    func: &'b Func,
    body: String,
    return_label: String,
    scopes: Vec<HashMap<String, LocalInfo>>,
    const_scopes: Vec<HashMap<String, ConstValue>>,
    stack_size: i32,
    temp_stack: i32,
    break_labels: Vec<String>,
    continue_labels: Vec<String>,
}

impl FunctionSymbols for FuncEmitter<'_, '_> {
    fn module_ctx(&self) -> &ModuleCtx<'_> {
        &self.parent.ctx
    }

    fn scopes(&self) -> &[HashMap<String, LocalInfo>] {
        &self.scopes
    }

    fn const_scopes(&self) -> &[HashMap<String, ConstValue>] {
        &self.const_scopes
    }
}

impl<'a, 'b> FuncEmitter<'a, 'b> {
    fn new(parent: &'a mut Riscv64Emitter<'b>, func: &'b Func) -> Self {
        Self {
            parent,
            func,
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
            scopes: vec![HashMap::new()],
            const_scopes: vec![HashMap::new()],
            stack_size: 16,
            temp_stack: 0,
            break_labels: Vec::new(),
            continue_labels: Vec::new(),
        }
    }

    fn emit(mut self) {
        self.emit_params();
        self.emit_block_items(&self.func.body.items);
        if self.func.ret == Type::Void {
            self.body.push_str(&format!("  j {}\n", self.return_label));
        }

        let stack_size = align_to(self.stack_size, 16);
        self.parent.out.push_str(&format!(
            ".globl {0}\n.type {0}, @function\n{0}:\n  addi sp, sp, -16\n  sd ra, 8(sp)\n  sd s0, 0(sp)\n  addi s0, sp, 16\n",
            self.func.name
        ));
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&format!("  li t6, {}\n  sub sp, sp, t6\n", stack_size));
        }
        self.parent.out.push_str(&self.body);
        self.parent
            .out
            .push_str(&format!("{}:\n", self.return_label));
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&format!("  li t6, {}\n  add sp, sp, t6\n", stack_size));
        }
        self.parent
            .out
            .push_str("  ld ra, -8(s0)\n  ld t6, -16(s0)\n  mv sp, s0\n  mv s0, t6\n  ret\n\n");
    }

    fn emit_params(&mut self) {
        let mut int_idx = 0usize;
        let mut float_idx = 0usize;
        let mut stack_idx = 0usize;
        for param in &self.func.params {
            let is_array = !param.dims.is_empty();
            let dims = param
                .dims
                .iter()
                .skip(1)
                .filter_map(|dim| dim.as_ref().map(|dim| self.eval_const_int(dim)))
                .collect::<Vec<_>>();
            let size = if is_array { 8 } else { 4 };
            let offset = self.alloc(size);
            self.define(
                param.name.clone(),
                LocalInfo {
                    ty: param.ty.clone(),
                    offset,
                    dims,
                    is_array_param: is_array,
                },
            );
            if is_array {
                if int_idx < 8 {
                    self.store_frame_x(&format!("a{}", int_idx), offset);
                } else {
                    let src = (stack_idx as i32) * 8;
                    self.load_frame_x("t0", src);
                    self.store_frame_x("t0", offset);
                    stack_idx += 1;
                }
                int_idx += 1;
            } else if param.ty == Type::Float {
                if float_idx < 8 {
                    self.store_frame_s(&format!("fa{}", float_idx), offset);
                } else {
                    let src = (stack_idx as i32) * 8;
                    self.load_frame_w("t0", src);
                    self.store_frame_w("t0", offset);
                    stack_idx += 1;
                }
                float_idx += 1;
            } else {
                if int_idx < 8 {
                    self.store_frame_w(&format!("a{}", int_idx), offset);
                } else {
                    let src = (stack_idx as i32) * 8;
                    self.load_frame_w("t0", src);
                    self.store_frame_w("t0", offset);
                    stack_idx += 1;
                }
                int_idx += 1;
            }
        }
    }

    fn emit_block_items(&mut self, items: &[BlockItem]) {
        self.push_scope();
        for item in items {
            match item {
                BlockItem::Decl(decl) => self.emit_decl(decl),
                BlockItem::Stmt(stmt) => self.emit_stmt(stmt),
            }
        }
        self.pop_scope();
    }

    fn emit_decl(&mut self, decl: &Decl) {
        for def in &decl.defs {
            self.emit_def(decl, def);
        }
    }

    fn emit_def(&mut self, decl: &Decl, def: &Def) {
        let dims = self.const_dims(&def.dims);
        let size = value_size(&dims);
        let offset = self.alloc(size);
        self.define(
            def.name.clone(),
            LocalInfo {
                ty: decl.ty.clone(),
                offset,
                dims: dims.clone(),
                is_array_param: false,
            },
        );
        if let Some(init) = &def.init {
            let values = flatten_init_exprs(init, &dims);
            for (idx, expr) in values.iter().enumerate() {
                self.emit_array_element_addr(offset, &dims, idx);
                self.push_x0();
                if let Some(expr) = expr {
                    if decl.ty == Type::Float {
                        self.emit_float_expr(expr);
                    } else {
                        self.emit_int_expr(expr);
                    }
                } else if decl.ty == Type::Float {
                    self.body.push_str("  fmv.w.x fa0, zero\n");
                } else {
                    self.body.push_str("  li a0, 0\n");
                }
                self.pop_x1();
                if decl.ty == Type::Float {
                    self.body.push_str("  fsw fa0, 0(a1)\n");
                } else {
                    self.body.push_str("  sw a0, 0(a1)\n");
                }
            }
        }
        if decl.is_const && dims.is_empty() {
            if let Some(Init::Expr(expr)) = &def.init {
                let value = self.eval_const_value(expr, &decl.ty);
                self.define_const(def.name.clone(), value);
            }
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Assign { target, value } => {
                self.emit_lvalue_addr(target);
                self.push_x0();
                let target_ty = self.lvalue_base_type(target);
                if target_ty == Type::Float {
                    self.emit_float_expr(value);
                } else {
                    self.emit_int_expr(value);
                }
                self.pop_x1();
                if target_ty == Type::Float {
                    self.body.push_str("  fsw fa0, 0(a1)\n");
                } else {
                    self.body.push_str("  sw a0, 0(a1)\n");
                }
            }
            Stmt::Expr(expr) => {
                if let Some(expr) = expr {
                    if self.expr_type(expr) == Type::Float {
                        self.emit_float_expr(expr);
                    } else {
                        self.emit_int_expr(expr);
                    }
                }
            }
            Stmt::Block(block) => self.emit_block_items(&block.items),
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let else_label = self.parent.ctx.fresh_label("else");
                let end_label = self.parent.ctx.fresh_label("endif");
                self.emit_cond(cond);
                self.body.push_str(&format!("  beqz a0, {}\n", else_label));
                self.emit_stmt(then_branch);
                self.body
                    .push_str(&format!("  j {}\n{}:\n", end_label, else_label));
                if let Some(else_branch) = else_branch {
                    self.emit_stmt(else_branch);
                }
                self.body.push_str(&format!("{}:\n", end_label));
            }
            Stmt::While { cond, body } => {
                let start = self.parent.ctx.fresh_label("while");
                let end = self.parent.ctx.fresh_label("endwhile");
                self.continue_labels.push(start.clone());
                self.break_labels.push(end.clone());
                self.body.push_str(&format!("{}:\n", start));
                self.emit_cond(cond);
                self.body.push_str(&format!("  beqz a0, {}\n", end));
                self.emit_stmt(body);
                self.body.push_str(&format!("  j {}\n{}:\n", start, end));
                self.continue_labels.pop();
                self.break_labels.pop();
            }
            Stmt::Break => {
                let label = self.break_labels.last().unwrap();
                self.body.push_str(&format!("  j {}\n", label));
            }
            Stmt::Continue => {
                let label = self.continue_labels.last().unwrap();
                self.body.push_str(&format!("  j {}\n", label));
            }
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    if self.func.ret == Type::Float {
                        self.emit_float_expr(expr);
                    } else {
                        self.emit_int_expr(expr);
                    }
                }
                self.body.push_str(&format!("  j {}\n", self.return_label));
            }
        }
    }

    fn emit_int_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Int(v) => self.body.push_str(&format!("  li a0, {}\n", *v as i32)),
            Expr::Float(v) => self.body.push_str(&format!("  li a0, {}\n", *v as i32)),
            Expr::String(_) => panic!("string codegen is not implemented yet"),
            Expr::LValue(lvalue) => {
                self.emit_lvalue_addr(lvalue);
                if self.lvalue_yields_pointer(lvalue) {
                    return;
                }
                if self.lvalue_base_type(lvalue) == Type::Float {
                    self.body
                        .push_str("  flw fa0, 0(a0)\n  fcvt.w.s a0, fa0, rtz\n");
                } else {
                    self.body.push_str("  lw a0, 0(a0)\n");
                }
            }
            Expr::Call { name, args } => {
                let ret = self.emit_call(name, args);
                if ret == Type::Float {
                    self.body.push_str("  fcvt.w.s a0, fa0, rtz\n");
                }
            }
            Expr::Unary { op, expr } => match op {
                UnaryOp::Pos => self.emit_int_expr(expr),
                UnaryOp::Neg => {
                    self.emit_int_expr(expr);
                    self.body.push_str("  negw a0, a0\n");
                }
                UnaryOp::Not => {
                    self.emit_cond(expr);
                    self.body.push_str("  seqz a0, a0\n");
                }
            },
            Expr::Binary { op, lhs, rhs } => self.emit_int_binary(*op, lhs, rhs),
        }
    }

    fn emit_int_binary(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) {
        if op == BinaryOp::And {
            let false_label = self.parent.ctx.fresh_label("land_false");
            let end_label = self.parent.ctx.fresh_label("land_end");
            self.emit_cond(lhs);
            self.body.push_str(&format!("  beqz a0, {}\n", false_label));
            self.emit_cond(rhs);
            self.body.push_str(&format!("  beqz a0, {}\n", false_label));
            self.body.push_str(&format!(
                "  li a0, 1\n  j {}\n{}:\n  li a0, 0\n{}:\n",
                end_label, false_label, end_label
            ));
            return;
        }
        if op == BinaryOp::Or {
            let true_label = self.parent.ctx.fresh_label("lor_true");
            let end_label = self.parent.ctx.fresh_label("lor_end");
            self.emit_cond(lhs);
            self.body.push_str(&format!("  bnez a0, {}\n", true_label));
            self.emit_cond(rhs);
            self.body.push_str(&format!("  bnez a0, {}\n", true_label));
            self.body.push_str(&format!(
                "  li a0, 0\n  j {}\n{}:\n  li a0, 1\n{}:\n",
                end_label, true_label, end_label
            ));
            return;
        }

        if self.expr_may_be_float(lhs) || self.expr_may_be_float(rhs) {
            if matches!(
                op,
                BinaryOp::Lt
                    | BinaryOp::Gt
                    | BinaryOp::Le
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne
            ) {
                self.emit_float_compare(op, lhs, rhs);
                return;
            }
            self.emit_float_binary_value(op, lhs, rhs);
            self.body.push_str("  fcvt.w.s a0, fa0, rtz\n");
            return;
        }

        if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            self.emit_int_add_sub_chain(op, lhs, rhs);
            return;
        }

        self.emit_int_expr(lhs);
        self.push_x0();
        self.emit_int_expr(rhs);
        self.pop_x1();
        match op {
            BinaryOp::Add => self.body.push_str("  addw a0, a1, a0\n"),
            BinaryOp::Sub => self.body.push_str("  subw a0, a1, a0\n"),
            BinaryOp::Mul => self.body.push_str("  mulw a0, a1, a0\n"),
            BinaryOp::Div | BinaryOp::Mod => {
                self.body.push_str("  divw t0, a1, a0\n");
                if op == BinaryOp::Mod {
                    self.body.push_str("  remw a0, a1, a0\n");
                } else {
                    self.body.push_str("  mv a0, t0\n");
                }
            }
            BinaryOp::Lt
            | BinaryOp::Gt
            | BinaryOp::Le
            | BinaryOp::Ge
            | BinaryOp::Eq
            | BinaryOp::Ne => match op {
                BinaryOp::Lt => self.body.push_str("  slt a0, a1, a0\n"),
                BinaryOp::Gt => self.body.push_str("  slt a0, a0, a1\n"),
                BinaryOp::Le => self.body.push_str("  slt a0, a0, a1\n  xori a0, a0, 1\n"),
                BinaryOp::Ge => self.body.push_str("  slt a0, a1, a0\n  xori a0, a0, 1\n"),
                BinaryOp::Eq => self.body.push_str("  subw a0, a1, a0\n  seqz a0, a0\n"),
                BinaryOp::Ne => self.body.push_str("  subw a0, a1, a0\n  snez a0, a0\n"),
                _ => unreachable!(),
            },
            BinaryOp::And | BinaryOp::Or => unreachable!(),
        }
    }

    fn emit_int_add_sub_chain(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) {
        let mut first = lhs;
        let mut rest = vec![(op, rhs)];
        while let Expr::Binary {
            op: child_op,
            lhs: child_lhs,
            rhs: child_rhs,
        } = first
        {
            if !matches!(child_op, BinaryOp::Add | BinaryOp::Sub)
                || self.expr_may_be_float(child_lhs)
                || self.expr_may_be_float(child_rhs)
            {
                break;
            }
            rest.push((*child_op, child_rhs));
            first = child_lhs;
        }

        self.emit_int_expr(first);
        for (op, expr) in rest.into_iter().rev() {
            self.push_x0();
            self.emit_int_expr(expr);
            self.pop_x1();
            match op {
                BinaryOp::Add => self.body.push_str("  addw a0, a1, a0\n"),
                BinaryOp::Sub => self.body.push_str("  subw a0, a1, a0\n"),
                _ => unreachable!(),
            }
        }
    }

    fn emit_float_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Int(v) => self
                .body
                .push_str(&format!("  li a0, {}\n  fcvt.s.w fa0, a0\n", *v as i32)),
            Expr::Float(v) => {
                let label = self.parent.ctx.fresh_label("float");
                self.parent.out.push_str(&format!(
                    ".section .rodata\n.align 2\n{}:\n  .word {}\n.text\n",
                    label,
                    v.to_bits()
                ));
                self.body
                    .push_str(&format!("  la t0, {}\n  flw fa0, 0(t0)\n", label));
            }
            Expr::String(_) => panic!("string codegen is not implemented yet"),
            Expr::LValue(lvalue) => {
                self.emit_lvalue_addr(lvalue);
                if self.lvalue_yields_pointer(lvalue) {
                    self.body.push_str("  fcvt.s.l fa0, a0\n");
                } else if self.lvalue_base_type(lvalue) == Type::Float {
                    self.body.push_str("  flw fa0, 0(a0)\n");
                } else {
                    self.body.push_str("  lw a0, 0(a0)\n  fcvt.s.w fa0, a0\n");
                }
            }
            Expr::Call { name, args } => {
                let ret = self.emit_call(name, args);
                if ret != Type::Float {
                    self.body.push_str("  fcvt.s.w fa0, a0\n");
                }
            }
            Expr::Unary { op, expr } => match op {
                UnaryOp::Pos => self.emit_float_expr(expr),
                UnaryOp::Neg => {
                    self.emit_float_expr(expr);
                    self.body.push_str("  fneg.s fa0, fa0\n");
                }
                UnaryOp::Not => {
                    self.emit_cond(expr);
                    self.body.push_str("  seqz a0, a0\n  fcvt.s.w fa0, a0\n");
                }
            },
            Expr::Binary { op, lhs, rhs } => {
                if matches!(
                    op,
                    BinaryOp::Lt
                        | BinaryOp::Gt
                        | BinaryOp::Le
                        | BinaryOp::Ge
                        | BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::And
                        | BinaryOp::Or
                ) {
                    self.emit_int_binary(*op, lhs, rhs);
                    self.body.push_str("  fcvt.s.w fa0, a0\n");
                } else {
                    self.emit_float_binary_value(*op, lhs, rhs);
                }
            }
        }
    }

    fn emit_float_binary_value(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) {
        self.emit_float_expr(lhs);
        self.push_s0();
        self.emit_float_expr(rhs);
        self.pop_s1();
        match op {
            BinaryOp::Add => self.body.push_str("  fadd.s fa0, fa1, fa0\n"),
            BinaryOp::Sub => self.body.push_str("  fsub.s fa0, fa1, fa0\n"),
            BinaryOp::Mul => self.body.push_str("  fmul.s fa0, fa1, fa0\n"),
            BinaryOp::Div => self.body.push_str("  fdiv.s fa0, fa1, fa0\n"),
            BinaryOp::Mod => panic!("float modulo is not supported"),
            _ => unreachable!(),
        }
    }

    fn emit_float_compare(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) {
        self.emit_float_expr(lhs);
        self.push_s0();
        self.emit_float_expr(rhs);
        self.pop_s1();
        match op {
            BinaryOp::Lt => self.body.push_str("  flt.s a0, fa1, fa0\n"),
            BinaryOp::Gt => self.body.push_str("  flt.s a0, fa0, fa1\n"),
            BinaryOp::Le => self.body.push_str("  fle.s a0, fa1, fa0\n"),
            BinaryOp::Ge => self.body.push_str("  fle.s a0, fa0, fa1\n"),
            BinaryOp::Eq => self.body.push_str("  feq.s a0, fa1, fa0\n"),
            BinaryOp::Ne => self.body.push_str("  feq.s a0, fa1, fa0\n  seqz a0, a0\n"),
            _ => unreachable!(),
        }
    }

    fn emit_cond(&mut self, expr: &Expr) {
        if self.expr_may_be_float(expr) && self.expr_type(expr) == Type::Float {
            self.emit_float_expr(expr);
            self.body
                .push_str("  fmv.w.x ft0, zero\n  feq.s a0, fa0, ft0\n  seqz a0, a0\n");
        } else {
            self.emit_int_expr(expr);
        }
    }

    fn emit_call(&mut self, name: &str, args: &[Expr]) -> Type {
        let sig = self
            .parent
            .ctx
            .funcs
            .get(name)
            .cloned()
            .unwrap_or_else(|| FuncSig {
                ret: Type::Int,
                params: args
                    .iter()
                    .map(|arg| ParamSig {
                        ty: self.expr_type(arg),
                        is_array: self.expr_yields_pointer(arg),
                    })
                    .collect(),
            });
        let arg_sigs = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| {
                sig.params.get(idx).cloned().unwrap_or_else(|| ParamSig {
                    ty: self.expr_type(arg),
                    is_array: self.expr_yields_pointer(arg),
                })
            })
            .collect::<Vec<_>>();

        let locations = assign_arg_locations(&arg_sigs, 8, 8);
        let stack_count = locations
            .iter()
            .filter(|location| matches!(location, ArgLocation::Stack))
            .count();

        for (arg, arg_sig) in args.iter().zip(arg_sigs.iter()) {
            if arg_sig.ty == Type::Float && !arg_sig.is_array {
                self.emit_float_expr(arg);
                self.push_s0();
            } else {
                self.emit_int_expr(arg);
                self.push_x0();
            }
        }

        let saved_bytes = (args.len() as i32) * 16;
        if stack_count != 0 {
            self.adjust_sp(-((stack_count as i32) * 8));
            self.temp_stack += (stack_count as i32) * 8;
        }

        let mut pushed_stack = 0usize;
        for (idx, location) in locations.iter().enumerate() {
            if matches!(location, ArgLocation::Stack) {
                let saved_offset = (args.len() - 1 - idx) * 16 + stack_count * 8;
                self.load_sp_x("t0", saved_offset as i32);
                self.store_sp_x("t0", (pushed_stack * 8) as i32);
                pushed_stack += 1;
            }
        }

        for (idx, location) in locations.iter().enumerate() {
            let saved_offset = stack_count * 8 + (args.len() - 1 - idx) * 16;
            match location {
                ArgLocation::IntReg(reg_idx) => {
                    self.load_sp_x(&format!("a{}", reg_idx), saved_offset as i32);
                }
                ArgLocation::FloatReg(reg_idx) => {
                    self.load_sp_s(&format!("fa{}", reg_idx), saved_offset as i32);
                }
                ArgLocation::Stack => {}
            }
        }
        self.body.push_str(&format!("  call {}\n", name));
        let cleanup = saved_bytes + (stack_count as i32) * 8;
        if cleanup != 0 {
            self.adjust_sp(cleanup);
            self.temp_stack -= cleanup;
        }
        sig.ret
    }

    fn emit_lvalue_addr(&mut self, lvalue: &LValue) {
        if let Some(local) = self.lookup(&lvalue.name).cloned() {
            if local.is_array_param {
                self.load_frame_x("a0", local.offset);
            } else {
                self.frame_addr("a0", local.offset);
            }
            self.emit_index_offset(&local.dims, &lvalue.indices, local.is_array_param);
        } else if let Some(global) = self.parent.ctx.globals.get(&lvalue.name).cloned() {
            self.body.push_str(&format!("  la a0, {}\n", lvalue.name));
            self.emit_index_offset(&global.dims, &lvalue.indices, false);
        } else {
            panic!("undefined symbol {}", lvalue.name);
        }
    }

    fn emit_index_offset(&mut self, dims: &[i32], indices: &[Expr], is_array_param: bool) {
        for (idx, index) in indices.iter().enumerate() {
            let stride_dims = if is_array_param {
                dims.iter().skip(idx)
            } else {
                dims.iter().skip(idx + 1)
            };
            let stride = stride_dims.product::<i32>().max(1) * 4;
            self.push_x0();
            self.emit_int_expr(index);
            if stride != 1 {
                self.body
                    .push_str(&format!("  li t0, {}\n  mul a0, a0, t0\n", stride));
            }
            self.pop_x1();
            self.body.push_str("  add a0, a1, a0\n");
        }
    }

    fn push_x0(&mut self) {
        self.body.push_str("  addi sp, sp, -16\n  sd a0, 0(sp)\n");
        self.temp_stack += 16;
    }

    fn pop_x1(&mut self) {
        self.body.push_str("  ld a1, 0(sp)\n  addi sp, sp, 16\n");
        self.temp_stack -= 16;
    }

    fn push_s0(&mut self) {
        self.body.push_str("  addi sp, sp, -16\n  fsw fa0, 0(sp)\n");
        self.temp_stack += 16;
    }

    fn pop_s1(&mut self) {
        self.body.push_str("  flw fa1, 0(sp)\n  addi sp, sp, 16\n");
        self.temp_stack -= 16;
    }

    fn adjust_sp(&mut self, amount: i32) {
        if (-2048..=2047).contains(&amount) {
            self.body.push_str(&format!("  addi sp, sp, {}\n", amount));
        } else {
            self.body.push_str(&format!("  li t0, {}\n", amount));
            self.body.push_str("  add sp, sp, t0\n");
        }
    }

    fn load_sp_x(&mut self, dst: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  ld {}, {}(sp)\n", dst, offset));
        } else {
            self.body.push_str(&format!("  li t1, {}\n", offset));
            self.body.push_str("  add t1, sp, t1\n");
            self.body.push_str(&format!("  ld {}, 0(t1)\n", dst));
        }
    }

    fn load_sp_s(&mut self, dst: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  flw {}, {}(sp)\n", dst, offset));
        } else {
            self.body.push_str(&format!("  li t1, {}\n", offset));
            self.body.push_str("  add t1, sp, t1\n");
            self.body.push_str(&format!("  flw {}, 0(t1)\n", dst));
        }
    }

    fn store_sp_x(&mut self, src: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  sd {}, {}(sp)\n", src, offset));
        } else {
            self.body.push_str(&format!("  li t1, {}\n", offset));
            self.body.push_str("  add t1, sp, t1\n");
            self.body.push_str(&format!("  sd {}, 0(t1)\n", src));
        }
    }

    fn emit_array_element_addr(&mut self, offset: i32, _dims: &[i32], flat_idx: usize) {
        let byte_offset = (flat_idx as i32) * 4;
        self.frame_addr("a0", offset + byte_offset);
    }

    fn frame_addr(&mut self, dst: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  addi {}, s0, {}\n", dst, offset));
        } else {
            self.body.push_str(&format!("  li t0, {}\n", offset));
            self.body.push_str(&format!("  add {}, s0, t0\n", dst));
        }
    }

    fn load_frame_x(&mut self, dst: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  ld {}, {}(s0)\n", dst, offset));
        } else {
            self.frame_addr("t0", offset);
            self.body.push_str(&format!("  ld {}, 0(t0)\n", dst));
        }
    }

    fn load_frame_w(&mut self, dst: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  lw {}, {}(s0)\n", dst, offset));
        } else {
            self.frame_addr("t0", offset);
            self.body.push_str(&format!("  lw {}, 0(t0)\n", dst));
        }
    }

    fn store_frame_x(&mut self, src: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  sd {}, {}(s0)\n", src, offset));
        } else {
            self.body.push_str(&format!("  li t1, {}\n", offset));
            self.body.push_str("  add t1, s0, t1\n");
            self.body.push_str(&format!("  sd {}, 0(t1)\n", src));
        }
    }

    fn store_frame_w(&mut self, src: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  sw {}, {}(s0)\n", src, offset));
        } else {
            self.body.push_str(&format!("  li t1, {}\n", offset));
            self.body.push_str("  add t1, s0, t1\n");
            self.body.push_str(&format!("  sw {}, 0(t1)\n", src));
        }
    }

    fn store_frame_s(&mut self, src: &str, offset: i32) {
        if (-2048..=2047).contains(&offset) {
            self.body
                .push_str(&format!("  fsw {}, {}(s0)\n", src, offset));
        } else {
            self.body.push_str(&format!("  li t1, {}\n", offset));
            self.body.push_str("  add t1, s0, t1\n");
            self.body.push_str(&format!("  fsw {}, 0(t1)\n", src));
        }
    }

    fn alloc(&mut self, size: i32) -> i32 {
        self.stack_size += align_to(size, 4);
        -self.stack_size
    }

    fn define(&mut self, name: String, info: LocalInfo) {
        self.scopes.last_mut().unwrap().insert(name, info);
    }

    fn define_const(&mut self, name: String, value: ConstValue) {
        self.const_scopes.last_mut().unwrap().insert(name, value);
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.const_scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.const_scopes.pop();
    }
}
