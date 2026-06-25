use crate::ast::{
    BinaryOp, BlockItem, Decl, Def, Expr, Func, Init, Item, LValue, Program, Stmt, Type, UnaryOp,
};
use crate::codegen::Target;
use crate::codegen::common::{
    ArgLocation, ConstValue, FuncSig, FunctionSymbols, LocalInfo, ModuleCtx, ParamSig, align_to,
    assign_arg_locations, emit_data_section, flatten_init_exprs, value_size,
};
use std::collections::HashMap;

pub fn emit_asm(target: Target, prog: &Program, _opt_o1: bool) -> String {
    match target {
        Target::X86_64 => X86Emitter::new(prog).emit(),
        Target::AArch64 => crate::codegen::aarch64::emit_asm(prog),
        Target::Riscv64 => crate::codegen::riscv64::emit_asm(prog),
    }
}

struct X86Emitter<'a> {
    ctx: ModuleCtx<'a>,
    out: String,
}

impl<'a> X86Emitter<'a> {
    fn new(prog: &'a Program) -> Self {
        Self {
            ctx: ModuleCtx::new(prog),
            out: String::new(),
        }
    }

    fn emit(mut self) -> String {
        self.out.push_str(&emit_data_section(&self.ctx, ".long"));
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
    parent: &'a mut X86Emitter<'b>,
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
    fn new(parent: &'a mut X86Emitter<'b>, func: &'b Func) -> Self {
        Self {
            parent,
            func,
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
            scopes: vec![HashMap::new()],
            const_scopes: vec![HashMap::new()],
            stack_size: 0,
            temp_stack: 0,
            break_labels: Vec::new(),
            continue_labels: Vec::new(),
        }
    }

    fn emit(mut self) {
        self.emit_params();
        self.emit_block_items(&self.func.body.items);
        if self.func.ret == Type::Void {
            self.body
                .push_str(&format!("  jmp {}\n", self.return_label));
        }

        let stack_size = align_to(self.stack_size, 16);
        self.parent.out.push_str(&format!(
            ".globl {0}\n.type {0}, @function\n{0}:\n  pushq %rbp\n  movq %rsp, %rbp\n",
            self.func.name
        ));
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&format!("  subq ${}, %rsp\n", stack_size));
        }
        self.parent.out.push_str(&self.body);
        self.parent
            .out
            .push_str(&format!("{}:\n  leave\n  ret\n\n", self.return_label));
    }

    fn emit_params(&mut self) {
        let int_regs = ["%edi", "%esi", "%edx", "%ecx", "%r8d", "%r9d"];
        let ptr_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        let float_regs = [
            "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        ];
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
                if int_idx < 6 {
                    self.body
                        .push_str(&format!("  movq {}, {}(%rbp)\n", ptr_regs[int_idx], offset));
                } else {
                    let src = 16 + (stack_idx as i32) * 8;
                    self.body.push_str(&format!("  movq {}(%rbp), %rax\n", src));
                    self.body
                        .push_str(&format!("  movq %rax, {}(%rbp)\n", offset));
                    stack_idx += 1;
                }
                int_idx += 1;
            } else if param.ty == Type::Float {
                if float_idx < 8 {
                    self.body.push_str(&format!(
                        "  movss {}, {}(%rbp)\n",
                        float_regs[float_idx], offset
                    ));
                } else {
                    let src = 16 + (stack_idx as i32) * 8;
                    self.body.push_str(&format!("  movl {}(%rbp), %eax\n", src));
                    self.body
                        .push_str(&format!("  movl %eax, {}(%rbp)\n", offset));
                    stack_idx += 1;
                }
                float_idx += 1;
            } else {
                if int_idx < 6 {
                    self.body
                        .push_str(&format!("  movl {}, {}(%rbp)\n", int_regs[int_idx], offset));
                } else {
                    let src = 16 + (stack_idx as i32) * 8;
                    self.body.push_str(&format!("  movl {}(%rbp), %eax\n", src));
                    self.body
                        .push_str(&format!("  movl %eax, {}(%rbp)\n", offset));
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
                self.push_rax();
                if let Some(expr) = expr {
                    if decl.ty == Type::Float {
                        self.emit_float_expr(expr);
                    } else {
                        self.emit_int_expr(expr);
                    }
                } else {
                    if decl.ty == Type::Float {
                        self.body.push_str("  pxor %xmm0, %xmm0\n");
                    } else {
                        self.body.push_str("  movl $0, %eax\n");
                    }
                }
                self.pop_rcx();
                if decl.ty == Type::Float {
                    self.body.push_str("  movss %xmm0, (%rcx)\n");
                } else {
                    self.body.push_str("  movl %eax, (%rcx)\n");
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
                self.push_rax();
                let target_ty = self.lvalue_base_type(target);
                if target_ty == Type::Float {
                    self.emit_float_expr(value);
                } else {
                    self.emit_int_expr(value);
                }
                self.pop_rcx();
                if target_ty == Type::Float {
                    self.body.push_str("  movss %xmm0, (%rcx)\n");
                } else {
                    self.body.push_str("  movl %eax, (%rcx)\n");
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
                self.body
                    .push_str(&format!("  cmpl $0, %eax\n  je {}\n", else_label));
                self.emit_stmt(then_branch);
                self.body
                    .push_str(&format!("  jmp {}\n{}:\n", end_label, else_label));
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
                self.body
                    .push_str(&format!("  cmpl $0, %eax\n  je {}\n", end));
                self.emit_stmt(body);
                self.body.push_str(&format!("  jmp {}\n{}:\n", start, end));
                self.continue_labels.pop();
                self.break_labels.pop();
            }
            Stmt::Break => {
                let label = self.break_labels.last().unwrap();
                self.body.push_str(&format!("  jmp {}\n", label));
            }
            Stmt::Continue => {
                let label = self.continue_labels.last().unwrap();
                self.body.push_str(&format!("  jmp {}\n", label));
            }
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    if self.func.ret == Type::Float {
                        self.emit_float_expr(expr);
                    } else {
                        self.emit_int_expr(expr);
                    }
                }
                self.body
                    .push_str(&format!("  jmp {}\n", self.return_label));
            }
        }
    }

    fn emit_int_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Int(v) => self
                .body
                .push_str(&format!("  movl ${}, %eax\n", *v as i32)),
            Expr::Float(v) => self
                .body
                .push_str(&format!("  movl ${}, %eax\n", *v as i32)),
            Expr::String(_) => panic!("string codegen is not implemented yet"),
            Expr::LValue(lvalue) => {
                self.emit_lvalue_addr(lvalue);
                if self.lvalue_yields_pointer(lvalue) {
                    return;
                }
                if self.lvalue_base_type(lvalue) == Type::Float {
                    self.body.push_str("  cvttss2sil (%rax), %eax\n");
                } else {
                    self.body.push_str("  movl (%rax), %eax\n");
                }
            }
            Expr::Call { name, args } => {
                let ret = self.emit_call(name, args);
                if ret == Type::Float {
                    self.body.push_str("  cvttss2si %xmm0, %eax\n");
                }
            }
            Expr::Unary { op, expr } => match op {
                UnaryOp::Pos => self.emit_int_expr(expr),
                UnaryOp::Neg => {
                    self.emit_int_expr(expr);
                    self.body.push_str("  negl %eax\n");
                }
                UnaryOp::Not => {
                    self.emit_cond(expr);
                    self.body
                        .push_str("  cmpl $0, %eax\n  sete %al\n  movzbl %al, %eax\n");
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
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  je {}\n", false_label));
            self.emit_cond(rhs);
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  je {}\n", false_label));
            self.body.push_str(&format!(
                "  movl $1, %eax\n  jmp {}\n{}:\n  movl $0, %eax\n{}:\n",
                end_label, false_label, end_label
            ));
            return;
        }
        if op == BinaryOp::Or {
            let true_label = self.parent.ctx.fresh_label("lor_true");
            let end_label = self.parent.ctx.fresh_label("lor_end");
            self.emit_cond(lhs);
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  jne {}\n", true_label));
            self.emit_cond(rhs);
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  jne {}\n", true_label));
            self.body.push_str(&format!(
                "  movl $0, %eax\n  jmp {}\n{}:\n  movl $1, %eax\n{}:\n",
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
            self.body.push_str("  cvttss2si %xmm0, %eax\n");
            return;
        }

        if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            self.emit_int_add_sub_chain(op, lhs, rhs);
            return;
        }

        self.emit_int_expr(lhs);
        self.push_rax();
        self.emit_int_expr(rhs);
        self.pop_rcx();
        match op {
            BinaryOp::Add => self.body.push_str("  addl %ecx, %eax\n"),
            BinaryOp::Sub => self.body.push_str("  subl %eax, %ecx\n  movl %ecx, %eax\n"),
            BinaryOp::Mul => self.body.push_str("  imull %ecx, %eax\n"),
            BinaryOp::Div | BinaryOp::Mod => {
                self.body
                    .push_str("  movl %eax, %r8d\n  movl %ecx, %eax\n  cltd\n  idivl %r8d\n");
                if op == BinaryOp::Mod {
                    self.body.push_str("  movl %edx, %eax\n");
                }
            }
            BinaryOp::Lt
            | BinaryOp::Gt
            | BinaryOp::Le
            | BinaryOp::Ge
            | BinaryOp::Eq
            | BinaryOp::Ne => {
                self.body.push_str("  cmpl %eax, %ecx\n");
                let cc = match op {
                    BinaryOp::Lt => "setl",
                    BinaryOp::Gt => "setg",
                    BinaryOp::Le => "setle",
                    BinaryOp::Ge => "setge",
                    BinaryOp::Eq => "sete",
                    BinaryOp::Ne => "setne",
                    _ => unreachable!(),
                };
                self.body
                    .push_str(&format!("  {} %al\n  movzbl %al, %eax\n", cc));
            }
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
            self.push_rax();
            self.emit_int_expr(expr);
            self.pop_rcx();
            match op {
                BinaryOp::Add => self.body.push_str("  addl %ecx, %eax\n"),
                BinaryOp::Sub => self.body.push_str("  subl %eax, %ecx\n  movl %ecx, %eax\n"),
                _ => unreachable!(),
            }
        }
    }

    fn emit_float_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Int(v) => self.body.push_str(&format!(
                "  movl ${}, %eax\n  cvtsi2ssl %eax, %xmm0\n",
                *v as i32
            )),
            Expr::Float(v) => {
                let label = self.parent.ctx.fresh_label("float");
                self.parent.out.push_str(&format!(
                    ".section .rodata\n{}:\n  .long {}\n.text\n",
                    label,
                    v.to_bits()
                ));
                self.body
                    .push_str(&format!("  movss {}(%rip), %xmm0\n", label));
            }
            Expr::String(_) => panic!("string codegen is not implemented yet"),
            Expr::LValue(lvalue) => {
                self.emit_lvalue_addr(lvalue);
                if self.lvalue_yields_pointer(lvalue) {
                    self.body.push_str("  cvtsi2ssq %rax, %xmm0\n");
                } else if self.lvalue_base_type(lvalue) == Type::Float {
                    self.body.push_str("  movss (%rax), %xmm0\n");
                } else {
                    self.body.push_str("  cvtsi2ssl (%rax), %xmm0\n");
                }
            }
            Expr::Call { name, args } => {
                let ret = self.emit_call(name, args);
                if ret != Type::Float {
                    self.body.push_str("  cvtsi2ssl %eax, %xmm0\n");
                }
            }
            Expr::Unary { op, expr } => match op {
                UnaryOp::Pos => self.emit_float_expr(expr),
                UnaryOp::Neg => {
                    self.emit_float_expr(expr);
                    self.body.push_str(
                        "  movl $-2147483648, %eax\n  movd %eax, %xmm1\n  xorps %xmm1, %xmm0\n",
                    );
                }
                UnaryOp::Not => {
                    self.emit_cond(expr);
                    self.body.push_str("  cmpl $0, %eax\n  sete %al\n  movzbl %al, %eax\n  cvtsi2ssl %eax, %xmm0\n");
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
                    self.body.push_str("  cvtsi2ssl %eax, %xmm0\n");
                } else {
                    self.emit_float_binary_value(*op, lhs, rhs);
                }
            }
        }
    }

    fn emit_float_binary_value(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) {
        self.emit_float_expr(lhs);
        self.body
            .push_str("  subq $8, %rsp\n  movss %xmm0, (%rsp)\n");
        self.temp_stack += 8;
        self.emit_float_expr(rhs);
        self.body
            .push_str("  movss (%rsp), %xmm1\n  addq $8, %rsp\n");
        self.temp_stack -= 8;
        match op {
            BinaryOp::Add => self.body.push_str("  addss %xmm1, %xmm0\n"),
            BinaryOp::Sub => self
                .body
                .push_str("  subss %xmm0, %xmm1\n  movaps %xmm1, %xmm0\n"),
            BinaryOp::Mul => self.body.push_str("  mulss %xmm1, %xmm0\n"),
            BinaryOp::Div => self
                .body
                .push_str("  divss %xmm0, %xmm1\n  movaps %xmm1, %xmm0\n"),
            BinaryOp::Mod => panic!("float modulo is not supported"),
            _ => unreachable!(),
        }
    }

    fn emit_float_compare(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) {
        self.emit_float_expr(lhs);
        self.body
            .push_str("  subq $8, %rsp\n  movss %xmm0, (%rsp)\n");
        self.temp_stack += 8;
        self.emit_float_expr(rhs);
        self.body
            .push_str("  movss (%rsp), %xmm1\n  addq $8, %rsp\n");
        self.temp_stack -= 8;
        self.body.push_str("  ucomiss %xmm0, %xmm1\n");
        let cc = match op {
            BinaryOp::Lt => "setb",
            BinaryOp::Gt => "seta",
            BinaryOp::Le => "setbe",
            BinaryOp::Ge => "setae",
            BinaryOp::Eq => "sete",
            BinaryOp::Ne => "setne",
            _ => unreachable!(),
        };
        self.body
            .push_str(&format!("  {} %al\n  movzbl %al, %eax\n", cc));
    }

    fn emit_cond(&mut self, expr: &Expr) {
        if self.expr_may_be_float(expr) && self.expr_type(expr) == Type::Float {
            self.emit_float_expr(expr);
            self.body.push_str(
                "  pxor %xmm1, %xmm1\n  ucomiss %xmm1, %xmm0\n  setne %al\n  movzbl %al, %eax\n",
            );
        } else {
            self.emit_int_expr(expr);
        }
    }

    fn emit_call(&mut self, name: &str, args: &[Expr]) -> Type {
        let int_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        let float_regs = [
            "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        ];
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
        let locations = assign_arg_locations(&arg_sigs, int_regs.len(), float_regs.len());
        let stack_count = locations
            .iter()
            .filter(|location| matches!(location, ArgLocation::Stack))
            .count();
        let float_reg_count = locations
            .iter()
            .filter(|location| matches!(location, ArgLocation::FloatReg(_)))
            .count();

        for (arg, arg_sig) in args.iter().zip(arg_sigs.iter()) {
            if arg_sig.ty == Type::Float && !arg_sig.is_array {
                self.emit_float_expr(arg);
                self.body
                    .push_str("  subq $8, %rsp\n  movq $0, (%rsp)\n  movss %xmm0, (%rsp)\n");
            } else {
                self.emit_int_expr(arg);
                if !arg_sig.is_array {
                    self.body.push_str("  cltq\n");
                }
                self.body.push_str("  pushq %rax\n");
            }
            self.temp_stack += 8;
        }

        let saved_bytes = (args.len() as i32) * 8;
        let call_stack = self.temp_stack + (stack_count as i32) * 8;
        let pad = if call_stack % 16 == 0 { 0 } else { 8 };
        if pad != 0 {
            self.body.push_str("  subq $8, %rsp\n");
            self.temp_stack += 8;
        }

        let mut pushed_stack = 0usize;
        for (idx, location) in locations.iter().enumerate().rev() {
            if matches!(location, ArgLocation::Stack) {
                let offset = pad + (pushed_stack as i32) * 8 + ((args.len() - 1 - idx) as i32) * 8;
                self.body
                    .push_str(&format!("  movq {}(%rsp), %rax\n  pushq %rax\n", offset));
                self.temp_stack += 8;
                pushed_stack += 1;
            }
        }

        let saved_base = pad + (stack_count as i32) * 8;
        for (idx, location) in locations.iter().enumerate() {
            let offset = saved_base + ((args.len() - 1 - idx) as i32) * 8;
            match location {
                ArgLocation::IntReg(reg_idx) => {
                    self.body.push_str(&format!(
                        "  movq {}(%rsp), {}\n",
                        offset, int_regs[*reg_idx]
                    ));
                }
                ArgLocation::FloatReg(reg_idx) => {
                    self.body.push_str(&format!(
                        "  movss {}(%rsp), {}\n",
                        offset, float_regs[*reg_idx]
                    ));
                }
                ArgLocation::Stack => {}
            }
        }
        self.body
            .push_str(&format!("  movb ${}, %al\n", float_reg_count));
        self.body.push_str(&format!("  call {}\n", name));
        let cleanup = saved_bytes + (stack_count as i32) * 8 + pad;
        if cleanup != 0 {
            self.body.push_str(&format!("  addq ${}, %rsp\n", cleanup));
            self.temp_stack -= cleanup as i32;
        }
        sig.ret
    }

    fn emit_lvalue_addr(&mut self, lvalue: &LValue) {
        if let Some(local) = self.lookup(&lvalue.name).cloned() {
            if local.is_array_param {
                self.body
                    .push_str(&format!("  movq {}(%rbp), %rax\n", local.offset));
            } else {
                self.body
                    .push_str(&format!("  leaq {}(%rbp), %rax\n", local.offset));
            }
            self.emit_index_offset(&local.dims, &lvalue.indices, local.is_array_param);
        } else if let Some(global) = self.parent.ctx.globals.get(&lvalue.name).cloned() {
            self.body
                .push_str(&format!("  leaq {}(%rip), %rax\n", lvalue.name));
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
            self.push_rax();
            self.emit_int_expr(index);
            self.body.push_str("  cltq\n");
            if stride != 1 {
                self.body.push_str(&format!("  imulq ${}, %rax\n", stride));
            }
            self.pop_rcx();
            self.body.push_str("  addq %rcx, %rax\n");
        }
    }

    fn push_rax(&mut self) {
        self.body.push_str("  pushq %rax\n");
        self.temp_stack += 8;
    }

    fn pop_rcx(&mut self) {
        self.pop_reg("%rcx");
    }

    fn pop_reg(&mut self, reg: &str) {
        self.body.push_str(&format!("  popq {}\n", reg));
        self.temp_stack -= 8;
    }

    fn emit_array_element_addr(&mut self, offset: i32, dims: &[i32], flat_idx: usize) {
        let byte_offset = (flat_idx as i32) * 4;
        self.body
            .push_str(&format!("  leaq {}(%rbp), %rax\n", offset + byte_offset));
        let _ = dims;
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
