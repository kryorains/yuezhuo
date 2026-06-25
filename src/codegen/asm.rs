use crate::ast::{
    BinaryOp, BlockItem, Decl, Def, Expr, Func, Init, Item, LValue, Program, Stmt, Type, UnaryOp,
};
use crate::codegen::Target;
use std::collections::HashMap;

pub fn emit_asm(target: Target, prog: &Program, _opt_o1: bool) -> String {
    match target {
        Target::X86_64 => X86Emitter::new(prog).emit(),
        Target::AArch64 | Target::Riscv64 => {
            panic!("Only x86_64 assembly generation is implemented for now")
        }
    }
}

struct X86Emitter<'a> {
    prog: &'a Program,
    out: String,
    globals: HashMap<String, GlobalInfo>,
    label_id: usize,
}

impl<'a> X86Emitter<'a> {
    fn new(prog: &'a Program) -> Self {
        Self {
            prog,
            out: String::new(),
            globals: HashMap::new(),
            label_id: 0,
        }
    }

    fn emit(mut self) -> String {
        self.collect_globals();
        self.emit_data();
        self.out.push_str(".text\n");
        for item in &self.prog.items {
            if let Item::Func(func) = item {
                FuncEmitter::new(&mut self, func).emit();
            }
        }
        self.out
    }

    fn collect_globals(&mut self) {
        for item in &self.prog.items {
            let Item::Decl(decl) = item else { continue };
            for def in &decl.defs {
                let dims = const_dims(&def.dims);
                self.globals.insert(def.name.clone(), GlobalInfo { dims });
            }
        }
    }

    fn emit_data(&mut self) {
        let mut any = false;
        for item in &self.prog.items {
            let Item::Decl(decl) = item else { continue };
            if !any {
                self.out.push_str(".data\n");
                any = true;
            }
            for def in &decl.defs {
                self.out
                    .push_str(&format!(".globl {}\n{}:\n", def.name, def.name));
                let dims = const_dims(&def.dims);
                let count = element_count(&dims);
                let values = flatten_const_init(def.init.as_ref(), count);
                for value in values {
                    self.out.push_str(&format!("  .long {}\n", value));
                }
            }
        }
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        let label = format!(".L_{}_{}", prefix, self.label_id);
        self.label_id += 1;
        label
    }
}

struct FuncEmitter<'a, 'b> {
    parent: &'a mut X86Emitter<'b>,
    func: &'b Func,
    body: String,
    return_label: String,
    scopes: Vec<HashMap<String, LocalInfo>>,
    stack_size: i32,
    temp_stack: i32,
    break_labels: Vec<String>,
    continue_labels: Vec<String>,
}

impl<'a, 'b> FuncEmitter<'a, 'b> {
    fn new(parent: &'a mut X86Emitter<'b>, func: &'b Func) -> Self {
        Self {
            parent,
            func,
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
            scopes: vec![HashMap::new()],
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
        let regs = ["%edi", "%esi", "%edx", "%ecx", "%r8d", "%r9d"];
        let ptr_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        for (idx, param) in self.func.params.iter().enumerate() {
            let is_array = !param.dims.is_empty();
            let dims = param
                .dims
                .iter()
                .skip(1)
                .filter_map(|dim| dim.as_ref().map(eval_const_int))
                .collect::<Vec<_>>();
            let size = if is_array { 8 } else { 4 };
            let offset = self.alloc(size);
            self.define(
                param.name.clone(),
                LocalInfo {
                    offset,
                    dims,
                    is_array_param: is_array,
                },
            );
            if idx < 6 {
                if is_array {
                    self.body
                        .push_str(&format!("  movq {}, {}(%rbp)\n", ptr_regs[idx], offset));
                } else {
                    self.body
                        .push_str(&format!("  movl {}, {}(%rbp)\n", regs[idx], offset));
                }
            } else {
                let src = 16 + ((idx - 6) as i32) * 8;
                if is_array {
                    self.body.push_str(&format!("  movq {}(%rbp), %rax\n", src));
                    self.body
                        .push_str(&format!("  movq %rax, {}(%rbp)\n", offset));
                } else {
                    self.body.push_str(&format!("  movl {}(%rbp), %eax\n", src));
                    self.body
                        .push_str(&format!("  movl %eax, {}(%rbp)\n", offset));
                }
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
            self.emit_def(def);
        }
    }

    fn emit_def(&mut self, def: &Def) {
        let dims = const_dims(&def.dims);
        let size = value_size(&dims);
        let offset = self.alloc(size);
        self.define(
            def.name.clone(),
            LocalInfo {
                offset,
                dims: dims.clone(),
                is_array_param: false,
            },
        );
        if let Some(init) = &def.init {
            let values = flatten_init_exprs(init, &dims);
            for (idx, expr) in values.iter().enumerate() {
                let Some(expr) = expr else { continue };
                self.emit_array_element_addr(offset, &dims, idx);
                self.push_rax();
                self.emit_expr(expr);
                self.pop_rcx();
                self.body.push_str("  movl %eax, (%rcx)\n");
            }
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Assign { target, value } => {
                self.emit_lvalue_addr(target);
                self.push_rax();
                self.emit_expr(value);
                self.pop_rcx();
                self.body.push_str("  movl %eax, (%rcx)\n");
            }
            Stmt::Expr(expr) => {
                if let Some(expr) = expr {
                    self.emit_expr(expr);
                }
            }
            Stmt::Block(block) => self.emit_block_items(&block.items),
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let else_label = self.parent.fresh_label("else");
                let end_label = self.parent.fresh_label("endif");
                self.emit_expr(cond);
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
                let start = self.parent.fresh_label("while");
                let end = self.parent.fresh_label("endwhile");
                self.continue_labels.push(start.clone());
                self.break_labels.push(end.clone());
                self.body.push_str(&format!("{}:\n", start));
                self.emit_expr(cond);
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
                    self.emit_expr(expr);
                }
                self.body
                    .push_str(&format!("  jmp {}\n", self.return_label));
            }
        }
    }

    fn emit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Int(v) => self
                .body
                .push_str(&format!("  movl ${}, %eax\n", *v as i32)),
            Expr::Float(_) => panic!("float codegen is not implemented yet"),
            Expr::String(_) => panic!("string codegen is not implemented yet"),
            Expr::LValue(lvalue) => {
                self.emit_lvalue_addr(lvalue);
                self.body.push_str("  movl (%rax), %eax\n");
            }
            Expr::Call { name, args } => self.emit_call(name, args),
            Expr::Unary { op, expr } => {
                self.emit_expr(expr);
                match op {
                    UnaryOp::Pos => {}
                    UnaryOp::Neg => self.body.push_str("  negl %eax\n"),
                    UnaryOp::Not => {
                        self.body
                            .push_str("  cmpl $0, %eax\n  sete %al\n  movzbl %al, %eax\n");
                    }
                }
            }
            Expr::Binary { op, lhs, rhs } => self.emit_binary(*op, lhs, rhs),
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) {
        if op == BinaryOp::And {
            let false_label = self.parent.fresh_label("land_false");
            let end_label = self.parent.fresh_label("land_end");
            self.emit_expr(lhs);
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  je {}\n", false_label));
            self.emit_expr(rhs);
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  je {}\n", false_label));
            self.body.push_str(&format!(
                "  movl $1, %eax\n  jmp {}\n{}:\n  movl $0, %eax\n{}:\n",
                end_label, false_label, end_label
            ));
            return;
        }
        if op == BinaryOp::Or {
            let true_label = self.parent.fresh_label("lor_true");
            let end_label = self.parent.fresh_label("lor_end");
            self.emit_expr(lhs);
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  jne {}\n", true_label));
            self.emit_expr(rhs);
            self.body
                .push_str(&format!("  cmpl $0, %eax\n  jne {}\n", true_label));
            self.body.push_str(&format!(
                "  movl $0, %eax\n  jmp {}\n{}:\n  movl $1, %eax\n{}:\n",
                end_label, true_label, end_label
            ));
            return;
        }

        self.emit_expr(rhs);
        self.push_rax();
        self.emit_expr(lhs);
        self.pop_rcx();
        match op {
            BinaryOp::Add => self.body.push_str("  addl %ecx, %eax\n"),
            BinaryOp::Sub => self.body.push_str("  subl %ecx, %eax\n"),
            BinaryOp::Mul => self.body.push_str("  imull %ecx, %eax\n"),
            BinaryOp::Div | BinaryOp::Mod => {
                self.body.push_str("  cltd\n  idivl %ecx\n");
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
                self.body.push_str("  cmpl %ecx, %eax\n");
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

    fn emit_call(&mut self, name: &str, args: &[Expr]) {
        let regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        let stack_args = args.len().saturating_sub(6);
        let pad = if stack_args % 2 == 1 { 8 } else { 0 };
        if pad != 0 {
            self.body.push_str("  subq $8, %rsp\n");
            self.temp_stack += 8;
        }
        for arg in args.iter().rev() {
            self.emit_expr(arg);
            self.body.push_str("  cltq\n");
            self.push_rax();
        }
        for idx in 0..args.len().min(6) {
            self.pop_reg(regs[idx]);
        }
        self.body.push_str("  movb $0, %al\n");
        self.body.push_str(&format!("  call {}\n", name));
        let cleanup = stack_args * 8 + pad;
        if cleanup != 0 {
            self.body.push_str(&format!("  addq ${}, %rsp\n", cleanup));
            self.temp_stack -= cleanup as i32;
        }
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
            self.emit_index_offset(&local.dims, &lvalue.indices);
        } else if let Some(global) = self.parent.globals.get(&lvalue.name).cloned() {
            self.body
                .push_str(&format!("  leaq {}(%rip), %rax\n", lvalue.name));
            self.emit_index_offset(&global.dims, &lvalue.indices);
        } else {
            panic!("undefined symbol {}", lvalue.name);
        }
    }

    fn emit_index_offset(&mut self, dims: &[i32], indices: &[Expr]) {
        for (idx, index) in indices.iter().enumerate() {
            let stride = dims.iter().skip(idx + 1).product::<i32>().max(1) * 4;
            self.push_rax();
            self.emit_expr(index);
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

    fn lookup(&self, name: &str) -> Option<&LocalInfo> {
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
struct LocalInfo {
    offset: i32,
    dims: Vec<i32>,
    is_array_param: bool,
}

#[derive(Debug, Clone)]
struct GlobalInfo {
    dims: Vec<i32>,
}

fn const_dims(dims: &[Expr]) -> Vec<i32> {
    dims.iter().map(eval_const_int).collect()
}

fn value_size(dims: &[i32]) -> i32 {
    element_count(dims) as i32 * 4
}

fn element_count(dims: &[i32]) -> usize {
    if dims.is_empty() {
        1
    } else {
        dims.iter().product::<i32>() as usize
    }
}

fn flatten_const_init(init: Option<&Init>, count: usize) -> Vec<i32> {
    let mut values = init
        .map(|init| {
            flatten_init_exprs(init, &[count as i32])
                .into_iter()
                .map(|expr| expr.map(eval_const_int).unwrap_or(0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    values.resize(count, 0);
    values
}

fn flatten_init_exprs<'a>(init: &'a Init, dims: &[i32]) -> Vec<Option<&'a Expr>> {
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

fn eval_const_int(expr: &Expr) -> i32 {
    match expr {
        Expr::Int(v) => *v as i32,
        Expr::Unary { op, expr } => {
            let v = eval_const_int(expr);
            match op {
                UnaryOp::Pos => v,
                UnaryOp::Neg => -v,
                UnaryOp::Not => (v == 0) as i32,
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let l = eval_const_int(lhs);
            let r = eval_const_int(rhs);
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
        _ => 0,
    }
}

fn align_to(value: i32, align: i32) -> i32 {
    (value + align - 1) / align * align
}
