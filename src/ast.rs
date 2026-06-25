#![allow(dead_code)]

use crate::token::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}

impl Program {
    pub fn funcs(&self) -> impl Iterator<Item = &Func> {
        self.items.iter().filter_map(|item| match item {
            Item::Func(func) => Some(func),
            Item::Decl(_) => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Decl(Decl),
    Func(Func),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Void,
    Float,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decl {
    pub is_const: bool,
    pub ty: Type,
    pub defs: Vec<Def>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Def {
    pub name: String,
    pub dims: Vec<Expr>,
    pub init: Option<Init>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Init {
    Expr(Expr),
    List(Vec<Init>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    pub name: String,
    pub ret: Type,
    pub params: Vec<FuncParam>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncParam {
    pub name: String,
    pub ty: Type,
    pub dims: Vec<Option<Expr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub items: Vec<BlockItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockItem {
    Decl(Decl),
    Stmt(Stmt),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign {
        target: LValue,
        value: Expr,
    },
    Expr(Option<Expr>),
    Block(Block),
    If {
        cond: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    While {
        cond: Expr,
        body: Box<Stmt>,
    },
    Break,
    Continue,
    Return(Option<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LValue {
    pub name: String,
    pub indices: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f32),
    String(String),
    LValue(LValue),
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Pos,
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}
