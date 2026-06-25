#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Void,
    Float,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<Func>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    pub name: String,
    pub ret: Type,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Return(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
}
