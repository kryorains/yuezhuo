mod builder;
mod const_eval;
mod error;
mod expr;
mod function;
mod init;
mod program;
mod stmt;
mod types;

pub use error::LowerError;

use super::{BlockId, Const, Function, Module, Type, ValueId};
use crate::ast;
use std::collections::HashMap;

pub fn lower_program(program: &ast::Program) -> Result<Module, LowerError> {
    Lowerer::new(program).lower()
}

#[derive(Debug, Clone)]
pub(super) struct FunctionSig {
    ret: Type,
    params: Vec<Type>,
}

pub(super) struct Lowerer<'a> {
    program: &'a ast::Program,
    module: Module,
    funcs: HashMap<String, FunctionSig>,
    globals: HashMap<String, Type>,
    consts: HashMap<String, Const>,
}

pub(super) struct FunctionLowerer<'a> {
    ast_func: &'a ast::Func,
    sigs: &'a HashMap<String, FunctionSig>,
    globals: &'a HashMap<String, Type>,
    consts: &'a HashMap<String, Const>,
    func: Function,
    scopes: Vec<HashMap<String, Symbol>>,
    current: BlockId,
    loop_stack: Vec<LoopTargets>,
}

#[derive(Debug, Clone)]
pub(super) struct Symbol {
    ptr: ValueId,
    ty: Type,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LoopTargets {
    break_target: BlockId,
    continue_target: BlockId,
}
