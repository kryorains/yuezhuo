use super::types::{lower_param_type, lower_type};
use super::{FunctionLowerer, FunctionSig, LowerError, Symbol};
use crate::ast;
use crate::ir::{Const, Function, Terminator, Type};
use std::collections::HashMap;

impl<'a> FunctionLowerer<'a> {
    pub(super) fn new(
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

    pub(super) fn lower(mut self) -> Result<Function, LowerError> {
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
}
