use super::const_eval::lower_const_init;
use super::types::{array_type, lower_param_type, lower_type};
use super::{FunctionLowerer, FunctionSig, LowerError, Lowerer};
use crate::ast::{self, Item};
use crate::ir::{Const, Global, Module, Type};
use std::collections::HashMap;

impl<'a> Lowerer<'a> {
    pub(super) fn new(program: &'a ast::Program) -> Self {
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

    pub(super) fn lower(mut self) -> Result<Module, LowerError> {
        self.collect_global_consts()?;
        self.collect_func_sigs()?;

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

    fn collect_global_consts(&mut self) -> Result<(), LowerError> {
        for item in &self.program.items {
            let Item::Decl(decl) = item else {
                continue;
            };
            if !decl.is_const {
                continue;
            }

            let base = lower_type(&decl.ty);
            for def in &decl.defs {
                let ty = array_type(base.clone(), &def.dims, &self.consts)?;
                if let Some(init) = &def.init {
                    let init = lower_const_init(init, &ty, &self.consts)?;
                    self.consts.insert(def.name.clone(), init);
                }
            }
        }
        Ok(())
    }

    fn collect_func_sigs(&mut self) -> Result<(), LowerError> {
        for func in self.program.funcs() {
            self.funcs.insert(
                func.name.clone(),
                FunctionSig {
                    ret: lower_type(&func.ret),
                    params: func
                        .params
                        .iter()
                        .map(|param| lower_param_type(param, &self.consts))
                        .collect::<Result<Vec<_>, _>>()?,
                },
            );
        }
        Ok(())
    }

    fn add_runtime_sigs(&mut self) {
        for (name, ret, params) in [
            ("getint", Type::I32, vec![]),
            ("getch", Type::I32, vec![]),
            ("getfloat", Type::F32, vec![]),
            ("getarray", Type::I32, vec![Type::Ptr(Box::new(Type::I32))]),
            ("getfarray", Type::I32, vec![Type::Ptr(Box::new(Type::F32))]),
            ("putint", Type::Void, vec![Type::I32]),
            ("putch", Type::Void, vec![Type::I32]),
            ("putfloat", Type::Void, vec![Type::F32]),
            (
                "putarray",
                Type::Void,
                vec![Type::I32, Type::Ptr(Box::new(Type::I32))],
            ),
            (
                "putfarray",
                Type::Void,
                vec![Type::I32, Type::Ptr(Box::new(Type::F32))],
            ),
            ("putf", Type::Void, vec![Type::Ptr(Box::new(Type::I32))]),
            ("starttime", Type::Void, vec![]),
            ("stoptime", Type::Void, vec![]),
            ("_sysy_starttime", Type::Void, vec![Type::I32]),
            ("_sysy_stoptime", Type::Void, vec![Type::I32]),
        ] {
            self.funcs
                .insert(name.to_string(), FunctionSig { ret, params });
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
