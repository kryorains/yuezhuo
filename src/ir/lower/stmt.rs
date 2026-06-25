use super::types::array_type;
use super::{FunctionLowerer, LoopTargets, LowerError, Symbol};
use crate::ast::{self, BlockItem, Expr, Stmt};
use crate::ir::Terminator;

impl<'a> FunctionLowerer<'a> {
    pub(super) fn lower_block_items(&mut self, items: &[BlockItem]) -> Result<(), LowerError> {
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
        let base = super::types::lower_type(&decl.ty);
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

    pub(super) fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), LowerError> {
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
}
