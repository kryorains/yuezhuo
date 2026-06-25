use super::types::pointee;
use super::{FunctionLowerer, LowerError, Symbol};
use crate::ir::{
    BinaryOp as IrBinaryOp, BlockId, CastOp, CmpOp, Function, InstKind, Terminator, Type,
    UnaryOp as IrUnaryOp, ValueId,
};
use std::collections::HashMap;

impl<'a> FunctionLowerer<'a> {
    pub(super) fn alloca(&mut self, name: String, ty: Type) -> ValueId {
        self.func
            .append_inst(
                self.func.entry,
                InstKind::Alloca { ty: ty.clone() },
                Some(Type::Ptr(Box::new(ty))),
            )
            .unwrap()
            .tap_name(&mut self.func, name)
    }

    pub(super) fn load(&mut self, ptr: ValueId, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Load { ptr }, Some(ty))
            .unwrap()
    }

    pub(super) fn store(&mut self, ptr: ValueId, value: ValueId) {
        self.func
            .append_inst(self.current, InstKind::Store { ptr, value }, None);
    }

    pub(super) fn memzero(&mut self, ptr: ValueId, bytes: usize) {
        if bytes != 0 {
            self.func
                .append_inst(self.current, InstKind::MemZero { ptr, bytes }, None);
        }
    }

    pub(super) fn unary(&mut self, op: IrUnaryOp, value: ValueId, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Unary { op, value }, Some(ty))
            .unwrap()
    }

    pub(super) fn binary(
        &mut self,
        op: IrBinaryOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: Type,
    ) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Binary { op, lhs, rhs }, Some(ty))
            .unwrap()
    }

    pub(super) fn icmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.func
            .append_inst(
                self.current,
                InstKind::Icmp { op, lhs, rhs },
                Some(Type::I1),
            )
            .unwrap()
    }

    pub(super) fn fcmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.func
            .append_inst(
                self.current,
                InstKind::Fcmp { op, lhs, rhs },
                Some(Type::I1),
            )
            .unwrap()
    }

    pub(super) fn cast(&mut self, op: CastOp, value: ValueId, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Cast { op, value }, Some(ty))
            .unwrap()
    }

    pub(super) fn gep(&mut self, base: ValueId, indices: Vec<ValueId>, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Gep { base, indices }, Some(ty))
            .unwrap()
    }

    pub(super) fn phi(&mut self, incomings: Vec<(BlockId, ValueId)>, ty: Type) -> ValueId {
        self.func
            .append_inst(self.current, InstKind::Phi { incomings }, Some(ty))
            .unwrap()
    }

    pub(super) fn const_int(&mut self, value: i32) -> ValueId {
        self.func.add_const(crate::ir::Const::Int(value))
    }

    pub(super) fn const_zero(&mut self, ty: Type) -> ValueId {
        self.func.add_const(crate::ir::Const::Zero(ty))
    }

    pub(super) fn terminate(&mut self, terminator: Terminator) {
        if !self.is_terminated(self.current) {
            self.func.set_terminator(self.current, terminator);
        }
    }

    pub(super) fn is_terminated(&self, block: BlockId) -> bool {
        self.func.block(block).terminator.is_some()
    }

    pub(super) fn value_type(&self, value: ValueId) -> Type {
        self.func.value(value).ty.clone()
    }

    pub(super) fn pointee_type(&self, ptr: ValueId) -> Result<Type, LowerError> {
        pointee(&self.value_type(ptr)).ok_or_else(|| {
            LowerError::new(format!("expected pointer, got {:?}", self.value_type(ptr)))
        })
    }

    pub(super) fn define(&mut self, name: String, symbol: Symbol) -> Result<(), LowerError> {
        let scope = self.scopes.last_mut().unwrap();
        if scope.contains_key(&name) {
            return Err(LowerError::new(format!("redefined symbol '{}'", name)));
        }
        scope.insert(name, symbol);
        Ok(())
    }

    pub(super) fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    pub(super) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub(super) fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

trait NameValue {
    fn tap_name(self, func: &mut Function, name: String) -> Self;
}

impl NameValue for ValueId {
    fn tap_name(self, func: &mut Function, name: String) -> Self {
        func.values[self.0].name = Some(name);
        self
    }
}
