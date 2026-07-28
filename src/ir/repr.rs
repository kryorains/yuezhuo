use super::verify::{verify_function, VerifyError};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Void,
    I1,
    I32,
    F32,
    Ptr(Box<Type>),
    Array { elem: Box<Type>, len: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub globals: Vec<Global>,
    pub funcs: Vec<Function>,
}

impl Module {
    pub fn new() -> Self {
        Self {
            globals: Vec::new(),
            funcs: Vec::new(),
        }
    }

    pub fn add_func(&mut self, func: Function) -> FunctionId {
        let id = FunctionId(self.funcs.len());
        self.funcs.push(func);
        id
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Global {
    pub name: String,
    pub ty: Type,
    pub is_const: bool,
    pub init: Option<Const>,
}

#[derive(Clone)]
pub struct Function {
    pub name: String,
    pub ret: Type,
    pub params: Vec<ValueId>,
    pub values: Vec<Value>,
    pub blocks: Vec<Block>,
    pub entry: BlockId,
    recursive_cfg_inline_decided: bool,
    reduction_jammed: bool,
    simple_loop_unroll_decided: bool,
}

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.ret == other.ret
            && self.params == other.params
            && self.values == other.values
            && self.blocks == other.blocks
            && self.entry == other.entry
    }
}

impl fmt::Debug for Function {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Function")
            .field("name", &self.name)
            .field("ret", &self.ret)
            .field("params", &self.params)
            .field("values", &self.values)
            .field("blocks", &self.blocks)
            .field("entry", &self.entry)
            .finish()
    }
}

impl Function {
    pub fn new(name: impl Into<String>, ret: Type) -> Self {
        let mut func = Self {
            name: name.into(),
            ret,
            params: Vec::new(),
            values: Vec::new(),
            blocks: Vec::new(),
            entry: BlockId(0),
            recursive_cfg_inline_decided: false,
            reduction_jammed: false,
            simple_loop_unroll_decided: false,
        };
        func.entry = func.add_block("entry");
        func
    }

    pub fn add_param(&mut self, name: impl Into<String>, ty: Type) -> ValueId {
        let id = self.add_value(Value {
            name: Some(name.into()),
            ty,
            kind: ValueKind::Param,
        });
        self.params.push(id);
        id
    }

    pub fn add_const(&mut self, value: Const) -> ValueId {
        let ty = value.ty();
        self.add_value(Value {
            name: None,
            ty,
            kind: ValueKind::Const(value),
        })
    }

    pub fn add_global_ref(&mut self, name: impl Into<String>, ty: Type) -> ValueId {
        self.add_value(Value {
            name: None,
            ty,
            kind: ValueKind::Global(name.into()),
        })
    }

    pub fn add_block(&mut self, name: impl Into<String>) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block {
            name: name.into(),
            insts: Vec::new(),
            terminator: None,
        });
        id
    }

    pub fn append_inst(
        &mut self,
        block: BlockId,
        kind: InstKind,
        result_ty: Option<Type>,
    ) -> Option<ValueId> {
        let result = result_ty.map(|ty| {
            self.add_value(Value {
                name: None,
                ty,
                kind: ValueKind::Inst(block, self.block(block).insts.len()),
            })
        });
        self.block_mut(block).insts.push(Inst { result, kind });
        result
    }

    /// Inserts an instruction at an arbitrary position and keeps every
    /// instruction-backed value location consistent.
    pub fn insert_inst(
        &mut self,
        block: BlockId,
        inst_idx: usize,
        kind: InstKind,
        result_ty: Option<Type>,
    ) -> Option<ValueId> {
        let inst_len = self.block(block).insts.len();
        assert!(
            inst_idx <= inst_len,
            "instruction insertion index out of bounds"
        );
        if inst_idx == inst_len {
            return self.append_inst(block, kind, result_ty);
        }
        for value in &mut self.values {
            if let ValueKind::Inst(owner, owner_idx) = &mut value.kind {
                if *owner == block && *owner_idx >= inst_idx {
                    *owner_idx += 1;
                }
            }
        }

        let result = result_ty.map(|ty| {
            self.add_value(Value {
                name: None,
                ty,
                kind: ValueKind::Inst(block, inst_idx),
            })
        });
        self.block_mut(block)
            .insts
            .insert(inst_idx, Inst { result, kind });
        result
    }

    pub fn set_terminator(&mut self, block: BlockId, terminator: Terminator) {
        let slot = &mut self.block_mut(block).terminator;
        if slot.is_some() {
            panic!("Block {} already has a terminator", block);
        }
        *slot = Some(terminator);
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0]
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.0]
    }

    fn add_value(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len());
        self.values.push(value);
        id
    }

    pub fn verify(&self) -> Result<(), Vec<VerifyError>> {
        verify_function(self)
    }

    pub(crate) fn has_recursive_cfg_inline_decision(&self) -> bool {
        self.recursive_cfg_inline_decided
    }

    pub(crate) fn mark_recursive_cfg_inline_decision(&mut self) {
        self.recursive_cfg_inline_decided = true;
    }

    pub(crate) fn has_reduction_jam(&self) -> bool {
        self.reduction_jammed
    }

    pub(crate) fn mark_reduction_jammed(&mut self) {
        self.reduction_jammed = true;
    }

    pub(crate) fn simple_loop_unroll_decided(&self) -> bool {
        self.simple_loop_unroll_decided
    }

    pub(crate) fn mark_simple_loop_unroll_decided(&mut self) {
        self.simple_loop_unroll_decided = true;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub name: String,
    pub insts: Vec<Inst>,
    pub terminator: Option<Terminator>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: Option<String>,
    pub ty: Type,
    pub kind: ValueKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueKind {
    Param,
    Global(String),
    Const(Const),
    Inst(BlockId, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Const {
    Int(i32),
    Bool(bool),
    Float(u32),
    Zero(Type),
    String(String),
    Array(Vec<Const>),
}

impl Const {
    pub fn ty(&self) -> Type {
        match self {
            Const::Int(_) => Type::I32,
            Const::Bool(_) => Type::I1,
            Const::Float(_) => Type::F32,
            Const::Zero(ty) => ty.clone(),
            Const::String(s) => Type::Array {
                elem: Box::new(Type::I32),
                len: s.len() + 1,
            },
            Const::Array(values) => {
                let elem = values.first().map_or(Type::I32, Const::ty);
                Type::Array {
                    elem: Box::new(elem),
                    len: values.len(),
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inst {
    pub result: Option<ValueId>,
    pub kind: InstKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InstKind {
    Nop,
    Phi {
        incomings: Vec<(BlockId, ValueId)>,
    },
    Alloca {
        ty: Type,
    },
    Load {
        ptr: ValueId,
    },
    Store {
        ptr: ValueId,
        value: ValueId,
    },
    MemZero {
        ptr: ValueId,
        bytes: usize,
    },
    Unary {
        op: UnaryOp,
        value: ValueId,
    },
    Binary {
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    Icmp {
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    Fcmp {
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    Cast {
        op: CastOp,
        value: ValueId,
    },
    Gep {
        base: ValueId,
        indices: Vec<ValueId>,
    },
    Call {
        name: String,
        args: Vec<ValueId>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    Return(Option<ValueId>),
    Jump(BlockId),
    Branch {
        cond: ValueId,
        then_target: BlockId,
        else_target: BlockId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Ineg,
    Fneg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Iadd,
    Isub,
    Imul,
    Idiv,
    Imod,
    Iand,
    Ior,
    Ixor,
    /// Variable i32 shifts use the low five bits of the shift count.
    Ishl,
    Iashr,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastOp {
    I32ToF32,
    F32ToI32,
    BoolToI32,
    I32ToBool,
    F32ToBool,
}

impl fmt::Display for FunctionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}
