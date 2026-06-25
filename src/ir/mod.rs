#![allow(dead_code)]

pub mod lower;

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

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub ret: Type,
    pub params: Vec<ValueId>,
    pub values: Vec<Value>,
    pub blocks: Vec<Block>,
    pub entry: BlockId,
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
        let mut verifier = Verifier::new(self);
        verifier.verify();
        if verifier.errors.is_empty() {
            Ok(())
        } else {
            Err(verifier.errors)
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError {
    pub message: String,
}

impl VerifyError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

struct Verifier<'a> {
    func: &'a Function,
    errors: Vec<VerifyError>,
}

impl<'a> Verifier<'a> {
    fn new(func: &'a Function) -> Self {
        Self {
            func,
            errors: Vec::new(),
        }
    }

    fn verify(&mut self) {
        self.check_block_id(self.func.entry, "entry block");

        for param in &self.func.params {
            self.check_value_id(*param, "function parameter");
        }

        for (block_idx, block) in self.func.blocks.iter().enumerate() {
            let block_id = BlockId(block_idx);
            if block.terminator.is_none() {
                self.error(format!("{} has no terminator", block_id));
            }
            self.verify_phi_placement(block_id, block);

            for (inst_idx, inst) in block.insts.iter().enumerate() {
                if let Some(result) = inst.result {
                    self.verify_inst_result(result, block_id, inst_idx);
                }
                self.verify_inst(block_id, inst);
            }

            if let Some(terminator) = &block.terminator {
                self.verify_terminator(block_id, terminator);
            }
        }
    }

    fn verify_phi_placement(&mut self, block_id: BlockId, block: &Block) {
        let mut seen_non_phi = false;
        for inst in &block.insts {
            if matches!(inst.kind, InstKind::Phi { .. }) {
                if seen_non_phi {
                    self.error(format!("{} has phi after non-phi instruction", block_id));
                }
            } else {
                seen_non_phi = true;
            }
        }
    }

    fn verify_inst_result(&mut self, result: ValueId, block: BlockId, inst_idx: usize) {
        self.check_value_id(result, "instruction result");
        if let Some(value) = self.func.values.get(result.0) {
            match value.kind {
                ValueKind::Inst(owner_block, owner_inst) => {
                    if owner_block != block || owner_inst != inst_idx {
                        self.error(format!(
                            "{} is attached to {} inst {}, but used as result of {} inst {}",
                            result, owner_block, owner_inst, block, inst_idx
                        ));
                    }
                }
                _ => self.error(format!(
                    "{} used as instruction result but is not Inst",
                    result
                )),
            }
        }
    }

    fn verify_inst(&mut self, block: BlockId, inst: &Inst) {
        match &inst.kind {
            InstKind::Phi { incomings } => {
                if inst.result.is_none() {
                    self.error(format!("{} has phi without result", block));
                }
                for (pred, value) in incomings {
                    self.check_block_id(*pred, "phi predecessor");
                    self.check_value_id(*value, "phi incoming value");
                }
            }
            InstKind::Alloca { .. } => {}
            InstKind::Load { ptr } => self.check_value_id(*ptr, "load pointer"),
            InstKind::Store { ptr, value } => {
                self.check_value_id(*ptr, "store pointer");
                self.check_value_id(*value, "store value");
                if inst.result.is_some() {
                    self.error(format!("{} has store with result", block));
                }
            }
            InstKind::Unary { value, .. } => self.check_value_id(*value, "unary operand"),
            InstKind::Binary { lhs, rhs, .. }
            | InstKind::Icmp { lhs, rhs, .. }
            | InstKind::Fcmp { lhs, rhs, .. } => {
                self.check_value_id(*lhs, "lhs operand");
                self.check_value_id(*rhs, "rhs operand");
            }
            InstKind::Cast { value, .. } => self.check_value_id(*value, "cast operand"),
            InstKind::Gep { base, indices } => {
                self.check_value_id(*base, "gep base");
                for index in indices {
                    self.check_value_id(*index, "gep index");
                }
            }
            InstKind::Call { args, .. } => {
                for arg in args {
                    self.check_value_id(*arg, "call argument");
                }
            }
        }
    }

    fn verify_terminator(&mut self, _block: BlockId, terminator: &Terminator) {
        match terminator {
            Terminator::Return(value) => {
                if let Some(value) = value {
                    self.check_value_id(*value, "return value");
                }
            }
            Terminator::Jump(target) => self.check_block_id(*target, "jump target"),
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                self.check_value_id(*cond, "branch condition");
                self.check_block_id(*then_target, "then target");
                self.check_block_id(*else_target, "else target");
            }
        }
    }

    fn check_block_id(&mut self, id: BlockId, context: &str) {
        if id.0 >= self.func.blocks.len() {
            self.error(format!("Invalid {} block id {}", context, id));
        }
    }

    fn check_value_id(&mut self, id: ValueId, context: &str) {
        if id.0 >= self.func.values.len() {
            self.error(format!("Invalid {} value id {}", context, id));
        }
    }

    fn error(&mut self, message: impl Into<String>) {
        self.errors.push(VerifyError::new(message));
    }
}
