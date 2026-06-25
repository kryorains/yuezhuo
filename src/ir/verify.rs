use super::{Block, BlockId, Function, Inst, InstKind, Terminator, ValueId, ValueKind};

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

pub(super) fn verify_function(func: &Function) -> Result<(), Vec<VerifyError>> {
    let mut verifier = Verifier::new(func);
    verifier.verify();
    if verifier.errors.is_empty() {
        Ok(())
    } else {
        Err(verifier.errors)
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
                self.verify_terminator(terminator);
            }
        }
    }

    fn verify_phi_placement(&mut self, block_id: BlockId, block: &Block) {
        let mut seen_non_phi = false;
        for inst in &block.insts {
            match inst.kind {
                InstKind::Nop => {}
                InstKind::Phi { .. } if seen_non_phi => {
                    self.error(format!("{} has phi after non-phi instruction", block_id));
                }
                InstKind::Phi { .. } => {}
                _ => seen_non_phi = true,
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
            InstKind::Nop => {
                if inst.result.is_some() {
                    self.error(format!("{} has nop with result", block));
                }
            }
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
            InstKind::MemZero { ptr, .. } => {
                self.check_value_id(*ptr, "memzero pointer");
                if inst.result.is_some() {
                    self.error(format!("{} has memzero with result", block));
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

    fn verify_terminator(&mut self, terminator: &Terminator) {
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
