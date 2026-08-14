use super::pass::dominators::{ControlFlowGraph, Dominators};
use super::{
    BinaryOp, Block, BlockId, Function, Inst, InstKind, Terminator, Type, ValueId, ValueKind,
};
use std::collections::HashSet;

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
    cfg_valid: bool,
}

impl<'a> Verifier<'a> {
    fn new(func: &'a Function) -> Self {
        Self {
            func,
            errors: Vec::new(),
            cfg_valid: true,
        }
    }

    fn verify(&mut self) {
        self.check_block_id(self.func.entry, "entry block");
        self.verify_parameters();

        let mut instruction_results = HashSet::new();
        for (block_idx, block) in self.func.blocks.iter().enumerate() {
            let block_id = BlockId(block_idx);
            if block.terminator.is_none() {
                self.error(format!("{} has no terminator", block_id));
            }
            self.verify_phi_placement(block_id, block);

            for (inst_idx, inst) in block.insts.iter().enumerate() {
                if let Some(result) = inst.result {
                    if !instruction_results.insert(result) {
                        self.error(format!(
                            "{} is the result of more than one instruction",
                            result
                        ));
                    }
                    self.verify_inst_result(result, block_id, inst_idx);
                }
                self.verify_inst(block_id, inst);
            }

            if let Some(terminator) = &block.terminator {
                self.verify_terminator(block_id, terminator);
            }
        }

        // Never construct the CFG with malformed block IDs: its compact tables
        // deliberately assume structurally valid terminators and phi labels.
        if !self.cfg_valid || self.func.blocks.is_empty() {
            return;
        }
        let cfg = ControlFlowGraph::new(self.func);
        if !cfg.preds[self.func.entry.0].is_empty() {
            self.error(format!(
                "entry block {} has predecessors {:?}",
                self.func.entry, cfg.preds[self.func.entry.0]
            ));
        }
        let dom = Dominators::new(self.func, &cfg);
        self.verify_phi_predecessors(&cfg);
        self.verify_ssa_uses(&dom);
    }

    fn verify_parameters(&mut self) {
        let mut params = HashSet::new();
        for param in &self.func.params {
            self.check_value_id(*param, "function parameter");
            if !params.insert(*param) {
                self.error(format!(
                    "{} occurs more than once in function parameters",
                    param
                ));
            }
            if let Some(value) = self.func.values.get(param.0) {
                if value.kind != ValueKind::Param {
                    self.error(format!(
                        "{} is listed as a function parameter but has kind {:?}",
                        param, value.kind
                    ));
                }
            }
        }
        for (value_idx, value) in self.func.values.iter().enumerate() {
            if value.kind == ValueKind::Param && !params.contains(&ValueId(value_idx)) {
                self.error(format!(
                    "{} has Param kind but is absent from function parameters",
                    ValueId(value_idx)
                ));
            }
        }
    }

    fn verify_phi_placement(&mut self, block_id: BlockId, block: &Block) {
        let mut seen_non_phi = false;
        for inst in &block.insts {
            match inst.kind {
                InstKind::Nop => {}
                InstKind::Phi { .. } if block_id == self.func.entry => {
                    self.error(format!("entry block {} contains a phi", block_id));
                }
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
                let Some(result) = inst.result else {
                    self.error(format!("{} has phi without result", block));
                    for (pred, value) in incomings {
                        self.check_block_id(*pred, "phi predecessor");
                        self.check_value_id(*value, "phi incoming value");
                    }
                    return;
                };
                for (pred, incoming) in incomings {
                    self.check_block_id(*pred, "phi predecessor");
                    self.check_value_id(*incoming, "phi incoming value");
                    let Some(result_ty) = self.func.values.get(result.0).map(|value| &value.ty)
                    else {
                        continue;
                    };
                    if let Some(incoming_ty) =
                        self.func.values.get(incoming.0).map(|value| &value.ty)
                    {
                        if incoming_ty != result_ty {
                            self.error(format!(
                                "{} phi {} incoming {} from {} has type {:?}, expected {:?}",
                                block, result, incoming, pred, incoming_ty, result_ty
                            ));
                        }
                    }
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
            InstKind::MemZero { ptr, count, .. } => {
                self.check_value_id(*ptr, "memzero pointer");
                if let Some(count) = count {
                    self.check_value_id(*count, "memzero element count");
                    if self
                        .func
                        .values
                        .get(count.0)
                        .is_some_and(|value| value.ty != Type::I32)
                    {
                        self.error(format!("{} has non-i32 memzero element count", block));
                    }
                }
                if inst.result.is_some() {
                    self.error(format!("{} has memzero with result", block));
                }
            }
            InstKind::MemCopy {
                dst, src, count, ..
            } => {
                self.check_value_id(*dst, "memcopy destination");
                self.check_value_id(*src, "memcopy source");
                self.check_value_id(*count, "memcopy element count");
                if self
                    .func
                    .values
                    .get(count.0)
                    .is_some_and(|value| value.ty != Type::I32)
                {
                    self.error(format!("{} has non-i32 memcopy element count", block));
                }
                if inst.result.is_some() {
                    self.error(format!("{} has memcopy with result", block));
                }
            }
            InstKind::Unary { value, .. } => self.check_value_id(*value, "unary operand"),
            InstKind::Binary { op, lhs, rhs } => {
                self.check_value_id(*lhs, "lhs operand");
                self.check_value_id(*rhs, "rhs operand");
                self.verify_binary_types(block, inst, *op, *lhs, *rhs);
            }
            InstKind::Icmp { lhs, rhs, .. } | InstKind::Fcmp { lhs, rhs, .. } => {
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

    fn verify_binary_types(
        &mut self,
        block: BlockId,
        inst: &Inst,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let expected = match op {
            BinaryOp::Iadd
            | BinaryOp::Isub
            | BinaryOp::Imul
            | BinaryOp::Idiv
            | BinaryOp::Imod
            | BinaryOp::Iand
            | BinaryOp::Ior
            | BinaryOp::Ixor
            | BinaryOp::Ishl
            | BinaryOp::Iashr => Type::I32,
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => Type::F32,
            BinaryOp::And | BinaryOp::Or => Type::I1,
        };
        let Some(result) = inst.result else {
            self.error(format!("{} has binary instruction without result", block));
            return;
        };
        for (label, value) in [("lhs", lhs), ("rhs", rhs), ("result", result)] {
            if let Some(actual) = self.func.values.get(value.0).map(|value| &value.ty) {
                if actual != &expected {
                    self.error(format!(
                        "{} binary {:?} {} {} has type {:?}, expected {:?}",
                        block, op, label, value, actual, expected
                    ));
                }
            }
        }
    }

    fn verify_terminator(&mut self, block: BlockId, terminator: &Terminator) {
        match terminator {
            Terminator::Return(value) => match (&self.func.ret, value) {
                (Type::Void, None) => {}
                (Type::Void, Some(value)) => {
                    self.check_value_id(*value, "return value");
                    self.error(format!(
                        "{} returns {}, but function return type is Void",
                        block, value
                    ));
                }
                (expected, None) => self.error(format!(
                    "{} returns no value, but function return type is {:?}",
                    block, expected
                )),
                (expected, Some(value)) => {
                    self.check_value_id(*value, "return value");
                    if let Some(actual) = self.func.values.get(value.0).map(|value| &value.ty) {
                        if actual != expected {
                            self.error(format!(
                                "{} returns {} with type {:?}, expected {:?}",
                                block, value, actual, expected
                            ));
                        }
                    }
                }
            },
            Terminator::Jump(target) => self.check_block_id(*target, "jump target"),
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                self.check_value_id(*cond, "branch condition");
                if let Some(actual) = self.func.values.get(cond.0).map(|value| &value.ty) {
                    if actual != &Type::I1 {
                        self.error(format!(
                            "{} branch condition {} has type {:?}, expected I1",
                            block, cond, actual
                        ));
                    }
                }
                self.check_block_id(*then_target, "then target");
                self.check_block_id(*else_target, "else target");
            }
        }
    }

    fn verify_phi_predecessors(&mut self, cfg: &ControlFlowGraph) {
        for (block_idx, block) in self.func.blocks.iter().enumerate() {
            let block_id = BlockId(block_idx);
            let expected = cfg.preds[block_idx].iter().copied().collect::<HashSet<_>>();
            for inst in &block.insts {
                let InstKind::Phi { incomings } = &inst.kind else {
                    continue;
                };
                let mut seen = HashSet::new();
                for (pred, _) in incomings {
                    if !seen.insert(*pred) {
                        self.error(format!(
                            "{} phi {:?} has duplicate incoming predecessor {}",
                            block_id, inst.result, pred
                        ));
                    }
                    if !expected.contains(pred) {
                        self.error(format!(
                            "{} phi {:?} has incoming from non-predecessor {}",
                            block_id, inst.result, pred
                        ));
                    }
                }
                for pred in &expected {
                    if !seen.contains(pred) {
                        self.error(format!(
                            "{} phi {:?} is missing incoming predecessor {}",
                            block_id, inst.result, pred
                        ));
                    }
                }
            }
        }
    }

    fn verify_ssa_uses(&mut self, dom: &Dominators) {
        for (block_idx, block) in self.func.blocks.iter().enumerate() {
            let block_id = BlockId(block_idx);
            for (inst_idx, inst) in block.insts.iter().enumerate() {
                if let InstKind::Phi { incomings } = &inst.kind {
                    for (pred, value) in incomings {
                        self.verify_phi_edge_use(*value, *pred, block_id, inst.result, dom);
                    }
                    continue;
                }
                for operand in inst_operands(&inst.kind) {
                    self.verify_ordinary_use(
                        operand,
                        block_id,
                        inst_idx,
                        &format!("{} inst {}", block_id, inst_idx),
                        dom,
                    );
                }
            }
            if let Some(terminator) = &block.terminator {
                for operand in terminator_operands(terminator) {
                    self.verify_ordinary_use(
                        operand,
                        block_id,
                        block.insts.len(),
                        &format!("{} terminator", block_id),
                        dom,
                    );
                }
            }
        }
    }

    fn verify_ordinary_use(
        &mut self,
        value: ValueId,
        use_block: BlockId,
        use_inst: usize,
        context: &str,
        dom: &Dominators,
    ) {
        let (def_block, def_inst) = match self.instruction_definition(value) {
            Ok(Some(definition)) => definition,
            Ok(None) => return,
            Err(reason) => {
                self.error(format!("{} uses {} with {}", context, value, reason));
                return;
            }
        };
        if def_block == use_block {
            if def_inst >= use_inst {
                self.error(format!(
                    "{} uses {} before its definition at {} inst {}",
                    context, value, def_block, def_inst
                ));
            }
            return;
        }

        if !dom.dominates_for_availability(def_block, use_block) {
            self.error(format!(
                "definition of {} in {} does not dominate its use in {}",
                value, def_block, context
            ));
        }
    }

    fn verify_phi_edge_use(
        &mut self,
        value: ValueId,
        predecessor: BlockId,
        phi_block: BlockId,
        phi: Option<ValueId>,
        dom: &Dominators,
    ) {
        let (def_block, _) = match self.instruction_definition(value) {
            Ok(Some(definition)) => definition,
            Ok(None) => return,
            Err(reason) => {
                self.error(format!(
                    "phi {:?} in {} uses {} with {} on edge from {}",
                    phi, phi_block, value, reason, predecessor
                ));
                return;
            }
        };
        // Every instruction in the predecessor is available at its outgoing
        // edge. Otherwise the definition must dominate that predecessor.
        if def_block != predecessor && !dom.dominates_for_availability(def_block, predecessor) {
            self.error(format!(
                "definition of {} in {} is not available on edge {} -> {} for phi {:?}",
                value, def_block, predecessor, phi_block, phi
            ));
        }
    }

    fn instruction_definition(&self, value: ValueId) -> Result<Option<(BlockId, usize)>, String> {
        let Some(value_data) = self.func.values.get(value.0) else {
            return Err("an invalid ValueId".to_string());
        };
        let ValueKind::Inst(block, inst_idx) = value_data.kind else {
            // Param, Const, and Global values are available in every block.
            return Ok(None);
        };
        let Some(owner) = self.func.blocks.get(block.0) else {
            return Err(format!("an invalid definition block {}", block));
        };
        let Some(inst) = owner.insts.get(inst_idx) else {
            return Err(format!(
                "a missing definition at {} inst {}",
                block, inst_idx
            ));
        };
        if inst.result != Some(value) {
            return Err(format!(
                "no definition at {} inst {} (result is {:?})",
                block, inst_idx, inst.result
            ));
        }
        Ok(Some((block, inst_idx)))
    }

    fn check_block_id(&mut self, id: BlockId, context: &str) {
        if id.0 >= self.func.blocks.len() {
            self.cfg_valid = false;
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

fn inst_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Phi { .. } | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Load { ptr } => vec![*ptr],
        InstKind::MemZero { ptr, count, .. } => {
            std::iter::once(*ptr).chain(count.iter().copied()).collect()
        }
        InstKind::MemCopy {
            dst, src, count, ..
        } => vec![*dst, *src, *count],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut values = Vec::with_capacity(indices.len() + 1);
            values.push(*base);
            values.extend(indices.iter().copied());
            values
        }
        InstKind::Call { args, .. } => args.clone(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<ValueId> {
    match terminator {
        Terminator::Return(Some(value)) => vec![*value],
        Terminator::Branch { cond, .. } => vec![*cond],
        Terminator::Return(None) | Terminator::Jump(_) => Vec::new(),
    }
}
