use super::Riscv64IrFuncEmitter;
use crate::ir::{
    BinaryOp, BlockId, CastOp, CmpOp, Const, Inst, InstKind, Terminator, Type, UnaryOp, ValueId,
    ValueKind,
};

const INT_PHI_CYCLE_SCRATCH: &str = "t2";
const FLOAT_PHI_CYCLE_SCRATCH: &str = "ft0";
const INT_PHI_MOVE_SCRATCHES: [&str; 4] = ["a0", "t0", "t1", INT_PHI_CYCLE_SCRATCH];
const MAX_PARALLEL_PHI_COPIES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhiLocation {
    IntReg(&'static str),
    FloatReg(&'static str),
    StackSlot { offset: i32, ty: PhiCopyType },
}

#[derive(Clone, Copy, Debug)]
enum PhiSource {
    Location(PhiLocation),
    Rematerialize(ValueId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhiCopyType {
    Word,
    Doubleword,
    Float,
}

#[derive(Clone, Copy, Debug)]
struct PhiCopy {
    destination: PhiLocation,
    source: PhiSource,
    ty: PhiCopyType,
}

#[derive(Clone, Copy, Debug)]
struct PhiSnapshotCopy {
    result: ValueId,
    incoming: ValueId,
    ty: PhiCopyType,
}

enum PhiCopyPlan {
    Parallel(Vec<PhiCopy>),
    Snapshot(Vec<PhiSnapshotCopy>),
}

impl PhiCopyPlan {
    fn is_empty(&self) -> bool {
        match self {
            Self::Parallel(copies) => copies.is_empty(),
            Self::Snapshot(copies) => copies.is_empty(),
        }
    }
}

enum NormalizedPhiCopy {
    Noop,
    Copy(PhiCopy),
    Unsupported,
}

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_inst(&mut self, inst: &Inst) {
        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Phi { incomings } => {
                let result = inst.result.unwrap();
                for (pred, value) in incomings {
                    self.body
                        .push_str(&format!("# phi {} from {} = {}\n", result, pred, value));
                }
            }
            InstKind::Alloca { ty } => {
                let result = inst.result.unwrap();
                self.frame_addr("a0", self.object_offset(result, ty));
                self.store_result(result);
            }
            InstKind::Load { ptr } => {
                let result = inst.result.unwrap();
                if !self.emit_assigned_load(result, *ptr) {
                    let (base, offset) = self.memory_address(*ptr);
                    self.load_value(base);
                    let ty = self.func.value(result).ty.clone();
                    self.load_indirect(&ty, offset);
                    if ty == Type::F32 {
                        self.store_float_result(result, "fa0");
                    } else {
                        self.store_result(result);
                    }
                }
            }
            InstKind::Store { ptr, value } => {
                if !self.emit_assigned_store(*ptr, *value) {
                    let (base, offset) = self.memory_address(*ptr);
                    self.load_value_into(base, "a1");
                    let ty = self.func.value(*value).ty.clone();
                    if ty == Type::F32 {
                        self.load_float_value(*value, "fa0");
                    } else {
                        self.load_value(*value);
                    }
                    self.store_indirect(&ty, offset);
                }
            }
            InstKind::MemZero { ptr, bytes } => self.emit_memzero(*ptr, *bytes),
            InstKind::Unary { op, value } => {
                let result = inst.result.unwrap();
                if !self.emit_assigned_unary(result, *op, *value) {
                    self.emit_unary(*op, *value);
                    if *op == UnaryOp::Fneg {
                        self.store_float_result(result, "fa0");
                    } else {
                        self.store_result(result);
                    }
                }
            }
            InstKind::Binary { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                if !self.emit_assigned_binary(result, *op, *lhs, *rhs) {
                    self.emit_binary(*op, *lhs, *rhs);
                    if matches!(
                        op,
                        BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv
                    ) {
                        self.store_float_result(result, "fa0");
                    } else {
                        self.store_result(result);
                    }
                }
            }
            InstKind::Icmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                if self.direct_branch_icmp(result).is_none()
                    && !self.emit_assigned_icmp(result, *op, *lhs, *rhs)
                {
                    self.emit_icmp(*op, *lhs, *rhs);
                    self.store_result(result);
                }
            }
            InstKind::Fcmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                if !self.emit_assigned_fcmp(result, *op, *lhs, *rhs) {
                    self.emit_fcmp(*op, *lhs, *rhs);
                    self.store_result(result);
                }
            }
            InstKind::Cast { op, value } => {
                let result = inst.result.unwrap();
                if !self.emit_assigned_cast(result, *op, *value) {
                    self.emit_cast(*op, *value);
                    if *op == CastOp::I32ToF32 {
                        self.store_float_result(result, "fa0");
                    } else {
                        self.store_result(result);
                    }
                }
            }
            InstKind::Gep { base, indices } => {
                let result = inst.result.unwrap();
                if self.skips_folded_memory_gep(result) {
                    return;
                }
                if !self.emit_assigned_gep(result, *base, indices) {
                    self.emit_gep(result, *base, indices);
                    self.store_result(result);
                }
            }
            InstKind::Call { name, args } => {
                let ret = self.emit_call(name, args, inst.result);
                if let Some(result) = inst.result {
                    if ret != Type::F32 {
                        self.store_result(result);
                    }
                }
            }
        }
    }

    pub(super) fn emit_terminator(
        &mut self,
        block_idx: usize,
        terminator: &Terminator,
        next_block: Option<usize>,
    ) {
        match terminator {
            Terminator::Return(value) => {
                if let Some(value) = value {
                    if self.func.value(*value).ty == Type::F32 {
                        self.load_float_value(*value, "fa0");
                    } else {
                        self.load_value(*value);
                    }
                }
                if next_block.is_some() {
                    self.body.push_str(&format!("  j {}\n", self.return_label));
                }
            }
            Terminator::Jump(target) => {
                self.emit_phi_copies(block_idx, target.0);
                if next_block != Some(target.0) {
                    self.body
                        .push_str(&format!("  j {}\n", self.block_label(target.0)));
                }
            }
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                if then_target == else_target {
                    self.emit_phi_copies(block_idx, then_target.0);
                    if next_block != Some(then_target.0) {
                        self.body
                            .push_str(&format!("  j {}\n", self.block_label(then_target.0)));
                    }
                    return;
                }

                if !self.edge_has_phi_copy(block_idx, then_target.0)
                    && !self.edge_has_phi_copy(block_idx, else_target.0)
                {
                    if next_block == Some(then_target.0) {
                        self.emit_branch_if_false(*cond, self.block_label(else_target.0));
                    } else if next_block == Some(else_target.0) {
                        self.emit_branch_if_true(*cond, self.block_label(then_target.0));
                    } else {
                        self.emit_branch_if_false(*cond, self.block_label(else_target.0));
                        self.body
                            .push_str(&format!("  j {}\n", self.block_label(then_target.0)));
                    }
                    return;
                }

                if next_block == Some(then_target.0) {
                    let then_edge = self.parent.ctx.fresh_label("then_edge");
                    self.emit_branch_if_true(*cond, then_edge.clone());
                    self.emit_phi_copies(block_idx, else_target.0);
                    self.body.push_str(&format!(
                        "  j {}\n{}:\n",
                        self.block_label(else_target.0),
                        then_edge
                    ));
                    self.emit_phi_copies(block_idx, then_target.0);
                } else {
                    let else_edge = self.parent.ctx.fresh_label("else_edge");
                    self.emit_branch_if_false(*cond, else_edge.clone());
                    self.emit_phi_copies(block_idx, then_target.0);
                    self.body.push_str(&format!(
                        "  j {}\n{}:\n",
                        self.block_label(then_target.0),
                        else_edge
                    ));
                    self.emit_phi_copies(block_idx, else_target.0);
                    if next_block != Some(else_target.0) {
                        self.body
                            .push_str(&format!("  j {}\n", self.block_label(else_target.0)));
                    }
                }
            }
        }
    }

    pub(super) fn emit_phi_copies(&mut self, pred_idx: usize, target_idx: usize) {
        match self.collect_phi_copies(pred_idx, target_idx) {
            PhiCopyPlan::Parallel(copies) => self.emit_parallel_phi_copies(copies),
            PhiCopyPlan::Snapshot(copies) => self.emit_snapshot_phi_copies(&copies),
        }
    }

    fn edge_has_phi_copy(&self, pred_idx: usize, target_idx: usize) -> bool {
        !self.collect_phi_copies(pred_idx, target_idx).is_empty()
    }

    fn collect_phi_copies(&self, pred_idx: usize, target_idx: usize) -> PhiCopyPlan {
        let mut raw_copies = Vec::new();
        let mut copies = Vec::new();
        let mut unsupported = false;

        for inst in &self.func.block(BlockId(target_idx)).insts {
            let (Some(result), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
                continue;
            };
            let Some((_, incoming)) = incomings.iter().find(|(pred, _)| pred.0 == pred_idx) else {
                continue;
            };
            match self.normalize_phi_copy(result, *incoming) {
                NormalizedPhiCopy::Noop => {}
                NormalizedPhiCopy::Copy(copy) => {
                    raw_copies.push(PhiSnapshotCopy {
                        result,
                        incoming: *incoming,
                        ty: copy.ty,
                    });
                    copies.push(copy);
                }
                NormalizedPhiCopy::Unsupported => {
                    raw_copies.push(PhiSnapshotCopy {
                        result,
                        incoming: *incoming,
                        ty: snapshot_phi_copy_type(&self.func.value(result).ty),
                    });
                    unsupported = true;
                }
            }
        }

        if unsupported
            || raw_copies.len() > MAX_PARALLEL_PHI_COPIES
            || !parallel_phi_invariants_hold(&copies)
        {
            PhiCopyPlan::Snapshot(raw_copies)
        } else {
            PhiCopyPlan::Parallel(copies)
        }
    }

    fn normalize_phi_copy(&self, result: ValueId, incoming: ValueId) -> NormalizedPhiCopy {
        if result == incoming {
            return NormalizedPhiCopy::Noop;
        }

        let result_ty = &self.func.value(result).ty;
        if self.func.value(incoming).ty != *result_ty {
            return NormalizedPhiCopy::Unsupported;
        }
        let ty = match result_ty {
            Type::I1 | Type::I32 => PhiCopyType::Word,
            Type::Ptr(_) => PhiCopyType::Doubleword,
            Type::F32 => PhiCopyType::Float,
            Type::Void | Type::Array { .. } => return NormalizedPhiCopy::Unsupported,
        };

        // Local registers are deliberately not physical edge locations. Their
        // allocator normally rejects phi uses; keep the old snapshot path if
        // that invariant ever changes.
        if self.local_regs.reg(result).is_some()
            || (self.regalloc.reg(incoming).is_none() && self.local_regs.reg(incoming).is_some())
        {
            return NormalizedPhiCopy::Unsupported;
        }

        let destination = if ty == PhiCopyType::Float {
            self.assigned_float_reg(result)
                .map(PhiLocation::FloatReg)
                .unwrap_or_else(|| PhiLocation::StackSlot {
                    offset: self.layout.offset(result),
                    ty,
                })
        } else {
            self.regalloc
                .reg(result)
                .map(PhiLocation::IntReg)
                .unwrap_or_else(|| PhiLocation::StackSlot {
                    offset: self.layout.offset(result),
                    ty,
                })
        };
        let assigned_source = match ty {
            PhiCopyType::Float => self.assigned_float_reg(incoming).map(PhiLocation::FloatReg),
            PhiCopyType::Word | PhiCopyType::Doubleword => {
                self.regalloc.reg(incoming).map(PhiLocation::IntReg)
            }
        };
        let source = if let Some(location) = assigned_source {
            PhiSource::Location(location)
        } else if matches!(
            self.func.value(incoming).kind,
            ValueKind::Const(_) | ValueKind::Global(_)
        ) {
            PhiSource::Rematerialize(incoming)
        } else {
            PhiSource::Location(PhiLocation::StackSlot {
                offset: self.layout.offset(incoming),
                ty,
            })
        };

        if matches!(source, PhiSource::Location(location) if location == destination) {
            NormalizedPhiCopy::Noop
        } else {
            let copy = PhiCopy {
                destination,
                source,
                ty,
            };
            if phi_copy_locations_match(copy) {
                NormalizedPhiCopy::Copy(copy)
            } else {
                NormalizedPhiCopy::Unsupported
            }
        }
    }

    fn emit_parallel_phi_copies(&mut self, mut copies: Vec<PhiCopy>) {
        while !copies.is_empty() {
            if let Some(ready) = copies.iter().position(|copy| {
                !copies.iter().any(|remaining| {
                    matches!(remaining.source, PhiSource::Location(source) if source == copy.destination)
                })
            }) {
                let copy = copies.remove(ready);
                self.emit_phi_copy(copy);
                continue;
            }

            // Every blocked destination is still a source. Preserve one such
            // old value in the reserved scratch, then schedule the opened cycle.
            let location = copies[0].destination;
            let ty = copies
                .iter()
                .find_map(|copy| match copy.source {
                    PhiSource::Location(source) if source == location => Some(copy.ty),
                    PhiSource::Location(_) | PhiSource::Rematerialize(_) => None,
                })
                .expect("blocked phi destination must be a remaining source");
            self.save_phi_cycle_location(location, ty);
            let scratch = match ty {
                PhiCopyType::Word | PhiCopyType::Doubleword => {
                    PhiLocation::IntReg(INT_PHI_CYCLE_SCRATCH)
                }
                PhiCopyType::Float => PhiLocation::FloatReg(FLOAT_PHI_CYCLE_SCRATCH),
            };
            for copy in &mut copies {
                if matches!(copy.source, PhiSource::Location(source) if source == location) {
                    copy.source = PhiSource::Location(scratch);
                }
            }
        }
    }

    fn emit_phi_copy(&mut self, copy: PhiCopy) {
        match copy.source {
            PhiSource::Rematerialize(value) => match copy.ty {
                PhiCopyType::Float => match copy.destination {
                    PhiLocation::FloatReg(destination) => {
                        self.load_float_value(value, destination);
                    }
                    PhiLocation::StackSlot { .. } => {
                        self.load_float_value(value, "fa0");
                        self.store_phi_location(copy.destination, "fa0", copy.ty);
                    }
                    PhiLocation::IntReg(_) => unreachable!("float phi destination must be float"),
                },
                PhiCopyType::Word | PhiCopyType::Doubleword => match copy.destination {
                    PhiLocation::IntReg(destination) => {
                        if !self.rematerialize_into(value, destination) {
                            self.load_value_into(value, "a0");
                            self.store_phi_location(copy.destination, "a0", copy.ty);
                        }
                    }
                    PhiLocation::StackSlot { .. } => {
                        self.load_value_into(value, "a0");
                        self.store_phi_location(copy.destination, "a0", copy.ty);
                    }
                    PhiLocation::FloatReg(_) => {
                        unreachable!("integer phi destination must be integer")
                    }
                },
            },
            PhiSource::Location(PhiLocation::IntReg(source)) => {
                self.store_phi_location(copy.destination, source, copy.ty);
            }
            PhiSource::Location(PhiLocation::FloatReg(source)) => {
                self.store_phi_location(copy.destination, source, copy.ty);
            }
            PhiSource::Location(PhiLocation::StackSlot { offset, ty }) => match copy.destination {
                PhiLocation::IntReg(destination) => match ty {
                    PhiCopyType::Word => self.load_frame_w(destination, offset),
                    PhiCopyType::Doubleword => self.load_frame_x(destination, offset),
                    PhiCopyType::Float => {
                        unreachable!("float stack phi destination must be float")
                    }
                },
                PhiLocation::FloatReg(destination) => {
                    debug_assert_eq!(ty, PhiCopyType::Float);
                    self.load_raw_frame_s(destination, offset - self.saved_area_size);
                }
                PhiLocation::StackSlot { .. } => {
                    let scratch = match ty {
                        PhiCopyType::Word => {
                            self.load_frame_w("a0", offset);
                            "a0"
                        }
                        PhiCopyType::Doubleword => {
                            self.load_frame_x("a0", offset);
                            "a0"
                        }
                        PhiCopyType::Float => {
                            self.load_raw_frame_s("fa0", offset - self.saved_area_size);
                            "fa0"
                        }
                    };
                    self.store_phi_location(copy.destination, scratch, copy.ty);
                }
            },
        }
    }

    fn store_phi_location(&mut self, destination: PhiLocation, source: &str, ty: PhiCopyType) {
        match destination {
            PhiLocation::IntReg(destination) => self
                .body
                .push_str(&format!("  mv {}, {}\n", destination, source)),
            PhiLocation::FloatReg(destination) => self
                .body
                .push_str(&format!("  fmv.s {}, {}\n", destination, source)),
            PhiLocation::StackSlot {
                offset,
                ty: slot_ty,
            } => {
                debug_assert_eq!(slot_ty, ty);
                match ty {
                    PhiCopyType::Word => self.store_frame_w(source, offset),
                    PhiCopyType::Doubleword => self.store_frame_x(source, offset),
                    PhiCopyType::Float => self.store_frame_s(source, offset),
                }
            }
        }
    }

    fn save_phi_cycle_location(&mut self, location: PhiLocation, ty: PhiCopyType) {
        match location {
            PhiLocation::IntReg(source) => self
                .body
                .push_str(&format!("  mv {}, {}\n", INT_PHI_CYCLE_SCRATCH, source)),
            PhiLocation::FloatReg(source) => self.body.push_str(&format!(
                "  fmv.s {}, {}\n",
                FLOAT_PHI_CYCLE_SCRATCH, source
            )),
            PhiLocation::StackSlot {
                offset,
                ty: slot_ty,
            } => {
                debug_assert_eq!(slot_ty, ty);
                match ty {
                    PhiCopyType::Word => self.load_frame_w(INT_PHI_CYCLE_SCRATCH, offset),
                    PhiCopyType::Doubleword => self.load_frame_x(INT_PHI_CYCLE_SCRATCH, offset),
                    PhiCopyType::Float => self
                        .load_raw_frame_s(FLOAT_PHI_CYCLE_SCRATCH, offset - self.saved_area_size),
                }
            }
        }
    }

    fn emit_snapshot_phi_copies(&mut self, copies: &[PhiSnapshotCopy]) {
        for copy in copies {
            match copy.ty {
                PhiCopyType::Float => {
                    self.load_float_value(copy.incoming, "fa0");
                    self.push_s0();
                }
                PhiCopyType::Word | PhiCopyType::Doubleword => {
                    self.load_value(copy.incoming);
                    self.push_x0();
                }
            }
        }
        for copy in copies.iter().rev() {
            match copy.ty {
                PhiCopyType::Float => {
                    self.body.push_str("  flw fa0, 0(sp)\n  addi sp, sp, 16\n");
                    self.store_float_result(copy.result, "fa0");
                }
                PhiCopyType::Word | PhiCopyType::Doubleword => {
                    self.pop_x1();
                    self.body.push_str("  mv a0, a1\n");
                    self.store_result(copy.result);
                }
            }
        }
    }

    fn emit_branch_if_false(&mut self, cond: ValueId, target: String) {
        if let Some((op, lhs, rhs)) = self.direct_branch_icmp(cond) {
            if self.emit_zero_icmp_branch_if_false(op, lhs, rhs, &target) {
                return;
            }
            let lhs = self.load_or_assigned(lhs, "a1");
            let rhs = self.load_or_assigned(rhs, "a0");
            let (branch, first, second) = match op {
                CmpOp::Lt => ("bge", lhs, rhs),
                CmpOp::Gt => ("bge", rhs, lhs),
                CmpOp::Le => ("blt", rhs, lhs),
                CmpOp::Ge => ("blt", lhs, rhs),
                CmpOp::Eq => ("bne", lhs, rhs),
                CmpOp::Ne => ("beq", lhs, rhs),
            };
            self.body
                .push_str(&format!("  {} {}, {}, {}\n", branch, first, second, target));
        } else {
            self.load_value(cond);
            self.body.push_str(&format!("  beqz a0, {}\n", target));
        }
    }

    fn emit_branch_if_true(&mut self, cond: ValueId, target: String) {
        if let Some((op, lhs, rhs)) = self.direct_branch_icmp(cond) {
            if self.emit_zero_icmp_branch_if_true(op, lhs, rhs, &target) {
                return;
            }
            let lhs = self.load_or_assigned(lhs, "a1");
            let rhs = self.load_or_assigned(rhs, "a0");
            let (branch, first, second) = match op {
                CmpOp::Lt => ("blt", lhs, rhs),
                CmpOp::Gt => ("blt", rhs, lhs),
                CmpOp::Le => ("bge", rhs, lhs),
                CmpOp::Ge => ("bge", lhs, rhs),
                CmpOp::Eq => ("beq", lhs, rhs),
                CmpOp::Ne => ("bne", lhs, rhs),
            };
            self.body
                .push_str(&format!("  {} {}, {}, {}\n", branch, first, second, target));
        } else {
            self.load_value(cond);
            self.body.push_str(&format!("  bnez a0, {}\n", target));
        }
    }

    fn emit_zero_icmp_branch_if_false(
        &mut self,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
        target: &str,
    ) -> bool {
        let (value, branch) = if let Some(branch) =
            const_i32(self.func, rhs).and_then(|constant| match (op, constant) {
                (CmpOp::Lt, 0) => Some("bgez"),
                (CmpOp::Gt, 0) => Some("blez"),
                (CmpOp::Le, 0) => Some("bgtz"),
                (CmpOp::Ge, 0) => Some("bltz"),
                (CmpOp::Eq, 0) => Some("bnez"),
                (CmpOp::Ne, 0) => Some("beqz"),
                (CmpOp::Lt, 1) => Some("bgtz"),
                (CmpOp::Ge, 1) => Some("blez"),
                (CmpOp::Gt, -1) => Some("bltz"),
                (CmpOp::Le, -1) => Some("bgez"),
                _ => None,
            }) {
            (lhs, branch)
        } else if const_i32(self.func, lhs) == Some(0) {
            let branch = match op {
                CmpOp::Lt => "blez",
                CmpOp::Gt => "bgez",
                CmpOp::Le => "bltz",
                CmpOp::Ge => "bgtz",
                CmpOp::Eq => "bnez",
                CmpOp::Ne => "beqz",
            };
            (rhs, branch)
        } else {
            return false;
        };
        let reg = self.load_or_assigned(value, "a0");
        self.body
            .push_str(&format!("  {} {}, {}\n", branch, reg, target));
        true
    }

    fn emit_zero_icmp_branch_if_true(
        &mut self,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
        target: &str,
    ) -> bool {
        let (value, branch) = if let Some(branch) =
            const_i32(self.func, rhs).and_then(|constant| match (op, constant) {
                (CmpOp::Lt, 0) => Some("bltz"),
                (CmpOp::Gt, 0) => Some("bgtz"),
                (CmpOp::Le, 0) => Some("blez"),
                (CmpOp::Ge, 0) => Some("bgez"),
                (CmpOp::Eq, 0) => Some("beqz"),
                (CmpOp::Ne, 0) => Some("bnez"),
                (CmpOp::Lt, 1) => Some("blez"),
                (CmpOp::Ge, 1) => Some("bgtz"),
                (CmpOp::Gt, -1) => Some("bgez"),
                (CmpOp::Le, -1) => Some("bltz"),
                _ => None,
            }) {
            (lhs, branch)
        } else if const_i32(self.func, lhs) == Some(0) {
            let branch = match op {
                CmpOp::Lt => "bgtz",
                CmpOp::Gt => "bltz",
                CmpOp::Le => "bgez",
                CmpOp::Ge => "blez",
                CmpOp::Eq => "beqz",
                CmpOp::Ne => "bnez",
            };
            (rhs, branch)
        } else {
            return false;
        };
        let reg = self.load_or_assigned(value, "a0");
        self.body
            .push_str(&format!("  {} {}, {}\n", branch, reg, target));
        true
    }

    fn emit_unary(&mut self, op: UnaryOp, value: ValueId) {
        match op {
            UnaryOp::Ineg => {
                self.load_value(value);
                self.body.push_str("  negw a0, a0\n");
            }
            UnaryOp::Fneg => {
                self.load_float_value(value, "fa0");
                self.body.push_str("  fneg.s fa0, fa0\n");
            }
            UnaryOp::Not => {
                self.load_value(value);
                self.body.push_str("  seqz a0, a0\n");
            }
        }
    }

    fn emit_assigned_unary(&mut self, result: ValueId, op: UnaryOp, value: ValueId) -> bool {
        if op == UnaryOp::Fneg {
            let Some(destination) = self.assigned_float_reg(result) else {
                return false;
            };
            self.load_float_value(value, "fa0");
            self.body
                .push_str(&format!("  fneg.s {}, fa0\n", destination));
            return true;
        }

        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };
        let source = self.load_or_assigned(value, "a0");
        match op {
            UnaryOp::Ineg => self
                .body
                .push_str(&format!("  negw {}, {}\n", destination, source)),
            UnaryOp::Not => self
                .body
                .push_str(&format!("  seqz {}, {}\n", destination, source)),
            UnaryOp::Fneg => return false,
        }
        true
    }

    fn emit_assigned_binary(
        &mut self,
        result: ValueId,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        if matches!(
            op,
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv
        ) {
            let Some(destination) = self.assigned_float_reg(result) else {
                return false;
            };
            self.load_float_value(lhs, "fa1");
            self.load_float_value(rhs, "fa0");
            let instruction = match op {
                BinaryOp::Fadd => "fadd.s",
                BinaryOp::Fsub => "fsub.s",
                BinaryOp::Fmul => "fmul.s",
                BinaryOp::Fdiv => "fdiv.s",
                _ => unreachable!(),
            };
            self.body
                .push_str(&format!("  {} {}, fa1, fa0\n", instruction, destination));
            return true;
        }

        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };

        match op {
            BinaryOp::Iadd => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_i12(*imm)) {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body
                        .push_str(&format!("  addiw {}, {}, {}\n", destination, source, imm));
                    return true;
                }
                if let Some(imm) = const_i32(self.func, lhs).filter(|imm| fits_i12(*imm)) {
                    let source = self.load_or_assigned(rhs, "a0");
                    self.body
                        .push_str(&format!("  addiw {}, {}, {}\n", destination, source, imm));
                    return true;
                }
            }
            BinaryOp::Isub => {
                if let Some(imm) = const_i32(self.func, rhs)
                    .and_then(i32::checked_neg)
                    .filter(|imm| fits_i12(*imm))
                {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body
                        .push_str(&format!("  addiw {}, {}, {}\n", destination, source, imm));
                    return true;
                }
            }
            BinaryOp::Imul => {
                if let Some(shift) = pow2_shift(self.func, rhs) {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body
                        .push_str(&format!("  slliw {}, {}, {}\n", destination, source, shift));
                    return true;
                }
                if let Some(shift) = pow2_shift(self.func, lhs) {
                    let source = self.load_or_assigned(rhs, "a0");
                    self.body
                        .push_str(&format!("  slliw {}, {}, {}\n", destination, source, shift));
                    return true;
                }
                if let Some((shift, add)) = const_i32(self.func, rhs).and_then(shift_add_pattern) {
                    let source = self.load_or_assigned(lhs, "a0");
                    let instruction = if add { "addw" } else { "subw" };
                    self.body.push_str(&format!(
                        "  slliw t0, {}, {}\n  {} {}, t0, {}\n",
                        source, shift, instruction, destination, source
                    ));
                    return true;
                }
                if let Some((shift, add)) = const_i32(self.func, lhs).and_then(shift_add_pattern) {
                    let source = self.load_or_assigned(rhs, "a0");
                    let instruction = if add { "addw" } else { "subw" };
                    self.body.push_str(&format!(
                        "  slliw t0, {}, {}\n  {} {}, t0, {}\n",
                        source, shift, instruction, destination, source
                    ));
                    return true;
                }
            }
            BinaryOp::Iand | BinaryOp::Ior | BinaryOp::Ixor => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_i12(*imm)) {
                    let source = self.load_or_assigned(lhs, "a0");
                    let instruction = match op {
                        BinaryOp::Iand => "andi",
                        BinaryOp::Ior => "ori",
                        BinaryOp::Ixor => "xori",
                        _ => unreachable!(),
                    };
                    self.body.push_str(&format!(
                        "  {} {}, {}, {}\n",
                        instruction, destination, source, imm
                    ));
                    return true;
                }
            }
            BinaryOp::Ishl | BinaryOp::Iashr => {
                if let Some(shift) = const_i32(self.func, rhs).and_then(word_shift_amount) {
                    let source = self.load_or_assigned(lhs, "a0");
                    let instruction = if op == BinaryOp::Ishl {
                        "slliw"
                    } else {
                        "sraiw"
                    };
                    self.body.push_str(&format!(
                        "  {} {}, {}, {}\n",
                        instruction, destination, source, shift
                    ));
                    return true;
                }
            }
            _ => {}
        }

        if matches!(op, BinaryOp::Idiv | BinaryOp::Imod) {
            if let Some(divisor) = const_i32(self.func, rhs) {
                self.emit_signed_divmod_const_into(lhs, divisor, op == BinaryOp::Imod, destination);
                return true;
            }
        }

        let lhs_reg = self.load_or_assigned(lhs, "a1");
        let rhs_reg = self.load_or_assigned(rhs, "a0");
        match op {
            BinaryOp::Iadd => self.body.push_str(&format!(
                "  addw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Isub => self.body.push_str(&format!(
                "  subw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Imul => self.body.push_str(&format!(
                "  mulw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Idiv => self.body.push_str(&format!(
                "  divw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Imod => self.body.push_str(&format!(
                "  remw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::And | BinaryOp::Or => {
                let instruction = if op == BinaryOp::And { "and" } else { "or" };
                if self.func.value(lhs).ty == Type::I1 && self.func.value(rhs).ty == Type::I1 {
                    self.body.push_str(&format!(
                        "  {} {}, {}, {}\n",
                        instruction, destination, lhs_reg, rhs_reg
                    ));
                } else {
                    self.body.push_str(&format!(
                        "  snez a1, {}\n  snez a0, {}\n  {} {}, a1, a0\n",
                        lhs_reg, rhs_reg, instruction, destination
                    ));
                }
            }
            BinaryOp::Iand => self.body.push_str(&format!(
                "  and {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Ior => self
                .body
                .push_str(&format!("  or {}, {}, {}\n", destination, lhs_reg, rhs_reg)),
            BinaryOp::Ixor => self.body.push_str(&format!(
                "  xor {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Ishl => self.body.push_str(&format!(
                "  sllw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Iashr => self.body.push_str(&format!(
                "  sraw {}, {}, {}\n",
                destination, lhs_reg, rhs_reg
            )),
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => return false,
        }
        true
    }

    pub(super) fn load_or_assigned(
        &mut self,
        value: ValueId,
        scratch: &'static str,
    ) -> &'static str {
        if let Some(reg) = self.assigned_reg(value) {
            reg
        } else {
            self.load_value_into(value, scratch);
            scratch
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) {
        if self.emit_binary_imm(op, lhs, rhs) {
            return;
        }

        match op {
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => {
                self.load_float_value(lhs, "fa1");
                self.load_float_value(rhs, "fa0");
                match op {
                    BinaryOp::Fadd => self.body.push_str("  fadd.s fa0, fa1, fa0\n"),
                    BinaryOp::Fsub => self.body.push_str("  fsub.s fa0, fa1, fa0\n"),
                    BinaryOp::Fmul => self.body.push_str("  fmul.s fa0, fa1, fa0\n"),
                    BinaryOp::Fdiv => self.body.push_str("  fdiv.s fa0, fa1, fa0\n"),
                    _ => unreachable!(),
                }
            }
            BinaryOp::And | BinaryOp::Or => {
                self.load_value_into(lhs, "a1");
                self.load_value(rhs);
                let instruction = if op == BinaryOp::And { "and" } else { "or" };
                if self.func.value(lhs).ty == Type::I1 && self.func.value(rhs).ty == Type::I1 {
                    self.body
                        .push_str(&format!("  {} a0, a1, a0\n", instruction));
                } else {
                    self.body.push_str(&format!(
                        "  snez a1, a1\n  snez a0, a0\n  {} a0, a1, a0\n",
                        instruction
                    ));
                }
            }
            _ => {
                self.load_value_into(lhs, "a1");
                self.load_value(rhs);
                match op {
                    BinaryOp::Iadd => self.body.push_str("  addw a0, a1, a0\n"),
                    BinaryOp::Isub => self.body.push_str("  subw a0, a1, a0\n"),
                    BinaryOp::Imul => self.body.push_str("  mulw a0, a1, a0\n"),
                    BinaryOp::Idiv => self.body.push_str("  divw a0, a1, a0\n"),
                    BinaryOp::Imod => self.body.push_str("  remw a0, a1, a0\n"),
                    BinaryOp::Iand => self.body.push_str("  and a0, a1, a0\n"),
                    BinaryOp::Ior => self.body.push_str("  or a0, a1, a0\n"),
                    BinaryOp::Ixor => self.body.push_str("  xor a0, a1, a0\n"),
                    BinaryOp::Ishl => self.body.push_str("  sllw a0, a1, a0\n"),
                    BinaryOp::Iashr => self.body.push_str("  sraw a0, a1, a0\n"),
                    _ => unreachable!(),
                }
            }
        }
    }

    fn emit_binary_imm(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) -> bool {
        match op {
            BinaryOp::Iadd => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_i12(*imm)) {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  addiw a0, a0, {}\n", imm));
                    return true;
                }
                if let Some(imm) = const_i32(self.func, lhs).filter(|imm| fits_i12(*imm)) {
                    self.load_value(rhs);
                    self.body.push_str(&format!("  addiw a0, a0, {}\n", imm));
                    return true;
                }
            }
            BinaryOp::Isub => {
                if let Some(imm) = const_i32(self.func, rhs)
                    .and_then(|imm| imm.checked_neg())
                    .filter(|imm| fits_i12(*imm))
                {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  addiw a0, a0, {}\n", imm));
                    return true;
                }
            }
            BinaryOp::Imul => {
                if let Some(shift) = pow2_shift(self.func, rhs) {
                    self.load_value(lhs);
                    self.body.push_str(&format!("  slliw a0, a0, {}\n", shift));
                    return true;
                }
                if let Some(shift) = pow2_shift(self.func, lhs) {
                    self.load_value(rhs);
                    self.body.push_str(&format!("  slliw a0, a0, {}\n", shift));
                    return true;
                }
                if let Some((shift, add)) = const_i32(self.func, rhs).and_then(shift_add_pattern) {
                    self.load_value(lhs);
                    let instruction = if add { "addw" } else { "subw" };
                    self.body.push_str(&format!(
                        "  slliw t0, a0, {}\n  {} a0, t0, a0\n",
                        shift, instruction
                    ));
                    return true;
                }
                if let Some((shift, add)) = const_i32(self.func, lhs).and_then(shift_add_pattern) {
                    self.load_value(rhs);
                    let instruction = if add { "addw" } else { "subw" };
                    self.body.push_str(&format!(
                        "  slliw t0, a0, {}\n  {} a0, t0, a0\n",
                        shift, instruction
                    ));
                    return true;
                }
            }
            BinaryOp::Iand | BinaryOp::Ior | BinaryOp::Ixor => {
                if let Some(imm) = const_i32(self.func, rhs).filter(|imm| fits_i12(*imm)) {
                    self.load_value(lhs);
                    let instruction = match op {
                        BinaryOp::Iand => "andi",
                        BinaryOp::Ior => "ori",
                        BinaryOp::Ixor => "xori",
                        _ => unreachable!(),
                    };
                    self.body
                        .push_str(&format!("  {} a0, a0, {}\n", instruction, imm));
                    return true;
                }
            }
            BinaryOp::Ishl | BinaryOp::Iashr => {
                if let Some(shift) = const_i32(self.func, rhs).and_then(word_shift_amount) {
                    self.load_value(lhs);
                    let instruction = if op == BinaryOp::Ishl {
                        "slliw"
                    } else {
                        "sraiw"
                    };
                    self.body
                        .push_str(&format!("  {} a0, a0, {}\n", instruction, shift));
                    return true;
                }
            }
            BinaryOp::Idiv | BinaryOp::Imod => {
                if let Some(divisor) = const_i32(self.func, rhs) {
                    self.emit_signed_divmod_const_into(lhs, divisor, op == BinaryOp::Imod, "a0");
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    fn emit_signed_divmod_const_into(
        &mut self,
        value: ValueId,
        divisor: i32,
        remainder: bool,
        destination: &'static str,
    ) {
        if divisor == 0 {
            if remainder {
                let source = self.load_or_assigned(value, "a0");
                if source != destination {
                    self.body
                        .push_str(&format!("  mv {}, {}\n", destination, source));
                }
            } else {
                self.body.push_str(&format!("  li {}, -1\n", destination));
            }
            return;
        }

        if divisor == 1 || divisor == -1 {
            if remainder {
                self.body.push_str(&format!("  li {}, 0\n", destination));
            } else {
                let source = self.load_or_assigned(value, "a0");
                if divisor < 0 {
                    self.body
                        .push_str(&format!("  negw {}, {}\n", destination, source));
                } else if source != destination {
                    self.body
                        .push_str(&format!("  mv {}, {}\n", destination, source));
                }
            }
            return;
        }

        let source = self.load_or_assigned(value, "a0");
        let abs_divisor = divisor.unsigned_abs();
        if remainder && abs_divisor.is_power_of_two() {
            let shift = abs_divisor.trailing_zeros();
            if shift == 1 {
                self.body.push_str(&format!("  srliw t0, {}, 31\n", source));
            } else {
                self.body.push_str(&format!(
                    "  sraiw t0, {}, 31\n  srliw t0, t0, {}\n",
                    source,
                    32 - shift
                ));
            }
            self.body.push_str(&format!("  addw t1, {}, t0\n", source));
            let mask = abs_divisor.wrapping_sub(1);
            if mask <= 2047 {
                self.body.push_str(&format!("  andi t1, t1, {}\n", mask));
            } else {
                self.body.push_str(&format!(
                    "  slliw t1, t1, {}\n  srliw t1, t1, {}\n",
                    32 - shift,
                    32 - shift
                ));
            }
            self.body
                .push_str(&format!("  subw {}, t1, t0\n", destination));
            return;
        }
        if remainder {
            self.body.push_str(&format!("  mv t2, {}\n", source));
        }

        // Bias negative dividends before shifting so the quotient truncates toward zero.
        if abs_divisor.is_power_of_two() {
            let shift = abs_divisor.trailing_zeros();
            if shift == 1 {
                self.body.push_str(&format!(
                    "  srliw t0, {}, 31\n  addw t0, {}, t0\n  sraiw t0, t0, 1\n",
                    source, source
                ));
            } else {
                self.body.push_str(&format!(
                    "  sraiw t0, {}, 31\n  srliw t0, t0, {}\n  addw t0, {}, t0\n  sraiw t0, t0, {}\n",
                    source,
                    32 - shift,
                    source,
                    shift
                ));
            }
        } else {
            // Widen to 64 bits: RV64 mulh is a 64x64 high multiply, while this
            // lowering needs the high half of a signed 32x32 product.
            let magic = signed_magic_positive(abs_divisor);
            self.body.push_str(&format!(
                "  li t0, {}\n  mul t0, {}, t0\n  srai t0, t0, 32\n",
                magic.multiplier, source
            ));
            if magic.add_dividend {
                self.body.push_str(&format!("  addw t0, t0, {}\n", source));
            }
            if magic.shift != 0 {
                self.body
                    .push_str(&format!("  sraiw t0, t0, {}\n", magic.shift));
            }
            self.body
                .push_str("  srliw t1, t0, 31\n  addw t0, t0, t1\n");
        }

        if divisor < 0 && !remainder {
            self.body.push_str("  negw t0, t0\n");
        }
        if remainder {
            if abs_divisor.is_power_of_two() {
                self.body.push_str(&format!(
                    "  slliw t0, t0, {}\n  subw {}, t2, t0\n",
                    abs_divisor.trailing_zeros(),
                    destination
                ));
            } else if abs_divisor.checked_add(1).is_some_and(u32::is_power_of_two) {
                self.body.push_str(&format!(
                    "  slliw t1, t0, {}\n  subw t0, t1, t0\n  subw {}, t2, t0\n",
                    (abs_divisor + 1).trailing_zeros(),
                    destination
                ));
            } else {
                self.body.push_str(&format!(
                    "  li t1, {}\n  mulw t0, t0, t1\n  subw {}, t2, t0\n",
                    abs_divisor as i32, destination
                ));
            }
        } else {
            self.body.push_str(&format!("  mv {}, t0\n", destination));
        }
    }

    fn direct_branch_icmp(&self, value: ValueId) -> Option<(CmpOp, ValueId, ValueId)> {
        if self.value_use_counts[value.0] != 1 {
            return None;
        }
        let ValueKind::Inst(block, inst_idx) = self.func.value(value).kind else {
            return None;
        };
        let Terminator::Branch { cond, .. } = self.func.block(block).terminator.as_ref()? else {
            return None;
        };
        if *cond != value {
            return None;
        }
        match self.func.block(block).insts.get(inst_idx)?.kind {
            InstKind::Icmp { op, lhs, rhs } => Some((op, lhs, rhs)),
            _ => None,
        }
    }

    fn emit_assigned_icmp(
        &mut self,
        result: ValueId,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };
        self.emit_icmp_into(destination, op, lhs, rhs);
        true
    }

    fn emit_icmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.emit_icmp_into("a0", op, lhs, rhs);
    }

    fn emit_icmp_into(
        &mut self,
        destination: &'static str,
        mut op: CmpOp,
        mut lhs: ValueId,
        mut rhs: ValueId,
    ) {
        if const_i32(self.func, lhs).is_some() && const_i32(self.func, rhs).is_none() {
            std::mem::swap(&mut lhs, &mut rhs);
            op = reverse_icmp(op);
        }
        if let Some(immediate) = const_i32(self.func, rhs) {
            match op {
                CmpOp::Eq | CmpOp::Ne if immediate == 0 => {
                    let source = self.load_or_assigned(lhs, "a0");
                    let instruction = if op == CmpOp::Eq { "seqz" } else { "snez" };
                    self.body
                        .push_str(&format!("  {} {}, {}\n", instruction, destination, source));
                    return;
                }
                CmpOp::Eq | CmpOp::Ne if fits_i12(immediate) => {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body.push_str(&format!(
                        "  xori {}, {}, {}\n",
                        destination, source, immediate
                    ));
                    let instruction = if op == CmpOp::Eq { "seqz" } else { "snez" };
                    self.body.push_str(&format!(
                        "  {} {}, {}\n",
                        instruction, destination, destination
                    ));
                    return;
                }
                CmpOp::Lt if fits_i12(immediate) => {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body.push_str(&format!(
                        "  slti {}, {}, {}\n",
                        destination, source, immediate
                    ));
                    return;
                }
                CmpOp::Le => {
                    if let Some(next) = immediate.checked_add(1).filter(|next| fits_i12(*next)) {
                        let source = self.load_or_assigned(lhs, "a0");
                        self.body
                            .push_str(&format!("  slti {}, {}, {}\n", destination, source, next));
                        return;
                    }
                }
                CmpOp::Gt => {
                    if let Some(next) = immediate.checked_add(1).filter(|next| fits_i12(*next)) {
                        let source = self.load_or_assigned(lhs, "a0");
                        self.body.push_str(&format!(
                            "  slti {}, {}, {}\n  seqz {}, {}\n",
                            destination, source, next, destination, destination
                        ));
                        return;
                    }
                }
                CmpOp::Ge if fits_i12(immediate) => {
                    let source = self.load_or_assigned(lhs, "a0");
                    self.body.push_str(&format!(
                        "  slti {}, {}, {}\n  seqz {}, {}\n",
                        destination, source, immediate, destination, destination
                    ));
                    return;
                }
                _ => {}
            }
        }

        let lhs = self.load_or_assigned(lhs, "a1");
        let rhs = self.load_or_assigned(rhs, "a0");
        match op {
            CmpOp::Lt => self
                .body
                .push_str(&format!("  slt {}, {}, {}\n", destination, lhs, rhs)),
            CmpOp::Gt => self
                .body
                .push_str(&format!("  slt {}, {}, {}\n", destination, rhs, lhs)),
            CmpOp::Le => self.body.push_str(&format!(
                "  slt {}, {}, {}\n  seqz {}, {}\n",
                destination, rhs, lhs, destination, destination
            )),
            CmpOp::Ge => self.body.push_str(&format!(
                "  slt {}, {}, {}\n  seqz {}, {}\n",
                destination, lhs, rhs, destination, destination
            )),
            CmpOp::Eq | CmpOp::Ne => {
                self.body
                    .push_str(&format!("  sub {}, {}, {}\n", destination, lhs, rhs));
                let instruction = if op == CmpOp::Eq { "seqz" } else { "snez" };
                self.body.push_str(&format!(
                    "  {} {}, {}\n",
                    instruction, destination, destination
                ));
            }
        }
    }

    fn emit_assigned_fcmp(
        &mut self,
        result: ValueId,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };
        self.emit_fcmp_into(destination, op, lhs, rhs);
        true
    }

    fn emit_fcmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.emit_fcmp_into("a0", op, lhs, rhs);
    }

    fn emit_fcmp_into(&mut self, destination: &'static str, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.load_float_value(lhs, "fa1");
        self.load_float_value(rhs, "fa0");
        match op {
            CmpOp::Lt => self
                .body
                .push_str(&format!("  flt.s {}, fa1, fa0\n", destination)),
            CmpOp::Gt => self
                .body
                .push_str(&format!("  flt.s {}, fa0, fa1\n", destination)),
            CmpOp::Le => self
                .body
                .push_str(&format!("  fle.s {}, fa1, fa0\n", destination)),
            CmpOp::Ge => self
                .body
                .push_str(&format!("  fle.s {}, fa0, fa1\n", destination)),
            CmpOp::Eq => self
                .body
                .push_str(&format!("  feq.s {}, fa1, fa0\n", destination)),
            CmpOp::Ne => self.body.push_str(&format!(
                "  feq.s {}, fa1, fa0\n  seqz {}, {}\n",
                destination, destination, destination
            )),
        }
    }

    fn emit_assigned_cast(&mut self, result: ValueId, op: CastOp, value: ValueId) -> bool {
        if op == CastOp::I32ToF32 {
            let Some(destination) = self.assigned_float_reg(result) else {
                return false;
            };
            let source = self.load_or_assigned(value, "a0");
            self.body
                .push_str(&format!("  fcvt.s.w {}, {}\n", destination, source));
            return true;
        }

        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };
        match op {
            CastOp::BoolToI32 => {
                let source = self.load_or_assigned(value, "a0");
                if source != destination {
                    self.body
                        .push_str(&format!("  mv {}, {}\n", destination, source));
                }
            }
            CastOp::I32ToBool => {
                let source = self.load_or_assigned(value, "a0");
                self.body
                    .push_str(&format!("  snez {}, {}\n", destination, source));
            }
            CastOp::F32ToI32 => {
                self.load_float_value(value, "fa0");
                self.body
                    .push_str(&format!("  fcvt.w.s {}, fa0, rtz\n", destination));
            }
            CastOp::F32ToBool => {
                self.load_float_value(value, "fa0");
                self.body.push_str(&format!(
                    "  fmv.w.x ft0, zero\n  feq.s {}, fa0, ft0\n  seqz {}, {}\n",
                    destination, destination, destination
                ));
            }
            CastOp::I32ToF32 => return false,
        }
        true
    }

    fn emit_cast(&mut self, op: CastOp, value: ValueId) {
        match op {
            CastOp::I32ToF32 | CastOp::BoolToI32 | CastOp::I32ToBool => {
                self.load_value(value);
                match op {
                    CastOp::I32ToF32 => self.body.push_str("  fcvt.s.w fa0, a0\n"),
                    CastOp::BoolToI32 => {}
                    CastOp::I32ToBool => self.body.push_str("  snez a0, a0\n"),
                    _ => unreachable!(),
                }
            }
            CastOp::F32ToI32 => {
                self.load_float_value(value, "fa0");
                self.body.push_str("  fcvt.w.s a0, fa0, rtz\n");
            }
            CastOp::F32ToBool => {
                self.load_float_value(value, "fa0");
                self.body
                    .push_str("  fmv.w.x ft0, zero\n  feq.s a0, fa0, ft0\n  seqz a0, a0\n");
            }
        }
    }
}

fn parallel_phi_invariants_hold(copies: &[PhiCopy]) -> bool {
    for (idx, copy) in copies.iter().enumerate() {
        if copies[..idx]
            .iter()
            .any(|previous| previous.destination == copy.destination)
            || !phi_copy_locations_match(*copy)
            || location_uses_move_scratch(copy.destination)
            || matches!(copy.source, PhiSource::Location(location) if location_uses_move_scratch(location))
        {
            return false;
        }
    }
    true
}

fn phi_copy_locations_match(copy: PhiCopy) -> bool {
    phi_location_matches_type(copy.destination, copy.ty)
        && match copy.source {
            PhiSource::Location(source) => phi_location_matches_type(source, copy.ty),
            PhiSource::Rematerialize(_) => true,
        }
}

fn phi_location_matches_type(location: PhiLocation, ty: PhiCopyType) -> bool {
    match location {
        PhiLocation::IntReg(_) => matches!(ty, PhiCopyType::Word | PhiCopyType::Doubleword),
        PhiLocation::FloatReg(_) => ty == PhiCopyType::Float,
        PhiLocation::StackSlot { ty: slot_ty, .. } => slot_ty == ty,
    }
}

fn location_uses_move_scratch(location: PhiLocation) -> bool {
    match location {
        PhiLocation::IntReg(reg) => INT_PHI_MOVE_SCRATCHES.contains(&reg),
        PhiLocation::FloatReg(reg) => reg == FLOAT_PHI_CYCLE_SCRATCH,
        PhiLocation::StackSlot { .. } => false,
    }
}

fn snapshot_phi_copy_type(ty: &Type) -> PhiCopyType {
    match ty {
        Type::Ptr(_) => PhiCopyType::Doubleword,
        Type::F32 => PhiCopyType::Float,
        Type::I1 | Type::I32 | Type::Void | Type::Array { .. } => PhiCopyType::Word,
    }
}

fn const_i32(func: &crate::ir::Function, value: ValueId) -> Option<i32> {
    match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => Some(*value),
        ValueKind::Const(Const::Bool(value)) => Some(*value as i32),
        _ => None,
    }
}

fn pow2_shift(func: &crate::ir::Function, value: ValueId) -> Option<u32> {
    let imm = const_i32(func, value)?;
    (imm > 0 && (imm & (imm - 1)) == 0).then_some(imm.trailing_zeros())
}

fn shift_add_pattern(value: i32) -> Option<(u32, bool)> {
    if value <= 2 {
        return None;
    }
    if value == i32::MAX {
        return Some((31, false));
    }
    if let Some(power) = value
        .checked_sub(1)
        .filter(|power| (*power as u32).is_power_of_two())
    {
        return Some((power.trailing_zeros(), true));
    }
    value
        .checked_add(1)
        .filter(|power| (*power as u32).is_power_of_two())
        .map(|power| (power.trailing_zeros(), false))
}

fn word_shift_amount(value: i32) -> Option<u32> {
    (0..32).contains(&value).then_some(value as u32)
}

fn reverse_icmp(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Ge => CmpOp::Le,
    }
}

fn fits_i12(value: i32) -> bool {
    (-2048..=2047).contains(&value)
}

#[derive(Debug, Clone, Copy)]
struct SignedMagic {
    multiplier: i32,
    shift: u32,
    add_dividend: bool,
}

// Hacker's Delight, chapter 10: magic multiplier for signed division by a
// positive, non-power-of-two 32-bit constant.
fn signed_magic_positive(divisor: u32) -> SignedMagic {
    debug_assert!((2..=i32::MAX as u32).contains(&divisor));
    debug_assert!(!divisor.is_power_of_two());

    let divisor = u64::from(divisor);
    let two31 = 1u64 << 31;
    let anc = two31 - 1 - two31 % divisor;
    let mut p = 31u32;
    let (mut q1, mut r1) = (two31 / anc, two31 % anc);
    let (mut q2, mut r2) = (two31 / divisor, two31 % divisor);

    loop {
        p += 1;
        q1 <<= 1;
        r1 <<= 1;
        if r1 >= anc {
            q1 += 1;
            r1 -= anc;
        }
        q2 <<= 1;
        r2 <<= 1;
        if r2 >= divisor {
            q2 += 1;
            r2 -= divisor;
        }

        let delta = divisor - r2;
        if q1 > delta || (q1 == delta && r1 != 0) {
            break;
        }
    }

    let multiplier = (q2 + 1) as u32 as i32;
    SignedMagic {
        multiplier,
        shift: p - 32,
        add_dividend: multiplier < 0,
    }
}
