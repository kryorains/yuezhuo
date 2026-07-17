use super::X86IrFuncEmitter;
use crate::ir::{
    BinaryOp, BlockId, CastOp, CmpOp, Inst, InstKind, Terminator, Type, UnaryOp, ValueId,
};

impl<'a, 'b> X86IrFuncEmitter<'a, 'b> {
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
                self.body.push_str(&format!(
                    "  leaq {}(%rbp), %rax\n",
                    self.object_offset(result, ty)
                ));
                self.store_result(result);
            }
            InstKind::Load { ptr } => {
                let result = inst.result.unwrap();
                self.load_value(*ptr);
                let ty = self.func.value(result).ty.clone();
                self.load_indirect(&ty);
                self.store_result(result);
            }
            InstKind::Store { ptr, value } => {
                self.load_value(*ptr);
                self.push_rax();
                self.load_value(*value);
                self.pop_rcx();
                let ty = self.func.value(*value).ty.clone();
                self.store_indirect(&ty);
            }
            InstKind::MemZero { ptr, bytes } => {
                self.emit_memzero(*ptr, *bytes);
            }
            InstKind::Unary { op, value } => {
                let result = inst.result.unwrap();
                self.emit_unary(*op, *value);
                self.store_result(result);
            }
            InstKind::Binary { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                self.emit_binary(*op, *lhs, *rhs);
                self.store_result(result);
            }
            InstKind::Icmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                self.emit_icmp(*op, *lhs, *rhs);
                self.store_result(result);
            }
            InstKind::Fcmp { op, lhs, rhs } => {
                let result = inst.result.unwrap();
                self.emit_fcmp(*op, *lhs, *rhs);
                self.store_result(result);
            }
            InstKind::Cast { op, value } => {
                let result = inst.result.unwrap();
                self.emit_cast(*op, *value);
                self.store_result(result);
            }
            InstKind::Gep { base, indices } => {
                let result = inst.result.unwrap();
                self.emit_gep(result, *base, indices);
                self.store_result(result);
            }
            InstKind::Call { name, args } => {
                let ret = self.emit_call(name, args);
                if let Some(result) = inst.result {
                    if ret == Type::F32 {
                        self.body.push_str(&format!(
                            "  movss %xmm0, {}(%rbp)\n",
                            self.layout.offset(result)
                        ));
                    } else {
                        self.store_result(result);
                    }
                }
            }
        }
    }

    pub(super) fn emit_terminator(&mut self, block_idx: usize, terminator: &Terminator) {
        match terminator {
            Terminator::Return(value) => {
                if let Some(value) = value {
                    self.load_value(*value);
                    if self.func.value(*value).ty == Type::F32 {
                        self.body.push_str("  movd %eax, %xmm0\n");
                    }
                }
                self.body
                    .push_str(&format!("  jmp {}\n", self.return_label));
            }
            Terminator::Jump(target) => {
                self.emit_phi_copies(block_idx, target.0);
                self.body
                    .push_str(&format!("  jmp {}\n", self.block_label(target.0)));
            }
            Terminator::Branch {
                cond,
                then_target,
                else_target,
            } => {
                let else_edge = self.parent.ctx.fresh_label("else_edge");
                self.load_value(*cond);
                self.body
                    .push_str(&format!("  cmpl $0, %eax\n  je {}\n", else_edge,));
                self.emit_phi_copies(block_idx, then_target.0);
                self.body.push_str(&format!(
                    "  jmp {}\n{}:\n",
                    self.block_label(then_target.0),
                    else_edge
                ));
                self.emit_phi_copies(block_idx, else_target.0);
                self.body
                    .push_str(&format!("  jmp {}\n", self.block_label(else_target.0)));
            }
        }
    }

    fn emit_phi_copies(&mut self, pred_idx: usize, target_idx: usize) {
        let copies = self
            .func
            .block(BlockId(target_idx))
            .insts
            .iter()
            .filter_map(|inst| {
                let incomings = match &inst.kind {
                    InstKind::Nop => return None,
                    InstKind::Phi { incomings } => incomings,
                    _ => return None,
                };
                let result = inst.result.unwrap();
                incomings
                    .iter()
                    .find(|(pred, _)| pred.0 == pred_idx)
                    .map(|(_, value)| (result, *value))
            })
            .collect::<Vec<_>>();

        for (_, value) in &copies {
            self.load_value(*value);
            self.push_rax();
        }
        for (result, _) in copies.iter().rev() {
            self.body.push_str("  popq %rax\n");
            self.store_result(*result);
        }
    }

    fn emit_unary(&mut self, op: UnaryOp, value: ValueId) {
        self.load_value(value);
        match op {
            UnaryOp::Ineg => self.body.push_str("  negl %eax\n"),
            UnaryOp::Fneg => self.body.push_str(
                "  movd %eax, %xmm0\n  movl $-2147483648, %eax\n  movd %eax, %xmm1\n  xorps %xmm1, %xmm0\n  movd %xmm0, %eax\n",
            ),
            UnaryOp::Not => self
                .body
                .push_str("  cmpl $0, %eax\n  sete %al\n  movzbl %al, %eax\n"),
        }
    }

    fn emit_binary(&mut self, op: BinaryOp, lhs: ValueId, rhs: ValueId) {
        match op {
            BinaryOp::Fadd | BinaryOp::Fsub | BinaryOp::Fmul | BinaryOp::Fdiv => {
                self.load_float_value(lhs, "%xmm0");
                self.body
                    .push_str("  subq $8, %rsp\n  movss %xmm0, (%rsp)\n");
                self.load_float_value(rhs, "%xmm0");
                self.body
                    .push_str("  movss (%rsp), %xmm1\n  addq $8, %rsp\n");
                match op {
                    BinaryOp::Fadd => self.body.push_str("  addss %xmm1, %xmm0\n"),
                    BinaryOp::Fsub => self
                        .body
                        .push_str("  subss %xmm0, %xmm1\n  movaps %xmm1, %xmm0\n"),
                    BinaryOp::Fmul => self.body.push_str("  mulss %xmm1, %xmm0\n"),
                    BinaryOp::Fdiv => self
                        .body
                        .push_str("  divss %xmm0, %xmm1\n  movaps %xmm1, %xmm0\n"),
                    _ => unreachable!(),
                }
                self.body.push_str("  movd %xmm0, %eax\n");
            }
            BinaryOp::And | BinaryOp::Or => {
                self.load_value(lhs);
                self.body
                    .push_str("  cmpl $0, %eax\n  setne %al\n  movzbl %al, %eax\n");
                self.push_rax();
                self.load_value(rhs);
                self.body
                    .push_str("  cmpl $0, %eax\n  setne %al\n  movzbl %al, %eax\n");
                self.pop_rcx();
                if op == BinaryOp::And {
                    self.body.push_str("  andl %ecx, %eax\n");
                } else {
                    self.body.push_str("  orl %ecx, %eax\n");
                }
            }
            BinaryOp::Ishl | BinaryOp::Iashr => {
                self.load_value(lhs);
                self.push_rax();
                self.load_value(rhs);
                self.body.push_str("  movl %eax, %ecx\n");
                self.pop_rax();
                if op == BinaryOp::Ishl {
                    self.body.push_str("  sall %cl, %eax\n");
                } else {
                    self.body.push_str("  sarl %cl, %eax\n");
                }
            }
            _ => {
                self.load_value(lhs);
                self.push_rax();
                self.load_value(rhs);
                self.pop_rcx();
                match op {
                    BinaryOp::Iadd => self.body.push_str("  addl %ecx, %eax\n"),
                    BinaryOp::Isub => self.body.push_str("  subl %eax, %ecx\n  movl %ecx, %eax\n"),
                    BinaryOp::Imul => self.body.push_str("  imull %ecx, %eax\n"),
                    BinaryOp::Idiv | BinaryOp::Imod => {
                        self.body.push_str(
                            "  movl %eax, %r8d\n  movl %ecx, %eax\n  cltd\n  idivl %r8d\n",
                        );
                        if op == BinaryOp::Imod {
                            self.body.push_str("  movl %edx, %eax\n");
                        }
                    }
                    BinaryOp::Iand => self.body.push_str("  andl %ecx, %eax\n"),
                    BinaryOp::Ior => self.body.push_str("  orl %ecx, %eax\n"),
                    BinaryOp::Ixor => self.body.push_str("  xorl %ecx, %eax\n"),
                    _ => unreachable!(),
                }
            }
        }
    }

    fn emit_icmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.load_value(lhs);
        self.push_rax();
        self.load_value(rhs);
        self.pop_rcx();
        self.body.push_str("  cmpl %eax, %ecx\n");
        let cc = match op {
            CmpOp::Lt => "setl",
            CmpOp::Gt => "setg",
            CmpOp::Le => "setle",
            CmpOp::Ge => "setge",
            CmpOp::Eq => "sete",
            CmpOp::Ne => "setne",
        };
        self.body
            .push_str(&format!("  {} %al\n  movzbl %al, %eax\n", cc));
    }

    fn emit_fcmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId) {
        self.load_float_value(lhs, "%xmm0");
        self.body
            .push_str("  subq $8, %rsp\n  movss %xmm0, (%rsp)\n");
        self.load_float_value(rhs, "%xmm0");
        self.body
            .push_str("  movss (%rsp), %xmm1\n  addq $8, %rsp\n  ucomiss %xmm0, %xmm1\n");
        let cc = match op {
            CmpOp::Lt => "setb",
            CmpOp::Gt => "seta",
            CmpOp::Le => "setbe",
            CmpOp::Ge => "setae",
            CmpOp::Eq => "sete",
            CmpOp::Ne => "setne",
        };
        self.body
            .push_str(&format!("  {} %al\n  movzbl %al, %eax\n", cc));
    }

    fn emit_cast(&mut self, op: CastOp, value: ValueId) {
        match op {
            CastOp::I32ToF32 | CastOp::BoolToI32 | CastOp::I32ToBool => {
                self.load_value(value);
                match op {
                    CastOp::I32ToF32 => {
                        self.body
                            .push_str("  cvtsi2ssl %eax, %xmm0\n  movd %xmm0, %eax\n");
                    }
                    CastOp::BoolToI32 => {}
                    CastOp::I32ToBool => self
                        .body
                        .push_str("  cmpl $0, %eax\n  setne %al\n  movzbl %al, %eax\n"),
                    _ => unreachable!(),
                }
            }
            CastOp::F32ToI32 => {
                self.load_float_value(value, "%xmm0");
                self.body.push_str("  cvttss2si %xmm0, %eax\n");
            }
            CastOp::F32ToBool => {
                self.load_float_value(value, "%xmm0");
                self.body.push_str(
                    "  pxor %xmm1, %xmm1\n  ucomiss %xmm1, %xmm0\n  setne %al\n  movzbl %al, %eax\n",
                );
            }
        }
    }
}
