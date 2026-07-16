use super::imm::{mov_w_imm, mov_x_imm};
use super::AArch64IrFuncEmitter;
use crate::codegen::common::{gep_elem_type, ir_size, pointee};
use crate::ir::{Const, Type, ValueId, ValueKind};

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_memzero(&mut self, ptr: ValueId, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let loop_label = self.parent.ctx.fresh_label("memzero");
        let end_label = self.parent.ctx.fresh_label("memzero_end");
        self.load_value(ptr);
        self.body.push_str(&mov_x_imm("x1", bytes as i64));
        self.body.push_str(&format!(
            "{}:\n  cbz x1, {}\n  strb wzr, [x0], #1\n  subs x1, x1, #1\n  b {}\n{}:\n",
            loop_label, end_label, loop_label, end_label
        ));
    }

    pub(super) fn emit_gep(&mut self, result: ValueId, base: ValueId, indices: &[ValueId]) {
        self.load_value(base);
        let mut ty = self.func.value(base).ty.clone();
        for (idx, index) in indices.iter().enumerate() {
            let elem_ty = if indices.len() == 1 {
                pointee(&self.func.value(result).ty).unwrap_or_else(|| gep_elem_type(&ty))
            } else {
                gep_elem_type(&ty)
            };
            let stride = ir_size(&elem_ty).max(1);
            self.push_x0();
            self.load_value(*index);
            self.body.push_str("  sxtw x0, w0\n");
            if stride != 1 {
                if stride > 0 && (stride & (stride - 1)) == 0 {
                    self.body
                        .push_str(&format!("  lsl x0, x0, #{}\n", stride.trailing_zeros()));
                } else {
                    self.body.push_str(&mov_x_imm("x16", stride as i64));
                    self.body.push_str("  mul x0, x0, x16\n");
                }
            }
            self.pop_x1();
            self.body.push_str("  add x0, x1, x0\n");
            ty = if idx + 1 == indices.len() {
                self.func.value(result).ty.clone()
            } else {
                elem_ty
            };
        }
    }

    pub(super) fn load_value(&mut self, value: ValueId) {
        if let Some(reg) = self.phi_regs.reg(value) {
            match self.func.value(value).ty {
                Type::Ptr(_) => self.body.push_str(&format!("  mov x0, {}\n", reg)),
                _ => self.body.push_str(&format!("  mov w0, {}\n", w_reg(reg))),
            }
            return;
        }

        match &self.func.value(value).kind {
            ValueKind::Const(value) => self.load_const(value),
            ValueKind::Global(name) => self.body.push_str(&format!(
                "  adrp x0, {}\n  add x0, x0, :lo12:{}\n",
                name, name
            )),
            _ => {
                let offset = self.layout.offset(value);
                match self.func.value(value).ty {
                    Type::Ptr(_) => self.load_frame_x("x0", offset),
                    _ => self.load_frame_w("w0", offset),
                }
            }
        }
    }

    pub(super) fn load_float_value(&mut self, value: ValueId, reg: &str) {
        match &self.func.value(value).kind {
            ValueKind::Const(Const::Float(bits)) => {
                let label = self.parent.ctx.fresh_label("float");
                self.parent.out.push_str(&format!(
                    ".section .rodata\n.align 2\n{}:\n  .word {}\n.text\n",
                    label, bits
                ));
                self.body.push_str(&format!(
                    "  adrp x16, {}\n  add x16, x16, :lo12:{}\n  ldr {}, [x16]\n",
                    label, label, reg
                ));
            }
            _ => {
                self.load_value(value);
                self.body.push_str(&format!("  fmov {}, w0\n", reg));
            }
        }
    }

    fn load_const(&mut self, value: &Const) {
        match value {
            Const::Int(value) => self.body.push_str(&mov_w_imm("w0", *value)),
            Const::Bool(value) => self.body.push_str(&mov_w_imm("w0", *value as i32)),
            Const::Float(bits) => self.body.push_str(&mov_w_imm("w0", *bits as i32)),
            Const::Zero(_) => self.body.push_str("  mov w0, wzr\n"),
            Const::String(_) | Const::Array(_) => self.body.push_str("  mov w0, wzr\n"),
        }
    }

    pub(super) fn load_indirect(&mut self, ty: &Type) {
        match ty {
            Type::Ptr(_) => self.body.push_str("  ldr x0, [x0]\n"),
            _ => self.body.push_str("  ldr w0, [x0]\n"),
        }
    }

    pub(super) fn store_indirect(&mut self, ty: &Type) {
        match ty {
            Type::Ptr(_) => self.body.push_str("  str x0, [x1]\n"),
            _ => self.body.push_str("  str w0, [x1]\n"),
        }
    }

    pub(super) fn store_result(&mut self, value: ValueId) {
        if let Some(reg) = self.phi_regs.reg(value) {
            match self.func.value(value).ty {
                Type::Ptr(_) => self.body.push_str(&format!("  mov {}, x0\n", reg)),
                _ => self.body.push_str(&format!("  mov {}, w0\n", w_reg(reg))),
            }
            return;
        }

        let offset = self.layout.offset(value);
        match self.func.value(value).ty {
            Type::Ptr(_) => self.store_frame_x("x0", offset),
            _ => self.store_frame_w("w0", offset),
        }
    }

    pub(super) fn push_x0(&mut self) {
        self.body.push_str("  str x0, [sp, #-16]!\n");
    }

    pub(super) fn pop_x1(&mut self) {
        self.body.push_str("  ldr x1, [sp], #16\n");
    }

    pub(super) fn push_s0(&mut self) {
        self.body
            .push_str("  sub sp, sp, #16\n  str xzr, [sp]\n  str s0, [sp]\n");
    }

    pub(super) fn pop_s1(&mut self) {
        self.body.push_str("  ldr s1, [sp]\n  add sp, sp, #16\n");
    }

    pub(super) fn adjust_sp(&mut self, amount: i32) {
        if amount == 0 {
            return;
        }
        let op = if amount < 0 { "sub" } else { "add" };
        let amount = amount.abs();
        if amount <= 4095 {
            self.body
                .push_str(&format!("  {} sp, sp, #{}\n", op, amount));
        } else {
            self.body.push_str(&mov_x_imm("x16", amount as i64));
            self.body.push_str(&format!("  {} sp, sp, x16\n", op));
        }
    }

    pub(super) fn frame_addr(&mut self, dst: &str, offset: i32) {
        self.base_addr(dst, "x29", offset);
    }

    fn base_addr(&mut self, dst: &str, base: &str, offset: i32) {
        if offset == 0 {
            self.body.push_str(&format!("  mov {}, {}\n", dst, base));
        } else if (1..=4095).contains(&offset) {
            self.body
                .push_str(&format!("  add {}, {}, #{}\n", dst, base, offset));
        } else if (-4095..=-1).contains(&offset) {
            self.body
                .push_str(&format!("  sub {}, {}, #{}\n", dst, base, -offset));
        } else if offset > 0 {
            self.body.push_str(&mov_x_imm("x16", offset as i64));
            self.body
                .push_str(&format!("  add {}, {}, x16\n", dst, base));
        } else {
            self.body.push_str(&mov_x_imm("x16", (-offset) as i64));
            self.body
                .push_str(&format!("  sub {}, {}, x16\n", dst, base));
        }
    }

    pub(super) fn load_frame_x(&mut self, dst: &str, offset: i32) {
        self.load_base_x(dst, "x29", offset);
    }

    pub(super) fn load_frame_w(&mut self, dst: &str, offset: i32) {
        self.load_base_w(dst, "x29", offset);
    }

    pub(super) fn load_frame_s(&mut self, dst: &str, offset: i32) {
        self.load_base_s(dst, "x29", offset);
    }

    pub(super) fn store_frame_x(&mut self, src: &str, offset: i32) {
        self.store_base_x(src, "x29", offset);
    }

    pub(super) fn store_frame_w(&mut self, src: &str, offset: i32) {
        self.store_base_w(src, "x29", offset);
    }

    pub(super) fn store_frame_s(&mut self, src: &str, offset: i32) {
        self.store_base_s(src, "x29", offset);
    }

    pub(super) fn load_sp_x(&mut self, dst: &str, offset: i32) {
        self.load_base_x(dst, "sp", offset);
    }

    pub(super) fn load_sp_s(&mut self, dst: &str, offset: i32) {
        self.load_base_s(dst, "sp", offset);
    }

    pub(super) fn store_sp_x(&mut self, src: &str, offset: i32) {
        self.store_base_x(src, "sp", offset);
    }

    fn load_base_x(&mut self, dst: &str, base: &str, offset: i32) {
        if let Some(op) = direct_mem_op(offset, 8) {
            self.body.push_str(&format!(
                "  {} {}, {}\n",
                op,
                dst,
                mem_operand(base, offset)
            ));
        } else {
            self.base_addr("x17", base, offset);
            self.body.push_str(&format!("  ldr {}, [x17]\n", dst));
        }
    }

    fn load_base_w(&mut self, dst: &str, base: &str, offset: i32) {
        if let Some(op) = direct_mem_op(offset, 4) {
            self.body.push_str(&format!(
                "  {} {}, {}\n",
                op,
                dst,
                mem_operand(base, offset)
            ));
        } else {
            self.base_addr("x17", base, offset);
            self.body.push_str(&format!("  ldr {}, [x17]\n", dst));
        }
    }

    fn load_base_s(&mut self, dst: &str, base: &str, offset: i32) {
        if let Some(op) = direct_mem_op(offset, 4) {
            self.body.push_str(&format!(
                "  {} {}, {}\n",
                op,
                dst,
                mem_operand(base, offset)
            ));
        } else {
            self.base_addr("x17", base, offset);
            self.body.push_str(&format!("  ldr {}, [x17]\n", dst));
        }
    }

    fn store_base_x(&mut self, src: &str, base: &str, offset: i32) {
        if let Some(op) = direct_store_op(offset, 8) {
            self.body.push_str(&format!(
                "  {} {}, {}\n",
                op,
                src,
                mem_operand(base, offset)
            ));
        } else {
            self.base_addr("x17", base, offset);
            self.body.push_str(&format!("  str {}, [x17]\n", src));
        }
    }

    fn store_base_w(&mut self, src: &str, base: &str, offset: i32) {
        if let Some(op) = direct_store_op(offset, 4) {
            self.body.push_str(&format!(
                "  {} {}, {}\n",
                op,
                src,
                mem_operand(base, offset)
            ));
        } else {
            self.base_addr("x17", base, offset);
            self.body.push_str(&format!("  str {}, [x17]\n", src));
        }
    }

    fn store_base_s(&mut self, src: &str, base: &str, offset: i32) {
        if let Some(op) = direct_store_op(offset, 4) {
            self.body.push_str(&format!(
                "  {} {}, {}\n",
                op,
                src,
                mem_operand(base, offset)
            ));
        } else {
            self.base_addr("x17", base, offset);
            self.body.push_str(&format!("  str {}, [x17]\n", src));
        }
    }

    pub(super) fn block_label(&self, block_idx: usize) -> String {
        format!(".L_{}_bb{}", self.func.name, block_idx)
    }

    pub(super) fn object_offset(&self, value: ValueId, _ty: &Type) -> i32 {
        self.layout.offset(value) + 8
    }
}

fn w_reg(x_reg: &str) -> String {
    x_reg.replacen('x', "w", 1)
}

fn direct_mem_op(offset: i32, size: i32) -> Option<&'static str> {
    if (-256..=255).contains(&offset) {
        return Some("ldur");
    }
    (offset >= 0 && offset % size == 0 && offset / size <= 4095).then_some("ldr")
}

fn direct_store_op(offset: i32, size: i32) -> Option<&'static str> {
    if (-256..=255).contains(&offset) {
        return Some("stur");
    }
    (offset >= 0 && offset % size == 0 && offset / size <= 4095).then_some("str")
}

fn mem_operand(base: &str, offset: i32) -> String {
    if offset == 0 {
        format!("[{}]", base)
    } else {
        format!("[{}, #{}]", base, offset)
    }
}
