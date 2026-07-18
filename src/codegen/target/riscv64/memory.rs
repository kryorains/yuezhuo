use super::Riscv64IrFuncEmitter;
use crate::codegen::common::{gep_elem_type, ir_size, pointee};
use crate::ir::{Const, Type, ValueId, ValueKind};

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_memzero(&mut self, ptr: ValueId, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let loop_label = self.parent.ctx.fresh_label("memzero");
        let end_label = self.parent.ctx.fresh_label("memzero_end");
        self.load_value(ptr);
        self.body.push_str(&format!("  li a1, {}\n", bytes));
        self.body.push_str(&format!(
            "{}:\n  beqz a1, {}\n  sb zero, 0(a0)\n  addi a0, a0, 1\n  addi a1, a1, -1\n  j {}\n{}:\n",
            loop_label, end_label, loop_label, end_label
        ));
    }

    pub(super) fn assigned_reg(&self, value: ValueId) -> Option<&'static str> {
        self.regalloc
            .reg(value)
            .or_else(|| self.local_regs.reg(value))
    }

    pub(super) fn emit_assigned_load(&mut self, result: ValueId, ptr: ValueId) -> bool {
        let (Some(destination), Some(pointer)) =
            (self.assigned_reg(result), self.assigned_reg(ptr))
        else {
            return false;
        };
        match self.func.value(result).ty {
            Type::Ptr(_) => self
                .body
                .push_str(&format!("  ld {}, 0({})\n", destination, pointer)),
            _ => self
                .body
                .push_str(&format!("  lw {}, 0({})\n", destination, pointer)),
        }
        true
    }

    pub(super) fn emit_assigned_store(&mut self, ptr: ValueId, value: ValueId) -> bool {
        let (Some(pointer), Some(source)) = (self.assigned_reg(ptr), self.assigned_reg(value))
        else {
            return false;
        };
        match self.func.value(value).ty {
            Type::Ptr(_) => self
                .body
                .push_str(&format!("  sd {}, 0({})\n", source, pointer)),
            _ => self
                .body
                .push_str(&format!("  sw {}, 0({})\n", source, pointer)),
        }
        true
    }

    pub(super) fn emit_assigned_gep(
        &mut self,
        result: ValueId,
        base: ValueId,
        indices: &[ValueId],
    ) -> bool {
        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };
        if indices.len() != 1 {
            return false;
        }
        let mut current_base = if let Some(base_reg) = self.assigned_reg(base) {
            base_reg
        } else {
            self.load_value_into(base, "a1");
            "a1"
        };
        let mut ty = self.func.value(base).ty.clone();
        if indices.is_empty() {
            self.body
                .push_str(&format!("  mv {}, {}\n", destination, current_base));
            return true;
        }
        for (idx, index) in indices.iter().enumerate() {
            let elem_ty = if indices.len() == 1 {
                pointee(&self.func.value(result).ty).unwrap_or_else(|| gep_elem_type(&ty))
            } else {
                gep_elem_type(&ty)
            };
            let stride = ir_size(&elem_ty).max(1);
            if let Some(offset) = const_gep_offset(self.func, *index, stride) {
                self.emit_gep_const_add(destination, current_base, offset);
                current_base = destination;
                ty = if idx + 1 == indices.len() {
                    self.func.value(result).ty.clone()
                } else {
                    elem_ty
                };
                continue;
            }
            let index_reg = if let Some(index_reg) = self.assigned_reg(*index) {
                index_reg
            } else {
                self.load_value(*index);
                "a0"
            };
            if current_base == destination && stride != 1 {
                self.body.push_str(&format!("  mv a1, {}\n", current_base));
                current_base = "a1";
            }
            if stride == 1 {
                self.body.push_str(&format!(
                    "  add {}, {}, {}\n",
                    destination, current_base, index_reg
                ));
            } else if (stride & (stride - 1)) == 0 {
                self.body.push_str(&format!(
                    "  slli {}, {}, {}\n  add {}, {}, {}\n",
                    destination,
                    index_reg,
                    stride.trailing_zeros(),
                    destination,
                    current_base,
                    destination
                ));
            } else {
                self.body.push_str(&format!(
                    "  li t0, {}\n  mul {}, {}, t0\n  add {}, {}, {}\n",
                    stride, destination, index_reg, destination, current_base, destination
                ));
            }
            current_base = destination;
            ty = if idx + 1 == indices.len() {
                self.func.value(result).ty.clone()
            } else {
                elem_ty
            };
        }
        true
    }

    pub(super) fn emit_gep(&mut self, result: ValueId, base: ValueId, indices: &[ValueId]) {
        self.load_value_into(base, "a1");
        let mut ty = self.func.value(base).ty.clone();
        for (idx, index) in indices.iter().enumerate() {
            let elem_ty = if indices.len() == 1 {
                pointee(&self.func.value(result).ty).unwrap_or_else(|| gep_elem_type(&ty))
            } else {
                gep_elem_type(&ty)
            };
            let stride = ir_size(&elem_ty).max(1);
            if let Some(offset) = const_gep_offset(self.func, *index, stride) {
                self.emit_gep_const_add("a1", "a1", offset);
                ty = if idx + 1 == indices.len() {
                    self.func.value(result).ty.clone()
                } else {
                    elem_ty
                };
                continue;
            }
            self.load_value(*index);
            if stride != 1 {
                if stride > 0 && (stride & (stride - 1)) == 0 {
                    self.body
                        .push_str(&format!("  slli a0, a0, {}\n", stride.trailing_zeros()));
                } else {
                    self.body
                        .push_str(&format!("  li t0, {}\n  mul a0, a0, t0\n", stride));
                }
            }
            self.body.push_str("  add a1, a1, a0\n");
            ty = if idx + 1 == indices.len() {
                self.func.value(result).ty.clone()
            } else {
                elem_ty
            };
        }
        self.body.push_str("  mv a0, a1\n");
    }

    fn emit_gep_const_add(&mut self, destination: &str, base: &str, offset: i64) {
        if offset == 0 {
            self.body
                .push_str(&format!("  mv {}, {}\n", destination, base));
        } else if i32::try_from(offset).is_ok_and(fits_i12) {
            self.body
                .push_str(&format!("  addi {}, {}, {}\n", destination, base, offset));
        } else {
            self.body.push_str(&format!(
                "  li t0, {}\n  add {}, {}, t0\n",
                offset, destination, base
            ));
        }
    }

    pub(super) fn load_value(&mut self, value: ValueId) {
        self.load_value_into(value, "a0");
    }

    pub(super) fn load_value_into(&mut self, value: ValueId, destination: &str) {
        debug_assert!(matches!(destination, "a0" | "a1"));

        if let Some(reg) = self.regalloc.reg(value) {
            self.body
                .push_str(&format!("  mv {}, {}\n", destination, reg));
            return;
        }
        if let Some(reg) = self.local_regs.reg(value) {
            self.body
                .push_str(&format!("  mv {}, {}\n", destination, reg));
            return;
        }

        match &self.func.value(value).kind {
            ValueKind::Const(value) => self.load_const_into(value, destination),
            ValueKind::Global(name) => self
                .body
                .push_str(&format!("  la {}, {}\n", destination, name)),
            _ => {
                let offset = self.layout.offset(value);
                match self.func.value(value).ty {
                    Type::Ptr(_) => self.load_frame_x(destination, offset),
                    _ => self.load_frame_w(destination, offset),
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
                self.body
                    .push_str(&format!("  la t0, {}\n  flw {}, 0(t0)\n", label, reg));
            }
            _ => {
                self.load_value(value);
                self.body.push_str(&format!("  fmv.w.x {}, a0\n", reg));
            }
        }
    }

    fn load_const_into(&mut self, value: &Const, destination: &str) {
        let value = match value {
            Const::Int(value) => *value,
            Const::Bool(value) => *value as i32,
            Const::Float(bits) => *bits as i32,
            Const::Zero(_) | Const::String(_) | Const::Array(_) => 0,
        };
        self.body
            .push_str(&format!("  li {}, {}\n", destination, value));
    }

    pub(super) fn load_indirect(&mut self, ty: &Type) {
        match ty {
            Type::Ptr(_) => self.body.push_str("  ld a0, 0(a0)\n"),
            _ => self.body.push_str("  lw a0, 0(a0)\n"),
        }
    }

    pub(super) fn store_indirect(&mut self, ty: &Type) {
        match ty {
            Type::Ptr(_) => self.body.push_str("  sd a0, 0(a1)\n"),
            _ => self.body.push_str("  sw a0, 0(a1)\n"),
        }
    }

    pub(super) fn store_result(&mut self, value: ValueId) {
        if let Some(reg) = self.regalloc.reg(value) {
            self.body.push_str(&format!("  mv {}, a0\n", reg));
            return;
        }
        if let Some(reg) = self.local_regs.reg(value) {
            self.body.push_str(&format!("  mv {}, a0\n", reg));
            return;
        }

        let offset = self.layout.offset(value);
        match self.func.value(value).ty {
            Type::Ptr(_) => self.store_frame_x("a0", offset),
            _ => self.store_frame_w("a0", offset),
        }
    }

    pub(super) fn push_x0(&mut self) {
        self.body.push_str("  addi sp, sp, -16\n  sd a0, 0(sp)\n");
    }

    pub(super) fn pop_x1(&mut self) {
        self.body.push_str("  ld a1, 0(sp)\n  addi sp, sp, 16\n");
    }

    pub(super) fn push_s0(&mut self) {
        self.body
            .push_str("  addi sp, sp, -16\n  sd zero, 0(sp)\n  fsw fa0, 0(sp)\n");
    }

    pub(super) fn adjust_sp(&mut self, amount: i32) {
        if amount == 0 {
            return;
        }
        if fits_i12(amount) {
            self.body.push_str(&format!("  addi sp, sp, {}\n", amount));
        } else {
            self.body.push_str(&format!("  li t0, {}\n", amount));
            self.body.push_str("  add sp, sp, t0\n");
        }
    }

    pub(super) fn frame_addr(&mut self, dst: &str, offset: i32) {
        self.base_addr(dst, "s0", self.frame_slot_offset(offset));
    }

    fn base_addr(&mut self, dst: &str, base: &str, offset: i32) {
        if offset == 0 {
            self.body.push_str(&format!("  mv {}, {}\n", dst, base));
        } else if fits_i12(offset) {
            self.body
                .push_str(&format!("  addi {}, {}, {}\n", dst, base, offset));
        } else {
            self.body.push_str(&format!("  li t0, {}\n", offset));
            self.body
                .push_str(&format!("  add {}, {}, t0\n", dst, base));
        }
    }

    pub(super) fn load_frame_x(&mut self, dst: &str, offset: i32) {
        self.load_base_x(dst, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn load_frame_w(&mut self, dst: &str, offset: i32) {
        self.load_base_w(dst, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn load_raw_frame_x(&mut self, dst: &str, offset: i32) {
        self.load_base_x(dst, "s0", offset);
    }

    pub(super) fn load_raw_frame_w(&mut self, dst: &str, offset: i32) {
        self.load_base_w(dst, "s0", offset);
    }

    pub(super) fn load_raw_frame_s(&mut self, dst: &str, offset: i32) {
        self.load_base_s(dst, "s0", offset);
    }

    pub(super) fn store_frame_x(&mut self, src: &str, offset: i32) {
        self.store_base_x(src, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn store_frame_w(&mut self, src: &str, offset: i32) {
        self.store_base_w(src, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn store_frame_s(&mut self, src: &str, offset: i32) {
        self.store_base_s(src, "s0", self.frame_slot_offset(offset));
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
        if fits_i12(offset) {
            self.body
                .push_str(&format!("  ld {}, {}({})\n", dst, offset, base));
        } else {
            self.base_addr("t1", base, offset);
            self.body.push_str(&format!("  ld {}, 0(t1)\n", dst));
        }
    }

    fn load_base_w(&mut self, dst: &str, base: &str, offset: i32) {
        if fits_i12(offset) {
            self.body
                .push_str(&format!("  lw {}, {}({})\n", dst, offset, base));
        } else {
            self.base_addr("t1", base, offset);
            self.body.push_str(&format!("  lw {}, 0(t1)\n", dst));
        }
    }

    fn load_base_s(&mut self, dst: &str, base: &str, offset: i32) {
        if fits_i12(offset) {
            self.body
                .push_str(&format!("  flw {}, {}({})\n", dst, offset, base));
        } else {
            self.base_addr("t1", base, offset);
            self.body.push_str(&format!("  flw {}, 0(t1)\n", dst));
        }
    }

    fn store_base_x(&mut self, src: &str, base: &str, offset: i32) {
        if fits_i12(offset) {
            self.body
                .push_str(&format!("  sd {}, {}({})\n", src, offset, base));
        } else {
            self.base_addr("t1", base, offset);
            self.body.push_str(&format!("  sd {}, 0(t1)\n", src));
        }
    }

    fn store_base_w(&mut self, src: &str, base: &str, offset: i32) {
        if fits_i12(offset) {
            self.body
                .push_str(&format!("  sw {}, {}({})\n", src, offset, base));
        } else {
            self.base_addr("t1", base, offset);
            self.body.push_str(&format!("  sw {}, 0(t1)\n", src));
        }
    }

    fn store_base_s(&mut self, src: &str, base: &str, offset: i32) {
        if fits_i12(offset) {
            self.body
                .push_str(&format!("  fsw {}, {}({})\n", src, offset, base));
        } else {
            self.base_addr("t1", base, offset);
            self.body.push_str(&format!("  fsw {}, 0(t1)\n", src));
        }
    }

    pub(super) fn block_label(&self, block_idx: usize) -> String {
        format!(".L_{}_bb{}", self.func.name, block_idx)
    }

    pub(super) fn object_offset(&self, value: ValueId, _ty: &Type) -> i32 {
        self.layout.offset(value) + 8
    }

    fn frame_slot_offset(&self, offset: i32) -> i32 {
        offset - self.regalloc.saved_area_size()
    }
}

fn const_gep_offset(func: &crate::ir::Function, index: ValueId, stride: i32) -> Option<i64> {
    let index = match &func.value(index).kind {
        ValueKind::Const(Const::Int(value)) => *value,
        ValueKind::Const(Const::Bool(value)) => *value as i32,
        _ => return None,
    };
    Some(i64::from(index) * i64::from(stride))
}

fn fits_i12(value: i32) -> bool {
    (-2048..=2047).contains(&value)
}
