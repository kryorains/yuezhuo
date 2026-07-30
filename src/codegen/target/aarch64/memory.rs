use super::imm::{mov_w_imm, mov_x_imm};
use super::AArch64IrFuncEmitter;
use crate::codegen::common::{gep_elem_type, ir_size, pointee};
use crate::ir::{Const, Function, InstKind, Type, ValueId, ValueKind};
use std::collections::HashMap;

const MAX_FOLDED_MEMORY_BLOCKS: usize = 1024;
const MAX_FOLDED_MEMORY_VALUES: usize = 8192;
const MAX_FOLDED_MEMORY_INSTRUCTIONS: usize = 32_768;
const MAX_FOLDED_MEMORY_USES: usize = 65_536;
const MAX_FOLDED_MEMORY_GEPS: usize = 1024;
const MAX_FOLDED_MEMORY_TYPE_NODES: usize = 128;
const MAX_FOLDED_MEMORY_CLONE_TYPE_NODES: usize = 65_536;

#[derive(Clone, Copy)]
pub(super) struct FoldedMemoryGep {
    pub(super) base: ValueId,
    offset: i32,
}

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

    pub(super) fn assigned_x_reg(&self, value: ValueId) -> Option<&'static str> {
        self.phi_regs
            .reg(value)
            .or_else(|| self.local_regs.reg(value))
    }

    pub(super) fn assigned_w_reg(&self, value: ValueId) -> Option<String> {
        self.assigned_x_reg(value).map(w_reg)
    }

    pub(super) fn assigned_float_reg(&self, value: ValueId) -> Option<&'static str> {
        self.float_regs.reg(value)
    }

    pub(super) fn emit_assigned_load(&mut self, result: ValueId, ptr: ValueId) -> bool {
        let (base, offset) = self.memory_address(ptr);
        if self.func.value(result).ty == Type::F32 {
            let Some(destination) = self.assigned_float_reg(result) else {
                return false;
            };
            let pointer = if let Some(pointer) = self.assigned_x_reg(base) {
                pointer
            } else {
                self.load_value_into(base, "x1");
                "x1"
            };
            if offset == 0 {
                self.body
                    .push_str(&format!("  ldr {}, [{}]\n", destination, pointer));
            } else {
                self.load_base_s(destination, pointer, offset);
            }
            return true;
        }
        let Some(destination) = self.assigned_x_reg(result) else {
            return false;
        };
        let pointer = if let Some(pointer) = self.assigned_x_reg(base) {
            pointer
        } else {
            self.load_value_into(base, "x1");
            "x1"
        };
        match self.func.value(result).ty {
            Type::Ptr(_) if offset == 0 => self
                .body
                .push_str(&format!("  ldr {}, [{}]\n", destination, pointer)),
            Type::Ptr(_) => self.load_base_x(destination, pointer, offset),
            _ if offset == 0 => {
                self.body
                    .push_str(&format!("  ldr {}, [{}]\n", w_reg(destination), pointer))
            }
            _ => self.load_base_w(&w_reg(destination), pointer, offset),
        }
        true
    }

    pub(super) fn emit_assigned_store(&mut self, ptr: ValueId, value: ValueId) -> bool {
        let (base, offset) = self.memory_address(ptr);
        if self.func.value(value).ty == Type::F32 {
            let Some(source) = self.assigned_float_reg(value) else {
                return false;
            };
            let pointer = if let Some(pointer) = self.assigned_x_reg(base) {
                pointer
            } else {
                self.load_value_into(base, "x1");
                "x1"
            };
            if offset == 0 {
                self.body
                    .push_str(&format!("  str {}, [{}]\n", source, pointer));
            } else {
                self.store_base_s(source, pointer, offset);
            }
            return true;
        }
        let Some(source) = self.assigned_x_reg(value) else {
            return false;
        };
        let pointer = if let Some(pointer) = self.assigned_x_reg(base) {
            pointer
        } else {
            self.load_value_into(base, "x1");
            "x1"
        };
        match self.func.value(value).ty {
            Type::Ptr(_) if offset == 0 => self
                .body
                .push_str(&format!("  str {}, [{}]\n", source, pointer)),
            Type::Ptr(_) => self.store_base_x(source, pointer, offset),
            _ if offset == 0 => {
                self.body
                    .push_str(&format!("  str {}, [{}]\n", w_reg(source), pointer))
            }
            _ => self.store_base_w(&w_reg(source), pointer, offset),
        }
        true
    }

    pub(super) fn memory_address(&self, ptr: ValueId) -> (ValueId, i32) {
        self.folded_memory_geps
            .get(&ptr)
            .map_or((ptr, 0), |address| (address.base, address.offset))
    }

    pub(super) fn skips_folded_memory_gep(&self, result: ValueId) -> bool {
        self.folded_memory_geps.contains_key(&result)
    }

    pub(super) fn emit_assigned_gep(
        &mut self,
        result: ValueId,
        base: ValueId,
        indices: &[ValueId],
    ) -> bool {
        let Some(destination) = self.assigned_x_reg(result) else {
            return false;
        };
        if indices.len() != 1 {
            return false;
        }
        let mut current_base = if let Some(base_reg) = self.assigned_x_reg(base) {
            base_reg.to_string()
        } else {
            self.load_value_into(base, "x1");
            "x1".to_string()
        };
        let mut ty = self.func.value(base).ty.clone();
        for (idx, index) in indices.iter().enumerate() {
            let elem_ty = if indices.len() == 1 {
                pointee(&self.func.value(result).ty).unwrap_or_else(|| gep_elem_type(&ty))
            } else {
                gep_elem_type(&ty)
            };
            let stride = ir_size(&elem_ty).max(1);
            if let Some(offset) = const_gep_offset(self.func, *index, stride) {
                self.emit_gep_const_add(destination, &current_base, offset);
                current_base = destination.to_string();
                ty = if idx + 1 == indices.len() {
                    self.func.value(result).ty.clone()
                } else {
                    elem_ty
                };
                continue;
            }
            let index_reg = if let Some(index_reg) = self.assigned_w_reg(*index) {
                index_reg
            } else {
                self.load_value(*index);
                "w0".to_string()
            };
            if (stride & (stride - 1)) == 0 {
                let shift = stride.trailing_zeros();
                if shift <= 4 {
                    self.body.push_str(&format!(
                        "  add {}, {}, {}, sxtw #{}\n",
                        destination, current_base, index_reg, shift
                    ));
                } else {
                    self.body.push_str(&format!(
                        "  sxtw x0, {}\n  add {}, {}, x0, lsl #{}\n",
                        index_reg, destination, current_base, shift
                    ));
                }
            } else {
                self.body.push_str(&format!("  sxtw x0, {}\n", index_reg));
                if stride != 1 {
                    self.body.push_str(&mov_x_imm("x16", stride as i64));
                    self.body.push_str("  mul x0, x0, x16\n");
                }
                self.body
                    .push_str(&format!("  add {}, {}, x0\n", destination, current_base));
            }
            current_base = destination.to_string();
            ty = if idx + 1 == indices.len() {
                self.func.value(result).ty.clone()
            } else {
                elem_ty
            };
        }
        true
    }

    pub(super) fn emit_gep(&mut self, result: ValueId, base: ValueId, indices: &[ValueId]) {
        self.load_value_into(base, "x1");
        let mut ty = self.func.value(base).ty.clone();
        for (idx, index) in indices.iter().enumerate() {
            let elem_ty = if indices.len() == 1 {
                pointee(&self.func.value(result).ty).unwrap_or_else(|| gep_elem_type(&ty))
            } else {
                gep_elem_type(&ty)
            };
            let stride = ir_size(&elem_ty).max(1);
            if let Some(offset) = const_gep_offset(self.func, *index, stride) {
                self.emit_gep_const_add("x1", "x1", offset);
                ty = if idx + 1 == indices.len() {
                    self.func.value(result).ty.clone()
                } else {
                    elem_ty
                };
                continue;
            }
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
            self.body.push_str("  add x1, x1, x0\n");
            ty = if idx + 1 == indices.len() {
                self.func.value(result).ty.clone()
            } else {
                elem_ty
            };
        }
        self.body.push_str("  mov x0, x1\n");
    }

    fn emit_gep_const_add(&mut self, destination: &str, base: &str, offset: i64) {
        if offset == 0 {
            self.body
                .push_str(&format!("  mov {}, {}\n", destination, base));
            return;
        }
        let (op, magnitude) = if offset > 0 {
            ("add", offset as u64)
        } else {
            ("sub", (-offset) as u64)
        };
        if let Some((immediate, shifted)) = aarch64_addsub_immediate(magnitude) {
            let shift = if shifted { ", lsl #12" } else { "" };
            self.body.push_str(&format!(
                "  {} {}, {}, #{}{}\n",
                op, destination, base, immediate, shift
            ));
        } else {
            self.body.push_str(&mov_x_imm("x16", magnitude as i64));
            self.body
                .push_str(&format!("  {} {}, {}, x16\n", op, destination, base));
        }
    }

    pub(super) fn load_value(&mut self, value: ValueId) {
        self.load_value_into(value, "x0");
    }

    pub(super) fn load_value_into(&mut self, value: ValueId, x_reg: &str) {
        assert!(matches!(x_reg, "x0" | "x1" | "x2"));
        let is_pointer = matches!(self.func.value(value).ty, Type::Ptr(_));
        let value_reg = if is_pointer {
            x_reg.to_string()
        } else {
            w_reg(x_reg)
        };

        if let Some(float_reg) = self.assigned_float_reg(value) {
            self.body
                .push_str(&format!("  fmov {}, {}\n", value_reg, float_reg));
            return;
        }
        if let Some(phi_reg) = self.phi_regs.reg(value) {
            let phi_reg = if is_pointer {
                phi_reg.to_string()
            } else {
                w_reg(phi_reg)
            };
            self.body
                .push_str(&format!("  mov {}, {}\n", value_reg, phi_reg));
            return;
        }
        if let Some(local_reg) = self.local_regs.reg(value) {
            let local_reg = if is_pointer {
                local_reg.to_string()
            } else {
                w_reg(local_reg)
            };
            self.body
                .push_str(&format!("  mov {}, {}\n", value_reg, local_reg));
            return;
        }

        match &self.func.value(value).kind {
            ValueKind::Const(value) => self.load_const(value, &value_reg),
            ValueKind::Global(name) => self.body.push_str(&format!(
                "  adrp {0}, {1}\n  add {0}, {0}, :lo12:{1}\n",
                x_reg, name
            )),
            _ => {
                let offset = self.layout.offset(value);
                if is_pointer {
                    self.load_frame_x(x_reg, offset);
                } else {
                    self.load_frame_w(&value_reg, offset);
                }
            }
        }
    }

    pub(super) fn load_float_value(&mut self, value: ValueId, reg: &str) {
        if let Some(source) = self.assigned_float_reg(value) {
            if source != reg {
                self.body.push_str(&format!("  fmov {}, {}\n", reg, source));
            }
            return;
        }
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

    pub(super) fn store_float_result(&mut self, value: ValueId, source: &str) {
        if let Some(destination) = self.assigned_float_reg(value) {
            if destination != source {
                self.body
                    .push_str(&format!("  fmov {}, {}\n", destination, source));
            }
        } else {
            self.store_frame_s(source, self.layout.offset(value));
        }
    }

    fn load_const(&mut self, value: &Const, reg: &str) {
        match value {
            Const::Int(value) => self.body.push_str(&mov_w_imm(reg, *value)),
            Const::Bool(value) => self.body.push_str(&mov_w_imm(reg, *value as i32)),
            Const::Float(bits) => self.body.push_str(&mov_w_imm(reg, *bits as i32)),
            Const::Zero(_) | Const::String(_) | Const::Array(_) => {
                let zero_reg = if reg.starts_with('x') { "xzr" } else { "wzr" };
                self.body
                    .push_str(&format!("  mov {}, {}\n", reg, zero_reg));
            }
        }
    }

    pub(super) fn load_indirect_offset(&mut self, ty: &Type, offset: i32) {
        match ty {
            Type::Ptr(_) if offset == 0 => self.body.push_str("  ldr x0, [x0]\n"),
            Type::Ptr(_) => self.load_base_x("x0", "x0", offset),
            _ if offset == 0 => self.body.push_str("  ldr w0, [x0]\n"),
            _ => self.load_base_w("w0", "x0", offset),
        }
    }

    pub(super) fn store_indirect_offset(&mut self, ty: &Type, offset: i32) {
        match ty {
            Type::Ptr(_) if offset == 0 => self.body.push_str("  str x0, [x1]\n"),
            Type::Ptr(_) => self.store_base_x("x0", "x1", offset),
            _ if offset == 0 => self.body.push_str("  str w0, [x1]\n"),
            _ => self.store_base_w("w0", "x1", offset),
        }
    }

    pub(super) fn store_result(&mut self, value: ValueId) {
        if let Some(reg) = self.assigned_float_reg(value) {
            self.body.push_str(&format!("  fmov {}, w0\n", reg));
            return;
        }
        if let Some(reg) = self.phi_regs.reg(value) {
            match self.func.value(value).ty {
                Type::Ptr(_) => self.body.push_str(&format!("  mov {}, x0\n", reg)),
                _ => self.body.push_str(&format!("  mov {}, w0\n", w_reg(reg))),
            }
            return;
        }
        if let Some(reg) = self.local_regs.reg(value) {
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
        self.frame_accessed = true;
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
        self.frame_accessed = true;
        self.load_base_x(dst, "x29", offset);
    }

    pub(super) fn load_frame_w(&mut self, dst: &str, offset: i32) {
        self.frame_accessed = true;
        self.load_base_w(dst, "x29", offset);
    }

    pub(super) fn load_frame_s(&mut self, dst: &str, offset: i32) {
        self.frame_accessed = true;
        self.load_base_s(dst, "x29", offset);
    }

    pub(super) fn store_frame_x(&mut self, src: &str, offset: i32) {
        self.frame_accessed = true;
        self.store_base_x(src, "x29", offset);
    }

    pub(super) fn store_frame_w(&mut self, src: &str, offset: i32) {
        self.frame_accessed = true;
        self.store_base_w(src, "x29", offset);
    }

    pub(super) fn store_frame_s(&mut self, src: &str, offset: i32) {
        self.frame_accessed = true;
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

/// Selects constant single-index GEPs whose complete use set consists of
/// correctly typed loads/stores and whose byte offset fits an AArch64 memory
/// operand. Selected GEPs can be omitted from both allocation and emission.
pub(super) fn collect_folded_memory_geps(
    func: &Function,
    value_use_counts: &[usize],
) -> HashMap<ValueId, FoldedMemoryGep> {
    let instruction_count = func
        .blocks
        .iter()
        .try_fold(0usize, |total, block| total.checked_add(block.insts.len()));
    let use_count = value_use_counts
        .iter()
        .try_fold(0usize, |total, count| total.checked_add(*count));
    if value_use_counts.len() != func.values.len()
        || !function_types_fit_clone_budget(func)
        || func.blocks.len() > MAX_FOLDED_MEMORY_BLOCKS
        || func.values.len() > MAX_FOLDED_MEMORY_VALUES
        || instruction_count.is_none_or(|count| count > MAX_FOLDED_MEMORY_INSTRUCTIONS)
        || use_count.is_none_or(|count| count > MAX_FOLDED_MEMORY_USES)
    {
        return HashMap::new();
    }

    let mut memory_pointer_uses = vec![0usize; func.values.len()];
    let mut exact_pointer_types = vec![true; func.values.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            let (ptr, access_ty) = match &inst.kind {
                InstKind::Load { ptr } => {
                    let Some(result) = inst.result else {
                        if let Some(exact) = exact_pointer_types.get_mut(ptr.0) {
                            *exact = false;
                        }
                        continue;
                    };
                    (*ptr, func.values.get(result.0).map(|value| &value.ty))
                }
                InstKind::Store { ptr, value } => {
                    (*ptr, func.values.get(value.0).map(|value| &value.ty))
                }
                _ => continue,
            };
            let Some(count) = memory_pointer_uses.get_mut(ptr.0) else {
                continue;
            };
            *count = count.saturating_add(1);
            let exact = match (func.values.get(ptr.0), access_ty) {
                (Some(pointer), Some(access_ty)) => match &pointer.ty {
                    Type::Ptr(pointee) => types_equal_bounded(pointee, access_ty),
                    _ => false,
                },
                _ => false,
            };
            if !exact {
                exact_pointer_types[ptr.0] = false;
            }
        }
    }

    let mut folded = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let (Some(result), InstKind::Gep { base, indices }) = (inst.result, &inst.kind) else {
                continue;
            };
            let [index] = indices.as_slice() else {
                continue;
            };
            let Some(total_uses) = value_use_counts.get(result.0).copied() else {
                continue;
            };
            if total_uses == 0
                || memory_pointer_uses.get(result.0).copied() != Some(total_uses)
                || exact_pointer_types.get(result.0) != Some(&true)
            {
                continue;
            }
            let Some(stride) = strict_single_index_stride(func, *base, result) else {
                continue;
            };
            let Some(offset) = const_gep_offset(func, *index, stride)
                .and_then(|offset| i32::try_from(offset).ok())
                .filter(|offset| *offset != 0 && direct_mem_op(*offset, stride).is_some())
            else {
                continue;
            };
            if folded.len() == MAX_FOLDED_MEMORY_GEPS {
                return HashMap::new();
            }
            folded.insert(
                result,
                FoldedMemoryGep {
                    base: *base,
                    offset,
                },
            );
        }
    }
    folded
}

pub(super) fn rewrite_folded_memory_uses_for_allocation(
    func: &mut Function,
    folded: &HashMap<ValueId, FoldedMemoryGep>,
) {
    for block in &mut func.blocks {
        for inst in &mut block.insts {
            match &mut inst.kind {
                InstKind::Load { ptr } | InstKind::Store { ptr, .. } => {
                    if let Some(address) = folded.get(ptr) {
                        *ptr = address.base;
                    }
                }
                _ => {}
            }
            if inst
                .result
                .is_some_and(|result| folded.contains_key(&result))
            {
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
        }
    }
}

fn function_types_fit_clone_budget(func: &Function) -> bool {
    let mut roots = func
        .values
        .iter()
        .map(|value| &value.ty)
        .collect::<Vec<_>>();
    roots.extend(func.blocks.iter().flat_map(|block| {
        block.insts.iter().filter_map(|inst| match &inst.kind {
            InstKind::Alloca { ty } => Some(ty),
            _ => None,
        })
    }));
    let mut total = 0usize;
    for root in roots {
        let mut worklist = vec![(root, 1usize)];
        while let Some((ty, depth)) = worklist.pop() {
            total = match total.checked_add(1) {
                Some(total) if total <= MAX_FOLDED_MEMORY_CLONE_TYPE_NODES => total,
                _ => return false,
            };
            if depth > MAX_FOLDED_MEMORY_TYPE_NODES {
                return false;
            }
            match ty {
                Type::Ptr(inner) => worklist.push((inner, depth + 1)),
                Type::Array { elem, .. } => worklist.push((elem, depth + 1)),
                Type::Void | Type::I1 | Type::I32 | Type::F32 => {}
            }
        }
    }
    true
}

fn strict_single_index_stride(func: &Function, base: ValueId, result: ValueId) -> Option<i32> {
    let Type::Ptr(base_pointee) = &func.values.get(base.0)?.ty else {
        return None;
    };
    let Type::Ptr(result_pointee) = &func.values.get(result.0)?.ty else {
        return None;
    };
    let compatible = types_equal_bounded(base_pointee, result_pointee)
        || matches!(
            base_pointee.as_ref(),
            Type::Array { elem, .. } if types_equal_bounded(elem, result_pointee)
        );
    compatible
        .then(|| checked_type_size(result_pointee))
        .flatten()
        .filter(|size| *size > 0)
}

fn checked_type_size(ty: &Type) -> Option<i32> {
    let mut current = ty;
    let mut elements = 1i32;
    for _ in 0..MAX_FOLDED_MEMORY_TYPE_NODES {
        match current {
            Type::Void => return Some(0),
            Type::I1 | Type::I32 | Type::F32 => return elements.checked_mul(4),
            Type::Ptr(_) => return elements.checked_mul(8),
            Type::Array { elem, len } => {
                elements = elements.checked_mul(i32::try_from(*len).ok()?)?;
                current = elem;
            }
        }
    }
    None
}

fn types_equal_bounded(mut lhs: &Type, mut rhs: &Type) -> bool {
    for _ in 0..MAX_FOLDED_MEMORY_TYPE_NODES {
        match (lhs, rhs) {
            (Type::Void, Type::Void)
            | (Type::I1, Type::I1)
            | (Type::I32, Type::I32)
            | (Type::F32, Type::F32) => return true,
            (Type::Ptr(lhs_inner), Type::Ptr(rhs_inner)) => {
                lhs = lhs_inner;
                rhs = rhs_inner;
            }
            (
                Type::Array {
                    elem: lhs_elem,
                    len: lhs_len,
                },
                Type::Array {
                    elem: rhs_elem,
                    len: rhs_len,
                },
            ) if lhs_len == rhs_len => {
                lhs = lhs_elem;
                rhs = rhs_elem;
            }
            _ => return false,
        }
    }
    false
}

fn w_reg(x_reg: &str) -> String {
    x_reg.replacen('x', "w", 1)
}

fn const_gep_offset(func: &crate::ir::Function, index: ValueId, stride: i32) -> Option<i64> {
    let index = match &func.value(index).kind {
        ValueKind::Const(Const::Int(value)) => *value,
        ValueKind::Const(Const::Bool(value)) => *value as i32,
        _ => return None,
    };
    Some(i64::from(index) * i64::from(stride))
}

// Keep the modulo form to support Rust toolchains predating u64::is_multiple_of in 1.87.
#[allow(clippy::manual_is_multiple_of)]
fn aarch64_addsub_immediate(magnitude: u64) -> Option<(u64, bool)> {
    const MAX_IMMEDIATE: u64 = (1 << 12) - 1;
    const SHIFTED_UNIT: u64 = 1 << 12;

    if magnitude <= MAX_IMMEDIATE {
        Some((magnitude, false))
    } else if magnitude % SHIFTED_UNIT == 0 && magnitude / SHIFTED_UNIT <= MAX_IMMEDIATE {
        Some((magnitude / SHIFTED_UNIT, true))
    } else {
        None
    }
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
