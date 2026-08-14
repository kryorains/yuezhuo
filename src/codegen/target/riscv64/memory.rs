use super::Riscv64IrFuncEmitter;
use crate::codegen::common::{gep_elem_type, ir_size, pointee};
use crate::ir::{Const, Function, InstKind, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const MAX_FOLDED_MEMORY_BLOCKS: usize = 1024;
const MAX_FOLDED_MEMORY_VALUES: usize = 8192;
const MAX_FOLDED_MEMORY_INSTRUCTIONS: usize = 32_768;
const MAX_FOLDED_MEMORY_USES: usize = 65_536;
const MAX_FOLDED_MEMORY_GEPS: usize = 1024;
const MAX_FOLDED_MEMORY_GEP_CHAIN: usize = 64;
const MAX_FOLDED_MEMORY_TYPE_NODES: usize = 128;
const MAX_FOLDED_MEMORY_CLONE_TYPE_NODES: usize = 65_536;

#[derive(Clone, Copy)]
pub(super) struct FoldedMemoryGep {
    pub(super) base: ValueId,
    offset: i32,
}

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_regalloc_materializations(&mut self) {
        let materialized = self.regalloc.materialized().to_vec();
        for (value, reg) in materialized {
            match self.func.value(value).kind.clone() {
                ValueKind::Const(value) => self.load_const_into(&value, reg),
                ValueKind::Global(name) => {
                    self.body.push_str(&format!("  la {reg}, {name}\n"));
                }
                ValueKind::Param | ValueKind::Inst(_, _) => {
                    unreachable!("only rematerializable values are initialized here")
                }
            }
        }
        for (offset, reg) in self.regalloc.address_offsets().to_vec() {
            self.body.push_str(&format!("  li {reg}, {offset}\n"));
        }
        let float_materialized = self.float_regalloc.materialized().to_vec();
        for (value, reg) in float_materialized {
            let bits = match self.func.value(value).kind {
                ValueKind::Const(Const::Float(bits)) => bits,
                ValueKind::Const(Const::Zero(Type::F32)) => 0,
                _ => unreachable!("only floating constants are initialized here"),
            };
            if bits == 0 {
                self.body.push_str(&format!("  fmv.w.x {reg}, zero\n"));
            } else {
                let label = self.parent.ctx.fresh_label("float");
                self.parent.out.push_str(&format!(
                    ".section .rodata\n.align 2\n{}:\n  .word {}\n.text\n",
                    label, bits
                ));
                self.body
                    .push_str(&format!("  la t0, {label}\n  flw {reg}, 0(t0)\n"));
            }
        }
    }

    pub(super) fn emit_memzero(&mut self, ptr: ValueId, bytes: usize) {
        if bytes == 0 {
            return;
        }
        const CHUNK_BYTES: usize = 8;
        let is_aligned_alloca = matches!(
            self.func.value(ptr).kind,
            ValueKind::Inst(block, inst_index)
                if matches!(
                    self.func.blocks[block.0].insts[inst_index].kind,
                    InstKind::Alloca { .. }
                )
        );
        self.load_value(ptr);
        let chunks = bytes / CHUNK_BYTES;
        let tail = bytes % CHUNK_BYTES;
        if chunks != 0 {
            let loop_label = self.parent.ctx.fresh_label("memzero");
            self.body.push_str(&format!("  li a1, {}\n", chunks));
            self.body.push_str(&format!("{}:\n", loop_label));
            if is_aligned_alloca {
                self.body.push_str("  sd zero, 0(a0)\n");
            } else {
                for offset in 0..CHUNK_BYTES {
                    self.body.push_str(&format!("  sb zero, {}(a0)\n", offset));
                }
            }
            self.body.push_str(&format!(
                "  addi a0, a0, {}\n  addi a1, a1, -1\n  bnez a1, {}\n",
                CHUNK_BYTES, loop_label
            ));
        }
        for offset in 0..tail {
            self.body.push_str(&format!("  sb zero, {}(a0)\n", offset));
        }
    }

    pub(super) fn emit_memzero_elements(
        &mut self,
        ptr: ValueId,
        count: ValueId,
        element_bytes: usize,
    ) {
        debug_assert_ne!(element_bytes, 0);
        self.load_value_into(ptr, "a0");
        let count = self.load_or_assigned(count, "a2");
        if element_bytes.is_power_of_two() {
            self.body.push_str(&format!(
                "  slli a2, {}, {}\n",
                count,
                element_bytes.trailing_zeros()
            ));
        } else {
            self.body.push_str(&format!(
                "  li t0, {}\n  mul a2, {}, t0\n",
                element_bytes, count
            ));
        }
        self.body.push_str("  li a1, 0\n  call memset\n");
    }

    pub(super) fn emit_memcopy_elements(
        &mut self,
        dst: ValueId,
        src: ValueId,
        count: ValueId,
        element_bytes: usize,
    ) {
        debug_assert_ne!(element_bytes, 0);
        self.load_value_into(dst, "a0");
        self.load_value_into(src, "a1");
        let count = self.load_or_assigned(count, "a2");
        if element_bytes.is_power_of_two() {
            self.body.push_str(&format!(
                "  slli a2, {}, {}\n",
                count,
                element_bytes.trailing_zeros()
            ));
        } else {
            self.body.push_str(&format!(
                "  li t0, {}\n  mul a2, {}, t0\n",
                element_bytes, count
            ));
        }
        self.body.push_str("  call memcpy\n");
    }

    pub(super) fn assigned_reg(&self, value: ValueId) -> Option<&'static str> {
        self.regalloc
            .reg(value)
            .or_else(|| self.local_regs.reg(value))
    }

    pub(super) fn assigned_float_reg(&self, value: ValueId) -> Option<&'static str> {
        self.float_regalloc.reg(value)
    }

    pub(super) fn emit_assigned_load(&mut self, result: ValueId, ptr: ValueId) -> bool {
        if self.func.value(result).ty == Type::F32 {
            let (base, offset) = self.memory_address(ptr);
            let Some(pointer) = self.assigned_reg(base) else {
                return false;
            };
            let Some(destination) = self.assigned_float_reg(result) else {
                return false;
            };
            self.body
                .push_str(&format!("  flw {}, {}({})\n", destination, offset, pointer));
            return true;
        }
        let Some(destination) = self.assigned_reg(result) else {
            return false;
        };
        let (base, offset) = self.memory_address(ptr);
        let pointer = self.load_or_assigned(base, "a0");
        match self.func.value(result).ty {
            Type::Ptr(_) => self
                .body
                .push_str(&format!("  ld {}, {}({})\n", destination, offset, pointer)),
            _ => self
                .body
                .push_str(&format!("  lw {}, {}({})\n", destination, offset, pointer)),
        }
        true
    }

    pub(super) fn emit_assigned_store(&mut self, ptr: ValueId, value: ValueId) -> bool {
        let (base, offset) = self.memory_address(ptr);
        if self.func.value(value).ty == Type::F32 {
            let Some(pointer) = self.assigned_reg(base) else {
                return false;
            };
            let Some(source) = self.assigned_float_reg(value) else {
                return false;
            };
            self.body
                .push_str(&format!("  fsw {}, {}({})\n", source, offset, pointer));
            return true;
        }
        let pointer = self.assigned_reg(base);
        let source = self.assigned_reg(value).or_else(|| {
            matches!(
                self.func.value(value).kind,
                ValueKind::Const(Const::Int(0) | Const::Bool(false) | Const::Zero(_))
            )
            .then_some("zero")
        });
        if pointer.is_none() && source.is_none() {
            return false;
        }
        let pointer = if let Some(pointer) = pointer {
            pointer
        } else {
            self.load_value_into(base, "a1");
            "a1"
        };
        let source = if let Some(source) = source {
            source
        } else {
            self.load_value_into(value, "a0");
            "a0"
        };
        match self.func.value(value).ty {
            Type::Ptr(_) => self
                .body
                .push_str(&format!("  sd {}, {}({})\n", source, offset, pointer)),
            _ => self
                .body
                .push_str(&format!("  sw {}, {}({})\n", source, offset, pointer)),
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
        } else if let Some(offset_reg) = self.regalloc.address_offset_reg(offset) {
            self.body.push_str(&format!(
                "  add {}, {}, {}\n",
                destination, base, offset_reg
            ));
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
        debug_assert!(matches!(
            destination,
            "a0" | "a1" | "a2" | "a3" | "a4" | "a5" | "a6" | "a7"
        ));

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
        if let Some(source) = self.assigned_float_reg(value) {
            if source != reg {
                self.body
                    .push_str(&format!("  fmv.s {}, {}\n", reg, source));
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
                self.body
                    .push_str(&format!("  la t0, {}\n  flw {}, 0(t0)\n", label, reg));
            }
            _ => {
                self.load_value(value);
                self.body.push_str(&format!("  fmv.w.x {}, a0\n", reg));
            }
        }
    }

    pub(super) fn store_float_result(&mut self, value: ValueId, source: &str) {
        if let Some(destination) = self.assigned_float_reg(value) {
            if destination != source {
                self.body
                    .push_str(&format!("  fmv.s {}, {}\n", destination, source));
            }
            return;
        }
        self.store_frame_s(source, self.layout.offset(value));
    }

    pub(super) fn rematerialize_into(&mut self, value: ValueId, destination: &str) -> bool {
        match &self.func.value(value).kind {
            ValueKind::Const(value) => self.load_const_into(value, destination),
            ValueKind::Global(name) => self
                .body
                .push_str(&format!("  la {}, {}\n", destination, name)),
            ValueKind::Param | ValueKind::Inst(_, _) => return false,
        }
        true
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

    pub(super) fn load_indirect(&mut self, ty: &Type, offset: i32) {
        match ty {
            Type::Ptr(_) => self.body.push_str(&format!("  ld a0, {}(a0)\n", offset)),
            Type::F32 => self.body.push_str(&format!("  flw fa0, {}(a0)\n", offset)),
            _ => self.body.push_str(&format!("  lw a0, {}(a0)\n", offset)),
        }
    }

    pub(super) fn store_indirect(&mut self, ty: &Type, offset: i32) {
        match ty {
            Type::Ptr(_) => self.body.push_str(&format!("  sd a0, {}(a1)\n", offset)),
            Type::F32 => self.body.push_str(&format!("  fsw fa0, {}(a1)\n", offset)),
            _ => self.body.push_str(&format!("  sw a0, {}(a1)\n", offset)),
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
        self.frame_accessed = true;
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
        self.frame_accessed = true;
        self.load_base_x(dst, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn load_frame_w(&mut self, dst: &str, offset: i32) {
        self.frame_accessed = true;
        self.load_base_w(dst, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn load_raw_frame_x(&mut self, dst: &str, offset: i32) {
        self.frame_accessed = true;
        self.load_base_x(dst, "s0", offset);
    }

    pub(super) fn load_raw_frame_w(&mut self, dst: &str, offset: i32) {
        self.frame_accessed = true;
        self.load_base_w(dst, "s0", offset);
    }

    pub(super) fn load_raw_frame_s(&mut self, dst: &str, offset: i32) {
        self.frame_accessed = true;
        self.load_base_s(dst, "s0", offset);
    }

    pub(super) fn store_frame_x(&mut self, src: &str, offset: i32) {
        self.frame_accessed = true;
        self.store_base_x(src, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn store_frame_w(&mut self, src: &str, offset: i32) {
        self.frame_accessed = true;
        self.store_base_w(src, "s0", self.frame_slot_offset(offset));
    }

    pub(super) fn store_frame_s(&mut self, src: &str, offset: i32) {
        self.frame_accessed = true;
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
        offset - self.saved_area_size
    }
}

/// Precomputes the only GEPs that may be omitted during emission. Comparing
/// the complete generic use count with typed load/store pointer uses makes a
/// missed phi, call, address, value, or terminator use disable folding.
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
    let mut memory_pointer_types_are_exact = vec![true; func.values.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            let (ptr, access_ty) = match &inst.kind {
                InstKind::Load { ptr } => {
                    let Some(result) = inst.result else {
                        if let Some(exact) = memory_pointer_types_are_exact.get_mut(ptr.0) {
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
                memory_pointer_types_are_exact[ptr.0] = false;
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
                || memory_pointer_types_are_exact.get(result.0) != Some(&true)
            {
                continue;
            }
            let Some(stride) = strict_single_index_stride(func, *base, result) else {
                continue;
            };
            let Some(offset) = const_gep_offset(func, *index, stride)
                .and_then(|offset| i32::try_from(offset).ok())
                .filter(|offset| fits_i12(*offset))
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
    fold_same_block_affine_memory_geps(
        func,
        value_use_counts,
        &memory_pointer_uses,
        &memory_pointer_types_are_exact,
        &mut folded,
    );
    folded
}

#[derive(PartialEq, Eq)]
struct AffineIndex {
    terms: HashMap<ValueId, i32>,
    constant: i32,
}

struct AffineMemoryGep {
    result: ValueId,
    base: ValueId,
    stride: i64,
    index: AffineIndex,
}

/// Reuses an earlier address in the same straight-line block when two
/// memory-only GEPs have identical affine dynamic terms and differ only by a
/// load/store-encodable constant. Valid source accesses cannot cross the
/// signed i32 boundary without leaving their object, so reassociating that
/// constant into the byte offset preserves every defined access.
fn fold_same_block_affine_memory_geps(
    func: &Function,
    value_use_counts: &[usize],
    memory_pointer_uses: &[usize],
    memory_pointer_types_are_exact: &[bool],
    folded: &mut HashMap<ValueId, FoldedMemoryGep>,
) {
    for block in &func.blocks {
        let mut earlier = Vec::<AffineMemoryGep>::new();
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
                || memory_pointer_types_are_exact.get(result.0) != Some(&true)
            {
                continue;
            }
            let Some(stride) = strict_single_index_stride(func, *base, result) else {
                continue;
            };
            let stride = i64::from(stride);
            let Some(index) = affine_i32_index(func, *index, 0) else {
                continue;
            };
            if !folded.contains_key(&result) {
                let replacement = earlier.iter().rev().find_map(|candidate| {
                    if candidate.base != *base
                        || candidate.stride != stride
                        || candidate.index.terms != index.terms
                        || func.value(candidate.result).ty != func.value(result).ty
                    {
                        return None;
                    }
                    let delta_indices =
                        i64::from(index.constant) - i64::from(candidate.index.constant);
                    let offset = delta_indices
                        .checked_mul(stride)
                        .and_then(|offset| i32::try_from(offset).ok())?;
                    let (base, base_offset) =
                        resolve_folded_memory_address(candidate.result, folded)?;
                    let offset = base_offset
                        .checked_add(offset)
                        .filter(|offset| fits_i12(*offset))?;
                    Some(FoldedMemoryGep { base, offset })
                });
                if let Some(replacement) = replacement {
                    if folded.len() == MAX_FOLDED_MEMORY_GEPS {
                        return;
                    }
                    folded.insert(result, replacement);
                }
            }
            earlier.push(AffineMemoryGep {
                result,
                base: *base,
                stride,
                index,
            });
        }
    }
}

/// Follows already-selected address-mode folds so a later affine access can
/// reuse the same retained root with a combined encodable displacement.
fn resolve_folded_memory_address(
    value: ValueId,
    folded: &HashMap<ValueId, FoldedMemoryGep>,
) -> Option<(ValueId, i32)> {
    let mut base = value;
    let mut offset = 0i32;
    for _ in 0..MAX_FOLDED_MEMORY_GEP_CHAIN {
        let Some(address) = folded.get(&base) else {
            return Some((base, offset));
        };
        offset = offset.checked_add(address.offset)?;
        base = address.base;
    }
    None
}

fn affine_i32_index(func: &Function, value: ValueId, depth: usize) -> Option<AffineIndex> {
    if depth > 16 || func.value(value).ty != Type::I32 {
        return None;
    }
    if let ValueKind::Const(Const::Int(constant)) = func.value(value).kind {
        return Some(AffineIndex {
            terms: HashMap::new(),
            constant,
        });
    }
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return Some(AffineIndex {
            terms: HashMap::from([(value, 1)]),
            constant: 0,
        });
    };
    let inst = func
        .blocks
        .get(block.0)
        .and_then(|block| block.insts.get(inst_idx))?;
    let InstKind::Binary { op, lhs, rhs } = inst.kind else {
        return Some(AffineIndex {
            terms: HashMap::from([(value, 1)]),
            constant: 0,
        });
    };
    if !matches!(op, crate::ir::BinaryOp::Iadd | crate::ir::BinaryOp::Isub) {
        return Some(AffineIndex {
            terms: HashMap::from([(value, 1)]),
            constant: 0,
        });
    }
    let mut lhs = affine_i32_index(func, lhs, depth + 1)?;
    let rhs = affine_i32_index(func, rhs, depth + 1)?;
    let sign = if op == crate::ir::BinaryOp::Iadd {
        1i32
    } else {
        -1i32
    };
    lhs.constant = lhs.constant.wrapping_add(rhs.constant.wrapping_mul(sign));
    for (term, coefficient) in rhs.terms {
        let next = lhs
            .terms
            .get(&term)
            .copied()
            .unwrap_or_default()
            .wrapping_add(coefficient.wrapping_mul(sign));
        if next == 0 {
            lhs.terms.remove(&term);
        } else {
            lhs.terms.insert(term, next);
        }
    }
    Some(lhs)
}

/// Makes register allocation see the same base live ranges that folded memory
/// emission will use. Value IDs and instruction slots stay stable; this is a
/// private allocation view and does not mutate the verified IR.
pub(super) fn rewrite_folded_memory_uses_for_allocation(
    func: &mut Function,
    folded: &HashMap<ValueId, FoldedMemoryGep>,
) {
    if folded.is_empty() {
        return;
    }
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

/// Removes pure address-building instructions that become dead only after
/// folded GEP uses are rewritten to their retained base address. This runs on
/// the private allocation view; the returned stable ValueIds tell emission to
/// skip the corresponding instructions in the original verified IR.
pub(super) fn eliminate_dead_folded_address_values(func: &mut Function) -> HashSet<ValueId> {
    let mut use_counts = crate::codegen::common::ir_value_use_counts(func);
    let mut worklist = Vec::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let Some(result) = inst
                .result
                .filter(|result| use_counts[result.0] == 0 && is_removable(inst))
            {
                worklist.push(result);
            }
        }
    }

    let mut removed = HashSet::new();
    while let Some(result) = worklist.pop() {
        if removed.contains(&result) || use_counts[result.0] != 0 {
            continue;
        }
        let ValueKind::Inst(block, inst_idx) = func.value(result).kind else {
            continue;
        };
        let Some(inst) = func
            .blocks
            .get(block.0)
            .and_then(|block| block.insts.get(inst_idx))
        else {
            continue;
        };
        if inst.result != Some(result) || !is_removable(inst) {
            continue;
        }
        let operands = instruction_operands(&inst.kind);
        let inst = &mut func.blocks[block.0].insts[inst_idx];
        inst.result = None;
        inst.kind = InstKind::Nop;
        removed.insert(result);

        for operand in operands {
            let Some(count) = use_counts.get_mut(operand.0) else {
                continue;
            };
            if *count == 0 {
                continue;
            }
            *count -= 1;
            if *count != 0 {
                continue;
            }
            let ValueKind::Inst(owner, owner_idx) = func.value(operand).kind else {
                continue;
            };
            if func.blocks[owner.0]
                .insts
                .get(owner_idx)
                .is_some_and(is_removable)
            {
                worklist.push(operand);
            }
        }
    }
    removed
}

fn is_removable(inst: &crate::ir::Inst) -> bool {
    matches!(
        inst.kind,
        InstKind::Nop
            | InstKind::Phi { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Unary { .. }
            | InstKind::Binary { .. }
            | InstKind::Icmp { .. }
            | InstKind::Fcmp { .. }
            | InstKind::Cast { .. }
            | InstKind::Gep { .. }
    )
}

fn instruction_operands(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
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
            let mut operands = Vec::with_capacity(indices.len() + 1);
            operands.push(*base);
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Call { args, .. } => args.clone(),
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
            let Some(next_total) = total.checked_add(1) else {
                return false;
            };
            total = next_total;
            if total > MAX_FOLDED_MEMORY_CLONE_TYPE_NODES || depth > MAX_FOLDED_MEMORY_TYPE_NODES {
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
    if !compatible {
        return None;
    }
    checked_type_size(result_pointee).filter(|size| *size > 0)
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

fn const_gep_offset(func: &Function, index: ValueId, stride: i32) -> Option<i64> {
    let value = func.values.get(index.0)?;
    if value.ty != Type::I32 {
        return None;
    }
    let index = match &value.kind {
        ValueKind::Const(Const::Int(value)) => *value,
        ValueKind::Const(Const::Zero(Type::I32)) => 0,
        _ => return None,
    };
    i64::from(index).checked_mul(i64::from(stride))
}

fn fits_i12(value: i32) -> bool {
    (-2048..=2047).contains(&value)
}
