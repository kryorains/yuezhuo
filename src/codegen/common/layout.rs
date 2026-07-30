use crate::ir::{Function, InstKind, Type, ValueId, ValueKind};
use std::collections::HashMap;

/// 函数内所有“需要落栈的 IR 值”到栈帧偏移的映射。
///
/// 约定：offset 是相对目标平台 frame pointer 的负偏移；`alloca` 结果本身保存的是
/// 被分配对象的地址，所以会额外预留一个 8 字节槽位来存这个地址，真实对象从
/// `object_offset = value_offset + 8` 开始。
pub(crate) struct IrFuncLayout {
    offsets: HashMap<ValueId, i32>,
    pub(crate) stack_size: i32,
}

impl IrFuncLayout {
    pub(crate) fn new(func: &Function) -> Self {
        Self::new_with_stack_slots(func, |_| true)
    }

    /// Builds a compact frame for values that really need spill storage.
    ///
    /// `alloca` objects are always retained because the predicate describes the
    /// result pointer, not the separately allocated object that it addresses.
    pub(crate) fn new_with_stack_slots(
        func: &Function,
        mut needs_stack_slot: impl FnMut(ValueId) -> bool,
    ) -> Self {
        let mut layout = Self {
            offsets: HashMap::new(),
            stack_size: 0,
        };
        for (idx, value) in func.values.iter().enumerate() {
            let value_id = ValueId(idx);
            let size = match &value.kind {
                ValueKind::Param if needs_stack_slot(value_id) => ir_slot_size(&value.ty),
                ValueKind::Param => 0,
                ValueKind::Inst(block, inst_idx) => {
                    let inst = &func.block(*block).insts[*inst_idx];
                    match &inst.kind {
                        InstKind::Nop => 0,
                        InstKind::Alloca { ty } => 8 + ir_align_to(ir_size(ty), 8),
                        _ if needs_stack_slot(value_id) => ir_slot_size(&value.ty),
                        _ => 0,
                    }
                }
                ValueKind::Const(_) | ValueKind::Global(_) => 0,
            };
            if size != 0 {
                layout.alloc(value_id, size);
            }
        }
        layout
    }

    pub(crate) fn offset(&self, value: ValueId) -> i32 {
        *self
            .offsets
            .get(&value)
            .unwrap_or_else(|| panic!("missing stack slot for {}", value))
    }

    fn alloc(&mut self, value: ValueId, size: i32) {
        // 当前后端统一按 8 字节对齐分配栈槽，方便同时容纳 i32/f32 和指针。
        self.stack_size += ir_align_to(size, 8);
        self.offsets.insert(value, -self.stack_size);
    }
}

pub(crate) fn ir_size(ty: &Type) -> i32 {
    match ty {
        Type::Void => 0,
        Type::I1 | Type::I32 | Type::F32 => 4,
        Type::Ptr(_) => 8,
        Type::Array { elem, len } => ir_size(elem) * (*len as i32),
    }
}

pub(crate) fn ir_align_to(value: i32, align: i32) -> i32 {
    (value + align - 1) / align * align
}

/// IR 值存回栈帧时需要的槽位大小。
///
/// 数组类型表示“对象本体”，按完整大小预留；指针是地址，固定 8 字节；
/// i1/i32/f32 当前都按 4 字节保存。
fn ir_slot_size(ty: &Type) -> i32 {
    match ty {
        Type::Void => 0,
        Type::Array { .. } => ir_size(ty),
        Type::Ptr(_) => 8,
        _ => 4,
    }
}
