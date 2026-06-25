use crate::ir::{Function, InstKind, Type, ValueId, ValueKind};
use std::collections::HashMap;

pub(crate) struct IrFuncLayout {
    offsets: HashMap<ValueId, i32>,
    pub(crate) stack_size: i32,
}

impl IrFuncLayout {
    pub(crate) fn new(func: &Function) -> Self {
        let mut layout = Self {
            offsets: HashMap::new(),
            stack_size: 0,
        };
        for (idx, value) in func.values.iter().enumerate() {
            let size = match &value.kind {
                ValueKind::Inst(block, inst_idx) => {
                    let inst = &func.block(*block).insts[*inst_idx];
                    match &inst.kind {
                        InstKind::Nop => 0,
                        InstKind::Alloca { ty } => 8 + ir_align_to(ir_size(ty), 8),
                        _ => ir_slot_size(&value.ty),
                    }
                }
                _ => ir_slot_size(&value.ty),
            };
            if size != 0 {
                layout.alloc(ValueId(idx), size);
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

fn ir_slot_size(ty: &Type) -> i32 {
    match ty {
        Type::Void => 0,
        Type::Array { .. } => ir_size(ty),
        Type::Ptr(_) => 8,
        _ => 4,
    }
}
