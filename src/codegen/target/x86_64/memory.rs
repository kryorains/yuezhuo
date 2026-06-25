use super::X86IrFuncEmitter;
use crate::codegen::common::ir_size;
use crate::ir::{Const, Type, ValueId, ValueKind};

impl<'a, 'b> X86IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_memzero(&mut self, ptr: ValueId, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.load_value(ptr);
        self.body.push_str(&format!(
            "  movq %rax, %rdi\n  xorl %eax, %eax\n  movq ${}, %rcx\n  rep stosb\n",
            bytes
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
            self.push_rax();
            self.load_value(*index);
            self.body.push_str("  cltq\n");
            if stride != 1 {
                self.body.push_str(&format!("  imulq ${}, %rax\n", stride));
            }
            self.pop_rcx();
            self.body.push_str("  addq %rcx, %rax\n");
            ty = if idx + 1 == indices.len() {
                self.func.value(result).ty.clone()
            } else {
                elem_ty
            };
        }
    }

    pub(super) fn load_value(&mut self, value: ValueId) {
        match &self.func.value(value).kind {
            ValueKind::Const(value) => self.load_const(value),
            ValueKind::Global(name) => self
                .body
                .push_str(&format!("  leaq {}(%rip), %rax\n", name)),
            _ => {
                let offset = self.layout.offset(value);
                match self.func.value(value).ty {
                    Type::Ptr(_) => self
                        .body
                        .push_str(&format!("  movq {}(%rbp), %rax\n", offset)),
                    _ => self
                        .body
                        .push_str(&format!("  movl {}(%rbp), %eax\n", offset)),
                }
            }
        }
    }

    pub(super) fn load_float_value(&mut self, value: ValueId, reg: &str) {
        match &self.func.value(value).kind {
            ValueKind::Const(Const::Float(bits)) => {
                let label = self.parent.ctx.fresh_label("float");
                self.parent.out.push_str(&format!(
                    ".section .rodata\n{}:\n  .long {}\n.text\n",
                    label, bits
                ));
                self.body
                    .push_str(&format!("  movss {}(%rip), {}\n", label, reg));
            }
            _ => {
                self.load_value(value);
                self.body.push_str(&format!("  movd %eax, {}\n", reg));
            }
        }
    }

    fn load_const(&mut self, value: &Const) {
        match value {
            Const::Int(value) => self.body.push_str(&format!("  movl ${}, %eax\n", value)),
            Const::Bool(value) => self
                .body
                .push_str(&format!("  movl ${}, %eax\n", *value as i32)),
            Const::Float(bits) => self.body.push_str(&format!("  movl ${}, %eax\n", bits)),
            Const::Zero(_) => self.body.push_str("  movl $0, %eax\n"),
            Const::String(_) | Const::Array(_) => self.body.push_str("  movl $0, %eax\n"),
        }
    }

    pub(super) fn load_indirect(&mut self, ty: &Type) {
        match ty {
            Type::Ptr(_) => self.body.push_str("  movq (%rax), %rax\n"),
            _ => self.body.push_str("  movl (%rax), %eax\n"),
        }
    }

    pub(super) fn store_indirect(&mut self, ty: &Type) {
        match ty {
            Type::Ptr(_) => self.body.push_str("  movq %rax, (%rcx)\n"),
            _ => self.body.push_str("  movl %eax, (%rcx)\n"),
        }
    }

    pub(super) fn store_result(&mut self, value: ValueId) {
        let offset = self.layout.offset(value);
        match self.func.value(value).ty {
            Type::Ptr(_) => self
                .body
                .push_str(&format!("  movq %rax, {}(%rbp)\n", offset)),
            _ => self
                .body
                .push_str(&format!("  movl %eax, {}(%rbp)\n", offset)),
        }
    }

    pub(super) fn push_rax(&mut self) {
        self.body.push_str("  pushq %rax\n");
    }

    pub(super) fn pop_rcx(&mut self) {
        self.body.push_str("  popq %rcx\n");
    }

    pub(super) fn block_label(&self, block_idx: usize) -> String {
        format!(".L_{}_bb{}", self.func.name, block_idx)
    }

    pub(super) fn object_offset(&self, value: ValueId, _ty: &Type) -> i32 {
        self.layout.offset(value) + 8
    }
}

fn gep_elem_type(ty: &Type) -> Type {
    match ty {
        Type::Ptr(inner) => (**inner).clone(),
        Type::Array { elem, .. } => (**elem).clone(),
        _ => Type::I32,
    }
}

fn pointee(ty: &Type) -> Option<Type> {
    match ty {
        Type::Ptr(inner) => Some((**inner).clone()),
        _ => None,
    }
}
