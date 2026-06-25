use super::layout::ir_size;
use crate::ir::{Const, Module, Type};

pub(crate) fn emit_ir_data_section(module: &Module, word_directive: &str) -> String {
    if module.globals.is_empty() {
        return String::new();
    }
    let mut out = String::from(".data\n");
    for global in &module.globals {
        out.push_str(&format!(".globl {}\n{}:\n", global.name, global.name));
        emit_const_words(&mut out, global.init.as_ref(), &global.ty, word_directive);
    }
    out
}

fn emit_const_words(out: &mut String, init: Option<&Const>, ty: &Type, word_directive: &str) {
    match init {
        Some(Const::Array(values)) => {
            if let Type::Array { elem, len } = ty {
                for idx in 0..*len {
                    emit_const_words(out, values.get(idx), elem, word_directive);
                }
            }
        }
        Some(Const::Zero(_)) | None => {
            for _ in 0..(ir_size(ty) / 4).max(1) {
                out.push_str(&format!("  {} 0\n", word_directive));
            }
        }
        Some(value) => out.push_str(&format!("  {} {}\n", word_directive, const_bits(value))),
    }
}

fn const_bits(value: &Const) -> u32 {
    match value {
        Const::Int(value) => *value as u32,
        Const::Bool(value) => *value as u32,
        Const::Float(bits) => *bits,
        Const::String(_) => 0,
        Const::Zero(_) | Const::Array(_) => 0,
    }
}
