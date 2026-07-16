use super::layout::ir_size;
use crate::ir::{Const, Module, Type};

pub(crate) fn emit_ir_data_section(module: &Module, word_directive: &str) -> String {
    let mut out = String::new();
    let mut current_section = None;

    for global in &module.globals {
        let section = if global.init.as_ref().map_or(true, is_zero_const) {
            ".bss"
        } else {
            ".data"
        };
        if current_section != Some(section) {
            out.push_str(section);
            out.push('\n');
            current_section = Some(section);
        }

        out.push_str(&format!(".globl {}\n{}:\n", global.name, global.name));
        emit_const_words(&mut out, global.init.as_ref(), &global.ty, word_directive);
    }
    out
}

fn emit_const_words(out: &mut String, init: Option<&Const>, ty: &Type, word_directive: &str) {
    match init {
        Some(value) if is_zero_const(value) => {
            out.push_str(&format!("  .zero {}\n", ir_size(ty).max(4)));
        }
        Some(Const::Array(values)) => {
            if let Type::Array { elem, len } = ty {
                for idx in 0..*len {
                    emit_const_words(out, values.get(idx), elem, word_directive);
                }
            }
        }
        Some(Const::Zero(_)) | None => {
            out.push_str(&format!("  .zero {}\n", ir_size(ty).max(4)));
        }
        Some(value) => out.push_str(&format!("  {} {}\n", word_directive, const_bits(value))),
    }
}

fn is_zero_const(value: &Const) -> bool {
    match value {
        Const::Int(value) => *value == 0,
        Const::Bool(value) => !*value,
        Const::Float(bits) => *bits == 0,
        Const::Zero(_) => true,
        Const::Array(values) => values.iter().all(is_zero_const),
        Const::String(_) => false,
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
