mod data;
mod layout;
mod sig;

pub(crate) use data::emit_ir_data_section;
pub(crate) use layout::{IrFuncLayout, ir_align_to, ir_size};
pub(crate) use sig::{IrFuncSig, IrModuleCtx, IrParamSig};
