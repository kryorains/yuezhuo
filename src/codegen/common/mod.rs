mod abi;
mod bitwise;
mod call;
mod data;
mod layout;
mod shift;
mod sig;
mod types;

pub(crate) use abi::{assign_arg_locations, IrArgLocation};
pub(crate) use bitwise::{emulated_bitwise_op, EmulatedBitwiseOp};
pub(crate) use call::resolve_call_sig;
pub(crate) use data::emit_ir_data_section;
pub(crate) use layout::{ir_align_to, ir_size, IrFuncLayout};
pub(crate) use shift::{emulated_shift_op, EmulatedShiftOp};
pub(crate) use sig::{IrFuncSig, IrModuleCtx, IrParamSig};
pub(crate) use types::{gep_elem_type, pointee};
