use crate::ir::Type;

use super::sig::IrParamSig;

/// 目标 ABI 中一个实参最终被放置的位置。
///
/// 目前两个后端都采用“整型/指针寄存器”和“浮点寄存器”两条独立通道；
/// 超出寄存器数量后，剩余参数落到调用栈上。因此这里可以作为公共逻辑，
/// 各目标只需要把 `usize` 映射到自己的物理寄存器名称即可。
#[derive(Debug, Clone, Copy)]
pub(crate) enum IrArgLocation {
    IntReg(usize),
    FloatReg(usize),
    Stack,
}

/// 按 SysY 当前支持类型的简化 ABI 规则给参数分配位置。
///
/// - 指针总是走整型寄存器通道；
/// - `f32` 且不是指针时走浮点寄存器通道；
/// - 其他标量类型走整型寄存器通道；
/// - 对应通道的寄存器耗尽后改走栈。
///
/// x86_64 和 RISC-V64 在“通道选择”上相同，只是寄存器个数和名字不同，
/// 所以把这段重复逻辑放在 common 里，避免两个 abi.rs/call.rs 各维护一份。
pub(crate) fn assign_arg_locations(
    arg_sigs: &[IrParamSig],
    int_reg_count: usize,
    float_reg_count: usize,
) -> Vec<IrArgLocation> {
    let mut int_idx = 0usize;
    let mut float_idx = 0usize;
    let mut locations = Vec::with_capacity(arg_sigs.len());
    for arg_sig in arg_sigs {
        if arg_sig.is_pointer || arg_sig.ty != Type::F32 {
            if int_idx < int_reg_count {
                locations.push(IrArgLocation::IntReg(int_idx));
                int_idx += 1;
            } else {
                locations.push(IrArgLocation::Stack);
            }
        } else if float_idx < float_reg_count {
            locations.push(IrArgLocation::FloatReg(float_idx));
            float_idx += 1;
        } else {
            locations.push(IrArgLocation::Stack);
        }
    }
    locations
}
