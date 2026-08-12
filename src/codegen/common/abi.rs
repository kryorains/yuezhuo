/// 目标 ABI 中一个实参最终被放置的位置。
///
/// 后端采用“整型/指针寄存器”和“浮点寄存器”两条独立通道；
/// 超出寄存器数量后，剩余参数落到调用栈上。因此这里可以作为公共逻辑，
/// 各目标只需要把 `usize` 映射到自己的物理寄存器名称即可。
#[derive(Debug, Clone, Copy)]
pub(crate) enum IrArgLocation {
    IntReg(usize),
    FloatReg(usize),
    Stack,
}
