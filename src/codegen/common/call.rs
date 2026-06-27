use crate::ir::{Function, Type, ValueId};

use super::{IrFuncSig, IrModuleCtx, IrParamSig};

/// 根据模块里登记的函数签名，为一次 call 指令补齐实参签名。
///
/// 这里处理两类情况：
/// 1. 已知函数（用户函数或运行时函数）：优先使用 `IrModuleCtx` 中的声明；
/// 2. 未知函数：按实参 Value 的 IR 类型临时推导，返回类型按历史行为默认为 `i32`。
///
/// 三个目标后端在 call.rs 里都需要这段逻辑；抽到 common 后，目标代码可以专注于
/// “怎么搬运参数到寄存器/栈”这一件事。
pub(crate) fn resolve_call_sig(
    ctx: &IrModuleCtx<'_>,
    func: &Function,
    name: &str,
    args: &[ValueId],
) -> (IrFuncSig, Vec<IrParamSig>) {
    let sig = ctx.funcs.get(name).cloned().unwrap_or_else(|| IrFuncSig {
        ret: Type::I32,
        params: args
            .iter()
            .map(|arg| param_sig_from_value(func, *arg))
            .collect(),
    });
    let arg_sigs = args
        .iter()
        .enumerate()
        .map(|(idx, arg)| {
            sig.params
                .get(idx)
                .cloned()
                .unwrap_or_else(|| param_sig_from_value(func, *arg))
        })
        .collect();
    (sig, arg_sigs)
}

fn param_sig_from_value(func: &Function, value: ValueId) -> IrParamSig {
    let ty = func.value(value).ty.clone();
    IrParamSig {
        is_pointer: matches!(ty, Type::Ptr(_)),
        ty,
    }
}
