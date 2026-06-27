use crate::ir::Type;

/// GEP 下降时，从“当前索引前的容器类型”推导这一次索引跨过的元素类型。
///
/// 例如：
/// - `ptr<T>` 的元素是 `T`；
/// - `[N x T]` 的元素是 `T`；
/// - 兜底返回 `i32` 是为了让错误 IR 不至于在 codegen 阶段直接崩溃。
pub(crate) fn gep_elem_type(ty: &Type) -> Type {
    match ty {
        Type::Ptr(inner) => (**inner).clone(),
        Type::Array { elem, .. } => (**elem).clone(),
        _ => Type::I32,
    }
}

/// 如果一个类型是指针，返回其指向的类型。
///
/// 单索引 GEP 经常用于“把指针按目标结果类型偏移”的场景，此时优先参考结果类型的
/// pointee，能让数组退化成指针后的寻址步长更符合预期。
pub(crate) fn pointee(ty: &Type) -> Option<Type> {
    match ty {
        Type::Ptr(inner) => Some((**inner).clone()),
        _ => None,
    }
}
