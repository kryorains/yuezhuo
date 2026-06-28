use crate::ir::{Function, Module, Type};
use std::collections::HashMap;

/// codegen 关注的参数签名。
///
/// `Type::Ptr(_)` 已经能表达指针，但这里额外缓存 `is_pointer`，是为了让 ABI 分配和
/// load/store 逻辑能直接区分“地址值”和普通标量，减少各后端反复 `matches!`。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IrParamSig {
    pub(crate) ty: Type,
    pub(crate) is_pointer: bool,
}

/// 函数调用所需的最小签名信息：返回类型 + 参数列表。
#[derive(Debug, Clone)]
pub(crate) struct IrFuncSig {
    pub(crate) ret: Type,
    pub(crate) params: Vec<IrParamSig>,
}

/// 一次模块级汇编生成过程共享的上下文。
///
/// 这里集中保存：
/// - 原始 IR module；
/// - 用户函数和 SysY 运行时函数的签名表；
/// - 后端生成局部标签时使用的递增编号。
pub(crate) struct IrModuleCtx<'a> {
    pub(crate) module: &'a Module,
    pub(crate) funcs: HashMap<String, IrFuncSig>,
    label_id: usize,
}

impl<'a> IrModuleCtx<'a> {
    pub(crate) fn new(module: &'a Module) -> Self {
        let mut ctx = Self {
            module,
            funcs: HashMap::new(),
            label_id: 0,
        };
        ctx.add_runtime_sigs();
        ctx.collect_func_sigs();
        ctx
    }

    pub(crate) fn fresh_label(&mut self, prefix: &str) -> String {
        let label = format!(".L_{}_{}", prefix, self.label_id);
        self.label_id += 1;
        label
    }

    fn collect_func_sigs(&mut self) {
        for func in &self.module.funcs {
            let params = func
                .params
                .iter()
                .map(|param| param_sig(func, *param))
                .collect();
            self.funcs.insert(
                func.name.clone(),
                IrFuncSig {
                    ret: func.ret.clone(),
                    params,
                },
            );
        }
    }

    fn add_runtime_sigs(&mut self) {
        // SysY 运行时函数不会出现在 IR module 的 funcs 里，但 call lowering 仍然需要
        // 知道它们的返回值和参数 ABI 分类，所以在收集用户函数前先注册内建签名。
        for (name, ret, params) in [
            ("getint", Type::I32, vec![]),
            ("getch", Type::I32, vec![]),
            ("getfloat", Type::F32, vec![]),
            (
                "getarray",
                Type::I32,
                vec![IrParamSig {
                    ty: Type::Ptr(Box::new(Type::I32)),
                    is_pointer: true,
                }],
            ),
            (
                "getfarray",
                Type::I32,
                vec![IrParamSig {
                    ty: Type::Ptr(Box::new(Type::F32)),
                    is_pointer: true,
                }],
            ),
            (
                "putint",
                Type::Void,
                vec![IrParamSig {
                    ty: Type::I32,
                    is_pointer: false,
                }],
            ),
            (
                "putch",
                Type::Void,
                vec![IrParamSig {
                    ty: Type::I32,
                    is_pointer: false,
                }],
            ),
            (
                "putfloat",
                Type::Void,
                vec![IrParamSig {
                    ty: Type::F32,
                    is_pointer: false,
                }],
            ),
            (
                "putarray",
                Type::Void,
                vec![
                    IrParamSig {
                        ty: Type::I32,
                        is_pointer: false,
                    },
                    IrParamSig {
                        ty: Type::Ptr(Box::new(Type::I32)),
                        is_pointer: true,
                    },
                ],
            ),
            (
                "putfarray",
                Type::Void,
                vec![
                    IrParamSig {
                        ty: Type::I32,
                        is_pointer: false,
                    },
                    IrParamSig {
                        ty: Type::Ptr(Box::new(Type::F32)),
                        is_pointer: true,
                    },
                ],
            ),
            ("starttime", Type::Void, vec![]),
            ("stoptime", Type::Void, vec![]),
            (
                "_sysy_starttime",
                Type::Void,
                vec![IrParamSig {
                    ty: Type::I32,
                    is_pointer: false,
                }],
            ),
            (
                "_sysy_stoptime",
                Type::Void,
                vec![IrParamSig {
                    ty: Type::I32,
                    is_pointer: false,
                }],
            ),
        ] {
            self.funcs
                .insert(name.to_string(), IrFuncSig { ret, params });
        }
    }
}

fn param_sig(func: &Function, value: crate::ir::ValueId) -> IrParamSig {
    let ty = func.value(value).ty.clone();
    IrParamSig {
        is_pointer: matches!(ty, Type::Ptr(_)),
        ty,
    }
}
