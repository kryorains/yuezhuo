use super::dominators::{ControlFlowGraph, Dominators};
use super::function_effects::FunctionEffects;
use super::util::{resolve_replacement, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{
    BinaryOp, CastOp, CmpOp, Function, FunctionId, InstKind, Module, Type, UnaryOp, ValueId,
    ValueKind,
};

const MAX_CSE_KEY_OPERANDS: usize = 262_144;
const MAX_CSE_COMPARISON_WORK: usize = 4_194_304;

/// 公共子表达式消除。
///
/// 这是一个保守的全局 CSE：沿 dominator tree 维护“当前路径上可用的纯表达式”。
/// 如果新指令的表达式 key 已经出现过，就把新结果替换成已有结果，并把新指令改成 Nop。
///
/// Load/store 不参与；有结果且模块摘要严格证明为 NoMemory 的直接调用也可参与，
/// 但只复用已支配当前位置且实参完全相同的既有返回值。
pub(super) struct CsePass;

impl CsePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for CsePass {
    fn run(&mut self, module: &mut Module) {
        let effects = FunctionEffects::analyze(module);
        for func in &mut module.funcs {
            cse_function(func, &effects);
        }
    }
}

fn cse_function(func: &mut Function, effects: &FunctionEffects) {
    if func.blocks.len() > 1024 || func.values.len() > 8192 || !cse_work_within_budget(func) {
        return;
    }

    canonicalize_global_values(func);
    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let mut replacements = ValueReplacements::new();
    // 用栈式 Vec 保存从入口到当前基本块路径上的可用表达式。
    // 进入子节点前追加，离开基本块后 truncate 回原长度，天然满足支配关系约束。
    let mut available = Vec::<(ExprKey, ValueId)>::new();

    visit_dom_tree(
        func,
        func.entry,
        &dom,
        effects,
        &mut replacements,
        &mut available,
    );
    rewrite_function_uses(func, &replacements);

    if let Err(errors) = func.verify() {
        panic!("cse produced invalid IR in {}: {:?}", func.name, errors);
    }
}

/// Lowering and CFG inlining may materialize the same global symbol more than
/// once in a function. They are the same address when both the symbol and its
/// complete pointer type agree, so using one canonical SSA root exposes GEP
/// and pointer-recurrence CSE without making any new alias assumption.
fn canonicalize_global_values(func: &mut Function) {
    let mut canonical = Vec::<(String, Type, ValueId)>::new();
    let mut replacements = ValueReplacements::new();
    for (index, value) in func.values.iter().enumerate() {
        let ValueKind::Global(name) = &value.kind else {
            continue;
        };
        if let Some((_, _, existing)) = canonical.iter().find(|(existing_name, existing_ty, _)| {
            existing_name == name && existing_ty == &value.ty
        }) {
            replacements.insert(ValueId(index), *existing);
        } else {
            canonical.push((name.clone(), value.ty.clone(), ValueId(index)));
        }
    }
    rewrite_function_uses(func, &replacements);
}

fn visit_dom_tree(
    func: &mut Function,
    block: crate::ir::BlockId,
    dom: &Dominators,
    effects: &FunctionEffects,
    replacements: &mut ValueReplacements,
    available: &mut Vec<(ExprKey, ValueId)>,
) {
    let available_base = available.len();

    for inst_idx in 0..func.blocks[block.0].insts.len() {
        let inst = func.blocks[block.0].insts[inst_idx].clone();
        let Some(result) = inst.result else {
            continue;
        };
        let Some(key) = ExprKey::from_inst(&inst.kind, result, func, effects, replacements) else {
            continue;
        };

        if let Some(replacement) = find_available(available, &key) {
            // 已有表达式支配当前指令，可以安全复用已有结果。
            replacements.insert(result, resolve_replacement(replacement, replacements));
            func.blocks[block.0].insts[inst_idx].result = None;
            func.blocks[block.0].insts[inst_idx].kind = InstKind::Nop;
        } else {
            available.push((key, result));
        }
    }

    for child in &dom.children[block.0] {
        visit_dom_tree(func, *child, dom, effects, replacements, available);
    }

    available.truncate(available_base);
}

fn find_available(available: &[(ExprKey, ValueId)], key: &ExprKey) -> Option<ValueId> {
    available
        .iter()
        .rev()
        .find_map(|(existing, value)| (existing == key).then_some(*value))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExprKey {
    Unary(UnaryOp, ValueId),
    Binary(BinaryOp, ValueId, ValueId),
    Icmp(CmpOp, ValueId, ValueId),
    Fcmp(CmpOp, ValueId, ValueId),
    Cast(CastOp, ValueId),
    Gep(Type, ValueId, Vec<ValueId>),
    Call(FunctionId, Vec<ValueId>),
}

impl ExprKey {
    fn from_inst(
        kind: &InstKind,
        result: ValueId,
        func: &Function,
        effects: &FunctionEffects,
        replacements: &ValueReplacements,
    ) -> Option<Self> {
        match kind {
            InstKind::Unary { op, value } => {
                Some(Self::Unary(*op, resolve_replacement(*value, replacements)))
            }
            InstKind::Binary { op, lhs, rhs } => {
                let lhs = resolve_replacement(*lhs, replacements);
                let rhs = resolve_replacement(*rhs, replacements);
                let (lhs, rhs) = normalize_binary_args(*op, lhs, rhs);
                Some(Self::Binary(*op, lhs, rhs))
            }
            InstKind::Icmp { op, lhs, rhs } => {
                let lhs = resolve_replacement(*lhs, replacements);
                let rhs = resolve_replacement(*rhs, replacements);
                let (op, lhs, rhs) = normalize_cmp_args(*op, lhs, rhs);
                Some(Self::Icmp(op, lhs, rhs))
            }
            InstKind::Fcmp { op, lhs, rhs } => Some(Self::Fcmp(
                *op,
                resolve_replacement(*lhs, replacements),
                resolve_replacement(*rhs, replacements),
            )),
            InstKind::Cast { op, value } => {
                Some(Self::Cast(*op, resolve_replacement(*value, replacements)))
            }
            InstKind::Gep { base, indices } => Some(Self::Gep(
                func.values.get(result.0)?.ty.clone(),
                resolve_replacement(*base, replacements),
                indices
                    .iter()
                    .map(|index| resolve_replacement(*index, replacements))
                    .collect(),
            )),
            InstKind::Call { name, args } => {
                let args = args
                    .iter()
                    .map(|arg| resolve_replacement(*arg, replacements))
                    .collect::<Vec<_>>();
                let callee = effects.resolve_no_memory_call(func, name, result, &args)?;
                Some(Self::Call(callee, args))
            }
            InstKind::Nop
            | InstKind::Phi { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::MemZero { .. } => None,
        }
    }
}

/// Conservatively bounds the existing linear available-expression lookup.
/// The estimate orders all candidate expressions as one path, so it is an
/// upper bound for every dominator-tree path without first building that tree.
fn cse_work_within_budget(func: &Function) -> bool {
    let mut prior_keys = 0usize;
    let mut key_operands = 0usize;
    let mut comparison_work = 0usize;

    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if inst.result.is_none() {
            continue;
        }
        let Some(operands) = potential_key_operands(func, inst.result.unwrap(), &inst.kind) else {
            continue;
        };
        key_operands = key_operands.saturating_add(operands);
        comparison_work =
            comparison_work.saturating_add(prior_keys.saturating_mul(operands.saturating_add(1)));
        if key_operands > MAX_CSE_KEY_OPERANDS || comparison_work > MAX_CSE_COMPARISON_WORK {
            return false;
        }
        prior_keys = prior_keys.saturating_add(1);
    }
    true
}

fn potential_key_operands(func: &Function, result: ValueId, kind: &InstKind) -> Option<usize> {
    match kind {
        InstKind::Unary { .. } | InstKind::Cast { .. } => Some(1),
        InstKind::Binary { .. } | InstKind::Icmp { .. } | InstKind::Fcmp { .. } => Some(2),
        InstKind::Gep { indices, .. } => Some(
            indices
                .len()
                .saturating_add(1)
                .saturating_add(type_nodes(&func.values.get(result.0)?.ty)),
        ),
        InstKind::Call { args, .. } => Some(args.len().saturating_add(1)),
        InstKind::Nop
        | InstKind::Phi { .. }
        | InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::MemZero { .. } => None,
    }
}

fn type_nodes(mut ty: &Type) -> usize {
    let mut nodes = 0usize;
    loop {
        nodes = nodes.saturating_add(1);
        if nodes > MAX_CSE_KEY_OPERANDS {
            return nodes;
        }
        match ty {
            Type::Ptr(inner) => ty = inner,
            Type::Array { elem, .. } => ty = elem,
            Type::Void | Type::I1 | Type::I32 | Type::F32 => return nodes,
        }
    }
}

/// 对可交换的整数/布尔二元运算做操作数排序，提升命中率。
///
/// 浮点加乘暂时不归一化，避免引入浮点语义上的额外假设。
fn normalize_binary_args(op: BinaryOp, lhs: ValueId, rhs: ValueId) -> (ValueId, ValueId) {
    if matches!(
        op,
        BinaryOp::Iadd
            | BinaryOp::Imul
            | BinaryOp::Iand
            | BinaryOp::Ior
            | BinaryOp::Ixor
            | BinaryOp::And
            | BinaryOp::Or
    ) && rhs.0 < lhs.0
    {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

/// `eq/ne` 可交换，其它大小比较不交换。
fn normalize_cmp_args(op: CmpOp, lhs: ValueId, rhs: ValueId) -> (CmpOp, ValueId, ValueId) {
    if matches!(op, CmpOp::Eq | CmpOp::Ne) && rhs.0 < lhs.0 {
        (op, rhs, lhs)
    } else {
        (op, lhs, rhs)
    }
}
