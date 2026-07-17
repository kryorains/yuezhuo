use super::dominators::{ControlFlowGraph, Dominators};
use super::util::{resolve_replacement, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::{BinaryOp, CastOp, CmpOp, Function, InstKind, Module, UnaryOp, ValueId};

/// 公共子表达式消除。
///
/// 这是一个保守的全局 CSE：沿 dominator tree 维护“当前路径上可用的纯表达式”。
/// 如果新指令的表达式 key 已经出现过，就把新结果替换成已有结果，并把新指令改成 Nop。
///
/// 为了避免内存 alias 和副作用问题，当前只处理纯表达式，不处理 load/store/call。
pub(super) struct CsePass;

impl CsePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for CsePass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            cse_function(func);
        }
    }
}

fn cse_function(func: &mut Function) {
    if func.blocks.len() > 1024 || func.values.len() > 8192 {
        return;
    }

    let cfg = ControlFlowGraph::new(func);
    let dom = Dominators::new(func, &cfg);
    let mut replacements = ValueReplacements::new();
    // 用栈式 Vec 保存从入口到当前基本块路径上的可用表达式。
    // 进入子节点前追加，离开基本块后 truncate 回原长度，天然满足支配关系约束。
    let mut available = Vec::<(ExprKey, ValueId)>::new();

    visit_dom_tree(func, func.entry, &dom, &mut replacements, &mut available);
    rewrite_function_uses(func, &replacements);

    if let Err(errors) = func.verify() {
        panic!("cse produced invalid IR in {}: {:?}", func.name, errors);
    }
}

fn visit_dom_tree(
    func: &mut Function,
    block: crate::ir::BlockId,
    dom: &Dominators,
    replacements: &mut ValueReplacements,
    available: &mut Vec<(ExprKey, ValueId)>,
) {
    let available_base = available.len();

    for inst_idx in 0..func.blocks[block.0].insts.len() {
        let inst = func.blocks[block.0].insts[inst_idx].clone();
        let Some(result) = inst.result else {
            continue;
        };
        let Some(key) = ExprKey::from_inst(&inst.kind, replacements) else {
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
        visit_dom_tree(func, *child, dom, replacements, available);
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
    Gep(ValueId, Vec<ValueId>),
}

impl ExprKey {
    fn from_inst(kind: &InstKind, replacements: &ValueReplacements) -> Option<Self> {
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
                resolve_replacement(*base, replacements),
                indices
                    .iter()
                    .map(|index| resolve_replacement(*index, replacements))
                    .collect(),
            )),
            InstKind::Nop
            | InstKind::Phi { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::MemZero { .. }
            | InstKind::Call { .. } => None,
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
