use super::ModulePass;
use crate::ir::{
    BinaryOp, BlockId, CmpOp, Const, Function, Global, InstKind, Module, Terminator, Type,
};
use std::collections::{HashSet, VecDeque};

const FIRST_PARAM_LIMIT: i32 = 64;
const SECOND_PARAM_LIMIT: i32 = 1024;
const CACHE_ENTRIES: usize = FIRST_PARAM_LIMIT as usize * SECOND_PARAM_LIMIT as usize;

/// Memoizes pure, branching recursion with two integer state parameters.
///
/// A small wrapper clears its validity table for every outermost call, so
/// reads from mutable globals cannot leak cached results across calls. States
/// outside the fixed cache bounds execute the original body unchanged.
pub(super) struct RecursiveMemoizePass;

impl RecursiveMemoizePass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RecursiveMemoizePass {
    fn run(&mut self, module: &mut Module) {
        let candidate_indices = module
            .funcs
            .iter()
            .enumerate()
            .filter_map(|(idx, func)| is_pure_branching_recursion(func).then_some(idx))
            .collect::<Vec<_>>();

        for (ordinal, func_idx) in candidate_indices.into_iter().enumerate() {
            let original_name = module.funcs[func_idx].name.clone();
            let stem = unique_cache_stem(module, &original_name, ordinal);
            let body_name = format!("{}_body", stem);
            let tags_name = format!("{}_tags", stem);
            let values_name = format!("{}_values", stem);
            let depth_name = format!("{}_depth", stem);

            module.funcs[func_idx].name = body_name.clone();
            append_cache_global(module, &tags_name, cache_array_type());
            append_cache_global(module, &values_name, cache_array_type());
            append_cache_global(module, &depth_name, Type::I32);
            module.funcs.push(build_wrapper(
                original_name,
                body_name,
                tags_name,
                values_name,
                depth_name,
            ));
        }
    }
}

fn unique_cache_stem(module: &Module, function_name: &str, mut nonce: usize) -> String {
    loop {
        let stem = format!("__recursive_memo_{}_{}", function_name, nonce);
        let reserved = [
            format!("{}_body", stem),
            format!("{}_tags", stem),
            format!("{}_values", stem),
            format!("{}_depth", stem),
        ];
        let conflicts = reserved.iter().any(|name| {
            module.funcs.iter().any(|func| &func.name == name)
                || module.globals.iter().any(|global| &global.name == name)
        });
        if !conflicts {
            return stem;
        }
        nonce += 1;
    }
}

fn is_pure_branching_recursion(func: &Function) -> bool {
    if func.ret != Type::I32
        || func.params.len() != 2
        || func
            .params
            .iter()
            .any(|param| func.value(*param).ty != Type::I32)
    {
        return false;
    }

    let mut recursive_calls = 0usize;
    for block in reachable_blocks(func) {
        for inst in &func.block(block).insts {
            match &inst.kind {
                InstKind::Store { .. } | InstKind::MemZero { .. } => return false,
                InstKind::Call { name, .. } if name == &func.name => recursive_calls += 1,
                InstKind::Call { .. } => return false,
                InstKind::Nop
                | InstKind::Phi { .. }
                | InstKind::Alloca { .. }
                | InstKind::Load { .. }
                | InstKind::Unary { .. }
                | InstKind::Binary { .. }
                | InstKind::Icmp { .. }
                | InstKind::Fcmp { .. }
                | InstKind::Cast { .. }
                | InstKind::Gep { .. } => {}
            }
        }
    }
    recursive_calls >= 2
}

fn append_cache_global(module: &mut Module, name: &str, ty: Type) {
    module.globals.push(Global {
        name: name.to_string(),
        ty: ty.clone(),
        is_const: false,
        init: Some(Const::Zero(ty)),
    });
}

fn cache_array_type() -> Type {
    Type::Array {
        elem: Box::new(Type::I32),
        len: CACHE_ENTRIES,
    }
}

fn build_wrapper(
    name: String,
    body_name: String,
    tags_name: String,
    values_name: String,
    depth_name: String,
) -> Function {
    let mut wrapper = Function::new(name, Type::I32);
    let first = wrapper.add_param("memo.first", Type::I32);
    let second = wrapper.add_param("memo.second", Type::I32);
    let entry = wrapper.entry;
    let clear = wrapper.add_block("memo.clear");
    let prepare = wrapper.add_block("memo.prepare");
    let lookup = wrapper.add_block("memo.lookup");
    let hit = wrapper.add_block("memo.hit");
    let miss = wrapper.add_block("memo.miss");
    let uncached = wrapper.add_block("memo.uncached");
    let finish = wrapper.add_block("memo.finish");

    let zero = wrapper.add_const(Const::Int(0));
    let one = wrapper.add_const(Const::Int(1));
    let first_limit = wrapper.add_const(Const::Int(FIRST_PARAM_LIMIT));
    let second_limit = wrapper.add_const(Const::Int(SECOND_PARAM_LIMIT));
    let array_ty = cache_array_type();
    let array_ptr_ty = Type::Ptr(Box::new(array_ty));
    let depth_ptr = wrapper.add_global_ref(&depth_name, Type::Ptr(Box::new(Type::I32)));
    let tags_ptr = wrapper.add_global_ref(&tags_name, array_ptr_ty.clone());
    let values_ptr = wrapper.add_global_ref(&values_name, array_ptr_ty);

    let depth = wrapper
        .append_inst(entry, InstKind::Load { ptr: depth_ptr }, Some(Type::I32))
        .unwrap();
    let outermost = wrapper
        .append_inst(
            entry,
            InstKind::Icmp {
                op: CmpOp::Eq,
                lhs: depth,
                rhs: zero,
            },
            Some(Type::I1),
        )
        .unwrap();
    wrapper.set_terminator(
        entry,
        Terminator::Branch {
            cond: outermost,
            then_target: clear,
            else_target: prepare,
        },
    );

    wrapper.append_inst(
        clear,
        InstKind::MemZero {
            ptr: tags_ptr,
            bytes: CACHE_ENTRIES * 4,
        },
        None,
    );
    wrapper.set_terminator(clear, Terminator::Jump(prepare));

    let next_depth = wrapper
        .append_inst(
            prepare,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: depth,
                rhs: one,
            },
            Some(Type::I32),
        )
        .unwrap();
    wrapper.append_inst(
        prepare,
        InstKind::Store {
            ptr: depth_ptr,
            value: next_depth,
        },
        None,
    );
    let first_nonnegative = append_compare(&mut wrapper, prepare, CmpOp::Ge, first, zero);
    let first_in_range = append_compare(&mut wrapper, prepare, CmpOp::Lt, first, first_limit);
    let second_nonnegative = append_compare(&mut wrapper, prepare, CmpOp::Ge, second, zero);
    let second_in_range = append_compare(&mut wrapper, prepare, CmpOp::Lt, second, second_limit);
    let first_valid = append_bool_and(&mut wrapper, prepare, first_nonnegative, first_in_range);
    let second_valid = append_bool_and(&mut wrapper, prepare, second_nonnegative, second_in_range);
    let cacheable = append_bool_and(&mut wrapper, prepare, first_valid, second_valid);
    wrapper.set_terminator(
        prepare,
        Terminator::Branch {
            cond: cacheable,
            then_target: lookup,
            else_target: uncached,
        },
    );

    let stride = wrapper.add_const(Const::Int(SECOND_PARAM_LIMIT));
    let row_offset = wrapper
        .append_inst(
            lookup,
            InstKind::Binary {
                op: BinaryOp::Imul,
                lhs: first,
                rhs: stride,
            },
            Some(Type::I32),
        )
        .unwrap();
    let index = wrapper
        .append_inst(
            lookup,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: row_offset,
                rhs: second,
            },
            Some(Type::I32),
        )
        .unwrap();
    let tag_slot = append_cache_slot(&mut wrapper, lookup, tags_ptr, zero, index);
    let value_slot = append_cache_slot(&mut wrapper, lookup, values_ptr, zero, index);
    let tag = wrapper
        .append_inst(lookup, InstKind::Load { ptr: tag_slot }, Some(Type::I32))
        .unwrap();
    let present = append_compare(&mut wrapper, lookup, CmpOp::Ne, tag, zero);
    wrapper.set_terminator(
        lookup,
        Terminator::Branch {
            cond: present,
            then_target: hit,
            else_target: miss,
        },
    );

    let hit_value = wrapper
        .append_inst(hit, InstKind::Load { ptr: value_slot }, Some(Type::I32))
        .unwrap();
    wrapper.set_terminator(hit, Terminator::Jump(finish));

    let miss_value = append_body_call(&mut wrapper, miss, &body_name, first, second);
    wrapper.append_inst(
        miss,
        InstKind::Store {
            ptr: value_slot,
            value: miss_value,
        },
        None,
    );
    wrapper.append_inst(
        miss,
        InstKind::Store {
            ptr: tag_slot,
            value: one,
        },
        None,
    );
    wrapper.set_terminator(miss, Terminator::Jump(finish));

    let uncached_value = append_body_call(&mut wrapper, uncached, &body_name, first, second);
    wrapper.set_terminator(uncached, Terminator::Jump(finish));

    let result = wrapper
        .append_inst(
            finish,
            InstKind::Phi {
                incomings: vec![
                    (hit, hit_value),
                    (miss, miss_value),
                    (uncached, uncached_value),
                ],
            },
            Some(Type::I32),
        )
        .unwrap();
    wrapper.append_inst(
        finish,
        InstKind::Store {
            ptr: depth_ptr,
            value: depth,
        },
        None,
    );
    wrapper.set_terminator(finish, Terminator::Return(Some(result)));
    debug_assert!(wrapper.verify().is_ok());
    wrapper
}

fn append_compare(
    func: &mut Function,
    block: BlockId,
    op: CmpOp,
    lhs: crate::ir::ValueId,
    rhs: crate::ir::ValueId,
) -> crate::ir::ValueId {
    func.append_inst(block, InstKind::Icmp { op, lhs, rhs }, Some(Type::I1))
        .unwrap()
}

fn append_bool_and(
    func: &mut Function,
    block: BlockId,
    lhs: crate::ir::ValueId,
    rhs: crate::ir::ValueId,
) -> crate::ir::ValueId {
    func.append_inst(
        block,
        InstKind::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
        },
        Some(Type::I1),
    )
    .unwrap()
}

fn append_cache_slot(
    func: &mut Function,
    block: BlockId,
    base: crate::ir::ValueId,
    zero: crate::ir::ValueId,
    index: crate::ir::ValueId,
) -> crate::ir::ValueId {
    func.append_inst(
        block,
        InstKind::Gep {
            base,
            indices: vec![zero, index],
        },
        Some(Type::Ptr(Box::new(Type::I32))),
    )
    .unwrap()
}

fn append_body_call(
    func: &mut Function,
    block: BlockId,
    name: &str,
    first: crate::ir::ValueId,
    second: crate::ir::ValueId,
) -> crate::ir::ValueId {
    func.append_inst(
        block,
        InstKind::Call {
            name: name.to_string(),
            args: vec![first, second],
        },
        Some(Type::I32),
    )
    .unwrap()
}

fn reachable_blocks(func: &Function) -> Vec<BlockId> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([func.entry]);
    while let Some(block) = queue.pop_front() {
        if !seen.insert(block) {
            continue;
        }
        match func.block(block).terminator.as_ref() {
            Some(Terminator::Jump(target)) => queue.push_back(*target),
            Some(Terminator::Branch {
                then_target,
                else_target,
                ..
            }) => {
                queue.push_back(*then_target);
                queue.push_back(*else_target);
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }
    seen.into_iter().collect()
}
