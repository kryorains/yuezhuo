use super::util::{const_i32, get_or_add_i32_const, rewrite_function_uses, ValueReplacements};
use super::ModulePass;
use crate::ir::int_range::{collect_function_return_ranges, collect_value_ranges};
use crate::ir::{BinaryOp, InstKind, Module};

pub(super) struct RangeIntegerSimplifyPass;

impl RangeIntegerSimplifyPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for RangeIntegerSimplifyPass {
    fn run(&mut self, module: &mut Module) {
        let return_ranges = collect_function_return_ranges(module);
        for func in &mut module.funcs {
            let ranges = collect_value_ranges(func, &return_ranges);
            let mut replacements = ValueReplacements::new();
            let mut removed = Vec::new();
            let mut zero_results = Vec::new();

            for (block_idx, block) in func.blocks.iter().enumerate() {
                for (inst_idx, inst) in block.insts.iter().enumerate() {
                    let (Some(result), InstKind::Binary { op, lhs, rhs }) =
                        (inst.result, &inst.kind)
                    else {
                        continue;
                    };
                    if !matches!(op, BinaryOp::Idiv | BinaryOp::Imod) {
                        continue;
                    }
                    let Some(divisor) = const_i32(func, *rhs).filter(|divisor| *divisor != 0)
                    else {
                        continue;
                    };
                    let Some(range) = ranges[lhs.0] else {
                        continue;
                    };
                    let bound = i64::from(divisor).abs();
                    if range.min > -bound && range.max < bound {
                        if *op == BinaryOp::Imod {
                            replacements.insert(result, *lhs);
                        } else {
                            zero_results.push(result);
                        }
                        removed.push((block_idx, inst_idx));
                    }
                }
            }

            if !zero_results.is_empty() {
                let zero = get_or_add_i32_const(func, 0);
                replacements.extend(zero_results.into_iter().map(|result| (result, zero)));
            }
            if removed.is_empty() {
                continue;
            }
            rewrite_function_uses(func, &replacements);
            for (block_idx, inst_idx) in removed {
                let inst = &mut func.blocks[block_idx].insts[inst_idx];
                inst.result = None;
                inst.kind = InstKind::Nop;
            }
            if let Err(errors) = func.verify() {
                panic!(
                    "range integer simplification produced invalid IR in {}: {:?}",
                    func.name, errors
                );
            }
        }
    }
}
