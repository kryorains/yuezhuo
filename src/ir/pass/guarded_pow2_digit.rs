use super::util::{const_i32, defining_inst};
use super::ModulePass;
use crate::ir::{BinaryOp, BlockId, CmpOp, Function, InstKind, Module, Terminator, Type, ValueId};
use std::collections::{HashSet, VecDeque};

/// Recognizes a loop that divides an integer by a fixed power of two `pos`
/// times and then takes the remainder by that same base.
pub(super) struct GuardedPow2DigitPass;

impl GuardedPow2DigitPass {
    pub(super) fn new() -> Self {
        Self
    }
}

impl ModulePass for GuardedPow2DigitPass {
    fn run(&mut self, module: &mut Module) {
        for func in &mut module.funcs {
            if let Some(shift) = match_pow2_digit_loop(func) {
                func.mark_guarded_pow2_digit(shift);
            }
        }
    }
}

fn match_pow2_digit_loop(func: &Function) -> Option<u32> {
    if func.ret != Type::I32 || func.params.len() != 2 {
        return None;
    }
    let number = func.params[0];
    let position = func.params[1];
    if func.value(number).ty != Type::I32 || func.value(position).ty != Type::I32 {
        return None;
    }

    let entry = func.entry;
    let Terminator::Jump(header) = func.block(entry).terminator.as_ref()? else {
        return None;
    };
    if func
        .block(entry)
        .insts
        .iter()
        .any(|inst| !matches!(inst.kind, InstKind::Nop))
    {
        return None;
    }

    let (number_phi, next_number, number_backedge) =
        phi_with_initial(func, *header, entry, number)?;
    let zero =
        func.values.iter().enumerate().find_map(|(idx, _)| {
            (const_i32(func, ValueId(idx)) == Some(0)).then_some(ValueId(idx))
        })?;
    let (index_phi, next_index, index_backedge) = phi_with_initial(func, *header, entry, zero)?;
    if number_backedge != index_backedge {
        return None;
    }
    let body = number_backedge;

    let Terminator::Branch {
        cond,
        then_target,
        else_target,
    } = func.block(*header).terminator.as_ref()?
    else {
        return None;
    };
    let InstKind::Icmp { op, lhs, rhs } = defining_inst(func, *cond)? else {
        return None;
    };
    if *op != CmpOp::Lt || *lhs != index_phi || *rhs != position || *then_target != body {
        return None;
    }
    let exit = *else_target;
    if func.block(body).terminator.as_ref() != Some(&Terminator::Jump(*header)) {
        return None;
    }

    let (dividend, base_value) = binary_operands(func, next_number, BinaryOp::Idiv)?;
    if dividend != number_phi {
        return None;
    }
    let base = const_i32(func, base_value)?;
    if base < 2 || !(base as u32).is_power_of_two() {
        return None;
    }
    let shift = (base as u32).trailing_zeros();
    if shift == 0 || shift >= 31 {
        return None;
    }
    let (incremented, increment) = binary_operands(func, next_index, BinaryOp::Iadd)?;
    if incremented != index_phi || const_i32(func, increment) != Some(1) {
        return None;
    }

    let Terminator::Return(Some(result)) = func.block(exit).terminator.as_ref()? else {
        return None;
    };
    let (remainder_value, remainder_base) = binary_operands(func, *result, BinaryOp::Imod)?;
    if remainder_value != number_phi || remainder_base != base_value {
        return None;
    }

    for block in reachable_blocks(func) {
        for inst in &func.block(block).insts {
            if !matches!(
                inst.kind,
                InstKind::Nop
                    | InstKind::Phi { .. }
                    | InstKind::Binary { .. }
                    | InstKind::Icmp { .. }
            ) {
                return None;
            }
        }
    }
    Some(shift)
}

fn phi_with_initial(
    func: &Function,
    header: BlockId,
    entry: BlockId,
    initial: ValueId,
) -> Option<(ValueId, ValueId, BlockId)> {
    func.block(header).insts.iter().find_map(|inst| {
        let (Some(result), InstKind::Phi { incomings }) = (inst.result, &inst.kind) else {
            return None;
        };
        if incomings.len() != 2 || !incomings.contains(&(entry, initial)) {
            return None;
        }
        incomings
            .iter()
            .find(|(pred, _)| *pred != entry)
            .map(|(pred, value)| (result, *value, *pred))
    })
}

fn binary_operands(
    func: &Function,
    value: ValueId,
    expected: BinaryOp,
) -> Option<(ValueId, ValueId)> {
    let InstKind::Binary { op, lhs, rhs } = defining_inst(func, value)? else {
        return None;
    };
    (*op == expected).then_some((*lhs, *rhs))
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

#[cfg(test)]
mod tests {
    use super::super::{run_pipeline, OptLevel, PassOptions};
    use crate::ir::{lower::lower_program, InstKind};
    use crate::parser::Parser;

    fn optimized(source: &str) -> crate::ir::Module {
        let program = Parser::new(source).parse_program();
        let mut module = lower_program(&program).unwrap();
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                cfg_inline_rounds: 1,
                cfg_inline_global_loads: false,
                enable_loop_call_memoize: false,
                enable_repeated_overwrite_elision: false,
                enable_guarded_mulmod_idiom: false,
                enable_guarded_pow2_digit_idiom: true,
                enable_write_only_alloca_cleanup_before_inline: true,
            },
        );
        module
    }

    #[test]
    fn marks_power_of_two_digit_extraction_and_preserves_the_call_boundary() {
        let module = optimized(
            r#"
            const int base = 16;
            int digit(int num, int pos) {
                int i = 0;
                while (i < pos) {
                    num = num / base;
                    i = i + 1;
                }
                return num % base;
            }
            int main() { return digit(1234, 2); }
            "#,
        );
        let func = module
            .funcs
            .iter()
            .find(|func| func.name == "digit")
            .unwrap();
        assert_eq!(func.guarded_pow2_digit_shift(), Some(4));
        let main = module
            .funcs
            .iter()
            .find(|func| func.name == "main")
            .unwrap();
        assert!(main
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .any(|inst| matches!(&inst.kind, InstKind::Call { name, .. } if name == "digit")));
    }

    #[test]
    fn rejects_a_non_power_of_two_base() {
        let module = optimized(
            r#"
            const int base = 10;
            int digit(int num, int pos) {
                int i = 0;
                while (i < pos) {
                    num = num / base;
                    i = i + 1;
                }
                return num % base;
            }
            int main() { return digit(1234, 2); }
            "#,
        );
        let func = module
            .funcs
            .iter()
            .find(|func| func.name == "digit")
            .unwrap();
        assert_eq!(func.guarded_pow2_digit_shift(), None);
    }
}
