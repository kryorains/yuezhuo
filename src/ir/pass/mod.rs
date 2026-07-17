mod bit_idiom;
mod const_fold;
mod cse;
mod dce;
mod dominators;
mod inline;
mod invariant_load;
mod licm;
mod local_forward;
mod piecewise_expr;
mod repeat_reduction;
mod scalar_promote;
mod simple_loop_unroll;
mod simplify_cfg;
mod tail_recursion;
mod util;

use super::Module;
use bit_idiom::BitwiseIdiomPass;
use const_fold::ConstFoldPass;
use cse::CsePass;
use dce::DcePass;
use inline::InlineSmallExprPass;
use invariant_load::InvariantLoadForwardPass;
use licm::LicmPass;
use local_forward::LocalForwardPass;
use piecewise_expr::PiecewiseExprPass;
use repeat_reduction::RepeatReductionPass;
use scalar_promote::ScalarPromotePass;
use simple_loop_unroll::SimpleLoopUnrollPass;
use simplify_cfg::SimplifyCfgPass;
use tail_recursion::TailRecursionPass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassOptions {
    pub enable_simple_loop_unroll: bool,
}

pub fn run_pipeline(module: &mut Module, opt_level: OptLevel, options: PassOptions) {
    // 所有优化 pass 都在这里排队，方便统一调整执行顺序。
    let mut pipeline = PassPipeline::new();
    match opt_level {
        OptLevel::O0 => {
            pipeline.add(ScalarPromotePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(DcePass::new());
            pipeline.add(RepeatReductionPass::new());
            pipeline.add(DcePass::new());
        }
        OptLevel::O1 => {
            // 先折叠常量和死分支，再做标量提升/局部转发，最后再清一次新产生的机会。
            pipeline.add(ConstFoldPass::new());
            pipeline.add(SimplifyCfgPass::new());
            pipeline.add(TailRecursionPass::new());
            pipeline.add(ScalarPromotePass::new());
            pipeline.add(InlineSmallExprPass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(CsePass::new());
            pipeline.add(LicmPass::new());
            pipeline.add(InvariantLoadForwardPass::new());
            pipeline.add(DcePass::new());
            pipeline.add(BitwiseIdiomPass::new());
            pipeline.add(PiecewiseExprPass::new());
            pipeline.add(RepeatReductionPass::new());
            if options.enable_simple_loop_unroll {
                pipeline.add(SimpleLoopUnrollPass::new());
            }
            pipeline.add(ConstFoldPass::new());
            pipeline.add(SimplifyCfgPass::new());
            pipeline.add(DcePass::new());
        }
    }
    pipeline.run(module);
}

struct PassPipeline {
    passes: Vec<Box<dyn ModulePass>>,
}

impl PassPipeline {
    fn new() -> Self {
        Self { passes: Vec::new() }
    }

    fn add(&mut self, pass: impl ModulePass + 'static) {
        self.passes.push(Box::new(pass));
    }

    fn run(&mut self, module: &mut Module) {
        // pass 之间直接共享同一个可变 Module，前一个 pass 的结果会喂给后一个 pass。
        for pass in &mut self.passes {
            pass.run(module);
        }
    }
}

trait ModulePass {
    fn run(&mut self, module: &mut Module);
}

#[cfg(test)]
mod tests {
    use super::{
        run_pipeline, BitwiseIdiomPass, ConstFoldPass, ModulePass, OptLevel, PassOptions,
        PiecewiseExprPass,
    };
    use crate::ir::{
        self, BinaryOp, Const, Function, InstKind, Module, Terminator, Type, ValueKind,
    };
    use crate::parser::Parser;

    const BITWISE_VARIANT: &str = r#"
        int combine(int right, int left) {
          int output = 0;
          int remaining = 32;
          int weight = 1;
          int right_bit;
          int left_bit;
          while (remaining > 0) {
            right_bit = right % 2;
            left_bit = left % 2;
            right = right / 2;
            left = left / 2;
            if (left_bit == 1 && right_bit == 1) output = output + weight;
            remaining = remaining - 1;
            weight = weight * 2;
          }
          return output;
        }
        int main() { return combine(3, 6); }
    "#;

    const REORDERED_DIVISION: &str = r#"
        int scale(int amount, int value) {
          if (amount == 3) return value / 8;
          if (1 == amount) return value / 2;
          if (amount != 2) return value;
          return value / 4;
        }
        int main() { return scale(2, -9); }
    "#;

    fn lower(source: &str) -> Module {
        let mut parser = Parser::new(source);
        ir::lower::lower_program(&parser.parse_program()).expect("test source must lower")
    }

    fn optimize(source: &str, opt_level: OptLevel) -> Module {
        let mut module = lower(source);
        run_pipeline(
            &mut module,
            opt_level,
            PassOptions {
                enable_simple_loop_unroll: false,
            },
        );
        module
    }

    fn count_binary(module: &Module, expected: BinaryOp) -> usize {
        module
            .funcs
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.insts)
            .filter(|inst| matches!(inst.kind, InstKind::Binary { op, .. } if op == expected))
            .count()
    }

    #[test]
    fn bitwise_idiom_uses_semantics_instead_of_source_names() {
        let module = optimize(BITWISE_VARIANT, OptLevel::O1);
        assert_eq!(count_binary(&module, BinaryOp::Iand), 1);
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    #[test]
    fn bitwise_idiom_is_not_enabled_at_o0() {
        let module = optimize(BITWISE_VARIANT, OptLevel::O0);
        assert_eq!(count_binary(&module, BinaryOp::Iand), 0);
    }

    #[test]
    fn bitwise_idiom_rejects_observable_loop_effects() {
        let source = BITWISE_VARIANT.replace(
            "right_bit = right % 2;",
            "right_bit = right % 2; putint(right_bit);",
        );
        let module = optimize(&source, OptLevel::O1);
        assert_eq!(count_binary(&module, BinaryOp::Iand), 0);
    }

    #[test]
    fn bitwise_idiom_pass_is_idempotent() {
        let mut module = optimize(BITWISE_VARIANT, OptLevel::O1);
        BitwiseIdiomPass::new().run(&mut module);
        assert_eq!(count_binary(&module, BinaryOp::Iand), 1);
    }

    #[test]
    fn piecewise_expr_interprets_reordered_decision_trees() {
        let module = optimize(REORDERED_DIVISION, OptLevel::O1);
        assert_eq!(count_binary(&module, BinaryOp::Iashr), 1);
        assert!(module.funcs.iter().all(|func| func.verify().is_ok()));
    }

    #[test]
    fn piecewise_expr_is_not_enabled_at_o0() {
        let module = optimize(REORDERED_DIVISION, OptLevel::O0);
        assert_eq!(count_binary(&module, BinaryOp::Iashr), 0);
    }

    #[test]
    fn piecewise_expr_accepts_ranges_starting_at_zero() {
        let source = r#"
            int scale(int value, int amount) {
              if (amount == 0) return value * 1;
              if (amount == 2) return value * 4;
              if (amount != 1) return value;
              return value * 2;
            }
            int main() { return scale(3, 2); }
        "#;
        let module = optimize(source, OptLevel::O1);
        assert_eq!(count_binary(&module, BinaryOp::Ishl), 1);
    }

    #[test]
    fn piecewise_expr_rejects_noncontiguous_mappings() {
        let source = r#"
            int scale(int value, int amount) {
              if (amount == 1) return value * 2;
              if (amount == 3) return value * 8;
              return value;
            }
            int main() { return scale(3, 1); }
        "#;
        let module = optimize(source, OptLevel::O1);
        assert_eq!(count_binary(&module, BinaryOp::Ishl), 0);
    }

    #[test]
    fn piecewise_expr_pass_is_idempotent() {
        let mut module = optimize(REORDERED_DIVISION, OptLevel::O1);
        PiecewiseExprPass::new().run(&mut module);
        assert_eq!(count_binary(&module, BinaryOp::Iashr), 1);
    }

    #[test]
    fn integer_bitwise_and_shift_constants_fold() {
        let mut func = Function::new("fold_integer_bits", Type::I32);
        let lhs = func.add_const(Const::Int(0x55));
        let rhs = func.add_const(Const::Int(0x0f));
        let shift = func.add_const(Const::Int(4));
        let negative = func.add_const(Const::Int(-16));
        let two = func.add_const(Const::Int(2));
        let and = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Iand,
                    lhs,
                    rhs,
                },
                Some(Type::I32),
            )
            .unwrap();
        let or = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Ior,
                    lhs,
                    rhs,
                },
                Some(Type::I32),
            )
            .unwrap();
        let xor = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Ixor,
                    lhs,
                    rhs,
                },
                Some(Type::I32),
            )
            .unwrap();
        let shifted = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Ishl,
                    lhs: and,
                    rhs: shift,
                },
                Some(Type::I32),
            )
            .unwrap();
        let arithmetic_shift = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Iashr,
                    lhs: negative,
                    rhs: two,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(xor)));
        let mut module = Module::new();
        module.add_func(func);

        ConstFoldPass::new().run(&mut module);
        assert!(matches!(
            module.funcs[0].value(and).kind,
            ValueKind::Const(Const::Int(5))
        ));
        assert!(matches!(
            module.funcs[0].value(or).kind,
            ValueKind::Const(Const::Int(95))
        ));
        assert!(matches!(
            module.funcs[0].value(xor).kind,
            ValueKind::Const(Const::Int(90))
        ));
        assert!(matches!(
            module.funcs[0].value(shifted).kind,
            ValueKind::Const(Const::Int(80))
        ));
        assert!(matches!(
            module.funcs[0].value(arithmetic_shift).kind,
            ValueKind::Const(Const::Int(-4))
        ));
    }

    #[test]
    fn verifier_rejects_non_i32_integer_bitwise_operands() {
        let mut func = Function::new("bad_integer_bits", Type::I32);
        let lhs = func.add_const(Const::Bool(true));
        let rhs = func.add_const(Const::Bool(false));
        let result = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Iand,
                    lhs,
                    rhs,
                },
                Some(Type::I32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(result)));
        assert!(func.verify().is_err());
    }
}
