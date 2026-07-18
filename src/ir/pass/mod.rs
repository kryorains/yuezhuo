mod bit_idiom;
mod const_fold;
mod cse;
mod dce;
pub(crate) mod dominators;
mod gep_induction;
mod global_scalar_localize;
mod inline;
mod inst_combine;
mod invariant_load;
mod licm;
mod local_forward;
mod loop_analysis;
mod piecewise_expr;
mod recursive_inline;
mod reduction_jam;
mod repeat_reduction;
mod scalar_promote;
mod simple_loop_unroll;
mod simplify_cfg;
mod tail_recursion;
mod util;

use super::Module;
use bit_idiom::LoopIdiomPass;
use const_fold::ConstFoldPass;
use cse::CsePass;
use dce::DcePass;
use gep_induction::GepInductionPass;
use global_scalar_localize::GlobalScalarLocalizePass;
use inline::InlineSmallExprPass;
use inst_combine::InstCombinePass;
use invariant_load::InvariantLoadForwardPass;
use licm::LicmPass;
use local_forward::LocalForwardPass;
use piecewise_expr::PiecewiseExprPass;
use recursive_inline::RecursiveInlinePass;
use reduction_jam::ReductionJamPass;
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
            pipeline.add(GlobalScalarLocalizePass::new());
            pipeline.add(ScalarPromotePass::new());
            pipeline.add(RecursiveInlinePass::new());
            pipeline.add(InlineSmallExprPass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(CsePass::new());
            pipeline.add(LicmPass::new());
            pipeline.add(InvariantLoadForwardPass::new());
            pipeline.add(DcePass::new());
            pipeline.add(ReductionJamPass::new());
            pipeline.add(CsePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(InvariantLoadForwardPass::new());
            pipeline.add(DcePass::new());
            pipeline.add(PiecewiseExprPass::new());
            pipeline.add(RepeatReductionPass::new());
            if options.enable_simple_loop_unroll {
                pipeline.add(SimpleLoopUnrollPass::new());
            }
            pipeline.add(InstCombinePass::new());
            pipeline.add(ConstFoldPass::new());
            pipeline.add(LoopIdiomPass::new());
            pipeline.add(ConstFoldPass::new());
            // Run address strength reduction after transforms whose matching
            // intentionally expects the source loop-phi set. In particular,
            // this preserves the existing simple-unroll profitability gate.
            pipeline.add(GepInductionPass::new());
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
