mod const_fold;
mod const_specialize;
mod cse;
mod dce;
pub(crate) mod dominators;
mod function_effects;
mod gep_induction;
mod global_const_prop;
mod global_scalar_localize;
mod inline;
mod inst_combine;
mod invariant_load;
mod licm;
mod local_forward;
mod loop_analysis;
mod pointer_recurrence_coalesce;
mod recursive_inline;
mod reduction_jam;
mod repeat_reduction;
mod scalar_promote;
mod simple_loop_unroll;
mod simplify_cfg;
mod tail_recursion;
mod util;

use super::Module;
use const_fold::ConstFoldPass;
use const_specialize::ConstSpecializePass;
use cse::CsePass;
use dce::DcePass;
use gep_induction::GepInductionPass;
use global_const_prop::GlobalConstPropPass;
use global_scalar_localize::GlobalScalarLocalizePass;
use inline::InlineSmallExprPass;
use inst_combine::InstCombinePass;
use invariant_load::InvariantLoadForwardPass;
use licm::LicmPass;
use local_forward::LocalForwardPass;
use pointer_recurrence_coalesce::PointerRecurrenceCoalescePass;
use recursive_inline::{CfgInlinePass, RecursiveInlinePass};
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
    pub enable_write_only_alloca_cleanup_before_inline: bool,
}

pub fn run_pipeline(module: &mut Module, opt_level: OptLevel, options: PassOptions) {
    run_pipeline_with_reduction_jam_factor(module, opt_level, options, 2);
}

pub fn run_pipeline_with_reduction_jam_factor(
    module: &mut Module,
    opt_level: OptLevel,
    options: PassOptions,
    max_reduction_jam_factor: usize,
) {
    assert!(matches!(max_reduction_jam_factor, 2 | 4));
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
            // 先传播只读全局常量、折叠常量和清理死代码，再做标量提升/局部转发。
            pipeline.add(GlobalConstPropPass::new());
            pipeline.add(ConstFoldPass::new());
            if options.enable_write_only_alloca_cleanup_before_inline {
                pipeline.add(DcePass::new());
            } else {
                pipeline.add(DcePass::preserving_write_only_allocas());
            }
            pipeline.add(SimplifyCfgPass::preserving_loop_preheaders());
            pipeline.add(TailRecursionPass::new());
            pipeline.add(GlobalScalarLocalizePass::new());
            pipeline.add(ScalarPromotePass::new());
            pipeline.add(GlobalScalarLocalizePass::new_across_no_memory_calls());
            pipeline.add(ScalarPromotePass::new());
            pipeline.add(ConstSpecializePass::new());
            pipeline.add(ConstFoldPass::new());
            pipeline.add(SimplifyCfgPass::preserving_loop_preheaders());
            if options.enable_write_only_alloca_cleanup_before_inline {
                pipeline.add(DcePass::new());
            } else {
                pipeline.add(DcePass::preserving_write_only_allocas());
            }
            pipeline.add(RecursiveInlinePass::new());
            pipeline.add(InlineSmallExprPass::new());
            pipeline.add(CfgInlinePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(CsePass::new());
            pipeline.add(LicmPass::new());
            pipeline.add(InvariantLoadForwardPass::new());
            pipeline.add(InstCombinePass::divisibility_only());
            pipeline.add(ConstFoldPass::new());
            pipeline.add(DcePass::new());
            pipeline.add(ReductionJamPass::new(max_reduction_jam_factor));
            pipeline.add(CsePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(CsePass::new());
            pipeline.add(InvariantLoadForwardPass::new());
            pipeline.add(DcePass::new());
            pipeline.add(RepeatReductionPass::new());
            if options.enable_simple_loop_unroll {
                pipeline.add(SimpleLoopUnrollPass::new(max_reduction_jam_factor));
            }
            pipeline.add(InstCombinePass::new());
            pipeline.add(ConstFoldPass::new());
            // Run address strength reduction after transforms whose matching
            // intentionally expects the source loop-phi set. In particular,
            // this preserves the existing simple-unroll profitability gate.
            pipeline.add(GepInductionPass::new());
            pipeline.add(PointerRecurrenceCoalescePass::new());
            pipeline.add(DcePass::new());
            pipeline.add(CsePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(CsePass::new());
            pipeline.add(DcePass::new());
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
