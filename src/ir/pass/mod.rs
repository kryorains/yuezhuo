mod adjacent_loop_fusion;
mod const_fold;
mod const_specialize;
mod constant_address_reduction;
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
mod local_memzero_sink;
mod loop_analysis;
mod loop_call_memoize;
mod pointer_recurrence_coalesce;
mod range_integer;
mod recursive_inline;
mod reduction_jam;
mod regional_global_scalar;
mod repeat_reduction;
mod scalar_promote;
mod simple_loop_unroll;
mod simplify_cfg;
mod small_loop_full_unroll;
mod tail_recursion;
mod util;

use super::Module;
use adjacent_loop_fusion::AdjacentLoopFusionPass;
use const_fold::ConstFoldPass;
use const_specialize::ConstSpecializePass;
use constant_address_reduction::ConstantAddressReductionPass;
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
use local_memzero_sink::LocalMemzeroSinkPass;
use loop_call_memoize::LoopCallMemoizePass;
use pointer_recurrence_coalesce::PointerRecurrenceCoalescePass;
use range_integer::RangeIntegerSimplifyPass;
use recursive_inline::{CfgInlinePass, RecursiveInlinePass};
use reduction_jam::ReductionJamPass;
use regional_global_scalar::RegionalGlobalScalarPass;
use repeat_reduction::RepeatReductionPass;
use scalar_promote::ScalarPromotePass;
use simple_loop_unroll::SimpleLoopUnrollPass;
use simplify_cfg::SimplifyCfgPass;
use small_loop_full_unroll::SmallLoopFullUnrollPass;
use tail_recursion::TailRecursionPass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassOptions {
    pub enable_simple_loop_unroll: bool,
    pub small_expr_inline_rounds: usize,
    pub cfg_inline_rounds: usize,
    pub cfg_inline_global_loads: bool,
    pub recursive_inline_rounds: usize,
    pub enable_constant_address_count_reduction: bool,
    pub enable_recursive_const_specialization: bool,
    pub enable_initialized_global_propagation: bool,
    pub enable_uniform_constant_arguments: bool,
    pub enable_loop_call_memoize: bool,
    pub enable_loop_invariant_call_memoize: bool,
    pub enable_regional_global_scalar_promotion: bool,
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
            pipeline.add(GlobalConstPropPass::new(
                options.enable_initialized_global_propagation,
            ));
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
            pipeline.add(ConstSpecializePass::new(
                options.enable_recursive_const_specialization,
                options.enable_uniform_constant_arguments,
            ));
            pipeline.add(ConstFoldPass::new());
            pipeline.add(SmallLoopFullUnrollPass::new());
            pipeline.add(ConstFoldPass::new());
            pipeline.add(SimplifyCfgPass::preserving_loop_preheaders());
            if options.enable_write_only_alloca_cleanup_before_inline {
                pipeline.add(DcePass::new());
            } else {
                pipeline.add(DcePass::preserving_write_only_allocas());
            }
            pipeline.add(RecursiveInlinePass::with_rounds(
                options.recursive_inline_rounds,
            ));
            for _ in 0..options.small_expr_inline_rounds {
                pipeline.add(InlineSmallExprPass::new());
            }
            for _ in 0..options.cfg_inline_rounds {
                pipeline.add(CfgInlinePass::new(options.cfg_inline_global_loads));
            }
            pipeline.add(SimplifyCfgPass::preserving_loop_preheaders());
            pipeline.add(RangeIntegerSimplifyPass::new());
            pipeline.add(DcePass::new());
            pipeline.add(AdjacentLoopFusionPass::new());
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
            if options.enable_loop_call_memoize {
                pipeline.add(LoopCallMemoizePass::new());
            }
            if options.enable_loop_invariant_call_memoize {
                pipeline.add(LoopCallMemoizePass::new_invariant_calls());
            }
            if options.enable_regional_global_scalar_promotion {
                pipeline.add(RegionalGlobalScalarPass::new());
            }
            if options.enable_constant_address_count_reduction {
                pipeline.add(ConstantAddressReductionPass::new());
            }
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
            if options.enable_regional_global_scalar_promotion {
                pipeline.add(RegionalGlobalScalarPass::new());
            }
            pipeline.add(DcePass::new());
            pipeline.add(CsePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(InstCombinePass::new());
            pipeline.add(CsePass::new());
            pipeline.add(DcePass::new());
            pipeline.add(LocalMemzeroSinkPass::new());
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
