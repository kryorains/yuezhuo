mod const_fold;
mod cse;
mod dce;
mod dominators;
mod licm;
mod local_forward;
mod scalar_promote;
mod simplify_cfg;
mod tail_recursion;
mod util;

use super::Module;
use const_fold::ConstFoldPass;
use cse::CsePass;
use dce::DcePass;
use licm::LicmPass;
use local_forward::LocalForwardPass;
use scalar_promote::ScalarPromotePass;
use simplify_cfg::SimplifyCfgPass;
use tail_recursion::TailRecursionPass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
}

pub fn run_pipeline(module: &mut Module, opt_level: OptLevel) {
    // 所有优化 pass 都在这里排队，方便统一调整执行顺序。
    let mut pipeline = PassPipeline::new();
    match opt_level {
        OptLevel::O0 => {}
        OptLevel::O1 => {
            // 先折叠常量和死分支，再做标量提升/局部转发，最后再清一次新产生的机会。
            pipeline.add(ConstFoldPass::new());
            pipeline.add(SimplifyCfgPass::new());
            pipeline.add(TailRecursionPass::new());
            pipeline.add(ScalarPromotePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(CsePass::new());
            pipeline.add(LicmPass::new());
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
