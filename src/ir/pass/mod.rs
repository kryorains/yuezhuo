mod const_fold;
mod dce;
mod local_forward;
mod scalar_promote;
mod simplify_cfg;
mod util;

use super::Module;
use const_fold::ConstFoldPass;
use dce::DcePass;
use local_forward::LocalForwardPass;
use scalar_promote::ScalarPromotePass;
use simplify_cfg::SimplifyCfgPass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
}

pub fn run_pipeline(module: &mut Module, opt_level: OptLevel) {
    let mut pipeline = PassPipeline::new();
    match opt_level {
        OptLevel::O0 => {}
        OptLevel::O1 => {
            // Keep this centralized so passes can be repeated or reordered without
            // hiding pipeline policy inside codegen.
            pipeline.add(ConstFoldPass::new());
            pipeline.add(SimplifyCfgPass::new());
            pipeline.add(ScalarPromotePass::new());
            pipeline.add(LocalForwardPass::new());
            pipeline.add(ConstFoldPass::new());
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
        for pass in &mut self.passes {
            pass.run(module);
        }
    }
}

trait ModulePass {
    fn run(&mut self, module: &mut Module);
}
