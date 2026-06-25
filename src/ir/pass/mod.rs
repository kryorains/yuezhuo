mod dce;

use super::Module;
use dce::DcePass;

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
