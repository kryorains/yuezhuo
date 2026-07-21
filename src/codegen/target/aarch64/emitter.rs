use super::imm::mov_x_imm;
use super::phi_regs::AArch64PhiRegs;
use super::{AArch64IrEmitter, AArch64IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, entry_early_return, ir_align_to, loop_rotated_block_order, IrFuncLayout,
    IrModuleCtx,
};
use crate::ir::{AArch64ThreadPlan, Function, FunctionId, Module};

pub(super) fn emit_asm(module: &Module) -> String {
    AArch64IrEmitter::new(module).emit()
}

impl<'a> AArch64IrEmitter<'a> {
    fn new(module: &'a Module) -> Self {
        let thread_plans = module
            .aarch64_thread_plans
            .iter()
            .filter(|plan| super::thread::valid_plan(module, plan))
            .cloned()
            .collect();
        Self {
            ctx: IrModuleCtx::new(module),
            thread_plans,
            out: String::new(),
        }
    }

    fn emit(mut self) -> String {
        self.out
            .push_str(&emit_ir_data_section(self.ctx.module, ".word"));
        self.out
            .push_str(&super::thread::emit_contexts(&self.thread_plans));
        self.out.push_str(".text\n");
        for function_idx in 0..self.ctx.module.funcs.len() {
            let func = &self.ctx.module.funcs[function_idx];
            let plan = self
                .thread_plans
                .iter()
                .find(|plan| plan.parent == FunctionId(function_idx))
                .cloned();
            AArch64IrFuncEmitter::new(&mut self, func, plan).emit();
        }
        self.out.push_str(&super::thread::emit_workers(
            self.ctx.module,
            &self.thread_plans,
        ));
        self.out
    }
}

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    fn new(
        parent: &'a mut AArch64IrEmitter<'b>,
        func: &'b Function,
        thread_plan: Option<AArch64ThreadPlan>,
    ) -> Self {
        Self {
            parent,
            func,
            layout: IrFuncLayout::new(func),
            phi_regs: AArch64PhiRegs::new(func, thread_plan.is_some()),
            local_regs: crate::codegen::common::IrLocalRegs::new(
                func,
                &["x3", "x4", "x5", "x6", "x7"],
                false,
            ),
            value_use_counts: crate::codegen::common::ir_value_use_counts(func),
            thread_plan,
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
        }
    }

    fn emit(mut self) {
        let early_return = self
            .thread_plan
            .is_none()
            .then(|| entry_early_return(self.func))
            .flatten()
            .and_then(|plan| {
                self.pre_prologue_early_return(&plan)
                    .map(|prelude| (plan, prelude))
            });
        self.emit_params();
        if let Some((plan, _)) = &early_return {
            self.emit_phi_copies(self.func.entry.0, plan.slow_block.0);
            self.body
                .push_str(&format!("  b {}\n", self.block_label(plan.slow_block.0)));
        }
        let setup_end = self.body.len();
        let block_order = loop_rotated_block_order(self.func)
            .into_iter()
            .filter(|block_idx| {
                !early_return.as_ref().is_some_and(|(plan, _)| {
                    *block_idx == self.func.entry.0 || *block_idx == plan.fast_block.0
                })
            })
            .collect::<Vec<_>>();
        for (order_idx, block_idx) in block_order.iter().copied().enumerate() {
            let block = &self.func.blocks[block_idx];
            self.body
                .push_str(&format!("{}:\n", self.block_label(block_idx)));
            for inst in &block.insts {
                self.emit_inst(inst);
            }
            if let Some(terminator) = &block.terminator {
                if self
                    .thread_plan
                    .as_ref()
                    .is_some_and(|plan| plan.preheader.0 == block_idx)
                {
                    let plan = self.thread_plan.clone().unwrap();
                    self.emit_thread_dispatch(&plan);
                } else {
                    self.emit_terminator(
                        block_idx,
                        terminator,
                        block_order.get(order_idx + 1).copied(),
                    );
                }
            }
        }
        let setup = self.body[..setup_end].to_string();
        let blocks = self.body[setup_end..].to_string();
        let saved_regs = self.phi_regs.saved_regs().to_vec();
        let saved_bytes = (saved_regs.len() as i32) * 8;
        let stack_size = ir_align_to(self.layout.stack_size + saved_bytes, 16);
        self.parent.out.push_str(&format!(
            ".globl {0}\n.type {0}, %function\n{0}:\n",
            self.func.name
        ));
        if let Some((_, prelude)) = &early_return {
            self.parent.out.push_str(prelude);
        }
        self.parent
            .out
            .push_str("  stp x29, x30, [sp, #-16]!\n  mov x29, sp\n");
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&mov_x_imm("x16", stack_size as i64));
            self.parent.out.push_str("  sub sp, sp, x16\n");
        }
        let mut saved_idx = 0;
        while saved_idx + 1 < saved_regs.len() {
            self.parent.out.push_str(&format!(
                "  stp {}, {}, [sp, #{}]\n",
                saved_regs[saved_idx],
                saved_regs[saved_idx + 1],
                saved_idx * 8
            ));
            saved_idx += 2;
        }
        if let Some(reg) = saved_regs.get(saved_idx) {
            self.parent
                .out
                .push_str(&format!("  str {}, [sp, #{}]\n", reg, saved_idx * 8));
        }
        self.parent.out.push_str(&setup);
        self.parent.out.push_str(&blocks);
        self.parent
            .out
            .push_str(&format!("{}:\n", self.return_label));
        let mut saved_idx = 0;
        while saved_idx + 1 < saved_regs.len() {
            self.parent.out.push_str(&format!(
                "  ldp {}, {}, [sp, #{}]\n",
                saved_regs[saved_idx],
                saved_regs[saved_idx + 1],
                saved_idx * 8
            ));
            saved_idx += 2;
        }
        if let Some(reg) = saved_regs.get(saved_idx) {
            self.parent
                .out
                .push_str(&format!("  ldr {}, [sp, #{}]\n", reg, saved_idx * 8));
        }
        if stack_size != 0 {
            self.parent
                .out
                .push_str(&mov_x_imm("x16", stack_size as i64));
            self.parent.out.push_str("  add sp, sp, x16\n");
        }
        self.parent
            .out
            .push_str("  ldp x29, x30, [sp], #16\n  ret\n\n");
    }
}
