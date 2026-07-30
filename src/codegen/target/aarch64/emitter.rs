use super::imm::mov_x_imm;
use super::phi_regs::AArch64PhiRegs;
use super::{AArch64IrEmitter, AArch64IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, entry_early_return, ir_align_to, loop_rotated_block_order, IrFuncLayout,
    IrModuleCtx,
};
use crate::ir::{Function, Module};

pub(super) fn emit_asm(module: &Module) -> String {
    AArch64IrEmitter::new(module).emit()
}

impl<'a> AArch64IrEmitter<'a> {
    fn new(module: &'a Module) -> Self {
        Self {
            ctx: IrModuleCtx::new(module),
            out: String::new(),
        }
    }

    fn emit(mut self) -> String {
        self.out
            .push_str(&emit_ir_data_section(self.ctx.module, ".word"));
        self.out.push_str(".text\n");
        for func in &self.ctx.module.funcs {
            AArch64IrFuncEmitter::new(&mut self, func).emit();
        }
        self.out
    }
}

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    fn new(parent: &'a mut AArch64IrEmitter<'b>, func: &'b Function) -> Self {
        let phi_regs = AArch64PhiRegs::new(func);
        let float_regs = super::float_regs::AArch64FloatRegs::new(func);
        let saved_slot_count = phi_regs.saved_regs().len() + float_regs.used_callee_saved().len();
        let saved_area_size = ir_align_to((saved_slot_count as i32) * 8, 16);
        Self {
            parent,
            func,
            layout: IrFuncLayout::new(func),
            phi_regs,
            float_regs,
            saved_area_size,
            local_regs: crate::codegen::common::IrLocalRegs::new(
                func,
                &["x3", "x4", "x5", "x6", "x7"],
                false,
            ),
            value_use_counts: crate::codegen::common::ir_value_use_counts(func),
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
        }
    }

    fn emit(mut self) {
        let early_return = entry_early_return(self.func).and_then(|plan| {
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
                self.emit_terminator(
                    block_idx,
                    terminator,
                    block_order.get(order_idx + 1).copied(),
                );
            }
        }
        let setup = self.body[..setup_end].to_string();
        let blocks = self.body[setup_end..].to_string();
        let saved_regs = self.phi_regs.saved_regs().to_vec();
        let saved_float_regs = self.float_regs.used_callee_saved().to_vec();
        let stack_size = ir_align_to(self.layout.stack_size + self.saved_area_size, 16);
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
        for (idx, reg) in saved_float_regs.iter().enumerate() {
            let offset = (saved_regs.len() + idx) * 8;
            self.parent
                .out
                .push_str(&format!("  str {}, [sp, #{}]\n", d_reg_name(reg), offset));
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
        for (idx, reg) in saved_float_regs.iter().enumerate() {
            let offset = (saved_regs.len() + idx) * 8;
            self.parent
                .out
                .push_str(&format!("  ldr {}, [sp, #{}]\n", d_reg_name(reg), offset));
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

fn d_reg_name(s_reg: &str) -> String {
    s_reg.replacen('s', "d", 1)
}
