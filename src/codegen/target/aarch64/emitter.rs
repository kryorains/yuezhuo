use super::imm::{mov_w_imm, mov_x_imm};
use super::phi_regs::AArch64PhiRegs;
use super::{AArch64IrEmitter, AArch64IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, entry_early_return, ir_align_to, loop_rotated_block_order, IrFuncLayout,
    IrLocalRegs, IrModuleCtx,
};
use crate::ir::{
    BinaryOp, CastOp, Const, Function, InstKind, Module, Type, UnaryOp, ValueId, ValueKind,
};
use std::collections::HashSet;

pub(super) fn emit_asm(module: &Module) -> String {
    AArch64IrEmitter::new(module).emit()
}

impl<'a> AArch64IrEmitter<'a> {
    fn new(module: &'a Module) -> Self {
        Self {
            ctx: IrModuleCtx::new(module),
            int_return_ranges: crate::ir::int_range::collect_function_return_ranges(module),
            nonnegative_globals: collect_nonnegative_globals(module),
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
        let value_use_counts = crate::codegen::common::ir_value_use_counts(func);
        let int_ranges =
            crate::ir::int_range::collect_value_ranges(func, &parent.int_return_ranges);
        let folded_memory_geps = super::memory::collect_folded_memory_geps(func, &value_use_counts);
        let mut allocation_view = (!folded_memory_geps.is_empty()).then(|| func.clone());
        if let Some(allocation_view) = &mut allocation_view {
            super::memory::rewrite_folded_memory_uses_for_allocation(
                allocation_view,
                &folded_memory_geps,
            );
        }
        let allocation_func = allocation_view.as_ref().unwrap_or(func);
        let phi_regs = AArch64PhiRegs::new(allocation_func);
        let float_regs = super::float_regs::AArch64FloatRegs::new(allocation_func);
        let local_regs = IrLocalRegs::new(allocation_func, &["x3", "x4", "x5", "x6", "x7"], false);
        let additive_divisor = additive_remainder_divisor(func);
        let cached_i32_constant = additive_divisor.zip(phi_regs.unused_callee_saved(0));
        let cached_i32_double_constant = additive_divisor
            .and_then(|divisor| divisor.checked_mul(2))
            .zip(phi_regs.unused_callee_saved(1));
        let layout = IrFuncLayout::new_with_stack_slots(func, |value| {
            phi_regs.reg(value).is_none()
                && float_regs.reg(value).is_none()
                && local_regs.reg(value).is_none()
                && !folded_memory_geps.contains_key(&value)
        });
        let saved_slot_count = phi_regs.saved_regs().len()
            + usize::from(cached_i32_constant.is_some())
            + usize::from(cached_i32_double_constant.is_some())
            + float_regs.used_callee_saved().len();
        let saved_area_size = ir_align_to((saved_slot_count as i32) * 8, 16);
        Self {
            parent,
            func,
            layout,
            phi_regs,
            float_regs,
            saved_area_size,
            local_regs,
            value_use_counts,
            int_ranges,
            folded_memory_geps,
            cached_i32_constant,
            cached_i32_double_constant,
            frame_accessed: false,
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
            for inst in &self.func.blocks[self.func.entry.0].insts {
                self.emit_inst(inst);
            }
            if let Some(chain) = plan.chained {
                for inst in &self.func.blocks[chain.guard_block.0].insts {
                    self.emit_inst(inst);
                }
            }
            let slow_predecessor = plan
                .chained
                .map_or(self.func.entry, |chain| chain.guard_block);
            self.emit_phi_copies(slow_predecessor.0, plan.slow_block.0);
            self.body
                .push_str(&format!("  b {}\n", self.block_label(plan.slow_block.0)));
        }
        let setup_end = self.body.len();
        let mut block_order = loop_rotated_block_order(self.func)
            .into_iter()
            .filter(|block_idx| {
                !early_return.as_ref().is_some_and(|(plan, _)| {
                    *block_idx == self.func.entry.0
                        || plan
                            .chained
                            .is_some_and(|chain| *block_idx == chain.guard_block.0)
                        || (plan.fast_block_exclusive && *block_idx == plan.fast_block.0)
                })
            })
            .collect::<Vec<_>>();
        let mut order_idx = 0;
        while order_idx < block_order.len() {
            let block_idx = block_order[order_idx];
            let final_merge = match self.func.blocks[block_idx].terminator.as_ref() {
                Some(crate::ir::Terminator::Branch {
                    cond,
                    then_target,
                    else_target,
                }) => self.safe_branchless_phi_final_merge(
                    block_idx,
                    *cond,
                    *then_target,
                    *else_target,
                ),
                _ => None,
            };
            if let Some(final_merge) = final_merge {
                if let Some(merge_idx) = block_order
                    .iter()
                    .position(|candidate| *candidate == final_merge.0)
                    .filter(|merge_idx| *merge_idx > order_idx + 1)
                {
                    let final_merge = block_order.remove(merge_idx);
                    block_order.insert(order_idx + 1, final_merge);
                }
                if let Some(crate::ir::Terminator::Jump(successor)) =
                    self.func.blocks[final_merge.0].terminator.as_ref()
                {
                    if let Some(successor_idx) = block_order
                        .iter()
                        .position(|candidate| *candidate == successor.0)
                        .filter(|successor_idx| *successor_idx > order_idx + 2)
                    {
                        let successor = block_order.remove(successor_idx);
                        block_order.insert(order_idx + 2, successor);
                    }

                    let loop_exit = match self.func.blocks[successor.0].terminator.as_ref() {
                        Some(crate::ir::Terminator::Branch {
                            then_target,
                            else_target,
                            ..
                        }) if then_target.0 == block_idx => Some(*else_target),
                        Some(crate::ir::Terminator::Branch {
                            then_target,
                            else_target,
                            ..
                        }) if else_target.0 == block_idx => Some(*then_target),
                        _ => None,
                    };
                    if let Some(loop_exit) = loop_exit {
                        if let Some(exit_idx) = block_order
                            .iter()
                            .position(|candidate| *candidate == loop_exit.0)
                            .filter(|exit_idx| *exit_idx > order_idx + 3)
                        {
                            let loop_exit = block_order.remove(exit_idx);
                            block_order.insert(order_idx + 3, loop_exit);
                        }
                    }
                }
            }
            order_idx += 1;
        }
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
        let mut saved_regs = self.phi_regs.saved_regs().to_vec();
        if let Some((_, reg)) = self.cached_i32_constant {
            saved_regs.push(reg);
        }
        if let Some((_, reg)) = self.cached_i32_double_constant {
            saved_regs.push(reg);
        }
        let saved_float_regs = self.float_regs.used_callee_saved().to_vec();
        let spill_area_size = if self.frame_accessed {
            self.layout.stack_size
        } else {
            0
        };
        let stack_size = ir_align_to(spill_area_size + self.saved_area_size, 16);
        let has_calls = self.func.blocks.iter().any(|block| {
            block
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, InstKind::Call { .. }))
        });
        let needs_frame_record = stack_size != 0 || has_calls || self.frame_accessed;
        self.parent.out.push_str(&format!(
            ".globl {0}\n.type {0}, %function\n{0}:\n",
            self.func.name
        ));
        if let Some((_, prelude)) = &early_return {
            self.parent.out.push_str(prelude);
        }
        if needs_frame_record {
            self.parent
                .out
                .push_str("  stp x29, x30, [sp, #-16]!\n  mov x29, sp\n");
        }
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
        if let Some((value, reg)) = self.cached_i32_constant {
            self.parent
                .out
                .push_str(&mov_w_imm(&reg.replacen('x', "w", 1), value));
        }
        if let Some((value, reg)) = self.cached_i32_double_constant {
            self.parent
                .out
                .push_str(&mov_w_imm(&reg.replacen('x', "w", 1), value));
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
        if needs_frame_record {
            self.parent.out.push_str("  ldp x29, x30, [sp], #16\n");
        }
        self.parent.out.push_str("  ret\n\n");
    }
}

fn d_reg_name(s_reg: &str) -> String {
    s_reg.replacen('s', "d", 1)
}

fn additive_remainder_divisor(func: &Function) -> Option<i32> {
    for block in &func.blocks {
        for inst in &block.insts {
            let (
                Some(remainder),
                InstKind::Binary {
                    op: BinaryOp::Imod,
                    lhs: dividend,
                    rhs,
                },
            ) = (inst.result, &inst.kind)
            else {
                continue;
            };
            let ValueKind::Const(Const::Int(divisor)) = func.value(*rhs).kind else {
                continue;
            };
            if divisor < 1024 || divisor.checked_mul(2).is_none() {
                continue;
            }
            let ValueKind::Inst(add_block, add_idx) = func.value(*dividend).kind else {
                continue;
            };
            let Some(InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            }) = func
                .blocks
                .get(add_block.0)
                .and_then(|block| block.insts.get(add_idx))
                .map(|inst| &inst.kind)
            else {
                continue;
            };
            if func.blocks.iter().any(|block| {
                block.insts.iter().any(|inst| {
                    let InstKind::Phi { incomings } = &inst.kind else {
                        return false;
                    };
                    inst.result.is_some_and(|phi| *lhs == phi || *rhs == phi)
                        && incomings.iter().any(|(_, incoming)| *incoming == remainder)
                })
            }) {
                return Some(divisor);
            }
        }
    }
    None
}

fn collect_nonnegative_globals(module: &Module) -> HashSet<String> {
    let mut candidates = module
        .globals
        .iter()
        .filter(|global| {
            global.ty == Type::I32
                && global.init.as_ref().is_some_and(|init| match init {
                    Const::Int(value) => *value >= 0,
                    Const::Bool(_) | Const::Zero(Type::I32 | Type::I1) => true,
                    _ => false,
                })
        })
        .map(|global| global.name.clone())
        .collect::<HashSet<_>>();

    loop {
        let mut rejected = HashSet::new();
        for func in &module.funcs {
            let global_values = func
                .values
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| match &value.kind {
                    ValueKind::Global(name) if candidates.contains(name) => {
                        Some((ValueId(idx), name.as_str()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            for block in &func.blocks {
                for inst in &block.insts {
                    for (global, name) in &global_values {
                        if !inst_operands_for_global_analysis(&inst.kind).contains(global) {
                            continue;
                        }
                        match inst.kind {
                            InstKind::Load { ptr } if ptr == *global => {}
                            InstKind::Store { ptr, value } if ptr == *global => {
                                if !value_is_nonnegative(
                                    func,
                                    value,
                                    &candidates,
                                    &mut HashSet::new(),
                                ) {
                                    rejected.insert((*name).to_owned());
                                }
                            }
                            _ => {
                                rejected.insert((*name).to_owned());
                            }
                        }
                    }
                }
            }
        }
        if rejected.is_empty() {
            return candidates;
        }
        candidates.retain(|name| !rejected.contains(name));
    }
}

pub(super) fn value_is_nonnegative(
    func: &Function,
    value: ValueId,
    nonnegative_globals: &HashSet<String>,
    visiting: &mut HashSet<ValueId>,
) -> bool {
    match &func.value(value).kind {
        ValueKind::Const(Const::Int(value)) => return *value >= 0,
        ValueKind::Const(Const::Bool(_)) | ValueKind::Const(Const::Zero(Type::I32 | Type::I1)) => {
            return true;
        }
        _ => {}
    }
    if !visiting.insert(value) {
        return true;
    }
    let proven = match func.value(value).kind {
        ValueKind::Inst(block, inst_idx) => {
            let Some(inst) = func
                .blocks
                .get(block.0)
                .and_then(|block| block.insts.get(inst_idx))
            else {
                visiting.remove(&value);
                return false;
            };
            match &inst.kind {
                InstKind::Phi { incomings } => incomings.iter().all(|(_, incoming)| {
                    value_is_nonnegative(func, *incoming, nonnegative_globals, visiting)
                }),
                InstKind::Load { ptr } => matches!(
                    &func.value(*ptr).kind,
                    ValueKind::Global(name) if nonnegative_globals.contains(name)
                ),
                InstKind::Icmp { .. }
                | InstKind::Fcmp { .. }
                | InstKind::Unary {
                    op: UnaryOp::Not, ..
                }
                | InstKind::Cast {
                    op: CastOp::BoolToI32 | CastOp::I32ToBool | CastOp::F32ToBool,
                    ..
                } => true,
                InstKind::Binary { op, lhs, rhs } => match op {
                    BinaryOp::Iadd => {
                        [(*lhs, *rhs), (*rhs, *lhs)]
                            .into_iter()
                            .any(|(source, adjustment)| {
                                is_signed_division_adjustment(func, source, adjustment)
                                    && value_is_nonnegative(
                                        func,
                                        source,
                                        nonnegative_globals,
                                        visiting,
                                    )
                            })
                    }
                    BinaryOp::Iand => {
                        is_nonnegative_mask(func, *lhs)
                            || is_nonnegative_mask(func, *rhs)
                            || is_negative_sanitize_value(func, value)
                    }
                    BinaryOp::Ior => {
                        value_is_nonnegative(func, *lhs, nonnegative_globals, visiting)
                            && value_is_nonnegative(func, *rhs, nonnegative_globals, visiting)
                    }
                    BinaryOp::Iashr => {
                        value_is_nonnegative(func, *lhs, nonnegative_globals, visiting)
                    }
                    BinaryOp::Idiv | BinaryOp::Imod => {
                        value_is_nonnegative(func, *lhs, nonnegative_globals, visiting)
                            && matches!(func.value(*rhs).kind, ValueKind::Const(Const::Int(v)) if v > 0)
                    }
                    BinaryOp::And | BinaryOp::Or => true,
                    _ => false,
                },
                _ => false,
            }
        }
        _ => false,
    };
    visiting.remove(&value);
    proven
}

fn is_nonnegative_mask(func: &Function, value: ValueId) -> bool {
    matches!(func.value(value).kind, ValueKind::Const(Const::Int(mask)) if mask >= 0)
}

fn is_signed_division_adjustment(func: &Function, source: ValueId, adjustment: ValueId) -> bool {
    let is_sign = |candidate: ValueId| {
        let ValueKind::Inst(block, inst_idx) = func.value(candidate).kind else {
            return false;
        };
        matches!(
            func.blocks
                .get(block.0)
                .and_then(|block| block.insts.get(inst_idx))
                .map(|inst| &inst.kind),
            Some(InstKind::Binary {
                op: BinaryOp::Iashr,
                lhs,
                rhs,
            }) if *lhs == source
                && matches!(func.value(*rhs).kind, ValueKind::Const(Const::Int(31)))
        )
    };
    if is_sign(adjustment) {
        return true;
    }
    let ValueKind::Inst(block, inst_idx) = func.value(adjustment).kind else {
        return false;
    };
    matches!(
        func.blocks
            .get(block.0)
            .and_then(|block| block.insts.get(inst_idx))
            .map(|inst| &inst.kind),
        Some(InstKind::Binary {
            op: BinaryOp::Iand,
            lhs,
            rhs,
        }) if is_sign(*lhs) || is_sign(*rhs)
    )
}

fn is_negative_sanitize_value(func: &Function, value: ValueId) -> bool {
    let ValueKind::Inst(block, inst_idx) = func.value(value).kind else {
        return false;
    };
    let Some(InstKind::Binary {
        op: BinaryOp::Iand,
        lhs,
        rhs,
    }) = func
        .blocks
        .get(block.0)
        .and_then(|block| block.insts.get(inst_idx))
        .map(|inst| &inst.kind)
    else {
        return false;
    };
    [(*lhs, *rhs), (*rhs, *lhs)]
        .into_iter()
        .any(|(source, mask)| {
            let ValueKind::Inst(mask_block, mask_idx) = func.value(mask).kind else {
                return false;
            };
            let Some(InstKind::Binary {
                op: BinaryOp::Ixor,
                lhs: xor_lhs,
                rhs: xor_rhs,
            }) = func
                .blocks
                .get(mask_block.0)
                .and_then(|block| block.insts.get(mask_idx))
                .map(|inst| &inst.kind)
            else {
                return false;
            };
            let sign = if matches!(func.value(*xor_lhs).kind, ValueKind::Const(Const::Int(-1))) {
                *xor_rhs
            } else if matches!(func.value(*xor_rhs).kind, ValueKind::Const(Const::Int(-1))) {
                *xor_lhs
            } else {
                return false;
            };
            let ValueKind::Inst(sign_block, sign_idx) = func.value(sign).kind else {
                return false;
            };
            matches!(
                func.blocks
                    .get(sign_block.0)
                    .and_then(|block| block.insts.get(sign_idx))
                    .map(|inst| &inst.kind),
                Some(InstKind::Binary {
                    op: BinaryOp::Iashr,
                    lhs,
                    rhs,
                }) if *lhs == source
                    && matches!(func.value(*rhs).kind, ValueKind::Const(Const::Int(31)))
            )
        })
}

fn inst_operands_for_global_analysis(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Nop | InstKind::Alloca { .. } => Vec::new(),
        InstKind::Phi { incomings } => incomings.iter().map(|(_, value)| *value).collect(),
        InstKind::Load { ptr } | InstKind::MemZero { ptr, .. } => vec![*ptr],
        InstKind::Store { ptr, value } => vec![*ptr, *value],
        InstKind::Unary { value, .. } | InstKind::Cast { value, .. } => vec![*value],
        InstKind::Binary { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        InstKind::Gep { base, indices } => {
            let mut operands = vec![*base];
            operands.extend(indices.iter().copied());
            operands
        }
        InstKind::Call { args, .. } => args.clone(),
    }
}
