use super::{Riscv64IrEmitter, Riscv64IrFuncEmitter};
use crate::codegen::common::{
    emit_ir_data_section, entry_early_return, ir_align_to, loop_rotated_block_order,
    natural_loop_depths, IrFuncLayout, IrModuleCtx,
};
use crate::ir::{BinaryOp, BlockId, Const, Function, InstKind, Module, Terminator, ValueKind};
use std::collections::{HashMap, HashSet};

pub(super) fn emit_asm(module: &Module) -> String {
    Riscv64IrEmitter::new(module).emit()
}

impl<'a> Riscv64IrEmitter<'a> {
    fn new(module: &'a Module) -> Self {
        Self {
            ctx: IrModuleCtx::new(module),
            int_return_ranges: crate::ir::int_range::collect_function_return_ranges(module),
            out: String::new(),
        }
    }

    fn emit(mut self) -> String {
        self.out
            .push_str(&emit_ir_data_section(self.ctx.module, ".word"));
        self.out.push_str(".text\n");
        for func in &self.ctx.module.funcs {
            Riscv64IrFuncEmitter::new(&mut self, func).emit();
        }
        self.out
    }
}

fn deepest_loop_targets(func: &Function, block_order: &[usize]) -> Vec<bool> {
    const MAX_ALIGNMENT_BLOCKS: usize = 1024;
    const MAX_ALIGNED_TARGETS: usize = 8;

    let block_count = func.blocks.len();
    let mut aligned = vec![false; block_count];
    if block_count == 0 || block_count > MAX_ALIGNMENT_BLOCKS {
        return aligned;
    }
    let depths = natural_loop_depths(func);
    let Some(max_depth) = depths.iter().copied().max().filter(|depth| *depth > 0) else {
        return aligned;
    };
    let mut positions = vec![usize::MAX; block_count];
    for (position, block) in block_order.iter().copied().enumerate() {
        positions[block] = position;
    }
    let mut count = 0usize;
    for source in block_order.iter().copied() {
        for target in terminator_targets(func.blocks[source].terminator.as_ref()) {
            if positions[target.0] == usize::MAX
                || positions[source] <= positions[target.0]
                || depths[target.0] != max_depth
            {
                continue;
            }
            if positions[target.0] > 0 {
                let previous = block_order[positions[target.0] - 1];
                if terminator_targets(func.blocks[previous].terminator.as_ref()).contains(&target) {
                    continue;
                }
            }
            if !aligned[target.0] {
                count += 1;
                if count > MAX_ALIGNED_TARGETS {
                    return vec![false; block_count];
                }
                aligned[target.0] = true;
            }
        }
    }
    aligned
}

fn terminator_targets(terminator: Option<&Terminator>) -> Vec<BlockId> {
    match terminator {
        Some(Terminator::Jump(target)) => vec![*target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) if then_target == else_target => vec![*then_target],
        Some(Terminator::Branch {
            then_target,
            else_target,
            ..
        }) => vec![*then_target, *else_target],
        Some(Terminator::Return(_)) | None => Vec::new(),
    }
}

fn collect_division_magic_regs(
    func: &Function,
    int_ranges: &[Option<crate::ir::int_range::IntRange>],
    occupied: &HashSet<&'static str>,
) -> (HashMap<i32, &'static str>, Vec<&'static str>) {
    if func.blocks.iter().any(|block| {
        block
            .insts
            .iter()
            .any(|inst| matches!(&inst.kind, InstKind::Call { name, .. } if name == &func.name))
    }) {
        // Reserving another callee-saved register is paid at every recursive
        // invocation. Leave magic constants rematerializable in recursive
        // functions instead of extending every frame for a local loop saving.
        return (HashMap::new(), Vec::new());
    }
    let use_scores = crate::codegen::common::weighted_use_scores(func);
    let mut scores = HashMap::<i32, usize>::new();
    for block in &func.blocks {
        for inst in &block.insts {
            let InstKind::Binary { op, lhs, rhs, .. } = inst.kind else {
                continue;
            };
            if !matches!(op, BinaryOp::Idiv | BinaryOp::Imod) {
                continue;
            }
            let ValueKind::Const(Const::Int(divisor)) = func.value(rhs).kind else {
                continue;
            };
            let magnitude = divisor.unsigned_abs();
            if magnitude <= 1 || magnitude > i32::MAX as u32 || magnitude.is_power_of_two() {
                continue;
            }
            let bound = i64::from(magnitude);
            if op == BinaryOp::Imod
                && int_ranges[lhs.0].is_some_and(|range| {
                    range.min > -bound && range.max < bound
                        || range.min > -2 * bound && range.max < 2 * bound
                })
            {
                continue;
            }
            let magic = crate::codegen::common::signed_magic_positive(magnitude).multiplier;
            *scores.entry(magic).or_default() = scores
                .get(&magic)
                .copied()
                .unwrap_or_default()
                .saturating_add(use_scores[inst.result.unwrap().0]);
        }
    }
    let mut candidates = scores
        .into_iter()
        .filter(|(_, score)| *score >= 16)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(magic, score)| (std::cmp::Reverse(*score), *magic));
    let available = [
        "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    ]
    .into_iter()
    .filter(|reg| !occupied.contains(reg))
    .collect::<Vec<_>>();
    let assignments = candidates
        .into_iter()
        .zip(available)
        .map(|((magic, _), reg)| (magic, reg))
        .collect::<HashMap<_, _>>();
    let mut saved = assignments.values().copied().collect::<Vec<_>>();
    saved.sort_by_key(|reg| {
        [
            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        ]
        .iter()
        .position(|candidate| candidate == reg)
        .unwrap()
    });
    (assignments, saved)
}

impl<'a, 'b> Riscv64IrFuncEmitter<'a, 'b> {
    fn new(parent: &'a mut Riscv64IrEmitter<'b>, func: &'b Function) -> Self {
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
        let elided_values = allocation_view
            .as_mut()
            .map(super::memory::eliminate_dead_folded_address_values)
            .unwrap_or_default();
        let allocation_func = allocation_view.as_ref().unwrap_or(func);
        let regalloc = super::regalloc::Riscv64RegAlloc::new(allocation_func);
        let float_regalloc = super::regalloc::Riscv64FloatRegAlloc::new(allocation_func);
        let occupied_int_regs = regalloc.used_regs().iter().copied().collect::<HashSet<_>>();
        let (division_magic_regs, division_magic_saved_regs) =
            collect_division_magic_regs(allocation_func, &int_ranges, &occupied_int_regs);
        let local_regs = crate::codegen::common::IrLocalRegs::new(
            allocation_func,
            regalloc.local_regs(),
            true,
            |value| regalloc.reg(value).is_some(),
        );
        let layout = IrFuncLayout::new_with_stack_slots(func, |value| {
            !elided_values.contains(&value)
                && !folded_memory_geps.contains_key(&value)
                && regalloc.reg(value).is_none()
                && float_regalloc.reg(value).is_none()
                && local_regs.reg(value).is_none()
        });
        let saved_slot_count = regalloc.used_regs().len()
            + division_magic_saved_regs.len()
            + float_regalloc.used_callee_saved().len();
        let saved_area_size = ir_align_to((saved_slot_count as i32) * 8, 16);
        Self {
            parent,
            func,
            layout,
            regalloc,
            float_regalloc,
            division_magic_regs,
            division_magic_saved_regs,
            saved_area_size,
            local_regs,
            value_use_counts,
            int_ranges,
            folded_memory_geps,
            elided_values,
            frame_accessed: false,
            body: String::new(),
            return_label: format!(".L_return_{}", func.name),
        }
    }

    fn emit(mut self) {
        let early_return = entry_early_return(self.func).and_then(|plan| {
            let prelude = self.pre_prologue_early_return(&plan)?;
            Some((plan, prelude))
        });
        self.emit_params();
        self.emit_regalloc_materializations();
        let mut division_magic = self
            .division_magic_regs
            .iter()
            .map(|(magic, reg)| (*reg, *magic))
            .collect::<Vec<_>>();
        division_magic.sort_by_key(|(reg, _)| *reg);
        for (reg, magic) in division_magic {
            self.body.push_str(&format!("  li {reg}, {magic}\n"));
        }
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
                .map_or(plan.guard_block, |chain| chain.guard_block);
            if plan.guard_block != self.func.entry {
                self.emit_phi_copies(self.func.entry.0, plan.guard_block.0);
                for inst in &self.func.blocks[plan.guard_block.0].insts {
                    self.emit_inst(inst);
                }
                self.emit_phi_copies(plan.guard_block.0, plan.slow_block.0);
            } else {
                self.emit_phi_copies(slow_predecessor.0, plan.slow_block.0);
            }
            self.body
                .push_str(&format!("  j {}\n", self.block_label(plan.slow_block.0)));
        }
        let setup_end = self.body.len();
        let block_order = loop_rotated_block_order(self.func)
            .into_iter()
            .filter(|block_idx| {
                !early_return.as_ref().is_some_and(|(plan, _)| {
                    *block_idx == self.func.entry.0
                        || (*block_idx == plan.guard_block.0 && plan.guard_block == self.func.entry)
                        || plan
                            .chained
                            .is_some_and(|chain| *block_idx == chain.guard_block.0)
                        || (plan.fast_block_exclusive && *block_idx == plan.fast_block.0)
                })
            })
            .collect::<Vec<_>>();
        let aligned_loop_targets = deepest_loop_targets(self.func, &block_order);
        for (order_idx, block_idx) in block_order.iter().copied().enumerate() {
            let block = &self.func.blocks[block_idx];
            if aligned_loop_targets[block_idx] {
                self.body.push_str(".p2align 4\n");
            }
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
        let saved_int_regs = self.regalloc.used_regs().to_vec();
        let saved_magic_regs = self.division_magic_saved_regs.clone();
        let saved_float_regs = self.float_regalloc.used_callee_saved().to_vec();
        let stack_size = ir_align_to(self.layout.stack_size + self.saved_area_size, 16);
        let leaf = !self.func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                matches!(
                    inst.kind,
                    crate::ir::InstKind::Call { .. }
                        | crate::ir::InstKind::MemZero { count: Some(_), .. }
                        | crate::ir::InstKind::MemCopy { .. }
                )
            })
        });
        let has_stack_params = {
            let sig = self.parent.ctx.funcs.get(&self.func.name).unwrap();
            super::abi::assign_riscv_arg_locations(&sig.params)
                .into_iter()
                .any(|location| matches!(location, crate::codegen::common::IrArgLocation::Stack))
        };
        // Stack-slot assignment happens before instruction emission. Register
        // allocation and address folding can make every assigned slot dead, so
        // decide whether a leaf needs a frame from the emitted accesses rather
        // than the conservative layout size.
        let frameless_leaf = leaf
            && !self.frame_accessed
            && saved_int_regs.is_empty()
            && saved_magic_regs.is_empty()
            && saved_float_regs.is_empty()
            && !has_stack_params;
        let recursive = self.func.blocks.iter().any(|block| {
            block.insts.iter().any(|inst| {
                matches!(&inst.kind, crate::ir::InstKind::Call { name, .. } if name == &self.func.name)
            })
        });
        // Keep large/recursive hot functions on a stable fetch boundary. Small
        // functions use a modest alignment to limit padding growth.
        let function_alignment = if self.func.blocks.len() >= 16 || recursive {
            7
        } else {
            4
        };
        self.parent.out.push_str(&format!(
            ".p2align {1}\n.globl {0}\n.type {0}, @function\n{0}:\n",
            self.func.name, function_alignment
        ));
        if let Some((_, prelude)) = &early_return {
            self.parent.out.push_str(prelude);
        }
        if !frameless_leaf {
            self.parent.out.push_str("  addi sp, sp, -16\n");
            if !leaf {
                self.parent.out.push_str("  sd ra, 8(sp)\n");
            }
            self.parent.out.push_str("  sd s0, 0(sp)\n  mv s0, sp\n");
            if stack_size != 0 {
                if stack_size <= 2048 {
                    self.parent
                        .out
                        .push_str(&format!("  addi sp, sp, -{}\n", stack_size));
                } else {
                    self.parent
                        .out
                        .push_str(&format!("  li t6, {}\n  sub sp, sp, t6\n", stack_size));
                }
            }
        }
        for (idx, reg) in saved_int_regs.iter().enumerate() {
            self.parent
                .out
                .push_str(&format!("  sd {}, -{}(s0)\n", reg, (idx + 1) * 8));
        }
        for (idx, reg) in saved_magic_regs.iter().enumerate() {
            let slot = saved_int_regs.len() + idx + 1;
            self.parent
                .out
                .push_str(&format!("  sd {}, -{}(s0)\n", reg, slot * 8));
        }
        for (idx, reg) in saved_float_regs.iter().enumerate() {
            let slot = saved_int_regs.len() + saved_magic_regs.len() + idx + 1;
            self.parent
                .out
                .push_str(&format!("  fsd {}, -{}(s0)\n", reg, slot * 8));
        }
        self.parent.out.push_str(&setup);
        self.parent.out.push_str(&blocks);
        self.parent
            .out
            .push_str(&format!("{}:\n", self.return_label));
        if frameless_leaf {
            self.parent.out.push_str("  ret\n\n");
            return;
        }
        for (idx, reg) in saved_int_regs.iter().enumerate() {
            self.parent
                .out
                .push_str(&format!("  ld {}, -{}(s0)\n", reg, (idx + 1) * 8));
        }
        for (idx, reg) in saved_magic_regs.iter().enumerate() {
            let slot = saved_int_regs.len() + idx + 1;
            self.parent
                .out
                .push_str(&format!("  ld {}, -{}(s0)\n", reg, slot * 8));
        }
        for (idx, reg) in saved_float_regs.iter().enumerate() {
            let slot = saved_int_regs.len() + saved_magic_regs.len() + idx + 1;
            self.parent
                .out
                .push_str(&format!("  fld {}, -{}(s0)\n", reg, slot * 8));
        }
        self.parent.out.push_str("  mv sp, s0\n");
        if !leaf {
            self.parent.out.push_str("  ld ra, 8(sp)\n");
        }
        self.parent
            .out
            .push_str("  ld s0, 0(sp)\n  addi sp, sp, 16\n  ret\n\n");
    }
}
