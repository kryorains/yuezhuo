use super::imm::mov_w_imm;
use super::AArch64IrFuncEmitter;
use crate::ir::{AArch64ThreadCapture, AArch64ThreadPlan, Module, Terminator, Type, ValueKind};

const TID_OFFSET: usize = 0;
const BEGIN_OFFSET: usize = 8;
const END_OFFSET: usize = 12;
const DONE_OFFSET: usize = 16;
const CAPTURE_OFFSET: usize = 24;
const CAPTURE_STRIDE: usize = 8;
const MAX_PARALLEL_THRESHOLD: i32 = 65_536;

pub(super) fn valid_plan(module: &Module, plan: &AArch64ThreadPlan) -> bool {
    let Some(parent) = module.funcs.get(plan.parent.0) else {
        return false;
    };
    let Some(helper) = module.funcs.get(plan.helper.0) else {
        return false;
    };
    if plan.parent == plan.helper
        || helper.verify().is_err()
        || plan.captures.len() > 6
        || parent.values.get(plan.bound.0).map(|value| &value.ty) != Some(&Type::I32)
        || parent.blocks.get(plan.preheader.0).is_none()
        || parent.blocks.get(plan.header.0).is_none()
        || parent.blocks.get(plan.body.0).is_none()
        || parent.blocks.get(plan.exit.0).is_none()
        || !(2..=MAX_PARALLEL_THRESHOLD).contains(&plan.parallel_threshold)
        || parent.blocks[plan.preheader.0].terminator != Some(Terminator::Jump(plan.header))
        || parent.blocks[plan.body.0].terminator != Some(Terminator::Jump(plan.header))
    {
        return false;
    }
    let Some(Terminator::Branch {
        then_target,
        else_target,
        ..
    }) = parent.blocks[plan.header.0].terminator.as_ref()
    else {
        return false;
    };
    if *else_target != plan.exit
        || *then_target == plan.exit
        || parent.blocks.get(then_target.0).is_none()
    {
        return false;
    }
    if plan.captures.iter().any(|capture| {
        !matches!(capture.ty, Type::I32 | Type::Ptr(_))
            || parent
                .values
                .get(capture.value.0)
                .is_none_or(|value| value.ty != capture.ty)
    }) {
        return false;
    }
    if plan.dispatch_setup.iter().any(|value| {
        !matches!(
            parent.values.get(value.0).map(|value| &value.kind),
            Some(ValueKind::Inst(owner, inst_idx))
                if *owner == plan.header
                    && parent.blocks.get(owner.0).and_then(|block| block.insts.get(*inst_idx))
                        .is_some_and(|inst| inst.result == Some(*value))
        )
    }) {
        return false;
    }

    let expected_params = 2usize.saturating_add(plan.captures.len());
    if helper.params.len() != expected_params
        || helper
            .params
            .first()
            .is_none_or(|param| helper.value(*param).ty != Type::I32)
        || helper
            .params
            .get(1)
            .is_none_or(|param| helper.value(*param).ty != Type::I32)
    {
        return false;
    }
    plan.captures.iter().enumerate().all(|(idx, capture)| {
        helper
            .params
            .get(idx + 2)
            .is_some_and(|param| helper.value(*param).ty == capture.ty)
    })
}

pub(super) fn emit_contexts(plans: &[AArch64ThreadPlan]) -> String {
    if plans.is_empty() {
        return String::new();
    }
    let mut out = String::from(".bss\n");
    for plan in plans {
        let size = context_size(plan.captures.len());
        out.push_str(&format!(
            ".balign 8\n.local {0}\n.type {0}, %object\n{0}:\n  .zero {1}\n.size {0}, {1}\n",
            plan.context_symbol, size
        ));
    }
    out
}

pub(super) fn emit_workers(module: &Module, plans: &[AArch64ThreadPlan]) -> String {
    let mut out = String::new();
    for plan in plans {
        let helper = &module.funcs[plan.helper.0];
        out.push_str(&format!(
            ".p2align 2\n.type {worker}, %function\n{worker}:\n  stp x29, x30, [sp, #-16]!\n  mov x29, sp\n  adrp x8, {context}\n  add x8, x8, :lo12:{context}\n  ldr w0, [x8, #{begin}]\n  ldr w1, [x8, #{end}]\n",
            worker = plan.worker_symbol,
            context = plan.context_symbol,
            begin = BEGIN_OFFSET,
            end = END_OFFSET,
        ));
        for (idx, capture) in plan.captures.iter().enumerate() {
            let offset = capture_offset(idx);
            let reg = idx + 2;
            match capture.ty {
                Type::Ptr(_) => out.push_str(&format!("  ldr x{}, [x8, #{}]\n", reg, offset)),
                Type::I32 => out.push_str(&format!("  ldr w{}, [x8, #{}]\n", reg, offset)),
                _ => unreachable!("validated thread capture type"),
            }
        }
        out.push_str(&format!(
            "  bl {helper}\n  adrp x8, {context}\n  add x8, x8, :lo12:{context}\n  add x8, x8, #{done}\n  mov w9, #1\n  stlr w9, [x8]\n  mov x0, xzr\n  ldp x29, x30, [sp], #16\n  ret\n.size {worker}, .-{worker}\n\n",
            helper = helper.name,
            context = plan.context_symbol,
            done = DONE_OFFSET,
            worker = plan.worker_symbol,
        ));
    }
    out
}

impl<'a, 'b> AArch64IrFuncEmitter<'a, 'b> {
    pub(super) fn emit_thread_dispatch(&mut self, plan: &AArch64ThreadPlan) {
        debug_assert_eq!(
            self.func as *const _,
            &self.parent.ctx.module.funcs[plan.parent.0]
        );
        for value in &plan.dispatch_setup {
            let ValueKind::Inst(block, inst_idx) = self.func.value(*value).kind else {
                unreachable!("validated dispatch setup definition");
            };
            let inst = self.func.blocks[block.0].insts[inst_idx].clone();
            self.emit_inst(&inst);
        }

        let fallback = self.parent.ctx.fresh_label("thread_fallback");
        let join_wait = self.parent.ctx.fresh_label("thread_join_wait");
        let parallel_exit = self.parent.ctx.fresh_label("thread_parallel_exit");
        let helper = &self.parent.ctx.module.funcs[plan.helper.0].name;
        self.load_value(plan.bound);
        self.body
            .push_str(&mov_w_imm("w17", plan.parallel_threshold));
        self.body.push_str(&format!(
            "  cmp w0, w17\n  blt {fallback}\n  lsr w17, w0, #1\n  adrp x15, {context}\n  add x15, x15, :lo12:{context}\n  str w17, [x15, #{begin}]\n  str w0, [x15, #{end}]\n",
            fallback = fallback,
            context = plan.context_symbol,
            begin = BEGIN_OFFSET,
            end = END_OFFSET,
        ));
        for (idx, capture) in plan.captures.iter().enumerate() {
            self.load_value(capture.value);
            let offset = capture_offset(idx);
            match capture.ty {
                Type::Ptr(_) => self
                    .body
                    .push_str(&format!("  str x0, [x15, #{}]\n", offset)),
                Type::I32 => self
                    .body
                    .push_str(&format!("  str w0, [x15, #{}]\n", offset)),
                _ => unreachable!("validated thread capture type"),
            }
        }
        self.body.push_str(&format!(
            "  str wzr, [x15, #{done}]\n  mov x0, x15\n  mov x1, xzr\n  adrp x2, {worker}\n  add x2, x2, :lo12:{worker}\n  mov x3, xzr\n  bl pthread_create\n  cbnz w0, {fallback}\n  adrp x8, {context}\n  add x8, x8, :lo12:{context}\n  mov w0, wzr\n  ldr w1, [x8, #{begin}]\n",
            done = DONE_OFFSET,
            worker = plan.worker_symbol,
            fallback = fallback,
            context = plan.context_symbol,
            begin = BEGIN_OFFSET,
        ));
        emit_capture_loads(&mut self.body, &plan.captures, "x8");
        self.body.push_str(&format!(
            "  bl {helper}\n  adrp x8, {context}\n  add x8, x8, :lo12:{context}\n  ldr x0, [x8, #{tid}]\n  mov x1, xzr\n  bl pthread_join\n  cbnz w0, {join_wait}\n  b {parallel_exit}\n{join_wait}:\n  adrp x8, {context}\n  add x8, x8, :lo12:{context}\n  add x8, x8, #{done}\n{join_wait}_spin:\n  ldar w9, [x8]\n  cbz w9, {join_wait}_spin\n  b {parallel_exit}\n{parallel_exit}:\n",
            helper = helper,
            context = plan.context_symbol,
            tid = TID_OFFSET,
            join_wait = join_wait,
            done = DONE_OFFSET,
            parallel_exit = parallel_exit,
        ));
        // The parallel path semantically takes the original header -> exit
        // edge, including any phi copies whose incoming values are proven
        // region-external. The proof rejects all region-defined live-outs.
        self.emit_phi_copies(plan.header.0, plan.exit.0);
        self.body.push_str(&format!(
            "  b {}\n{}:\n",
            self.block_label(plan.exit.0),
            fallback
        ));
        self.emit_phi_copies(plan.preheader.0, plan.header.0);
        self.body
            .push_str(&format!("  b {}\n", self.block_label(plan.header.0)));
    }
}

fn emit_capture_loads(out: &mut String, captures: &[AArch64ThreadCapture], base: &str) {
    for (idx, capture) in captures.iter().enumerate() {
        let offset = capture_offset(idx);
        let reg = idx + 2;
        match capture.ty {
            Type::Ptr(_) => out.push_str(&format!("  ldr x{}, [{}, #{}]\n", reg, base, offset)),
            Type::I32 => out.push_str(&format!("  ldr w{}, [{}, #{}]\n", reg, base, offset)),
            _ => unreachable!("validated thread capture type"),
        }
    }
}

fn capture_offset(index: usize) -> usize {
    CAPTURE_OFFSET + index * CAPTURE_STRIDE
}

fn context_size(capture_count: usize) -> usize {
    CAPTURE_OFFSET + capture_count * CAPTURE_STRIDE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::pass::{run_pipeline, OptLevel, PassOptions};
    use crate::parser::Parser;

    fn threaded_module() -> Module {
        let source = r#"
            int data[100000];
            int transform(int n) {
                int i = 0;
                while (i < n) {
                    int x = data[i];
                    data[i] = x * x + x * 3 + 1;
                    i = i + 1;
                }
                return 0;
            }
        "#;
        let mut parser = Parser::new(source);
        let mut module = crate::ir::lower::lower_program(&parser.parse_program()).unwrap();
        run_pipeline(
            &mut module,
            OptLevel::O1,
            PassOptions {
                enable_simple_loop_unroll: false,
                enable_aarch64_threading: true,
            },
        );
        module
    }

    #[test]
    fn emits_create_failure_scalar_edge_and_exact_pthread_abi() {
        let module = threaded_module();
        let plan = &module.aarch64_thread_plans[0];
        let asm = super::super::emit_ir_asm(&module);
        assert!(asm.contains("mov x1, xzr\n  adrp x2, __yuezhuo_parallel_worker_0"));
        assert!(asm.contains("mov x3, xzr\n  bl pthread_create\n  cbnz w0, .L_thread_fallback"));
        assert!(asm.contains("ldr x0, [x8, #0]\n  mov x1, xzr\n  bl pthread_join"));

        let branch = asm
            .lines()
            .find(|line| line.trim_start().starts_with("cbnz w0, .L_thread_fallback"))
            .unwrap();
        let fallback_label = branch.split_whitespace().last().unwrap();
        let fallback_definition = format!("{}:\n", fallback_label);
        let fallback = asm.find(&fallback_definition).unwrap();
        assert!(asm[fallback..].contains(&format!(
            "b .L_{}_bb{}",
            module.funcs[plan.parent.0].name, plan.header.0
        )));
    }

    #[test]
    fn emits_release_completion_and_join_failure_acquire_wait() {
        let asm = super::super::emit_ir_asm(&threaded_module());
        assert!(asm.contains("stlr w9, [x8]"));
        assert!(asm.contains("ldar w9, [x8]"));
        assert!(asm.contains("cbz w9, .L_thread_join_wait"));
    }

    #[test]
    fn leaves_unplanned_modules_free_of_pthread_references() {
        let mut module = threaded_module();
        module.aarch64_thread_plans.clear();
        let asm = super::super::emit_ir_asm(&module);
        assert!(!asm.contains("pthread_create"));
        assert!(!asm.contains("pthread_join"));
    }

    #[test]
    fn implicit_parent_calls_disable_leaf_phi_registers() {
        let module = threaded_module();
        let plan = &module.aarch64_thread_plans[0];
        let regs = super::super::phi_regs::AArch64PhiRegs::new(&module.funcs[plan.parent.0], true);
        for value in 0..module.funcs[plan.parent.0].values.len() {
            assert!(!matches!(
                regs.reg(crate::ir::ValueId(value)),
                Some("x9" | "x10" | "x11" | "x12" | "x13" | "x14" | "x15")
            ));
        }
    }
}
