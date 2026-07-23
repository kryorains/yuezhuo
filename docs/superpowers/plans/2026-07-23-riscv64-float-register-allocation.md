# RISC-V64 Float Register Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allocate RISC-V64 physical floating-point registers to `F32` IR values so hot loops stop spilling every float intermediate and phi value to the stack.

**Architecture:** Keep the existing integer `Riscv64RegAlloc` behavior intact, add a focused `Riscv64FloatRegAlloc`, and reuse the existing CFG liveness, call-liveness, interference, phi-affinity, and loop-weight analysis. The emitter uses separate integer and float register lookup APIs, preserves used `fs*` registers in the frame, and extends typed loads, results, calls, returns, and phi copies while retaining the existing stack fallback.

**Tech Stack:** Rust 2021, the custom SSA IR, RISC-V64 hard-float ABI, existing codegen test helpers, `cargo test`, WSL, `riscv64-linux-gnu-gcc`, and `qemu-riscv64`.

**Implementation Base:** Create the implementation worktree from commit `701084f` (`origin/feat/matrix-foundation` as of 2026-07-23). Do not implement on the current `main`, because PR #30 has unmerged changes in every RISC-V64 emitter file touched by this feature.

---

### Task 1: Add the floating-point allocator

**Files:**
- Create: `src/codegen/target/riscv64/regalloc/float.rs`
- Modify: `src/codegen/target/riscv64/regalloc.rs:1-170`
- Test: `src/codegen/target/riscv64/regalloc/float.rs`

- [ ] **Step 1: Write failing allocator tests**

Add tests in the new module for a block-local value, a loop phi, and a value live across a call. Use helpers with these exact shapes:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Const, Function, InstKind, Terminator, Type};

    fn float_const(func: &mut Function, value: f32) -> ValueId {
        func.add_const(Const::Float(value.to_bits()))
    }

    #[test]
    fn assigns_block_local_float_to_caller_saved_register() {
        let mut func = Function::new("local_float", Type::F32);
        let lhs = func.add_param("lhs", Type::F32);
        let rhs = float_const(&mut func, 1.0);
        let sum = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs,
                    rhs,
                },
                Some(Type::F32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(sum)));

        let regs = Riscv64FloatRegAlloc::new(&func);
        assert!(regs.reg(sum).is_some_and(|reg| reg.starts_with("ft")));
    }

    #[test]
    fn keeps_call_crossing_float_in_callee_saved_register() {
        let mut func = Function::new("call_crossing", Type::F32);
        let value = func.add_param("value", Type::F32);
        func.append_inst(
            func.entry,
            InstKind::Call {
                name: "side_effect".into(),
                args: vec![],
            },
            None,
        );
        let one = float_const(&mut func, 1.0);
        let result = func
            .append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs: value,
                    rhs: one,
                },
                Some(Type::F32),
            )
            .unwrap();
        func.set_terminator(func.entry, Terminator::Return(Some(result)));

        let regs = Riscv64FloatRegAlloc::new(&func);
        assert!(regs.reg(value).is_some_and(|reg| reg.starts_with("fs")));
    }
}
```

Add the phi-affinity test with an explicit backedge patch:

```rust
#[test]
fn coalesces_non_interfering_float_phi_incoming() {
    let mut func = Function::new("float_phi", Type::Void);
    let initial = func.add_param("initial", Type::F32);
    let header = func.add_block("header");
    func.set_terminator(func.entry, Terminator::Jump(header));
    let phi = func
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(func.entry, initial)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let one = float_const(&mut func, 1.0);
    let next = func
        .append_inst(
            header,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: phi,
                rhs: one,
            },
            Some(Type::F32),
        )
        .unwrap();
    func.block_mut(header).insts[0].kind = InstKind::Phi {
        incomings: vec![(func.entry, initial), (header, next)],
    };
    func.set_terminator(header, Terminator::Jump(header));

    let regs = Riscv64FloatRegAlloc::new(&func);
    assert_eq!(regs.reg(phi), regs.reg(next));
}
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
cargo test codegen::target::riscv64::regalloc::float --locked
```

Expected: FAIL because `regalloc::float` and `Riscv64FloatRegAlloc` do not exist.

- [ ] **Step 3: Expose the existing shared analysis to the child module**

At the top of `regalloc.rs`, add:

```rust
mod float;
pub(super) use float::Riscv64FloatRegAlloc;
```

Make the following existing items visible to the child module without changing their algorithms:

```rust
pub(super) struct InterferenceAnalysis {
    pub(super) interference: Vec<HashSet<ValueId>>,
    pub(super) call_operands: HashSet<ValueId>,
    pub(super) live_across_calls: HashSet<ValueId>,
}

pub(super) fn interference_graph(
    func: &Function,
    candidates: &HashSet<ValueId>,
) -> Option<InterferenceAnalysis>;

pub(super) fn phi_affinities(
    func: &Function,
    candidates: &HashSet<ValueId>,
    interference: &[HashSet<ValueId>],
) -> Vec<Vec<ValueId>>;
pub(super) fn weighted_use_scores(func: &Function) -> Vec<usize>;
```

Keep `inst_uses`, `terminator_uses`, and the analysis budgets in the parent module. The float child should call the same analysis rather than duplicating CFG dataflow.

- [ ] **Step 4: Implement minimal float coloring**

Create `regalloc/float.rs` with these register classes and API:

```rust
use super::{interference_graph, phi_affinities, weighted_use_scores};
use crate::ir::{Function, InstKind, Type, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

const CALLER_SAVED: [&str; 11] = [
    "ft1", "ft2", "ft3", "ft4", "ft5", "ft6",
    "ft7", "ft8", "ft9", "ft10", "ft11",
];
const CALLEE_SAVED: [&str; 12] = [
    "fs0", "fs1", "fs2", "fs3", "fs4", "fs5",
    "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
];
const MIN_CALLEE_SAVED_SCORE: usize = 16;

pub(super) struct Riscv64FloatRegAlloc {
    regs: HashMap<ValueId, &'static str>,
    used_callee_saved: Vec<&'static str>,
}

impl Riscv64FloatRegAlloc {
    pub(super) fn new(func: &Function) -> Self {
        let scores = weighted_use_scores(func);
        let candidates = func
            .values
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                let value_id = ValueId(idx);
                if value.ty != Type::F32 || scores[idx] == 0 {
                    return None;
                }
                match &value.kind {
                    ValueKind::Param => Some(value_id),
                    ValueKind::Inst(block, inst_idx) => {
                        let inst = func.blocks.get(block.0)?.insts.get(*inst_idx)?;
                        (!matches!(inst.kind, InstKind::Nop | InstKind::Alloca { .. }))
                            .then_some(value_id)
                    }
                    ValueKind::Const(_) | ValueKind::Global(_) => None,
                }
            })
            .collect::<HashSet<_>>();
        if candidates.is_empty() {
            return Self::empty();
        }

        let Some(analysis) = interference_graph(func, &candidates) else {
            return Self::empty();
        };
        let affinities = phi_affinities(func, &candidates, &analysis.interference);
        let mut order = candidates
            .iter()
            .copied()
            .map(|value| (value, scores[value.0]))
            .collect::<Vec<_>>();
        order.sort_by_key(|(value, score)| (std::cmp::Reverse(*score), value.0));

        let mut regs = HashMap::<ValueId, &'static str>::new();
        for (value, score) in order {
            let unavailable = analysis.interference[value.0]
                .iter()
                .filter_map(|neighbor| regs.get(neighbor).copied())
                .collect::<HashSet<_>>();
            let live_across_call = analysis.live_across_calls.contains(&value);
            let allowed = |reg: &'static str| {
                !unavailable.contains(reg)
                    && (!CALLER_SAVED.contains(&reg) || !live_across_call)
                    && (!CALLEE_SAVED.contains(&reg)
                        || live_across_call
                        || score >= MIN_CALLEE_SAVED_SCORE)
            };
            let preferred = affinities[value.0]
                .iter()
                .find_map(|neighbor| regs.get(neighbor).copied().filter(|reg| allowed(*reg)));
            let selected = preferred
                .or_else(|| CALLER_SAVED.iter().copied().find(|reg| allowed(*reg)))
                .or_else(|| CALLEE_SAVED.iter().copied().find(|reg| allowed(*reg)));
            if let Some(reg) = selected {
                regs.insert(value, reg);
            }
        }

        let occupied = regs.values().copied().collect::<HashSet<_>>();
        let used_callee_saved = CALLEE_SAVED
            .iter()
            .copied()
            .filter(|reg| occupied.contains(reg))
            .collect();
        Self {
            regs,
            used_callee_saved,
        }
    }

    fn empty() -> Self {
        Self {
            regs: HashMap::new(),
            used_callee_saved: Vec::new(),
        }
    }

    pub(super) fn reg(&self, value: ValueId) -> Option<&'static str> {
        self.regs.get(&value).copied()
    }
    pub(super) fn used_callee_saved(&self) -> &[&'static str] {
        &self.used_callee_saved
    }
}
```

Build the candidate set from used `Type::F32` params and instruction results. Exclude constants, globals, `Nop`, `Alloca`, and unused values. Prefer an affinity register, then `CALLER_SAVED`, then `CALLEE_SAVED`. Reject caller-saved registers for values in `analysis.live_across_calls`. Permit callee-saved registers for call-crossing values; for other values, require a weighted score of at least `MIN_CALLEE_SAVED_SCORE`.

- [ ] **Step 5: Run allocator tests**

Run:

```bash
cargo test codegen::target::riscv64::regalloc --locked
```

Expected: all existing integer tests and all new float allocator tests PASS.

- [ ] **Step 6: Commit the allocator**

```bash
git add src/codegen/target/riscv64/regalloc.rs src/codegen/target/riscv64/regalloc/float.rs
git commit -m "feat: allocate RISC-V64 float registers"
```

### Task 2: Integrate float allocation with function state and ABI

**Files:**
- Modify: `src/codegen/target/riscv64/mod.rs:8-29`
- Modify: `src/codegen/target/riscv64/emitter.rs:85-205`
- Modify: `src/codegen/target/riscv64/abi.rs:1-55`
- Create: `src/codegen/target/riscv64/tests.rs`

- [ ] **Step 1: Add a failing prologue and parameter test**

Register a test module in `riscv64/mod.rs`:

```rust
#[cfg(test)]
mod tests;
```

In `tests.rs`, add a reusable emitter helper and a function whose `F32` parameter remains live across an internal call:

```rust
use super::emit_ir_asm;
use crate::ir::{BinaryOp, Const, Function, InstKind, Module, Terminator, Type};

fn function_asm<'a>(asm: &'a str, name: &str) -> &'a str {
    let start = asm.find(&format!("{name}:\n")).unwrap();
    let rest = &asm[start..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn preserves_call_crossing_float_parameter() {
    let mut callee = Function::new("side_effect", Type::Void);
    callee.set_terminator(callee.entry, Terminator::Return(None));

    let mut caller = Function::new("float_caller", Type::F32);
    let value = caller.add_param("value", Type::F32);
    caller.append_inst(
        caller.entry,
        InstKind::Call {
            name: "side_effect".into(),
            args: vec![],
        },
        None,
    );
    let one = caller.add_const(Const::Float(1.0f32.to_bits()));
    let result = caller
        .append_inst(
            caller.entry,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: value,
                rhs: one,
            },
            Some(Type::F32),
        )
        .unwrap();
    caller.set_terminator(caller.entry, Terminator::Return(Some(result)));

    let mut module = Module::new();
    module.add_func(callee);
    module.add_func(caller);
    let asm = emit_ir_asm(&module);
    let caller_asm = function_asm(&asm, "float_caller");

    assert!(caller_asm.contains("fmv.s fs0, fa0"));
    assert!(caller_asm.contains("fsw fs0, -8(s0)"));
    assert!(caller_asm.contains("flw fs0, -8(s0)"));
}
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
cargo test codegen::target::riscv64::tests::preserves_call_crossing_float_parameter --locked
```

Expected: FAIL because the emitter does not own or save the float allocation.

- [ ] **Step 3: Add float allocation to the function emitter**

Update imports and state in `mod.rs`:

```rust
use regalloc::{Riscv64FloatRegAlloc, Riscv64RegAlloc};

struct Riscv64IrFuncEmitter<'a, 'b> {
    // existing fields
    regalloc: Riscv64RegAlloc,
    float_regalloc: Riscv64FloatRegAlloc,
    // existing fields
}
```

Construct both allocators from the same `allocation_func` in `Riscv64IrFuncEmitter::new`.

- [ ] **Step 4: Save and restore used `fs*` registers**

In `emitter.rs`, replace the integer-only saved-area calculation with:

```rust
let saved_int_regs = self.regalloc.used_regs().to_vec();
let saved_float_regs = self.float_regalloc.used_callee_saved().to_vec();
let saved_slot_count = saved_int_regs.len() + saved_float_regs.len();
let saved_area_size = ir_align_to((saved_slot_count as i32) * 8, 16);
let stack_size = ir_align_to(self.layout.stack_size + saved_area_size, 16);
```

Keep integer offsets first. Save and restore float registers at subsequent 8-byte slots:

```rust
let slot = saved_int_regs.len() + idx + 1;
self.parent.out.push_str(&format!("  fsw {}, -{}(s0)\n", reg, slot * 8));
// tail uses the same slot with flw
```

- [ ] **Step 5: Receive float parameters into assigned registers**

In `abi.rs`, query `self.float_regalloc.reg(*param)` for `Type::F32`. For `IrArgLocation::FloatReg`, emit `fmv.s destination, faN`; for stack float args, load directly into the assigned destination with `load_raw_frame_s`. Keep the current stack-slot path when no float register is assigned.

- [ ] **Step 6: Run the focused and allocator tests**

Run:

```bash
cargo test codegen::target::riscv64 --locked
```

Expected: PASS, including matching `fsw/flw` save slots.

- [ ] **Step 7: Commit ABI integration**

```bash
git add src/codegen/target/riscv64/mod.rs src/codegen/target/riscv64/emitter.rs src/codegen/target/riscv64/abi.rs src/codegen/target/riscv64/tests.rs
git commit -m "feat: preserve assigned RISC-V64 float registers"
```

### Task 3: Route float values through assigned registers

**Files:**
- Modify: `src/codegen/target/riscv64/memory.rs:52-90,281-350`
- Modify: `src/codegen/target/riscv64/inst.rs:57-145,600-930`
- Modify: `src/codegen/target/riscv64/call.rs`
- Test: `src/codegen/target/riscv64/tests.rs`

- [ ] **Step 1: Add a failing straight-line float assembly test**

Add the following test to `tests.rs`:

```rust
#[test]
fn keeps_straight_line_float_intermediates_in_registers() {
    let mut func = Function::new("float_expr", Type::I32);
    let lhs = func.add_param("lhs", Type::F32);
    let rhs = func.add_param("rhs", Type::F32);
    let addend = func.add_param("addend", Type::F32);
    let product = func
        .append_inst(
            func.entry,
            InstKind::Binary {
                op: BinaryOp::Fmul,
                lhs,
                rhs,
            },
            Some(Type::F32),
        )
        .unwrap();
    let sum = func
        .append_inst(
            func.entry,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: product,
                rhs: addend,
            },
            Some(Type::F32),
        )
        .unwrap();
    let result = func
        .append_inst(
            func.entry,
            InstKind::Cast {
                op: crate::ir::CastOp::F32ToI32,
                value: sum,
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(func.entry, Terminator::Return(Some(result)));

    let mut module = Module::new();
    module.add_func(func);
    let asm = emit_ir_asm(&module);
    let body = function_asm(&asm, "float_expr");
    assert!(body.contains("fmul.s"));
    assert!(body.contains("fadd.s"));
    assert!(body.contains("fmv.s ft"));
    assert!(!body.contains("fsw fa0"));
}
```

- [ ] **Step 2: Run the test and verify failure**

```bash
cargo test codegen::target::riscv64::tests::keeps_straight_line_float_intermediates_in_registers --locked
```

Expected: FAIL because every float result still uses the integer bit-pattern/stack path.

- [ ] **Step 3: Add typed float lookup and storage helpers**

In `memory.rs`, add:

```rust
pub(super) fn assigned_float_reg(&self, value: ValueId) -> Option<&'static str> {
    self.float_regalloc.reg(value)
}

pub(super) fn store_float_result(&mut self, value: ValueId, source: &str) {
    if let Some(destination) = self.assigned_float_reg(value) {
        if destination != source {
            self.body.push_str(&format!("  fmv.s {}, {}\n", destination, source));
        }
    } else {
        self.store_frame_s(source, self.layout.offset(value));
    }
}
```

Change `load_float_value` to check `assigned_float_reg` first and emit `fmv.s` only when source and destination differ. Do not route assigned floats through `load_value` or `fmv.w.x`.

- [ ] **Step 4: Emit float load/store directly**

Extend `emit_assigned_load` and `emit_assigned_store` so `Type::F32` uses `flw`/`fsw` with the assigned float result/source and the existing integer pointer register. Preserve existing folded-address handling and return `false` when the required pointer or float register is unavailable.

- [ ] **Step 5: Store all float-producing instructions with the typed helper**

In `emit_inst`, use `store_float_result(result, "fa0")` for:

- `UnaryOp::Fneg`
- `BinaryOp::Fadd/Fsub/Fmul/Fdiv`
- `CastOp::I32ToF32`
- calls returning `Type::F32`

Keep comparisons and `F32ToI32/F32ToBool` on the integer result path. For float returns, call `load_float_value(value, "fa0")` directly instead of loading integer bits through `a0`.

- [ ] **Step 6: Keep float call operands and results type-correct**

In `call.rs`, continue assigning ABI operands to `fa0-fa7`, but let `load_float_value` source them from `ft*`/`fs*`. After a float-returning call, immediately call `store_float_result(result, "fa0")`; never write the result unconditionally to its stack slot.

- [ ] **Step 7: Run focused and full Rust tests**

```bash
cargo test codegen::target::riscv64 --locked
cargo test --locked
```

Expected: all tests PASS; the straight-line test shows register-resident intermediates.

- [ ] **Step 8: Commit typed float emission**

```bash
git add src/codegen/target/riscv64/memory.rs src/codegen/target/riscv64/inst.rs src/codegen/target/riscv64/call.rs src/codegen/target/riscv64/tests.rs
git commit -m "feat: emit RISC-V64 float values from registers"
```

### Task 4: Support float phi parallel copies

**Files:**
- Modify: `src/codegen/target/riscv64/inst.rs:8-45,281-445`
- Test: `src/codegen/target/riscv64/tests.rs`

- [ ] **Step 1: Add failing loop-phi and cycle tests**

Add these explicit CFG tests:

```rust
#[test]
fn keeps_float_accumulator_phi_off_the_stack() {
    let mut func = Function::new("float_loop", Type::F32);
    let initial = func.add_param("initial", Type::F32);
    let count = func.add_param("count", Type::I32);
    let zero_i = func.add_const(Const::Int(0));
    let one_i = func.add_const(Const::Int(1));
    let one_f = func.add_const(Const::Float(1.0f32.to_bits()));
    let header = func.add_block("header");
    let exit = func.add_block("exit");
    func.set_terminator(func.entry, Terminator::Jump(header));
    let acc = func
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(func.entry, initial)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let index = func
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(func.entry, zero_i)],
            },
            Some(Type::I32),
        )
        .unwrap();
    let next_acc = func
        .append_inst(
            header,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: acc,
                rhs: one_f,
            },
            Some(Type::F32),
        )
        .unwrap();
    let next_index = func
        .append_inst(
            header,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs: index,
                rhs: one_i,
            },
            Some(Type::I32),
        )
        .unwrap();
    let cond = func
        .append_inst(
            header,
            InstKind::Icmp {
                op: crate::ir::CmpOp::Lt,
                lhs: next_index,
                rhs: count,
            },
            Some(Type::I1),
        )
        .unwrap();
    func.block_mut(header).insts[0].kind = InstKind::Phi {
        incomings: vec![(func.entry, initial), (header, next_acc)],
    };
    func.block_mut(header).insts[1].kind = InstKind::Phi {
        incomings: vec![(func.entry, zero_i), (header, next_index)],
    };
    func.set_terminator(
        header,
        Terminator::Branch {
            cond,
            then_target: header,
            else_target: exit,
        },
    );
    func.set_terminator(exit, Terminator::Return(Some(next_acc)));

    let mut module = Module::new();
    module.add_func(func);
    let asm = emit_ir_asm(&module);
    let body = function_asm(&asm, "float_loop");
    assert!(body.contains("fadd.s"));
    assert!(body.contains("fmv.s ft"));
    assert!(!body.contains("fsw fa0"));
}

#[test]
fn breaks_float_phi_copy_cycle_with_ft0() {
    let mut func = Function::new("float_swap", Type::Void);
    let lhs = func.add_param("lhs", Type::F32);
    let rhs = func.add_param("rhs", Type::F32);
    let header = func.add_block("header");
    func.set_terminator(func.entry, Terminator::Jump(header));
    let left_phi = func
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(func.entry, lhs)],
            },
            Some(Type::F32),
        )
        .unwrap();
    let right_phi = func
        .append_inst(
            header,
            InstKind::Phi {
                incomings: vec![(func.entry, rhs)],
            },
            Some(Type::F32),
        )
        .unwrap();
    func.block_mut(header).insts[0].kind = InstKind::Phi {
        incomings: vec![(func.entry, lhs), (header, right_phi)],
    };
    func.block_mut(header).insts[1].kind = InstKind::Phi {
        incomings: vec![(func.entry, rhs), (header, left_phi)],
    };
    func.set_terminator(header, Terminator::Jump(header));

    let mut module = Module::new();
    module.add_func(func);
    let asm = emit_ir_asm(&module);
    let body = function_asm(&asm, "float_swap");
    assert!(body.contains("fmv.s ft0,"));
    assert!(body.lines().any(|line| line.ends_with(", ft0")));
}
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
cargo test codegen::target::riscv64::tests::keeps_float_accumulator_phi_off_the_stack --locked
cargo test codegen::target::riscv64::tests::breaks_float_phi_copy_cycle_with_ft0 --locked
```

Expected: FAIL because `PhiLocation::Reg` and the snapshot fallback only understand integer locations.

- [ ] **Step 3: Make phi locations register-class aware**

Replace the location enum and scratch constants with:

```rust
const INT_PHI_CYCLE_SCRATCH: &str = "t2";
const FLOAT_PHI_CYCLE_SCRATCH: &str = "ft0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhiLocation {
    IntReg(&'static str),
    FloatReg(&'static str),
    StackSlot { offset: i32, ty: PhiCopyType },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhiCopyType {
    Word,
    Doubleword,
    Float,
}
```

Map `Type::F32` results and incoming values through `assigned_float_reg`; map integer types through the existing integer allocator.

- [ ] **Step 4: Emit typed parallel copies**

Use:

- `mv`, `lw`, `sw`, `ld`, `sd` for integer locations.
- `fmv.s`, `flw`, `fsw` for float locations.
- `t2` to break integer cycles.
- `ft0` to break float cycles.

Reject a copy plan if source and destination register classes differ. Rematerialize `Const::Float` directly into a float scratch/destination; keep integer constants on the existing path.

- [ ] **Step 5: Make snapshot fallback type-aware**

Replace the integer-only push/pop loop with a typed snapshot record. Float incoming values use `load_float_value(value, "fa0")` plus the existing 16-byte `push_s0`; restoration uses `flw fa0, 0(sp)`, advances `sp`, and calls `store_float_result`. Integer values retain `push_x0/pop_x1`.

- [ ] **Step 6: Run all RISC-V64 and full Rust tests**

```bash
cargo test codegen::target::riscv64 --locked
cargo test --locked
```

Expected: all phi tests PASS and existing integer phi tests remain unchanged.

- [ ] **Step 7: Commit float phi support**

```bash
git add src/codegen/target/riscv64/inst.rs src/codegen/target/riscv64/tests.rs
git commit -m "feat: copy RISC-V64 float phis in registers"
```

### Task 5: Add budget, profitability, and documentation coverage

**Files:**
- Modify: `src/codegen/target/riscv64/regalloc/float.rs`
- Modify: `src/codegen/README.md`
- Test: `src/codegen/target/riscv64/regalloc/float.rs`
- Test: `src/codegen/target/riscv64/tests.rs`

- [ ] **Step 1: Add failing conservative-fallback tests**

Add tests named:

```rust
does_not_pay_callee_saved_cost_for_cold_single_use_value
falls_back_to_stack_when_float_candidate_budget_is_exceeded
does_not_add_float_frame_to_integer_only_function
```

Implement the allocator tests with 12 simultaneously live and used float parameters for the cold case, and `MAX_CANDIDATES + 1` used float parameters for the budget case:

```rust
#[test]
fn does_not_pay_callee_saved_cost_for_cold_single_use_value() {
    let mut func = Function::new("cold_pressure", Type::F32);
    let params = (0..12)
        .map(|idx| func.add_param(format!("p{idx}"), Type::F32))
        .collect::<Vec<_>>();
    let mut sums = Vec::new();
    for pair in params.chunks_exact(2) {
        sums.push(
            func.append_inst(
                func.entry,
                InstKind::Binary {
                    op: BinaryOp::Fadd,
                    lhs: pair[0],
                    rhs: pair[1],
                },
                Some(Type::F32),
            )
            .unwrap(),
        );
    }
    func.set_terminator(func.entry, Terminator::Return(Some(sums[0])));

    let regs = Riscv64FloatRegAlloc::new(&func);
    assert!(regs.used_callee_saved().is_empty());
    assert!(params.iter().any(|value| regs.reg(*value).is_none()));
}

#[test]
fn falls_back_to_stack_when_float_candidate_budget_is_exceeded() {
    let mut func = Function::new("float_budget", Type::Void);
    let zero = float_const(&mut func, 0.0);
    let mut params = Vec::new();
    for idx in 0..=MAX_CANDIDATES {
        let param = func.add_param(format!("p{idx}"), Type::F32);
        params.push(param);
        func.append_inst(
            func.entry,
            InstKind::Binary {
                op: BinaryOp::Fadd,
                lhs: param,
                rhs: zero,
            },
            Some(Type::F32),
        );
    }
    func.set_terminator(func.entry, Terminator::Return(None));

    let regs = Riscv64FloatRegAlloc::new(&func);
    assert!(params.iter().all(|value| regs.reg(*value).is_none()));
}
```

Add the integer-only emitter test:

```rust
#[test]
fn does_not_add_float_frame_to_integer_only_function() {
    let mut func = Function::new("integer_only", Type::I32);
    let lhs = func.add_param("lhs", Type::I32);
    let rhs = func.add_param("rhs", Type::I32);
    let sum = func
        .append_inst(
            func.entry,
            InstKind::Binary {
                op: BinaryOp::Iadd,
                lhs,
                rhs,
            },
            Some(Type::I32),
        )
        .unwrap();
    func.set_terminator(func.entry, Terminator::Return(Some(sum)));
    let mut module = Module::new();
    module.add_func(func);
    let asm = emit_ir_asm(&module);
    let body = function_asm(&asm, "integer_only");
    assert!(body.contains("addw"));
    assert!(!body.contains("fsw fs"));
    assert!(!body.contains("flw fs"));
}
```

- [ ] **Step 2: Run the tests and verify the intended failures**

```bash
cargo test codegen::target::riscv64::regalloc::float --locked
cargo test codegen::target::riscv64::tests::does_not_add_float_frame_to_integer_only_function --locked
```

Expected: budget/profitability assertions fail until the hard limits and fallback are complete; the integer-only output test must pass without adding a float save area.

- [ ] **Step 3: Add explicit float allocation budgets**

Define separate limits in `float.rs`, initially matching the integer allocator:

```rust
const MAX_BLOCKS: usize = 1024;
const MAX_VALUES: usize = 8192;
const MAX_CANDIDATES: usize = 512;
const MAX_LIVENESS_CELLS: usize = 262_144;
```

Return an empty allocator before invoking shared analysis when any limit is exceeded. Never use a conservative exclusive-register assignment that ignores interference.

- [ ] **Step 4: Document the final behavior**

Update `src/codegen/README.md` to state:

- RISC-V64 allocates `F32` values separately from integer values.
- `ft1-ft11` are used only for values not live across calls.
- `fs0-fs11` are saved/restored when allocated.
- `fa0/fa1` remain fixed evaluation/ABI registers and `ft0` remains the float phi scratch.
- analysis budget failures retain the existing stack path.

- [ ] **Step 5: Run formatting, tests, and Clippy**

```bash
cargo fmt
cargo fmt -- --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Expected: all commands succeed with no warnings.

- [ ] **Step 6: Commit budgets and docs**

```bash
git add src/codegen/target/riscv64/regalloc/float.rs src/codegen/target/riscv64/tests.rs src/codegen/README.md
git commit -m "docs: describe RISC-V64 float register allocation"
```

### Task 6: Verify RISC-V64 correctness and performance

**Files:**
- No source changes expected unless a measured regression requires a focused profitability adjustment

- [ ] **Step 1: Install/check the WSL cross toolchain**

Run:

```bash
command -v riscv64-linux-gnu-gcc
command -v qemu-riscv64

if [ ! -e .test-set/.git ]; then
  git clone --branch pref/test-set --single-branch \
    https://github.com/kryorains/yuezhuo.git .test-set
fi
git -C .test-set lfs pull
```

Expected: both commands print executable paths and `.test-set/examples/perf-riscv64` contains materialized LFS test files. If the tools are missing, install `gcc-riscv64-linux-gnu`, `qemu-user`, and `git-lfs` before continuing.

- [ ] **Step 2: Build matching baseline and candidate compilers**

Create a sibling baseline worktree pinned to the implementation base and build both compilers with separate target directories:

```bash
git worktree add ../yuezhuo-float-baseline 701084f

(cd ../yuezhuo-float-baseline && \
  CARGO_TARGET_DIR=target/wsl-baseline cargo build --locked)

CARGO_TARGET_DIR=target/wsl cargo build --locked
```

Expected: baseline compiler at `../yuezhuo-float-baseline/target/wsl-baseline/debug/compiler` and candidate compiler at `target/wsl/debug/compiler`.

- [ ] **Step 3: Run RISC-V64 functional tests**

```bash
COMPILER="$PWD/target/wsl/debug/compiler" TARGET=riscv64 COMPILER_FLAGS=-O0 \
  WORK_DIR="$PWD/target/test-work/riscv64-float-O0" tools/run_functional_tests.sh

COMPILER="$PWD/target/wsl/debug/compiler" TARGET=riscv64 COMPILER_FLAGS=-O1 \
  WORK_DIR="$PWD/target/test-work/riscv64-float-O1" tools/run_functional_tests.sh
```

Expected: both summaries report `FAIL=0`.

- [ ] **Step 4: Benchmark `h-2-*` repeatedly against the pinned compiler**

Use the checked-out LFS performance set and run five baseline/candidate pairs per case:

```bash
for case in h-2-01 h-2-02 h-2-03; do
  for sample in 1 2 3 4 5; do
    COMPILER="$PWD/../yuezhuo-float-baseline/target/wsl-baseline/debug/compiler" \
      TARGET=riscv64 OPT_LEVELS=O1 TEST_FILTER="$case" \
      WORK_DIR="$PWD/target/test-work/float-baseline-$case-$sample" \
      tools/run_perf_tests.sh "$PWD/.test-set/examples/perf-riscv64"

    COMPILER="$PWD/target/wsl/debug/compiler" \
      TARGET=riscv64 OPT_LEVELS=O1 TEST_FILTER="$case" \
      WORK_DIR="$PWD/target/test-work/float-candidate-$case-$sample" \
      tools/run_perf_tests.sh "$PWD/.test-set/examples/perf-riscv64"
  done
done
```

Expected: compute medians from each set of five `RUN_MS` values. At least two cases improve by 20%, and the three-case geometric mean improves by at least 15% relative to the pinned compiler.

- [ ] **Step 5: Run the full RISC-V64 performance set**

```bash
COMPILER="$PWD/target/wsl/debug/compiler" TARGET=riscv64 OPT_LEVELS=O1 BASELINE=1 \
  WORK_DIR="$PWD/target/test-work/riscv64-float-full" \
  tools/run_perf_tests.sh "$PWD/.test-set/examples/perf-riscv64"
```

Expected: no new `DIFF_FAIL`, crash, or timeout. Investigate any material regression by comparing generated assembly; only adjust the float profitability threshold or register eligibility when the evidence identifies save/restore overhead or register pressure as the cause.

- [ ] **Step 6: Inspect the final diff**

```bash
git diff --check origin/feat/matrix-foundation...HEAD
git status --short
git log --oneline origin/feat/matrix-foundation..HEAD
```

Expected: no whitespace errors, only planned source/docs changes, and one focused commit per completed task.
