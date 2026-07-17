# codegen 结构说明

`src/codegen` 负责把已经 lowered 的 IR 输出成目标平台汇编。当前支持：

- `x86_64`
- `aarch64`
- `riscv64`

## 目录分层

- `mod.rs`：选择目标平台并分发到具体后端。
- `common/`：与具体指令集无关的公共逻辑。
  - `layout.rs`：计算每个 IR value 在栈帧里的位置。
  - `local_regs.rs`：为完整生命周期局限在单一基本块、且不跨调用的值分配 caller-saved 临时寄存器。
  - `loop_depths.rs`：从 CFG 回边恢复自然循环嵌套深度，供热点分配启发式使用。
  - `value_uses.rs`：线性预计算每个 IR value 的使用次数，供指令融合与寄存器分配共享。
  - `sig.rs`：收集用户函数与 SysY 运行时函数签名。
  - `abi.rs`：公共参数位置分配逻辑，只决定“整型寄存器 / 浮点寄存器 / 栈”。
  - `call.rs`：公共 call 签名解析逻辑。
  - `data.rs`：输出全局变量的数据段。
  - `types.rs`：GEP 等地址计算会用到的类型辅助函数。
- `target/<arch>/`：目标相关逻辑。
  - `emitter.rs`：模块/函数级输出框架，以及序言/尾声。
  - `abi.rs`：函数入口参数搬运。
  - `call.rs`：调用点参数搬运。
  - `memory.rs`：load/store、GEP、栈帧寻址等。
  - `inst.rs`：大多数 IR 指令和 terminator 的翻译。
  - `phi_regs.rs` / `regalloc.rs`：保守寄存器分配、phi incoming 合并及保存区计算。

## 关键约定

1. 大多数后端把“当前表达式结果”放在固定临时寄存器里：
   - x86_64：整数/指针在 `%rax`，浮点临时常用 `%xmm0`。
   - AArch64：整数/指针结果在 `x0/w0`，直接求值的左操作数在 `x1/w1`；浮点对应使用 `s0` 和 `s1`。GEP 也用 `x1` 保存地址累加器，避免与大偏移寻址使用的 `x16/x17` 临时寄存器冲突。
   - RISC-V64：整数/指针结果在 `a0`，直接求值的左操作数在 `a1`；浮点对应使用 `fa0` 和 `fa1`。
2. 单基本块内、不跨调用的整数/指针值可使用 AArch64 `x3-x7` 或 RISC-V64 `t3-t6`；分配器按块计算短生命周期，避免把普通表达式反复写回帧栈。
3. Phi 结果拥有函数级专用寄存器；有限寄存器按自然循环深度加权后的使用成本优先分给热点 phi，安全的单用途 backedge incoming 会与目标 phi 合并。AArch64 还会给高频跨块值分配唯一的函数级寄存器；RISC-V64 在剩余 `s` 寄存器以及叶函数的 `a2-a7` 中保守分配跨块 load/GEP 值。
4. 无法证明寄存器分配安全的 IR value 仍落到 `IrFuncLayout` 栈槽。`alloca` 的 value 栈槽保存对象地址，对象本体紧跟在地址槽之后。
5. 普通二元运算、比较、store 和 GEP 直接从两个临时寄存器求值，不再借助运行时求值栈；已有寄存器分配时，load/store、GEP、立即数运算以及 AArch64 `madd` 会直接使用最终寄存器。
6. 整数 `Iand/Ior/Ixor/Ishl/Iashr` 是普通 IR 指令，由各目标的 `inst.rs` 直接选择原生指令；后端不再按函数 CFG 或源码变量形状注入整函数快速路径。
7. Phi 在 terminator 边上生成并行 copy；同寄存器 incoming 不再生成 copy，多值环仍使用栈暂存保证并行语义。无 phi copy 的热分支会直接跳向目标并优先使用顺序后继作为 fallthrough。
