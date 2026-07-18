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
2. 单基本块内、不跨调用的整数/指针值可使用 AArch64 `x3-x7` 或 RISC-V64 `t3-t6`；RISC-V64 还允许调用参数和不跨后续调用的调用结果直接使用这些局部寄存器，避免多余的栈往返。
3. 普通函数继续让热点 phi 独占函数级寄存器，并安全合并单用途 entry 参数或 backedge incoming。经过 reduction jam、寄存器需求更高的函数会构造 CFG live-in/live-out 与 phi-edge interference graph；AArch64 `x19-x28`、RISC-V64 `s1-s11` 可在不干涉的顺序循环间复用，并优先按 phi affinity 共色，避免每次回边执行并行栈 copy。叶函数还可使用 caller-saved 跨块寄存器；无法着色的值仍回退到栈。
4. 无法证明寄存器分配安全的 IR value 仍落到 `IrFuncLayout` 栈槽。`alloca` 的 value 栈槽保存对象地址，对象本体紧跟在地址槽之后。
5. 普通二元运算、比较、store 和 GEP 直接从两个临时寄存器求值，不再借助运行时求值栈；已有寄存器分配时，load/store、GEP、整数运算、比较以及 AArch64 `madd` 会直接使用最终寄存器。单索引常量 GEP 会先按元素类型计算通用 byte offset；AArch64 在 add/sub immediate 编码范围内直接更新地址，RISC-V64 在 signed 12-bit 范围内选择 `addi`，范围外也只 materialize 最终 byte offset 后做一次寄存器 add/sub，不再重复生成 index scaling。
6. 整数 `Iand/Ior/Ixor/Ishl/Iashr` 是普通 IR 指令，由各目标的 `inst.rs` 直接选择原生指令。只有当 use-def 精确形成单用途低位掩码比较时，AArch64 才进一步合成为 `tbz/tbnz`（多位掩码用 `tst`），RISC-V64 使用 `andi` 加零分支；不按函数 CFG 或源码变量形状注入整函数快速路径。
7. Phi 在 terminator 边上生成并行 copy；同寄存器 incoming 不再生成 copy，多值环仍使用栈暂存保证并行语义。AArch64 与 RISC-V64 的规范化自然循环会采用 `body .. latch, header, exit` 汇编布局：preheader 保留零次检查，热点回边由 latch fallthrough 和一条反向条件分支完成。多 latch、非专用入口或不规范循环保持原顺序。
8. AArch64 和 RISC-V64 会识别只依赖寄存器参数的入口守卫；当一侧是无副作用且只有入口前驱的短返回表达式时，该路径会在函数序言前执行，只有慢路径才建立栈帧。有帧路径先生成原入口到慢分支的 phi copy，再绕过已执行的守卫；不满足结构、类型、ABI 寄存器或规模证明时完全保留普通发射。
9. RISC-V64 的少量纯整数/指针调用参数从后向前直接装入 `a0-a7`，零比较直接选择 `beqz`/`bnez` 等分支指令；递归或大 CFG 函数使用 128 字节函数对齐，降低热点代码跨取指边界造成的波动。
