# codegen 结构说明

`src/codegen` 负责把已经 lowered 的 IR 输出成目标平台汇编。当前支持：

- `riscv64`

## 目录分层

- `mod.rs`：选择目标平台并分发到具体后端。
- `common/`：与具体指令集无关的公共逻辑。
  - `layout.rs`：计算每个 IR value 在栈帧里的位置。
  - `local_regs.rs`：为完整生命周期局限在单一基本块、且不跨调用的值分配 caller-saved 临时寄存器。
  - `loop_depths.rs`：从 CFG 回边恢复自然循环嵌套深度，供热点分配启发式使用。
  - `value_uses.rs`：线性预计算每个 IR value 的使用次数，供指令融合与寄存器分配共享。
  - `sig.rs`：收集用户函数与 SysY 运行时函数签名。
  - `abi.rs`：公共默认参数位置分配逻辑，只决定“整型寄存器 / 浮点寄存器 / 栈”；RISC-V hard-float 由目标后端补充 `fa*` 耗尽后回退 `a*` 的规则。
  - `call.rs`：公共 call 签名解析逻辑。
  - `data.rs`：输出全局变量的数据段。
  - `types.rs`：GEP 等地址计算会用到的类型辅助函数。
- `target/<arch>/`：目标相关逻辑。
  - `emitter.rs`：模块/函数级输出框架，以及序言/尾声。
  - `abi.rs`：函数入口参数搬运。
  - `call.rs`：调用点参数搬运。
  - `memory.rs`：load/store、GEP、栈帧寻址等。
  - `inst.rs`：大多数 IR 指令和 terminator 的翻译。
  - `phi_regs.rs` / `regalloc.rs`：CFG 活跃性/干涉着色、保守回退及保存区计算。

## 关键约定

1. 大多数后端把“当前表达式结果”放在固定临时寄存器里：
   - RISC-V64：整数/指针结果在 `a0`，直接求值的左操作数在 `a1`；浮点对应使用 `fa0` 和 `fa1`。
2. 单基本块内、不跨调用的整数/指针值可使用 RISC-V64 `t3-t6`；调用参数和不跨后续调用的调用结果也可直接使用这些局部寄存器，避免多余的栈往返。
3. RISC-V64 在固定块数、值数量、候选数量、`block * candidate` 工作量和活跃性迭代预算内，对所有函数统一构造 CFG live-in/live-out、phi-edge interference graph 和 copy affinity，使物理寄存器可在不干涉的控制流区域间复用；不收敛或超预算时回退到不做生命周期复用的独占分配。精确分析会在反向扫描每个 call 时先移除 call result、记录完整 live-across-call 集合，再加入 call operands；叶函数和 non-leaf 函数都可把 `a2-a7` 分配给非参数、非任何 call operand、且不跨任何 call 的 i1/i32/ptr 候选，跨 call 的值仍只使用 `s1-s11`。copy affinity 与普通选色经过相同限制，保存区也只统计实际使用的 `s` 寄存器；不收敛或超预算的 non-leaf 保守回退不使用 `a2-a7`。F32 值另行分配：不跨 call 的值优先使用 `ft1-ft11`，跨 call 的值只能使用 `fs0-fs11` 或栈，实际分配的 `fs0-fs11` 会在函数序言和尾声中保存、恢复；`fa0/fa1` 保留为固定求值及 ABI 寄存器，`ft0` 保留为浮点 phi copy 环的 scratch。浮点分析使用独立的块数、值数量、候选数量和 `block * candidate` 预算，任一预算失败都放弃浮点寄存器分配并保留原有栈路径，不做忽略干涉关系的保守独占分配。无法着色的值仍回退到栈。
4. 无法证明寄存器分配安全的 IR value 仍落到 `IrFuncLayout` 栈槽。`alloca` 的 value 栈槽保存对象地址，对象本体紧跟在地址槽之后。
5. 普通二元运算、比较、store 和 GEP 直接从两个临时寄存器求值，不再借助运行时求值栈。RISC-V64 的任一已分配 load 结果及至少一侧已分配的 store 都直接使用物理寄存器，即使另一侧需要从栈槽或全局地址物化，也不再经 `a0/a1` 做结果往返；小立即数比较、`or/xor` 和常量移位分别选择 `slti/xori/ori/slliw/sraiw`。单索引常量 GEP 会先按元素类型计算通用 byte offset；RISC-V64 在 signed 12-bit 范围内选择 `addi`，范围外也只 materialize 最终 byte offset 后做一次寄存器 add/sub，不再重复生成 index scaling。发射前还会比较完整 value use-count：只有 typed GEP 结果的每一次 use 都是 pointee/access type 精确一致的 load/store pointer，且 byte offset 可编码为 signed 12-bit 时，才省略 GEP 并直接发射带 offset 的 `lw/ld/sw/sd`；regalloc 与 local-reg 使用相同的 base-use 视图延长活跃区间，assigned/unassigned 路径遵守同一折叠结论；函数规模、指令、use、候选 GEP、单类型深度或 allocation-view 全函数类型节点总量超出固定预算时完整回退。
6. 整数 `Iand/Ior/Ixor/Ishl/Iashr` 是普通 IR 指令，由各目标的 `inst.rs` 直接选择原生指令。单用途低位掩码比较在 RISC-V64 上使用 `andi` 加零分支；不按函数 CFG 或源码变量形状注入整函数快速路径。
7. Phi 在 terminator 边上生成并行 copy；同一物理位置的 incoming 不生成 copy。RISC-V64 以 regalloc 寄存器或唯一 layout 栈槽作为跨边位置，Const/Global 在需要时直接向目标寄存器重新物化；预算内的无环 copy 按位置依赖顺序直接发射，环由保留的 `t2` 打断，不动态调整 `sp`；超出 copy 数量预算或无法证明位置/类型不变量时回退到全量快照。带 phi copy 的条件边会让布局中的下一块从内部 edge label 自然落入，普通最后返回块也直接落入共享尾声，避免无意义跳转。布局分析在预算内只发射入口可达块，并让规范化自然循环采用 `body .. latch, header, exit` 汇编布局：preheader 保留零次检查，热点回边由 latch fallthrough 和一条反向条件分支完成。RISC-V64 还在固定块数和目标数预算内只对最大嵌套深度的反向分支目标做 16-byte 对齐，并拒绝会让前一块经 padding fallthrough 的目标；多 latch、非专用入口或不规范循环保持原顺序。
8. RISC-V64 会识别只依赖寄存器参数的入口守卫；当一侧是无副作用且只有入口前驱的短返回表达式时，该路径会在函数序言前执行，只有慢路径才建立栈帧。有帧路径先生成原入口到慢分支的 phi copy，再绕过已执行的守卫；不满足结构、类型、ABI 寄存器或规模证明时完全保留普通发射。
9. RISC-V64 的少量纯整数/指针调用参数从后向前直接物化或加载到最终 `a0-a7`，call operand 不参与这些 ABI 寄存器的通用分配，因此不会覆盖尚未装载的实参，也不再统一经过 `a0` 中转；零比较直接选择 `beqz`/`bnez` 等分支指令；递归或大 CFG 函数使用 128 字节函数对齐，降低热点代码跨取指边界造成的波动。小于等于 2048 字节的固定帧用单条 `addi sp` 建立；leaf 函数不保存从未改写的 `ra`，但仍保留相同 16-byte header 和 `s0` 基址，因此栈参数及 frame slot ABI 不变。
10. RISC-V64 常量有符号除法/余数保持 `divw/remw` 的向零截断、负除数、`INT_MIN/-1` 和除零语义：除以 2 时直接从 sign bit 形成 bias，其它二次幂保持完整 bias；二次幂余数直接计算 `(x + bias) & mask - bias`，大 mask 用等价左右移位提取低位，不再先构造 quotient。输入直接从已分配源寄存器参与 magic/shift 序列，结果也直接写入已分配目标寄存器。正的 `2^k+1` / `2^k-1` 常量乘法选择 `slliw + addw/subw`，避免 RV64GC 整数乘法延迟并与 GCC O1 的普通目标成本一致。`MemZero` 对后端栈布局保证八字节对齐的直接 `alloca` 使用 `sd` 清零；其它指针不假设额外对齐，仍以八个 byte store 清零固定八字节 chunk，再处理 0..7 字节静态 tail。
