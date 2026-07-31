# IR Pass 说明

这个目录放的是 IR 层的优化 pass，以及这些 pass 共享的一些控制流/改写工具。

## 优化流水线

入口在 `mod.rs` 的 `run_pipeline`：

- `OptLevel::O0`：仍保留源码层面的控制流与表达式，但会执行后端所需的标量规范化：
  1. `ScalarPromotePass`
  2. `LocalForwardPass`
  3. `DcePass`
  4. `RepeatReductionPass`
  5. `DcePass`
- `OptLevel::O1`：按下面顺序执行：
  1. `GlobalConstPropPass`
  2. `ConstFoldPass`
  3. `DcePass`
  4. `SimplifyCfgPass`
  5. `TailRecursionPass`
  6. `GlobalScalarLocalizePass`
  7. `ScalarPromotePass`
  8. `GlobalScalarLocalizePass`
  9. `ScalarPromotePass`
  10. `ConstSpecializePass`
  11. `ConstFoldPass`
  12. `SimplifyCfgPass`
  13. `DcePass`
  14. `RecursiveInlinePass`
  15. `InlineSmallExprPass`
  16. `CfgInlinePass`
  17. `LocalForwardPass`
  18. `CsePass`
  19. `LicmPass`
  20. `InvariantLoadForwardPass`
  21. `InstCombinePass`
  22. `ConstFoldPass`
  23. `DcePass`
  24. `ReductionJamPass`
  25. `CsePass`
  26. `LocalForwardPass`
  27. `CsePass`
  28. `InvariantLoadForwardPass`
  29. `DcePass`
  30. `RepeatReductionPass`
  31. `SimpleLoopUnrollPass`（按目标收益门控）
  32. `InstCombinePass`
  33. `ConstFoldPass`
  34. `GepInductionPass`
  35. `PointerRecurrenceCoalescePass`
  36. `DcePass`
  37. `CsePass`
  38. `LocalForwardPass`
  39. `CsePass`
  40. `DcePass`
  41. `SimplifyCfgPass`
  42. `DcePass`

O0 也做标量提升，是为了给代码生成器提供统一的 SSA/phi 形式；不会执行 `GepInductionPass`、`InstCombinePass`、全局常量传播、常量特化、CFG 内联、常量折叠、CSE、LICM 等主动优化。O1 开头先传播已证明不可变的标量全局初值，让紧邻的 ConstFold 和 DCE 折叠其用户并清理原 load；第一次标量提升会暴露 `NoMemory` 调用摘要，随后第二轮全局标量局部化和提升可安全跨纯调用保留状态。常量实参特化先把闭世界全部调用点一致的标量参数直接传播进原函数，再为其余有收益的调用建立有预算变体；随后立即折叠常量、清理 CFG 和死代码，再按统一成本模型执行递归及普通纯标量 CFG 内联。ReductionJam 前的一轮 InstCombine/ConstFold 会先把可证明安全的整除观察规范成无陷阱位运算，使严格 conditional reduction 可进入通用 loop matcher。O1 末尾再由 InstCombine 和 ConstFold 规范化局部整数算术，并执行通用 GEP 地址归纳强度削弱与 pointer recurrence 合并；第一轮 DCE 专门清掉被合并的 pointer `phi -> GEP -> phi` SCC，最后由 SimplifyCfg 和 DCE 清理新暴露的控制流及死指令。地址变换放在 simple unroll 和其它依赖原始 loop-phi 集合的标准循环变换之后，避免新增 pointer phi 屏蔽已有收益。

## Pass 列表

### `GlobalConstPropPass` (`global_const_prop.rs`)

通用的不可变标量全局初值传播，仅接入 O1。pass 通过模块符号解析取得唯一全局对象；对象类型必须是 i1/i32/f32，且初始化常量的 IR 类型与对象类型完全一致。源码 `const` 在所有目标进入证明流程；普通标量声明的“未写且不逃逸”闭世界扩展由目标成本模型控制，当前仅在经过整套回归的 RISC-V64 启用。

模块中对候选地址的唯一合法使用是直接读取完整对象的 `load`；一旦地址参与 `store`、`memzero`、GEP、phi、调用或其它逃逸，整个对象都会保守退出传播。只有指针类型和结果类型都与对象精确一致的直接 load 会被替换；其它 load 保持原样。数组对象、数组元素和任何派生地址都不会传播。满足证明时，pass 在函数内建立对应的 IR `Const`，用共享 rewrite 工具改写用户并删除原 load；随后由 ConstFold 和 DCE 清理新暴露的常量表达式。变换不读取符号拼写，不依赖目标架构，重复运行幂等，并在改写后运行 verifier。

### `ConstFoldPass` (`const_fold.rs`)

常量折叠和代数化简。

它主要做两类事情：

- 如果一条指令的操作数全是常量，就直接算出结果，把原指令替换成常量值，并把指令改成 `Nop`。
- 如果一条指令能用更简单的已有值表示，就记录 `ValueId -> ValueId` 的替换关系，再统一改写所有使用点。

典型例子：

- `1 + 2` 折叠成常量 `3`。
- `x + 0` 简化成 `x`。
- `x * 1` 简化成 `x`。
- `x == x` 简化成 `true`。
- 所有 incoming 都相同的 `phi` 简化成那个唯一值。

### `ConstSpecializePass` (`const_specialize.rs`)

目标成本允许时，先对唯一内部目标执行无代码增长的闭世界实参传播：只有全部直接调用点在某个参数位置都传入同一个 i1/i32/f32 常量，才把原函数体中的该形参使用替换为常量；任一未知、非常量或不同常量调用都会让该参数保守退出。这条路径不受函数大小限制，并优先于变体克隆，因此既能优化过去无法克隆的大函数，也避免为只有一种实参状态的函数保留重复代码。当前只在 RISC-V64 启用；AArch64/x86-64 保持原有有预算变体路径，避免目标相关代码布局回退。

随后对其余具有标量常量实参的唯一内部调用目标创建有预算的函数变体，并在变体中把对应形参使用替换为常量。变换只使用符号唯一性、完整调用签名、参数 use-def 和统一的函数/模块代码增长预算；不读取函数名含义、变量名、块名或调用所在测例。只有至少被使用两次的形参参与变体特化；直接自递归函数保守跳过。每个函数的总变体数、轮数、块数、活动指令数和完整指令槽数均有固定上限。紧随其后的 ConstFold、SimplifyCfg 和 DCE 负责折叠分支及清理不可达路径。

### `InstCombinePass` (`inst_combine.rs`)

逐指令执行的通用整数表达式规范化。它遍历所有函数，只检查当前指令及其操作数的直接定义，不读取函数名、变量名、块名，也不匹配整函数或 CFG 形状。目前支持：

- 把可交换 i32 运算规范到稳定操作数顺序，并把常量放在 RHS；
- 把 `x + x` 改写为 wrapping `x * 2`；
- 在 wrapping 语义下重关联嵌套常量加法和乘法，例如 `(x * 2) * 2` 变为 `x * 4`；
- 当 use-def 链精确证明相同 dividend 和相同已知非零 i32 常量除数时，把 `x - (x / d) * d` 改写为 `x % d`；
- 当有符号余数的全部观察都只是与零比较，且除数绝对值为二次幂时，把 divisibility test 改写为低位掩码；负 dividend 的余数值若在其它位置可见则拒绝；
- 把 `icmp` 常量换到 RHS，并同步反转大小比较谓词。

pass 不重关联或以其它方式改写浮点运算，也不把局部常量二次幂乘除主动改写成 shift；这类局部指令选择继续由现有后端负责。IR 的动态 shift 仍采用计数低 5 位语义。

### `SimplifyCfgPass` (`simplify_cfg.rs`)

简化控制流图。

目前主要处理条件跳转：

- 如果 `branch` 的 then/else 目标相同，改成无条件 `jump`。
- 如果 `branch` 条件是常量，直接选定唯一可达的目标块。
- 当删掉某条 CFG 边时，同步删除目标块里 `phi` 对应的 incoming。
- 对仅合并布尔值并立即分支的 `phi` 块，如果所有动态 incoming 都能证明等于分支条件，则把前驱边直接穿透到分支目标；目前要求至少一个目标块直接返回，并拒绝会改变后继 `phi` 的情况。
- 对可安全推测的一条布尔 RHS 做局部短路 if-conversion，并把无副作用的条件加减 diamond 改写为 wrapping 算术选择；每次改写后都用 verifier 复核，证明失败则原样回退。
- 迭代移除不可达块、转发空跳转块并合并唯一前驱的无 phi 线性基本块；同步重映射 BlockId、指令定义位置和后继 phi incoming。

O1 在首轮及标量提升/常量特化后的中间清理阶段，仅保护 `LoopInfo` 已证明为 dedicated preheader 的空跳转块；其它安全的空跳转仍照常转发，从而保留后续标准循环变换所需形状而不扩大中间 CFG。ReductionJam、地址归纳等 shape-sensitive 变换完成后，末尾的完整 SimplifyCfg 会再次转发所有安全的空块。为限制每轮重建支配树/循环森林的最坏工作量，超过 256 blocks 的中间函数会把全部空块转发延迟到末轮，而不是反复执行无预算循环分析。

### `TailRecursionPass` (`tail_recursion.rs`)

尾递归消除。

它只处理保守的自递归尾调用：

- 非 `void` 函数中形如 `return self(args...)` 的 IR 形态；
- `void` 函数中尾部的 `self(args...); return;` 形态；
- 调用结果除了对应 `return` 之外没有其它使用；
- 尾调用参数不能包含由当前栈帧局部 `alloca` 派生的指针，否则保留递归调用，避免循环复用局部数组时改变语义。

变换时会把函数入口拆成一次性执行的初始化块和可回跳的函数体入口块，然后把尾自调用改写为：

1. 按原调用求值顺序得到的参数值依次存回形参槽位；
2. 跳回函数体入口。

后续 `ScalarPromotePass` 会把这些形参槽位提升成 loop phi，使最终 IR 更接近普通循环。

### `GlobalScalarLocalizePass` (`global_scalar_localize.rs`)

把函数中反复访问的标量全局变量暂存为局部标量：入口加载一次，每个正常返回点写回一次，再由后续 `ScalarPromotePass` 提升为 SSA。候选全局必须只被直接 `load`/`store` 使用，候选地址不能参与 `memzero` 或逃逸；函数可以包含经闭世界摘要和完整签名证明为 `NoMemory` 的有结果调用，其它调用仍保守拒绝，从而保证局部状态不会被调用观察或改写。处理顺序按全局名排序并受固定数量和下游标量提升规模预算约束，保证代码生成稳定且避免大函数编译时间膨胀。

### `ScalarPromotePass` (`scalar_promote.rs`)

标量提升，类似简化版 mem2reg。

它会把安全的栈上标量变量从内存形式提升成 SSA 值：

- 找出只被普通 `load`/`store` 使用、没有逃逸的标量 `alloca`。
- 利用支配树和 dominance frontier 决定在哪里插入 `phi`。
- 沿支配树重命名：
  - `store` 变成“当前值入栈”；
  - `load` 替换成当前栈顶 SSA 值；
  - 被提升掉的 `alloca`/`load`/`store` 改成 `Nop`。

这个 pass 会跳过过大的函数，避免朴素算法在大 IR 上耗时过高。

### `RecursiveInlinePass` (`recursive_inline.rs`)

对小型自递归函数的非尾调用做一层标准 CFG 内联。IR 的直接调用目标目前用符号表示，pass 仅用符号的精确相等关系解析唯一调用目标，不依赖函数名的具体拼写，也不读取值名、块名、固定块编号或源码信息；每个候选、调用点、调用方及整个模块都有显式的指令/基本块增长上限。

变换在调用处拆出 continuation，克隆 pass 开始时快照中的全部 callee CFG、SSA 值、phi 和 terminator，把形参映射到实参，并让所有克隆 return 跳到 continuation。非 void 返回统一用 phi 合并，因此多个返回路径保持原 edge 语义。原后继 phi 的前驱边会从调用块改到 continuation。每个调用点完成后立即执行 IR verifier；函数级标记保证重复运行不会再增加递归深度。

当前证明边界有意保守：只接受 i1/i32 返回，入口不可达的死 CFG 不参与调用点、指令增长和克隆，但函数总块上限及禁用类型检查仍覆盖整个 IR；拒绝 f32、void/result-less call、active `alloca`、`memzero`、歧义调用目标和签名不一致的自调用。普通整数、指针、load/store/GEP、phi 及有结果的调用会按原控制流与副作用顺序克隆。pass 放在 `ScalarPromotePass` 之后，使已安全提升的标量栈槽不会阻挡候选；只接入 O1。

### `CfgInlinePass` (`recursive_inline.rs`)

对唯一内部调用目标执行有预算的普通 CFG 内联。候选必须是无内存、无调用、无浮点和无栈对象的纯 i1/i32 CFG；允许 phi、分支和多返回。pass 使用与递归 CFG 内联相同的 continuation 拆分、全值预分配、phi/terminator 克隆和返回值合并逻辑，但按独立的普通函数块数、指令数、调用点、调用方及模块增长预算决定收益。所有候选和调用点来自 pass 入口快照，符号名只用于唯一调用解析，不参与收益判断。

### `InlineSmallExprPass` (`inline.rs`)

内联小型、单基本块的纯标量表达式函数。

候选函数不能包含调用、控制流、phi 或可观察的内存操作；允许忽略没有读取或逃逸的局部只写槽位，其余只保留少量一元/二元运算、比较和类型转换。调用点会克隆这些纯指令并改写返回值的使用点，让后续 CSE、LICM 和 DCE 继续优化展开后的表达式。每个候选和调用方都有固定的代码增长预算，避免病态膨胀。

### `LocalForwardPass` (`local_forward.rs`)

有预算的局部及 CFG available-load 前向传播。

它在基本块内跟踪“某个可跟踪指针当前存着哪个值”：

- 遇到 `store ptr, value` 时，记录 `ptr -> value`。
- 遇到后续 `load ptr` 时，如果类型一致，就把 load 的结果替换成已知的 `value`；同一基本块内、没有 intervening clobber 的相同指针重复 load 也会复用第一次结果。
- 遇到 `call`、`memzero` 或可能别名的 `store` 时清空相应 load 记录；typed scalar pointer 若有一侧来自当前函数新建的 `alloca`、另一侧来自不同对象根，则按普通 C 对象生命周期证明不别名并保留 load。参数或全局不可能预先指向尚未存在的 callee-local allocation；无法追溯完整根时仍清空。本地标量 `alloca` 的 store-to-load 转发保持精确。
- 对 exact SSA pointer 的重复 load，pass 还在 CFG 上计算前驱 OUT 状态的交集；只有每条进入路径都携带同一个、类型一致的支配 load 且路径内没有 clobber 时才跨块复用。状态项、迭代次数和总 transfer work 均有固定预算，超限回退块内逻辑。

同一 pass 还执行严格局部的冗余回写 DSE：若 `v = load p` 之后出现精确相同 SSA 指针上的 `store p, v`，并且中间没有 `call`、`memzero` 或写入其它值的 `store`，则把最终 store 改成 `Nop`。中间的 `store q, v` 只有在 `p/q` 都能通过单索引、结果类型一致的 typed GEP 链追溯到同一对象根、访问相同四字节标量类型时才允许保留；此时两地址之差必为四字节倍数，等宽访问只能完全相同或互不重叠，写入相同值不会破坏原值。任何未知 provenance、部分重叠风险或写入其它 SSA 值的 store 都会保守终止候选，不依据名称、类型维度或跨块关系。

跨块转发只处理 exact pointer 的 must-available 事实；对写入只使用上述 callee-local allocation NoAlias，不猜测参数间、全局间或同一对象内的 may-alias。新 DSE 与数据流使用固定的块、值、指令、pointer-chain、状态、transfer-work 和 fixed-point 预算；超出 DSE 尺寸预算时原有转发仍会执行。两种局部变换运行到共同不动点；若在迭代预算内未收敛则恢复函数入口快照，因此重复运行幂等，发生改写后执行 verifier。

### `CsePass` (`cse.rs`)

沿入口可达区域的支配树维护当前路径上可用的表达式，复用一元/二元运算、比较、cast、GEP，以及严格证明为 `NoMemory` 的有结果直接调用。调用 key 是模块快照中的唯一 `FunctionId` 和经过已有 replacement 链解析后的完整实参序列，不使用函数名拼写分类；调用签名必须与唯一目标完全一致。已有调用必须支配当前调用，块内还必须位于其之前，因此 pass 只删除第二次调用，不做移动、LICM 或推测外提；即使纯递归调用可能不终止或触发异常，也只会在第一次相同调用已经返回后复用结果。

`NoMemory` 由 `function_effects.rs` 在闭世界模块上计算。函数全部块内都不能出现 `load`、`store` 或 `memzero`，每个 call 还必须通过精确符号相等唯一解析到同样被证明为 `NoMemory` 的模块函数；`alloca`、phi、纯算术/比较、cast、GEP 和控制流允许。分析从所有局部可接受函数出发做 greatest-fixed-point 反向淘汰，因此没有内存操作或未知调用的递归 SCC 可以保留证明；任一成员含内存访问、未知外部调用或歧义目标时，结论会传播到全部调用者。重复函数符号没有唯一 identity，readonly 函数因包含 load 也不满足此最严格摘要，两者的调用均不会被消除。

函数数和调用边数受固定的通用预算约束；任一模块超预算时全部摘要保守为 `MayMemory`，不会保留部分、顺序相关的证明。CSE 还对 key 操作数总量和支配路径可用表达式比较工作量设置统一上限，超限函数保持不变。GEP key 包含结果指针类型，避免把相同 base/index 但元素步长或结果类型不同的地址计算合并。分析基于每次 pass 入口的不可变模块快照重建，CSE 只把被消除指令改成 Nop 并统一改写使用，重复运行幂等，变换后执行 verifier。

### `InvariantLoadForwardPass` (`invariant_load.rs`)

转发支配当前位置且来自只读对象的重复加载。

它在闭世界的 SysY 模块内检查所有直接调用点：只有当一对指针形参在每个调用点都来自两个不同的完整全局对象时，才把它们视为不别名；若某个指针形参在全部调用点都来自已知全局对象集合，则它还可与集合之外的唯一全局对象证明不别名。直接指向同一唯一全局 symbol 的不同 SSA 地址值会按对象 identity 统一 load key。函数内含未知调用、写入来源不明，或任一调用点可能别名时都会放弃。满足条件后，pass 沿支配树复用完全相同指针的已有 `load`，不会把加载推测执行到原控制流之前。

### `GepInductionPass` (`gep_induction.rs`)

对自然循环中的仿射 GEP 地址构造 pointer recurrence。候选循环必须有唯一 entering predecessor、唯一 latch 和专用 preheader；pass 从 header 的直接 i32 induction、低深度 `iv ± constant` 或 `iv + invariant` AddRec、GEP 类型大小、use-def 和支配关系证明地址为 `base + induction * constant stride + invariant offset`。derived index 只有在精确 trip endpoint（包含最终失败 header）或完整 modulo-`2^32` 同余类极值证明 signed 加减不回绕时才接受；动态 invariant 必须是另一个具有常量 trip count 的已证明 induction，并对两者完整取值区间求和验证不溢出。循环内的 nested GEP chain 可以包含任意循环不变量索引，数组长度和正负 induction step 都不参与形状匹配。

变换在 preheader 用 induction 初值重建完整地址，在 header 插入 pointer phi，在 latch 用单索引常量 GEP 生成固定步长 next pointer。只有全部使用都位于循环内且新 phi 支配普通使用及 phi edge 时才替换原 GEP；仅作为其它已选 nested GEP base 的中间地址不会单独生成死 recurrence。常量 trip-count 分析或 header signed comparison 还必须证明 i32 induction 在每个回边不 wrap，否则 sign extension 后的重算地址与 pointer increment 并不等价，pass 会拒绝。

pass 不推测执行内存访问，也不要求唯一 exit；side exit 上没有 live-out 地址时，header pointer state 仍只沿唯一 backedge 更新。含调用的循环不会变换，因为跨调用 pointer recurrence 会占用 callee-saved 寄存器或增加 spill，而地址重算通常不是这类循环的主要成本。动态大步长、超过一层的标量 affine DAG、循环内变化的 offset、非 GEP 派生链、不能在 preheader 重建的定义、不可整除为最终 pointee stride 的步长及类型大小溢出都保守保留。函数指令、候选数、GEP chain/index、预计新增 values 和最终函数 values 均有硬预算，避免多个共享前缀候选产生二次方 IR 膨胀。变换后立即运行 verifier，重复执行不会再次匹配生成的 pointer recurrence。

### `PointerRecurrenceCoalescePass` (`pointer_recurrence_coalesce.rs`)

合并同一自然循环中保持固定 byte distance 的 pointer recurrence。候选必须具有同一 header/latch、精确相同的 pointer type 和单索引常量 GEP step；初值会沿完整 typed GEP chain 展开为对象根、SSA 动态索引项及 checked 常量 byte offset，只有对象根和全部动态项系数完全相同的 recurrence 才能配对。索引中的 `iv + constant` 仅在 i32 addrec 初值/步长同余类的完整 signed 极值范围证明该加减不回绕时才提取常量；证明失败时该 SSA 表达式保持为不透明动态项，不假设源码维度或循环上界。

secondary pointer 除唯一 backedge update 外，只能作为循环内、pointee/access type 精确一致的 load/store pointer；update 自身也只能回到该 phi。pass 在 header 从 primary pointer 建立一个可表示该 byte distance 的单索引常量 GEP，并只改写已证明的 memory pointer use；旧 secondary `phi/update` SCC 交给紧随其后的 mark-and-sweep DCE 删除。函数规模、use 数、recurrence 数、GEP chain/index、affine 深度、type nodes、pair proof 和总 work 都有固定预算；所有计划在改写前完成，预算或算术证明失败不产生部分 IR。改写后运行 verifier，重复执行不会再次处理已失去 memory use 的 secondary。

### `ReductionJamPass` (`reduction_jam.rs`)

对严格规范化的双层自然循环执行二路或四路 reduction unroll-and-jam。候选外层与内层都必须是从非负常量初值开始、步长为 1 的直接 signed `<` 归纳循环；既有路径要求内层只有一个 i32 accumulator、一个无副作用基本块，外层每轮唯一副作用必须是归约结果 store。已建立的二路路径要求外层初值模 2 对齐；只有额外合法性与收益证明全部成功才选择模 4 对齐的四路路径。fast header 直接形成 `iv+1..iv+factor-1` 并以最后一 lane 的 signed `< bound` 为动态 guard，taken exit 才形成 `iv+factor`：对齐性质证明每次失败的 header 求值也不回绕，成功 guard 又证明 next 不回绕且确实至少剩余 factor 次。完整原循环从 fast phi 接手所有动态 tail，四路时仍可执行 0..3 次，绝不假设输入次数能整除 factor；已知常量 trip count 小于 4 时保留二路，小于 2 时不增加 fast path。

RISC-V64 的 `max_reduction_jam_factor >= 4` 成本入口还允许严格 canonical conditional reduction 使用保守 factor 2：inner header 恰有共享 i32 induction 和 accumulator 两个 phi，循环 taken edge 进入纯 condition block；condition 按原方向分支到纯、无 trapping arithmetic 的 update block 或直接进入 merge；update 也只进入 merge；merge 恰有选择 old/new accumulator 的 phi 和原 induction next，再以唯一 backedge 返回 header。变换为两个 lane 建立独立 accumulator，按 lane 原指令顺序分别执行完整 diamond，绝不把 update-only load 提升到分支前；共享 inner induction 每两个 outer lane 只更新一次。fast guard 失败后原循环处理动态 0..1 tail。额外 phi、live-out、side exit、call/store/memzero、整数除法/余数、浮点除法及非唯一 final store 均拒绝。x86-64/AArch64 的 `max_factor == 2` 入口不启用该形状。

内存证明采用普通 SysY/C 有效对象前提：实际执行的 load/store 必须位于其完整 typed allocation 内，越界或无效指针访问属于未定义行为；IR 也没有 volatile/atomic memory access。因此证明为不别名的有效 load 可以跨更早 lane 的 store 重排。证明不按符号拼写分类或匹配固定维度，只把完整全局 symbol identity 当作对象身份：所有访问都要有预算地展开完整 nested GEP 链，检查每层 base/result 指针类型、i32 index、load/store 与 pointee 的精确类型和 checked byte size；不同全局对象随后直接不别名。store 以及需要证明的同对象 load，其 terminal 必须是以外层 induction 为唯一索引的 GEP；prefix 常量折入 byte offset，所有非常量 prefix 即使 SSA 相同也分别视为任意整数。对 store lane `s` 与后续 load lane `l` 的地址差，常量项为 `C = load_offset - store_offset + l*load_stride - s*store_stride`，再以所有 prefix stride、terminal stride 差和机器地址模数 `2^64` 的绝对 gcd `G` 构造同余类；只有该同余类在可能重叠区间 `[-(load_width-1), store_width-1]` 内无解才证明不别名，因此即使只通过 64 位地址回绕才能重叠也会保守拒绝。四路会逐对证明 lane-0/1/2 store 与所有后续 lane load；store 本身仍按原 lane 顺序发出，因而不需要 store-store 不别名。

四路当前只在 RISC-V64 目标成本模型中启用；x86-64/AArch64 继续使用既有二路路径。具体收益门控只读取目标无关 SSA/类型成本，不按后端指令形状、名称、维度或输入分类：必须至少有一条可由后续 CSE/LocalForward 共享的 lane-invariant load stream；最多 4 个活跃 accumulator；pointer-state 上限 12，包括每 lane 的输出地址/索引状态、一个共享对象 base，以及按 lane-dependence（outer index 或 accumulator）放大为四份的不同 load pointer stream；accumulator、pointer state、按 lane 复制的 setup 标量、直接供 inner body 使用的外部 invariant 标量、outer/inner induction、两个动态 bound 和共享 load 结果的保守峰值不超过 20；每 lane clone map 总项数不超过 96，预计结果寄存器候选不超过 96，投影后的整函数 key 数、GEP 结果 type nodes、操作数总量及 key×operand 工作量也必须落在保守 CSE 预算内；变换后值数量及 `block×value` 上界还必须保证不会跨过 RISC-V 精确 regalloc 的候选/工作量悬崖；新增 fast-path 指令不超过 128 且不超过原五块区域的 5 倍，加入 5 个块和保守估计的新常量/结果后也必须仍在函数级 block/value/instruction 预算内。conditional factor 2 另设更紧的 10 pointer、64 map、64 register candidate、20 peak-live、96 code-growth、原区域 4 倍和新增 10 blocks 上限；condition block 中每轮必执行的 load 只有在两 lane remap 后 pointer identity 精确相同时才直接复用 lane 0 结果，update-only load 始终保留在原分支并按两份 pointer/result pressure 计费。二路 fallback 同样执行变换后硬预算检查。所有计数使用 checked arithmetic；任一上限、模 4 nowrap 或六组跨 lane 内存证明失败都无条件回退已有二路路径，不会关闭原收益。未知根、类型或算术溢出、超出 GEP chain/index/type/work 预算、循环内 store/call/memzero、多 latch/exit、额外 phi、非专用 preheader 或代码预算超限都会拒绝整个变换。fast path 会重建而非克隆 loop header，因此两个 header 还必须只包含 Nop、phi 和 terminator 使用的精确 i32 比较；inner exit 除唯一 store 外也只允许可直接 remap 的纯标量/GEP 指令。setup、inner body 和 exit 中会跨 store 重排的整数除法/余数及浮点除法全部拒绝，避免后续 lane trap 提前。lane cloning 统一使用 factor-sized value-map vector；克隆后重新运行 CSE、块内 load forwarding、只读 load forwarding 和 DCE，再由 verifier 检查，jam mark 保证重复运行幂等。

### `RepeatReductionPass` (`repeat_reduction.rs`)

无副作用重复归约折叠。

它通过共享的 `LoopInfo` 与 i32 归纳变量分析识别从 0 开始、每轮加 1 的计数循环；当循环体没有 `store`、`call`、`memzero` 等副作用，并且唯一可观察的循环状态满足 `acc' = acc + delta` 时，把重复执行改写为：

1. 执行一次原循环体，得到单轮增量 `delta`；
2. 计算 `initial + delta * count`；
3. 把计数器直接推进到循环上界并退出。

变换使用 i32 环绕算术，因此与逐轮累加在模 2^32 下等价。计数器参与循环体计算、非线性累加、额外活跃 loop phi、依赖累加器的分支、循环内存写入及侧出口都会让该 pass 保守退出。

### `SimpleLoopUnrollPass` (`simple_loop_unroll.rs`)

对严格规范化的单基本块计数循环做目标成本驱动的二倍或四倍展开。

候选循环通过共享的 `LoopInfo` 与 i32 归纳变量分析取得结构和计数器信息，但仍严格要求从 0 开始、每轮加 1、以动态上界做有符号小于比较，并且只有一个活跃 loop phi；含调用、`memzero`、侧出口或额外循环状态时不会展开。x86-64 使用二路快速循环；RV64GC 缺少 AArch64 的 scaled-address、整数 `madd` 和条件执行能力，因而只对不超过 8 条活跃指令且不含整数除法/余数或浮点除法的小循环使用四路快速循环，以减少循环控制和重复地址更新；更大或单指令机器成本很高的循环不展开，避免代码/I-cache 与寄存器压力抵消收益。原标量循环继续处理负数、小于 factor 的次数和 0..factor-1 动态尾项。各 lane 严格按迭代顺序克隆，因此即使相邻迭代的内存访问互相别名或先前 lane 发生同步 trap，也不会改变可观察顺序。clone、最终 block/value/instruction 和 checked-arithmetic 均有单循环及单函数预算；函数级 decision mark 让预算拒绝和成功变换都不会在重复运行时继续累积另一批候选。当前 AArch64 后端会让展开后的中间值产生额外栈流量，因此目标收益门控不启用该 pass。

### `DcePass` (`dce.rs`)

死代码删除。

它从 `store`、`call`、`memzero` 等可观察副作用以及 terminator 操作数反向标记完整依赖闭包，再把未标记的纯结果指令改成 `Nop`。这种 mark-and-sweep 形式不仅删除普通零使用值，也能删除 `phi -> add -> phi` 这类内部互相引用、但没有任何可观察出口的死 SSA 环。

`store`、`call`、`memzero` 本身始终保留；删除仍以 `Nop` 占位，不重排 `ValueKind::Inst` 的指令索引。

## 支撑模块

### `function_effects.rs`

构建闭世界模块的通用函数副作用摘要和唯一直接调用目标，不是独立变换 pass。当前只公开最严格的 `NoMemory / MayMemory` 边界；后续若增加 readonly 等层级，不会自动放宽 CSE 的合法性。

### `dominators.rs`

构建控制流图和支配信息，不是一个独立 pass。

它提供：

- `ControlFlowGraph`：每个基本块的前驱和后继。
- `Dominators`：
  - 在入口可达 CFG 的 reverse postorder 上用 Cooper-Harvey-Kennedy 算法求 immediate dominator；
  - 支配树 children 和 DFS 区间，`dominates` 查询为 O(1)；
  - dominance frontier。

不可达块不加入入口支配树；IR verifier 另把虚拟根视为位于 physical entry 指令之后，并连接不可达 CFG 各 source SCC 的全部成员，保守检查 dead region 的跨块定义及 phi-edge 可用性，既不放过 dead diamond 的错误 incoming，也允许 entry 定义及结构正确的 `dead.def -> dead.use`。`ScalarPromotePass` 用入口支配信息判断定义是否支配使用，以及在哪里需要插入 `phi`；`LoopInfo`、LICM 及 IR verifier 都不依赖基本块在 `Function.blocks` 中的存储顺序。

### `loop_analysis.rs`

共享的自然循环与 i32 归纳变量分析，不是独立 pass，也不读取函数名、块名或固定块编号。

- `LoopInfo` 从 CFG 中收集“循环头支配回边源”的 backedge，并按 header 合并成 `NaturalLoop`。
- `NaturalLoop` 提供 header、所有 latch/backedge、只含入口可达且被 header 支配的 loop blocks、唯一 entering predecessor 与专用 preheader（若存在）、全部退出边和唯一 exit（若存在）；多 latch、多入口或多 exit 会显式保留，而不是猜测某个固定布局。
- `analyze_i32_induction` 只需从 header phi 的唯一 entering predecessor/latch incoming 识别 `next = phi + constant` 形式，统一处理 `add` 两种操作数顺序及 `sub phi, constant`，支持任意非零 i32 环绕步长，并返回 phi、initial、next、step。
- `analyze_const_i32_trip_count` 对常量初值、常量步长和 header 直接 signed `icmp` 计算精确迭代次数，统一比较操作数反转和 true/false continuation；依赖 i32 回绕才能终止、越界或不终止的情况会拒绝。

LICM 和 `SimpleLoopUnrollPass` 的 CFG/区域改写要求 dedicated preheader；`RepeatReductionPass` 与归纳分析只要求 unique entering predecessor。各 pass 复用循环结构和归纳变量描述，再施加自身的严格变换门控。LICM 按 invariant use-def 拓扑一次把定义先于使用移入 preheader，避免 BlockId 逆序依赖链上的反复全循环扫描。因此该模块是普通循环优化的公共基础设施。

### `util.rs`

共享的 Value 改写工具，不是一个独立 pass。

它提供：

- `ValueReplacements = HashMap<ValueId, ValueId>`：记录值替换关系。
- `rewrite_function_uses`：把函数里所有操作数按替换表改写。
- `resolve_replacement`：追踪链式替换，保证 `a -> b -> c` 最终解析到 `c`。
