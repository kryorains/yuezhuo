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
  21. `DcePass`
  22. `ReductionJamPass`
  23. `CsePass`
  24. `LocalForwardPass`
  25. `InvariantLoadForwardPass`
  26. `DcePass`
  27. `RepeatReductionPass`
  28. `SimpleLoopUnrollPass`（按目标收益门控）
  29. `InstCombinePass`
  30. `ConstFoldPass`
  31. `AArch64ThreadOutlinePass`（仅显式启用的 AArch64 O1）
  32. `GepInductionPass`
  33. `SimplifyCfgPass`
  34. `DcePass`

O0 也做标量提升，是为了给代码生成器提供统一的 SSA/phi 形式；不会执行 `GepInductionPass`、`InstCombinePass`、全局常量传播、常量特化、CFG 内联、常量折叠、CSE、LICM 等主动优化。O1 开头先传播只读标量全局常量，让紧邻的 ConstFold 和 DCE 折叠其用户并清理原 load；第一次标量提升会暴露 `NoMemory` 调用摘要，随后第二轮全局标量局部化和提升可安全跨纯调用保留状态。常量实参特化后立即折叠常量、清理 CFG 和死代码，再按统一成本模型执行递归及普通纯标量 CFG 内联。O1 末尾由 InstCombine 和 ConstFold 规范化局部整数算术，再执行通用 GEP 地址归纳强度削弱，最后由 SimplifyCfg 和 DCE 清理新暴露的控制流及死指令。GEP 变换放在 simple unroll 和其它依赖原始 loop-phi 集合的标准循环变换之后，避免新增 pointer phi 屏蔽已有收益。

## Pass 列表

### `GlobalConstPropPass` (`global_const_prop.rs`)

通用的只读标量全局常量传播，仅接入 O1。pass 通过模块符号解析取得唯一全局对象；只有对象带有 `is_const`、对象类型是 i1/i32/f32，且初始化常量的 IR 类型与对象类型完全一致时，才会成为候选。

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

对具有标量常量实参的唯一内部调用目标创建有预算的函数变体，并在变体中把对应形参使用替换为常量。变换只使用符号唯一性、完整调用签名、参数 use-def 和统一的函数/模块代码增长预算；不读取函数名含义、变量名、块名或调用所在测例。只有至少被使用两次的形参参与特化；直接自递归函数保守跳过。每个函数的总变体数、轮数、块数、活动指令数和完整指令槽数均有固定上限，只传播 i1/i32/f32 标量常量。紧随其后的 ConstFold、SimplifyCfg 和 DCE 负责折叠分支及清理不可达路径。

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

基本块内的局部前向传播。

它在单个基本块内跟踪“某个可跟踪指针当前存着哪个值”：

- 遇到 `store ptr, value` 时，记录 `ptr -> value`。
- 遇到后续 `load ptr` 时，如果类型一致，就把 load 的结果替换成已知的 `value`；同一基本块内、没有 intervening clobber 的相同指针重复 load 也会复用第一次结果。
- 遇到任意可能别名的 `store`、`call` 或 `memzero` 时，清空通用 load 记录；本地标量 `alloca` 的 store-to-load 转发仍保持精确。

跨块读写关系继续交给更强的 SSA 与 alias 分析，局部 load CSE 不越过任何未知写入。

### `CsePass` (`cse.rs`)

沿入口可达区域的支配树维护当前路径上可用的表达式，复用一元/二元运算、比较、cast、GEP，以及严格证明为 `NoMemory` 的有结果直接调用。调用 key 是模块快照中的唯一 `FunctionId` 和经过已有 replacement 链解析后的完整实参序列，不使用函数名拼写分类；调用签名必须与唯一目标完全一致。已有调用必须支配当前调用，块内还必须位于其之前，因此 pass 只删除第二次调用，不做移动、LICM 或推测外提；即使纯递归调用可能不终止或触发异常，也只会在第一次相同调用已经返回后复用结果。

`NoMemory` 由 `function_effects.rs` 在闭世界模块上计算。函数全部块内都不能出现 `load`、`store` 或 `memzero`，每个 call 还必须通过精确符号相等唯一解析到同样被证明为 `NoMemory` 的模块函数；`alloca`、phi、纯算术/比较、cast、GEP 和控制流允许。分析从所有局部可接受函数出发做 greatest-fixed-point 反向淘汰，因此没有内存操作或未知调用的递归 SCC 可以保留证明；任一成员含内存访问、未知外部调用或歧义目标时，结论会传播到全部调用者。重复函数符号没有唯一 identity，readonly 函数因包含 load 也不满足此最严格摘要，两者的调用均不会被消除。

函数数和调用边数受固定的通用预算约束；任一模块超预算时全部摘要保守为 `MayMemory`，不会保留部分、顺序相关的证明。CSE 还对 key 操作数总量和支配路径可用表达式比较工作量设置统一上限，超限函数保持不变。GEP key 包含结果指针类型，避免把相同 base/index 但元素步长或结果类型不同的地址计算合并。分析基于每次 pass 入口的不可变模块快照重建，CSE 只把被消除指令改成 Nop 并统一改写使用，重复运行幂等，变换后执行 verifier。

### `InvariantLoadForwardPass` (`invariant_load.rs`)

转发支配当前位置且来自只读对象的重复加载。

它在闭世界的 SysY 模块内检查所有直接调用点：只有当一对指针形参在每个调用点都来自两个不同的完整全局对象时，才把它们视为不别名。函数内含未知调用、写入来源不明，或任一调用点可能别名时都会放弃。满足条件后，pass 沿支配树复用完全相同指针的已有 `load`，不会把加载推测执行到原控制流之前。

### `AArch64ThreadOutlinePass` (`aarch64_thread.rs`)

仅在 driver 通过 `PassOptions` 显式选择 AArch64 O1 时运行的保守双核 outlining。候选是入口可达、专用 preheader、唯一 outer latch 且仅有 `outer header -> exit` 唯一退出边的自然外层循环；region 可以完整包含规范嵌套自然循环和 diamond，但不能与其它自然循环部分重叠。outer header 仅有从 0 开始、步长 1 的 i32 counter phi、可安全复制的纯 bound setup 和直接 signed `counter < bound`。其它 phi 的所有 predecessor 必须在 region 内，嵌套循环还必须从 outer region 内唯一进入，因此 inner reduction 每次 outer iteration 都重新初始化。call、alloca、memzero、return/side exit、region scalar live-out、Fdiv 及动态/零整数除数均拒绝。

内存证明把每个 4 字节 scalar load/store 的地址沿 region 内完整单层 typed GEP chain 展开到模块中唯一的 Global object；每层都验证 array base/result pointee、i32 index、固定对象大小和 GEP use，pointer parameter/alloca/未知 root 不参与首版证明。每个 written root 的 store chain 必须有且仅有一层 index 直接等于 outer counter，任何其它 counter-dependent、偏移、flatten 或间接 selector 都拒绝；同一 written root 的全部 load 必须使用相同 selector depth 和 typed subobject，所以 `A[i][j]` 读改写可行而 `A[j][i]` cross-slice load 不可行。不同且各自唯一的 Global object 证明 NoAlias；region 内未写的 root 可使用任意完整 typed in-object 地址。

helper 是带 `begin/end` 和至多 6 个 preheader-available i32/指针 capture 的普通 verified IR 函数。它先预分配完整 block/value map，再克隆整个 outer region：entry 跳 cloned header，counter 初值换成 `begin`，outer condition bound 换成 `end`，原 outer exit edge 换成 helper return，其它 CFG、phi、指令和 terminator 保持。原 scalar 循环不改写。默认运行门仍为 65,536 outer iterations；只有严格证明一个必经嵌套循环有足够 active work，且其 trip count 为常量或规范化内层 bound 与 outer bound 是同一 SSA 值时，才按已证明的总 work（常量乘数或 `outer²`）保守降低门槛。每个函数仍只生成一个 plan；若有多个合法 region，统一选择证明门槛最低、再选择 region 更完整者，不依赖符号名、维度或算法形状。

pass 对函数、region block/instruction/value、memory、typed chain/type depth、capture、module candidate、use/dependence/provenance work 和 helper 增长均有 checked hard budget；生成符号与用户函数/global 冲突时禁用。每函数最多一个、每模块最多 16 个 plan，重复执行不会继续增长。它位于 GEP induction 之前以证明原始 typed counter GEP；后续语义保持的地址强度削弱不改变已经建立的迭代独立性证明。

### `GepInductionPass` (`gep_induction.rs`)

对自然循环中的仿射 GEP 地址构造 pointer recurrence。候选循环必须有唯一 entering predecessor、唯一 latch 和专用 preheader；pass 从 header 的直接 i32 induction、GEP 类型大小、use-def 和支配关系证明地址为 `base + induction * constant stride + invariant offset`。循环内的 nested GEP chain 可以包含任意循环不变量索引，数组长度和正负 induction step 都不参与形状匹配。

变换在 preheader 用 induction 初值重建完整地址，在 header 插入 pointer phi，在 latch 用单索引常量 GEP 生成固定步长 next pointer。只有全部使用都位于循环内且新 phi 支配普通使用及 phi edge 时才替换原 GEP；仅作为其它已选 nested GEP base 的中间地址不会单独生成死 recurrence。常量 trip-count 分析或 header signed comparison 还必须证明 i32 induction 在每个回边不 wrap，否则 sign extension 后的重算地址与 pointer increment 并不等价，pass 会拒绝。

pass 不推测执行内存访问，也不要求唯一 exit；side exit 上没有 live-out 地址时，header pointer state 仍只沿唯一 backedge 更新。含调用的循环不会变换，因为跨调用 pointer recurrence 会占用 callee-saved 寄存器或增加 spill，而地址重算通常不是这类循环的主要成本。动态大步长、循环内变化的 offset、非 GEP 派生链、不能在 preheader 使用的定义、不可整除为最终 pointee stride 的步长及类型大小溢出都保守保留。变换后立即运行 verifier，重复执行不会再次匹配生成的 pointer recurrence。

### `ReductionJamPass` (`reduction_jam.rs`)

对严格规范化的双层自然循环执行二路 reduction unroll-and-jam。候选外层与内层都必须是从非负初值开始、步长为 1 的直接 signed `<` 归纳循环；内层只有一个 i32 accumulator、一个无副作用基本块，外层每轮唯一副作用必须是归约结果 store。fast loop 一次处理相邻两个外层迭代并共享内层 induction，完整原循环作为动态零/一迭代 tail，因此负边界、奇数次数和小次数仍保持原语义。

内存证明不读取符号名或固定维度：store 地址必须追溯到完整全局对象且最终 GEP 索引就是外层 induction；内层 load 若来自同一全局对象，也必须具有相同最终索引，才能证明两个 lane 访问不同列。未知根、不同索引、循环内 store/call/memzero、多 latch/exit、额外 phi、非专用 preheader 或超出代码预算都会拒绝。克隆后重新运行 CSE、块内 load forwarding、只读 load forwarding和 DCE，再由 verifier 检查；fast loop 的步长 2 也使 pass 重复运行幂等。

### `RepeatReductionPass` (`repeat_reduction.rs`)

无副作用重复归约折叠。

它通过共享的 `LoopInfo` 与 i32 归纳变量分析识别从 0 开始、每轮加 1 的计数循环；当循环体没有 `store`、`call`、`memzero` 等副作用，并且唯一可观察的循环状态满足 `acc' = acc + delta` 时，把重复执行改写为：

1. 执行一次原循环体，得到单轮增量 `delta`；
2. 计算 `initial + delta * count`；
3. 把计数器直接推进到循环上界并退出。

变换使用 i32 环绕算术，因此与逐轮累加在模 2^32 下等价。计数器参与循环体计算、非线性累加、额外活跃 loop phi、依赖累加器的分支、循环内存写入及侧出口都会让该 pass 保守退出。

### `SimpleLoopUnrollPass` (`simple_loop_unroll.rs`)

对严格规范化的单基本块计数循环做二倍展开。

候选循环通过共享的 `LoopInfo` 与 i32 归纳变量分析取得结构和计数器信息，但仍严格要求从 0 开始、每轮加 1、以动态上界做有符号小于比较，并且只有一个活跃 loop phi；含调用、`memzero`、侧出口或额外循环状态时不会展开。pass 在原标量循环前插入两路快速循环，原循环继续处理负数、小于 2 的次数和奇数尾项。两份循环体严格按迭代顺序克隆，因此即使相邻迭代的内存访问互相别名，也不会改变可观察顺序。代码增长受单循环和单函数预算限制。当前 AArch64 后端会让展开后的中间值产生额外栈流量，因此目标收益门控暂时只在 x86-64 和 RISC-V64 启用该 pass。

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
