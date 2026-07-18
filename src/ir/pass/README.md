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
  1. `ConstFoldPass`
  2. `SimplifyCfgPass`
  3. `TailRecursionPass`
  4. `GlobalScalarLocalizePass`
  5. `ScalarPromotePass`
  6. `RecursiveInlinePass`
  7. `InlineSmallExprPass`
  8. `LocalForwardPass`
  9. `CsePass`
  10. `LicmPass`
  11. `InvariantLoadForwardPass`
  12. `DcePass`
  13. `PiecewiseExprPass`
  14. `RepeatReductionPass`
  15. `SimpleLoopUnrollPass`（按目标收益门控）
  16. `InstCombinePass`
  17. `ConstFoldPass`
  18. `LoopIdiomPass`
  19. `ConstFoldPass`
  20. `GepInductionPass`
  21. `SimplifyCfgPass`
  22. `DcePass`

O0 也做标量提升，是为了给代码生成器提供统一的 SSA/phi 形式；不会执行 `PiecewiseExprPass`、`LoopIdiomPass`、`GepInductionPass`、`InstCombinePass`、常量折叠、CSE、LICM 等主动优化。流水线中的前置 DCE 会先清掉标量提升遗留的死 phi；O1 末尾先由 InstCombine 和 ConstFold 把局部整数算术规范化，再让 LoopIdiom 识别循环区域，随后执行 GEP 地址归纳强度削弱，最后由 SimplifyCfg 和 DCE 清理新暴露的控制流及死指令。GEP 变换放在 simple unroll 和其它依赖原始 loop-phi 集合的 matcher 之后，避免新增 pointer phi 屏蔽已有收益。

## Pass 列表

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

### `InstCombinePass` (`inst_combine.rs`)

逐指令执行的通用整数表达式规范化。它遍历所有函数，只检查当前指令及其操作数的直接定义，不读取函数名、变量名、块名，也不匹配整函数或 CFG 形状。目前支持：

- 把可交换 i32 运算规范到稳定操作数顺序，并把常量放在 RHS；
- 把 `x + x` 改写为 wrapping `x * 2`；
- 在 wrapping 语义下重关联嵌套常量加法和乘法，例如 `(x * 2) * 2` 变为 `x * 4`；
- 当 use-def 链精确证明相同 dividend 和相同已知非零 i32 常量除数时，把 `x - (x / d) * d` 改写为 `x % d`；
- 把 `icmp` 常量换到 RHS，并同步反转大小比较谓词。

pass 不重关联或以其它方式改写浮点运算，也不把局部常量二次幂乘除主动改写成 shift；这类局部指令选择继续由现有后端负责。IR 的动态 shift 仍采用计数低 5 位语义。

### `SimplifyCfgPass` (`simplify_cfg.rs`)

简化控制流图。

目前主要处理条件跳转：

- 如果 `branch` 的 then/else 目标相同，改成无条件 `jump`。
- 如果 `branch` 条件是常量，直接选定唯一可达的目标块。
- 当删掉某条 CFG 边时，同步删除目标块里 `phi` 对应的 incoming。
- 对仅合并布尔值并立即分支的 `phi` 块，如果所有动态 incoming 都能证明等于分支条件，则把前驱边直接穿透到分支目标；目前要求至少一个目标块直接返回，并拒绝会改变后继 `phi` 的情况。

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

把叶函数中反复访问的标量全局变量暂存为局部标量：入口加载一次，每个正常返回点写回一次，再由后续 `ScalarPromotePass` 提升为 SSA。候选全局必须只被直接 `load`/`store` 使用，函数内不能有调用，且候选地址不能参与 `memzero`，从而避免指针逃逸、调用可见性及别名问题。处理顺序按全局名排序并受固定数量和下游标量提升规模预算约束，保证代码生成稳定且避免大函数编译时间膨胀。

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

### `InlineSmallExprPass` (`inline.rs`)

内联小型、单基本块的纯标量表达式函数。

候选函数不能包含调用、控制流、phi 或可观察的内存操作；允许忽略没有读取或逃逸的局部只写槽位，其余只保留少量一元/二元运算、比较和类型转换。调用点会克隆这些纯指令并改写返回值的使用点，让后续 CSE、LICM 和 DCE 继续优化展开后的表达式。每个候选和调用方都有固定的代码增长预算，避免病态膨胀。

### `LocalForwardPass` (`local_forward.rs`)

基本块内的局部前向传播。

它在单个基本块内跟踪“某个可跟踪指针当前存着哪个值”：

- 遇到 `store ptr, value` 时，记录 `ptr -> value`。
- 遇到后续 `load ptr` 时，如果类型一致，就把 load 的结果替换成已知的 `value`。
- 遇到 `call` 或 `memzero` 这类可能改写内存的指令时，清空记录。

目前只跟踪非数组的本地 `alloca` 指针，避免别名关系不清导致错误优化。

### `InvariantLoadForwardPass` (`invariant_load.rs`)

转发支配当前位置且来自只读对象的重复加载。

它在闭世界的 SysY 模块内检查所有直接调用点：只有当一对指针形参在每个调用点都来自两个不同的完整全局对象时，才把它们视为不别名。函数内含未知调用、写入来源不明，或任一调用点可能别名时都会放弃。满足条件后，pass 沿支配树复用完全相同指针的已有 `load`，不会把加载推测执行到原控制流之前。

### `LoopIdiomPass` (`bit_idiom.rs`)

循环区域级整数 idiom 变换，仅接入 O1。它通过共享的 `LoopInfo`、`NaturalLoop`、i32 归纳变量与精确常量 trip-count 分析，在任意函数中逐个识别两个输入每轮除以 2、位权每轮乘以 2、按输入低位更新 accumulator 的自然循环。单轮控制流会对四组输入位做符号求值，因此可以推导全部二输入布尔真值表；局部余数、倍增和乘法重关联只接受前置 InstCombine/ConstFold 产生的规范形式。

变换只 version 目标 loop region：preheader 新增两个输入非负的 guard，fast block 合成整数位运算和不足 32 位时的掩码，负数继续进入完整原循环；唯一 exit 新增 accumulator 合并 phi，循环后的原有计算继续使用合并结果。函数返回类型、参数数量、输入是否直接来自参数、循环数量和结果是否直接返回都不参与匹配。

证明边界是保守的：必须有唯一专用 preheader、唯一 latch、唯一 exiting edge 和唯一 exit，不能有返回、内存访问、调用等 side exit/副作用；fast operands 必须在 preheader 可用；除 accumulator 外不能有 loop-defined live-out。exit 的已有 phi 只有在 fast edge 能复用其原 incoming 时才会补边，否则拒绝。变换后 preheader 不再是原循环的专用 preheader，因此重复运行幂等。

当循环结果所在的 exit 块只包含 Nop 并直接返回 accumulator 时，fast block 会直接返回综合结果，避免为未改写的 fallback 循环增加合并 phi 和寄存器压力；其它区域仍使用 exit phi 合并 live-out。

### `GepInductionPass` (`gep_induction.rs`)

对自然循环中的仿射 GEP 地址构造 pointer recurrence。候选循环必须有唯一 entering predecessor、唯一 latch 和专用 preheader；pass 从 header 的直接 i32 induction、GEP 类型大小、use-def 和支配关系证明地址为 `base + induction * constant stride + invariant offset`。循环内的 nested GEP chain 可以包含任意循环不变量索引，数组长度和正负 induction step 都不参与形状匹配。

变换在 preheader 用 induction 初值重建完整地址，在 header 插入 pointer phi，在 latch 用单索引常量 GEP 生成固定步长 next pointer。只有全部使用都位于循环内且新 phi 支配普通使用及 phi edge 时才替换原 GEP；仅作为其它已选 nested GEP base 的中间地址不会单独生成死 recurrence。常量 trip-count 分析或 header signed comparison 还必须证明 i32 induction 在每个回边不 wrap，否则 sign extension 后的重算地址与 pointer increment 并不等价，pass 会拒绝。

pass 不推测执行内存访问，也不要求唯一 exit；side exit 上没有 live-out 地址时，header pointer state 仍只沿唯一 backedge 更新。动态大步长、循环内变化的 offset、非 GEP 派生链、不能在 preheader 使用的定义、不可整除为最终 pointee stride 的步长及类型大小溢出都保守保留。变换后立即运行 verifier，重复执行不会再次匹配生成的 pointer recurrence。

### `PiecewiseExprPass` (`piecewise_expr.rs`)

解释无环纯函数中的 selector 等值决策树。当连续的 selector 范围分别返回 `x * 2^selector` 或 `x / 2^selector`，范围外返回 `x` 时，将整条决策树版本化为范围检查和动态移位快速路径。比较顺序、`if` 组织方式、参数顺序及范围端点均不固定。

左移在通过 `0..31` 范围检查后可直接使用；有符号除法只在 `x >= 0` 时使用算术右移，负数和范围外输入继续执行原函数，保持向零截断语义。非连续映射、混合乘除、额外副作用或未知分支都会让 pass 保守退出。

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

它先收集所有被指令或 terminator 使用到的 `ValueId`，然后删除那些：

- 有结果值；
- 结果值没有被使用；
- 指令本身没有副作用、可以安全移除；

的指令。删除方式是把指令结果清空，并把指令改成 `Nop`。

`store`、`call`、`memzero` 等有副作用或可能影响外部状态的指令不会被删除。

## 支撑模块

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

LICM、`SimpleLoopUnrollPass` 和 `LoopIdiomPass` 的 CFG/区域改写要求 dedicated preheader；`RepeatReductionPass` 与归纳分析只要求 unique entering predecessor。各 pass 复用循环结构和归纳变量描述，再施加自身的严格变换门控。LICM 按 invariant use-def 拓扑一次把定义先于使用移入 preheader，避免 BlockId 逆序依赖链上的反复全循环扫描。因此该模块是普通循环优化的公共基础设施，不是某个整数 idiom 的私有 matcher。

### `util.rs`

共享的 Value 改写工具，不是一个独立 pass。

它提供：

- `ValueReplacements = HashMap<ValueId, ValueId>`：记录值替换关系。
- `rewrite_function_uses`：把函数里所有操作数按替换表改写。
- `resolve_replacement`：追踪链式替换，保证 `a -> b -> c` 最终解析到 `c`。
