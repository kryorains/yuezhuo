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
  4. `ScalarPromotePass`
  5. `InlineSmallExprPass`
  6. `LocalForwardPass`
  7. `CsePass`
  8. `LicmPass`
  9. `InvariantLoadForwardPass`
  10. `DcePass`
  11. `BitwiseIdiomPass`
  12. `PiecewiseExprPass`
  13. `RepeatReductionPass`
  14. `SimpleLoopUnrollPass`
  15. `ConstFoldPass`
  16. `SimplifyCfgPass`
  17. `DcePass`

O0 也做标量提升，是为了给代码生成器提供统一的 SSA/phi 形式；不会执行常量折叠、CSE、LICM 等主动改写表达式的优化。流水线中的前置 DCE 会先清掉标量提升遗留的死 phi，末尾 DCE 再清理归约折叠后失效的循环记账指令。

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

### `SimplifyCfgPass` (`simplify_cfg.rs`)

简化控制流图。

目前主要处理条件跳转：

- 如果 `branch` 的 then/else 目标相同，改成无条件 `jump`。
- 如果 `branch` 条件是常量，直接选定唯一可达的目标块。
- 当删掉某条 CFG 边时，同步删除目标块里 `phi` 对应的 incoming。

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

### `BitwiseIdiomPass` (`bit_idiom.rs`)

按 SSA 递推语义识别 bit-sliced 整数运算循环。候选必须是无副作用的纯标量 helper，并包含两个逐轮 `/ 2` 的输入、逐轮 `* 2` 的位权、有限位数倒计时和按条件累加位权的结果。pass 对一轮循环的四组输入位做符号执行，从真值表综合出整数 `and/or/xor` 表达式；它不读取变量名、块名、块编号，也不依赖固定的指令数量。

迭代次数可以是 1 到 32；不足 32 位时自动添加低位掩码。原循环作为慢路径完整保留，只有两个输入都非负时才执行原生位运算快速路径，从而保持 SysY 有符号除法和取模对负数的语义。

### `PiecewiseExprPass` (`piecewise_expr.rs`)

解释无环纯函数中的 selector 等值决策树。当连续的 selector 范围分别返回 `x * 2^selector` 或 `x / 2^selector`，范围外返回 `x` 时，将整条决策树版本化为范围检查和动态移位快速路径。比较顺序、`if` 组织方式、参数顺序及范围端点均不固定。

左移在通过 `0..31` 范围检查后可直接使用；有符号除法只在 `x >= 0` 时使用算术右移，负数和范围外输入继续执行原函数，保持向零截断语义。非连续映射、混合乘除、额外副作用或未知分支都会让 pass 保守退出。

### `RepeatReductionPass` (`repeat_reduction.rs`)

无副作用重复归约折叠。

它识别从 0 开始、每轮加 1 的计数循环；当循环体没有 `store`、`call`、`memzero` 等副作用，并且唯一可观察的循环状态满足 `acc' = acc + delta` 时，把重复执行改写为：

1. 执行一次原循环体，得到单轮增量 `delta`；
2. 计算 `initial + delta * count`；
3. 把计数器直接推进到循环上界并退出。

变换使用 i32 环绕算术，因此与逐轮累加在模 2^32 下等价。计数器参与循环体计算、非线性累加、额外活跃 loop phi、依赖累加器的分支、循环内存写入及侧出口都会让该 pass 保守退出。

### `SimpleLoopUnrollPass` (`simple_loop_unroll.rs`)

对严格规范化的单基本块计数循环做二倍展开。

候选循环必须从 0 开始、每轮加 1、以动态上界做有符号小于比较，并且只有一个活跃 loop phi；含调用、`memzero`、侧出口或额外循环状态时不会展开。pass 在原标量循环前插入两路快速循环，原循环继续处理负数、小于 2 的次数和奇数尾项。两份循环体严格按迭代顺序克隆，因此即使相邻迭代的内存访问互相别名，也不会改变可观察顺序。代码增长受单循环和单函数预算限制。当前 AArch64 后端会让展开后的中间值产生额外栈流量，因此目标收益门控暂时只在 x86-64 和 RISC-V64 启用该 pass。

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
  - 每个块的支配集合；
  - 支配树 children；
  - dominance frontier。

`ScalarPromotePass` 用它来判断定义是否支配使用，以及在哪里需要插入 `phi`。

### `util.rs`

共享的 Value 改写工具，不是一个独立 pass。

它提供：

- `ValueReplacements = HashMap<ValueId, ValueId>`：记录值替换关系。
- `rewrite_function_uses`：把函数里所有操作数按替换表改写。
- `resolve_replacement`：追踪链式替换，保证 `a -> b -> c` 最终解析到 `c`。
