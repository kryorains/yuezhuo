# IR Pass 说明

这个目录放的是 IR 层的优化 pass，以及这些 pass 共享的一些控制流/改写工具。

## 优化流水线

入口在 `mod.rs` 的 `run_pipeline`：

- `OptLevel::O0`：不跑任何优化 pass。
- `OptLevel::O1`：按下面顺序执行：
  1. `ConstFoldPass`
  2. `SimplifyCfgPass`
  3. `TailRecursionPass`
  4. `ScalarPromotePass`
  5. `LocalForwardPass`
  6. `CsePass`
  7. `LicmPass`
  8. `ConstFoldPass`
  9. `SimplifyCfgPass`
  10. `DcePass`

这里会重复跑常量折叠和 CFG 简化，因为前面的 pass 可能制造新的常量、死分支或无用指令，后面的 pass 再把这些机会吃掉。

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

### `LocalForwardPass` (`local_forward.rs`)

基本块内的局部前向传播。

它在单个基本块内跟踪“某个可跟踪指针当前存着哪个值”：

- 遇到 `store ptr, value` 时，记录 `ptr -> value`。
- 遇到后续 `load ptr` 时，如果类型一致，就把 load 的结果替换成已知的 `value`。
- 遇到 `call` 或 `memzero` 这类可能改写内存的指令时，清空记录。

目前只跟踪非数组的本地 `alloca` 指针，避免别名关系不清导致错误优化。

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
