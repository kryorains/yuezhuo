# codegen 结构说明

`src/codegen` 负责把已经 lowered 的 IR 输出成目标平台汇编。当前支持：

- `x86_64`
- `aarch64`
- `riscv64`

## 目录分层

- `mod.rs`：选择目标平台并分发到具体后端。
- `common/`：与具体指令集无关的公共逻辑。
  - `layout.rs`：计算每个 IR value 在栈帧里的位置。
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

## 关键约定

1. 大多数后端把“当前表达式结果”放在固定临时寄存器里：
   - x86_64：整数/指针在 `%rax`，浮点临时常用 `%xmm0`。
   - AArch64：整数/指针在 `x0/w0`，浮点在 `s0`。
   - RISC-V64：整数/指针在 `a0`，浮点在 `fa0`。
2. IR value 若需要跨指令使用，都会落到 `IrFuncLayout` 分配的栈槽。
3. `alloca` 的 value 栈槽保存“对象地址”，对象本体紧跟在这个地址槽之后。
4. Phi 不是 SSA 消除 pass 里处理，而是在 terminator 跳边时生成 copy。
