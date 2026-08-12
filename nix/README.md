# Nix 开发环境

进入默认开发环境：

```sh
nix develop
```

默认 Git 身份为 `kryorains <kryorains@kryorains.io>`。环境包含 Rust 工具链、
RISC-V64 Linux 交叉编译器，以及测试脚本依赖的 GNU 工具。

## Linux 测试虚拟机

Darwin 无法运行 QEMU user-mode，因此开发环境提供了基于 QEMU、Alpine Linux
AArch64 cloud image 和 HVF 加速的轻量虚拟机：

```sh
yuezhuo-vm up
yuezhuo-vm ssh
yuezhuo-vm status
yuezhuo-vm down
```

第一次执行 `up` 会初始化位于 `target/qemu-vm` 的可写磁盘、cloud-init seed 和
专用 SSH 密钥。guest 使用镜像规定的 `alpine` 登录用户。基础镜像由 Nix 下载并
校验，项目不会修改 Nix store 中的镜像。

运行单个由项目交叉链接出的 Linux ELF：

```sh
yuezhuo-vm run --target riscv64 target/test-work/example
```

运行现有测试脚本：

```sh
yuezhuo-vm test-functional riscv64
yuezhuo-vm test-perf riscv64
```

VM 通过只读 9p 挂载宿主的 `/nix/store`，因此可以使用 Nix 交叉工具链的 glibc，
也能在 guest 内使用 Linux 版 QEMU user-mode 执行 RISC-V64 ELF。

可通过以下环境变量调整运行参数：

- `YUEZHUO_VM_STATE`：运行状态目录。
- `YUEZHUO_VM_SSH_PORT`：SSH 转发端口，默认 `2222`。
- `YUEZHUO_VM_MEMORY`：内存 MiB，默认 `2048`。
- `YUEZHUO_VM_CPUS`：虚拟 CPU 数量，默认 `4`。
- `YUEZHUO_VM_ACCELERATOR`、`YUEZHUO_VM_CPU`：覆盖 QEMU 加速器和 CPU。
