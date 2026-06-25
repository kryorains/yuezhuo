pub mod aarch64;
pub mod asm;
pub mod common;
pub mod riscv64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    X86_64,
    AArch64,
    Riscv64,
}
