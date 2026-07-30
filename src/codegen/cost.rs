use super::Target;

/// Target-specific profitability thresholds shared by IR transforms and
/// register allocation.
///
/// Hard analysis budgets remain next to the analysis that they protect. This
/// model contains only target costs: decisions that may legitimately differ
/// because an optimization has a different runtime price on each ISA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetCostModel {
    simple_loop_unroll: bool,
    cfg_inline_rounds: usize,
    cfg_inline_global_loads: bool,
    loop_call_memoize: bool,
    repeated_overwrite_elision: bool,
    guarded_mulmod_idiom: bool,
    guarded_pow2_digit_idiom: bool,
    cleanup_write_only_allocas_before_inline: bool,
    max_reduction_jam_factor: usize,
    min_global_register_score: usize,
    callee_saved_register_score: usize,
}

impl TargetCostModel {
    pub(super) const fn for_target(target: Target) -> Self {
        match target {
            Target::X86_64 => Self {
                simple_loop_unroll: true,
                cfg_inline_rounds: 1,
                cfg_inline_global_loads: false,
                loop_call_memoize: false,
                repeated_overwrite_elision: false,
                guarded_mulmod_idiom: false,
                guarded_pow2_digit_idiom: false,
                cleanup_write_only_allocas_before_inline: true,
                max_reduction_jam_factor: 2,
                min_global_register_score: 1,
                callee_saved_register_score: 16,
            },
            Target::AArch64 => Self {
                simple_loop_unroll: false,
                cfg_inline_rounds: 2,
                cfg_inline_global_loads: true,
                loop_call_memoize: true,
                repeated_overwrite_elision: true,
                guarded_mulmod_idiom: false,
                guarded_pow2_digit_idiom: false,
                cleanup_write_only_allocas_before_inline: false,
                max_reduction_jam_factor: 2,
                min_global_register_score: 2,
                callee_saved_register_score: 16,
            },
            Target::Riscv64 => Self {
                simple_loop_unroll: true,
                cfg_inline_rounds: 1,
                cfg_inline_global_loads: false,
                loop_call_memoize: false,
                repeated_overwrite_elision: false,
                guarded_mulmod_idiom: true,
                guarded_pow2_digit_idiom: true,
                cleanup_write_only_allocas_before_inline: true,
                max_reduction_jam_factor: 4,
                min_global_register_score: 1,
                callee_saved_register_score: 16,
            },
        }
    }

    pub const fn enable_simple_loop_unroll(self) -> bool {
        self.simple_loop_unroll
    }

    pub const fn cfg_inline_rounds(self) -> usize {
        self.cfg_inline_rounds
    }

    pub const fn cfg_inline_global_loads(self) -> bool {
        self.cfg_inline_global_loads
    }

    pub const fn enable_loop_call_memoize(self) -> bool {
        self.loop_call_memoize
    }

    pub const fn enable_repeated_overwrite_elision(self) -> bool {
        self.repeated_overwrite_elision
    }

    pub const fn enable_guarded_mulmod_idiom(self) -> bool {
        self.guarded_mulmod_idiom
    }

    pub const fn enable_guarded_pow2_digit_idiom(self) -> bool {
        self.guarded_pow2_digit_idiom
    }

    pub const fn cleanup_write_only_allocas_before_inline(self) -> bool {
        self.cleanup_write_only_allocas_before_inline
    }

    pub const fn max_reduction_jam_factor(self) -> usize {
        self.max_reduction_jam_factor
    }

    pub(crate) const fn should_allocate_global_register(
        self,
        weighted_score: usize,
        is_phi: bool,
        is_phi_copy: bool,
        live_across_call: bool,
    ) -> bool {
        is_phi
            || is_phi_copy
            || live_across_call
            || weighted_score >= self.min_global_register_score
    }

    pub(crate) const fn should_use_callee_saved_register(
        self,
        weighted_score: usize,
        live_across_call: bool,
    ) -> bool {
        live_across_call || weighted_score >= self.callee_saved_register_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_profiles_keep_measured_transform_differences_explicit() {
        let aarch64 = TargetCostModel::for_target(Target::AArch64);
        let riscv64 = TargetCostModel::for_target(Target::Riscv64);

        assert!(!aarch64.enable_simple_loop_unroll());
        assert_eq!(aarch64.cfg_inline_rounds(), 2);
        assert!(aarch64.cfg_inline_global_loads());
        assert!(aarch64.enable_loop_call_memoize());
        assert!(aarch64.enable_repeated_overwrite_elision());
        assert!(!aarch64.cleanup_write_only_allocas_before_inline());
        assert!(riscv64.enable_simple_loop_unroll());
        assert_eq!(riscv64.cfg_inline_rounds(), 1);
        assert!(!riscv64.cfg_inline_global_loads());
        assert!(!riscv64.enable_loop_call_memoize());
        assert!(!riscv64.enable_repeated_overwrite_elision());
        assert!(riscv64.enable_guarded_mulmod_idiom());
        assert!(riscv64.enable_guarded_pow2_digit_idiom());
        assert!(riscv64.cleanup_write_only_allocas_before_inline());
        assert_eq!(riscv64.max_reduction_jam_factor(), 4);
    }

    #[test]
    fn phi_copy_savings_can_pay_for_a_cold_global_register() {
        let aarch64 = TargetCostModel::for_target(Target::AArch64);

        assert!(!aarch64.should_allocate_global_register(1, false, false, false));
        assert!(aarch64.should_allocate_global_register(1, false, true, false));
        assert!(aarch64.should_allocate_global_register(1, false, false, true));
    }
}
