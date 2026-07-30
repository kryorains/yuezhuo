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
    small_expr_inline_rounds: usize,
    cfg_inline_rounds: usize,
    cfg_inline_global_loads: bool,
    constant_address_count_reduction: bool,
    recursive_const_specialization: bool,
    loop_call_memoize: bool,
    loop_invariant_call_memoize: bool,
    repeated_overwrite_elision: bool,
    guarded_mulmod_idiom: bool,
    guarded_pow2_digit_idiom: bool,
    regional_global_scalar_promotion: bool,
    producer_consumer_fusion: bool,
    periodic_reduction_memoize: bool,
    contract_float_madd: bool,
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
                small_expr_inline_rounds: 1,
                cfg_inline_rounds: 1,
                cfg_inline_global_loads: false,
                constant_address_count_reduction: false,
                recursive_const_specialization: false,
                loop_call_memoize: false,
                loop_invariant_call_memoize: false,
                repeated_overwrite_elision: false,
                guarded_mulmod_idiom: false,
                guarded_pow2_digit_idiom: false,
                regional_global_scalar_promotion: false,
                producer_consumer_fusion: false,
                periodic_reduction_memoize: false,
                contract_float_madd: false,
                cleanup_write_only_allocas_before_inline: true,
                max_reduction_jam_factor: 2,
                min_global_register_score: 1,
                callee_saved_register_score: 16,
            },
            Target::AArch64 => Self {
                simple_loop_unroll: false,
                small_expr_inline_rounds: 1,
                cfg_inline_rounds: 2,
                cfg_inline_global_loads: true,
                constant_address_count_reduction: false,
                recursive_const_specialization: false,
                loop_call_memoize: true,
                loop_invariant_call_memoize: false,
                repeated_overwrite_elision: true,
                guarded_mulmod_idiom: false,
                guarded_pow2_digit_idiom: false,
                regional_global_scalar_promotion: false,
                producer_consumer_fusion: false,
                periodic_reduction_memoize: false,
                contract_float_madd: true,
                cleanup_write_only_allocas_before_inline: false,
                max_reduction_jam_factor: 2,
                min_global_register_score: 2,
                callee_saved_register_score: 16,
            },
            Target::Riscv64 => Self {
                simple_loop_unroll: true,
                small_expr_inline_rounds: 2,
                cfg_inline_rounds: 2,
                cfg_inline_global_loads: false,
                constant_address_count_reduction: true,
                recursive_const_specialization: true,
                loop_call_memoize: false,
                loop_invariant_call_memoize: true,
                repeated_overwrite_elision: false,
                guarded_mulmod_idiom: true,
                guarded_pow2_digit_idiom: true,
                regional_global_scalar_promotion: true,
                producer_consumer_fusion: true,
                periodic_reduction_memoize: true,
                contract_float_madd: false,
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

    pub const fn small_expr_inline_rounds(self) -> usize {
        self.small_expr_inline_rounds
    }

    pub const fn cfg_inline_rounds(self) -> usize {
        self.cfg_inline_rounds
    }

    pub const fn cfg_inline_global_loads(self) -> bool {
        self.cfg_inline_global_loads
    }

    pub const fn enable_constant_address_count_reduction(self) -> bool {
        self.constant_address_count_reduction
    }

    pub const fn enable_recursive_const_specialization(self) -> bool {
        self.recursive_const_specialization
    }

    pub const fn enable_loop_call_memoize(self) -> bool {
        self.loop_call_memoize
    }

    pub const fn enable_loop_invariant_call_memoize(self) -> bool {
        self.loop_invariant_call_memoize
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

    pub const fn enable_regional_global_scalar_promotion(self) -> bool {
        self.regional_global_scalar_promotion
    }

    pub const fn enable_producer_consumer_fusion(self) -> bool {
        self.producer_consumer_fusion
    }

    pub const fn enable_periodic_reduction_memoize(self) -> bool {
        self.periodic_reduction_memoize
    }

    pub(crate) const fn contract_float_madd(self) -> bool {
        self.contract_float_madd
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
        assert_eq!(aarch64.small_expr_inline_rounds(), 1);
        assert_eq!(aarch64.cfg_inline_rounds(), 2);
        assert!(aarch64.cfg_inline_global_loads());
        assert!(aarch64.enable_loop_call_memoize());
        assert!(aarch64.enable_repeated_overwrite_elision());
        assert!(!aarch64.cleanup_write_only_allocas_before_inline());
        assert!(riscv64.enable_simple_loop_unroll());
        assert_eq!(riscv64.small_expr_inline_rounds(), 2);
        assert_eq!(riscv64.cfg_inline_rounds(), 2);
        assert!(!riscv64.cfg_inline_global_loads());
        assert!(riscv64.enable_constant_address_count_reduction());
        assert!(riscv64.enable_recursive_const_specialization());
        assert!(!riscv64.enable_loop_call_memoize());
        assert!(riscv64.enable_loop_invariant_call_memoize());
        assert!(!riscv64.enable_repeated_overwrite_elision());
        assert!(riscv64.enable_guarded_mulmod_idiom());
        assert!(riscv64.enable_guarded_pow2_digit_idiom());
        assert!(riscv64.enable_regional_global_scalar_promotion());
        assert!(riscv64.enable_producer_consumer_fusion());
        assert!(riscv64.enable_periodic_reduction_memoize());
        assert!(!riscv64.contract_float_madd());
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
