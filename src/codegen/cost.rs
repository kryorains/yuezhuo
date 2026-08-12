use super::Target;

/// Target-specific profitability thresholds shared by IR transforms and
/// register allocation.
///
/// Hard analysis budgets remain next to the analysis that they protect. This
/// model contains only target costs: decisions that may legitimately differ
/// because an optimization has a different runtime price on each ISA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetCostModel {
    small_expr_inline_rounds: usize,
    cfg_inline_rounds: usize,
    cfg_inline_global_loads: bool,
    cfg_inline_global_stores: bool,
    recursive_inline_rounds: usize,
    constant_address_count_reduction: bool,
    recursive_const_specialization: bool,
    initialized_global_propagation: bool,
    uniform_constant_arguments: bool,
    loop_invariant_call_memoize: bool,
    regional_global_scalar_promotion: bool,
    full_domain_bitwise_digit: bool,
    cleanup_write_only_allocas_before_inline: bool,
    max_reduction_jam_factor: usize,
    callee_saved_register_score: usize,
}

impl TargetCostModel {
    pub(super) const fn for_target(target: Target) -> Self {
        match target {
            Target::Riscv64 => Self {
                small_expr_inline_rounds: 2,
                cfg_inline_rounds: 2,
                cfg_inline_global_loads: true,
                cfg_inline_global_stores: true,
                recursive_inline_rounds: 2,
                constant_address_count_reduction: true,
                recursive_const_specialization: true,
                initialized_global_propagation: true,
                uniform_constant_arguments: true,
                loop_invariant_call_memoize: true,
                regional_global_scalar_promotion: true,
                full_domain_bitwise_digit: true,
                cleanup_write_only_allocas_before_inline: true,
                max_reduction_jam_factor: 4,
                callee_saved_register_score: 16,
            },
        }
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

    pub const fn cfg_inline_global_stores(self) -> bool {
        self.cfg_inline_global_stores
    }

    pub const fn recursive_inline_rounds(self) -> usize {
        self.recursive_inline_rounds
    }

    pub const fn enable_constant_address_count_reduction(self) -> bool {
        self.constant_address_count_reduction
    }

    pub const fn enable_recursive_const_specialization(self) -> bool {
        self.recursive_const_specialization
    }

    pub const fn enable_initialized_global_propagation(self) -> bool {
        self.initialized_global_propagation
    }

    pub const fn enable_uniform_constant_arguments(self) -> bool {
        self.uniform_constant_arguments
    }

    pub const fn enable_loop_invariant_call_memoize(self) -> bool {
        self.loop_invariant_call_memoize
    }

    pub const fn enable_regional_global_scalar_promotion(self) -> bool {
        self.regional_global_scalar_promotion
    }

    pub const fn enable_full_domain_bitwise_digit(self) -> bool {
        self.full_domain_bitwise_digit
    }

    pub const fn cleanup_write_only_allocas_before_inline(self) -> bool {
        self.cleanup_write_only_allocas_before_inline
    }

    pub const fn max_reduction_jam_factor(self) -> usize {
        self.max_reduction_jam_factor
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
    fn riscv64_profile_keeps_measured_transform_choices_explicit() {
        let riscv64 = TargetCostModel::for_target(Target::Riscv64);

        assert_eq!(riscv64.small_expr_inline_rounds(), 2);
        assert_eq!(riscv64.cfg_inline_rounds(), 2);
        assert!(riscv64.cfg_inline_global_loads());
        assert!(riscv64.cfg_inline_global_stores());
        assert_eq!(riscv64.recursive_inline_rounds(), 2);
        assert!(riscv64.enable_constant_address_count_reduction());
        assert!(riscv64.enable_recursive_const_specialization());
        assert!(riscv64.enable_initialized_global_propagation());
        assert!(riscv64.enable_uniform_constant_arguments());
        assert!(riscv64.enable_loop_invariant_call_memoize());
        assert!(riscv64.enable_regional_global_scalar_promotion());
        assert!(riscv64.enable_full_domain_bitwise_digit());
        assert!(riscv64.cleanup_write_only_allocas_before_inline());
        assert_eq!(riscv64.max_reduction_jam_factor(), 4);
    }
}
