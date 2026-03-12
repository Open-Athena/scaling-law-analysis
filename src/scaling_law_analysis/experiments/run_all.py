"""Run all experiments."""

from scaling_law_analysis.experiments import (
    exp0_reproductions,
    exp1_empirical_error,
    exp2_exponent_imbalance,
    exp3_drift_sensitivity,
    exp4_extrapolation_error,
    exp5_parameter_recovery,
    exp6_analytical_error,
    exp7_exponent_inference,
    exp8_conditioning,
    exp9_data_efficiency,
    exp10_compounding_errors,
    exp11_cost_estimates,
)


def main():
    """Run all experiments in sequence."""
    print("\n" + "=" * 80)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 80)

    exp0_reproductions.main()

    print("\n")
    exp1_empirical_error.main()

    print("\n")
    exp2_exponent_imbalance.main()

    print("\n")
    exp3_drift_sensitivity.main()

    print("\n")
    exp4_extrapolation_error.main()

    print("\n")
    exp5_parameter_recovery.main()

    print("\n")
    exp6_analytical_error.main()

    print("\n")
    exp7_exponent_inference.main()

    print("\n")
    exp8_conditioning.main()

    print("\n")
    exp9_data_efficiency.main()

    print("\n")
    exp10_compounding_errors.main()

    print("\n")
    exp11_cost_estimates.main()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
