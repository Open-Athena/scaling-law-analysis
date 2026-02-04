"""Run all experiments."""

from scaling_law_analysis.experiments import (
    exp1_empirical_error,
    exp2_exponent_imbalance,
    exp3_drift_sensitivity,
    exp4_extrapolation_error,
)


def main():
    """Run all experiments in sequence."""
    print("\n" + "=" * 80)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 80)

    exp1_empirical_error.main()

    print("\n")
    exp2_exponent_imbalance.main()

    print("\n")
    exp3_drift_sensitivity.main()

    print("\n")
    exp4_extrapolation_error.main()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
