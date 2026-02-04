"""Experiment 4: Extrapolation Error Analysis.

This experiment investigates how the accuracy of scaling law exponent inference
degrades when extrapolating to compute budgets beyond those used for fitting.

Hypothesis: Errors in inferred scaling exponents compound when extrapolating
to compute budgets beyond the fitting regime, with magnitude depending on
sampling range and loss surface geometry.
"""

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import fit_approach2
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    LOSS_SURFACES,
    BIAS_CONFIGS,
    DISPLAY_LOG_RANGES,
    DISPLAY_LOG_RANGE_NAMES,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    EXTRAPOLATION_BUDGETS,
    N_POINTS,
    run_extrapolation_analysis,
    create_extrapolation_figure,
)


def approach2_fitter(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_range: float,
    n_points: int,
):
    """Fit using Approach 2 and return D_opt function."""
    result = fit_approach2(
        compute_budgets=compute_budgets,
        surface=sim_config.loss,
        drift_rate=sim_config.drift_rate,
        center_scale=sim_config.center_scale,
        n_points=n_points,
        log_range=log_range,
    )
    return result.D_opt


def main():
    """Run Experiment 4: analyze extrapolation error across configurations."""
    print("=" * 70)
    print("Experiment 4: Extrapolation Error Analysis")
    print("=" * 70)

    # Prepare output directory
    output_dir = prepare_output_dir(config.RESULTS_DIR / "exp4")

    print(f"\nFitting compute budgets: {COMPUTE_BUDGETS}")
    print(
        f"Extrapolation budgets: {EXTRAPOLATION_BUDGETS[0]:.0e} to {EXTRAPOLATION_BUDGETS[-1]:.0e}"
    )
    print(f"\nLoss surfaces: {[name for name, _ in LOSS_SURFACES]}")

    # Run extrapolation analysis
    all_results = run_extrapolation_analysis(
        fitter=approach2_fitter,
        compute_budgets=COMPUTE_BUDGETS,
        extrapolation_budgets=EXTRAPOLATION_BUDGETS,
        log_ranges=DISPLAY_LOG_RANGES,
        log_range_names=DISPLAY_LOG_RANGE_NAMES,
        n_points=N_POINTS,
        loss_surfaces=LOSS_SURFACES,
        bias_configs=BIAS_CONFIGS,
    )

    # Create and save extrapolation error figure
    print(f"\n{'─' * 70}")
    print("Generating extrapolation error figure...")
    extrap_fig = create_extrapolation_figure(
        all_results=all_results,
        loss_surfaces=LOSS_SURFACES,
        log_range_names=DISPLAY_LOG_RANGE_NAMES,
        log_ranges=DISPLAY_LOG_RANGES,
        title="Experiment 4: Extrapolation Error Analysis (Approach 2)",
        subtitle="Fitting: 10¹⁷-10²¹ FLOPs → Extrapolating to 10²²-10²⁵ FLOPs",
    )
    extrap_path = output_dir / "extrapolation_error.png"
    extrap_fig.savefig(extrap_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {extrap_path}")
    plt.close(extrap_fig)

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Maximum D* extrapolation errors")
    print("=" * 70)

    for range_name in DISPLAY_LOG_RANGE_NAMES:
        print(f"\n{range_name.upper()} sampling range:")

        for surface_name, _ in LOSS_SURFACES:
            print(f"\n  {surface_name}:")
            print(f"  {'Config':<15} {'max_D_err%':>12}")
            print(f"  {'-' * 30}")

            for results in all_results[range_name][surface_name]:
                sim_config = results["config"]
                max_D_err = np.abs(results["D_rel_errors"]).max() * 100
                print(f"  {sim_config.name:<15} {max_D_err:>12.2f}")

    print("\nExperiment 4 complete.")
    return all_results


if __name__ == "__main__":
    main()
