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
from scaling_law_analysis.chinchilla import LossSurface, fit_approach2
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    LOSS_SURFACES,
    BIAS_CONFIGS,
    DISPLAY_LOG_RANGES,
    DISPLAY_LOG_RANGE_NAMES,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    N_POINTS,
)
from scaling_law_analysis.experiments.exp1_empirical_error import log_range_to_label

# Extrapolation compute budgets: 10^22 to 10^25 FLOPs (beyond fitting range)
EXTRAPOLATION_BUDGETS = np.geomspace(1e22, 1e25, 16)


def compute_extrapolation_errors(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    extrapolation_budgets: np.ndarray,
    log_range: float,
    n_points: int,
) -> dict:
    """Compute extrapolation errors for a given configuration.

    Fits scaling law exponents using the provided compute budgets, then
    extrapolates to higher compute budgets and compares inferred vs true D*.

    Args:
        sim_config: Simulation configuration
        compute_budgets: Compute budgets for fitting (FLOPs)
        extrapolation_budgets: Compute budgets for extrapolation (FLOPs)
        log_range: Sampling range in log10 space
        n_points: Number of points per IsoFLOP curve

    Returns:
        Dictionary with extrapolation error results
    """
    loss = sim_config.loss

    # Fit Approach 2 to get inferred exponents and intercepts
    result = fit_approach2(
        compute_budgets=compute_budgets,
        surface=loss,
        drift_rate=sim_config.drift_rate,
        center_scale=sim_config.center_scale,
        n_points=n_points,
        log_range=log_range,
    )

    # Compute true and inferred D* at each extrapolation budget
    true_D_opts = np.array([loss.D_opt(C) for C in extrapolation_budgets])
    inferred_D_opts = np.array([result.D_opt(C) for C in extrapolation_budgets])

    # Relative error in D*
    D_rel_errors = (inferred_D_opts - true_D_opts) / true_D_opts

    return {
        "config": sim_config,
        "log_range": log_range,
        "extrapolation_budgets": extrapolation_budgets,
        "true_D_opts": true_D_opts,
        "inferred_D_opts": inferred_D_opts,
        "D_rel_errors": D_rel_errors,
        "fitted_a": result.a,
        "fitted_b": result.b,
        "true_a": loss.a,
        "true_b": loss.b,
        "approach2_result": result,
    }


def run_all_configurations(
    compute_budgets: np.ndarray,
    extrapolation_budgets: np.ndarray,
    log_ranges: list[float],
    n_points: int,
) -> dict[str, dict[str, list[dict]]]:
    """Run extrapolation analysis for all configurations.

    Returns:
        Nested dict: log_range_name -> surface_name -> list of results (one per bias config)
    """
    all_results: dict[str, dict[str, list[dict]]] = {}

    for log_range, range_name in zip(log_ranges, DISPLAY_LOG_RANGE_NAMES):
        print(f"\n{'#' * 70}")
        print(f"Sampling Range: {range_name} ({log_range_to_label(log_range)})")
        print(f"{'#' * 70}")

        all_results[range_name] = {}

        for surface_name, loss in LOSS_SURFACES:
            print(f"\n{'=' * 70}")
            print(f"Loss Surface: {surface_name}")
            print(f"  α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}")
            print("=" * 70)

            surface_results = []

            for drift_rate, center_scale, bias_name in BIAS_CONFIGS:
                sim_config = SimulationConfig(
                    name=bias_name,
                    loss=loss,
                    drift_rate=drift_rate,
                    center_scale=center_scale,
                )

                print(f"\n  Configuration: {bias_name}")

                results = compute_extrapolation_errors(
                    sim_config=sim_config,
                    compute_budgets=compute_budgets,
                    extrapolation_budgets=extrapolation_budgets,
                    log_range=log_range,
                    n_points=n_points,
                )
                surface_results.append(results)

                # Print summary
                print(f"    Fitted: a={results['fitted_a']:.4f} (true={results['true_a']:.4f})")
                print(f"    Fitted: b={results['fitted_b']:.4f} (true={results['true_b']:.4f})")
                print(f"    Max D* error: {np.abs(results['D_rel_errors']).max() * 100:.2f}%")

            all_results[range_name][surface_name] = surface_results

    return all_results


def create_extrapolation_figure(
    all_results: dict[str, dict[str, list[dict]]],
) -> plt.Figure:
    """Create extrapolation error figure (3 rows × 3 cols).

    Rows: one per sampling range (narrow, medium, wide)
    Cols: one per loss surface (symmetric, chinchilla, high_imbalance)
    """
    n_ranges = len(DISPLAY_LOG_RANGE_NAMES)
    n_surfaces = len(LOSS_SURFACES)
    fig, axes = plt.subplots(n_ranges, n_surfaces, figsize=(5 * n_surfaces, 4 * n_ranges))

    for row, range_name in enumerate(DISPLAY_LOG_RANGE_NAMES):
        log_range = DISPLAY_LOG_RANGES[row]
        range_label = log_range_to_label(log_range)

        for col, (surface_name, loss) in enumerate(LOSS_SURFACES):
            ax = axes[row, col]
            results_list = all_results[range_name][surface_name]
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(results_list)))

            for i, results in enumerate(results_list):
                sim_config = results["config"]
                budgets = results["extrapolation_budgets"]
                D_rel_errors = results["D_rel_errors"] * 100  # Convert to %

                label = sim_config.name
                ax.plot(
                    budgets,
                    D_rel_errors,
                    "o-",
                    color=colors[i],
                    markersize=4,
                    label=label,
                )

            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Compute budget (FLOPs)")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

            # Title with loss surface and sampling range info
            ratio = loss.alpha / loss.beta
            ax.set_title(
                f"{surface_name} / {range_name} ({range_label})\n"
                f"α={loss.alpha:.2f}, β={loss.beta:.2f}, ratio={ratio:.2f}",
                fontsize=9,
            )

            ax.set_ylabel("Relative error in D* (%)")

            # Show legend only in top-right panel
            if row == 0 and col == n_surfaces - 1:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Experiment 4: Extrapolation Error Analysis\n"
        "Fitting: compute budgets 10¹⁷-10²¹ FLOPs → Extrapolating to 10²²-10²⁵ FLOPs",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()

    return fig


def main():
    """Run Experiment 4: analyze extrapolation error across configurations."""
    print("=" * 70)
    print("Experiment 4: Extrapolation Error Analysis")
    print("=" * 70)

    compute_budgets = COMPUTE_BUDGETS
    extrapolation_budgets = EXTRAPOLATION_BUDGETS
    log_ranges = DISPLAY_LOG_RANGES
    n_points = N_POINTS

    # Prepare output directory
    output_dir = prepare_output_dir(config.RESULTS_DIR / "exp4")

    print(f"\nFitting compute budgets: {compute_budgets}")
    print(f"Extrapolation budgets: {extrapolation_budgets[0]:.0e} to {extrapolation_budgets[-1]:.0e}")
    print(f"Sampling ranges: {[log_range_to_label(lr) for lr in log_ranges]}")
    print(f"\nLoss surfaces: {[name for name, _ in LOSS_SURFACES]}")

    # Run all configurations
    all_results = run_all_configurations(
        compute_budgets=compute_budgets,
        extrapolation_budgets=extrapolation_budgets,
        log_ranges=log_ranges,
        n_points=n_points,
    )

    # Create and save extrapolation error figure
    print(f"\n{'─' * 70}")
    print("Generating extrapolation error figure...")
    extrap_fig = create_extrapolation_figure(all_results)
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
            print(f"  {'Config':<15} {'drift':>8} {'scale':>8} {'max_D_err%':>12}")
            print(f"  {'-' * 50}")

            for results in all_results[range_name][surface_name]:
                sim_config = results["config"]
                max_D_err = np.abs(results["D_rel_errors"]).max() * 100
                print(
                    f"  {sim_config.name:<15} "
                    f"{sim_config.drift_rate:>8.1f} "
                    f"{sim_config.center_scale:>8.1f} "
                    f"{max_D_err:>12.2f}"
                )

    print("\nExperiment 4 complete.")
    return all_results


if __name__ == "__main__":
    main()
