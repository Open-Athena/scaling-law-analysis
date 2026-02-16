"""Experiment 7: Data Efficiency Comparison.

Compares how Chinchilla Approach 2, Approach 3, and VPNLS recover the
scaling exponents a = beta/(alpha+beta) and b = alpha/(alpha+beta) as a
function of the number of points per isoflop curve.

Uses the Asymmetric surface (alpha/beta ratio = 3) with a drifting sampling
bias (drift_rate = log10(3)), matching the "Compounding Errors" section
of the article.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    FitError,
    fit_approach2,
    fit_approach3,
    fit_surface,
)
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    ASYMMETRIC_CONFIG,
    COMPUTE_BUDGETS,
    prepare_output_dir,
    sample_isoflop_data,
    log_range_to_label,
)

# Drift rate matching the "Compounding Errors" article section
DRIFT_RATE = np.log10(3)

# Asymmetric surface with drift bias
SIM_CONFIG = SimulationConfig(
    name="asymmetric_drift",
    loss=ASYMMETRIC_CONFIG.loss,
    drift_rate=DRIFT_RATE,
    center_scale=1.0,
)

# Points per isoflop curve to sweep
N_POINTS_RANGE = np.arange(3, 51)

# Representative log_range values (narrow, medium, wide)
EXPERIMENT_LOG_RANGES = [0.3, 1.0, 2.0]
EXPERIMENT_LOG_RANGE_NAMES = ["narrow", "medium", "wide"]

# Method display configuration
METHOD_STYLES = {
    "Approach 2": {"color": "#d62728", "marker": "s"},
    "Approach 3": {"color": "#ff7f0e", "marker": "^"},
    "VPNLS": {"color": "#1f77b4", "marker": "o"},
}


def compute_ab_errors(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_range: float,
    n_points_range: np.ndarray,
) -> dict[str, dict]:
    """Compute a/b exponent errors for all methods across n_points values.

    For each n_points value, fits Approach 2, Approach 3, and VPNLS to data
    sampled from the configured surface, then computes the relative error
    in the recovered scaling exponents a and b.

    Args:
        sim_config: Simulation configuration (surface + bias)
        compute_budgets: Compute budgets for isoflop sampling
        log_range: Sampling range in log10 space
        n_points_range: Array of n_points values to sweep

    Returns:
        Dict mapping method name -> dict with "a_errors", "b_errors", "failed"
    """
    loss = sim_config.loss
    true_a = loss.a
    true_b = loss.b

    methods = list(METHOD_STYLES.keys())
    results = {
        m: {
            "a_errors": np.full(len(n_points_range), np.nan),
            "b_errors": np.full(len(n_points_range), np.nan),
            "failed": np.ones(len(n_points_range), dtype=bool),
        }
        for m in methods
    }

    for i, n_points in enumerate(n_points_range):
        n_pts = int(n_points)

        # Approach 2: does its own per-budget isoflop sampling internally
        try:
            a2_result = fit_approach2(
                compute_budgets=compute_budgets,
                surface=sim_config.loss,
                drift_rate=sim_config.drift_rate,
                center_scale=sim_config.center_scale,
                n_points=n_pts,
                log_range=log_range,
            )
            results["Approach 2"]["a_errors"][i] = (a2_result.a - true_a) / true_a
            results["Approach 2"]["b_errors"][i] = (a2_result.b - true_b) / true_b
            results["Approach 2"]["failed"][i] = False
        except FitError:
            pass

        # Shared pooled data for surface-fitting methods
        N, D, L = sample_isoflop_data(sim_config, compute_budgets, log_range, n_pts)

        # Approach 3: 5D direct optimization
        try:
            a3_result = fit_approach3(N, D, L)
            fitted_a3 = a3_result.to_loss_surface()
            results["Approach 3"]["a_errors"][i] = (fitted_a3.a - true_a) / true_a
            results["Approach 3"]["b_errors"][i] = (fitted_a3.b - true_b) / true_b
            results["Approach 3"]["failed"][i] = False
        except FitError:
            pass

        # VPNLS: variable projection + Nelder-Mead
        try:
            vpnls_result = fit_surface(N, D, L)
            fitted_vpnls = vpnls_result.to_loss_surface()
            results["VPNLS"]["a_errors"][i] = (fitted_vpnls.a - true_a) / true_a
            results["VPNLS"]["b_errors"][i] = (fitted_vpnls.b - true_b) / true_b
            results["VPNLS"]["failed"][i] = False
        except FitError:
            pass

    return results


def create_data_efficiency_figure(
    all_results: dict[str, dict[str, dict]],
    n_points_range: np.ndarray,
    log_ranges: list[float],
    log_range_names: list[str],
) -> plt.Figure:
    """Create the data efficiency comparison figure.

    Layout: 2 rows (a, b exponents) x len(log_ranges) columns.
    Each subplot shows absolute relative error vs n_points for all methods.

    Args:
        all_results: Dict mapping log_range_name -> method_name -> error dict
        n_points_range: Array of n_points values (x-axis)
        log_ranges: Log range values (for subplot titles)
        log_range_names: Names for each log_range column

    Returns:
        matplotlib Figure
    """
    n_cols = len(log_ranges)
    exponent_labels = ["a", "b"]
    error_keys = ["a_errors", "b_errors"]

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(5 * n_cols, 7),
        gridspec_kw={"hspace": 0.3, "wspace": 0.25},
    )

    loss = SIM_CONFIG.loss

    for col, (range_name, log_range) in enumerate(zip(log_range_names, log_ranges)):
        range_label = log_range_to_label(log_range)
        method_results = all_results[range_name]

        for row, (exp_label, err_key) in enumerate(zip(exponent_labels, error_keys)):
            ax = axes[row, col]

            for method_name, style in METHOD_STYLES.items():
                errors = np.abs(method_results[method_name][err_key]) * 100
                valid = ~np.isnan(errors)
                ax.plot(
                    n_points_range[valid],
                    errors[valid],
                    marker=style["marker"],
                    color=style["color"],
                    markersize=3,
                    linewidth=1.2,
                    label=method_name,
                )

            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(n_points_range[0] - 1, n_points_range[-1] + 1)

            true_val = loss.a if exp_label == "a" else loss.b
            ax.set_title(
                f"Exponent {exp_label} (true={true_val:.3f})\n"
                f"Grid width: {range_label}",
                fontsize=9,
            )

            if row == 1:
                ax.set_xlabel("Points per isoflop curve", fontsize=9)

            if col == 0:
                ax.set_ylabel("Absolute relative error (%)", fontsize=9)

            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=8, loc="best")

            ax.tick_params(labelsize=8)

    n_budgets = len(COMPUTE_BUDGETS)
    c_min = COMPUTE_BUDGETS.min()
    c_max = COMPUTE_BUDGETS.max()
    fig.suptitle(
        "Data Efficiency: Scaling Exponent Recovery\n"
        f"Asymmetric surface "
        f"($\\alpha$={loss.alpha:.3f}, $\\beta$={loss.beta:.3f}), "
        f"drift rate = log$_{{10}}$(3) $\\approx$ {DRIFT_RATE:.3f}, "
        f"{n_budgets} isoflop curves "
        f"({c_min:.0e}\u2013{c_max:.0e} FLOPs)",
        fontsize=11,
        y=1.02,
    )

    return fig


def main():
    """Run Experiment 7: data efficiency comparison."""
    print("=" * 70)
    print("Experiment 7: Data Efficiency Comparison")
    print("=" * 70)

    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp7")

    loss = SIM_CONFIG.loss
    print(f"\nSurface: Asymmetric")
    print(f"  alpha={loss.alpha:.3f}, beta={loss.beta:.3f}")
    print(f"  True a={loss.a:.4f}, True b={loss.b:.4f}")
    print(f"  Drift rate: log10(3) = {DRIFT_RATE:.4f}")
    print(f"  Compute budgets: {COMPUTE_BUDGETS}")
    print(f"  n_points range: {N_POINTS_RANGE[0]} to {N_POINTS_RANGE[-1]}")

    all_results: dict[str, dict[str, dict]] = {}

    for range_name, log_range in zip(EXPERIMENT_LOG_RANGE_NAMES, EXPERIMENT_LOG_RANGES):
        range_label = log_range_to_label(log_range)
        print(f"\n{'#' * 70}")
        print(f"Log range: {range_name} ({range_label})")
        print(f"{'#' * 70}")

        results = compute_ab_errors(
            sim_config=SIM_CONFIG,
            compute_budgets=COMPUTE_BUDGETS,
            log_range=log_range,
            n_points_range=N_POINTS_RANGE,
        )
        all_results[range_name] = results

        # Print summary
        for method_name in METHOD_STYLES:
            n_failed = results[method_name]["failed"].sum()
            a_err = results[method_name]["a_errors"]
            b_err = results[method_name]["b_errors"]
            valid_a = a_err[~np.isnan(a_err)]
            valid_b = b_err[~np.isnan(b_err)]

            a_summary = (
                f"max |err|={np.max(np.abs(valid_a))*100:.2f}%"
                if len(valid_a) > 0
                else "all failed"
            )
            b_summary = (
                f"max |err|={np.max(np.abs(valid_b))*100:.2f}%"
                if len(valid_b) > 0
                else "all failed"
            )
            print(
                f"  {method_name:<15} "
                f"a: {a_summary}, b: {b_summary}, "
                f"failures: {n_failed}/{len(N_POINTS_RANGE)}"
            )

    print(f"\n{'─' * 70}")
    print("Generating data efficiency figure...")
    fig = create_data_efficiency_figure(
        all_results=all_results,
        n_points_range=N_POINTS_RANGE,
        log_ranges=EXPERIMENT_LOG_RANGES,
        log_range_names=EXPERIMENT_LOG_RANGE_NAMES,
    )
    fig_path = output_dir / "data_efficiency.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fig_path}")
    plt.close(fig)

    print("\nExperiment 7 complete.")
    return all_results


if __name__ == "__main__":
    main()
