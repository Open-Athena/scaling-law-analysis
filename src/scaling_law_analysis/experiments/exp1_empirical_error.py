"""Experiment 1: Empirical Error Analysis of Chinchilla Approach 2.

This experiment demonstrates that the accuracy of Approach 2 parameter recovery
depends on the grid step size (log_range) used for IsoFLOP sampling.

The hypothesis is that systematic error arises from the second-order Taylor
expansion underlying the validity of parabolic fits.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    chinchilla_loss,
    compute_drift_offset,
    isoflop_sample,
    optimal_allocation,
    approach2_recover,
    CHINCHILLA_PARAMS,
)


def log_range_to_label(log_range: float) -> str:
    """Convert log_range to human-readable N sampling range.
    
    log_range=1.0 means N spans 10^-1 to 10^1 around optimal = 0.1x to 10x.
    """
    factor = 10 ** log_range
    if factor >= 10:
        return f"±{factor:.0f}x"
    else:
        return f"±{factor:.1f}x"

def run_experiment(
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int = 10,
    use_log_loss: bool = False,
    drift_rate: float = 0.0,
) -> dict:
    """Run Approach 2 recovery across multiple grid step sizes.

    Args:
        compute_budgets: Compute budgets to use for fitting
        log_ranges: Array of log_range values (grid step sizes) to test
        n_points: Number of points per IsoFLOP curve
        use_log_loss: If True, fit parabola to log(L) vs log(N).
                      If False (default), fit parabola to L vs log(N).
        drift_rate: Rate at which sampling center drifts from optimal.

    Returns:
        Dictionary with results for a and b exponents
    """
    # True values derived from alpha and beta
    true_alpha = CHINCHILLA_PARAMS["alpha"]
    true_beta = CHINCHILLA_PARAMS["beta"]
    true_a = true_beta / (true_alpha + true_beta)  # N* exponent
    true_b = true_alpha / (true_alpha + true_beta)  # D* exponent

    a_recovered = []
    b_recovered = []
    results = []

    for log_range in log_ranges:
        result = approach2_recover(
            compute_budgets=compute_budgets,
            n_points=n_points,
            log_range=log_range,
            drift_rate=drift_rate,
            use_log_loss=use_log_loss,
        )
        a_recovered.append(result.a)
        b_recovered.append(result.b)
        results.append(result)

    a_recovered = np.array(a_recovered)
    b_recovered = np.array(b_recovered)

    return {
        "log_ranges": log_ranges,
        "true_a": true_a,
        "true_b": true_b,
        "a_recovered": a_recovered,
        "b_recovered": b_recovered,
        "a_error": (a_recovered - true_a) / true_a,
        "b_error": (b_recovered - true_b) / true_b,
        "results": results,
        "use_log_loss": use_log_loss,
        "drift_rate": drift_rate,
    }


def plot_isoflop_fits(
    ax: plt.Axes,
    result,
    compute_budgets: np.ndarray,
    log_range: float,
    title: str,
    use_log_loss: bool = False,
    drift_rate: float = 0.0,
):
    """Plot IsoFLOP curves with parabola fits for a single grid step size.

    Args:
        ax: Matplotlib axes to plot on
        result: Approach2Result from recovery
        compute_budgets: Compute budgets used
        log_range: Grid step size used
        title: Plot title
        use_log_loss: Whether log loss was used for fitting
        drift_rate: Rate at which sampling center drifts from optimal
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(compute_budgets)))

    for i, (C, fit) in enumerate(zip(compute_budgets, result.parabola_fits_N)):
        center_offset = compute_drift_offset(C, compute_budgets, drift_rate)

        # Get sampled data
        N, D, L = isoflop_sample(C, 20, log_range, center_offset)
        log_N = np.log10(N)

        # Transform y-axis based on fitting mode
        if use_log_loss:
            y = np.log10(L)
            y_true_opt = lambda L_val: np.log10(L_val)
            y_fit_opt = fit.log_L_min
        else:
            y = L
            y_true_opt = lambda L_val: L_val
            y_fit_opt = fit.L_min

        # Plot data points
        ax.scatter(log_N, y, c=[colors[i]], s=20, alpha=0.7, zorder=3)

        # Plot parabola fit
        log_N_fine = np.linspace(log_N.min(), log_N.max(), 100)
        y_fit = np.polyval(fit.coeffs, log_N_fine)
        ax.plot(log_N_fine, y_fit, c=colors[i], lw=1.5, alpha=0.8)

        # Mark true optimum
        N_true, _ = optimal_allocation(C)
        L_true = chinchilla_loss(N_true, C / (6 * N_true))
        ax.scatter(
            [np.log10(N_true)],
            [y_true_opt(L_true)],
            c="red",
            marker="x",
            s=60,
            zorder=4,
            linewidths=2,
        )

        # Mark inferred optimum
        ax.scatter(
            [fit.log_x_opt],
            [y_fit_opt],
            c="blue",
            marker="+",
            s=60,
            zorder=4,
            linewidths=2,
        )

    ax.set_xlabel("log₁₀(N)")
    ax.set_ylabel("log₁₀(L)" if use_log_loss else "L")
    ax.set_title(title)

    # Custom legend
    ax.scatter([], [], c="red", marker="x", s=60, label="True optimum")
    ax.scatter([], [], c="blue", marker="+", s=60, label="Inferred optimum")
    ax.legend(loc="upper right", fontsize=8)


def plot_error_vs_grid_size(
    ax: plt.Axes,
    experiment_results: dict,
):
    """Plot exponent recovery error vs grid step size.

    Args:
        ax: Matplotlib axes to plot on
        experiment_results: Results from run_experiment
    """
    log_ranges = experiment_results["log_ranges"]
    a_error = experiment_results["a_error"] * 100  # Convert to %
    b_error = experiment_results["b_error"] * 100

    ax.plot(log_ranges, a_error, "o-", label="a error (N* exponent)", color="C0", markersize=4)
    ax.plot(log_ranges, b_error, "s-", label="b error (D* exponent)", color="C1", markersize=4)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    
    # Primary x-axis: log_range values
    ax.set_xlabel("Sampling range around optimal N")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Exponent Recovery Error vs Sampling Range")
    
    # Add secondary x-axis labels showing intuitive range
    tick_positions = [0.05, 0.5, 1.0, 1.5, 2.0]
    tick_labels = [log_range_to_label(lr) for lr in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_figure(
    experiment_results: dict,
    display_log_ranges: list[float],
    compute_budgets: np.ndarray,
) -> plt.Figure:
    """Create the main experiment figure.

    Top row: IsoFLOP curves with fits for 3 grid sizes
    Bottom row: Error plot

    Args:
        experiment_results: Results from run_experiment
        display_log_ranges: 3 log_range values to display in top row
        compute_budgets: Compute budgets used

    Returns:
        Matplotlib figure
    """
    use_log_loss = experiment_results.get("use_log_loss", False)
    drift_rate = experiment_results.get("drift_rate", 0.0)
    fit_mode = "log(L)" if use_log_loss else "L"
    
    fig = plt.figure(figsize=(14, 8))

    # Create grid: 3 columns for top row, 1 spanning column for bottom
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.3, wspace=0.3)

    # Top row: IsoFLOP fits for 3 grid sizes
    size_labels = ["Small", "Medium", "Large"]
    for i, log_range in enumerate(display_log_ranges):
        # Find the closest result
        idx = np.argmin(np.abs(experiment_results["log_ranges"] - log_range))
        result = experiment_results["results"][idx]
        actual_log_range = experiment_results["log_ranges"][idx]

        ax = fig.add_subplot(gs[0, i])
        plot_isoflop_fits(
            ax,
            result,
            compute_budgets,
            actual_log_range,
            f"{size_labels[i]} grid ({log_range_to_label(actual_log_range)})",
            use_log_loss=use_log_loss,
            drift_rate=drift_rate,
        )

    # Bottom row: Error plot (spans all columns)
    ax_error = fig.add_subplot(gs[1, :])
    plot_error_vs_grid_size(ax_error, experiment_results)

    # Add true values annotation
    true_alpha = CHINCHILLA_PARAMS["alpha"]
    true_beta = CHINCHILLA_PARAMS["beta"]
    true_a = experiment_results["true_a"]
    true_b = experiment_results["true_b"]
    drift_str = f", drift_rate={drift_rate}" if drift_rate != 0.0 else ""
    fig.suptitle(
        f"Experiment 1: Approach 2 Accuracy vs Grid Resolution (fit on {fit_mode}{drift_str})\n"
        f"True: α={true_alpha}, β={true_beta} → a=β/(α+β)={true_a:.4f}, b=α/(α+β)={true_b:.4f}",
        fontsize=11,
        y=0.98,
    )

    return fig


def main():
    """Run Experiment 1 and generate output figure."""
    print("=" * 70)
    print("Experiment 1: Empirical Error Analysis")
    print("=" * 70)

    # Configuration
    compute_budgets = np.array([1e17, 1e18, 1e19, 1e20, 1e21])
    log_ranges = np.linspace(0.05, 2.0, 20)  # Grid step sizes to test
    n_points = 15  # Points per IsoFLOP curve
    use_log_loss = False  # If True, fit log(L) vs log(N); if False, fit L vs log(N)
    drift_rate = 0.1  # Drift rate for sampling center offset

    fit_mode = "log(L) vs log(N)" if use_log_loss else "L vs log(N)"
    print(f"\nFitting mode: {fit_mode}")
    print(f"Compute budgets: {compute_budgets}")
    print(f"Grid step sizes (log_range): {log_ranges[0]:.2f} to {log_ranges[-1]:.2f}")
    print(f"Points per IsoFLOP curve: {n_points}")
    print(f"Drift rate: {drift_rate}")

    # Run experiment
    print("\nRunning parameter recovery...")
    results = run_experiment(compute_budgets, log_ranges, n_points, use_log_loss=use_log_loss, drift_rate=drift_rate)

    true_alpha = CHINCHILLA_PARAMS["alpha"]
    true_beta = CHINCHILLA_PARAMS["beta"]
    true_a = results["true_a"]
    true_b = results["true_b"]

    # Print header explaining the exponents
    print("\n" + "-" * 70)
    print("Approach 2 recovers power law exponents:")
    print("  N* ∝ C^a  where a = β/(α+β)")
    print("  D* ∝ C^b  where b = α/(α+β)")
    print(f"\nGround truth: α={true_alpha}, β={true_beta}")
    print(f"  → True a = {true_a:.4f} (N* exponent)")
    print(f"  → True b = {true_b:.4f} (D* exponent)")
    print("-" * 70)

    # Print results table
    print(f"\n{'log_range':>10} {'a_rec':>10} {'b_rec':>10} {'a_err%':>10} {'b_err%':>10}")
    print("-" * 70)

    for i, lr in enumerate(log_ranges):
        a_rec = results["a_recovered"][i]
        b_rec = results["b_recovered"][i]
        a_err = results["a_error"][i] * 100
        b_err = results["b_error"][i] * 100
        print(f"{lr:>10.2f} {a_rec:>10.4f} {b_rec:>10.4f} {a_err:>+10.2f} {b_err:>+10.2f}")

    print("-" * 70)
    print(f"{'True':>10} {true_a:>10.4f} {true_b:>10.4f}")

    # Create output directory
    output_dir = config.RESULTS_DIR

    # Generate figure
    print("\nGenerating figure...")
    display_log_ranges = [log_ranges[0], log_ranges[len(log_ranges) // 2], log_ranges[-1]]
    fig = create_figure(results, display_log_ranges, compute_budgets)

    # Save figure
    output_path = output_dir / "exp1_empirical_error.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to: {output_path}")

    plt.close(fig)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    smallest_error_a = results["a_error"][0] * 100
    smallest_error_b = results["b_error"][0] * 100
    largest_error_a = results["a_error"][-1] * 100
    largest_error_b = results["b_error"][-1] * 100

    print(f"At smallest grid step (log_range={log_ranges[0]:.2f}):")
    print(f"  a error: {smallest_error_a:+.4f}%")
    print(f"  b error: {smallest_error_b:+.4f}%")
    print(f"\nAt largest grid step (log_range={log_ranges[-1]:.2f}):")
    print(f"  a error: {largest_error_a:+.4f}%")
    print(f"  b error: {largest_error_b:+.4f}%")


if __name__ == "__main__":
    main()
