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
    compute_center_offset,
    isoflop_sample,
    approach2_recover,
)
from scaling_law_analysis.experiments.common import SimulationConfig, DEFAULT_CONFIG


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
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int,
) -> dict:
    """Run Approach 2 recovery across multiple grid step sizes.

    Args:
        sim_config: Simulation configuration
        compute_budgets: Compute budgets to use for fitting
        log_ranges: Array of log_range values (grid step sizes) to test
        n_points: Number of points per IsoFLOP curve

    Returns:
        Dictionary with results for a and b exponents
    """
    loss = sim_config.loss
    a_recovered = []
    b_recovered = []
    results = []

    for log_range in log_ranges:
        result = approach2_recover(
            compute_budgets=compute_budgets,
            surface=loss,
            drift_rate=sim_config.drift_rate,
            center_scale=sim_config.center_scale,
            n_points=n_points,
            log_range=log_range,
        )
        a_recovered.append(result.a)
        b_recovered.append(result.b)
        results.append(result)

    a_recovered = np.array(a_recovered)
    b_recovered = np.array(b_recovered)

    return {
        "config": sim_config,
        "log_ranges": log_ranges,
        "true_a": loss.a,
        "true_b": loss.b,
        "a_recovered": a_recovered,
        "b_recovered": b_recovered,
        "a_error": (a_recovered - loss.a) / loss.a,
        "b_error": (b_recovered - loss.b) / loss.b,
        "results": results,
    }


def plot_isoflop_fits(
    ax: plt.Axes,
    result,
    compute_budgets: np.ndarray,
    log_range: float,
    sim_config: SimulationConfig,
    title: str,
    show_ylabel: bool = True,
):
    """Plot IsoFLOP curves with parabola fits for a single grid step size.

    Args:
        ax: Matplotlib axes to plot on
        result: Approach2Result from recovery
        compute_budgets: Compute budgets used
        log_range: Grid step size used
        sim_config: Simulation configuration
        title: Plot title
        show_ylabel: Whether to show y-axis label
    """
    loss = sim_config.loss
    colors = plt.cm.viridis(np.linspace(0, 1, len(compute_budgets)))

    for i, (C, fit) in enumerate(zip(compute_budgets, result.parabola_fits_N)):
        center_offset = compute_center_offset(
            C=C,
            compute_budgets=compute_budgets,
            drift_rate=sim_config.drift_rate,
            center_scale=sim_config.center_scale,
        )

        # Get sampled data
        N, D, L = isoflop_sample(
            C=C,
            n_points=20,
            log_range=log_range,
            center_offset=center_offset,
            surface=loss,
        )
        log_N = np.log10(N)

        # Plot data points
        ax.scatter(log_N, L, c=[colors[i]], s=20, alpha=0.7, zorder=3)

        # Plot parabola fit
        log_N_fine = np.linspace(log_N.min(), log_N.max(), 100)
        L_fit = np.polyval(fit.coeffs, log_N_fine)
        ax.plot(log_N_fine, L_fit, c=colors[i], lw=1.5, alpha=0.8)

        # Mark true optimum
        N_true, D_true = loss.N_opt(C), loss.D_opt(C)
        L_true = loss.loss(N_true, D_true)
        ax.scatter(
            [np.log10(N_true)],
            [L_true],
            c="red",
            marker="x",
            s=60,
            zorder=4,
            linewidths=2,
        )

        # Mark inferred optimum
        ax.scatter(
            [fit.log_x_opt],
            [fit.L_min],
            c="blue",
            marker="+",
            s=60,
            zorder=4,
            linewidths=2,
        )

    ax.set_xlabel("log₁₀(N)")
    if show_ylabel:
        ax.set_ylabel("Loss")
    ax.set_title(title)

    # Custom legend
    ax.scatter([], [], c="red", marker="x", s=60, label="True optimum")
    ax.scatter([], [], c="blue", marker="+", s=60, label="Inferred optimum")
    ax.legend(loc="upper right", fontsize=8)


def plot_power_law_fits(
    ax: plt.Axes,
    result,
    compute_budgets: np.ndarray,
    sim_config: SimulationConfig,
    title: str = "",
    show_ylabel: bool = True,
):
    """Plot inferred N* and D* vs compute budget with power-law fits.

    Shows the parabola minima (N* and D*) and the power-law fits used to
    recover scaling exponents a and b.

    Args:
        ax: Matplotlib axes to plot on (will create twin for D*)
        result: Approach2Result with parabola fits and power-law coefficients
        compute_budgets: Compute budgets used
        sim_config: Simulation configuration
        title: Plot title
        show_ylabel: Whether to show y-axis labels
    """
    loss = sim_config.loss
    log_C = np.log10(compute_budgets)

    # Get inferred optima from parabola fits
    N_opts = result.N_opts
    D_opts = result.D_opts

    # Get true optima
    true_N_opts = np.array([loss.N_opt(C) for C in compute_budgets])
    true_D_opts = np.array([loss.D_opt(C) for C in compute_budgets])

    # Colors: blue for N*, green for D*
    color_N = "C0"  # blue
    color_D = "C2"  # green

    # Create twin axis for D*
    ax2 = ax.twinx()

    # Plot N* (left axis)
    ax.scatter(log_C, np.log10(N_opts), c=color_N, s=40, zorder=3, label="Inferred N*")
    ax.scatter(log_C, np.log10(true_N_opts), c=color_N, marker="x", s=60, zorder=3, label="True N*")

    # Plot N* fit line (solid for inferred)
    log_C_fine = np.linspace(log_C.min(), log_C.max(), 100)
    log_N_fit = result.a * log_C_fine + result.a_intercept
    ax.plot(log_C_fine, log_N_fit, c=color_N, lw=1.5, alpha=0.8)

    # Plot true N* line (dashed)
    ax.plot(log_C, np.log10(true_N_opts), c=color_N, lw=1.5, alpha=0.8, linestyle="--")

    # Plot D* (right axis)
    ax2.scatter(log_C, np.log10(D_opts), c=color_D, s=40, zorder=3, marker="s", label="Inferred D*")
    ax2.scatter(log_C, np.log10(true_D_opts), c=color_D, marker="x", s=60, zorder=3, label="True D*")

    # Plot D* fit line (solid for inferred)
    log_D_fit = result.b * log_C_fine + result.b_intercept
    ax2.plot(log_C_fine, log_D_fit, c=color_D, lw=1.5, alpha=0.8)

    # Plot true D* line (dashed)
    ax2.plot(log_C, np.log10(true_D_opts), c=color_D, lw=1.5, alpha=0.8, linestyle="--")

    # Labels and formatting
    ax.set_xlabel("log₁₀(C)")
    if show_ylabel:
        ax.set_ylabel("log₁₀(N*)", color=color_N, labelpad=2)
        ax2.set_ylabel("log₁₀(D*)", color=color_D, labelpad=2)
    ax.tick_params(axis="y", labelcolor=color_N, pad=2)
    ax2.tick_params(axis="y", labelcolor=color_D, pad=2)

    # Add exponent annotations
    ax.text(
        0.05, 0.95,
        f"a={result.a:.3f} (true={loss.a:.3f})",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        color=color_N,
    )
    ax.text(
        0.05, 0.85,
        f"b={result.b:.3f} (true={loss.b:.3f})",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        color=color_D,
    )

    ax.set_title(title)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=7)


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
    ax.plot(log_ranges, b_error, "s-", label="b error (D* exponent)", color="C2", markersize=4)

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

    Row 1: IsoFLOP curves with parabola fits for 3 grid sizes
    Row 2: Power-law fits (N* and D* vs compute) for same 3 grid sizes
    Row 3: Error plot spanning all columns

    Args:
        experiment_results: Results from run_experiment
        display_log_ranges: 3 log_range values to display in top rows
        compute_budgets: Compute budgets used

    Returns:
        Matplotlib figure
    """
    sim_config = experiment_results["config"]
    loss = sim_config.loss

    fig = plt.figure(figsize=(14, 12))

    # Create grid: 3 columns, 3 rows
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.4)

    size_labels = ["Small", "Medium", "Large"]

    # Collect results for each display log_range
    results_for_display = []
    actual_log_ranges = []
    for log_range in display_log_ranges:
        idx = np.argmin(np.abs(experiment_results["log_ranges"] - log_range))
        results_for_display.append(experiment_results["results"][idx])
        actual_log_ranges.append(experiment_results["log_ranges"][idx])

    # Row 1: IsoFLOP fits for 3 grid sizes
    for i, (result, actual_log_range) in enumerate(zip(results_for_display, actual_log_ranges)):
        ax = fig.add_subplot(gs[0, i])
        plot_isoflop_fits(
            ax=ax,
            result=result,
            compute_budgets=compute_budgets,
            log_range=actual_log_range,
            sim_config=sim_config,
            title=f"{size_labels[i]} grid ({log_range_to_label(actual_log_range)})",
            show_ylabel=(i == 0),
        )

    # Row 2: Power-law fits for same 3 grid sizes
    for i, (result, actual_log_range) in enumerate(zip(results_for_display, actual_log_ranges)):
        ax = fig.add_subplot(gs[1, i])
        plot_power_law_fits(
            ax=ax,
            result=result,
            compute_budgets=compute_budgets,
            sim_config=sim_config,
            title=f"N*, D* vs C ({log_range_to_label(actual_log_range)})",
            show_ylabel=(i == 0),
        )

    # Row 3: Error plot (spans all columns)
    ax_error = fig.add_subplot(gs[2, :])
    plot_error_vs_grid_size(ax_error, experiment_results)

    bias_parts = []
    if sim_config.drift_rate != 0.0:
        bias_parts.append(f"drift_rate={sim_config.drift_rate}")
    if sim_config.center_scale != 1.0:
        bias_parts.append(f"center_scale={sim_config.center_scale}")
    bias_str = f", {', '.join(bias_parts)}" if bias_parts else ""
    fig.suptitle(
        f"Experiment 1: Approach 2 Accuracy vs Grid Resolution{bias_str}\n"
        f"True: α={loss.alpha}, β={loss.beta} → a=β/(α+β)={loss.a:.4f}, b=α/(α+β)={loss.b:.4f}",
        fontsize=11,
        y=0.98,
    )

    return fig


def main(
    sim_config: SimulationConfig | None = None,
    output_path: Path | None = None,
):
    """Run Experiment 1 and generate output figure.

    Args:
        sim_config: Simulation configuration. Defaults to DEFAULT_CONFIG.
        output_path: Path to save the figure. Defaults to results/exp1_empirical_error.png.
    """
    if sim_config is None:
        sim_config = DEFAULT_CONFIG

    loss = sim_config.loss

    print("=" * 70)
    print(f"Experiment 1: Empirical Error Analysis ({sim_config.name})")
    print("=" * 70)

    # Experiment parameters
    compute_budgets = np.array([1e17, 1e18, 1e19, 1e20, 1e21])
    log_ranges = np.linspace(0.05, 2.0, 20)  # Grid step sizes to test
    n_points = 15  # Points per IsoFLOP curve

    print(f"\nCompute budgets: {compute_budgets}")
    print(f"Grid step sizes (log_range): {log_ranges[0]:.2f} to {log_ranges[-1]:.2f}")
    print(f"Points per IsoFLOP curve: {n_points}")
    print(f"Drift rate: {sim_config.drift_rate}")
    print(f"Center scale: {sim_config.center_scale}")
    print(f"Alpha: {loss.alpha}, Beta: {loss.beta}")
    print(f"A: {loss.A}, B: {loss.B}, E: {loss.E}")

    # Run experiment
    print("\nRunning parameter recovery...")
    results = run_experiment(
        sim_config=sim_config,
        compute_budgets=compute_budgets,
        log_ranges=log_ranges,
        n_points=n_points,
    )

    true_a = results["true_a"]
    true_b = results["true_b"]

    # Print header explaining the exponents
    print("\n" + "-" * 70)
    print("Approach 2 recovers power law exponents:")
    print("  N* ∝ C^a  where a = β/(α+β)")
    print("  D* ∝ C^b  where b = α/(α+β)")
    print(f"\nGround truth: α={loss.alpha}, β={loss.beta}")
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

    # Generate figure
    print("\nGenerating figure...")
    display_log_ranges = [log_ranges[0], log_ranges[len(log_ranges) // 2], log_ranges[-1]]
    fig = create_figure(results, display_log_ranges, compute_budgets)

    # Save figure
    if output_path is None:
        output_path = config.RESULTS_DIR / "exp1_empirical_error.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    return results


if __name__ == "__main__":
    main()
