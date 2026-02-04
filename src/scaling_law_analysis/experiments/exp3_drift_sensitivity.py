"""Experiment 3: Sampling Drift Sensitivity Analysis.

This experiment investigates how the accuracy of Chinchilla Approach 2 varies
with systematic biases in the sampling center (drift and scale).

Hypothesis: Both linear drift and constant center scaling degrade exponent recovery.
"""

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.experiments.common import (
    EXP3_CONFIGS,
    EXP3_LOSS_SURFACE,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    LOG_RANGES,
    N_POINTS,
    TICK_POSITIONS,
)
from scaling_law_analysis.experiments.exp1_empirical_error import (
    run_experiment,
    log_range_to_label,
    create_figure as create_exp1_figure,
)


def _configure_ax(ax: plt.Axes, title: str, show_legend: bool = True):
    """Apply common axis configuration."""
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sampling range")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(title)
    if show_legend:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    tick_positions = TICK_POSITIONS
    tick_labels = [log_range_to_label(lr) for lr in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")


def create_figure(all_results: list[dict]) -> plt.Figure:
    """Create combined error analysis figure for all configurations.

    Shows four rows of error analysis:
    - Row 1: Exponent error (a, b)
    - Row 2: Intercept error (a₀, b₀)
    - Row 3: Optimum error relative (N*, D*) - per-budget dots for baseline
    - Row 4: Optimum error absolute (N*, D*) - per-budget dots for baseline

    Args:
        all_results: List of experiment results, one per configuration

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_results)))

    for i, results in enumerate(all_results):
        sim_config = results["config"]
        log_ranges = results["log_ranges"]
        label = sim_config.name
        color = colors[i]

        # Row 1: Exponent errors
        axes[0, 0].plot(log_ranges, results["a_error"] * 100, "o-", color=color, markersize=3, label=label)
        axes[0, 1].plot(log_ranges, results["b_error"] * 100, "o-", color=color, markersize=3, label=label)

        # Row 2: Intercept errors
        axes[1, 0].plot(log_ranges, results["a_intercept_error"] * 100, "o-", color=color, markersize=3, label=label)
        axes[1, 1].plot(log_ranges, results["b_intercept_error"] * 100, "o-", color=color, markersize=3, label=label)

    # Rows 3 & 4: Optimum errors - only scale_2.0, per-budget dots like exp1
    baseline_results = next(r for r in all_results if r["config"].name == "scale_2.0")
    log_ranges = baseline_results["log_ranges"]
    compute_budgets = baseline_results["compute_budgets"]
    loss = baseline_results["config"].loss

    # Compute true optima for absolute error calculation
    true_N_opts = np.array([loss.N_opt(C) for C in compute_budgets])
    true_D_opts = np.array([loss.D_opt(C) for C in compute_budgets])

    N_opt_rel_errors = baseline_results["N_opt_errors"] * 100  # shape: (n_log_ranges, n_budgets)
    D_opt_rel_errors = baseline_results["D_opt_errors"] * 100
    N_opt_signed_errors = baseline_results["N_opt_errors"] * true_N_opts  # signed error (not absolute)
    D_opt_signed_errors = baseline_results["D_opt_errors"] * true_D_opts

    n_budgets = len(compute_budgets)
    alphas = np.linspace(0.2, 1.0, n_budgets)
    sizes = np.linspace(10, 36, n_budgets)  # Size also varies with compute budget

    for j, C in enumerate(compute_budgets):
        # Row 3: Relative errors
        axes[2, 0].scatter(log_ranges, N_opt_rel_errors[:, j], c="C0", alpha=alphas[j], s=sizes[j], marker="o")
        axes[2, 1].scatter(log_ranges, D_opt_rel_errors[:, j], c="C2", alpha=alphas[j], s=sizes[j], marker="s")
        # Row 4: Signed errors (N_opt - true_N_opt)
        axes[3, 0].scatter(log_ranges, N_opt_signed_errors[:, j], c="C0", alpha=alphas[j], s=sizes[j], marker="o")
        axes[3, 1].scatter(log_ranges, D_opt_signed_errors[:, j], c="C2", alpha=alphas[j], s=sizes[j], marker="s")

    # Configure axes
    _configure_ax(axes[0, 0], "Exponent error: a (N* exponent)")
    _configure_ax(axes[0, 1], "Exponent error: b (D* exponent)")
    _configure_ax(axes[1, 0], "Intercept error: a₀ (N* intercept)")
    _configure_ax(axes[1, 1], "Intercept error: b₀ (D* intercept)")
    _configure_ax(axes[2, 0], "Optimum relative error: N* (scale_2.0, light→dark = C↑)", show_legend=False)
    _configure_ax(axes[2, 1], "Optimum relative error: D* (scale_2.0, light→dark = C↑)", show_legend=False)
    _configure_ax(axes[3, 0], "Optimum signed error: N* (scale_2.0, light→dark = C↑)", show_legend=False)
    _configure_ax(axes[3, 1], "Optimum signed error: D* (scale_2.0, light→dark = C↑)", show_legend=False)
    axes[3, 0].set_ylabel("Signed error")
    axes[3, 1].set_ylabel("Signed error")

    loss = EXP3_LOSS_SURFACE
    fig.suptitle(
        "Experiment 3: Error Analysis vs Sampling Bias\n"
        f"Loss surface: α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f} (symmetric)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()

    return fig


def main():
    """Run Experiment 3: analyze sensitivity to sampling drift and center scale."""
    print("=" * 70)
    print("Experiment 3: Sampling Drift Sensitivity Analysis")
    print("=" * 70)

    compute_budgets = COMPUTE_BUDGETS
    log_ranges = LOG_RANGES
    n_points = N_POINTS

    # Prepare output directory
    output_dir = prepare_output_dir(config.RESULTS_DIR / "exp3")

    all_results = []

    for sim_config in EXP3_CONFIGS:
        loss = sim_config.loss
        print(f"\n{'─' * 70}")
        print(f"Configuration: {sim_config.name}")
        print(f"  drift_rate={sim_config.drift_rate}, center_scale={sim_config.center_scale}")
        print(f"  α={loss.alpha}, β={loss.beta}")
        print(f"{'─' * 70}")

        results = run_experiment(
            sim_config=sim_config,
            compute_budgets=compute_budgets,
            log_ranges=log_ranges,
            n_points=n_points,
        )
        all_results.append(results)

        # Generate individual figure (same format as Experiment 1)
        display_log_ranges = [log_ranges[0], log_ranges[len(log_ranges) // 2], log_ranges[-1]]
        fig = create_exp1_figure(results, display_log_ranges, compute_budgets)

        # Update title for Experiment 3 context (include loss surface params to show it's symmetric)
        fig.suptitle(
            f"Experiment 3: {sim_config.name} configuration\n"
            f"Loss surface: α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f} (symmetric)\n"
            f"drift_rate={sim_config.drift_rate}, center_scale={sim_config.center_scale}",
            fontsize=11,
            y=0.995,
        )

        # Save individual figure
        fig_path = output_dir / f"{sim_config.name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {fig_path}")
        plt.close(fig)

    # Create and save combined error analysis figure
    # Only include subset of configs for cleaner visualization
    combined_configs = {"baseline", "drift_0.4", "scale_2.0"}
    combined_results = [r for r in all_results if r["config"].name in combined_configs]

    print(f"\n{'─' * 70}")
    print("Generating combined error analysis figure...")
    combined_fig = create_figure(combined_results)
    combined_path = output_dir / "combined_errors.png"
    combined_fig.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {combined_path}")
    plt.close(combined_fig)

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Maximum errors at largest sampling range")
    print("=" * 70)
    print(f"{'Config':<15} {'drift':>8} {'scale':>8} {'a_err%':>10} {'b_err%':>10}")
    print("-" * 70)

    for results in all_results:
        sim_config = results["config"]
        max_a_err = results["a_error"][-1] * 100
        max_b_err = results["b_error"][-1] * 100
        print(f"{sim_config.name:<15} {sim_config.drift_rate:>8.1f} {sim_config.center_scale:>8.1f} {max_a_err:>+10.2f} {max_b_err:>+10.2f}")

    print("\nExperiment 3 complete.")
    return all_results


if __name__ == "__main__":
    main()
