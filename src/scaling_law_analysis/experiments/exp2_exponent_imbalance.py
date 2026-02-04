"""Experiment 2: Exponent Imbalance Sensitivity Analysis.

This experiment investigates how the accuracy of Chinchilla Approach 2 varies
with the degree of imbalance between scaling exponents α and β.

Hypothesis: Greater asymmetry between α and β leads to larger recovery errors.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scaling_law_analysis import config
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    EXP2_CONFIGS,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    LOG_RANGES,
    N_POINTS,
    TICK_POSITIONS,
    log_range_to_label,
)
from scaling_law_analysis.experiments.exp1_empirical_error import (
    run_experiment,
    create_figure,
)


def plot_combined_errors(
    all_results: list[dict],
    ax_a: plt.Axes,
    ax_b: plt.Axes,
):
    """Plot exponent recovery errors for all configurations on shared axes.

    Args:
        all_results: List of experiment results, one per configuration
        ax_a: Axes for 'a' exponent errors
        ax_b: Axes for 'b' exponent errors
    """
    # Sort by imbalance ratio for consistent legend ordering
    sorted_results = sorted(
        all_results, key=lambda r: r["config"].loss.imbalance_ratio
    )
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_results)))

    for i, results in enumerate(sorted_results):
        sim_config = results["config"]
        loss = sim_config.loss
        log_ranges = results["log_ranges"]
        a_error = results["a_error"] * 100
        b_error = results["b_error"] * 100

        label = f"{sim_config.name} (α={loss.alpha:.2f}, β={loss.beta:.2f}, ratio={loss.imbalance_ratio:.2f})"

        ax_a.plot(
            log_ranges, a_error,
            "o-", color=colors[i], markersize=4, label=label,
        )
        ax_b.plot(
            log_ranges, b_error,
            "s-", color=colors[i], markersize=4, label=label,
        )

    # Configure axes
    for ax, title in [(ax_a, "N* exponent (a) error"), (ax_b, "D* exponent (b) error")]:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sampling range around optimal N")
        ax.set_ylabel("Relative error (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Set x-axis labels
        tick_positions = TICK_POSITIONS
        tick_labels = [log_range_to_label(lr) for lr in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")


def create_combined_figure(all_results: list[dict]) -> plt.Figure:
    """Create combined error analysis figure for all configurations.

    Args:
        all_results: List of experiment results, one per configuration

    Returns:
        Matplotlib figure
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    plot_combined_errors(all_results, ax_a, ax_b)

    fig.suptitle(
        "Experiment 2: Exponent Recovery Error vs Imbalance\n"
        "Comparing recovery accuracy across different α/β configurations",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    return fig


def main():
    """Run Experiment 2: analyze sensitivity to exponent imbalance."""
    print("=" * 70)
    print("Experiment 2: Exponent Imbalance Sensitivity Analysis")
    print("=" * 70)

    # Experiment parameters
    compute_budgets = COMPUTE_BUDGETS
    log_ranges = LOG_RANGES
    n_points = N_POINTS

    # Output directory for Experiment 2
    output_dir = prepare_output_dir(config.RESULTS_DIR / "exp2")

    all_results = []

    for sim_config in EXP2_CONFIGS:
        loss = sim_config.loss
        print(f"\n{'─' * 70}")
        print(f"Configuration: {sim_config.name}")
        print(f"  α={loss.alpha:.2f}, β={loss.beta:.2f} (ratio={loss.imbalance_ratio:.2f})")
        print(f"  a={loss.a:.4f}, b={loss.b:.4f}")
        print(f"{'─' * 70}")

        # Run experiment for this configuration
        results = run_experiment(
            sim_config=sim_config,
            compute_budgets=compute_budgets,
            log_ranges=log_ranges,
            n_points=n_points,
        )
        all_results.append(results)

        # Generate individual figure (same format as Experiment 1)
        display_log_ranges = [log_ranges[0], log_ranges[len(log_ranges) // 2], log_ranges[-1]]
        fig = create_figure(results, display_log_ranges, compute_budgets)

        # Update title for Experiment 2 context
        fig.suptitle(
            f"Experiment 2: {sim_config.name} configuration (α/β ratio = {loss.imbalance_ratio:.2f})\n"
            f"α={loss.alpha:.2f}, β={loss.beta:.2f} → a={loss.a:.4f}, b={loss.b:.4f}",
            fontsize=11,
            y=0.98,
        )

        # Save individual figure
        fig_path = output_dir / f"{sim_config.name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {fig_path}")
        plt.close(fig)

    # Create and save combined error analysis figure
    print(f"\n{'─' * 70}")
    print("Generating combined error analysis figure...")
    combined_fig = create_combined_figure(all_results)
    combined_path = output_dir / "combined_errors.png"
    combined_fig.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {combined_path}")
    plt.close(combined_fig)

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Maximum errors at largest sampling range")
    print("=" * 70)
    print(f"{'Config':<20} {'α/β ratio':>10} {'a_err%':>10} {'b_err%':>10}")
    print("-" * 70)

    for results in all_results:
        sim_config = results["config"]
        loss = sim_config.loss
        max_a_err = results["a_error"][-1] * 100
        max_b_err = results["b_error"][-1] * 100
        print(f"{sim_config.name:<20} {loss.imbalance_ratio:>10.2f} {max_a_err:>+10.2f} {max_b_err:>+10.2f}")

    print("\nExperiment 2 complete.")
    return all_results


if __name__ == "__main__":
    main()
