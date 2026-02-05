"""Experiment 3: Sampling Drift Sensitivity Analysis.

This experiment investigates how the accuracy of Chinchilla Approach 2 varies
with systematic biases in the sampling center (drift and scale) across
different loss surface geometries.

Hypothesis: Both linear drift and constant center scaling degrade exponent recovery,
with the magnitude depending on loss surface geometry.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import LossSurface
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    LOSS_SURFACES,
    BIAS_CONFIGS,
    DISPLAY_LOG_RANGES,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    LOG_RANGES,
    N_POINTS,
    TICK_POSITIONS,
    log_range_to_label,
)
from scaling_law_analysis.experiments.exp1_empirical_error import (
    run_experiment,
    create_figure as create_exp1_figure,
)


def _configure_ax(ax: plt.Axes, title: str, show_legend: bool = False):
    """Apply common axis configuration."""
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sampling range")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(title, fontsize=9)
    if show_legend:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    tick_labels = [log_range_to_label(lr) for lr in TICK_POSITIONS]
    ax.set_xticks(TICK_POSITIONS)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")


def run_all_configurations(
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int,
    output_dir: Path,
) -> dict[str, list[dict]]:
    """Run experiments for all loss surface and bias configurations.

    Args:
        compute_budgets: Compute budgets for IsoFLOP sampling
        log_ranges: Sampling ranges to sweep
        n_points: Number of points per IsoFLOP curve
        output_dir: Directory to save individual figures

    Returns:
        Dict mapping surface_name -> list of results (one per bias config)
    """
    all_results: dict[str, list[dict]] = {}

    for surface_name, loss in LOSS_SURFACES:
        print(f"\n{'=' * 70}")
        print(f"Loss Surface: {surface_name}")
        print(
            f"  α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}"
        )
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
            print(f"    drift_rate={drift_rate}, center_scale={center_scale}")

            results = run_experiment(
                sim_config=sim_config,
                compute_budgets=compute_budgets,
                log_ranges=log_ranges,
                n_points=n_points,
            )
            surface_results.append(results)

            # Generate individual exp1-style figure
            fig = create_exp1_figure(results, DISPLAY_LOG_RANGES, compute_budgets)
            fig.suptitle(
                f"Experiment 3: {surface_name} / {bias_name}\n"
                f"Loss surface: α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}\n"
                f"drift_rate={drift_rate}, center_scale={center_scale}",
                fontsize=11,
                y=0.995,
            )
            fig_path = output_dir / f"{surface_name}_{bias_name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"    Saved: {fig_path}")
            plt.close(fig)

        all_results[surface_name] = surface_results

    return all_results


def create_param_errors_figure(
    all_results: dict[str, list[dict]],
) -> plt.Figure:
    """Create parameter estimation errors figure (3 rows × 4 cols).

    Rows: one per loss surface (symmetric, chinchilla, high_imbalance)
    Cols: N* exponent, D* exponent, N* intercept, D* intercept
    """
    n_surfaces = len(LOSS_SURFACES)
    fig, axes = plt.subplots(n_surfaces, 4, figsize=(16, 4 * n_surfaces))

    for row, (surface_name, loss) in enumerate(LOSS_SURFACES):
        results_list = all_results[surface_name]
        colors = plt.colormaps["viridis"](np.linspace(0, 0.9, len(results_list)))

        for i, results in enumerate(results_list):
            log_ranges = results["log_ranges"]
            label = results["config"].name
            color = colors[i]

            # Col 0: N* exponent error
            axes[row, 0].plot(
                log_ranges,
                results["a_error"] * 100,
                "o-",
                color=color,
                markersize=3,
                label=label,
            )
            # Col 1: D* exponent error
            axes[row, 1].plot(
                log_ranges,
                results["b_error"] * 100,
                "o-",
                color=color,
                markersize=3,
                label=label,
            )
            # Col 2: N* intercept error
            axes[row, 2].plot(
                log_ranges,
                results["a_intercept_error"] * 100,
                "o-",
                color=color,
                markersize=3,
                label=label,
            )
            # Col 3: D* intercept error
            axes[row, 3].plot(
                log_ranges,
                results["b_intercept_error"] * 100,
                "o-",
                color=color,
                markersize=3,
                label=label,
            )

        # Configure axes for this row
        ratio = loss.alpha / loss.beta
        row_label = (
            f"{surface_name} (α={loss.alpha:.2f}, β={loss.beta:.2f}, ratio={ratio:.2f})"
        )
        _configure_ax(axes[row, 0], f"N* exponent error\n{row_label}")
        _configure_ax(axes[row, 1], f"D* exponent error\n{row_label}")
        _configure_ax(axes[row, 2], f"N* intercept error\n{row_label}")
        _configure_ax(
            axes[row, 3], f"D* intercept error\n{row_label}", show_legend=(row == 0)
        )

    fig.suptitle(
        "Experiment 3: Parameter Estimation Errors vs Sampling Bias",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()

    return fig


def create_optima_errors_figure(
    surface_name: str,
    loss: LossSurface,
    results_list: list[dict],
) -> plt.Figure:
    """Create optimal value estimation errors figure for a single loss surface.

    Rows: baseline, drift_0.4 (highest drift), scale_2.0 (highest scale) - 3 rows
    Cols: N* relative, D* relative, N* signed, D* signed - 4 cols
    Distinct colors for N* vs D*; opacity/size denote compute budget.

    Args:
        surface_name: Name of the loss surface
        loss: The LossSurface object
        results_list: List of results for each bias configuration

    Returns:
        Matplotlib figure
    """
    # Filter to only baseline, drift_0.4, scale_2.0
    selected_configs = {"baseline", "drift_0.4", "scale_2.0"}
    filtered_results = [r for r in results_list if r["config"].name in selected_configs]
    # Sort to ensure consistent order: baseline, drift_0.4, scale_2.0
    config_order = ["baseline", "drift_0.4", "scale_2.0"]
    filtered_results = sorted(
        filtered_results, key=lambda r: config_order.index(r["config"].name)
    )

    compute_budgets = filtered_results[0]["compute_budgets"]
    n_budgets = len(compute_budgets)
    n_configs = len(filtered_results)

    # Compute true optima for signed error calculation
    true_N_opts = np.array([loss.N_opt(C) for C in compute_budgets])
    true_D_opts = np.array([loss.D_opt(C) for C in compute_budgets])

    fig, axes = plt.subplots(n_configs, 4, figsize=(16, 3.5 * n_configs))

    # Opacity and size increase with compute budget (consistent with exp1)
    alphas = np.linspace(0.2, 1.0, n_budgets)
    sizes = np.linspace(10, 36, n_budgets)

    # Distinct colors for N* and D*
    color_N = "C0"  # Blue
    color_D = "C2"  # Green

    for row, results in enumerate(filtered_results):
        config_name = results["config"].name
        log_ranges = results["log_ranges"]

        for j, C in enumerate(compute_budgets):
            alpha = alphas[j]
            size = sizes[j]

            # N_opt_errors and D_opt_errors have shape (n_log_ranges, n_budgets)
            # Relative errors: (inferred - true) / true
            N_rel_errors = results["N_opt_errors"][:, j] * 100
            D_rel_errors = results["D_opt_errors"][:, j] * 100

            # Signed errors: inferred - true (in absolute units)
            N_signed_errors = results["N_opt_errors"][:, j] * true_N_opts[j]
            D_signed_errors = results["D_opt_errors"][:, j] * true_D_opts[j]

            # Col 0: N* relative error
            axes[row, 0].scatter(
                log_ranges, N_rel_errors, c=color_N, alpha=alpha, s=size, marker="o"
            )
            # Col 1: D* relative error
            axes[row, 1].scatter(
                log_ranges, D_rel_errors, c=color_D, alpha=alpha, s=size, marker="s"
            )
            # Col 2: N* signed error
            axes[row, 2].scatter(
                log_ranges, N_signed_errors, c=color_N, alpha=alpha, s=size, marker="o"
            )
            # Col 3: D* signed error
            axes[row, 3].scatter(
                log_ranges, D_signed_errors, c=color_D, alpha=alpha, s=size, marker="s"
            )

        # Configure axes for this row
        _configure_ax(axes[row, 0], f"N* relative error (%)\n{config_name}")
        _configure_ax(axes[row, 1], f"D* relative error (%)\n{config_name}")
        _configure_ax(axes[row, 2], f"N* signed error\n{config_name}")
        axes[row, 2].set_ylabel("Signed error")
        _configure_ax(axes[row, 3], f"D* signed error\n{config_name}")
        axes[row, 3].set_ylabel("Signed error")

    ratio = loss.alpha / loss.beta
    fig.suptitle(
        f"Experiment 3: Optimal Value Errors - {surface_name}\n"
        f"α={loss.alpha:.2f}, β={loss.beta:.2f}, ratio={ratio:.2f}\n"
        f"(opacity/size increase with compute budget: 10¹⁷ → 10²¹)",
        fontsize=12,
        y=0.995,
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
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp3")

    # Run all configurations (also saves individual figures)
    all_results = run_all_configurations(
        compute_budgets, log_ranges, n_points, output_dir
    )

    # Create and save parameter errors figure
    print(f"\n{'─' * 70}")
    print("Generating parameter errors figure...")
    param_fig = create_param_errors_figure(all_results)
    param_path = output_dir / "combined_param_errors.png"
    param_fig.savefig(param_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {param_path}")
    plt.close(param_fig)

    # Create and save optima errors figures (one per loss surface)
    print(f"\n{'─' * 70}")
    print("Generating optima errors figures (one per surface)...")
    for surface_name, loss in LOSS_SURFACES:
        optima_fig = create_optima_errors_figure(
            surface_name, loss, all_results[surface_name]
        )
        optima_path = output_dir / f"{surface_name}_optima_errors.png"
        optima_fig.savefig(optima_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {optima_path}")
        plt.close(optima_fig)

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Maximum errors at largest sampling range")
    print("=" * 70)

    for surface_name, _ in LOSS_SURFACES:
        print(f"\n{surface_name}:")
        print(f"{'Config':<15} {'drift':>8} {'scale':>8} {'a_err%':>10} {'b_err%':>10}")
        print("-" * 60)

        for results in all_results[surface_name]:
            sim_config = results["config"]
            max_a_err = results["a_error"][-1] * 100
            max_b_err = results["b_error"][-1] * 100
            print(
                f"{sim_config.name:<15} {sim_config.drift_rate:>8.1f} {sim_config.center_scale:>8.1f} {max_a_err:>+10.2f} {max_b_err:>+10.2f}"
            )

    print("\nExperiment 3 complete.")
    return all_results


if __name__ == "__main__":
    main()
