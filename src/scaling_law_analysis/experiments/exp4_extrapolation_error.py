"""Experiment 4: Extrapolation Error Analysis.

This experiment investigates how the accuracy of scaling law exponent inference
degrades when extrapolating to compute budgets beyond those used for fitting.

Hypothesis: Errors in inferred scaling exponents compound when extrapolating
to compute budgets beyond the fitting regime.
"""

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import LossSurface, DEFAULT_LOSS_SURFACE, fit_approach2
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    EXP3_LOSS_SURFACE,
    HIGH_IMBALANCE_CONFIG,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    LOG_RANGES,
    N_POINTS,
)
from scaling_law_analysis.experiments.exp1_empirical_error import (
    run_experiment,
    create_figure as create_exp1_figure,
)

# Sampling range used for the extrapolation error figure: ±10x around optimal
EXP4_EXTRAP_LOG_RANGE = 1.0

# Extrapolation compute budgets: 10^22 to 10^25 FLOPs (beyond fitting range)
EXP4_EXTRAPOLATION_BUDGETS = np.geomspace(1e22, 1e25, 16)

# Loss surfaces for Experiment 4
EXP4_LOSS_SURFACES = [
    ("symmetric", EXP3_LOSS_SURFACE),
    ("chinchilla", DEFAULT_LOSS_SURFACE),
    ("high_imbalance", HIGH_IMBALANCE_CONFIG.loss),
]

# Sampling bias configurations (drift_rate, center_scale, name)
EXP4_BIAS_CONFIGS = [
    (0.0, 1.0, "baseline"),
    (0.2, 1.0, "drift_0.2"),
    (0.4, 1.0, "drift_0.4"),
    (0.0, 1.5, "scale_1.5"),
    (0.0, 2.0, "scale_2.0"),
]


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


def create_faceted_figure(
    all_results_by_surface: dict[str, list[dict]],
    loss_surfaces: list[tuple[str, LossSurface]],
) -> plt.Figure:
    """Create faceted extrapolation error figure across loss surfaces.

    Args:
        all_results_by_surface: Dict mapping surface name to list of results
        loss_surfaces: List of (name, LossSurface) tuples

    Returns:
        Matplotlib figure with one facet per loss surface
    """
    n_surfaces = len(loss_surfaces)
    fig, axes = plt.subplots(1, n_surfaces, figsize=(5 * n_surfaces, 5))

    if n_surfaces == 1:
        axes = [axes]

    for ax, (surface_name, loss) in zip(axes, loss_surfaces):
        results_list = all_results_by_surface[surface_name]
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

        # Title with loss surface info
        ax.set_title(
            f"{surface_name}\n"
            f"α={loss.alpha:.2f}, β={loss.beta:.2f}, ratio={loss.imbalance_ratio:.2f}",
            fontsize=10,
        )

    for ax in axes:
        ax.set_ylabel("Relative error in D* (%)")
    axes[-1].legend(fontsize=8, loc="best")

    fig.suptitle(
        "Experiment 4: Extrapolation Error Analysis\n"
        "Fitting: ±10x sampling, compute budgets 10¹⁷-10²¹ FLOPs",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    return fig


def main():
    """Run Experiment 4: analyze extrapolation error across loss surfaces."""
    print("=" * 70)
    print("Experiment 4: Extrapolation Error Analysis")
    print("=" * 70)

    compute_budgets = COMPUTE_BUDGETS
    extrapolation_budgets = EXP4_EXTRAPOLATION_BUDGETS
    log_ranges = LOG_RANGES  # Full range for exp1-style figures
    extrap_log_range = EXP4_EXTRAP_LOG_RANGE  # ±10x for extrapolation figure
    n_points = N_POINTS

    # Prepare output directory
    output_dir = prepare_output_dir(config.RESULTS_DIR / "exp4")

    print(f"\nFitting compute budgets: {compute_budgets}")
    print(f"Extrapolation budgets: {extrapolation_budgets[0]:.0e} to {extrapolation_budgets[-1]:.0e}")
    print(f"Sampling ranges: ±{10**log_ranges[0]:.1f}x to ±{10**log_ranges[-1]:.0f}x")
    print(f"Extrapolation figure uses: ±{10**extrap_log_range:.0f}x")
    print(f"\nLoss surfaces: {[name for name, _ in EXP4_LOSS_SURFACES]}")

    # Store results by surface name (for extrapolation figure, using extrap_log_range)
    all_results_by_surface: dict[str, list[dict]] = {}

    for surface_name, loss in EXP4_LOSS_SURFACES:
        print(f"\n{'=' * 70}")
        print(f"Loss Surface: {surface_name}")
        print(f"  α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}")
        print(f"  imbalance ratio={loss.imbalance_ratio:.2f}")
        print("=" * 70)

        surface_results = []

        for drift_rate, center_scale, bias_name in EXP4_BIAS_CONFIGS:
            sim_config = SimulationConfig(
                name=bias_name,
                loss=loss,
                drift_rate=drift_rate,
                center_scale=center_scale,
            )

            print(f"\n{'─' * 70}")
            print(f"Configuration: {bias_name}")
            print(f"  drift_rate={drift_rate}, center_scale={center_scale}")
            print(f"{'─' * 70}")

            # Compute extrapolation errors (using fixed extrap_log_range for the figure)
            extrap_results = compute_extrapolation_errors(
                sim_config=sim_config,
                compute_budgets=compute_budgets,
                extrapolation_budgets=extrapolation_budgets,
                log_range=extrap_log_range,
                n_points=n_points,
            )
            surface_results.append(extrap_results)

            # Print summary
            print(f"  Fitted: a={extrap_results['fitted_a']:.4f} (true={extrap_results['true_a']:.4f})")
            print(f"  Fitted: b={extrap_results['fitted_b']:.4f} (true={extrap_results['true_b']:.4f})")
            print(f"  Max D* error: {np.abs(extrap_results['D_rel_errors']).max() * 100:.2f}%")

            # Generate individual exp1-style figure using full log_ranges
            exp_results = run_experiment(
                sim_config=sim_config,
                compute_budgets=compute_budgets,
                log_ranges=log_ranges,
                n_points=n_points,
            )

            # Display 3 representative sampling ranges (small, medium, large)
            display_log_ranges = [log_ranges[0], log_ranges[len(log_ranges) // 2], log_ranges[-1]]
            fig = create_exp1_figure(exp_results, display_log_ranges, compute_budgets)

            # Update title for Experiment 4 context
            fig.suptitle(
                f"Experiment 4: {surface_name} / {bias_name}\n"
                f"Loss surface: α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}\n"
                f"drift_rate={drift_rate}, center_scale={center_scale}",
                fontsize=11,
                y=0.995,
            )

            # Save individual figure
            fig_path = output_dir / f"{surface_name}_{bias_name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {fig_path}")
            plt.close(fig)

        all_results_by_surface[surface_name] = surface_results

    # Create and save main faceted extrapolation error figure
    print(f"\n{'─' * 70}")
    print("Generating faceted extrapolation error figure...")
    extrap_fig = create_faceted_figure(all_results_by_surface, EXP4_LOSS_SURFACES)
    extrap_path = output_dir / "extrapolation_error.png"
    extrap_fig.savefig(extrap_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {extrap_path}")
    plt.close(extrap_fig)

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Extrapolation errors by surface")
    print("=" * 70)

    for surface_name, _ in EXP4_LOSS_SURFACES:
        print(f"\n{surface_name}:")
        print(f"{'Config':<15} {'drift':>8} {'scale':>8} {'a_err%':>10} {'b_err%':>10} {'max_D_err%':>12}")
        print("-" * 70)

        for results in all_results_by_surface[surface_name]:
            sim_config = results["config"]
            a_err = (results["fitted_a"] - results["true_a"]) / results["true_a"] * 100
            b_err = (results["fitted_b"] - results["true_b"]) / results["true_b"] * 100
            max_D_err = np.abs(results["D_rel_errors"]).max() * 100
            print(
                f"{sim_config.name:<15} "
                f"{sim_config.drift_rate:>8.1f} "
                f"{sim_config.center_scale:>8.1f} "
                f"{a_err:>+10.2f} "
                f"{b_err:>+10.2f} "
                f"{max_D_err:>12.2f}"
            )

    print("\nExperiment 4 complete.")
    return all_results_by_surface


if __name__ == "__main__":
    main()
