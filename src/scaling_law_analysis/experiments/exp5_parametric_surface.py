"""Experiment 5: Parametric Surface Fitting.

This experiment investigates fitting the loss surface L(N, D) = E + A/N^α + B/D^β
directly via variable projection (grid search over α/β + NNLS for E/A/B).

Hypothesis: Variable projection with grid search (over α/β) provides stable and
accurate scaling law parameter recovery.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    LossSurface,
    fit_surface,
    isoflop_sample,
    compute_center_offset,
)
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    LOSS_SURFACES as ALL_LOSS_SURFACES,
    BIAS_CONFIGS,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    LOG_RANGES,
    N_POINTS,
    TICK_POSITIONS,
)
from scaling_law_analysis.experiments.exp1_empirical_error import log_range_to_label

# TODO: Use all loss surfaces once performance is verified
# LOSS_SURFACES = ALL_LOSS_SURFACES
LOSS_SURFACES = [ALL_LOSS_SURFACES[1]]  # chinchilla only for now


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


def sample_isoflop_data(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_range: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample IsoFLOP data across all compute budgets.

    Args:
        sim_config: Simulation configuration with loss surface and bias parameters
        compute_budgets: Array of compute budgets (FLOPs)
        log_range: Sampling range in log10 space around optimal N
        n_points: Number of points per IsoFLOP curve

    Returns:
        Tuple of (N, D, L) arrays pooled across all compute budgets
    """
    all_N = []
    all_D = []
    all_L = []

    for C in compute_budgets:
        center_offset = compute_center_offset(
            C=C,
            compute_budgets=compute_budgets,
            drift_rate=sim_config.drift_rate,
            center_scale=sim_config.center_scale,
        )
        N, D, L = isoflop_sample(
            C=C,
            n_points=n_points,
            log_range=log_range,
            center_offset=center_offset,
            surface=sim_config.loss,
        )
        all_N.append(N)
        all_D.append(D)
        all_L.append(L)

    return np.concatenate(all_N), np.concatenate(all_D), np.concatenate(all_L)


def compute_param_errors(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int,
) -> dict:
    """Compute parameter errors across sampling ranges.

    Args:
        sim_config: Simulation configuration
        compute_budgets: Compute budgets for IsoFLOP sampling
        log_ranges: Array of sampling ranges to sweep
        n_points: Number of points per IsoFLOP curve

    Returns:
        Dictionary with arrays of relative errors for each parameter
    """
    loss = sim_config.loss
    n_ranges = len(log_ranges)

    # Arrays to store relative errors
    E_errors = np.zeros(n_ranges)
    A_errors = np.zeros(n_ranges)
    B_errors = np.zeros(n_ranges)
    alpha_errors = np.zeros(n_ranges)
    beta_errors = np.zeros(n_ranges)

    for i, log_range in enumerate(log_ranges):
        # Sample data
        N, D, L = sample_isoflop_data(sim_config, compute_budgets, log_range, n_points)

        # Fit surface
        result = fit_surface(N, D, L)

        # Compute relative errors
        E_errors[i] = (result.E - loss.E) / loss.E
        A_errors[i] = (result.A - loss.A) / loss.A
        B_errors[i] = (result.B - loss.B) / loss.B
        alpha_errors[i] = (result.alpha - loss.alpha) / loss.alpha
        beta_errors[i] = (result.beta - loss.beta) / loss.beta

    return {
        "config": sim_config,
        "log_ranges": log_ranges,
        "E_error": E_errors,
        "A_error": A_errors,
        "B_error": B_errors,
        "alpha_error": alpha_errors,
        "beta_error": beta_errors,
    }


def run_all_configurations(
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int,
) -> dict[str, list[dict]]:
    """Run experiments for all loss surface and bias configurations.

    Args:
        compute_budgets: Compute budgets for IsoFLOP sampling
        log_ranges: Sampling ranges to sweep
        n_points: Number of points per IsoFLOP curve

    Returns:
        Dict mapping surface_name -> list of results (one per bias config)
    """
    all_results: dict[str, list[dict]] = {}

    for surface_name, loss in LOSS_SURFACES:
        print(f"\n{'=' * 70}")
        print(f"Loss Surface: {surface_name}")
        print(f"  α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}, E={loss.E:.2f}")
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

            results = compute_param_errors(
                sim_config=sim_config,
                compute_budgets=compute_budgets,
                log_ranges=log_ranges,
                n_points=n_points,
            )
            surface_results.append(results)

            # Print summary at widest sampling range
            print(f"    Errors at widest range:")
            print(f"      E: {results['E_error'][-1]*100:+.2f}%")
            print(f"      A: {results['A_error'][-1]*100:+.2f}%")
            print(f"      B: {results['B_error'][-1]*100:+.2f}%")
            print(f"      α: {results['alpha_error'][-1]*100:+.2f}%")
            print(f"      β: {results['beta_error'][-1]*100:+.2f}%")

        all_results[surface_name] = surface_results

    return all_results


def create_param_errors_figure(
    all_results: dict[str, list[dict]],
) -> plt.Figure:
    """Create parameter estimation errors figure (n_surfaces rows × 5 cols).

    Rows: one per loss surface
    Cols: E, A, B, α, β
    """
    n_surfaces = len(LOSS_SURFACES)
    param_names = ["E", "A", "B", "α", "β"]
    error_keys = ["E_error", "A_error", "B_error", "alpha_error", "beta_error"]

    fig, axes = plt.subplots(n_surfaces, 5, figsize=(20, 4 * n_surfaces))
    # Ensure axes is 2D even with single row
    if n_surfaces == 1:
        axes = axes[np.newaxis, :]

    for row, (surface_name, loss) in enumerate(LOSS_SURFACES):
        results_list = all_results[surface_name]
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(results_list)))

        for col, (param_name, error_key) in enumerate(zip(param_names, error_keys)):
            ax = axes[row, col]

            for i, results in enumerate(results_list):
                log_ranges = results["log_ranges"]
                label = results["config"].name
                color = colors[i]

                errors = results[error_key] * 100  # Convert to %
                ax.plot(log_ranges, errors, "o-", color=color, markersize=3, label=label)

            # Configure axis
            ratio = loss.alpha / loss.beta
            row_label = f"{surface_name} (α={loss.alpha:.2f}, β={loss.beta:.2f})"
            _configure_ax(
                ax,
                f"{param_name} error\n{row_label}",
                show_legend=(row == 0 and col == 4),
            )

    fig.suptitle(
        "Experiment 5: Parameter Estimation Errors via Surface Fitting",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()

    return fig


def main():
    """Run Experiment 5: parametric surface fitting analysis."""
    print("=" * 70)
    print("Experiment 5: Parametric Surface Fitting")
    print("=" * 70)

    compute_budgets = COMPUTE_BUDGETS
    log_ranges = LOG_RANGES
    n_points = N_POINTS

    # Prepare output directory
    output_dir = prepare_output_dir(config.RESULTS_DIR / "exp5")

    # Run all configurations
    all_results = run_all_configurations(compute_budgets, log_ranges, n_points)

    # Create and save parameter errors figure
    print(f"\n{'─' * 70}")
    print("Generating parameter errors figure...")
    param_fig = create_param_errors_figure(all_results)
    param_path = output_dir / "surface_param_errors.png"
    param_fig.savefig(param_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {param_path}")
    plt.close(param_fig)

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Maximum errors at largest sampling range")
    print("=" * 70)

    for surface_name, _ in LOSS_SURFACES:
        print(f"\n{surface_name}:")
        print(f"{'Config':<15} {'E%':>8} {'A%':>8} {'B%':>8} {'α%':>8} {'β%':>8}")
        print("-" * 60)

        for results in all_results[surface_name]:
            sim_config = results["config"]
            E_err = results["E_error"][-1] * 100
            A_err = results["A_error"][-1] * 100
            B_err = results["B_error"][-1] * 100
            alpha_err = results["alpha_error"][-1] * 100
            beta_err = results["beta_error"][-1] * 100
            print(
                f"{sim_config.name:<15} "
                f"{E_err:>+8.2f} {A_err:>+8.2f} {B_err:>+8.2f} "
                f"{alpha_err:>+8.2f} {beta_err:>+8.2f}"
            )

    print("\nExperiment 5 complete.")
    return all_results


if __name__ == "__main__":
    main()
