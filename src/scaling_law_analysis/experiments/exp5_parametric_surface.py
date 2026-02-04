"""Experiment 5: Parametric Surface Fitting.

This experiment investigates fitting the loss surface L(N, D) = E + A/N^α + B/D^β
directly via variable projection (grid search over α/β + NNLS for E/A/B).

Hypothesis: Variable projection with grid search (over α/β) provides stable and
accurate scaling law parameter recovery, and extrapolation using fitted parameters
remains accurate even at compute budgets far beyond the fitting range.
"""

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import fit_surface
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    LOSS_SURFACES,
    BIAS_CONFIGS,
    DISPLAY_LOG_RANGES,
    DISPLAY_LOG_RANGE_NAMES,
    prepare_output_dir,
    COMPUTE_BUDGETS,
    EXTRAPOLATION_BUDGETS,
    LOG_RANGES,
    N_POINTS,
    TICK_POSITIONS,
    sample_isoflop_data,
    run_extrapolation_analysis,
    create_extrapolation_figure,
    log_range_to_label,
)


def _configure_ax(ax: plt.Axes, title: str, show_legend: bool = False):
    """Apply common axis configuration for parameter error plots."""
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


def surface_fitter(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_range: float,
    n_points: int,
):
    """Fit surface parameters and return D_opt function."""
    N, D, L = sample_isoflop_data(sim_config, compute_budgets, log_range, n_points)
    result = fit_surface(N, D, L)
    fitted_surface = result.to_loss_surface()
    return fitted_surface.D_opt


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
    errors = {key: np.zeros(n_ranges) for key in ["E", "A", "B", "alpha", "beta"]}

    for i, log_range in enumerate(log_ranges):
        N, D, L = sample_isoflop_data(sim_config, compute_budgets, log_range, n_points)
        result = fit_surface(N, D, L)

        # Compute relative errors
        errors["E"][i] = (result.E - loss.E) / loss.E
        errors["A"][i] = (result.A - loss.A) / loss.A
        errors["B"][i] = (result.B - loss.B) / loss.B
        errors["alpha"][i] = (result.alpha - loss.alpha) / loss.alpha
        errors["beta"][i] = (result.beta - loss.beta) / loss.beta

    return {"config": sim_config, "log_ranges": log_ranges, **errors}


def run_param_error_analysis(
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int,
) -> dict[str, list[dict]]:
    """Run parameter error analysis for all configurations.

    Returns:
        Dict mapping surface_name -> list of results (one per bias config)
    """
    all_results: dict[str, list[dict]] = {}

    for surface_name, loss in LOSS_SURFACES:
        print(f"\n{'=' * 70}")
        print(f"Loss Surface: {surface_name}")
        print(
            f"  α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}, E={loss.E:.2f}"
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

            results = compute_param_errors(
                sim_config=sim_config,
                compute_budgets=compute_budgets,
                log_ranges=log_ranges,
                n_points=n_points,
            )
            surface_results.append(results)

            # Print summary at widest sampling range
            print(f"    Errors at widest range:")
            for key in ["E", "A", "B", "alpha", "beta"]:
                label = "α" if key == "alpha" else ("β" if key == "beta" else key)
                print(f"      {label}: {results[key][-1]*100:+.2f}%")

        all_results[surface_name] = surface_results

    return all_results


def create_param_errors_figure(all_results: dict[str, list[dict]]) -> plt.Figure:
    """Create parameter estimation errors figure (n_surfaces rows × 5 cols)."""
    n_surfaces = len(LOSS_SURFACES)
    param_names = ["E", "A", "B", "α", "β"]
    error_keys = ["E", "A", "B", "alpha", "beta"]

    fig, axes = plt.subplots(n_surfaces, 5, figsize=(20, 4 * n_surfaces))
    if n_surfaces == 1:
        axes = axes[np.newaxis, :]

    for row, (surface_name, loss) in enumerate(LOSS_SURFACES):
        results_list = all_results[surface_name]
        colors = plt.colormaps["viridis"](np.linspace(0, 0.9, len(results_list)))

        for col, (param_name, error_key) in enumerate(zip(param_names, error_keys)):
            ax = axes[row, col]

            for i, results in enumerate(results_list):
                log_ranges = results["log_ranges"]
                label = results["config"].name
                errors = results[error_key] * 100  # Convert to %
                ax.plot(
                    log_ranges, errors, "o-", color=colors[i], markersize=3, label=label
                )

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

    output_dir = prepare_output_dir(config.RESULTS_DIR / "exp5")

    # --- Part 1: Parameter Error Analysis ---
    print("\n" + "#" * 70)
    print("Part 1: Parameter Error Analysis")
    print("#" * 70)

    param_results = run_param_error_analysis(COMPUTE_BUDGETS, LOG_RANGES, N_POINTS)

    print(f"\n{'─' * 70}")
    print("Generating parameter errors figure...")
    param_fig = create_param_errors_figure(param_results)
    param_path = output_dir / "surface_param_errors.png"
    param_fig.savefig(param_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {param_path}")
    plt.close(param_fig)

    # --- Part 2: Extrapolation Error Analysis ---
    print("\n" + "#" * 70)
    print("Part 2: Extrapolation Error Analysis")
    print("#" * 70)

    print(f"\nFitting compute budgets: {COMPUTE_BUDGETS}")
    print(
        f"Extrapolation budgets: {EXTRAPOLATION_BUDGETS[0]:.0e} to {EXTRAPOLATION_BUDGETS[-1]:.0e}"
    )

    extrap_results = run_extrapolation_analysis(
        fitter=surface_fitter,
        compute_budgets=COMPUTE_BUDGETS,
        extrapolation_budgets=EXTRAPOLATION_BUDGETS,
        log_ranges=DISPLAY_LOG_RANGES,
        log_range_names=DISPLAY_LOG_RANGE_NAMES,
        n_points=N_POINTS,
        loss_surfaces=LOSS_SURFACES,
        bias_configs=BIAS_CONFIGS,
    )

    print(f"\n{'─' * 70}")
    print("Generating extrapolation error figure...")
    extrap_fig = create_extrapolation_figure(
        all_results=extrap_results,
        loss_surfaces=LOSS_SURFACES,
        log_range_names=DISPLAY_LOG_RANGE_NAMES,
        log_ranges=DISPLAY_LOG_RANGES,
        title="Experiment 5: Extrapolation Error Analysis (Surface Fitting)",
        subtitle="Fitting: 10¹⁷-10²¹ FLOPs → Extrapolating to 10²²-10²⁵ FLOPs",
    )
    extrap_path = output_dir / "surface_extrapolation_error.png"
    extrap_fig.savefig(extrap_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {extrap_path}")
    plt.close(extrap_fig)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary: Parameter errors at largest sampling range")
    print("=" * 70)

    for surface_name, _ in LOSS_SURFACES:
        print(f"\n{surface_name}:")
        print(f"{'Config':<15} {'E%':>8} {'A%':>8} {'B%':>8} {'α%':>8} {'β%':>8}")
        print("-" * 60)

        for results in param_results[surface_name]:
            vals = [results[k][-1] * 100 for k in ["E", "A", "B", "alpha", "beta"]]
            print(
                f"{results['config'].name:<15} " + " ".join(f"{v:>+8.2f}" for v in vals)
            )

    print("\n" + "=" * 70)
    print("Summary: Maximum D* extrapolation errors")
    print("=" * 70)

    for range_name in DISPLAY_LOG_RANGE_NAMES:
        print(f"\n{range_name.upper()} sampling range:")
        for surface_name, _ in LOSS_SURFACES:
            print(f"\n  {surface_name}:")
            print(f"  {'Config':<15} {'max_D_err%':>12}")
            print(f"  {'-' * 30}")
            for results in extrap_results[range_name][surface_name]:
                max_D_err = np.abs(results["D_rel_errors"]).max() * 100
                print(f"  {results['config'].name:<15} {max_D_err:>12.2f}")

    print("\nExperiment 5 complete.")
    return {"param_results": param_results, "extrap_results": extrap_results}


if __name__ == "__main__":
    main()
