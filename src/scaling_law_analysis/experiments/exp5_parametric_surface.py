"""Experiment 5: Parametric Surface Fitting.

This experiment investigates fitting the loss surface L(N, D) = E + A/N^α + B/D^β
directly via variable projection (grid search over α/β + NNLS for E/A/B).

Hypothesis: Variable projection with grid search (over α/β) provides stable and
accurate scaling law parameter recovery, and extrapolation using fitted parameters
remains accurate even at compute budgets far beyond the fitting range.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    DEFAULT_APPROACH3_GRID,
    DEFAULT_VPNLS_GRID,
    FINE_VPNLS_GRID,
    FitError,
    FitStatus,
    LBFGSBOptions,
    LBFGSB_DEFAULT_EPS,
    fit_approach3,
    fit_vpnls,
)
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


# Both methods use equal-sized initialization grids so that any accuracy
# difference reflects the optimizer and loss-landscape geometry, not an
# initialization budget advantage.
def _nanmax_or_raise(arr: np.ndarray, label: str) -> float:
    """Return nanmax of arr, raising ValueError if all values are NaN."""
    if not np.any(np.isfinite(arr)):
        raise ValueError(
            f"All values for '{label}' are NaN — every fit failed for this parameter"
        )
    return float(np.nanmax(arr))


assert DEFAULT_APPROACH3_GRID.total_size == DEFAULT_VPNLS_GRID.total_size, (
    f"Approach 3 grid ({DEFAULT_APPROACH3_GRID.total_size}) must equal "
    f"VPNLS grid ({DEFAULT_VPNLS_GRID.total_size})"
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


def _verify_default_eps() -> None:
    """Assert that LBFGSB_DEFAULT_EPS matches scipy's actual default.

    The method comparison labels and article exposition reference this value
    explicitly, so we need to ensure it stays correct across scipy versions.
    """
    import inspect

    # Private API import used intentionally for exposition (verifying the default eps value).
    from scipy.optimize._lbfgsb_py import _minimize_lbfgsb

    sig = inspect.signature(_minimize_lbfgsb)
    scipy_default = sig.parameters["eps"].default
    assert (
        scipy_default == LBFGSB_DEFAULT_EPS
    ), f"LBFGSB_DEFAULT_EPS={LBFGSB_DEFAULT_EPS} != scipy default={scipy_default}"


def surface_fitter(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_range: float,
    n_points: int,
):
    """Fit surface parameters and return D_opt function."""
    N, D, L = sample_isoflop_data(sim_config, compute_budgets, log_range, n_points)
    result = fit_vpnls(N, D, L)
    fitted_surface = result.to_loss_surface()
    return fitted_surface.D_opt


def compute_param_errors(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int,
    method: str = "nelder-mead",
    jac: str | None = None,
    eps: float | None = None,
    use_grad: bool = True,
) -> dict:
    """Compute parameter errors across sampling ranges.

    Args:
        sim_config: Simulation configuration
        compute_budgets: Compute budgets for IsoFLOP sampling
        log_ranges: Array of sampling ranges to sweep
        n_points: Number of points per IsoFLOP curve
        method: Optimization method for fit_vpnls
        jac: Jacobian scheme for L-BFGS-B (e.g. "3-point")
        eps: Override finite-difference step size for L-BFGS-B
        use_grad: For approach3, whether to use analytical gradients

    Returns:
        Dictionary with arrays of relative errors for each parameter
    """
    loss = sim_config.loss
    n_ranges = len(log_ranges)

    # Arrays to store relative errors and per-range failure flags
    errors = {key: np.zeros(n_ranges) for key in ["E", "A", "B", "alpha", "beta"]}
    failed = np.zeros(n_ranges, dtype=bool)

    for i, log_range in enumerate(log_ranges):
        N, D, L = sample_isoflop_data(sim_config, compute_budgets, log_range, n_points)
        try:
            if method == "approach3":
                a3_options = LBFGSBOptions(jac=jac) if jac is not None else None
                result = fit_approach3(
                    N,
                    D,
                    L,
                    grid=DEFAULT_APPROACH3_GRID,
                    use_grad=use_grad,
                    options=a3_options,
                )
            else:
                vpnls_grid = FINE_VPNLS_GRID if method == "grid" else None
                vpnls_options: LBFGSBOptions | None = None
                if method == "l-bfgs-b" and (jac is not None or eps is not None):
                    vpnls_options = LBFGSBOptions(jac=jac, eps=eps)
                result = fit_vpnls(
                    N, D, L, grid=vpnls_grid, method=method, options=vpnls_options
                )
        except FitError:
            failed[i] = True
            for key in errors:
                errors[key][i] = np.nan
            continue

        # Treat ABNORMAL and BOUND_HIT as failures — this is noise-free data
        # where neither condition should occur.
        if result.status in (FitStatus.ABNORMAL, FitStatus.BOUND_HIT):
            failed[i] = True
            for key in errors:
                errors[key][i] = np.nan
            continue

        # Compute relative errors
        errors["E"][i] = (result.E - loss.E) / loss.E
        errors["A"][i] = (result.A - loss.A) / loss.A
        errors["B"][i] = (result.B - loss.B) / loss.B
        errors["alpha"][i] = (result.alpha - loss.alpha) / loss.alpha
        errors["beta"][i] = (result.beta - loss.beta) / loss.beta

    return {
        "config": sim_config,
        "log_ranges": log_ranges,
        "failed": failed,
        "n_failures": int(failed.sum()),
        **errors,
    }


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


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for a single optimization method in the comparison."""

    label: str
    color: str
    method: str
    jac: str | None = None
    eps: float | None = None
    use_grad: bool = True


# Method configurations for the comparison figure.
# The default L-BFGS-B uses scipy's default eps=1e-8. The two eps variants
# bracket it symmetrically in log space (100x above and below) to demonstrate
# sensitivity to the finite-difference step size.
METHOD_CONFIGS = [
    # Approach 3 — 5D direct optimization (orange family)
    MethodConfig("5D L-BFGS-B (analytical grad)", "#e65100", "approach3"),
    MethodConfig("5D L-BFGS-B (finite diff)", "#ff9800", "approach3", use_grad=False),
    MethodConfig(
        "5D L-BFGS-B (central diff)",
        "#e6b800",
        "approach3",
        use_grad=False,
        jac="3-point",
    ),
    # VPNLS — our method: variable projection + NNLS + Nelder-Mead (standalone, dark blue)
    MethodConfig("2D Nelder-Mead (VPNLS)", "#1a237e", "nelder-mead"),
    # Variable projection — 2D L-BFGS-B variants (blue family)
    MethodConfig(
        f"2D L-BFGS-B (default eps={LBFGSB_DEFAULT_EPS:.0e})", "#1976d2", "l-bfgs-b"
    ),
    MethodConfig("2D L-BFGS-B (central diff)", "#64b5f6", "l-bfgs-b", jac="3-point"),
    MethodConfig("2D L-BFGS-B (eps=1e-6)", "#90caf9", "l-bfgs-b", eps=1e-6),
    MethodConfig("2D L-BFGS-B (eps=1e-10)", "#42a5f5", "l-bfgs-b", eps=1e-10),
    # Variable projection — 2D Grid (standalone, green)
    MethodConfig("2D Grid (256²)", "#2e7d32", "grid"),
]


def run_method_comparison(
    compute_budgets: np.ndarray,
    log_ranges: np.ndarray,
    n_points: int,
) -> dict[str, list[tuple[MethodConfig, dict]]]:
    """Run parameter error analysis for all methods (baseline bias only).

    Returns:
        Dict mapping surface_name -> list of (MethodConfig, results) tuples
    """
    # Verify that our claimed default eps matches what scipy actually uses.
    # Run a trivial fit with and without explicit eps and check identical results.
    _verify_default_eps()

    all_results: dict[str, list[tuple[MethodConfig, dict]]] = {}

    for surface_name, loss in LOSS_SURFACES:
        print(f"\n{'=' * 70}")
        print(f"Loss Surface: {surface_name}")
        print(f"  α={loss.alpha:.2f}, β={loss.beta:.2f}")
        print("=" * 70)

        surface_results: list[tuple[MethodConfig, dict]] = []
        sim_config = SimulationConfig(
            name="baseline", loss=loss, drift_rate=0.0, center_scale=1.0
        )

        for mc in METHOD_CONFIGS:
            print(f"  Method: {mc.label} ...", end=" ", flush=True)
            results = compute_param_errors(
                sim_config=sim_config,
                compute_budgets=compute_budgets,
                log_ranges=log_ranges,
                n_points=n_points,
                method=mc.method,
                jac=mc.jac,
                eps=mc.eps,
                use_grad=mc.use_grad,
            )
            surface_results.append((mc, results))

            max_errs = {
                k: _nanmax_or_raise(np.abs(results[k]), k) * 100
                for k in ["E", "A", "B", "alpha", "beta"]
            }
            failures = results["n_failures"]
            fail_str = f" ({failures}/{len(log_ranges)} failed)" if failures else ""
            print(
                f"max errors: "
                + ", ".join(f"{k}={v:.2e}%" for k, v in max_errs.items())
                + fail_str
            )

        all_results[surface_name] = surface_results

    return all_results


def create_method_comparison_figure(
    all_results: dict[str, list[tuple[MethodConfig, dict]]],
) -> plt.Figure:
    """Create method comparison figure (n_surfaces rows × 5 param cols)."""
    n_surfaces = len(LOSS_SURFACES)
    param_names = ["E", "A", "B", "α", "β"]
    error_keys = ["E", "A", "B", "alpha", "beta"]

    fig, axes = plt.subplots(
        n_surfaces,
        5,
        figsize=(16, 3 * n_surfaces),
    )
    if n_surfaces == 1:
        axes = axes[np.newaxis, :]

    for row, (surface_name, loss) in enumerate(LOSS_SURFACES):
        method_results = all_results[surface_name]
        is_last_row = row == n_surfaces - 1

        for col, (param_name, error_key) in enumerate(zip(param_names, error_keys)):
            ax = axes[row, col]

            for mc, results in method_results:
                errors = np.abs(results[error_key]) * 100  # Absolute relative error %
                ax.plot(
                    results["log_ranges"],
                    errors,
                    "o-",
                    color=mc.color,
                    markersize=3,
                    label=mc.label,
                )

            ax.set_yscale("log")
            row_label = f"{surface_name} (α={loss.alpha:.2f}, β={loss.beta:.2f})"
            ax.set_title(f"|{param_name} error|\n{row_label}", fontsize=9)
            ax.grid(True, alpha=0.3)

            # x-axis: ticks on all rows, labels only on last row
            tick_labels = [log_range_to_label(lr) for lr in TICK_POSITIONS]
            ax.set_xticks(TICK_POSITIONS)
            if is_last_row:
                ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
                ax.set_xlabel("Sampling range")
            else:
                ax.set_xticklabels([])

            # y-axis: label only on left-most column
            if col == 0:
                ax.set_ylabel("Absolute relative error (%)")
            else:
                ax.set_ylabel("")

    # Legend below the plots in a horizontal layout
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        borderaxespad=0.3,
        frameon=True,
    )

    fig.suptitle(
        "Experiment 5: Method Comparison — Parameter Recovery (baseline, no bias)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    return fig


def export_method_comparison_csv(
    all_results: dict[str, list[tuple[MethodConfig, dict]]],
    path: str | Path,
) -> None:
    """Export method comparison summary as CSV."""
    param_keys = ["E", "A", "B", "alpha", "beta"]
    rows = []
    for surface_name, _ in LOSS_SURFACES:
        for mc, results in all_results[surface_name]:
            n_ranges = len(results["log_ranges"])
            n_failures = results["n_failures"]
            max_errs = {
                k: _nanmax_or_raise(np.abs(results[k]), k) * 100 for k in param_keys
            }
            rows.append(
                f"{surface_name},{mc.method},{mc.jac or ''},{mc.eps or ''},"
                f"{mc.label},{n_failures}/{n_ranges},"
                + ",".join(f"{max_errs[k]:.2e}" for k in param_keys)
            )

    header = (
        "surface,method,jac,eps,label,failures,"
        "max_E_err%,max_A_err%,max_B_err%,max_alpha_err%,max_beta_err%"
    )
    Path(path).write_text(header + "\n" + "\n".join(rows) + "\n")


def main():
    """Run Experiment 5: parametric surface fitting analysis."""
    print("=" * 70)
    print("Experiment 5: Parametric Surface Fitting")
    print("=" * 70)

    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp5")

    # --- Part 1: Parameter Error Analysis (Nelder-Mead) ---
    print("\n" + "#" * 70)
    print("Part 1: Parameter Error Analysis (Nelder-Mead)")
    print("#" * 70)

    param_results = run_param_error_analysis(COMPUTE_BUDGETS, LOG_RANGES, N_POINTS)

    print(f"\n{'─' * 70}")
    print("Generating parameter errors figure...")
    param_fig = create_param_errors_figure(param_results)
    param_path = output_dir / "surface_param_errors.png"
    param_fig.savefig(param_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {param_path}")
    plt.close(param_fig)

    # --- Part 2: Extrapolation Error Analysis (Nelder-Mead) ---
    print("\n" + "#" * 70)
    print("Part 2: Extrapolation Error Analysis (Nelder-Mead)")
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

    # --- Part 3: Method Comparison ---
    print("\n" + "#" * 70)
    print("Part 3: Method Comparison (baseline, no bias)")
    print("#" * 70)

    method_results = run_method_comparison(COMPUTE_BUDGETS, LOG_RANGES, N_POINTS)

    print(f"\n{'─' * 70}")
    print("Generating method comparison figure...")
    method_fig = create_method_comparison_figure(method_results)
    method_path = output_dir / "parameter_recovery_detailed.png"
    method_fig.savefig(method_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {method_path}")
    plt.close(method_fig)

    csv_path = output_dir / "parameter_recovery_detailed.csv"
    export_method_comparison_csv(method_results, csv_path)
    print(f"Saved: {csv_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary: Parameter errors at largest sampling range (Nelder-Mead)")
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
    print("Summary: Method comparison at largest sampling range (baseline)")
    print("=" * 70)

    for surface_name, _ in LOSS_SURFACES:
        print(f"\n{surface_name}:")
        print(
            f"{'Method':<25} {'E%':>10} {'A%':>10} {'B%':>10} {'α%':>10} {'β%':>10} {'fails':>6}"
        )
        print("-" * 80)

        for mc, results in method_results[surface_name]:
            vals = [results[k][-1] * 100 for k in ["E", "A", "B", "alpha", "beta"]]
            fails = results["n_failures"]
            print(
                f"{mc.label:<25} "
                + " ".join(f"{v:>+10.2e}" for v in vals)
                + f" {fails:>6}"
            )

    print("\nExperiment 5 complete.")
    return {
        "param_results": param_results,
        "extrap_results": extrap_results,
        "method_results": method_results,
    }


if __name__ == "__main__":
    main()
