"""Experiment 10: Compounding Errors.

Measures how multiple bias sources compound when extrapolating D* predictions
to 10²⁴ FLOPs. Combines loss surface asymmetry, sampling grid width, and
sampling center bias (constant offset or compute-dependent drift) to show
that individually small errors can interact to produce large extrapolation
errors.

Usage:
    uv run python -m scaling_law_analysis.experiments.exp10_compounding_errors
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    LossSurface,
    ParabolaFitResult,
)
from scaling_law_analysis.experiments.common import (
    ASYMMETRIC_SURFACE,
    CHINCHILLA_SURFACE,
    SYMMETRIC_SURFACE,
    fit_simulated_approach2,
    prepare_output_dir,
)

# ── Surfaces ─────────────────────────────────────────────────────────────────

SURFACES = [
    ("Symmetric", SYMMETRIC_SURFACE),
    ("Chinchilla", CHINCHILLA_SURFACE),
    ("Asymmetric", ASYMMETRIC_SURFACE),
]

SurfaceList = list[tuple[str, LossSurface]]

# ── Sampling grids ───────────────────────────────────────────────────────────

GRID_WIDTHS = [
    ("XS (±2×)", np.log10(2)),
    ("S (±4×)", np.log10(4)),
    ("L (±8×)", np.log10(8)),
    ("XL (±16×)", np.log10(16)),
]

TRAINING_BUDGETS = np.array([1e17, 1e18, 1e19, 1e20, 1e21])
EVAL_BUDGET = 1e24

# ── Bias configurations ──────────────────────────────────────────────────────

OFF_CENTER_SCALE = 3.0
OFF_CENTER_DRIFT_RATE = np.log10(3)

COMPOUNDING_CONFIGS = [
    {
        "label": f"Offset by {OFF_CENTER_SCALE:.0f}×",
        "drift_rate": 0.0,
        "center_scale": OFF_CENTER_SCALE,
    },
    {
        "label": f"Drift to {10**OFF_CENTER_DRIFT_RATE:.0f}×",
        "drift_rate": OFF_CENTER_DRIFT_RATE,
        "center_scale": 1.0,
    },
]

# ── Cost assumptions ─────────────────────────────────────────────────────────

# H100 SXM bf16 theoretical peak: 1,979 TFLOPS
# Source: https://www.nvidia.com/en-us/data-center/h100
H100_BF16_TFLOPS = 1979

# Model FLOPs Utilization: 50%
# Source: arxiv:2401.00448 (Beyond Chinchilla-Optimal)
MFU = 0.50

# Cloud GPU cost: $2/GPU-hour
# Source: arxiv:2512.13961 (OLMo 3)
GPU_COST_PER_HOUR = 2.0

# Derived: dollars per FLOP
DOLLARS_PER_FLOP = GPU_COST_PER_HOUR / (H100_BF16_TFLOPS * 1e12 * MFU * 3600)


# ── Core computation ─────────────────────────────────────────────────────────


def compare_allocations(
    surface: LossSurface, fit_result: ParabolaFitResult, eval_budget: float
) -> dict:
    """Compare true vs inferred optimal allocation at a given compute budget.

    Returns dict with true/inferred N, D, loss, C, and derived errors.
    The key FLOP metric is "equivalent_C": the budget that would achieve
    the same (suboptimal) loss with optimal allocation. The difference
    eval_budget - equivalent_C represents wasted compute.
    """
    true_N = surface.N_opt(eval_budget)
    true_D = surface.D_opt(eval_budget)
    inf_D = fit_result.D_opt(eval_budget)
    true_loss = surface.loss(true_N, true_D)
    # Loss at the inferred allocation, constrained to the actual budget:
    # N is forced by C = 6*N*D, so N = C / (6 * D_inferred)
    constrained_N = eval_budget / (6 * inf_D)
    inf_loss = surface.loss(constrained_N, inf_D)
    # What budget would achieve inf_loss with optimal allocation?
    equivalent_C = surface.C_from_loss(inf_loss)
    wasted_flops = eval_budget - equivalent_C
    return {
        "true_N": true_N,
        "true_D": true_D,
        "inferred_N": constrained_N,
        "inferred_D": inf_D,
        "true_loss": true_loss,
        "inferred_loss": inf_loss,
        "equivalent_C": equivalent_C,
        "rel_error_pct": (inf_D - true_D) / true_D * 100,
        "loss_error_nats": inf_loss - true_loss,
        "wasted_flops": wasted_flops,
        "wasted_dollars": wasted_flops * DOLLARS_PER_FLOP,
    }


def fit_and_compare(
    surface: LossSurface,
    eval_budget: float,
    training_budgets: np.ndarray,
    *,
    log_range: float,
    center_scale: float,
    drift_rate: float,
) -> dict:
    """Fit Approach 2 on training data and compare allocation at eval budget."""
    fit_result = fit_simulated_approach2(
        compute_budgets=training_budgets,
        surface=surface,
        drift_rate=drift_rate,
        center_scale=center_scale,
        log_range=log_range,
    )
    return compare_allocations(surface, fit_result, eval_budget)


def run(
    eval_budget: float,
    training_budgets: np.ndarray,
    surfaces: SurfaceList = SURFACES,
) -> dict:
    """Run compounding errors analysis.

    Returns:
        Dict keyed by config label → surface key → grid name → comparison data.
    """
    results: dict = {}
    for cfg in COMPOUNDING_CONFIGS:
        config_label = cfg["label"]
        results[config_label] = {}
        for surface_name, surface in surfaces:
            key = surface_name.lower().replace(" ", "_")
            results[config_label][key] = {}
            for grid_name, log_range in GRID_WIDTHS:
                results[config_label][key][grid_name] = fit_and_compare(
                    surface=surface,
                    eval_budget=eval_budget,
                    training_budgets=training_budgets,
                    log_range=log_range,
                    center_scale=cfg["center_scale"],
                    drift_rate=cfg["drift_rate"],
                )
    return results


# ── Visualization ────────────────────────────────────────────────────────────


def setup_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def plot(
    results: dict,
    output_path: str | Path,
    eval_budget: float,
    training_budgets: np.ndarray,
    surfaces: SurfaceList = SURFACES,
) -> None:
    """Create the compounding errors bar chart figure."""
    setup_style()
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    n_configs = len(COMPOUNDING_CONFIGS)
    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 4.5), sharey=True)
    if n_configs == 1:
        axes = [axes]

    x_positions = np.arange(len(surfaces))
    n_grids = len(GRID_WIDTHS)
    bar_width = 0.20
    offsets = [(i - (n_grids - 1) / 2) * bar_width for i in range(n_grids)]

    for ax_idx, cfg in enumerate(COMPOUNDING_CONFIGS):
        ax = axes[ax_idx]
        config_label = cfg["label"]

        for (grid_name, _log_range), color, offset in zip(GRID_WIDTHS, colors, offsets):
            rel_errors = []
            for surface_name, _surface in surfaces:
                key = surface_name.lower().replace(" ", "_")
                rel_errors.append(
                    results[config_label][key][grid_name]["rel_error_pct"]
                )

            ax.bar(
                x_positions + offset,
                rel_errors,
                bar_width,
                label=grid_name,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

        if n_configs <= 2 or ax_idx == n_configs // 2:
            ax.set_xlabel("Loss Surface")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [f"{name}\n(α={s.alpha:.2f}, β={s.beta:.2f})" for name, s in surfaces],
            fontsize=9,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(config_label, fontsize=11)

        if ax_idx == 0:
            ax.set_ylabel("Relative Error in D* (%)")
            ax.legend(title="Sampling Grid", loc="upper left", fontsize=8)

    lo = int(np.log10(training_budgets[0]))
    hi = int(np.log10(training_budgets[-1]))
    if eval_budget == 1e24:
        budget_str = "10²⁴"
    else:
        budget_str = f"{eval_budget:.1e}"
    fig.suptitle(
        "Compounding Errors: D* Prediction Error by Bias Configuration\n"
        f"(Fitting: 10^{lo}–10^{hi} FLOPs → Extrapolating to {budget_str} FLOPs)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_csv(
    results: dict,
    csv_path: str | Path,
    eval_budget: float,
    training_budgets: np.ndarray,
    surfaces: SurfaceList = SURFACES,
) -> None:
    """Export raw error data to CSV."""
    TRAINING_RANGE = f"{training_budgets[0]:.0e}-{training_budgets[-1]:.0e}"
    with open(csv_path, "w") as f:
        f.write(
            "config,drift_rate,center_scale,surface,alpha,beta,"
            "grid_name,log_range,training_range,eval_budget,"
            "true_D,inferred_D,abs_error,rel_error_pct\n"
        )
        for cfg in COMPOUNDING_CONFIGS:
            config_label = cfg["label"]
            for surface_name, surface in surfaces:
                key = surface_name.lower().replace(" ", "_")
                for grid_name, log_range in GRID_WIDTHS:
                    data = results[config_label][key][grid_name]
                    true_D = data["true_D"]
                    inferred_D = data["inferred_D"]
                    abs_error = inferred_D - true_D
                    rel_error = data["rel_error_pct"]
                    f.write(
                        f'"{config_label}",{cfg["drift_rate"]},'
                        f'{cfg["center_scale"]},'
                        f"{surface_name},{surface.alpha},{surface.beta},"
                        f'"{grid_name}",{log_range},{TRAINING_RANGE},'
                        f"{eval_budget:.0e},"
                        f"{true_D:.15e},{inferred_D:.15e},"
                        f"{abs_error:.15e},{rel_error:.15f}\n"
                    )
    print(f"Saved: {csv_path}")


# ── Report formatting ────────────────────────────────────────────────────────


def fmt_direction(value: float) -> str:
    """Describe whether a signed error implies over- or under-utilization."""
    if value > 0:
        return "over"
    elif value < 0:
        return "under"
    return "exact"


def fmt_flops(flops: float) -> str:
    """Format a FLOP count in human-readable scientific notation."""
    exp = int(np.floor(np.log10(abs(flops))))
    mantissa = flops / 10**exp
    return f"{mantissa:.2f}e{exp}"


def fmt_count(value: float) -> str:
    """Format a large count with T/B/M/K suffix."""
    v = abs(value)
    if v >= 1e12:
        return f"{value / 1e12:.1f}T"
    if v >= 1e9:
        return f"{value / 1e9:.1f}B"
    if v >= 1e6:
        return f"{value / 1e6:.0f}M"
    if v >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def fmt_dollars(dollars: float) -> str:
    """Format a dollar amount."""
    v = abs(dollars)
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    if v >= 1:
        return f"${v:.0f}"
    if v >= 0.01:
        return f"${v:.2f}"
    return f"<$0.01"


def fmt_surface_result(d: dict, eval_budget: float) -> str:
    """Format a single surface's comparison results."""
    d_dir = fmt_direction(d["inferred_D"] - d["true_D"])
    n_dir = fmt_direction(d["inferred_N"] - d["true_N"])
    n_err_pct = abs(d["inferred_N"] - d["true_N"]) / d["true_N"] * 100
    true_ratio = d["true_D"] / d["true_N"]
    inf_ratio = d["inferred_D"] / d["inferred_N"]
    dcl_pct = d["wasted_flops"] / eval_budget * 100
    lines = [
        f"  D* (tokens):  {fmt_count(d['true_D']):>10s} true → "
        f"{fmt_count(d['inferred_D']):>10s} inferred  "
        f"({d_dir}, {abs(d['rel_error_pct']):.1f}%)",
        f"  N* (params):  {fmt_count(d['true_N']):>10s} true → "
        f"{fmt_count(d['inferred_N']):>10s} inferred  "
        f"({n_dir}, {n_err_pct:.1f}%)",
        f"  D*/N* ratio:  {true_ratio:>10.1f} true → " f"{inf_ratio:>10.1f} inferred",
        f"  Loss (nats):  {d['true_loss']:.6f} true → "
        f"{d['inferred_loss']:.6f} inferred  "
        f"(+{d['loss_error_nats']:.6f} penalty)",
        f"  DCL: {fmt_flops(d['wasted_flops'])} FLOPs "
        f"({dcl_pct:.1f}% of budget, {fmt_dollars(d['wasted_dollars'])})",
    ]
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp10")

    results = run(
        eval_budget=EVAL_BUDGET,
        training_budgets=TRAINING_BUDGETS,
        surfaces=SURFACES,
    )
    plot(
        results,
        output_dir / "compounding_errors.png",
        eval_budget=EVAL_BUDGET,
        training_budgets=TRAINING_BUDGETS,
        surfaces=SURFACES,
    )
    save_csv(
        results,
        output_dir / "compounding_errors.csv",
        eval_budget=EVAL_BUDGET,
        training_budgets=TRAINING_BUDGETS,
        surfaces=SURFACES,
    )


if __name__ == "__main__":
    main()
