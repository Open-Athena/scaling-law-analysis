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
from scaling_law_analysis.chinchilla import LossSurface
from scaling_law_analysis.experiments.common import (
    fit_simulated_approach2,
    prepare_output_dir,
)

# ── Surfaces ────────────────────────────────────────────────────────────────

SYMMETRIC_SURFACE = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)
CHINCHILLA_SURFACE = LossSurface(alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69)


def _exponents_from_ratio(ratio: float) -> tuple[float, float]:
    """Compute alpha and beta from ratio, keeping sum = 0.62."""
    exponent_sum = 0.62
    beta = exponent_sum / (1 + ratio)
    alpha = exponent_sum * ratio / (1 + ratio)
    return alpha, beta


ASYMMETRIC_SURFACE = LossSurface.from_chinchilla(*_exponents_from_ratio(3))

SURFACES = [
    ("Symmetric", SYMMETRIC_SURFACE),
    ("Chinchilla", CHINCHILLA_SURFACE),
    ("Asymmetric", ASYMMETRIC_SURFACE),
]

# ── Sampling grids ──────────────────────────────────────────────────────────

GRID_WIDTHS = [
    ("XS (±2×)", np.log10(2)),
    ("S (±4×)", np.log10(4)),
    ("L (±8×)", np.log10(8)),
    ("XL (±16×)", np.log10(16)),
]

TRAINING_BUDGETS = np.array([1e17, 1e18, 1e19, 1e20, 1e21])
EVAL_BUDGET = 1e24

# ── Bias configurations ────────────────────────────────────────────────────

OFF_CENTER_SCALE = 3.0
OFF_CENTER_DRIFT_RATE = np.log10(3)

COMPOUNDING_CONFIGS = [
    {
        "label": f"Offset by {OFF_CENTER_SCALE:.0f}×",
        "drift_rate": 0.0,
        "center_scale": OFF_CENTER_SCALE,
    },
    {
        "label": f"Drift to {OFF_CENTER_SCALE:.0f}×",
        "drift_rate": OFF_CENTER_DRIFT_RATE,
        "center_scale": 1.0,
    },
]


# ── Core computation ────────────────────────────────────────────────────────


def compute_extrapolation_errors(
    surface: LossSurface,
    training_budgets: np.ndarray,
    eval_budgets: np.ndarray,
    log_range: float,
    n_points: int = 15,
    center_scale: float = 1.0,
    drift_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute D* errors at evaluation budgets.

    Returns:
        Tuple of (true_D, inferred_D, abs_errors) arrays at each eval budget.
    """
    result = fit_simulated_approach2(
        compute_budgets=training_budgets,
        surface=surface,
        drift_rate=drift_rate,
        center_scale=center_scale,
        n_points=n_points,
        log_range=log_range,
    )
    true_D = np.array([surface.D_opt(C) for C in eval_budgets])
    inferred_D = np.array([result.D_opt(C) for C in eval_budgets])
    abs_errors = inferred_D - true_D
    return true_D, inferred_D, abs_errors


def run() -> dict:
    """Run compounding errors analysis.

    Returns:
        Dict keyed by config label → surface key → grid name → error data.
    """
    results: dict = {}
    for config_entry in COMPOUNDING_CONFIGS:
        config_label = config_entry["label"]
        results[config_label] = {}
        for surface_name, surface in SURFACES:
            key = surface_name.lower().replace(" ", "_")
            results[config_label][key] = {}
            for grid_name, log_range in GRID_WIDTHS:
                true_D, inferred_D, errors = compute_extrapolation_errors(
                    surface=surface,
                    training_budgets=TRAINING_BUDGETS,
                    eval_budgets=np.array([EVAL_BUDGET]),
                    log_range=log_range,
                    center_scale=config_entry["center_scale"],
                    drift_rate=config_entry["drift_rate"],
                )
                rel_error = errors[0] / true_D[0] * 100
                results[config_label][key][grid_name] = {
                    "true_D": true_D[0],
                    "inferred_D": inferred_D[0],
                    "rel_error_pct": rel_error,
                }
    return results


# ── Visualization ───────────────────────────────────────────────────────────


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


def plot(results: dict, output_path: str | Path) -> None:
    """Create the compounding errors bar chart figure."""
    setup_style()
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    n_configs = len(COMPOUNDING_CONFIGS)
    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 4.5), sharey=True)
    if n_configs == 1:
        axes = [axes]

    x_positions = np.arange(len(SURFACES))
    n_grids = len(GRID_WIDTHS)
    bar_width = 0.20
    offsets = [(i - (n_grids - 1) / 2) * bar_width for i in range(n_grids)]

    for ax_idx, cfg in enumerate(COMPOUNDING_CONFIGS):
        ax = axes[ax_idx]
        config_label = cfg["label"]

        for i, ((grid_name, _log_range), color, offset) in enumerate(
            zip(GRID_WIDTHS, colors, offsets)
        ):
            rel_errors = []
            for surface_name, _surface in SURFACES:
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
            [f"{name}\n(α={s.alpha:.2f}, β={s.beta:.2f})" for name, s in SURFACES],
            fontsize=9,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(config_label, fontsize=11)

        if ax_idx == 0:
            ax.set_ylabel("Relative Error in D* (%)")
            ax.legend(title="Sampling Grid", loc="upper left", fontsize=8)

    fig.suptitle(
        "Compounding Errors: D* Prediction Error by Bias Configuration\n"
        "(Fitting: 10¹⁷–10²¹ FLOPs → Extrapolating to 10²⁴ FLOPs)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_csv(results: dict, csv_path: str | Path) -> None:
    """Export raw error data to CSV."""
    TRAINING_RANGE = "1e17-1e21"
    with open(csv_path, "w") as f:
        f.write(
            "config,drift_rate,center_scale,surface,alpha,beta,"
            "grid_name,log_range,training_range,eval_budget,"
            "true_D,inferred_D,abs_error,rel_error_pct\n"
        )
        for cfg in COMPOUNDING_CONFIGS:
            config_label = cfg["label"]
            for surface_name, surface in SURFACES:
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
                        f"{EVAL_BUDGET:.0e},"
                        f"{true_D:.15e},{inferred_D:.15e},"
                        f"{abs_error:.15e},{rel_error:.15f}\n"
                    )
    print(f"Saved: {csv_path}")


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp10")
    results = run()
    plot(results, output_dir / "compounding_errors.png")
    save_csv(results, output_dir / "compounding_errors.csv")


if __name__ == "__main__":
    main()
