"""Experiment 12: Residual Distributions.

Fits Approach 3 (with and without log-loss) to each experiment in the unified
isoFLOP dataset, then visualises per-budget residual distributions as layered
rug + boxplot + KDE plots.  Outliers are flagged with the modified Z-score
(MAD-based) method.

Usage:
    uv run python -m scaling_law_analysis.experiments.exp12_residual_distributions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    ParameterGrid,
    SurfaceFitResult,
    fit_approach3,
)
from scaling_law_analysis.experiments.common import prepare_output_dir
from scaling_law_analysis.experiments.exp10_compounding_errors import setup_style

# ── Constants ────────────────────────────────────────────────────────────────

ISOFLOPS_CSV = config.PROJECT_ROOT / "data" / "isoflops" / "isoflops.csv"

RESIDUAL_FIT_GRID = ParameterGrid(
    E=np.linspace(0.1, 5.0, 8),
    A=np.logspace(1, 6, 8),
    B=np.logspace(1, 6, 8),
    alpha=np.linspace(0.05, 0.95, 8),
    beta=np.linspace(0.05, 0.95, 8),
)

# Modified Z-score threshold (Iglewicz & Hoaglin, 1993).
# 3.5 is the recommended "conservative" cutoff.
MAD_THRESHOLD = 3.5

# Minimum points required to draw a KDE.
MIN_KDE_POINTS = 5

EXPERIMENT_SHORT_NAMES: dict[str, str] = {
    "epochai_chinchilla__massivetext__chinchilla": "Epoch AI / Chinchilla",
    "llama3__llama_3__llama_3__exp_loss": "Llama 3 (exp loss)",
    "llama3__llama_3__llama_3__raw_loss": "Llama 3 (raw loss)",
    "marin_202603__comma__llama_2": "Marin / CoMMA",
    "marin_202603__dclm__llama_2": "Marin / DCLM",
    "marin_202603__nemotron__llama_2": "Marin / Nemotron",
    "misfitting__fineweb_c4__transformer": "Misfitting / FineWeb-C4",
    "ml_scalefit__massivetext__chinchilla": "ML-Scalefit / Chinchilla",
}

OUTLIER_COLOR = "#d62728"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _modified_z_scores(x: np.ndarray) -> np.ndarray:
    """Modified Z-scores using Median Absolute Deviation (MAD).

    The constant 0.6745 (= Phi^{-1}(0.75)) normalises MAD so that it is
    consistent with the standard deviation for normally distributed data.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0.0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def _outlier_mask(x: np.ndarray, threshold: float = MAD_THRESHOLD) -> np.ndarray:
    """Boolean mask – True for outliers."""
    return np.abs(_modified_z_scores(x)) > threshold


def _fmt_budget(budget: float) -> str:
    """Format a compute budget for y-axis tick labels, e.g. '6e18'."""
    exp = int(np.floor(np.log10(budget)))
    mantissa = budget / 10**exp
    if abs(mantissa - round(mantissa)) < 0.05:
        return f"{round(mantissa):.0f}e{exp}"
    return f"{mantissa:.1f}e{exp}"


# ── Fitting & residuals ─────────────────────────────────────────────────────


def _fit_experiment(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    use_logloss: bool,
) -> SurfaceFitResult:
    """Fit Approach 3 to one experiment's data."""
    return fit_approach3(
        N,
        D,
        L,
        grid=RESIDUAL_FIT_GRID,
        use_lse=True,
        use_grad=True,
        use_logloss=use_logloss,
    )


def _compute_residuals(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    fit: SurfaceFitResult,
) -> np.ndarray:
    """Compute ``L_obs - L_pred`` for every data point.

    ``to_loss_surface().loss()`` always returns predictions in natural scale
    (E + A/N^α + B/D^β) regardless of whether the fit used log-loss.
    """
    surface = fit.to_loss_surface()
    predicted = surface.E + surface.A / N**surface.alpha + surface.B / D**surface.beta
    return L - predicted


# ── Visualisation ────────────────────────────────────────────────────────────


def _draw_budget_row(
    ax: plt.Axes,
    residuals: np.ndarray,
    outlier: np.ndarray,
    y: float,
    color: str,
    row_height: float,
) -> None:
    """Draw rug + boxplot + KDE for one compute-budget row."""
    normal = residuals[~outlier]
    outliers = residuals[outlier]

    # ── Rug (bottom slice of the row) ────────────────────────────────────
    rug_y = y - 0.30 * row_height
    rug_h = 0.12 * row_height
    if len(normal) > 0:
        ax.vlines(
            normal,
            rug_y - rug_h / 2,
            rug_y + rug_h / 2,
            colors=color,
            linewidth=0.6,
            alpha=0.7,
            zorder=1,
        )
    if len(outliers) > 0:
        ax.scatter(
            outliers,
            np.full_like(outliers, rug_y),
            s=16,
            facecolors="none",
            edgecolors=OUTLIER_COLOR,
            linewidths=0.9,
            zorder=2,
        )

    # ── Boxplot (middle slice) ───────────────────────────────────────────
    if len(residuals) >= 2:
        q1, med, q3 = np.percentile(residuals, [25, 50, 75])
        iqr = q3 - q1
        whislo = residuals[residuals >= q1 - 1.5 * iqr]
        whislo = float(whislo.min()) if len(whislo) > 0 else float(q1)
        whishi = residuals[residuals <= q3 + 1.5 * iqr]
        whishi = float(whishi.max()) if len(whishi) > 0 else float(q3)
        stats = dict(
            med=float(med),
            q1=float(q1),
            q3=float(q3),
            whislo=whislo,
            whishi=whishi,
            fliers=[],
        )
        box_width = 0.22 * row_height
        bp = ax.bxp(
            [stats],
            positions=[y],
            widths=[box_width],
            vert=False,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(color)
                line.set_linewidth(1.0)
        # Mark the mean
        ax.plot(
            np.mean(residuals),
            y,
            marker="D",
            markersize=3.5,
            color=color,
            zorder=4,
        )

    # ── KDE (top slice) ──────────────────────────────────────────────────
    if len(residuals) >= MIN_KDE_POINTS:
        try:
            kde = gaussian_kde(residuals, bw_method="silverman")
        except np.linalg.LinAlgError:
            return
        pad = (residuals.max() - residuals.min()) * 0.15
        x_eval = np.linspace(residuals.min() - pad, residuals.max() + pad, 200)
        density = kde(x_eval)
        kde_base = y + 0.10 * row_height
        kde_max_h = 0.32 * row_height
        if density.max() > 0:
            scaled = density / density.max() * kde_max_h
            ax.fill_between(
                x_eval,
                kde_base,
                kde_base + scaled,
                color=color,
                alpha=0.25,
                zorder=2,
            )
            ax.plot(
                x_eval,
                kde_base + scaled,
                color=color,
                linewidth=0.8,
                zorder=2,
            )


def plot_residual_distributions(
    results: dict[str, dict],
    output_path: Path,
    use_logloss: bool,
) -> None:
    """Create the multi-panel residual distribution figure."""
    setup_style()

    experiments = sorted(results.keys())
    n_exp = len(experiments)
    n_cols = 4
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(24, 5.5 * n_rows),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)

    cmap = plt.colormaps["viridis"]

    for idx, experiment in enumerate(experiments):
        ax = axes[idx // n_cols, idx % n_cols]
        edf: pd.DataFrame = results[experiment]["df"]
        fit: SurfaceFitResult = results[experiment]["fit"]

        budgets = sorted(edf["budget"].unique(), reverse=True)
        n_budgets = len(budgets)
        n_rows_plot = n_budgets + 1  # +1 for "Overall"
        colors = cmap(np.linspace(0.15, 0.85, n_budgets))
        row_height = 1.0

        for yi, budget in enumerate(budgets):
            mask = edf["budget"] == budget
            residuals = np.asarray(edf.loc[mask, "residual"])
            outlier = np.asarray(edf.loc[mask, "outlier"], dtype=bool)
            _draw_budget_row(ax, residuals, outlier, float(yi), colors[yi], row_height)

        # "Overall" row pooling all budgets, placed after the last budget
        overall_y = float(n_budgets)
        all_residuals = np.asarray(edf["residual"])
        all_outlier = np.asarray(edf["outlier"], dtype=bool)
        _draw_budget_row(
            ax, all_residuals, all_outlier, overall_y, "#555555", row_height
        )
        # Separator line between budgets and overall
        ax.axhline(
            n_budgets - 0.5, color="grey", linestyle=":", linewidth=0.6, zorder=0
        )

        # Zero-residual reference line
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.7, zorder=0)

        # Axes
        labels = [_fmt_budget(b) for b in budgets] + ["Overall"]
        ax.set_yticks(range(n_rows_plot))
        ax.set_yticklabels(
            labels,
            fontsize=max(6, 9 - n_budgets // 8),
        )
        ax.set_ylim(-0.6, n_rows_plot - 0.4)
        ax.set_xlabel("Residual: observed \u2212 predicted (nats)", fontsize=9)
        ax.set_ylabel("Compute budget (FLOPs)", fontsize=9)
        ax.grid(True, axis="x", alpha=0.2)

        # Title
        short = EXPERIMENT_SHORT_NAMES.get(experiment, experiment)
        s = fit.to_loss_surface()
        ax.set_title(
            f"{short}\n" rf"E={s.E:.3f}  $\alpha$={s.alpha:.3f}  $\beta$={s.beta:.3f}",
            fontsize=9,
        )

    # Hide unused axes
    for idx in range(n_exp, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    logloss_label = "log-loss" if use_logloss else "raw-loss"
    fig.suptitle(
        f"Approach 3 Residual Distributions by Compute Budget ({logloss_label})",
        fontsize=13,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp12")
    df = pd.read_csv(ISOFLOPS_CSV)
    experiments = sorted(df["experiment"].unique())

    for use_logloss in [False, True]:
        logloss_tag = "true" if use_logloss else "false"
        print(f"\n{'=' * 60}")
        print(f"Fitting with use_logloss={use_logloss}")
        print(f"{'=' * 60}")

        results: dict[str, dict] = {}
        for experiment in experiments:
            edf = df[df["experiment"] == experiment].copy()
            N = np.asarray(edf["params"])
            D = np.asarray(edf["tokens"])
            L = np.asarray(edf["loss"])

            print(f"\n  {experiment} ({len(edf)} points) ...")
            fit = _fit_experiment(N, D, L, use_logloss=use_logloss)
            residuals = _compute_residuals(N, D, L, fit)
            outlier = _outlier_mask(residuals)

            edf["residual"] = residuals
            edf["outlier"] = outlier

            s = fit.to_loss_surface()
            print(
                f"    E={s.E:.4f}, A={s.A:.1f}, B={s.B:.1f}, "
                f"\u03b1={s.alpha:.4f}, \u03b2={s.beta:.4f}"
            )
            print(
                f"    Residuals: mean={residuals.mean():.6f}, "
                f"std={residuals.std():.6f}, "
                f"outliers={outlier.sum()}/{len(edf)}"
            )

            results[experiment] = {"fit": fit, "df": edf}

        plot_residual_distributions(
            results,
            output_dir / f"residuals_logloss_{logloss_tag}.png",
            use_logloss=use_logloss,
        )


if __name__ == "__main__":
    main()
