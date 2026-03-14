"""Experiment 12: Residual Distributions.

Fits Approach 3 (with and without log-loss) to each experiment in the unified
isoFLOP dataset, then visualises per-budget residual distributions as layered
rug + boxplot + KDE plots.  Outliers are detected *before* fitting using a
three-stage pipeline: minimum-budget-size, negative-curvature, and Akima
leave-one-out spline checks.

Usage:
    uv run python -m scaling_law_analysis.experiments.exp12_residual_distributions
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import gaussian_kde, t as t_dist  # pyrefly: ignore

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    LossSurface,
    ParameterGrid,
    SurfaceFitResult,
    fit_approach3,
)
from scaling_law_analysis.data.transform import display_name, ordered_experiments
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

# Pre-fit outlier detection constants.
MIN_BUDGET_POINTS = 6
LOO_ZSCORE_THRESHOLD = 6.0
NEAR_DUP_LOG_TOL = 0.01  # relative tolerance on log(N) for near-duplicate binning
# Confidence level for the curvature parameter CI.  Higher → wider CI → more
# budgets flagged as having insignificant curvature.  Set to 0.0 to disable.
CURVATURE_CI = 0.95

REASON_NONE = ""
REASON_EXACT_DUP = "exact_duplicate"
REASON_DUP_PARAMS = "near_duplicate_params"
REASON_TOO_FEW = "too_few_points"
REASON_NEG_CURVATURE = "negative_curvature"
REASON_WEAK_CURVATURE = "weak_curvature"
REASON_SPLINE = "spline_outlier"

# Minimum points required to draw a KDE.
MIN_KDE_POINTS = 5

OUTLIER_COLOR = "#d62728"

# Per-experiment overrides for outlier detection thresholds.
# Keys: experiment name → dict with optional "min_budget_points" and/or "loo_zscore_threshold".
EXPERIMENT_OVERRIDES: dict[str, dict] = {
    # Effectively disables LOO spline outlier detection for misfitting.
    "misfitting__fineweb_c4__transformer": {"loo_zscore_threshold": 100.0},
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def detect_outliers(
    edf: pd.DataFrame,
    *,
    min_budget_points: int = MIN_BUDGET_POINTS,
    loo_zscore_threshold: float = LOO_ZSCORE_THRESHOLD,
    curvature_ci: float = CURVATURE_CI,
) -> pd.DataFrame:
    """Pre-fit outlier detection. Adds ``outlier`` and ``reason`` columns.

    Stage 0a — exact duplicate params: keep point closest to nominal budget.
    Stage 0b — near-duplicate params: same, within log(N) tolerance.
    Stage 1  — too few points: budgets with < *min_budget_points* are flagged.
    Stage 2a — negative curvature: quadratic fit log(L) ~ log(N)^2 with a <= 0.
    Stage 2b — weak curvature: CI for quadratic coefficient includes zero at
               *curvature_ci* confidence level (0.0 to disable).
    Stage 3  — Akima LOO: leave-one-out spline residuals, experiment-wide MAD.
    """
    edf = edf.copy()
    edf["outlier"] = pd.Series(False, index=edf.index, dtype=bool)
    edf["reason"] = REASON_NONE

    budgets = edf["budget"].unique()

    # Stage 0a: exact duplicate params within a budget — keep the point whose
    # implied compute (6·N·D) is closest to the nominal budget, ties by loss.
    for budget in budgets:
        mask = edf["budget"] == budget
        bdf = edf.loc[mask]
        dup_mask = bdf.duplicated(subset="params", keep=False)
        if dup_mask.any():
            for _, grp in bdf[dup_mask].groupby("params"):
                implied_c = 6 * grp["params"] * grp["tokens"]
                c_err = (implied_c - budget).abs()
                keep_idx = (
                    grp.assign(_c_err=c_err).sort_values(["_c_err", "loss"]).index[0]
                )
                flag_idx = grp.index.difference([keep_idx])
                edf.loc[flag_idx, "outlier"] = True
                edf.loc[flag_idx, "reason"] = REASON_EXACT_DUP

    # Stage 0b: near-duplicate params within a budget — keep the point whose
    # implied compute (6·N·D) is closest to the nominal budget.  Ties broken
    # by lowest loss.  Two points are "near-duplicates" if their log(N) values
    # differ by less than NEAR_DUP_LOG_TOL.  Only considers unflagged rows.
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        bdf = edf.loc[mask].sort_values("params")
        log_n = np.log(bdf["params"].to_numpy())
        indices = bdf.index.tolist()
        # Greedy binning: walk sorted log(N), start a new bin when gap > tol
        bins: list[list[int]] = []
        current_bin: list[int] = [0]
        for i in range(1, len(log_n)):
            if log_n[i] - log_n[current_bin[0]] <= NEAR_DUP_LOG_TOL:
                current_bin.append(i)
            else:
                bins.append(current_bin)
                current_bin = [i]
        bins.append(current_bin)
        for b in bins:
            if len(b) < 2:
                continue
            grp_idx = [indices[i] for i in b]
            grp = edf.loc[grp_idx]
            implied_c = 6 * grp["params"] * grp["tokens"]
            c_err = (implied_c - budget).abs()
            # Keep the point closest to nominal budget; break ties by loss
            keep_idx = grp.assign(_c_err=c_err).sort_values(["_c_err", "loss"]).index[0]
            flag_idx = [ix for ix in grp_idx if ix != keep_idx]
            edf.loc[flag_idx, "outlier"] = True
            edf.loc[flag_idx, "reason"] = REASON_DUP_PARAMS

    # Stage 1: too few clean points
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        if mask.sum() < min_budget_points:
            budget_mask = edf["budget"] == budget
            # Only flag rows not already flagged by an earlier stage
            unflagged = budget_mask & ~edf["outlier"]
            edf.loc[unflagged, "outlier"] = True
            edf.loc[unflagged, "reason"] = REASON_TOO_FEW

    # Stage 2a: negative curvature (only clean rows in surviving budgets)
    surviving_budgets = [
        b for b in budgets if not edf.loc[edf["budget"] == b, "outlier"].all()
    ]
    for budget in surviving_budgets:
        clean = edf.loc[(edf["budget"] == budget) & ~edf["outlier"]]
        log_n = np.log(clean["params"].to_numpy())
        log_l = np.log(clean["loss"].to_numpy())
        if len(log_n) >= 3:
            coeffs = np.polyfit(log_n, log_l, 2)
            if coeffs[0] <= 0:
                unflagged = (edf["budget"] == budget) & ~edf["outlier"]
                edf.loc[unflagged, "outlier"] = True
                edf.loc[unflagged, "reason"] = REASON_NEG_CURVATURE

    # Stage 2b: weak curvature — CI for the quadratic coefficient `a` includes
    # zero at the requested confidence level.  The point estimate â may be
    # positive (passed stage 2a) but statistically indistinguishable from zero,
    # meaning the isoflop curve is too flat to reliably locate an optimum.
    # Skipped when curvature_ci <= 0.
    if curvature_ci > 0:
        surviving_budgets = [
            b for b in budgets if not edf.loc[edf["budget"] == b, "outlier"].all()
        ]
        for budget in surviving_budgets:
            clean = edf.loc[(edf["budget"] == budget) & ~edf["outlier"]]
            log_n = np.log(clean["params"].to_numpy())
            log_l = np.log(clean["loss"].to_numpy())
            n_pts = len(log_n)
            if n_pts < 4:
                # Need at least 4 points for df = n-3 >= 1
                continue
            coeffs, cov = np.polyfit(log_n, log_l, 2, cov=True)
            a_hat = coeffs[0]
            if a_hat <= 0:
                # Already caught by stage 2a
                continue
            se_a = float(np.sqrt(cov[0, 0]))
            df = n_pts - 3
            t_crit = float(t_dist.ppf((1 + curvature_ci) / 2, df))
            ci_lower = a_hat - t_crit * se_a
            if ci_lower <= 0:
                unflagged = (edf["budget"] == budget) & ~edf["outlier"]
                edf.loc[unflagged, "outlier"] = True
                edf.loc[unflagged, "reason"] = REASON_WEAK_CURVATURE

    # Stage 3: Akima LOO (only clean rows in surviving budgets)
    surviving_budgets = [
        b for b in budgets if not edf.loc[edf["budget"] == b, "outlier"].all()
    ]
    # Collect LOO residuals for all surviving (clean) points
    loo_residuals: dict[int, float] = {}  # index -> residual
    for budget in surviving_budgets:
        clean = edf.loc[(edf["budget"] == budget) & ~edf["outlier"]].sort_values(
            "params"
        )
        log_n = np.log(clean["params"].to_numpy())
        loss = clean["loss"].to_numpy()
        indices = clean.index.tolist()
        for i in range(len(indices)):
            rest = np.delete(np.arange(len(indices)), i)
            if len(rest) < 2:
                continue
            log_n_rest = log_n[rest]
            l_rest = loss[rest]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interp = Akima1DInterpolator(log_n_rest, l_rest)
                interp.extrapolate = True
            predicted = float(interp(log_n[i], extrapolate=True))
            loo_residuals[indices[i]] = loss[i] - predicted

    if loo_residuals:
        resid_arr = np.array(list(loo_residuals.values()))
        med = np.median(resid_arr)
        mad = np.median(np.abs(resid_arr - med))
        if mad > 0:
            for ix, r in loo_residuals.items():
                z = 0.6745 * abs(r - med) / mad
                if z > loo_zscore_threshold:
                    edf.loc[ix, "outlier"] = True
                    edf.loc[ix, "reason"] = REASON_SPLINE

    return edf


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


def _flop_factor_predictions(
    edf: pd.DataFrame, surface: LossSurface
) -> tuple[np.ndarray, dict[float, float]]:
    """Per-budget FLOP-factor (k) estimation (clean points only).

    Given a global fit ``L = E + A·N^(-α) + B·D^(-β)`` with ``D = C/(k·N)``,
    estimates the per-budget k that best explains the data.  When k = 6 the
    standard C ≈ 6ND assumption holds.

    The substitution ``κ = k^β`` makes the problem linear:
        ``min_κ  Σᵢ (Lᵢ - E - A·Nᵢ^(-α) - κ·B·Nᵢ^β/C^β)²``

    Returns predictions array and a dict mapping budget → fitted k.
    """

    predictions = pd.Series(np.nan, index=edf.index)
    k_values: dict[float, float] = {}

    E, A, B = surface.E, surface.A, surface.B
    alpha, beta = surface.alpha, surface.beta

    for budget in edf["budget"].unique():
        bmask = edf["budget"] == budget
        bdf = edf.loc[bmask]
        cmask = ~bdf["outlier"]
        cdf = bdf[cmask].sort_values("params")

        if len(cdf) < 2:
            continue

        N = cdf["params"].to_numpy()
        L = cdf["loss"].to_numpy()

        # Residual after subtracting E and the param term
        r = L - E - A * N ** (-alpha)
        # Basis vector for the data term: g = B·N^β / C^β
        g = B * N**beta / budget**beta

        # OLS for κ = k^β:  κ = Σ(rᵢ·gᵢ) / Σ(gᵢ²)
        kappa = float(np.dot(r, g) / np.dot(g, g))
        if kappa <= 0:
            raise ValueError(
                f"Negative κ={kappa:.6f} for budget {budget:.2e}: "
                f"the global fit cannot explain this budget's data term"
            )
        k = kappa ** (1 / beta)

        k_values[budget] = k

        # Predict with the fitted k: D = C/(k·N)
        D = budget / (k * N)
        predictions.loc[cdf.index] = surface.loss(N, D)  # pyrefly: ignore

    return predictions.to_numpy(), k_values


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
    predicted = fit.to_loss_surface().loss(N, D)  # pyrefly: ignore
    return L - predicted


# ── Visualisation ────────────────────────────────────────────────────────────


def _draw_budget_row(
    ax: plt.Axes,
    residuals: np.ndarray,
    y: float,
    color: str,
    row_height: float,
) -> None:
    """Draw rug + boxplot + KDE for one compute-budget row (clean points only)."""
    clean = residuals

    # ── Rug (bottom slice of the row) ────────────────────────────────────
    rug_y = y - 0.30 * row_height
    rug_h = 0.12 * row_height

    if len(clean) > 0:
        ax.vlines(
            clean,
            rug_y - rug_h / 2,
            rug_y + rug_h / 2,
            colors=color,
            linewidth=0.6,
            alpha=0.7,
            zorder=1,
        )

    # ── Boxplot (middle slice) ───────────────────────────────────────────
    if len(clean) >= 2:
        q1, med, q3 = np.percentile(clean, [25, 50, 75])
        iqr = q3 - q1
        whislo = clean[clean >= q1 - 1.5 * iqr]
        whislo = float(whislo.min()) if len(whislo) > 0 else float(q1)
        whishi = clean[clean <= q3 + 1.5 * iqr]
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
            np.mean(clean),
            y,
            marker="D",
            markersize=3.5,
            color=color,
            zorder=4,
        )

    # ── KDE (top slice) — clean points only ──────────────────────────────
    if len(clean) >= MIN_KDE_POINTS:
        try:
            kde = gaussian_kde(clean, bw_method="silverman")
        except np.linalg.LinAlgError:
            return
        pad = (clean.max() - clean.min()) * 0.15
        x_eval = np.linspace(clean.min() - pad, clean.max() + pad, 200)
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
    *,
    title: str | None = None,
) -> None:
    """Create the multi-panel residual distribution figure."""
    setup_style()

    experiments = ordered_experiments(results.keys())
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
        fit: SurfaceFitResult | None = results[experiment].get("fit")

        all_budgets = sorted(edf["budget"].unique(), reverse=True)
        n_all_budgets = len(all_budgets)
        # Assign viridis colors based on original budget index (consistent across variants)
        all_colors = cmap(np.linspace(0.15, 0.85, n_all_budgets))

        # Only draw rows for budgets that have at least one clean point
        clean_edf = edf[edf["reason"] == REASON_NONE]
        drawn_budgets: list[tuple[float, str]] = []  # (budget, hex color)
        for bi, budget in enumerate(all_budgets):
            if (clean_edf["budget"] == budget).any():
                drawn_budgets.append((budget, mcolors.to_hex(all_colors[bi])))

        n_drawn = len(drawn_budgets)
        n_rows_plot = n_drawn + 1  # +1 for "Overall"
        row_height = 1.0

        for yi, (budget, bcolor) in enumerate(drawn_budgets):
            residuals = np.asarray(
                clean_edf.loc[clean_edf["budget"] == budget, "residual"]
            )
            _draw_budget_row(ax, residuals, float(yi), bcolor, row_height)

        # "Overall" row pooling all clean points
        overall_y = float(n_drawn)
        _draw_budget_row(
            ax, np.asarray(clean_edf["residual"]), overall_y, "#555555", row_height
        )
        # Separator line between budgets and overall
        ax.axhline(n_drawn - 0.5, color="grey", linestyle=":", linewidth=0.6, zorder=0)

        # Zero-residual reference line
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.7, zorder=0)

        # Axes
        labels = [_fmt_budget(b) for b, _ in drawn_budgets] + ["Overall"]
        ax.set_yticks(range(n_rows_plot))
        ax.set_yticklabels(
            labels,
            fontsize=max(6, 9 - n_drawn // 8),
        )
        ax.set_ylim(-0.6, n_rows_plot - 0.4)
        ax.set_xlabel("Residual: observed \u2212 predicted (nats)", fontsize=9)
        ax.set_ylabel("Compute budget (FLOPs)", fontsize=9)
        ax.grid(True, axis="x", alpha=0.2)

        # Title
        short = display_name(experiment)
        if fit is not None:
            s = fit.to_loss_surface()
            ax.set_title(
                f"{short}\n"
                rf"E={s.E:.3f}  $\alpha$={s.alpha:.3f}  $\beta$={s.beta:.3f}",
                fontsize=9,
            )
        else:
            ax.set_title(short, fontsize=9)

    # Hide unused axes
    for idx in range(n_exp, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # Figure-level legend
    legend_handles = [
        Line2D(
            [],
            [],
            color="grey",
            marker="|",
            linestyle="None",
            markersize=7,
            label="Rug",
        ),
        Line2D(
            [],
            [],
            color="grey",
            marker="D",
            linestyle="None",
            markersize=4,
            label="Mean",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(legend_handles),
        fontsize=9,
        frameon=True,
        borderpad=0.4,
    )

    if title is None:
        logloss_label = "log-loss" if use_logloss else "raw-loss"
        title = f"Approach 3 Residual Distributions by Compute Budget ({logloss_label})"
    fig.suptitle(title, fontsize=13)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_isoflop_curves(
    results: dict[str, dict],
    output_path: Path,
    *,
    title: str,
    method: str,
) -> None:
    """Plot isoflop curves with data, parabolic fits, and method predictions.

    Parameters
    ----------
    method : ``"approach3"`` or ``"akima"``
    """
    setup_style()

    experiments = ordered_experiments(results.keys())
    n_exp = len(experiments)
    n_cols = 4
    n_rows_grid = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows_grid,
        n_cols,
        figsize=(24, 5.5 * n_rows_grid),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)

    cmap = plt.colormaps["viridis"]

    for idx, experiment in enumerate(experiments):
        ax = axes[idx // n_cols, idx % n_cols]
        edf: pd.DataFrame = results[experiment]["df"]
        fit: SurfaceFitResult | None = results[experiment].get("fit")

        all_budgets = sorted(edf["budget"].unique(), reverse=True)
        n_all_budgets = len(all_budgets)
        all_colors = cmap(np.linspace(0.15, 0.85, n_all_budgets))

        for bi, budget in enumerate(all_budgets):
            color = all_colors[bi]
            bmask = edf["budget"] == budget
            bdf = edf.loc[bmask]

            N = bdf["params"].to_numpy()
            L = bdf["loss"].to_numpy()
            reasons = bdf["reason"].to_numpy()

            clean_mask = reasons == REASON_NONE
            N_clean = N[clean_mask]
            L_clean = L[clean_mask]

            # ── Data points ──────────────────────────────────────────
            if len(N_clean) > 0:
                ax.scatter(
                    N_clean,
                    L_clean,
                    s=20,
                    color=color,
                    marker="o",
                    zorder=5,
                    linewidths=0.5,
                    edgecolors=color,
                )

            _reason_marker_map = {
                REASON_EXACT_DUP: (
                    "s",
                    dict(facecolors="none", edgecolors=OUTLIER_COLOR),
                ),
                REASON_DUP_PARAMS: ("+", dict(color=OUTLIER_COLOR)),
                REASON_TOO_FEW: ("x", dict(color=OUTLIER_COLOR)),
                REASON_NEG_CURVATURE: (
                    "v",
                    dict(facecolors="none", edgecolors=OUTLIER_COLOR),
                ),
                REASON_WEAK_CURVATURE: (
                    "D",
                    dict(facecolors="none", edgecolors=OUTLIER_COLOR),
                ),
                REASON_SPLINE: (
                    "o",
                    dict(facecolors="none", edgecolors=OUTLIER_COLOR),
                ),
            }
            for reason, (marker, style) in _reason_marker_map.items():
                rmask = reasons == reason
                if rmask.any():
                    ax.scatter(
                        N[rmask],
                        L[rmask],
                        s=20,
                        marker=marker,
                        linewidths=0.9,
                        zorder=5,
                        **style,
                    )

            # ── Parabolic fit (clean points, log-log) ────────────────
            if len(N_clean) >= 3:
                log_n = np.log(N_clean)
                log_l = np.log(L_clean)
                coeffs = np.polyfit(log_n, log_l, 2)
                n_sweep = np.linspace(log_n.min(), log_n.max(), 200)
                ax.plot(
                    np.exp(n_sweep),
                    np.exp(np.polyval(coeffs, n_sweep)),
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=3,
                )
                # Budget annotation at geometric midpoint of the parabola
                n_mid = (log_n.min() + log_n.max()) / 2
                l_mid = np.exp(np.polyval(coeffs, n_mid))
                ax.annotate(
                    _fmt_budget(budget),
                    (np.exp(n_mid), l_mid),
                    textcoords="offset points",
                    xytext=(0, 6),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color="black",
                    alpha=0.8,
                )

            # ── Method prediction curve ──────────────────────────────
            if method == "approach3" and fit is not None and len(N_clean) >= 2:
                s = fit.to_loss_surface()
                n_range = np.logspace(
                    np.log10(N_clean.min()),
                    np.log10(N_clean.max()),
                    200,
                )
                d_range = budget / (6 * n_range)
                l_pred = s.loss(n_range, d_range)  # pyrefly: ignore
                ax.plot(
                    n_range,
                    l_pred,
                    color=color,
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=4,
                )
            elif method == "akima" and len(N_clean) >= 2:
                order = np.argsort(N_clean)
                log_n_sorted = np.log(N_clean[order])
                l_sorted = L_clean[order]
                if len(np.unique(log_n_sorted)) == len(log_n_sorted):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        interp = Akima1DInterpolator(log_n_sorted, l_sorted)
                    n_sweep = np.linspace(
                        log_n_sorted.min(),
                        log_n_sorted.max(),
                        200,
                    )
                    l_spline = interp(n_sweep)
                    # Detect spikes: spline should stay within a
                    # reasonable multiple of the data range
                    l_lo, l_hi = L_clean.min(), L_clean.max()
                    l_range = l_hi - l_lo if l_hi > l_lo else l_hi
                    spike_lo = l_lo - 2 * l_range
                    spike_hi = l_hi + 2 * l_range
                    if l_spline.min() < spike_lo or l_spline.max() > spike_hi:
                        short = display_name(experiment)
                        raise RuntimeError(
                            f"Akima spike detected in {short}, "
                            f"budget={budget:.2e}: spline range "
                            f"[{l_spline.min():.4f}, {l_spline.max():.4f}] "
                            f"exceeds data range "
                            f"[{l_lo:.4f}, {l_hi:.4f}] by >2x. "
                            f"Near-duplicate params likely survived binning."
                        )
                    ax.plot(
                        np.exp(n_sweep),
                        l_spline,
                        color=color,
                        linestyle="-",
                        linewidth=1.2,
                        alpha=0.8,
                        zorder=4,
                    )
            elif method == "flop_factor":
                surface = results[experiment].get("surface")
                k_values = results[experiment].get("k_values", {})
                k = k_values.get(budget)
                if surface is not None and k is not None and len(N_clean) >= 2:
                    n_range = np.logspace(
                        np.log10(N_clean.min()),
                        np.log10(N_clean.max()),
                        200,
                    )
                    d_range = budget / (k * n_range)
                    l_pred = surface.loss(n_range, d_range)
                    ax.plot(
                        n_range,
                        l_pred,
                        color=color,
                        linestyle="-",
                        linewidth=1.2,
                        alpha=0.8,
                        zorder=4,
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters (N)", fontsize=9)
        ax.set_ylabel("Loss (nats)", fontsize=9)
        ax.grid(True, alpha=0.2)

        short = display_name(experiment)
        if fit is not None:
            s = fit.to_loss_surface()
            ax.set_title(
                f"{short}\n"
                rf"E={s.E:.3f}  $\alpha$={s.alpha:.3f}  $\beta$={s.beta:.3f}",
                fontsize=9,
            )
        else:
            ax.set_title(short, fontsize=9)

    # Hide unused axes
    for idx in range(n_exp, n_rows_grid * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # Legend
    legend_handles = [
        Line2D(
            [],
            [],
            color="grey",
            marker="o",
            linestyle="None",
            markersize=5,
            markerfacecolor="grey",
            label="Clean",
        ),
        Line2D(
            [],
            [],
            color=OUTLIER_COLOR,
            marker="s",
            linestyle="None",
            markersize=7,
            markerfacecolor="none",
            label="Exact duplicate",
        ),
        Line2D(
            [],
            [],
            color=OUTLIER_COLOR,
            marker="+",
            linestyle="None",
            markersize=7,
            label="Near-duplicate params",
        ),
        Line2D(
            [],
            [],
            color=OUTLIER_COLOR,
            marker="x",
            linestyle="None",
            markersize=7,
            label="Too few points",
        ),
        Line2D(
            [],
            [],
            color=OUTLIER_COLOR,
            marker="v",
            linestyle="None",
            markersize=7,
            markerfacecolor="none",
            label="Negative curvature",
        ),
        Line2D(
            [],
            [],
            color=OUTLIER_COLOR,
            marker="d",
            linestyle="None",
            markersize=7,
            markerfacecolor="none",
            label="Weak curvature",
        ),
        Line2D(
            [],
            [],
            color=OUTLIER_COLOR,
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor="none",
            label="Spline outlier",
        ),
        Line2D(
            [], [], color="grey", linestyle="--", linewidth=1.0, label="Parabolic fit"
        ),
        Line2D([], [], color="grey", linestyle="-", linewidth=1.2, label="Prediction"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(legend_handles) // 2,
        fontsize=9,
        frameon=True,
        borderpad=0.4,
    )

    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_flop_factors(
    results: dict[str, dict],
    output_path: Path,
) -> None:
    """Plot per-budget FLOP factor k vs compute budget for each experiment."""
    setup_style()

    experiments = ordered_experiments(results.keys())
    n_exp = len(experiments)
    n_cols = 4
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(24, 5 * n_rows),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)

    for idx, experiment in enumerate(experiments):
        ax = axes[idx // n_cols, idx % n_cols]
        k_values: dict[float, float] = results[experiment].get("k_values", {})

        budgets = np.array(sorted(k_values.keys()))
        ks = np.array([k_values[b] for b in budgets])

        ax.plot(budgets, ks, "o-", color="steelblue", markersize=5, linewidth=1.2)
        ax.axhline(6.0, color="grey", linestyle="--", linewidth=0.8, label="k = 6")

        ax.set_xscale("log")
        ax.set_xlabel("Compute budget C (FLOPs)", fontsize=9)
        ax.set_ylabel("k  (C = k·N·D)", fontsize=9)
        ax.grid(True, alpha=0.2)

        short = display_name(experiment)
        if len(ks) > 0:
            ax.set_title(f"{short}\nmedian k = {np.median(ks):.3f}", fontsize=9)
        else:
            ax.set_title(short, fontsize=9)

    for idx in range(n_exp, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Per-Budget FLOP Factor k (C = k·N·D)", fontsize=13)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp12")
    df = pd.read_csv(ISOFLOPS_CSV)
    all_experiments = set(df["experiment"].unique())
    experiments = ordered_experiments(all_experiments)

    # Pre-compute outlier detection (independent of fit method)
    print("Pre-fit outlier detection")
    print("=" * 60)
    experiment_edfs: dict[str, pd.DataFrame] = {}
    for experiment in experiments:
        edf = df[df["experiment"] == experiment].copy()
        overrides = EXPERIMENT_OVERRIDES.get(experiment, {})
        edf = detect_outliers(edf, **overrides)
        clean = edf[~edf["outlier"]]
        print(f"\n  {experiment} ({len(edf)} points)")
        for reason in [
            REASON_EXACT_DUP,
            REASON_DUP_PARAMS,
            REASON_TOO_FEW,
            REASON_NEG_CURVATURE,
            REASON_WEAK_CURVATURE,
            REASON_SPLINE,
        ]:
            cnt = (edf["reason"] == reason).sum()
            if cnt > 0:
                print(f"    {reason}: {cnt}")
        print(f"    clean: {len(clean)}/{len(edf)} points")
        experiment_edfs[experiment] = edf

    # ── Approach 3 fits (raw-loss and log-loss) ──────────────────────────
    global_fits: dict[str, SurfaceFitResult] = {}  # from logloss=False
    for use_logloss in [False, True]:
        logloss_tag = "true" if use_logloss else "false"
        print(f"\n{'=' * 60}")
        print(f"Approach 3 with use_logloss={use_logloss}")
        print(f"{'=' * 60}")

        results: dict[str, dict] = {}
        for experiment in experiments:
            edf = experiment_edfs[experiment].copy()
            clean = edf[~edf["outlier"]]

            if len(clean) == 0:
                print(f"\n  {experiment}: no clean points, skipping")
                continue

            N_clean = np.asarray(clean["params"])
            D_clean = np.asarray(clean["tokens"])
            L_clean = np.asarray(clean["loss"])

            fit = _fit_experiment(N_clean, D_clean, L_clean, use_logloss=use_logloss)

            # Compute residuals for ALL points (including outliers)
            N_all = np.asarray(edf["params"])
            D_all = np.asarray(edf["tokens"])
            L_all = np.asarray(edf["loss"])
            residuals = _compute_residuals(N_all, D_all, L_all, fit)
            edf["residual"] = residuals

            s = fit.to_loss_surface()
            clean_resid = residuals[~edf["outlier"].to_numpy()]
            print(
                f"\n  {experiment}"
                f"\n    E={s.E:.4f}, A={s.A:.1f}, B={s.B:.1f}, "
                f"\u03b1={s.alpha:.4f}, \u03b2={s.beta:.4f}"
                f"\n    Residuals (clean): mean={clean_resid.mean():.6f}, "
                f"std={clean_resid.std():.6f}"
            )

            results[experiment] = {"fit": fit, "df": edf}
            if not use_logloss:
                global_fits[experiment] = fit

        plot_residual_distributions(
            results,
            output_dir / f"residuals_logloss_{logloss_tag}.png",
            use_logloss=use_logloss,
        )

        logloss_label = "log-loss" if use_logloss else "raw-loss"
        plot_isoflop_curves(
            results,
            output_dir / f"isoflops_approach3_logloss_{logloss_tag}.png",
            title=f"IsoFLOP Curves — Approach 3 ({logloss_label})",
            method="approach3",
        )

    # ── Akima isoflop curves (no residual plot) ────────────────────
    results_akima: dict[str, dict] = {}
    for experiment in experiments:
        edf = experiment_edfs[experiment].copy()
        if (~edf["outlier"]).any():
            results_akima[experiment] = {"df": edf}

    plot_isoflop_curves(
        results_akima,
        output_dir / "isoflops_akima.png",
        title="IsoFLOP Curves — Akima",
        method="akima",
    )

    # ── Per-budget FLOP factor (k) estimation ──────────────────────
    print(f"\n{'=' * 60}")
    print("Per-budget FLOP factor estimation (k in C = k·N·D)")
    print(f"{'=' * 60}")

    results_kfactor: dict[str, dict] = {}
    for experiment in experiments:
        edf = experiment_edfs[experiment].copy()
        clean = edf[~edf["outlier"]]

        if len(clean) == 0 or experiment not in global_fits:
            print(f"\n  {experiment}: no clean points or no global fit, skipping")
            continue

        surface = global_fits[experiment].to_loss_surface()
        predictions, k_values = _flop_factor_predictions(edf, surface)
        edf["residual"] = edf["loss"].to_numpy() - predictions

        valid_clean = ~np.isnan(predictions) & ~edf["outlier"].to_numpy()
        clean_resid = edf.loc[valid_clean, "residual"].to_numpy()
        print(f"\n  {experiment}")
        for budget in sorted(k_values):
            print(f"    C={budget:.2e}: k={k_values[budget]:.4f}")
        print(
            f"    Residuals (clean): mean={clean_resid.mean():.6f}, "
            f"std={clean_resid.std():.6f}"
        )

        results_kfactor[experiment] = {
            "df": edf,
            "surface": surface,
            "k_values": k_values,
        }

    plot_residual_distributions(
        results_kfactor,
        output_dir / "residuals_flop_factor.png",
        use_logloss=False,
        title="Per-Budget FLOP Factor Residuals (k in C = k·N·D)",
    )

    plot_isoflop_curves(
        results_kfactor,
        output_dir / "isoflops_flop_factor.png",
        title="Per-Budget FLOP Factor Curves (k in C = k·N·D)",
        method="flop_factor",
    )

    plot_flop_factors(
        results_kfactor,
        output_dir / "flop_factors.png",
    )


if __name__ == "__main__":
    main()
