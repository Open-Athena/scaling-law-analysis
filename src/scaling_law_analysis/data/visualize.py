"""Visualization for isoflop data."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

from scaling_law_analysis.data.schema import OutlierReason
from scaling_law_analysis.data.transform import display_name, ordered_experiments

OUTLIER_COLOR = "#d62728"


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


def fmt_budget(b: float) -> str:
    """Format a FLOP budget for annotation labels, e.g. 6e+18 -> 6e18."""
    return f"{b:.0e}".replace("e+0", "e").replace("e+", "e")


def plot_isoflops(
    df: pd.DataFrame,
    fits: dict[tuple[str, str, float], dict],
    out_path: Path,
) -> None:
    """Plot loss vs params for each (source, experiment), colored by budget.

    Draws fitted parabolas (valid only) over the data range at each budget
    and annotates the compute budget at the geometric midpoint of the param range.
    """
    # Build (source, experiment) pairs in canonical order
    pairs = (
        df[["source", "experiment"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    exp_to_source = {exp: src for src, exp in pairs}
    panel_exps = ordered_experiments(exp_to_source.keys())
    panel_keys = [(exp_to_source[e], e) for e in panel_exps]

    n_panels = len(panel_keys)
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, (source, experiment) in enumerate(panel_keys):
        ax = axes[idx // ncols][idx % ncols]
        edf = df[(df["source"] == source) & (df["experiment"] == experiment)]
        budgets = sorted(edf["budget"].unique())
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=np.log10(min(budgets)), vmax=np.log10(max(budgets)))

        for budget in budgets:
            bdf = edf[edf["budget"] == budget].sort_values("params")
            color = cmap(norm(np.log10(budget)))
            fit = fits.get((source, experiment, budget))

            # Circles for budgets with valid fits, x for invalid/missing
            marker = "o" if (fit and fit["valid"]) else "x"
            ms = 3 if marker == "o" else 5
            mew = None if marker == "o" else 1.2
            ax.plot(
                bdf["params"],
                bdf["loss"],
                marker,
                color=color,
                ms=ms,
                mew=mew,
                zorder=3,
            )

            # Parabola curve + budget annotation (valid fits only)
            # Draw over the param range of this panel's data, not the fit's
            # pooled range (which may span multiple sources).
            if fit and fit["valid"]:
                panel_n_min = bdf["params"].min()
                panel_n_max = bdf["params"].max()
                n_curve = np.logspace(
                    np.log10(panel_n_min),
                    np.log10(panel_n_max),
                    100,
                )
                l_curve = np.exp(np.polyval(fit["coeffs"], np.log(n_curve)))
                ax.plot(n_curve, l_curve, "-", color=color, lw=0.8, zorder=2)

                # Annotate budget at geometric midpoint
                n_mid = np.sqrt(panel_n_min * panel_n_max)
                l_mid = np.exp(np.polyval(fit["coeffs"], np.log(n_mid)))
                ax.annotate(
                    fmt_budget(budget),
                    (n_mid, l_mid),
                    textcoords="offset points",
                    xytext=(0, 6),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color="black",
                    alpha=0.8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Params (N)")
        ax.set_ylabel("Loss")
        ax.set_title(display_name(experiment), fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt_budget_short(budget: float) -> str:
    """Format a compute budget for annotations, e.g. '6e18'."""
    exp = int(np.floor(np.log10(budget)))
    mantissa = budget / 10**exp
    if abs(mantissa - round(mantissa)) < 0.05:
        return f"{round(mantissa):.0f}e{exp}"
    return f"{mantissa:.1f}e{exp}"


def plot_isoflops_akima(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot isoflop curves with Akima splines and outlier annotations.

    Reads an annotated DataFrame (with ``outlier`` and ``reason`` columns).
    For each experiment panel, draws:
    - Clean points as filled circles
    - Outlier points with reason-specific markers
    - Parabolic fit (dashed) on clean points per budget
    - Akima spline (solid) on clean points per budget
    - Budget annotation at geometric midpoint
    """
    setup_style()

    R = OutlierReason

    df = df.copy()
    df["reason"] = df["reason"].fillna(R.NONE)

    experiments = ordered_experiments(df["experiment"].unique())
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
        edf = df[df["experiment"] == experiment]

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

            clean_mask = reasons == R.NONE
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
                R.EXACT_DUP: (
                    "s",
                    dict(facecolors="none", edgecolors=OUTLIER_COLOR),
                ),
                R.DUP_PARAMS: ("+", dict(color=OUTLIER_COLOR)),
                R.TOO_FEW: ("x", dict(color=OUTLIER_COLOR)),
                R.NEG_CURVATURE: (
                    "v",
                    dict(facecolors="none", edgecolors=OUTLIER_COLOR),
                ),
                R.WEAK_CURVATURE: (
                    "D",
                    dict(facecolors="none", edgecolors=OUTLIER_COLOR),
                ),
                R.SPLINE: (
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
                    _fmt_budget_short(budget),
                    (np.exp(n_mid), l_mid),
                    textcoords="offset points",
                    xytext=(0, 6),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color="black",
                    alpha=0.8,
                )

            # ── Akima spline curve ───────────────────────────────────
            if len(N_clean) >= 2:
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

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters (N)", fontsize=9)
        ax.set_ylabel("Loss (nats)", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_title(display_name(experiment), fontsize=9)

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
        Line2D(
            [], [], color="grey", linestyle="-", linewidth=1.2, label="Akima spline"
        ),
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

    fig.suptitle("IsoFLOP Curves — Akima", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")
