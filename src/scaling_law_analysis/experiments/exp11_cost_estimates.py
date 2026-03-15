"""Experiment 11: Cost Estimates.

Estimates wasted compute and dollar costs from Approach 2 extrapolation errors
on published loss surfaces (Chinchilla, SODA, Sparse-NMM) and Llama 3 isoFLOP
data. Compares VPNLS and Approach 3 as ground truth against Approach 2 under
both log-scale and raw-nats interpretations of Llama 3 losses.

Usage:
    uv run python -m scaling_law_analysis.experiments.exp11_cost_estimates
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaling_law_analysis import config
from scaling_law_analysis.data.schema import OutlierReason, QCStage
from scaling_law_analysis.data.transform import DEFAULT_STAGES, STAGE_REASONS
from scaling_law_analysis.chinchilla import (
    ExponentGrid,
    LossSurface,
    ParameterGrid,
    ParabolaFitResult,
    fit_approach2,
    fit_approach3,
    fit_vpnls,
)
from scaling_law_analysis.experiments.common import (
    CHINCHILLA_SURFACE,
    SODA_SURFACE,
    SPARSE_NMM_SURFACE,
    prepare_output_dir,
)
from scaling_law_analysis.experiments.exp10_compounding_errors import (
    GRID_WIDTHS,
    H100_BF16_TFLOPS,
    MFU,
    GPU_COST_PER_HOUR,
    SurfaceList,
    compare_allocations,
    fit_and_compare,
    fmt_count,
    fmt_dollars,
    fmt_surface_result,
    plot,
    run,
    save_csv,
    setup_style,
)

# ── Llama 3 data ─────────────────────────────────────────────────────────────

from scaling_law_analysis.data.common import ISOFLOPS_CSV

LLAMA3_LOSS_LOG_SCALE = True  # if True, interpret validation_loss as ln(loss)

FIT_GRID = ParameterGrid(
    E=np.linspace(0.1, 5.0, 8),
    A=np.logspace(1, 6, 8),
    B=np.logspace(1, 6, 8),
    alpha=np.linspace(0.05, 0.95, 8),
    beta=np.linspace(0.05, 0.95, 8),
)

VPNLS_GRID = ExponentGrid(
    alpha=np.linspace(0.05, 0.95, 256),
    beta=np.linspace(0.05, 0.95, 256),
)


@dataclass(frozen=True)
class IsoFlopModelConfig:
    """Configuration for a model's isoFLOP progressive filter analysis."""

    name: str  # "Llama 3", "Chinchilla"
    experiment: str  # experiment name in isoflops CSV
    eval_budget: float  # FLOPs for extrapolation
    file_prefix: str  # "llama3", "chinchilla"


LLAMA3_MODEL = IsoFlopModelConfig(
    name="Llama 3",
    experiment="llama_3__raw_loss",
    eval_budget=3.8e25,
    file_prefix="llama3",
)

# Eval budget from Figure 2, arxiv:2203.15556
CHINCHILLA_MODEL = IsoFlopModelConfig(
    name="Chinchilla",
    experiment="epochai_chinchilla__massivetext__chinchilla",
    eval_budget=5.76e23,
    file_prefix="chinchilla",
)


FILTER_OUTLIERS = False

_LLAMA3_EXPERIMENT = {
    True: "llama_3__exp_loss",
    False: "llama_3__raw_loss",
}


def _load_isoflop_data(
    experiment: str,
    filter_outliers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load isoFLOP data from the data pipeline CSV.

    Returns (N, D, L, C) arrays.
    """
    df = pd.read_csv(ISOFLOPS_CSV)
    edf = df[df["experiment"] == experiment]
    if filter_outliers:
        edf = edf[~edf["outlier"]]
    N = edf["params"].to_numpy()
    D = edf["tokens"].to_numpy()
    L = edf["loss"].to_numpy()
    C = edf["budget"].to_numpy()
    return N, D, L, C


def _fit_vpnls(
    name: str,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
) -> LossSurface:
    """Fit isoFLOP data via VPNLS with L-BFGS-B."""
    result = fit_vpnls(N, D, L, grid=VPNLS_GRID, method="l-bfgs-b", use_grad=True)
    print(
        f"{name} VPNLS: E={result.E:.4f}, A={result.A:.1f}, B={result.B:.1f}, "
        f"α={result.alpha:.4f}, β={result.beta:.4f} "
        f"(RSS={result.residual_sum_squares:.6f})"
    )
    return LossSurface(
        alpha=result.alpha, beta=result.beta, A=result.A, B=result.B, E=result.E
    )


def _fit_a3(
    name: str,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
) -> LossSurface:
    """Fit a loss surface to isoFLOP data via Approach 3."""
    result = fit_approach3(
        N, D, L, grid=FIT_GRID, use_lse=True, use_logloss=True, use_grad=True
    )
    print(
        f"{name} Approach 3: E={result.E:.4f}, A={result.A:.1f}, B={result.B:.1f}, "
        f"α={result.alpha:.4f}, β={result.beta:.4f} "
        f"(RSS={result.residual_sum_squares:.6f})"
    )
    return LossSurface(
        alpha=result.alpha, beta=result.beta, A=result.A, B=result.B, E=result.E
    )


def _fit_a2(
    name: str,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
) -> ParabolaFitResult:
    """Fit isoFLOP data via Approach 2."""
    result = fit_approach2(N, D, L, C)
    print(
        f"{name} Approach 2: a={result.a:.4f}, b={result.b:.4f} "
        f"(N* ∝ C^a, D* ∝ C^b)"
    )
    return result


# ── Progressive filter analysis ──────────────────────────────────────────────

# Grouped filter stages: (group_label, list of OutlierReasons collapsed into one level).
# Order follows DEFAULT_STAGES.
_FILTER_GROUPS: list[tuple[str, list[OutlierReason]]] = [
    (
        "Pre-QC",
        [r for stage in (QCStage.DEDUP, QCStage.TOO_FEW) for r in STAGE_REASONS[stage]],
    ),
    (
        "Off Center",
        [
            r
            for stage in (QCStage.OFF_CENTER, QCStage.SPLINE)
            for r in STAGE_REASONS[stage]
        ],
    ),
    (
        "Weak Curvature",
        [r for r in STAGE_REASONS[QCStage.CURVATURE]],
    ),
    (
        "Post-QC",
        [r for r in STAGE_REASONS[QCStage.POST_QC]],
    ),
]

# Verify all QCStages are covered by _FILTER_GROUPS.
_covered_stages = {
    stage
    for _, reasons in _FILTER_GROUPS
    for stage in QCStage
    if any(r in reasons for r in STAGE_REASONS[stage])
}
_missing_stages = set(QCStage) - _covered_stages
if _missing_stages:
    raise RuntimeError(
        f"_FILTER_GROUPS is missing QCStage(s): {sorted(_missing_stages)}"
    )


@dataclass
class _FilterLevel:
    """One level in the progressive filter analysis."""

    label: str
    dcl_flops: float
    pts_removed: int
    pts_remaining: int
    budgets_removed: int
    budgets_remaining: int


def _progressive_filter_dcl(
    surface: LossSurface,
    eval_budget: float,
    experiment: str,
) -> list[_FilterLevel]:
    """Compute DCL and data statistics at each cumulative filter level.

    *surface* is the ground-truth surface (A3 or VPNLS, fit on raw data).
    Returns list of _FilterLevel starting with "Raw" (no filter).
    """
    df = pd.read_csv(ISOFLOPS_CSV)
    edf = df[df["experiment"] == experiment].copy()

    results: list[_FilterLevel] = []
    total_pts = len(edf)
    total_budgets = edf["budget"].nunique()

    # Level 0: no filtering
    a2 = fit_approach2(
        edf["params"].to_numpy(),
        edf["tokens"].to_numpy(),
        edf["loss"].to_numpy(),
        edf["budget"].to_numpy(),
    )
    d = compare_allocations(surface, a2, eval_budget)
    results.append(
        _FilterLevel(
            label="Raw",
            dcl_flops=d["wasted_flops"],
            pts_removed=0,
            pts_remaining=total_pts,
            budgets_removed=0,
            budgets_remaining=total_budgets,
        )
    )

    # Cumulative filter levels (grouped)
    excluded: set[str] = set()
    prev_pts = total_pts
    prev_budgets = total_budgets
    for label, reasons in _FILTER_GROUPS:
        for reason in reasons:
            excluded.add(reason.value)
        filtered = edf[~edf["reason"].isin(excluded)]
        cur_pts = len(filtered)
        cur_budgets = filtered["budget"].nunique() if cur_pts > 0 else 0
        if cur_pts == 0:
            break
        a2 = fit_approach2(
            filtered["params"].to_numpy(),
            filtered["tokens"].to_numpy(),
            filtered["loss"].to_numpy(),
            filtered["budget"].to_numpy(),
        )
        d = compare_allocations(surface, a2, eval_budget)
        results.append(
            _FilterLevel(
                label=f"+{label}",
                dcl_flops=d["wasted_flops"],
                pts_removed=prev_pts - cur_pts,
                pts_remaining=cur_pts,
                budgets_removed=prev_budgets - cur_budgets,
                budgets_remaining=cur_budgets,
            )
        )
        prev_pts = cur_pts
        prev_budgets = cur_budgets

    return results


def plot_progressive_filter(
    results: list[_FilterLevel],
    output_path: str | Path,
    title: str,
    eval_budget: float,
    show_convergence_annotation: bool = True,
) -> None:
    """Progressive filter: horizontal bar + dot-line chart with detail table."""
    setup_style()
    plt.rcParams["text.usetex"] = False

    n_rows = len(results)
    y_pos = np.arange(n_rows)

    dcl_dollars = [_flops_to_dollars(r.dcl_flops) for r in results]
    dcl_pcts = [r.dcl_flops / eval_budget * 100 for r in results]

    fig_height = max(0.45 * n_rows + 1.2, 3.5)
    fig, (ax_bar, ax_tbl) = plt.subplots(
        1,
        2,
        figsize=(12, fig_height),
        gridspec_kw={"width_ratios": [6, 4], "wspace": 0.02},
        sharey=True,
        layout="constrained",
    )

    # ── Left panel: thin bars (background) + dot-line (foreground) ──
    ax_bar.barh(
        y_pos,
        dcl_pcts,
        color="#d0d0d0",
        edgecolor="white",
        linewidth=0.8,
        height=0.15,
        zorder=1,
    )
    ax_bar.plot(
        dcl_pcts,
        y_pos,
        "o-",
        color="#333333",
        markersize=7,
        linewidth=1.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
        label="Approach 2",
    )

    # Approach 3 reference line at DCL=0%
    ax_bar.plot(
        [0] * n_rows,
        y_pos,
        "s-",
        color="#333333",
        markersize=5,
        linewidth=1.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=2,
        label="Approach 3",
    )

    # $ annotations to the right of Approach 2 dots
    for idx, (pct, dollars) in enumerate(zip(dcl_pcts, dcl_dollars)):
        if results[idx].label == "+Weak Curvature" and pct < 1.0:
            # Place to the left of the Approach 3 line
            x_off, y_off, ha = -6, 4, "right"
        elif results[idx].label == "+Off Center":
            # Shift down to avoid overlap with neighboring annotations
            x_off, y_off, ha = 6, -2, "left"
        else:
            x_off, y_off, ha = 6, 4, "left"
        ax_bar.annotate(
            fmt_dollars(dollars),
            xy=(pct, y_pos[idx]),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
            va="top",
            ha=ha,
            zorder=4,
        )

    max_val = max(dcl_pcts)
    ax_bar.set_xlim(-max_val * 0.15, max_val * 1.6)

    # Tick labels with brief descriptions
    _GROUP_DESCRIPTIONS: dict[str, str] = {
        "Raw": "unfiltered",
        "+Pre-QC": "dedup & min points",
        "+Off Center": "geometric outliers",
        "+Weak Curvature": "flat/inverted curves",
        "+Post-QC": "sparse curve cleanup",
    }
    tick_labels = []
    for r in results:
        desc = _GROUP_DESCRIPTIONS.get(r.label, "")
        tick_labels.append(f"{r.label}\n{desc}" if desc else r.label)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(tick_labels, fontsize=9)
    ax_bar.set_xlabel("Deadweight Compute Loss (%)", fontsize=10)
    ax_bar.set_ylabel("Approach 2 QC Pipeline", fontsize=10)
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.invert_yaxis()

    ax_bar.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Convergence annotation on last row
    if show_convergence_annotation:
        last_y = y_pos[-1]
        eval_exp = int(np.floor(np.log10(eval_budget)))
        eval_mantissa = eval_budget / 10**eval_exp
        ax_bar.annotate(
            "QC bias corrections yield nearly convergent\n"
            "compute-optimal estimates at "
            rf"${eval_mantissa:.1f} \times 10^{{{eval_exp}}}$ FLOPs",
            xy=(max_val * 0.12, last_y),
            xytext=(max_val * 0.35, last_y),
            fontsize=8.5,
            color="#555555",
            fontstyle="italic",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0),
            zorder=4,
        )

    # ── Right panel: detail table ──
    headers = ["\u0394 points", "points", "\u0394 budgets", "budgets", "DCL %", "DCL $"]
    grad_cols = {4, 5}
    n_cols = len(headers)

    cell_values: list[list[str]] = []
    for i, r in enumerate(results):
        if r.label == "Raw":
            delta_pts = "\u2014"
            delta_bdg = "\u2014"
        else:
            delta_pts = f"\u2212{r.pts_removed}" if r.pts_removed > 0 else "0"
            delta_bdg = f"\u2212{r.budgets_removed}" if r.budgets_removed > 0 else "0"
        cell_values.append(
            [
                delta_pts,
                str(r.pts_remaining),
                delta_bdg,
                str(r.budgets_remaining),
                f"{dcl_pcts[i]:.2f}%",
                _fmt_dollars_2dp(dcl_dollars[i]),
            ]
        )

    vmin, vmax = min(dcl_pcts), max(dcl_pcts)

    ax_tbl.set_xlim(-0.15, n_cols + 0.3)
    ax_tbl.set_ylim(n_rows - 0.5, -0.5)
    ax_tbl.set_axis_off()

    ax_tbl.text(
        0.5,
        -0.02,
        "Filter Details",
        transform=ax_tbl.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    for j, header in enumerate(headers):
        ax_tbl.text(
            j + 0.5,
            -0.7,
            header,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Alternating row backgrounds
    for idx in range(n_rows):
        if idx % 2 == 0:
            ax_tbl.add_patch(
                plt.Rectangle(
                    (0, idx - 0.5),
                    n_cols,
                    1,
                    facecolor="#f0f0f0",
                    edgecolor="none",
                    zorder=0,
                )
            )

    # Gradient cell backgrounds for DCL columns
    for j in grad_cols:
        for idx in range(n_rows):
            bg = _gradient_color(dcl_pcts[idx], vmin, vmax)
            ax_tbl.add_patch(
                plt.Rectangle(
                    (j, idx - 0.5),
                    1,
                    1,
                    facecolor=bg,
                    edgecolor="none",
                    zorder=1,
                )
            )

    # Cell values
    for idx in range(n_rows):
        for j in range(n_cols):
            if j in grad_cols:
                bg = _gradient_color(dcl_pcts[idx], vmin, vmax)
                color = _text_color_for_bg(bg)
                fontweight = "bold"
            else:
                color = "#000000"
                fontweight = "normal"
            ax_tbl.text(
                j + 0.5,
                idx,
                cell_values[idx][j],
                ha="center",
                va="center",
                fontsize=8,
                fontweight=fontweight,
                color=color,
                zorder=2,
            )

    # Table border
    ax_tbl.add_patch(
        plt.Rectangle(
            (0, -0.5),
            n_cols,
            n_rows,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
    )

    # Vertical divider between count columns and DCL columns
    first_grad = min(grad_cols)
    ax_tbl.plot(
        [first_grad, first_grad],
        [-0.5, n_rows - 0.5],
        color="black",
        linewidth=1.0,
        zorder=3,
    )

    if show_convergence_annotation:
        fig.suptitle(title, fontsize=12)
    else:
        eval_exp = int(np.floor(np.log10(eval_budget)))
        eval_mantissa = eval_budget / 10**eval_exp
        eval_latex = rf"${eval_mantissa:.1f} \times 10^{{{eval_exp}}}$"
        fig.suptitle(
            f"{title}\n(Extrapolated to {eval_latex} FLOPs)",
            fontsize=12,
        )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _flops_to_dollars(flops: float) -> float:
    """Convert FLOPs to dollar cost using H100 assumptions."""
    flops_per_second = H100_BF16_TFLOPS * 1e12 * MFU
    seconds = flops / flops_per_second
    hours = seconds / 3600
    return hours * GPU_COST_PER_HOUR


def _fmt_dollars_2dp(dollars: float) -> str:
    """Format a dollar amount with 1 decimal place for table display."""
    v = abs(dollars)
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.1f}K"
    if v >= 1:
        return f"${v:.1f}"
    return f"${v:.1f}"


# ── Configuration ────────────────────────────────────────────────────────────

COST_REPORT_SURFACES: SurfaceList = [
    ("Chinchilla", CHINCHILLA_SURFACE),
    ("SODA", SODA_SURFACE),
    ("Sparse-NMM", SPARSE_NMM_SURFACE),
]

TRAINING_BUDGETS = np.array([1e18, 1e19, 1e20, 1e21, 1e22])
LLAMA3_405B_FLOPS = 3.8e25

REPORT_DRIFT_RATE = np.log10(3)  # drift to 3×
REPORT_GRID = "XS (±2×)"

Llama3FitSet = tuple[bool, LossSurface, LossSurface, ParabolaFitResult]


# ── DCL summary figure ───────────────────────────────────────────────────────


@dataclass
class DCLRow:
    """One row in the DCL summary figure."""

    label: str
    group: str  # "real" or "simulated"
    d: dict  # from compare_allocations / fit_and_compare
    surface: LossSurface
    eval_budget: float
    ba_ratio: float  # b/a from Approach 3 fit (α/(α+β)) / (β/(α+β)) = α/β


def _collect_dcl_rows(
    eval_budget: float,
    training_budgets: np.ndarray,
    surfaces: SurfaceList,
    llama3_comparisons: list[Llama3FitSet],
) -> list[DCLRow]:
    """Collect all DCL rows for the summary figure."""
    _, log_range = next(g for g in GRID_WIDTHS if g[0] == REPORT_GRID)
    rows: list[DCLRow] = []

    # Real Llama 3 fits (VPNLS/A3 × log-scale/raw-nats)
    # Use b/a from Approach 3 for all real rows in each scale group.
    for log_scale, vpnls_surface, a3_surface, llama3_a2 in llama3_comparisons:
        scale_tag = "log" if log_scale else "raw"
        a3_ba = a3_surface.b / a3_surface.a
        for method_label, surface in [
            ("VPNLS", vpnls_surface),
            ("Approach 3", a3_surface),
        ]:
            d = compare_allocations(surface, llama3_a2, eval_budget)
            rows.append(
                DCLRow(
                    label=f"Llama 3: {method_label} ({scale_tag})",
                    group="real",
                    d=d,
                    surface=surface,
                    eval_budget=eval_budget,
                    ba_ratio=a3_ba,
                )
            )

    # Simulated fits
    for sname, surface in surfaces:
        d = fit_and_compare(
            surface=surface,
            eval_budget=eval_budget,
            training_budgets=training_budgets,
            log_range=log_range,
            center_scale=1.0,
            drift_rate=REPORT_DRIFT_RATE,
        )
        rows.append(
            DCLRow(
                label=f"{sname} (sim.)",
                group="simulated",
                d=d,
                surface=surface,
                eval_budget=eval_budget,
                ba_ratio=surface.b / surface.a,
            )
        )

    return rows


def _fmt_flops_latex(flops: float) -> str:
    """Format a FLOP count as a LaTeX string like $3.93 \\times 10^{24}$."""
    exp = int(np.floor(np.log10(abs(flops))))
    mantissa = flops / 10**exp
    return rf"${mantissa:.2f} \times 10^{{{exp}}}$"


def _fmt_tokens(value: float) -> str:
    """Format token count with 1 decimal place."""
    v = abs(value)
    if v >= 1e12:
        return f"{value / 1e12:.1f}T"
    if v >= 1e9:
        return f"{value / 1e9:.1f}B"
    return fmt_count(value)


def _fmt_params(value: float) -> str:
    """Format parameter count with 0 decimal places."""
    v = abs(value)
    if v >= 1e12:
        return f"{value / 1e12:.0f}T"
    if v >= 1e9:
        return f"{value / 1e9:.0f}B"
    return fmt_count(value)


def _gradient_color(val: float, vmin: float, vmax: float) -> str:
    """Map a value to a white→dark grey gradient. Returns hex color."""
    if vmax == vmin:
        return "#ffffff"
    t = (val - vmin) / (vmax - vmin)
    # White (#ffffff) to dark grey (#404040)
    r = int(255 - t * (255 - 64))
    g = int(255 - t * (255 - 64))
    b = int(255 - t * (255 - 64))
    return f"#{r:02x}{g:02x}{b:02x}"


def _text_color_for_bg(bg_hex: str) -> str:
    """Return black or white text depending on background luminance."""
    r = int(bg_hex[1:3], 16)
    g = int(bg_hex[3:5], 16)
    b = int(bg_hex[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 140 else "#ffffff"


def plot_dcl_summary(
    rows: list[DCLRow],
    output_path: str | Path,
    eval_budget: float,
) -> None:
    """DCL summary: horizontal bar chart (left) + detail table (right)."""
    setup_style()
    plt.rcParams["text.usetex"] = False  # use mathtext, not system LaTeX

    # Split and sort within groups (largest DCL first)
    real = sorted(
        [r for r in rows if r.group == "real"],
        key=lambda r: -r.d["wasted_flops"],
    )
    sim = sorted(
        [r for r in rows if r.group == "simulated"],
        key=lambda r: -r.d["wasted_flops"],
    )
    ordered = real + sim
    n_real = len(real)
    n_rows = len(ordered)

    y_pos = np.arange(n_rows)
    fig_height = max(0.45 * n_rows + 1.5, 3.8)
    fig, (ax_bar, ax_tbl) = plt.subplots(
        1,
        2,
        figsize=(12, fig_height),
        gridspec_kw={"width_ratios": [4, 5], "wspace": 0.02},
        sharey=True,
        layout="constrained",
    )

    # ── Left panel: horizontal bar chart ──
    dcl_vals = [r.d["wasted_flops"] for r in ordered]
    # Shades of grey: darker for real, lighter for simulated
    colors = ["#333333"] * n_real + ["#999999"] * len(sim)
    ax_bar.barh(
        y_pos,
        dcl_vals,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.6,
    )

    # Set x-axis limits with padding before drawing labels
    max_val = max(dcl_vals)
    min_val = min(dcl_vals)
    ax_bar.set_xscale("log")
    ax_bar.set_xlim(min_val * 0.3, max_val * 3)

    # Value labels — inside bar for large bars, just outside for small
    threshold = max_val * 0.05  # bars < 5% of max get outside labels
    for idx, val in enumerate(dcl_vals):
        if val > threshold:
            ax_bar.text(
                val * 0.85,
                y_pos[idx],
                _fmt_flops_latex(val),
                va="center",
                ha="right",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        else:
            ax_bar.text(
                val * 1.08,
                y_pos[idx],
                _fmt_flops_latex(val),
                va="center",
                ha="left",
                fontsize=8,
                color="#333333",
                fontweight="bold",
            )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([r.label for r in ordered], fontsize=9)
    ax_bar.set_xlabel("Deadweight Compute Loss (FLOPs)", fontsize=10)
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.invert_yaxis()

    # Dashed separator between real and simulated
    if n_real > 0 and len(sim) > 0:
        sep_y = n_real - 0.5
        ax_bar.axhline(sep_y, color="gray", linestyle="--", linewidth=1)
        xhi = ax_bar.get_xlim()[1]
        ax_bar.text(
            xhi,
            sep_y - 0.08,
            "empirical \u2191 ",
            fontsize=7,
            color="gray",
            ha="right",
            va="bottom",
        )
        ax_bar.text(
            xhi,
            sep_y + 0.08,
            "simulated \u2193 ",
            fontsize=7,
            color="gray",
            ha="right",
            va="top",
        )

    # ── Right panel: detail table ──
    # Column definitions: (header, format_fn, gradient_key_or_None)
    # gradient_key links columns that share the same gradient scale
    col_defs = [
        ("D* true", lambda r: _fmt_tokens(r.d["true_D"]), None),
        ("D* inf.", lambda r: _fmt_tokens(r.d["inferred_D"]), None),
        ("N* true", lambda r: _fmt_params(r.d["true_N"]), None),
        ("N* inf.", lambda r: _fmt_params(r.d["inferred_N"]), None),
        (
            "\u0394Loss (\u00d710\u00b3)",
            lambda r: f"+{r.d['loss_error_nats'] * 1e3:.1f}",
            None,
        ),
        (
            "DCL %",
            lambda r: f"{r.d['wasted_flops'] / r.eval_budget * 100:.1f}%",
            "dcl",
        ),
        ("DCL $", lambda r: fmt_dollars(r.d["wasted_dollars"]), "dcl"),
        ("b/a", lambda r: f"{r.ba_ratio:.2f}", "ba"),
    ]

    # Precompute gradient ranges
    dcl_pcts = [r.d["wasted_flops"] / r.eval_budget * 100 for r in ordered]
    ba_vals = [r.ba_ratio for r in ordered]
    grad_ranges = {
        "dcl": (min(dcl_pcts), max(dcl_pcts)),
        "ba": (min(ba_vals), max(ba_vals)),
    }
    # Raw values for gradient mapping (by gradient key)
    grad_row_vals = {
        "dcl": dcl_pcts,
        "ba": ba_vals,
    }

    n_cols = len(col_defs)
    ax_tbl.set_xlim(-0.15, n_cols + 0.3)
    ax_tbl.set_ylim(n_rows - 0.5, -0.5)
    ax_tbl.set_axis_off()

    # Table label (aligned with bar chart x-axis label)
    ax_tbl.text(
        0.5,
        -0.02,
        "Allocation Details",
        transform=ax_tbl.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    # Column headers
    for j, (header, _, _) in enumerate(col_defs):
        ax_tbl.text(
            j + 0.5,
            -0.7,
            header,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Alternating row backgrounds (base layer)
    for idx in range(n_rows):
        if idx % 2 == 0:
            ax_tbl.add_patch(
                plt.Rectangle(
                    (0, idx - 0.5),
                    n_cols,
                    1,
                    facecolor="#f0f0f0",
                    edgecolor="none",
                    zorder=0,
                )
            )

    # Gradient cell backgrounds
    for j, (_, _, grad_key) in enumerate(col_defs):
        if grad_key is None:
            continue
        vmin, vmax = grad_ranges[grad_key]
        for idx in range(n_rows):
            val = grad_row_vals[grad_key][idx]
            bg = _gradient_color(val, vmin, vmax)
            ax_tbl.add_patch(
                plt.Rectangle(
                    (j, idx - 0.5),
                    1,
                    1,
                    facecolor=bg,
                    edgecolor="none",
                    zorder=1,
                )
            )

    # Cell values
    for idx, row in enumerate(ordered):
        for j, (_, fmt_fn, grad_key) in enumerate(col_defs):
            if grad_key is not None:
                vmin, vmax = grad_ranges[grad_key]
                val = grad_row_vals[grad_key][idx]
                bg = _gradient_color(val, vmin, vmax)
                color = _text_color_for_bg(bg)
                fontweight = "bold"
            else:
                color = "#000000"
                fontweight = "normal"
            ax_tbl.text(
                j + 0.5,
                idx,
                fmt_fn(row),
                ha="center",
                va="center",
                fontsize=8,
                fontweight=fontweight,
                color=color,
                zorder=2,
            )

    # Table border
    ax_tbl.add_patch(
        plt.Rectangle(
            (0, -0.5),
            n_cols,
            n_rows,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
    )

    # Vertical divider between non-gradient and gradient columns
    first_grad_col = next(j for j, (_, _, gk) in enumerate(col_defs) if gk is not None)
    ax_tbl.plot(
        [first_grad_col, first_grad_col],
        [-0.5, n_rows - 0.5],
        color="black",
        linewidth=1.0,
        zorder=3,
    )

    # Separator line in table too
    if n_real > 0 and len(sim) > 0:
        ax_tbl.plot(
            [0, n_cols],
            [n_real - 0.5, n_real - 0.5],
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=3,
        )

    eval_exp = int(np.floor(np.log10(eval_budget)))
    eval_mantissa = eval_budget / 10**eval_exp
    eval_latex = rf"${eval_mantissa:.1f} \times 10^{{{eval_exp}}}$"
    fig.suptitle(
        "Deadweight Compute Loss: Approach 2 Misallocation at Frontier Scale\n"
        f"(Extrapolated to {eval_latex} FLOPs)",
        fontsize=12,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Report ───────────────────────────────────────────────────────────────────


def save_report(
    output_path: str | Path,
    eval_budget: float,
    training_budgets: np.ndarray,
    surfaces: SurfaceList,
    llama3_comparisons: list[Llama3FitSet],
) -> None:
    """Write cost estimate report with simulated and Llama 3 comparisons."""
    _, log_range = next(g for g in GRID_WIDTHS if g[0] == REPORT_GRID)
    drift_scale = 10**REPORT_DRIFT_RATE
    drift_label = (
        f"Drift to {drift_scale:.1f}×"
        if drift_scale < 1
        else f"Drift to {drift_scale:.0f}×"
    )

    lines: list[str] = []
    w = lines.append

    # ── Section 1: Simulated compounding errors ──
    w("Cost Estimates")
    w("=" * 60)
    w("")
    lo = int(np.log10(training_budgets[0]))
    hi = int(np.log10(training_budgets[-1]))
    w(
        f"Approach 2 fit on 10^{lo}-10^{hi} FLOPs, extrapolated to "
        f"{eval_budget:.1e} FLOPs."
    )
    w(f"Sampling: {REPORT_GRID} grid, {drift_label} bias.")
    w("Loss: L(N,D) = E + A/N^α + B/D^β (nats, base e).")
    w("")
    w(
        "Cost basis: H100 SXM bf16 ({} TFLOPS), {}% MFU, ${}/GPU-hr".format(
            H100_BF16_TFLOPS, int(MFU * 100), GPU_COST_PER_HOUR
        )
    )
    w("  TFLOPS: https://www.nvidia.com/en-us/data-center/h100")
    w("  MFU:    arxiv:2401.00448")
    w("  $/hr:   arxiv:2512.13961")
    w("")
    w("'over' = inferred exceeds true optimum; 'under' = falls short.")
    w("")
    w("Given inferred D*, actual N is forced by the budget: N = C/(6D).")
    w("The resulting loss L(N,D) is suboptimal. Deadweight Compute Loss")
    w("(DCL) inverts L_opt(C) to find the smaller budget that would reach")
    w("the same loss with optimal allocation; DCL = budget − equivalent_C.")
    w("")

    for sname, surface in surfaces:
        d = fit_and_compare(
            surface=surface,
            eval_budget=eval_budget,
            training_budgets=training_budgets,
            log_range=log_range,
            center_scale=1.0,
            drift_rate=REPORT_DRIFT_RATE,
        )
        w(
            f"{sname} (E={surface.E:.4f}, A={surface.A:.1f}, B={surface.B:.1f}, "
            f"α={surface.alpha:.4f}, β={surface.beta:.4f}, "
            f"a={surface.a:.4f}, b={surface.b:.4f})"
        )
        w("-" * 60)
        w(fmt_surface_result(d, eval_budget))
        w("")

    # ── Section 2: Llama 3 fits comparison ──
    w("")
    w("Llama 3: Approach 2 vs Parametric Fits on Real IsoFLOP Data")
    w("=" * 60)
    w("")
    # Use first entry to get n_curves (same data for all).
    n_curves = len(llama3_comparisons[0][3].parabola_fits_N)
    w("Each parametric fit is treated as ground truth in turn;")
    w("Approach 2 fit provides the inferred D* allocation,")
    w("extrapolated to the same budget.")
    w(f"Training data: {n_curves} isoFLOP curves from 6×10^18 to 10^22 FLOPs.")
    w("IsoFLOP points digitized from Llama 3 paper (Fig. 2), which plots")
    w("negative log-likelihood on a log-scale y-axis.")

    for log_scale, vpnls_surface, a3_surface, llama3_a2 in llama3_comparisons:
        if log_scale:
            scale_label = "log-scale (exp transform)"
        else:
            scale_label = "raw nats (no transform)"
        w("")
        w(f"── {scale_label} ──")
        w("")
        w(
            f"VPNLS surface (E={vpnls_surface.E:.4f}, "
            f"A={vpnls_surface.A:.1f}, B={vpnls_surface.B:.1f}, "
            f"α={vpnls_surface.alpha:.4f}, β={vpnls_surface.beta:.4f}, "
            f"a={vpnls_surface.a:.4f}, b={vpnls_surface.b:.4f})"
        )
        w(
            f"Approach 3 surface (E={a3_surface.E:.4f}, "
            f"A={a3_surface.A:.1f}, B={a3_surface.B:.1f}, "
            f"α={a3_surface.alpha:.4f}, β={a3_surface.beta:.4f}, "
            f"a={a3_surface.a:.4f}, b={a3_surface.b:.4f})"
        )
        w(
            f"Approach 2 power laws: a={llama3_a2.a:.4f}, b={llama3_a2.b:.4f} "
            f"(N* ∝ C^a, D* ∝ C^b)"
        )
        w("")
        for label, surface in [
            ("VPNLS", vpnls_surface),
            ("Approach 3", a3_surface),
        ]:
            d = compare_allocations(surface, llama3_a2, eval_budget)
            w(f"Approach 2 vs {label}:")
            w("-" * 60)
            w(fmt_surface_result(d, eval_budget))
            w("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


def _run_progressive_filter(model: IsoFlopModelConfig, output_dir: Path) -> None:
    """Fit A3/VPNLS on raw data and run progressive filter analysis for *model*."""
    N, D, L, _ = _load_isoflop_data(model.experiment)
    raw_a3 = _fit_a3(model.name, N, D, L)
    raw_vpnls = _fit_vpnls(model.name, N, D, L)

    for method_label, surface, suffix in [
        ("A3", raw_a3, "a3"),
        ("VPNLS", raw_vpnls, "vpnls"),
    ]:
        print(f"\n── Progressive filter ({model.name}): DCL vs {method_label} ──")
        results = _progressive_filter_dcl(surface, model.eval_budget, model.experiment)
        for r in results:
            print(
                f"  {r.label:>12s}: DCL={r.dcl_flops:.2e} "
                f"({r.dcl_flops / model.eval_budget * 100:.1f}%)"
            )
        plot_progressive_filter(
            results,
            output_dir / f"progressive_filter_{model.file_prefix}_{suffix}.png",
            f"Deadweight Compute Loss: Approach 2 vs {method_label} "
            f"Progressive Filtering ({model.name})",
            model.eval_budget,
            show_convergence_annotation=(model is LLAMA3_MODEL),
        )


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp11")

    # Fit Llama 3 isoFLOP data under both log-scale assumptions.
    llama3_comparisons: list[Llama3FitSet] = []
    primary_vpnls: LossSurface | None = None
    for log_scale in [True, False]:
        label = "log-scale" if log_scale else "raw nats"
        print(f"\n── Llama 3 fits ({label}) ──")
        experiment = _LLAMA3_EXPERIMENT[log_scale]
        N, D, L, C = _load_isoflop_data(experiment, filter_outliers=FILTER_OUTLIERS)
        vpnls = _fit_vpnls("Llama 3", N, D, L)
        a3 = _fit_a3("Llama 3", N, D, L)
        a2 = _fit_a2("Llama 3", N, D, L, C)
        llama3_comparisons.append((log_scale, vpnls, a3, a2))
        if log_scale == LLAMA3_LOSS_LOG_SCALE:
            primary_vpnls = vpnls
    assert primary_vpnls is not None
    surfaces: SurfaceList = [("Llama 3", primary_vpnls)] + COST_REPORT_SURFACES

    results = run(
        eval_budget=LLAMA3_405B_FLOPS,
        training_budgets=TRAINING_BUDGETS,
        surfaces=surfaces,
    )
    plot(
        results,
        output_dir / "token_allocation_errors.png",
        eval_budget=LLAMA3_405B_FLOPS,
        training_budgets=TRAINING_BUDGETS,
        surfaces=surfaces,
    )
    save_csv(
        results,
        output_dir / "token_allocation_errors.csv",
        eval_budget=LLAMA3_405B_FLOPS,
        training_budgets=TRAINING_BUDGETS,
        surfaces=surfaces,
    )
    save_report(
        output_dir / "compute_allocation_errors.txt",
        eval_budget=LLAMA3_405B_FLOPS,
        training_budgets=TRAINING_BUDGETS,
        surfaces=surfaces,
        llama3_comparisons=llama3_comparisons,
    )

    # DCL summary figure
    dcl_rows = _collect_dcl_rows(
        eval_budget=LLAMA3_405B_FLOPS,
        training_budgets=TRAINING_BUDGETS,
        surfaces=surfaces,
        llama3_comparisons=llama3_comparisons,
    )
    plot_dcl_summary(
        dcl_rows,
        output_dir / "compute_allocation_errors.png",
        eval_budget=LLAMA3_405B_FLOPS,
    )

    # ── Progressive filter impact on DCL ──
    for model in [LLAMA3_MODEL, CHINCHILLA_MODEL]:
        _run_progressive_filter(model, output_dir)


if __name__ == "__main__":
    main()
