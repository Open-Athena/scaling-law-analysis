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

LLAMA3_FIT_GRID = ParameterGrid(
    E=np.linspace(0.1, 5.0, 8),
    A=np.logspace(1, 6, 8),
    B=np.logspace(1, 6, 8),
    alpha=np.linspace(0.05, 0.95, 8),
    beta=np.linspace(0.05, 0.95, 8),
)

LLAMA3_VPNLS_GRID = ExponentGrid(
    alpha=np.linspace(0.05, 0.95, 256),
    beta=np.linspace(0.05, 0.95, 256),
)


FILTER_OUTLIERS = False

_LLAMA3_EXPERIMENT = {
    True: "llama3__llama_3__exp_loss",
    False: "llama3__llama_3__raw_loss",
}


def _load_llama3_data(
    log_scale: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Llama 3 isoFLOP data from the data pipeline CSV.

    Returns (N, D, L, C) arrays.
    """
    df = pd.read_csv(ISOFLOPS_CSV)
    edf = df[df["experiment"] == _LLAMA3_EXPERIMENT[log_scale]]
    if FILTER_OUTLIERS:
        edf = edf[~edf["outlier"]]
    N = edf["params"].to_numpy()
    D = edf["tokens"].to_numpy()
    L = edf["loss"].to_numpy()
    C = edf["budget"].to_numpy()
    return N, D, L, C


def _fit_llama3_vpnls(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
) -> LossSurface:
    """Fit Llama 3 isoFLOP data via VPNLS with L-BFGS-B."""
    result = fit_vpnls(
        N, D, L, grid=LLAMA3_VPNLS_GRID, method="l-bfgs-b", use_grad=True
    )
    print(
        f"Llama 3 VPNLS: E={result.E:.4f}, A={result.A:.1f}, B={result.B:.1f}, "
        f"α={result.alpha:.4f}, β={result.beta:.4f} "
        f"(RSS={result.residual_sum_squares:.6f})"
    )
    return LossSurface(
        alpha=result.alpha, beta=result.beta, A=result.A, B=result.B, E=result.E
    )


def _fit_llama3_a3(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
) -> LossSurface:
    """Fit a loss surface to Llama 3 isoFLOP data via Approach 3."""
    result = fit_approach3(
        N, D, L, grid=LLAMA3_FIT_GRID, use_lse=True, use_logloss=True, use_grad=True
    )
    print(
        f"Llama 3 Approach 3: E={result.E:.4f}, A={result.A:.1f}, B={result.B:.1f}, "
        f"α={result.alpha:.4f}, β={result.beta:.4f} "
        f"(RSS={result.residual_sum_squares:.6f})"
    )
    return LossSurface(
        alpha=result.alpha, beta=result.beta, A=result.A, B=result.B, E=result.E
    )


def _fit_llama3_a2(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
) -> ParabolaFitResult:
    """Fit Llama 3 isoFLOP data via Approach 2."""
    result = fit_approach2(N, D, L, C)
    print(
        f"Llama 3 Approach 2: a={result.a:.4f}, b={result.b:.4f} "
        f"(N* ∝ C^a, D* ∝ C^b)"
    )
    return result


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
        figsize=(13, fig_height),
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


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp11")

    # Fit Llama 3 isoFLOP data under both log-scale assumptions.
    llama3_comparisons: list[Llama3FitSet] = []
    primary_vpnls: LossSurface | None = None
    for log_scale in [True, False]:
        label = "log-scale" if log_scale else "raw nats"
        print(f"\n── Llama 3 fits ({label}) ──")
        N, D, L, C = _load_llama3_data(log_scale=log_scale)
        vpnls = _fit_llama3_vpnls(N, D, L)
        a3 = _fit_llama3_a3(N, D, L)
        a2 = _fit_llama3_a2(N, D, L, C)
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


if __name__ == "__main__":
    main()
