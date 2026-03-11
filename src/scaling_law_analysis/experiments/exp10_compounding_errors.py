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

import csv
import io
import urllib.request

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
    ASYMMETRIC_SURFACE,
    CHINCHILLA_SURFACE,
    SODA_SURFACE,
    SPARSE_NMM_SURFACE,
    SYMMETRIC_SURFACE,
    fit_simulated_approach2,
    prepare_output_dir,
)

# Llama 3 isoFLOP data (digitized from paper figure).
# Source: https://github.com/eric-czech/llama3_isoflop_extraction
LLAMA3_ISOFLOP_CSV_URL = (
    "https://raw.githubusercontent.com/eric-czech/llama3_isoflop_extraction/"
    "1bc1755b76e6ee55a911549c8ec52b71cb480320/isoflops_points.csv"
)

LLAMA3_LOSS_LOG_SCALE = True  # if True, interpret validation_loss as ln(loss)

LLAMA3_FIT_GRID = ParameterGrid(
    E=np.linspace(0.1, 5.0, 8),
    A=np.logspace(1, 6, 8),
    B=np.logspace(1, 6, 8),
    alpha=np.linspace(0.05, 0.95, 8),
    beta=np.linspace(0.05, 0.95, 8),
)


def _download_llama3_csv() -> str:
    """Download Llama 3 isoFLOP CSV and return raw text."""
    with urllib.request.urlopen(LLAMA3_ISOFLOP_CSV_URL) as resp:
        return resp.read().decode("utf-8")


def _parse_llama3_data(
    text: str, log_scale: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse Llama 3 CSV text into (N, D, L, C) arrays."""
    reader = csv.DictReader(io.StringIO(text))
    C_list, D_list, L_list = [], [], []
    for row in reader:
        C_list.append(float(row["compute_budget"]))
        D_list.append(float(row["training_tokens"]))
        L_list.append(float(row["validation_loss"]))
    C = np.array(C_list)
    D = np.array(D_list)
    L = np.array(L_list)
    if log_scale:
        L = np.exp(L)
    N = C / (6 * D)  # C = 6ND
    return N, D, L, C


def fit_llama3_surface(
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


LLAMA3_VPNLS_GRID = ExponentGrid(
    alpha=np.linspace(0.05, 0.95, 256),
    beta=np.linspace(0.05, 0.95, 256),
)


def fit_llama3_vpnls(
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


def fit_llama3_approach2(
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


SURFACES = [
    ("Symmetric", SYMMETRIC_SURFACE),
    ("Chinchilla", CHINCHILLA_SURFACE),
    ("Asymmetric", ASYMMETRIC_SURFACE),
]

COST_REPORT_SURFACES = [
    ("Chinchilla", CHINCHILLA_SURFACE),
    ("SODA", SODA_SURFACE),
    ("Sparse-NMM", SPARSE_NMM_SURFACE),
]

# ── Sampling grids ──────────────────────────────────────────────────────────

GRID_WIDTHS = [
    ("XS (±2×)", np.log10(2)),
    ("S (±4×)", np.log10(4)),
    ("L (±8×)", np.log10(8)),
    ("XL (±16×)", np.log10(16)),
]

TRAINING_BUDGETS = np.array([1e18, 1e19, 1e20, 1e21, 1e22])

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
        "label": f"Drift to {10**OFF_CENTER_DRIFT_RATE:.0f}×",
        "drift_rate": OFF_CENTER_DRIFT_RATE,
        "center_scale": 1.0,
    },
]

# ── Evaluation scenarios ───────────────────────────────────────────────────

LLAMA3_405B_FLOPS = 3.8e25

SCENARIOS = {
    "default": {
        "eval_budget": 1e24,
        "file_stem": "compounding_errors",
        "surfaces": SURFACES,
        "report": False,
    },
    "costs": {
        "eval_budget": LLAMA3_405B_FLOPS,
        "file_stem": "compounding_error_costs",
        "surfaces": COST_REPORT_SURFACES,
        "report": True,
    },
}

# ── Cost assumptions ──────────────────────────────────────────────────────

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


# ── Core computation ────────────────────────────────────────────────────────


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
    inf_N = fit_result.N_opt(eval_budget)
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
    *,
    log_range: float,
    center_scale: float,
    drift_rate: float,
) -> dict:
    """Fit Approach 2 on training data and compare allocation at eval budget."""
    fit_result = fit_simulated_approach2(
        compute_budgets=TRAINING_BUDGETS,
        surface=surface,
        drift_rate=drift_rate,
        center_scale=center_scale,
        log_range=log_range,
    )
    return compare_allocations(surface, fit_result, eval_budget)


SurfaceList = list[tuple[str, LossSurface]]


def run(eval_budget: float, surfaces: SurfaceList = SURFACES) -> dict:
    """Run compounding errors analysis.

    Args:
        eval_budget: Compute budget (FLOPs) to extrapolate D* to.
        surfaces: List of (name, LossSurface) pairs to evaluate.

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
                    log_range=log_range,
                    center_scale=cfg["center_scale"],
                    drift_rate=cfg["drift_rate"],
                )
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


def plot(
    results: dict,
    output_path: str | Path,
    eval_budget: float,
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

        for i, ((grid_name, _log_range), color, offset) in enumerate(
            zip(GRID_WIDTHS, colors, offsets)
        ):
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

    if eval_budget == 1e24:
        budget_str = "10²⁴"
    else:
        budget_str = f"{eval_budget:.1e}"
    fig.suptitle(
        "Compounding Errors: D* Prediction Error by Bias Configuration\n"
        f"(Fitting: 10¹⁸–10²² FLOPs → Extrapolating to {budget_str} FLOPs)",
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
    surfaces: SurfaceList = SURFACES,
) -> None:
    """Export raw error data to CSV."""
    TRAINING_RANGE = "1e18-1e22"
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


# ── Report ──────────────────────────────────────────────────────────────────

# Report configuration: which (config, surface, grid) combinations to include.
# Report configuration: which (config, grid) to use for the .txt report.
REPORT_DRIFT_RATE = np.log10(3)  # drift to 3×
REPORT_GRID = "XS (±2×)"


def _direction(value: float) -> str:
    """Describe whether a signed error implies over- or under-utilization."""
    if value > 0:
        return "over"
    elif value < 0:
        return "under"
    return "exact"


def _fmt_flops(flops: float) -> str:
    """Format a FLOP count in human-readable scientific notation."""
    exp = int(np.floor(np.log10(abs(flops))))
    mantissa = flops / 10**exp
    return f"{mantissa:.2f}e{exp}"


def _fmt_count(value: float) -> str:
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


def _fmt_dollars(dollars: float) -> str:
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


def _fmt_surface_result(d: dict, eval_budget: float) -> str:
    """Format a single surface's comparison results."""
    d_dir = _direction(d["inferred_D"] - d["true_D"])
    n_dir = _direction(d["inferred_N"] - d["true_N"])
    n_err_pct = abs(d["inferred_N"] - d["true_N"]) / d["true_N"] * 100
    waste_pct = d["wasted_flops"] / eval_budget * 100
    lines = [
        f"  D* (tokens):  {_fmt_count(d['true_D']):>10s} true → "
        f"{_fmt_count(d['inferred_D']):>10s} inferred  "
        f"({d_dir}, {abs(d['rel_error_pct']):.1f}%)",
        f"  N* (params):  {_fmt_count(d['true_N']):>10s} true → "
        f"{_fmt_count(d['inferred_N']):>10s} inferred  "
        f"({n_dir}, {n_err_pct:.1f}%)",
        f"  Loss (nats):  {d['true_loss']:.6f} true → "
        f"{d['inferred_loss']:.6f} inferred  "
        f"(+{d['loss_error_nats']:.6f} penalty)",
        f"  Wasted compute: {_fmt_flops(d['wasted_flops'])} FLOPs "
        f"({waste_pct:.1f}% of budget, {_fmt_dollars(d['wasted_dollars'])})",
    ]
    return "\n".join(lines)


Llama3FitSet = tuple[bool, LossSurface, LossSurface, ParabolaFitResult]


def save_report(
    output_path: str | Path,
    eval_budget: float,
    surfaces: SurfaceList = COST_REPORT_SURFACES,
    llama3_comparisons: list[Llama3FitSet] | None = None,
) -> None:
    """Write a human-readable report for selected compounding error results.

    Computes results independently from the main run() so that the report
    can include surfaces not in the main SURFACES list.

    If llama3_comparisons is provided, appends a section comparing VPNLS and
    Approach 3 fits on real Llama 3 isoFLOP data for each log-scale assumption.
    Each entry is (log_scale, vpnls_surface, a3_surface, a2_result).
    """
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
    w("Compounding Error Costs")
    w("=" * 60)
    w("")
    w(
        "Approach 2 fit on 10^18-10^22 FLOPs, extrapolated to "
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
    w("The resulting loss L(N,D) is suboptimal. 'Wasted compute' inverts")
    w("L_opt(C) to find the smaller budget that would reach the same loss")
    w("with optimal allocation; the difference is wasted.")
    w("")

    for sname, surface in surfaces:
        d = fit_and_compare(
            surface=surface,
            eval_budget=eval_budget,
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
        w(_fmt_surface_result(d, eval_budget))
        w("")

    # ── Section 2: Llama 3 fits comparison ──
    if llama3_comparisons is not None:
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
                w(_fmt_surface_result(d, eval_budget))
                w("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp10")

    # Fit Llama 3 isoFLOP data under both log-scale assumptions.
    csv_text = _download_llama3_csv()
    llama3_comparisons: list[Llama3FitSet] = []
    primary_vpnls: LossSurface | None = None
    for log_scale in [True, False]:
        label = "log-scale" if log_scale else "raw nats"
        print(f"\n── Llama 3 fits ({label}) ──")
        N, D, L, C = _parse_llama3_data(csv_text, log_scale=log_scale)
        vpnls = fit_llama3_vpnls(N, D, L)
        a3 = fit_llama3_surface(N, D, L)
        a2 = fit_llama3_approach2(N, D, L, C)
        llama3_comparisons.append((log_scale, vpnls, a3, a2))
        if log_scale == LLAMA3_LOSS_LOG_SCALE:
            primary_vpnls = vpnls
    assert primary_vpnls is not None
    cost_surfaces = COST_REPORT_SURFACES + [("Llama 3", primary_vpnls)]

    for name, scenario in SCENARIOS.items():
        eb = scenario["eval_budget"]
        stem = scenario["file_stem"]
        surfs = cost_surfaces if name == "costs" else scenario["surfaces"]
        results = run(eval_budget=eb, surfaces=surfs)
        plot(results, output_dir / f"{stem}.png", eval_budget=eb, surfaces=surfs)
        save_csv(results, output_dir / f"{stem}.csv", eval_budget=eb, surfaces=surfs)
        if scenario["report"]:
            save_report(
                output_dir / f"{stem}.txt",
                eval_budget=eb,
                surfaces=surfs,
                llama3_comparisons=llama3_comparisons,
            )


if __name__ == "__main__":
    main()
