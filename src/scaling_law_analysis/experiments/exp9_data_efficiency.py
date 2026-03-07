"""Experiment 9: Data Efficiency.

Monte Carlo comparison of Approach 2, Approach 3, and VPNLS exponent recovery
under noisy, data-limited conditions on a symmetric loss surface with perfectly
centered sampling.

Reimplements the comparison from demo/chinchilla_fit_methods.py using codebase
fitting utilities and simulation infrastructure.

Usage:
    uv run python -m scaling_law_analysis.experiments.exp9_data_efficiency
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    FitError,
    LossSurface,
    fit_approach2,
    fit_approach3,
    fit_vpnls,
    isoflop_sample,
)
from scaling_law_analysis.experiments.common import prepare_output_dir

# ── Surface ──────────────────────────────────────────────────────────────────

SURFACE = LossSurface(alpha=0.31, beta=0.31, A=400.0, B=400.0, E=1.69)
TRUE_B = SURFACE.b
assert TRUE_B == 0.5

# ── Experimental grid ────────────────────────────────────────────────────────

NOISE_STDS = [0.01, 0.02, 0.05]
N_POINTS_LIST = [21, 31, 41]
N_BUDGETS_LIST = [3, 5, 7]
LOG_WIDTHS = [np.log10(2), np.log10(4), np.log10(8)]
N_TRIALS = 10


# ── Data generation ──────────────────────────────────────────────────────────


def _generate_noisy_data(
    budgets: np.ndarray,
    n_points: int,
    log_width: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate centered IsoFLOP data with additive Gaussian noise.

    Returns (N, D, L, C) arrays pooled across all budgets.
    """
    all_N, all_D, all_L, all_C = [], [], [], []
    for C_val in budgets:
        N, D, L = isoflop_sample(
            C=C_val,
            n_points=n_points,
            log_range=log_width,
            center_offset=0.0,
            surface=SURFACE,
        )
        L = L + rng.normal(0, noise_std, L.shape)
        all_N.append(N)
        all_D.append(D)
        all_L.append(L)
        all_C.append(np.full(len(N), C_val))
    return (
        np.concatenate(all_N),
        np.concatenate(all_D),
        np.concatenate(all_L),
        np.concatenate(all_C),
    )


# ── Fitting ──────────────────────────────────────────────────────────────────


def _fit_all(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, C: np.ndarray
) -> dict[str, float]:
    """Fit all three methods and return {name: b_hat}.

    Returns NaN for methods that raise FitError (e.g. inverted parabola
    under high noise).
    """
    try:
        r2 = fit_approach2(N=N, D=D, L=L, C=C)
        b2 = r2.b
    except FitError:
        b2 = np.nan

    try:
        r3 = fit_approach3(N=N, D=D, L=L, use_lse=True, use_grad=True)
        b3 = r3.b
    except FitError:
        b3 = np.nan

    try:
        rv = fit_vpnls(N=N, D=D, L=L, method="l-bfgs-b", use_grad=True)
        bv = rv.b
    except FitError:
        bv = np.nan

    return {"Approach 2": b2, "Approach 3": b3, "VPNLS": bv}


# ── Monte Carlo ──────────────────────────────────────────────────────────────


Errors = dict[tuple[str, float], list[float]]


@dataclass(frozen=True)
class MethodVariance:
    method: str
    variance: float
    per_noise_var: dict[float, float]


def run() -> Errors:
    """Run Monte Carlo sweep; return {(method, noise_std): [signed_errors]}."""
    rng = np.random.default_rng(42)
    methods = ["Approach 2", "Approach 3", "VPNLS"]
    errors: Errors = {(m, s): [] for m in methods for s in NOISE_STDS}

    total = (
        len(NOISE_STDS)
        * len(LOG_WIDTHS)
        * len(N_POINTS_LIST)
        * len(N_BUDGETS_LIST)
        * N_TRIALS
    )
    done = 0
    t0 = time.time()

    for noise in NOISE_STDS:
        for lw in LOG_WIDTHS:
            for npts in N_POINTS_LIST:
                for nb in N_BUDGETS_LIST:
                    budgets = np.logspace(17, 21, nb)
                    for _ in range(N_TRIALS):
                        N, D, L, C = _generate_noisy_data(budgets, npts, lw, noise, rng)
                        results = _fit_all(N, D, L, C)
                        for name in methods:
                            b_hat = results[name]
                            if np.isfinite(b_hat):
                                errors[(name, noise)].append(b_hat - TRUE_B)
                        done += 1
                        pct = done / total * 100
                        elapsed = time.time() - t0
                        print(
                            f"\r  [{pct:5.1f}%] {done}/{total} | {elapsed:.0f}s",
                            end="",
                            flush=True,
                        )

    print(
        f"\n  Done in {time.time() - t0:.1f}s  ({total} conditions, {total * 3} fits)\n"
    )
    return errors


# ── Visualization ────────────────────────────────────────────────────────────


def plot(errors: Errors, output_path: str | Path) -> None:
    """Grouped boxplot: noise levels within each method."""
    methods = ["Approach 2", "Approach 3", "VPNLS"]
    method_colors = {
        "Approach 2": "#4287f5",
        "Approach 3": "#f5a442",
        "VPNLS": "#42c978",
    }
    n_noise = len(NOISE_STDS)
    box_width = 0.5
    group_gap = 1.2  # gap between method groups
    box_spacing = 0.65  # spacing between boxes within a group

    fig, ax = plt.subplots(figsize=(9, 4.5))

    positions = []
    group_centers = []
    for gi, method in enumerate(methods):
        group_start = gi * (n_noise * box_spacing + group_gap)
        group_positions = [group_start + j * box_spacing for j in range(n_noise)]
        positions.extend(group_positions)
        group_centers.append(float(np.mean(group_positions)))

    data = []
    for method in methods:
        for noise in NOISE_STDS:
            data.append(errors[(method, noise)])

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
        medianprops=dict(color="black", linewidth=1.2),
    )

    # Color boxes by method
    for i, (method, noise) in enumerate([(m, s) for m in methods for s in NOISE_STDS]):
        bp["boxes"][i].set_facecolor(method_colors[method])
        bp["boxes"][i].set_alpha(0.7)

    # Overlay mean ± 2σ error bars
    for i, (method, noise) in enumerate([(m, s) for m in methods for s in NOISE_STDS]):
        vals = np.array(errors[(method, noise)])
        if len(vals) > 0:
            mean = np.mean(vals)
            std2 = 2 * np.std(vals)
            ax.errorbar(
                positions[i],
                mean,
                yerr=std2,
                fmt="D",
                color="black",
                markersize=3,
                capsize=4,
                capthick=1,
                linewidth=1,
                zorder=5,
            )

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_yscale("symlog", linthresh=0.01)
    # Tick labels: method name on center, noise σ under each box
    all_ticks = list(positions)
    all_labels = []
    for gi, method in enumerate(methods):
        for noise in NOISE_STDS:
            all_labels.append(f"\u03c3={noise}")
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=7, color="black")
    # Add method names as a second row via text
    for gi, method in enumerate(methods):
        ax.text(
            group_centers[gi],
            -0.10,
            method,
            ha="center",
            va="top",
            fontsize=11,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_ylabel(r"Signed error  ($\hat{b} - b_{true}$)  [symlog scale]", fontsize=11)

    # Overall stats per method (pooled across noise levels)
    for gi, method in enumerate(methods):
        all_vals = []
        for noise in NOISE_STDS:
            all_vals.extend(errors[(method, noise)])
        vals = np.array(all_vals)
        n = len(vals)
        std = np.std(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        ax.text(
            group_centers[gi],
            -0.18,
            f"n={n}  \u03c3\u0302={std:.4f}  IQR={iqr:.4f}",
            fontsize=7,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"
            ),
        )

    total_estimates = sum(len(v) for v in errors.values())
    ax.set_title(
        r"Estimator Efficiency: Recovery of $b = \alpha/(\alpha+\beta)$"
        "\n"
        r"Symmetric surface: $\alpha = \beta = 0.31$, $A = B = 400$, $E = 1.69$  "
        r"(true $b = 0.5$)"
        "\n"
        f"Centered IsoFLOP sampling  |  "
        f"pts/curve \u2208 {{{', '.join(str(n) for n in N_POINTS_LIST)}}}  |  "
        f"budgets \u2208 {{{', '.join(str(n) for n in N_BUDGETS_LIST)}}}  |  "
        f"log-widths \u2208 {{\u00b12\u00d7, \u00b14\u00d7, \u00b18\u00d7}}\n"
        f"{N_TRIALS} trials/config  |  "
        f"{total_estimates:,} total estimates  ($b$ exponent only)",
        fontsize=9,
        linespacing=1.5,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_summary(errors: Errors, output_path: str | Path) -> None:
    """Summary figure: horizontal bar chart of variance + heatmap by noise level.

    Left panel: horizontal bars showing overall variance per method.
    Right panel: heatmap of std dev by noise level, white-to-black colormap.
    """
    methods = ["Approach 2", "Approach 3", "VPNLS"]

    # ── Compute per-method stats ──
    row_data: list[MethodVariance] = []
    for method in methods:
        all_vals: list[float] = []
        per_noise_var: dict[float, float] = {}
        for noise in NOISE_STDS:
            vals = np.array(errors[(method, noise)])
            if len(vals) > 0:
                per_noise_var[noise] = float(np.var(vals))
                all_vals.extend(vals.tolist())
            else:
                per_noise_var[noise] = np.nan
        arr = np.array(all_vals)
        row_data.append(
            MethodVariance(
                method=method,
                variance=float(np.var(arr)) if len(arr) > 0 else np.nan,
                per_noise_var=per_noise_var,
            )
        )

    # Sort by variance descending (worst at top)
    row_data.sort(key=lambda rd: -(rd.variance if np.isfinite(rd.variance) else 0.0))

    n_rows = len(row_data)
    n_noise = len(NOISE_STDS)
    y_positions = np.arange(n_rows)

    fig_height = 0.7 * n_rows + 1.5
    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(10, fig_height),
        gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.02},
        sharey=True,
        layout="constrained",
    )

    # ── Left panel: horizontal bar chart of variance (×10⁻⁴) ──
    scale = 1e4
    variances = [rd.variance * scale for rd in row_data]
    ax_bar.barh(
        y_positions,
        variances,
        color="#555555",
        edgecolor="black",
        linewidth=0.8,
        height=0.5,
    )
    for idx, v in enumerate(variances):
        if np.isfinite(v):
            ax_bar.text(
                v - 0.15,
                y_positions[idx],
                f"{v:.1f}",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
            )

    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels([rd.method for rd in row_data], fontsize=10)
    ax_bar.set_xlabel(
        r"Variance of $\hat{b} - b_{\mathrm{true}}$ ($\times 10^{-4}$)", fontsize=11
    )
    ax_bar.set_title("Overall Estimator Variance (pooled)", fontsize=11)
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.invert_yaxis()

    # ── Right panel: variance heatmap by noise level ──
    col_labels = [f"σ={s}" for s in NOISE_STDS]

    heat_data = np.full((n_rows, n_noise), np.nan)
    for idx, rd in enumerate(row_data):
        for j, noise in enumerate(NOISE_STDS):
            heat_data[idx, j] = rd.per_noise_var[noise] * scale

    finite_vals = heat_data[np.isfinite(heat_data)]
    vmin = max(finite_vals.min(), 1e-10) if len(finite_vals) > 0 else 1e-10
    vmax = finite_vals.max() if len(finite_vals) > 0 else 1e2

    cmap_wb = LinearSegmentedColormap.from_list("white_black", ["#ffffff", "#000000"])
    ax_heat.imshow(
        heat_data,
        aspect="auto",
        cmap=cmap_wb,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )

    for idx in range(n_rows):
        for j in range(n_noise):
            val = heat_data[idx, j]
            if np.isfinite(val) and val > 0:
                txt = f"{val:.1f}"
            else:
                txt = "—"
            log_frac = 0.0
            if np.isfinite(val) and val > 0 and vmax > vmin:
                log_frac = (np.log10(val) - np.log10(vmin)) / (
                    np.log10(vmax) - np.log10(vmin)
                )
            text_color = "white" if log_frac > 0.5 else "black"
            ax_heat.text(
                j,
                idx,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax_heat.set_xticks(np.arange(n_noise))
    ax_heat.set_xticklabels(col_labels, fontsize=9)
    ax_heat.set_xlabel("Noise level", fontsize=11)
    ax_heat.set_title(r"Variance by Noise ($\times 10^{-4}$)", fontsize=11)
    ax_heat.tick_params(left=False)

    fig.suptitle(
        r"Data Efficiency: Recovery of $b = \alpha/(\alpha+\beta)$",
        fontsize=13,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp9")
    errors = run()
    plot(errors, output_dir / "data_efficiency_detailed.png")
    plot_summary(errors, output_dir / "data_efficiency.png")


if __name__ == "__main__":
    main()
