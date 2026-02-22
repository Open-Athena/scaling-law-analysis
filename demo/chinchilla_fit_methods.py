#!/usr/bin/env python3
"""
Standalone comparison of scaling law estimator efficiency.

Compares three fitting methods — Approach 2, Approach 3, and VPNLS —
on perfectly centered IsoFLOP data from a symmetric loss surface.
All fitting logic is self-contained (no codebase imports).

Outputs: demo/chinchilla_fit_methods.png
Usage:   python demo/chinchilla_fit_methods.py
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, nnls

# =============================================================================
# Surface: L(N, D) = E + A/N^α + B/D^β  (symmetric: α = β)
# =============================================================================

ALPHA, BETA = 0.31, 0.31
A, B, E = 400.0, 400.0, 1.69
TRUE_A = BETA / (ALPHA + BETA)  # 0.5
TRUE_B = ALPHA / (ALPHA + BETA)  # 0.5
_G = (ALPHA * A / (BETA * B)) ** (1.0 / (ALPHA + BETA))  # = 1.0 for symmetric


def loss(N: float, D: float) -> float:
    return E + A / N**ALPHA + B / D**BETA


def N_opt(C: float) -> float:
    return _G * (C / 6) ** TRUE_A


# =============================================================================
# Data generation
# =============================================================================


@dataclass
class Data:
    C: np.ndarray
    N: np.ndarray
    D: np.ndarray
    L: np.ndarray


def generate_data(
    budgets: np.ndarray,
    n_points: int,
    log_width: float,
    noise_std: float,
    rng: np.random.Generator,
) -> Data:
    """Perfectly centered, noise-injected IsoFLOP samples."""
    Cs, Ns, Ds, Ls = [], [], [], []
    for C_val in budgets:
        center = np.log10(N_opt(C_val))
        log_Ns = np.linspace(center - log_width, center + log_width, n_points)
        N_arr = 10.0**log_Ns
        D_arr = C_val / (6.0 * N_arr)
        L_arr = np.array([loss(n, d) for n, d in zip(N_arr, D_arr)])
        Cs.append(np.full(n_points, float(C_val)))
        Ns.append(N_arr)
        Ds.append(D_arr)
        Ls.append(L_arr)
    L_all = np.concatenate(Ls)
    return Data(
        np.concatenate(Cs),
        np.concatenate(Ns),
        np.concatenate(Ds),
        L_all + rng.normal(0, noise_std, L_all.shape),
    )


# =============================================================================
# Approach 2: parabola per budget → power-law regression
# =============================================================================


def fit_approach2(d: Data) -> tuple[float, float]:
    unique_C = np.unique(d.C)
    valid_C, N_opts, D_opts = [], [], []
    for c in unique_C:
        m = d.C == c
        n_coeff = np.polyfit(np.log10(d.N[m]), d.L[m], 2)
        d_coeff = np.polyfit(np.log10(d.D[m]), d.L[m], 2)
        if n_coeff[0] < 1e-10 or d_coeff[0] < 1e-10:
            continue  # non-positive curvature (noise inverted the parabola) → no valid minimum
        n_opt = 10.0 ** (-n_coeff[1] / (2 * n_coeff[0]))
        d_opt = 10.0 ** (-d_coeff[1] / (2 * d_coeff[0]))
        if n_opt <= 0 or d_opt <= 0:
            continue  # degenerate optimum (negative N* or D*) → discard this budget
        valid_C.append(c)
        N_opts.append(n_opt)
        D_opts.append(d_opt)
    if len(valid_C) < 2:
        return np.nan, np.nan
    valid_C = np.array(valid_C)
    fit_exp = lambda x, y: float(np.polyfit(np.log10(x), np.log10(np.array(y)), 1)[0])
    return fit_exp(valid_C, N_opts), fit_exp(valid_C, D_opts)


# =============================================================================
# Approach 3: direct 5-param L-BFGS-B
# =============================================================================

_A3_BOUNDS = [(1e-6, 10), (1e-6, 1e6), (1e-6, 1e6), (0.01, 0.99), (0.01, 0.99)]


def fit_approach3(d: Data) -> tuple[float, float]:
    log_N, log_D, L = np.log(d.N), np.log(d.D), d.L

    # Vectorized 4^5 grid search for initialization
    grid = np.array(
        list(
            itertools.product(
                np.linspace(0.1, 5, 4),
                np.logspace(1, 4, 4),
                np.logspace(1, 4, 4),
                np.linspace(0.05, 0.95, 4),
                np.linspace(0.05, 0.95, 4),
            )
        )
    ).T  # grid[i]: 0=E, 1=A, 2=B, 3=α, 4=β
    preds = (
        grid[0]
        + grid[1] * np.exp(-grid[3] * log_N[:, None])
        + grid[2] * np.exp(-grid[4] * log_D[:, None])
    )
    rss_vals = np.sum((L[:, None] - preds) ** 2, axis=0)
    x0 = grid[:, np.argmin(rss_vals)]

    def rss(x):
        p = x[0] + x[1] * np.exp(-x[3] * log_N) + x[2] * np.exp(-x[4] * log_D)
        return float(np.sum((L - p) ** 2))

    def grad(x):
        tN, tD = np.exp(-x[3] * log_N), np.exp(-x[4] * log_D)
        r = x[0] + x[1] * tN + x[2] * tD - L
        return np.array(
            [
                2 * np.sum(r),
                2 * np.sum(r * tN),
                2 * np.sum(r * tD),
                2 * np.sum(r * x[1] * tN * (-log_N)),
                2 * np.sum(r * x[2] * tD * (-log_D)),
            ]
        )

    res = minimize(
        rss,
        x0,
        jac=grad,
        method="L-BFGS-B",
        bounds=_A3_BOUNDS,
        options={"ftol": 1e-15, "gtol": 1e-15, "maxiter": 1_000},
    )
    if res is None or not np.all(np.isfinite(res.x)):
        return np.nan, np.nan
    al, be = res.x[3], res.x[4]
    return be / (al + be), al / (al + be)


# =============================================================================
# VPNLS: 2D Nelder-Mead over (α,β) with NNLS inner solve
# =============================================================================


def fit_vpnls(d: Data) -> tuple[float, float]:
    log_N, log_D, L = np.log(d.N), np.log(d.D), d.L

    def rss_nnls(al, be):
        dm = np.column_stack(
            [np.ones(len(L)), np.exp(-al * log_N), np.exp(-be * log_D)]
        )
        return nnls(dm, L)[1] ** 2

    # 32×32 grid search
    best, ba, bb = np.inf, 0.5, 0.5
    for al in np.linspace(0.05, 0.95, 32):
        for be in np.linspace(0.05, 0.95, 32):
            r = rss_nnls(al, be)
            if r < best:
                best, ba, bb = r, al, be

    res = minimize(
        lambda x: rss_nnls(x[0], x[1]),
        [ba, bb],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-15, "maxiter": 1_000},
    )
    if res is None or not np.all(np.isfinite(res.x)):
        return np.nan, np.nan
    al, be = res.x
    return be / (al + be), al / (al + be)


# =============================================================================
# Experiment
# =============================================================================

METHODS = {"Approach 2": fit_approach2, "Approach 3": fit_approach3, "VPNLS": fit_vpnls}

# Experimental grid
NOISE_STDS = [0.05, 0.1, 0.2, 0.3]
N_POINTS = [21, 31, 41]
N_BUDGETS = [3, 5, 7]
LOG_WIDTHS = [np.log10(2), np.log10(4), np.log10(8)]
N_TRIALS = 10


def run() -> dict[str, list[float]]:
    """Run Monte Carlo; return {method: [signed_errors]} pooled over all conditions."""
    rng = np.random.default_rng(42)
    errors = {m: [] for m in METHODS}
    total = (
        len(NOISE_STDS)
        * len(LOG_WIDTHS)
        * len(N_POINTS)
        * len(N_BUDGETS)
        * N_TRIALS
        * len(METHODS)
    )
    done, t0 = 0, time.time()

    for noise in NOISE_STDS:
        for lw in LOG_WIDTHS:
            for npts in N_POINTS:
                for nb in N_BUDGETS:
                    budgets = np.logspace(17, 21, nb)
                    for _ in range(N_TRIALS):
                        data = generate_data(budgets, npts, lw, noise, rng)
                        for name, fn in METHODS.items():
                            a_hat, b_hat = fn(data)
                            for est, true in [(a_hat, TRUE_A), (b_hat, TRUE_B)]:
                                if np.isfinite(est):
                                    errors[name].append(est - true)
                            done += 1
                    pct = done / total * 100
                    print(
                        f"\r  [{pct:5.1f}%] {done}/{total} | {time.time()-t0:.0f}s",
                        end="",
                        flush=True,
                    )

    print(f"\n  Done in {time.time()-t0:.1f}s  ({total} fits)\n")
    return errors


# =============================================================================
# Visualization
# =============================================================================


def plot(errors: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(METHODS.keys())
    colors = ["#4287f5", "#f5a442", "#42c978"]
    data = [errors[n] for n in names]

    bp = ax.boxplot(
        data,
        tick_labels=names,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
        medianprops=dict(color="black", linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlim(0.4, len(names) + 0.9)
    ax.set_ylabel(r"Signed error  ($\hat{e} - e_{true}$)  [symlog scale]", fontsize=11)
    ax.set_title(
        r"Estimator Efficiency: Recovery of $a = \beta/(\alpha+\beta)$, "
        r"$b = \alpha/(\alpha+\beta)$"
        "\n"
        r"Symmetric surface: $\alpha = \beta = 0.31$, $A = B = 400$, $E = 1.69$  "
        r"(true $a = b = 0.5$)"
        "\n"
        f"Centered IsoFLOP sampling  |  "
        f"noise σ ∈ {{{', '.join(str(s) for s in NOISE_STDS)}}}  |  "
        f"pts/curve ∈ {{{', '.join(str(n) for n in N_POINTS)}}}  |  "
        f"budgets ∈ {{{', '.join(str(n) for n in N_BUDGETS)}}}\n"
        f"log-widths ∈ {{±2×, ±4×, ±8×}}  |  "
        f"{N_TRIALS} trials/config  |  "
        f"{sum(len(v) for v in errors.values()):,} total exponent estimates"
        f"  (pooled $a$ and $b$)",
        fontsize=9,
        linespacing=1.5,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Annotate stats
    for i, name in enumerate(names):
        vals = np.array(errors[name])
        mean = np.mean(vals)
        med = np.median(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        std = np.std(vals)
        n = len(vals)
        ax.annotate(
            f"n={n}\nμ={mean:+.4f}\nσ̂={std:.4f}\nmed={med:+.4f}\nIQR={iqr:.4f}",
            xy=(i + 1.22, 0),
            fontsize=7.5,
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"
            ),
        )

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def main() -> None:
    errors = run()
    plot(errors, Path(__file__).parent / "chinchilla_fit_methods.png")


if __name__ == "__main__":
    main()
