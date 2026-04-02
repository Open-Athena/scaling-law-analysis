"""LOO leverage analysis of the A2 minima power law fit."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from analysis.delphi_scaling_analysis import (  # pyrefly: ignore
    FORECAST_TARGETS,
    MinimaPowerLaw,
    _fit_power_law_to_minima,
    fit_dataset,
    fmt_budget,
    load_delphi,
)
from scaling_law_analysis.common import save_figure

OUT_DIR = __import__("pathlib").Path(__file__).parent / "results"

# Budgets with positive full-fit residuals (above the power law curve)
HIGHLIGHT_BUDGETS: list[float] | None = None  # None = all budgets


@dataclass
class LOODrop:
    """Results from dropping one vertex from the power law fit."""

    budget: float
    d_star: float
    l_min: float
    full_pred: float
    loo_pl: MinimaPowerLaw
    loo_pred: float
    loo_fc: float  # LOO power law evaluated at forecast D*

    @property
    def full_resid(self) -> float:
        return self.l_min - self.full_pred

    @property
    def loo_resid(self) -> float:
        return self.l_min - self.loo_pred

    @property
    def fc_delta(self) -> float:
        """LOO forecast minus full forecast at 1e23 FLOP budget."""
        return self.loo_fc - self._full_fc

    @property
    def fc_pct(self) -> float:
        return self.fc_delta / self._full_fc * 100

    _full_fc: float = 0.0  # set after construction


def run_loo(
    pl_full: MinimaPowerLaw,
    D_stars: np.ndarray,
    L_mins: np.ndarray,
    budgets: np.ndarray,
) -> list[LOODrop]:
    """Run LOO on the minima power law fit."""
    fc_target = FORECAST_TARGETS[1e23]
    D_fc = fc_target.tokens
    full_fc = pl_full.x1 * D_fc**pl_full.x2 + pl_full.x0
    full_pred = pl_full.x1 * D_stars**pl_full.x2 + pl_full.x0

    drops: list[LOODrop] = []
    for i in range(len(budgets)):
        mask = np.ones(len(budgets), dtype=bool)
        mask[i] = False
        pl_loo = _fit_power_law_to_minima(D_stars[mask], L_mins[mask])
        loo_pred = pl_loo.x1 * D_stars[i] ** pl_loo.x2 + pl_loo.x0
        loo_fc = pl_loo.x1 * D_fc**pl_loo.x2 + pl_loo.x0
        drop = LOODrop(
            budget=budgets[i],
            d_star=D_stars[i],
            l_min=L_mins[i],
            full_pred=full_pred[i],
            loo_pl=pl_loo,
            loo_pred=loo_pred,
            loo_fc=loo_fc,
        )
        drop._full_fc = full_fc
        drops.append(drop)
    return drops


def print_table(pl_full: MinimaPowerLaw, drops: list[LOODrop], full_fc: float) -> None:
    """Print the LOO leverage table."""
    print(
        f"Full-fit params: x0={pl_full.x0:.6f}  x1={pl_full.x1:.6f}  x2={pl_full.x2:.6f}"
    )
    print(
        f"Full PL evaluated at D*={FORECAST_TARGETS[1e23].tokens:.2e} "
        f"(1e23 FLOP budget): {full_fc:.6f}"
    )
    print()
    print(
        f"| Budget    | D*       | L_min   | Full Resid | LOO Resid "
        f"| FC \u0394%     | \u0394x0      | \u0394x1        | \u0394x2      |"
    )
    print(
        f"|-----------|----------|---------|------------|-----------|"
        f"----------|----------|------------|----------|"
    )
    for d in drops:
        dx0 = d.loo_pl.x0 - pl_full.x0
        dx1 = d.loo_pl.x1 - pl_full.x1
        dx2 = d.loo_pl.x2 - pl_full.x2
        print(
            f"| {fmt_budget(d.budget):>9s} | {d.d_star:.2e} | {d.l_min:.4f} "
            f"| {d.full_resid:+.5f}    | {d.loo_resid:+.5f}   "
            f"| {d.fc_pct:+6.2f}%  | {dx0:+.6f} | {dx1:+10.4f}   | {dx2:+.6f} |"
        )


def plot_leverage(
    pl_full: MinimaPowerLaw,
    drops: list[LOODrop],
    D_stars: np.ndarray,
    L_mins: np.ndarray,
) -> plt.Figure:
    """7x2 grid: left = PL curves, right = residuals per dropped budget."""
    D_fc = FORECAST_TARGETS[1e23].tokens
    actual = FORECAST_TARGETS[1e23].actual

    plot_drops = sorted(drops, key=lambda d: d.budget)
    n = len(plot_drops)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(12, 2 * n),
        gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.12},
    )

    # Curve spans training D* through forecast D*
    d_curve = np.logspace(np.log10(D_stars.min() * 0.7), np.log10(D_fc * 1.3), 500)
    full_curve = pl_full.x1 * d_curve**pl_full.x2 + pl_full.x0
    full_fc = pl_full.x1 * D_fc**pl_full.x2 + pl_full.x0

    # Full-fit residuals at each vertex
    full_resids = L_mins - (pl_full.x1 * D_stars**pl_full.x2 + pl_full.x0)

    # Compute global y-limits for consistent scales across rows
    # PL column: spans training vertices through forecast point
    all_loo_fcs = [d.loo_fc for d in plot_drops]
    pl_ymin = min(min(L_mins), full_fc, actual, *all_loo_fcs) - 0.05
    pl_ymax = max(L_mins) + 0.15

    # Residuals column: spans all full-fit and LOO residuals
    all_resids = [full_resids]
    for d in plot_drops:
        loo_pred = d.loo_pl.x1 * D_stars**d.loo_pl.x2 + d.loo_pl.x0
        all_resids.append(L_mins - loo_pred)
    res_abs_max = max(np.abs(r).max() for r in all_resids) * 1.1
    res_ylim = (-res_abs_max, res_abs_max)

    for i, drop in enumerate(plot_drops):
        ax_pl = axes[i, 0]
        ax_res = axes[i, 1]
        bl = fmt_budget(drop.budget)
        loo_pl = drop.loo_pl
        loo_curve = loo_pl.x1 * d_curve**loo_pl.x2 + loo_pl.x0

        # --- Left: PL curves ---
        mask = D_stars != drop.d_star
        ax_pl.plot(D_stars[mask], L_mins[mask], "ko", ms=3, zorder=5)
        ax_pl.plot(drop.d_star, drop.l_min, "rx", ms=7, mew=1.5, zorder=6)
        ax_pl.plot(d_curve, full_curve, "k-", lw=1, zorder=3, label="Full")
        ax_pl.plot(
            d_curve,
            loo_curve,
            "--",
            color="tab:blue",
            lw=1,
            zorder=2,
            label=f"LOO ($\\Delta x_2$={loo_pl.x2 - pl_full.x2:+.3f})",
        )
        ax_pl.plot(D_fc, full_fc, "k*", ms=6, zorder=7)
        ax_pl.plot(D_fc, drop.loo_fc, "D", color="tab:blue", ms=4, zorder=7)
        ax_pl.plot(D_fc, actual, "gs", ms=4, zorder=7)
        ax_pl.axvline(D_stars.max(), color="grey", ls=":", lw=0.5, alpha=0.5)
        ax_pl.set_xscale("log")
        ax_pl.set_ylim(pl_ymin, pl_ymax)
        ax_pl.set_ylabel(f"Drop {bl}\n({drop.fc_pct:+.1f}%)", fontsize=8)
        ax_pl.legend(fontsize=6, loc="upper right", ncol=2)
        ax_pl.tick_params(labelsize=7)

        # --- Right: residuals (LOO PL - actual L_min at each vertex) ---
        loo_pred_all = loo_pl.x1 * D_stars**loo_pl.x2 + loo_pl.x0
        loo_resids = L_mins - loo_pred_all

        ax_res.bar(
            range(len(D_stars)),
            full_resids,
            width=0.35,
            align="edge",
            color="black",
            alpha=0.5,
            label="Full",
        )
        ax_res.bar(
            [x + 0.35 for x in range(len(D_stars))],
            loo_resids,
            width=0.35,
            align="edge",
            color="tab:blue",
            alpha=0.5,
            label="LOO",
        )
        # Highlight the dropped budget's bar
        drop_idx = np.argmin(np.abs(D_stars - drop.d_star))
        ax_res.bar(
            drop_idx + 0.35,
            loo_resids[drop_idx],
            width=0.35,
            align="edge",
            color="tab:red",
            alpha=0.7,
        )
        ax_res.axhline(0, color="grey", ls="-", lw=0.5)
        ax_res.set_ylim(res_ylim)
        ax_res.set_xticks(range(len(D_stars)))
        ax_res.set_xticklabels([fmt_budget(d, decimals=0) for d in D_stars], fontsize=6)
        ax_res.tick_params(labelsize=7)
        if i == 0:
            ax_res.legend(fontsize=6, loc="upper right")

    axes[0, 0].set_title("Power Law Fits", fontsize=9, pad=2)
    axes[0, 1].set_title("Residuals ($L_{min}$ \u2212 PL)", fontsize=9, pad=2)
    axes[-1, 0].set_xlabel("Tokens ($D$)", fontsize=9)
    axes[-1, 1].set_xlabel("Budget", fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(top=0.955)
    fig.suptitle("A2 Minima Power Law: LOO by Dropped Budget", fontsize=12)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    delphi = load_delphi()
    fit = fit_dataset(delphi, fit_asymptote=True)
    a2 = fit.a2
    pl_full = fit.minima_pl_a2

    D_stars = a2.D_opts
    L_mins = np.array([f.L_min for f in a2.parabola_fits_D])
    budgets = a2.compute_budgets

    D_fc = FORECAST_TARGETS[1e23].tokens
    full_fc = pl_full.x1 * D_fc**pl_full.x2 + pl_full.x0

    drops = run_loo(pl_full, D_stars, L_mins, budgets)
    print_table(pl_full, drops, full_fc)

    fig = plot_leverage(pl_full, drops, D_stars, L_mins)
    save_figure(fig, OUT_DIR / "pl_leverage.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
