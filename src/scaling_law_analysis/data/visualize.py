"""Visualization for isoflop data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    panel_keys = sorted(
        df[["source", "experiment"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    n_panels = len(panel_keys)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
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
        title = experiment if source == experiment else f"{source} / {experiment}"
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
