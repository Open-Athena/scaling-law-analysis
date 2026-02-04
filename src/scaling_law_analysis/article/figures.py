"""Figure generation for the scaling law analysis article.

This module generates clean, publication-ready figures for the blog post.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scaling_law_analysis.chinchilla import (
    LossSurface,
    isoflop_sample,
    fit_approach2,
    Approach2Result,
)


# =============================================================================
# Symmetric Surface Configuration
# =============================================================================

SYMMETRIC_SURFACE = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)
COMPUTE_BUDGETS = np.array([1e17, 1e18, 1e19, 1e20, 1e21])
LOG_RANGE = 1.2  # ±16x sampling range
N_POINTS = 15


# =============================================================================
# Figure Styling
# =============================================================================


def setup_style():
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


# =============================================================================
# Section 2: Happy Path Figure (Token Scaling Only)
# =============================================================================


def plot_isoflop_curves_D(
    ax: plt.Axes,
    result: Approach2Result,
    surface: LossSurface,
    compute_budgets: np.ndarray,
    log_range: float,
):
    """Plot IsoFLOP curves with parabola fits (L vs log D).

    Shows sampled points and fitted parabolas for each compute budget,
    with true and inferred D* marked.
    """
    colors = plt.colormaps["viridis"](np.linspace(0.1, 0.9, len(compute_budgets)))

    for i, (C, fit) in enumerate(zip(compute_budgets, result.parabola_fits_D)):
        # Sample data (no offset for ideal case)
        N, D, L = isoflop_sample(
            C=C,
            n_points=N_POINTS,
            log_range=log_range,
            center_offset=0.0,
            surface=surface,
        )
        log_D = np.log10(D)

        # Plot sampled points
        ax.scatter(log_D, L, c=[colors[i]], s=30, alpha=0.8, zorder=3)

        # Plot parabola fit
        log_D_fine = np.linspace(log_D.min(), log_D.max(), 100)
        L_fit = np.polyval(fit.coeffs, log_D_fine)
        ax.plot(log_D_fine, L_fit, c=colors[i], lw=1.5, alpha=0.9)

        # Mark true optimum (red X)
        D_true = surface.D_opt(C)
        L_true = float(surface.loss(surface.N_opt(C), D_true))
        ax.scatter(
            [np.log10(D_true)],
            [L_true],
            c="red",
            marker="x",
            s=100,
            zorder=4,
            linewidths=2.5,
        )

        # Mark inferred optimum (blue +)
        ax.scatter(
            [fit.log_x_opt],
            [fit.L_min],
            c="blue",
            marker="+",
            s=100,
            zorder=4,
            linewidths=2.5,
        )

    ax.set_xlabel(r"$\log_{10}(D)$  [tokens]")
    ax.set_ylabel("Loss")
    ax.set_title("IsoFLOP Curves with Parabola Fits")

    # Legend
    ax.scatter([], [], c="red", marker="x", s=80, linewidths=2, label="True $D^*$")
    ax.scatter([], [], c="blue", marker="+", s=80, linewidths=2, label="Inferred $D^*$")
    ax.legend(loc="upper right")


def plot_power_law_fit_D(
    ax: plt.Axes,
    result: Approach2Result,
    surface: LossSurface,
    compute_budgets: np.ndarray,
):
    """Plot D* vs compute budget with power-law fit.

    Shows inferred D* from parabola fits and the power-law
    regression used to extract scaling exponent b.
    """
    log_C = np.log10(compute_budgets)

    # True optima
    true_D = np.array([surface.D_opt(C) for C in compute_budgets])

    # Color
    color_D = "#2ca02c"  # green

    # Plot D*
    ax.scatter(
        log_C,
        np.log10(result.D_opts),
        c=color_D,
        s=60,
        zorder=3,
        marker="s",
        label="Inferred $D^*$",
    )
    ax.scatter(
        log_C,
        np.log10(true_D),
        c="red",
        marker="x",
        s=80,
        zorder=3,
        linewidths=2,
        label="True $D^*$",
    )

    # D* power-law fit line (solid)
    log_C_fine = np.linspace(log_C.min(), log_C.max(), 100)
    log_D_fit = result.b * log_C_fine + result.b_intercept
    ax.plot(
        log_C_fine,
        log_D_fit,
        c=color_D,
        lw=2,
        alpha=0.9,
        label=f"Fit: $D^* \\propto C^{{{result.b:.4f}}}$",
    )

    # True D* line (dashed)
    ax.plot(
        log_C,
        np.log10(true_D),
        c="red",
        lw=2,
        alpha=0.6,
        linestyle="--",
        label=f"True: $D^* \\propto C^{{{surface.b:.4f}}}$",
    )

    # Labels
    ax.set_xlabel(r"$\log_{10}(C)$  [FLOPs]")
    ax.set_ylabel(r"$\log_{10}(D^*)$  [tokens]")
    ax.set_title(r"Power-Law Fit: $D^* \propto C^b$")

    ax.legend(loc="lower right", fontsize=9)


def compute_true_intercepts(surface: LossSurface) -> tuple[float, float]:
    """Compute true power-law intercepts for N* and D* vs C.

    The true power laws are:
        N* = G · (C/6)^a  →  log10(N*) = a·log10(C) + (log10(G) - a·log10(6))
        D* = (1/G) · (C/6)^b  →  log10(D*) = b·log10(C) + (-log10(G) - b·log10(6))
    """
    log_G = np.log10(surface.G)
    log_6 = np.log10(6)
    true_a_intercept = log_G - surface.a * log_6
    true_b_intercept = -log_G - surface.b * log_6
    return true_a_intercept, true_b_intercept


def create_happy_path_figure(output_dir: Path) -> dict:
    """Create the Happy Path figure for Section 2 (D* only).

    Returns a dict with the results for the comparison table.
    """
    setup_style()

    surface = SYMMETRIC_SURFACE
    compute_budgets = COMPUTE_BUDGETS
    log_range = LOG_RANGE

    # Fit using Approach 2 (no drift, no scale)
    result = fit_approach2(
        compute_budgets=compute_budgets,
        surface=surface,
        drift_rate=0.0,
        center_scale=1.0,
        n_points=N_POINTS,
        log_range=log_range,
    )

    # Compute true intercepts
    true_a_intercept, true_b_intercept = compute_true_intercepts(surface)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: IsoFLOP curves (D*)
    plot_isoflop_curves_D(axes[0], result, surface, compute_budgets, log_range)

    # Right: Power-law fit (D*)
    plot_power_law_fit_D(axes[1], result, surface, compute_budgets)

    fig.suptitle(
        f"Approach 2 on Symmetric Surface: $\\alpha = \\beta = {surface.alpha:.2f}$",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "happy_path.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Return comparison data for table (D* only)
    return {
        "surface": surface,
        "result": result,
        "true_b": surface.b,
        "true_b_intercept": true_b_intercept,
        "inferred_b": result.b,
        "inferred_b_intercept": result.b_intercept,
    }


def format_comparison_table(data: dict) -> str:
    """Format the comparison table as HTML (D* only)."""
    rows = [
        ("b (D* exponent)", data["true_b"], data["inferred_b"]),
        ("b₀ (D* intercept)", data["true_b_intercept"], data["inferred_b_intercept"]),
    ]

    html = """<table class="comparison-table">
    <thead>
        <tr>
            <th>Parameter</th>
            <th>True Value</th>
            <th>Inferred Value</th>
            <th>Relative Error</th>
        </tr>
    </thead>
    <tbody>"""

    for name, true_val, inferred_val in rows:
        rel_error = (inferred_val - true_val) / abs(true_val) * 100
        html += f"""
        <tr>
            <td>{name}</td>
            <td>{true_val:.6f}</td>
            <td>{inferred_val:.6f}</td>
            <td>{rel_error:+.2e}%</td>
        </tr>"""

    html += """
    </tbody>
</table>"""
    return html


# =============================================================================
# Main
# =============================================================================

# Loss surface configurations for asymmetric section
CHINCHILLA_SURFACE = LossSurface(alpha=0.34, beta=0.28, A=406.4, B=410.7, E=1.69)


# High imbalance: ratio = 3 (α/β = 3, keeping α+β = 0.62)
def _exponents_from_ratio(ratio: float) -> tuple[float, float]:
    """Compute alpha and beta from ratio, keeping sum = 0.62."""
    exponent_sum = 0.62
    beta = exponent_sum / (1 + ratio)
    alpha = exponent_sum * ratio / (1 + ratio)
    return alpha, beta


HIGH_IMBALANCE_SURFACE = LossSurface.from_chinchilla(*_exponents_from_ratio(3))


def create_asymmetric_figure(output_dir: Path) -> dict:
    """Create the Asymmetric Surfaces figure for Section 3.

    Shows Chinchilla and High Imbalance surfaces side-by-side.
    Layout: 2 rows x 2 columns
        Row 1: IsoFLOP curves (Chinchilla, High Imbalance)
        Row 2: Power-law fits (Chinchilla, High Imbalance)

    Returns a dict with results for both surfaces.
    """
    setup_style()

    surfaces = [
        ("Chinchilla", CHINCHILLA_SURFACE),
        ("High Imbalance", HIGH_IMBALANCE_SURFACE),
    ]

    compute_budgets = COMPUTE_BUDGETS
    log_range = LOG_RANGE

    results = {}

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, (name, surface) in enumerate(surfaces):
        # Fit using Approach 2 (no drift, no scale)
        result = fit_approach2(
            compute_budgets=compute_budgets,
            surface=surface,
            drift_rate=0.0,
            center_scale=1.0,
            n_points=N_POINTS,
            log_range=log_range,
        )

        # Compute true intercepts
        true_a_intercept, true_b_intercept = compute_true_intercepts(surface)

        # Store results
        results[name.lower().replace(" ", "_")] = {
            "surface": surface,
            "result": result,
            "true_b": surface.b,
            "true_b_intercept": true_b_intercept,
            "inferred_b": result.b,
            "inferred_b_intercept": result.b_intercept,
        }

        # Row 1: IsoFLOP curves
        ax_iso = axes[0, col]
        plot_isoflop_curves_D(ax_iso, result, surface, compute_budgets, log_range)
        ax_iso.set_title(
            f"{name}: IsoFLOP Curves\n($\\alpha$={surface.alpha:.2f}, $\\beta$={surface.beta:.2f})"
        )

        # Row 2: Power-law fits
        ax_pow = axes[1, col]
        plot_power_law_fit_D(ax_pow, result, surface, compute_budgets)
        ax_pow.set_title(f"{name}: Power-Law Fit")

    fig.suptitle(
        "Approach 2 on Asymmetric Surfaces",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()

    # Save
    fig_path = output_dir / "asymmetric.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    return results


def generate_all_figures(output_dir: Path) -> dict:
    """Generate all article figures.

    Returns dict of data needed for the HTML article.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {}

    # Section 2: Happy Path
    data["happy_path"] = create_happy_path_figure(output_dir)

    # Section 3: Asymmetric Surfaces
    data["asymmetric"] = create_asymmetric_figure(output_dir)

    return data


if __name__ == "__main__":
    from scaling_law_analysis import config

    output_dir = config.RESULTS_DIR / "article"
    data = generate_all_figures(output_dir)

    print("\n=== Happy Path ===")
    print(format_comparison_table(data["happy_path"]))

    print("\n=== Asymmetric: Chinchilla ===")
    print(format_comparison_table(data["asymmetric"]["chinchilla"]))

    print("\n=== Asymmetric: High Imbalance ===")
    print(format_comparison_table(data["asymmetric"]["high_imbalance"]))
