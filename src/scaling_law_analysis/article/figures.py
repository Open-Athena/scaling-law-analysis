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
    compute_center_offset,
    Approach2Result,
)


# =============================================================================
# Symmetric Surface Configuration
# =============================================================================

SYMMETRIC_SURFACE = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)
COMPUTE_BUDGETS = np.array([1e17, 1e18, 1e19, 1e20, 1e21])
LOG_RANGE = np.log10(16)  # Extra Large (XL) ±16× sampling grid
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
        "true_b_intercept": surface.b_intercept,
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
# Section 3: Asymmetric Surface Configurations
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

        # Store results
        results[name.lower().replace(" ", "_")] = {
            "surface": surface,
            "result": result,
            "true_b": surface.b,
            "true_b_intercept": surface.b_intercept,
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


# =============================================================================
# Section 3b: Extrapolation Error Figure ("Why It Matters")
# =============================================================================

# Sampling grid widths for extrapolation analysis ("Why It Matters" section)
# XS: ±2×, S: ±4×, L: ±8×, XL: ±16×
GRID_WIDTHS = [
    ("XS (±2×)", np.log10(2)),
    ("S (±4×)", np.log10(4)),
    ("L (±8×)", np.log10(8)),
    ("XL (±16×)", np.log10(16)),
]

# Training budget range for fitting
TRAINING_BUDGETS = np.array([1e17, 1e18, 1e19, 1e20, 1e21])


def compute_extrapolation_errors(
    surface: LossSurface,
    training_budgets: np.ndarray,
    eval_budgets: np.ndarray,
    log_range: float,
    n_points: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute D* errors at evaluation budgets.

    Args:
        surface: Loss surface to use
        training_budgets: Compute budgets for fitting
        eval_budgets: Compute budgets for evaluation
        log_range: Sampling grid width in log10 space
        n_points: Points per IsoFLOP curve

    Returns:
        Tuple of (true_D, inferred_D, abs_errors) arrays at each eval budget
    """
    # Fit using Approach 2 on training budgets
    result = fit_approach2(
        compute_budgets=training_budgets,
        surface=surface,
        drift_rate=0.0,
        center_scale=1.0,
        n_points=n_points,
        log_range=log_range,
    )

    # Compute errors at evaluation budgets
    true_D = np.array([surface.D_opt(C) for C in eval_budgets])
    inferred_D = np.array([result.D_opt(C) for C in eval_budgets])
    abs_errors = inferred_D - true_D

    return true_D, inferred_D, abs_errors


def create_extrapolation_error_figure(output_dir: Path) -> dict:
    """Create the extrapolation error figure for "Why It Matters" section.

    Bar chart showing relative error in D* prediction at 10²⁴ FLOPs,
    grouped by loss surface and sampling grid width.

    Also exports raw data to CSV for transparency.

    Returns:
        Dict with error data for each surface and grid width.
    """
    setup_style()

    surfaces = [
        ("Symmetric", SYMMETRIC_SURFACE),
        ("Chinchilla", CHINCHILLA_SURFACE),
        ("High Imbalance", HIGH_IMBALANCE_SURFACE),
    ]

    # Extrapolate to a single budget
    EVAL_BUDGET = 1e24
    TRAINING_RANGE = "1e17-1e21"

    # Colors for grid widths (4 grids: XS, S, L, XL)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]  # green, blue, orange, red

    fig, ax = plt.subplots(figsize=(11, 4))

    results = {}
    x_positions = np.arange(len(surfaces))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    # Collect token annotations to place later with arrows
    token_annotations = []

    for i, ((grid_name, log_range), color, offset) in enumerate(
        zip(GRID_WIDTHS, colors, offsets)
    ):
        rel_errors = []
        annotations = []

        for surface_name, surface in surfaces:
            # Compute error at single budget
            true_D, inferred_D, errors = compute_extrapolation_errors(
                surface=surface,
                training_budgets=TRAINING_BUDGETS,
                eval_budgets=np.array([EVAL_BUDGET]),
                log_range=log_range,
            )

            rel_error = errors[0] / true_D[0] * 100
            rel_errors.append(rel_error)
            annotations.append((true_D[0], inferred_D[0]))

            # Store results
            key = surface_name.lower().replace(" ", "_")
            if key not in results:
                results[key] = {}
            results[key][grid_name] = {
                "true_D": true_D[0],
                "inferred_D": inferred_D[0],
                "rel_error_pct": rel_error,
            }

        # Plot bars
        bars = ax.bar(
            x_positions + offset,
            rel_errors,
            bar_width,
            label=grid_name,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Annotate ALL bars with percentage values at y=0
        for j, (bar, (true_D, inferred_D)) in enumerate(zip(bars, annotations)):
            height = bar.get_height()
            surface_name = surfaces[j][0]

            # Percentage annotation - 1 decimal unless 0
            if abs(height) < 0.05:
                pct_label = "0%"
            else:
                pct_label = f"{height:.1f}%"

            ax.annotate(
                pct_label,
                xy=(bar.get_x() + bar.get_width() / 2, 0.8),
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

            # Collect token count annotations for Chinchilla surface only (all grid widths)
            if surface_name == "Chinchilla":
                # Format token counts with 2 decimal places
                def format_tokens(t):
                    if t >= 1e15:
                        return f"{t/1e15:.2f}Q"  # Quadrillion
                    elif t >= 1e12:
                        return f"{t/1e12:.2f}T"  # Trillion
                    else:
                        return f"{t/1e9:.2f}B"  # Billion

                true_str = format_tokens(true_D)
                inf_str = format_tokens(inferred_D)

                # Store bar position and annotation text for later
                bar_x = bar.get_x() + bar.get_width() / 2
                bar_bottom = height  # Bottom of bar (bars go negative from 0)
                token_annotations.append(
                    {
                        "text": f"$D^*$ = {true_str}\n$\\hat{{D}}^*$ = {inf_str}",
                        "bar_x": bar_x,
                        "bar_bottom": bar_bottom,
                        "grid_name": grid_name,
                    }
                )

    # Styling
    ax.set_xlabel("Loss Surface")
    ax.set_ylabel("Relative Error in D* (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{name}\n(α={s.alpha:.2f}, β={s.beta:.2f})" for name, s in surfaces],
        fontsize=10,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Sampling Grid", loc="lower left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Set y-axis to show negative values prominently
    y_min = min(ax.get_ylim()[0], -55)
    ax.set_ylim(y_min, 5)

    # Add token annotations for Chinchilla in a fan shape below the bars
    # Fan out: spread annotations horizontally below the Chinchilla bars
    annotation_y_outer = -18  # y position for outer (1st, 4th) text boxes
    annotation_y_inner = (
        -28
    )  # y position for inner (2nd, 3rd) text boxes, slightly lower
    annotation_ys = [
        annotation_y_outer,
        annotation_y_inner,
        annotation_y_inner,
        annotation_y_outer,
    ]
    # Fan positions: spread out horizontally for 4 bars
    # Outer annotations pulled inward, inner annotations stay close to their bars
    fan_x_offsets = [-0.30, -0.10, 0.10, 0.30]

    # Inner annotations (S, L grids) get a thicker border to highlight practical ranges
    bbox_styles = [
        dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            linewidth=1,
            alpha=0.9,
        ),
        dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#333333",
            linewidth=1.8,
            alpha=0.95,
        ),
        dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#333333",
            linewidth=1.8,
            alpha=0.95,
        ),
        dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            linewidth=1,
            alpha=0.9,
        ),
    ]

    for idx, ann in enumerate(token_annotations):
        bar_x = float(ann["bar_x"])
        bar_bottom = float(ann["bar_bottom"])
        x_text = bar_x + fan_x_offsets[idx]
        ax.annotate(
            ann["text"],
            xy=(bar_x, bar_bottom),  # Arrow points to bar bottom
            xytext=(x_text, annotation_ys[idx]),  # Text position in fan shape below
            ha="center",
            va="top",
            fontsize=10,
            color="black",
            bbox=bbox_styles[idx],
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                lw=1.2,
            ),
        )

    fig.suptitle(
        "Token Prediction Error by Surface and Grid Width\n"
        f"(Fitting: 10¹⁷-10²¹ FLOPs → Extrapolating to 10²⁴ FLOPs)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout()

    # Save figure
    fig_path = output_dir / "extrapolation_error.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Export raw data to CSV for transparency
    csv_path = output_dir / "extrapolation_error_data.csv"
    with open(csv_path, "w") as f:
        f.write(
            "surface,alpha,beta,grid_name,log_range,training_range,eval_budget,"
            "true_D,inferred_D,abs_error,rel_error_pct\n"
        )
        for surface_name, surface in surfaces:
            key = surface_name.lower().replace(" ", "_")
            for grid_name, log_range in GRID_WIDTHS:
                data = results[key][grid_name]
                true_D = data["true_D"]
                inferred_D = data["inferred_D"]
                abs_error = inferred_D - true_D
                rel_error = data["rel_error_pct"]
                f.write(
                    f"{surface_name},{surface.alpha},{surface.beta},"
                    f'"{grid_name}",{log_range},{TRAINING_RANGE},{EVAL_BUDGET:.0e},'
                    f"{true_D:.15e},{inferred_D:.15e},{abs_error:.15e},{rel_error:.15f}\n"
                )
    print(f"Saved: {csv_path}")

    return results


# =============================================================================
# Section 4: Off-Center Sampling — Constant Multiplicative Bias
# =============================================================================

# 16 grid widths from XS (±2×) to XL (±16×)
OFF_CENTER_LOG_RANGES = np.linspace(np.log10(2), np.log10(16), 16)
OFF_CENTER_SCALE = 2.0


def create_off_center_constant_bias_figure(output_dir: Path) -> dict:
    """Create the constant multiplicative bias figure for Section 4.

    Shows that a constant multiplicative bias in sampling centers
    preserves exponents perfectly but biases intercepts, on a
    symmetric surface where asymmetry effects are absent.

    Layout: 1 row × 3 columns
        Col 1: IsoFLOP contours at L (±8×) with center_scale=2
        Col 2: D* exponent error vs grid width (16 points, XS to XL)
        Col 3: D* intercept error vs grid width (16 points, XS to XL)

    Returns:
        Dict with error sweep data.
    """
    setup_style()

    surface = SYMMETRIC_SURFACE
    compute_budgets = COMPUTE_BUDGETS
    center_scale = OFF_CENTER_SCALE
    log_ranges = OFF_CENTER_LOG_RANGES

    # --- Compute D* errors across grid widths ---
    b_errors = []
    b_intercept_errors = []

    for lr in log_ranges:
        result = fit_approach2(
            compute_budgets=compute_budgets,
            surface=surface,
            drift_rate=0.0,
            center_scale=center_scale,
            n_points=N_POINTS,
            log_range=lr,
        )
        b_errors.append((result.b - surface.b) / surface.b * 100)
        b_intercept_errors.append(
            (result.b_intercept - surface.b_intercept) / abs(surface.b_intercept) * 100
        )

    b_errors = np.array(b_errors)
    b_intercept_errors = np.array(b_intercept_errors)

    # --- IsoFLOP contour data at L (±8×) grid ---
    log_range_display = np.log10(8)
    result_display = fit_approach2(
        compute_budgets=compute_budgets,
        surface=surface,
        drift_rate=0.0,
        center_scale=center_scale,
        n_points=N_POINTS,
        log_range=log_range_display,
    )

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: IsoFLOP contours (L vs log D)
    ax_iso = axes[0]
    colors = plt.colormaps["viridis"](np.linspace(0.1, 0.9, len(compute_budgets)))

    for i, (C, fit) in enumerate(zip(compute_budgets, result_display.parabola_fits_D)):
        center_offset = compute_center_offset(
            C=C,
            compute_budgets=compute_budgets,
            drift_rate=0.0,
            center_scale=center_scale,
        )
        N, D, L = isoflop_sample(
            C=C,
            n_points=N_POINTS,
            log_range=log_range_display,
            center_offset=center_offset,
            surface=surface,
        )
        log_D = np.log10(D)

        # Sampled points
        ax_iso.scatter(log_D, L, c=[colors[i]], s=30, alpha=0.8, zorder=3)

        # Parabola fit
        log_D_fine = np.linspace(log_D.min(), log_D.max(), 100)
        L_fit = np.polyval(fit.coeffs, log_D_fine)
        ax_iso.plot(log_D_fine, L_fit, c=colors[i], lw=1.5, alpha=0.9)

        # Sampling center marker (black diamond)
        N_center = surface.N_opt(C) / center_scale
        D_center = C / (6 * N_center)
        L_center = float(surface.loss(N_center, D_center))
        ax_iso.scatter(
            [np.log10(D_center)],
            [L_center],
            c="black",
            marker="D",
            s=90,
            zorder=5,
            linewidths=1,
            edgecolors="white",
        )

        # True optimum (red X)
        D_true = surface.D_opt(C)
        L_true = float(surface.loss(surface.N_opt(C), D_true))
        ax_iso.scatter(
            [np.log10(D_true)],
            [L_true],
            c="red",
            marker="x",
            s=100,
            zorder=4,
            linewidths=2.5,
        )

        # Inferred optimum (blue +)
        ax_iso.scatter(
            [fit.log_x_opt],
            [fit.L_min],
            c="blue",
            marker="+",
            s=100,
            zorder=4,
            linewidths=2.5,
        )

    ax_iso.set_xlabel(r"$\log_{10}(D)$  [tokens]")
    ax_iso.set_ylabel("Loss")
    ax_iso.set_title(f"IsoFLOP Curves — grid=±8× (Large), offset={center_scale:.0f}×")
    ax_iso.scatter(
        [],
        [],
        c="black",
        marker="D",
        s=100,
        linewidths=1,
        edgecolors="white",
        label="Sampling center",
    )
    ax_iso.scatter([], [], c="red", marker="x", s=80, linewidths=2, label="True $D^*$")
    ax_iso.scatter(
        [], [], c="blue", marker="+", s=80, linewidths=2, label="Inferred $D^*$"
    )
    ax_iso.legend(loc="upper right")

    # Shared y-axis range for error panels (driven by intercept errors)
    all_errors = np.concatenate([b_errors, b_intercept_errors])
    y_margin = max(0.5, 0.15 * (all_errors.max() - all_errors.min()))
    y_min = all_errors.min() - y_margin
    y_max = all_errors.max() + y_margin

    # Tick positions for error plot x-axes
    tick_log_ranges = [np.log10(2), np.log10(4), np.log10(8), np.log10(16)]
    tick_labels = ["±2×", "±4×", "±8×", "±16×"]

    # Panel 2: D* exponent error
    ax_exp = axes[1]
    ax_exp.plot(
        log_ranges, b_errors, "s-", label="b (D* exponent)", color="C2", markersize=4
    )
    ax_exp.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_exp.set_xlabel("Sampling range")
    ax_exp.set_ylabel("Relative error (%)")
    ax_exp.set_title("Exponent Error (b)")
    ax_exp.set_xticks(tick_log_ranges)
    ax_exp.set_xticklabels(tick_labels, fontsize=9)
    ax_exp.set_ylim(y_min, y_max)
    ax_exp.legend(fontsize=9)
    ax_exp.grid(True, alpha=0.3)

    # Panel 3: D* intercept error
    ax_int = axes[2]
    ax_int.plot(
        log_ranges,
        b_intercept_errors,
        "s-",
        label="b₀ (D* intercept)",
        color="C2",
        markersize=4,
    )
    ax_int.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_int.set_xlabel("Sampling range")
    ax_int.set_ylabel("Relative error (%)")
    ax_int.set_title("Intercept Error (b₀)")
    ax_int.set_xticks(tick_log_ranges)
    ax_int.set_xticklabels(tick_labels, fontsize=9)
    ax_int.set_ylim(y_min, y_max)
    ax_int.legend(fontsize=9)
    ax_int.grid(True, alpha=0.3)

    fig.suptitle(
        "Off-Center Sampling: Constant Multiplicative Bias\n"
        f"Symmetric surface ($\\alpha = \\beta = {surface.alpha:.2f}$),"
        f" center offset = {center_scale:.0f}×",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    # Save
    fig_path = output_dir / "off_center_constant_bias.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    return {
        "log_ranges": log_ranges,
        "b_errors": b_errors,
        "b_intercept_errors": b_intercept_errors,
    }


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

    # Section 3b: Extrapolation Error ("Why It Matters")
    data["extrapolation"] = create_extrapolation_error_figure(output_dir)

    # Section 4: Off-Center Sampling — Constant Multiplicative Bias
    data["off_center_constant"] = create_off_center_constant_bias_figure(output_dir)

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

    print("\n=== Extrapolation Error Summary ===")
    for surface_name, surface_data in data["extrapolation"].items():
        print(f"\n{surface_name}:")
        for grid_name, grid_data in surface_data.items():
            true_D = grid_data["true_D"]
            inferred_D = grid_data["inferred_D"]
            rel_error_pct = grid_data["rel_error_pct"]
            abs_error = abs(inferred_D - true_D)
            print(
                f"  {grid_name}: |error| = {abs_error:.2e} tokens ({rel_error_pct:.1f}%)"
            )

    print("\n=== Off-Center Constant Bias Summary ===")
    oc = data["off_center_constant"]
    print(f"Exponent error (b): max |error| = {np.abs(oc['b_errors']).max():.2e}%")
    print(
        f"Intercept error (b₀): range [{oc['b_intercept_errors'].min():.2f}%, "
        f"{oc['b_intercept_errors'].max():.2f}%]"
    )
