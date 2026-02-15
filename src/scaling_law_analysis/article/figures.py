"""Figure generation for the scaling law analysis article.

This module generates clean, publication-ready figures for the blog post.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path

from scaling_law_analysis.chinchilla import (
    LossSurface,
    isoflop_sample,
    fit_approach2,
    compute_center_offset,
    Approach2Result,
)
from scaling_law_analysis.config import prepare_output_dir


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
    figure_dir = prepare_output_dir(output_dir / "happy_path")
    fig_path = figure_dir / "happy_path.png"
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


# Asymmetric surface: ratio = 3 (α/β = 3, keeping α+β = 0.62)
def _exponents_from_ratio(ratio: float) -> tuple[float, float]:
    """Compute alpha and beta from ratio, keeping sum = 0.62."""
    exponent_sum = 0.62
    beta = exponent_sum / (1 + ratio)
    alpha = exponent_sum * ratio / (1 + ratio)
    return alpha, beta


ASYMMETRIC_SURFACE = LossSurface.from_chinchilla(*_exponents_from_ratio(3))


def create_asymmetric_figure(output_dir: Path) -> dict:
    """Create the Asymmetric Surfaces figure for Section 3.

    Shows Chinchilla and Asymmetric surfaces side-by-side.
    Layout: 2 rows x 2 columns
        Row 1: IsoFLOP curves (Chinchilla, Asymmetric)
        Row 2: Power-law fits (Chinchilla, Asymmetric)

    Returns a dict with results for both surfaces.
    """
    setup_style()

    surfaces = [
        ("Chinchilla", CHINCHILLA_SURFACE),
        ("Asymmetric", ASYMMETRIC_SURFACE),
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
    figure_dir = prepare_output_dir(output_dir / "asymmetric")
    fig_path = figure_dir / "asymmetric.png"
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
    center_scale: float = 1.0,
    drift_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute D* errors at evaluation budgets.

    Args:
        surface: Loss surface to use
        training_budgets: Compute budgets for fitting
        eval_budgets: Compute budgets for evaluation
        log_range: Sampling grid width in log10 space
        n_points: Points per IsoFLOP curve
        center_scale: Constant multiplicative offset for sampling centers
        drift_rate: Rate at which sampling center drifts from optimal

    Returns:
        Tuple of (true_D, inferred_D, abs_errors) arrays at each eval budget
    """
    # Fit using Approach 2 on training budgets
    result = fit_approach2(
        compute_budgets=training_budgets,
        surface=surface,
        drift_rate=drift_rate,
        center_scale=center_scale,
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
        ("Asymmetric", ASYMMETRIC_SURFACE),
    ]

    # Extrapolate to a single budget
    EVAL_BUDGET = 1e24
    TRAINING_RANGE = "1e17-1e21"

    # Colors for grid widths (4 grids: XS, S, L, XL)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]  # green, blue, orange, red

    fig, ax = plt.subplots(figsize=(11, 5.5))

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

    # Set y-axis tight to data
    y_data_min = min(p.get_height() for p in ax.patches if hasattr(p, "get_height"))
    ax.set_ylim(y_data_min - 2, 2)

    # Add token annotations for Chinchilla bars, fanned just below bar tips
    chinchilla_min = min(float(ann["bar_bottom"]) for ann in token_annotations)
    ann_y_outer = chinchilla_min - 2.5
    ann_y_inner = chinchilla_min - 5.5
    annotation_ys = [ann_y_outer, ann_y_inner, ann_y_inner, ann_y_outer]
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
    figure_dir = prepare_output_dir(output_dir / "extrapolation_error")
    fig_path = figure_dir / "extrapolation_error.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Export raw data to CSV for transparency
    csv_path = figure_dir / "extrapolation_error_data.csv"
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
# Section 4: Off-Center Sampling — Shared Infrastructure
# =============================================================================

# 16 grid widths from XS (±2×) to XL (±16×)
OFF_CENTER_LOG_RANGES = np.linspace(np.log10(2), np.log10(16), 16)

# Tick positions for error plot x-axes
_TICK_LOG_RANGES = [np.log10(2), np.log10(4), np.log10(8), np.log10(16)]
_TICK_LABELS = ["±2× (XS)", "±4× (S)", "±8× (L)", "±16× (XL)"]


def _create_off_center_bias_figure(
    output_dir: Path,
    drift_rate: float,
    center_scale: float,
    title: str,
    figure_subdir: str,
    fig_filename: str,
    csv_filename: str,
) -> dict:
    """Shared implementation for off-center bias figures (constant and drifting).

    Layout: 2 rows × 2 columns (top row 2× height of bottom row)
        (0,0): IsoFLOP contours at L (±8×) grid
        (0,1): Extrapolation error bar chart (D* at 10²⁴ FLOPs) by grid width
        (1,0): D* exponent error vs grid width (16 points, XS to XL)
        (1,1): D* intercept error vs grid width (16 points, XS to XL)

    Args:
        output_dir: Base directory for output files
        drift_rate: Rate at which sampling center drifts from optimal
        center_scale: Constant multiplier applied to all sampling centers
        title: Figure suptitle
        figure_subdir: Subdirectory name for this figure's outputs
        fig_filename: Output filename for the figure PNG
        csv_filename: Output filename for the CSV data

    Returns:
        Dict with error sweep data and extrapolation results.
    """
    setup_style()

    surface = SYMMETRIC_SURFACE
    compute_budgets = COMPUTE_BUDGETS
    log_ranges = OFF_CENTER_LOG_RANGES

    # --- Compute D* errors across grid widths (for exponent/intercept plots) ---
    b_errors = []
    b_intercept_errors = []

    for lr in log_ranges:
        result = fit_approach2(
            compute_budgets=compute_budgets,
            surface=surface,
            drift_rate=drift_rate,
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

    # --- Compute extrapolation errors at 10²⁴ FLOPs (for bar chart) ---
    EVAL_BUDGET = 1e24
    extrap_data = (
        []
    )  # list of (grid_name, log_range, rel_error_pct, true_D, inferred_D)
    for grid_name, grid_log_range in GRID_WIDTHS:
        true_D, inferred_D, errors = compute_extrapolation_errors(
            surface=surface,
            training_budgets=TRAINING_BUDGETS,
            eval_budgets=np.array([EVAL_BUDGET]),
            log_range=grid_log_range,
            center_scale=center_scale,
            drift_rate=drift_rate,
        )
        rel_error = errors[0] / true_D[0] * 100
        extrap_data.append(
            (grid_name, grid_log_range, rel_error, true_D[0], inferred_D[0])
        )

    # --- IsoFLOP contour data at L (±8×) grid ---
    log_range_display = np.log10(8)
    result_display = fit_approach2(
        compute_budgets=compute_budgets,
        surface=surface,
        drift_rate=drift_rate,
        center_scale=center_scale,
        n_points=N_POINTS,
        log_range=log_range_display,
    )

    # --- Create figure (2x2 layout, top row 2x height of bottom row) ---
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        height_ratios=[2, 1],
    )

    # --- Panel (0,0): IsoFLOP contours (L vs log D) ---
    ax_iso = axes[0, 0]
    viridis_colors = plt.colormaps["viridis"](
        np.linspace(0.1, 0.9, len(compute_budgets))
    )

    for i, (C, fit) in enumerate(zip(compute_budgets, result_display.parabola_fits_D)):
        center_offset = compute_center_offset(
            C=C,
            compute_budgets=compute_budgets,
            drift_rate=drift_rate,
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
        ax_iso.scatter(log_D, L, c=[viridis_colors[i]], s=30, alpha=0.8, zorder=3)

        # Parabola fit
        log_D_fine = np.linspace(log_D.min(), log_D.max(), 100)
        L_fit = np.polyval(fit.coeffs, log_D_fine)
        ax_iso.plot(log_D_fine, L_fit, c=viridis_colors[i], lw=1.5, alpha=0.9)

        # Sampling center marker (black diamond)
        N_center = surface.N_opt(C) * 10**center_offset
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
    ax_iso.set_title("IsoFLOP Curves (±8× grid)")
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

    # --- Panel (0,1): Extrapolation error bar chart ---
    ax_bar = axes[0, 1]
    grid_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    grid_names = [name for name, _, _, _, _ in extrap_data]
    rel_errors = [err for _, _, err, _, _ in extrap_data]
    x_bar = np.arange(len(extrap_data))

    bars = ax_bar.bar(
        x_bar,
        rel_errors,
        color=grid_colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar in bars:
        height = bar.get_height()
        if abs(height) < 0.05:
            pct_label = "0%"
        elif height > 0:
            pct_label = f"+{height:.1f}%"
        else:
            pct_label = f"{height:.1f}%"
        va = "bottom" if height >= 0 else "top"
        y_offset = 0.05 if height >= 0 else -0.05
        ax_bar.annotate(
            pct_label,
            xy=(bar.get_x() + bar.get_width() / 2, height + y_offset),
            ha="center",
            va=va,
            fontsize=12,
            color="black",
        )

    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(grid_names, fontsize=9)
    ax_bar.set_xlabel("Sampling grid width")
    ax_bar.set_ylabel("Relative Error in D* (%)")
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.grid(True, axis="y", alpha=0.3)
    # Add margin around tallest bar for annotation clearance
    if rel_errors:
        y_max_bar = max(rel_errors)
        y_min_bar = min(rel_errors)
        y_top = y_max_bar * 1.15 if y_max_bar > 0 else 2.0
        y_bot = y_min_bar * 1.15 if y_min_bar < 0 else ax_bar.get_ylim()[0]
        ax_bar.set_ylim(
            min(ax_bar.get_ylim()[0], y_bot), max(ax_bar.get_ylim()[1], y_top)
        )
    ax_bar.set_title(
        "Token Prediction Error at $10^{24}$ FLOPs\n"
        f"(Fitting: $10^{{17}}$-$10^{{21}}$ FLOPs)"
    )

    # --- Panel (1,0): D* exponent error ---
    ax_exp = axes[1, 0]
    ax_exp.plot(
        log_ranges, b_errors, "s-", label="b (D* exponent)", color="C2", markersize=4
    )
    ax_exp.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_exp.set_xlabel("Sampling grid width")
    ax_exp.set_ylabel("Relative error (%)")
    ax_exp.set_title("Exponent Error (b)")
    ax_exp.set_xticks(_TICK_LOG_RANGES)
    ax_exp.set_xticklabels(_TICK_LABELS, fontsize=9)
    ax_exp.legend(fontsize=9)
    ax_exp.grid(True, alpha=0.3)

    # --- Panel (1,1): D* intercept error ---
    ax_int = axes[1, 1]
    ax_int.plot(
        log_ranges,
        b_intercept_errors,
        "s-",
        label="b₀ (D* intercept)",
        color="C2",
        markersize=4,
    )
    ax_int.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_int.set_xlabel("Sampling grid width")
    ax_int.set_ylabel("Relative error (%)")
    ax_int.set_title("Intercept Error (b₀)")
    ax_int.set_xticks(_TICK_LOG_RANGES)
    ax_int.set_xticklabels(_TICK_LABELS, fontsize=9)
    ax_int.legend(fontsize=9)
    ax_int.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    # Save figure
    figure_dir = prepare_output_dir(output_dir / figure_subdir)
    fig_path = figure_dir / fig_filename
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Export extrapolation data to CSV
    csv_path = figure_dir / csv_filename
    with open(csv_path, "w") as f:
        f.write(
            "surface,alpha,beta,drift_rate,center_scale,grid_name,log_range,"
            "training_range,eval_budget,"
            "true_D,inferred_D,abs_error,rel_error_pct\n"
        )
        for grid_name, grid_log_range, rel_error, true_D, inferred_D in extrap_data:
            abs_error = inferred_D - true_D
            f.write(
                f"Symmetric,{surface.alpha},{surface.beta},"
                f"{drift_rate},{center_scale},"
                f'"{grid_name}",{grid_log_range},'
                f"1e17-1e21,{EVAL_BUDGET:.0e},"
                f"{true_D:.15e},{inferred_D:.15e},"
                f"{abs_error:.15e},{rel_error:.15f}\n"
            )
    print(f"Saved: {csv_path}")

    return {
        "log_ranges": log_ranges,
        "b_errors": b_errors,
        "b_intercept_errors": b_intercept_errors,
        "extrapolation": extrap_data,
    }


# =============================================================================
# Section 4a: Off-Center Sampling — Constant Multiplicative Bias
# =============================================================================

OFF_CENTER_SCALE = 3.0


def create_off_center_constant_bias_figure(output_dir: Path) -> dict:
    """Create the constant multiplicative bias figure for Section 4.

    Shows that a constant multiplicative bias in sampling centers
    preserves exponents perfectly but biases intercepts, on a
    symmetric surface where asymmetry effects are absent.
    """
    surface = SYMMETRIC_SURFACE
    return _create_off_center_bias_figure(
        output_dir=output_dir,
        drift_rate=0.0,
        center_scale=OFF_CENTER_SCALE,
        title=(
            "Off-Center Sampling: Constant Multiplicative Bias\n"
            f"Symmetric surface ($\\alpha = \\beta = {surface.alpha:.2f}$),"
            f" center offset = {OFF_CENTER_SCALE:.0f}×"
        ),
        figure_subdir="off_center_constant_bias",
        fig_filename="off_center_constant_bias.png",
        csv_filename="off_center_constant_bias_data.csv",
    )


# =============================================================================
# Section 4b: Off-Center Sampling — Drifting Bias
# =============================================================================

# Linear drift to 3×: at max compute, sampling center has drifted to 3× the
# true optimum. drift_rate is in log10 units.
OFF_CENTER_DRIFT_RATE = np.log10(3)


def create_off_center_drifting_bias_figure(output_dir: Path) -> dict:
    """Create the drifting bias figure for Section 4.

    Shows that a compute-dependent drift in sampling centers
    corrupts both exponents and intercepts, on a symmetric surface
    where asymmetry effects are absent.
    """
    surface = SYMMETRIC_SURFACE
    return _create_off_center_bias_figure(
        output_dir=output_dir,
        drift_rate=OFF_CENTER_DRIFT_RATE,
        center_scale=1.0,
        title=(
            "Off-Center Sampling: Drifting Bias\n"
            f"Symmetric surface ($\\alpha = \\beta = {surface.alpha:.2f}$),"
            f" linear drift to {10**OFF_CENTER_DRIFT_RATE:.0f}× at max compute"
        ),
        figure_subdir="off_center_drifting_bias",
        fig_filename="off_center_drifting_bias.png",
        csv_filename="off_center_drifting_bias_data.csv",
    )


# =============================================================================
# Section 5: Method Comparison (Variable Projection Optimizer Analysis)
# =============================================================================


def create_method_comparison_figure(output_dir: Path) -> dict:
    """Create the method comparison figure and companion CSVs.

    Single horizontal dot-range plot: geometric mean of |relative error| pooled
    across all surfaces, sampling ranges, and parameters, with min-max bars.
    Methods with convergence failures are drawn with reduced opacity.
    """
    from scipy.stats import gmean

    from scaling_law_analysis.experiments.common import (
        COMPUTE_BUDGETS as EXP_BUDGETS,
        LOG_RANGES,
        N_POINTS as EXP_N_POINTS,
        LOSS_SURFACES,
    )
    from scaling_law_analysis.experiments.exp5_parametric_surface import (
        METHOD_CONFIGS,
        MethodConfig,
        run_method_comparison,
    )

    setup_style()

    # Run the experiment
    all_results = run_method_comparison(
        compute_budgets=EXP_BUDGETS,
        log_ranges=LOG_RANGES,
        n_points=EXP_N_POINTS,
    )

    surface_names = [name for name, _ in LOSS_SURFACES]
    param_keys = ["E", "A", "B", "alpha", "beta"]
    n_surfaces = len(surface_names)
    n_ranges = len(LOG_RANGES)
    n_params = len(param_keys)
    total_fits = n_surfaces * n_ranges  # 60 attempts per method

    # --- Collect per-method aggregated stats and failure info ---
    method_stats = []  # list of dicts per method
    raw_rows = []  # for raw CSV

    for m_idx, mc in enumerate(METHOD_CONFIGS):
        all_errors = []  # flat pool of |relative error| % values
        total_failures = 0
        surfaces_with_failures: set[str] = set()

        # Per-parameter max errors and success counts (across all surfaces/ranges)
        param_max_errs: dict[str, float] = {pk: 0.0 for pk in param_keys}
        param_n_succeeded: dict[str, int] = {pk: 0 for pk in param_keys}

        for sname in surface_names:
            _, results = all_results[sname][m_idx]
            n_fail = results["n_failures"]
            total_failures += n_fail
            if n_fail > 0:
                surfaces_with_failures.add(sname)

            for range_idx, log_range in enumerate(LOG_RANGES):
                range_failed = results["failed"][range_idx]
                for pk in param_keys:
                    err_val = results[pk][range_idx]
                    abs_err_pct = abs(err_val) * 100
                    raw_rows.append(
                        {
                            "method": mc.label,
                            "surface": sname,
                            "log_range": float(log_range),
                            "parameter": pk,
                            "relative_error_pct": (
                                float(err_val * 100) if not range_failed else ""
                            ),
                            "abs_relative_error_pct": (
                                float(abs_err_pct) if not range_failed else ""
                            ),
                            "convergence_failure": range_failed,
                        }
                    )
                    if not range_failed:
                        all_errors.append(abs_err_pct)
                        param_max_errs[pk] = max(param_max_errs[pk], abs_err_pct)
                        param_n_succeeded[pk] += 1

        if all_errors:
            arr = np.array(all_errors)
            stats = {
                "gmean": float(gmean(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        else:
            stats = {"gmean": np.nan, "min": np.nan, "max": np.nan}

        method_stats.append(
            {
                "mc": mc,
                "stats": stats,
                "total_failures": total_failures,
                "n_surfaces_failed": len(surfaces_with_failures),
                "param_max_errs": param_max_errs,
                "param_n_succeeded": param_n_succeeded,
            }
        )

    # --- Sort by gmean descending (worst at top) ---
    def _sort_key(ms: dict) -> float:
        g: float = ms["stats"]["gmean"]
        return g if not np.isnan(g) else float("inf")

    method_stats.sort(key=_sort_key, reverse=True)

    # --- Build 1×2 figure: dot-range plot (left) + max-error heatmap (right) ---
    n_methods = len(METHOD_CONFIGS)
    y_positions = np.arange(n_methods)

    fig_height = (0.5 * n_methods + 1.5) * 1.0
    fig, (ax_dot, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(12, fig_height),
        gridspec_kw={"width_ratios": [3, 1.5], "wspace": 0.02},
        sharey=True,
        layout="constrained",
    )

    # ---- Left panel: dot-range plot ----
    for m_idx, ms in enumerate(method_stats):
        mc: MethodConfig = ms["mc"]  # type: ignore[assignment]
        stats: dict[str, float] = ms["stats"]  # type: ignore[assignment]
        n_fail: int = ms["total_failures"]  # type: ignore[assignment]
        fillstyle = "full" if n_fail == 0 else "none"
        y = y_positions[m_idx]

        if np.isnan(stats["gmean"]):
            ax_dot.scatter(
                [1e-1],
                [y],
                marker="x",
                s=60,
                color="black",
                zorder=5,
            )
        else:
            xerr_lo = max(0.0, stats["gmean"] - stats["min"])
            xerr_hi = max(0.0, stats["max"] - stats["gmean"])
            ax_dot.errorbar(
                stats["gmean"],
                y,
                xerr=[[xerr_lo], [xerr_hi]],
                fmt="o",
                color="black",
                markersize=7,
                capsize=4,
                linewidth=1.5,
                fillstyle=fillstyle,
                markeredgewidth=1.5,
                zorder=5,
            )

        # Failure count annotation below the marker
        if n_fail > 0:
            ax_dot.annotate(
                f"{n_fail}/{total_fits} fits failed",
                xy=(stats["gmean"] if not np.isnan(stats["gmean"]) else 1e-1, y),
                xytext=(0, -10),
                textcoords="offset points",
                fontsize=9,
                color="#555555",
                ha="center",
                va="top",
            )

        # "Chinchilla Approach 3" callout for the 5D analytical grad method
        if mc.label == "5D L-BFGS-B (analytical grad)" and not np.isnan(stats["gmean"]):
            ax_dot.annotate(
                "(Chinchilla Approach 3)",
                xy=(stats["gmean"], y),
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=9,
                color="#333333",
                ha="center",
                va="bottom",
            )

    # Legend for marker shapes
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            markersize=7,
            linestyle="None",
            label=f"Converged on all fits ({n_ranges} grid widths, "
            f"{n_surfaces} loss surfaces)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            markersize=7,
            markeredgewidth=1.5,
            linestyle="None",
            label="Failed to converge in at least one simulation",
        ),
    ]
    ax_dot.legend(
        handles=legend_handles,
        fontsize=8.5,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    ax_dot.set_xscale("log")
    ax_dot.set_yticks(y_positions)
    ax_dot.set_yticklabels(
        [ms["mc"].label for ms in method_stats],  # type: ignore[union-attr]
        fontsize=10,
    )
    ax_dot.set_xlabel("Absolute relative error (%)", fontsize=11)
    ax_dot.set_title("Geometric Mean Error (min–max range)", fontsize=11)
    ax_dot.grid(True, axis="x", alpha=0.3)
    ax_dot.invert_yaxis()

    # ---- Right panel: max-error heatmap ----
    param_display = {"E": "E", "A": "A", "B": "B", "alpha": "\u03b1", "beta": "\u03b2"}
    col_labels = [param_display[pk] for pk in param_keys]

    # Build the data matrix (n_methods × n_params)
    heat_data = np.full((n_methods, n_params), np.nan)
    for m_idx, ms in enumerate(method_stats):
        pme: dict[str, float] = ms["param_max_errs"]  # type: ignore[assignment]
        for p_idx, pk in enumerate(param_keys):
            val = pme[pk]
            if val > 0:
                heat_data[m_idx, p_idx] = val

    # Determine color scale bounds from finite data
    finite_vals = heat_data[np.isfinite(heat_data)]
    vmin = max(finite_vals.min(), 1e-10) if len(finite_vals) > 0 else 1e-10
    vmax = finite_vals.max() if len(finite_vals) > 0 else 1e2

    # Custom colormap: white → black
    cmap_wb = LinearSegmentedColormap.from_list("white_black", ["#ffffff", "#000000"])

    im = ax_heat.imshow(
        heat_data,
        aspect="auto",
        cmap=cmap_wb,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )

    # Cell text: max error only (no fit count)
    for m_idx, ms in enumerate(method_stats):
        pme: dict[str, float] = ms["param_max_errs"]  # type: ignore[assignment]
        for p_idx, pk in enumerate(param_keys):
            val = pme[pk]
            if val > 0:
                # Format the error value compactly
                if val > 100:
                    txt = ">100%"
                elif val >= 1.0:
                    txt = f"{val:.1f}%"
                elif val >= 0.01:
                    txt = f"{val:.2f}%"
                else:
                    exp = int(np.floor(np.log10(val)))
                    mantissa = val / 10**exp
                    txt = f"{mantissa:.1f}e{exp}%"
            else:
                txt = "—"
            # Choose text color for readability against background
            cell_val = heat_data[m_idx, p_idx]
            if np.isfinite(cell_val):
                log_frac = (np.log10(cell_val) - np.log10(vmin)) / (
                    np.log10(vmax) - np.log10(vmin)
                )
                text_color = "white" if log_frac > 0.5 else "black"
            else:
                text_color = "black"
            ax_heat.text(
                p_idx,
                m_idx,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax_heat.set_xticks(np.arange(n_params))
    ax_heat.set_xticklabels(col_labels, fontsize=11)
    ax_heat.set_xlabel("Estimated parameter", fontsize=11)
    ax_heat.set_title("Max Error by Parameter", fontsize=11)
    ax_heat.tick_params(left=False)  # hide y-axis ticks (shared from left)

    fig.suptitle("Optimizer Comparison: Parameter Recovery Accuracy", fontsize=13)

    # Save figure
    figure_dir = prepare_output_dir(output_dir / "method_comparison")
    fig_path = figure_dir / "method_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # --- Export CSV 1: Raw data ---
    raw_csv_path = figure_dir / "method_comparison_raw.csv"
    with open(raw_csv_path, "w") as f:
        f.write(
            "method,surface,log_range,parameter,relative_error_pct,"
            "abs_relative_error_pct,convergence_failure\n"
        )
        for row in raw_rows:
            f.write(
                f'"{row["method"]}",{row["surface"]},{row["log_range"]:.6f},'
                f'{row["parameter"]},{row["relative_error_pct"]},'
                f'{row["abs_relative_error_pct"]},{row["convergence_failure"]}\n'
            )
    print(f"Saved: {raw_csv_path}")

    # --- Export CSV 2: Max error pivot (method × parameter) ---
    max_err_csv_path = figure_dir / "method_comparison_max_errors.csv"
    with open(max_err_csv_path, "w") as f:
        f.write("method," + ",".join(f"max_{pk}_err_pct" for pk in param_keys) + "\n")
        for m_idx, mc in enumerate(METHOD_CONFIGS):
            max_errs = []
            for pk in param_keys:
                vals = []
                for sname in surface_names:
                    _, results = all_results[sname][m_idx]
                    succeeded = ~results["failed"]
                    if succeeded.any():
                        vals.append(np.max(np.abs(results[pk][succeeded])) * 100)
                max_errs.append(max(vals) if vals else "")
            f.write(
                f'"{mc.label}",'
                + ",".join(
                    f"{v:.6e}" if isinstance(v, float) else str(v) for v in max_errs
                )
                + "\n"
            )
    print(f"Saved: {max_err_csv_path}")

    # --- Export CSV 3: Failure counts pivot ---
    fail_csv_path = figure_dir / "method_comparison_failures.csv"
    with open(fail_csv_path, "w") as f:
        f.write(
            "method,total_failures,total_fits,failure_rate,"
            "surfaces_with_failures,total_surfaces\n"
        )
        for ms in method_stats:
            mc: MethodConfig = ms["mc"]  # type: ignore[assignment]
            n_fail: int = ms["total_failures"]  # type: ignore[assignment]
            n_surf_fail: int = ms["n_surfaces_failed"]  # type: ignore[assignment]
            f.write(
                f'"{mc.label}",{n_fail},{total_fits},'
                f"{n_fail/total_fits:.4f},"
                f"{n_surf_fail},{n_surfaces}\n"
            )
    print(f"Saved: {fail_csv_path}")

    return {"all_results": all_results, "method_stats": method_stats}


def generate_all_figures(output_dir: Path) -> dict:
    """Generate all article figures.

    Each figure's outputs are placed in a dedicated subdirectory under
    output_dir (e.g. output_dir/happy_path/, output_dir/asymmetric/).
    Subdirectories are cleared before each figure is generated.

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

    # Section 4a: Off-Center Sampling — Constant Multiplicative Bias
    data["off_center_constant"] = create_off_center_constant_bias_figure(output_dir)

    # Section 4b: Off-Center Sampling — Drifting Bias
    data["off_center_drifting"] = create_off_center_drifting_bias_figure(output_dir)

    # Section 5: Method Comparison
    data["method_comparison"] = create_method_comparison_figure(output_dir)

    return data


if __name__ == "__main__":
    from scaling_law_analysis import config

    output_dir = config.RESULTS_DIR / "article"
    data = generate_all_figures(output_dir)

    print("\n=== Happy Path ===")
    print(format_comparison_table(data["happy_path"]))

    print("\n=== Asymmetric: Chinchilla ===")
    print(format_comparison_table(data["asymmetric"]["chinchilla"]))

    print("\n=== Asymmetric: Asymmetric ===")
    print(format_comparison_table(data["asymmetric"]["asymmetric"]))

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

    for label, key in [
        ("Off-Center Constant Bias", "off_center_constant"),
        ("Off-Center Drifting Bias", "off_center_drifting"),
    ]:
        print(f"\n=== {label} Summary ===")
        oc = data[key]
        print(f"Exponent error (b): max |error| = {np.abs(oc['b_errors']).max():.2e}%")
        print(
            f"Intercept error (b₀): range [{oc['b_intercept_errors'].min():.2f}%, "
            f"{oc['b_intercept_errors'].max():.2f}%]"
        )
