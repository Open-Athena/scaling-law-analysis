"""Common configuration and utilities for scaling law experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis.chinchilla import (
    LossSurface,
    DEFAULT_LOSS_SURFACE,
    isoflop_sample,
    compute_center_offset,
)
from scaling_law_analysis.config import prepare_output_dir


# =============================================================================
# Experiment Parameters
# =============================================================================

# Compute budgets (in FLOPs) for IsoFLOP sampling
# Spans 4 orders of magnitude from 10^17 to 10^21
COMPUTE_BUDGETS = np.array([1e17, 1e18, 1e19, 1e20, 1e21])

# Sampling range parameter (log_range) controls how far from optimal N* we sample.
# For a given log_range value, N is sampled from N*/factor to N*×factor,
# where factor = 10^log_range.
#
# Examples:
#   log_range=0.3 → factor≈2   → N spans [N*/2, N*×2]     (±2x)
#   log_range=1.0 → factor=10  → N spans [N*/10, N*×10]   (±10x)
#   log_range=2.0 → factor=100 → N spans [N*/100, N*×100] (±100x)
#
# We sweep from narrow (±2x) to wide (±100x) sampling ranges.
LOG_RANGES = np.linspace(0.3, 2.0, 20)

# Number of points sampled along each IsoFLOP contour
N_POINTS = 15

# X-axis tick positions for plots (in log_range units)
TICK_POSITIONS = [0.3, 0.5, 1.0, 1.5, 2.0]


# =============================================================================
# Utilities
# =============================================================================


def log_range_to_label(log_range: float) -> str:
    """Convert log_range to human-readable N sampling range.

    log_range=1.0 means N spans 10^-1 to 10^1 around optimal = 0.1x to 10x.
    """
    factor = 10**log_range
    if factor >= 10:
        return f"±{factor:.0f}x"
    else:
        return f"±{factor:.1f}x"


@dataclass
class SimulationConfig:
    """Configuration for experiment simulation parameters.

    Attributes:
        name: Human-readable name for the configuration
        loss: Loss surface configuration
        drift_rate: Rate at which sampling center drifts from optimal
        center_scale: Constant multiplier applied to all sampling centers
    """

    name: str
    loss: LossSurface
    drift_rate: float = 0.2
    center_scale: float = 1.0


# Sum of alpha + beta (matches Chinchilla paper: 0.34 + 0.28 = 0.62)
EXPONENT_SUM = 0.62


def exponents_from_ratio(ratio: float) -> tuple[float, float]:
    """Compute alpha and beta from their ratio, keeping sum constant.

    Args:
        ratio: Desired alpha/beta ratio

    Returns:
        Tuple of (alpha, beta) where alpha + beta = EXPONENT_SUM
    """
    beta = EXPONENT_SUM / (1 + ratio)
    alpha = EXPONENT_SUM * ratio / (1 + ratio)
    return alpha, beta


# Predefined configurations for Experiment 2
# Chinchilla: paper values (ratio ≈ 1.21)
CHINCHILLA_CONFIG = SimulationConfig(
    name="chinchilla",
    loss=DEFAULT_LOSS_SURFACE,
)

# Balanced: ratio = 1 (equal exponents)
BALANCED_CONFIG = SimulationConfig(
    name="balanced",
    loss=LossSurface.from_chinchilla(*exponents_from_ratio(1)),
)

# Small imbalance: ratio = 1.5
SMALL_IMBALANCE_CONFIG = SimulationConfig(
    name="small_imbalance",
    loss=LossSurface.from_chinchilla(*exponents_from_ratio(1.5)),
)

# Moderate imbalance: ratio = 2
MODERATE_IMBALANCE_CONFIG = SimulationConfig(
    name="moderate_imbalance",
    loss=LossSurface.from_chinchilla(*exponents_from_ratio(2)),
)

# High imbalance: ratio = 3 (used by Experiment 2 in the imbalance series)
HIGH_IMBALANCE_CONFIG = SimulationConfig(
    name="high_imbalance",
    loss=LossSurface.from_chinchilla(*exponents_from_ratio(3)),
)

# Asymmetric surface: same loss surface as HIGH_IMBALANCE_CONFIG,
# but named "asymmetric" for use in experiments 3-6 and the article.
ASYMMETRIC_CONFIG = SimulationConfig(
    name="asymmetric",
    loss=HIGH_IMBALANCE_CONFIG.loss,
)

# Extreme imbalance: ratio = 9
EXTREME_IMBALANCE_CONFIG = SimulationConfig(
    name="extreme_imbalance",
    loss=LossSurface.from_chinchilla(*exponents_from_ratio(9)),
)

# All Experiment 2 configurations
EXP2_CONFIGS = [
    CHINCHILLA_CONFIG,
    BALANCED_CONFIG,
    SMALL_IMBALANCE_CONFIG,
    MODERATE_IMBALANCE_CONFIG,
    HIGH_IMBALANCE_CONFIG,
    EXTREME_IMBALANCE_CONFIG,
]


# =============================================================================
# Shared Configurations for Experiments 3 & 4
# =============================================================================

# Loss surfaces shared by Experiments 3 & 4
SYMMETRIC_LOSS_SURFACE = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)

LOSS_SURFACES: list[tuple[str, LossSurface]] = [
    ("symmetric", SYMMETRIC_LOSS_SURFACE),
    ("chinchilla", DEFAULT_LOSS_SURFACE),
    ("asymmetric", ASYMMETRIC_CONFIG.loss),
]

# Sampling bias configurations shared by Experiments 3 & 4
# Each tuple is (drift_rate, center_scale, name)
BIAS_CONFIGS: list[tuple[float, float, str]] = [
    (0.0, 1.0, "baseline"),
    (0.2, 1.0, "drift_0.2"),
    (0.4, 1.0, "drift_0.4"),
    (0.0, 1.5, "scale_1.5"),
    (0.0, 2.0, "scale_2.0"),
]

# Display log ranges (narrow, medium, wide) - indices into LOG_RANGES
# Used by Experiments 1 (for display) and 4 (for extrapolation analysis)
DISPLAY_LOG_RANGE_INDICES = [0, len(LOG_RANGES) // 2, -1]
DISPLAY_LOG_RANGES = [LOG_RANGES[i] for i in DISPLAY_LOG_RANGE_INDICES]
DISPLAY_LOG_RANGE_NAMES = ["narrow", "medium", "wide"]


# Legacy aliases for backward compatibility
EXP3_LOSS_SURFACE = SYMMETRIC_LOSS_SURFACE

EXP3_CONFIGS = [
    SimulationConfig(
        name=name,
        loss=SYMMETRIC_LOSS_SURFACE,
        drift_rate=drift_rate,
        center_scale=center_scale,
    )
    for drift_rate, center_scale, name in BIAS_CONFIGS
]


# =============================================================================
# Extrapolation Analysis (shared by Experiments 4 & 5)
# =============================================================================

# Extrapolation compute budgets: 10^22 to 10^25 FLOPs (beyond fitting range)
EXTRAPOLATION_BUDGETS = np.geomspace(1e22, 1e25, 16)


def sample_isoflop_data(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_range: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample IsoFLOP data across all compute budgets.

    Args:
        sim_config: Simulation configuration with loss surface and bias parameters
        compute_budgets: Array of compute budgets (FLOPs)
        log_range: Sampling range in log10 space around optimal N
        n_points: Number of points per IsoFLOP curve

    Returns:
        Tuple of (N, D, L) arrays pooled across all compute budgets
    """
    all_N = []
    all_D = []
    all_L = []

    for C in compute_budgets:
        center_offset = compute_center_offset(
            C=C,
            compute_budgets=compute_budgets,
            drift_rate=sim_config.drift_rate,
            center_scale=sim_config.center_scale,
        )
        N, D, L = isoflop_sample(
            C=C,
            n_points=n_points,
            log_range=log_range,
            center_offset=center_offset,
            surface=sim_config.loss,
        )
        all_N.append(N)
        all_D.append(D)
        all_L.append(L)

    return np.concatenate(all_N), np.concatenate(all_D), np.concatenate(all_L)


# Type alias for extrapolation fitter: takes config and returns D_opt function
ExtrapolationFitter = Callable[
    [
        SimulationConfig,
        np.ndarray,
        float,
        int,
    ],  # sim_config, compute_budgets, log_range, n_points
    Callable[[float], float],  # Returns D_opt(C) function
]


def compute_extrapolation_errors(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    extrapolation_budgets: np.ndarray,
    log_range: float,
    n_points: int,
    fitter: ExtrapolationFitter,
) -> dict:
    """Compute extrapolation errors using a given fitting method.

    Args:
        sim_config: Simulation configuration
        compute_budgets: Compute budgets for fitting (FLOPs)
        extrapolation_budgets: Compute budgets for extrapolation (FLOPs)
        log_range: Sampling range in log10 space
        n_points: Number of points per IsoFLOP curve
        fitter: Function that fits the data and returns a D_opt(C) callable

    Returns:
        Dictionary with extrapolation error results
    """
    loss = sim_config.loss

    # Fit using the provided fitter and get D_opt function
    D_opt_fn = fitter(sim_config, compute_budgets, log_range, n_points)

    # Compute true and inferred D* at each extrapolation budget
    true_D_opts = np.array([loss.D_opt(C) for C in extrapolation_budgets])
    inferred_D_opts = np.array([D_opt_fn(C) for C in extrapolation_budgets])

    # Relative error in D*
    D_rel_errors = (inferred_D_opts - true_D_opts) / true_D_opts

    return {
        "config": sim_config,
        "log_range": log_range,
        "extrapolation_budgets": extrapolation_budgets,
        "true_D_opts": true_D_opts,
        "inferred_D_opts": inferred_D_opts,
        "D_rel_errors": D_rel_errors,
    }


def run_extrapolation_analysis(
    fitter: ExtrapolationFitter,
    compute_budgets: np.ndarray,
    extrapolation_budgets: np.ndarray,
    log_ranges: list[float],
    log_range_names: list[str],
    n_points: int,
    loss_surfaces: list[tuple[str, LossSurface]],
    bias_configs: list[tuple[float, float, str]],
) -> dict[str, dict[str, list[dict]]]:
    """Run extrapolation analysis for all configurations.

    Args:
        fitter: Function that fits the data and returns a D_opt(C) callable
        compute_budgets: Compute budgets for fitting
        extrapolation_budgets: Compute budgets for extrapolation
        log_ranges: Sampling ranges to test
        log_range_names: Names for each sampling range
        n_points: Number of points per IsoFLOP curve
        loss_surfaces: List of (name, LossSurface) tuples
        bias_configs: List of (drift_rate, center_scale, name) tuples

    Returns:
        Nested dict: log_range_name -> surface_name -> list of results
    """
    all_results: dict[str, dict[str, list[dict]]] = {}

    for log_range, range_name in zip(log_ranges, log_range_names):
        print(f"\n{'#' * 70}")
        print(f"Sampling Range: {range_name} ({log_range_to_label(log_range)})")
        print(f"{'#' * 70}")

        all_results[range_name] = {}

        for surface_name, loss in loss_surfaces:
            print(f"\n{'=' * 70}")
            print(f"Loss Surface: {surface_name}")
            print(
                f"  α={loss.alpha:.2f}, β={loss.beta:.2f}, A={loss.A:.1f}, B={loss.B:.1f}"
            )
            print("=" * 70)

            surface_results = []

            for drift_rate, center_scale, bias_name in bias_configs:
                sim_config = SimulationConfig(
                    name=bias_name,
                    loss=loss,
                    drift_rate=drift_rate,
                    center_scale=center_scale,
                )

                print(f"\n  Configuration: {bias_name}")

                results = compute_extrapolation_errors(
                    sim_config=sim_config,
                    compute_budgets=compute_budgets,
                    extrapolation_budgets=extrapolation_budgets,
                    log_range=log_range,
                    n_points=n_points,
                    fitter=fitter,
                )
                surface_results.append(results)

                # Print summary
                print(
                    f"    Max D* error: {np.abs(results['D_rel_errors']).max() * 100:.2f}%"
                )

            all_results[range_name][surface_name] = surface_results

    return all_results


def create_extrapolation_figure(
    all_results: dict[str, dict[str, list[dict]]],
    loss_surfaces: list[tuple[str, LossSurface]],
    log_range_names: list[str],
    log_ranges: list[float],
    title: str,
    subtitle: str,
) -> plt.Figure:
    """Create extrapolation error figure (rows=sampling ranges, cols=loss surfaces).

    Args:
        all_results: Nested dict from run_extrapolation_analysis
        loss_surfaces: List of (name, LossSurface) tuples
        log_range_names: Names for each sampling range
        log_ranges: Sampling range values (for labels)
        title: Main figure title
        subtitle: Subtitle with additional context

    Returns:
        matplotlib Figure
    """
    n_ranges = len(log_range_names)
    n_surfaces = len(loss_surfaces)
    fig, axes = plt.subplots(
        n_ranges,
        n_surfaces,
        figsize=(4.5 * n_surfaces, 2.8 * n_ranges),
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )

    is_bottom_row = n_ranges - 1
    bottom_mid_col = n_surfaces // 2

    for row, (range_name, log_range) in enumerate(zip(log_range_names, log_ranges)):
        range_label = log_range_to_label(log_range)

        for col, (surface_name, loss) in enumerate(loss_surfaces):
            ax = axes[row, col]
            results_list = all_results[range_name][surface_name]
            colors = plt.colormaps["viridis"](np.linspace(0, 0.9, len(results_list)))

            for i, results in enumerate(results_list):
                sim_config = results["config"]
                budgets = results["extrapolation_budgets"]
                D_rel_errors = results["D_rel_errors"] * 100  # Convert to %

                label = sim_config.name
                ax.plot(
                    budgets,
                    D_rel_errors,
                    "o-",
                    color=colors[i],
                    markersize=3,
                    linewidth=1.2,
                    label=label,
                )

            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

            # X-axis: label only on bottom-middle subplot,
            # tick labels only on bottom row
            if row == is_bottom_row and col == bottom_mid_col:
                ax.set_xlabel("Compute budget (FLOPs)", fontsize=9)
            elif row != is_bottom_row:
                ax.tick_params(axis="x", labelbottom=False)

            # Y-axis: label only on left-most subplot per row
            if col == 0:
                ax.set_ylabel("Relative error in D* (%)", fontsize=9)

            # Compact subplot title
            ratio = loss.alpha / loss.beta
            ax.set_title(
                f"{surface_name} / {range_name} ({range_label})\n"
                f"α={loss.alpha:.2f}, β={loss.beta:.2f}, ratio={ratio:.2f}",
                fontsize=8,
                pad=4,
            )

            ax.tick_params(labelsize=8)

            # Show legend only in top-right panel
            if row == 0 and col == n_surfaces - 1:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, y=1.02)

    return fig
