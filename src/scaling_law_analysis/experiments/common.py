"""Common configuration and utilities for scaling law experiments."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scaling_law_analysis.chinchilla import LossSurface, DEFAULT_LOSS_SURFACE


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

def prepare_output_dir(output_dir: Path) -> Path:
    """Clear and recreate an output directory for experiment results.

    Args:
        output_dir: Path to the output directory

    Returns:
        The output directory path
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
# Reference: Chinchilla paper values (ratio ≈ 1.21)
REFERENCE_CONFIG = SimulationConfig(
    name="reference",
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

# High imbalance: ratio = 3
HIGH_IMBALANCE_CONFIG = SimulationConfig(
    name="high_imbalance",
    loss=LossSurface.from_chinchilla(*exponents_from_ratio(3)),
)

# Extreme imbalance: ratio = 9
EXTREME_IMBALANCE_CONFIG = SimulationConfig(
    name="extreme_imbalance",
    loss=LossSurface.from_chinchilla(*exponents_from_ratio(9)),
)

# All Experiment 2 configurations
EXP2_CONFIGS = [
    REFERENCE_CONFIG,
    BALANCED_CONFIG,
    SMALL_IMBALANCE_CONFIG,
    MODERATE_IMBALANCE_CONFIG,
    HIGH_IMBALANCE_CONFIG,
    EXTREME_IMBALANCE_CONFIG,
]


# Experiment 3 configurations: Sampling Drift Sensitivity
# Use symmetric loss surface: A=B=400, alpha=beta=0.31
EXP3_LOSS_SURFACE = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)

EXP3_CONFIGS = [
    SimulationConfig(name="baseline", loss=EXP3_LOSS_SURFACE, drift_rate=0.0, center_scale=1.0),
    SimulationConfig(name="drift_0.2", loss=EXP3_LOSS_SURFACE, drift_rate=0.2, center_scale=1.0),
    SimulationConfig(name="drift_0.4", loss=EXP3_LOSS_SURFACE, drift_rate=0.4, center_scale=1.0),
    SimulationConfig(name="scale_1.5", loss=EXP3_LOSS_SURFACE, drift_rate=0.0, center_scale=1.5),
    SimulationConfig(name="scale_2.0", loss=EXP3_LOSS_SURFACE, drift_rate=0.0, center_scale=2.0),
]
