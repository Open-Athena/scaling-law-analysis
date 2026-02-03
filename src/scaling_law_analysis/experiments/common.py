"""Common configuration and utilities for scaling law experiments."""

from dataclasses import dataclass

from scaling_law_analysis.chinchilla import LossSurface, DEFAULT_LOSS_SURFACE


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
