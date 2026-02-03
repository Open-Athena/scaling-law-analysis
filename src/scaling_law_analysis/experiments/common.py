"""Common configuration and utilities for scaling law experiments."""

from dataclasses import dataclass

from scaling_law_analysis.chinchilla import LossSurface, DEFAULT_LOSS_SURFACE, CHINCHILLA_PARAMS


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
    drift_rate: float = 0.05
    center_scale: float = 1.0


# Predefined configurations for Experiment 2
SYMMETRIC_CONFIG = SimulationConfig(
    name="symmetric",
    loss=LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=CHINCHILLA_PARAMS["E"]),
)

BALANCED_CONFIG = SimulationConfig(
    name="balanced",
    loss=LossSurface.from_chinchilla(alpha=0.31, beta=0.31),
)

DEFAULT_CONFIG = SimulationConfig(
    name="default",
    loss=DEFAULT_LOSS_SURFACE,
)

MODERATE_IMBALANCE_CONFIG = SimulationConfig(
    name="moderate_imbalance",
    loss=LossSurface.from_chinchilla(alpha=0.372, beta=0.248),
)

HIGH_IMBALANCE_CONFIG = SimulationConfig(
    name="high_imbalance",
    loss=LossSurface.from_chinchilla(alpha=0.496, beta=0.124),
)

# All Experiment 2 configurations
EXP2_CONFIGS = [
    SYMMETRIC_CONFIG,
    BALANCED_CONFIG,
    DEFAULT_CONFIG,
    MODERATE_IMBALANCE_CONFIG,
    HIGH_IMBALANCE_CONFIG,
]
