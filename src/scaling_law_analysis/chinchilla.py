"""Chinchilla loss function and parameter recovery implementations.

This module provides:
- The Chinchilla loss function L(N, D) = E + A/N^α + B/D^β
- IsoFLOP sampling along constant compute contours
- Four fitting methods with a standardized (N, D, L, ...) interface:
  - Approach 2: Parabolic IsoFLOP fits → power-law regression
  - Grid Search: 5D exhaustive parameter search
  - VPNLS: Variable Projection with Non-negative Least Squares
  - Approach 3: Direct 5-parameter L-BFGS-B optimization
"""

import enum
import itertools

import numpy as np
from dataclasses import dataclass
from typing import Union

from scipy.optimize import minimize, nnls


# =============================================================================
# Fit status and exceptions
# =============================================================================


class FitStatus(enum.Enum):
    """Outcome of a surface-fitting procedure.

    Non-CONVERGED statuses indicate that the optimizer flagged an issue,
    but a usable parameter estimate was still produced.  Callers should
    inspect ``SurfaceFitResult.status`` rather than relying on exceptions
    for these soft issues.

    Only hard failures (non-finite results, degenerate fits) raise
    ``FitError``; everything else returns normally with status metadata.
    """

    CONVERGED = "converged"
    """Optimizer reported success and all diagnostics passed."""

    MAX_ITER = "max_iter"
    """Optimizer exhausted its iteration budget before meeting convergence
    tolerances.  The result is typically still usable — the optimizer
    simply ran out of budget."""

    ABNORMAL = "abnormal"
    """Optimizer reported ``success=False`` for reasons other than hitting
    the iteration limit.  The most common cause is L-BFGS-B returning
    "ABNORMAL_TERMINATION_IN_LNSRCH" when its line search fails to find
    sufficient decrease along the computed direction.  This happens when
    the objective landscape is extremely flat or ill-conditioned in some
    directions, causing gradient-based line search to break down even
    though the current iterate may be very close to the optimum.  Despite
    the alarming name, the fitted parameters are often still usable."""

    BOUND_HIT = "bound_hit"
    """One or more fitted parameters landed at or near an optimizer bound
    or grid edge.  This suggests the true optimum may lie outside the
    search region.  The estimate is still returned but should be treated
    with caution."""


class FitError(Exception):
    """A fitting procedure failed — no usable result is available.

    Raised for hard failures where no meaningful parameter estimate can
    be constructed: non-finite (NaN/Inf) values, degenerate fits (e.g.
    non-positive curvature in parabola fitting), or underdetermined
    systems.

    Does NOT cover soft issues (max iterations, abnormal termination,
    bound hits) — those return normally via ``SurfaceFitResult.status``.
    Does NOT cover programmer errors (invalid arguments), which remain
    ``ValueError``.
    """


class NonFiniteFitError(FitError):
    """A fitted parameter is NaN or Inf."""


# =============================================================================
# Loss surface model
# =============================================================================

# Chinchilla paper ground truth parameters;
# see "D.2. Approach 3: Parametric fitting of the loss"
CHINCHILLA_PARAMS = {
    "A": 406.4,
    "B": 410.7,
    "E": 1.69,
    "alpha": 0.34,
    "beta": 0.28,
}


@dataclass(frozen=True)
class LossSurface:
    """Configuration for the loss function L(N, D) = E + A/N^α + B/D^β.

    Attributes:
        alpha: Parameter scaling exponent
        beta: Data scaling exponent
        A: Parameter scaling coefficient
        B: Data scaling coefficient
        E: Irreducible loss (entropy of natural text)
    """

    alpha: float
    beta: float
    A: float
    B: float
    E: float

    @property
    def a(self) -> float:
        """N* scaling exponent: a = β/(α+β)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """D* scaling exponent: b = α/(α+β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of alpha to beta (α/β)."""
        return self.alpha / self.beta

    @property
    def G(self) -> float:
        """Scaling constant relating optimal N* and D* to compute.

        From the Chinchilla paper (Appendix A), minimizing L(N,D) subject to
        C = 6ND yields the optimal allocation where:
            N* = G · (C/6)^a
            D* = (1/G) · (C/6)^b

        The constant G is defined as:
            G = (αA / βB)^(1/(α+β))

        This ensures N* · D* = C/6 holds exactly.
        """
        return ((self.alpha * self.A) / (self.beta * self.B)) ** (
            1.0 / (self.alpha + self.beta)
        )

    @property
    def a_intercept(self) -> float:
        """Intercept of log₁₀(N*) vs log₁₀(C) power law.

        From N* = G · (C/6)^a:
            log₁₀(N*) = a·log₁₀(C) + (log₁₀(G) - a·log₁₀(6))

        The intercept is: log₁₀(G) - a·log₁₀(6)
        """
        return np.log10(self.G) - (self.a * np.log10(6))

    @property
    def b_intercept(self) -> float:
        """Intercept of log₁₀(D*) vs log₁₀(C) power law.

        From D* = (1/G) · (C/6)^b:
            log₁₀(D*) = b·log₁₀(C) + (-log₁₀(G) - b·log₁₀(6))

        The intercept is: -log₁₀(G) - b·log₁₀(6)
        """
        return -np.log10(self.G) - (self.b * np.log10(6))

    def N_opt(self, C: float) -> float:
        """Compute optimal parameter count N* for a given compute budget.

        From the Chinchilla paper, the optimal allocation is:
            N* = G · (C/6)^(β/(α+β))

        where G = (αA/βB)^(1/(α+β)) and C = 6ND is the compute approximation.

        Args:
            C: Compute budget in FLOPs

        Returns:
            Optimal number of parameters N*
        """
        return self.G * ((C / 6) ** self.a)

    def D_opt(self, C: float) -> float:
        """Compute optimal token count D* for a given compute budget.

        From the Chinchilla paper, the optimal allocation is:
            D* = (1/G) · (C/6)^(α/(α+β))

        where G = (αA/βB)^(1/(α+β)) and C = 6ND is the compute approximation.

        Args:
            C: Compute budget in FLOPs

        Returns:
            Optimal number of training tokens D*
        """
        return (1 / self.G) * ((C / 6) ** self.b)

    def loss(self, N: float, D: float) -> float:
        """Compute loss L(N, D) = E + A/N^α + B/D^β.

        Args:
            N: Number of parameters
            D: Number of training tokens

        Returns:
            Loss value for the (N, D) pair
        """
        return self.E + (self.A / (N**self.alpha)) + (self.B / (D**self.beta))

    @classmethod
    def from_chinchilla(cls, alpha: float, beta: float) -> "LossSurface":
        """Create a LossSurface with Chinchilla paper A, B, E values.

        Args:
            alpha: Parameter scaling exponent
            beta: Data scaling exponent

        Returns:
            LossSurface with Chinchilla coefficients
        """
        return cls(
            alpha=alpha,
            beta=beta,
            A=CHINCHILLA_PARAMS["A"],
            B=CHINCHILLA_PARAMS["B"],
            E=CHINCHILLA_PARAMS["E"],
        )


# Default loss surface using Chinchilla paper parameters
DEFAULT_LOSS_SURFACE = LossSurface(
    alpha=CHINCHILLA_PARAMS["alpha"],
    beta=CHINCHILLA_PARAMS["beta"],
    A=CHINCHILLA_PARAMS["A"],
    B=CHINCHILLA_PARAMS["B"],
    E=CHINCHILLA_PARAMS["E"],
)


# =============================================================================
# IsoFLOP sampling
# =============================================================================


def compute_center_offset(
    C: float,
    compute_budgets: np.ndarray,
    drift_rate: float,
    center_scale: float,
) -> float:
    """Compute the sampling center offset for a given compute budget.

    Combines two independent effects:
    1. drift_rate: Linear drift in log-compute space starting from zero at the
       lowest compute budget
       - At min compute: offset = 0 (no drift)
       - At max compute: offset = -drift_rate (smaller N)
       This means all budgets except the first undershoot toward smaller N.
    2. center_scale: Constant multiplicative factor applied to all centers
       - scale > 1: all centers shifted left (smaller N)
       - scale < 1: all centers shifted right (larger N)

    Both effects are subtractive in log10 space (positive values shift toward smaller N).

    Args:
        C: Compute budget for which to compute offset
        compute_budgets: Array of all compute budgets (for drift normalization)
        drift_rate: Rate at which sampling center drifts from optimal
        center_scale: Constant multiplier applied to all sampling centers

    Returns:
        Total center offset in log10 units
    """
    offset = 0.0

    # Subtract constant scale offset (scale > 1 shifts left toward smaller N)
    if center_scale != 1.0:
        offset -= np.log10(center_scale)

    # Subtract linear drift offset (0 at min compute, -drift_rate at max compute)
    if drift_rate != 0.0:
        log_C = np.log10(C)
        log_C_min = np.log10(compute_budgets.min())
        log_C_max = np.log10(compute_budgets.max())
        if log_C_max > log_C_min:
            fraction = (log_C - log_C_min) / (log_C_max - log_C_min)
        else:
            fraction = 0.0
        offset -= drift_rate * fraction

    return offset


def isoflop_sample(
    C: float,
    n_points: int,
    log_range: float,
    center_offset: float,
    surface: LossSurface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points along an IsoFLOP contour (constant compute budget).

    Points are sampled logarithmically around the optimal N* for the
    given compute budget. D is derived from the constraint C = 6ND.

    Args:
        C: Compute budget in FLOPs
        n_points: Number of points to sample
        log_range: Range in log10 space around optimal N (±log_range)
        center_offset: Offset in log10 space to shift sampling center from optimal N*
                       Positive values shift center to larger N, negative to smaller N
        surface: Loss surface configuration

    Returns:
        Tuple of (N, D, L) arrays - parameter counts, token counts, and losses
    """
    # Sample N logarithmically around (possibly offset) center
    log_N_center = np.log10(surface.N_opt(C)) + center_offset
    log_N_min = log_N_center - log_range
    log_N_max = log_N_center + log_range
    N = np.logspace(log_N_min, log_N_max, n_points)

    # Derive D from constraint C = 6ND
    D = C / (6 * N)

    # Compute loss at each point
    L = np.array([surface.loss(n, d) for n, d in zip(N, D)])

    return N, D, L


# =============================================================================
# Configuration: grids, bounds, optimizer options
# =============================================================================


@dataclass(frozen=True)
class ExponentGrid:
    """Initialization grid for 2D searches over (alpha, beta)."""

    alpha: np.ndarray
    beta: np.ndarray

    @property
    def total_size(self) -> int:
        return len(self.alpha) * len(self.beta)


@dataclass(frozen=True)
class ParameterGrid:
    """Initialization grid for 5D searches over (E, A, B, alpha, beta)."""

    E: np.ndarray
    A: np.ndarray
    B: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    @property
    def total_size(self) -> int:
        return (
            len(self.E) * len(self.A) * len(self.B) * len(self.alpha) * len(self.beta)
        )


@dataclass(frozen=True)
class SurfaceBounds:
    """Optimization bounds for all 5 surface parameters."""

    E: tuple[float, float] = (1e-6, 10.0)
    A: tuple[float, float] = (1e-6, 1e6)
    B: tuple[float, float] = (1e-6, 1e6)
    alpha: tuple[float, float] = (0.01, 0.99)
    beta: tuple[float, float] = (0.01, 0.99)

    def to_list(self) -> list[tuple[float, float]]:
        """Return bounds in [E, A, B, alpha, beta] order for scipy."""
        return [self.E, self.A, self.B, self.alpha, self.beta]


@dataclass(frozen=True)
class NelderMeadOptions:
    """Options for the Nelder-Mead optimizer."""

    xatol: float = 1e-10
    fatol: float = 1e-15
    maxiter: int = 10_000

    def to_dict(self) -> dict:
        return {"xatol": self.xatol, "fatol": self.fatol, "maxiter": self.maxiter}


@dataclass(frozen=True)
class LBFGSBOptions:
    """Options for the L-BFGS-B optimizer."""

    ftol: float = 1e-15
    gtol: float = 1e-15
    maxiter: int = 10_000
    jac: str | None = None
    eps: float | None = None

    def to_dict(self) -> dict:
        d: dict = {"ftol": self.ftol, "gtol": self.gtol, "maxiter": self.maxiter}
        if self.eps is not None:
            d["eps"] = self.eps
        return d


# Default instances

DEFAULT_EXPONENT_GRID = ExponentGrid(
    alpha=np.linspace(0.05, 0.95, 32),
    beta=np.linspace(0.05, 0.95, 32),
)

DEFAULT_PARAMETER_GRID = ParameterGrid(
    E=np.linspace(0.1, 5.0, 4),
    A=np.logspace(1, 4, 4),
    B=np.logspace(1, 4, 4),
    alpha=np.linspace(0.05, 0.95, 4),
    beta=np.linspace(0.05, 0.95, 4),
)

FINE_EXPONENT_GRID = ExponentGrid(
    alpha=np.linspace(0.05, 0.95, 256),
    beta=np.linspace(0.05, 0.95, 256),
)

DEFAULT_SURFACE_BOUNDS = SurfaceBounds()

DEFAULT_NELDER_MEAD_OPTIONS = NelderMeadOptions()
DEFAULT_LBFGSB_OPTIONS = LBFGSBOptions()

assert DEFAULT_PARAMETER_GRID.total_size == DEFAULT_EXPONENT_GRID.total_size, (
    f"Parameter grid ({DEFAULT_PARAMETER_GRID.total_size}) must equal "
    f"exponent grid ({DEFAULT_EXPONENT_GRID.total_size})"
)

LBFGSB_DEFAULT_EPS = 1e-8


# =============================================================================
# Result types
# =============================================================================


@dataclass(frozen=True)
class PowerLawFit:
    """Results from fitting a power law relationship in log-log space.

    Fits the model: log₁₀(y) = exponent · log₁₀(x) + intercept
    Equivalently: y = 10^intercept · x^exponent
    """

    exponent: float  # Slope in log-log space (power law exponent)
    intercept: float  # Intercept in log₁₀ space


@dataclass(frozen=True)
class ParabolaFit:
    """Results from fitting a parabola to IsoFLOP data."""

    coeffs: np.ndarray  # Polynomial coefficients [a, b, c] for ax² + bx + c
    log_x_opt: float  # Log10 of optimal x from parabola minimum
    x_opt: float  # Optimal x from parabola minimum
    L_min: float  # Minimum loss from parabola


@dataclass(frozen=True)
class ParabolaFitResult:
    """Results from Approach 2 (parabolic IsoFLOP fits → power-law regression).

    Approach 2 fits power laws:
        N* ∝ C^a  where a = β/(α+β)
        D* ∝ C^b  where b = α/(α+β)

    Note: a and b satisfy a + b = 1.
    """

    a: float  # Exponent of N* vs C power law, equals β/(α+β)
    b: float  # Exponent of D* vs C power law, equals α/(α+β)
    a_intercept: float  # Intercept of log(N*) vs log(C) fit
    b_intercept: float  # Intercept of log(D*) vs log(C) fit
    parabola_fits_N: tuple[ParabolaFit, ...]  # Per-budget parabola fits for L vs log(N)
    parabola_fits_D: tuple[ParabolaFit, ...]  # Per-budget parabola fits for L vs log(D)
    compute_budgets: np.ndarray  # Unique compute budgets used
    N_opts: np.ndarray  # Optimal N* from each parabola fit
    D_opts: np.ndarray  # Optimal D* from each parabola fit

    def N_opt(self, C: float) -> float:
        """Compute inferred optimal parameter count N* for a given compute budget.

        Uses the fitted power law from Approach 2:
            log₁₀(N*) = a · log₁₀(C) + a_intercept

        Args:
            C: Compute budget in FLOPs

        Returns:
            Inferred optimal number of parameters N*
        """
        log10_N = self.a * np.log10(C) + self.a_intercept
        return 10**log10_N

    def D_opt(self, C: float) -> float:
        """Compute inferred optimal token count D* for a given compute budget.

        Uses the fitted power law from Approach 2:
            log₁₀(D*) = b · log₁₀(C) + b_intercept

        This can be used to extrapolate D* to compute budgets beyond those
        used for fitting.

        Args:
            C: Compute budget in FLOPs

        Returns:
            Inferred optimal number of training tokens D*
        """
        log10_D = self.b * np.log10(C) + self.b_intercept
        return 10**log10_D


@dataclass(frozen=True)
class SurfaceFitResult:
    """Results from fitting the loss surface L(N, D) = E + A/N^α + B/D^β.

    Returned by fit_grid_search, fit_vpnls, and fit_approach3.

    Attributes:
        E: Fitted irreducible loss
        A: Fitted parameter scaling coefficient
        B: Fitted data scaling coefficient
        alpha: Fitted parameter scaling exponent
        beta: Fitted data scaling exponent
        residual_sum_squares: Sum of squared residuals at optimal fit
        n_points: Number of data points used in fit
        method: Fitting method used (e.g. "nelder-mead", "approach3")
        status: Outcome of the fitting procedure (see FitStatus)
        status_message: Human-readable detail when status is not CONVERGED
    """

    E: float
    A: float
    B: float
    alpha: float
    beta: float
    residual_sum_squares: float
    n_points: int
    method: str = "nelder-mead"
    status: FitStatus = FitStatus.CONVERGED
    status_message: str = ""

    @property
    def a(self) -> float:
        """N* scaling exponent: β/(α+β)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """D* scaling exponent: α/(α+β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def converged(self) -> bool:
        """True if the optimizer reported full convergence."""
        return self.status == FitStatus.CONVERGED

    def to_loss_surface(self) -> LossSurface:
        """Convert fit result to a LossSurface object."""
        return LossSurface(
            alpha=self.alpha,
            beta=self.beta,
            A=self.A,
            B=self.B,
            E=self.E,
        )


# =============================================================================
# Shared utilities (validation, diagnostics, internal grid searches)
# =============================================================================


def _validate_positive_finite(name: str, arr: np.ndarray) -> None:
    """Raise ValueError if *arr* contains non-finite or non-positive values."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive for log-space fitting")


def _validate_ndl_inputs(
    N: np.ndarray, D: np.ndarray, L: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce N, D, L inputs.

    Returns:
        (N, D, L) as float arrays after validation.

    Raises:
        ValueError: If lengths mismatch or values are invalid.
    """
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    if not (len(N) == len(D) == len(L)):
        raise ValueError(
            f"N, D, L must have same length, got {len(N)}, {len(D)}, {len(L)}"
        )
    _validate_positive_finite("N", N)
    _validate_positive_finite("D", D)
    if not np.all(np.isfinite(L)):
        raise ValueError("L contains non-finite values (NaN or Inf)")
    return N, D, L


def _check_at_grid_edge(name: str, idx: int, grid: np.ndarray) -> str | None:
    """Return a message if index is at grid edge, else None."""
    if idx == 0 or idx == len(grid) - 1:
        return (
            f"Best {name}={grid[idx]:.4f} is at grid edge "
            f"[{grid[0]:.2f}, {grid[-1]:.2f}]."
        )
    return None


def _check_at_bounds(
    name: str, val: float, lo: float, hi: float, tol: float = 1e-6
) -> str | None:
    """Return a message if value is at or near bounds, else None."""
    if val - lo < tol:
        return f"Optimized {name}={val:.6f} is at lower bound {lo:.2f}."
    if hi - val < tol:
        return f"Optimized {name}={val:.6f} is at upper bound {hi:.2f}."
    return None


def _check_positive(name: str, val: float, tol: float = 1e-6) -> str | None:
    """Return a message if value is at or near zero, else None."""
    if val < tol:
        return f"Fitted {name}={val:.2e} is at or near zero."
    return None


def _check_finite(**params: float) -> None:
    """Raise if any parameter is NaN or Inf."""
    bad = {k: v for k, v in params.items() if np.isnan(v) or np.isinf(v)}
    if bad:
        raise NonFiniteFitError(f"Optimization produced non-finite values: {bad}")


def fit_power_law(x: np.ndarray, y: np.ndarray) -> PowerLawFit:
    """Fit a power law relationship y = a · x^b via linear regression in log-log space.

    Fits: log₁₀(y) = exponent · log₁₀(x) + intercept

    This is used in Chinchilla Approach 2 to fit:
        N* = 10^a_intercept · C^a  (optimal parameters vs compute)
        D* = 10^b_intercept · C^b  (optimal tokens vs compute)

    Args:
        x: Independent variable (e.g., compute budgets)
        y: Dependent variable (e.g., optimal N* or D* values)

    Returns:
        PowerLawFit with exponent and intercept

    Raises:
        ValueError: If x or y contain non-finite or non-positive values
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    _validate_positive_finite("x", x)
    _validate_positive_finite("y", y)
    log_x = np.log10(x)
    log_y = np.log10(y)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    return PowerLawFit(exponent=float(exponent), intercept=float(intercept))


def fit_parabola(
    log_x: np.ndarray,
    L: np.ndarray,
    min_curvature: float = 1e-10,
) -> ParabolaFit:
    """Fit a parabola to loss vs log-x data.

    Fits L = a*log(x)² + b*log(x) + c and finds the minimum.

    Args:
        log_x: Log10 of x values (e.g., parameter counts N or token counts D)
        L: Loss values
        min_curvature: Minimum value for quadratic coefficient (must be positive
                       for an upward-facing parabola with a minimum).

    Returns:
        ParabolaFit with coefficients and minimum location

    Raises:
        FitError: If fewer than 3 points or curvature is non-positive
    """
    if len(log_x) < 3:
        raise FitError(f"Parabola fit requires at least 3 points, got {len(log_x)}.")
    # Fit quadratic: L = a*log(x)² + b*log(x) + c
    coeffs = np.polyfit(log_x, L, 2)
    a, b, c = coeffs

    # Require positive curvature (upward-facing parabola) to have a minimum
    if a < min_curvature:
        raise FitError(
            f"Parabola fit has non-positive curvature (a={a:.2e}). "
            f"A minimum requires positive curvature (upward-facing parabola). "
            f"This may indicate a flat loss surface, insufficient data, or inverted curvature."
        )

    # Minimum at log(x*) = -b / (2a)
    log_x_opt = -b / (2 * a)
    x_opt = 10**log_x_opt

    # Minimum loss value
    L_min = np.polyval(coeffs, log_x_opt)

    return ParabolaFit(
        coeffs=coeffs,
        log_x_opt=log_x_opt,
        x_opt=x_opt,
        L_min=L_min,
    )


def _compute_rss_and_params(
    alpha: float,
    beta: float,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute RSS and NNLS parameters for given (α, β).

    For fixed (α, β), solves the non-negative least squares problem for (E, A, B).
    NNLS enforces E, A, B >= 0, which is physically required for the loss surface.

    Returns:
        Tuple of (residual_sum_squares, params_array[E, A, B])
    """
    N_neg_alpha = np.exp(-alpha * log_N)
    D_neg_beta = np.exp(-beta * log_D)

    design_matrix = np.column_stack(
        [
            np.ones(len(L)),
            N_neg_alpha,
            D_neg_beta,
        ]
    )

    params, rnorm = nnls(design_matrix, L)
    # nnls returns the 2-norm ||Ax - b||, but we need RSS = ||Ax - b||^2.
    # Using rnorm directly would break Nelder-Mead convergence: at the optimum,
    # RSS ≈ 1e-28 but rnorm ≈ 1e-14, which exceeds fatol=1e-15 and causes
    # the optimizer to run forever trying to improve.
    rss = rnorm**2

    return rss, params


def _cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """Build a (K, N) matrix whose columns are the Cartesian product of K 1-D arrays.

    N is the product of all array lengths. Column order matches nested
    for-loops: the first array varies slowest and the last varies fastest.
    """
    result = np.array(list(itertools.product(*arrays))).T
    assert result.shape == (len(arrays), np.prod([len(a) for a in arrays]))
    return result


def _grid_search_2d(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[int, int]:
    """Find best (α, β) indices via exhaustive 2D grid search."""
    best_rss = np.inf
    best_i, best_j = 0, 0

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            rss, _ = _compute_rss_and_params(alpha, beta, log_N, log_D, L)
            if rss < best_rss:
                best_rss = rss
                best_i, best_j = i, j

    return best_i, best_j


def _grid_search_5d(
    grid: ParameterGrid,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Vectorized 5D grid search for the best Chinchilla surface parameters.

    Evaluates RSS = sum((L - pred)^2) over the Cartesian product of the five
    grids and returns the parameter vector with the lowest RSS.

    Args:
        grid: 5D parameter grid.
        log_N: Log parameter counts (natural log), shape (n_data,).
        log_D: Log token counts (natural log), shape (n_data,).
        L: Loss values, shape (n_data,).

    Returns:
        Tuple of (best_params, best_rss) where best_params is
        a 1-D array [E, A, B, α, β].
    """
    cart = _cartesian_product(grid.E, grid.A, grid.B, grid.alpha, grid.beta)
    preds = (
        cart[0]
        + cart[1] * np.exp(-cart[3] * log_N[:, None])
        + cart[2] * np.exp(-cart[4] * log_D[:, None])
    )  # (n_data, N_grid)
    rss_vals = np.sum((L[:, None] - preds) ** 2, axis=0)
    best_idx = int(np.argmin(rss_vals))
    return cart[:, best_idx], float(rss_vals[best_idx])


def _surface_rss(
    x: np.ndarray, log_N: np.ndarray, log_D: np.ndarray, L: np.ndarray
) -> float:
    """RSS objective for 5-parameter Chinchilla loss surface."""
    E, A, B, alpha, beta = x
    pred = E + A * np.exp(-alpha * log_N) + B * np.exp(-beta * log_D)
    return float(np.sum((L - pred) ** 2))


def _surface_rss_grad(
    x: np.ndarray, log_N: np.ndarray, log_D: np.ndarray, L: np.ndarray
) -> np.ndarray:
    """Analytical gradient of RSS for 5-parameter Chinchilla loss surface."""
    E, A, B, alpha, beta = x
    term_N = np.exp(-alpha * log_N)  # N^(-alpha)
    term_D = np.exp(-beta * log_D)  # D^(-beta)
    pred = E + A * term_N + B * term_D
    resid = pred - L  # (pred - observed)
    # d(RSS)/dx = 2 * sum(resid * d(pred)/dx)
    grad_E = 2 * np.sum(resid)
    grad_A = 2 * np.sum(resid * term_N)
    grad_B = 2 * np.sum(resid * term_D)
    grad_alpha = 2 * np.sum(resid * A * term_N * (-log_N))
    grad_beta = 2 * np.sum(resid * B * term_D * (-log_D))
    return np.array([grad_E, grad_A, grad_B, grad_alpha, grad_beta])


# =============================================================================
# Method: Approach 2 (Parabolic IsoFLOP → Power Law)
# =============================================================================


def fit_approach2(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
) -> ParabolaFitResult:
    """Recover scaling exponents using Chinchilla Approach 2.

    Stage 1: For each compute budget, fit parabolas to find optimal N* and D*
    Stage 2: Fit power laws N* ∝ C^a and D* ∝ C^b

    Args:
        N: Array of parameter counts (one per data point)
        D: Array of token counts (one per data point)
        L: Array of loss values (one per data point)
        C: Array of compute budgets (one per data point, same length as N/D/L).
            Points are grouped by unique values of C.

    Returns:
        ParabolaFitResult with recovered exponents a and b
    """
    N, D, L = _validate_ndl_inputs(N, D, L)
    C = np.asarray(C, dtype=float)
    if len(C) != len(N):
        raise ValueError(f"C must have same length as N/D/L, got {len(C)} vs {len(N)}")

    # Group by unique compute budgets
    unique_C = np.unique(C)

    # Stage 1: Parabola fits to find N* and D* at each compute budget
    parabola_fits_N = []
    parabola_fits_D = []
    N_opts = []
    D_opts = []

    for c in unique_C:
        mask = C == c
        N_group = N[mask]
        D_group = D[mask]
        L_group = L[mask]

        # Fit parabola to L vs log(N) to find N*
        fit_N = fit_parabola(np.log10(N_group), L_group)
        parabola_fits_N.append(fit_N)
        N_opts.append(fit_N.x_opt)

        # Fit parabola to L vs log(D) to find D*
        fit_D = fit_parabola(np.log10(D_group), L_group)
        parabola_fits_D.append(fit_D)
        D_opts.append(fit_D.x_opt)

    N_opts_arr = np.array(N_opts)
    D_opts_arr = np.array(D_opts)

    # Stage 2: Fit power laws N* ∝ C^a and D* ∝ C^b in log-log space
    N_fit = fit_power_law(unique_C, N_opts_arr)
    D_fit = fit_power_law(unique_C, D_opts_arr)

    return ParabolaFitResult(
        a=N_fit.exponent,
        b=D_fit.exponent,
        a_intercept=N_fit.intercept,
        b_intercept=D_fit.intercept,
        parabola_fits_N=tuple(parabola_fits_N),
        parabola_fits_D=tuple(parabola_fits_D),
        compute_budgets=unique_C,
        N_opts=N_opts_arr,
        D_opts=D_opts_arr,
    )


# =============================================================================
# Method: Grid Search (5D Exhaustive)
# =============================================================================


def fit_grid_search(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    grid: ParameterGrid | None = None,
) -> SurfaceFitResult:
    """Fit the loss surface via exhaustive 5D grid search.

    Evaluates RSS = sum((L - pred)^2) over the Cartesian product of the
    parameter grid and returns the best-fitting point.

    Args:
        N: Array of parameter counts
        D: Array of token counts
        L: Array of loss values (same length as N and D)
        grid: 5D parameter grid. Defaults to DEFAULT_PARAMETER_GRID.

    Returns:
        SurfaceFitResult with the best grid point parameters.
    """
    if grid is None:
        grid = DEFAULT_PARAMETER_GRID

    N, D, L = _validate_ndl_inputs(N, D, L)
    log_N = np.log(N)
    log_D = np.log(D)

    best_params, best_rss = _grid_search_5d(grid, log_N, log_D, L)
    E, A, B, alpha, beta = best_params

    return SurfaceFitResult(
        E=float(E),
        A=float(A),
        B=float(B),
        alpha=float(alpha),
        beta=float(beta),
        residual_sum_squares=best_rss,
        n_points=len(N),
        method="grid-search",
        status=FitStatus.CONVERGED,
        status_message="",
    )


# =============================================================================
# Method: VPNLS (Variable Projection + NNLS)
# =============================================================================


def fit_vpnls(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    grid: ExponentGrid | None = None,
    bounds: SurfaceBounds | None = None,
    method: str = "nelder-mead",
    options: Union[NelderMeadOptions, LBFGSBOptions, None] = None,
) -> SurfaceFitResult:
    """Fit the loss surface L(N, D) = E + A/N^α + B/D^β via variable projection.

    All methods search only over (α, β) and solve (E, A, B) via NNLS at each
    candidate, so comparisons isolate the optimizer rather than the parameterization.

    Methods:
        - "nelder-mead": Coarse grid search for initialization, then Nelder-Mead
          refinement. Gradient-free.
        - "l-bfgs-b": Coarse grid search for initialization, then L-BFGS-B
          refinement. Gradient scheme and step size configurable via options.
        - "grid": Grid search only with no local refinement. Pass a high-
          resolution grid (e.g. FINE_EXPONENT_GRID) for better precision.

    Args:
        N: Array of parameter counts
        D: Array of token counts
        L: Array of loss values (same length as N and D)
        grid: Initialization grid for (α, β). Defaults to DEFAULT_EXPONENT_GRID.
        bounds: Parameter bounds for optimization and post-fit checking.
            Defaults to DEFAULT_SURFACE_BOUNDS.
        method: Optimization method — "nelder-mead", "l-bfgs-b", or "grid"
        options: Optimizer options. Pass NelderMeadOptions for "nelder-mead",
            LBFGSBOptions for "l-bfgs-b". Ignored for "grid". If None, uses
            the default options for the chosen method.

    Returns:
        SurfaceFitResult with fitted parameters.  Soft issues (max iterations,
        abnormal termination, bound/grid-edge hits) are reported via
        ``SurfaceFitResult.status`` rather than exceptions.

    Raises:
        ValueError: If arguments are invalid (unknown method, mismatched lengths,
            non-finite or non-positive N/D values)
        NonFiniteFitError: If the fitted parameters contain NaN or Inf
    """
    valid_methods = ("nelder-mead", "l-bfgs-b", "grid")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    if grid is None:
        grid = DEFAULT_EXPONENT_GRID
    if bounds is None:
        bounds = DEFAULT_SURFACE_BOUNDS

    N, D, L = _validate_ndl_inputs(N, D, L)

    alpha_grid = grid.alpha
    beta_grid = grid.beta
    log_N, log_D = np.log(N), np.log(D)

    # Stage 1: Grid search (initialization for local methods, or final for "grid")
    best_i, best_j = _grid_search_2d(alpha_grid, beta_grid, log_N, log_D, L)
    grid_edge_msgs = [
        msg
        for msg in [
            _check_at_grid_edge("α", best_i, alpha_grid),
            _check_at_grid_edge("β", best_j, beta_grid),
        ]
        if msg is not None
    ]

    # Track status for non-optimizer issues (grid edge, bounds)
    status = FitStatus.CONVERGED
    status_message = ""

    if method == "grid":
        alpha = float(alpha_grid[best_i])
        beta = float(beta_grid[best_j])
        if grid_edge_msgs:
            status = FitStatus.BOUND_HIT
            status_message = "; ".join(grid_edge_msgs)

    else:

        def objective(x):
            rss, _ = _compute_rss_and_params(x[0], x[1], log_N, log_D, L)
            return rss

        x0 = [alpha_grid[best_i], beta_grid[best_j]]

        if method == "nelder-mead":
            opts = options if options is not None else DEFAULT_NELDER_MEAD_OPTIONS
            result = minimize(
                objective, x0=x0, method="Nelder-Mead", options=opts.to_dict()
            )
        else:  # l-bfgs-b
            lbfgs_opts: LBFGSBOptions = (
                options
                if isinstance(options, LBFGSBOptions)
                else DEFAULT_LBFGSB_OPTIONS
            )
            opts = lbfgs_opts
            lbfgs_kwargs: dict = {
                "method": "L-BFGS-B",
                "bounds": [bounds.alpha, bounds.beta],
                "options": lbfgs_opts.to_dict(),
            }
            if lbfgs_opts.jac is not None:
                lbfgs_kwargs["jac"] = lbfgs_opts.jac
            result = minimize(objective, x0=x0, **lbfgs_kwargs)

        if result is None:
            raise FitError("Optimizer returned None.")

        status = FitStatus.CONVERGED
        status_message = ""
        if not result.success:
            if result.nit >= opts.maxiter:
                status = FitStatus.MAX_ITER
                status_message = (
                    f"Optimization hit maxiter ({opts.maxiter}): "
                    f"{result.message} (iterations: {result.nit})"
                )
            else:
                status = FitStatus.ABNORMAL
                status_message = (
                    f"Optimization failed: {result.message} "
                    f"(iterations: {result.nit})"
                )

        alpha, beta = float(result.x[0]), float(result.x[1])

    # Extract final parameters at optimized (α, β)
    rss, (E, A, B) = _compute_rss_and_params(alpha, beta, log_N, log_D, L)

    _check_finite(E=E, A=A, B=B, alpha=alpha, beta=beta, rss=rss)

    if status == FitStatus.CONVERGED:
        bound_msgs = [
            msg
            for msg in [
                _check_at_bounds("α", alpha, *bounds.alpha),
                _check_at_bounds("β", beta, *bounds.beta),
                _check_positive("E", E),
                _check_positive("A", A),
                _check_positive("B", B),
            ]
            if msg is not None
        ]
        if bound_msgs:
            status = FitStatus.BOUND_HIT
            status_message = "; ".join(bound_msgs)

    return SurfaceFitResult(
        E=E,
        A=A,
        B=B,
        alpha=alpha,
        beta=beta,
        residual_sum_squares=rss,
        n_points=len(N),
        method=method,
        status=status,
        status_message=status_message,
    )


# =============================================================================
# Method: Approach 3 (Direct 5-parameter L-BFGS-B)
# =============================================================================


def fit_approach3(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    grid: ParameterGrid | None = None,
    bounds: SurfaceBounds | None = None,
    options: LBFGSBOptions | None = None,
    use_grad: bool = True,
    random_init: bool = False,
    rng: np.random.Generator | None = None,
) -> SurfaceFitResult:
    """Fit the loss surface via direct L-BFGS-B over all 5 parameters.

    This is the standard approach used in the Chinchilla paper and others:
    optimize E, A, B, α, β jointly without exploiting linear structure.
    Uses RSS (not Huber loss) for direct comparison with variable projection.

    Initialization is via a coarse 5D grid search, or a single random point
    when ``random_init`` is True.

    Args:
        N: Array of parameter counts
        D: Array of token counts
        L: Array of loss values (same length as N and D)
        grid: Initialization grid. Defaults to DEFAULT_PARAMETER_GRID.
        bounds: Parameter bounds. Defaults to DEFAULT_SURFACE_BOUNDS.
        options: L-BFGS-B options. Defaults to DEFAULT_LBFGSB_OPTIONS.
            When use_grad is False, ``options.jac`` sets the finite-difference
            scheme (e.g. "3-point" for central differences).
        use_grad: If True (default), use analytical gradients. If False,
            use finite-difference gradients via ``options.jac``.
        random_init: If True, skip grid search and use a random starting
            point within bounds. Requires ``rng``.
        rng: Random generator (required when ``random_init`` is True).

    Returns:
        SurfaceFitResult with fitted parameters.  Soft issues (max iterations,
        abnormal termination, bound hits) are reported via
        ``SurfaceFitResult.status`` rather than exceptions.

    Raises:
        ValueError: If inputs have mismatched lengths or contain non-finite /
            non-positive N/D values
        NonFiniteFitError: If the fitted parameters contain NaN or Inf
    """
    if grid is None:
        grid = DEFAULT_PARAMETER_GRID
    if bounds is None:
        bounds = DEFAULT_SURFACE_BOUNDS
    if options is None:
        options = DEFAULT_LBFGSB_OPTIONS

    N, D, L = _validate_ndl_inputs(N, D, L)

    log_N = np.log(N)
    log_D = np.log(D)

    def rss(x: np.ndarray) -> float:
        return _surface_rss(x, log_N, log_D, L)

    def rss_grad(x: np.ndarray) -> np.ndarray:
        return _surface_rss_grad(x, log_N, log_D, L)

    bounds_list = bounds.to_list()

    if random_init:
        if rng is None:
            raise ValueError("rng is required when random_init=True")
        best_x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds_list])
    else:
        best_x0, _ = _grid_search_5d(grid, log_N, log_D, L)

    result = minimize(
        rss,
        x0=best_x0,
        jac=rss_grad if use_grad else options.jac,
        method="L-BFGS-B",
        bounds=bounds_list,
        options=options.to_dict(),
    )

    if result is None:
        raise FitError("Optimizer returned None.")

    status = FitStatus.CONVERGED
    status_message = ""
    if not result.success:
        if result.nit >= options.maxiter:
            status = FitStatus.MAX_ITER
            status_message = (
                f"Optimization hit maxiter ({options.maxiter}): "
                f"{result.message} (iterations: {result.nit})"
            )
        else:
            status = FitStatus.ABNORMAL
            status_message = (
                f"Optimization failed: {result.message} " f"(iterations: {result.nit})"
            )

    E, A, B, alpha, beta = result.x
    final_rss = float(result.fun)
    E, A, B, alpha, beta = float(E), float(A), float(B), float(alpha), float(beta)

    _check_finite(E=E, A=A, B=B, alpha=alpha, beta=beta, rss=final_rss)

    if status == FitStatus.CONVERGED:
        param_names = ["E", "A", "B", "α", "β"]
        bound_msgs = [
            msg
            for name, val, (lo, hi) in zip(
                param_names, [E, A, B, alpha, beta], bounds_list
            )
            if (msg := _check_at_bounds(name, val, lo, hi)) is not None
        ]
        if bound_msgs:
            status = FitStatus.BOUND_HIT
            status_message = "; ".join(bound_msgs)

    return SurfaceFitResult(
        E=E,
        A=A,
        B=B,
        alpha=alpha,
        beta=beta,
        residual_sum_squares=final_rss,
        n_points=len(N),
        method="approach3",
        status=status,
        status_message=status_message,
    )
