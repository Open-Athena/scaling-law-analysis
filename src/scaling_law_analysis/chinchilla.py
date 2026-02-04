"""Chinchilla loss function and parameter recovery implementations.

This module provides:
- The Chinchilla loss function L(N, D) = E + A/N^α + B/D^β
- IsoFLOP sampling along constant compute contours
- Approach 2 parameter recovery via parabolic fits
- Surface fitting via variable projection (grid search + NNLS)
"""

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from scipy.optimize import minimize


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
        return (self.alpha * self.A / (self.beta * self.B)) ** (1 / (self.alpha + self.beta))

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

    def loss(self, N: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Compute loss L(N, D) = E + A/N^α + B/D^β.

        Args:
            N: Number of parameters (can be array)
            D: Number of training tokens (can be array)

        Returns:
            Loss values corresponding to each (N, D) pair
        """
        return self.E + self.A / (N**self.alpha) + self.B / (D**self.beta)

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
        log_C_all = np.log10(compute_budgets)
        log_C_min = log_C_all.min()
        log_C_range = log_C_all.max() - log_C_min

        if log_C_range > 0:
            normalized_log_C = (np.log10(C) - log_C_min) / log_C_range
            offset -= drift_rate * normalized_log_C

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
    L = surface.loss(N, D)

    return N, D, L


class ParabolaFit(NamedTuple):
    """Results from fitting a parabola to IsoFLOP data."""

    coeffs: np.ndarray  # Polynomial coefficients [a, b, c] for ax² + bx + c
    log_x_opt: float  # Log10 of optimal x from parabola minimum
    x_opt: float  # Optimal x from parabola minimum
    L_min: float  # Minimum loss from parabola


@dataclass
class Approach2Result:
    """Results from Approach 2 parameter recovery.
    
    Approach 2 fits power laws:
        N* ∝ C^a  where a = β/(α+β)
        D* ∝ C^b  where b = α/(α+β)
    
    Note: a and b satisfy a + b = 1.
    """

    a: float  # Exponent of N* vs C power law, equals β/(α+β)
    b: float  # Exponent of D* vs C power law, equals α/(α+β)
    a_intercept: float  # Intercept of log(N*) vs log(C) fit
    b_intercept: float  # Intercept of log(D*) vs log(C) fit
    parabola_fits_N: list[ParabolaFit]  # Per-budget parabola fits for L vs log(N)
    parabola_fits_D: list[ParabolaFit]  # Per-budget parabola fits for L vs log(D)
    compute_budgets: np.ndarray  # Compute budgets used
    N_opts: np.ndarray  # Optimal N* from each parabola fit
    D_opts: np.ndarray  # Optimal D* from each parabola fit

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
        return 10 ** log10_D


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
        min_curvature: Minimum absolute value for quadratic coefficient.
                       Raises ValueError if |a| < min_curvature.

    Returns:
        ParabolaFit with coefficients and minimum location

    Raises:
        ValueError: If quadratic coefficient is too small (degenerate/flat parabola)
    """
    # Fit quadratic: L = a*log(x)² + b*log(x) + c
    coeffs = np.polyfit(log_x, L, 2)
    a, b, c = coeffs

    # Check for degenerate parabola (flat loss surface)
    if abs(a) < min_curvature:
        raise ValueError(
            f"Parabola fit has near-zero curvature (a={a:.2e}). "
            f"This indicates a flat loss surface or insufficient data. "
            f"Consider using more points or a narrower sampling range."
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


def fit_approach2(
    compute_budgets: np.ndarray,
    surface: LossSurface,
    drift_rate: float = 0.0,
    center_scale: float = 1.0,
    n_points: int = 15,
    log_range: float = 1.0,
) -> Approach2Result:
    """Recover scaling exponents using Chinchilla Approach 2.

    Stage 1: For each compute budget, fit parabolas to find optimal N* and D*
    Stage 2: Fit power laws N* ∝ C^a and D* ∝ C^b

    Args:
        compute_budgets: Array of compute budgets (FLOPs)
        surface: Loss surface configuration
        drift_rate: Rate at which sampling center drifts from optimal as a
                    function of compute budget. When non-zero, centers are shifted
                    left (smaller N) at all budgets except the lowest, where drift
                    is zero. The drift is linear in log-compute space (0 at min,
                    -drift_rate at max) and measured in log10 units of N.
        center_scale: Constant multiplier applied to all sampling centers.
                      When 1.0, centers are at true optimal N*.
                      When >1, all centers are shifted left (smaller N).
                      When <1, all centers are shifted right (larger N).
        n_points: Number of points per IsoFLOP curve
        log_range: Sampling range in log10 space around optimal N (and D)

    Returns:
        Approach2Result with recovered exponents a and b
    """
    # Stage 1: Parabola fits to find N* and D* at each compute budget
    parabola_fits_N = []
    parabola_fits_D = []
    N_opts = []
    D_opts = []

    for C in compute_budgets:
        center_offset = compute_center_offset(
            C=C,
            compute_budgets=compute_budgets,
            drift_rate=drift_rate,
            center_scale=center_scale,
        )
        N, D, L = isoflop_sample(
            C=C,
            n_points=n_points,
            log_range=log_range,
            center_offset=center_offset,
            surface=surface,
        )

        # Fit parabola to L vs log(N) to find N*
        fit_N = fit_parabola(np.log10(N), L)
        parabola_fits_N.append(fit_N)
        N_opts.append(fit_N.x_opt)

        # Fit parabola to L vs log(D) to find D*
        fit_D = fit_parabola(np.log10(D), L)
        parabola_fits_D.append(fit_D)
        D_opts.append(fit_D.x_opt)

    N_opts = np.array(N_opts)
    D_opts = np.array(D_opts)

    # Stage 2: Linear fits in log-log space
    log_C = np.log10(compute_budgets)
    a, a_intercept = np.polyfit(log_C, np.log10(N_opts), 1)
    b, b_intercept = np.polyfit(log_C, np.log10(D_opts), 1)

    return Approach2Result(
        a=a,
        b=b,
        a_intercept=a_intercept,
        b_intercept=b_intercept,
        parabola_fits_N=parabola_fits_N,
        parabola_fits_D=parabola_fits_D,
        compute_budgets=compute_budgets,
        N_opts=N_opts,
        D_opts=D_opts,
    )


# =============================================================================
# Surface Fitting via Variable Projection
# =============================================================================


@dataclass
class SurfaceFitResult:
    """Results from fitting the loss surface L(N, D) = E + A/N^α + B/D^β.

    Attributes:
        E: Fitted irreducible loss
        A: Fitted parameter scaling coefficient
        B: Fitted data scaling coefficient
        alpha: Fitted parameter scaling exponent
        beta: Fitted data scaling exponent
        residual_sum_squares: Sum of squared residuals at optimal fit
        n_points: Number of data points used in fit
    """

    E: float
    A: float
    B: float
    alpha: float
    beta: float
    residual_sum_squares: float
    n_points: int

    def to_loss_surface(self) -> LossSurface:
        """Convert fit result to a LossSurface object."""
        return LossSurface(
            alpha=self.alpha,
            beta=self.beta,
            A=self.A,
            B=self.B,
            E=self.E,
        )


# Default grid search parameters
DEFAULT_ALPHA_GRID = np.linspace(0.05, 0.95, 256)
DEFAULT_BETA_GRID = np.linspace(0.05, 0.95, 256)


def _compute_rss_and_params(
    alpha: float,
    beta: float,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute RSS and OLS parameters for given (α, β).

    For fixed (α, β), solves the linear least squares problem for (E, A, B).

    Returns:
        Tuple of (residual_sum_squares, params_array[E, A, B])
    """
    N_neg_alpha = np.exp(-alpha * log_N)
    D_neg_beta = np.exp(-beta * log_D)

    design_matrix = np.column_stack([
        np.ones(len(L)),
        N_neg_alpha,
        D_neg_beta,
    ])

    params, _, _, _ = np.linalg.lstsq(design_matrix, L, rcond=None)
    rss = np.sum((L - design_matrix @ params) ** 2)

    return rss, params


def fit_surface(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    alpha_grid: np.ndarray = DEFAULT_ALPHA_GRID,
    beta_grid: np.ndarray = DEFAULT_BETA_GRID,
) -> SurfaceFitResult:
    """Fit the loss surface L(N, D) = E + A/N^α + B/D^β via variable projection.

    Uses a 2D grid search over (α, β) to find a good starting point, then
    refines with local optimization. At each (α, β) evaluation, solves the
    linear least squares problem for (E, A, B).

    Args:
        N: Array of parameter counts
        D: Array of token counts
        L: Array of loss values (same length as N and D)
        alpha_grid: Grid of α values for initial search
        beta_grid: Grid of β values for initial search

    Returns:
        SurfaceFitResult with fitted parameters

    Raises:
        ValueError: If grid search finds best (α, β) at edge (refinement may be unreliable)
    """
    N = np.asarray(N)
    D = np.asarray(D)
    L = np.asarray(L)

    if not (len(N) == len(D) == len(L)):
        raise ValueError(f"N, D, L must have same length, got {len(N)}, {len(D)}, {len(L)}")

    n_points = len(N)

    # Precompute log values for efficiency
    log_N = np.log(N)
    log_D = np.log(D)

    # Stage 1: Grid search to find starting point
    best_rss = np.inf
    best_alpha_idx = None
    best_beta_idx = None

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            rss, _ = _compute_rss_and_params(alpha, beta, log_N, log_D, L)
            if rss < best_rss:
                best_rss = rss
                best_alpha_idx = i
                best_beta_idx = j

    # Diagnostic check: Best (α, β) at grid edge
    for name, idx, grid in [("α", best_alpha_idx, alpha_grid), ("β", best_beta_idx, beta_grid)]:
        if idx == 0 or idx == len(grid) - 1:
            raise ValueError(
                f"Best {name}={grid[idx]:.4f} is at grid edge. "
                f"Consider expanding grid range [{grid[0]:.2f}, {grid[-1]:.2f}]."
            )

    # Stage 2: Local refinement from best grid point
    def objective(x):
        alpha, beta = x
        rss, _ = _compute_rss_and_params(alpha, beta, log_N, log_D, L)
        return rss

    x0 = [alpha_grid[best_alpha_idx], beta_grid[best_beta_idx]]
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[(alpha_grid[0], alpha_grid[-1]), (beta_grid[0], beta_grid[-1])],
    )

    alpha_opt, beta_opt = result.x
    rss_opt, params_opt = _compute_rss_and_params(alpha_opt, beta_opt, log_N, log_D, L)
    E, A, B = params_opt

    return SurfaceFitResult(
        E=E,
        A=A,
        B=B,
        alpha=alpha_opt,
        beta=beta_opt,
        residual_sum_squares=rss_opt,
        n_points=n_points,
    )
