"""Chinchilla loss function and parameter recovery implementations.

This module provides:
- The Chinchilla loss function L(N, D) = E + A/N^α + B/D^β
- IsoFLOP sampling along constant compute contours
- Approach 2 parameter recovery via parabolic fits
- Surface fitting via variable projection (grid search + NNLS)
"""

import numpy as np
from dataclasses import dataclass
from typing import Union

from scipy.optimize import minimize, nnls


class FitError(Exception):
    """A fitting or optimization procedure failed to produce a valid result.

    This covers all runtime failure modes: optimizer non-convergence, solutions
    at parameter bounds or grid edges, non-finite values, and degenerate fits
    (e.g. non-positive curvature in parabola fitting). It does NOT cover
    programmer errors like invalid arguments, which remain ValueError.
    """


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
        return (self.alpha * self.A / (self.beta * self.B)) ** (
            1 / (self.alpha + self.beta)
        )

    @property
    def a_intercept(self) -> float:
        """Intercept of log₁₀(N*) vs log₁₀(C) power law.

        From N* = G · (C/6)^a:
            log₁₀(N*) = a·log₁₀(C) + (log₁₀(G) - a·log₁₀(6))

        The intercept is: log₁₀(G) - a·log₁₀(6)
        """
        return np.log10(self.G) - self.a * np.log10(6)

    @property
    def b_intercept(self) -> float:
        """Intercept of log₁₀(D*) vs log₁₀(C) power law.

        From D* = (1/G) · (C/6)^b:
            log₁₀(D*) = b·log₁₀(C) + (-log₁₀(G) - b·log₁₀(6))

        The intercept is: -log₁₀(G) - b·log₁₀(6)
        """
        return -np.log10(self.G) - self.b * np.log10(6)

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

    def loss(
        self, N: Union[float, np.ndarray], D: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute loss L(N, D) = E + A/N^α + B/D^β.

        Args:
            N: Number of parameters (scalar or array)
            D: Number of training tokens (scalar or array)

        Returns:
            Loss values corresponding to each (N, D) pair
        """
        return self.E + self.A / (N**self.alpha) + self.B / (D**self.beta)  # type: ignore[operator]

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
    if not isinstance(L, np.ndarray):
        L = np.array([L])

    return N, D, L


@dataclass(frozen=True)
class PowerLawFit:
    """Results from fitting a power law relationship in log-log space.

    Fits the model: log₁₀(y) = exponent · log₁₀(x) + intercept
    Equivalently: y = 10^intercept · x^exponent
    """

    exponent: float  # Slope in log-log space (power law exponent)
    intercept: float  # Intercept in log₁₀ space


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
    """
    log_x = np.log10(x)
    log_y = np.log10(y)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    return PowerLawFit(exponent=float(exponent), intercept=float(intercept))


@dataclass(frozen=True)
class ParabolaFit:
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
        return 10**log10_D


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
        ValueError: If quadratic coefficient is not sufficiently positive
    """
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

    # Stage 2: Fit power laws N* ∝ C^a and D* ∝ C^b in log-log space
    N_fit = fit_power_law(compute_budgets, N_opts)
    D_fit = fit_power_law(compute_budgets, D_opts)

    return Approach2Result(
        a=N_fit.exponent,
        b=D_fit.exponent,
        a_intercept=N_fit.intercept,
        b_intercept=D_fit.intercept,
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
    method: str = "nelder-mead"

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
# Coarse grid is sufficient since Nelder-Mead refinement finds the true optimum.
# 32x32 gives ~56x speedup over 256x256 with identical accuracy.
DEFAULT_ALPHA_GRID = np.linspace(0.05, 0.95, 32)
DEFAULT_BETA_GRID = np.linspace(0.05, 0.95, 32)

# Fine grid for grid-only search (no local refinement).
# 256x256 = 65,536 evaluations — higher resolution to compensate for
# the lack of a continuous optimizer.
FINE_ALPHA_GRID = np.linspace(0.05, 0.95, 256)
FINE_BETA_GRID = np.linspace(0.05, 0.95, 256)


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


def _check_at_grid_edge(name: str, idx: int, grid: np.ndarray) -> None:
    """Raise if index is at grid edge."""
    if idx == 0 or idx == len(grid) - 1:
        raise FitError(
            f"Best {name}={grid[idx]:.4f} is at grid edge. "
            f"Consider expanding grid range [{grid[0]:.2f}, {grid[-1]:.2f}]."
        )


def _check_at_bounds(
    name: str, val: float, lo: float, hi: float, tol: float = 1e-6
) -> None:
    """Raise if value is at or near bounds."""
    if val - lo < tol:
        raise FitError(f"Optimized {name}={val:.6f} is at lower bound {lo:.2f}.")
    if hi - val < tol:
        raise FitError(f"Optimized {name}={val:.6f} is at upper bound {hi:.2f}.")


def _check_positive(name: str, val: float, tol: float = 1e-6) -> None:
    """Raise if value is at or near zero."""
    if val < tol:
        raise FitError(f"Fitted {name}={val:.2e} is at or near zero.")


def _check_finite(**params: float) -> None:
    """Raise if any parameter is NaN or Inf."""
    bad = {k: v for k, v in params.items() if np.isnan(v) or np.isinf(v)}
    if bad:
        raise FitError(f"Optimization produced non-finite values: {bad}")


def _grid_search(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[int, int]:
    """Find best (α, β) indices via exhaustive grid search."""
    best_rss = np.inf
    best_i, best_j = 0, 0

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            rss, _ = _compute_rss_and_params(alpha, beta, log_N, log_D, L)
            if rss < best_rss:
                best_rss = rss
                best_i, best_j = i, j

    return best_i, best_j


# Nelder-Mead optimizer rationale:
#
# We use Nelder-Mead rather than L-BFGS-B because:
# - L-BFGS-B uses numerical gradients which have limited precision (~1e-8)
# - L-BFGS-B's line search fails sporadically on this problem, returning
#   success=False with message "ABNORMAL" after only a few iterations.
#   This occurred at specific sampling ranges (e.g., log_range=1.64, 1.73)
#   but not others, even with identical tolerance settings. The failure
#   appears related to the extremely flat RSS surface near the optimum
#   where numerical gradients become unreliable.
# - Nelder-Mead is gradient-free and reliably converges for 2D optimization
#
# Tolerances are set tight because we're fitting noise-free synthetic data
# where the true minimum has RSS ≈ 0:
# - xatol=1e-10: stop when parameters change by less than this between iterations
# - fatol=1e-15: stop when RSS changes by less than this between iterations;
#   set near machine epsilon (~2e-16) because RSS→0 at the true optimum
NELDER_MEAD_OPTIONS = {"xatol": 1e-10, "fatol": 1e-15, "maxiter": 10000}

# L-BFGS-B uses numerical finite-difference gradients.
# Default step size is eps=1e-8 (scipy's default for L-BFGS-B).
# Tolerances are set tight to match Nelder-Mead intent, though gradient
# precision may limit actual convergence.
LBFGSB_DEFAULT_EPS = 1e-8
LBFGSB_OPTIONS = {"ftol": 1e-15, "gtol": 1e-15, "maxiter": 10000}


def fit_surface(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    alpha_grid: np.ndarray = DEFAULT_ALPHA_GRID,
    beta_grid: np.ndarray = DEFAULT_BETA_GRID,
    method: str = "nelder-mead",
    jac: str | None = None,
    eps: float | None = None,
) -> SurfaceFitResult:
    """Fit the loss surface L(N, D) = E + A/N^α + B/D^β via variable projection.

    All methods search only over (α, β) and solve (E, A, B) via NNLS at each
    candidate, so comparisons isolate the optimizer rather than the parameterization.

    Methods:
        - "nelder-mead": Coarse grid search for initialization, then Nelder-Mead
          refinement. Gradient-free.
        - "l-bfgs-b": Coarse grid search for initialization, then L-BFGS-B
          refinement. Gradient scheme and step size configurable via jac and eps.
        - "grid": Grid search only with no local refinement. The caller-provided
          alpha_grid/beta_grid define the search resolution directly (use
          FINE_ALPHA_GRID/FINE_BETA_GRID for higher resolution).

    Args:
        N: Array of parameter counts
        D: Array of token counts
        L: Array of loss values (same length as N and D)
        alpha_grid: Grid of α values (init grid for local methods, search grid for "grid")
        beta_grid: Grid of β values (init grid for local methods, search grid for "grid")
        method: Optimization method — "nelder-mead", "l-bfgs-b", or "grid"
        jac: Jacobian scheme for L-BFGS-B (e.g. "3-point" for central differences).
            If None, scipy uses forward differences. Ignored for other methods.
        eps: Override finite-difference step size for L-BFGS-B.
            If None, uses scipy default (1e-8). Ignored for other methods.

    Returns:
        SurfaceFitResult with fitted parameters

    Raises:
        ValueError: If any diagnostic check fails (grid edge, bounds, convergence, etc.)
    """
    valid_methods = ("nelder-mead", "l-bfgs-b", "grid")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    N, D, L = np.asarray(N), np.asarray(D), np.asarray(L)
    if not (len(N) == len(D) == len(L)):
        raise ValueError(
            f"N, D, L must have same length, got {len(N)}, {len(D)}, {len(L)}"
        )

    log_N, log_D = np.log(N), np.log(D)

    # Stage 1: Grid search (initialization for local methods, or final for "grid")
    best_i, best_j = _grid_search(alpha_grid, beta_grid, log_N, log_D, L)
    _check_at_grid_edge("α", best_i, alpha_grid)
    _check_at_grid_edge("β", best_j, beta_grid)

    if method == "grid":
        # Grid search only — no local refinement
        alpha = float(alpha_grid[best_i])
        beta = float(beta_grid[best_j])

    else:
        # Local refinement from best grid point
        def objective(x):
            rss, _ = _compute_rss_and_params(x[0], x[1], log_N, log_D, L)
            return rss

        x0 = [alpha_grid[best_i], beta_grid[best_j]]

        if method == "nelder-mead":
            result = minimize(
                objective,
                x0=x0,
                method="Nelder-Mead",
                options=NELDER_MEAD_OPTIONS,
            )
        else:  # l-bfgs-b
            lbfgs_kwargs: dict = {
                "method": "L-BFGS-B",
                "bounds": [
                    (alpha_grid[0], alpha_grid[-1]),
                    (beta_grid[0], beta_grid[-1]),
                ],
                "options": dict(LBFGSB_OPTIONS),
            }
            if jac is not None:
                lbfgs_kwargs["jac"] = jac
            if eps is not None:
                lbfgs_kwargs["options"]["eps"] = eps
            result = minimize(objective, x0=x0, **lbfgs_kwargs)

        if result is None or not result.success:
            message = result.message if result else "Unknown error"
            nit = result.nit if result else 0
            raise FitError(f"Optimization failed: {message} (iterations: {nit})")

        alpha, beta = float(result.x[0]), float(result.x[1])

    # Extract final parameters at optimized (α, β)
    rss, (E, A, B) = _compute_rss_and_params(alpha, beta, log_N, log_D, L)

    # Diagnostic checks
    _check_at_bounds("α", alpha, alpha_grid[0], alpha_grid[-1])
    _check_at_bounds("β", beta, beta_grid[0], beta_grid[-1])
    _check_positive("E", E)
    _check_positive("A", A)
    _check_positive("B", B)
    _check_finite(E=E, A=A, B=B, alpha=alpha, beta=beta, rss=rss)

    return SurfaceFitResult(
        E=E,
        A=A,
        B=B,
        alpha=alpha,
        beta=beta,
        residual_sum_squares=rss,
        n_points=len(N),
        method=method,
    )


# =============================================================================
# Approach 3: Direct nonlinear optimization over all 5 parameters
# =============================================================================

# Coarse 5D grid for initialization (8 values per parameter = 8^5 = 32768 points)
_A3_ALPHA_GRID = np.linspace(0.05, 0.95, 8)
_A3_BETA_GRID = np.linspace(0.05, 0.95, 8)
_A3_E_GRID = np.linspace(0.1, 5.0, 8)
_A3_A_GRID = np.logspace(1, 4, 8)  # 10 to 10000
_A3_B_GRID = np.logspace(1, 4, 8)  # 10 to 10000


def _a3_rss(
    x: np.ndarray, log_N: np.ndarray, log_D: np.ndarray, L: np.ndarray
) -> float:
    """RSS objective for Approach 3 (5-parameter Chinchilla surface)."""
    E, A, B, alpha, beta = x
    pred = E + A * np.exp(-alpha * log_N) + B * np.exp(-beta * log_D)
    return float(np.sum((L - pred) ** 2))


def _a3_rss_grad(
    x: np.ndarray, log_N: np.ndarray, log_D: np.ndarray, L: np.ndarray
) -> np.ndarray:
    """Analytical gradient of RSS for Approach 3."""
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


def fit_approach3(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    use_grad: bool = True,
    jac: str | None = None,
) -> SurfaceFitResult:
    """Fit the loss surface via direct L-BFGS-B over all 5 parameters.

    This is the standard approach used in the Chinchilla paper and others:
    optimize E, A, B, α, β jointly without exploiting linear structure.
    Uses RSS (not Huber loss) for direct comparison with variable projection.

    Initialization is via a coarse 5D grid search (8 values per parameter).

    Args:
        N: Array of parameter counts
        D: Array of token counts
        L: Array of loss values (same length as N and D)
        use_grad: If True (default), use analytical gradients. If False,
            L-BFGS-B uses finite-difference gradients (forward differences
            unless jac is set).
        jac: Finite-difference scheme when use_grad is False (e.g. "3-point"
            for central differences). Ignored when use_grad is True.

    Returns:
        SurfaceFitResult with fitted parameters

    Raises:
        ValueError: If optimization fails or produces invalid results
    """
    N, D, L = (
        np.asarray(N, dtype=float),
        np.asarray(D, dtype=float),
        np.asarray(L, dtype=float),
    )
    if not (len(N) == len(D) == len(L)):
        raise ValueError(
            f"N, D, L must have same length, got {len(N)}, {len(D)}, {len(L)}"
        )

    log_N = np.log(N)
    log_D = np.log(D)

    def rss(x: np.ndarray) -> float:
        return _a3_rss(x, log_N, log_D, L)

    def rss_grad(x: np.ndarray) -> np.ndarray:
        return _a3_rss_grad(x, log_N, log_D, L)

    # Stage 1: Coarse 5D grid search for initialization
    best_rss = np.inf
    best_x0 = None
    for E_init in _A3_E_GRID:
        for A_init in _A3_A_GRID:
            for B_init in _A3_B_GRID:
                for alpha_init in _A3_ALPHA_GRID:
                    for beta_init in _A3_BETA_GRID:
                        x = np.array([E_init, A_init, B_init, alpha_init, beta_init])
                        r = rss(x)
                        if r < best_rss:
                            best_rss = r
                            best_x0 = x

    # Stage 2: L-BFGS-B refinement from best grid point
    bounds = [
        (1e-6, 10.0),  # E
        (1e-6, 1e6),  # A
        (1e-6, 1e6),  # B
        (0.01, 0.99),  # alpha
        (0.01, 0.99),  # beta
    ]

    result = minimize(
        rss,
        x0=best_x0,
        jac=rss_grad if use_grad else jac,
        method="L-BFGS-B",
        bounds=bounds,
        options=LBFGSB_OPTIONS,
    )

    if result is None or not result.success:
        message = result.message if result else "Unknown error"
        nit = result.nit if result else 0
        raise FitError(f"Optimization failed: {message} (iterations: {nit})")

    E, A, B, alpha, beta = result.x

    # Diagnostic checks — all 5 parameters are bounded
    param_names = ["E", "A", "B", "α", "β"]
    for name, val, (lo, hi) in zip(param_names, result.x, bounds):
        _check_at_bounds(name, val, lo, hi)
    _check_finite(E=E, A=A, B=B, alpha=alpha, beta=beta, rss=result.fun)

    return SurfaceFitResult(
        E=float(E),
        A=float(A),
        B=float(B),
        alpha=float(alpha),
        beta=float(beta),
        residual_sum_squares=float(result.fun),
        n_points=len(N),
        method="approach3",
    )
