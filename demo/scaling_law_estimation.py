"""Standalone scaling law estimation demo.

Generates synthetic IsoFLOP data from the Chinchilla loss surface
L(N, D) = E + A/N^α + B/D^β and recovers scaling exponents
a = β/(α+β) and b = α/(α+β) using three methods:

1. Approach 2 — Parabolic fits to IsoFLOP curves → power law regression
2. Approach 3 — Direct 5-parameter L-BFGS-B optimization
3. VPNLS — Variable Projection with Non-negative Least Squares (2D Nelder-Mead)

Primary experimentation variables: number of points per IsoFLOP curve (n_points)
and number of compute budgets (n_budgets).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.optimize import minimize, nnls


# ── Exceptions ────────────────────────────────────────────────────────────────


class FitError(Exception):
    """A fitting procedure failed to produce a valid result.

    Hard errors — no usable estimate is available. Callers should treat these
    as NaN / missing data.

    Subclasses:
        FitWarning  — soft issue; an estimate is still attached and usable.
        NonFiniteFitError — a parameter is NaN or Inf (hard error).
    """


class FitWarning(FitError):
    """Base for soft fit issues that still produce a usable estimate.

    All FitWarning subclasses carry an ``estimate`` attribute.
    Callers can ``except FitWarning`` to handle any soft issue uniformly.
    """

    def __init__(self, message: str, *, estimate=None):
        super().__init__(message)
        self.estimate = estimate


class MaxIterFitWarning(FitWarning):
    """Optimizer reached the maximum iteration limit.

    The result is typically still usable — the optimizer simply ran out of
    budget before meeting its convergence tolerances.
    """

    def __init__(self, message: str, *, result, estimate=None):
        super().__init__(message, estimate=estimate)
        self.result = result


class OptimizerFitWarning(FitWarning):
    """Optimizer reported failure (result.success=False) for reasons other than maxiter.

    Common cause: L-BFGS-B "ABNORMAL" on noisy surfaces. The result may
    still be usable.
    """

    def __init__(self, message: str, *, result=None, estimate=None):
        super().__init__(message, estimate=estimate)
        self.result = result


class BoundHitFitWarning(FitWarning):
    """A fitted parameter landed at or near an optimizer or grid bound.

    The estimate is still usable but should be treated with caution.
    """

    def __init__(self, message: str, *, estimate=None):
        super().__init__(message, estimate=estimate)


class NonFiniteFitError(FitError):
    """A fitted parameter is NaN or Inf (hard error — no usable estimate)."""


# ── Loss Surface ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LossSurface:
    """Chinchilla loss surface L(N, D) = E + A/N^α + B/D^β."""

    A: float
    B: float
    E: float
    alpha: float
    beta: float

    @property
    def a(self) -> float:
        """N* scaling exponent: a = β/(α+β)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """D* scaling exponent: b = α/(α+β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def G(self) -> float:
        """Optimal allocation constant G = (αA/βB)^(1/(α+β))."""
        return (self.alpha * self.A / (self.beta * self.B)) ** (
            1 / (self.alpha + self.beta)
        )

    def N_opt(self, C: float) -> float:
        """Optimal parameter count N* for compute budget C (where C = 6ND)."""
        return self.G * ((C / 6) ** self.a)

    def D_opt(self, C: float) -> float:
        """Optimal token count D* for compute budget C (where C = 6ND)."""
        return (1 / self.G) * ((C / 6) ** self.b)

    def loss(
        self, N: Union[float, np.ndarray], D: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute L(N, D) = E + A/N^α + B/D^β."""
        return self.E + self.A / (N**self.alpha) + self.B / (D**self.beta)


# ── Data Generation ───────────────────────────────────────────────────────────


def _compute_center_offset(
    C: float,
    compute_budgets: np.ndarray,
    drift_rate: float,
    center_scale: float,
) -> float:
    """Compute the sampling center offset for a given compute budget.

    Combines two independent effects:
    1. drift_rate: Linear drift in log-compute space starting from zero at the
       lowest compute budget. At min compute: offset = 0, at max: offset = -drift_rate.
    2. center_scale: Constant multiplicative factor applied to all centers.
       scale > 1 shifts left (smaller N), scale < 1 shifts right (larger N).

    Both effects are subtractive in log10 space.
    """
    offset = 0.0

    if center_scale != 1.0:
        offset -= np.log10(center_scale)

    if drift_rate != 0.0:
        log_C_all = np.log10(compute_budgets)
        log_C_min = log_C_all.min()
        log_C_range = log_C_all.max() - log_C_min
        if log_C_range > 0:
            normalized_log_C = (np.log10(C) - log_C_min) / log_C_range
            offset -= drift_rate * normalized_log_C

    return offset


def _isoflop_sample(
    C: float,
    n_points: int,
    log_range: float,
    center_offset: float,
    surface: LossSurface,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points along an IsoFLOP contour (constant compute budget).

    Points are sampled logarithmically around the optimal N* for the
    given compute budget. D is derived from the constraint C = 6ND.
    Gaussian noise with standard deviation noise_std is added to L.
    """
    log_N_center = np.log10(surface.N_opt(C)) + center_offset
    log_N_min = log_N_center - log_range
    log_N_max = log_N_center + log_range
    N = np.logspace(log_N_min, log_N_max, n_points)
    D = C / (6 * N)
    L = surface.loss(N, D)
    if not isinstance(L, np.ndarray):
        L = np.array([L])
    if noise_std > 0:
        L = L + rng.normal(0, noise_std, size=L.shape)
    return N, D, L


@dataclass
class IsoFlopData:
    """IsoFLOP samples across multiple compute budgets.

    Stores per-budget arrays of (N, D, L) for methods that need per-budget
    structure (Approach 2), and provides pooled arrays for methods that
    operate on all data at once (Approach 3, VPNLS).
    """

    per_budget: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    """Per-budget list of (N, D, L) arrays, one per compute budget."""

    @property
    def N(self) -> np.ndarray:
        """Pooled parameter counts across all budgets."""
        return np.concatenate([d[0] for d in self.per_budget])

    @property
    def D(self) -> np.ndarray:
        """Pooled token counts across all budgets."""
        return np.concatenate([d[1] for d in self.per_budget])

    @property
    def L(self) -> np.ndarray:
        """Pooled loss values across all budgets."""
        return np.concatenate([d[2] for d in self.per_budget])


def generate_isoflop_data(
    surface: LossSurface,
    compute_budgets: np.ndarray,
    n_points: int,
    log_range: float,
    drift_rate: float,
    center_scale: float,
    noise_std: float,
    rng: np.random.Generator,
) -> IsoFlopData:
    """Generate IsoFLOP samples across all compute budgets.

    For each compute budget, samples n_points along the IsoFLOP contour
    (constant C = 6ND) in log-space around the optimal N*, with optional
    sampling bias via drift_rate and center_scale. Gaussian noise with
    standard deviation noise_std is added to the loss values.

    Args:
        surface: Loss surface configuration.
        compute_budgets: Array of compute budgets in FLOPs.
        n_points: Number of points per IsoFLOP curve.
        log_range: Range in log10 space around optimal N (±log_range).
        drift_rate: Linear drift of sampling center across compute budgets
            (0 at min budget, -drift_rate at max budget, in log10 units of N).
        center_scale: Constant multiplier on all sampling centers.
            1.0 = centered on true N*. >1 shifts left (smaller N).
        noise_std: Standard deviation of Gaussian noise added to loss values.
            0.0 = no noise (deterministic).
        rng: NumPy random generator for reproducibility.

    Returns:
        IsoFlopData with per-budget and pooled access.
    """
    per_budget = []
    for C in compute_budgets:
        center_offset = _compute_center_offset(
            C, compute_budgets, drift_rate, center_scale
        )
        N, D, L = _isoflop_sample(
            C, n_points, log_range, center_offset, surface, noise_std, rng
        )
        per_budget.append((N, D, L))
    return IsoFlopData(per_budget=per_budget)


# ── Fitting: Common Result ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExponentEstimate:
    """Recovered scaling exponents from any fitting method."""

    a: float  # Estimate of β/(α+β)
    b: float  # Estimate of α/(α+β)
    method: str
    converged: bool  # Whether the optimizer reported convergence


# ── Fitting: Approach 2 ──────────────────────────────────────────────────────


def _fit_parabola(
    log_x: np.ndarray,
    L: np.ndarray,
    min_curvature: float = 1e-10,
) -> tuple[np.ndarray, float, float]:
    """Fit a parabola to loss vs log-x data.

    Returns (coeffs, x_opt, L_min). Raises FitError if fewer than 3 points
    (underdetermined for degree 2) or curvature is non-positive.
    """
    if len(log_x) < 3:
        raise FitError(f"Parabola fit requires at least 3 points, got {len(log_x)}.")
    coeffs = np.polyfit(log_x, L, 2)
    a_coeff, b_coeff, _ = coeffs
    if a_coeff < min_curvature:
        raise FitError(f"Parabola fit has non-positive curvature (a={a_coeff:.2e}).")
    log_x_opt = -b_coeff / (2 * a_coeff)
    x_opt = 10**log_x_opt
    L_min = np.polyval(coeffs, log_x_opt)
    return coeffs, x_opt, L_min


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit power law y = 10^intercept · x^exponent in log-log space.

    Returns (exponent, intercept).
    """
    log_x = np.log10(x)
    log_y = np.log10(y)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    return float(exponent), float(intercept)


def fit_approach2(
    data: IsoFlopData,
    compute_budgets: np.ndarray,
) -> ExponentEstimate:
    """Recover exponents via Chinchilla Approach 2 (parabolic IsoFLOP fits).

    Fits L vs log(N) parabolas per budget to find N* and D*, then fits
    power laws N* ∝ C^a and D* ∝ C^b across budgets.

    Args:
        data: IsoFlopData from generate_isoflop_data.
        compute_budgets: Array of compute budgets in FLOPs (same order as data).

    Returns:
        ExponentEstimate with recovered a and b.

    Raises:
        FitError: If any parabola fit has non-positive curvature.
    """
    N_opts = []
    D_opts = []

    for N, D, L in data.per_budget:
        _, N_opt, _ = _fit_parabola(np.log10(N), L)
        N_opts.append(N_opt)

        _, D_opt, _ = _fit_parabola(np.log10(D), L)
        D_opts.append(D_opt)

    N_opts_arr = np.array(N_opts)
    D_opts_arr = np.array(D_opts)

    a_exp, _ = _fit_power_law(compute_budgets, N_opts_arr)
    b_exp, _ = _fit_power_law(compute_budgets, D_opts_arr)

    return ExponentEstimate(
        a=a_exp, b=b_exp, method="Chinchilla Approach 2", converged=True
    )


# ── Shared fitting diagnostics ────────────────────────────────────────────────


def _make_estimate(
    alpha: float, beta: float, method: str, converged: bool = True
) -> ExponentEstimate:
    """Compute ExponentEstimate from recovered alpha and beta."""
    return ExponentEstimate(
        a=beta / (alpha + beta),
        b=alpha / (alpha + beta),
        method=method,
        converged=converged,
    )


def _validate_optimizer_result(
    result,
    method_name: str,
    maxiter: int,
    alpha_beta_from_result,
) -> None:
    """Check optimizer result for None, maxiter, or general failure.

    Args:
        result: scipy.optimize.OptimizeResult (or None).
        method_name: Human-readable name for error messages.
        maxiter: The maxiter setting used for this optimizer.
        alpha_beta_from_result: Callable(result) → (alpha, beta) to extract
            exponents from an incomplete result for MaxIterationsFitError.

    Raises:
        OptimizerFitWarning: If result is None or optimizer reported failure.
        MaxIterFitWarning: If optimizer hit the iteration limit.
            Carries an ExponentEstimate computed from the partial result.
    """
    if result is None:
        raise OptimizerFitWarning(f"{method_name} optimization returned None.")
    if not result.success:
        alpha, beta = alpha_beta_from_result(result)
        estimate = _make_estimate(alpha, beta, method_name, converged=False)
        if result.nit >= maxiter:
            raise MaxIterFitWarning(
                f"{method_name} hit maxiter ({maxiter}): "
                f"{result.message} (iterations: {result.nit})",
                result=result,
                estimate=estimate,
            )
        raise OptimizerFitWarning(
            f"{method_name} optimization failed: {result.message} "
            f"(iterations: {result.nit})",
            result=result,
            estimate=estimate,
        )


def _check_params_finite(method_name: str, **params: float) -> None:
    """Raise NonFiniteFitError if any parameter is NaN or Inf."""
    for name, val in params.items():
        if np.isnan(val) or np.isinf(val):
            raise NonFiniteFitError(f"{method_name}: {name}={val} is non-finite.")


def _check_params_at_bounds(
    method_name: str,
    named_bounds: list[tuple[str, float, float, float]],
    estimate: Union["ExponentEstimate", None] = None,
    tol: float = 1e-6,
) -> None:
    """Raise BoundHitFitWarning if any parameter is at or near a bound.

    Args:
        method_name: Human-readable name for error messages.
        named_bounds: List of (name, value, lo, hi) tuples.
        estimate: ExponentEstimate to attach to the warning (still usable).
        tol: Tolerance for bound proximity.
    """
    for name, val, lo, hi in named_bounds:
        if val - lo < tol or hi - val < tol:
            raise BoundHitFitWarning(
                f"{method_name}: {name}={val:.6f} at bound [{lo}, {hi}].",
                estimate=estimate,
            )


# ── Optimizer defaults ────────────────────────────────────────────────────────

_OPT_TOL = 1e-15
_OPT_MAXITER = 1000

# Shared parameter bounds (used by both Approach 3 and VPNLS)
_E_BOUNDS = (1e-6, 10.0)
_A_BOUNDS = (1e-6, 1e6)
_B_BOUNDS = (1e-6, 1e6)
_ALPHA_BOUNDS = (0.01, 0.99)
_BETA_BOUNDS = (0.01, 0.99)

# Grid search endpoints for α/β (interior to optimizer bounds)
_ALPHA_GRID_MIN, _ALPHA_GRID_MAX = 0.05, 0.95
_BETA_GRID_MIN, _BETA_GRID_MAX = 0.05, 0.95


# ── Fitting: Approach 3 ──────────────────────────────────────────────────────

# Coarse 5D grid for initialization (4 values per parameter = 4^5 = 1024 points)
_A3_ALPHA_GRID = np.linspace(_ALPHA_GRID_MIN, _ALPHA_GRID_MAX, 4)
_A3_BETA_GRID = np.linspace(_BETA_GRID_MIN, _BETA_GRID_MAX, 4)
_A3_E_GRID = np.linspace(0.1, 5.0, 4)
_A3_A_GRID = np.logspace(1, 4, 4)
_A3_B_GRID = np.logspace(1, 4, 4)

_A3_BOUNDS = [_E_BOUNDS, _A_BOUNDS, _B_BOUNDS, _ALPHA_BOUNDS, _BETA_BOUNDS]
_A3_LBFGSB_OPTIONS = {"ftol": _OPT_TOL, "gtol": _OPT_TOL, "maxiter": _OPT_MAXITER}


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
    term_N = np.exp(-alpha * log_N)
    term_D = np.exp(-beta * log_D)
    pred = E + A * term_N + B * term_D
    resid = pred - L
    return np.array(
        [
            2 * np.sum(resid),
            2 * np.sum(resid * term_N),
            2 * np.sum(resid * term_D),
            2 * np.sum(resid * A * term_N * (-log_N)),
            2 * np.sum(resid * B * term_D * (-log_D)),
        ]
    )


def fit_approach3(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    random_init: bool = False,
    rng: np.random.Generator | None = None,
) -> ExponentEstimate:
    """Recover exponents via direct 5-parameter L-BFGS-B optimization.

    Optimizes E, A, B, α, β jointly using analytical gradients.
    Initialization via coarse 5D grid search (4^5 = 1024 evaluations) or,
    when ``random_init`` is True, a single random starting point within bounds.

    Args:
        N: Array of parameter counts.
        D: Array of token counts.
        L: Array of loss values (same length as N and D).
        random_init: If True, use a random starting point instead of grid search.
        rng: Random generator (required when ``random_init`` is True).

    Returns:
        ExponentEstimate with recovered a and b.

    Raises:
        FitError: If optimization fails or hits parameter bounds.
    """
    method_name = "Chinchilla Approach 3"
    N, D, L = (
        np.asarray(N, dtype=float),
        np.asarray(D, dtype=float),
        np.asarray(L, dtype=float),
    )
    log_N, log_D = np.log(N), np.log(D)

    def rss(x: np.ndarray) -> float:
        return _surface_rss(x, log_N, log_D, L)

    def rss_grad(x: np.ndarray) -> np.ndarray:
        return _surface_rss_grad(x, log_N, log_D, L)

    if random_init:
        if rng is None:
            raise ValueError("rng is required when random_init=True")
        best_x0 = np.array([rng.uniform(lo, hi) for lo, hi in _A3_BOUNDS])
    else:
        best_rss = np.inf
        best_x0 = None
        for E_init in _A3_E_GRID:
            for A_init in _A3_A_GRID:
                for B_init in _A3_B_GRID:
                    for alpha_init in _A3_ALPHA_GRID:
                        for beta_init in _A3_BETA_GRID:
                            x = np.array(
                                [E_init, A_init, B_init, alpha_init, beta_init]
                            )
                            r = rss(x)
                            if r < best_rss:
                                best_rss = r
                                best_x0 = x

    result = minimize(
        rss,
        x0=best_x0,
        jac=rss_grad,
        method="L-BFGS-B",
        bounds=_A3_BOUNDS,
        options=_A3_LBFGSB_OPTIONS,
    )

    _validate_optimizer_result(
        result,
        method_name,
        _A3_LBFGSB_OPTIONS["maxiter"],
        alpha_beta_from_result=lambda r: (r.x[3], r.x[4]),
    )
    assert result is not None  # guaranteed by _validate_optimizer_result

    E, A, B, alpha, beta = result.x
    _check_params_finite(method_name, E=E, A=A, B=B, alpha=alpha, beta=beta)
    estimate = _make_estimate(alpha, beta, method_name)
    _check_params_at_bounds(
        method_name,
        [
            (name, val, lo, hi)
            for name, val, (lo, hi) in zip(
                ["E", "A", "B", "α", "β"], result.x, _A3_BOUNDS
            )
        ],
        estimate=estimate,
    )

    return estimate


# ── Fitting: VPNLS ───────────────────────────────────────────────────────────

_VP_ALPHA_GRID = np.linspace(_ALPHA_GRID_MIN, _ALPHA_GRID_MAX, 32)
_VP_BETA_GRID = np.linspace(_BETA_GRID_MIN, _BETA_GRID_MAX, 32)
_VP_NELDER_MEAD_OPTIONS = {
    "xatol": _OPT_TOL,
    "fatol": _OPT_TOL,
    "maxiter": _OPT_MAXITER,
}

_A3_GRID_SIZE = (
    len(_A3_ALPHA_GRID)
    * len(_A3_BETA_GRID)
    * len(_A3_E_GRID)
    * len(_A3_A_GRID)
    * len(_A3_B_GRID)
)
_VP_GRID_SIZE = len(_VP_ALPHA_GRID) * len(_VP_BETA_GRID)
assert (
    _A3_GRID_SIZE == _VP_GRID_SIZE
), f"Grid search sizes differ: Approach 3 has {_A3_GRID_SIZE}, VPNLS has {_VP_GRID_SIZE}"


def _vp_compute_rss_and_params(
    alpha: float,
    beta: float,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[float, np.ndarray]:
    """For fixed (α, β), solve NNLS for (E, A, B) and return (RSS, [E, A, B])."""
    N_neg_alpha = np.exp(-alpha * log_N)
    D_neg_beta = np.exp(-beta * log_D)
    design_matrix = np.column_stack([np.ones(len(L)), N_neg_alpha, D_neg_beta])
    params, rnorm = nnls(design_matrix, L)
    rss = rnorm**2
    return rss, params


def _vp_grid_search(
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
            rss, _ = _vp_compute_rss_and_params(alpha, beta, log_N, log_D, L)
            if rss < best_rss:
                best_rss = rss
                best_i, best_j = i, j
    return best_i, best_j


def fit_vpnls(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
) -> ExponentEstimate:
    """Recover exponents via Variable Projection + NNLS + Nelder-Mead.

    Searches over (α, β) only; solves (E, A, B) via NNLS at each candidate.
    Coarse 32×32 grid initialization followed by Nelder-Mead refinement.

    Args:
        N: Array of parameter counts.
        D: Array of token counts.
        L: Array of loss values (same length as N and D).

    Returns:
        ExponentEstimate with recovered a and b.

    Raises:
        FitError: If optimization fails or hits parameter bounds.
    """
    N, D, L = np.asarray(N), np.asarray(D), np.asarray(L)
    log_N, log_D = np.log(N), np.log(D)

    alpha_grid = _VP_ALPHA_GRID
    beta_grid = _VP_BETA_GRID

    method_name = "VPNLS"
    best_i, best_j = _vp_grid_search(alpha_grid, beta_grid, log_N, log_D, L)

    def objective(x):
        rss, _ = _vp_compute_rss_and_params(x[0], x[1], log_N, log_D, L)
        return rss

    x0 = [alpha_grid[best_i], beta_grid[best_j]]
    result = minimize(
        objective, x0=x0, method="Nelder-Mead", options=_VP_NELDER_MEAD_OPTIONS
    )

    _validate_optimizer_result(
        result,
        method_name,
        _VP_NELDER_MEAD_OPTIONS["maxiter"],
        alpha_beta_from_result=lambda r: (float(r.x[0]), float(r.x[1])),
    )
    assert result is not None  # guaranteed by _validate_optimizer_result

    alpha, beta = float(result.x[0]), float(result.x[1])
    rss, (E, A, B) = _vp_compute_rss_and_params(alpha, beta, log_N, log_D, L)

    _check_params_finite(method_name, E=E, A=A, B=B, alpha=alpha, beta=beta)
    estimate = _make_estimate(alpha, beta, method_name)
    _check_params_at_bounds(
        method_name,
        [
            ("E", E, *_E_BOUNDS),
            ("A", A, *_A_BOUNDS),
            ("B", B, *_B_BOUNDS),
            ("α", alpha, *_ALPHA_BOUNDS),
            ("β", beta, *_BETA_BOUNDS),
        ],
        estimate=estimate,
    )

    return estimate


# ── Experiment ────────────────────────────────────────────────────────────────

METHOD_NAMES = ["Chinchilla Approach 2", "Chinchilla Approach 3", "VPNLS"]

METHOD_STYLES = {
    "Chinchilla Approach 2": {"color": "#d62728", "marker": "s"},
    "Chinchilla Approach 3": {"color": "#ff7f0e", "marker": "^"},
    "VPNLS": {"color": "#1f77b4", "marker": "o"},
}


def compute_exponent_errors(
    surface: LossSurface,
    compute_budgets: np.ndarray,
    n_points_range: np.ndarray,
    log_range: float,
    drift_rate: float,
    center_scale: float,
    noise_std: float,
    seed: int,
    n_repeats: int,
    a3_random_init: bool = False,
) -> dict[str, dict]:
    """Compute a/b exponent errors for all methods across n_points and seeds.

    For each seed and n_points in n_points_range, generates data and fits
    all three methods, recording relative error in recovered a and b exponents.

    Args:
        surface: Loss surface configuration.
        compute_budgets: Array of compute budgets in FLOPs.
        n_points_range: Array of n_points values to sweep.
        log_range: Sampling range in log10 space around optimal N.
        drift_rate: Sampling center drift rate.
        center_scale: Sampling center scale factor.
        noise_std: Standard deviation of Gaussian noise added to loss values.
        seed: Base random seed (each repeat uses seed + r).
        n_repeats: Number of independent noise realizations per n_points.
        a3_random_init: If True, initialize Approach 3 from a random point
            within bounds instead of the default 5D grid search.

    Returns:
        Dict mapping method name → dict with 2D arrays (n_repeats × n_points):
            "a_errors", "b_errors" — relative errors (NaN for hard failures).
            Warnings (estimate still counted):
                "maxiter"   — optimizer hit iteration limit.
                "optimizer" — optimizer reported failure (e.g. ABNORMAL).
                "bound_hit" — parameter at bound.
            Errors (estimate is NaN):
                "fail"      — hard failure (non-finite, underdetermined, etc.).
    """
    true_a = surface.a
    true_b = surface.b
    n_pts_len = len(n_points_range)

    results: dict[str, dict] = {
        m: {
            "a_errors": np.zeros((n_repeats, n_pts_len)),
            "b_errors": np.zeros((n_repeats, n_pts_len)),
            "maxiter": np.zeros((n_repeats, n_pts_len), dtype=bool),
            "optimizer": np.zeros((n_repeats, n_pts_len), dtype=bool),
            "bound_hit": np.zeros((n_repeats, n_pts_len), dtype=bool),
            "fail": np.zeros((n_repeats, n_pts_len), dtype=bool),
        }
        for m in METHOD_NAMES
    }

    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        for i, n_points in enumerate(n_points_range):
            n_pts = int(n_points)

            data = generate_isoflop_data(
                surface,
                compute_budgets,
                n_pts,
                log_range,
                drift_rate,
                center_scale,
                noise_std,
                rng,
            )

            a3_rng = rng.spawn(1)[0] if a3_random_init else None
            for method_name, fit_fn in [
                ("Chinchilla Approach 2", lambda d: fit_approach2(d, compute_budgets)),
                (
                    "Chinchilla Approach 3",
                    lambda d: fit_approach3(
                        d.N, d.D, d.L, random_init=a3_random_init, rng=a3_rng
                    ),
                ),
                ("VPNLS", lambda d: fit_vpnls(d.N, d.D, d.L)),
            ]:
                res = results[method_name]
                est = None
                try:
                    est = fit_fn(data)
                except MaxIterFitWarning as e:
                    res["maxiter"][r, i] = True
                    est = e.estimate
                except OptimizerFitWarning as e:
                    res["optimizer"][r, i] = True
                    est = e.estimate
                except BoundHitFitWarning as e:
                    res["bound_hit"][r, i] = True
                    est = e.estimate
                except FitError:
                    res["fail"][r, i] = True
                if est is not None:
                    res["a_errors"][r, i] = (est.a - true_a) / true_a
                    res["b_errors"][r, i] = (est.b - true_b) / true_b
                else:
                    res["a_errors"][r, i] = np.nan
                    res["b_errors"][r, i] = np.nan

    return results


def compute_pairwise_frequencies(
    all_results: dict[int, dict[str, dict]],
    n_budgets_range: list[int],
    tol: float = 1e-12,
) -> dict[int, tuple[dict[str, int], int]]:
    """Compute pairwise |b| error comparison frequencies across trials.

    For each (budget, repeat, n_points) trial, compares methods pairwise
    by absolute b error. NaN errors are treated as infinite (worst).

    Returns:
        Dict mapping n_budgets → (counts dict, total trial count).
        Counts dict keys: "vpnls_beats_a3", "a3_beats_vpnls", "vpnls_ties_a3",
                          "vpnls_beats_or_ties_a2", "a3_beats_or_ties_a2".
    """
    a2, a3, vp = METHOD_NAMES

    def _cmp(va: float, vb: float) -> int:
        """Return -1 if a < b, +1 if a > b, 0 if tied (within tol)."""
        a_abs = np.inf if np.isnan(va) else abs(va)
        b_abs = np.inf if np.isnan(vb) else abs(vb)
        if a_abs == np.inf and b_abs == np.inf:
            return 0
        if abs(a_abs - b_abs) < tol:
            return 0
        return -1 if a_abs < b_abs else 1

    out: dict[int, tuple[dict[str, int], int]] = {}

    for nb in n_budgets_range:
        results = all_results[nb]
        n_repeats = results[a2]["b_errors"].shape[0]
        n_pts_len = results[a2]["b_errors"].shape[1]
        counts = {
            "vpnls_beats_a3": 0,
            "a3_beats_vpnls": 0,
            "vpnls_ties_a3": 0,
            "vpnls_beats_or_ties_a2": 0,
            "a3_beats_or_ties_a2": 0,
        }
        for r in range(n_repeats):
            for i in range(n_pts_len):
                e = {m: results[m]["b_errors"][r, i] for m in METHOD_NAMES}
                c_vp_a3 = _cmp(e[vp], e[a3])
                if c_vp_a3 < 0:
                    counts["vpnls_beats_a3"] += 1
                elif c_vp_a3 > 0:
                    counts["a3_beats_vpnls"] += 1
                else:
                    counts["vpnls_ties_a3"] += 1
                if _cmp(e[vp], e[a2]) <= 0:
                    counts["vpnls_beats_or_ties_a2"] += 1
                if _cmp(e[a3], e[a2]) <= 0:
                    counts["a3_beats_or_ties_a2"] += 1
        out[nb] = (counts, n_repeats * n_pts_len)

    return out


# ── Visualization ─────────────────────────────────────────────────────────────


def _log_range_to_label(log_range: float) -> str:
    """Convert log_range to human-readable N sampling range."""
    factor = 10**log_range
    if factor >= 10:
        return f"±{factor:.0f}x"
    return f"±{factor:.1f}x"


def create_figure(
    all_results: dict[int, dict[str, dict]],
    surface: LossSurface,
    n_points_range: np.ndarray,
    n_budgets_range: list[int],
    c_min: float,
    c_max: float,
    log_range: float,
    drift_rate: float,
    noise_std: float,
    n_repeats: int,
    output_path: str,
) -> None:
    """Create and save the data efficiency comparison figure using boxplots.

    Layout: len(n_budgets_range) rows × 2 columns (exponent a, exponent b).
    Each subplot shows boxplots of |relative error| vs n_points for all methods.

    Args:
        all_results: Dict mapping n_budgets → method results dict.
            Each method dict has 2D arrays (n_repeats × n_points).
        surface: Loss surface (for ground truth display).
        n_points_range: Array of n_points values (x-axis).
        n_budgets_range: List of n_budgets values (one row per value).
        c_min: Minimum compute budget in FLOPs.
        c_max: Maximum compute budget in FLOPs.
        log_range: Sampling grid width in log10 space.
        drift_rate: Drift rate used (for title).
        noise_std: Noise standard deviation (for title).
        n_repeats: Number of repeats per configuration (for title).
        output_path: Path to save the figure.
    """
    range_label = _log_range_to_label(log_range)
    exponent_labels = ["a", "b"]
    error_keys = ["a_errors", "b_errors"]
    n_rows = len(n_budgets_range)
    n_methods = len(METHOD_NAMES)
    log_positions = np.log2(n_points_range.astype(float))
    box_width = 0.2
    offsets = np.linspace(
        -(n_methods - 1) / 2 * box_width, (n_methods - 1) / 2 * box_width, n_methods
    )

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 2.8 * n_rows),
        gridspec_kw={"hspace": 0.5, "wspace": 0.2},
        squeeze=False,
    )

    for row, nb in enumerate(n_budgets_range):
        results = all_results[nb]

        for col, (exp_label, err_key) in enumerate(zip(exponent_labels, error_keys)):
            ax = axes[row, col]

            for m_idx, method_name in enumerate(METHOD_NAMES):
                style = METHOD_STYLES[method_name]
                errors_2d = np.abs(results[method_name][err_key]) * 100
                positions = log_positions + offsets[m_idx]

                bp = ax.boxplot(
                    [
                        errors_2d[:, i][~np.isnan(errors_2d[:, i])]
                        for i in range(errors_2d.shape[1])
                    ],
                    positions=positions,
                    widths=box_width * 0.85,
                    patch_artist=True,
                    showfliers=True,
                    flierprops={
                        "marker": ".",
                        "markersize": 3,
                        "alpha": 0.5,
                        "markerfacecolor": style["color"],
                    },
                    medianprops={"color": "white", "linewidth": 1.2},
                    whiskerprops={"color": style["color"], "linewidth": 0.8},
                    capprops={"color": style["color"], "linewidth": 0.8},
                    boxprops={
                        "facecolor": style["color"],
                        "alpha": 0.6,
                        "edgecolor": style["color"],
                    },
                )

            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(log_positions)
            ax.set_xticklabels(n_points_range)
            ax.set_xlim(log_positions[0] - 0.5, log_positions[-1] + 0.5)

            true_val = surface.a if exp_label == "a" else surface.b
            ax.set_title(
                f"Exponent {exp_label} (true={true_val:.3f}), " f"{nb} compute budgets",
                fontsize=9,
            )

            if row == n_rows - 1:
                ax.set_xlabel("Points per isoflop curve", fontsize=9)
            if col == 0:
                ax.set_ylabel("Absolute relative error (%)", fontsize=9)
            ax.tick_params(labelsize=8)

    fig.suptitle(
        "Data Efficiency: Scaling Exponent Recovery\n"
        f"$\\alpha$={surface.alpha:.3f}, $\\beta$={surface.beta:.3f}, "
        f"grid width = {range_label}, "
        f"drift rate = {drift_rate:.3f}, "
        f"noise $\\sigma$ = {noise_std}, "
        f"{n_repeats} seeds, "
        f"budgets: {c_min:.0e}\u2013{c_max:.0e} FLOPs",
        fontsize=11,
        y=1.01,
    )

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=METHOD_STYLES[m]["color"], alpha=0.6, label=m)
        for m in METHOD_NAMES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METHOD_NAMES),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_isoflop_figure(
    data: IsoFlopData,
    surface: LossSurface,
    compute_budgets: np.ndarray,
    log_range: float,
    drift_rate: float,
    center_scale: float,
    noise_std: float,
    output_path: str,
) -> None:
    """Create and save a visualization of the IsoFLOP curves with noise.

    Shows L vs log10(N) and L vs log10(D) for each compute budget, with the
    true (noiseless) curve overlaid, plus markers for the true optimum and the
    sampling center.

    Args:
        data: IsoFlopData from generate_isoflop_data.
        surface: Loss surface (for noiseless reference curves and true optima).
        compute_budgets: Compute budgets used (for labels).
        log_range: Sampling grid width in log10 space (for title).
        drift_rate: Drift rate used for sampling bias.
        center_scale: Center scale used for sampling bias.
        noise_std: Noise standard deviation (for title).
        output_path: Path to save the figure.
    """
    range_label = _log_range_to_label(log_range)
    n_budgets = len(compute_budgets)
    colors = plt.colormaps["viridis"](np.linspace(0.1, 0.9, n_budgets))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: L vs log10(N)
    ax_n = axes[0]
    for i, ((N, D, L), C) in enumerate(zip(data.per_budget, compute_budgets)):
        log_N = np.log10(N)
        ax_n.scatter(
            log_N, L, color=colors[i], s=30, alpha=0.7, zorder=2, label=f"C = {C:.0e}"
        )
        L_true = surface.loss(N, D)
        ax_n.plot(log_N, L_true, color=colors[i], linewidth=1, alpha=0.5, zorder=1)

        # True optimum (red X)
        N_opt = surface.N_opt(C)
        L_opt = float(surface.loss(N_opt, surface.D_opt(C)))
        ax_n.scatter(
            [np.log10(N_opt)],
            [L_opt],
            c="red",
            marker="x",
            s=100,
            zorder=4,
            linewidths=2.5,
        )

        # Sampling center (black diamond)
        center_offset = _compute_center_offset(
            C, compute_budgets, drift_rate, center_scale
        )
        N_center = N_opt * 10**center_offset
        D_center = C / (6 * N_center)
        L_center = float(surface.loss(N_center, D_center))
        ax_n.scatter(
            [np.log10(N_center)],
            [L_center],
            c="black",
            marker="D",
            s=90,
            zorder=5,
            linewidths=1,
            edgecolors="white",
        )

    ax_n.set_xlabel("log$_{10}$(N)", fontsize=10)
    ax_n.set_ylabel("Loss L(N, D)", fontsize=10)
    ax_n.set_title("IsoFLOP curves: L vs N", fontsize=10)
    ax_n.legend(fontsize=7, loc="upper right")
    ax_n.grid(True, alpha=0.3)
    ax_n.tick_params(labelsize=8)
    ax_n.scatter([], [], c="red", marker="x", s=80, linewidths=2, label="True $N^*$")
    ax_n.scatter(
        [],
        [],
        c="black",
        marker="D",
        s=60,
        linewidths=1,
        edgecolors="white",
        label="Sampling center",
    )
    ax_n.legend(fontsize=7, loc="upper right")

    # Right: L vs log10(D)
    ax_d = axes[1]
    for i, ((N, D, L), C) in enumerate(zip(data.per_budget, compute_budgets)):
        log_D = np.log10(D)
        ax_d.scatter(
            log_D, L, color=colors[i], s=30, alpha=0.7, zorder=2, label=f"C = {C:.0e}"
        )
        L_true = surface.loss(N, D)
        ax_d.plot(log_D, L_true, color=colors[i], linewidth=1, alpha=0.5, zorder=1)

        # True optimum (red X)
        D_opt = surface.D_opt(C)
        L_opt = float(surface.loss(surface.N_opt(C), D_opt))
        ax_d.scatter(
            [np.log10(D_opt)],
            [L_opt],
            c="red",
            marker="x",
            s=100,
            zorder=4,
            linewidths=2.5,
        )

        # Sampling center (black diamond)
        center_offset = _compute_center_offset(
            C, compute_budgets, drift_rate, center_scale
        )
        N_center = surface.N_opt(C) * 10**center_offset
        D_center = C / (6 * N_center)
        L_center = float(surface.loss(N_center, D_center))
        ax_d.scatter(
            [np.log10(D_center)],
            [L_center],
            c="black",
            marker="D",
            s=90,
            zorder=5,
            linewidths=1,
            edgecolors="white",
        )

    ax_d.set_xlabel("log$_{10}$(D)", fontsize=10)
    ax_d.set_ylabel("Loss L(N, D)", fontsize=10)
    ax_d.set_title("IsoFLOP curves: L vs D", fontsize=10)
    ax_d.grid(True, alpha=0.3)
    ax_d.tick_params(labelsize=8)
    ax_d.scatter([], [], c="red", marker="x", s=80, linewidths=2, label="True $D^*$")
    ax_d.scatter(
        [],
        [],
        c="black",
        marker="D",
        s=60,
        linewidths=1,
        edgecolors="white",
        label="Sampling center",
    )
    ax_d.legend(fontsize=7, loc="upper right")

    n_pts = len(data.per_budget[0][0])
    fig.suptitle(
        f"IsoFLOP Samples ({n_budgets} budgets, {n_pts} points/curve, "
        f"grid width = {range_label}, noise $\\sigma$ = {noise_std})",
        fontsize=11,
        y=1.01,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Console Output ────────────────────────────────────────────────────────────


def print_summary(
    all_results: dict[int, dict[str, dict]],
    surface: LossSurface,
    n_points_range: np.ndarray,
    n_budgets_range: list[int],
    c_min: float,
    c_max: float,
    log_range: float,
    drift_rate: float,
    noise_std: float,
    n_repeats: int,
) -> None:
    """Print rich summary tables to console.

    Args:
        all_results: Dict mapping n_budgets → method results dict.
            Each method dict has 2D arrays (n_repeats × n_points).
        surface: Loss surface (for ground truth display).
        n_points_range: Array of n_points values.
        n_budgets_range: List of n_budgets values tested.
        c_min: Minimum compute budget in FLOPs.
        c_max: Maximum compute budget in FLOPs.
        log_range: Sampling grid width in log10 space.
        drift_rate: Drift rate used.
        noise_std: Standard deviation of Gaussian noise on loss values.
        n_repeats: Number of repeats per configuration.
    """
    range_label = _log_range_to_label(log_range)

    print("=" * 60)
    print("Scaling Law Estimation — Data Efficiency")
    print("=" * 60)

    config = {
        "Surface": f"α={surface.alpha:.4f}, β={surface.beta:.4f}, A={surface.A}, B={surface.B}, E={surface.E}",
        "True exponents": f"a={surface.a:.6f}, b={surface.b:.6f}",
        "Grid width": f"{range_label} (log_range={log_range:.4f})",
        "Drift rate": f"{drift_rate:.6f}",
        "Noise σ": f"{noise_std}",
        "Budget range": f"{c_min:.0e} – {c_max:.0e} FLOPs",
        "n_budgets": f"{n_budgets_range}",
        "n_points range": f"{n_points_range[0]} – {n_points_range[-1]}",
        "n_repeats": f"{n_repeats}",
    }
    config_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
    print(config_df.to_string(index=False))
    print()

    for nb in n_budgets_range:
        results = all_results[nb]
        rows = []
        for method_name in METHOD_NAMES:
            res = results[method_name]
            a_abs = np.abs(res["a_errors"]) * 100
            b_abs = np.abs(res["b_errors"]) * 100
            n_total = res["maxiter"].size
            rows.append(
                {
                    "Method": method_name,
                    "Median |a| %": f"{np.nanmedian(a_abs):.2f}",
                    "Median |b| %": f"{np.nanmedian(b_abs):.2f}",
                    "P90 |a| %": f"{np.nanpercentile(a_abs, 90):.2f}",
                    "P90 |b| %": f"{np.nanpercentile(b_abs, 90):.2f}",
                    "MaxIter (w)": f"{res['maxiter'].sum() / n_total * 100:.1f}%",
                    "Optimizer (w)": f"{res['optimizer'].sum() / n_total * 100:.1f}%",
                    "BoundHit (w)": f"{res['bound_hit'].sum() / n_total * 100:.1f}%",
                    "Fail": f"{res['fail'].sum() / n_total * 100:.1f}%",
                }
            )
        print(f"--- {nb} compute budgets ({range_label}) ---")
        print(pd.DataFrame(rows).to_string(index=False))
        print("  (w) = warning only — estimate still counted in statistics")
        print()

    # ── Pairwise comparison frequencies for |b| error ──
    all_pairwise = compute_pairwise_frequencies(all_results, n_budgets_range)

    print("=" * 60)
    print("Pairwise |b| Error Comparisons")
    print("=" * 60)

    for nb in n_budgets_range:
        counts, n_total = all_pairwise[nb]
        rows = [
            {
                "Comparison": "VPNLS better than Approach 3",
                "Frequency": f"{counts['vpnls_beats_a3'] / n_total * 100:.1f}%",
            },
            {
                "Comparison": "Approach 3 better than VPNLS",
                "Frequency": f"{counts['a3_beats_vpnls'] / n_total * 100:.1f}%",
            },
            {
                "Comparison": "VPNLS tied with Approach 3",
                "Frequency": f"{counts['vpnls_ties_a3'] / n_total * 100:.1f}%",
            },
            {
                "Comparison": "VPNLS >= Approach 2",
                "Frequency": f"{counts['vpnls_beats_or_ties_a2'] / n_total * 100:.1f}%",
            },
            {
                "Comparison": "Approach 3 >= Approach 2",
                "Frequency": f"{counts['a3_beats_or_ties_a2'] / n_total * 100:.1f}%",
            },
        ]
        print(f"--- {nb} budgets (n={n_total}) ---")
        print(pd.DataFrame(rows).to_string(index=False))
        print()


# ── Entry Point ───────────────────────────────────────────────────────────────


def main():
    # ── Loss surface (asymmetric, ratio=3, Chinchilla A/B/E) ──
    exponent_sum = 0.62
    exponent_ratio = 3.0
    alpha = exponent_sum * exponent_ratio / (1 + exponent_ratio)
    beta = exponent_sum / (1 + exponent_ratio)
    A = 406.4
    B = 410.7
    E = 1.69

    # ── Compute budget grid ──
    n_budgets_range = [2, 3, 4, 5]
    c_min = 1e17
    c_max = 1e21

    # ── IsoFLOP sampling ──
    n_points_max = 32
    log_range = np.log10(8)  # ±8x grid width

    # ── Sampling bias ──
    drift_rate = np.log10(3)
    center_scale = 1.0

    # ── Noise ──
    noise_std = 0.1
    seed = 42
    n_repeats = 32

    # ── Approach 3 initialization ──
    a3_random_init = True  # True = random point within bounds; False = 5D grid search

    # ── Output ──
    output_path = "demo/scaling_law_estimation.png"
    isoflop_output_path = "demo/isoflop_curves.png"

    # ── Derived ──
    surface = LossSurface(A=A, B=B, E=E, alpha=alpha, beta=beta)
    n_points_range = np.array([4, 8, 16, 32])

    # ── Run ──
    console = Console()
    range_label = _log_range_to_label(log_range)
    all_results: dict[int, dict[str, dict]] = {}

    for nb in n_budgets_range:
        compute_budgets = np.geomspace(c_min, c_max, nb)
        console.print(
            f"[bold]Running:[/bold] {nb} budgets, {range_label}, {n_repeats} seeds ...",
            end=" ",
        )
        all_results[nb] = compute_exponent_errors(
            surface=surface,
            compute_budgets=compute_budgets,
            n_points_range=n_points_range,
            log_range=log_range,
            drift_rate=drift_rate,
            center_scale=center_scale,
            noise_std=noise_std,
            seed=seed,
            n_repeats=n_repeats,
            a3_random_init=a3_random_init,
        )
        console.print("[green]done[/green]")

    # ── IsoFLOP visualization (max budgets, max n_points, first seed only) ──
    isoflop_budgets = np.geomspace(c_min, c_max, max(n_budgets_range))
    isoflop_rng = np.random.default_rng(seed)
    isoflop_data = generate_isoflop_data(
        surface,
        isoflop_budgets,
        n_points_max,
        log_range,
        drift_rate,
        center_scale,
        noise_std,
        isoflop_rng,
    )
    create_isoflop_figure(
        isoflop_data,
        surface,
        isoflop_budgets,
        log_range,
        drift_rate,
        center_scale,
        noise_std,
        isoflop_output_path,
    )
    console.print(f"[bold green]Figure saved:[/bold green] {isoflop_output_path}")

    print_summary(
        all_results,
        surface,
        n_points_range,
        n_budgets_range,
        c_min,
        c_max,
        log_range,
        drift_rate,
        noise_std,
        n_repeats,
    )
    create_figure(
        all_results,
        surface,
        n_points_range,
        n_budgets_range,
        c_min,
        c_max,
        log_range,
        drift_rate,
        noise_std,
        n_repeats,
        output_path,
    )
    console.print(f"[bold green]Figure saved:[/bold green] {output_path}")


if __name__ == "__main__":
    main()
