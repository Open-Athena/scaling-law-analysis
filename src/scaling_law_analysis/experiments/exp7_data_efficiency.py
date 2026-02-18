"""Experiment 7: Data Efficiency Comparison.

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

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.optimize import minimize, nnls
from scipy.stats import gaussian_kde, gmean

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    FitError,
    FitStatus,
    LossSurface,
    NonFiniteFitError,
    fit_grid_search,
    compute_center_offset,
    fit_parabola,
    fit_power_law,
    isoflop_sample,
)
from scaling_law_analysis.experiments.common import (
    ASYMMETRIC_CONFIG,
    log_range_to_label,
    prepare_output_dir,
)


@dataclass
class _PooledStats:
    gmean: float
    min: float
    max: float


@dataclass
class _ComparisonRow:
    method: str
    stats: _PooledStats
    pooled_errors: np.ndarray
    max_a: float
    max_b: float


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
        offset = compute_center_offset(C, compute_budgets, drift_rate, center_scale)
        N, D, L = isoflop_sample(C, n_points, log_range, offset, surface)
        if noise_std > 0:
            L = L + rng.normal(0, noise_std, size=L.shape)
        per_budget.append((N, D, L))
    return IsoFlopData(per_budget=per_budget)


# ── Fitting: Common Result ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExponentEstimate:
    """Recovered scaling exponents from any fitting method."""

    a: float  # Estimate of β/(α+β)
    b: float  # Estimate of α/(α+β)
    method: str
    status: FitStatus = FitStatus.CONVERGED
    status_message: str = ""

    @property
    def converged(self) -> bool:
        """True if the optimizer reported full convergence."""
        return self.status == FitStatus.CONVERGED


# ── Fitting: Approach 2 ──────────────────────────────────────────────────────


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
        N_opts.append(fit_parabola(np.log10(N), L).x_opt)
        D_opts.append(fit_parabola(np.log10(D), L).x_opt)

    N_opts_arr = np.array(N_opts)
    D_opts_arr = np.array(D_opts)

    a_exp = fit_power_law(compute_budgets, N_opts_arr).exponent
    b_exp = fit_power_law(compute_budgets, D_opts_arr).exponent

    return ExponentEstimate(a=a_exp, b=b_exp, method="Chinchilla Approach 2")


# ── Shared fitting diagnostics ────────────────────────────────────────────────


def _make_estimate(
    alpha: float,
    beta: float,
    method: str,
    status: FitStatus = FitStatus.CONVERGED,
    status_message: str = "",
) -> ExponentEstimate:
    """Compute ExponentEstimate from recovered alpha and beta."""
    return ExponentEstimate(
        a=beta / (alpha + beta),
        b=alpha / (alpha + beta),
        method=method,
        status=status,
        status_message=status_message,
    )


def _check_optimizer_result(
    result,
    method_name: str,
    maxiter: int,
) -> tuple[FitStatus, str]:
    """Determine FitStatus from a scipy optimizer result.

    Args:
        result: scipy.optimize.OptimizeResult.
        method_name: Human-readable name for status messages.
        maxiter: The maxiter setting used for this optimizer.

    Returns:
        (status, status_message) tuple.

    Raises:
        FitError: If result is None (optimizer produced no output at all).
    """
    if result is None:
        raise FitError(f"{method_name} optimization returned None.")
    if not result.success:
        if result.nit >= maxiter:
            return FitStatus.MAX_ITER, (
                f"{method_name} hit maxiter ({maxiter}): "
                f"{result.message} (iterations: {result.nit})"
            )
        return FitStatus.ABNORMAL, (
            f"{method_name} optimization failed: {result.message} "
            f"(iterations: {result.nit})"
        )
    return FitStatus.CONVERGED, ""


def _check_params_finite(method_name: str, **params: float) -> None:
    """Raise NonFiniteFitError if any parameter is NaN or Inf."""
    for name, val in params.items():
        if np.isnan(val) or np.isinf(val):
            raise NonFiniteFitError(f"{method_name}: {name}={val} is non-finite.")


def _check_params_at_bounds(
    method_name: str,
    named_bounds: list[tuple[str, float, float, float]],
    tol: float = 1e-6,
) -> str | None:
    """Return a message if any parameter is at or near a bound, else None.

    Args:
        method_name: Human-readable name for messages.
        named_bounds: List of (name, value, lo, hi) tuples.
        tol: Tolerance for bound proximity.
    """
    msgs = []
    for name, val, lo, hi in named_bounds:
        if val - lo < tol or hi - val < tol:
            msgs.append(f"{method_name}: {name}={val:.6f} at bound [{lo}, {hi}].")
    return "; ".join(msgs) if msgs else None


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

# Coarse 5D grid for initialization (8 values per parameter = 8^5 = 32768 points)
_A3_ALPHA_GRID = np.linspace(_ALPHA_GRID_MIN, _ALPHA_GRID_MAX, 8)
_A3_BETA_GRID = np.linspace(_BETA_GRID_MIN, _BETA_GRID_MAX, 8)
_A3_E_GRID = np.linspace(0.1, 5.0, 8)
_A3_A_GRID = np.logspace(1, 4, 8)
_A3_B_GRID = np.logspace(1, 4, 8)

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
    Initialization via coarse 5D grid search (8^5 = 32768 evaluations) or,
    when ``random_init`` is True, a single random starting point within bounds.

    Args:
        N: Array of parameter counts.
        D: Array of token counts.
        L: Array of loss values (same length as N and D).
        random_init: If True, use a random starting point instead of grid search.
        rng: Random generator (required when ``random_init`` is True).

    Returns:
        ExponentEstimate with recovered a and b.  Check ``status`` for
        non-CONVERGED outcomes (max_iter, abnormal, bound_hit).

    Raises:
        FitError: Only for hard failures (non-finite parameters).
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
        best_x0 = fit_grid_search(
            E_grid=_A3_E_GRID,
            A_grid=_A3_A_GRID,
            B_grid=_A3_B_GRID,
            alpha_grid=_A3_ALPHA_GRID,
            beta_grid=_A3_BETA_GRID,
            log_N=log_N,
            log_D=log_D,
            L=L,
        )

    result = minimize(
        rss,
        x0=best_x0,
        jac=rss_grad,
        method="L-BFGS-B",
        bounds=_A3_BOUNDS,
        options=_A3_LBFGSB_OPTIONS,
    )

    status, status_message = _check_optimizer_result(
        result, method_name, _A3_LBFGSB_OPTIONS["maxiter"]
    )
    assert result is not None  # ensured by _check_optimizer_result

    E, A, B, alpha, beta = (
        float(result.x[0]),
        float(result.x[1]),
        float(result.x[2]),
        float(result.x[3]),
        float(result.x[4]),
    )
    _check_params_finite(method_name, E=E, A=A, B=B, alpha=alpha, beta=beta)

    if status == FitStatus.CONVERGED:
        bound_msg = _check_params_at_bounds(
            method_name,
            [
                (name, val, lo, hi)
                for name, val, (lo, hi) in zip(
                    ["E", "A", "B", "α", "β"], [E, A, B, alpha, beta], _A3_BOUNDS
                )
            ],
        )
        if bound_msg is not None:
            status = FitStatus.BOUND_HIT
            status_message = bound_msg

    return _make_estimate(
        alpha, beta, method_name, status=status, status_message=status_message
    )


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
    _A3_GRID_SIZE == 32 * _VP_GRID_SIZE
), f"Expected Approach 3 grid to be 32× VPNLS grid, got {_A3_GRID_SIZE}/{_VP_GRID_SIZE}={_A3_GRID_SIZE/_VP_GRID_SIZE}×"


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
        ExponentEstimate with recovered a and b.  Check ``status`` for
        non-CONVERGED outcomes (max_iter, abnormal, bound_hit).

    Raises:
        FitError: Only for hard failures (non-finite parameters).
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

    status, status_message = _check_optimizer_result(
        result, method_name, _VP_NELDER_MEAD_OPTIONS["maxiter"]
    )
    assert result is not None  # ensured by _check_optimizer_result

    alpha, beta = float(result.x[0]), float(result.x[1])

    rss, (E, A, B) = _vp_compute_rss_and_params(alpha, beta, log_N, log_D, L)

    _check_params_finite(method_name, E=E, A=A, B=B, alpha=alpha, beta=beta)

    if status == FitStatus.CONVERGED:
        bound_msg = _check_params_at_bounds(
            method_name,
            [
                ("E", E, *_E_BOUNDS),
                ("A", A, *_A_BOUNDS),
                ("B", B, *_B_BOUNDS),
                ("α", alpha, *_ALPHA_BOUNDS),
                ("β", beta, *_BETA_BOUNDS),
            ],
        )
        if bound_msg is not None:
            status = FitStatus.BOUND_HIT
            status_message = bound_msg

    return _make_estimate(
        alpha, beta, method_name, status=status, status_message=status_message
    )


# ── Experiment ────────────────────────────────────────────────────────────────

METHOD_NAMES = [
    "Chinchilla Approach 2",
    "Chinchilla Approach 3 (grid init)",
    "Chinchilla Approach 3 (random init)",
    "VPNLS",
]

METHOD_STYLES = {
    "Chinchilla Approach 2": {"color": "#d62728", "marker": "s"},
    "Chinchilla Approach 3 (grid init)": {"color": "#ff7f0e", "marker": "^"},
    "Chinchilla Approach 3 (random init)": {"color": "#9467bd", "marker": "v"},
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
) -> dict[str, dict]:
    """Compute a/b exponent errors for all methods across n_points and seeds.

    For each seed and n_points in n_points_range, generates data and fits
    all four methods, recording relative error in recovered a and b exponents.

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

    Returns:
        Dict mapping method name → dict with 2D arrays (n_repeats × n_points):
            "a_errors", "b_errors" — relative errors (NaN for hard failures).
            Status flags (estimate still counted):
                "maxiter"   — optimizer hit iteration limit.
                "abnormal"  — optimizer reported failure (e.g. ABNORMAL).
                "bound_hit" — parameter at bound.
            Hard failures (estimate is NaN):
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
            "abnormal": np.zeros((n_repeats, n_pts_len), dtype=bool),
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

            a3_random_rng = rng.spawn(1)[0]
            for method_name, fit_fn in [
                ("Chinchilla Approach 2", lambda d: fit_approach2(d, compute_budgets)),
                (
                    "Chinchilla Approach 3 (grid init)",
                    lambda d: fit_approach3(d.N, d.D, d.L),
                ),
                (
                    "Chinchilla Approach 3 (random init)",
                    lambda d: fit_approach3(
                        d.N, d.D, d.L, random_init=True, rng=a3_random_rng
                    ),
                ),
                ("VPNLS", lambda d: fit_vpnls(d.N, d.D, d.L)),
            ]:
                res = results[method_name]
                try:
                    est = fit_fn(data)
                except FitError:
                    res["fail"][r, i] = True
                    res["a_errors"][r, i] = np.nan
                    res["b_errors"][r, i] = np.nan
                    continue

                # Record status flags
                if est.status == FitStatus.MAX_ITER:
                    res["maxiter"][r, i] = True
                elif est.status == FitStatus.ABNORMAL:
                    res["abnormal"][r, i] = True
                elif est.status == FitStatus.BOUND_HIT:
                    res["bound_hit"][r, i] = True

                res["a_errors"][r, i] = (est.a - true_a) / true_a
                res["b_errors"][r, i] = (est.b - true_b) / true_b

    return results


# ── Visualization ─────────────────────────────────────────────────────────────


def create_figure(
    noise_results: dict[float, dict[int, dict[str, dict]]],
    noise_std_levels: list[float],
    surface: LossSurface,
    n_points_range: np.ndarray,
    n_budgets_range: list[int],
    c_min: float,
    c_max: float,
    log_range: float,
    drift_rate: float,
    n_repeats: int,
    output_path: str,
) -> None:
    """Create combined exponent recovery boxplot figure.

    Shows only the lowest and highest noise levels.
    Layout: len(n_budgets_range) rows × 4 columns (2 noise × 2 exponents),
    with a subtle vertical separator between the two noise groups.
    """
    range_label = log_range_to_label(log_range)
    exponent_labels = ["a", "b"]
    error_keys = ["a_errors", "b_errors"]
    n_rows = len(n_budgets_range)

    shown_noise = [min(noise_std_levels), max(noise_std_levels)]
    n_shown = len(shown_noise)
    n_cols = 2 * n_shown
    n_methods = len(METHOD_NAMES)
    log_positions = np.log2(n_points_range.astype(float))
    box_width = 0.15
    offsets = np.linspace(
        -(n_methods - 1) / 2 * box_width, (n_methods - 1) / 2 * box_width, n_methods
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.8 * n_cols, 2.3 * n_rows),
        gridspec_kw={"hspace": 0.45, "wspace": 0.25},
        squeeze=False,
    )

    for noise_idx, noise_std in enumerate(shown_noise):
        all_results = noise_results[noise_std]
        for row, nb in enumerate(n_budgets_range):
            results = all_results[nb]

            for exp_idx, (exp_label, err_key) in enumerate(
                zip(exponent_labels, error_keys)
            ):
                col = noise_idx * 2 + exp_idx
                ax = axes[row, col]

                for m_idx, method_name in enumerate(METHOD_NAMES):
                    style = METHOD_STYLES[method_name]
                    errors_2d = np.abs(results[method_name][err_key]) * 100
                    positions = log_positions + offsets[m_idx]

                    ax.boxplot(
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
                            "markersize": 2,
                            "alpha": 0.4,
                            "markerfacecolor": style["color"],
                        },
                        medianprops={"color": "white", "linewidth": 0.8},
                        whiskerprops={"color": style["color"], "linewidth": 0.6},
                        capprops={"color": style["color"], "linewidth": 0.6},
                        boxprops={
                            "facecolor": style["color"],
                            "alpha": 0.6,
                            "edgecolor": style["color"],
                        },
                    )

                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                ax.set_xticks(log_positions)
                ax.set_xticklabels(n_points_range, fontsize=6)
                ax.set_xlim(log_positions[0] - 0.5, log_positions[-1] + 0.5)

                ax.set_title(
                    f"{exp_label} (σ={noise_std}), {nb} budgets",
                    fontsize=7,
                )

                if row == n_rows - 1:
                    ax.set_xlabel("Points/curve", fontsize=7)
                if col == 0:
                    ax.set_ylabel("Abs. rel. error (%)", fontsize=7)
                ax.tick_params(labelsize=6)

    # Subtle vertical line between the two noise groups
    mid_x = 0.5
    fig.add_artist(
        plt.Line2D(
            [mid_x, mid_x],
            [0.02, 0.95],
            transform=fig.transFigure,
            color="#cccccc",
            linewidth=1,
            linestyle="--",
            zorder=0,
        )
    )

    fig.suptitle(
        "Exponent Recovery: Boxplots by Noise Level\n"
        f"$\\alpha$={surface.alpha:.3f}, $\\beta$={surface.beta:.3f}, "
        f"grid = {range_label}, "
        f"drift = {drift_rate:.3f}, "
        f"{n_repeats} seeds, "
        f"budgets: {c_min:.0e}\u2013{c_max:.0e} FLOPs",
        fontsize=10,
        y=1.01,
    )

    handles = [
        Patch(facecolor=METHOD_STYLES[m]["color"], alpha=0.6, label=m)
        for m in METHOD_NAMES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METHOD_NAMES),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_isoflop_figure(
    datasets: list[tuple[float, IsoFlopData]],
    surface: LossSurface,
    compute_budgets: np.ndarray,
    log_range: float,
    drift_rate: float,
    center_scale: float,
    output_path: str,
) -> None:
    """Create and save IsoFLOP curves for multiple noise levels.

    Layout: 2 rows (L vs N, L vs D) × len(datasets) columns (one per noise
    level).  Noise level is shown as a prominent column header.

    Args:
        datasets: List of (noise_std, IsoFlopData) pairs, one per noise level.
        surface: Loss surface (for noiseless reference curves and true optima).
        compute_budgets: Compute budgets used (for labels).
        log_range: Sampling grid width in log10 space (for title).
        drift_rate: Drift rate used for sampling bias.
        center_scale: Center scale used for sampling bias.
        output_path: Path to save the figure.
    """
    range_label = log_range_to_label(log_range)
    n_budgets = len(compute_budgets)
    n_noise = len(datasets)
    colors = plt.colormaps["viridis"](np.linspace(0.1, 0.9, n_budgets))

    x_configs = [
        ("log$_{10}$(N)", "N", surface.N_opt),
        ("log$_{10}$(D)", "D", surface.D_opt),
    ]

    fig, axes = plt.subplots(
        2,
        n_noise,
        figsize=(4.0 * n_noise, 6),
        squeeze=False,
        gridspec_kw={"hspace": 0.2, "wspace": 0.2},
    )

    for col_idx, (noise_std, data) in enumerate(datasets):
        for row_idx, (x_label, x_param, opt_fn) in enumerate(x_configs):
            ax = axes[row_idx, col_idx]
            for i, ((N, D, L), C) in enumerate(zip(data.per_budget, compute_budgets)):
                x_vals = np.log10(N) if x_param == "N" else np.log10(D)
                ax.scatter(
                    x_vals,
                    L,
                    color=colors[i],
                    s=25,
                    alpha=0.7,
                    zorder=2,
                    label=f"C = {C:.0e}" if col_idx == 0 and row_idx == 0 else None,
                )
                L_true = surface.loss(N, D)
                ax.plot(
                    x_vals, L_true, color=colors[i], linewidth=1, alpha=0.5, zorder=1
                )

                x_opt = np.log10(opt_fn(C))
                L_opt = float(surface.loss(surface.N_opt(C), surface.D_opt(C)))
                ax.scatter(
                    [x_opt],
                    [L_opt],
                    c="red",
                    marker="x",
                    s=80,
                    zorder=4,
                    linewidths=2,
                )

                center_offset = compute_center_offset(
                    C, compute_budgets, drift_rate, center_scale
                )
                N_center = surface.N_opt(C) * 10**center_offset
                D_center = C / (6 * N_center)
                L_center = float(surface.loss(N_center, D_center))
                x_center = np.log10(N_center) if x_param == "N" else np.log10(D_center)
                ax.scatter(
                    [x_center],
                    [L_center],
                    c="black",
                    marker="D",
                    s=70,
                    zorder=5,
                    linewidths=1,
                    edgecolors="white",
                )

            if row_idx == 1:
                ax.set_xlabel(x_label, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Loss L(N, D)", fontsize=9)

            if row_idx == 0:
                ax.set_title(f"σ = {noise_std}", fontsize=11)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Row labels on the right edge of the last column
            if col_idx == n_noise - 1:
                ax.annotate(
                    f"L vs {x_param}",
                    xy=(1.02, 0.5),
                    xycoords="axes fraction",
                    fontsize=9,
                    ha="left",
                    va="center",
                    rotation=-90,
                    color="#555555",
                )

            if col_idx == 0 and row_idx == 0:
                ax.scatter(
                    [],
                    [],
                    c="red",
                    marker="x",
                    s=60,
                    linewidths=2,
                    label="True optimum",
                )
                ax.scatter(
                    [],
                    [],
                    c="black",
                    marker="D",
                    s=50,
                    linewidths=1,
                    edgecolors="white",
                    label="Sampling center",
                )
                ax.legend(fontsize=6, loc="upper right")

    n_pts = len(datasets[0][1].per_budget[0][0])
    fig.suptitle(
        f"IsoFLOP Samples ({n_budgets} budgets, {n_pts} points/curve, "
        f"grid width = {range_label})",
        fontsize=11,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Console Output ────────────────────────────────────────────────────────────


def _safe_gmean(arr: np.ndarray) -> float:
    """Geometric mean of positive values, ignoring NaN and zeros."""
    vals = arr[np.isfinite(arr) & (arr > 0)]
    if len(vals) == 0:
        return np.nan
    return float(gmean(vals))


def print_summary(
    noise_results: dict[float, dict[int, dict[str, dict]]],
    surface: LossSurface,
    n_points_range: np.ndarray,
    noise_std_levels: list[float],
    n_budgets_range: list[int],
    c_min: float,
    c_max: float,
    log_range: float,
    drift_rate: float,
    n_repeats: int,
) -> None:
    """Print summary tables to console.

    Args:
        noise_results: Dict mapping noise_std → n_budgets → method results.
        surface: Loss surface (for ground truth display).
        n_points_range: Array of n_points values.
        noise_std_levels: Noise standard deviations tested.
        n_budgets_range: List of n_budgets values tested.
        c_min: Minimum compute budget in FLOPs.
        c_max: Maximum compute budget in FLOPs.
        log_range: Sampling grid width in log10 space.
        drift_rate: Drift rate used.
        n_repeats: Number of repeats per configuration.
    """
    range_label = log_range_to_label(log_range)

    print("=" * 60)
    print("Scaling Law Estimation — Data Efficiency")
    print("=" * 60)

    cfg = {
        "Surface": f"α={surface.alpha:.4f}, β={surface.beta:.4f}, A={surface.A}, B={surface.B}, E={surface.E}",
        "True exponents": f"a={surface.a:.6f}, b={surface.b:.6f}",
        "Grid width": f"{range_label} (log_range={log_range:.4f})",
        "Drift rate": f"{drift_rate:.6f}",
        "Noise σ": f"{noise_std_levels}",
        "Budget range": f"{c_min:.0e} – {c_max:.0e} FLOPs",
        "n_budgets": f"{n_budgets_range}",
        "n_points range": f"{n_points_range[0]} – {n_points_range[-1]}",
        "n_repeats": f"{n_repeats}",
    }
    config_df = pd.DataFrame(list(cfg.items()), columns=["Parameter", "Value"])
    print(config_df.to_string(index=False))
    print()

    rows = []
    for noise_std in noise_std_levels:
        all_results = noise_results[noise_std]
        for nb in n_budgets_range:
            results = all_results[nb]
            for method_name in METHOD_NAMES:
                res = results[method_name]
                a_abs = np.abs(res["a_errors"]) * 100
                b_abs = np.abs(res["b_errors"]) * 100
                n_total = res["maxiter"].size
                rows.append(
                    {
                        "Noise": noise_std,
                        "Budgets": nb,
                        "Method": method_name,
                        "GMean |a| %": f"{_safe_gmean(a_abs):.2f}",
                        "GMean |b| %": f"{_safe_gmean(b_abs):.2f}",
                        "Max |a| %": f"{np.nanmax(a_abs):.2f}",
                        "Max |b| %": f"{np.nanmax(b_abs):.2f}",
                        "MaxIter (s)": f"{res['maxiter'].sum() / n_total * 100:.1f}%",
                        "Abnormal (s)": f"{res['abnormal'].sum() / n_total * 100:.1f}%",
                        "BoundHit (s)": f"{res['bound_hit'].sum() / n_total * 100:.1f}%",
                        "Fail": f"{res['fail'].sum() / n_total * 100:.1f}%",
                    }
                )
    n_trials = n_repeats * len(n_points_range)
    print(f"  n = {n_trials} trials per (noise, budget, method)")
    print(pd.DataFrame(rows).to_string(index=False))
    print("  (s) = status flag only — estimate still counted in statistics")
    print()


# ── Entry Point ───────────────────────────────────────────────────────────────


def _pool_method_errors(
    noise_results: dict[float, dict[int, dict[str, dict]]],
    noise_std_levels: list[float],
    n_budgets_range: list[int],
) -> list[_ComparisonRow]:
    """Pool absolute relative errors across all noise levels, budgets, n_points, seeds."""
    row_data: list[_ComparisonRow] = []
    for method_name in METHOD_NAMES:
        all_a: list[np.ndarray] = []
        all_b: list[np.ndarray] = []

        for noise_std in noise_std_levels:
            for nb in n_budgets_range:
                res = noise_results[noise_std][nb][method_name]
                all_a.append(np.abs(res["a_errors"]).ravel() * 100)
                all_b.append(np.abs(res["b_errors"]).ravel() * 100)

        pooled_a = np.concatenate(all_a)
        pooled_b = np.concatenate(all_b)
        pooled_all = np.concatenate([pooled_a, pooled_b])

        valid = pooled_all[np.isfinite(pooled_all) & (pooled_all > 0)]
        if len(valid) > 0:
            stats = _PooledStats(
                gmean=float(gmean(valid)),
                min=float(valid.min()),
                max=float(valid.max()),
            )
        else:
            stats = _PooledStats(gmean=np.nan, min=np.nan, max=np.nan)

        row_data.append(
            _ComparisonRow(
                method=method_name,
                stats=stats,
                pooled_errors=pooled_all,
                max_a=float(np.nanmax(pooled_a)),
                max_b=float(np.nanmax(pooled_b)),
            )
        )
    return row_data


def create_method_comparison_figure(
    noise_results: dict[float, dict[int, dict[str, dict]]],
    noise_std_levels: list[float],
    n_budgets_range: list[int],
    output_path: str,
) -> None:
    """Create method comparison figure: dot-range plot + max-error heatmap.

    Layout: 1×2 figure with one row per method (4 rows), pooled across all
    noise levels, compute budgets, n_points, and seeds.  Sorted by max error
    descending.
    - Left: geometric mean with min-max bars, rug plot below, KDE line above.
    - Right: narrow heatmap of max |a| and max |b| errors.
    """
    row_data = _pool_method_errors(noise_results, noise_std_levels, n_budgets_range)

    # Sort descending by max error
    row_data.sort(key=lambda rd: -rd.stats.max if not np.isnan(rd.stats.max) else 0.0)

    n_rows = len(row_data)
    y_positions = np.arange(n_rows)
    dot_color = "#333333"
    rug_kde_alpha = 0.35

    fig_height = 0.7 * n_rows + 1.5
    fig, (ax_dot, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(12, fig_height),
        gridspec_kw={"width_ratios": [5, 1], "wspace": 0.02},
        sharey=True,
        layout="constrained",
    )

    # Compute shared x-axis limits in log space for KDE
    all_finite = np.concatenate(
        [rd.pooled_errors[np.isfinite(rd.pooled_errors)] for rd in row_data]
    )
    if len(all_finite) > 0:
        x_lo = max(all_finite[all_finite > 0].min() * 0.5, 1e-4)
        x_hi = all_finite.max() * 2.0
    else:
        x_lo, x_hi = 1e-2, 1e2
    kde_x = np.geomspace(x_lo, x_hi, 300)

    # ── Left panel: dot-range plot with rug and KDE ──
    for idx, rd in enumerate(row_data):
        s = rd.stats
        y = y_positions[idx]
        valid = rd.pooled_errors[np.isfinite(rd.pooled_errors) & (rd.pooled_errors > 0)]

        # Rug plot (small ticks below the dot row)
        if len(valid) > 0:
            ax_dot.scatter(
                valid,
                np.full_like(valid, y + 0.25),
                marker="|",
                s=15,
                color=dot_color,
                alpha=1.0,
                zorder=2,
                linewidths=0.7,
            )

        # KDE line (above the dot row), clipped to actual data range
        if len(valid) > 20:
            try:
                kde = gaussian_kde(np.log10(valid), bw_method=0.3)
                v_min, v_max = valid.min(), valid.max()
                kde_x_clip = kde_x[(kde_x >= v_min) & (kde_x <= v_max)]
                if len(kde_x_clip) > 1:
                    density = kde(np.log10(kde_x_clip))
                    density_norm = density / density.max() * 0.3
                    ax_dot.plot(
                        kde_x_clip,
                        y - density_norm,
                        color=dot_color,
                        alpha=rug_kde_alpha,
                        linewidth=0.8,
                        zorder=2,
                    )
            except np.linalg.LinAlgError:
                pass

        # Dot-range (geometric mean with min-max error bars)
        if not np.isnan(s.gmean):
            xerr_lo = max(0.0, s.gmean - s.min)
            xerr_hi = max(0.0, s.max - s.gmean)
            ax_dot.errorbar(
                s.gmean,
                y,
                xerr=[[xerr_lo], [xerr_hi]],
                fmt="o",
                color=dot_color,
                markersize=7,
                capsize=4,
                linewidth=1.5,
                markeredgewidth=1.5,
                zorder=5,
            )
        else:
            ax_dot.scatter([1e-1], [y], marker="x", s=60, color=dot_color, zorder=5)

    ax_dot.set_xscale("log")
    ax_dot.set_yticks(y_positions)
    ax_dot.set_yticklabels([rd.method for rd in row_data], fontsize=9)
    ax_dot.set_xlabel("Absolute relative error (%)", fontsize=11)
    ax_dot.set_title("Geometric Mean Error (min–max range)", fontsize=11)
    ax_dot.grid(True, axis="x", alpha=0.3)
    ax_dot.invert_yaxis()

    # ── Right panel: max-error heatmap ──
    col_labels = ["a", "b"]
    n_params = len(col_labels)

    heat_data = np.full((n_rows, n_params), np.nan)
    for idx, rd in enumerate(row_data):
        heat_data[idx, 0] = rd.max_a
        heat_data[idx, 1] = rd.max_b

    finite_vals = heat_data[np.isfinite(heat_data)]
    vmin = max(finite_vals.min(), 1e-10) if len(finite_vals) > 0 else 1e-10
    vmax = finite_vals.max() if len(finite_vals) > 0 else 1e2

    cmap_wb = LinearSegmentedColormap.from_list("white_black", ["#ffffff", "#000000"])
    ax_heat.imshow(
        heat_data,
        aspect="auto",
        cmap=cmap_wb,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )

    for idx in range(n_rows):
        for p_idx in range(n_params):
            val = heat_data[idx, p_idx]
            if np.isfinite(val) and val > 0:
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
            log_frac = 0.0
            if np.isfinite(val) and val > 0 and vmax > vmin:
                log_frac = (np.log10(val) - np.log10(vmin)) / (
                    np.log10(vmax) - np.log10(vmin)
                )
            text_color = "white" if log_frac > 0.5 else "black"
            ax_heat.text(
                p_idx,
                idx,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax_heat.set_xticks(np.arange(n_params))
    ax_heat.set_xticklabels(col_labels, fontsize=11)
    ax_heat.set_xlabel("Exponent", fontsize=11)
    ax_heat.set_title("Max Error", fontsize=11)
    ax_heat.tick_params(left=False)

    fig.suptitle(
        "Method Comparison: Exponent Inference Accuracy",
        fontsize=13,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_method_comparison_csv(
    noise_results: dict[float, dict[int, dict[str, dict]]],
    noise_std_levels: list[float],
    n_budgets_range: list[int],
    n_points_range: np.ndarray,
    output_path: str,
) -> None:
    """Export method comparison data to CSV.

    Includes error statistics, status/error counts per type, top 8 largest
    errors, and metadata about what was aggregated over.
    """
    rows = []
    for method_name in METHOD_NAMES:
        all_a: list[np.ndarray] = []
        all_b: list[np.ndarray] = []
        n_maxiter = 0
        n_abnormal = 0
        n_bound_hit = 0
        n_fail = 0
        total_fits = 0

        for noise_std in noise_std_levels:
            for nb in n_budgets_range:
                res = noise_results[noise_std][nb][method_name]
                all_a.append(np.abs(res["a_errors"]).ravel() * 100)
                all_b.append(np.abs(res["b_errors"]).ravel() * 100)
                n_maxiter += int(res["maxiter"].sum())
                n_abnormal += int(res["abnormal"].sum())
                n_bound_hit += int(res["bound_hit"].sum())
                n_fail += int(res["fail"].sum())
                total_fits += res["fail"].size

        pooled_a = np.concatenate(all_a)
        pooled_b = np.concatenate(all_b)
        pooled_all = np.concatenate([pooled_a, pooled_b])

        valid = pooled_all[np.isfinite(pooled_all) & (pooled_all > 0)]
        gm = float(gmean(valid)) if len(valid) > 0 else np.nan

        sorted_desc = np.sort(pooled_all[np.isfinite(pooled_all)])[::-1]
        top8 = [
            float(sorted_desc[i]) if i < len(sorted_desc) else np.nan for i in range(8)
        ]

        row: dict[str, object] = {
            "method": method_name,
            "n_noise_levels": len(noise_std_levels),
            "n_budgets": len(n_budgets_range),
            "n_points_per_curve": len(n_points_range),
            "total_fits": total_fits,
            "min": (
                float(np.nanmin(pooled_all))
                if np.any(np.isfinite(pooled_all))
                else np.nan
            ),
            "gmean": gm,
            "max": (
                float(np.nanmax(pooled_all))
                if np.any(np.isfinite(pooled_all))
                else np.nan
            ),
            "max_a": (
                float(np.nanmax(pooled_a)) if np.any(np.isfinite(pooled_a)) else np.nan
            ),
            "max_b": (
                float(np.nanmax(pooled_b)) if np.any(np.isfinite(pooled_b)) else np.nan
            ),
            "n_maxiter": n_maxiter,
            "n_abnormal": n_abnormal,
            "n_bound_hit": n_bound_hit,
            "n_fail": n_fail,
        }
        for rank, val in enumerate(top8, 1):
            row[f"top{rank}"] = val
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    """Run Experiment 7: data efficiency comparison."""
    print("=" * 70)
    print("Experiment 7: Data Efficiency Comparison")
    print("=" * 70)

    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp7")

    surface = ASYMMETRIC_CONFIG.loss

    n_budgets_range = [2, 3, 4]
    c_min = 1e17
    c_max = 1e21

    n_points_range = np.array([4, 8, 16, 32])
    isoflop_fig_n_points = 16
    if isoflop_fig_n_points not in n_points_range:
        raise ValueError(
            f"isoflop_fig_n_points={isoflop_fig_n_points} must be one of "
            f"n_points_range={n_points_range.tolist()}"
        )
    log_range = np.log10(8)  # ±8x grid width

    drift_rate = np.log10(3)
    center_scale = 1.0

    noise_std_levels = [0.05, 0.1, 0.2]
    seed = 42
    n_repeats = 32

    console = Console()
    range_label = log_range_to_label(log_range)
    noise_results: dict[float, dict[int, dict[str, dict]]] = {}

    for noise_std in noise_std_levels:
        console.print(f"\n[bold cyan]Noise σ = {noise_std}[/bold cyan]")
        all_results: dict[int, dict[str, dict]] = {}
        for nb in n_budgets_range:
            compute_budgets = np.geomspace(c_min, c_max, nb)
            console.print(
                f"  [bold]Running:[/bold] {nb} budgets, {range_label}, "
                f"{n_repeats} seeds ...",
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
            )
            console.print("[green]done[/green]")
        noise_results[noise_std] = all_results

    # IsoFLOP visualization (all noise levels, max budgets, cherry-picked n_points)
    isoflop_budgets = np.geomspace(c_min, c_max, max(n_budgets_range))
    isoflop_datasets: list[tuple[float, IsoFlopData]] = []
    for ns in noise_std_levels:
        isoflop_rng = np.random.default_rng(seed)
        isoflop_datasets.append(
            (
                ns,
                generate_isoflop_data(
                    surface,
                    isoflop_budgets,
                    isoflop_fig_n_points,
                    log_range,
                    drift_rate,
                    center_scale,
                    ns,
                    isoflop_rng,
                ),
            )
        )
    isoflop_path = output_dir / "isoflop_curves.png"
    create_isoflop_figure(
        isoflop_datasets,
        surface,
        isoflop_budgets,
        log_range,
        drift_rate,
        center_scale,
        str(isoflop_path),
    )
    print(f"Saved: {isoflop_path}")

    print_summary(
        noise_results,
        surface,
        n_points_range,
        noise_std_levels,
        n_budgets_range,
        c_min,
        c_max,
        log_range,
        drift_rate,
        n_repeats,
    )

    # Combined boxplot figure (all noise levels)
    errors_path = output_dir / "exponent_inference_errors.png"
    create_figure(
        noise_results,
        noise_std_levels,
        surface,
        n_points_range,
        n_budgets_range,
        c_min,
        c_max,
        log_range,
        drift_rate,
        n_repeats,
        str(errors_path),
    )
    print(f"Saved: {errors_path}")

    # Method comparison figure
    comparison_path = output_dir / "exponent_inference.png"
    create_method_comparison_figure(
        noise_results,
        noise_std_levels,
        n_budgets_range,
        str(comparison_path),
    )
    print(f"Saved: {comparison_path}")

    # Method comparison CSV
    csv_path = output_dir / "exponent_inference.csv"
    export_method_comparison_csv(
        noise_results,
        noise_std_levels,
        n_budgets_range,
        n_points_range,
        str(csv_path),
    )

    print("\nExperiment 7 complete.")


if __name__ == "__main__":
    main()
