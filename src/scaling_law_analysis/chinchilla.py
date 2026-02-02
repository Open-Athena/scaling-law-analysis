"""Chinchilla loss function and parameter recovery implementations.

This module provides:
- The Chinchilla loss function L(N, D) = E + A/N^α + B/D^β
- IsoFLOP sampling along constant compute contours
- Approach 2 parameter recovery via parabolic fits
"""

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple


# Chinchilla paper ground truth parameters
CHINCHILLA_PARAMS = {
    "A": 406.4,
    "B": 410.7,
    "E": 1.69,
    "alpha": 0.34,
    "beta": 0.28,
}


def chinchilla_loss(
    N: np.ndarray,
    D: np.ndarray,
    A: float = CHINCHILLA_PARAMS["A"],
    B: float = CHINCHILLA_PARAMS["B"],
    E: float = CHINCHILLA_PARAMS["E"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> np.ndarray:
    """Compute Chinchilla loss L(N, D) = E + A/N^α + B/D^β.

    Args:
        N: Number of parameters (can be array)
        D: Number of training tokens (can be array)
        A: Parameter scaling coefficient
        B: Data scaling coefficient
        E: Irreducible loss (entropy of natural text)
        alpha: Parameter scaling exponent
        beta: Data scaling exponent

    Returns:
        Loss values corresponding to each (N, D) pair
    """
    return E + A / (N**alpha) + B / (D**beta)


def optimal_allocation(
    C: float,
    A: float = CHINCHILLA_PARAMS["A"],
    B: float = CHINCHILLA_PARAMS["B"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> tuple[float, float]:
    """Compute optimal N* and D* for a given compute budget.

    The optimal allocation minimizes L(N, D) subject to C = 6ND.

    From the Chinchilla paper, the closed-form solution is:
        N* = G * (C/6)^(β/(α+β))
        D* = (1/G) * (C/6)^(α/(α+β))
    where G = (α*A / (β*B))^(1/(α+β))

    Args:
        C: Compute budget in FLOPs
        A, B: Scaling coefficients
        alpha, beta: Scaling exponents

    Returns:
        Tuple of (N*, D*) optimal parameter and token counts
    """
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    C_eff = C / 6  # Effective compute (C = 6ND approximation)

    N_opt = G * (C_eff ** (beta / (alpha + beta)))
    D_opt = (1 / G) * (C_eff ** (alpha / (alpha + beta)))

    return N_opt, D_opt


def isoflop_sample(
    C: float,
    n_points: int,
    log_range: float = 1.0,
    A: float = CHINCHILLA_PARAMS["A"],
    B: float = CHINCHILLA_PARAMS["B"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points along an IsoFLOP contour (constant compute budget).

    Points are sampled logarithmically around the optimal N* for the
    given compute budget. D is derived from the constraint C = 6ND.

    Args:
        C: Compute budget in FLOPs
        n_points: Number of points to sample
        log_range: Range in log10 space around optimal N (±log_range)
        A, B, alpha, beta: Chinchilla parameters

    Returns:
        Tuple of (N, D, L) arrays - parameter counts, token counts, and losses
    """
    N_opt, _ = optimal_allocation(C, A, B, alpha, beta)

    # Sample N logarithmically around optimal
    log_N_min = np.log10(N_opt) - log_range
    log_N_max = np.log10(N_opt) + log_range
    N = np.logspace(log_N_min, log_N_max, n_points)

    # Derive D from constraint C = 6ND
    D = C / (6 * N)

    # Compute loss at each point
    L = chinchilla_loss(N, D, A, B, alpha=alpha, beta=beta)

    return N, D, L


class ParabolaFit(NamedTuple):
    """Results from fitting a parabola to IsoFLOP data."""

    coeffs: np.ndarray  # Polynomial coefficients [a, b, c] for ax² + bx + c
    log_N_opt: float  # Log10 of optimal N from parabola minimum
    N_opt: float  # Optimal N from parabola minimum
    log_L_min: float  # Log10 of minimum loss
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
    parabola_fits: list[ParabolaFit]  # Per-budget parabola fits
    compute_budgets: np.ndarray  # Compute budgets used
    N_opts: np.ndarray  # Optimal N* from each parabola
    D_opts: np.ndarray  # Optimal D* derived from constraint


def fit_parabola(log_N: np.ndarray, log_L: np.ndarray) -> ParabolaFit:
    """Fit a parabola to log-loss vs log-N data.

    Args:
        log_N: Log10 of parameter counts
        log_L: Log10 of losses

    Returns:
        ParabolaFit with coefficients and minimum location
    """
    # Fit quadratic: log(L) = a*log(N)² + b*log(N) + c
    coeffs = np.polyfit(log_N, log_L, 2)
    a, b, c = coeffs

    # Minimum at log(N*) = -b / (2a)
    log_N_opt = -b / (2 * a)
    N_opt = 10**log_N_opt

    # Minimum loss value
    log_L_min = np.polyval(coeffs, log_N_opt)
    L_min = 10**log_L_min

    return ParabolaFit(
        coeffs=coeffs,
        log_N_opt=log_N_opt,
        N_opt=N_opt,
        log_L_min=log_L_min,
        L_min=L_min,
    )


def approach2_recover(
    compute_budgets: np.ndarray,
    n_points: int = 10,
    log_range: float = 1.0,
    A: float = CHINCHILLA_PARAMS["A"],
    B: float = CHINCHILLA_PARAMS["B"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> Approach2Result:
    """Recover scaling exponents using Chinchilla Approach 2.

    Stage 1: For each compute budget, fit a parabola to find optimal N*
    Stage 2: Fit power laws N* ∝ C^a and D* ∝ C^b

    Args:
        compute_budgets: Array of compute budgets (FLOPs)
        n_points: Number of points per IsoFLOP curve
        log_range: Sampling range in log10 space around optimal N
        A, B, alpha, beta: Chinchilla parameters for data generation

    Returns:
        Approach2Result with recovered exponents a and b
    """
    # Stage 1: Parabola fits to find N* at each compute budget
    parabola_fits = []
    N_opts = []

    for C in compute_budgets:
        N, D, L = isoflop_sample(C, n_points, log_range, A, B, alpha, beta)
        fit = fit_parabola(np.log10(N), np.log10(L))
        parabola_fits.append(fit)
        N_opts.append(fit.N_opt)

    N_opts = np.array(N_opts)
    D_opts = compute_budgets / (6 * N_opts)  # From constraint C = 6ND

    # Stage 2: Linear fits in log-log space
    log_C = np.log10(compute_budgets)
    a, a_intercept = np.polyfit(log_C, np.log10(N_opts), 1)
    b, b_intercept = np.polyfit(log_C, np.log10(D_opts), 1)

    return Approach2Result(
        a=a,
        b=b,
        a_intercept=a_intercept,
        b_intercept=b_intercept,
        parabola_fits=parabola_fits,
        compute_budgets=compute_budgets,
        N_opts=N_opts,
        D_opts=D_opts,
    )
