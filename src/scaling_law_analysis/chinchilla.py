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
    alpha: float,
    beta: float,
    A: float,
    B: float,
    E: float,
) -> np.ndarray:
    """Compute Chinchilla loss L(N, D) = E + A/N^α + B/D^β.

    Args:
        N: Number of parameters (can be array)
        D: Number of training tokens (can be array)
        alpha: Parameter scaling exponent
        beta: Data scaling exponent
        A: Parameter scaling coefficient
        B: Data scaling coefficient
        E: Irreducible loss (entropy of natural text)

    Returns:
        Loss values corresponding to each (N, D) pair
    """
    return E + A / (N**alpha) + B / (D**beta)


def optimal_allocation(
    C: float,
    alpha: float,
    beta: float,
    A: float,
    B: float,
) -> tuple[float, float]:
    """Compute optimal N* and D* for a given compute budget.

    The optimal allocation minimizes L(N, D) subject to C = 6ND.

    From the Chinchilla paper, the closed-form solution is:
        N* = G * (C/6)^(β/(α+β))
        D* = (1/G) * (C/6)^(α/(α+β))
    where G = (α*A / (β*B))^(1/(α+β))

    Args:
        C: Compute budget in FLOPs
        alpha, beta: Scaling exponents
        A, B: Scaling coefficients

    Returns:
        Tuple of (N*, D*) optimal parameter and token counts
    """
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    C_eff = C / 6  # Effective compute (C = 6ND approximation)

    N_opt = G * (C_eff ** (beta / (alpha + beta)))
    D_opt = (1 / G) * (C_eff ** (alpha / (alpha + beta)))

    return N_opt, D_opt


def compute_center_offset(
    C: float,
    compute_budgets: np.ndarray,
    drift_rate: float,
    center_scale: float,
) -> float:
    """Compute the sampling center offset for a given compute budget.

    Combines two independent effects:
    1. drift_rate: Linear drift in log-compute space
       - At min compute: offset = -drift_rate (smaller N)
       - At max compute: offset = +drift_rate (larger N)
    2. center_scale: Constant multiplicative factor applied to all centers
       - scale > 1: all centers shifted right (larger N)
       - scale < 1: all centers shifted left (smaller N)

    Both effects are additive in log10 space.

    Args:
        C: Compute budget for which to compute offset
        compute_budgets: Array of all compute budgets (for drift normalization)
        drift_rate: Rate at which sampling center drifts from optimal
        center_scale: Constant multiplier applied to all sampling centers

    Returns:
        Total center offset in log10 units
    """
    offset = 0.0

    # Add constant scale offset
    if center_scale != 1.0:
        offset += np.log10(center_scale)

    # Add linear drift offset
    if drift_rate != 0.0:
        log_C_all = np.log10(compute_budgets)
        log_C_mid = (log_C_all.min() + log_C_all.max()) / 2
        log_C_half_range = (log_C_all.max() - log_C_all.min()) / 2

        if log_C_half_range > 0:
            normalized_log_C = (np.log10(C) - log_C_mid) / log_C_half_range
            offset += drift_rate * normalized_log_C

    return offset


def isoflop_sample(
    C: float,
    n_points: int,
    log_range: float,
    center_offset: float,
    alpha: float,
    beta: float,
    A: float,
    B: float,
    E: float,
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
        alpha, beta: Scaling exponents
        A, B: Scaling coefficients
        E: Irreducible loss

    Returns:
        Tuple of (N, D, L) arrays - parameter counts, token counts, and losses
    """
    N_opt, _ = optimal_allocation(C=C, alpha=alpha, beta=beta, A=A, B=B)

    # Sample N logarithmically around (possibly offset) center
    log_N_center = np.log10(N_opt) + center_offset
    log_N_min = log_N_center - log_range
    log_N_max = log_N_center + log_range
    N = np.logspace(log_N_min, log_N_max, n_points)

    # Derive D from constraint C = 6ND
    D = C / (6 * N)

    # Compute loss at each point
    L = chinchilla_loss(N=N, D=D, alpha=alpha, beta=beta, A=A, B=B, E=E)

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


def approach2_recover(
    compute_budgets: np.ndarray,
    drift_rate: float,
    center_scale: float,
    n_points: int,
    log_range: float,
    alpha: float,
    beta: float,
    A: float,
    B: float,
    E: float,
) -> Approach2Result:
    """Recover scaling exponents using Chinchilla Approach 2.

    Stage 1: For each compute budget, fit parabolas to find optimal N* and D*
    Stage 2: Fit power laws N* ∝ C^a and D* ∝ C^b

    Args:
        compute_budgets: Array of compute budgets (FLOPs)
        drift_rate: Rate at which sampling center drifts from optimal as a
                    function of compute budget. When non-zero, centers are shifted
                    left (smaller N) at low compute and right (larger N) at high
                    compute. The drift is linear in log-compute space and measured
                    in log10 units of N. The effect is constant regardless of log_range.
        center_scale: Constant multiplier applied to all sampling centers.
                      When 1.0, centers are at true optimal N*.
                      When >1, all centers are shifted right (larger N).
                      When <1, all centers are shifted left (smaller N).
                      This is independent of and additive with drift_rate in log space.
        n_points: Number of points per IsoFLOP curve
        log_range: Sampling range in log10 space around optimal N (and D)
        alpha, beta: Scaling exponents
        A, B: Scaling coefficients
        E: Irreducible loss

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
            alpha=alpha,
            beta=beta,
            A=A,
            B=B,
            E=E,
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
