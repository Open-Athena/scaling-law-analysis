"""Validation of analytical scaling exponent/intercept error derivations.

This module validates the derived expressions for systematic errors in
Chinchilla Approach 2 when fitting N* scaling exponents and intercepts.

The derivation provides a closed-form expression for the vertex shift δw
as a function of α, β, and grid width W.
"""

import numpy as np
import matplotlib.pyplot as plt

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import LossSurface
from scaling_law_analysis.experiments.common import (
    COMPUTE_BUDGETS,
    LOG_RANGES,
    N_POINTS,
    SYMMETRIC_LOSS_SURFACE,
    DEFAULT_LOSS_SURFACE,
    ASYMMETRIC_CONFIG,
    prepare_output_dir,
    fit_simulated_approach2,
)


def predicted_vertex_shift(
    alpha: float,
    beta: float,
    W: float,
    n_points: int,
) -> float:
    """Compute predicted vertex shift from closed-form expression.

    For the loss function L(w) = E + P·10^(-αw) + R·10^(βw) with αP = βR,
    sampled at n equally-spaced points in [-W, W], the parabola fit has:

        a₁ = Σᵢ wᵢ·L(wᵢ) / Σᵢ wᵢ²

    where the numerator simplifies (using αP = βR) to:

        Σᵢ wᵢ·L(wᵢ) = 2R · Σⱼ₊ wⱼ·[sinh(β·ln10·wⱼ) - (β/α)·sinh(α·ln10·wⱼ)]

    The vertex shift is δw = -a₁/(2a₂), which depends only on α, β, W, n.

    Args:
        alpha: Parameter scaling exponent
        beta: Data scaling exponent
        W: Grid half-width in log10 space
        n_points: Number of sample points

    Returns:
        Vertex shift δw in log10 space
    """
    # Grid points symmetric around 0
    w = np.linspace(-W, W, n_points)
    ln10 = np.log(10)

    # Compute the sums that appear in least-squares fitting
    # Using the substitution that αP = βR, so P = (β/α)R
    # The R factor cancels out in the ratio a₁/a₂

    # Loss at each point (with P = β/α, R = 1 for normalization)
    # L(w) = E + (β/α)·10^(-αw) + 10^(βw)
    # The E term doesn't affect the vertex position
    P_norm = beta / alpha  # normalized P (with R=1)
    L_contribution = P_norm * (10 ** (-alpha * w)) + (10 ** (beta * w))

    # Least-squares parabola fit: L = a₀ + a₁w + a₂w²
    # For symmetric points:
    #   a₁ = Σ(wᵢ·Lᵢ) / Σ(wᵢ²)
    #   a₂ = [n·Σ(wᵢ²·Lᵢ) - Σ(wᵢ²)·Σ(Lᵢ)] / [n·Σ(wᵢ⁴) - (Σwᵢ²)²]

    sum_w = np.sum(w)  # = 0 for symmetric grid
    sum_w2 = np.sum(w**2)
    sum_w3 = np.sum(w**3)  # = 0 for symmetric grid
    sum_w4 = np.sum(w**4)

    sum_L = np.sum(L_contribution)
    sum_wL = np.sum(w * L_contribution)
    sum_w2L = np.sum(w**2 * L_contribution)

    # Linear coefficient
    a1 = sum_wL / sum_w2

    # Quadratic coefficient (for symmetric grid, the formula simplifies)
    n = n_points
    a2 = (n * sum_w2L - sum_w2 * sum_L) / (n * sum_w4 - sum_w2**2)

    # Vertex shift
    delta_w = -a1 / (2 * a2)

    return delta_w


def predicted_intercept_error(
    alpha: float,
    beta: float,
    W: float,
    n_points: int,
) -> float:
    """Compute predicted relative intercept error from the derived formula.

    The vertex shift δw means N̂* = N* · 10^δw, so:
        (â₀ - a₀) / a₀ = 10^δw - 1

    Args:
        alpha: Parameter scaling exponent
        beta: Data scaling exponent
        W: Grid half-width in log10 space
        n_points: Number of sample points

    Returns:
        Predicted relative intercept error
    """
    delta_w = predicted_vertex_shift(alpha, beta, W, n_points)
    return 10**delta_w - 1


def compute_numerical_errors(
    surface: LossSurface,
    log_ranges: np.ndarray,
    compute_budgets: np.ndarray,
    n_points: int,
) -> dict:
    """Compute numerical errors from Approach 2 fitting.

    Args:
        surface: Loss surface configuration
        log_ranges: Array of sampling ranges to test
        compute_budgets: Compute budgets for IsoFLOP sampling
        n_points: Number of points per IsoFLOP contour

    Returns:
        Dictionary with arrays of exponent and intercept errors
    """
    exponent_errors = []
    intercept_errors = []

    for log_range in log_ranges:
        result = fit_simulated_approach2(
            compute_budgets=compute_budgets,
            surface=surface,
            drift_rate=0.0,  # No drift (symmetric sampling)
            center_scale=1.0,  # No scaling (centered on true optimum)
            n_points=n_points,
            log_range=log_range,
        )

        # Exponent error (should be ~0)
        exponent_errors.append(result.a - surface.a)

        # Intercept error (relative)
        intercept_errors.append(
            (10**result.a_intercept - 10**surface.a_intercept) / 10**surface.a_intercept
        )

    return {
        "log_ranges": log_ranges,
        "exponent_errors": np.array(exponent_errors),
        "intercept_errors": np.array(intercept_errors),
        "true_a": surface.a,
        "true_a_intercept": surface.a_intercept,
    }


def run_validation() -> dict:
    """Run validation comparing predicted vs numerical errors.

    Returns:
        Dictionary with validation results for each surface configuration
    """
    surfaces = [
        ("symmetric", SYMMETRIC_LOSS_SURFACE),
        ("chinchilla", DEFAULT_LOSS_SURFACE),
        ("asymmetric", ASYMMETRIC_CONFIG.loss),
    ]

    results = {}

    for name, surface in surfaces:
        print(f"\n{'=' * 60}")
        print(f"Surface: {name}")
        print(f"  α = {surface.alpha:.4f}, β = {surface.beta:.4f}")
        print(f"  α - β = {surface.alpha - surface.beta:.4f}")
        print("=" * 60)

        # Compute numerical errors
        numerical = compute_numerical_errors(
            surface=surface,
            log_ranges=LOG_RANGES,
            compute_budgets=COMPUTE_BUDGETS,
            n_points=N_POINTS,
        )

        # Compute predicted errors from derived formula (function of α, β, W only)
        predicted = np.array(
            [
                predicted_intercept_error(surface.alpha, surface.beta, W, N_POINTS)
                for W in LOG_RANGES
            ]
        )

        # Compare
        max_exp_error = np.abs(numerical["exponent_errors"]).max()
        print(f"\nExponent error (should be ~0):")
        print(f"  Max |â - a|: {max_exp_error:.2e}")

        print(f"\nIntercept error comparison:")
        print(f"  {'W (log10)':>10} {'predicted':>12} {'numerical':>12} {'diff':>12}")
        print(f"  {'-' * 50}")

        for i in [0, len(LOG_RANGES) // 2, -1]:  # Sample a few
            W = LOG_RANGES[i]
            pred = predicted[i]
            num = numerical["intercept_errors"][i]
            diff = num - pred
            print(f"  {W:>10.2f} {pred:>12.6f} {num:>12.6f} {diff:>12.2e}")

        # Max deviation
        max_deviation = np.abs(numerical["intercept_errors"] - predicted).max()
        print(f"\n  Max absolute deviation: {max_deviation:.2e}")

        results[name] = {
            "surface": surface,
            "numerical": numerical,
            "predicted": predicted,
            "max_deviation": max_deviation,
        }

    return results


def create_validation_figure(results: dict) -> plt.Figure:
    """Create figure comparing predicted vs numerical errors.

    Args:
        results: Dictionary from run_validation()

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (name, data) in zip(axes, results.items()):
        surface = data["surface"]
        numerical = data["numerical"]
        predicted = data["predicted"]
        log_ranges = numerical["log_ranges"]

        # Plot numerical vs predicted intercept errors
        ax.plot(
            log_ranges,
            numerical["intercept_errors"] * 100,
            "o-",
            label="Numerical (Approach 2)",
            markersize=4,
        )
        ax.plot(
            log_ranges,
            predicted * 100,
            "--",
            label="Predicted (derived formula)",
            linewidth=2,
        )

        ax.set_xlabel("W (log₁₀ grid half-width)")
        ax.set_ylabel("Intercept error (%)")
        ax.set_title(
            f"{name}\n"
            f"α={surface.alpha:.2f}, β={surface.beta:.2f}, "
            f"max dev={data['max_deviation']:.2e}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Validation: Derived Formula vs Numerical Intercept Errors\n"
        "δw = f(α, β, W, n) — independent of compute budget C",
        fontsize=11,
    )
    fig.tight_layout()

    return fig


def main():
    """Run validation and save results."""
    print("Running validation of scaling exponent/intercept error derivations...")

    # Run validation
    results = run_validation()

    # Check success criteria
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    tolerance = 1e-10  # Strict tolerance for exact match

    for name, data in results.items():
        numerical = data["numerical"]
        max_exp_error = np.abs(numerical["exponent_errors"]).max()
        max_int_deviation = data["max_deviation"]

        # Criteria 1: Exponent error should be ~0
        exp_ok = max_exp_error < tolerance

        # Criteria 2: Intercept prediction should be exact
        int_ok = max_int_deviation < tolerance

        passed = exp_ok and int_ok
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"\n{name}: {status}")
        print(
            f"  Exponent error < {tolerance:.0e}: {'✓' if exp_ok else '✗'} ({max_exp_error:.2e})"
        )
        print(
            f"  Intercept deviation < {tolerance:.0e}: {'✓' if int_ok else '✗'} ({max_int_deviation:.2e})"
        )

    # Create and save figure
    fig = create_validation_figure(results)
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp6")
    output_path = output_dir / "validation_intercept_errors.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
