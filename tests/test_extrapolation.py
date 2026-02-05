"""Tests for extrapolation accuracy with perfect D* inputs.

These tests validate that when using true analytical D* values (from LossSurface)
instead of parabola-derived estimates, the log-linear regression extrapolation
produces exact results. This isolates the extrapolation error to the parabola
fitting step in Chinchilla Approach 2.
"""

import numpy as np
import pytest

from scaling_law_analysis.chinchilla import (
    LossSurface,
    fit_power_law,
    DEFAULT_LOSS_SURFACE,
)
from scaling_law_analysis.experiments.common import (
    SimulationConfig,
    LOSS_SURFACES,
    COMPUTE_BUDGETS,
    EXTRAPOLATION_BUDGETS,
    compute_extrapolation_errors,
)


def true_dopt_fitter(
    sim_config: SimulationConfig,
    compute_budgets: np.ndarray,
    log_range: float,  # unused - true D* doesn't depend on sampling
    n_points: int,  # unused - true D* doesn't depend on sampling
):
    """Fitter that uses true analytical D* values instead of parabola fits.

    This fitter:
    1. Gets true D* values from LossSurface.D_opt() at each compute budget
    2. Fits a log-linear regression using fit_power_law()
    3. Returns a callable that extrapolates D* to any compute budget

    When used with compute_extrapolation_errors(), this should produce
    zero extrapolation error (within floating-point precision).
    """
    loss = sim_config.loss

    # Get true D* values at fitting compute budgets
    true_D_opts = np.array([loss.D_opt(C) for C in compute_budgets])

    # Fit power law: D* = 10^intercept * C^exponent
    D_fit = fit_power_law(compute_budgets, true_D_opts)

    # Return callable that extrapolates D* using the fitted power law
    def D_opt_extrapolate(C: float) -> float:
        log10_D = D_fit.exponent * np.log10(C) + D_fit.intercept
        return 10**log10_D

    return D_opt_extrapolate


class TestPowerLawFitExactness:
    """Test that fit_power_law recovers exact parameters for perfect power law data."""

    def test_fit_power_law_recovers_exact_exponent(self):
        """fit_power_law should recover exact exponent for perfect power law data."""
        # Create perfect power law data: y = 10^2 * x^0.5
        x = np.array([1e10, 1e12, 1e14, 1e16, 1e18])
        true_exponent = 0.5
        true_intercept = 2.0
        y = (10**true_intercept) * (x**true_exponent)

        fit = fit_power_law(x, y)

        assert fit.exponent == pytest.approx(true_exponent, rel=1e-12)
        assert fit.intercept == pytest.approx(true_intercept, rel=1e-12)

    def test_fit_power_law_with_loss_surface_d_opt(self):
        """fit_power_law on true D* values should recover exact b exponent."""
        surface = DEFAULT_LOSS_SURFACE
        compute_budgets = COMPUTE_BUDGETS

        # Get true D* values
        true_D_opts = np.array([surface.D_opt(C) for C in compute_budgets])

        # Fit power law
        fit = fit_power_law(compute_budgets, true_D_opts)

        # The exponent should match the analytical value b = α/(α+β)
        expected_b = surface.b
        assert fit.exponent == pytest.approx(expected_b, rel=1e-10)


class TestTrueDoptExtrapolation:
    """Test that extrapolation with true D* values produces zero error.

    This is the core validation: if we use analytical D* values (bypassing
    parabola fitting) and fit a log-linear regression, extrapolating to
    higher compute budgets should give exactly the same results as computing
    D* analytically at those budgets.
    """

    @pytest.mark.parametrize("surface_name,surface", LOSS_SURFACES)
    def test_extrapolation_error_is_zero_for_true_dopt(self, surface_name, surface):
        """Extrapolation error should be ~0 when using true D* values."""
        sim_config = SimulationConfig(
            name=f"true_dopt_{surface_name}",
            loss=surface,
            drift_rate=0.0,
            center_scale=1.0,
        )

        results = compute_extrapolation_errors(
            sim_config=sim_config,
            compute_budgets=COMPUTE_BUDGETS,
            extrapolation_budgets=EXTRAPOLATION_BUDGETS,
            log_range=1.0,  # unused by true_dopt_fitter
            n_points=15,  # unused by true_dopt_fitter
            fitter=true_dopt_fitter,
        )

        # Relative errors should be essentially zero (machine precision)
        # Using 1e-10 as threshold to account for floating-point accumulation
        max_rel_error = np.abs(results["D_rel_errors"]).max()
        assert max_rel_error < 1e-10, (
            f"Expected zero extrapolation error for {surface_name}, "
            f"but got max relative error: {max_rel_error:.2e}"
        )


class TestFitPowerLawIntercept:
    """Test that fit_power_law recovers the correct intercept for D* scaling."""

    def test_intercept_matches_analytical_formula(self):
        """The fitted intercept should match LossSurface.b_intercept."""
        surface = DEFAULT_LOSS_SURFACE
        compute_budgets = COMPUTE_BUDGETS

        # Get true D* values and fit
        true_D_opts = np.array([surface.D_opt(C) for C in compute_budgets])
        fit = fit_power_law(compute_budgets, true_D_opts)

        assert fit.intercept == pytest.approx(surface.b_intercept, rel=1e-10)
