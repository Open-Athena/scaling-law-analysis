"""Tests for core chinchilla module functions."""

import numpy as np
import pytest

from scaling_law_analysis.chinchilla import (
    LossSurface,
    fit_parabola,
    fit_approach2,
    fit_surface,
    isoflop_sample,
    compute_center_offset,
)


class TestFitParabola:
    """Tests for fit_parabola function."""

    def test_fit_parabola_recovers_known_minimum(self):
        """fit_parabola should recover the true minimum from synthetic data."""
        # Fit a known parabola: L = a*(log_x - log_x_opt)² + L_min
        # Expanding: L = a*log_x² - 2*a*log_x_opt*log_x + a*log_x_opt² + L_min
        # Standard form coeffs [a, b, c]: a, -2*a*log_x_opt, a*log_x_opt² + L_min
        log_x = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
        a_true = 0.5
        log_x_opt_true = 3.0
        L_min_true = 2.0
        L = a_true * (log_x - log_x_opt_true) ** 2 + L_min_true

        fit = fit_parabola(log_x, L)

        assert fit.log_x_opt == pytest.approx(log_x_opt_true, rel=1e-10)
        assert fit.x_opt == pytest.approx(10**log_x_opt_true, rel=1e-10)
        assert fit.L_min == pytest.approx(L_min_true, rel=1e-10)

        # Validate all polynomial coefficients [a, b, c] for ax² + bx + c
        expected_coeffs = [
            a_true,  # a
            -2 * a_true * log_x_opt_true,  # b
            a_true * log_x_opt_true**2 + L_min_true,  # c
        ]
        assert len(fit.coeffs) == 3
        for i, (actual, expected) in enumerate(zip(fit.coeffs, expected_coeffs)):
            assert actual == pytest.approx(expected, rel=1e-10), f"coeffs[{i}] mismatch"

    def test_fit_parabola_rejects_non_positive_curvature(self):
        """fit_parabola should raise ValueError for flat or downward-facing data."""
        log_x = np.array([2.0, 2.5, 3.0, 3.5, 4.0])

        # Flat curve (zero curvature)
        L_flat = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        with pytest.raises(ValueError, match="non-positive curvature"):
            fit_parabola(log_x, L_flat)

        # Downward-facing parabola (negative curvature) - has maximum, not minimum
        L_inverted = -0.5 * (log_x - 3.0) ** 2 + 5.0
        with pytest.raises(ValueError, match="non-positive curvature"):
            fit_parabola(log_x, L_inverted)


class TestComputeCenterOffset:
    """Tests for compute_center_offset function."""

    def test_compute_center_offset_drift_and_scale_effects(self):
        """compute_center_offset should handle drift, scale, and baseline correctly."""
        compute_budgets = np.array([1e17, 1e18, 1e19, 1e20, 1e21])

        # Baseline: no drift, no scale → zero offset everywhere
        for C in compute_budgets:
            offset = compute_center_offset(
                C, compute_budgets, drift_rate=0.0, center_scale=1.0
            )
            assert offset == 0.0

        # Drift only: 0 at min compute, -drift_rate at max compute
        drift_rate = 0.3
        offset_min = compute_center_offset(
            compute_budgets[0], compute_budgets, drift_rate=drift_rate, center_scale=1.0
        )
        offset_max = compute_center_offset(
            compute_budgets[-1],
            compute_budgets,
            drift_rate=drift_rate,
            center_scale=1.0,
        )
        assert offset_min == pytest.approx(0.0, abs=1e-12)
        assert offset_max == pytest.approx(-drift_rate, rel=1e-10)

        # Scale only: constant offset = -log10(scale) everywhere
        center_scale = 2.0
        expected_scale_offset = -np.log10(center_scale)
        for C in compute_budgets:
            offset = compute_center_offset(
                C, compute_budgets, drift_rate=0.0, center_scale=center_scale
            )
            assert offset == pytest.approx(expected_scale_offset, rel=1e-10)

        # Combined: drift + scale are additive
        offset_combined = compute_center_offset(
            compute_budgets[-1],
            compute_budgets,
            drift_rate=drift_rate,
            center_scale=center_scale,
        )
        expected_combined = expected_scale_offset - drift_rate  # both subtractive
        assert offset_combined == pytest.approx(expected_combined, rel=1e-10)


class TestFitApproach2:
    """Tests for fit_approach2 function."""

    def test_fit_approach2_perfect_recovery_on_symmetric_surface(self):
        """fit_approach2 should perfectly recover parameters on a symmetric loss surface.

        With α = β and A = B, the loss surface is symmetric, and centered sampling
        (no drift, no scale) should yield perfect parabola fits and exact exponents.
        """
        # Symmetric surface: α = β, A = B → a = b = 0.5, G = 1
        surface = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)
        compute_budgets = np.array([1e17, 1e18, 1e19, 1e20, 1e21])

        result = fit_approach2(
            compute_budgets=compute_budgets,
            surface=surface,
            drift_rate=0.0,
            center_scale=1.0,
            n_points=15,
            log_range=1.0,
        )

        # Exponents should be exactly 0.5 for symmetric surface
        assert result.a == pytest.approx(0.5, rel=1e-10)
        assert result.b == pytest.approx(0.5, rel=1e-10)

        # N* and D* from parabola fits should match analytical values
        for i, C in enumerate(compute_budgets):
            true_N = surface.N_opt(C)
            true_D = surface.D_opt(C)
            assert result.N_opts[i] == pytest.approx(true_N, rel=1e-10)
            assert result.D_opts[i] == pytest.approx(true_D, rel=1e-10)

        # Intercepts should match analytical values from LossSurface
        assert result.a_intercept == pytest.approx(surface.a_intercept, rel=1e-10)
        assert result.b_intercept == pytest.approx(surface.b_intercept, rel=1e-10)


class TestFitSurface:
    """Tests for fit_surface function."""

    def test_fit_surface_recovers_symmetric_surface_parameters(self):
        """fit_surface should perfectly recover all 5 parameters on a symmetric surface."""
        # Create symmetric surface
        surface = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)
        compute_budgets = np.array([1e17, 1e18, 1e19, 1e20, 1e21])

        # Generate sample data from the surface using IsoFLOP sampling
        all_N, all_D, all_L = [], [], []
        for C in compute_budgets:
            N, D, L = isoflop_sample(
                C=C,
                n_points=15,
                log_range=1.0,
                center_offset=0.0,
                surface=surface,
            )
            all_N.extend(N)
            all_D.extend(D)
            all_L.extend(L)

        N_arr = np.array(all_N)
        D_arr = np.array(all_D)
        L_arr = np.array(all_L)

        # Fit the surface
        result = fit_surface(N_arr, D_arr, L_arr)

        # All 5 parameters should be recovered exactly
        assert result.alpha == pytest.approx(surface.alpha, rel=1e-6)
        assert result.beta == pytest.approx(surface.beta, rel=1e-6)
        assert result.A == pytest.approx(surface.A, rel=1e-6)
        assert result.B == pytest.approx(surface.B, rel=1e-6)
        assert result.E == pytest.approx(surface.E, rel=1e-6)

        # RSS should be essentially zero for noise-free data
        assert result.residual_sum_squares < 1e-18

        # Verify to_loss_surface() creates equivalent surface
        fitted_surface = result.to_loss_surface()
        assert fitted_surface.alpha == pytest.approx(surface.alpha, rel=1e-6)
        assert fitted_surface.beta == pytest.approx(surface.beta, rel=1e-6)
