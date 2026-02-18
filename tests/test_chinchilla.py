"""Tests for core chinchilla module functions."""

import numpy as np
import pytest

from scaling_law_analysis.chinchilla import (
    DEFAULT_LOSS_SURFACE,
    FINE_ALPHA_GRID,
    FINE_BETA_GRID,
    FitError,
    LossSurface,
    _cartesian_product,
    _surface_rss,
    _surface_rss_grad,
    fit_grid_search,
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
        """fit_parabola should raise FitError for flat or downward-facing data."""
        log_x = np.array([2.0, 2.5, 3.0, 3.5, 4.0])

        # Flat curve (zero curvature)
        L_flat = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        with pytest.raises(FitError, match="non-positive curvature"):
            fit_parabola(log_x, L_flat)

        # Downward-facing parabola (negative curvature) - has maximum, not minimum
        L_inverted = -0.5 * (log_x - 3.0) ** 2 + 5.0
        with pytest.raises(FitError, match="non-positive curvature"):
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


def _generate_isoflop_data(
    surface: LossSurface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate IsoFLOP sample data from a loss surface for testing."""
    compute_budgets = np.array([1e17, 1e18, 1e19, 1e20, 1e21])
    all_N, all_D, all_L = [], [], []
    for C in compute_budgets:
        N, D, L = isoflop_sample(
            C=C, n_points=15, log_range=1.0, center_offset=0.0, surface=surface
        )
        all_N.extend(N)
        all_D.extend(D)
        all_L.extend(L)
    return np.array(all_N), np.array(all_D), np.array(all_L)


class TestFitSurface:
    """Tests for fit_surface across optimization methods."""

    SYMMETRIC = LossSurface(alpha=0.31, beta=0.31, A=400, B=400, E=1.69)
    CHINCHILLA = DEFAULT_LOSS_SURFACE  # α=0.34, β=0.28

    @pytest.mark.parametrize(
        "surface_name,surface",
        [("symmetric", SYMMETRIC), ("chinchilla", CHINCHILLA)],
    )
    @pytest.mark.parametrize("method", ["nelder-mead", "l-bfgs-b", "grid"])
    def test_method_recovers_parameters(self, surface_name, surface, method):
        """Each method should recover surface parameters within its precision."""
        N, D, L = _generate_isoflop_data(surface)

        if method == "grid":
            result = fit_surface(
                N,
                D,
                L,
                alpha_grid=FINE_ALPHA_GRID,
                beta_grid=FINE_BETA_GRID,
                method=method,
            )
        else:
            result = fit_surface(N, D, L, method=method)

        assert result.method == method

        if method == "grid":
            # Grid step ~0.0035 over [0.05, 0.95] at 256 points.
            # Exponent errors ≤0.4%; coefficient errors up to ~2% due to
            # exponential amplification of exponent discretization.
            exp_tol = 5e-3
            coeff_tol = 2.5e-2
        else:
            exp_tol = 1e-6
            coeff_tol = 1e-6

        assert result.alpha == pytest.approx(surface.alpha, rel=exp_tol)
        assert result.beta == pytest.approx(surface.beta, rel=exp_tol)
        assert result.A == pytest.approx(surface.A, rel=coeff_tol)
        assert result.B == pytest.approx(surface.B, rel=coeff_tol)
        assert result.E == pytest.approx(surface.E, rel=coeff_tol)

    def test_nelder_mead_rss_near_zero(self):
        """Nelder-Mead should achieve near-zero RSS on noise-free data."""
        N, D, L = _generate_isoflop_data(self.SYMMETRIC)
        result = fit_surface(N, D, L, method="nelder-mead")
        assert result.residual_sum_squares < 1e-18

    def test_to_loss_surface_roundtrip(self):
        """to_loss_surface() should produce an equivalent surface."""
        N, D, L = _generate_isoflop_data(self.SYMMETRIC)
        result = fit_surface(N, D, L, method="nelder-mead")
        fitted = result.to_loss_surface()
        assert fitted.alpha == pytest.approx(self.SYMMETRIC.alpha, rel=1e-6)
        assert fitted.beta == pytest.approx(self.SYMMETRIC.beta, rel=1e-6)


class TestSurfaceRssGrad:
    """Test analytical gradient of surface RSS against finite differences."""

    def test_gradient_matches_finite_differences(self):
        """Analytical gradient should match central finite differences to high precision."""
        rng = np.random.default_rng(42)
        # Moderate scales keep all gradient components well-conditioned
        log_N = rng.uniform(1, 5, size=50)
        log_D = rng.uniform(1, 5, size=50)
        L = rng.uniform(2.0, 4.0, size=50)

        x = np.array([1.5, 50.0, 30.0, 0.35, 0.28])

        analytic = _surface_rss_grad(x, log_N, log_D, L)

        # Optimal step for central differences: h ~ eps^(1/3).
        # Balances O(h²) truncation error against O(eps/h) roundoff error.
        # See Press et al., Numerical Recipes (3rd ed.), §5.7 "Numerical Derivatives".
        h = float(np.finfo(float).eps) ** (1 / 3)
        fd = np.zeros_like(x)
        for i in range(len(x)):
            x_fwd = x.copy()
            x_bwd = x.copy()
            x_fwd[i] += h
            x_bwd[i] -= h
            fd[i] = (
                _surface_rss(x_fwd, log_N, log_D, L)
                - _surface_rss(x_bwd, log_N, log_D, L)
            ) / (2 * h)

        np.testing.assert_allclose(analytic, fd, rtol=1e-9)


class TestCartesianProduct:
    """Tests for _cartesian_product helper."""

    def test_matches_nested_loops(self):
        """Columns match the order of nested for-loops (first=slowest)."""
        a = np.array([1, 2])
        b = np.array([10, 20, 30])
        c = np.array([100, 200])

        result = _cartesian_product(a, b, c)

        expected = np.array(
            [
                [1, 10, 100],
                [1, 10, 200],
                [1, 20, 100],
                [1, 20, 200],
                [1, 30, 100],
                [1, 30, 200],
                [2, 10, 100],
                [2, 10, 200],
                [2, 20, 100],
                [2, 20, 200],
                [2, 30, 100],
                [2, 30, 200],
            ]
        ).T  # (3, 12)

        assert result.shape == (3, 2 * 3 * 2)
        np.testing.assert_array_equal(result, expected)


class TestFitGridSearch:
    """Tests for fit_grid_search vectorized grid initialization."""

    @pytest.mark.parametrize(
        "target_idx",
        [(0, 0, 0, 0, 0), (2, 1, 2, 1, 2), (1, 0, 1, 1, 0)],
        ids=["first", "last", "interior"],
    )
    def test_finds_planted_zero_rss_point(self, target_idx):
        """Plant a grid point as the exact data generator; function must find it."""
        E_grid = np.array([1.0, 2.0, 3.0])
        A_grid = np.array([10.0, 100.0])
        B_grid = np.array([10.0, 50.0, 100.0])
        alpha_grid = np.array([0.2, 0.5])
        beta_grid = np.array([0.3, 0.6, 0.9])

        grids = [E_grid, A_grid, B_grid, alpha_grid, beta_grid]
        expected = np.array([g[i] for g, i in zip(grids, target_idx)])
        E, A, B, alpha, beta = expected

        N = np.array([1e8, 1e9, 1e10])
        D = np.array([1e9, 1e10, 1e11])
        log_N, log_D = np.log(N), np.log(D)
        L = E + A * N ** (-alpha) + B * D ** (-beta)

        best_x0 = fit_grid_search(
            E_grid=E_grid,
            A_grid=A_grid,
            B_grid=B_grid,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            log_N=log_N,
            log_D=log_D,
            L=L,
        )

        np.testing.assert_array_equal(best_x0, expected)
