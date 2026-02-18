"""Experiment 8: Optimizer conditioning — Approach 3 vs VPNLS.

Demonstrates why VPNLS achieves higher precision than Approach 3 on
noise-free Chinchilla-style loss surfaces. The root cause is the extreme
ill-conditioning of the 5-parameter optimization landscape that
Approach 3 must navigate, which VPNLS avoids through variable projection.

Output: a text log file (conditioning_analysis.txt) with step-by-step
diagnostics, written to results/experiments/exp8/.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import approx_fprime, minimize

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_BETA_GRID,
    LBFGSB_OPTIONS,
    _A3_A_GRID,
    _A3_ALPHA_GRID,
    _A3_B_GRID,
    _A3_BETA_GRID,
    _A3_E_GRID,
    _compute_rss_and_params,
    _surface_rss,
    _surface_rss_grad,
    fit_approach3,
    fit_surface,
)
from scaling_law_analysis.experiments.common import (
    ASYMMETRIC_CONFIG,
    COMPUTE_BUDGETS,
    prepare_output_dir,
    sample_isoflop_data,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _vp_rss(
    alpha: float,
    beta: float,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Variable-projection RSS: for fixed (α, β), solve E/A/B via NNLS."""
    return _compute_rss_and_params(alpha, beta, log_N, log_D, L)


def _hessian_5d(
    x: np.ndarray,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """Numerical Hessian of the 5D RSS at x."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):

        def grad_i(xv: np.ndarray, idx: int = i) -> float:
            return float(_surface_rss_grad(xv, log_N, log_D, L)[idx])

        H[i] = np.asarray(approx_fprime(x, grad_i, 1e-8))
    return (H + H.T) / 2


def _hessian_2d_vp(
    alpha: float,
    beta: float,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """Numerical Hessian of the 2D VP objective at (α, β)."""

    def obj(x: np.ndarray) -> float:
        rss, _ = _vp_rss(x[0], x[1], log_N, log_D, L)
        return rss

    x0 = np.array([alpha, beta])
    H = np.zeros((2, 2))
    for i in range(2):

        def grad_i(xv: np.ndarray, idx: int = i) -> float:
            return float(np.asarray(approx_fprime(xv, obj, 1e-8))[idx])

        H[i] = np.asarray(approx_fprime(x0, grad_i, 1e-8))
    return (H + H.T) / 2


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> str:
    """Run the conditioning analysis and return the report as a string."""
    loss = ASYMMETRIC_CONFIG.loss

    lines: list[str] = []

    def p(text: str = "") -> None:
        lines.append(text)

    def rule(char: str = "=", width: int = 72) -> None:
        p(char * width)

    # ── Header ────────────────────────────────────────────────────────────

    rule()
    p("Experiment 8: Optimizer Conditioning — Approach 3 vs VPNLS")
    rule()
    p()
    p("This experiment explains why VPNLS achieves higher precision than")
    p("Approach 3 (direct 5-parameter L-BFGS-B) when fitting noise-free")
    p("Chinchilla-style loss surfaces. The difference is not due to grid search")
    p("quality or optimizer settings — it is a fundamental consequence of")
    p("the ill-conditioning of the 5D optimization landscape.")
    p()

    # ── Setup ─────────────────────────────────────────────────────────────

    rule("-")
    p("Setup")
    rule("-")
    p()
    p(
        f"Loss surface: E={loss.E}, A={loss.A}, B={loss.B}, "
        f"α={loss.alpha}, β={loss.beta}"
    )
    p(
        f"Compute budgets: {len(COMPUTE_BUDGETS)} budgets, "
        f"{COMPUTE_BUDGETS[0]:.0e} to {COMPUTE_BUDGETS[-1]:.0e} FLOPs"
    )
    p(f"Points per curve: 15")
    p(f"Noise: σ = 0 (perfect data from the true loss surface)")
    p()

    N, D, L = sample_isoflop_data(ASYMMETRIC_CONFIG, COMPUTE_BUDGETS, np.log10(8), 15)
    log_N, log_D = np.log(N), np.log(D)
    true_x = np.array([loss.E, loss.A, loss.B, loss.alpha, loss.beta])
    true_rss = _surface_rss(true_x, log_N, log_D, L)

    p(f"RSS at true parameters: {true_rss:.2e}")
    p(f"  (This should be ~0; any residual is floating-point noise.)")
    p()

    # ── Step 1: Fit both methods ──────────────────────────────────────────

    rule("-")
    p("Step 1: Fit both methods on noise-free data")
    rule("-")
    p()
    p("Both methods are given identical, perfect data generated from the")
    p("true loss surface with zero noise. We use the library's standard")
    p("fitting functions with default settings.")
    p()

    est_a3 = fit_approach3(N, D, L)
    est_vp = fit_surface(N, D, L, method="nelder-mead")

    a3_errs = {
        "E": (est_a3.E - loss.E) / loss.E,
        "A": (est_a3.A - loss.A) / loss.A,
        "B": (est_a3.B - loss.B) / loss.B,
        "α": (est_a3.alpha - loss.alpha) / loss.alpha,
        "β": (est_a3.beta - loss.beta) / loss.beta,
    }
    vp_errs = {
        "E": (est_vp.E - loss.E) / loss.E,
        "A": (est_vp.A - loss.A) / loss.A,
        "B": (est_vp.B - loss.B) / loss.B,
        "α": (est_vp.alpha - loss.alpha) / loss.alpha,
        "β": (est_vp.beta - loss.beta) / loss.beta,
    }

    p("Approach 3 (L-BFGS-B, 5 parameters):")
    for name, err in a3_errs.items():
        p(f"  {name:>1}:  relative error = {err:+.2e}")
    p(
        f"  status: {est_a3.status.value}"
        + (f" — {est_a3.status_message}" if est_a3.status_message else "")
    )
    p()
    p("VPNLS (Nelder-Mead, 2 parameters + NNLS):")
    for name, err in vp_errs.items():
        p(f"  {name:>1}:  relative error = {err:+.2e}")
    p(
        f"  status: {est_vp.status.value}"
        + (f" — {est_vp.status_message}" if est_vp.status_message else "")
    )
    p()
    alpha_ratio = (
        abs(a3_errs["α"] / vp_errs["α"]) if vp_errs["α"] != 0 else float("inf")
    )
    beta_ratio = abs(a3_errs["β"] / vp_errs["β"]) if vp_errs["β"] != 0 else float("inf")
    p(
        f"VPNLS is ~{alpha_ratio:.0f}× more precise on α "
        f"and ~{beta_ratio:.0f}× more precise on β."
    )
    p()
    p("Why? Both methods use tight optimizer tolerances (ftol=1e-15,")
    p("gtol=1e-15). The difference is not in the settings — it is in")
    p("the conditioning of the problem each optimizer sees.")
    p()

    # ── Step 2: 5D Hessian analysis ──────────────────────────────────────

    rule("-")
    p("Step 2: Conditioning of the 5D landscape (Approach 3)")
    rule("-")
    p()
    p("The Hessian of the RSS objective captures how curved the loss")
    p("surface is in each direction. Its eigenvalues tell us which")
    p("parameter directions are 'steep' (high eigenvalue = sensitive)")
    p("vs 'flat' (low eigenvalue = insensitive).")
    p()
    p("A high condition number (ratio of largest to smallest eigenvalue)")
    p("means the optimizer must simultaneously handle directions where a")
    p("tiny step causes a huge change AND directions where a large step")
    p("barely changes anything. L-BFGS-B's limited-memory Hessian")
    p("approximation (~10 vector pairs) cannot accurately represent")
    p("such extreme curvature spread, so it cannot compute steps that")
    p("make correct progress along all directions simultaneously.")
    p()

    H5 = _hessian_5d(true_x, log_N, log_D, L)
    eigvals_5d, eigvecs_5d = np.linalg.eigh(H5)
    cond_5d = eigvals_5d[-1] / eigvals_5d[0]
    param_names = ["E", "A", "B", "α", "β"]

    p("Hessian eigenvalues at the true parameters:")
    p()
    for k in range(5):
        direction_parts = []
        for j in range(5):
            v = eigvecs_5d[j, k]
            if abs(v) > 0.01:
                direction_parts.append(f"{v:+.4f}·{param_names[j]}")
        direction_str = ", ".join(direction_parts)
        p(f"  λ_{k+1} = {eigvals_5d[k]:.2e}  " f"(direction: {direction_str})")
    p()
    p(f"Condition number: κ = λ_max / λ_min = {cond_5d:.2e}")
    p()
    p("Interpretation:")
    p(
        f"  - The two flattest directions "
        f"(λ₁ ≈ {eigvals_5d[0]:.0e}, λ₂ ≈ {eigvals_5d[1]:.0e}) point almost"
    )
    p(f"    entirely along A and B. This means perturbing the linear")
    p(f"    coefficients A or B barely changes the RSS — the loss surface")
    p(f"    is extremely insensitive to these parameters near the optimum.")
    p(f"  - The steepest direction (λ₅ ≈ {eigvals_5d[-1]:.2e}) is")
    p(f"    dominated by β. The next steepest (λ₄ ≈ {eigvals_5d[-2]:.2e})")
    p(f"    is dominated by α.")
    p(f"  - The condition number κ ≈ {cond_5d:.0e} means L-BFGS-B must")
    p(
        f"    resolve curvature differences spanning {np.log10(cond_5d):.0f} "
        f"orders of magnitude."
    )
    p()
    p("We can verify this directly by examining the gradient.")
    p()

    # -- Gradient at the true parameters --
    grad_true = _surface_rss_grad(true_x, log_N, log_D, L)
    p("Gradient at the TRUE parameters (should be ~0 everywhere):")
    for j, name in enumerate(param_names):
        p(f"  ∂RSS/∂{name:>1} = {grad_true[j]:+.2e}")
    p()
    ab_max = max(abs(grad_true[1]), abs(grad_true[2]))
    ab_exp = int(np.floor(np.log10(ab_max)))
    p(f"The A and B components are near machine epsilon (~10^{ab_exp}),")
    p("confirming these directions are truly flat at the minimum.")
    alpha_grad = abs(grad_true[3])
    beta_grad = abs(grad_true[4])
    p(f"The α component ({alpha_grad:.0e}) and β component ({beta_grad:.0e})")
    p("are larger because the surface is steeper there — even floating-")
    p("point noise in the evaluation produces a measurable gradient.")
    p("Now look at what L-BFGS-B sees at its converged solution")
    p("(which is NOT the true minimum).")
    p()

    # -- Gradient at the Approach 3 solution --
    a3_x = np.array([est_a3.E, est_a3.A, est_a3.B, est_a3.alpha, est_a3.beta])
    grad_a3 = _surface_rss_grad(a3_x, log_N, log_D, L)
    p("Gradient at the Approach 3 converged solution:")
    for j, name in enumerate(param_names):
        p(f"  ∂RSS/∂{name:>1} = {grad_a3[j]:+.2e}")
    p()

    # -- Project onto eigenvectors --
    grad_proj = eigvecs_5d.T @ grad_a3
    p("Same gradient projected onto Hessian eigenvectors:")
    for k in range(5):
        p(
            f"  eigenvector {k+1} (λ={eigvals_5d[k]:.2e}): "
            f"component = {grad_proj[k]:+.2e}"
        )
    p()
    flat_range = (abs(grad_proj[0]), abs(grad_proj[1]))
    steep_range = (
        min(abs(grad_proj[2]), abs(grad_proj[3]), abs(grad_proj[4])),
        max(abs(grad_proj[2]), abs(grad_proj[3]), abs(grad_proj[4])),
    )
    p(
        f"The flat-direction components (eigenvectors 1–2) are "
        f"~{min(flat_range):.0e} to {max(flat_range):.0e},"
    )
    p(
        f"while the steep-direction components (3–5) are "
        f"~{steep_range[0]:.0e} to {steep_range[1]:.0e}."
    )
    p("L-BFGS-B's limited-memory Hessian approximation (~10 vector")
    p(f"pairs) in a κ ≈ {cond_5d:.0e} landscape cannot accurately resolve")
    p("these differences — the step it computes is unreliable.")
    p()

    # -- Perturbation experiment --
    delta = 1e-4
    p(f"Perturbation test: displace by δ={delta} along each eigenvector")
    p(f"from the true parameters, then measure |∇RSS|:")
    p()
    pert_norms = []
    for k in range(5):
        x_pert = true_x + delta * eigvecs_5d[:, k]
        grad_pert = _surface_rss_grad(x_pert, log_N, log_D, L)
        grad_norm = float(np.linalg.norm(grad_pert))
        pert_norms.append(grad_norm)
        p(f"  eigenvector {k+1} (λ={eigvals_5d[k]:.2e}): " f"|∇RSS| = {grad_norm:.2e}")
    p()
    ratio = pert_norms[-1] / pert_norms[0]
    p("The gradient response scales with the eigenvalue: a perturbation")
    p(
        f"along the flat A/B directions (λ ≈ {eigvals_5d[0]:.0e}–{eigvals_5d[1]:.0e}) produces a gradient"
    )
    p(f"~{ratio:.0e}× smaller than the same perturbation along the steep β")
    ab_grad_at_soln = max(abs(grad_a3[1]), abs(grad_a3[2]))
    p(f"direction. At Approach 3's solution, the A/B gradient IS nonzero")
    p(f"(~{ab_grad_at_soln:.0e}), but L-BFGS-B's limited-memory Hessian approximation")
    p(f"(~10 vector pairs) cannot accurately represent κ ≈ {cond_5d:.0e}. It")
    p("cannot convert those small A/B gradients into correctly-sized")
    p("steps, and function-value changes from the flat directions are")
    p("negligible, so convergence criteria trigger with A/B unresolved.")
    p()

    # ── Step 3: 2D Hessian analysis ──────────────────────────────────────

    rule("-")
    p("Step 3: Conditioning of the 2D landscape (VPNLS)")
    rule("-")
    p()
    p("VPNLS avoids the ill-conditioned directions entirely. For each")
    p("candidate (α, β), it solves for the linear coefficients E, A, B")
    p("using Non-Negative Least Squares (NNLS) — a direct linear algebra")
    p("solve, not iterative optimization. This eliminates E, A, B from")
    p("the search space, including the two flattest directions.")
    p()
    p("The remaining 2D surface over (α, β) has its own Hessian:")
    p()

    H2 = _hessian_2d_vp(loss.alpha, loss.beta, log_N, log_D, L)
    eigvals_2d = np.linalg.eigvalsh(H2)
    cond_2d = eigvals_2d[-1] / eigvals_2d[0]

    p(f"  λ₁ = {eigvals_2d[0]:.2e}")
    p(f"  λ₂ = {eigvals_2d[1]:.2e}")
    p(f"  Condition number: κ = {cond_2d:.1f}")
    p()
    p(f"A condition number of {cond_2d:.0f} is excellent — both directions")
    p("have comparable curvature, so Nelder-Mead can optimize (α, β)")
    p("to machine precision without numerical issues.")
    p()

    # ── Step 4: Direct comparison ─────────────────────────────────────────

    rule("-")
    p("Step 4: The conditioning gap explains the precision gap")
    rule("-")
    p()
    p("Summary of condition numbers:")
    p(f"  Approach 3 (5D):  κ = {cond_5d:.2e}")
    p(f"  VPNLS (2D):       κ = {cond_2d:.1f}")
    p(f"  Ratio:            {cond_5d / cond_2d:.0e}x worse for Approach 3")
    p()
    p("Summary of precision achieved (relative error in α and β):")
    p(
        f"  Approach 3:  |α| error ≈ {abs(a3_errs['α']):.0e},  "
        f"|β| error ≈ {abs(a3_errs['β']):.0e}"
    )
    p(
        f"  VPNLS:       |α| error ≈ {abs(float(vp_errs['α'])):.0e},  "
        f"|β| error ≈ {abs(float(vp_errs['β'])):.0e}"
    )
    p()
    p(
        f"The ~{cond_5d / cond_2d:.0e} gap in conditioning explains the "
        f"~{alpha_ratio:.0f}×/~{beta_ratio:.0f}× gap in α/β precision."
    )
    p(f"At κ ≈ {cond_5d:.0e}, L-BFGS-B's limited-memory Hessian")
    p("approximation cannot convert the small A/B gradients into")
    p("correctly-sized steps. Convergence criteria trigger based on the")
    p("dominant steep directions, leaving the flat directions unresolved.")
    p()

    # ── Step 5: Verify it's not the grid search ──────────────────────────

    rule("-")
    p("Step 5: Verify the grid search is not the bottleneck")
    rule("-")
    p()
    p("One might suspect that Approach 3's grid search (8⁵ = 32768 points")
    p("in 5D vs 32² = 1024 in 2D for VPNLS) gives it a worse starting point.")
    p("We test this by initializing L-BFGS-B at the TRUE parameters.")
    p()

    result_true_init = minimize(
        lambda x: _surface_rss(x, log_N, log_D, L),
        x0=true_x,
        jac=lambda x: _surface_rss_grad(x, log_N, log_D, L),
        method="L-BFGS-B",
        bounds=[(1e-6, 1e6)] * 3 + [(0.01, 0.99)] * 2,
        options=LBFGSB_OPTIONS,
    )
    assert result_true_init is not None

    p(f"L-BFGS-B initialized at true parameters:")
    p(f"  Iterations: {result_true_init.nit}")
    p(f"  Final RSS:  {result_true_init.fun:.2e}")
    p(f"  Message:    {result_true_init.message}")
    p()

    if result_true_init.nit == 0:
        p("L-BFGS-B took ZERO iterations — its line search could not find")
        p("a descent direction from the true parameters. This is because at")
        p(f"RSS = {true_rss:.2e}, the gradient along the flat A/B directions is")
        p("pure floating-point noise, and L-BFGS-B correctly identifies that")
        p("no reliable improvement is possible.")
    else:
        E, A, B, alpha, beta = result_true_init.x
        alpha_err = (alpha - loss.alpha) / loss.alpha
        beta_err = (beta - loss.beta) / loss.beta
        p(f"  α error: {alpha_err:+.2e},  β error: {beta_err:+.2e}")
    p()
    p("Now compare grid search initialization quality:")
    p()

    # Re-do the Approach 3 grid search to get the starting point
    best_rss_grid = np.inf
    best_x0 = true_x.copy()
    for E_init in _A3_E_GRID:
        for A_init in _A3_A_GRID:
            for B_init in _A3_B_GRID:
                for alpha_init in _A3_ALPHA_GRID:
                    for beta_init in _A3_BETA_GRID:
                        x0 = np.array([E_init, A_init, B_init, alpha_init, beta_init])
                        r = _surface_rss(x0, log_N, log_D, L)
                        if r < best_rss_grid:
                            best_rss_grid = r
                            best_x0 = x0

    # VPNLS grid search for comparison
    best_rss_vp = np.inf
    for alpha_g in DEFAULT_ALPHA_GRID:
        for beta_g in DEFAULT_BETA_GRID:
            rss_g, _ = _compute_rss_and_params(alpha_g, beta_g, log_N, log_D, L)
            if rss_g < best_rss_vp:
                best_rss_vp = rss_g

    p(
        f"  Approach 3 grid search: 8⁵ = {len(_A3_E_GRID)**5} evaluations, "
        f"best RSS = {best_rss_grid:.2e}"
    )
    p(
        f"  VPNLS grid search:     32² = {len(DEFAULT_ALPHA_GRID)**2} evaluations, "
        f"best RSS = {best_rss_vp:.2e}"
    )
    p()
    p("VPNLS starts much closer to the optimum despite fewer evaluations")
    p("because it only needs to search 2D, and NNLS solves E/A/B exactly")
    p("for each (α, β) candidate. But the grid search quality alone does")
    p("not explain the final precision gap — let's verify with ftol=0:")
    p()

    result_ftol0 = minimize(
        lambda x: _surface_rss(x, log_N, log_D, L),
        x0=best_x0,
        jac=lambda x: _surface_rss_grad(x, log_N, log_D, L),
        method="L-BFGS-B",
        bounds=[(1e-6, 1e6)] * 3 + [(0.01, 0.99)] * 2,
        options={"ftol": 0, "gtol": 1e-15, "maxiter": 10000},
    )
    assert result_ftol0 is not None

    E, A, B, alpha, beta = result_ftol0.x
    ftol0_alpha_err = (alpha - loss.alpha) / loss.alpha
    ftol0_beta_err = (beta - loss.beta) / loss.beta
    grad_norm = float(
        np.linalg.norm(_surface_rss_grad(result_ftol0.x, log_N, log_D, L))
    )

    p(f"L-BFGS-B with ftol=0 (grid search init):")
    p(f"  Iterations: {result_ftol0.nit}")
    p(f"  Final RSS:  {result_ftol0.fun:.2e}")
    p(f"  |α| error:  {abs(ftol0_alpha_err):.2e}")
    p(f"  |β| error:  {abs(ftol0_beta_err):.2e}")
    p(f"  Grad norm:  {grad_norm:.2e}")
    p(f"  Message:    {result_ftol0.message}")
    p()
    p("Even with ftol=0 (no function-value stopping), L-BFGS-B improves")
    p(
        f"only modestly (|α| error {abs(a3_errs['α']):.0e} → "
        f"{abs(ftol0_alpha_err):.0e}). The gradient norm"
    )
    p(f"of {grad_norm:.2e} is small but nonzero — descent signal exists, but")
    p("L-BFGS-B's Hessian approximation cannot convert it into a useful step.")
    p()

    # ── Step 6: Why this matters less with noise ──────────────────────────

    rule("-")
    p("Step 6: Why noise masks the conditioning problem")
    rule("-")
    p()
    p("With realistic noise (σ > 0), the true minimum of the RSS shifts")
    p("away from the true parameters. The relevant precision is now set")
    a3_limit = max(abs(a3_errs["α"]), abs(a3_errs["β"]))
    p("by the noise floor, not machine epsilon. Since noise-induced errors")
    p(f"are typically ~10⁻² to 10⁻¹ (much larger than the ~{a3_limit:.0e} limit")
    p("from conditioning), both methods achieve similar accuracy.")
    p()
    p("The conditioning problem only matters when you need precision")
    p("beyond what the noise floor allows — which is exactly the case")
    p("for noise-free synthetic data.")
    p()

    # ── Conclusion ────────────────────────────────────────────────────────

    rule("=")
    p("Conclusion")
    rule("=")
    p()
    p("VPNLS achieves higher precision than Approach 3 because variable")
    p("projection eliminates the linear parameters (E, A, B) — including")
    p(
        "the two most ill-conditioned directions — from the search space. This reduces the"
    )
    p(f"condition number from ~{cond_5d:.0e} (5D) to ~{cond_2d:.0f} (2D),")
    p("allowing the optimizer to converge to machine precision.")
    p()
    grad_ab_mag = max(abs(grad_a3[1]), abs(grad_a3[2]))
    grad_steep_mag = max(abs(grad_a3[0]), abs(grad_a3[4]))
    grad_spread = grad_steep_mag / grad_ab_mag if grad_ab_mag > 0 else float("inf")
    p("Approach 3's precision is limited not by its grid search or")
    p("optimizer settings, but by L-BFGS-B's inability to accurately")
    p(f"approximate the inverse Hessian of a κ ≈ {cond_5d:.0e} landscape.")
    p(f"The gradient signal along A/B exists but is ~{grad_spread:.0f}× smaller")
    p("than along E/β; the limited-memory Hessian cannot amplify it correctly.")
    p()
    p("For practitioners: when fitting Chinchilla-style loss surfaces,")
    p("variable projection (VPNLS) should be preferred over direct 5D")
    p("optimization whenever high precision is needed. The advantage is")
    p("structural, not algorithmic — it comes from reformulating the")
    p("problem to avoid ill-conditioning rather than from a better")
    p("optimizer.")

    return "\n".join(lines)


if __name__ == "__main__":
    report = main()

    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp8")
    output_path = output_dir / "conditioning_analysis.txt"
    output_path.write_text(report + "\n")

    # Also print to console
    print(report)
    print()
    print(f"Report saved: {output_path}")
