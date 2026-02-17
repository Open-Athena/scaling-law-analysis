"""Experiment 8: Optimizer conditioning — Approach 3 vs VPNLS.

Demonstrates why VPNLS achieves higher precision than Approach 3 on
noise-free Chinchilla loss surfaces. The root cause is the extreme
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
    p("Chinchilla loss surfaces. The difference is not due to grid search")
    p("quality or optimizer settings — it is a fundamental consequence of")
    p("the ill-conditioning of the 5D optimization landscape.")
    p()

    # ── Setup ─────────────────────────────────────────────────────────────

    rule("-")
    p("Setup")
    rule("-")
    p()
    p(
        f"Loss surface: α={loss.alpha}, β={loss.beta}, "
        f"A={loss.A}, B={loss.B}, E={loss.E}"
    )
    p(f"True scaling exponents: a={loss.a:.6f}, b={loss.b:.6f}")
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

    # Both methods now return a result with status metadata —
    # no exceptions for soft issues like ABNORMAL termination.
    est_a3 = fit_approach3(N, D, L)
    a3_a = est_a3.beta / (est_a3.alpha + est_a3.beta)
    a3_b = est_a3.alpha / (est_a3.alpha + est_a3.beta)
    a3_a_err = (a3_a - loss.a) / loss.a
    a3_b_err = (a3_b - loss.b) / loss.b

    est_vp = fit_surface(N, D, L, method="nelder-mead")
    vp_a = est_vp.beta / (est_vp.alpha + est_vp.beta)
    vp_b = est_vp.alpha / (est_vp.alpha + est_vp.beta)
    vp_a_err = (vp_a - loss.a) / loss.a
    vp_b_err = (vp_b - loss.b) / loss.b

    p("Approach 3 (L-BFGS-B, 5 parameters):")
    p(f"  a = {a3_a:.15f}  (relative error: {a3_a_err:+.2e})")
    p(f"  b = {a3_b:.15f}  (relative error: {a3_b_err:+.2e})")
    p(
        f"  status: {est_a3.status.value}"
        + (f" — {est_a3.status_message}" if est_a3.status_message else "")
    )
    p()
    p("VPNLS (Nelder-Mead, 2 parameters + NNLS):")
    p(f"  a = {vp_a:.15f}  (relative error: {vp_a_err:+.2e})")
    p(f"  b = {vp_b:.15f}  (relative error: {vp_b_err:+.2e})")
    p(
        f"  status: {est_vp.status.value}"
        + (f" — {est_vp.status_message}" if est_vp.status_message else "")
    )
    p()
    p(
        f"VPNLS is ~{abs(a3_a_err / vp_a_err):.0f}x more precise on 'a' "
        f"and ~{abs(a3_b_err / vp_b_err):.0f}x more precise on 'b'."
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
    p("barely changes anything. Gradient-based optimizers like L-BFGS-B")
    p("struggle with this because finite floating-point precision cannot")
    p("accurately represent gradient information along the flat directions.")
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
    p(f"  - The two flattest directions (λ₁, λ₂ ≈ 10⁻⁵) point almost")
    p(f"    entirely along A and B. This means perturbing the linear")
    p(f"    coefficients A or B barely changes the RSS — the loss surface")
    p(f"    is extremely insensitive to these parameters near the optimum.")
    p(f"  - The steepest direction (λ₅ ≈ {eigvals_5d[-1]:.2e}) is")
    p(f"    dominated by α, the most sensitive parameter.")
    p(f"  - The condition number κ ≈ {cond_5d:.0e} means L-BFGS-B must")
    p(
        f"    resolve curvature differences spanning {np.log10(cond_5d):.0f} "
        f"orders of magnitude."
    )
    p()
    p("At RSS values near 10⁻¹⁶, the gradient along the flat A/B")
    p("directions is dominated by floating-point rounding error. L-BFGS-B")
    p("cannot distinguish true descent from numerical noise, so it stops")
    p("early (or reports 'ABNORMAL' line search failure).")
    p()

    # ── Step 3: 2D Hessian analysis ──────────────────────────────────────

    rule("-")
    p("Step 3: Conditioning of the 2D landscape (VPNLS)")
    rule("-")
    p()
    p("VPNLS avoids the ill-conditioned directions entirely. For each")
    p("candidate (α, β), it solves for E, A, B using Non-Negative Least")
    p("Squares (NNLS) — a direct linear algebra solve, not iterative")
    p("optimization. This eliminates the 3 flattest dimensions from the")
    p("search space.")
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
    p("Summary of precision achieved:")
    p(
        f"  Approach 3:  |a| error ≈ {abs(a3_a_err):.0e},  "
        f"|b| error ≈ {abs(a3_b_err):.0e}"
    )
    p(
        f"  VPNLS:       |a| error ≈ {abs(vp_a_err):.0e},  "
        f"|b| error ≈ {abs(vp_b_err):.0e}"
    )
    p()
    precision_ratio = abs(a3_a_err / vp_a_err) if vp_a_err != 0 else float("inf")
    p(
        f"The ~{cond_5d / cond_2d:.0e} gap in conditioning explains the ~{precision_ratio:.0f}x gap"
    )
    p("in precision. L-BFGS-B's convergence rate degrades with condition")
    p("number, and at κ ≈ 10¹¹, floating-point noise in gradient")
    p("evaluation becomes the binding constraint before the optimizer")
    p("reaches the true minimum.")
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
        a_err = (beta / (alpha + beta) - loss.a) / loss.a
        p(f"  a error: {a_err:.2e}")
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
    a_ftol0 = beta / (alpha + beta)
    a_ftol0_err = (a_ftol0 - loss.a) / loss.a
    grad_norm = float(
        np.linalg.norm(_surface_rss_grad(result_ftol0.x, log_N, log_D, L))
    )

    p(f"L-BFGS-B with ftol=0 (grid search init):")
    p(f"  Iterations: {result_ftol0.nit}")
    p(f"  Final RSS:  {result_ftol0.fun:.2e}")
    p(f"  |a| error:  {abs(a_ftol0_err):.2e}")
    p(f"  Grad norm:  {grad_norm:.2e}")
    p(f"  Message:    {result_ftol0.message}")
    p()
    p("Even with ftol=0 (no function-value stopping), L-BFGS-B improves")
    p(
        f"only modestly (|a| error {abs(a3_a_err):.0e} → {abs(a_ftol0_err):.0e}). The gradient norm"
    )
    p(f"of {grad_norm:.2e} shows the optimizer is stuck — not because it")
    p("truly converged, but because the gradient is too noisy to follow.")
    p()

    # ── Step 6: Why this matters less with noise ──────────────────────────

    rule("-")
    p("Step 6: Why noise masks the conditioning problem")
    rule("-")
    p()
    p("With realistic noise (σ > 0), the true minimum of the RSS shifts")
    p("away from the true parameters. The relevant precision is now set")
    p("by the noise floor, not machine epsilon. Since noise-induced errors")
    p("are typically ~10⁻² to 10⁻¹ (much larger than the ~10⁻¹⁰ limit"),
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
    p("projection eliminates the 3 most ill-conditioned parameter")
    p("directions (E, A, B) from the search space. This reduces the")
    p(f"condition number from ~{cond_5d:.0e} (5D) to ~{cond_2d:.0f} (2D),")
    p("allowing the optimizer to converge to machine precision.")
    p()
    p("Approach 3's precision is limited not by its grid search or")
    p("optimizer settings, but by the fundamental inability of gradient-")
    p("based methods to navigate an extremely ill-conditioned landscape")
    p("when floating-point arithmetic is the binding constraint.")
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
