# Experiments Spec

This document specifies all experiments for analyzing scaling law inference methods.

## Experiment 1: Empirical Error

**Hypothesis**: The accuracy of Chinchilla Approach 2 is dependent on the validity of the parabolic approximation. Sampling farther from the optimum introduces systematic bias.

**Method**:

1. Generate synthetic loss data from: L(N, D) = E + A/N^α + B/D^β
   - Ground truth: α=0.34, β=0.28, A=406.4, B=410.7, E=1.69
   - No statistical noise (pure deterministic loss surface)

2. For each sampling range (from ±2x to ±100x around the optimum):
   - Sample IsoFLOP contours at 5 compute budgets: 10^17 to 10^21 FLOPs
   - Optionally apply sampling center bias via:
     - drift_rate: asymmetric linear drift from optimal (0 at lowest compute budget, -drift_rate at highest, causing undershoot toward smaller N)
     - center_scale: constant multiplier applied to all sampling centers
   - Fit parabolas to L vs log(N) for each budget → extract N*
   - Fit parabolas to L vs log(D) for each budget → extract D*
   - Fit power laws: N* ∝ C^a and D* ∝ C^b
   - Note: The C=6ND approximation should ONLY be used for IsoFLOP sampling and inferring true optimal values, never for exponent inference/fitting

3. Compare recovered exponents to true values:
   - True a = β/(α+β) (N* exponent)
   - True b = α/(α+β) (D* exponent)

**Visualization**:

Produce a single figure with three rows:

1. **IsoFLOP curves panel**: For 3 representative sampling ranges (narrow, medium, wide), show:
   - Sampled loss values along each IsoFLOP contour
   - Fitted parabolas overlaid on the data
   - True optimal N* marked distinctly from inferred N*

2. **Power-law fits panel**: For the same 3 sampling ranges, show:
   - Inferred N* and D* vs compute budget (dual y-axes)
   - Power-law fit lines for inferred values (solid)
   - True N* and D* with connecting lines (dashed)
   - This reveals slope divergence between true and inferred optima

3. **Error analysis panel**: Three subplots showing relative error as a function of sampling range:
   - **Exponent error**: Relative error in recovered scaling exponents a and b
   - **Intercept error**: Relative error in power-law fit intercepts a₀ and b₀ (from N* = a₀·C^a and D* = b₀·C^b)
   - **Optimum error**: Relative error in parabola-inferred N* and D* compared to true optimal values, shown per compute budget (not aggregated)
   
   This reveals whether error grows systematically with wider sampling and which quantities are most sensitive.


## Experiment 2: Exponent Imbalance Sensitivity

**Hypothesis**: The accuracy of Chinchilla Approach 2 is sensitive to scaling exponent imbalance. Greater asymmetry between α and β leads to larger recovery errors.

**Method**:

1. Generate synthetic loss data using the same procedure as Experiment 1, across six loss surface configurations defined by their α/β ratio (keeping α+β=0.62 constant):
   - **Reference**: α=0.34, β=0.28 (Chinchilla paper defaults, ratio≈1.21)
   - **Balanced**: ratio=1.0 (equal exponents)
   - **Small imbalance**: ratio=1.5
   - **Moderate imbalance**: ratio=2.0
   - **High imbalance**: ratio=3.0
   - **Extreme imbalance**: ratio=9.0

2. For each configuration:
   - Use fixed drift_rate=0.2 and center_scale=1.0
   - Sweep sampling ranges from ±2x to ±100x around the optimum
   - Recover exponents a and b via Approach 2

**Visualization**:

Produce a single figure showing relative error in recovered exponents (a and b) as a function of sampling range, with one curve per configuration. This reveals how exponent imbalance affects sensitivity to sampling range.


## Experiment 3: Sampling Drift Sensitivity

**Hypothesis**: The accuracy of Chinchilla Approach 2 is sensitive to systematic biases in the sampling center.

**Method**:

1. Generate synthetic loss data across three loss surface configurations:
   - **Symmetric**: α=β=0.31, A=B=400, E=1.69 (balanced)
   - **Chinchilla**: α=0.34, β=0.28, A=406.4, B=410.7, E=1.69 (paper defaults)
   - **Asymmetric**: α/β ratio = 3.0 (keeping α+β=0.62)

2. For each loss surface, test five sampling bias configurations:
   - **Baseline**: no drift, no center scaling
   - **Drift 0.2** and **Drift 0.4**: linear drift only
   - **Scale 1.5** and **Scale 2.0**: center scaling only

3. For each configuration, sweep sampling ranges from ±2x to ±100x (as in Experiment 2) and recover exponents via Approach 2.

**Visualization**:

Produce the following figures:

1. **Parameter estimation errors** (one combined figure):
   - Grid with one row per loss surface (3 rows) and 4 columns
   - Columns show: (1) N* exponent error, (2) D* exponent error, (3) N* intercept error, (4) D* intercept error
   - Each plot shows relative error vs sampling range, with one curve per sampling bias configuration

2. **Optimal value estimation errors** (one figure per loss surface):
   - Each loss surface gets its own figure showing errors across all compute budgets
   - One row per sampling bias configuration, using only baseline, highest drift (drift_0.4), and highest scale (scale_2.0) for clarity (3 rows)
   - Four columns: (1) N* relative error, (2) D* relative error, (3) N* signed error, (4) D* signed error
   - Relative error: (inferred - true) / true, expressed as percentage
   - Signed error: inferred - true, in absolute units (parameter count or token count)
   - Use distinct colors for N* vs D* columns to make results visually distinct; use increasing opacity and marker size to denote higher compute budgets (consistent with Experiment 1)
   - This reveals how optima errors vary with both sampling range and compute scale


## Experiment 4: Extrapolation Error

**Hypothesis**: The accuracy of scaling law exponent inference degrades when extrapolating to compute budgets beyond those used for fitting. The magnitude of this error depends on the loss surface geometry.

**Method**:

1. Use the same three loss surface configurations as Experiment 3 (symmetric, chinchilla, asymmetric).

2. Use the same five sampling bias configurations as Experiment 3 (baseline, drift_0.2, drift_0.4, scale_1.5, scale_2.0).

3. Use the same three sampling ranges as Experiment 1 (narrow, medium, wide).

4. For each combination of loss surface, sampling bias, and sampling range:
   - Fit scaling law exponents using existing compute budgets (10^17 to 10^21 FLOPs)
   - Extrapolate to higher compute budgets (10^22 to 10^25 FLOPs)
   - Compare inferred vs true optimal token counts D* at each extrapolation budget

**Visualization**:

Produce a single figure organized as a grid with one row per sampling range (3 rows: narrow, medium, wide) and one column per loss surface (3 columns: symmetric, chinchilla, asymmetric). Each panel shows the relative error in inferred D* as a function of extrapolation compute budget, with one curve per sampling bias configuration. This reveals how extrapolation error depends on the sampling range, loss surface geometry, and sampling biases.

## Experiment 5: Parameter Recovery

**Hypothesis**: Variable projection with grid search (over α/β) provides stable and accurate scaling law parameter recovery, and extrapolation using fitted parameters remains accurate even at compute budgets far beyond the fitting range.

**Method**:

1. Generate synthetic loss data using the same procedure as Experiments 3 and 4.

2. Fit all 5 parameters simultaneously via **variable projection**: for fixed (α, β), the loss L = E + A·N^(-α) + B·D^(-β) is linear in (E, A, B) and solved via NNLS.
   - All methods search only over (α, β) and solve (E, A, B) via NNLS at each candidate, isolating the optimizer as the sole variable.
   - Six optimization configurations:
     - **Nelder-Mead**: coarse grid search init (32×32) + Nelder-Mead refinement (gradient-free)
     - **L-BFGS-B (default eps=1e-8)**: same coarse grid init + L-BFGS-B refinement with default forward finite differences
     - **L-BFGS-B (central diff)**: same, but with 3-point central finite differences
     - **L-BFGS-B (eps=1e-6)**: forward diff with step size 100x above default
     - **L-BFGS-B (eps=1e-10)**: forward diff with step size 100x below default
     - **Grid**: fine grid search (256×256) with no local refinement
   - Convergence failures (ABNORMAL) are recorded as NaN and shown as gaps in the figure
   - Diagnostic checks: convergence failure, (α, β) at bounds, E/A/B near zero, non-finite values

3. For each configuration (loss surface × sampling bias × sampling range):
   - Pool the sampled (N, D, L) data across all compute budgets
   - Fit using all six method configurations
   - Compute relative errors for all 5 parameters compared to ground truth

4. For extrapolation analysis, use the same setup as Experiment 4.

5. L-BFGS-B sensitivity findings:
   - **Precision ceiling from numerical gradients**: forward-diff L-BFGS-B converges reliably but plateaus at ~1e-5% error (vs ~1e-8% for Nelder-Mead). This ceiling comes from the ~1e-8 precision of forward finite-difference gradient estimates.
   - **Central diff closes the precision gap but introduces failures**: 3-point central differences achieve Nelder-Mead-level precision (~1e-8%) but cause sporadic ABNORMAL line search failures (1/60 in our sweep). These are false negatives: the optimizer has already reached the true minimum (RSS ~1e-19) but the line search cannot verify progress because function values are near machine zero.
   - **Custom eps values cause failures in both directions**: eps=1e-6 (100x above default) produces ABNORMAL failures in 20/60 trials with worse precision where it converges. eps=1e-10 (100x below default) is slightly more precise than default but introduces failures on the Asymmetric surface (3/20). There is no eps value that reliably improves on the default.
   - **Forward diff "succeeds" by stopping early**: default L-BFGS-B reports success at RSS ~1e-14 because noisy gradients satisfy termination criteria prematurely. Central diff reaches RSS ~1e-19 (100,000x better) but fails the convergence check because the line search cannot distinguish progress at that scale.
   - **Takeaway**: L-BFGS-B exposes multiple interacting settings — `eps` (FD step size), `jac` (FD scheme), `ftol` (objective tolerance), `gtol` (gradient tolerance), `maxcor` (Hessian corrections), `maxls` (line search steps) — where choices interact: tight `gtol` demands accurate gradients, which demands careful `eps`, which risks cancellation. Nelder-Mead has only `xatol`/`fatol` (simplex convergence) and `maxiter`, with no gradient-related settings and no interactions between them. On noise-free data, Nelder-Mead achieves machine-precision accuracy with defaults; L-BFGS-B cannot match this without coordinated tuning, and no single configuration avoids all failures.

**Visualization**:

Produce three figures:

1. **Parameter estimation errors** (one figure, Nelder-Mead only):
   - Grid with one row per loss surface (3 rows) and 5 columns (one per parameter: E, A, B, α, β)
   - Each panel shows relative error vs sampling range, with one curve per sampling bias configuration
   - Follow the same style as the parameter errors figure from Experiment 3

2. **Extrapolation errors** (one figure, Nelder-Mead only):
   - Same layout and style as Experiment 4's extrapolation figure
   - Allows direct comparison of extrapolation error when using surface fitting vs Approach 2

3. **Method comparison** (one figure + CSV):
   - Grid with one row per loss surface (3 rows) and 5 columns (one per parameter: E, A, B, α, β)
   - Each panel shows absolute relative error (log scale) vs sampling range, with one curve per method configuration; gaps indicate convergence failures
   - Baseline (no bias) only, to isolate optimizer precision from sampling effects
   - Accompanying CSV summarizes failure rate and max error per method per surface


## Experiment 6: Analytical Error

**Hypothesis**: It is possible to analytically derive the error in the inferred exponents and intercepts via Approach 2 as a function of loss surface parameters and grid specification.

**Goal**: Derive a closed-form expression for errors in Approach 2 N* inference (both exponent and intercept) as a function of loss surface parameters and grid specification.

**Inputs**: α, β (loss surface exponents), W (grid half-width in log₁₀ space), n (number of sample points).

**Approach**:

1. Work in log-space coordinates centered on the true optimum: w = log₁₀(N/N*), so N = N*·10^w
   - w measures how many orders of magnitude a sample point is from the true optimum: w=0 at the optimum, w=1 means 10× larger, w=-1 means 10× smaller
2. Substitute into the IsoFLOP loss L(N;C) = E + A·N^(-α) + B·(6N/C)^β to get L(w)
   - This gives L(w) = E + P·10^(-αw) + R·10^(βw) where P and R are functions of N*, C, and loss surface parameters
   - Note: L(w) is not a parabola in w — it contains exponential terms
3. Use the first-order optimality condition (dL/dN = 0 at N*) to relate P and R, which may allow simplification
   - At the true optimum, the loss is minimized, so derivatives balance. This constrains the ratio of P to R, potentially letting you eliminate one in favor of the other
4. Derive parabola coefficients from least-squares fitting:
   - Fitting L̂ = a₀ + a₁w + a₂w² to points (wᵢ, Lᵢ) minimizes Σ(Lᵢ - a₀ - a₁wᵢ - a₂wᵢ²)²
   - Taking derivatives w.r.t. a₀, a₁, a₂ and setting to zero gives normal equations
   - The solution involves sums: Σwᵢ, Σwᵢ², Σwᵢ³, Σwᵢ⁴, ΣLᵢ, ΣwᵢLᵢ, Σwᵢ²Lᵢ
   - For equally-spaced points in [-W, W], the grid is symmetric about 0: for each +wⱼ there is a -wⱼ
   - Odd-power sums vanish by cancellation: wⱼ + (-wⱼ) = 0, so Σwᵢ = 0 and Σwᵢ³ = 0
   - This decouples some of the normal equations, yielding simpler formulas for a₁ and a₂
   - The parabola vertex is at w = -a₁/(2a₂)
5. Express a₁ and a₂ in terms of sums involving L(wᵢ), then substitute the loss function L(w) = E + P·10^(-αw) + R·10^(βw)
   - The sums ΣLᵢ, ΣwᵢLᵢ, Σwᵢ²Lᵢ each become sums of exponential terms evaluated at the grid points
   - Use the P/R relationship from step 3 to simplify; look for terms that cancel or factor out
   - The goal is an expression for a₁ and a₂ (and hence the vertex shift) that depends only on α, β, W, n
6. Connect the vertex shift to inference errors
   - If the fitted parabola's minimum is at w = δw instead of w = 0, then the inferred optimum is N̂* = N*·10^δw (off by a multiplicative factor)
   - Approach 2 repeats this at multiple compute budgets, then fits log(N̂*) vs log(C) to get exponent and intercept
   - Write log(N̂*) in terms of log(N*) and δw, then see how δw affects the slope (exponent) and intercept of that fit
7. Check whether the vertex shift δw depends on C
   - From step 6, log(N̂*) = log(N*) + δw·log(10). If δw varies with C, it changes the slope of log(N̂*) vs log(C), corrupting the exponent
   - If δw is constant across C, it only adds a constant offset, affecting the intercept but not the exponent
   - Look back at your expression for δw from step 5: does it contain C, or only α, β, W, n?

**Visualization**:

Produce a single figure with one panel per loss surface configuration (symmetric, chinchilla, Asymmetric). Each panel shows:
- Numerical intercept error from Approach 2 vs grid half-width W
- Predicted intercept error from the derived formula vs grid half-width W
- Maximum deviation between numerical and predicted values annotated

This validates that the derived closed-form expression exactly matches numerical results.

**Validation**: 
- Compare derived expressions against numerical Approach 2 results across multiple surface configurations (symmetric, chinchilla, Asymmetric)
- Target machine precision agreement (1e-10)
- Sanity check: symmetric surfaces (α = β) should produce zero error


## Experiment 7: Exponent Inference

**Hypothesis**: VPNLS and Approach 3 recover the scaling exponents a and b more accurately than Approach 2 when given fewer data points per isoflop curve, especially under sampling drift. Variable projection (VPNLS) may be more data-efficient than full 5D optimization (Approach 3) because it searches a lower-dimensional space.

**Method**:

1. Use the Asymmetric surface (α/β ratio = 3) with a drifting sampling bias (drift_rate = log₁₀(3)), matching the "Compounding Errors" section of the article.

2. Sweep the number of points per isoflop curve from 3 to 50, using 5 compute budgets (10¹⁷–10²¹ FLOPs).

3. At each sample size, fit Approach 2, Approach 3, and VPNLS, then compute the absolute relative error in recovered exponents a = β/(α+β) and b = α/(α+β).

4. Repeat across three grid widths (narrow ±2×, medium ±16×, wide ±100×) to understand interaction between sample density and sampling range.

**Visualization**:

Produce a single figure with 2 rows (exponent a, exponent b) × 3 columns (narrow, medium, wide grid width). Each panel shows absolute relative error (log scale) vs points per isoflop curve, with one curve per fitting method.


## Experiment 8: Conditioning Analysis

**Hypothesis**: Approach 3's lower precision on noise-free data stems from ill-conditioning of the 5D problem, not from poor initialization or optimizer choice. VPNLS avoids this via a well-conditioned 2D reformulation.

**Key finding**: Near the optimum, A and B are nearly interchangeable — perturbing either barely changes the RSS. The 5D Hessian at the true parameters has eigenvalues spanning 12 orders of magnitude: the two flattest (λ ≈ 10⁻⁵) point along A/B, the steepest (λ ≈ 3×10⁶) along β, giving κ ≈ 3.5×10¹¹. VPNLS eliminates A/B/E via NNLS and optimizes only (α, β), whose 2D Hessian has κ ≈ 11.

At Approach 3's converged solution, the gradient along A/B is ~10⁻⁹ — nonzero and well above machine epsilon, so the descent signal exists. But L-BFGS-B cannot use it: its limited-memory Hessian approximation (~10 vector pairs) cannot accurately represent κ ≈ 10¹¹, so it cannot convert those small A/B gradients into appropriately-scaled steps. Meanwhile, function-value changes are dominated by the steep E/α/β directions; once those converge, the ftol criterion triggers with the flat directions unresolved.

A perturbation test confirms the scale of the problem: displacing by δ=10⁻⁴ along each eigenvector produces gradient norms from ~5×10⁻⁹ (flat A/B) to ~3×10² (steep β) — a ~6×10¹⁰× spread matching κ.

**Method**: Generate noise-free isoflop data from the Asymmetric surface (α/β = 3) using 5 compute budgets (10¹⁷–10²¹ FLOPs), 15 points per curve, ±8× sampling range. Steps:

1. Fit both methods with default settings and compare recovered exponent precision.
2. Compute numerical Hessian of the 5D RSS at true parameters; report eigenvalues, eigenvector directions, and condition number. Do the same for the 2D VPNLS Hessian.
3. Compute the gradient at the true parameters, at Approach 3's converged solution (in both parameter space and projected onto Hessian eigenvectors), and under equal perturbations along each eigenvector — directly demonstrating the signal-vs-curvature mismatch.
4. Summarize the conditioning-vs-precision correspondence.
5. Initialize L-BFGS-B at the true parameters to rule out grid search quality as the cause. Compare grid search starting-point quality for both methods.
6. Re-run L-BFGS-B with ftol=0 from grid init to rule out early stopping as the cause.
7. Explain why statistical noise masks the conditioning gap in practice.

**Output**: Text report at `results/experiments/exp8/conditioning_analysis.txt`.
