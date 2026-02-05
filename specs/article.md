# Article Spec: Scaling Law Fitting Pitfalls

HTML blog post, ~10 minute read. Demonstrates systematic biases in Chinchilla Approach 2 using noise-free synthetic data. Audience: ML practitioners familiar with scaling laws but not Approach 2/3 nuances. Uses custom code extractions, not direct experiment outputs. Taylor math deferred to future work.

---

## Approaches to Fitting Scaling Laws

Brief recap of Chinchilla loss surface: `L(N, D) = E + A/N^α + B/D^β`

**Approach 2** (IsoFLOP Parabolic Fitting): Sample loss along fixed-compute contours. Fit parabolas to `L vs. log(N)` at each compute budget. Extract optimal N* from parabola vertices. Fit power law `N* ∝ C^a` to infer scaling exponents. Key assumption: parabolas accurately approximate loss near optimum.

**Approach 3** (Direct Surface Fitting): Fit all 5 parameters simultaneously via nonlinear optimization. Known to be unstable and initialization-sensitive.

Article focus: pitfalls of Approach 2 on noiseless synthetic data.

Figure: Schematic of Approach 2 pipeline (sample → fit parabolas → extract minima → fit power law)

---

## The Happy Path — Symmetric Surfaces

Conditions: symmetric surface (α = β, A = B), perfect sampling centers, no noise.

Show isoflop curves and parabola fits. Exponents a, b perfectly recovered. Intercepts a₀, b₀ perfectly recovered. Extrapolation is perfect.

Key message: under symmetric conditions, Approach 2 is flawless.

Figures: IsoFLOP curves with parabola fits (symmetric). Power-law fits showing perfect alignment.

---

## Asymmetric Surfaces — Intercepts Go Wrong

Conditions: asymmetric surface (Chinchilla: α=0.34, β=0.28), perfect sampling centers, no noise.

**Surprising finding**: Using the Chinchilla paper's own reported parameters, Approach 2 produces systematically wrong intercepts — even with perfect data. This isn't statistical noise; it's a deterministic bias from fitting parabolas to a non-parabolic surface. Exponents are still exact, but intercept error grows with sampling range.

Key message: Approach 2 applied to the Chinchilla loss surface introduces systematic intercept errors that affect extrapolation.

Figures: Chinchilla vs. high imbalance isoflops. Exponent error (~0) vs. intercept error (nonzero) by sampling range. True vs. inferred parabola minima overlay.

---

## Sampling Bias — When Exponents Break Too

**Constant multiplicative bias**: All sampling centers shifted by constant factor (e.g., 1.5×). Exponents still perfect, intercepts biased. Why: constant multiplicative offset in log-space is constant additive → shifts intercept only.

**Drifting bias**: Bias varies with compute budget (undershoots at low C, overshoots at high C). BOTH exponents and intercepts are wrong.

Key message: only constant multiplicative bias preserves exponents; any other pattern introduces errors in both.

Figures: Comparison of no bias vs. constant scale vs. drift. Exponent error curves for each bias type.

---

## Extrapolation — Where It Falls Apart

Extrapolation uses both exponents AND intercepts: `D* = b₀ × C^b`. Small intercept error → large extrapolation error at high compute. Quantify relative error in D* at 10²² to 10²⁵ FLOPs.

Key message: intercept errors that seem small become catastrophic at scale.

Figures: Extrapolation error vs. compute budget. Annotate magnitude (e.g., "2× overestimate of tokens needed").

---

## The Fix — Variable Projection Surface Fitting

Challenge: naive nonlinear optimization is unstable.

Solution: variable projection. For fixed (α, β), loss function is linear in (E, A, B). Grid search over (α, β), solve linear system for each. Select best, optionally refine with local optimizer.

Result: all 5 parameters recovered perfectly. Extrapolation is perfect.

Key message: variable projection makes direct fitting robust.

Figures: Extrapolation comparison — Approach 2 vs. surface fitting.


