# Article Spec: Scaling Law Fitting Pitfalls

## Agent Instructions

- Create blog post as single html file based on this spec.
- Length: target a ~20 minute read. 
- Audience: ML practitioners familiar with scaling laws but not Approach 2/3 nuances. 
- Purpose: Demonstrate systematic biases in Chinchilla Approach 2 using noise-free synthetic data. 
- Figures: Use custom code extractions to generate figures or new data, not direct experiment outputs from other parts of this project.
- Tone: Use a soft, neutral tone; avoid strong language like "catastrophic", "disastrous", "corrupted", etc. when referring to critiques of Approach 2; target a more balanced, informative tone.
- Grammar: Avoid em dashes; use other grammatical devices instead.

---

## Motivation

- **Where it comes from**: TBD — origin of Approach 2, the Chinchilla paper context
- **Who uses this method**: TBD — labs, researchers, practitioners applying scaling laws
- **Why they use it**: TBD — simplicity, avoids nonlinear optimization, interpretable steps
- **Why I'm personally using it**: TBD — specific context for this analysis, scaling laws for scientific data modalities

**Article focus**: This article examines pitfalls of Approach 2 using noise-free synthetic data. By eliminating statistical noise, we isolate the systematic biases inherent to the method itself.

---

## Approaches to Fitting Scaling Laws

Brief recap of Chinchilla loss surface: `L(N, D) = E + A/N^α + B/D^β`

**Approach 2** (IsoFLOP Parabolic Fitting): Sample loss along fixed-compute contours. Fit parabolas to `L vs. log(N)` at each compute budget. Extract optimal N* from parabola vertices. Fit power law `N* ∝ C^a` to infer scaling exponents. Key assumption: parabolas accurately approximate loss near optimum.

**Approach 3** (Direct Surface Fitting): Fit all 5 parameters simultaneously via nonlinear optimization. Known to be unstable and initialization-sensitive.

Figure: Schematic of Approach 2 pipeline (sample → fit parabolas → extract minima → fit power law)

---

## The Happy Path — Symmetric Surfaces

Conditions: symmetric surface (α = β, A = B), perfect sampling centers, no noise.

Show isoflop curves and parabola fits. Exponents a, b perfectly recovered. Intercepts a₀, b₀ perfectly recovered. Extrapolation is perfect.

Key message: under symmetric conditions, Approach 2 is flawless.

Figures: IsoFLOP curves with parabola fits (symmetric). Power-law fits showing perfect alignment.

---

## Asymmetric Surfaces — Intercepts Go Wrong

Conditions: asymmetric surface (α ≠ β), perfect sampling centers, no noise.

### What Happens

Simulation results show that when the loss surface is asymmetric, Approach 2 produces systematically wrong intercepts while exponents remain accurate. This isn't statistical noise — it's a deterministic bias from fitting parabolas to a non-parabolic surface.

Figures: Exponent error (~0) vs. intercept error (nonzero) by sampling range. True vs. inferred parabola minima overlay.

### Why This Is Surprising

This happens for the Chinchilla paper's own reported parameters (α=0.34, β=0.28). Even with perfect, noiseless data sampled exactly at the true optimum, Approach 2 returns biased intercepts.

### Why It Happens

The IsoFLOP loss curve is not a true parabola — it contains exponential terms. When a parabola is fit to this curve, the parabola's minimum (vertex) doesn't land exactly at the true optimum. It shifts slightly — and the key insight is that this shift depends only on the loss surface shape (α, β) and the sampling grid. It does not depend on compute budget. The sampling grid size becomes important here: wider grids amplify the mismatch between the true curve and its parabolic approximation, increasing the vertex shift.

Since the vertex shift is constant across all compute budgets, it biases every inferred N* by the same multiplicative factor. When fitting log(N*) vs log(C) to extract scaling exponents:
- The slope (exponent) is unchanged — multiplying all N* values by a constant factor adds a constant to log(N*), which doesn't affect the slope
- The intercept absorbs the entire error — it's biased by exactly that multiplicative factor

**Exact derivation**: The intercept error can be derived analytically in closed form as a function of only α, β, and the sampling grid — no other parameters affect it. Link to full derivation document. Include key results from the linked derivation (e.g., the closed-form expression for vertex shift and how it scales with grid width).

**Intuition via Taylor expansion**: A parabola is a 2nd-order polynomial, which is equivalent to a 2nd-order Taylor expansion around the optimum. The approximation L(w) ≈ L(0) + ½L''(0)w² is only valid when higher-order terms are negligible — i.e., when samples are close to the true minimum. As sampling range increases, 3rd and 4th order terms grow. For symmetric surfaces (α = β), odd-order terms cancel by symmetry, preserving the vertex location. For asymmetric surfaces, they don't cancel, shifting the fitted vertex away from the true optimum.

### Why It Matters

Extrapolation uses both exponents AND intercepts: `D* = b₀ × C^b`. Small intercept error → large extrapolation error at high compute. Quantify relative error in D* at 10²² to 10²⁵ FLOPs.

Key message: intercept errors that seem small can compound significantly at scale.

Figures: Extrapolation error vs. compute budget. Annotate magnitude (e.g., "2× overestimate of tokens needed").

---

## Sampling Bias — When Exponents Break Too

Conditions: symmetric surface (α = β), with intentional sampling center bias.

**Constant multiplicative bias**: All sampling centers shifted by constant factor (e.g., 1.5×). Exponents still perfect, intercepts biased. Why: constant multiplicative offset in log-space is constant additive → shifts intercept only. This is the same mechanism as asymmetric surfaces with no sampling bias — both produce a constant vertex shift across compute budgets.

**Drifting bias**: Bias varies with compute budget (undershoots at low C, overshoots at high C). BOTH exponents and intercepts are wrong.

Key message: only constant multiplicative bias preserves exponents; any other pattern introduces errors in both.

Figures: Comparison of no bias vs. constant scale vs. drift. Exponent error curves for each bias type.

---

## The Fix — Variable Projection Surface Fitting

Challenge: naive nonlinear optimization is unstable.

Solution: variable projection. For fixed (α, β), loss function is linear in (E, A, B). Grid search over (α, β), solve linear system for each. Select best, optionally refine with local optimizer.

Result: all 5 parameters recovered perfectly. Extrapolation is perfect.

Key message: variable projection makes direct fitting robust.

Figures: Extrapolation comparison — Approach 2 vs. surface fitting.


