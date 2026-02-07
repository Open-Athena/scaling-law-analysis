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

Conditions: symmetric surface (α = β, A = B), perfect sampling centers, extra large (±16×) sampling grid, no noise.

Show isoflop curves and parabola fits. Exponents a, b perfectly recovered. Intercepts a₀, b₀ perfectly recovered. Extrapolation is perfect.

Key message: under symmetric conditions, Approach 2 is flawless.

Figures: IsoFLOP curves with parabola fits (symmetric). Power-law fits showing perfect alignment.

---

## Asymmetric Surfaces — Intercept and Extrapolation Errors

Conditions: asymmetric surface (α ≠ β), perfect sampling centers, extra large (±16×) sampling grid, no noise.

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

Extrapolation to higher compute budgets requires both exponents and intercepts to be correct. The previous section established that asymmetric loss surfaces produce provably biased intercepts even under ideal experimental conditions. Here we quantify what those errors mean in practical terms by examining compute-optimal token prediction: given a compute budget, how many tokens does the inferred scaling law predict?

Up to this point, all analysis has assumed a single fixed sampling grid width. We now examine how token prediction error varies with both compute budget and sampling grid width. For surfaces with asymmetric exponents, wider sampling grids amplify the parabola-fitting mismatch, increasing the constant vertex shift and thus the intercept bias.

A sampling grid of "±k×" means model sizes range from 1/k× to k× the true optimum at each compute budget. The total range covered is k² (the ratio of largest to smallest model size). The log₁₀ of that ratio tells you how many factors of 10, or "decades," the grid spans end-to-end (e.g. a value of 1.81 means the largest model is 10^1.81 ≈ 64× the smallest). The table below shows the four grid widths used in this analysis:

| Grid Name          | ±k×  | Sampling Range     | Total Ratio | Decade Span (factors of 10) |
|--------------------|------|--------------------|-------------|-----------------------------|
| Extra Small (XS)    | ±2×  | 1/2× to 2×        | 4×          | 0.60                        |
| Small (S)          | ±4×  | 1/4× to 4×        | 16×         | 1.20                        |
| Large (L)          | ±8×  | 1/8× to 8×        | 64×         | 1.81                        |
| Extra Large (XL)    | ±16× | 1/16× to 16×      | 256×        | 2.41                        |

In practice, scaling law experiments typically sample across 1 to 2 decades in token count, placing the Small and Large grids squarely within the realistic range. The Extra Small and Extra Large grids bracket this range on either side, illustrating how the biases shrink or grow as the sampling window narrows or widens. The Extra Large grid (±16×, ~2.4 decades) is the default used in all single-grid analyses in the preceding sections.

Figures: Bar chart with x-axis = loss surface (symmetric, chinchilla, high_imbalance), y-axis = relative error in D* (%). Bars grouped by sampling grid width (extra small ±2×, small ±4×, large ±8×, extra large ±16×). Single extrapolation budget (10²⁴ FLOPs). Negative bars = underestimation. Annotate with true D* scale.

---

## Off-Center Sampling — Exponent and Extrapolation Errors

### Introduction

All analysis so far has assumed that IsoFLOP sampling grids are centered exactly at the true compute-optimal model size N* for each compute budget. In practice, you don't know N* before running the experiment; that's what you're trying to infer. The sampling center is chosen based on prior estimates, heuristics, or earlier smaller-scale runs, and these are rarely exactly right. We call the systematic offset between the assumed sampling center and the true optimum **off-center sampling**.

This bias is distinct from the surface asymmetry errors studied in the previous section. Asymmetry errors arise from the shape of the loss surface itself (α ≠ β) and are present even when sampling is perfectly centered. Off-center sampling errors arise from where you place the sampling grid, regardless of the surface shape. In practice both effects compound, but to understand each in isolation, this section studies off-center sampling on **symmetric loss surfaces only** (α = β). On a symmetric surface, the previous section established that Approach 2 produces zero error with perfect centering. Any errors introduced here are therefore purely attributable to off-center sampling, with no confounding from surface asymmetry.

Two forms of off-center sampling are studied, corresponding to distinct failure modes a practitioner might encounter:

- **Constant multiplicative bias** (center_scale): every sampling center is shifted by the same multiplicative factor across all compute budgets (e.g., always sampling at 1.5× the true optimal N*). This models a persistent miscalibration, such as using a fixed rule-of-thumb that consistently overestimates or underestimates the optimal size.

- **Drifting bias** (drift_rate): the sampling center diverges progressively from the true optimum as compute budget increases. At low compute, the guess is close; at high compute, it undershoots (toward smaller N). This models the more realistic scenario where prior knowledge degrades at scales beyond what has been explored, causing sampling centers to become increasingly stale as the experiment pushes to larger budgets.

The key question is how each form of bias propagates through the Approach 2 pipeline: does it corrupt only the intercept (as surface asymmetry does), or does it also distort the exponents?

Conditions: symmetric surface (α = β), extra large (±16×) sampling grid, with intentional off-center sampling.

### Constant Multiplicative Bias

All sampling centers shifted by constant factor (e.g., 1.5×). Exponents still perfect, intercepts biased. Why: constant multiplicative offset in log-space is constant additive → shifts intercept only. This is the same mechanism as asymmetric surfaces with perfectly centered sampling — both produce a constant vertex shift across compute budgets.

### Drifting Bias

Bias varies with compute budget (undershoots at low C, overshoots at high C). BOTH exponents and intercepts are wrong. Unlike a constant shift, a drift introduces a compute-dependent vertex shift, which changes the slope of log(N*) vs log(C), corrupting the exponent.

Key message: only constant multiplicative bias preserves exponents; any other pattern introduces errors in both. Constant bias is structurally equivalent to the asymmetry effect (both produce uniform vertex shifts), while drift is qualitatively worse because it distorts the relationship between N* and C itself.

Figures: Comparison of no bias vs. constant scale vs. drift. Exponent error curves for each bias type.

---

## The Fix — Variable Projection Surface Fitting

Challenge: naive nonlinear optimization is unstable.

Solution: variable projection. For fixed (α, β), loss function is linear in (E, A, B). Grid search over (α, β), solve linear system for each. Select best, optionally refine with local optimizer.

Result: all 5 parameters recovered perfectly. Extrapolation is perfect.

Key message: variable projection makes direct fitting robust.

Figures: Extrapolation comparison — Approach 2 vs. surface fitting.


