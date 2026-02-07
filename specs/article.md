# Article Spec: Problems with Chinchilla Approach 2

> **Editorial Guidelines**
>
> - Format: single self-contained HTML file
> - Length: target a ~20 minute read
> - Audience: ML practitioners familiar with scaling laws but not Approach 2/3 nuances
> - Purpose: demonstrate systematic biases in Chinchilla Approach 2 using noise-free synthetic data
> - Figures: use custom code extractions to generate figures or new data, not direct experiment outputs from other parts of this project
> - Tone: soft, neutral; avoid strong language like "catastrophic", "disastrous", "corrupted" when referring to critiques of Approach 2; target a balanced, informative register
> - Grammar: avoid em dashes; use other grammatical devices instead

---

## Motivation

- TBD: origin of Approach 2 and the Chinchilla paper context
- TBD: who uses this method (labs, researchers, practitioners)
- TBD: why they use it (simplicity, avoids nonlinear optimization, interpretable steps)
- TBD: personal motivation (scaling laws for scientific data modalities)
- Article focus: examine pitfalls of Approach 2 using noise-free synthetic data; by eliminating noise, isolate systematic biases inherent to the method itself

---

## Approaches to Fitting Scaling Laws

- Introduce the Chinchilla loss surface: L(N, D) = E + A/N^α + B/D^β; define each term (N = parameters, D = tokens, E = irreducible loss, A/B/α/β = scaling coefficients)
- State the compute-optimal allocation: N* ∝ C^a where a = β/(α+β), D* ∝ C^b where b = α/(α+β); recovering a and b from empirical runs is the goal
- **Approach 2: IsoFLOP Parabolic Fitting**
  - Key insight: along a fixed-compute contour, loss as a function of log N is approximately parabolic near the optimum
  - Three-step pipeline: (1) sample IsoFLOP contours at various (N, D) pairs for each compute budget, (2) fit parabolas and extract vertex N* for each budget, (3) regress log N* against log C to recover scaling exponent
  - Appeal is simplicity: only polynomial fits, no nonlinear optimization; parabolic approximation comes from Taylor expansion around the optimum
- **Approach 3: Direct Surface Fitting**
  - Fit all five parameters (E, A, B, α, β) simultaneously via nonlinear least squares
  - Avoids the parabolic approximation entirely but is notoriously unstable: sensitive to initialization, prone to spurious local minima

---

## The Happy Path — Symmetric Surfaces

- Frame as establishing a baseline before examining failure modes
- Use a concrete symmetric surface: L(N, D) = 1.69 + 400/N^0.31 + 400/D^0.31
- Note that equal exponents (α = β) mean compute splits evenly; true scaling exponents are a = b = 0.5
- Describe the experiment: five IsoFLOP contours from 10^17 to 10^21 FLOPs, fit parabolas, extract optimal D*
- Figure: IsoFLOP curves with fitted parabolas (left) and power-law fit (right); true (×) and inferred (+) optima indistinguishable
- Table: show perfect recovery of b (D* exponent) and b₀ (D* intercept) with machine-precision relative errors (~10⁻¹⁰ %)
- Key result: on a symmetric surface with perfectly crafted IsoFLOP grid sampling, Approach 2 recovers both exponents and intercepts with machine-precision accuracy; the parabolic approximation is exact when α = β
- Close by noting this baseline is precisely correct under ideal conditions that are unrealistic in practice; the following sections perturb these conditions in controlled ways

---

## Asymmetric Surfaces — Intercept and Extrapolation Errors

- Frame as repeating the exact same procedure as the Happy Path; only change is α ≠ β

- **What Happens**
  - Asymmetric surfaces produce systematically wrong intercepts while exponents remain accurate
  - Two test configurations: Chinchilla (α=0.34, β=0.28, ratio ≈ 1.2) and High Imbalance (α=0.465, β=0.155, ratio = 3.0)
  - Figure: Approach 2 on both asymmetric surfaces; visible gap between true (dashed) and inferred (solid) power-law lines, especially for High Imbalance
  - Tables for each surface showing b exponent with negligible error but b₀ intercept with meaningful error; error larger for High Imbalance than Chinchilla

- **Why This Is Surprising**
  - Acknowledge that a few percent may seem minor, then enumerate the ideal advantages given to Approach 2: perfect data (no noise, every point exactly on the true surface), perfect sampling (centered at true optimum), and standard parameters (from the Chinchilla paper, not contrived)
  - Key result: even under these ideal conditions, Approach 2 produces biased intercepts; the error is inherent to the parabolic approximation

- **Why It Happens**
  - IsoFLOP loss curve is not a true parabola; it contains exponential terms
  - Parabola vertex shift depends only on surface shape (α, β) and sampling grid, not on compute budget; wider grids amplify the mismatch
  - Because the vertex shift is constant across compute budgets, it biases every N* by the same multiplicative factor:
    - Slope (exponent) is unchanged (constant additive shift in log-space doesn't affect slope)
    - Intercept absorbs the entire error
  - Exact derivation: intercept error = 10^(δw) − 1, where δw = f(α, β, W, n) depends only on surface exponents and sampling grid (width W in log-space, n points per IsoFLOP curve); properties: δw = 0 when α = β, grows with |α − β|, grows with W
  - Concrete example: show how Chinchilla parameters yield small intercept error at narrow grid vs. larger error at wide grid
  - Link to full closed-form derivation document
  - Taylor expansion intuition: parabola = 2nd-order Taylor expansion; higher-order terms grow with sampling range; odd-order terms cancel for symmetric surfaces (preserving vertex) but not for asymmetric ones (shifting vertex)

- **Why It Matters**
  - Transition: extrapolation requires both exponents and intercepts to be correct; now quantify the practical impact via compute-optimal token prediction
  - Introduce varying grid widths; define the ±kx notation (range from 1/k to k times optimum, total ratio k², decade span = log₁₀(k²))
  - Table of four grid widths:

    | Grid Name          | ±k×  | Sampling Range     | Total Ratio | Decade Span (factors of 10) |
    |--------------------|------|--------------------|-------------|-----------------------------|
    | Extra Small (XS)   | ±2×  | 1/2× to 2×        | 4×          | 0.60                        |
    | Small (S)          | ±4×  | 1/4× to 4×        | 16×         | 1.20                        |
    | Large (L)          | ±8×  | 1/8× to 8×        | 64×         | 1.81                        |
    | Extra Large (XL)   | ±16× | 1/16× to 16×      | 256×        | 2.41                        |

  - Note that real experiments typically span 1–2 decades, making S and L the realistic range; XS and XL bracket either side; XL is the default used in preceding single-grid analyses
  - Figure: bar chart of relative D* error at 10²⁴ FLOPs, grouped by grid width across all three surfaces; negative bars = underestimation
  - Collapsible raw data table with full-precision values for all surface/grid combinations
  - Key observations from the figure:
    - Symmetric surfaces are immune (zero error at all grid widths)
    - Asymmetric surfaces always underestimate (predicting fewer tokens than optimal → undertraining)
    - Wider grids amplify error
    - More asymmetry magnifies everything (High Imbalance shows roughly 4–5x larger errors than Chinchilla at each grid width)
  - Key result: highlight a concrete case using the Chinchilla surface with a practical grid width; show the absolute token shortfall at 10²⁴ FLOPs; emphasize these are ideal conditions, real experiments can only do worse

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


