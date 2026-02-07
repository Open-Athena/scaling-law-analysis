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
> - References:
>   - Source YAML: `docs/references/references.yaml`; generated HTML: `results/article/references/references.html` (run `uv run python -m scaling_law_analysis.references` to regenerate)
>   - Inline citations: `<sup><a href="#ref-KEY">[N]</a></sup>` where KEY and N match the generated references list
>   - To include in the article: copy the contents of `references.html` into the `<article>` as the last `<section>` before the closing `</article>` tag
>   - In the outline, cite references as `[KEY]` (e.g. `[chinchilla]`); these map to keys in the YAML
>   - All inline citations belong in the Motivation section unless explicitly noted otherwise; that section introduces every source referenced later in the article

---

## Motivation

- Chinchilla Approach 2 is arguably the most widely adopted method for fitting scaling laws in practice today
- Used by top AI labs including DeepMind [chinchilla] [sovit] (its creators), Meta [llama3] [optibert], DeepSeek [deepseek], Microsoft [ehr_scaling], Amazon [il_scaling], Waymo [waymo_scaling], and Arc Institute [evo], among others
  - TODO: Continue to pad this list out later
- Also a workhorse method for academic studies [dit_scaling] [dlm_scaling] [biosignal_scaling] and high-profile practitioner tutorials (e.g. Andrej Karpathy)
- Its appeal lies in stability and data efficiency relative to nonlinear optimization over all loss surface parameters; this owes to its reliance on 2nd-order Taylor approximations fit as parabolas and the fact that it estimates only the more actionable scaling exponents rather than the full set of surface parameters
- To our knowledge, the sensitivity of these approximations and the method's behavior on loss surfaces that are less symmetric than the original Chinchilla form (where token and parameter scaling exponents are roughly equal) have not been studied in detail
- We investigate this through noise-free synthetic simulations that isolate systematic biases inherent to the method itself
- We show how these biases impact downstream decisions like dataset size selection for final training runs at large compute budgets
- We show how extrapolation errors trace back to suboptimal isoflop experiment design, and that pathologies in these designs can be observed in real, high-profile scaling law studies even if they are difficult to quantify precisely
- We propose an alternative fitting method that is simple, stable, and free of these biases while building on the same intuitive computational shortcut: optimizing exponential terms separately from linear terms

---

## Preliminaries — Loss Surface, Notation, and Fitting Methods

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
- Figure (1 row × 2 columns): IsoFLOP curves with fitted parabolas (left) and power-law fit (right); true (×) and inferred (+) optima indistinguishable
- Table: show perfect recovery of b (D* exponent) and b₀ (D* intercept) with machine-precision relative errors (~10⁻¹⁰ %)
- Key result: on a symmetric surface with perfectly crafted IsoFLOP grid sampling, Approach 2 recovers both exponents and intercepts with machine-precision accuracy; the parabolic approximation is exact when α = β
- Close by noting this baseline is precisely correct under ideal conditions that are unrealistic in practice; the following sections perturb these conditions in controlled ways

---

## Asymmetric Surfaces — Intercept and Extrapolation Errors

- Frame as repeating the exact same procedure as the Happy Path; only change is α ≠ β

- **What Happens**
  - Asymmetric surfaces produce systematically wrong intercepts while exponents remain accurate
  - Two test configurations: Chinchilla (α=0.34, β=0.28, ratio ≈ 1.2) and High Imbalance (α=0.465, β=0.155, ratio = 3.0)
  - Figure (2 rows × 2 columns): Approach 2 on both asymmetric surfaces; rows = IsoFLOP curves, power-law fits; columns = Chinchilla, High Imbalance; visible gap between true (dashed) and inferred (solid) power-law lines, especially for High Imbalance
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
  - Figure (1 row × 1 column): bar chart of relative D* error at 10²⁴ FLOPs, grouped by grid width across all three surfaces; negative bars = underestimation
  - Collapsible raw data table with full-precision values for all surface/grid combinations
  - Key observations from the figure:
    - Symmetric surfaces are unaffected (zero error at all grid widths)
    - Asymmetric surfaces always underestimate (predicting fewer tokens than optimal → undertraining)
    - Wider grids amplify error
    - More asymmetry magnifies everything (High Imbalance shows roughly 4–5x larger errors than Chinchilla at each grid width)
  - Key result: highlight a concrete case using the Chinchilla surface with a practical grid width; show the absolute token shortfall at 10²⁴ FLOPs; emphasize these are ideal conditions, real experiments can only do worse

---

## Off-Center Sampling — Exponent and Extrapolation Errors

- In practice you don't know N* before running the experiment; sampling centers are guesses based on prior estimates or heuristics
- Distinct from asymmetry errors: this is about where you place the grid, not the shape of the surface
- Study on symmetric surfaces only (α = β) to isolate the effect from asymmetry bias
- **Constant multiplicative bias**: same factor at every compute budget; corrupts intercepts only (same mechanism as asymmetry errors)
  - Define "2× offset": each IsoFLOP grid is centered at 2×D* instead of D*, so the grid midpoint sits at twice the true optimum
  - Figure (1 row × 3 columns): IsoFLOP contours at L (±8×) grid with offset=2× on symmetric surface; D* exponent error and D* intercept error vs grid width (16 points from XS to XL), y-axes matched to show exponent is zero while intercept has systematic bias
- **Drifting bias**: offset grows with compute budget; corrupts both exponents and intercepts
- Key message: constant bias preserves exponents; any compute-dependent bias pattern distorts them

---

## Robust Surface Fitting via Variable Projection

- Naive Approach 3 (nonlinear least squares over all five parameters) is unstable
- Variable projection exploits the partially linear structure: for fixed (α, β), the loss is linear in (E, A, B)
- Algorithm: grid search over (α, β), solve the linear system at each grid point, select best fit, optionally refine with a local optimizer
- Result: all five parameters recovered perfectly; extrapolation is exact
- Key message: variable projection makes direct surface fitting robust and eliminates the biases introduced by the parabolic approximation
