# Article Spec: Problems with Chinchilla Approach 2

> **Editorial Guidelines**
>
> - Format: single self-contained HTML file (see `specs/build.md` for the full build workflow)
> - Length: target a ~20 minute read
> - Audience: ML practitioners familiar with scaling laws but not Approach 2/3 nuances
> - Purpose: demonstrate systematic biases in Chinchilla Approach 2 using noise-free synthetic data
> - Figures: use custom code extractions to generate figures or new data, not direct experiment outputs from other parts of this project
> - Implementation details: output paths, filenames, and other build artifacts should not be specified in this outline; those belong in code or `specs/build.md`
> - Tone: soft, neutral; avoid strong language like "catastrophic", "disastrous", "corrupted" when referring to critiques of Approach 2; target a balanced, informative register
> - Prose: avoid meta-commentary that tells the reader what is important or summarizes what they just read; let the content speak for itself and use callout boxes for key messages
> - Grammar: avoid em dashes; use other grammatical devices instead
> - References (see `specs/build.md` for regeneration steps):
>   - Source of truth: `docs/references/references.yaml`
>   - Inline citation format: `<sup><a href="#ref-KEY">[N]</a></sup>` where KEY and N match the generated references list
>   - In this outline, cite as `[KEY]` (e.g. `[chinchilla]`); these map to keys in the YAML
>   - All inline citations belong in the Motivation section unless explicitly noted otherwise

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
- We propose VPNLS (Variable Projection with Non-negative Least Squares), an alternative fitting method that is simple, stable, and free of these biases while building on the same intuitive computational shortcut: optimizing exponential terms separately from linear terms

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
- Use a concrete asymmetric surface: L(N, D) = 1.69 + 400/N^0.31 + 400/D^0.31
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
  - Define "3× offset": each IsoFLOP grid is centered at 3×D* instead of D*, so the grid midpoint sits at three times the true optimum
  - Figure (2 rows × 2 columns):
    - (0,0): IsoFLOP contours at L (±8×) grid with offset=3× on symmetric surface; black diamonds at (off-center) sampling centers, red × at true D*, blue + at inferred D*
    - (0,1): Extrapolation error bar chart (D* at 10²⁴ FLOPs) by grid width (XS through XL)
    - (1,0): D* exponent error vs grid width (16 points from XS to XL); flat at zero (exponent perfectly preserved)
    - (1,1): D* intercept error vs grid width (16 points from XS to XL); systematic bias that varies with grid width
    - Bottom row y-axes matched to show exponent is zero while intercept has systematic bias
- **Drifting bias**: offset grows with compute budget; corrupts both exponents and intercepts
  - Define "linear drift to 3×": sampling center starts at the true optimum (lowest budget) and drifts to 3× (highest budget), interpolating linearly in log-compute space
  - Figure (2 rows × 2 columns, same layout as constant bias):
    - (0,0): IsoFLOP contours at L (±8×) grid with linear drift on symmetric surface; sampling centers (black diamonds) visibly shift away from true D* (red ×) at higher compute budgets, unlike the constant bias case where the gap is uniform
    - (0,1): Extrapolation error bar chart (D* at 10²⁴ FLOPs) by grid width (XS through XL)
    - (1,0): D* exponent error vs grid width (16 points from XS to XL); now non-zero, unlike the flat-at-zero line in the constant bias figure — this is the key visual contrast
    - (1,1): D* intercept error vs grid width (16 points from XS to XL)
    - Bottom row y-axes matched to show relative magnitude of exponent vs intercept errors
- Key message: constant bias preserves exponents; any compute-dependent bias pattern distorts them; the distinction matters because exponent errors compound during extrapolation while intercept errors remain fixed

---

## Robust Fits — Unbiased Estimation with Linear Separation

- Overview
  - Naive Approach 3 (nonlinear least squares over all five parameters) is unstable
    - TODO: write this part up:
    - Typically, "unstable" means that results are sensitive to initialization, hyperparameters (e.g. Huber loss delta) or lack of optimization convergence
    - The most common approaches are BFGS or L-BFGS (cite (Mis)fitting scaling laws section 6) or other gradient-based SGD optimizers
    - Some studies forgo optimization entirely and use grid search instead due to instability (Goyal et al. (2024))
    - Some studies "opt to use a linear method" by taking the log on both sides, but "it is generally not advised because the log transformation also changes the distribution of error"
      - Some even still use a nonlinear optimizer even with a linear objective (Hashimoto, 2021)
      - It can be easily shown in simulations like those we use that log transformations of loss values lead to universal biases in parameter estimates
      - Similarly, use loss functions for fits other than MSE like MAE or Huber loss also induce biased parameter inference so we use MSE for all further fits
    - Use the above as segueue into variable projection
  - Variable projection exploits the partially linear structure: for fixed (α, β), the loss is linear in (E, A, B)
  - This is the same computational shortcut motivating Approach 2: optimizing exponential terms separately from linear terms; but here it is applied without the parabolic approximation
- Algorithm: search over (α, β) and solve for (E, A, B) analytically at each candidate; a coarse grid search seeds a local optimizer (Nelder-Mead) that refines (α, β) while maintaining the linear separation throughout, never optimizing the full five-parameter space. We call this method **VPNLS** (Variable Projection with Non-negative Least Squares)
- **Why Nelder-Mead over L-BFGS-B?** VPNLS uses NNLS for the inner solve to guarantee non-negative coefficients (E, A, B ≥ 0). This prevents physically meaningless fits but makes the objective non-smooth — the active-set transitions in NNLS create kinks that rule out analytical gradients. Switching to OLS would restore differentiability but cannot enforce non-negativity (the outer L-BFGS-B bounds only constrain α, β, not the inner solve's output). Deriving and implementing the analytical gradient through the normal equations also adds complexity for marginal benefit in a 2D search space. With NNLS, L-BFGS-B must use finite-difference gradients, which creates interacting tuning parameters (`eps`, `jac`, `ftol`, `gtol`) where tight tolerances demand gradient accuracy that finite differences cannot reliably deliver. Nelder-Mead avoids all of this — its few settings (`xatol`, `fatol`, `maxiter`) are independent and work out of the box. Nelder-Mead scales poorly to high dimensions, but variable projection reduces the search to 2D (α, β), which is exactly the regime where it excels
- We compare nine method configurations on noise-free synthetic data across three loss surfaces (symmetric, Chinchilla, high imbalance) and 20 sampling ranges (the best case for gradient methods):
  - **5D direct (Approach 3)**: L-BFGS-B with analytical gradients, finite-difference (forward), and finite-difference (central); no variable projection, optimizes all five parameters jointly
  - **2D variable projection**: VPNLS (Nelder-Mead), L-BFGS-B with four configurations (default eps, central diff, eps=1e-6, eps=1e-10), and a fine 256² grid search
- Even here, L-BFGS-B either sacrifices precision or introduces convergence failures depending on configuration, while VPNLS achieves machine-precision recovery with no tuning
- **Figure: Method Comparison** (1 × 2, shared y-axis; methods sorted by gmean error, worst at top)
  - **Left — dot-range plot**: geometric mean of |relative error| (%) pooled across all surfaces, grid widths, and parameters; horizontal bars span min–max. Filled dot = converged on all 60 fits; open dot = at least one failure (annotated with count)
  - **Right — max-error heatmap**: columns {E, A, B, α, β}, white-to-black log-scale colormap, cell text shows max |relative error| (%) over successful fits only
- **Companion CSVs**: raw per-(method, surface, grid width, parameter) errors, max-error pivot, and failure-count pivot
- Key result: all five loss surface parameters (E, A, B, α, β) recovered with machine precision; extrapolation is exact
- Key message: VPNLS makes direct surface fitting robust — it eliminates the biases from the parabolic approximation and avoids the fragile gradient tuning that makes L-BFGS-B impractical for this problem

---

## IsoFLOP Curves in the Wild — Evidence from Published Studies

- Figure (1 row × 3 columns): IsoFLOP curves from Chinchilla [chinchilla], Llama 3 [llama3], and DeepSeek [deepseek]; image at `results/article/static/isoflop_curve_examples.png`
- These curves exhibit visibly asymmetric shapes (steeper on one side of the minimum than the other), suggesting α ≠ β
- Sampling centers do not always coincide with the curve minima, and the degree of off-centering appears to vary across compute budgets
- This is not a criticism of these studies; these are some of the most careful and influential scaling law analyses published. The point is that the conditions under which Approach 2's biases activate are the norm, not the exception

- **Putting It All Together**: simulate combined asymmetry and sampling biases in a single extrapolation analysis (XS through XL grids, all three surfaces, all five bias configs)
- Figure (TODO: determine presentation/layout): D* extrapolation error across grid widths and surfaces with combined biases; temporary research image at `results/article/static/combined_extrapolation_error.png`
- Show how each bias source dominates at different grid widths; note that the two sources can partially offset or reinforce depending on offset direction
- TODO: add a configuration where bias sources reinforce rather than offset, to demonstrate the compounding case directly
- Key result: multiple bias sources act simultaneously in any real experiment; when they align, combined error exceeds either alone

---

## Conclusion

- **These biases are structural, not statistical**: the errors documented here exist on noise-free data with perfect experimental conditions; real experiments, which contend with measurement noise and unknown optima, can only make them worse
- **Two independent sources compound in practice**: surface asymmetry (α ≠ β) biases intercepts, and off-center sampling biases intercepts or exponents depending on whether the offset is constant or drifting; both act simultaneously in any real experiment
- **A practical alternative exists**: VPNLS recovers all five surface parameters with machine precision, uses the same intuitive linear separation that makes Approach 2 appealing, and is straightforward to implement
- **Takeaway for practitioners**: when using Approach 2, be aware that intercept estimates carry a systematic bias that grows with exponent asymmetry and sampling grid width; when precision matters for extrapolation to large compute budgets, consider VPNLS as a robust alternative
