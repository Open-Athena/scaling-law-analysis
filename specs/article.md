# Article Spec: Problems with Chinchilla Approach 2

> **Editorial Guidelines**
>
> - Format: the article is authored as an HTML file with relative image references and external dependencies (MathJax, Google Fonts); a separate build step inlines local images as base64 data URIs to produce a self-contained standalone HTML for deployment (see `specs/build.md` for the full build workflow)
> - Length: target a ~30 minute read
> - Audience: ML practitioners familiar with scaling laws but not Approach 2/3 nuances
> - Purpose: demonstrate systematic biases in Chinchilla Approach 2 using noise-free synthetic data
> - Figures: use custom code extractions to generate figures or new data, not direct experiment outputs from other parts of this project
> - Implementation details: output paths, filenames, and other build artifacts should not be specified in this outline; those belong in code or `specs/build.md`
> - Tone: soft, neutral; avoid strong language like "catastrophic", "disastrous", "corrupted" when referring to critiques of Approach 2; target a balanced, informative register
> - Prose: avoid meta-commentary that tells the reader what is important or summarizes what they just read; let the content speak for itself and use callout boxes for key messages
> - **Syntax (IMPORTANT):** favor direct, integrated sentences. Do not use em dashes or explanatory colons to append elaborations, asides, or restatements onto a clause. Instead, weave the information into the sentence itself, or use a separate sentence. If you find yourself reaching for "—" or "general statement: specific restatement", restructure.
> - References (see `specs/build.md` for regeneration steps):
>   - Source of truth: `docs/references/references.yaml`
>   - Inline citation format: `<sup><a href="#ref-KEY">[N]</a></sup>` where KEY and N match the generated references list
>   - In this outline, cite as `[KEY]` (e.g. `[chinchilla]`); these map to keys in the YAML

---

## Motivation

- Chinchilla Approach 2 is arguably the most widely adopted method for fitting scaling laws in practice today
- Used by top AI labs including DeepMind [chinchilla] [sovit] (its creators), Meta [llama3] [optibert], DeepSeek [deepseek], Microsoft [ehr_scaling], Amazon [il_scaling], Waymo [waymo_scaling], and Arc Institute [evo], among others
- Also a workhorse method for academic studies [dit_scaling] [dlm_scaling] [biosignal_scaling] and high-profile practitioner tutorials (e.g. Andrej Karpathy)
- Its appeal lies in stability and data efficiency relative to nonlinear optimization over all loss surface parameters; this owes to its reliance on 2nd-order Taylor approximations fit as parabolas and the fact that it estimates only the more actionable scaling exponents rather than the full set of surface parameters
- Many **analytical** extensions have since been formulated that add or modify terms in the original Chinchilla functional form: epochs [data_constrained] [data_filtering_scaling], overfitting [mupt], precision [precision_scaling], MoE sparsity [moe_scaling], data quality [quality_scaling], data mixtures [optimal_data_mixtures] [redundancy_scaling] [data_filtering_scaling], non-embedding parameters [reconciling_scaling], downstream task performance [ai2_task_scaling]; these prescribe explicit functional forms rather than inferring scaling law structure automatically, and build directly on Chinchilla as a foundation
  - We revisit basics here on how to best apply a simple model like Chinchilla with high precision and stability, to validation loss alone, before considering more advanced extensions
  - A fitting method that recovers the base surface with higher precision may offer a stronger starting point for these richer settings
- To our knowledge, the sensitivity of these approximations and the method's behavior on loss surfaces that are less symmetric than the original Chinchilla form (where token and parameter scaling exponents are roughly equal) have not been studied in detail
- We investigate this through noise-free synthetic simulations that isolate systematic biases inherent to the method itself
- We show how these biases impact downstream decisions like dataset size selection for final training runs at large compute budgets
- We show how extrapolation errors trace back to suboptimal isoflop experiment design, and that pathologies in these designs can be observed in real, high-profile scaling law studies even if they are difficult to quantify precisely
- We propose VPNLS (Variable Projection with Non-negative Least Squares), an alternative fitting method that is simple, stable, and free of these biases while building on the same intuitive computational shortcut: optimizing exponential terms separately from linear terms

---

## Preliminaries: Loss Surface, Notation, and Fitting Methods

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

## The Happy Path: Symmetric Surfaces

- Frame as establishing a baseline before examining failure modes
- Use a concrete asymmetric surface: L(N, D) = 1.69 + 400/N^0.31 + 400/D^0.31
- Note that equal exponents (α = β) mean compute splits evenly; true scaling exponents are a = b = 0.5
- Describe the experiment: five IsoFLOP contours from 10^17 to 10^21 FLOPs with 15 model sizes per curve, fit parabolas, extract optimal D*; note that this same configuration (five budgets, 15 points per curve) is used in all simulations throughout the article
- Figure (1 row × 2 columns): IsoFLOP curves with fitted parabolas (left) and power-law fit (right); true (×) and inferred (+) optima indistinguishable
- Table: show perfect recovery of b (D* exponent) and b₀ (D* intercept) with machine-precision relative errors (~10⁻¹⁰ %)
- Key result: on a symmetric surface with perfectly crafted IsoFLOP grid sampling, Approach 2 recovers both exponents and intercepts with machine-precision accuracy; the parabola vertex shift is zero when α = β, so the inferred optima coincide with the true optima
- Close by noting this baseline is precisely correct under ideal conditions that are unrealistic in practice; the following sections perturb these conditions in controlled ways

---

## Asymmetric Surfaces: Intercept and Extrapolation Errors

- Frame as repeating the exact same procedure as the Happy Path; only change is α ≠ β

- **What Happens**
  - Asymmetric surfaces produce systematically wrong intercepts while exponents remain accurate
  - Two test configurations: Chinchilla (α=0.34, β=0.28, ratio ≈ 1.2) and Asymmetric (α=0.465, β=0.155, ratio = 3.0); note that the Asymmetric surface's ratio of 3.0 is comparable to real-world findings (DeepSeek [deepseek] reports allocation exponents a=0.73, b=0.27 on OpenWebText2, implying a loss surface ratio of ~2.7)
  - Figure (2 rows × 2 columns): Approach 2 on both asymmetric surfaces; rows = IsoFLOP curves, power-law fits; columns = Chinchilla, Asymmetric; visible gap between true (dashed) and inferred (solid) power-law lines, especially for the Asymmetric surface
  - Tables for each surface showing b exponent with negligible error but b₀ intercept with meaningful error; error larger for the Asymmetric surface than Chinchilla

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
    - More asymmetry magnifies everything (the Asymmetric surface shows roughly 4–5x larger errors than Chinchilla at each grid width)
  - Key result: highlight a concrete case using the Chinchilla surface with a practical grid width; show the absolute token shortfall at 10²⁴ FLOPs; emphasize these are ideal conditions, real experiments can only do worse

---

## Off-Center Sampling: Exponent and Extrapolation Errors

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

## IsoFLOP Curves in the Wild: Evidence from Published Studies

- Figure (1 row × 3 columns): IsoFLOP curves from Chinchilla [chinchilla], Llama 3 [llama3], and DeepSeek [deepseek]; image at `results/article/static/isoflop_curve_examples.png`
- These curves exhibit visibly asymmetric shapes (steeper on one side of the minimum than the other), suggesting α ≠ β
- Sampling centers do not always coincide with the curve minima, and the degree of off-centering appears to vary across compute budgets
- This is not a criticism of these studies; these are some of the most careful and influential scaling law analyses published. The point is that the conditions under which Approach 2's biases activate are the norm, not the exception

- **Compounding Errors**: simulate combined asymmetry and sampling biases in a single extrapolation analysis using the same 3× drift and 3× center offset from the main-text off-center figures
- Figure (1×2 bar chart grid): one subplot per sampling configuration (offset by 3×, drift to 3×); loss surface on x-axis, bars grouped/colored by grid width (XS through XL); on the symmetric surface, constant offset results correspond to the constant bias figure and drift results correspond to the drifting bias figure
- Collapsible raw data table with full-precision values for all config/surface/grid combinations
- Describe interaction: off-center sampling pushes errors positive, asymmetry pushes negative; net error depends on which dominates; partial cancellation with wider grids is only coincidental
- Argue that 3× perturbations are representative of realistic conditions: IsoFLOP curves they produce are qualitatively similar to published studies; 3× offset is likely within the range of uncertainty practitioners face
- Cross-reference to appendix for detailed view of how errors trend with compute budget across a wider set of drift rates and center offset magnitudes
- Key result: multiple bias sources act simultaneously in any real experiment; when they align, combined error can exceed either one alone, even with the narrowest grid where the parabolic approximation is most accurate

---

## Robust Fits: Unbiased Estimation with Linear Separation

- Opening segue: the previous sections established the biases and showed they arise in practice; now address what to do about them
- **Problems with Direct Surface Fitting**
- Naive Approach 3 (nonlinear least squares over all five parameters) is unstable
  - The following summary of fitting practices and failure modes draws from [misfitting], a survey of over 50 scaling law papers; the problems documented apply to scaling law fitting in general (not just Chinchilla forms), but they are directly relevant because Approach 3 involves the same kind of nonlinear optimization
  - Over half of surveyed papers do not fully specify their fitting procedure (optimizer, loss, initialization), compounding reproducibility issues
  - The most common optimizers for scaling law fits are BFGS or L-BFGS; some studies use SGD-family optimizers (Adam, Adagrad), though these are noted as sometimes ill-suited for curve fitting due to poor data efficiency; at least one study [data_filtering_scaling] forgoes optimization entirely in favor of pure grid search due to instability of fitted solutions
  - "Unstable" in practice means: sensitivity to initialization, sensitivity to optimizer hyperparameters (e.g. convergence tolerance, gradient estimation method), and convergence to local minima rather than the global optimum
  - Initialization is a major source of variability; common mitigations include (a) grid search over initializations, running the optimizer from each of thousands of starting points and keeping the best fit, (b) random sampling of starting points, (c) evaluating a coarse grid without optimization and seeding the optimizer from the best candidate only, or (d) initializing from previously published parameter values
    - These mitigations do not reliably solve the problem; the survey's own experiments show that full-grid optimization over 4500 starting points sometimes yields the worst fit among all strategies tested, evidence of "the difficulty of optimizing over this space, and the presence of many local minima"
  - A simpler alternative is log-linearization: take the log of both sides of the power law and fit with linear regression; however, this changes the error distribution and exaggerates errors at small loss values, biasing parameter estimates in a way that is easily observed in simulations like ours
      - The survey also finds that loss function choice (Log-Huber, Huber, MSE, MAE) affects fitted parameters unpredictably across datasets, and non-MSE objectives can introduce systematic bias in parameter estimates; our goal is to identify a method that is simple, stable, and efficient rather than to address outliers or other statistical concerns, so we use MSE for all fits
  - The survey's experimental analysis varies optimizer, loss function, and initialization strategy across three datasets; the overarching finding is that none of these choices reliably eliminates instability, and results shift unpredictably between datasets
  - Segue: a key contributor to these problems is the high dimensionality of the joint 5-parameter optimization, which creates a complex loss landscape with many local minima and interacting sensitivities; variable projection reduces the nonlinear search to 2 dimensions (α, β), which shrinks the space enough to make simple grid-seeded optimization practical and greatly reduces (though does not eliminate) the risk of converging to poor local minima
  - Concrete example from Experiment 8: Hessian of 5D RSS on the Asymmetric surface (α=0.465, β=0.155, five IsoFLOP contours 10¹⁷–10²¹, 15 points/curve) has eigenvalues spanning ~8×10⁻⁶ to ~3×10⁶ (κ ≈ 3.5×10¹¹); flattest directions are A and B (underdetermined near optimum), steepest are α and β; 2D landscape after variable projection has κ ≈ 11 [hessian_optimization]
- **Variable Projection (VPNLS)**
- Variable projection exploits the partially linear structure: for fixed (α, β), the loss is linear in (E, A, B)
- This is the same computational shortcut motivating Approach 2: optimizing exponential terms separately from linear terms; but here it is applied without the parabolic approximation
- **Algorithm**: search over (α, β) and solve for (E, A, B) analytically at each candidate; a coarse grid search seeds a local optimizer (Nelder-Mead) that refines (α, β) while maintaining the linear separation throughout, never optimizing the full five-parameter space; we call this method VPNLS (Variable Projection with Non-negative Least Squares)
- **Why Nelder-Mead over L-BFGS-B?**
  - VPNLS uses NNLS for the inner solve to guarantee non-negative coefficients (E, A, B ≥ 0); this prevents physically meaningless fits but makes the objective non-smooth, as NNLS has no closed-form gradient with respect to the outer parameters
  - Switching to OLS would restore differentiability but cannot enforce non-negativity (the outer L-BFGS-B bounds only constrain α, β, not the inner solve's output); deriving and implementing the analytical gradient through the normal equations also adds complexity for marginal benefit in a 2D search space
  - With NNLS, L-BFGS-B must use finite-difference gradients, which creates interacting tuning parameters (`eps`, `jac`, `ftol`, `gtol`, `maxcor`, `maxls`) where tight tolerances demand gradient accuracy that finite differences cannot reliably provide
  - Nelder-Mead avoids all of this; its few settings (`xatol`, `fatol`, `maxiter`) are independent and work out of the box; it scales poorly to high dimensions, but variable projection reduces the search to 2D (α, β), which is exactly the regime where it excels
- **Method Comparison (Parameter Recovery)**
  - We compare nine method configurations on noise-free synthetic data across three loss surfaces (symmetric, Chinchilla, Asymmetric) and 20 sampling ranges (the best case for gradient methods):
    - 5D direct (Approach 3): L-BFGS-B with analytical gradients, finite-difference (forward), and finite-difference (central); no variable projection, optimizes all five parameters jointly
    - 2D variable projection: VPNLS (Nelder-Mead), L-BFGS-B with four configurations (default eps, central diff, eps=1e-6, eps=1e-10), and a fine 256² grid search
  - Figure (1 × 2, shared y-axis; methods sorted by gmean error, worst at top): dot-range plot (left) showing geometric mean of |relative error| (%) pooled across all surfaces, grid widths, and parameters, with horizontal bars spanning min to max, filled dots for methods that converged on all 60 fits and open dots for those with at least one failure (annotated with count); max-error heatmap (right) with columns {E, A, B, α, β}, white-to-black log-scale colormap, cell text showing max |relative error| (%) over successful fits only
  - Companion CSVs: raw per-(method, surface, grid width, parameter) errors, max-error pivot, and failure-count pivot
  - Key observations from the figure:
    - High-resolution grid search (256²) is stable across all conditions but provides the poorest overall precision, limited by grid resolution
    - 5D direct optimization (Approach 3) is more accurate on average than grid search but highly variable across conditions; 5D optimization without analytical gradients (finite-difference only) performs very poorly and serves as a negative control, demonstrating what high variability and instability look like for comparison to Approach 3 with analytical gradients, which is similarly variable
    - L-BFGS-B with 2D variable projection can match Nelder-Mead precision, though the optimizer fails to converge in a non-trivial fraction of the relatively small number of scenarios simulated here
    - Central differences are key: switching from forward to 3-point central finite differences closes the precision gap with Nelder-Mead (~1e-8% vs ~1e-5% error), but introduces sporadic line search failures; these failures can be false positives where the optimizer has already reached the true minimum (RSS near machine zero) but the line search cannot verify further progress because function values are too small to distinguish; scipy reports this as `result.success = False` with status `ABNORMAL` (abnormal termination in line search), even though the returned parameters are correct
    - L-BFGS-B is a viable alternative to Nelder-Mead if settings are tuned carefully and the practitioner understands that `result.success = False` from `scipy.optimize.minimize` does not always indicate a bad fit
    - VPNLS with Nelder-Mead is simpler, requires less tuning, and recovers parameter estimates with precision at least as high as any other method tested; it technically achieves the most precise estimates, though the margin over a well-configured L-BFGS-B with 3-point central differences is small
- Key message: VPNLS eliminates the biases inherent in the parabolic approximation and avoids the fragile gradient tuning that complicates L-BFGS-B; all five loss surface parameters (E, A, B, α, β) are recovered with machine precision and extrapolation is exact
- **Method Comparison (Exponent Inference)**
  - TBD

---

## Conclusion

- **These biases are structural, not statistical**: the errors documented here exist on noise-free data with perfect experimental conditions; real experiments, which contend with measurement noise and unknown optima, can only make them worse
- **Two independent sources compound in practice**: surface asymmetry (α ≠ β) biases intercepts, and off-center sampling biases intercepts or exponents depending on whether the offset is constant or drifting; both act simultaneously in any real experiment
- **A practical alternative exists**: VPNLS recovers all five surface parameters with machine precision, uses the same intuitive linear separation that makes Approach 2 appealing, and is straightforward to implement
- VPNLS may also provide a more precise foundation for the analytical extensions discussed in the Motivation; brief callback (not restated in full), noting they retain the partially linear structure and are a natural direction for future work
- **Takeaway for practitioners**: when using Approach 2, be aware that intercept estimates carry a systematic bias that grows with exponent asymmetry and sampling grid width; when precision matters for extrapolation to large compute budgets, consider VPNLS as a robust alternative

### Limitations

- Bullet list with bold labels per item
- **Irreducible loss dominance at large scale**: at sufficiently large compute budgets the Chinchilla surface reaches E asymptotically, making extrapolations irrelevant and all training configurations equally effective; study assumes practitioners are still in a regime where scaling law extrapolations inform model quality
- **No quantification of downstream cost**: no connection from token extrapolation error → under/over-training → model performance → cost in FLOPs/$; justified because alternatives to Approach 2 follow from theory and simulation and are easy to implement at no extra computational cost
- **Assumed correctness of the Chinchilla loss surface**: evidence supports the model [chinchilla_robustness] but alternatives exist, including the Kaplan loss model [kaplan_scaling], refined analytical surfaces like Farseer [farseer] and MuPT [mupt], and agent-discovered functional forms [sld_agent]
- **Qualitative characterization of published study errors**: likely errors in published studies are not quantified; the qualitative characterization is compelling but difficult to quantify because real pathologies don't follow the convenient theoretical model used in simulations
- TODO: add limitation for ignoring scaling laws about downstream evals

---

## Appendix

### A. Detailed Method Comparison

- Full per-parameter, per-surface, per-sampling-range error breakdown from Experiment 5's method comparison (see `specs/experiments.md`, Experiment 5 > Visualization > item 3)
- Figure (3 rows × 5 columns): rows = loss surfaces, columns = parameters (E, A, B, α, β); each panel shows absolute relative error vs sampling range for all nine method configurations; baseline (no bias) only

### B. Combined Extrapolation Error by Compute Budget

- Detailed view of D* extrapolation error as a function of compute budget, from Experiment 4
- Figure (3 rows × 3 columns): rows = sampling ranges (narrow ±2×, medium ±16×, wide ±100×), columns = loss surfaces (symmetric, Chinchilla, Asymmetric); each panel shows relative D* error vs extrapolation compute budget (10²²–10²⁵ FLOPs) with one curve per bias configuration (baseline, two drift rates, two constant offsets)
- Shows how drift-based biases produce errors that grow with extrapolation distance while surface asymmetry and constant offsets produce flat or slowly varying errors; also reveals how these patterns change across sampling ranges and bias magnitudes
