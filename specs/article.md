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

- Approach 2 is arguably the most widely adopted method for fitting scaling laws today
- Used by top AI labs including DeepMind [chinchilla] [sovit], Meta [llama3] [optibert] [beyond_language_modeling], DeepSeek [deepseek], Microsoft [ehr_scaling], Amazon [il_scaling], Waymo [waymo_scaling], and Arc Institute [evo], among others
- Also a workhorse for academic studies [dit_scaling] [dlm_scaling] [biosignal_scaling] and practitioner tutorials (e.g. Andrej Karpathy)
- Its appeal lies in stability and data efficiency relative to nonlinear optimization, owing to parabolic (2nd-order Taylor) approximations that estimate only the scaling exponents rather than the full surface
- Many analytical extensions build on the Chinchilla functional form: data repetition [data_constrained], overfitting [mupt], precision [precision_scaling], optimizers [optimizer_scaling], MoE sparsity [moe_memory_scaling], pruning [pruning_scaling], data quality [quality_scaling], data mixtures [optimal_data_mixtures] [atlas_multilingual], model shape/context length [icr_scaling], non-embedding parameters [reconciling_scaling], downstream task performance [ai2_task_scaling]
  - Similar studies extend individual terms in isolation [redundancy_scaling] [data_filtering_scaling] [moe_scaling] [subgroup_scaling] or propose modified Kaplan scaling laws [kaplan_scaling]
  - A fitting method that recovers the base surface with higher precision may offer a stronger starting point for these richer settings
- The sensitivity of the parabolic approximation on asymmetric loss surfaces (where α ≠ β) has not been studied in detail
- Four modes of investigation: (1) noise-free synthetic simulations, (2) closed-form error expressions, (3) noisy simulations with a validated noise model, (4) empirical fits to Llama 3 IsoFLOP data
- We propose VPNLS (Variable Projection with Non-negative Least Squares) as an alternative that builds on the same computational shortcut (optimizing exponential terms separately from linear terms) without the parabolic approximation

---

## Preliminaries: Loss Surface, Notation, and Fitting Methods

- Chinchilla loss surface: L(N, D) = E + A/N^α + B/D^β
- Compute-optimal allocation: N* ∝ C^a where a = β/(α+β), D* ∝ C^b where b = α/(α+β)
- Compute constraint: C = 6ND
- **Approach 2: IsoFLOP Parabolic Fitting**
  - Along a fixed-compute contour, loss as a function of log N is approximately parabolic near the optimum
  - Three-step pipeline: (1) sample IsoFLOP contours, (2) fit parabolas and extract vertex N* per budget, (3) regress log N* on log C to recover scaling exponent
  - Only polynomial fits, no nonlinear optimization
- **Approach 3: Direct Surface Fitting**
  - Minimize Σ(L_i − L̂(N_i, D_i))² over all five parameters (RSS objective)
  - Practical issues: E, A, B must remain positive; loss scale spans orders of magnitude
  - Chinchilla's adapted formulation [chinchilla]: LSE reparameterization enforces positivity via E=exp(e), A=exp(a), B=exp(b); predicted log-loss via logsumexp for numerical stability; Huber loss on log-predictions
  - Both objectives shown side by side; we use MSE rather than Huber throughout for MLE properties
  - Avoids the parabolic approximation but is notoriously unstable

---

## Error Costs: Misallocation at Scale

- Overview
  - Compute-optimal scaling laws are most directly relevant to the largest frontier models, which are still frequently trained near compute-optimality
  - Assess the cost of Approach 2 misallocations at the compute scale of Llama 3 405B (3.8×10²⁵ FLOPs) [llama3], one of the most compute-intensive open models as of early 2026 [epochai_open_model_compute]
  - Also examine Chinchilla and two multimodal models with greater asymmetry: SODA [audio_scaling] and Sparse-NMM [nmm_scaling]
  - Measure misallocation as Deadweight Compute Loss (DCL), the FLOPs needed to reach an optimal allocation from one provided by Approach 2
- Methods
  - For Llama 3: extract raw IsoFLOP data from Figure 2 of [llama3] via SVG coordinate extraction [epochai_chinchilla_replication]; fit Approach 2 and several Approach 3 variants; extrapolate to target compute; convert FLOPs to dollars assuming 50% MFU [beyond_chinchilla] and $2/H100 hour [olmo3]
  - For Chinchilla, SODA, Sparse-NMM: simulate IsoFLOP data from published Approach 3 fit statistics; fit with Approach 2; compare extrapolations
  - Llama 3 included in both approaches to measure how IsoFLOP experiment design degrades Approach 2 accuracy beyond the systematic biases
- Figure (1×2): horizontal bar chart of DCL (left); heatmap of token/param counts, loss differences, dollar costs (right); simulated vs real results separated by dashed line
- Results
  - Empirical Llama 3: DCL of 6–10% of budget ($1.3–2.2M) depending on Approach 3 configuration
  - Simulated Llama 3 (near-symmetric, b/a ≈ 0.97): only 0.2% DCL ($39K), implying ~$1–2M of empirical cost is attributable to Approach 2's poor fit to real IsoFLOP data rather than surface asymmetry alone
  - Asymmetric surfaces amplify errors: SODA (b/a = 1.56) reaches 8% DCL ($1.7M); Sparse-NMM (b/a = 1.91) reaches 10% DCL ($2.1M) under the same 3× drift bias
  - Multimodal models with more asymmetric surfaces face potentially much larger misallocations than text-only LLMs

---

## Symmetric Surfaces: Unbiased Estimation in Ideal Conditions

- Establishes a baseline before examining failure modes
- Symmetric surface: L(N, D) = 1.69 + 400/N^0.31 + 400/D^0.31; equal exponents mean a = b = 0.5
- Experiment: five IsoFLOP contours from 10¹⁷ to 10²¹ FLOPs, 15 points per curve (this configuration is used in all simulations throughout the article)
- Figure (1×2): IsoFLOP curves with fitted parabolas (left); power-law fit (right); true and inferred optima indistinguishable
- Table: machine-precision recovery of b (exponent) and b₀ (intercept), relative errors ~10⁻¹⁰ %
- Key result: the parabola vertex shift is zero when α = β; Approach 2 is exactly correct under these ideal conditions
- These conditions are unrealistic in practice; the following sections perturb them in controlled ways

---

## Asymmetric Surfaces: Intercept and Extrapolation Errors

- Same procedure as the symmetric baseline; only change is α ≠ β
- Two test surfaces: Chinchilla (α=0.34, β=0.28, ratio ≈ 1.2) and Asymmetric (α=0.465, β=0.155, ratio = 3.0, comparable to DeepSeek's reported allocation exponents [deepseek])
- Figure (2×2): Approach 2 on both surfaces; rows = IsoFLOP curves, power-law fits; columns = Chinchilla, Asymmetric; visible gap between true and inferred power-law lines
- Tables: b exponent has negligible error; b₀ intercept has meaningful error, larger for the Asymmetric surface
- Even under ideal conditions (no noise, centered sampling, standard parameters), Approach 2 produces biased intercepts

### Underlying Causes

- IsoFLOP curve is not a true parabola; higher-order Taylor terms shift the vertex when α ≠ β
- Vertex shift is constant across compute budgets, so it biases every N* by the same multiplicative factor
  - Slope (exponent) is unchanged; intercept absorbs the entire error
- Closed-form expression: intercept error = 10^(δw) − 1, where δw depends only on (α, β) and sampling grid; δw = 0 when α = β, grows with |α − β| and grid width
- Taylor expansion intuition: odd-order terms cancel for symmetric surfaces but not for asymmetric ones

### Error Implications

- Quantify extrapolation error via compute-optimal token prediction across four grid widths (XS ±2× through XL ±16×); real experiments typically span S to L range
- Figure (1×1): bar chart of relative D* error at 10²⁴ FLOPs, grouped by grid width across all three surfaces
- Collapsible raw data table
- Key observations: symmetric surfaces unaffected; asymmetric surfaces always underestimate; wider grids and more asymmetry amplify error

---

## Off-Center Sampling: Exponent and Extrapolation Errors

- In practice, sampling centers are guesses; this is about where you place the grid, distinct from asymmetry errors
- Studied on symmetric surfaces only (α = β) to isolate the effect

### Constant Multiplicative Bias

- Same offset factor at every compute budget (e.g. 3× offset: grid centered at 3×D* instead of D*)
- Corrupts intercepts only; exponents are perfectly preserved (same mechanism as asymmetry errors)
- Figure (2×2): IsoFLOP contours with off-center sampling (top-left); extrapolation error bar chart (top-right); exponent error vs grid width, flat at zero (bottom-left); intercept error vs grid width (bottom-right)

### Drifting Bias

- Offset grows with compute budget (e.g. linear drift from 1× to 3× across budgets)
- Corrupts both exponents and intercepts; the compute-dependent perturbation distorts the power-law slope
- Figure (2×2, same layout): key visual contrast is that exponent error is now non-zero
- Key message: constant bias preserves exponents; compute-dependent bias distorts them; exponent errors compound during extrapolation while intercept errors remain fixed

---

## Real IsoFLOP Curves: Evidence from Published Studies

- IsoFLOP curves from Chinchilla [chinchilla], Llama 3 [llama3], and DeepSeek [deepseek]
- Visibly asymmetric shapes and off-center sampling are common in these high-profile studies
- The conditions under which Approach 2's biases activate are the norm, not the exception
- Six published experiments processed through the IsoFLOP quality control pipeline

### Compounding Errors

- Simulate combined asymmetry and off-center sampling biases (3× drift, 3× offset) in a single extrapolation analysis
- Figure (1×2): one subplot per bias configuration; bars grouped by grid width across surfaces
- Collapsible raw data table
- Off-center sampling pushes errors positive, asymmetry pushes negative; partial cancellation is coincidental
- 3× perturbations are representative of realistic conditions based on comparison with published IsoFLOP curves
- Key result: multiple bias sources act simultaneously; combined error can exceed either source alone

---

## Robust Fits: Unbiased Estimation with Linear Separation

### Problems with Direct Surface Fitting

- Summary of fitting challenges from [misfitting], a survey of 50+ scaling law papers
  - Over half do not fully specify their fitting procedure
  - Most common optimizers are BFGS/L-BFGS; SGD-family noted as sometimes ill-suited for curve fitting
  - "Unstable" means: sensitivity to initialization, sensitivity to optimizer hyperparameters, convergence to local minima
  - Common mitigations (grid search over starting points, random init, seeding from published values) do not reliably solve the problem
  - Loss function choice (Log-Huber, Huber, MSE, MAE) affects parameters unpredictably across datasets
- Ill-conditioning example: Hessian of 5D RSS on the Asymmetric surface has κ ≈ 3.5×10¹¹; flattest directions are A and B (underdetermined near optimum); 2D landscape after variable projection has κ ≈ 11

### Variable Projection (VPNLS)

- For fixed (α, β), the loss is linear in (E, A, B); this is the same shortcut motivating Approach 2 but applied without the parabolic approximation
- Algorithm: search over (α, β), solve for (E, A, B) via least squares at each candidate; coarse grid seeds a local optimizer; never optimizes the full 5D space
- Grid search scalability: a 32² grid provides 1,024 candidates with fine 2D resolution vs 4⁵ = 1,024 points spread thinly in 5D; extensions adding linear terms enlarge the inner solve but not the outer grid
- Analytical gradients via the envelope theorem: switching from NNLS to OLS makes the objective differentiable with closed-form gradients
- Optimizer choice: L-BFGS-B (analytical gradients, OLS) and Nelder-Mead (gradient-free, NNLS) both achieve machine-precision recovery in the 2D search space

### Method Comparison (Parameter Recovery)

- Six configurations on noise-free data across three surfaces and 20 sampling ranges:
  - 5D direct (Approach 3): L-BFGS-B with analytical and numerical gradients; grid-seeded from 4⁵ = 1,024 points
  - 2D variable projection: L-BFGS-B with analytical gradients, L-BFGS-B with numerical gradients, Nelder-Mead, and 256² grid search; grid-seeded from 32² = 1,024 points
- Figure (1×2): dot-range plot (left) with method callouts; max-error heatmap (right) with columns {E, A, B, α, β}
- All 2D methods with local optimization recover parameters to machine precision (~1e-7%); 5D methods exhibit larger errors, especially with numerical gradients
- Dominant pattern is the gap between 2D variable projection (all variants) and 5D direct optimization

### Method Comparison (Exponent Inference)

- Extends to a statistical setting: Gaussian noise, varying noise levels, budget counts, and points per curve; thousands of fits per method
- Focus shifts to scaling exponents (a, b) for direct comparison with Approach 2; emphasis on worst-case errors
- Five methods: Approach 2; Naive Approach 3 (random init, MLE); MLE Approach 3 (grid init, MLE); canonical Approach 3 (grid init, LSE + log-loss); VPNLS
- Figure (1×2): dot-range plot with KDE (left); max-error heatmap (right)
- Results:
  - Approach 2 has consistently poor accuracy from structural bias
  - Naive Approach 3 is worse, confirming that uninitialized 5D optimization is unreliable
  - Canonical Approach 3 with grid init is a large improvement; LSE and grid init are the critical ingredients
  - VPNLS is roughly equivalent in typical accuracy to well-configured Approach 3 with the smallest max errors

### Method Comparison (Data Efficiency)

- Symmetric surface with centered sampling (no structural bias); compares estimator variance
- Approach 2's variance is ~8× higher than Approach 3 / VPNLS even under ideal conditions
- Figure: bar chart of pooled variance + heatmap by noise level

---

## Conclusion

- **Approach 2 biases are structural, not statistical**: errors exist on noise-free data with perfect experimental conditions and persist under realistic noise levels with varying amounts of data
- **Three sources of error compound in practice**: IsoFLOP sampling grid width, uncentered IsoFLOP sampling, and loss surface asymmetry all bias inference and extrapolations in different ways; published IsoFLOP curves show clear signs of both asymmetry and off-center sampling
- **Error costs at frontier scale**: at frontier compute scales, these biases translate to a potential 6.5% decrease in training FLOPs ($1.4M) on Llama 3 data and potentially more on multimodal surfaces with greater asymmetry
- **Well-configured Approach 3 works well**: grid initialization and LSE reparameterization (as specified in the original Chinchilla paper [chinchilla]) achieve typical accuracy comparable to VPNLS; a recent survey [misfitting] suggests these details may be omitted or not reported in some studies
- **VPNLS is at least as stable and accurate**: variable projection separates exponential from linear terms, reducing the nonlinear search to the exponential terms only; dense grid search is practical because the exponents occupy tight ranges (typically 0 to 1) unlike linear coefficients which span orders of magnitude
- **VPNLS scales naturally to extensions**: analytical extensions (epochs, data quality, MoE sparsity, etc.) often add linear terms that could be omitted from direct optimization; a simplified reference implementation is possible in ~70 lines of JavaScript with no dependencies
- **Takeaway for practitioners**: be aware of systematic bias with Approach 2 that grows with asymmetry, sampling offsets, and grid width; ensure grid init and LSE reparameterization when using Approach 3; VPNLS offers equivalent accuracy with simpler optimization that scales to richer formulations

### Limitations

- **Irreducible loss dominance at large scale**: at sufficiently large compute budgets, extrapolations become irrelevant as the surface approaches E asymptotically; assumes practitioners are in a regime where scaling law extrapolations still inform model quality
- **Assumed correctness of the Chinchilla loss surface**: evidence supports the model [chinchilla_robustness] but alternatives exist including Kaplan [kaplan_scaling], Farseer [farseer], MuPT [mupt], and agent-discovered forms [sld_agent]
- **Qualitative characterization of published study errors**: likely errors are not quantified; real pathologies don't follow the convenient theoretical model used in simulations

---

## Appendix

### A. VPNLS Implementation Validation

- Validates VPNLS against Apple's ml-scalefit [optimal_data_mixtures] on the Chinchilla dataset (217 points, C < 10²¹)
- Compares fitted parameters across VPNLS, Approach 3 variants (LSE with and without log-loss), and ml-scalefit configurations (MSE, Huber)
- MSE configuration matches VPNLS to three decimal places; divergence only under log-scaled loss

### B. IsoFLOP Quality Control Pipeline

- 8-step QC pipeline applied to published IsoFLOP data: deduplication, minimum curve size, curvature checks, spline-based outlier detection, progressive filtering
- Before/after visualizations for six published experiments

### C. IsoFLOP Samples with Noise

- Figure (2 rows × n columns): rows = L vs N and L vs D; columns = noise levels
- Shows noisy scatter points with noiseless reference curves, true optima, and drifting sampling centers

### D. Detailed Method Comparison

- Full per-parameter, per-surface, per-sampling-range error breakdown from parameter recovery
- Figure (3 rows × 5 columns): rows = surfaces, columns = parameters; absolute relative error vs sampling range for all six configurations

### E. Combined Extrapolation Error by Compute Budget

- D* extrapolation error as a function of compute budget across sampling ranges, surfaces, and bias configurations
- Figure (3×3): rows = sampling ranges, columns = surfaces; one curve per bias configuration
- Shows drift-based biases growing with extrapolation distance vs flat/slowly varying asymmetry errors

### F. Exponent Inference Error Breakdown

- Boxplots broken down by noise level, budget count, and points per curve
- Shows 4 of 5 methods (excludes Naive Approach 3)
- Figure (n_budgets rows × 4 columns): columns = noise level × exponent; per-method boxplots at each points-per-curve setting

### G. Data Efficiency Error Breakdown

- Per-noise-level signed error distributions for the data efficiency comparison

### H. Published Scaling Exponents

- Table of scaling exponents from published studies spanning language models, multimodal models, code, scientific domains, and other modalities
- Provides context for the range of asymmetry ratios observed in practice

### I. Progressive Filtering for Chinchilla

- Effect of progressively applying QC filters to the original Chinchilla IsoFLOP data
- Shows how each step changes the fitted surface parameters

### J. Residual Distributions by Budget

- Residual distributions from real IsoFLOP experiments grouped by compute budget
- Assesses whether residual patterns vary systematically across budgets

### K. Residual Variance Summary

- Summary statistics of residual variance across experiments and budgets
- Supports the uniform noise model used in simulations
