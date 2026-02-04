# Project Intent: Scaling Law Analysis

## Purpose

The purpose of this project is to:

- Demonstrate whether or not flaws in Chinchilla Approach 2 for scaling law exponent inference exist
- Evaluate alternative methods for scaling law exponent inference
- Produce balanced, concise, empirical and theoretical considerations for best practices in scaling law inference

## Experiments

### Experiment 1: Empirical Error

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


### Experiment 2: Exponent Imbalance Sensitivity

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


### Experiment 3: Sampling Drift Sensitivity

**Hypothesis**: The accuracy of Chinchilla Approach 2 is sensitive to systematic biases in the sampling center.

**Method**:

1. Generate synthetic loss data across three loss surface configurations:
   - **Symmetric**: α=β=0.31, A=B=400, E=1.69 (balanced)
   - **Chinchilla**: α=0.34, β=0.28, A=406.4, B=410.7, E=1.69 (paper defaults)
   - **High imbalance**: α/β ratio = 3.0 (keeping α+β=0.62)

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


### Experiment 4: Extrapolation Error

**Hypothesis**: The accuracy of scaling law exponent inference degrades when extrapolating to compute budgets beyond those used for fitting. The magnitude of this error depends on the loss surface geometry.

**Method**:

1. Use the same three loss surface configurations as Experiment 3 (symmetric, chinchilla, high_imbalance).

2. Use the same five sampling bias configurations as Experiment 3 (baseline, drift_0.2, drift_0.4, scale_1.5, scale_2.0).

3. Use the same three sampling ranges as Experiment 1 (narrow, medium, wide).

4. For each combination of loss surface, sampling bias, and sampling range:
   - Fit scaling law exponents using existing compute budgets (10^17 to 10^21 FLOPs)
   - Extrapolate to higher compute budgets (10^22 to 10^25 FLOPs)
   - Compare inferred vs true optimal token counts D* at each extrapolation budget

**Visualization**:

Produce a single figure organized as a grid with one row per sampling range (3 rows: narrow, medium, wide) and one column per loss surface (3 columns: symmetric, chinchilla, high_imbalance). Each panel shows the relative error in inferred D* as a function of extrapolation compute budget, with one curve per sampling bias configuration. This reveals how extrapolation error depends on the sampling range, loss surface geometry, and sampling biases.

### Experiment 5: Parametric Surface Fitting

**Hypothesis**: Variable projection with grid search (over α/β) provides stable and accurate scaling law parameter recovery.

**Method**:

1. Generate synthetic loss data using the same procedure as Experiments 3 and 4.

2. Implement a new `fit_surface` function in `chinchilla.py` that fits all 5 parameters simultaneously:
   - Input: Arrays of N, D, and L values (token counts, parameter counts, and loss) pooled across all compute budgets
   - Use **variable projection** (also known as separable least squares):
     - For fixed (α, β), the loss function L = E + A·N^(-α) + B·D^(-β) is linear in (E, A, B)
     - Perform a fine 2D grid search over (α, β) space
     - At each grid point, solve for (E, A, B) using non-negative least squares (NNLS) to enforce physical constraints
     - Select the (α, β) grid point that minimizes the total residual sum of squares
   - Grid search parameters:
     - α range: [0.05, 0.95] with 256 points
     - β range: [0.05, 0.95] with 256 points
   - Diagnostic checks (raise errors if violated):
     - Any of E, A, B are at or near zero (hitting NNLS constraint boundary)
     - Best (α, β) is at the edge of the grid search range
   - Return a result object containing all 5 fitted parameters (E, A, B, α, β)

3. For each configuration (loss surface × sampling bias × sampling range):
   - Pool the sampled (N, D, L) data across all compute budgets
   - Fit the surface using `fit_surface`
   - Compute relative errors for all 5 parameters compared to ground truth

**Visualization**:

Produce a single figure organized as a grid with one row per loss surface (3 rows) and 5 columns (one per parameter: E, A, B, α, β). Each panel shows relative error vs sampling range, with one curve per sampling bias configuration. This reveals:
- How parameter recovery accuracy depends on sampling range
- Which parameters are most sensitive to sampling biases
- How loss surface geometry affects parameter recovery

The figure should follow the same style as the parameter errors figure from Experiment 3.


### Experiment 6: Analytical Error

Hypothesis: It is possible to analytically model the error in the inferred exponents via Approach 2 as a function of compute budget and grid resolution

Steps:
- TODO: complete

## Background

- What is Chinchilla Approach 2?
  - This approach defines parameter and token count grids along IsoFLOP contours
  - Loss values from training configurations in that grid are then fit with parabolas (one per compute budget)
  - Minimal values from the inferred parabolas are then fit with a linear model to recover scaling exponents
  - No optimizer is used, only analytical polynomial fits
- What is Chinchilla Approach 3?
  - This approach fits the non-linear loss function directly given empirical data that is not necessarily sampled along IsoFLOP contours
  - It is known to be very sensitive to initialization and often unstable

## Core Objectives
- Simulate loss curves based on parameters (alpha, beta, etc.).
- Implement parameter recovery methods (e.g., Chinchilla Approach 2).
- Analyze the impact of compute budgets and sampling strategies on parameter estimation.
- Develop analytical models to quantify differences between empirical fits and theoretical scaling laws.
