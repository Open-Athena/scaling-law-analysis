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

2. For each sampling range (narrow to wide):
   - Sample IsoFLOP contours at 5 compute budgets: 10^17 to 10^21 FLOPs
   - Optionally apply sampling center bias via:
     - drift_rate: linear drift from optimal (left at low compute, right at high compute)
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

3. **Error analysis panel**: Plot relative error in recovered exponents (a and b) as a function of sampling range. This should reveal whether error grows systematically with wider sampling.


### Experiment 2: Empirical Sensitivity

Hypothesis: It is possible to analytically model the error in the inferred exponents via Approach 2 as a function of compute budget and grid resolution

Steps:
- TODO: complete

### Experiment 3: Parametric fits

Hypothesis: It is possible to fit scaling laws parametrically with variable projection and grid search (over alpha/beta) in a manner that is both more stable and more accurate than Chinchilla Approach 3

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
