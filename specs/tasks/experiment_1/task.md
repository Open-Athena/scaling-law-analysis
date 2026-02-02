# Task: Experiment 1 - Empirical Error Analysis

## Intent
Demonstrate that the accuracy of Chinchilla Approach 2 is dependent on the accuracy of the second-order Taylor expansion underlying the validity of parabolic fits. This will be shown by analyzing how error in the recovered scaling exponents changes with sampling range.

## Requirements
- Implement the Chinchilla loss function: L(N, D) = E + A/N^α + B/D^β
- Implement IsoFLOP sampling: generate (N, D) pairs along constant compute contours where C = 6ND
- Implement Chinchilla Approach 2:
  - Stage 1: Fit parabolas to log-loss vs log-N curves for each compute budget to find N*
  - Stage 2: Fit power laws N* ∝ C^a and D* ∝ C^b to recover exponents
- Analyze exponent recovery across varying sampling ranges
- Produce visualization showing:
  - IsoFLOP curves with parabola fits and true vs inferred minima (3 sampling ranges)
  - Error in recovered a/b vs sampling range

## Success Criteria
- Recovered a and b should match ground truth when sampling range is small
- Error should demonstrably increase with larger sampling ranges
- Output figure clearly shows the relationship between discretization and inference error

## Notes
- Use no statistical noise in sampled data (pure deterministic loss surface)
- Approach 2 recovers a = β/(α+β) and b = α/(α+β), not α and β directly
- Ground truth: α=0.34, β=0.28 → a=0.4516, b=0.5484

## Results

**Date**: 2026-02-02

### Approach 2 Output

Approach 2 fits power laws:
- N* ∝ C^a where **a = β/(α+β)** = 0.4516
- D* ∝ C^b where **b = α/(α+β)** = 0.5484

Note: a + b = 1 by construction.

### Findings

| Sampling Range | a recovered | b recovered | a error | b error |
|----------------|-------------|-------------|---------|---------|
| ±1.1x (narrow) | 0.4516      | 0.5484      | +0.00%  | -0.00%  |
| ±12x (medium)  | 0.4517      | 0.5483      | +0.03%  | -0.02%  |
| ±100x (wide)   | 0.4527      | 0.5473      | +0.24%  | -0.20%  |

### Interpretation

The parabolic approximation inherent in Approach 2 is valid only in the local neighborhood around the optimum. As the sampling range expands, points farther from the minimum deviate from the second-order Taylor approximation, introducing systematic bias.

The error magnitude is small (<0.3% even at ±100x range), indicating Approach 2 is robust in the noise-free case.

### Output

Figure: `experiments/outputs/experiment_1.png`
