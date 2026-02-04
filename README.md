# Scaling Law Analysis

Simulating systematic biases in neural scaling law fitting procedures, with focus on Chinchilla Approach 2.

## Overview

This project uses synthetic loss surfaces to study how sampling choices and experimental design affect scaling exponent recovery. By controlling ground truth parameters, we isolate sources of bias in the IsoFLOP parabolic fitting method (Approach 2) without confounding from training noise.

## Project Structure

- `specs/project.md` — Single source of truth for experimental design, hypotheses, and methodology
- `src/scaling_law_analysis/` — Implementation of loss surfaces, fitting procedures, and experiments
- `results/` — Generated figures from experiments

## Installation

```bash
uv sync
```

## Experiments

Run all experiments:

```bash
uv run python -m scaling_law_analysis.experiments.run_all
```

| Experiment | Focus |
|------------|-------|
| **Exp 1**: Empirical Error | How sampling range affects exponent recovery |
| **Exp 2**: Exponent Imbalance | How α/β asymmetry amplifies fitting errors |
| **Exp 3**: Drift Sensitivity | How systematic sampling center biases affect accuracy |
| **Exp 4**: Extrapolation Error | How inference degrades when extrapolating beyond fitting range |
| **Exp 5**: Parametric Surface | Whether direct surface fitting (variable projection) avoids Approach 2 biases |
| **Exp 6**: Analytical Error | Analytical modeling of inference error *(TODO)* |

## TODO

- Conclusions to emphasize:
  - It really only takes subtle differences in where parabola minima are to result in significant extrapolation errors
    - Emphasize this by showing a chinchilla surface with center scaling and correct exponents / incorrect intercepts
    - Discuss how close the true and inferred parabola minima are to each other, and how this would be virtually impossible to detect w/ statistical noise
  - Extrapolation relies on scaling exponents AND intercepts from log-linear (D_opt ~ A * C^alpha) fits 
    - See https://github.com/marin-community/marin/blob/ce63e88434e0a44eba787747dbdd6a97123fb650/lib/marin/src/marin/scaling_laws/isoflop_analysis.py#L396
  - Taylor approximation accuracy effects intercept inference, not exponent inference
  - Sampling bias affects both exponents and intercepts, depending on the nature of that bias
    - A constant multiplicative bias, i.e. constant offset from center, effects only the intercepts
    - Anything else that does not result in a constant, multiplicative change from true centers effects both exponents and intercepts (e.g. drift)
- Results post
  - Start with symmetric loss surface
    - Assume perfect knowledge of optimal sampling centers
    - Explain what isoflops from it look like
    - Explain the fitting and extrapolation process
    - Show that this is perfect for exponent and intercept recovery + extrapolation 
  - Now what happens if the surface has slighly assymmetric scaling?
    - Assume perfect knowledge of optimal sampling centers
    - Assume chinchilla params
    - Now the exponents are perfect, but the intercepts are off 
    - Compare to high imbalance loss surface to show more exaggerated effects (still with perfect exponents)
    - Explain how the width of the sampling grid now somehow determines the accuracy of the intercept recovery
    - Show data on the accuracy of N*/D* recovery vs compute budgets
    - Explain errors in Taylor approximation accuracy
  - Now what happens if the sampling is not centered?
    - Assume a constant multiplicative bias from center
    - Show that exponents are still perfect somehow, but intercepts are not
    - Show mathematically why this is
    - Assume a drifting bias instead and how that makes both erroneous
    - This error in both also occurs if sampling centers are moved by some constant N on a linear scale -- the change needs to be multiplicative and constant to exactly cancel out exponent inference errors
  - How much does this matter for extrapolation?
  - Show that if parabolas are avoided in favor of true curve fits, all of the above problems go away
    - Explain how it is very hard even in this noise-free synthetic case to get Approach 3 to work
    - Explain how variable projection + grid search is necessary to get perfect inference