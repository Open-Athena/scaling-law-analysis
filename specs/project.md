# Project Intent: Scaling Law Simulation

## Purpose

The purpose of this project is to:

- Demonstrate whether or not flaws in Chinchilla Approach 2 for scaling law exponent inference exist
- Evaluate alternative methods for scaling law exponent inference
- Produce balanced, concise, empirical and theoretical considerations for best practices in scaling law inference
- Create a single document at results.md very concisely explaining all experimental questions, findings, and conclusions

## Experiments

### Experiment 1: Empirical Error

Hypothesis: The accuracy of Chinchilla Approach 2 is dependent on the accuracy of the second-order Taylor expansion underlying the validity of parabolic fits.

Steps:
- Sample data from a Chinchilla loss surface with no statistical noise
- Use Chinchilla Approach 2 to infer exponents from the sampled data
- Show how error in this inferences changes with grid resolution and alpha/beta assymetry

### Experiment 2: Analytical Error

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
