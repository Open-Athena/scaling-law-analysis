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
