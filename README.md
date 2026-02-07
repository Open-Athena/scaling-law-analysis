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

On article:

- Discuss practical relevance of sampling grid width
  - Note that many experiments (like Llama3 and Marin) use token grids spanning 1-2 decades (OOMs base 10)
  - This contrasts with the 16x sampling grid in the current write-up, which spans ~2.4 decades
  - Make sure to show errors for more common grid widths (e.g. +/-10x)
  - Add section on what is a "normal" grid width
- Links to share from simple demo example:
  - https://gemini.google.com/share/67e761f19481
  - https://chatgpt.com/share/69853e09-8180-800e-8eaf-0840cd5d2d45
- Reduce decimal precision in figures where possible
- Move derivation to pdf in repo and link in article
- Expand on this:
  - > For surfaces with asymmetric exponents, wider sampling grids amplify the parabola-fitting mismatch, increasing the constant vertex shift and thus the intercept bias.
  - Mention that it affects extrapolation as well
- Rename "High Imbalance" to "Asymmetric"
- Replace references to scaling grid sizes with code blocks e.g. `Small` instead of "Small"
- Rephrase "Symmetric surfaces are immune"
- Change y-range of first extrapolation error plot (bars are too small)
- Update "The Happy Path — Symmetric Surfaces" section title
- Create combined figure of real isoflop curve plots from Llama, DeepSeek, etc. showing how these sampling biases are real
- Prior to final review:
  - Review figures.py for ways to use existing code utilities and then regen