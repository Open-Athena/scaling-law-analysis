# Scaling Law Analysis

Simulating systematic biases in neural scaling law fitting procedures, with focus on Chinchilla Approach 2.

## Overview

This project uses synthetic loss surfaces to study how sampling choices and experimental design affect scaling exponent recovery. By controlling ground truth parameters, we isolate sources of bias in the IsoFLOP parabolic fitting method (Approach 2) without confounding from training noise.

The main output is a self-contained HTML article suitable for publication as a blog post. See the [Article](#article) section below.

## Project Structure

```
specs/                       # Source of truth for project intent and design
  project.md                 #   High-level project overview
  experiments.md             #   Detailed experiment specifications
  article.md                 #   Article outline and editorial guidelines
src/scaling_law_analysis/    # All implementation code
  chinchilla.py              #   Loss surface and Approach 2 fitting
  config.py                  #   Shared surface configurations
  experiments/               #   Experiment scripts (exp1–exp6) and runner
  article/                   #   Article figure generation and standalone builder
  references.py              #   Reference list generator from YAML
docs/                        # Build and deployment documentation
  build.md                   #   Full build workflow (experiments → figures → article)
  references/                #   references.yaml (source of truth for citations)
results/                     # All generated outputs (git-tracked)
  experiments/               #   Per-experiment figures (exp1/–exp6/)
  article/                   #   Article HTML, figures, CSVs, and supplementary PDF
```

## Installation

```bash
uv sync
```

## Article

The article is a single self-contained HTML file demonstrating systematic biases in Chinchilla Approach 2 using noise-free synthetic data. It lives at `results/article/article.html` (source) and `results/article/article_standalone.html` (deployable, with inlined images).

Sections cover: symmetric baselines, asymmetric surface errors, off-center sampling biases, a robust alternative via variable projection, and evidence from published scaling law studies.

See [`docs/build.md`](docs/build.md) for the full build workflow, which covers:

1. Running experiments
2. Generating article figures and CSV data
3. Generating the references list
4. Syncing CSV data with article prose
5. Building the standalone HTML and supplementary PDF

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
