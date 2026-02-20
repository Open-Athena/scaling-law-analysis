# Scaling Law Analysis

Simulating systematic biases in neural scaling law fitting procedures, with focus on Chinchilla Approach 2.

## Overview

This project uses synthetic loss surfaces to study how sampling choices and experimental design affect scaling exponent recovery. By controlling ground truth parameters, we isolate sources of bias in the IsoFLOP parabolic fitting method (Approach 2) without confounding from training noise.

The main output is a self-contained HTML article suitable for publication as a blog post. See the [Article](#article) section below.

See [specs/project.md](specs/project.md#project-structure) for the full directory layout and implementation map.

## Installation

```bash
uv sync
```

## Article

The article is a single self-contained HTML file demonstrating systematic biases in Chinchilla Approach 2 using noise-free synthetic data. It lives at `results/article/article.html` (source) and `results/article/article_standalone.html` (deployable, with inlined images).

Sections cover: symmetric baselines, asymmetric surface errors, off-center sampling biases, a robust alternative via variable projection, and evidence from published scaling law studies.

See [`specs/build.md`](specs/build.md) for the full build workflow, which covers:

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
| **Exp 5**: Parameter Recovery | Whether direct surface fitting (variable projection) avoids Approach 2 biases |
| **Exp 6**: Analytical Error | Analytical modeling of inference error *(TODO)* |
