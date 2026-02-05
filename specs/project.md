# Project Intent: Scaling Law Analysis

## Purpose

The purpose of this project is to:

- Demonstrate whether or not flaws in Chinchilla Approach 2 for scaling law exponent inference exist
- Evaluate alternative methods for scaling law exponent inference
- Produce balanced, concise, empirical and theoretical considerations for best practices in scaling law inference

## Project Structure

This project consists of several components, each with its own detailed specification:

### Experiments

See **[specs/experiments.md](experiments.md)** for detailed experiment specifications.

The experiments systematically evaluate scaling law inference methods using synthetic data:

1. **Experiment 1**: Empirical Error — baseline accuracy of Approach 2
2. **Experiment 2**: Exponent Imbalance Sensitivity — effect of α/β asymmetry
3. **Experiment 3**: Sampling Drift Sensitivity — effect of sampling biases
4. **Experiment 4**: Extrapolation Error — accuracy at higher compute budgets
5. **Experiment 5**: Parametric Surface Fitting — alternative fitting approach
6. **Experiment 6**: Analytical Error — closed-form error derivations

Results are stored in `results/experiments/exp{N}/`.

### Article

See **[specs/article.md](article.md)** for the article specification.

An HTML blog post (~10 minute read) demonstrating systematic biases in Chinchilla Approach 2 using noise-free synthetic data. Target audience: ML practitioners familiar with scaling laws.

Results are stored in `results/article/`.

### Paper

*TBD* — A more formal academic treatment of the findings, intended for publication.
