## Overview

This project studies how neural scaling laws are fit, comparing methods across objectives, optimizers, reparameterizations, and experimental designs.

- **Bias analysis**: Systematic errors in Chinchilla Approach 2's parabolic approximation are isolated using noise-free synthetic loss surfaces, tracing them to IsoFLOP sampling grid width, uncentered sampling, and loss surface asymmetry.
- **Method comparison**: Multiple fitting methods are evaluated — including Approach 2, several Approach 3 variants (direct 5D optimization with different objectives, gradient strategies, and initializations), and Variable Projection (VPNLS) — under both noiseless and noisy conditions.
- **Empirical validation**: Fitting implementations are validated against Apple's [ml-scalefit](https://github.com/apple/ml-scalefit) reproductions. Biases are quantified against published Llama 3 IsoFLOP data and shown to produce even larger misallocations on simulated multimodal scaling surfaces.
- **Novel reparameterization**: VPNLS exploits the partially linear structure of the loss surface to reduce fitting to a well-conditioned 2D search, eliminating the biases of Approach 2 while avoiding the numerical difficulties of full 5D optimization.

See [`specs/project.md`](specs/project.md#project-structure) for the full directory layout and implementation map.

## Results

Results from this analysis can be found in a few places:

- [Article (HTML)](https://openathena.ai/scaling-law-analysis/)
- [Paper (PDF)](https://github.com/Open-Athena/scaling-law-analysis/blob/main/results/paper/paper.pdf)
- [Interactive demo](https://claude.ai/public/artifacts/ff4b6e45-cc20-4a96-b95c-57caac05bfff) ([prompt](https://gist.github.com/eric-czech/77cc21e825e19b7ac98b9a538da6ec99))

See [`specs/build.md`](specs/build.md) for build and reproduction details.

## Installation

```bash
uv sync
```

## Experiments

Experiments from this analysis each isolate a specific bias source or fitting method comparison, progressing from baseline validation through individual error modes to combined effects and practical cost implications.

| Experiment | Focus |
|------------|-------|
| **Exp 0**: Reproductions | Reproduce Apple ml-scalefit results to validate the fitting implementation |
| **Exp 1**: Empirical Error | How sampling range affects exponent and intercept recovery on a symmetric surface |
| **Exp 2**: Exponent Imbalance | How α/β asymmetry amplifies fitting errors across surface configurations |
| **Exp 3**: Drift Sensitivity | How systematic sampling center biases (constant offset and linear drift) affect exponent and intercept accuracy |
| **Exp 4**: Extrapolation Error | How intercept errors from asymmetry and off-center sampling translate into token count errors when extrapolating to large compute budgets |
| **Exp 5**: Parameter Recovery | Whether VPNLS (variable projection + NNLS) recovers all five surface parameters without the parabolic approximation's biases, and how optimizer choice affects precision and stability |
| **Exp 6**: Analytical Error | Closed-form derivation of Approach 2 intercept error as a function of surface exponents and grid specification, validated against numerical results |
| **Exp 7**: Exponent Inference | How VPNLS and Approach 3 compare to Approach 2 for recovering scaling exponents under noise, sampling drift, and varying data budgets |
| **Exp 8**: Conditioning Analysis | Why Approach 3's 5D optimization is ill-conditioned (κ ≈ 3.5×10¹¹) and how variable projection reduces the problem to a well-conditioned 2D search (κ ≈ 11) |
| **Exp 9**: Data Efficiency | How fitting methods compare in accuracy given limited IsoFLOP data budgets |
| **Exp 10**: Compounding Errors | How individual bias sources accumulate when present simultaneously |
| **Exp 11**: Cost Estimates | How fitting biases translate into compute allocation errors at training scale |
| **Exp 12**: Residual Distributions | Residual patterns across fitting methods as a diagnostic for model misspecification |

See [`specs/experiments.md`](specs/experiments.md) for full specifications.
