# ml-scalefit reproduction

Reproduces Chinchilla scaling law fits using Apple's
[ml-scalefit](https://github.com/apple/ml-scalefit) library for comparison
with our VPNLS and Approach 3 implementations.

## Setup

```bash
cd /tmp
git clone --depth 1 https://github.com/apple/ml-scalefit.git
```

## Run

```bash
cd /tmp/ml-scalefit
uv run --with . --with pandas --with jax --with joblib --with scipy \
    python /path/to/scaling-law-analysis/docs/reproductions/ml-scalefit/run.py
```

Then copy outputs back:

```bash
cp scalefit_results.txt scalefit_results.csv \
    /path/to/scaling-law-analysis/docs/reproductions/ml-scalefit/
```

## Notes

- Inputs are normalized (N/1e6, D/1e9) to match scalefit convention.
- Runs 6 configs: MSE/Huber × L-BFGS/basinhopping, plus bootstrap variants.
- Uses `n_starts=100` random restarts; `n_starts=10` is insufficient (~3% gaps on α/β).
- Results are loaded by `exp0_reproductions.py` for side-by-side comparison.
