"""Reproduce Chinchilla scaling law fits using scalefit.

Run from within a clone of https://github.com/apple/ml-scalefit:

    cd /tmp/ml-scalefit
    uv run --with . --with pandas --with jax --with joblib --with scipy \
        python /path/to/run.py
"""

import csv
import sys

import pandas as pd
from scalefit import ScalingLaw  # type: ignore[import]

CSV_FILE = "data/chinchilla.csv"
OUTPUT_TXT = "scalefit_results.txt"
OUTPUT_CSV = "scalefit_results.csv"


def chinchilla_model(params, inputs):
    N = inputs["model_size"] / 1e6
    D = inputs["n_tokens"] / 1e9
    return (
        params["bias"]
        + params["bias_tokens"] / (D ** params["pow_tokens"])
        + params["bias_size"] / (N ** params["pow_size"])
    )


BOUNDS = {
    "bias": (0.0, 2.0),
    "bias_tokens": (0.0, 10.0),
    "pow_tokens": (0.0, 1.0),
    "bias_size": (0.0, 10.0),
    "pow_size": (0.0, 1.0),
}

PARAM_NAMES = ["bias", "bias_size", "bias_tokens", "pow_size", "pow_tokens"]

# n_starts=100 ensures sufficient random restarts for convergence;
# n_starts=10 can land in slightly different local minima (~3% gaps on α/β).
N_STARTS = 100

CONFIGS = [
    {"loss": "mse", "optimizer": "lbfgs", "n_bootstraps": 1},
    {"loss": "mse", "optimizer": "basinhopping", "n_bootstraps": 1},
    {"loss": "mse", "optimizer": "basinhopping", "n_bootstraps": 10},
    {"loss": "huber", "optimizer": "lbfgs", "n_bootstraps": 1},
    {"loss": "huber", "optimizer": "basinhopping", "n_bootstraps": 1},
    {"loss": "huber", "optimizer": "basinhopping", "n_bootstraps": 10},
]


def fit_one(X, y, loss, optimizer, n_bootstraps):
    model = ScalingLaw(
        model_fn=chinchilla_model,
        bounds=BOUNDS,
        seed=42,
        n_bootstraps=n_bootstraps,
        resampling_method="bootstrap" if n_bootstraps > 1 else "none",
        loss=loss,
        optimizer=optimizer,
        n_starts=N_STARTS,
    )
    model.fit(X, y)
    return model.optimal_params_, model.score(X, y)


def main():
    df = pd.read_csv(CSV_FILE)
    compute = 6 * df["n_tokens"] * df["model_size"]
    mask = compute < 1e21
    features = ["model_size", "n_tokens"]
    X, y = df.loc[mask, features], df.loc[mask, "loss"]
    n = len(y)

    results = []
    rows = []
    for cfg in CONFIGS:
        bs = cfg["n_bootstraps"]
        label = f"{cfg['loss']}+{cfg['optimizer']}" + (f"+bs{bs}" if bs > 1 else "")
        print(f"Fitting {label} ...", file=sys.stderr)
        params, scores = fit_one(X, y, cfg["loss"], cfg["optimizer"], bs)
        results.append((label, params, scores))
        row = {"config": label, "n": n}
        for k in PARAM_NAMES:
            row[k] = params[k]
        row["r2"] = scores["r2"]
        row["mre"] = scores["mre"]
        row["mae"] = scores["mae"]
        rows.append(row)

    # Write text report
    w = []
    w.append("scalefit Chinchilla Reproductions")
    w.append("=" * 60)
    w.append("")
    w.append(f"Data: {CSV_FILE} (C < 1e21, {n} points)")
    w.append(
        "Model: bias + bias_size / (N/1e6)^pow_size + bias_tokens / (D/1e9)^pow_tokens"
    )
    w.append(f"n_starts={N_STARTS}, seed=42")
    w.append("")

    w.append("Parameters")
    w.append("-" * 90)
    w.append(
        f"{'config':<28s}  {'bias':>10s}  {'bias_size':>10s}  {'bias_tok':>10s}  {'pow_size':>10s}  {'pow_tok':>10s}"
    )
    for label, params, _ in results:
        w.append(
            f"{label:<28s}  {params['bias']:>10.4f}  {params['bias_size']:>10.4f}  "
            f"{params['bias_tokens']:>10.4f}  {params['pow_size']:>10.4f}  {params['pow_tokens']:>10.4f}"
        )
    w.append("")

    w.append("Scores")
    w.append("-" * 90)
    w.append(f"{'config':<28s}  {'R²':>10s}  {'MRE%':>10s}  {'MAE':>10s}")
    for label, _, scores in results:
        w.append(
            f"{label:<28s}  {scores['r2']:>10.6f}  {scores['mre']*100:>10.2f}  {scores['mae']:>10.6f}"
        )
    w.append("")

    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(w) + "\n")
    print(f"Saved: {OUTPUT_TXT}", file=sys.stderr)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {OUTPUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
