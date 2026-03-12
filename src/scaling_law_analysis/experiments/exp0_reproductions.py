"""Experiment 0: Reproductions.

Fits the Chinchilla data from the scalefit repository using VPNLS and
Approach 3, reproducing published results and comparing fitting methods.

Usage:
    uv run python -m scaling_law_analysis.experiments.exp0_reproductions
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scaling_law_analysis import config
from scaling_law_analysis.chinchilla import (
    ExponentGrid,
    ParameterGrid,
    SurfaceFitResult,
    fit_approach3,
    fit_vpnls,
)
from scaling_law_analysis.experiments.common import (
    CHINCHILLA_SURFACE,
    prepare_output_dir,
)

# ── Data source ──────────────────────────────────────────────────────────────

CHINCHILLA_CSV_URL = (
    "https://raw.githubusercontent.com/apple/ml-scalefit/"
    "ac4664af5db6c94e6ac7521a61dd3bbb0d91cc3a/data/chinchilla.csv"
)

SCALEFIT_CSV = (
    config.PROJECT_ROOT
    / "docs"
    / "reproductions"
    / "ml-scalefit"
    / "scalefit_results.csv"
)

# scalefit normalizes inputs: N→millions, D→billions. We do the same so that
# fitted A/B are directly comparable to scalefit's bias_size/bias_tokens.
N_NORM = 1e6
D_NORM = 1e9
C_MAX = 1e21


@dataclass
class ScalefitRow:
    """One row from the scalefit results CSV."""

    config: str
    n: int
    E: float  # bias
    A: float  # bias_size
    B: float  # bias_tokens
    alpha: float  # pow_size
    beta: float  # pow_tokens
    r2: float
    mre: float
    mae: float

    @property
    def a(self) -> float:
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        return self.alpha / (self.alpha + self.beta)


# ── Grids ────────────────────────────────────────────────────────────────────

VPNLS_GRID = ExponentGrid(
    alpha=np.linspace(0.05, 0.95, 256),
    beta=np.linspace(0.05, 0.95, 256),
)

A3_GRID = ParameterGrid(
    E=np.linspace(0.1, 5.0, 8),
    A=np.logspace(1, 6, 8),
    B=np.logspace(1, 6, 8),
    alpha=np.linspace(0.05, 0.95, 8),
    beta=np.linspace(0.05, 0.95, 8),
)

# ── Data loading ─────────────────────────────────────────────────────────────


def _download_chinchilla_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Download scalefit Chinchilla CSV and return (N, D, L, C) arrays.

    Inputs are normalized (N/1e6, D/1e9) to match scalefit convention.
    """
    with urllib.request.urlopen(CHINCHILLA_CSV_URL) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    N_list, D_list, L_list = [], [], []
    for row in reader:
        D_list.append(float(row["n_tokens"]))
        N_list.append(float(row["model_size"]))
        L_list.append(float(row["loss"]))
    N_raw = np.array(N_list)
    D_raw = np.array(D_list)
    L = np.array(L_list)
    C = 6 * N_raw * D_raw  # raw compute (before normalization)
    N = N_raw / N_NORM
    D = D_raw / D_NORM
    return N, D, L, C


def _load_scalefit_results() -> list[ScalefitRow]:
    """Load scalefit CSV results from docs/reproductions/."""
    rows: list[ScalefitRow] = []
    with open(SCALEFIT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ScalefitRow(
                    config=row["config"],
                    n=int(row["n"]),
                    E=float(row["bias"]),
                    A=float(row["bias_size"]),
                    B=float(row["bias_tokens"]),
                    alpha=float(row["pow_size"]),
                    beta=float(row["pow_tokens"]),
                    r2=float(row["r2"]),
                    mre=float(row["mre"]),
                    mae=float(row["mae"]),
                )
            )
    return rows


# ── Fitting ──────────────────────────────────────────────────────────────────

FIT_SPECS: list[tuple[str, dict]] = [
    (
        "VPNLS",
        {
            "func": fit_vpnls,
            "kwargs": {"grid": VPNLS_GRID, "method": "l-bfgs-b", "use_grad": True},
        },
    ),
    (
        "Approach 3 (lse+logloss)",
        {
            "func": fit_approach3,
            "kwargs": {
                "grid": A3_GRID,
                "use_grad": True,
                "use_lse": True,
                "use_logloss": True,
            },
        },
    ),
    (
        "Approach 3 (lse)",
        {
            "func": fit_approach3,
            "kwargs": {
                "grid": A3_GRID,
                "use_grad": True,
                "use_lse": True,
                "use_logloss": False,
            },
        },
    ),
]


def _run_fits(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
) -> list[tuple[str, SurfaceFitResult]]:
    """Run all fit specs and return (name, result) pairs."""
    results = []
    for name, spec in FIT_SPECS:
        func = spec["func"]
        result = func(N=N, D=D, L=L, **spec["kwargs"])
        print(
            f"  {name}: E={result.E:.4f}, A={result.A:.4f}, B={result.B:.4f}, "
            f"α={result.alpha:.4f}, β={result.beta:.4f} (RSS={result.residual_sum_squares:.6f})"
        )
        results.append((name, result))
    return results


# ── Report ───────────────────────────────────────────────────────────────────


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build a fixed-width text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}s}}" for w in widths)
    sep = "  ".join("-" * w for w in widths)
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*row))
    return lines


def save_report(
    output_path: str | Path,
    results: list[tuple[str, SurfaceFitResult]],
    scalefit_rows: list[ScalefitRow],
    n_points: int,
) -> None:
    """Write reproduction report."""
    lines: list[str] = []
    w = lines.append

    w("Chinchilla Scaling Law Reproductions")
    w("=" * 70)
    w("")
    w(f"Data: {CHINCHILLA_CSV_URL}")
    w("Compute: C = 6 * n_tokens * model_size")
    w(f"Train mask: C < {C_MAX:.0e} ({n_points} points)")
    w("Input normalization: N/1e6, D/1e9 (matches scalefit)")
    w("")

    # ── Our fits ──────────────────────────────────────────────────────────
    w("Our Fits")
    w("-" * 70)

    # Parameters table
    param_headers = ["method", "E", "A", "B", "α", "β", "a", "b", "RSS"]
    param_rows: list[list[str]] = []

    # Reference row
    s = CHINCHILLA_SURFACE
    param_rows.append(
        [
            "Chinchilla (Table A3)",
            f"{s.E:.4f}",
            f"{s.A:.1f}",
            f"{s.B:.1f}",
            f"{s.alpha:.4f}",
            f"{s.beta:.4f}",
            f"{s.a:.4f}",
            f"{s.b:.4f}",
            "",
        ]
    )

    # Fit rows
    for name, r in results:
        param_rows.append(
            [
                name,
                f"{r.E:.4f}",
                f"{r.A:.4f}",
                f"{r.B:.4f}",
                f"{r.alpha:.4f}",
                f"{r.beta:.4f}",
                f"{r.a:.4f}",
                f"{r.b:.4f}",
                f"{r.residual_sum_squares:.6f}",
            ]
        )

    for line in _table(param_headers, param_rows):
        w(line)
    w("")
    w(
        "Note: Chinchilla A/B use unnormalized inputs; not directly comparable to fitted A/B."
    )
    w("")

    # ── scalefit results ──────────────────────────────────────────────────
    w("scalefit Results (n_starts=100, seed=42)")
    w("-" * 70)

    sf_param_headers = ["config", "E", "A", "B", "α", "β", "a", "b"]
    sf_param_rows = [
        [
            r.config,
            f"{r.E:.4f}",
            f"{r.A:.4f}",
            f"{r.B:.4f}",
            f"{r.alpha:.4f}",
            f"{r.beta:.4f}",
            f"{r.a:.4f}",
            f"{r.b:.4f}",
        ]
        for r in scalefit_rows
    ]
    for line in _table(sf_param_headers, sf_param_rows):
        w(line)
    w("")

    sf_score_headers = ["config", "R²", "MRE%", "MAE"]
    sf_score_rows = [
        [
            r.config,
            f"{r.r2:.6f}",
            f"{r.mre * 100:.2f}",
            f"{r.mae:.6f}",
        ]
        for r in scalefit_rows
    ]
    for line in _table(sf_score_headers, sf_score_rows):
        w(line)
    w("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = prepare_output_dir(config.RESULTS_DIR / "experiments" / "exp0")

    N, D, L, C = _download_chinchilla_data()
    mask = C < C_MAX
    n = int(mask.sum())

    print(f"\nC < {C_MAX:.0e} ({n} points)")
    results = _run_fits(N[mask], D[mask], L[mask])
    scalefit_rows = _load_scalefit_results()

    save_report(
        output_dir / "reproductions.txt",
        results=results,
        scalefit_rows=scalefit_rows,
        n_points=n,
    )


if __name__ == "__main__":
    main()
