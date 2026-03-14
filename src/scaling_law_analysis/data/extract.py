"""Fetch isoflop data from published sources and standardize to common schema."""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from typing import Callable

import numpy as np
import pandas as pd

from scaling_law_analysis.data import DATA_DIR
from scaling_law_analysis.data.transform import make_df

# ── Shared helpers ───────────────────────────────────────────────────────────


def _budget_6nd(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Compute FLOPs budget as C = 6 * N * D."""
    return 6.0 * np.asarray(n, dtype=float) * np.asarray(d, dtype=float)


def _parse_budget_from_name(name: str) -> float:
    """Extract compute budget from marin run Name field.

    Patterns: '...-3e+19-...', '...-1e+20-...'
    """
    m = re.search(r"-(\d+(?:\.\d+)?e[+-]\d+)-", name)
    if m:
        return float(m.group(1))
    m = re.search(r"-(\d+(?:\.\d+)?e[+-]\d+)$", name)
    if m:
        return float(m.group(1))
    return float("nan")


# ── Chinchilla isoFLOP helpers ───────────────────────────────────────────────

# 9 isoFLOP budgets from Figure 3 of the Chinchilla paper
# (Hoffmann et al., https://arxiv.org/abs/2203.15556).
# The original experiments trained multiple model sizes at each of these
# fixed compute budgets.  The raw data (from ml-scalefit and Epoch AI)
# includes runs from all three Chinchilla approaches; we keep only the
# points that snap to one of these 9 budgets.
CHINCHILLA_ISOFLOP_BUDGETS: list[float] = [
    6e18,
    1e19,
    3e19,
    6e19,
    1e20,
    3e20,
    6e20,
    1e21,
    3e21,
]
_CHINCHILLA_BUDGET_TOLERANCE = 0.10  # 10% relative error


def _snap_to_chinchilla_budget(
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap compute values to the nearest known Chinchilla isoFLOP budget.

    Returns:
        (snapped_budgets, mask) where mask is True for points within tolerance.
    """
    budgets = np.array(CHINCHILLA_ISOFLOP_BUDGETS)
    snapped = np.empty_like(c)
    mask = np.zeros(len(c), dtype=bool)
    for i, ci in enumerate(c):
        rel_err = np.abs(ci / budgets - 1)
        best = int(np.argmin(rel_err))
        if rel_err[best] < _CHINCHILLA_BUDGET_TOLERANCE:
            snapped[i] = budgets[best]
            mask[i] = True
    return snapped, mask


# ── Source: ml_scalefit ──────────────────────────────────────────────────────

# Paper: https://arxiv.org/abs/2507.09404
ML_SCALEFIT_URL = (
    "https://raw.githubusercontent.com/apple/ml-scalefit/"
    "ac4664af5db6c94e6ac7521a61dd3bbb0d91cc3a/data/chinchilla.csv"
)


def fetch_ml_scalefit() -> pd.DataFrame:
    """Chinchilla isoFLOP data from apple/ml-scalefit.

    Budget is computed as C = 6*N*D then snapped to the 9 known isoFLOP
    budgets from Figure 3 of the Chinchilla paper (arxiv:2203.15556).
    Points that don't match any budget within 10% are discarded.
    """
    print("ml_scalefit: downloading from GitHub ...")
    with urllib.request.urlopen(ML_SCALEFIT_URL) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    n = np.array([float(r["model_size"]) for r in rows])
    d = np.array([float(r["n_tokens"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    c_raw = _budget_6nd(n, d)
    c_snapped, mask = _snap_to_chinchilla_budget(c_raw)
    print(f"  kept {mask.sum()}/{len(mask)} points matching isoFLOP budgets")
    return make_df(
        source="ml_scalefit",
        provenance="chinchilla",
        dataset="massivetext",
        model="chinchilla",
        tokens=d[mask],
        params=n[mask],
        budget=c_snapped[mask],
        loss=loss[mask],
    )


# ── Source: epochai_chinchilla ───────────────────────────────────────────────

# Paper: https://arxiv.org/abs/2404.10102
EPOCHAI_CSV_URL = (
    "https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/"
    "92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv"
)


def fetch_epochai_chinchilla() -> pd.DataFrame:
    """Chinchilla replication data from Epoch AI (SVG-extracted).

    N and C are rounded to integers (SVG extraction artifacts), then C is
    snapped to the 9 known isoFLOP budgets from Figure 3 of the Chinchilla
    paper (arxiv:2203.15556).  D is derived as C / (6*N) using the snapped
    budget.  Points that don't match any budget within 10% are discarded.
    """
    print("epochai_chinchilla: downloading from GitHub ...")
    with urllib.request.urlopen(EPOCHAI_CSV_URL) as resp:
        text = resp.read().decode("utf-8")
    raw = pd.read_csv(io.StringIO(text))
    n = np.round(raw["Model Size"].to_numpy(dtype=float))
    c_raw = np.round(raw["Training FLOP"].to_numpy(dtype=float))
    loss = raw["loss"].to_numpy(dtype=float)
    c_snapped, mask = _snap_to_chinchilla_budget(c_raw)

    # SVG extraction can produce near-duplicate points that snap to the same
    # (N, budget).  Keep the one whose original C was closest to the target.
    tmp = pd.DataFrame(
        {
            "params": n[mask],
            "budget": c_snapped[mask],
            "loss": loss[mask],
            "c_raw": c_raw[mask],
        }
    )
    tmp["rel_err"] = np.abs(tmp["c_raw"] / tmp["budget"] - 1)
    tmp = tmp.sort_values("rel_err").drop_duplicates(
        subset=["params", "budget"], keep="first"
    )
    print(f"  kept {len(tmp)}/{len(raw)} points matching isoFLOP budgets")

    # D derived from snapped budget via C = 6ND
    d = tmp["budget"].to_numpy() / (6.0 * tmp["params"].to_numpy())
    return make_df(
        source="epochai_chinchilla",
        provenance="chinchilla",
        dataset="massivetext",
        model="chinchilla",
        tokens=d,
        params=tmp["params"].to_numpy(),
        budget=tmp["budget"].to_numpy(),
        loss=tmp["loss"].to_numpy(),
    )


# ── Source: misfitting ───────────────────────────────────────────────────────

# Paper: https://arxiv.org/abs/2502.18969
# Code: https://github.com/hadasah/scaling_laws
# IsoFLOP construction follows the interpolation approach from
# hadasah/scaling_laws/paper_analysis_and_plots.py (fetch_flop / interp_flop).
MISFITTING_CSV_URL = (
    "https://raw.githubusercontent.com/hadasah/scaling_laws/"
    "1f3708c0a12df0effb0ee906b1da5f9f0ff4f4f1/data/scaling_results.csv"
)

# Budget grid: log-spaced values covering the data range where >= 3 model
# sizes have checkpoint data within the FLOP tolerance.
_MISFITTING_FLOP_TOLERANCE = 0.1  # reject checkpoints >10% from target budget
_MISFITTING_MIN_MODELS = 3  # minimum model sizes per budget to form an isoflop curve


def _interpolate_loss_at_budget(
    run_df: pd.DataFrame, target_c: float, flop_tolerance: float, n_context: int = 5
) -> float | None:
    """Interpolate a single run's loss at a target FLOP budget.

    Uses log-log interpolation on nearby checkpoints, following the approach
    in hadasah/scaling_laws fetch_flop().

    Args:
        run_df: Checkpoints for one (N, peak_lr, sweep) run, with C and loss columns.
        target_c: Target FLOP budget.
        flop_tolerance: Max relative error to accept (e.g. 0.1 = 10%).
        n_context: Number of checkpoints on each side for interpolation window.

    Returns:
        Interpolated loss, or None if no checkpoint is close enough.
    """
    run_df = run_df.sort_values("C")
    c_vals = run_df["C"].to_numpy(dtype=float)
    loss_vals = run_df["loss"].to_numpy(dtype=float)

    if len(c_vals) == 0:
        return None

    # Find nearest checkpoint
    idx = int(np.searchsorted(c_vals, target_c))
    if idx > 0:
        # Pick the closer of the two neighbors
        candidates = slice(max(0, idx - 1), min(len(c_vals), idx + 1))
        idx = max(0, idx - 1) + int(
            np.abs(np.log(c_vals[candidates] / target_c)).argmin()
        )

    idx = np.clip(idx, 0, len(c_vals) - 1)
    rel_err = np.exp(np.abs(np.log(c_vals[idx] / target_c))) - 1
    if rel_err > flop_tolerance:
        return None

    # Log-log interpolation using nearby checkpoints
    lo = max(0, idx - n_context)
    hi = min(len(c_vals), idx + n_context)
    c_window = c_vals[lo:hi]
    loss_window = loss_vals[lo:hi]

    if len(c_window) > 1:
        return float(
            np.exp(np.interp(np.log(target_c), np.log(c_window), np.log(loss_window)))
        )
    return float(loss_vals[idx])


def _build_isoflop_grid(
    df: pd.DataFrame, flop_tolerance: float, min_models: int
) -> np.ndarray:
    """Select budget grid points where enough model sizes have data."""
    c_min = df["C"].min()
    c_max = df["C"].max()
    # Dense candidate grid (40 log-spaced points across full range)
    candidates = np.logspace(np.log10(c_min), np.log10(c_max), 40)
    model_sizes = df["N"].unique()

    good = []
    for target_c in candidates:
        n_with_data = 0
        for n in model_sizes:
            sub = df[df["N"] == n]
            c_vals = sub["C"].to_numpy(dtype=float)
            if np.any(np.abs(np.log(c_vals / target_c)) < np.log(1 + flop_tolerance)):
                n_with_data += 1
        if n_with_data >= min_models:
            good.append(target_c)
    return np.array(good)


def fetch_misfitting() -> pd.DataFrame:
    """Misfitting scaling law survey data (Marghi et al.).

    Trained on FineWeb, evaluated on C4. IsoFLOP curves are constructed by
    interpolating each model's loss curve (across checkpoints) at target
    budget levels, then selecting the best LR for each model size.
    """
    print("misfitting: downloading from GitHub ...")
    with urllib.request.urlopen(MISFITTING_CSV_URL) as resp:
        text = resp.read().decode("utf-8")
    raw = pd.read_csv(io.StringIO(text))

    # Clean
    for col in ["N", "D", "C", "loss"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["N", "D", "C", "loss"])
    raw = raw[(raw[["N", "D", "C", "loss"]] > 0).all(axis=1)].copy()

    # Build budget grid
    budget_grid = _build_isoflop_grid(
        raw, _MISFITTING_FLOP_TOLERANCE, _MISFITTING_MIN_MODELS
    )
    print(
        f"  budget grid: {len(budget_grid)} levels from {budget_grid.min():.2e} to {budget_grid.max():.2e}"
    )

    # For each target budget, interpolate each run's loss, then take best LR per model size
    records: list[dict] = []
    for target_c in budget_grid:
        # Interpolate loss for every run (unique model + lr + sweep)
        run_results: list[dict] = []
        for (n, _lr, _sweep), run_df in raw.groupby(["N", "peak_lr", "sweep"]):
            loss = _interpolate_loss_at_budget(
                run_df, target_c, _MISFITTING_FLOP_TOLERANCE
            )
            if loss is not None:
                run_results.append({"N": n, "loss": loss})

        if len(run_results) == 0:
            continue

        # Best LR per model size (minimum loss)
        run_df = pd.DataFrame(run_results)
        best = run_df.loc[run_df.groupby("N")["loss"].idxmin()]

        for _, row in best.iterrows():
            n_val = float(row["N"])
            d_val = target_c / (6.0 * n_val)  # D derived from C = 6ND
            records.append({"N": n_val, "D": d_val, "C": target_c, "loss": row["loss"]})

    assert len(records) > 0, "No isoflop points produced"
    result = pd.DataFrame(records)
    n_budgets = result["C"].nunique()
    pts_per_budget = result.groupby("C").size()
    print(
        f"  {len(result)} points across {n_budgets} budgets "
        f"({pts_per_budget.min()}-{pts_per_budget.max()} models/budget)"
    )

    return make_df(
        source="misfitting",
        provenance="chinchilla",
        dataset="fineweb_c4",
        model="transformer",
        tokens=result["D"].to_numpy(),
        params=result["N"].to_numpy(),
        budget=result["C"].to_numpy(),
        loss=result["loss"].to_numpy(),
    )


# ── Source: llama3 ───────────────────────────────────────────────────────────

# Paper: https://arxiv.org/abs/2407.21783
LLAMA3_ISOFLOP_CSV_URL = (
    "https://raw.githubusercontent.com/eric-czech/llama3_isoflop_extraction/"
    "1bc1755b76e6ee55a911549c8ec52b71cb480320/isoflops_points.csv"
)


def fetch_llama3() -> pd.DataFrame:
    """Llama 3 isoFLOP data (digitized from paper figure).

    Produces two experiments via condition:
      - raw_loss: loss = validation_loss as-is
      - exp_loss: loss = exp(validation_loss)
    N is derived as C / (6*D) in both cases.
    """
    print("llama3: downloading from GitHub ...")
    with urllib.request.urlopen(LLAMA3_ISOFLOP_CSV_URL) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    c = np.array([float(r["compute_budget"]) for r in rows])
    d = np.array([float(r["training_tokens"]) for r in rows])
    raw_loss = np.array([float(r["validation_loss"]) for r in rows])
    # N derived from C = 6ND
    n = c / (6.0 * d)

    dfs: list[pd.DataFrame] = []
    for condition, loss_vals in [
        ("raw_loss", raw_loss),
        ("exp_loss", np.exp(raw_loss)),
    ]:
        dfs.append(
            make_df(
                source="llama3",
                provenance="llama3",
                dataset="llama_3",
                model="llama_3",
                condition=condition,
                tokens=d,
                params=n,
                budget=c,
                loss=loss_vals,
            )
        )
    return pd.concat(dfs, ignore_index=True)


# ── Source: marin_202603 ─────────────────────────────────────────────────────

# Report: https://wandb.ai/marin-community/marin/reports/Scaling-Ladders--VmlldzoxNTc0MjM1NQ

_MARIN_EXPERIMENTS: list[tuple[str, str, str]] = [
    # (csv_filename, dataset, model)
    ("isoflop-comma.csv", "comma", "llama_2"),
    ("isoflop-dclm.csv", "dclm", "llama_2"),
    ("isoflop-nemotron.csv", "nemotron", "llama_2"),
]


def fetch_marin_202603() -> pd.DataFrame:
    """Marin scaling ladder isoFLOP data.

    Budget is parsed from the run Name field and corrected to total FLOPs.

    The Marin data has two naming conventions:
      - Regular runs: budget in name ≈ 2ND (forward-pass FLOPs only).
        Empirically, 6ND/parsed_budget ≈ 3.0 across 258 runs.
      - "validation-optimal" runs (one per budget, comma & nemotron only):
        budget in name ≈ 6ND (already total FLOPs).
        Empirically, 6ND/parsed_budget ≈ 1.0 across 15 runs.

    We multiply parsed budgets by 3 (converting forward-pass to total FLOPs)
    and exclude the "validation-optimal" runs (1 per budget, comma & nemotron
    only) which use a different FLOPs convention and would create duplicates.
    After correction we validate that budget ≈ 6ND within 40%.
    """
    print("marin_202603: loading vendored CSVs ...")
    raw_dir = DATA_DIR / "marin_202603" / "raw"
    dfs: list[pd.DataFrame] = []
    for csv_name, dataset, model in _MARIN_EXPERIMENTS:
        raw = pd.read_csv(raw_dir / csv_name)

        # Exclude "validation-optimal" runs: these use a different FLOPs
        # convention (budget ≈ 6ND already) and would collide with regular
        # runs after 3x correction.  There is at most 1 per budget.
        is_optimal = raw["Name"].str.contains("validation-optimal")
        n_optimal = is_optimal.sum()
        if n_optimal > 0:
            print(f"  {csv_name}: dropping {n_optimal} validation-optimal runs")
        raw = raw[~is_optimal].copy()

        raw["budget_parsed"] = raw["Name"].astype(str).apply(_parse_budget_from_name)
        n = raw["parameter_count"].to_numpy(dtype=float)
        d = raw["throughput/total_tokens"].to_numpy(dtype=float)
        loss = raw["eval/paloma/macro_loss"].to_numpy(dtype=float)
        budget_parsed = raw["budget_parsed"].to_numpy(dtype=float)

        # Parsed budget is forward-pass FLOPs (≈ 2ND).  Multiply by 3 to
        # convert to total FLOPs (≈ 6ND).  Empirically, 6ND/parsed_budget
        # ≈ 3.0 across all 258 regular runs.
        c = budget_parsed * 3.0

        # Standardize: drop non-positive / NaN
        mask = (
            (n > 0)
            & (d > 0)
            & (c > 0)
            & (loss > 0)
            & np.isfinite(n)
            & np.isfinite(d)
            & np.isfinite(c)
            & np.isfinite(loss)
        )

        # Validate: corrected budget should be within 40% of C = 6ND
        c6nd = _budget_6nd(n[mask], d[mask])
        ratio = c6nd / c[mask]
        bad = np.abs(ratio - 1) > 0.40
        if bad.any():
            names = raw["Name"].to_numpy()[mask]
            for idx in np.where(bad)[0]:
                print(
                    f"  WARNING {csv_name}: 6ND/budget={ratio[idx]:.2f} for "
                    f"{names[idx]} (budget={c[mask][idx]:.2e}, 6ND={c6nd[idx]:.2e})"
                )
            print(
                f"  {bad.sum()}/{mask.sum()} rows exceed 40% tolerance in {csv_name} "
                f"(keeping all rows — outlier detection is downstream)"
            )

        n, d, c, loss = n[mask], d[mask], c[mask], loss[mask]
        print(f"  {csv_name}: {len(n)} rows")
        dfs.append(
            make_df(
                source="marin_202603",
                provenance="chinchilla",
                dataset=dataset,
                model=model,
                tokens=d,
                params=n,
                budget=c,
                loss=loss,
            )
        )
    return pd.concat(dfs, ignore_index=True)


# ── Registry ─────────────────────────────────────────────────────────────────

FETCH_FUNCTIONS: list[tuple[str, Callable[[], pd.DataFrame]]] = [
    ("ml_scalefit", fetch_ml_scalefit),
    ("epochai_chinchilla", fetch_epochai_chinchilla),
    ("misfitting", fetch_misfitting),
    ("llama3", fetch_llama3),
    ("marin_202603", fetch_marin_202603),
]
