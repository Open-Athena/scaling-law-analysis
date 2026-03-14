"""Pipeline: extract, transform, and visualize isoflop data.

Usage:
    uv run python -m scaling_law_analysis.data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scaling_law_analysis.data.common import EXTRACT_DIR, TRANSFORM_DIR
from scaling_law_analysis.data.extract import FETCH_FUNCTIONS
from scaling_law_analysis.data.schema import ANNOTATED_SCHEMA_COLS, SCHEMA_COLS
from scaling_law_analysis.data.transform import (
    EXPERIMENT_OVERRIDES,
    assert_unique,
    detect_outliers,
    fit_parabolas,
    ordered_experiments,
)
from scaling_law_analysis.data.visualize import plot_isoflops, plot_isoflops_akima


def _write_source(df: pd.DataFrame, source: str) -> Path:
    """Write per-source CSV to data/isoflops/extract/<source>/isoflops.csv."""
    out_dir = EXTRACT_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "isoflops.csv"
    df[SCHEMA_COLS].to_csv(out_path, index=False)
    print(f"  wrote {len(df):>5d} rows -> {out_path}")
    return out_path


def main() -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)

    # ── Extract ──────────────────────────────────────────────────────────
    all_dfs: list[pd.DataFrame] = []
    for source_name, fetch_fn in FETCH_FUNCTIONS:
        print(f"\n{'─' * 60}")
        df = fetch_fn()
        _write_source(df, source_name)
        all_dfs.append(df)

    # ── Aggregate ────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Aggregating all sources ...")
    combined = pd.concat(all_dfs, ignore_index=True)
    assert_unique(combined, "aggregate")

    agg_cols = [c for c in SCHEMA_COLS if c != "provenance"]
    extract_csv = EXTRACT_DIR / "isoflops.csv"
    combined[agg_cols].to_csv(extract_csv, index=False)
    print(f"  wrote {len(combined):>5d} rows -> {extract_csv}")

    # ── Transform (outlier detection) ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Outlier detection ...")
    experiments = ordered_experiments(combined["experiment"].unique())

    annotated_dfs: list[pd.DataFrame] = []
    for experiment in experiments:
        edf = combined[combined["experiment"] == experiment].copy()
        overrides = EXPERIMENT_OVERRIDES.get(experiment, {})
        edf = detect_outliers(edf, **overrides)
        clean = edf[~edf["outlier"]]
        print(f"  {experiment}: {len(clean)}/{len(edf)} clean")
        annotated_dfs.append(edf)

    annotated = pd.concat(annotated_dfs, ignore_index=True)
    assert len(annotated) == len(
        combined
    ), f"Row count mismatch: annotated={len(annotated)}, extract={len(combined)}"

    transform_csv = TRANSFORM_DIR / "isoflops.csv"
    annotated[[c for c in ANNOTATED_SCHEMA_COLS if c != "provenance"]].to_csv(
        transform_csv, index=False
    )
    print(f"  wrote {len(annotated):>5d} rows -> {transform_csv}")

    # ── Parabola fits (on all data, used by raw plot) ────────────────────
    print(f"\n{'─' * 60}")
    print("Fitting parabolas ...")
    fits = fit_parabolas(combined)

    # ── Visualize ────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    raw_fig = TRANSFORM_DIR / "isoflops_raw.png"
    plot_isoflops(combined, fits, raw_fig)
    print(f"  wrote {raw_fig}")

    akima_fig = TRANSFORM_DIR / "isoflops_akima.png"
    plot_isoflops_akima(annotated, akima_fig)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Summary:")
    for source_name, group in combined.groupby("source"):
        experiments = group["experiment"].unique()
        print(f"  {source_name}: {len(group)} rows, experiments={list(experiments)}")
    print(f"  TOTAL: {len(combined)} rows")
