"""Pipeline entry point: extract, transform, and visualize isoflop data.

Usage:
    uv run python -m scaling_law_analysis.data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scaling_law_analysis.data import DATA_DIR
from scaling_law_analysis.data.extract import FETCH_FUNCTIONS
from scaling_law_analysis.data.transform import (
    SCHEMA_COLS,
    assert_unique,
    fit_parabolas,
)
from scaling_law_analysis.data.visualize import plot_isoflops


def _write_source(df: pd.DataFrame, source: str) -> Path:
    """Write per-source CSV to data/isoflops/<source>/isoflops.csv."""
    out_dir = DATA_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "isoflops.csv"
    df[SCHEMA_COLS].to_csv(out_path, index=False)
    print(f"  wrote {len(df):>5d} rows -> {out_path}")
    return out_path


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    for source_name, fetch_fn in FETCH_FUNCTIONS:
        print(f"\n{'─' * 60}")
        df = fetch_fn()
        _write_source(df, source_name)
        all_dfs.append(df)

    # Aggregate (without provenance — per-source CSVs retain it)
    print(f"\n{'─' * 60}")
    print("Aggregating all sources ...")
    combined = pd.concat(all_dfs, ignore_index=True)
    assert_unique(combined, "aggregate")

    # Parabola fits (used by plot)
    print(f"\n{'─' * 60}")
    print("Fitting parabolas ...")
    fits = fit_parabolas(combined)

    agg_cols = [c for c in SCHEMA_COLS if c != "provenance"]
    out_path = DATA_DIR / "isoflops.csv"
    combined[agg_cols].to_csv(out_path, index=False)
    print(f"  wrote {len(combined):>5d} rows -> {out_path}")

    # Plot
    print(f"\n{'─' * 60}")
    fig_path = DATA_DIR / "isoflops.png"
    plot_isoflops(combined, fits, fig_path)
    print(f"  wrote {fig_path}")

    # Summary
    print(f"\n{'─' * 60}")
    print("Summary:")
    for source_name, group in combined.groupby("source"):
        experiments = group["experiment"].unique()
        print(f"  {source_name}: {len(group)} rows, experiments={list(experiments)}")
    print(f"  TOTAL: {len(combined)} rows")


if __name__ == "__main__":
    main()
