"""Schema definition, validation, and outlier detection for isoflop data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

# ── Schema ───────────────────────────────────────────────────────────────────

UNIQUE_KEY: list[str] = ["source", "experiment", "tokens", "params", "budget"]


class IsoFlopRecord(BaseModel):
    """Validated schema for a single isoflop data point."""

    source: str
    provenance: str
    dataset: str
    model: str
    condition: str | None = None
    experiment: str
    tokens: float
    params: float
    budget: float
    loss: float

    @field_validator("tokens", "params", "budget", "loss")
    @classmethod
    def must_be_positive_finite(cls, v: float) -> float:
        if not (v > 0 and np.isfinite(v)):
            raise ValueError(f"must be positive and finite, got {v}")
        return v

    @field_validator("dataset", "model", "experiment", "condition")
    @classmethod
    def must_be_lowercase(cls, v: str | None) -> str | None:
        if v is not None and v != v.lower():
            raise ValueError(f"must be lowercase, got {v!r}")
        return v

    @field_validator("experiment")
    @classmethod
    def must_be_snake_case(cls, v: str) -> str:
        if " " in v:
            raise ValueError(f"experiment must be snake_case, got {v!r}")
        if v.startswith(".") or v.endswith(".") or ".." in v:
            raise ValueError(f"experiment has invalid dot placement, got {v!r}")
        return v


SCHEMA_COLS = list(IsoFlopRecord.model_fields.keys())

# ── Helpers ──────────────────────────────────────────────────────────────────


def experiment_name(
    source: str,
    dataset: str,
    model: str,
    condition: str | None = None,
) -> str:
    """Build experiment name from constituent parts, deduplicating adjacent repeats.

    For example, ``("llama3", "llama_3", "llama_3", "exp_loss")`` becomes
    ``"llama3__llama_3__exp_loss"`` instead of ``"llama3__llama_3__llama_3__exp_loss"``.
    """
    parts = [source, dataset, model]
    if condition is not None:
        parts.append(condition)
    # Deduplicate adjacent identical parts
    deduped = [parts[0]]
    for p in parts[1:]:
        if p != deduped[-1]:
            deduped.append(p)
    return "__".join(deduped)


# ── Canonical experiment ordering & display names ────────────────────────────

EXPERIMENT_ORDER: list[str] = [
    "epochai_chinchilla__massivetext__chinchilla",
    "ml_scalefit__massivetext__chinchilla",
    "llama3__llama_3__exp_loss",
    "llama3__llama_3__raw_loss",
    "marin_202603__comma__llama_2",
    "marin_202603__dclm__llama_2",
    "marin_202603__nemotron__llama_2",
    "misfitting__fineweb_c4__transformer",
]

EXPERIMENT_DISPLAY_NAMES: dict[str, str] = {
    "epochai_chinchilla__massivetext__chinchilla": "Epoch AI / Chinchilla",
    "ml_scalefit__massivetext__chinchilla": "ML-Scalefit / Chinchilla",
    "llama3__llama_3__exp_loss": "Llama 3 (exp loss)",
    "llama3__llama_3__raw_loss": "Llama 3 (raw loss)",
    "marin_202603__comma__llama_2": "Marin / CoMMA",
    "marin_202603__dclm__llama_2": "Marin / DCLM",
    "marin_202603__nemotron__llama_2": "Marin / Nemotron",
    "misfitting__fineweb_c4__transformer": "Misfitting / FineWeb-C4",
}


def ordered_experiments(experiments: Iterable[str]) -> list[str]:
    """Return experiments in canonical order, raising on unknown names."""
    exp_set = set(experiments)
    unknown = exp_set - set(EXPERIMENT_ORDER)
    if unknown:
        raise ValueError(
            f"Unknown experiment(s): {sorted(unknown)}. "
            f"Add them to EXPERIMENT_ORDER and EXPERIMENT_DISPLAY_NAMES in transform.py."
        )
    return [e for e in EXPERIMENT_ORDER if e in exp_set]


def display_name(experiment: str) -> str:
    """Return the human-readable display name for an experiment."""
    if experiment not in EXPERIMENT_DISPLAY_NAMES:
        raise ValueError(
            f"Unknown experiment: {experiment!r}. "
            f"Add it to EXPERIMENT_DISPLAY_NAMES in transform.py."
        )
    return EXPERIMENT_DISPLAY_NAMES[experiment]


def make_df(
    *,
    source: str,
    provenance: str,
    dataset: str,
    model: str,
    condition: str | None = None,
    tokens: np.ndarray,
    params: np.ndarray,
    budget: np.ndarray,
    loss: np.ndarray,
) -> pd.DataFrame:
    """Build a standardized DataFrame with per-row pydantic validation."""
    exp = experiment_name(source, dataset, model, condition)
    tokens = np.asarray(tokens, dtype=float)
    params = np.asarray(params, dtype=float)
    budget = np.asarray(budget, dtype=float)
    loss = np.asarray(loss, dtype=float)

    # Validate every row through the pydantic model
    records: list[dict] = []
    for i in range(len(tokens)):
        record = IsoFlopRecord(
            source=source,
            provenance=provenance,
            dataset=dataset,
            model=model,
            condition=condition,
            experiment=exp,
            tokens=float(tokens[i]),
            params=float(params[i]),
            budget=float(budget[i]),
            loss=float(loss[i]),
        )
        records.append(record.model_dump())

    df = pd.DataFrame(records)
    assert_unique(df, source)
    return df


def assert_unique(df: pd.DataFrame, label: str) -> None:
    """Assert uniqueness on the composite key."""
    dupes = df.duplicated(subset=UNIQUE_KEY, keep=False)
    if dupes.any():
        examples = df.loc[dupes, UNIQUE_KEY].head(10).to_string()
        raise AssertionError(
            f"[{label}] {dupes.sum()} duplicate rows on {UNIQUE_KEY}:\n{examples}"
        )


# ── Parabola fitting ─────────────────────────────────────────────────────────


def fit_parabolas(
    df: pd.DataFrame,
) -> dict[tuple[str, str, float], dict]:
    """Fit log-log parabolas per (source, experiment, budget).

    For each group, fits log(loss) = a*log(N)^2 + b*log(N) + c.
    A fit is marked valid when there are >= 3 points and curvature is
    positive (a > 0, i.e. the parabola opens upward).

    Returns:
        fits dict keyed by (source, experiment, budget) with keys:
          coeffs, valid, n_min, n_max
    """
    fits: dict[tuple[str, str, float], dict] = {}

    for (source, experiment), sedf in df.groupby(["source", "experiment"]):
        for budget, bdf in sedf.groupby("budget"):
            log_n = np.log(bdf["params"].to_numpy())
            log_l = np.log(bdf["loss"].to_numpy())
            if len(log_n) >= 3:
                coeffs = np.polyfit(log_n, log_l, 2)
                valid = coeffs[0] > 0  # positive (upward) curvature
            else:
                coeffs = None
                valid = False
            fits[(str(source), str(experiment), float(budget))] = {  # pyrefly: ignore
                "coeffs": coeffs,
                "valid": valid,
                "n_min": bdf["params"].min(),
                "n_max": bdf["params"].max(),
            }

    n_valid = sum(1 for f in fits.values() if f["valid"])
    print(f"  {n_valid}/{len(fits)} budgets have valid parabola fits")

    return fits
