"""Schema definition, validation, and outlier detection for isoflop data."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import t as t_dist  # pyrefly: ignore

from scaling_law_analysis.data.schema import (
    UNIQUE_KEY,
    IsoFlopRecord,
    OutlierReason,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def experiment_name(
    source: str,
    dataset: str,
    model: str,
    condition: str | None = None,
) -> str:
    """Build experiment name from constituent parts, deduplicating adjacent repeats.

    For example, ``("llama_3", "llama_3", "llama_3", "exp_loss")`` becomes
    ``"llama_3__exp_loss"`` instead of ``"llama_3__llama_3__llama_3__exp_loss"``.
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
    "llama_3__exp_loss",
    "llama_3__raw_loss",
    "marin_202603__comma__llama_2",
    "marin_202603__dclm__llama_2",
    "marin_202603__nemotron__llama_2",
    "misfitting__fineweb_c4__transformer",
]

EXPERIMENT_DISPLAY_NAMES: dict[str, str] = {
    "epochai_chinchilla__massivetext__chinchilla": "Epoch AI / Chinchilla",
    "ml_scalefit__massivetext__chinchilla": "ML-Scalefit / Chinchilla",
    "llama_3__exp_loss": "Llama 3 (exp loss)",
    "llama_3__raw_loss": "Llama 3 (raw loss)",
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


# ── Pre-fit outlier detection ────────────────────────────────────────────────

# Detection threshold constants
MIN_BUDGET_POINTS = 6
LOO_ZSCORE_THRESHOLD = 6.0
NEAR_DUP_LOG_TOL = 0.01  # relative tolerance on log(N) for near-duplicate binning
# Confidence level for the curvature parameter CI.  Higher → wider CI → more
# budgets flagged as having insignificant curvature.  Set to 0.0 to disable.
CURVATURE_CI = 0.95
# Fractional margin added to the off-center symmetric radius.  A value of 0.25
# means a point can be up to 25% beyond the min half-distance and still be kept.
OFF_CENTER_MARGIN = 0.50

# Per-experiment overrides for outlier detection thresholds.
# Keys: experiment name → dict with optional "min_budget_points" and/or "loo_zscore_threshold".
EXPERIMENT_OVERRIDES: dict[str, dict] = {
    # Effectively disables LOO spline outlier detection for misfitting.
    "misfitting__fineweb_c4__transformer": {"loo_zscore_threshold": 100.0},
}


def detect_outliers(
    edf: pd.DataFrame,
    *,
    min_budget_points: int = MIN_BUDGET_POINTS,
    loo_zscore_threshold: float = LOO_ZSCORE_THRESHOLD,
    curvature_ci: float = CURVATURE_CI,
    off_center_margin: float = OFF_CENTER_MARGIN,
) -> pd.DataFrame:
    """Pre-fit outlier detection. Adds ``outlier`` and ``reason`` columns.

    Stage 0a — exact duplicate params: keep point closest to nominal budget.
    Stage 0b — near-duplicate params: same, within log(N) tolerance.
    Stage 1  — too few points: budgets with < *min_budget_points* are flagged.
    Stage 2a — negative curvature: quadratic fit log(L) ~ log(N)^2 with a <= 0.
    Stage 2b — weak curvature: CI for quadratic coefficient includes zero at
               *curvature_ci* confidence level (0.0 to disable).
    Stage 3  — Akima LOO: leave-one-out spline residuals, experiment-wide MAD.
    Stage 4  — off-center: trim to symmetric window around parabola minimum,
               with *off_center_margin* fractional slack (0.25 = 25% beyond).
    Stage 5  — post-QC too-few-points recheck (must be last).
    """
    R = OutlierReason

    edf = edf.copy()
    edf["outlier"] = pd.Series(False, index=edf.index, dtype=bool)
    edf["reason"] = R.NONE

    budgets = edf["budget"].unique()

    # Stage 0a: exact duplicate params within a budget — keep the point whose
    # implied compute (6·N·D) is closest to the nominal budget, ties by loss.
    for budget in budgets:
        mask = edf["budget"] == budget
        bdf = edf.loc[mask]
        dup_mask = bdf.duplicated(subset="params", keep=False)
        if dup_mask.any():
            for _, grp in bdf[dup_mask].groupby("params"):
                implied_c = 6 * grp["params"] * grp["tokens"]
                c_err = (implied_c - budget).abs()
                keep_idx = (
                    grp.assign(_c_err=c_err).sort_values(["_c_err", "loss"]).index[0]
                )
                flag_idx = grp.index.difference([keep_idx])
                edf.loc[flag_idx, "outlier"] = True
                edf.loc[flag_idx, "reason"] = R.EXACT_DUP

    # Stage 0b: near-duplicate params within a budget — keep the point whose
    # implied compute (6·N·D) is closest to the nominal budget.  Ties broken
    # by lowest loss.  Two points are "near-duplicates" if their log(N) values
    # differ by less than NEAR_DUP_LOG_TOL.  Only considers unflagged rows.
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        bdf = edf.loc[mask].sort_values("params")
        log_n = np.log(bdf["params"].to_numpy())
        indices = bdf.index.tolist()
        # Greedy binning: walk sorted log(N), start a new bin when gap > tol
        bins: list[list[int]] = []
        current_bin: list[int] = [0]
        for i in range(1, len(log_n)):
            if log_n[i] - log_n[current_bin[0]] <= NEAR_DUP_LOG_TOL:
                current_bin.append(i)
            else:
                bins.append(current_bin)
                current_bin = [i]
        bins.append(current_bin)
        for b in bins:
            if len(b) < 2:
                continue
            grp_idx = [indices[i] for i in b]
            grp = edf.loc[grp_idx]
            implied_c = 6 * grp["params"] * grp["tokens"]
            c_err = (implied_c - budget).abs()
            # Keep the point closest to nominal budget; break ties by loss
            keep_idx = grp.assign(_c_err=c_err).sort_values(["_c_err", "loss"]).index[0]
            flag_idx = [ix for ix in grp_idx if ix != keep_idx]
            edf.loc[flag_idx, "outlier"] = True
            edf.loc[flag_idx, "reason"] = R.DUP_PARAMS

    # Stage 1: too few clean points
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        if mask.sum() < min_budget_points:
            budget_mask = edf["budget"] == budget
            # Only flag rows not already flagged by an earlier stage
            unflagged = budget_mask & ~edf["outlier"]
            edf.loc[unflagged, "outlier"] = True
            edf.loc[unflagged, "reason"] = R.TOO_FEW

    # Stage 2a: negative curvature (only clean rows in surviving budgets)
    surviving_budgets = [
        b for b in budgets if not edf.loc[edf["budget"] == b, "outlier"].all()
    ]
    for budget in surviving_budgets:
        clean = edf.loc[(edf["budget"] == budget) & ~edf["outlier"]]
        log_n = np.log(clean["params"].to_numpy())
        log_l = np.log(clean["loss"].to_numpy())
        if len(log_n) >= 3:
            coeffs = np.polyfit(log_n, log_l, 2)
            if coeffs[0] <= 0:
                unflagged = (edf["budget"] == budget) & ~edf["outlier"]
                edf.loc[unflagged, "outlier"] = True
                edf.loc[unflagged, "reason"] = R.NEG_CURVATURE

    # Stage 2b: weak curvature — CI for the quadratic coefficient `a` includes
    # zero at the requested confidence level.  The point estimate â may be
    # positive (passed stage 2a) but statistically indistinguishable from zero,
    # meaning the isoflop curve is too flat to reliably locate an optimum.
    # Skipped when curvature_ci <= 0.
    if curvature_ci > 0:
        surviving_budgets = [
            b for b in budgets if not edf.loc[edf["budget"] == b, "outlier"].all()
        ]
        for budget in surviving_budgets:
            clean = edf.loc[(edf["budget"] == budget) & ~edf["outlier"]]
            log_n = np.log(clean["params"].to_numpy())
            log_l = np.log(clean["loss"].to_numpy())
            n_pts = len(log_n)
            if n_pts < 4:
                # Need at least 4 points for df = n-3 >= 1
                continue
            coeffs, cov = np.polyfit(log_n, log_l, 2, cov=True)
            a_hat = coeffs[0]
            if a_hat <= 0:
                # Already caught by stage 2a
                continue
            se_a = float(np.sqrt(cov[0, 0]))
            df = n_pts - 3
            t_crit = float(t_dist.ppf((1 + curvature_ci) / 2, df))
            ci_lower = a_hat - t_crit * se_a
            if ci_lower <= 0:
                unflagged = (edf["budget"] == budget) & ~edf["outlier"]
                edf.loc[unflagged, "outlier"] = True
                edf.loc[unflagged, "reason"] = R.WEAK_CURVATURE

    # Stage 3: Akima LOO (only clean rows in surviving budgets)
    surviving_budgets = [
        b for b in budgets if not edf.loc[edf["budget"] == b, "outlier"].all()
    ]
    # Collect LOO residuals for all surviving (clean) points
    loo_residuals: dict[int, float] = {}  # index -> residual
    for budget in surviving_budgets:
        clean = edf.loc[(edf["budget"] == budget) & ~edf["outlier"]].sort_values(
            "params"
        )
        log_n = np.log(clean["params"].to_numpy())
        loss = clean["loss"].to_numpy()
        indices = clean.index.tolist()
        for i in range(len(indices)):
            rest = np.delete(np.arange(len(indices)), i)
            if len(rest) < 2:
                continue
            log_n_rest = log_n[rest]
            l_rest = loss[rest]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interp = Akima1DInterpolator(log_n_rest, l_rest)
                interp.extrapolate = True
            predicted = float(interp(log_n[i], extrapolate=True))
            loo_residuals[indices[i]] = loss[i] - predicted

    if loo_residuals:
        resid_arr = np.array(list(loo_residuals.values()))
        med = np.median(resid_arr)
        mad = np.median(np.abs(resid_arr - med))
        if mad > 0:
            for ix, r in loo_residuals.items():
                z = 0.6745 * abs(r - med) / mad
                if z > loo_zscore_threshold:
                    edf.loc[ix, "outlier"] = True
                    edf.loc[ix, "reason"] = R.SPLINE

    # Stage 4: off-center — trim to a symmetric window around the parabola
    # minimum N*.  For each budget, fit a parabola to clean points in log-log
    # space, find N*, then compute the log-distance from N* to the nearest
    # clean point on each side.  The shorter side defines the symmetric radius;
    # any clean points beyond that radius are flagged.  If all clean points
    # fall on one side of N* (minimum is extrapolated), flag the entire budget.
    surviving_budgets = [
        b for b in budgets if not edf.loc[edf["budget"] == b, "outlier"].all()
    ]
    for budget in surviving_budgets:
        clean = edf.loc[(edf["budget"] == budget) & ~edf["outlier"]]
        log_n = np.log(clean["params"].to_numpy())
        log_l = np.log(clean["loss"].to_numpy())
        if len(log_n) < 3:
            continue
        coeffs = np.polyfit(log_n, log_l, 2)
        if coeffs[0] <= 0:
            # No valid minimum (negative/zero curvature) — skip
            continue
        log_n_star = -coeffs[1] / (2 * coeffs[0])
        left = log_n[log_n <= log_n_star]
        right = log_n[log_n >= log_n_star]
        if len(left) == 0 or len(right) == 0:
            # All points on one side — flag entire budget
            unflagged = (edf["budget"] == budget) & ~edf["outlier"]
            edf.loc[unflagged, "outlier"] = True
            edf.loc[unflagged, "reason"] = R.OFF_CENTER
            continue
        # Symmetric radius = min distance from N* to the nearest edge,
        # plus a fractional margin to allow slight overshoot.
        radius = min(log_n_star - left.min(), right.max() - log_n_star)
        radius *= 1.0 + off_center_margin
        outside = (log_n < log_n_star - radius) | (log_n > log_n_star + radius)
        if outside.any():
            flag_idx = clean.index[outside]
            edf.loc[flag_idx, "outlier"] = True
            edf.loc[flag_idx, "reason"] = R.OFF_CENTER

    # Stage 5: post-QC too-few-points recheck.  Earlier stages may have reduced
    # surviving budgets below the minimum.  This MUST be the last stage so that
    # it catches any budget left under-supported after all other filters.
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        if 0 < mask.sum() < min_budget_points:
            unflagged = (edf["budget"] == budget) & ~edf["outlier"]
            edf.loc[unflagged, "outlier"] = True
            edf.loc[unflagged, "reason"] = R.TOO_FEW_POST_QC

    # Invariant checks
    assert (
        edf["outlier"].notna().all()
    ), "outlier column has NA values"  # pyrefly: ignore
    assert (
        edf["reason"].notna().all()  # pyrefly: ignore
        and (edf["reason"] != "").all()  # pyrefly: ignore
    ), "reason column has empty/NA values"
    bad_clean = edf.loc[~edf["outlier"] & (edf["reason"] != R.NONE)]
    assert (
        bad_clean.empty
    ), f"outlier=False with non-none reason:\n{bad_clean[['outlier', 'reason']]}"
    bad_flagged = edf.loc[edf["outlier"] & (edf["reason"] == R.NONE)]
    assert (
        bad_flagged.empty
    ), f"outlier=True with reason=none:\n{bad_flagged[['outlier', 'reason']]}"

    return edf
