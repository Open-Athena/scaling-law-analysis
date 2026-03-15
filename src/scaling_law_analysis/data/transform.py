"""Schema definition, validation, and outlier detection for isoflop data."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import t as t_dist  # pyrefly: ignore

from scaling_law_analysis.data.schema import (
    UNIQUE_KEY,
    Experiment,
    IsoFlopRecord,
    OutlierReason,
    QCStage,
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

EXPERIMENT_ORDER: list[str] = [e.value for e in Experiment]

EXPERIMENT_DISPLAY_NAMES: dict[str, str] = {
    Experiment.EPOCHAI_CHINCHILLA: "Epoch AI / Chinchilla",
    Experiment.ML_SCALEFIT_CHINCHILLA: "ML-Scalefit / Chinchilla",
    Experiment.LLAMA3_EXP_LOSS: "Llama 3 (exp loss)",
    Experiment.LLAMA3_RAW_LOSS: "Llama 3",
    Experiment.MARIN_COMMA: "Marin / CoMMA",
    Experiment.MARIN_DCLM: "Marin / DCLM",
    Experiment.MARIN_NEMOTRON: "Marin / Nemotron",
    Experiment.MISFITTING_FINEWEB: "Misfitting / FineWeb-C4",
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

# Detection threshold constants (defaults for QCConfig)
MIN_BUDGET_POINTS = 6
LOO_ZSCORE_THRESHOLD = 6.0
NEAR_DUP_LOG_TOL = 0.01  # relative tolerance on log(N) for near-duplicate binning
# Confidence level for the curvature parameter CI.  Higher → wider CI → more
# budgets flagged as having insignificant curvature.  Set to 0.0 to disable.
CURVATURE_CI = 0.95
# Fractional margin added to the off-center symmetric radius.  A value of 0.25
# means a point can be up to 25% beyond the min half-distance and still be kept.
OFF_CENTER_MARGIN = 1.50


@dataclass(frozen=True)
class QCConfig:
    """Threshold configuration for outlier detection."""

    min_budget_points: int = MIN_BUDGET_POINTS
    loo_zscore_threshold: float = LOO_ZSCORE_THRESHOLD
    near_dup_log_tol: float = NEAR_DUP_LOG_TOL
    curvature_ci: float = CURVATURE_CI
    off_center_margin: float = OFF_CENTER_MARGIN


# Mapping from each QCStage to the OutlierReason(s) it produces.
STAGE_REASONS: dict[QCStage, list[OutlierReason]] = {
    QCStage.DEDUP: [OutlierReason.EXACT_DUP, OutlierReason.DUP_PARAMS],
    QCStage.TOO_FEW: [OutlierReason.TOO_FEW],
    QCStage.SPLINE: [OutlierReason.SPLINE],
    QCStage.CURVATURE: [OutlierReason.NEG_CURVATURE, OutlierReason.WEAK_CURVATURE],
    QCStage.OFF_CENTER: [OutlierReason.OFF_CENTER],
    QCStage.POST_QC: [OutlierReason.TOO_FEW_POST_QC],
}

# Canonical ordering of QC stages.
DEFAULT_STAGES: list[QCStage] = [
    QCStage.DEDUP,
    QCStage.TOO_FEW,
    QCStage.OFF_CENTER,
    QCStage.SPLINE,
    QCStage.CURVATURE,
    QCStage.POST_QC,
]

# Sync check: every non-NONE OutlierReason must be covered by exactly one stage.
_covered = {r for reasons in STAGE_REASONS.values() for r in reasons}
_all_reasons = {r for r in OutlierReason if r != OutlierReason.NONE}
assert _covered == _all_reasons, (
    f"STAGE_REASONS coverage mismatch: "
    f"missing={_all_reasons - _covered}, extra={_covered - _all_reasons}"
)

# Per-experiment overrides for outlier detection thresholds.
EXPERIMENT_OVERRIDES: dict[str, QCConfig] = {
    # Effectively disables LOO spline outlier detection for misfitting.
    Experiment.MISFITTING_FINEWEB: QCConfig(loo_zscore_threshold=100.0),
    # Lower LOO threshold for Marin DCLM to catch high-variance outliers.
    Experiment.MARIN_DCLM: QCConfig(loo_zscore_threshold=4.5),
}


# ── Stage functions ──────────────────────────────────────────────────────────
# Uniform signature: (edf, budgets, cfg) -> None  (mutates edf in place)


def _stage_dedup(edf: pd.DataFrame, budgets: np.ndarray, cfg: QCConfig) -> None:
    R = OutlierReason

    # Sub-stage 0a: exact duplicate params within a budget — keep the point whose
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

    # Sub-stage 0b: near-duplicate params within a budget — keep the point whose
    # implied compute (6·N·D) is closest to the nominal budget.  Ties broken
    # by lowest loss.  Two points are "near-duplicates" if their log(N) values
    # differ by less than near_dup_log_tol.  Only considers unflagged rows.
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        bdf = edf.loc[mask].sort_values("params")
        log_n = np.log(bdf["params"].to_numpy())
        indices = bdf.index.tolist()
        # Greedy binning: walk sorted log(N), start a new bin when gap > tol
        bins: list[list[int]] = []
        current_bin: list[int] = [0]
        for i in range(1, len(log_n)):
            if log_n[i] - log_n[current_bin[0]] <= cfg.near_dup_log_tol:
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


def _stage_too_few(edf: pd.DataFrame, budgets: np.ndarray, cfg: QCConfig) -> None:
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        if mask.sum() < cfg.min_budget_points:
            budget_mask = edf["budget"] == budget
            # Only flag rows not already flagged by an earlier stage
            unflagged = budget_mask & ~edf["outlier"]
            edf.loc[unflagged, "outlier"] = True
            edf.loc[unflagged, "reason"] = OutlierReason.TOO_FEW


def _stage_spline(edf: pd.DataFrame, budgets: np.ndarray, cfg: QCConfig) -> None:
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
                if z > cfg.loo_zscore_threshold:
                    edf.loc[ix, "outlier"] = True
                    edf.loc[ix, "reason"] = OutlierReason.SPLINE


def _stage_curvature(edf: pd.DataFrame, budgets: np.ndarray, cfg: QCConfig) -> None:
    R = OutlierReason

    # Sub-stage 3a: negative curvature (only clean rows in surviving budgets)
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

    # Sub-stage 3b: weak curvature — CI for the quadratic coefficient `a`
    # includes zero at the requested confidence level.  Skipped when
    # curvature_ci <= 0.
    if cfg.curvature_ci > 0:
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
                # Already caught by sub-stage 3a
                continue
            se_a = float(np.sqrt(cov[0, 0]))
            df = n_pts - 3
            t_crit = float(t_dist.ppf((1 + cfg.curvature_ci) / 2, df))
            ci_lower = a_hat - t_crit * se_a
            if ci_lower <= 0:
                unflagged = (edf["budget"] == budget) & ~edf["outlier"]
                edf.loc[unflagged, "outlier"] = True
                edf.loc[unflagged, "reason"] = R.WEAK_CURVATURE


def _stage_off_center(edf: pd.DataFrame, budgets: np.ndarray, cfg: QCConfig) -> None:
    R = OutlierReason
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
        radius *= 1.0 + cfg.off_center_margin
        outside = (log_n < log_n_star - radius) | (log_n > log_n_star + radius)
        if outside.any():
            flag_idx = clean.index[outside]
            edf.loc[flag_idx, "outlier"] = True
            edf.loc[flag_idx, "reason"] = R.OFF_CENTER


def _stage_post_qc(edf: pd.DataFrame, budgets: np.ndarray, cfg: QCConfig) -> None:
    for budget in budgets:
        mask = (edf["budget"] == budget) & ~edf["outlier"]
        if 0 < mask.sum() < cfg.min_budget_points:
            unflagged = (edf["budget"] == budget) & ~edf["outlier"]
            edf.loc[unflagged, "outlier"] = True
            edf.loc[unflagged, "reason"] = OutlierReason.TOO_FEW_POST_QC


_STAGE_DISPATCH: dict[QCStage, Callable[[pd.DataFrame, np.ndarray, QCConfig], None]] = {
    QCStage.DEDUP: _stage_dedup,
    QCStage.TOO_FEW: _stage_too_few,
    QCStage.SPLINE: _stage_spline,
    QCStage.CURVATURE: _stage_curvature,
    QCStage.OFF_CENTER: _stage_off_center,
    QCStage.POST_QC: _stage_post_qc,
}


def detect_outliers(
    edf: pd.DataFrame,
    *,
    stages: list[QCStage] | None = None,
    cfg: QCConfig | None = None,
) -> pd.DataFrame:
    """Pre-fit outlier detection. Adds ``outlier`` and ``reason`` columns.

    Parameters
    ----------
    stages : list of QCStage, optional
        Ordered list of QC stages to run.  Defaults to ``DEFAULT_STAGES``.
    cfg : QCConfig, optional
        Threshold configuration.  Defaults to ``QCConfig()`` (module defaults).
    """
    if stages is None:
        stages = DEFAULT_STAGES
    if cfg is None:
        cfg = QCConfig()

    R = OutlierReason

    edf = edf.copy()
    edf["outlier"] = pd.Series(False, index=edf.index, dtype=bool)
    edf["reason"] = R.NONE

    budgets = edf["budget"].unique()

    for stage in stages:
        _STAGE_DISPATCH[stage](edf, budgets, cfg)

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
