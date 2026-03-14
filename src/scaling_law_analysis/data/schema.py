"""Schema definitions for isoflop data."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
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


class OutlierReason(StrEnum):
    """Why a data point was flagged as an outlier (or not)."""

    NONE = "none"
    EXACT_DUP = "exact_duplicate"
    DUP_PARAMS = "near_duplicate_params"
    TOO_FEW = "too_few_points"
    NEG_CURVATURE = "negative_curvature"
    WEAK_CURVATURE = "weak_curvature"
    SPLINE = "spline_outlier"


class IsoFlopAnnotatedRecord(IsoFlopRecord):
    """Schema with outlier annotations appended."""

    outlier: bool = False
    reason: OutlierReason = OutlierReason.NONE


ANNOTATED_SCHEMA_COLS = SCHEMA_COLS + ["outlier", "reason"]
