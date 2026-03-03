"""Shared generic utilities for the scaling_law_analysis package."""

import numpy as np


def check_design_matrix(
    design_matrix: np.ndarray,
    *,
    min_samples: int = 4,
    max_condition_number: float = 1e12,
    exception_cls: type[Exception] = ValueError,
) -> None:
    """Check design matrix sample count, rank, and conditioning."""
    n_samples, _ = design_matrix.shape
    if n_samples < min_samples:
        raise exception_cls(
            f"Design matrix requires at least {min_samples} samples, got {n_samples}."
        )

    singular_values = np.linalg.svd(design_matrix, compute_uv=False)
    smallest_sv = float(singular_values[-1]) if len(singular_values) else 0.0
    condition_number = (
        np.inf if smallest_sv <= 0.0 else float(singular_values[0] / smallest_sv)
    )
    if condition_number > max_condition_number:
        raise exception_cls(
            f"Design matrix is ill-conditioned: cond={condition_number:.3e} "
            f"(max={max_condition_number:.3e})."
        )
