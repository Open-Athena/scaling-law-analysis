# Refactoring Plan: `chinchilla.py` Standardization

## Overview

Standardize all four fitting methods to a common `(N, D, L, ...)` interface, unify
result types with scaling exponent properties, eliminate exp7-specific wrappers, and
clean up internal utilities.

**No experiments or figures will be re-run.** Code changes only.

---

## Phase 1: Result Type Redesign

### 1a. `ParabolaFitResult` — make frozen, add `N_opt` method

```python
@dataclass(frozen=True)
class ParabolaFitResult:
    a: float                                  # N* exponent: β/(α+β)
    b: float                                  # D* exponent: α/(α+β)
    a_intercept: float                        # log₁₀ intercept of N* fit
    b_intercept: float                        # log₁₀ intercept of D* fit
    parabola_fits_N: tuple[ParabolaFit, ...]   # frozen-compatible
    parabola_fits_D: tuple[ParabolaFit, ...]
    compute_budgets: np.ndarray
    N_opts: np.ndarray
    D_opts: np.ndarray

    def D_opt(self, C: float) -> float: ...   # existing
    def N_opt(self, C: float) -> float: ...   # new, symmetric with D_opt
```

### 1b. `SurfaceFitResult` — make frozen, add `.a` and `.b` properties

```python
@dataclass(frozen=True)
class SurfaceFitResult:
    # ... existing fields unchanged ...

    @property
    def a(self) -> float:
        """N* scaling exponent: β/(α+β)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """D* scaling exponent: α/(α+β)."""
        return self.alpha / (self.alpha + self.beta)
```

This eliminates the need for exp7's `ExponentEstimate`, `_make_estimate`, and
`_result_to_estimate`.

---

## Phase 2: Grid Class Consolidation

| Current | New | Used by |
|---|---|---|
| `VPNLSInitGrid` | `ExponentGrid` | `fit_vpnls`, `_grid_search_2d` |
| `Approach3InitGrid` | `ParameterGrid` | `fit_approach3`, `fit_grid_search` |
| `DEFAULT_VPNLS_GRID` | `DEFAULT_EXPONENT_GRID` | |
| `DEFAULT_APPROACH3_GRID` | `DEFAULT_PARAMETER_GRID` | |
| `FINE_VPNLS_GRID` | `FINE_EXPONENT_GRID` | |

All references updated in one pass — no deprecated aliases.

---

## Phase 3: `fit_approach2` Refactoring

### 3a. New `fit_approach2` in `chinchilla.py`

```python
def fit_approach2(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,     # per-point compute budgets (same length as N, D, L)
) -> ParabolaFitResult:
```

- `C` is a 1D array with one entry per data point (same length as `N, D, L`).
- Groups are determined by unique values of `C`: `for c in np.unique(C): mask = C == c`.
- This makes no assumptions about equal group sizes.
- Fits parabolas per group, then power-law regression across unique budgets.
- Validates `len(N) == len(D) == len(L) == len(C)`.

### 3b. `fit_simulated_approach2` in `experiments/common.py`

```python
def fit_simulated_approach2(
    compute_budgets: np.ndarray,
    surface: LossSurface,
    *,
    drift_rate: float = 0.0,
    center_scale: float = 1.0,
    n_points: int = 15,
    log_range: float = 1.0,
) -> ParabolaFitResult:
```

- Contains the current `fit_approach2` simulation logic: generates isoflop samples,
  builds the per-point `C` array, then calls the new `fit_approach2(N, D, L, C)`.
- Lives alongside `sample_isoflop_data` and `SimulationConfig`.

---

## Phase 4: `fit_grid_search` Standardization

### 4a. Public `fit_grid_search` — promoted to primary method

```python
def fit_grid_search(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    grid: ParameterGrid | None = None,
) -> SurfaceFitResult:
```

- Takes raw `N, D` (converts to `log_N, log_D` internally).
- Uses `ParameterGrid` (renamed from `Approach3InitGrid`).
- Returns `SurfaceFitResult` (with `.a`, `.b` properties).
- Status is always `CONVERGED` (grid search is deterministic, no optimizer issues).

### 4b. Internal `_grid_search_5d` — raw helper

The vectorized Cartesian-product search logic stays as an internal function returning
the raw `[E, A, B, α, β]` array, used by both public `fit_grid_search` and
`fit_approach3`:

```python
def _grid_search_5d(
    grid: ParameterGrid,
    log_N: np.ndarray,
    log_D: np.ndarray,
    L: np.ndarray,
) -> tuple[np.ndarray, float]:
    # Returns (best_params, best_rss)
```

### 4c. Internal `_grid_search_2d` — private, standardized naming

Rename existing `_grid_search` → `_grid_search_2d` for clarity alongside
`_grid_search_5d`. Same interface, just a clearer name. Used only by `fit_vpnls`.

---

## Phase 5: Internal Cleanup

### 5a. Shared input validation

Extract `_validate_ndl_inputs(N, D, L)` used by all surface-fitting methods:

```python
def _validate_ndl_inputs(
    N: np.ndarray, D: np.ndarray, L: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce N, D, L inputs. Returns (N, D, L) as float arrays."""
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    if not (len(N) == len(D) == len(L)):
        raise ValueError(...)
    _validate_positive_finite("N", N)
    _validate_positive_finite("D", D)
    if not np.all(np.isfinite(L)):
        raise ValueError(...)
    return N, D, L
```

### 5b. Keep `itertools.product` in `_cartesian_product`

The existing `_cartesian_product` implementation using `itertools.product` is
well-tested and correct. Keep it as-is. Keep `import itertools`.

### 5c. `SurfaceFitResult` → frozen

Change from `@dataclass` to `@dataclass(frozen=True)`. All fields are set at
construction time.

---

## Phase 6: File Organization in `chinchilla.py`

Sections in order, with banner comments:

```
# =============================================================================
# Fit status and exceptions
# =============================================================================

# =============================================================================
# Loss surface model
# =============================================================================

# =============================================================================
# IsoFLOP sampling
# =============================================================================

# =============================================================================
# Configuration: grids, bounds, optimizer options
# =============================================================================

# =============================================================================
# Shared utilities (validation, diagnostics, internal grid searches)
# =============================================================================

# =============================================================================
# Method: Approach 2 (Parabolic IsoFLOP → Power Law)
# =============================================================================

# =============================================================================
# Method: Grid Search (5D Exhaustive)
# =============================================================================

# =============================================================================
# Method: VPNLS (Variable Projection + NNLS)
# =============================================================================

# =============================================================================
# Method: Approach 3 (Direct 5-parameter L-BFGS-B)
# =============================================================================
```

---

## Phase 7: Caller Updates

| File | Changes |
|---|---|
| **`experiments/common.py`** | Add `fit_simulated_approach2`. Update imports for renamed grid classes. |
| **`article/figures.py`** | `fit_approach2` → `fit_simulated_approach2` (import from `experiments.common`), 5 call sites unchanged in shape. |
| **`exp1_empirical_error.py`** | `fit_approach2` → `fit_simulated_approach2`, import from `experiments.common`. |
| **`exp4_extrapolation_error.py`** | `fit_approach2` → `fit_simulated_approach2`, import from `experiments.common`. |
| **`exp6_analytical_error.py`** | `fit_approach2` → `fit_simulated_approach2`, import from `experiments.common`. |
| **`exp5_parameter_recovery.py`** | Update grid class name imports (`DEFAULT_APPROACH3_GRID` → `DEFAULT_PARAMETER_GRID`, `FINE_VPNLS_GRID` → `FINE_EXPONENT_GRID`). |
| **`exp7_exponent_inference.py`** | **Major refactor**: Remove local `fit_approach2`, `fit_approach3`, `fit_vpnls` wrappers. Remove `ExponentEstimate`, `_make_estimate`, `_result_to_estimate`. Use chinchilla methods directly. Access `.a` and `.b` on results. Adjust `_run_single_repeat` to call `fit_approach2(data.N, data.D, data.L, C_repeated)` with per-point C array, and `fit_approach3(data.N, data.D, data.L)` / `fit_vpnls(data.N, data.D, data.L)` directly, reading `.a` and `.b` off the results. |
| **`exp8_conditioning.py`** | Update grid class name imports. |
| **`tests/test_chinchilla.py`** | Update `fit_approach2` test to use new `(N, D, L, C)` signature. Add test for `fit_simulated_approach2` (import from `common`). Update grid class names in `fit_grid_search` tests — note `fit_grid_search` now takes `(N, D, L, *, grid=)` instead of individual grid kwargs, and returns `SurfaceFitResult` not `np.ndarray`. |

---

## Execution Order

1. **`chinchilla.py`**: All internal changes — grid renames, result type changes,
   `_validate_ndl_inputs`, `_grid_search` → `_grid_search_2d`, new `_grid_search_5d`,
   new `fit_approach2(N,D,L,C)`, promoted `fit_grid_search`, section reordering,
   frozen dataclasses, `.a`/`.b` properties
2. **`experiments/common.py`**: Add `fit_simulated_approach2`
3. **`article/figures.py`**: Update imports and call sites
4. **`exp1`**: Update
5. **`exp4`**: Update
6. **`exp6`**: Update
7. **`exp5`**: Update grid names
8. **`exp7`**: Major refactor — remove wrappers, use chinchilla directly
9. **`exp8`**: Update grid names
10. **`tests/test_chinchilla.py`**: Update test signatures and assertions

---

## DELETE THIS FILE after implementation is complete.
