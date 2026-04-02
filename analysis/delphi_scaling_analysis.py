"""Compare Llama 3 and Marin Delphi isoFLOP scaling law fits."""

from __future__ import annotations

import csv
import io
import textwrap
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from scaling_law_analysis.chinchilla import (
    ExponentGrid,
    FitStatus,
    LossSurface,
    ParabolaFitResult,
    SurfaceFitResult,
    fit_approach2,
)
from scaling_law_analysis.common import save_figure

# =============================================================================
# Analysis configuration
# =============================================================================

# -- Data sources --
LLAMA3_URL = (
    "https://raw.githubusercontent.com/eric-czech/llama3_isoflop_extraction/"
    "1bc1755b76e6ee55a911549c8ec52b71cb480320/isoflops_points.csv"
)
DELPHI_URL = (
    "https://gist.githubusercontent.com/eric-czech/35e3b493c5d6a01dcea2f8dba8708a98/"
    "raw/a67640529a78c5f040fab1ab7b14b8894d414340/marin_delphi_isoflop_records.csv"
)

# -- Output --
OUT_DIR = Path(__file__).parent / "results"

# -- Delphi metric to use --
DELPHI_METRIC = "eval/paloma/macro_loss"
METRIC_LABELS = {
    "eval/paloma/c4_en/bpb": "Paloma C4-EN BPB",
    "eval/paloma/macro_loss": "Paloma Macro Loss",
}
DELPHI_METRIC_LABEL = METRIC_LABELS[DELPHI_METRIC]


# -- VPNLS grid resolution for alpha/beta --
GRID_RESOLUTION = 0.001

# -- Approach 3 flop factor mode --
# True: learn per-budget k via OLS; False: fix k=6 (standard C=6ND)
LEARN_K = True

# -- Param inference --
# True: derive N = C/(6D) for Delphi (same as Llama 3); False: use true params
INFER_PARAMS = False

# -- Visualization --
# Show Approach 3 prediction curves in the isoflop plot
SHOW_A3_CURVES = False

# Show power law fit through A2 parabola minima
SHOW_MINIMA_PL = False

# =============================================================================
# Data loading
# =============================================================================

FINE_EXPONENT_GRID = ExponentGrid(
    alpha=np.arange(0.01, 1.0, GRID_RESOLUTION),
    beta=np.arange(0.01, 1.0, GRID_RESOLUTION),
)


@dataclass
class IsoFlopData:
    """Unified isoFLOP dataset: arrays of N, D, L, C (all same length)."""

    name: str
    N: np.ndarray  # params
    D: np.ndarray  # tokens
    L: np.ndarray  # loss
    C: np.ndarray  # compute budget (FLOPs)

    def __post_init__(self):
        n = len(self.N)
        assert len(self.D) == n and len(self.L) == n and len(self.C) == n
        assert np.all(np.isfinite(self.N)) and np.all(self.N > 0)
        assert np.all(np.isfinite(self.D)) and np.all(self.D > 0)
        assert np.all(np.isfinite(self.L)) and np.all(self.L > 0)
        assert np.all(np.isfinite(self.C)) and np.all(self.C > 0)


def _fetch_csv(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url) as resp:
        return list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))


def load_llama3() -> IsoFlopData:
    """Load Llama 3 isoFLOP data. N derived as C/(6D), raw loss used directly."""
    rows = _fetch_csv(LLAMA3_URL)
    C = np.array([float(r["compute_budget"]) for r in rows])
    D = np.array([float(r["training_tokens"]) for r in rows])
    L = np.array([float(r["validation_loss"]) for r in rows])
    N = C / (6.0 * D)
    return IsoFlopData(name="Llama 3", N=N, D=D, L=L, C=C)


def load_delphi(metric: str = DELPHI_METRIC) -> IsoFlopData:
    """Load Marin Delphi isoFLOP data, filtered by metric."""
    all_rows = _fetch_csv(DELPHI_URL)
    rows = [r for r in all_rows if r["metric"] == metric]
    assert len(rows) > 0, f"No rows for metric={metric!r}"
    D = np.array([float(r["tokens"]) for r in rows])
    L = np.array([float(r["value"]) for r in rows])
    C = np.array([float(r["flops"]) for r in rows])
    N = C / (6.0 * D) if INFER_PARAMS else np.array([float(r["params"]) for r in rows])
    n_budgets = len(np.unique(C))
    assert n_budgets == 7, f"Expected 7 unique budgets, got {n_budgets}"
    return IsoFlopData(name="Delphi", N=N, D=D, L=L, C=C)


# =============================================================================
# Fitting
# =============================================================================


def fit_vpnls(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    grid: ExponentGrid,
    fit_asymptote: bool,
) -> SurfaceFitResult:
    """VPNLS grid search: L = E + A/N^α + B/D^β (or E=0 when fit_asymptote=False).

    2D grid over (α, β), NNLS inner solve for (E, A, B) or (A, B) at each candidate.
    """
    from scipy.optimize import nnls

    log_N, log_D = np.log(N), np.log(D)
    n_linear = 3 if fit_asymptote else 2
    best_rss, best_ab, best_params = np.inf, (0.0, 0.0), np.zeros(n_linear)
    for alpha in grid.alpha:
        for beta in grid.beta:
            cols = [np.exp(-alpha * log_N), np.exp(-beta * log_D)]
            if fit_asymptote:
                cols.insert(0, np.ones(len(L)))
            X = np.column_stack(cols)
            params, rnorm = nnls(X, L)
            rss = rnorm**2
            if rss < best_rss:
                best_rss, best_ab, best_params = rss, (alpha, beta), params
    if fit_asymptote:
        E, A, B = best_params
    else:
        E, A, B = 0.0, best_params[0], best_params[1]
    return SurfaceFitResult(
        E=float(E),
        A=float(A),
        B=float(B),
        alpha=float(best_ab[0]),
        beta=float(best_ab[1]),
        residual_sum_squares=best_rss,
        n_points=len(N),
        method=f"vpnls-grid{'' if fit_asymptote else '-no-E'}",
        status=FitStatus.CONVERGED,
    )


def estimate_k_hat(data: IsoFlopData, surface: LossSurface) -> dict[float, float]:
    """Per-budget flop factor estimation: C = k*N*D.

    Given global fit L = E + A*N^(-alpha) + B*D^(-beta) with D = C/(k*N),
    estimates per-budget k via OLS on the linearized data term (kappa = k^beta).
    """
    E, A, B = surface.E, surface.A, surface.B
    alpha, beta = surface.alpha, surface.beta
    k_values: dict[float, float] = {}
    for c in np.unique(data.C):
        mask = data.C == c
        N, L = data.N[mask], data.L[mask]
        r = L - E - A * N ** (-alpha)
        g = B * N**beta / c**beta
        kappa = float(np.dot(r, g) / np.dot(g, g))
        assert kappa > 0, f"Non-positive kappa={kappa:.6f} at budget {c:.2e}"
        k_values[c] = kappa ** (1 / beta)
    return k_values


def get_k_values(data: IsoFlopData, surface: LossSurface) -> dict[float, float]:
    """Return per-budget k based on LEARN_K config."""
    if LEARN_K:
        return estimate_k_hat(data, surface)
    return {c: 6.0 for c in np.unique(data.C)}


def compute_rss_by_budget(
    data: IsoFlopData, surface: LossSurface, k_values: dict[float, float]
) -> dict[float, float]:
    """RSS per budget using provided k values."""
    rss: dict[float, float] = {}
    for c in np.unique(data.C):
        mask = data.C == c
        N, L = data.N[mask], data.L[mask]
        D = c / (k_values[c] * N)
        pred = surface.loss(N, D)  # pyrefly: ignore
        rss[c] = float(np.sum((L - pred) ** 2))
    return rss


@dataclass
class MinimaPowerLaw:
    """Power law fit to A2 parabola minima: L = x1 * D^x2 + x0."""

    x0: float  # offset (irreducible loss along minima)
    x1: float  # coefficient
    x2: float  # exponent
    rss: float


def _fit_power_law_to_minima(D: np.ndarray, L: np.ndarray) -> MinimaPowerLaw:
    """Fit L = x1 * D^x2 + x0 to (D, L) pairs."""

    def model(d: np.ndarray, x0: float, x1: float, x2: float) -> np.ndarray:
        return x1 * d**x2 + x0

    popt, _ = curve_fit(  # pyrefly: ignore
        model, D, L, p0=[L.min(), 1.0, -0.1], maxfev=10000
    )
    pred = model(D, *popt)
    rss = float(np.sum((L - pred) ** 2))
    return MinimaPowerLaw(x0=popt[0], x1=popt[1], x2=popt[2], rss=rss)


def fit_minima_power_law_a2(a2: ParabolaFitResult) -> MinimaPowerLaw:
    """Fit power law to A2 parabola vertices (D*, L_min)."""
    D = a2.D_opts
    L = np.array([f.L_min for f in a2.parabola_fits_D])
    return _fit_power_law_to_minima(D, L)


def d_opt_a3(surface: LossSurface, C: float, k: float) -> float:
    """Analytical D* for approach 3: D* = (1/G) * (C/k)^b."""
    return (1.0 / surface.G) * (C / k) ** surface.b


@dataclass
class DOptDeltas:
    """D* comparison between Approach 3 and Approach 2.

    delta% = (D*_A3 - D*_A2) / D*_A2 * 100
    Positive means Approach 3 infers more tokens than Approach 2.
    """

    pct: dict[float, float]  # signed % delta per budget
    raw: dict[float, float]  # raw token delta per budget
    mean_signed: float
    mean_unsigned: float
    median_signed: float
    median_unsigned: float
    var_unsigned: float
    var_signed: float
    iqr: float  # IQR of signed % deltas


def compute_dopt_deltas(results: "FitResults") -> DOptDeltas:
    surface = results.vpnls.to_loss_surface()
    budgets = results.a2.compute_budgets
    d_a2 = results.a2.D_opts
    pct: dict[float, float] = {}
    raw: dict[float, float] = {}
    for i, c in enumerate(budgets):
        d3 = d_opt_a3(surface, c, results.k_values[c])
        raw[c] = d3 - d_a2[i]
        pct[c] = raw[c] / d_a2[i] * 100.0
    vals = np.array(list(pct.values()))
    abs_vals = np.abs(vals)
    return DOptDeltas(
        pct=pct,
        raw=raw,
        mean_signed=float(vals.mean()),
        mean_unsigned=float(abs_vals.mean()),
        median_signed=float(np.median(vals)),
        median_unsigned=float(np.median(abs_vals)),
        var_unsigned=float(abs_vals.var()),
        var_signed=float(vals.var()),
        iqr=float(np.subtract(*np.percentile(vals, [75, 25]))),
    )


@dataclass
class FitResults:
    """VPNLS grid + Approach 2 + k results for one dataset."""

    data: IsoFlopData
    vpnls: SurfaceFitResult
    a2: ParabolaFitResult
    minima_pl_a2: MinimaPowerLaw
    k_values: dict[float, float] = field(default_factory=dict)
    rss: dict[float, float] = field(default_factory=dict)


def fit_dataset(data: IsoFlopData, *, fit_asymptote: bool) -> FitResults:
    n_budgets = len(np.unique(data.C))
    print(f"Fitting: {data.name} ({len(data.N)} points, {n_budgets} budgets)")

    vpnls_result = fit_vpnls(
        data.N, data.D, data.L, grid=FINE_EXPONENT_GRID, fit_asymptote=fit_asymptote
    )
    a2_result = fit_approach2(data.N, data.D, data.L, data.C)

    surface = vpnls_result.to_loss_surface()
    k_values = get_k_values(data, surface)
    rss = compute_rss_by_budget(data, surface, k_values)

    minima_pl_a2 = fit_minima_power_law_a2(a2_result)

    return FitResults(
        data=data,
        vpnls=vpnls_result,
        a2=a2_result,
        minima_pl_a2=minima_pl_a2,
        k_values=k_values,
        rss=rss,
    )


# =============================================================================
# Shared visualization utilities
# =============================================================================


def fmt_budget(b: float, decimals: int = 1) -> str:
    exp = int(np.floor(np.log10(b)))
    mantissa = b / 10**exp
    return f"{mantissa:.{decimals}f}e{exp}" if decimals else f"{mantissa:.0f}e{exp}"


ISOFLOP_CMAPS = {"Llama 3": "Reds", "Delphi": "Blues"}
VERTEX_COLOR_A2 = "red"
VERTEX_COLOR_A3 = "blue"


# =============================================================================
# Figure 1: Forecast sensitivity
# =============================================================================


@dataclass(frozen=True)
class ForecastTarget:
    """True param/token counts for a forecast compute budget."""

    params: float
    tokens: float
    actual: float  # observed loss


# Known forecast targets (from Delphi scaling ladder)
FORECAST_TARGETS: dict[float, ForecastTarget] = {
    1e21: ForecastTarget(params=3_383_110_656, tokens=46_256_881_664, actual=2.75814),
    1e22: ForecastTarget(params=9_714_698_752, tokens=160_369_213_440, actual=2.53079),
    1e23: ForecastTarget(params=24_963_098_112, tokens=628_172_521_472, actual=2.3546),
}
FORECAST_BUDGETS = list(FORECAST_TARGETS.keys())


def forecast_loss_a2(results: FitResults, C: float) -> float:
    """Forecast loss at budget C using true token count + A2 minima power law."""
    assert C in FORECAST_TARGETS, f"No forecast target for C={C:.0e}"
    pl = results.minima_pl_a2
    d = FORECAST_TARGETS[C].tokens
    return pl.x1 * d**pl.x2 + pl.x0


def forecast_loss_a3(results: FitResults, C: float) -> float:
    """Forecast loss at budget C using A3 surface.

    When INFER_PARAMS: N = C/(6D), consistent with how the surface was fit.
    Otherwise: use true N and D from forecast targets.
    """
    assert C in FORECAST_TARGETS, f"No forecast target for C={C:.0e}"
    t = FORECAST_TARGETS[C]
    n = C / (6.0 * t.tokens) if INFER_PARAMS else t.params
    return float(results.vpnls.to_loss_surface().loss(n, t.tokens))  # pyrefly: ignore


def drop_budget(data: IsoFlopData, budget: float) -> IsoFlopData:
    """Return data with a specific compute budget removed."""
    mask = data.C != budget
    assert mask.sum() < len(data.C), f"Budget {budget:.2e} not found in data"
    return IsoFlopData(
        name=data.name,
        N=data.N[mask],
        D=data.D[mask],
        L=data.L[mask],
        C=data.C[mask],
    )


@dataclass
class LOOResult:
    """LOO sensitivity: per-forecast-budget arrays of % deltas across all drops."""

    # {forecast_budget: {dropped_budget: (a2_pct, a3_pct)}}
    per_drop: dict[float, dict[float, tuple[float, float]]]

    def stats(self, fc: float) -> dict[str, tuple[float, float]]:
        """Return {stat: (a2, a3)} for a given forecast budget."""
        vals = self.per_drop[fc]
        a2 = np.array([v[0] for v in vals.values()])
        a3 = np.array([v[1] for v in vals.values()])
        return {
            "mean": (float(a2.mean()), float(a3.mean())),
            "median": (float(np.median(a2)), float(np.median(a3))),
            "std": (float(a2.std()), float(a3.std())),
            "iqr": (
                float(np.subtract(*np.percentile(a2, [75, 25]))),
                float(np.subtract(*np.percentile(a3, [75, 25]))),
            ),
        }


@dataclass
class LOORaw:
    """Raw loss values for one LOO drop + forecast budget combination."""

    drop_c: float
    fc: float
    a2_full: float
    a2_drop: float
    a3_full: float
    a3_drop: float

    @property
    def a2_pct(self) -> float:
        return (self.a2_drop - self.a2_full) / self.a2_full * 100

    @property
    def a3_pct(self) -> float:
        return (self.a3_drop - self.a3_full) / self.a3_full * 100


def leave_one_out_sensitivity(
    data: IsoFlopData, fit_full: FitResults, *, fit_asymptote: bool
) -> tuple[LOOResult, list[LOORaw]]:
    """LOO: for each training budget, refit without it and compute
    % change in forecasted loss at each FORECAST_BUDGET.

    Returns (LOOResult, list of raw values for reporting).
    """
    training_budgets = np.unique(data.C)
    per_drop: dict[float, dict[float, tuple[float, float]]] = {
        fc: {} for fc in FORECAST_BUDGETS
    }
    raw_records: list[LOORaw] = []

    for drop_c in training_budgets:
        fit_drop = fit_dataset(drop_budget(data, drop_c), fit_asymptote=fit_asymptote)
        for fc in FORECAST_BUDGETS:
            r = LOORaw(
                drop_c=drop_c,
                fc=fc,
                a2_full=forecast_loss_a2(fit_full, fc),
                a2_drop=forecast_loss_a2(fit_drop, fc),
                a3_full=forecast_loss_a3(fit_full, fc),
                a3_drop=forecast_loss_a3(fit_drop, fc),
            )
            per_drop[fc][drop_c] = (r.a2_pct, r.a3_pct)
            raw_records.append(r)

    return LOOResult(per_drop=per_drop), raw_records


def generate_loo_report(
    fit_full: FitResults, loo: LOOResult, raw_records: list[LOORaw]
) -> str:
    data = fit_full.data
    v = fit_full.vpnls
    a2 = fit_full.a2
    pl = fit_full.minima_pl_a2
    fc_max = max(FORECAST_BUDGETS)
    training_budgets = a2.compute_budgets

    sections: list[str] = [f"# Forecast Sensitivity Report: {data.name}\n"]

    # -- Fit parameters --
    sections.append(textwrap.dedent(f"""\
        ## Fit Parameters

        ### Approach 2

        | Parameter | Value |
        |-----------|-------|
        | a (N* exponent) | {a2.a:.4f} |
        | b (D* exponent) | {a2.b:.4f} |
        | a + b | {a2.a + a2.b:.4f} |
        | a intercept | {a2.a_intercept:.4f} |
        | b intercept | {a2.b_intercept:.4f} |

        ### Approach 3 (VPNLS)

        | Parameter | Value |
        |-----------|-------|
        | E | {v.E:.4f} |
        | A | {v.A:.2f} |
        | B | {v.B:.2f} |
        | alpha | {v.alpha:.4f} |
        | beta | {v.beta:.4f} |
        | a | {v.a:.4f} |
        | b | {v.b:.4f} |
        | RSS | {v.residual_sum_squares:.6f} |

        ### Minima Power Law (L = x1 * D*^x2 + x0)

        | Parameter | Value |
        |-----------|-------|
        | x0 (offset) | {pl.x0:.6f} |
        | x1 (coefficient) | {pl.x1:.6f} |
        | x2 (exponent) | {pl.x2:.6f} |
        | RSS | {pl.rss:.8f} |
    """))

    # -- Forecast points with actuals --
    sections.append("## Forecast Points\n")
    sections.append(
        "| Budget | N (true) | D (true) | Actual | A2 | A2 Δ% | A3 | A3 Δ% |"
    )
    sections.append(
        "|--------|----------|----------|--------|-----|-------|-----|-------|"
    )
    for fc in FORECAST_BUDGETS:
        t = FORECAST_TARGETS[fc]
        l_a2 = forecast_loss_a2(fit_full, fc)
        l_a3 = forecast_loss_a3(fit_full, fc)
        a2_pct = (l_a2 - t.actual) / t.actual * 100
        a3_pct = (l_a3 - t.actual) / t.actual * 100
        sections.append(
            f"| {fmt_budget(fc, decimals=0)} | {t.params:.2e} | {t.tokens:.2e} "
            f"| {t.actual:.6f} | {l_a2:.6f} | {a2_pct:+.2f}% "
            f"| {l_a3:.6f} | {a3_pct:+.2f}% |"
        )
    sections.append("")

    # -- Per-forecast-budget LOO stats (what the annotations show) --
    sections.append("## LOO Stats by Forecast Budget\n")
    sections.append("| Budget | Stat | A2 | A3 |")
    sections.append("|--------|------|-----|-----|")
    for fc in FORECAST_BUDGETS:
        s = loo.stats(fc)
        b = fmt_budget(fc, decimals=0)
        sections.append(f"| {b} | mean | {s['mean'][0]:+.3f}% | {s['mean'][1]:+.3f}% |")
        sections.append(
            f"| | median | {s['median'][0]:+.3f}% | {s['median'][1]:+.3f}% |"
        )
        sections.append(f"| | std | {s['std'][0]:.3f}% | {s['std'][1]:.3f}% |")
        sections.append(f"| | IQR | {s['iqr'][0]:.3f}% | {s['iqr'][1]:.3f}% |")
    sections.append("")

    # -- Per-dropped-budget table at fc_max (what the plot table shows) --
    sections.append(
        f"## LOO by Dropped Budget (forecast @ {fmt_budget(fc_max, decimals=0)})\n"
    )
    sections.append("| Dropped | A2 % | A3 % |")
    sections.append("|---------|------|------|")
    a2_vals, a3_vals = [], []
    for drop_c in sorted(training_budgets):
        a2_pct, a3_pct = loo.per_drop[fc_max][drop_c]
        a2_vals.append(a2_pct)
        a3_vals.append(a3_pct)
        sections.append(f"| {fmt_budget(drop_c)} | {a2_pct:+.2f} | {a3_pct:+.2f} |")
    a2a, a3a = np.array(a2_vals), np.array(a3_vals)
    a2_abs, a3_abs = np.abs(a2a), np.abs(a3a)
    sections.append(
        f"| **mean/med** | **{a2a.mean():+.3f}/{np.median(a2a):+.3f}** "
        f"| **{a3a.mean():+.3f}/{np.median(a3a):+.3f}** |"
    )
    sections.append(
        f"| *std/IQR* | *{a2a.std():.3f}/{np.subtract(*np.percentile(a2a, [75, 25])):.3f}* "
        f"| *{a3a.std():.3f}/{np.subtract(*np.percentile(a3a, [75, 25])):.3f}* |"
    )
    sections.append(
        f"| **mean/med \\|%\\|** | **{a2_abs.mean():.3f}/{np.median(a2_abs):.3f}** "
        f"| **{a3_abs.mean():.3f}/{np.median(a3_abs):.3f}** |"
    )
    sections.append(
        f"| *std/IQR \\|%\\|* | *{a2_abs.std():.3f}/{np.subtract(*np.percentile(a2_abs, [75, 25])):.3f}* "
        f"| *{a3_abs.std():.3f}/{np.subtract(*np.percentile(a3_abs, [75, 25])):.3f}* |"
    )
    sections.append("")

    # -- Full raw values --
    sections.append("## Raw LOO Values\n")
    sections.append(
        "| Dropped | Forecast | A2 full | A2 drop | A2 Δ% | A3 full | A3 drop | A3 Δ% |"
    )
    sections.append(
        "|---------|----------|---------|---------|-------|---------|---------|-------|"
    )
    for r in raw_records:
        sections.append(
            f"| {fmt_budget(r.drop_c)} | {fmt_budget(r.fc, decimals=0)} "
            f"| {r.a2_full:.6f} | {r.a2_drop:.6f} | {r.a2_pct:+.2f}% "
            f"| {r.a3_full:.6f} | {r.a3_drop:.6f} | {r.a3_pct:+.2f}% |"
        )
    sections.append("")

    return "\n".join(sections)


def plot_forecast_sensitivity(
    fit_full: FitResults,
    loo: LOOResult,
) -> plt.Figure:
    """Isoflop panel with LOO per-drop table on right, per-forecast stats inset."""
    data = fit_full.data
    a2 = fit_full.a2
    training_budgets = a2.compute_budgets
    cmap = plt.get_cmap(ISOFLOP_CMAPS.get(data.name, "Blues"))
    norm = plt.Normalize(
        vmin=np.log10(training_budgets.min()) - 0.75,
        vmax=np.log10(training_budgets.max()) + 0.25,
    )

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])

    # Isoflop curves + A2 parabolas
    for i, c in enumerate(training_budgets):
        mask = data.C == c
        D_group, L_group = data.D[mask], data.L[mask]
        order = np.argsort(D_group)
        D_group, L_group = D_group[order], L_group[order]
        color = cmap(norm(np.log10(c)))

        ax.plot(D_group, L_group, "o", color=color, ms=3, zorder=5)
        fit_D = a2.parabola_fits_D[i]
        d_curve = np.logspace(np.log10(D_group.min()), np.log10(D_group.max()), 200)
        ax.plot(
            d_curve,
            np.polyval(fit_D.coeffs, np.log10(d_curve)),
            "-",
            color=color,
            lw=1.0,
            alpha=0.8,
            zorder=3,
        )
        ax.plot(
            a2.D_opts[i],
            fit_D.L_min,
            "x",
            color=VERTEX_COLOR_A2,
            ms=7,
            mew=1.5,
            zorder=7,
        )

    # Minima power law extending to max forecast budget
    pl = fit_full.minima_pl_a2
    d_max_forecast = FORECAST_TARGETS[max(FORECAST_BUDGETS)].tokens
    d_pl = np.logspace(np.log10(a2.D_opts.min()), np.log10(d_max_forecast), 300)
    ax.plot(
        d_pl,
        pl.x1 * d_pl**pl.x2 + pl.x0,
        "--",
        color="grey",
        lw=1.0,
        alpha=0.6,
        zorder=1,
    )

    # Forecast points with annotations
    # Per-budget annotation offsets: (xytext, va)
    ann_offsets: dict[float, tuple[tuple[int, int], str]] = {
        1e21: ((-57, -18), "center"),
        1e22: ((40, 25), "center"),
        1e23: ((-57, -18), "center"),
    }
    for fc in FORECAST_BUDGETS:
        t = FORECAST_TARGETS[fc]
        d_star = t.tokens
        l_a2 = forecast_loss_a2(fit_full, fc)
        l_a3 = forecast_loss_a3(fit_full, fc)
        a2_pct = (l_a2 - t.actual) / t.actual * 100
        a3_pct = (l_a3 - t.actual) / t.actual * 100
        # Unsigned LOO means
        loo_a2 = np.mean(np.abs([v[0] for v in loo.per_drop[fc].values()]))
        loo_a3 = np.mean(np.abs([v[1] for v in loo.per_drop[fc].values()]))
        xytext, va = ann_offsets.get(fc, ((8, 0), "center"))

        ax.plot(d_star, l_a2, "D", color="black", ms=5, zorder=8)
        ax.annotate(
            f"{fmt_budget(fc, decimals=0)} (actual={t.actual:.3f})\n"
            f"A2={l_a2:.3f} ($\\mathbf{{{a2_pct:+.1f}\\%}}$)\n"
            f"A3={l_a3:.3f} ($\\mathbf{{{a3_pct:+.1f}\\%}}$)\n"
            f"LOO $\\mu$ |%|: A2={loo_a2:.2f} A3={loo_a3:.2f}",
            (d_star, l_a2),
            textcoords="offset points",
            xytext=xytext,
            fontsize=6.5,
            ha="center",
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.5, alpha=0.7
            ),
            zorder=9,
        )

    ax.set_xscale("log")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.24, ymax)
    ax.set_xlabel("Tokens (D)")
    ax.set_ylabel(f"Loss ({DELPHI_METRIC_LABEL})")

    # Budget legend (upper right)
    budget_handles = [
        plt.Line2D(
            [], [], color=cmap(norm(np.log10(c))), marker="o", ms=4, ls="-", lw=1.0
        )
        for c in training_budgets
    ]
    budget_labels = [fmt_budget(c) for c in training_budgets]
    budget_legend = ax.legend(
        budget_handles,
        budget_labels,
        title="FLOPs",
        fontsize=7,
        title_fontsize=8,
        loc="upper right",
    )

    # Style legend (lower left)
    style_handles = [
        plt.Line2D(
            [],
            [],
            color="grey",
            ls="--",
            lw=1.0,
            label="Approach 2 power law\n$L = x_1 D^{x_2} + x_0$",
        ),
        plt.Line2D(
            [],
            [],
            color=VERTEX_COLOR_A2,
            marker="x",
            ls="None",
            ms=7,
            mew=1.5,
            label=r"$D^*$ Approach 2",
        ),
        plt.Line2D(
            [], [], color="black", marker="D", ls="None", ms=5, label="Forecast point"
        ),
    ]
    ax.legend(handles=style_handles, fontsize=7, loc="lower left")
    ax.add_artist(budget_legend)

    # Right panel: per-dropped-budget table at highest forecast budget
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.set_axis_off()

    fc_max = max(FORECAST_BUDGETS)
    headers = ["Dropped", "A2 %", "A3 %"]

    # Per-budget rows (footnote on first row's A2 value)
    min_budget = sorted(training_budgets)[0]
    a2_vals_list: list[float] = []
    a3_vals_list: list[float] = []
    rows: list[list[str]] = []
    for drop_c in sorted(training_budgets):
        a2_pct, a3_pct = loo.per_drop[fc_max][drop_c]
        a2_vals_list.append(a2_pct)
        a3_vals_list.append(a3_pct)
        a2_cell = f"{a2_pct:+.2f}*" if drop_c == min_budget else f"{a2_pct:+.2f}"
        rows.append([fmt_budget(drop_c), a2_cell, f"{a3_pct:+.2f}"])

    # Aggregate rows (unsigned)
    a2_abs, a3_abs = np.abs(np.array(a2_vals_list)), np.abs(np.array(a3_vals_list))
    agg_rows = [
        [
            r"$\mu$/med |%|",
            f"{a2_abs.mean():.2f}/{np.median(a2_abs):.2f}",
            f"{a3_abs.mean():.2f}/{np.median(a3_abs):.2f}",
        ],
        [
            r"$\sigma$/IQR |%|",
            f"{a2_abs.std():.2f}/{np.subtract(*np.percentile(a2_abs, [75, 25])):.2f}",
            f"{a3_abs.std():.2f}/{np.subtract(*np.percentile(a3_abs, [75, 25])):.2f}",
        ],
    ]

    all_rows = rows + agg_rows
    tbl = ax_tbl.table(
        cellText=[headers] + all_rows,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.31)

    # Header styling
    for j in range(len(headers)):
        tbl[0, j].set_text_props(fontweight="bold")
        tbl[0, j].set_facecolor("#e0e0e0")

    # Data row styling
    n_data = len(rows)
    for i in range(n_data):
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            tbl[i + 1, j].set_facecolor(bg)

    # Aggregate row styling: bold values only (not labels), italic throughout
    for i in range(len(agg_rows)):
        for j in range(len(headers)):
            cell = tbl[n_data + 1 + i, j]
            cell.set_facecolor("#d0d8e8")
            weight = "bold" if j > 0 else "normal"
            cell.set_text_props(fontweight=weight, fontstyle="italic")

    ax_tbl.set_title(f"Forecast @ {fmt_budget(fc_max, decimals=0)}", fontsize=9)

    # Footnote with worked example
    l_full = forecast_loss_a2(fit_full, fc_max)
    fit_drop = fit_dataset(drop_budget(data, min_budget), fit_asymptote=False)
    l_drop = forecast_loss_a2(fit_drop, fc_max)
    pct = (l_drop - l_full) / l_full * 100
    ax_tbl.text(
        0.5,
        -0.02,
        f"* Example: (L_drop \u2212 L_full) / L_full \u00d7 100\n"
        f"  ({l_drop:.4f} \u2212 {l_full:.4f}) / {l_full:.4f} \u00d7 100 = {pct:+.2f}%",
        transform=ax_tbl.transAxes,
        fontsize=8,
        ha="center",
        va="top",
        style="italic",
        alpha=0.7,
    )

    fig.text(
        0.5,
        0.99,
        f"Forecast Sensitivity: {data.name}",
        fontsize=12,
        ha="center",
        va="bottom",
    )
    fig.text(
        0.5,
        0.95,
        "LOO % change in forecasted loss when each training budget is dropped",
        fontsize=9,
        ha="center",
        va="bottom",
    )

    fig.tight_layout(rect=(0, 0.14, 1, 0.96))

    # Caption
    fig.text(
        0.5,
        -0.1,
        r"$\bf{Figure\ 1}$: Forecast sensitivity via leave-one-out (LOO) over training budgets."
        r" Loss at 1e21/1e22/1e23 FLOPs predicted by:"
        "\n"
        r"Approach 2 power law $L = x_1 D^{x_2} + x_0$ fit to parabola minima;"
        r" Approach 3 surface $L = A/N^\alpha + B/D^\beta$ (E=0) evaluated at true $(N, D)$."
        "\n"
        r"Table shows signed % change when each training budget is dropped."
        r" LOO $\mu$|%| = mean of unsigned % changes across all LOO drops.",
        fontsize=7,
        ha="center",
        va="bottom",
    )

    return fig


# =============================================================================
# Figure 2: FLOP factors
# =============================================================================


def plot_flop_factors(data: IsoFlopData) -> plt.Figure:
    """Horizontal violin+boxplot of k=C/(ND) by budget with stats table."""
    N, D, C = data.N, data.D, data.C
    k = C / (N * D)
    budgets = np.unique(C)
    n_b = len(budgets)
    groups = [k[C == c] for c in budgets]
    params = [N[C == c] for c in budgets]
    labels = [fmt_budget(c) for c in budgets]

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])

    ax.boxplot(
        groups,
        positions=range(n_b),
        vert=False,
        widths=0.3,
        showfliers=False,
        manage_ticks=False,
        whis=(0, 100),
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.5),
        medianprops=dict(color="black"),
    )
    rng = np.random.default_rng(42)
    for i, (g, p) in enumerate(zip(groups, params)):
        sizes = 20 + 80 * (p - p.min()) / (p.max() - p.min() + 1e-10)
        jitter = rng.uniform(-0.08, 0.08, len(g))
        ax.scatter(
            g,
            i + jitter,
            s=sizes,
            color="steelblue",
            alpha=0.7,
            zorder=5,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.axvline(6, color="grey", ls="--", lw=0.8, label="k=6")
    ax.set_yticks(range(n_b))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("k = C / (N × D)")
    ax.set_title(f"k by Budget — {data.name}")
    ax.legend(fontsize=8, loc="lower left")

    # Stats grid (imshow + text, labels at bottom via xticks)
    ax_tbl = fig.add_subplot(gs[0, 1])
    col_labels = [r"$\mu$", "med", "min", "max"]
    n_cols = len(col_labels)
    cell_data = np.array([[g.mean(), np.median(g), g.min(), g.max()] for g in groups])
    # Alternating row backgrounds (0=white, 0.05=light grey)
    bg = np.zeros((n_b, n_cols))
    bg[::2] = 0.15
    ax_tbl.imshow(
        bg, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest"
    )
    for i in range(n_b):
        for j in range(n_cols):
            weight = "bold" if j == 0 else "normal"
            ax_tbl.text(
                j,
                i,
                f"{cell_data[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight=weight,
            )
    ax_tbl.set_xticks(range(n_cols))
    ax_tbl.set_xticklabels(col_labels, fontsize=9)
    ax_tbl.xaxis.set_ticks_position("bottom")
    ax_tbl.set_yticks([])
    ax_tbl.set_title("Summary Stats")
    ax_tbl.tick_params(left=False, bottom=False)

    fig.text(0.5, 0.99, "Empirical FLOP Factors", fontsize=12, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))

    fig.text(
        0.5,
        -0.1,
        r"$\bf{Figure\ 2}$: Empirical FLOP factor $k = C/(N \cdot D)$ per training budget."
        r" Points sized by parameter count $N$."
        "\n"
        r"The standard $C = 6ND$ approximation holds at mid-to-high budgets"
        r" but overestimates at the lowest budget ($k \approx 4.8$).",
        fontsize=7,
        ha="center",
        va="bottom",
    )
    return fig


# =============================================================================
# Figure 3: IsoFLOP comparison
# =============================================================================


def plot_isoflop_panel(
    ax: plt.Axes, results: FitResults, *, show_vertex_legend: bool = False
) -> None:
    """Plot isoFLOP curves with Approach 2 parabolas (solid), Approach 3
    prediction (dashed), and D* vertex markers on the Approach 2 parabola."""
    data = results.data
    a2 = results.a2
    surface = results.vpnls.to_loss_surface()
    budgets = a2.compute_budgets
    cmap_name = ISOFLOP_CMAPS.get(data.name, "viridis")
    cmap = plt.get_cmap(cmap_name)
    # Start at 0.3 to avoid too-light colors, end at 0.9 to avoid near-black
    norm = plt.Normalize(
        vmin=np.log10(budgets.min()) - 0.75,
        vmax=np.log10(budgets.max()) + 0.25,
    )
    deltas = compute_dopt_deltas(results)

    for i, c in enumerate(budgets):
        mask = data.C == c
        D_group, L_group = data.D[mask], data.L[mask]
        order = np.argsort(D_group)
        D_group, L_group = D_group[order], L_group[order]
        color = cmap(norm(np.log10(c)))

        ax.plot(D_group, L_group, "o", color=color, ms=3, zorder=5)

        d_curve = np.logspace(np.log10(D_group.min()), np.log10(D_group.max()), 200)

        # Approach 2 parabola (solid, primary)
        fit_D = a2.parabola_fits_D[i]
        ax.plot(
            d_curve,
            np.polyval(fit_D.coeffs, np.log10(d_curve)),
            "-",
            color=color,
            lw=1.0,
            alpha=0.8,
            zorder=3,
        )

        # Approach 3 prediction (dashed, secondary)
        k = results.k_values[c]
        if SHOW_A3_CURVES:
            n_curve = c / (k * d_curve)
            ax.plot(
                d_curve,
                surface.loss(n_curve, d_curve),
                "--",
                color=color,
                lw=1.0,
                alpha=0.7,
                zorder=2,
            )

        # D* vertices on the Approach 2 parabola
        ax.plot(
            a2.D_opts[i],
            fit_D.L_min,
            "x",
            color=VERTEX_COLOR_A2,
            ms=7,
            mew=1.5,
            zorder=7,
        )
        d3 = d_opt_a3(surface, c, k)
        ax.plot(
            d3,
            np.polyval(fit_D.coeffs, np.log10(d3)),
            "+",
            color=VERTEX_COLOR_A3,
            ms=8,
            mew=1.5,
            zorder=7,
        )

        # Delta % annotation to the right of rightmost data point
        abs_delta = abs(deltas.pct[c])
        ann_color = "grey" if abs_delta < 10 else "black"
        ann_weight = "bold" if abs_delta >= 20 else "normal"
        ax.annotate(
            f"{deltas.pct[c]:+.0f}%",
            (D_group[-1], L_group[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=6,
            ha="left",
            va="center",
            color=ann_color,
            fontweight=ann_weight,
            zorder=8,
        )

    # Power law through A2 parabola minima
    if SHOW_MINIMA_PL:
        pl = results.minima_pl_a2
        d_pl = np.logspace(np.log10(a2.D_opts.min()), np.log10(a2.D_opts.max()), 200)
        ax.plot(
            d_pl,
            pl.x1 * d_pl**pl.x2 + pl.x0,
            "--",
            color="grey",
            lw=0.8,
            alpha=0.5,
            zorder=1,
        )

    ax.set_xscale("log")
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.5)
    ax.set_xlabel("Tokens (D)")
    ax.set_ylabel("Loss")
    title = data.name
    if data.name == "Delphi":
        title += f" ({DELPHI_METRIC_LABEL})"
    ax.set_title(title)

    # Budget legend
    handles = [
        plt.Line2D(
            [], [], color=cmap(norm(np.log10(c))), marker="o", ms=4, ls="-", lw=1.0
        )
        for c in budgets
    ]
    labels = [fmt_budget(c) for c in budgets]
    budget_legend = ax.legend(
        handles, labels, title="FLOPs", fontsize=7, title_fontsize=8, loc="upper right"
    )

    if show_vertex_legend:
        vertex_handles = [
            plt.Line2D([], [], color="grey", ls="-", lw=1.0, label="Approach 2"),
        ]
        if SHOW_A3_CURVES:
            vertex_handles.append(
                plt.Line2D([], [], color="grey", ls="--", lw=1.0, label="Approach 3"),
            )
        vertex_handles += [
            plt.Line2D(
                [],
                [],
                color="red",
                marker="x",
                ls="None",
                ms=7,
                mew=1.5,
                label=r"$D^*$ Approach 2",
            ),
            plt.Line2D(
                [],
                [],
                color="blue",
                marker="+",
                ls="None",
                ms=8,
                mew=1.5,
                label=r"$D^*$ Approach 3",
            ),
        ]
        ax.legend(
            handles=vertex_handles,
            fontsize=7,
            loc="lower left",
            framealpha=0.9,
        )
        ax.add_artist(budget_legend)


VIOLIN_COLORS = {
    "Llama 3": mcolors.to_hex(plt.get_cmap("Reds")(0.85)),
    "Delphi": mcolors.to_hex(plt.get_cmap("Blues")(0.85)),
}


def plot_comparison(results_list: list[FitResults]) -> plt.Figure:
    """Side-by-side isoFLOP comparison with violin plot of D* deltas."""
    n = len(results_list)
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[2.5] * n + [0.9], wspace=0.02)

    for j, res in enumerate(results_list):
        ax = fig.add_subplot(gs[0, j])
        plot_isoflop_panel(ax, res, show_vertex_legend=(j == 0))
        # Place y-tick labels just inside the left edge of the plot area
        if j > 0:
            ax.set_ylabel("")
        ax.tick_params(
            axis="y", direction="in", left=True, labelleft=True, labelsize=6, pad=-8
        )
        for label in ax.yaxis.get_ticklabels():
            label.set_zorder(0)
            label.set_alpha(0.4)
            label.set_horizontalalignment("left")

    # Violin panel
    ax_v = fig.add_subplot(gs[0, n])
    all_vals: list[np.ndarray] = []
    names: list[str] = []
    for res in results_list:
        d = compute_dopt_deltas(res)
        all_vals.append(np.array(list(d.pct.values())))
        names.append(res.data.name)

    positions = list(range(len(all_vals)))
    parts = ax_v.violinplot(all_vals, positions=positions, vert=True, showextrema=False)
    for i, body in enumerate(parts["bodies"]):  # pyrefly: ignore
        color = VIOLIN_COLORS.get(names[i], "grey")
        body.set_facecolor(color)
        body.set_alpha(0.4)
        body.set_edgecolor(color)

    # Skinny boxplots overlaid
    bp = ax_v.boxplot(
        all_vals,
        positions=positions,
        widths=0.08,
        vert=True,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        zorder=4,
        whis=(0, 100),  # whiskers span full data range
    )
    for i, patch in enumerate(bp["boxes"]):
        color = VIOLIN_COLORS.get(names[i], "grey")
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)
    for i in range(len(all_vals)):
        color = VIOLIN_COLORS.get(names[i], "grey")
        for element in ("whiskers", "caps"):
            for line in bp[element][2 * i : 2 * i + 2]:
                line.set_color(color)
                line.set_linewidth(0.8)
        bp["medians"][i].set_color("black")
        bp["medians"][i].set_linewidth(0.8)

    # Individual points
    for i, (vals, name) in enumerate(zip(all_vals, names)):
        color = VIOLIN_COLORS[name]
        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, size=len(vals))
        ax_v.scatter(i + jitter, vals, s=15, color=color, zorder=5, alpha=0.8)

    ax_v.axhline(0, color="grey", ls="--", lw=0.7)
    ax_v.set_xticks(positions)
    ax_v.set_xticklabels([n.split()[0] for n in names])
    ax_v.set_title(r"$\Delta$% $D^*$")
    ax_v.yaxis.set_label_position("right")
    ax_v.yaxis.tick_right()
    ax_v.tick_params(axis="y", labelsize=8)

    fig.text(
        0.5,
        1.02,
        "IsoFLOP Experiment Comparison",
        fontsize=12,
        ha="center",
        va="bottom",
    )
    fig.text(
        0.5,
        0.97,
        r"$\Delta$% $D^*$: per-budget % difference in optimal tokens $(D^*_{A3} - D^*_{A2})/D^*_{A2}$",
        fontsize=9,
        ha="center",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))

    # Caption
    fig.text(
        0.5,
        -0.08,
        r"$\bf{Figure\ 3}$: IsoFLOP curves with Approach 2 parabolic fits ($L$ vs $\log D$)"
        r" and inferred $D^*$ vertices from Approach 2 ($\times$) and Approach 3 ($+$)."
        "\n"
        r"Approach 3 fits $L = E + A/N^\alpha + B/D^\beta$ via VPNLS grid search."
        r" Violin plot shows distribution of per-budget $\Delta\% \ D^*$ between methods.",
        fontsize=7,
        ha="center",
        va="bottom",
    )

    # Annotations in empty space
    bbox_props = dict(boxstyle="round,pad=0.2", alpha=0.6, ec="black", lw=0.5)
    ann_positions = [(0.62, 0.90), (0.75, 0.24)]
    for i, ((px, py), name) in enumerate(zip(ann_positions, names)):
        color = VIOLIN_COLORS[name]
        d = compute_dopt_deltas(results_list[i])
        ax_v.text(
            px,
            py,
            f"$\\mu$={d.mean_signed:+.1f}%\nmed={d.median_signed:+.1f}%",
            transform=ax_v.transAxes,
            fontsize=7,
            ha="center",
            va="center",
            color="black",
            bbox={**bbox_props, "facecolor": "white"},
        )

    return fig


# =============================================================================
# Reports
# =============================================================================

EXPERIMENT_TEMPLATE = textwrap.dedent("""\
    ## {name}

    Data: {n_points} points, {n_budgets} budgets
    Flop factor mode: {k_mode}

    ### Approach 2 (Parabolic IsoFLOP Fits)

      a (N* exponent)   = {a2_a:.4f}
      b (D* exponent)   = {a2_b:.4f}
      a + b             = {a2_ab:.4f}
      a intercept       = {a2_a_int:.4f}
      b intercept       = {a2_b_int:.4f}

    ### Approach 3 (VPNLS Grid, resolution={grid_res})

      E     = {a3_E:.4f}
      A     = {a3_A:.2f}
      B     = {a3_B:.2f}
      alpha = {a3_alpha:.4f}
      beta  = {a3_beta:.4f}
      a     = {a3_a:.4f}
      b     = {a3_b:.4f}
      RSS   = {a3_rss:.6f}

    ### Minima Power Law (L = x1 * D*^x2 + x0)

    Fit to Approach 2 parabola vertices (D*, L_min).

      x0 (offset)     = {pl_x0:.6f}
      x1 (coefficient) = {pl_x1:.6f}
      x2 (exponent)   = {pl_x2:.6f}
      RSS             = {pl_rss:.8f}

    ### D* Comparison (Approach 3 vs Approach 2)

    Positive delta means Approach 3 infers more tokens than Approach 2.

    {budget_table}

      Mean delta%     = {mean_signed:+.1f}%
      Mean |delta%|   = {mean_unsigned:.1f}%
      Median delta%   = {median_signed:+.1f}%
      Median |delta%| = {median_unsigned:.1f}%
""")

BUDGET_TABLE_HEADER = "      {:<12s} {:>8s} {:>14s} {:>14s} {:>10s} {:>10s}".format(
    "Budget",
    "k",
    "D* (A2)",
    "D* (A3)",
    "Delta",
    "Delta%",
)


def _fmt_tokens(d: float) -> str:
    """Format token count compactly, e.g. 1.23e+10."""
    return f"{d:.2e}"


def generate_report(results_list: list[FitResults]) -> str:
    """Generate a text report for all experiments."""
    sections: list[str] = ["# Delphi Scaling Analysis Report\n"]
    for r in results_list:
        v, a2 = r.vpnls, r.a2
        surface = v.to_loss_surface()
        dd = compute_dopt_deltas(r)
        budgets = a2.compute_budgets

        # Build per-budget table
        table_lines = [BUDGET_TABLE_HEADER, "      " + "-" * 72]
        for i, c in enumerate(budgets):
            d2 = a2.D_opts[i]
            d3 = d_opt_a3(surface, c, r.k_values[c])
            raw_str = f"{dd.raw[c]:+.2e}"
            table_lines.append(
                "      {:<12s} {:>8.3f} {:>14s} {:>14s} {:>10s} {:>+9.1f}%".format(
                    fmt_budget(c),
                    r.k_values[c],
                    _fmt_tokens(d2),
                    _fmt_tokens(d3),
                    raw_str,
                    dd.pct[c],
                )
            )

        k_mode = "learned per-budget (OLS)" if LEARN_K else "fixed k=6"
        sections.append(
            EXPERIMENT_TEMPLATE.format(
                name=r.data.name,
                n_points=len(r.data.N),
                n_budgets=len(budgets),
                k_mode=k_mode,
                a2_a=a2.a,
                a2_b=a2.b,
                a2_ab=a2.a + a2.b,
                a2_a_int=a2.a_intercept,
                a2_b_int=a2.b_intercept,
                a3_E=v.E,
                a3_A=v.A,
                a3_B=v.B,
                a3_alpha=v.alpha,
                a3_beta=v.beta,
                a3_a=v.a,
                a3_b=v.b,
                grid_res=GRID_RESOLUTION,
                a3_rss=sum(r.rss.values()),
                pl_x0=r.minima_pl_a2.x0,
                pl_x1=r.minima_pl_a2.x1,
                pl_x2=r.minima_pl_a2.x2,
                pl_rss=r.minima_pl_a2.rss,
                budget_table="\n".join(table_lines),
                mean_signed=dd.mean_signed,
                mean_unsigned=dd.mean_unsigned,
                median_signed=dd.median_signed,
                median_unsigned=dd.median_unsigned,
            )
        )
    return "\n".join(sections)


# =============================================================================
# Main
# =============================================================================


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    llama3 = load_llama3()
    delphi = load_delphi()

    # Figure 1: Forecast sensitivity (Delphi only, E=0 surface for better extrapolation)
    delphi_fit_noE = fit_dataset(delphi, fit_asymptote=False)
    loo, loo_raw = leave_one_out_sensitivity(
        delphi, delphi_fit_noE, fit_asymptote=False
    )
    fig1 = plot_forecast_sensitivity(delphi_fit_noE, loo)
    save_figure(fig1, OUT_DIR / "forecast_sensitivity.png", dpi=300)
    plt.close(fig1)

    # Figure 2: FLOP factors (Delphi only)
    fig2 = plot_flop_factors(delphi)
    save_figure(fig2, OUT_DIR / "flop_factors.png", dpi=300)
    plt.close(fig2)

    # Figure 3: IsoFLOP comparison
    llama3_fit = fit_dataset(llama3, fit_asymptote=True)
    delphi_fit = fit_dataset(delphi, fit_asymptote=True)
    fig3 = plot_comparison([llama3_fit, delphi_fit])
    save_figure(fig3, OUT_DIR / "isoflop_comparison.png", dpi=300)
    plt.close(fig3)

    # Reports
    report = generate_report([llama3_fit, delphi_fit])
    report_path = OUT_DIR / "isoflop_comparison.md"
    report_path.write_text(report)
    print(report)
    print(f"\nReport saved to {report_path}")

    loo_report = generate_loo_report(delphi_fit_noE, loo, loo_raw)
    loo_report_path = OUT_DIR / "forecast_sensitivity.md"
    loo_report_path.write_text(loo_report)
    print(loo_report)
    print(f"\nLOO report saved to {loo_report_path}")


if __name__ == "__main__":
    main()
