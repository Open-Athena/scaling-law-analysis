# Spec ↔ Implementation Sync

After changing any spec or implementation, check this document to see what else needs updating. Changes may originate on either side. The sync process is responsible for recovering consistency: if a change affects a topic covered by a spec, the other side must be updated to match. Changes that fall outside the spec's scope do not require a sync (see [Spec-Driven Development](project.md#spec-driven-development)).

## Spec → Implementation

When a spec changes, the implementation it controls must be updated to match. See [build.md](build.md) for the commands to regenerate implementation artifacts.

- `specs/experiments.md` → `results/experiments/exp{N}/`: methodology changes require re-running experiments ([build.md > Full Workflow](build.md#full-workflow), step 1)
- `specs/article.md` → `results/article/`: outline or figure spec changes require updating `article.html` and regenerating figures ([build.md > Full Workflow](build.md#full-workflow), steps 2–7)

## Implementation → Spec

Required only for implementation changes that touch topics a spec covers. If the implementation evolves beyond the spec's scope, no update is needed — but if a change contradicts or modifies something a spec specifies, the spec must be updated to match.

- `results/experiments/exp{N}/` → `specs/experiments.md` — never add experimental results or findings to experiment specs. Experiment specs define intent and methodology only.
- `results/article/` → `specs/article.md`

## Implementation → Implementation

When one implementation artifact changes, other artifacts that depend on it may need updating.

- **CSV data in article text**: The figure generator exports numerical results alongside each figure. Specific values from these CSVs are hardcoded in `results/article/article.html` in both prose and data tables. After regenerating figures, check whether the CSV data has changed and update the corresponding text and tables in the article to match. The mapping of CSVs to article sections is:
  - `results/article/extrapolation_error/extrapolation_error_data.csv` → "Why It Matters" (extrapolation bar chart, collapsible data table)
  - `results/article/off_center_constant_bias/off_center_constant_bias_data.csv` → "Constant Multiplicative Bias" (key result callout numbers)
  - `results/article/parameter_recovery/parameter_recovery_max_errors.csv` → "Method Comparison (Parameter Recovery)" (prose error claims for Approach 3 and VPNLS, collapsible data table max-error columns)
  - `results/article/parameter_recovery/parameter_recovery_failures.csv` → "Method Comparison (Parameter Recovery)" (collapsible data table failure counts)
  - `results/article/exponent_inference/exponent_inference.csv` → "Method Comparison (Exponent Inference)" (prose error claims, collapsible data table)
  - `results/experiments/exp8/conditioning_analysis.txt` → "Problems with Direct Surface Fitting" (experimental setup: surface parameters, compute budgets, points per curve; Hessian eigenvalue ranges, condition numbers κ ≈ 3.5×10¹¹ and κ ≈ 11, underdetermined parameter directions)
- **Experiment outputs copied to article**: the figure generator's `copy_experiment_outputs` copies several experiment results directly into the article directory tree. Re-run step 2 after re-running experiments.
  - `results/experiments/exp5/parameter_recovery_detailed.png` → `results/article/appendix/parameter_recovery_detailed.png`
  - `results/experiments/exp4/extrapolation_error.png` → `results/article/appendix/combined_extrapolation_error.png`
  - `results/experiments/exp7/isoflop_curves.png` → `results/article/appendix/isoflop_curves_noisy.png`
  - `results/experiments/exp7/exponent_inference_errors.png` → `results/article/appendix/exponent_inference_errors.png`
  - `results/experiments/exp7/exponent_inference.png` → `results/article/exponent_inference/exponent_inference.png`
  - `results/experiments/exp7/exponent_inference.csv` → `results/article/exponent_inference/exponent_inference.csv`

## Spec → Spec

Specs reference each other and external files (e.g. `AGENTS.md`, `README.md`). When renaming, moving, or restructuring any spec, review links in all markdown files across the project — not just within `specs/`.

This includes `sync.md` itself: when specs are added or removed, when the mapping between specs and implementation artifacts changes, or when build steps are restructured, update this document to reflect the new relationships.
