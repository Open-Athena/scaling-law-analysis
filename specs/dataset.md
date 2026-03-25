# Dataset Spec

Instructions for uploading `data/isoflops/extract/isoflops.csv` to Hugging Face as `open-athena/isoflop-experiments`.

## Pre-upload Transformations

1. Drop all rows where `condition == "exp_loss"` (Llama 3 exponentiated loss variant — only raw values from the paper should be published).
2. Drop the `condition` column entirely (it is empty for all other experiments and `raw_loss` for Llama 3; no longer needed after filtering).

After these transformations the dataset has **814 rows** and **7 experiments**.

## Uniqueness

Each row is uniquely identified by `(experiment, tokens, params, budget)`.

Note: `(experiment, budget, params)` is *almost* unique but has 6 collisions in `ml_scalefit` where the same model size appears twice at the same snapped budget with slightly different token counts (a consequence of budget snapping). Including `tokens` resolves all collisions.

## Upload Instructions

1. Run the pre-upload transformations (drop `exp_loss` rows, drop `condition` column).
2. Verify uniqueness: assert that `(experiment, tokens, params, budget)` has no duplicates.
3. Verify row count: 814.
4. Create HF dataset repo `open-athena/isoflop-experiments` with:
   - `isoflops.csv` (the transformed CSV)
   - `README.md` (the exact content of the README section below, uploaded as-is)
5. Use `huggingface_hub` CLI or Python API to upload.

---

## README

Everything below this line is the HF dataset card README. Upload it verbatim as `README.md` in the dataset repo.

---

# IsoFLOP Scaling Law Experiments

Curated collection of IsoFLOP curve data from 6 experiments, standardized to a common schema.

Associated with [Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits](https://arxiv.org/abs/2603.22339). Extraction and transformation code: [Open-Athena/scaling-law-analysis](https://github.com/Open-Athena/scaling-law-analysis).

## Schema

| Field | Type | Description |
|---|---|---|
| `source` | string | Data source identifier. One of: `ml_scalefit`, `epochai_chinchilla`, `llama_3`, `marin_202603`, `misfitting`. |
| `dataset` | string | Training dataset. One of: `massivetext`, `llama_3`, `comma`, `dclm`, `nemotron`, `fineweb_c4`. |
| `model` | string | Model architecture. One of: `chinchilla`, `llama_3`, `llama_2`, `transformer`. |
| `experiment` | string | Canonical identifier, unique per study. Usually `source__dataset__model`. |
| `tokens` | float | Training tokens ($D$). Either from the source data or derived via $D = C / (6N)$. |
| `params` | float | Model parameter count ($N$). Either from the source data or derived via $N = C / (6D)$. |
| `budget` | float | Compute budget in FLOPs ($C$). |
| `loss` | float | Validation loss. Units and scale vary by source (see experiment details). |

Each row is uniquely identified by `(experiment, tokens, params, budget)`.

## Summary

| Experiment | Points | Budgets | Reference | Collection Method |
|---|---|---|---|---|
| `ml_scalefit__massivetext__chinchilla` | 124 | 9 | [arxiv:2507.09404](https://arxiv.org/abs/2507.09404) | GitHub CSV |
| `epochai_chinchilla__massivetext__chinchilla` | 123 | 9 | [arxiv:2404.10102](https://arxiv.org/abs/2404.10102) | SVG digitization |
| `llama_3__raw_loss` | 133 | 10 | [arxiv:2407.21783](https://arxiv.org/abs/2407.21783) | Manual figure digitization |
| `marin_202603__comma__llama_2` | 85 | 7 | [W&B report](https://wandb.ai/marin-community/marin/reports/Scaling-Ladders--VmlldzoxNTc0MjM1NQ) | W&B export |
| `marin_202603__dclm__llama_2` | 85 | 7 | [W&B report](https://wandb.ai/marin-community/marin/reports/Scaling-Ladders--VmlldzoxNTc0MjM1NQ) | W&B export |
| `marin_202603__nemotron__llama_2` | 88 | 8 | [W&B report](https://wandb.ai/marin-community/marin/reports/Scaling-Ladders--VmlldzoxNTc0MjM1NQ) | W&B export |
| `misfitting__fineweb_c4__transformer` | 176 | 26 | [arxiv:2502.18969](https://arxiv.org/abs/2502.18969) | Checkpoint interpolation |
| **Total** | **814** | | | |

## Experiment Details

### ml_scalefit

Chinchilla training data from Besiroglu et al. ([arxiv:2507.09404](https://arxiv.org/abs/2507.09404)). Raw data: [`apple/ml-scalefit/data/chinchilla.csv`](https://github.com/apple/ml-scalefit/blob/ac4664af5db6c94e6ac7521a61dd3bbb0d91cc3a/data/chinchilla.csv) with columns `model_size` ($N$), `n_tokens` ($D$), `loss`. Budget $C = 6ND$ is computed and snapped to the 9 Chinchilla IsoFLOP levels ($6 \times 10^{18}$ to $3 \times 10^{21}$); points >10% from the nearest budget are discarded. $N$, $D$, and loss are kept as-is.

### epochai_chinchilla

Independent extraction of the same Chinchilla experiments by Besiroglu et al. ([arxiv:2404.10102](https://arxiv.org/abs/2404.10102)), digitized from SVG figures in the original paper. Raw data: [`epoch-research/analyzing-chinchilla/data/svg_extracted_data.csv`](https://github.com/epoch-research/analyzing-chinchilla/blob/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv) with columns `Model Size` ($N$), `Training FLOP` ($C$), `loss`. $N$ and $C$ are rounded to integers (SVG artifact). $C$ is snapped to the same 9 budgets as ml_scalefit; near-duplicates from SVG extraction are resolved by keeping the point closest to the target budget. $D$ is derived as $C / (6N)$.

### llama_3

Manually digitized from the IsoFLOP figure in the Llama 3 technical report ([arxiv:2407.21783](https://arxiv.org/abs/2407.21783)). Raw data: [`eric-czech/llama3_isoflop_extraction/isoflops_points.csv`](https://github.com/eric-czech/llama3_isoflop_extraction/blob/1bc1755b76e6ee55a911549c8ec52b71cb480320/isoflops_points.csv) with columns `compute_budget` ($C$), `training_tokens` ($D$), `validation_loss`. Loss values are kept as-is from the figure (log-scale). $N$ is derived as $C / (6D)$.

### misfitting

Scaling law survey data from Marghi et al. ([arxiv:2502.18969](https://arxiv.org/abs/2502.18969)). Transformers trained on FineWeb, evaluated on C4. Raw data: [`hadasah/scaling_laws/data/scaling_results.csv`](https://github.com/hadasah/scaling_laws/blob/1f3708c0a12df0effb0ee906b1da5f9f0ff4f4f1/data/scaling_results.csv) — per-checkpoint training logs. IsoFLOP curves are constructed by: (1) building a grid of 40 log-spaced budget candidates, keeping levels where $\geq 3$ model sizes have data within 10% FLOP tolerance; (2) interpolating each run's loss at target budgets via log-log interpolation over nearby checkpoints; (3) selecting the best learning rate per model size. $D$ is derived from the target budget. Follows the interpolation approach in `hadasah/scaling_laws/paper_analysis_and_plots.py`.

### marin_202603

Marin community scaling ladder experiments: Llama 2 models trained on three datasets (Comma, DCLM, Nemotron). Raw data: vendored CSVs exported from the [Marin W&B project](https://wandb.ai/marin-community/marin/reports/Scaling-Ladders--VmlldzoxNTc0MjM1NQ). Budget is parsed from run names and multiplied by 3 to convert from forward-pass FLOPs ($\approx 2ND$) to total FLOPs ($\approx 6ND$); this factor was validated empirically across all runs. "Validation-optimal" runs (which use a different FLOPs convention) are excluded. Loss is `eval/paloma/macro_loss`.

## License

Apache 2.0

## Citation

```bibtex
@article{czech2025,
  title={Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits},
  author={Czech, Eric and Xu, Zhiwei and Elmatad, Yael and Wang, Yixin and Held, William},
  journal={arXiv preprint arXiv:2603.22339},
  year={2025}
}
```
