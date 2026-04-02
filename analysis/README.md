# Delphi Scaling Analysis

Comparison of Llama 3 and Marin Delphi isoFLOP scaling law fits.

## Data Sources

- **Llama 3**: Digitized isoFLOP data from [Meta Llama 3 paper](https://arxiv.org/abs/2407.21783) ([CSV](https://github.com/eric-czech/llama3_isoflop_extraction))
- **Delphi**: Marin Delphi isoFLOP records (adamh_scaling_v6) — 78 records, 7 FLOP budgets (~2.9e18 to ~3.1e20)
  - Links:
    - [Extraction gist](https://gist.github.com/eric-czech/35e3b493c5d6a01dcea2f8dba8708a98)
    - [Delphi scaling setup PR (marin#3292)](https://github.com/marin-community/marin/pull/3292)
    - [Delphi parent issue (marin#1337)](https://github.com/marin-community/marin/issues/1337#issuecomment-4085058300)
    - [Discord discussion](https://discord.com/channels/1354881461060243556/1375005693899309126/1483907317291552972)
  - Forecasts:
    - runs: https://wandb.ai/marin-community/marin/table (search `adamh-scaling-ladder-nemotron-optimal-.*`)
    - [original forecasts](https://github.com/marin-community/marin/issues/1337#issuecomment-3992593231):
      - `1e21`: 2.75, `1e22`: 2.55, `1e23`: 2.40
    - `1e21` forecast:
      - [marin#1337/4006185971](https://github.com/marin-community/marin/issues/1337#issuecomment-4006185971)
      - [wandb/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021](https://wandb.ai/marin-community/marin/runs/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021)
      - Total parameters: 3,383,110,656 (from `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/hf/step-22056`)
      - Total tokens: 46,256,881,664 (from wandb overview)
      - forecast: 2.7587, actual: 2.75814
    - `1e22` forecast:
      - [marin#1337/4016705345](https://github.com/marin-community/marin/issues/1337#issuecomment-4016705345)
      - [wandb/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e](https://wandb.ai/marin-community/marin/runs/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e)
      - Total parameters: 9,714,698,752 (from `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234`)
      - Total tokens: 160,369,213,440
      - forecast: 2.55, actual: 2.53295 (in gh post), 2.53079 (in wandb run at very end)
    - `1e23` forecast:
      - [marin#1337/4148827725](https://github.com/marin-community/marin/issues/1337#issuecomment-4148827725)
      - [wandb/adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb](https://wandb.ai/marin-community/marin/runs/adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb)
      - Total params: 24,963,098,112 (from `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb/hf/step-74883`)
      - Total tokens: 628,172,521,472
      - forecast: 2.3660 / 2.349 / 2.4 (original), actual (spiky run): 2.3546
  - Posts
    - [Announcement](https://x.com/percyliang/status/2034367256277533100)
    - [W&B tracking](https://x.com/WilliamBarrHeld/status/2037628853787738461)
- **Prior Nemotron results**: [Discord thread](https://discord.com/channels/1354881461060243556/1356490712199462912/1460423753895772274)

## Usage

```bash
uv run python3 analysis/delphi_scaling_analysis.py
```

Results are saved to `analysis/results/`.
