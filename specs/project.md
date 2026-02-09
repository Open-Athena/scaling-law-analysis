# Project: Scaling Law Analysis

## Purpose

Demonstrate whether flaws in Chinchilla Approach 2 for scaling law exponent inference exist, evaluate alternatives, and produce balanced empirical and theoretical guidance for practitioners.

## Spec-Driven Development

This project follows Spec-Driven Development (SDD). The `specs/` directory is a compressed, authoritative representation of project intent and design maintained collaboratively across human and agentic workflows. It serves as the primary entrypoint for development and as curated, persistent memory for agent execution. Adherence is mandatory for all agentic work.

**Process:**
1. **Consult specs** before starting any work. Start here, then consult the relevant spec below.
2. **Align planning** to the requirements in specs.
3. **Update specs first** when intent or methodology changes; seek approval before implementing.
4. **Verify** all work against the success criteria defined in specs.

**Never add experimental results or findings to specs.** Specs define what to build and test, not what was observed.

## Spec Index

| Spec | Purpose |
|------|---------|
| [experiments.md](experiments.md) | Experiment definitions: hypotheses, methods, visualizations |
| [article.md](article.md) | Article outline, editorial guidelines, figure specifications |
| [build.md](build.md) | Build & deploy workflow: experiments → figures → article → deploy |
| [sync.md](sync.md) | Spec ↔ output sync: which specs control which artifacts, update rules |

## Project Structure

```
specs/                       # This directory — see Spec Index above
src/scaling_law_analysis/    # All implementation code
  chinchilla.py              #   Loss surface and Approach 2 fitting
  config.py                  #   Shared surface configurations
  experiments/               #   Experiment scripts and runner
  article/                   #   Article figure generation and standalone builder
  references.py              #   Reference list generator from YAML
docs/                        # Supplementary documentation
  references/                #   references.yaml (source of truth for citations)
results/                     # All generated outputs (git-tracked)
  experiments/               #   Per-experiment figures (exp{N}/)
  article/                   #   Article HTML, figures, CSVs, and supplementary PDF
```

## Implementation Map

| Module | Role |
|--------|------|
| `chinchilla.py` | Loss surface model, IsoFLOP sampling, Approach 2 parabolic fitting, surface fitting via variable projection |
| `config.py` | Project paths (`PROJECT_ROOT`, `RESULTS_DIR`) |
| `experiments/common.py` | Shared experiment config: compute budgets, sampling ranges, surface configs, bias configs, plotting utilities |
| `experiments/exp{N}_*.py` | Individual experiment scripts; each writes to `results/experiments/exp{N}/` |
| `experiments/run_all.py` | Runs all experiments sequentially |
| `article/figures.py` | Generates article figures and CSV data to `results/article/` |
| `article/standalone.py` | Inlines images into self-contained HTML |
| `references.py` | Renders `docs/references/references.yaml` to HTML |
