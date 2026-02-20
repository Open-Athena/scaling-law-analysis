# Project: Scaling Law Analysis

## Purpose

Demonstrate whether flaws in Chinchilla Approach 2 for scaling law exponent inference exist, evaluate alternatives, and produce balanced empirical and theoretical guidance for practitioners.

## Spec-Driven Development

This project follows Spec-Driven Development (SDD). The `specs/` directory is an opinionated, operator-curated interface that captures project intent and design at a level of detail chosen by the operator. It serves as the primary entrypoint for development and as curated, persistent memory for agent execution. Specs are maintained collaboratively across operator and agentic workflows.

**Key principle — asymmetric fidelity:**
- **Spec → Implementation** is lossless. Everything stated in a spec must be faithfully reflected in the implementation. Adherence is mandatory for all agentic work.
- **Implementation → Spec** is lossy by design. The implementation may contain details, decisions, and elaborations that are not captured in specs. This is expected and acceptable — specs are not intended to be a comprehensive mirror of the codebase. The operator decides what belongs in specs and what does not. However, if an implementation change affects something the spec *does* cover, the spec must be updated to match. Specs and implementation must not drift on covered topics.

**Process:**
1. **Consult specs** before starting any work. Start here, then consult the relevant spec below.
2. **Align planning** to the requirements in specs.
3. **Keep specs in sync.** Changes may originate in specs or implementation. When a change affects a topic covered by a spec, update the other side to match (see [sync.md](sync.md)). Seek operator approval before making spec changes.
4. **Verify** all work against the success criteria defined in specs.

## Spec Index

| Spec | Purpose |
|------|---------|
| [experiments.md](experiments.md) | Experiment definitions: hypotheses, methods, visualizations |
| [article.md](article.md) | Article outline, editorial guidelines, figure specifications |
| [build.md](build.md) | Build & deploy workflow: experiments → figures → article → deploy |
| [paper.md](paper.md) | arXiv preprint: LaTeX conversion of the main article, build setup, section overrides |
| [sync.md](sync.md) | Spec ↔ implementation sync: which specs control which artifacts, update rules |

## Project Structure

| Path | Role |
|------|------|
| `specs/` | Spec directory — see Spec Index above |
| `src/scaling_law_analysis/` | All implementation code |
| `  chinchilla.py` | Loss surface model, IsoFLOP sampling, Approach 2 parabolic fitting, surface fitting via variable projection |
| `  config.py` | Project paths (`PROJECT_ROOT`, `RESULTS_DIR`), shared surface configurations |
| `  references.py` | Renders `docs/references/references.yaml` to HTML |
| `  experiments/common.py` | Shared experiment config: compute budgets, sampling ranges, surface configs, bias configs, plotting utilities |
| `  experiments/exp{N}_*.py` | Individual experiment scripts; each writes to `results/experiments/exp{N}/` |
| `  experiments/run_all.py` | Runs all experiments sequentially |
| `  article/figures.py` | Generates article figures and CSV data to `results/article/figures/` |
| `  article/standalone.py` | Inlines images into self-contained HTML |
| `  paper/build.py` | Compiles LaTeX paper to PDF via Tectonic |
| `docs/` | Supplementary documentation |
| `  references/` | `references.yaml` — source of truth for citations |
| `results/` | All generated outputs (git-tracked) |
| `  experiments/` | Per-experiment figures (`exp{N}/`) |
| `  article/` | Article HTML (`article.html`, `article_standalone.html`); figure PNGs and CSVs under `figures/`; `appendix/`, `static/`, `references/` |
| `  paper/` | LaTeX source (paper.tex, references.bib) and compiled PDF |
