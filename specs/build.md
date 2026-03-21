# Build & Deploy

Commands for generating outputs and deploying artifacts. Start with the [Full Workflow](#full-workflow) for end-to-end sequencing; step details are below. Manual sync steps from [sync.md](sync.md) are referenced where they apply.

## Full Workflow

1. **Run experiments**: `uv run python -m scaling_law_analysis.experiments.run_all` — all figures are saved as both PNG and PDF
2. **Generate article figures** (outputs to `results/article/`; includes copying appendix figures from experiment results; copies both PNG and PDF):
   `uv run python -m scaling_law_analysis.article.figures`
3. **Generate references HTML**: `uv run python -m scaling_law_analysis.references`
4. **Sync CSV data with article text** — skip if figures unchanged (see [sync.md > Implementation → Implementation](sync.md#implementation--implementation))
5. **Edit article**: `results/article/article.html`
6. **Build standalone HTML** (see [Standalone HTML](#standalone-html))
7. **Build paper PDF** (see [Paper PDF](#paper-pdf))
8. **Push to `main`**
9. **Deploy** (see [Deploy to GitHub Pages](#deploy-to-github-pages))

---

## Step Details

### Standalone HTML

Inlines local images as base64 into a single self-contained HTML file:

```bash
uv run python -m scaling_law_analysis.article.standalone
```

Reads `results/article/article.html` → writes `results/article/article_standalone.html`.

### Paper PDF

Compiles `results/paper/paper.tex` to PDF using [Tectonic](https://tectonic-typst.github.io/tectonic/), a self-contained TeX engine that auto-downloads packages on demand. The paper references PDF figures from `results/article/figures/` via relative paths (the article HTML uses the PNG versions), so the article figures step must run first.

```bash
brew install tectonic  # one-time
uv run python -m scaling_law_analysis.paper.build
```

Output: `results/paper/paper.pdf`.

### Deploy to GitHub Pages

Deploys `article_standalone.html` to the `gh-pages` branch as `index.html`:

```bash
gh workflow run deploy.yml
```

**Agent note — TLS errors**: The `gh` CLI may fail with `x509: OSStatus -26276` if the shell session cannot access the macOS system keychain for certificate verification (common in sandboxed environments). If your environment supports sandbox permissions, run `gh` commands without sandboxing (e.g. `required_permissions: ["all"]`).

One-time setup: in repo Settings → Pages, set source to **Deploy from a branch**, branch `gh-pages`, root `/`.
