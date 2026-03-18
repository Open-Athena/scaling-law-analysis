# Build & Deploy

Commands for generating outputs and deploying artifacts. Start with the [Full Workflow](#full-workflow) for end-to-end sequencing; step details are below. Manual sync steps from [sync.md](sync.md) are referenced where they apply.

## Full Workflow

1. **Run experiments**: `uv run python -m scaling_law_analysis.experiments.run_all`
2. **Generate article figures** (outputs to `results/article/`; includes copying appendix figures from experiment results):
   `uv run python -m scaling_law_analysis.article.figures`
3. **Generate references HTML**: `uv run python -m scaling_law_analysis.references`
4. **Sync CSV data with article text** — skip if figures unchanged (see [sync.md > Implementation → Implementation](sync.md#implementation--implementation))
5. **Edit article**: `results/article/article.html`
6. **Build standalone HTML** (see [Standalone HTML](#standalone-html))
8. **Push to `main`**
9. **Deploy** (when ready to publish) — trigger "Deploy Article" workflow from the Actions tab or via `gh workflow run deploy.yml`
   - **Agent note — TLS errors**: The `gh` CLI may fail with `x509: OSStatus -26276` if the shell session cannot access the macOS system keychain for certificate verification (common in sandboxed environments). If your environment supports sandbox permissions, run `gh` commands without sandboxing (e.g. `required_permissions: ["all"]`). If a combined command (e.g. `git push && gh workflow run`) is used, ensure the entire command runs without sandbox restrictions.

---

## Step Details

### Standalone HTML

Inlines local images as base64 into a single self-contained HTML file:

```bash
uv run python -m scaling_law_analysis.article.standalone
```

Reads `results/article/article.html` → writes `results/article/article_standalone.html`.

### Deploy to GitHub Pages

The standalone HTML is deployed to GitHub Pages via a manually triggered workflow (`.github/workflows/deploy.yml`). It copies `results/article/article_standalone.html` to the `gh-pages` branch as `index.html`. Trigger it from the Actions tab → "Deploy Article" → "Run workflow".

One-time setup: in repo Settings → Pages, set source to **Deploy from a branch**, branch `gh-pages`, root `/`.
