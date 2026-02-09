# Build & Deploy

Commands for generating outputs and deploying artifacts. Start with the [Full Workflow](#full-workflow) for end-to-end sequencing; step details are below. Manual sync steps from [sync.md](sync.md) are referenced where they apply.

## Full Workflow

1. **Run experiments**: `uv run python -m scaling_law_analysis.experiments.run_all`
2. **Generate article figures** (outputs to `results/article/`):
   `uv run python -m scaling_law_analysis.article.figures`
3. **Generate references HTML**: `uv run python -m scaling_law_analysis.references`
4. **Sync CSV data with article text** — skip if figures unchanged (see [sync.md > CSV Data in Article Text](sync.md#csv-data-in-article-text))
5. **Edit article**: `results/article/article.html`
6. **Build supplementary PDF** — skip if `scaling_parameter_errors.html` hasn't changed (see [Scaling Parameter Errors PDF](#scaling-parameter-errors-pdf))
7. **Build standalone HTML** (see [Standalone HTML](#standalone-html))
8. **Push to `main`**
9. **Deploy** (when ready to publish) — trigger "Deploy Article" workflow from the Actions tab or via `gh workflow run`
   - Agents: run without sandbox to avoid SSL errors

---

## Step Details

### Standalone HTML

Inlines local images as base64 into a single self-contained HTML file:

```bash
uv run python -m scaling_law_analysis.article.standalone
```

Reads `results/article/article.html` → writes `results/article/article_standalone.html`.

### Scaling Parameter Errors PDF

Requires Playwright (`uv pip install playwright && uv run playwright install chromium`):

```bash
uv run python -c "
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

async def main():
    html = Path('results/article/scaling_parameter_errors.html').resolve()
    pdf = html.with_suffix('.pdf')
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f'file://{html}', wait_until='networkidle')
        await page.evaluate('() => MathJax.startup.promise')
        await page.wait_for_timeout(2000)
        await page.pdf(path=str(pdf), format='Letter', print_background=True,
                       scale=0.95, margin=dict(top='12mm', bottom='12mm', left='12mm', right='12mm'))
        await browser.close()

asyncio.run(main())
"
```

The article links to this PDF via a `blob/main` URL on GitHub, so no special commit ordering is needed — the link always resolves to the latest version on `main`.

### Deploy to GitHub Pages

The standalone HTML is deployed to GitHub Pages via a manually triggered workflow (`.github/workflows/deploy.yml`). It copies `results/article/article_standalone.html` to the `gh-pages` branch as `index.html`. Trigger it from the Actions tab → "Deploy Article" → "Run workflow".

One-time setup: in repo Settings → Pages, set source to **Deploy from a branch**, branch `gh-pages`, root `/`.
