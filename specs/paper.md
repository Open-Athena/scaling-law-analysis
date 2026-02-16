# Paper Spec: arXiv Preprint

## Intent

Port the main article (`results/article/article.html`) to LaTeX for submission to arXiv as a preprint. The paper should be a faithful conversion of the article's content, structure, and figures. Deviations from the article are documented below as section-level overrides. When the article and this spec conflict, this spec takes precedence.

The paper emphasizes linear separability (optimizing exponential terms separately from linear terms) as the key mathematical insight underlying the proposed alternative to Approach 2. VPNLS is one specific implementation of this principle. Frame it accordingly: the contribution is the observation that linear separation eliminates the biases of the parabolic approximation, and VPNLS is our realization of that idea.

Some HTML-specific features (collapsible sections, interactive elements, callout box styling) may not transfer cleanly to LaTeX. See decisions below.

## Build

Requires [Tectonic](https://tectonic-typst.github.io/tectonic/), a self-contained TeX engine that auto-downloads packages on demand. Install: `brew install tectonic`.

Compile:

```bash
uv run python -m scaling_law_analysis.paper.build
```

Source files live in `results/paper/` (paper.tex, references.bib). Output PDF is `results/paper/paper.pdf`. Build script is `src/scaling_law_analysis/paper/build.py`.

## Author / Affiliation

```
Eric Czech
Open Athena AI Foundation
```

No email addresses unless explicitly instructed.

## Source Material

The primary source for all prose, math, figures, and tables is `results/article/article.html`. Consult `specs/article.md` for the editorial outline and structural intent behind each section.

## Section Mapping

The paper follows the article's structure with these renames and overrides:

### Title (override)

"Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits"

| Article Section | Paper Section | Notes |
|---|---|---|
| *(title/subtitle)* | `\title` / `\begin{abstract}` | See Title and Abstract overrides above/below |
| Motivation | Introduction | See Introduction override below |
| Preliminaries: Loss Surface, Notation, and Fitting Methods | Preliminaries | Direct conversion |
| The Happy Path: Symmetric Surfaces | The Happy Path: Symmetric Surfaces | Direct conversion |
| Asymmetric Surfaces: Intercept and Extrapolation Errors | Asymmetric Surfaces: Intercept and Extrapolation Errors | Direct conversion |
| Off-Center Sampling: Exponent and Extrapolation Errors | Off-Center Sampling: Exponent and Extrapolation Errors | Direct conversion |
| IsoFLOP Curves in the Wild: Evidence from Published Studies | IsoFLOP Curves in the Wild | Direct conversion |
| Robust Fits: Unbiased Estimation with Linear Separation | Robust Fits: Unbiased Estimation with Linear Separation | Direct conversion |
| Conclusion | Conclusion | Direct conversion |
| Appendix A, B | Appendix A, B | Direct conversion |

### Abstract (override)

The article has no abstract; it uses a subtitle and opens directly into Motivation. The paper needs a standalone abstract. Render the following as-is (with necessary LaTeX formatting):

> Chinchilla Approach 2 is arguably the most widely adopted method for fitting neural scaling laws, used by leading AI labs and academic groups to plan compute-optimal training configurations. The method fits parabolas to IsoFLOP loss curves and extracts scaling exponents through a sequence of simple polynomial and linear regressions. We show that this parabolic approximation introduces systematic biases in compute-optimal allocation estimates, even on noise-free synthetic data under ideal experimental conditions. Two independent bias sources are identified: surface asymmetry (alpha != beta), which shifts intercept estimates, and off-center sampling, which distorts intercepts or exponents depending on whether the offset is constant or varies with compute budget. These biases compound in practice and grow with sampling grid width. We show that exploiting the partially linear structure of the Chinchilla loss surface, by separating linear coefficients from nonlinear exponents, eliminates these biases entirely. Our realization of this approach, Variable Projection with Non-negative Least Squares (VPNLS), recovers all five surface parameters with machine precision across all conditions tested, while offering comparable data efficiency to Approach 2, high stability with full parametric inference, and no dependence on specialized IsoFLOP experiment designs.

### Introduction (override)

Rename "Motivation" to "Introduction." The content is the same, but restructure the opening to work as a standard paper introduction rather than a blog-style motivation section. The article's bullet-point flow should become connected prose paragraphs.

## References

### Source of truth

All references originate from `docs/references/references.yaml`. The rendered reference list in the article HTML (`results/article/article.html`, near the bottom) provides the full formatted entries. Use this to populate `results/paper/references.bib` with proper BibTeX entries (author, title, year, journal/booktitle, url).

### Inline hyperlinks

The article contains a small number of bare hyperlinks that are not formal references (e.g., a link to a Karpathy tutorial, a link to the derivation PDF on GitHub). In the paper, convert these to footnotes using `\footnote{\url{...}}`.

## Figures

Figures are generated by `src/scaling_law_analysis/article/figures.py` into `results/article/`. The paper references these via relative paths (e.g., `../article/figure_name.png`). No figure copying is needed; arXiv submission will be PDF-only.

## Collapsible Sections

The article uses HTML `<details>` elements for collapsible raw-data tables. In the paper, move these to the appendix.

## Key Result Boxes

The article uses styled callout boxes for key results. In the paper, render these as a brief bold heading (e.g., `\paragraph{Key Result.}`) followed by the text. No frames, boxes, or extra packages.
