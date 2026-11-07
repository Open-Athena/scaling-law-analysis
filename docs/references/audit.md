# Bibliography and citation audit, 2026-11-07

Baseline: commit `82cf54880ded6b2225c06e64751f15d9d7a57e07`, corresponding to the March 2026 paper sources and archived submissions. This revision audits the complete bibliography and its uses, including entries previously described as correct in an external chat audit.

## Coverage and evidence

- All **46 original bibliography entries** were checked against primary author/publication records. The current bibliography has **47 entries**: 45 retained originals, one withdrawn source removed, and two foundational numerical-method references added.
- All **76 original citation targets** were reviewed in context. The current article and LaTeX each have **77 targets** after removing the withdrawn reference and adding variable projection and NNLS attribution. Grouped LaTeX commands count as multiple targets.
- [citation-audit.json](citation-audit.json) records every current citation occurrence, its baseline context, source locations, supporting URLs, disposition, original BibTeX, and available source-file SHA-256 hashes. It separately records the removed occurrence. Generated copies inherit these decisions and are checked for matching citation targets and labels.
- [references.yaml](references.yaml) stores complete ordered authors, source type, title, year, venue, persistent identifiers, versioned URLs where available, and verification dates. Author names containing particles and collective authors were checked rather than accepted blindly from machine-parsed metadata. Display abbreviation is explicit; the canonical lists are complete.
- Evidence was read from the cited primary papers, arXiv records, bioRxiv, ACL Anthology, NeurIPS/PMLR publisher records, the University of Waterloo thesis repository, Epoch AI, and SIAM. Source PDFs were cached locally for inspection, not redistributed in this repository. Publisher records/abstracts and the book contents/front matter establish the two foundational citations; full paywalled book/article text was not required for those limited attributions.

The source-use inventory is a review record, not a claim that software can determine whether a paper supports an assertion. Automated checks enforce the reviewed identities and cross-format consistency; the assessments and source locators are the human-readable evidence trail.

## Bibliographic corrections

The 15 author-attribution problems identified in the shared audit are present in the baseline and corrected here:

| Key | Correct attribution |
| --- | --- |
| `sovit` | Ibrahim M. Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, Lucas Beyer |
| `atlas_multilingual` | Shayne Longpre and coauthors |
| `waymo_scaling` | Mustafa Baniodeh and coauthors |
| `icr_scaling` | Blake Bordelon, Mary I. Letey, Cengiz Pehlevan |
| `il_scaling` | Jens Tuyls and coauthors |
| `subgroup_scaling` | Esther Rolf, Theodora Worledge, Benjamin Recht, Michael I. Jordan |
| `moe_memory_scaling` | Jan Ludziejewski and coauthors |
| `ehr_scaling` | Sheng Zhang and coauthors |
| `dit_scaling` | Zhengyang Liang and coauthors |
| `misfitting` | Margaret Li, Sneha Kudugunta, Luke Zettlemoyer |
| `dlm_scaling` | Dimitri von Rütte and coauthors |
| `reconciling_scaling` | Tim Pearce, Jinyeop Song |
| `optibert` | Megi Dervishi, Alexandre Allauzen, Gabriel Synnaeve, Yann LeCun |
| `biosignal_scaling` | Thomas Fortin |
| `mupt` | Xingwei Qu and coauthors |

Additional metadata repairs include Colin Raffel's omitted authorship; complete collaboration lists and corporate/collective attribution; the official *Olmo 3* title; the title/year/author of Luke Emberson's January 2025 Epoch article; the 2026 *Olmo Hybrid: From Theory to Practice and Back* record; TMLR's 2024 publication year for the imitation-learning paper; proper master's-thesis typing for Fortin; and official publisher/DOI/URL details. Preprint and journal dates are kept consistent with the version cited. Bibliographic normalization and completion are recorded separately from scientific support judgments.

## Corrections to source use

1. **Withdrawn work.** `redundancy_scaling` was withdrawn on 2026-03-22. The [author's arXiv withdrawal record](https://arxiv.org/abs/2509.20721) is the evidence for removing its sole supporting citation. Other sources remain attached to the related-work statement. The original record remains in the audit inventory.
2. **Historical compute context.** The January 2025 Epoch article does not establish which open models ranked highest in early 2026. The text now describes its historical Llama 3 compute observation without asserting a later ranking.
3. **Related work versus direct extensions.** In-context regression theory and parameter-counting analysis are identified by their actual scope instead of being described as direct empirical Chinchilla-surface extensions.
4. **Fitting survey.** The text reports the survey's 51-paper scope and its count of 23 papers without an optimization-process description. The initialization discussion is located at Sections 6 and 7.4. An overbroad assertion about all optimizers and a malformed quotation were removed. BFGS is correctly described as approximating inverse-Hessian information from gradients, with a [primary SciPy documentation link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
5. **Foundational attribution.** The method section cites Golub and Pereyra's 1973 separable nonlinear least-squares work and Lawson and Hanson's 1995 SIAM edition for NNLS. VPNLS is presented as an application of established numerical methods.
6. **Strength of evidence.** Selected curve examples establish that bias conditions can occur; they do not establish prevalence or quantify errors in the original studies. Robustness of allocation prescriptions is distinguished from universal correctness of a functional form.
7. **Ratio notation and implementation claims.** The paper's cost section labeled the SODA and Sparse-NMM ratios as beta/alpha while reporting alpha/beta values (1.56 and 1.91); the labels now match the verified constants and article convention. The linked example is a bounded-grid TypeScript/OLS implementation. Its 147-line file does not certify a continuous global optimum, so the paper/article/post no longer claim such a guarantee or an unqualified 70-line implementation.

## Published-exponent table

Every numerical row was checked at the source locations recorded in the inventory, including the Evo extraction repository and both separate xLSTM figures. Corrections and qualifications are:

- Protein CLM's data exponent is **0.421**, as printed in Figure 2, rather than 0.422. Its allocation ratio is recomputed.
- The game examples are **imitation learning** on NetHack and Atari, not all Atari/reinforcement-learning experiments.
- DLM's pair **0.589/0.411** refers specifically to the **uniform-noise** configuration.
- DeepSeek and DLM measure model resource with non-embedding FLOPs per token, so their exponents are not silently treated as exact parameter-count exponents.
- The Olmo Hybrid public report's Table 18 gives rounded free-fit pairs **0.46/0.54**, **0.49/0.51**, **0.55/0.45**. The table uses this precision and recomputed ratios, and notes the wide, overlapping confidence intervals. The previous three-decimal values were consistent with the rounded estimates, so removing the rows would have been unwarranted. This revision cites the public 2026 report rather than claiming a verified 2025 version.
- The ratio column is explicitly **a/b** for empirical allocation estimates. Identifying that ratio with loss-surface **beta/alpha** requires the ideal Chinchilla model and compute convention. For example, NMM Table 2's late-fusion allocation pair does not equal the ratio of its separately printed surface exponents. The original published values are preserved and the interpretation is qualified.
- ProGen3 directly reports the **1.479** exponent of optimal model size versus data. Dense Retrieval's **2.3393** is calculated from its reported **1.31/0.56**. The caption now distinguishes the two cases.

The SODA and Sparse-NMM surface constants used in the numerical simulations match the cited primary tables/equation. Their values, the fitted Llama 3 data, and experimental procedures are unchanged. Consequently the principal computed DCL values and bootstrap intervals are unchanged; the changes concern attribution, ancillary table values, and interpretation of cited evidence.

## Other citation-bearing artifacts

- The generated 30-entry HTML reference fragment was stale; both it and the embedded article list now come from the 47-entry canonical source. All inline labels are refreshed by key, including the Kaplan footnote that previously displayed `[14]` for reference 16.
- LaTeX, article, standalone article, relevant outline/specification text, source-provenance comments, and the explanatory post were reviewed for affected references. The landing page contains the abstract and output links rather than an independent bibliography.
- Bare links were checked in context: the Karpathy tutorial, derivation document, Evo and Llama figure-extraction repositories, the public 15-item IsoFLOP curve catalogue, VPNLS repository, and JavaScript example. The figure catalogue is a collection of examples, not an additional quantified prevalence study. Mathematical derivations and this project's own results are not misclassified as third-party citations.
- The external Claude demonstration URL responds to retrieval, but its interactive runtime was not independently evaluated. It is retained as an optional demonstration link, not used as primary bibliographic evidence. The linked TypeScript source was read directly, including its bounded search and OLS inner solve.
- All four March submission archives contain 46 bibliography entries. Their paths, reference keys, and hashes are recorded in the inventory. They are preserved unchanged; the new submission package contains the corrected compiled bibliography.
- An independent compile exposed citation-style drift: removing `\bibliographystyle` from the submission changed square-bracket/comma citations to parentheses/semicolons. The `natbib` options are now explicit in the preamble, preserving citation formatting in the self-contained package.

## Reproduction and maintenance

```bash
uv run python -m scaling_law_analysis.references
uv run python -m scaling_law_analysis.article.standalone
uv run python -m scaling_law_analysis.references --check
uv run pytest tests/ -q
uv run pyrefly check .
uv run python -m scaling_law_analysis.paper.build
uv run python -m scaling_law_analysis.paper.submit
```

Builds use reviewed metadata without making network metadata requests. CI checks bibliography drift and the regression tests. The regression tests cover real defects from this audit: wrong citation labels, orphan/duplicate targets, mismatched use counts, incomplete attribution, unsafe metadata, stale generated files, capitalization/name escaping, and explicit author-list abbreviation.

The publication update is prepared as a PR with regenerated artifacts and a new dated arXiv source archive. Merging, deploying the website, and uploading an arXiv revision are separate publication actions.

## Validation of this revision

- Full unit suite: **67 passed, 3 skipped**; all **15 reference regressions pass**.
- Black, Pyrefly, the offline reference consistency check, and `git diff --check` pass.
- All 37 paper pages were rendered for layout review, with detailed inspection of all five bibliography pages and the exponent table. The final reference/table renders match the reviewed pages exactly; no overfull-box or missing-citation warnings remain. Existing underfull-box spacing warnings remain in long URL paragraphs.
- Browser inspection found 47 rendered references, no missing citation targets, no broken images, and no horizontal page overflow at the inspected desktop viewport. The standalone HTML is also checked byte-for-byte against the source article with images inlined.
- The new `20261107/arxiv.tar.gz` contains exactly 47 unique compiled bibliography entries. Its rewritten source matches the repository source. The extracted archive compiles independently, and its extracted PDF text matches the repository PDF text exactly, including citation punctuation.
- Principal simulation inputs and results were not changed, so simulations were not rerun as part of this bibliographic correction.
