# Bibliography and citation audit, 2026-11-07

Baseline: commit `82cf54880ded6b2225c06e64751f15d9d7a57e07`.

This revision changes bibliography metadata, formal citations, citation numbering and formatting, and their validation. On 2026-11-08, the author also approved correcting the two Error Costs ratio labels from beta/alpha to alpha/beta for SODA (1.56) and Sparse-NMM (1.91), and changing the shared IL domain cell to "Gaming". The HTML article already uses the equivalent b/a ratio labels; the domain cell is updated in both formats. All other bytes in the LaTeX and article sources match the baseline after removing citation markup and the HTML bibliography, normalizing the `natbib` options, and reversing these explicitly recorded exceptions for comparison. This includes the remaining original wording, equations, numerical table values, captions, and figure references. Figures, data, the post, editorial outline, PDF, and submission archives are unchanged.

## Coverage

- All **46 original bibliography records** were checked against primary author or publication records, including entries previously described as correct in the shared audit.
- All **76 original citation targets** were inventoried in both the paper and article. The current sources contain **47 references** and **77 citation targets**: one withdrawn source's citation was removed, two numerical-method citations were added, and the old Epoch article citation was replaced with the dataset citation without changing the surrounding words.
- [citation-audit.json](citation-audit.json) records all original source identities, baseline BibTeX, verification URLs and available source-file hashes, all current citation occurrences, the removed occurrence, and hashes of the unchanged non-citation source content. This is an identity and consistency record; it does not assert that every scientific claim is validated by its citation.
- [references.yaml](references.yaml) is the canonical metadata source: complete ordered authors, title, year, source type, venue, identifiers, versioned URLs where available, and verification provenance. Evidence comes from the primary papers, arXiv, bioRxiv, publisher records, the University of Waterloo thesis repository, Epoch AI, and SIAM.

## Bibliographic corrections

The 15 incorrect author attributions identified in the shared audit are corrected:

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

Additional metadata repairs include Colin Raffel's omitted authorship; complete collaboration lists and collective attribution; the official *Olmo 3* title; the 2026 *Olmo Hybrid: From Theory to Practice and Back* record; the imitation-learning paper's 2024 TMLR publication year; Fortin's master's-thesis source type and institution; and publisher, DOI, and URL details. Dates identify the cited edition or publication version.

## Citation markup and generated references

- The authors [withdrew *Scaling Laws are Redundancy Laws* on 2026-03-22](https://arxiv.org/abs/2509.20721). Its one citation and bibliography entry are removed; the surrounding sentence is unchanged, including its other citations.
- The existing variable-projection explanation now cites Golub and Pereyra (1973), and the existing NNLS mention cites Lawson and Hanson (the 1995 SIAM edition). No wording is added to either passage.
- The Epoch compute-threshold article's only active citation was this compute-ranking statement, mirrored in the outline. At the author's request, both now cite Epoch AI's *Data on AI models*, using the supplied 2026/11 citation and access note; the old article is removed from the active bibliography. The Notable AI models subset accessed on 2026-11-08 was filtered using Epoch's open-weight classifications to records published by 2026-03-27 with a training-compute estimate. Llama 3.1-405B ranks first among 217 such records at 3.8e25 FLOP, supporting the existing "one of the most compute-intensive" wording. This is a retrospective comparison of the current dataset, limited to known estimates, rather than an article's explicit ranking or a historical dataset snapshot. The CSV hash, filters, counts, and leading records are recorded in the citation inventory. The sentence is unchanged.
- The previously stale 30-entry reference fragment, embedded article bibliography, and BibTeX now derive from the same metadata. Labels are refreshed by stable key, including the Kaplan footnote's incorrect number. Complete author lists stay in YAML; HTML explicitly abbreviates after three authors and BibTeX after ten.
- Explicit `natbib` square-bracket/comma options preserve citation punctuation when a submission includes its compiled `.bbl` directly.

Run `uv run python -m scaling_law_analysis.references` to regenerate references and `--check` to reject stale outputs. Validation checks duplicate or unknown keys, citation numbers and targets, metadata provenance, and paper/article occurrence counts. Regression tests cover the corrected identities, generated and standalone HTML, citation formatting, the occurrence inventory, and preservation of the recorded non-citation source content. The build and submission commands check citations before compiling.

Only the text sources and their citation workflow, plus the explicitly approved ratio-label and "Gaming" cell corrections, are updated in this PR. The existing PDF and historical submission archives remain unchanged; their bibliography is consequently still the pre-update version. Other substantive observations from the source review remain in chat and are not proposed as edits here.
