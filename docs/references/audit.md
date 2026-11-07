# Bibliography and citation audit, 2026-11-07

Baseline: commit `82cf54880ded6b2225c06e64751f15d9d7a57e07`.

This revision changes bibliography metadata, formal citations, citation numbering and formatting, and their validation. All other bytes in the LaTeX and article sources match the baseline after removing citation markup and the HTML bibliography and normalizing the `natbib` options. This includes the original wording, equations, numerical table values, captions, and figure references. Figures, data, the post, editorial outline, PDF, and submission archives are unchanged.

## Coverage

- All **46 original bibliography records** were checked against primary author or publication records, including entries previously described as correct in the shared audit.
- All **76 original citation targets** were inventoried in both the paper and article. The current sources contain **47 references** and **77 citation targets**: one withdrawn source's citation was removed, and two numerical-method citations were added without changing the surrounding words.
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

Additional metadata repairs include Colin Raffel's omitted authorship; complete collaboration lists and collective attribution; the official *Olmo 3* title; Luke Emberson's title, authorship, and January 2025 publication date for the Epoch article; the 2026 *Olmo Hybrid: From Theory to Practice and Back* record; the imitation-learning paper's 2024 TMLR publication year; Fortin's master's-thesis source type and institution; and publisher, DOI, and URL details. Dates identify the cited edition or publication version.

## Citation markup and generated references

- The authors [withdrew *Scaling Laws are Redundancy Laws* on 2026-03-22](https://arxiv.org/abs/2509.20721). Its one citation and bibliography entry are removed; the surrounding sentence is unchanged, including its other citations.
- The existing variable-projection explanation now cites Golub and Pereyra (1973), and the existing NNLS mention cites Lawson and Hanson (the 1995 SIAM edition). No wording is added to either passage.
- The previously stale 30-entry reference fragment, embedded article bibliography, and BibTeX now derive from the same metadata. Labels are refreshed by stable key, including the Kaplan footnote's incorrect number. Complete author lists stay in YAML; HTML explicitly abbreviates after three authors and BibTeX after ten.
- Explicit `natbib` square-bracket/comma options preserve citation punctuation when a submission includes its compiled `.bbl` directly.

Run `uv run python -m scaling_law_analysis.references` to regenerate references and `--check` to reject stale outputs. Validation checks duplicate or unknown keys, citation numbers and targets, metadata provenance, and paper/article occurrence counts. Regression tests cover the corrected identities, generated and standalone HTML, citation formatting, the occurrence inventory, and preservation of the recorded non-citation source content. The build and submission commands check citations before compiling.

Only the text sources and their citation workflow are updated in this PR. The existing PDF and historical submission archives remain unchanged; their bibliography is consequently still the pre-update version. Any substantive observations from the source review are discussed separately in chat and are not proposed as edits here.
