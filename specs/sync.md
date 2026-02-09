# Spec ↔ Output Sync

After changing any spec or output, check this document to see what else needs manual updating. Syncs are bidirectional: spec changes may require output edits, and output edits may require spec updates.

## Experiments

- `specs/experiments.md` ↔ `results/experiments/exp{N}/`
- **Spec → Output**: methodology changes require re-running experiments
- **Output → Spec**: if experiment code evolves (e.g. new parameters, revised visualizations), update the spec to match

## Article

- `specs/article.md` ↔ `results/article/`
- **Spec → Output**: outline or figure spec changes require updating `article.html` and regenerating figures
- **Output → Spec**: if article prose or figure content is revised directly in `article.html`, update the spec to reflect the new intent

### CSV Data in Article Text

The figure generator exports numerical results alongside each figure:

- `results/article/extrapolation_error_data.csv` — token prediction errors by surface and grid width
- `results/article/off_center_extrapolation_data.csv` — off-center sampling errors by grid width

Specific values from these CSVs (e.g. error percentages, token counts) are hardcoded in `results/article/article.html` — both in prose and in data tables. After regenerating figures, check whether the CSV data has changed and update the corresponding text and tables in the article to match.

## Specs Cross-References

Specs reference each other and external files (e.g. `AGENTS.md`, `README.md`). When renaming, moving, or restructuring any spec, review links in all markdown files across the project — not just within `specs/`.

## Rules

- **Never add experimental results or findings to specs.** Specs define intent and methodology only.
- Changes to methodology or structure go through specs first; changes to presentation or content discovered during output editing get reflected back into specs.
