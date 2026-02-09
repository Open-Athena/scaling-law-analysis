# build

Build the project per [specs/build.md](../../specs/build.md). Skip deployment (use /deploy separately).

## Mode

- **`/build`** or **`/build changes`**: Build only what is affected by changes in the working tree.
- **`/build all`**: Run the full workflow regardless of local changes.

## Output

Summarize the mode used, then provide clickable `file://` URLs for:
- The article HTML (source)
- The standalone article HTML (self-contained, for browser review)
- The supplementary scaling parameter errors HTML and PDF
