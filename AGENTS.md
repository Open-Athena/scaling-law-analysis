## Workflow
- This project follows Spec-Driven Development (SDD) practices. Adherence to these instructions is mandatory for all agentic work.
- The `specs/` directory is the **source of truth** for project intent, requirements, and specifications:
  - `specs/project.md` — High-level project intent and structure overview
  - `specs/experiments.md` — Detailed experiment specifications
  - `specs/article.md` — Article/blog post specification
- **Never add experimental results or findings to specs.** The spec defines what to build and test, not what was observed. Results belong in code comments, notebooks, or separate analysis documents.
- Process:
    1.  **Consult Specs**: Always read the relevant spec file(s) before starting any work. Start with `specs/project.md` for an overview, then consult the specific spec (e.g., `specs/experiments.md` for experiment work).
    2.  **Align Planning**: Ensure your implementation plans directly address the requirements in the specs.
    3.  **Update Specs**: Any changes to intent or methodology must be reflected in the appropriate spec file. Update specs first and seek approval before implementing changes.
    4.  **Verification**: Verify all work against the success criteria defined in the specs.

## Environment
- Use uv to manage dependencies
- Use uv to run all python commands

## Style
- Always use absolute imports
- Do not create `__init__.py` files with re-exports; keep them empty or omit them
- Don't edit README.md unless explicitly asked