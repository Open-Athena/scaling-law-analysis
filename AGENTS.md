## Workflow
- This project follows Spec-Driven Development (SDD) practices. Adherence to these instructions is mandatory for all agentic work.
- `specs/project.md` is the **sole source of truth** for project intent, requirements, and specifications.
- Process:
    1.  **Consult Specs**: Always read `specs/project.md` before starting any work.
    2.  **Align Planning**: Ensure your implementation plans directly address the requirements in the specs.
    3.  **Update Specs**: Any changes to intent or methodology must be reflected in `specs/project.md`. Update specs first and seek approval before implementing changes.
    4.  **Verification**: Verify all work against the success criteria defined in the specs.

## Environment
- Use uv to manage dependencies
- Use uv to run all python commands

## Style
- Always use absolute imports
- Do not create `__init__.py` files with re-exports; keep them empty or omit them
- Don't edit README.md unless explicitly asked