# Agent Instructions: Spec-Driven Development (SDD)

This project follows SDD practices. Adherence to these instructions is mandatory for all agentic work.

## Source of Truth
- The `specs/` directory is the **sole source of truth** for project and task intent.
- `specs/project.md` contains global intent, principles, and core objectives.
- `specs/tasks/<task>/task.md` contains intent and results for incremental tasks.

## Workflow
1.  **Consult Specs**: Always read `specs/project.md` and create or read the relevant `specs/tasks/<task>/task.md` before starting any work.
2.  **Align Planning**: Ensure your implementation plans directly address the requirements in the specs.
3.  **Update Specs**: If a task requires a change in intent or discovery leads to new requirements, update the specs first and seek approval.
4.  **Verification**: Verify all work against the success criteria defined in the specs.
5.  **Update Results**: Update the `specs/tasks/<task>/task.md` with the results of your work.

## Implementation
- Use uv to manage dependencies
- Use uv to run all python commands
- Always use absolute imports (e.g., `from scaling_law_simulation.chinchilla import func`)
- Do not create `__init__.py` files with re-exports; keep them empty or omit them