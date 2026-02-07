"""Shared reference management.

Loads a canonical set of references from YAML and provides
format-specific renderers (currently HTML; LaTeX/BibTeX planned
for paper output).

Source: docs/references/references.yaml

Schema (references.yaml):

    - key: chinchilla           # unique id, used across all output formats
      title: Training Compute-Optimal Large Language Models
      venue: ArXiv              # journal, conference, or ArXiv
      url: https://...          # optional

Usage (HTML generation for the article):
    uv run python -m scaling_law_analysis.references
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from scaling_law_analysis.config import PROJECT_ROOT, RESULTS_DIR

REFERENCES_YAML = PROJECT_ROOT / "docs" / "references" / "references.yaml"
GENERATED_DIR = RESULTS_DIR / "article" / "references"


@dataclass
class Reference:
    """A single bibliographic reference."""

    key: str
    title: str
    venue: str
    url: str | None = None


def load_references(path: str | Path) -> list[Reference]:
    """Load references from a YAML file.

    Args:
        path: Path to the references YAML file.

    Returns:
        List of Reference objects in file order.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Reference(**entry) for entry in data]


def build_citation_map(references: list[Reference]) -> dict[str, int]:
    """Build a key-to-number mapping for inline citations.

    Args:
        references: List of Reference objects.

    Returns:
        Dict mapping each key to its 1-indexed position.
    """
    return {ref.key: i + 1 for i, ref in enumerate(references)}


def cite_html(key: str, number: int) -> str:
    """Render an inline HTML citation.

    Args:
        key: Reference key (must match a key in references.yaml).
        number: Display number (1-indexed, matches position in references list).

    Returns:
        HTML string like ``<sup><a href="#ref-chinchilla">[1]</a></sup>``.
    """
    return f'<sup><a href="#ref-{key}">[{number}]</a></sup>'


def render_references_html(references: list[Reference]) -> str:
    """Render references as an HTML section.

    Args:
        references: List of Reference objects (order determines numbering).

    Returns:
        Complete ``<section id="references">...</section>`` HTML string.
    """
    items = []
    for ref in references:
        entry = f'"{ref.title}," <em>{ref.venue}</em>.'
        if ref.url:
            entry += f' <a href="{ref.url}">{ref.url}</a>'
        items.append(f'    <li id="ref-{ref.key}">{entry}</li>')

    items_html = "\n".join(items)
    return f"""\
<section id="references">
    <h2>References</h2>
    <ol>
{items_html}
    </ol>
</section>"""


def main() -> None:
    """Load references from YAML and write generated outputs to disk."""
    refs = load_references(REFERENCES_YAML)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    out_path = GENERATED_DIR / "references.html"
    out_path.write_text(render_references_html(refs))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
