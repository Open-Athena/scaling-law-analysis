"""Render reviewed offline metadata to HTML and BibTeX; --check rejects drift."""

from __future__ import annotations

import argparse
import html
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import yaml

from scaling_law_analysis.config import PROJECT_ROOT, RESULTS_DIR

REFERENCES_YAML = PROJECT_ROOT / "docs/references/references.yaml"
GENERATED_DIR = RESULTS_DIR / "article/references"
ARTICLE_PATH = RESULTS_DIR / "article/article.html"
BIB_PATH = RESULTS_DIR / "paper/references.bib"
PAPER_PATH = RESULTS_DIR / "paper/paper.tex"
KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
SECTION_RE = re.compile(r'<section\s+id="references">.*?</section>', re.S)
LINK_RE = re.compile(
    r"(<a\b[^>]*\bhref=[\"\']#ref-([a-z0-9_]+)[\"\'][^>]*>)" r"\s*\[\d+\]\s*(</a>)"
)


@dataclass
class Reference:
    key: str
    title: str
    authors: list[str]
    year: int
    venue: str
    url: str
    entry_type: str = "article"
    doi: str | None = None
    eprint: str | None = None
    institution: str | None = None
    volume: str | None = None
    pages: str | None = None
    publisher: str | None = None
    month: int | None = None
    note: str | None = None
    verified_on: str = ""
    evidence_urls: list[str] = field(default_factory=list)


def load_references(path: str | Path = REFERENCES_YAML) -> list[Reference]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("References must be a YAML list")
    refs = [Reference(**entry) for entry in data]
    duplicates = [
        key for key, count in Counter(ref.key for ref in refs).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate reference keys: {duplicates}")
    identities: dict[str, str] = {}
    for ref in refs:
        if not KEY_RE.fullmatch(ref.key):
            raise ValueError(f"Invalid reference key: {ref.key}")
        if not ref.title.strip() or not ref.venue.strip():
            raise ValueError(f"Missing title or venue: {ref.key}")
        if (
            not isinstance(ref.authors, list)
            or not ref.authors
            or any(
                not isinstance(name, str) or not name.strip() for name in ref.authors
            )
        ):
            raise ValueError(f"Missing authors: {ref.key}")
        if "others" in ref.authors:
            raise ValueError(f"Canonical author list must be complete: {ref.key}")
        if type(ref.year) is not int or not 1900 <= ref.year <= 2100:
            raise ValueError(f"Invalid year: {ref.key}")
        if ref.entry_type not in {
            "article",
            "inproceedings",
            "mastersthesis",
            "book",
            "misc",
        }:
            raise ValueError(f"Invalid entry type: {ref.key}")
        for url in [ref.url, *ref.evidence_urls]:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"Invalid source URL for {ref.key}: {url}")
        if not ref.evidence_urls or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", ref.verified_on
        ):
            raise ValueError(f"Missing verification provenance: {ref.key}")
        if ref.entry_type == "mastersthesis" and not ref.institution:
            raise ValueError(f"Missing thesis institution: {ref.key}")
        identity = (
            ref.doi or re.sub(r"v\d+$", "", ref.eprint or "") or ref.url
        ).lower()
        if identity in identities:
            raise ValueError(
                f"Duplicate publication: {identities[identity]}, {ref.key}"
            )
        identities[identity] = ref.key
    return refs


def build_citation_map(references: list[Reference]) -> dict[str, int]:
    return {ref.key: i + 1 for i, ref in enumerate(references)}


def cite_html(key: str, number: int) -> str:
    if not KEY_RE.fullmatch(key) or number < 1:
        raise ValueError("Invalid citation key or number")
    return f'<sup><a href="#ref-{key}">[{number}]</a></sup>'


def _display_authors(authors: list[str]) -> str:
    names = []
    for author in authors[:3]:
        if ", " in author:
            family, given = author.split(", ", 1)
            names.append(f"{given} {family}")
        else:
            names.append(author.strip("{}"))
    return "; ".join(names) + (" et al." if len(authors) > 3 else "")


def render_references_html(references: list[Reference]) -> str:
    items = []
    for ref in references:
        authors = html.escape(_display_authors(ref.authors))
        title = html.escape(
            ref.title + ("" if ref.title.endswith((".", "?", "!")) else ".")
        )
        venue = html.escape(ref.venue)
        url = html.escape(ref.url, quote=True)
        entry = f'{authors} ({ref.year}). "{title}" <em>{venue}</em>.'
        if ref.note:
            entry += " " + html.escape(ref.note.rstrip(".")) + "."
        entry += f' <a href="{url}">{url}</a>'
        items.append(f'    <li id="ref-{ref.key}">{entry}</li>')
    return (
        '<section id="references">\n    <h2>References</h2>\n    <ol>\n'
        + "\n".join(items)
        + "\n    </ol>\n</section>"
    )


def _tex(value: str) -> str:
    # BibTeX/plainnat needs explicit TeX accents for sorting and author labels.
    accents = {
        "é": r"{\'e}",
        "è": r"{\`e}",
        "í": r"{\'i}",
        "ó": r"{\'o}",
        "á": r"{\'a}",
        "ú": r"{\'u}",
        "ö": r"{\"o}",
        "ü": r"{\"u}",
        "ä": r"{\"a}",
        "ë": r"{\"e}",
        "ï": r"{\"i}",
        "ç": r"{\c c}",
        "ñ": r"{\~n}",
        "Ç": r"{\c C}",
        "ł": r"{\l}",
        "Ł": r"{\L}",
        "ń": r"{\'n}",
        "ś": r"{\'s}",
        "ć": r"{\'c}",
        "ř": r"{\v r}",
        "š": r"{\v s}",
        "ž": r"{\v z}",
        "ø": r"{\o}",
        "å": r"{\aa}",
        "ý": r"{\'y}",
    }
    escaped = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}
    return "".join(accents.get(char, escaped.get(char, char)) for char in value)


def render_references_bibtex(references: list[Reference]) -> str:
    entries = ["% Generated from docs/references/references.yaml; do not edit.\n"]
    for ref in references:
        # Keep full lists in YAML and explicitly mark any display abbreviation.
        authors = ref.authors[:10] + (["others"] if len(ref.authors) > 10 else [])
        fields = {
            "title": "{" + _tex(ref.title) + "}",
            "author": " and ".join(_tex(name) for name in authors),
            "year": str(ref.year),
        }
        venue_field = {
            "article": "journal",
            "inproceedings": "booktitle",
            "mastersthesis": "school",
            "book": "publisher",
            "misc": "howpublished",
        }[ref.entry_type]
        fields[venue_field] = _tex(ref.institution or ref.venue)
        for key in ("volume", "pages", "publisher", "note"):
            if value := getattr(ref, key):
                fields[key] = _tex(value)
        if ref.month is not None:
            fields["month"] = str(ref.month)
        fields["url"] = ref.url
        if ref.doi:
            fields["doi"] = ref.doi
        if ref.eprint:
            fields["eprint"] = ref.eprint
            fields["archivePrefix"] = "arXiv"
        body = ",\n".join(f"    {key}={{{value}}}" for key, value in fields.items())
        entries.append(f"@{ref.entry_type}{{{ref.key},\n{body}\n}}\n")
    return "\n".join(entries)


def update_article_references(article: str, references: list[Reference]) -> str:
    if len(SECTION_RE.findall(article)) != 1:
        raise ValueError("Expected exactly one article references section")
    numbers = build_citation_map(references)

    def replace_link(match: re.Match[str]) -> str:
        key = match[2]
        if key not in numbers:
            raise ValueError(f"Unknown article citation: {key}")
        return f"{match[1]}[{numbers[key]}]{match[3]}"

    article = LINK_RE.sub(replace_link, article)
    return SECTION_RE.sub(lambda _: render_references_html(references), article)


class CitationParser(HTMLParser):
    """Collect citation IDs/labels independently of attribute order and spacing."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.citations: list[tuple[str, str]] = []
        self._key: str | None = None
        self._label = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier and identifier.startswith("ref-"):
            self.ids.append(identifier[4:])
        href = attributes.get("href") or ""
        if tag == "a" and href.startswith("#ref-"):
            self._key, self._label = href[5:], ""

    def handle_data(self, data: str) -> None:
        if self._key is not None:
            self._label += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._key is not None:
            self.citations.append((self._key, self._label.strip()))
            self._key = None


def validate_citations(article: str, paper: str, references: list[Reference]) -> None:
    numbers = build_citation_map(references)
    parser = CitationParser()
    parser.feed(article)
    if parser.ids != list(numbers):
        raise ValueError(
            "Article bibliography IDs/order differ from canonical references"
        )
    for key, label in parser.citations:
        if key not in numbers or label != f"[{numbers[key]}]":
            raise ValueError(f"Invalid article citation: {key} {label}")
    uncommented = re.sub(r"(?<!\\)%[^\n]*", "", paper)
    tex_keys = [
        key.strip()
        for match in CITE_RE.findall(uncommented)
        for key in match.split(",")
    ]
    if unknown := set(tex_keys) - set(numbers):
        raise ValueError(f"Unknown paper citations: {sorted(unknown)}")
    if Counter(tex_keys) != Counter(key for key, _ in parser.citations):
        raise ValueError("Paper/article citation occurrence counts differ")
    if unused := set(numbers) - set(tex_keys):
        raise ValueError(f"Uncited bibliography entries: {sorted(unused)}")


def generate_outputs(*, check: bool = False) -> None:
    refs = load_references()
    article = update_article_references(ARTICLE_PATH.read_text(encoding="utf-8"), refs)
    validate_citations(article, PAPER_PATH.read_text(encoding="utf-8"), refs)
    outputs = {
        GENERATED_DIR / "references.html": render_references_html(refs) + "\n",
        BIB_PATH: render_references_bibtex(refs),
        ARTICLE_PATH: article,
    }
    stale = [
        path
        for path, content in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if check and stale:
        raise ValueError(
            "Stale reference outputs; run python -m scaling_law_analysis.references: "
            + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in stale)
        )
    if not check:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(
        f"{'Checked' if check else 'Generated'} {len(refs)} references and validated paper/article citations"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail on stale outputs without writing"
    )
    generate_outputs(check=parser.parse_args().check)


if __name__ == "__main__":
    main()
