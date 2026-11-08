"""Regression checks for bibliography identity, generated output, and citation links."""

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace

import pytest
import yaml

from scaling_law_analysis.references import (
    ARTICLE_PATH,
    PAPER_PATH,
    CitationParser,
    Reference,
    generate_outputs,
    load_references,
    render_references_bibtex,
    render_references_html,
    update_article_references,
    validate_citations,
)


@pytest.fixture
def reference():
    return Reference(
        key="example",
        title="A & B: Case-Sensitive NNLS",
        authors=["Raffel, Colin"],
        year=2023,
        venue="Example",
        url="https://example.org/paper",
        verified_on="2026-11-07",
        evidence_urls=["https://example.org/paper"],
    )


def test_metadata_rejects_duplicate_keys_and_publications(tmp_path, reference):
    path = tmp_path / "references.yaml"
    path.write_text(yaml.safe_dump([asdict(reference), asdict(reference)]))
    with pytest.raises(ValueError, match="Duplicate reference keys"):
        load_references(path)
    path.write_text(
        yaml.safe_dump([asdict(reference), asdict(replace(reference, key="other"))])
    )
    with pytest.raises(ValueError, match="Duplicate publication"):
        load_references(path)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"authors": []}, "Missing authors"),
        ({"authors": ["others"]}, "must be complete"),
        ({"url": "javascript:alert(1)"}, "Invalid source URL"),
        ({"verified_on": ""}, "Missing verification"),
        ({"entry_type": "madeup"}, "Invalid entry type"),
    ],
)
def test_metadata_rejects_unreviewable_entries(tmp_path, reference, changes, message):
    path = tmp_path / "references.yaml"
    path.write_text(yaml.safe_dump([asdict(replace(reference, **changes))]))
    with pytest.raises(ValueError, match=message):
        load_references(path)


def test_html_escapes_metadata_and_bibtex_preserves_case_and_names(reference):
    ref = replace(
        reference,
        authors=["von Rütte, Dimitri", "Pióro, Maciej", "{Team Olmo}"],
        month=11,
        note="Accessed: 8 Nov 2026 & verified",
    )
    rendered = render_references_html([ref])
    assert "A &amp; B" in rendered
    assert "Dimitri von Rütte" in rendered
    assert "Accessed: 8 Nov 2026 &amp; verified" in rendered
    bib = render_references_bibtex([ref])
    assert r"title={{A \& B: Case-Sensitive NNLS}}" in bib
    assert r"von R{\"u}tte, Dimitri" in bib
    assert "{Team Olmo}" in bib
    assert "year={2023}" in bib
    assert "month={11}" in bib
    assert r"note={Accessed: 8 Nov 2026 \& verified}" in bib


def test_large_author_lists_are_explicitly_abbreviated_only_on_render(reference):
    authors = [f"Family{i}, Given{i}" for i in range(12)]
    ref = replace(reference, authors=authors)
    assert "Family9, Given9 and others" in render_references_bibtex([ref])
    assert "et al." in render_references_html([ref])
    assert ref.authors == authors


def test_refresh_repairs_numbers_by_key_and_keeps_surrounding_prose(reference):
    article = (
        '<p>Kaplan <a class="cite" href="#ref-example">[14]</a>.</p>\n'
        + render_references_html([reference])
        + "<footer>Keep me</footer>"
    )
    updated = update_article_references(article, [reference])
    assert '<a class="cite" href="#ref-example">[1]</a>' in updated
    assert updated.endswith("<footer>Keep me</footer>")
    assert update_article_references(updated, [reference]) == updated
    validate_citations(updated, r"\citep{example}", [reference])


def test_validation_catches_wrong_labels_duplicate_targets_and_cross_format_drift(
    reference,
):
    section = render_references_html([reference])
    valid = '<a href="#ref-example">[1]</a>' + section
    with pytest.raises(ValueError, match="Invalid article citation"):
        validate_citations(
            valid.replace("[1]", "[14]"), r"\citep{example}", [reference]
        )
    with pytest.raises(ValueError, match="IDs/order"):
        validate_citations(valid + section, r"\citep{example}", [reference])
    with pytest.raises(ValueError, match="Unknown paper citations"):
        validate_citations(valid, r"\citep{nonexistent}", [reference])
    with pytest.raises(ValueError, match="occurrence counts differ"):
        validate_citations(valid, r"\citep{example,example}", [reference])


def test_citation_parser_accepts_attribute_order_nested_label_and_comments(reference):
    section = render_references_html([reference])
    article = '<a title="reference" href="#ref-example"><span>[1]</span></a>' + section
    validate_citations(
        article,
        "% \\citep{not_a_real_key}\n" + r"\citep[see][p. 4]{example}",
        [reference],
    )


def test_checked_in_reference_outputs_and_standalone_are_current():
    from scaling_law_analysis.article.standalone import inline_local_images

    generate_outputs(check=True)
    refs = load_references()
    standalone = ARTICLE_PATH.with_name("article_standalone.html").read_text()
    validate_citations(standalone, PAPER_PATH.read_text(), refs)
    parser = CitationParser()
    parser.feed(standalone)
    assert len(parser.citations) > len(refs)
    assert render_references_html(refs) in standalone
    assert standalone == inline_local_images(
        ARTICLE_PATH.read_text(), ARTICLE_PATH.parent
    )


def test_audited_authorship_and_source_identity_regressions():
    refs = {ref.key: ref for ref in load_references()}
    corrected_first_authors = {
        "sovit": "Alabdulmohsin, Ibrahim M.",
        "atlas_multilingual": "Longpre, Shayne",
        "waymo_scaling": "Baniodeh, Mustafa",
        "icr_scaling": "Bordelon, Blake",
        "il_scaling": "Tuyls, Jens",
        "subgroup_scaling": "Rolf, Esther",
        "moe_memory_scaling": "Ludziejewski, Jan",
        "ehr_scaling": "Zhang, Sheng",
        "dit_scaling": "Liang, Zhengyang",
        "misfitting": "Li, Margaret",
        "dlm_scaling": "von Rütte, Dimitri",
        "reconciling_scaling": "Pearce, Tim",
        "optibert": "Dervishi, Megi",
        "biosignal_scaling": "Fortin, Thomas",
        "mupt": "Qu, Xingwei",
    }
    for key, author in corrected_first_authors.items():
        assert refs[key].authors[0] == author
    assert refs["reconciling_scaling"].authors == ["Pearce, Tim", "Song, Jinyeop"]
    assert refs["data_constrained"].authors[-1] == "Raffel, Colin"
    assert refs["biosignal_scaling"].entry_type == "mastersthesis"
    assert "epochai_open_model_compute" not in refs
    assert refs["epochai_model_database"].year == 2026
    assert refs["epochai_model_database"].month == 11
    assert refs["epochai_model_database"].note == "Accessed: 8 Nov 2026"
    assert refs["olmo_hybrid"].year == 2026
    assert "redundancy_scaling" not in refs


def test_review_inventory_covers_every_current_reference_and_citation():
    from scaling_law_analysis.config import PROJECT_ROOT
    from scaling_law_analysis.references import CITE_RE

    audit = json.loads(
        (PROJECT_ROOT / "docs/references/citation-audit.json").read_text()
    )
    refs = load_references()
    paper = PAPER_PATH.read_text()
    article_lines = ARTICLE_PATH.read_text().splitlines()
    paper_lines = paper.splitlines()
    current_sources = [
        item for item in audit["sources"] if item["status"] in {"retained", "added"}
    ]
    assert {item["key"] for item in current_sources} == {ref.key for ref in refs}
    original_sources = [item for item in audit["sources"] if item["original_entry"]]
    assert len(original_sources) == audit["baseline_reference_count"]
    keys = [key.strip() for group in CITE_RE.findall(paper) for key in group.split(",")]
    assert Counter(item["key"] for item in audit["occurrences"]) == Counter(keys)
    assert len(current_sources) == audit["current_reference_count"]
    assert len(keys) == audit["current_citation_targets"]
    for item in audit["occurrences"]:
        assert item["context"] in paper
        assert item["key"] in paper_lines[item["paper_line"] - 1]
        assert f'#ref-{item["key"]}' in article_lines[item["article_line"] - 1]
        assert item["source_urls"]


def test_citation_revision_preserves_recorded_non_citation_sources():
    from scaling_law_analysis.config import PROJECT_ROOT
    from scaling_law_analysis.references import CITE_RE

    audit = json.loads(
        (PROJECT_ROOT / "docs/references/citation-audit.json").read_text()
    )
    contents = {
        "results/paper/paper.tex": PAPER_PATH.read_text(),
        "results/article/article.html": ARTICLE_PATH.read_text(),
    }
    for change in audit["preservation"].get("approved_non_citation_changes", []):
        assert change["path"] in contents
        for replacement in change["replacements"]:
            assert contents[change["path"]].count(replacement["after"]) == 1
            contents[change["path"]] = contents[change["path"]].replace(
                replacement["after"], replacement["before"], 1
            )
    paper = re.sub(
        r"\\usepackage(?:\[[^]]*\])?\{natbib\}",
        r"\\usepackage{natbib}",
        contents["results/paper/paper.tex"],
    )
    paper = re.sub(r"~?" + CITE_RE.pattern, "", paper)
    article = re.sub(
        r'<section\s+id="references">.*?</section>',
        "",
        contents["results/article/article.html"],
        flags=re.S,
    )
    anchor = r'<a\b[^>]*\bhref=["\x27]#ref-[a-z0-9_]+["\x27][^>]*>.*?</a>'
    article = re.sub(
        r"<sup>\s*(?:" + anchor + r"\s*,?\s*)+</sup>", "", article, flags=re.S
    )
    article = re.sub(anchor, "", article, flags=re.S)
    assert (
        hashlib.sha256(paper.encode()).hexdigest()
        == audit["preservation"]["paper_non_citation_sha256"]
    )
    assert (
        hashlib.sha256(article.encode()).hexdigest()
        == audit["preservation"]["article_non_citation_sha256"]
    )


def test_arxiv_rewrite_keeps_explicit_citation_style():
    from scaling_law_analysis.paper.submit import _rewrite_tex

    source = PAPER_PATH.read_text()
    packaged = _rewrite_tex(source)
    assert r"\usepackage[square,comma]{natbib}" in source
    assert r"\usepackage[square,comma]{natbib}" in packaged
    assert r"\input{paper.bbl}" in packaged
    assert r"\bibliographystyle" not in packaged
