"""Build an arXiv submission archive from the paper source.

Creates a self-contained .tar.gz with all figures copied locally,
relative paths rewritten, and a pre-compiled .bbl for bibliography.

Usage:
    uv run python -m scaling_law_analysis.paper.submit
"""

import datetime
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from scaling_law_analysis import config
from scaling_law_analysis.paper.build import check_tectonic

PAPER_DIR = config.RESULTS_DIR / "paper"
TEX_FILE = PAPER_DIR / "paper.tex"
SUBMISSIONS_DIR = PAPER_DIR / "submissions"
FIGURES_SUBDIR = "figures"

INCLUDEGRAPHICS_RE = re.compile(r"(\\includegraphics\[[^\]]*\])\{([^}]+)\}")
BIBSTYLE_RE = re.compile(r"\\bibliographystyle\{[^}]*\}\n?")
BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\{[^}]*\}")


def _collect_figure_paths(tex: str) -> list[str]:
    """Extract all figure paths from \\includegraphics commands."""
    return [m.group(2) for m in INCLUDEGRAPHICS_RE.finditer(tex)]


def _rewrite_tex(tex: str) -> str:
    """Rewrite figure paths to local figures/ directory and inline .bbl."""

    def _rewrite_path(m: re.Match) -> str:
        cmd = m.group(1)
        old_path = m.group(2)
        filename = Path(old_path).name
        return f"{cmd}{{{FIGURES_SUBDIR}/{filename}}}"

    tex = INCLUDEGRAPHICS_RE.sub(_rewrite_path, tex)

    # Replace \bibliographystyle{...} / \bibliography{...} with \input{paper.bbl}
    tex = BIBSTYLE_RE.sub("", tex)
    tex = BIBLIOGRAPHY_RE.sub(r"\\input{paper.bbl}", tex)

    return tex


def _build_bbl(tectonic: str) -> Path:
    """Compile paper.tex with --keep-intermediates to generate .bbl."""
    bbl_file = PAPER_DIR / "paper.bbl"
    aux_file = PAPER_DIR / "paper.aux"
    out_file = PAPER_DIR / "paper.out"

    result = subprocess.run(
        [tectonic, "--keep-intermediates", str(TEX_FILE)],
        cwd=str(PAPER_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(
            f"Error: tectonic exited with code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    if not bbl_file.exists():
        print("Error: tectonic did not produce paper.bbl", file=sys.stderr)
        sys.exit(1)

    # Clean up other intermediates (keep only .bbl)
    for f in [aux_file, out_file]:
        f.unlink(missing_ok=True)

    return bbl_file


def build_submission() -> Path:
    """Build the arXiv submission archive.

    Creates a dated submission directory at results/paper/submissions/YYYYMMDD/
    containing the archive and its extracted contents for inspection.

    Returns:
        Path to the generated .tar.gz archive.
    """
    tectonic = check_tectonic()

    if not TEX_FILE.exists():
        print(f"Error: {TEX_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    # Create dated submission directory
    date_str = datetime.date.today().strftime("%Y%m%d")
    submit_dir = SUBMISSIONS_DIR / date_str
    if submit_dir.exists():
        shutil.rmtree(submit_dir)
    submit_dir.mkdir(parents=True)
    staging_dir = submit_dir / "staging"
    staging_dir.mkdir()
    figures_dir = staging_dir / FIGURES_SUBDIR
    figures_dir.mkdir()

    # Read and rewrite .tex
    tex = TEX_FILE.read_text()
    figure_paths = _collect_figure_paths(tex)
    rewritten_tex = _rewrite_tex(tex)

    # Verify no ../ paths remain
    if "../" in rewritten_tex:
        print(
            "Error: rewritten .tex still contains '../' paths",
            file=sys.stderr,
        )
        sys.exit(1)

    (staging_dir / "paper.tex").write_text(rewritten_tex)
    print(f"Rewrote paper.tex ({len(figure_paths)} figures)")

    # Copy figures
    missing = []
    for rel_path in figure_paths:
        src = (PAPER_DIR / rel_path).resolve()
        if not src.exists():
            missing.append(rel_path)
            continue
        shutil.copy2(src, figures_dir / src.name)

    if missing:
        print(
            f"Error: {len(missing)} figures not found:\n"
            + "\n".join(f"  {p}" for p in missing),
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Copied {len(figure_paths)} figures to {FIGURES_SUBDIR}/")

    # Build .bbl
    print("Building .bbl via tectonic...")
    bbl_file = _build_bbl(tectonic)
    shutil.copy2(bbl_file, staging_dir / "paper.bbl")
    bbl_file.unlink()
    print("Generated paper.bbl")

    # Create .tar.gz
    archive_file = submit_dir / "arxiv.tar.gz"
    with tarfile.open(archive_file, "w:gz") as tar:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(staging_dir))

    # Report
    size_kb = archive_file.stat().st_size / 1024
    print(f"\nSubmission: {submit_dir}")
    print(f"Archive: {archive_file} ({size_kb:.0f} KB)")
    print("Contents:")
    with tarfile.open(archive_file, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"  {member.name}")

    return archive_file


if __name__ == "__main__":
    build_submission()
