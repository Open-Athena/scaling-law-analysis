"""Build the LaTeX paper to PDF using Tectonic.

Tectonic is a self-contained TeX engine that auto-downloads packages on demand.
Install: brew install tectonic

Usage:
    uv run python -m scaling_law_analysis.paper.build
"""

import shutil
import subprocess
import sys
from pathlib import Path

from scaling_law_analysis import config

PAPER_DIR = config.RESULTS_DIR / "paper"
TEX_FILE = PAPER_DIR / "paper.tex"
PDF_FILE = PAPER_DIR / "paper.pdf"


def check_tectonic() -> str:
    """Verify tectonic is installed and return its path."""
    path = shutil.which("tectonic")
    if path is None:
        print(
            "Error: tectonic not found on PATH.\n"
            "Install with: brew install tectonic",
            file=sys.stderr,
        )
        sys.exit(1)
    return path


def build_paper() -> Path:
    """Compile paper.tex to PDF using tectonic.

    Returns:
        Path to the generated PDF.
    """
    tectonic = check_tectonic()

    if not TEX_FILE.exists():
        print(f"Error: {TEX_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Building paper: {TEX_FILE}")
    print(f"Using tectonic: {tectonic}")

    result = subprocess.run(
        [tectonic, str(TEX_FILE)],
        cwd=str(PAPER_DIR),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"Error: tectonic exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Output: {PDF_FILE} ({PDF_FILE.stat().st_size / 1024:.0f} KB)")
    return PDF_FILE


if __name__ == "__main__":
    build_paper()
