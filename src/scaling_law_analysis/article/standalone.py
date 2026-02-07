"""Build a standalone HTML file with all local images inlined as base64 data URIs.

This takes article.html (which references local PNGs via relative src paths) and
produces a single self-contained HTML file suitable for hosting on GCS or any
static file server without additional assets.

Usage:
    uv run python -m scaling_law_analysis.article.standalone
"""

import base64
import mimetypes
import re
from pathlib import Path


def inline_local_images(html: str, base_dir: Path) -> str:
    """Replace local <img src="..."> references with base64 data URIs.

    Only processes relative paths (i.e. skips URLs starting with http://, https://, or data:).

    Args:
        html: The HTML content to process.
        base_dir: Directory containing the HTML file (used to resolve relative paths).

    Returns:
        HTML with local image sources replaced by inline base64 data URIs.
    """
    img_pattern = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")')

    def replace_src(match: re.Match) -> str:
        prefix = match.group(1)
        src = match.group(2)
        suffix = match.group(3)

        # Skip URLs and already-inlined data URIs
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)

        img_path = base_dir / src
        if not img_path.exists():
            print(f"  WARNING: image not found, skipping: {img_path}")
            return match.group(0)

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Read and encode
        img_data = img_path.read_bytes()
        b64 = base64.b64encode(img_data).decode("ascii")

        print(f"  Inlined: {src} ({len(img_data) / 1024:.0f} KB)")
        return f"{prefix}data:{mime_type};base64,{b64}{suffix}"

    return img_pattern.sub(replace_src, html)


def build_standalone(
    input_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Build a standalone HTML file with inlined images.

    Args:
        input_path: Path to the source article.html.
        output_path: Path for the output file. Defaults to article_standalone.html
            in the same directory as input_path.

    Returns:
        Path to the generated standalone file.
    """
    if output_path is None:
        output_path = input_path.parent / "article_standalone.html"

    print(f"Building standalone HTML from: {input_path}")

    html = input_path.read_text(encoding="utf-8")
    html = inline_local_images(html, base_dir=input_path.parent)

    output_path.write_text(html, encoding="utf-8")

    input_size = input_path.stat().st_size / 1024
    output_size = output_path.stat().st_size / 1024
    print(f"Output: {output_path}")
    print(f"Size: {input_size:.0f} KB -> {output_size:.0f} KB")

    return output_path


if __name__ == "__main__":
    from scaling_law_analysis import config

    article_dir = config.RESULTS_DIR / "article"
    input_path = article_dir / "article.html"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Article not found at {input_path}. "
            "Generate it first before building the standalone version."
        )

    build_standalone(input_path)
