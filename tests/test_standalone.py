"""Tests for the standalone HTML image inliner."""

import base64
from pathlib import Path

from scaling_law_analysis.article.standalone import inline_local_images


def test_inline_local_images(tmp_path: Path):
    img_data = b"\x89PNG\r\n\x1a\nfake"
    (tmp_path / "pic.png").write_bytes(img_data)
    b64 = base64.b64encode(img_data).decode("ascii")

    html = (
        '<img src="pic.png">'
        "<img src='pic.png'>"
        '<img class="hero" src="pic.png">'
        '<img src="https://example.com/remote.png">'
        '<img src="missing.png">'
    )
    result = inline_local_images(html, tmp_path)

    expected_uri = f"data:image/png;base64,{b64}"

    assert f'<img src="{expected_uri}">' in result
    assert f"<img src='{expected_uri}'>" in result
    assert f'<img class="hero" src="{expected_uri}">' in result
    assert '<img src="https://example.com/remote.png">' in result
    assert '<img src="missing.png">' in result
