from __future__ import annotations

import json
from pathlib import Path

from core.apriltag_printable import available_pages, create_printable_target


def test_printable_target_supports_a4_a3_and_letter(tmp_path: Path) -> None:
    assert available_pages() == ("A4", "A3", "Letter")

    a4 = create_printable_target(tmp_path / "a4", family="tag36h11", tag_id=7, tag_size_m=0.160, page="A4")
    a3 = create_printable_target(tmp_path / "a3", family="tag36h11", tag_id=7, tag_size_m=0.160, page="A3")
    letter = create_printable_target(
        tmp_path / "letter",
        family="tag36h11",
        tag_id=7,
        tag_size_m=0.160,
        page="Letter",
    )

    assert a4.page_pdf.is_file()
    assert a3.page_pdf.is_file()
    assert letter.page_pdf.is_file()
    assert a4.page_pixels == (2480, 3508)
    assert a3.page_pixels == (3508, 4961)
    assert letter.page_pixels == (2550, 3300)

    spec = json.loads(a4.spec_json.read_text(encoding="utf-8"))
    assert spec["page"] == "A4"
    assert spec["tag_id"] == 7
    assert spec["tag_size_m"] == 0.160
