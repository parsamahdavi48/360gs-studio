from __future__ import annotations

from pathlib import Path

from core.input_sources import (
    SOURCE_KIND_IMAGE_SEQUENCE,
    SOURCE_KIND_VIDEO,
    InputSource,
    normalize_input_sources,
    replace_video_sources,
    sources_from_legacy_text,
)


def test_input_sources_parse_legacy_text_and_dedupe() -> None:
    sources = sources_from_legacy_text('"C:/take/a.mp4"; C:/take/a.mp4', "D:/stills")

    assert sources == [
        InputSource(SOURCE_KIND_VIDEO, Path("C:/take/a.mp4")),
        InputSource(SOURCE_KIND_IMAGE_SEQUENCE, Path("D:/stills")),
    ]


def test_replace_video_sources_preserves_image_sequences() -> None:
    original = normalize_input_sources(
        [
            InputSource(SOURCE_KIND_VIDEO, Path("old.mp4")),
            InputSource(SOURCE_KIND_IMAGE_SEQUENCE, Path("stills")),
        ]
    )

    replaced = replace_video_sources(original, [Path("new.mp4")])

    assert replaced == [
        InputSource(SOURCE_KIND_IMAGE_SEQUENCE, Path("stills")),
        InputSource(SOURCE_KIND_VIDEO, Path("new.mp4")),
    ]
