from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SOURCE_KIND_VIDEO = "video"
SOURCE_KIND_IMAGE_SEQUENCE = "image_sequence"
SOURCE_KINDS = {SOURCE_KIND_VIDEO, SOURCE_KIND_IMAGE_SEQUENCE}


@dataclass(frozen=True, slots=True)
class InputSource:
    kind: str
    path: Path


def parse_path_list(text: str) -> list[Path]:
    raw_paths = [part.strip().strip('"') for part in str(text or "").split(";")]
    return [Path(part) for part in raw_paths if part]


def source_key(source: InputSource) -> str:
    try:
        resolved = source.path.resolve(strict=False)
    except OSError:
        resolved = source.path
    return f"{source.kind}:{str(resolved).replace('\\', '/').casefold()}"


def normalize_input_sources(sources: list[InputSource]) -> list[InputSource]:
    unique: list[InputSource] = []
    seen: set[str] = set()
    for source in sources:
        if source.kind not in SOURCE_KINDS:
            continue
        key = source_key(source)
        if key in seen:
            continue
        seen.add(key)
        unique.append(InputSource(source.kind, Path(source.path)))
    return unique


def split_input_sources(sources: list[InputSource]) -> tuple[list[Path], list[Path]]:
    videos = [source.path for source in sources if source.kind == SOURCE_KIND_VIDEO]
    image_sequences = [source.path for source in sources if source.kind == SOURCE_KIND_IMAGE_SEQUENCE]
    return videos, image_sequences


def sources_from_legacy_text(video_text: str, image_sequence_text: str) -> list[InputSource]:
    sources = [InputSource(SOURCE_KIND_VIDEO, path) for path in parse_path_list(video_text)]
    sources.extend(InputSource(SOURCE_KIND_IMAGE_SEQUENCE, path) for path in parse_path_list(image_sequence_text))
    return normalize_input_sources(sources)


def replace_video_sources(sources: list[InputSource], videos: list[Path]) -> list[InputSource]:
    kept = [source for source in sources if source.kind != SOURCE_KIND_VIDEO]
    kept.extend(InputSource(SOURCE_KIND_VIDEO, video) for video in videos)
    return normalize_input_sources(kept)
