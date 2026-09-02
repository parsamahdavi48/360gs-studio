"""External JSON language packs with English fallback and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}")


@dataclass(frozen=True, slots=True)
class LanguagePack:
    language_id: str
    display_name: str
    strings: dict[str, str]
    schema_version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> LanguagePack:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported language pack schema: {path}")
        strings = payload.get("strings")
        if not isinstance(strings, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in strings.items()):
            raise ValueError(f"language pack strings must be a string mapping: {path}")
        return cls(str(payload["language_id"]), str(payload["display_name"]), dict(strings))

    def validate_against(self, fallback: LanguagePack) -> list[str]:
        problems: list[str] = []
        for key, fallback_text in fallback.strings.items():
            translated = self.strings.get(key)
            if translated is None:
                problems.append(f"missing key: {key}")
                continue
            expected = set(_PLACEHOLDER_RE.findall(fallback_text))
            actual = set(_PLACEHOLDER_RE.findall(translated))
            if actual != expected:
                problems.append(f"placeholder mismatch for {key}: expected {sorted(expected)}, found {sorted(actual)}")
        return problems


class Translator:
    def __init__(self, fallback: LanguagePack, selected: LanguagePack | None = None) -> None:
        self.fallback = fallback
        self.selected = selected or fallback

    def text(self, key: str, **values: object) -> str:
        template = self.selected.strings.get(key, self.fallback.strings.get(key, key))
        try:
            return template.format(**values)
        except (KeyError, ValueError):
            return self.fallback.strings.get(key, key).format(**values)


def bundled_english_pack() -> LanguagePack:
    return LanguagePack.load(Path(__file__).resolve().parent / "resources" / "i18n" / "en.json")
