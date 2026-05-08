"""UI文字列 — ロケール自動判定で日本語/英語を切り替え。

モジュールロード時にシステムロケールを判定し、日本語環境なら JA、
それ以外なら EN の文字列をモジュール変数としてエクスポートする。
"""

from __future__ import annotations

import locale
import os
import textwrap

from gui.i18n_en import STRINGS as _EN
from gui.i18n_en import TIPS as _TIPS_EN
from gui.i18n_ja import STRINGS as _JA
from gui.i18n_ja import TIPS as _TIPS_JA

# ---------------------------------------------------------------------------
# 文字列テーブル
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ロケール判定 & モジュール変数エクスポート
# ---------------------------------------------------------------------------


def _detect_lang() -> str:
    """環境変数 → システムロケールの順で言語を判定。'ja' or 'en'。"""
    override = os.environ.get("STUDIO_LANG", "").strip().lower()
    if override in ("ja", "jp", "ja_jp"):
        return "ja"
    if override in ("en", "en_us", "en_gb"):
        return "en"
    for env_name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        env_locale = os.environ.get(env_name, "").strip().lower()
        if env_locale.startswith(("ja", "japanese")):
            return "ja"
        if env_locale.startswith(("en", "english")):
            return "en"
    try:
        loc = locale.getlocale()[0] or ""
    except Exception:
        loc = ""
    loc = loc.lower()
    return "ja" if loc.startswith(("ja", "japanese")) else "en"


LANG = _detect_lang()
_table = _JA if LANG == "ja" else _EN
_tips = _TIPS_JA if LANG == "ja" else _TIPS_EN
_TOOLTIP_WRAP_WIDTH = 46 if LANG == "ja" else 78
_PATH_SEPARATORS = set("/\\")
_JA_FORBIDDEN_LINE_START = set("。、，．,.!?！？:：;；)]）】〕〉》」』”’/\\")


def _extend_path_separator_break(line: str, start: int, end: int) -> int:
    while start < end < len(line):
        if line[end] in _JA_FORBIDDEN_LINE_START:
            end += 1
            continue
        if line[end - 1] in _PATH_SEPARATORS:
            end += 1
            continue
        break
    return end


def _wrap_ja_tooltip_line(line: str) -> list[str]:
    if len(line) <= _TOOLTIP_WRAP_WIDTH:
        return [line]
    wrapped: list[str] = []
    start = 0
    while start < len(line):
        end = min(len(line), start + _TOOLTIP_WRAP_WIDTH)
        end = _extend_path_separator_break(line, start, end)
        wrapped.append(line[start:end])
        start = end
    return wrapped


def _wrap_tooltip(text: str) -> str:
    lines: list[str] = []
    for line in text.split("\n"):
        if not line or len(line) <= _TOOLTIP_WRAP_WIDTH:
            lines.append(line)
            continue
        if LANG == "ja":
            lines.extend(_wrap_ja_tooltip_line(line))
            continue
        lines.extend(
            textwrap.wrap(
                line,
                width=_TOOLTIP_WRAP_WIDTH,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


def t(key: str) -> str:
    """文字列キーからローカライズ済みテキストを取得。"""
    return _table.get(key, key)


def tip(key: str) -> str:
    """ツールチップキーからローカライズ済みテキストを取得。"""
    return _wrap_tooltip(_tips.get(key, ""))


# モジュール変数として全キーを公開 (既存コードとの互換性)
def _export_module_vars() -> None:
    g = globals()
    for key, value in _table.items():
        g[key] = value


_export_module_vars()
