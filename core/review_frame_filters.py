"""Filtering contracts for Step 2 frame review rows."""
from __future__ import annotations

from collections.abc import Sequence

REVIEW_ISSUE_TOKENS = frozenset(
    {"redundant_drop", "motion_blur", "borderline_blur", "low_texture", "weak_match"}
)

REVIEW_THUMBNAIL_FILTER_ALL = "all"
REVIEW_THUMBNAIL_FILTER_DROPS = "drops"
REVIEW_THUMBNAIL_FILTER_WARNINGS = "warnings"
REVIEW_THUMBNAIL_FILTER_ACTIONABLE = "actionable"
REVIEW_THUMBNAIL_FILTER_KEYS = (
    REVIEW_THUMBNAIL_FILTER_ALL,
    REVIEW_THUMBNAIL_FILTER_DROPS,
    REVIEW_THUMBNAIL_FILTER_WARNINGS,
    REVIEW_THUMBNAIL_FILTER_ACTIONABLE,
)


def normalize_review_decision(row: dict[str, str]) -> str:
    return "drop" if row.get("decision", "keep").strip().lower() == "drop" else "keep"


def review_status_tokens(row: dict[str, str]) -> set[str]:
    status = row.get("status", "").strip().lower()
    if status in {"", "ok"}:
        return set()
    return {token.strip() for token in status.replace(",", "+").split("+") if token.strip()}


def is_drop_marked(row: dict[str, str]) -> bool:
    return normalize_review_decision(row) == "drop"


def has_review_issue_status(row: dict[str, str]) -> bool:
    return bool(review_status_tokens(row) & REVIEW_ISSUE_TOKENS)


def is_review_warning(row: dict[str, str]) -> bool:
    return not is_drop_marked(row) and has_review_issue_status(row)


def is_problem_navigation_row(row: dict[str, str], *, include_added: bool = False) -> bool:
    tokens = review_status_tokens(row)
    if not tokens:
        return False
    if include_added:
        return True
    return bool(tokens & REVIEW_ISSUE_TOKENS)


def thumbnail_filter_accepts(row: dict[str, str], filter_key: str) -> bool:
    key = filter_key if filter_key in REVIEW_THUMBNAIL_FILTER_KEYS else REVIEW_THUMBNAIL_FILTER_ALL
    if key == REVIEW_THUMBNAIL_FILTER_DROPS:
        return is_drop_marked(row)
    if key == REVIEW_THUMBNAIL_FILTER_WARNINGS:
        return is_review_warning(row)
    if key == REVIEW_THUMBNAIL_FILTER_ACTIONABLE:
        return is_drop_marked(row) or is_review_warning(row)
    return True


def thumbnail_filter_indices(rows: Sequence[dict[str, str]], filter_key: str) -> list[int]:
    return [idx for idx, row in enumerate(rows) if thumbnail_filter_accepts(row, filter_key)]


def thumbnail_filter_counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    return {key: len(thumbnail_filter_indices(rows, key)) for key in REVIEW_THUMBNAIL_FILTER_KEYS}
