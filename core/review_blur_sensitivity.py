"""Step 2 blur-review sensitivity switching for selected_frames.csv rows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.frame_pair_analysis import (
    PAIR_MOTION_BLUR_BASELINE_MIN,
    PAIR_MOTION_BLUR_DROP_RATIO,
    PAIR_MOTION_BLUR_REVIEW_RATIO,
)

BLUR_REVIEW_MODE_STANDARD = "standard"
BLUR_REVIEW_MODE_LOW = "low"
BLUR_REVIEW_MODE_FIELD = "blur_review_mode"
REVIEW_DECISION_OVERRIDE_FIELD = "review_decision_override"

LOW_SENSITIVITY_DROP_RATIO = 0.50
LOW_SENSITIVITY_REVIEW_RATIO = 0.70

BLUR_REVIEW_MODES = frozenset({BLUR_REVIEW_MODE_STANDARD, BLUR_REVIEW_MODE_LOW})
_BLUR_STATUS_TOKENS = frozenset({"motion_blur", "borderline_blur"})
_RISK_FLAG_TOKENS = frozenset({"motion_blur", "borderline_blur", "low_texture", "weak_match"})


@dataclass(frozen=True)
class BlurReviewThresholds:
    drop_ratio: float
    review_ratio: float


@dataclass(frozen=True)
class BlurReviewApplyResult:
    mode: str
    changed_rows: int
    decision_changed_rows: int
    motion_blur_count: int
    borderline_blur_count: int
    applicable_rows: int


def normalize_blur_review_mode(mode: str | None) -> str:
    normalized = str(mode or "").strip().lower()
    return normalized if normalized in BLUR_REVIEW_MODES else BLUR_REVIEW_MODE_STANDARD


def blur_review_thresholds(mode: str | None) -> BlurReviewThresholds:
    normalized = normalize_blur_review_mode(mode)
    if normalized == BLUR_REVIEW_MODE_LOW:
        return BlurReviewThresholds(
            drop_ratio=LOW_SENSITIVITY_DROP_RATIO,
            review_ratio=LOW_SENSITIVITY_REVIEW_RATIO,
        )
    return BlurReviewThresholds(
        drop_ratio=PAIR_MOTION_BLUR_DROP_RATIO,
        review_ratio=PAIR_MOTION_BLUR_REVIEW_RATIO,
    )


def detect_blur_review_mode(rows: Iterable[dict[str, str]]) -> str:
    modes = {
        normalize_blur_review_mode(row.get(BLUR_REVIEW_MODE_FIELD))
        for row in rows
        if str(row.get(BLUR_REVIEW_MODE_FIELD, "")).strip()
    }
    if modes == {BLUR_REVIEW_MODE_LOW}:
        return BLUR_REVIEW_MODE_LOW
    return BLUR_REVIEW_MODE_STANDARD


def row_supports_blur_review(row: dict[str, str]) -> bool:
    return _row_blur_classification(row, BLUR_REVIEW_MODE_STANDARD) is not None


def apply_blur_review_mode(rows: list[dict[str, str]], mode: str | None) -> BlurReviewApplyResult:
    """Reclassify motion-blur review status in-place for Step 2.

    This intentionally updates only the blur tokens and automatic blur decisions.
    It does not recompute frame spacing, replacement search, low texture, or tracking
    quality. Rows with ``review_decision_override=1`` keep their current decision.
    """

    normalized_mode = normalize_blur_review_mode(mode)
    changed_rows = 0
    decision_changed_rows = 0
    motion_blur_count = 0
    borderline_blur_count = 0
    applicable_rows = 0

    for row in rows:
        old_row = dict(row)
        row[BLUR_REVIEW_MODE_FIELD] = normalized_mode
        classification = _row_blur_classification(row, normalized_mode)
        if classification is None:
            if row != old_row:
                changed_rows += 1
            continue

        applicable_rows += 1
        old_decision = _normalize_decision(row.get("decision"))
        status_tokens = _split_tokens(row.get("status", ""), separator="+")
        base_status_tokens = [token for token in status_tokens if token not in _BLUR_STATUS_TOKENS]

        if classification == "motion_blur":
            base_status_tokens.append("motion_blur")
            motion_blur_count += 1
        elif classification == "borderline_blur":
            base_status_tokens.append("borderline_blur")
            borderline_blur_count += 1

        row["status"] = _join_status(base_status_tokens)
        row["risk_flags"] = _join_risk_flags(_updated_risk_flags(row.get("risk_flags", ""), classification))
        row["review_required"] = "1" if row["status"] != "ok" else "0"

        if not _is_decision_override(row):
            row["decision"] = _automatic_decision_for_row(row, base_status_tokens, classification)
            if _normalize_decision(row.get("decision")) != old_decision:
                decision_changed_rows += 1

        if row != old_row:
            changed_rows += 1

    return BlurReviewApplyResult(
        mode=normalized_mode,
        changed_rows=changed_rows,
        decision_changed_rows=decision_changed_rows,
        motion_blur_count=motion_blur_count,
        borderline_blur_count=borderline_blur_count,
        applicable_rows=applicable_rows,
    )


def _row_blur_classification(row: dict[str, str], mode: str | None) -> str | None:
    if str(row.get("analysis_pipeline", "")).strip().lower() != "pair":
        return None
    ratio = _parse_float(row.get("sharpness_ratio"))
    baseline = _parse_float(row.get("sharpness_baseline"))
    if ratio is None or baseline is None or baseline < PAIR_MOTION_BLUR_BASELINE_MIN:
        return None

    thresholds = blur_review_thresholds(mode)
    if ratio <= thresholds.drop_ratio:
        return "motion_blur"
    if ratio <= thresholds.review_ratio:
        return "borderline_blur"
    return ""


def _automatic_decision_for_row(
    row: dict[str, str],
    status_tokens: list[str],
    classification: str | None,
) -> str:
    if classification == "motion_blur":
        return "drop"
    if "redundant_drop" in status_tokens:
        return "drop"
    if str(row.get("selection_reason", "")).strip().lower() == "motion_blur":
        return "keep"
    return _normalize_decision(row.get("decision"))


def _updated_risk_flags(value: str | None, classification: str | None) -> list[str]:
    flags = [token for token in _split_tokens(value or "", separator=",") if token not in _BLUR_STATUS_TOKENS]
    if classification in _BLUR_STATUS_TOKENS:
        flags.append(str(classification))
    return [token for token in flags if token in _RISK_FLAG_TOKENS]


def _split_tokens(value: str | None, *, separator: str) -> list[str]:
    text = str(value or "").strip().lower()
    if not text or text == "ok":
        return []
    normalized = text.replace(",", separator).replace("+", separator)
    tokens: list[str] = []
    for raw in normalized.split(separator):
        token = raw.strip()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _join_status(tokens: list[str]) -> str:
    return "+".join(tokens) if tokens else "ok"


def _join_risk_flags(tokens: list[str]) -> str:
    return ",".join(tokens)


def _normalize_decision(value: str | None) -> str:
    return "drop" if str(value or "").strip().lower() == "drop" else "keep"


def _is_decision_override(row: dict[str, str]) -> bool:
    return str(row.get(REVIEW_DECISION_OVERRIDE_FIELD, "")).strip().lower() in {"1", "true", "yes"}


def _parse_float(value: str | None) -> float | None:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None
