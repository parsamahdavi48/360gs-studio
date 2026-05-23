from core.review_blur_sensitivity import (
    BLUR_REVIEW_MODE_FIELD,
    BLUR_REVIEW_MODE_LOW,
    BLUR_REVIEW_MODE_STANDARD,
    REVIEW_DECISION_OVERRIDE_FIELD,
    apply_blur_review_mode,
    detect_blur_review_mode,
)


def _pair_row(
    ratio: str,
    *,
    status: str,
    decision: str,
    selection_reason: str = "fixed_interval",
    override: str = "",
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    row = {
        "analysis_pipeline": "pair",
        "blur_score_final": "55",
        "sharpness_baseline": "100",
        "sharpness_ratio": ratio,
        "status": status,
        "decision": decision,
        "selection_reason": selection_reason,
        "review_required": "1" if status != "ok" else "0",
        "risk_flags": status if "blur" in status else "",
        REVIEW_DECISION_OVERRIDE_FIELD: override,
    }
    if extra:
        row.update(extra)
    return row


def test_apply_blur_review_mode_switches_between_standard_and_low() -> None:
    rows = [
        _pair_row("0.55", status="motion_blur", decision="drop", selection_reason="motion_blur"),
        _pair_row("0.75", status="borderline_blur", decision="keep"),
        _pair_row("0.90", status="ok", decision="keep"),
    ]

    low = apply_blur_review_mode(rows, BLUR_REVIEW_MODE_LOW)

    assert low.mode == BLUR_REVIEW_MODE_LOW
    assert low.decision_changed_rows == 1
    assert [row["status"] for row in rows] == ["borderline_blur", "ok", "ok"]
    assert [row["decision"] for row in rows] == ["keep", "keep", "keep"]
    assert {row[BLUR_REVIEW_MODE_FIELD] for row in rows} == {BLUR_REVIEW_MODE_LOW}
    assert detect_blur_review_mode(rows) == BLUR_REVIEW_MODE_LOW

    standard = apply_blur_review_mode(rows, BLUR_REVIEW_MODE_STANDARD)

    assert standard.mode == BLUR_REVIEW_MODE_STANDARD
    assert standard.decision_changed_rows == 1
    assert [row["status"] for row in rows] == ["motion_blur", "borderline_blur", "ok"]
    assert [row["decision"] for row in rows] == ["drop", "keep", "keep"]
    assert {row[BLUR_REVIEW_MODE_FIELD] for row in rows} == {BLUR_REVIEW_MODE_STANDARD}


def test_apply_blur_review_mode_preserves_manual_decision_override() -> None:
    rows = [
        _pair_row(
            "0.55",
            status="motion_blur",
            decision="keep",
            selection_reason="motion_blur",
            override="1",
        )
    ]

    apply_blur_review_mode(rows, BLUR_REVIEW_MODE_LOW)
    apply_blur_review_mode(rows, BLUR_REVIEW_MODE_STANDARD)

    assert rows[0]["status"] == "motion_blur"
    assert rows[0]["decision"] == "keep"
    assert rows[0][REVIEW_DECISION_OVERRIDE_FIELD] == "1"


def test_apply_blur_review_mode_keeps_redundant_drop_when_blur_is_cleared() -> None:
    rows = [
        _pair_row(
            "0.75",
            status="redundant_drop+borderline_blur",
            decision="drop",
            selection_reason="redundant_drop",
        )
    ]

    apply_blur_review_mode(rows, BLUR_REVIEW_MODE_LOW)

    assert rows[0]["status"] == "redundant_drop"
    assert rows[0]["decision"] == "drop"
    assert rows[0]["risk_flags"] == ""


def test_apply_blur_review_mode_uses_local_ratio_and_tracking_for_new_rows() -> None:
    rows = [
        _pair_row(
            "0.55",
            status="motion_blur",
            decision="drop",
            selection_reason="motion_blur",
            extra={
                "local_sharpness_ratio": "0.95",
                "local_sharpness_count": "4",
                "track_count": "240",
                "track_coverage": "0.85",
                "match_confidence": "0.94",
            },
        ),
        _pair_row(
            "0.55",
            status="motion_blur",
            decision="drop",
            selection_reason="motion_blur",
            extra={
                "track_count": "240",
                "track_coverage": "0.85",
                "match_confidence": "0.94",
            },
        ),
    ]

    result = apply_blur_review_mode(rows, BLUR_REVIEW_MODE_STANDARD)

    assert result.motion_blur_count == 0
    assert result.borderline_blur_count == 1
    assert [row["status"] for row in rows] == ["ok", "borderline_blur"]
    assert [row["decision"] for row in rows] == ["keep", "keep"]
