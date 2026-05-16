from core.review_frame_filters import (
    REVIEW_THUMBNAIL_FILTER_ACTIONABLE,
    REVIEW_THUMBNAIL_FILTER_ALL,
    REVIEW_THUMBNAIL_FILTER_DROPS,
    REVIEW_THUMBNAIL_FILTER_WARNINGS,
    is_problem_navigation_row,
    is_review_warning,
    thumbnail_filter_counts,
    thumbnail_filter_indices,
)


def test_thumbnail_filters_split_drops_from_review_warnings() -> None:
    rows = [
        {"decision": "drop", "status": "ok"},
        {"decision": "keep", "status": "borderline_blur"},
        {"decision": "drop", "status": "motion_blur"},
        {"decision": "keep", "status": "novelty_added"},
        {"decision": "keep", "status": "ok"},
    ]

    assert thumbnail_filter_indices(rows, REVIEW_THUMBNAIL_FILTER_ALL) == [0, 1, 2, 3, 4]
    assert thumbnail_filter_indices(rows, REVIEW_THUMBNAIL_FILTER_DROPS) == [0, 2]
    assert thumbnail_filter_indices(rows, REVIEW_THUMBNAIL_FILTER_WARNINGS) == [1]
    assert thumbnail_filter_indices(rows, REVIEW_THUMBNAIL_FILTER_ACTIONABLE) == [0, 1, 2]

    counts = thumbnail_filter_counts(rows)
    assert counts[REVIEW_THUMBNAIL_FILTER_ALL] == 5
    assert counts[REVIEW_THUMBNAIL_FILTER_DROPS] == 2
    assert counts[REVIEW_THUMBNAIL_FILTER_WARNINGS] == 1
    assert counts[REVIEW_THUMBNAIL_FILTER_ACTIONABLE] == 3


def test_problem_navigation_keeps_existing_issue_contract() -> None:
    manual_drop = {"decision": "drop", "status": "ok"}
    planned_drop = {"decision": "drop", "status": "redundant_drop"}
    added = {"decision": "keep", "status": "novelty_added"}
    warning = {"decision": "keep", "status": "weak_match"}

    assert not is_review_warning(manual_drop)
    assert is_review_warning(warning)
    assert not is_problem_navigation_row(manual_drop)
    assert is_problem_navigation_row(planned_drop)
    assert not is_problem_navigation_row(added)
    assert is_problem_navigation_row(added, include_added=True)
