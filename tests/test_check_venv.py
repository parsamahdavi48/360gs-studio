from scripts.check_venv import split_pip_check_errors


def test_split_pip_check_errors_ignores_optional_sam3_numpy_conflict() -> None:
    errors, ignored = split_pip_check_errors(
        "sam3 0.0.1 has requirement numpy<2, but you have numpy 2.4.4.\n"
    )

    assert errors == []
    assert ignored == ["sam3 0.0.1 has requirement numpy<2, but you have numpy 2.4.4."]


def test_split_pip_check_errors_keeps_unrelated_errors() -> None:
    errors, ignored = split_pip_check_errors(
        "\n".join(
            [
                "sam3 0.0.1 has requirement numpy<2, but you have numpy 2.4.4.",
                "example 1.0 has requirement pillow<10, but you have pillow 13.0.",
            ]
        )
    )

    assert ignored == ["sam3 0.0.1 has requirement numpy<2, but you have numpy 2.4.4."]
    assert errors == ["example 1.0 has requirement pillow<10, but you have pillow 13.0."]
