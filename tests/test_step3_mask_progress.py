from __future__ import annotations

from gui.steps.step3_mask_progress import MaskProgressParser


def test_mask_progress_parser_uses_explicit_progress_counts() -> None:
    parser = MaskProgressParser()

    assert parser.on_line("[progress] 0/3") == (0, 3)
    assert parser.on_line("Processing: frame_0001.jpg") is None
    assert parser.on_line("Processed: frame_0001.jpg") == (1, 3)
    assert parser.on_line("[progress] 2/3") == (2, 3)
    assert parser.on_line("[progress] 3/3") == (3, 3)


def test_mask_progress_parser_resets_yolo_phase_after_success() -> None:
    parser = MaskProgressParser()
    assert parser.on_line("[progress] 2/3") == (2, 3)

    parser.on_phase_finished("yolo", 0)

    assert parser.phase_done == 0
    assert parser.phase_total == 0


def test_mask_progress_parser_handles_stitch_logs() -> None:
    parser = MaskProgressParser()

    assert parser.on_line("Processing 4 images with 2 workers...") == (0, 4)
    assert parser.on_line("anything | 2/4 [00:00<00:00, 10.0it/s]") == (2, 4)
    assert parser.on_line("anything | 4/4 [00:00<00:00, 10.0it/s]") == (4, 4)
