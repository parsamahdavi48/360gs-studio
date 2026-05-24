"""Progress parsing for Step 3 mask worker output."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MASK_PROGRESS_RE = re.compile(r"\[progress\]\s+(\d+)\s*/\s*(\d+)")
_YOLO_PROCESSED_RE = re.compile(r"^Processed:\s+")
_STITCH_TASK_RE = re.compile(r"^Processing\s+(\d+)\s+images\s+with\s+\d+\s+workers\.\.\.$")
_STITCH_TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


@dataclass
class MaskProgressParser:
    phase_total: int = 0
    phase_done: int = 0
    stitch_chunk_total: int = 0
    stitch_chunk_done: int = 0
    stitch_done_before: int = 0

    def on_line(self, line: str) -> tuple[int, int] | None:
        match = _MASK_PROGRESS_RE.search(line)
        if match:
            self.phase_done = int(match.group(1))
            self.phase_total = int(match.group(2))
            return self.phase_done, self.phase_total

        if _YOLO_PROCESSED_RE.match(line):
            self.phase_done += 1
            return self.phase_done, self.phase_total

        match = _STITCH_TASK_RE.match(line)
        if match:
            self.phase_total = int(match.group(1))
            self.phase_done = 0
            self.stitch_chunk_total = 0
            self.stitch_chunk_done = 0
            self.stitch_done_before = 0
            return 0, self.phase_total

        match = _STITCH_TQDM_RE.search(line)
        if match:
            done = int(match.group(1))
            total = int(match.group(2))
            self.stitch_chunk_done = done
            self.stitch_chunk_total = total
            overall = self.stitch_done_before + done
            return overall, self.phase_total if self.phase_total > 0 else total

        return None

    def on_phase_finished(self, phase: str, exit_code: int) -> None:
        if phase.startswith("yolo") and exit_code == 0:
            self.phase_total = 0
            self.phase_done = 0
