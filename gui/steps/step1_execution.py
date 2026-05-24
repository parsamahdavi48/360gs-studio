"""Step 1 extraction readiness, job construction, and progress parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.app_job import frame_app_job
from core.extract_sessions import sanitize_filename_prefix
from core.frame_job_spec import extract_video_job, import_image_sequence_job
from core.input_sources import SOURCE_KIND_IMAGE_SEQUENCE
from gui import i18n

_DEFAULT_CAPTURE_PROFILE = "walk_standard"
_SOURCE_KIND_IMAGE_SEQUENCE = SOURCE_KIND_IMAGE_SEQUENCE


class Step1ExecutionMixin:
    def _readiness(self) -> tuple[bool, str]:
        sources = self._selected_input_sources()
        if not sources:
            return False, i18n.t("EXTRACT_READY_NO_INPUT_SOURCE")
        if not self.scene_dir:
            return False, i18n.t("EXTRACT_READY_NO_SCENE")
        image_dirs = self._selected_image_sequence_dirs()
        for folder in image_dirs:
            if not folder.is_dir():
                return False, i18n.t("EXTRACT_READY_IMAGE_SEQUENCE_NOT_FOUND")
            if not self._image_sequence_files(folder):
                return False, i18n.t("EXTRACT_READY_IMAGE_SEQUENCE_EMPTY")
        videos = self._selected_video_paths()
        if not videos:
            count = sum(len(self._image_sequence_files(folder)) for folder in image_dirs)
            return True, i18n.t("EXTRACT_READY_IMAGE_SEQUENCE_OK").format(n=count)

        if len(videos) > 1:
            missing = [video for video in videos if not video.is_file()]
            if missing:
                return False, i18n.t("EXTRACT_READY_VIDEO_NOT_FOUND")
            if any(self._video_key(video) in self.video_info_failures for video in videos):
                return False, i18n.t("EXTRACT_READY_NO_VIDEO_INFO")
            if not self.quick_extract_cb.isChecked() and not self._analysis_width_valid():
                return False, i18n.t("EXTRACT_READY_BAD_ANALYSIS_WIDTH")
            queued, skipped = self._queued_selected_videos()
            mode = self._extract_output_mode()
            if mode == "append" and not queued and not image_dirs:
                return False, i18n.t("EXTRACT_READY_QUEUE_ALL_DUPLICATE").format(n=len(videos))
            if image_dirs:
                image_count = sum(len(self._image_sequence_files(folder)) for folder in image_dirs)
                return True, i18n.t("EXTRACT_READY_SOURCE_QUEUE_OK").format(
                    videos=len(queued) if mode == "append" else len(videos),
                    skipped=skipped if mode == "append" else 0,
                    image_folders=len(image_dirs),
                    images=self._format_number(image_count),
                )
            if mode == "append" and skipped:
                return True, i18n.t("EXTRACT_READY_QUEUE_PARTIAL").format(n=len(queued), skipped=skipped)
            if mode == "replace-video":
                replace_count = sum(1 for video in videos if self._matching_video_sessions_for_path(video))
                return True, i18n.t("EXTRACT_READY_QUEUE_REPLACE").format(n=len(videos), replace=replace_count)
            return True, i18n.t("EXTRACT_READY_QUEUE_OK").format(n=len(videos))

        if not videos[0].is_file():
            return False, i18n.t("EXTRACT_READY_VIDEO_NOT_FOUND")
        if self._video_key(videos[0]) in self.video_info_failures:
            return False, i18n.t("EXTRACT_READY_NO_VIDEO_INFO")
        if not self.quick_extract_cb.isChecked() and not self._analysis_width_valid():
            return False, i18n.t("EXTRACT_READY_BAD_ANALYSIS_WIDTH")
        if len(sources) == 1 and not self.video_info:
            return False, i18n.t("EXTRACT_READY_NO_VIDEO_INFO")
        matching_sessions = self._matching_video_sessions()
        output_mode = self._extract_output_mode()
        if image_dirs:
            queued, skipped = self._queued_selected_videos()
            image_count = sum(len(self._image_sequence_files(folder)) for folder in image_dirs)
            return True, i18n.t("EXTRACT_READY_SOURCE_QUEUE_OK").format(
                videos=len(queued) if output_mode == "append" else len(videos),
                skipped=skipped if output_mode == "append" else 0,
                image_folders=len(image_dirs),
                images=self._format_number(image_count),
            )
        if matching_sessions and output_mode == "append":
            return False, i18n.t("EXTRACT_READY_DUPLICATE_VIDEO").format(n=len(matching_sessions))
        if matching_sessions and output_mode == "replace-video":
            return True, i18n.t("EXTRACT_READY_DUPLICATE_REPLACE").format(n=len(matching_sessions))
        return True, i18n.t("EXTRACT_READY_OK")

    def _update_ready_status(self) -> None:
        ready, reason = self._readiness()
        self.ready_status_label.setText(reason)
        if ready:
            self.ready_status_label.setStyleSheet(
                "padding: 8px 10px; border-radius: 4px; color: #dcfce7; background-color: #14532d;"
            )
        else:
            self.ready_status_label.setStyleSheet(
                "padding: 8px 10px; border-radius: 4px; color: #fef3c7; background-color: #713f12;"
            )
        self.primary_action_state_changed.emit()

    def build_commands(self) -> list[tuple[str, object]]:
        sources = self._selected_input_sources()
        videos = self._selected_video_paths()
        missing = [video for video in videos if not video.is_file()]
        if missing:
            preview = ", ".join(str(video) for video in missing[:3])
            raise ValueError(f"{i18n.t('EXTRACT_READY_VIDEO_NOT_FOUND')}\n{preview}")

        commands: list[tuple[str, object]] = []
        runnable_videos, _skipped = self._queued_selected_videos()
        runnable_video_keys = {self._video_key(video) for video in runnable_videos}
        used_prefixes: set[str] = set()
        for source in sources:
            if source.kind == _SOURCE_KIND_IMAGE_SEQUENCE:
                phase = "image_sequence_import"
                if len(sources) > 1:
                    phase = f"image_sequence_import: {source.path.name or source.path}"
                commands.append((phase, self._build_image_sequence_import_cmd(source.path)))
                continue
            if self._extract_output_mode() == "append" and self._video_key(source.path) not in runnable_video_keys:
                continue
            phase = "extract" if len(sources) == 1 else f"extract: {source.path.name}"
            commands.append((phase, self._build_extract_cmd_for_video(source.path, used_prefixes)))

        if not commands:
            raise ValueError(i18n.t("EXTRACT_READY_QUEUE_ALL_DUPLICATE").format(n=len(self._selected_video_paths())))
        return commands

    def _build_image_sequence_import_cmd(self, source: Path | None = None) -> object:
        source = source or self._image_sequence_dir()
        if source is None or not source.is_dir():
            raise ValueError(i18n.t("EXTRACT_READY_IMAGE_SEQUENCE_NOT_FOUND"))
        if not self.scene_dir:
            raise ValueError(i18n.t("EXTRACT_READY_NO_SCENE"))
        prefix = sanitize_filename_prefix(self.prefix_edit.text())
        return frame_app_job(
            import_image_sequence_job(
                source_dir=source,
                scene_dir=self.scene_dir,
                prefix=prefix,
                recursive=False,
            )
        )

    def _build_extract_cmd(self) -> object:
        videos = self._selected_video_paths()
        if not videos:
            raise ValueError("入力動画が指定されていません")
        video = videos[0]
        if not video.is_file():
            raise ValueError(f"入力動画が見つかりません: {video}")
        if not self.scene_dir:
            raise ValueError("シーンフォルダが指定されていません")

        return self._build_extract_cmd_for_video(video, set())

    def _build_extract_cmd_for_video(self, video_path: Path, used_prefixes: set[str]) -> object:
        if not video_path.is_file():
            raise ValueError(f"入力動画が見つかりません: {video_path}")
        if not self.scene_dir:
            raise ValueError("シーンフォルダが指定されていません")

        output_mode = self._extract_output_mode()
        prefix = self._prefix_for_video(video_path, used_prefixes)
        quick_extract = self.quick_extract_cb.isChecked()
        analysis_width = 0 if quick_extract else int(self.analysis_width_edit.text().strip() or "0")

        return frame_app_job(
            extract_video_job(
                input_video=video_path,
                scene_dir=self.scene_dir,
                image_ext=self.image_ext_combo.currentText(),
                jpg_quality=int(self.jpg_quality_edit.value()),
                ffmpeg=self.ffmpeg_browse.text() or "ffmpeg",
                ffprobe=self.ffprobe_browse.text() or "ffprobe",
                output_mode=output_mode,
                filename_prefix=prefix,
                interval_sec=float(self.interval_edit.value()),
                quick_extract=quick_extract,
                pair_motion_profile=str(self.pair_motion_profile_combo.currentData() or _DEFAULT_CAPTURE_PROFILE),
                analysis_width=analysis_width,
                fixed_smart=bool((not quick_extract) and self.smart_fixed_cb.isChecked()),
                min_gap_sec=float(self.min_gap_edit.value()),
                max_gap_sec=float(self.max_gap_edit.value()),
            )
        )

    def phase_display_name(self, phase: str) -> str:
        if phase == "image_sequence_import":
            return i18n.t("EXTRACT_PHASE_IMAGE_SEQUENCE")
        if phase.startswith("image_sequence_import: "):
            return i18n.t("EXTRACT_PHASE_IMAGE_SEQUENCE_FOLDER").format(folder=phase.split(": ", 1)[1])
        if phase == "extract":
            return i18n.t("EXTRACT_PHASE")
        if phase.startswith("extract: "):
            return i18n.t("EXTRACT_PHASE_VIDEO").format(video=phase.split(": ", 1)[1])
        return phase

    def phase_status_text(self, phase: str, queue_index: int, queue_total: int) -> str:
        label = self.phase_display_name(phase)
        if queue_total > 1:
            return i18n.t("EXTRACT_PHASE_QUEUE_STATUS").format(
                status=i18n.STATUS_RUNNING,
                current=queue_index,
                total=queue_total,
                phase=label,
            )
        return f"{i18n.STATUS_RUNNING}: {label}"

    def on_line(self, line: str) -> tuple[int, int] | None:
        progress_prefix = "[progress] "
        if line.startswith(progress_prefix):
            text = line[len(progress_prefix) :]
            match = re.search(r"(\d+)/(\d+)", text)
            if match:
                return int(match.group(1)), int(match.group(2))

        if line.startswith("SUMMARY_JSON:"):
            payload = line[len("SUMMARY_JSON:") :]
            try:
                summary = json.loads(payload)
                self.last_estimate_summary = summary
                self._apply_summary(summary)
            except Exception:
                pass
        return None

    def on_queue_finished(self, success: bool) -> None:
        if success and self._selected_video_paths():
            self._save_source_video_registry()
            self._refresh_finished_run_state(revalidate_video_info=False)
        elif success:
            self._refresh_finished_run_state(revalidate_video_info=False)
        else:
            self._refresh_finished_run_state(revalidate_video_info=True)

    def _refresh_finished_run_state(self, *, revalidate_video_info: bool) -> None:
        videos = self._selected_video_paths()
        if self._prune_missing_selected_videos():
            return
        if revalidate_video_info and videos:
            self._load_video_info(show_error=False)
            return
        self._update_video_info_label()
        self._update_ready_status()
