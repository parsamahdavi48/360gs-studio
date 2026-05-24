"""Step 1 video probing, source registry, and estimate helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from core.scene_project import source_video_record, upsert_source_videos
from gui import i18n


class Step1VideoInfoMixin:
    # -- 動画情報 --

    @staticmethod
    def _parse_fraction(value: str) -> float:
        if not value:
            return 0.0
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(value)

    @staticmethod
    def _format_duration(sec: float) -> str:
        whole = int(max(0, sec))
        h, m, s = whole // 3600, (whole % 3600) // 60, whole % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _format_number(value: int) -> str:
        return f"{int(value):,}"

    @staticmethod
    def _video_key(video: Path) -> str:
        return str(video)

    @staticmethod
    def _estimated_total_frames(info: dict) -> int:
        total = int(info.get("total_frames", 0))
        if total > 0:
            return total
        dur = float(info.get("duration_sec", 0))
        fps = float(info.get("fps", 0))
        if dur > 0 and fps > 0:
            return max(1, int(round(dur * fps)))
        return 0

    def _fixed_estimate_count(self, info: dict) -> int:
        total = self._estimated_total_frames(info)
        fps = float(info.get("fps", 0))
        if total <= 0 or fps <= 0:
            return 0
        iv = self.interval_edit.value()
        if iv <= 0:
            return 0
        step = max(1, int(round(iv * fps)))
        indices = list(range(0, max(total, 1), step))
        last_index = max(total - 1, 0)
        if indices[-1] != last_index:
            indices.append(last_index)
        return len(indices)

    def _prune_video_info_cache(self, videos: list[Path]) -> None:
        keys = {self._video_key(video) for video in videos}
        self.video_infos = {key: value for key, value in self.video_infos.items() if key in keys}
        self.video_info_failures = {key: value for key, value in self.video_info_failures.items() if key in keys}

    def _load_video_info(self, show_error: bool = True) -> bool:
        videos = self._selected_video_paths()
        self._prune_video_info_cache(videos)
        if len(videos) > 1:
            return self._load_multi_video_info(videos, show_error=show_error)
        try:
            self.video_info = self._probe_video_info()
            if videos:
                key = self._video_key(videos[0])
                self.video_infos[key] = self.video_info
                self.video_info_failures.pop(key, None)
            self._update_video_info_label()
            self._mark_estimate_stale()
            self._update_ready_status()
            return True
        except Exception as e:
            self.video_info = None
            for video in videos[:1]:
                self.video_info_failures[self._video_key(video)] = str(e)
            self._update_video_info_label()
            self.instant_estimate_text = "-"
            self._refresh_estimate_label()
            self._update_ready_status()
            if show_error:
                QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))
            return False

    def _load_multi_video_info(self, videos: list[Path], show_error: bool = True) -> bool:
        self.video_info = None
        self._prune_video_info_cache(videos)
        failures: list[str] = []
        for video in videos:
            key = self._video_key(video)
            try:
                info = self._probe_video_info_for_path(video)
            except Exception as e:
                self.video_infos.pop(key, None)
                self.video_info_failures[key] = str(e)
                failures.append(f"{video.name}: {e}")
                continue
            self.video_infos[key] = info
            self.video_info_failures.pop(key, None)

        self._update_video_info_label()
        self._mark_estimate_stale()
        self._update_ready_status()
        if failures and show_error:
            QMessageBox.warning(self, i18n.INVALID_INPUT, "\n".join(failures))
        return bool(self.video_infos)

    def _reload_video_info_if_selected(self) -> None:
        videos = self._selected_video_paths()
        if videos:
            self._load_video_info(show_error=False)

    def _probe_video_info(self) -> dict:
        videos = self._selected_video_paths()
        if not videos:
            raise ValueError("入力動画が指定されていません")
        return self._probe_video_info_for_path(videos[0])

    def _probe_video_info_for_path(self, video_path: Path) -> dict:
        video = str(video_path)
        ffprobe = self.ffprobe_browse.text() or "ffprobe"
        if not video_path.exists():
            raise ValueError(f"入力動画が見つかりません: {video}")

        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            video,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffprobe 失敗")

        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError("動画ストリームが見つかりません")

        s = streams[0]
        fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
        w, h = int(s.get("width", 0)), int(s.get("height", 0))
        fps = self._parse_fraction(s.get("avg_frame_rate", "0"))
        if fps <= 0:
            fps = self._parse_fraction(s.get("r_frame_rate", "0"))
        dur = float(s.get("duration") or fmt.get("duration") or 0.0)
        nb = int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else 0

        if fps <= 0 and dur > 0 and nb > 0:
            fps = nb / dur
        if fps <= 0:
            raise RuntimeError("FPSを取得できません")
        if dur <= 0 and nb > 0:
            dur = nb / fps
        if nb <= 0 and dur > 0:
            nb = max(1, int(round(dur * fps)))

        return {
            "width": w,
            "height": h,
            "fps": fps,
            "duration_sec": dur,
            "total_frames": nb,
            "tags": s.get("tags") if isinstance(s.get("tags"), dict) else {},
            "format_tags": fmt.get("tags") if isinstance(fmt.get("tags"), dict) else {},
            "side_data_list": s.get("side_data_list") if isinstance(s.get("side_data_list"), list) else [],
        }

    def _save_source_video_registry(self) -> None:
        if not self.scene_dir:
            return
        records: list[dict] = []
        for video in self._selected_video_paths():
            if not video.is_file():
                continue
            info = self.video_infos.get(self._video_key(video))
            if info is None and len(self._selected_video_paths()) == 1:
                info = self.video_info
            if not isinstance(info, dict):
                continue
            try:
                records.append(source_video_record(video, info))
            except OSError:
                continue
        if records:
            upsert_source_videos(Path(self.scene_dir), records)

    def _update_video_info_label(self) -> None:
        image_dirs = self._selected_image_sequence_dirs()
        videos = self._selected_video_paths()
        if image_dirs and not videos:
            count = sum(len(self._image_sequence_files(folder)) for folder in image_dirs)
            self.video_info_label.setText(
                i18n.t("IMAGE_SEQUENCE_INFO_FORMAT").format(
                    folder="\n".join(str(folder) for folder in image_dirs),
                    count=self._format_number(count),
                )
            )
            return
        self._refresh_video_queue_list()
        if self._is_multi_video_input():
            videos = self._selected_video_paths()
            queued, skipped = self._queued_selected_videos()
            info_rows = [
                (video, self.video_infos[self._video_key(video)])
                for video in videos
                if self._video_key(video) in self.video_infos
            ]
            if info_rows:
                failed = len([video for video in videos if self._video_key(video) in self.video_info_failures])
                lines = [
                    i18n.t("VIDEO_INFO_MULTI_HEADER_FORMAT").format(
                        total=len(videos),
                        queued=len(queued),
                        skipped=skipped,
                        probed=len(info_rows),
                    )
                ]
                for video, info in info_rows:
                    lines.append(
                        i18n.t("VIDEO_INFO_MULTI_ITEM_FORMAT").format(
                            name=video.name,
                            status=self._video_queue_status_text(video),
                            projection=self._video_projection_text(info),
                            width=info["width"],
                            height=info["height"],
                            fps=info["fps"],
                            duration=self._format_duration(float(info.get("duration_sec", 0))),
                            frames=self._format_number(self._estimated_total_frames(info)),
                        )
                    )
                if failed:
                    lines[0] += i18n.t("VIDEO_INFO_FAILED_SUFFIX").format(failed=failed)
                self.video_info_label.setText("\n".join(lines))
                return
            self.video_info_label.setText(
                i18n.t("VIDEO_QUEUE_LABEL_FORMAT").format(
                    total=len(videos),
                    queued=len(queued),
                    skipped=skipped,
                )
            )
            return
        if not self.video_info:
            self.video_info_label.setText(i18n.t("VIDEO_LABEL_DEFAULT"))
            return
        i = self.video_info
        d = self._format_duration(float(i["duration_sec"]))
        videos = self._selected_video_paths()
        status = self._video_queue_status_text(videos[0]) if videos else i18n.t("VIDEO_QUEUE_STATUS_NEW")
        self.video_info_label.setText(
            i18n.t("VIDEO_INFO_SINGLE_FORMAT").format(
                status=status,
                projection=self._video_projection_text(i),
                width=i["width"],
                height=i["height"],
                fps=i["fps"],
                duration=d,
                frames=self._format_number(self._estimated_total_frames(i)),
            )
        )

    # -- フレーム数推定 --

    def _refresh_estimate_label(self) -> None:
        self.estimate_label.setText(f"{i18n.INSTANT_ESTIMATE}: {self.instant_estimate_text}")

    def _mark_estimate_stale(self, *_args) -> None:
        self.last_estimate_summary = None
        self._update_instant_estimate()
        self._update_ready_status()

    def _update_instant_estimate(self) -> None:
        image_dirs = self._selected_image_sequence_dirs()
        videos = self._selected_video_paths()
        if image_dirs and not videos:
            count = sum(len(self._image_sequence_files(folder)) for folder in image_dirs)
            self.instant_estimate_text = (
                i18n.t("IMAGE_SEQUENCE_ESTIMATE_FORMAT").format(count=self._format_number(count)) if count else "-"
            )
            self._refresh_estimate_label()
            return
        if image_dirs and videos:
            queued, skipped = self._queued_selected_videos()
            image_count = sum(len(self._image_sequence_files(folder)) for folder in image_dirs)
            self.instant_estimate_text = i18n.t("SOURCE_QUEUE_ESTIMATE_FORMAT").format(
                videos=len(queued),
                skipped=skipped,
                image_folders=len(image_dirs),
                images=self._format_number(image_count),
            )
            self._refresh_estimate_label()
            return
        if self._is_multi_video_input():
            queued, skipped = self._queued_selected_videos()
            info_rows = [
                (video, self.video_infos[self._video_key(video)])
                for video in queued
                if self._video_key(video) in self.video_infos
            ]
            if info_rows:
                counts = [(video, self._fixed_estimate_count(info)) for video, info in info_rows]
                total_estimated = sum(count for _video, count in counts)
                missing = max(0, len(queued) - len(info_rows))
                lines = [
                    i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_HEADER_FORMAT").format(
                        interval=f"{self.interval_edit.value():g}",
                    )
                ]
                for video, count in counts:
                    lines.append(
                        i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_ITEM_FORMAT").format(
                            name=video.name,
                            count=self._format_number(count),
                        )
                    )
                total_line = i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_TOTAL_FORMAT").format(
                    count=self._format_number(total_estimated),
                    videos=len(info_rows),
                )
                if self.quick_extract_cb.isChecked():
                    total_line += f" ({i18n.t('QUICK_EXTRACT_ESTIMATE')})"
                elif self.smart_fixed_cb.isChecked():
                    total_line += f" ({i18n.t('FIXED_SMART_ESTIMATE')})"
                if missing:
                    total_line += i18n.t("ESTIMATE_MISSING_INFO_SUFFIX").format(missing=missing)
                lines.append(total_line)
                self.instant_estimate_text = "\n".join(lines)
                self._refresh_estimate_label()
                return
            self.instant_estimate_text = i18n.t("QUEUE_ESTIMATE_FORMAT").format(
                queued=len(queued),
                skipped=skipped,
            )
            self._refresh_estimate_label()
            return
        if not self.video_info:
            self.instant_estimate_text = "-"
            self._refresh_estimate_label()
            return
        dur = float(self.video_info.get("duration_sec", 0))
        fps = float(self.video_info.get("fps", 0))
        total = int(self.video_info.get("total_frames", 0))

        try:
            iv = self.interval_edit.value()
            if iv <= 0:
                raise ValueError
            if total <= 0 and dur > 0 and fps > 0:
                total = max(1, int(round(dur * fps)))
            info = dict(self.video_info)
            info["total_frames"] = total
            estimated = self._fixed_estimate_count(info)
            text = i18n.t("FIXED_INTERVAL_ESTIMATE_FORMAT").format(
                interval=f"{iv:g}",
                count=self._format_number(estimated),
            )
            if self.quick_extract_cb.isChecked():
                text += f" ({i18n.t('QUICK_EXTRACT_ESTIMATE')})"
            elif self.smart_fixed_cb.isChecked():
                text += f" ({i18n.t('FIXED_SMART_ESTIMATE')})"
            self.instant_estimate_text = text
        except Exception:
            self.instant_estimate_text = "-"
        self._refresh_estimate_label()

    def _apply_summary(self, summary: dict) -> None:
        video = summary.get("video", {})
        if video:
            self.video_info = {
                "width": int(video.get("width", 0)),
                "height": int(video.get("height", 0)),
                "fps": float(video.get("fps", 0.0)),
                "duration_sec": float(video.get("duration_sec", 0.0)),
                "total_frames": int(video.get("total_frames", 0)),
            }
            self._update_video_info_label()

        result = summary.get("result", {})
        selected = int(result.get("selected_count", 0))
        total_f = int(video.get("total_frames", 0))
        ratio = (selected / total_f * 100.0) if total_f > 0 else 0.0
        parts = [f"{selected} {i18n.t('FRAMES_UNIT')} ({ratio:.1f}%)"]
        if result.get("novelty_added_count"):
            parts.append(f"+{int(result['novelty_added_count'])}")
        if result.get("dropped_count"):
            parts.append(f"-{int(result['dropped_count'])}")
        self.instant_estimate_text = " ".join(parts)
        self._refresh_estimate_label()
