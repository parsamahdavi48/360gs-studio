"""Step 1 input-source queue and scene autoload helpers."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem

from core.extract_sessions import load_manifest, matching_video_sessions, sanitize_filename_prefix
from core.input_sources import (
    SOURCE_KIND_IMAGE_SEQUENCE,
    SOURCE_KIND_VIDEO,
    InputSource,
    normalize_input_sources,
    parse_path_list,
    replace_video_sources,
    source_key,
    sources_from_legacy_text,
)
from core.scene_import_contracts import IMAGE_EXTS as _IMAGE_SEQUENCE_EXTS
from core.scene_layout import APP_DIR_NAME, scene_images_dir, source_videos_path
from core.scene_project import infer_video_projection, load_json, remove_source_videos
from gui import i18n
from gui.common import dialogs
from gui.common.icons import image_folder_source_icon, video_source_icon

_SOURCE_KIND_VIDEO = SOURCE_KIND_VIDEO
_SOURCE_KIND_IMAGE_SEQUENCE = SOURCE_KIND_IMAGE_SEQUENCE
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
_VIDEO_SCAN_EXCLUDED_DIRS = {
    APP_DIR_NAME.casefold(),
    ".git",
    ".venv",
    "__pycache__",
    "images",
    "masks",
    "output",
    "outputs",
}


class Step1InputSourcesMixin:
    @staticmethod
    def _source_key(source: InputSource) -> str:
        return source_key(source)

    def _video_paths_from_text(self) -> list[Path]:
        return parse_path_list(self.video_browse.text())

    def _image_sequence_dirs_from_text(self) -> list[Path]:
        return parse_path_list(self.image_sequence_browse.text())

    def _selected_input_sources(self) -> list[InputSource]:
        if self._input_sources:
            return list(self._input_sources)
        return sources_from_legacy_text(self.video_browse.text(), self.image_sequence_browse.text())

    def _selected_video_paths(self) -> list[Path]:
        return [source.path for source in self._selected_input_sources() if source.kind == _SOURCE_KIND_VIDEO]

    def _selected_image_sequence_dirs(self) -> list[Path]:
        return [
            source.path for source in self._selected_input_sources() if source.kind == _SOURCE_KIND_IMAGE_SEQUENCE
        ]

    def _set_input_sources(self, sources: list[InputSource]) -> None:
        unique = normalize_input_sources(sources)

        self._input_sources = unique
        self._syncing_input_source_widgets = True
        try:
            videos = [source.path for source in unique if source.kind == _SOURCE_KIND_VIDEO]
            folders = [source.path for source in unique if source.kind == _SOURCE_KIND_IMAGE_SEQUENCE]
            self.video_browse.set_text("; ".join(str(video) for video in videos))
            self.image_sequence_browse.set_text("; ".join(str(folder) for folder in folders))
        finally:
            self._syncing_input_source_widgets = False

        videos = self._selected_video_paths()
        self._prune_video_info_cache(videos)
        if not videos:
            self.video_info = None
        self._refresh_video_queue_list()
        self._suggest_scene_dir_from_sources(unique)
        self._update_source_mode_widgets()
        if videos:
            self._load_video_info(show_error=False)
        else:
            self._update_video_info_label()
            self._update_instant_estimate()
            self._update_ready_status()

    def _append_input_sources(self, kind: str, paths: list[Path]) -> None:
        if not paths:
            return
        self._set_input_sources([*self._selected_input_sources(), *(InputSource(kind, path) for path in paths)])

    def _set_video_queue_paths(self, videos: list[Path]) -> None:
        self._set_input_sources(replace_video_sources(self._selected_input_sources(), videos))

    def _queue_dialog_start_path(self) -> str:
        for source in reversed(self._selected_input_sources()):
            if source.kind == _SOURCE_KIND_VIDEO and source.path.is_file():
                return str(source.path.parent)
            if source.kind == _SOURCE_KIND_IMAGE_SEQUENCE and source.path.is_dir():
                return str(source.path)
        if self.scene_dir:
            return self.scene_dir
        return ""

    def _add_input_videos(self) -> None:
        paths, _selected_filter = dialogs.get_open_file_names(
            self,
            i18n.t("ADD_INPUT_VIDEO"),
            self._queue_dialog_start_path(),
            i18n.t("VIDEO_FILE_FILTER"),
        )
        if not paths:
            return
        self._append_input_sources(_SOURCE_KIND_VIDEO, [Path(path) for path in paths])

    def _add_input_image_sequence(self) -> None:
        folder = dialogs.get_existing_directory(
            self,
            i18n.t("ADD_INPUT_IMAGE_SEQUENCE"),
            self._queue_dialog_start_path(),
        )
        if not folder:
            return
        self._append_input_sources(_SOURCE_KIND_IMAGE_SEQUENCE, [Path(folder)])

    def _remove_selected_input_videos(self) -> None:
        selected_keys = {str(item.data(Qt.UserRole)) for item in self.video_queue_list.selectedItems()}
        if not selected_keys:
            return
        sources = self._selected_input_sources()
        removed = [source for source in sources if self._source_key(source) in selected_keys]
        removed_videos = [source.path for source in removed if source.kind == _SOURCE_KIND_VIDEO]
        if removed_videos:
            self._forget_source_videos(removed_videos)
        self._set_input_sources([source for source in sources if self._source_key(source) not in selected_keys])

    def _video_info_for_queue_item(self, video: Path) -> dict | None:
        info = self.video_infos.get(self._video_key(video))
        if info is None and len(self._selected_video_paths()) == 1:
            info = self.video_info
        return info if isinstance(info, dict) else None

    def _video_queue_item_text(self, video: Path) -> str:
        key = self._video_key(video)
        info = self._video_info_for_queue_item(video)
        if key in self.video_info_failures:
            status = i18n.t("VIDEO_QUEUE_STATUS_ERROR")
        else:
            status = self._video_queue_status_text(video)
        if info is not None:
            return i18n.t("VIDEO_QUEUE_ITEM_INFO_FORMAT").format(
                name=video.name or str(video),
                status=status,
                projection=self._video_projection_text(info),
                width=info["width"],
                height=info["height"],
                fps=info["fps"],
                duration=self._format_duration(float(info.get("duration_sec", 0))),
                frames=self._format_number(self._estimated_total_frames(info)),
                folder=str(video.parent),
            )
        return i18n.t("VIDEO_QUEUE_ITEM_FORMAT").format(
            name=video.name or str(video),
            status=status,
            projection=self._video_projection_text(info),
            folder=str(video.parent),
        )

    def _image_sequence_status_text(self, folder: Path) -> str:
        if not folder.is_dir():
            return i18n.t("IMAGE_SEQUENCE_QUEUE_STATUS_MISSING")
        if not self._image_sequence_files(folder):
            return i18n.t("IMAGE_SEQUENCE_QUEUE_STATUS_EMPTY")
        return i18n.t("IMAGE_SEQUENCE_QUEUE_STATUS_READY")

    def _image_sequence_queue_item_text(self, folder: Path) -> str:
        count = len(self._image_sequence_files(folder))
        return i18n.t("IMAGE_SEQUENCE_QUEUE_ITEM_FORMAT").format(
            name=folder.name or str(folder),
            status=self._image_sequence_status_text(folder),
            count=self._format_number(count),
            folder=str(folder),
        )

    def _input_source_item_text(self, source: InputSource) -> str:
        if source.kind == _SOURCE_KIND_IMAGE_SEQUENCE:
            return self._image_sequence_queue_item_text(source.path)
        return self._video_queue_item_text(source.path)

    def _update_video_queue_summary_label(self) -> None:
        if not hasattr(self, "video_queue_summary_label"):
            return
        sources = self._selected_input_sources()
        if not sources:
            self.video_queue_summary_label.setText(i18n.t("NO_INPUT_SOURCE"))
            return
        videos = self._selected_video_paths()
        image_dirs = self._selected_image_sequence_dirs()
        queued, skipped = self._queued_selected_videos()
        probed = sum(1 for video in videos if self._video_info_for_queue_item(video) is not None)
        failed = sum(1 for video in videos if self._video_key(video) in self.video_info_failures)
        if image_dirs:
            image_count = sum(len(self._image_sequence_files(folder)) for folder in image_dirs)
            text = i18n.t("INPUT_SOURCE_QUEUE_SUMMARY_FORMAT").format(
                total=len(sources),
                videos=len(videos),
                queued=len(queued),
                skipped=skipped,
                image_folders=len(image_dirs),
                images=self._format_number(image_count),
                probed=probed,
            )
        else:
            text = i18n.t("VIDEO_QUEUE_SUMMARY_FORMAT").format(
                total=len(videos),
                queued=len(queued),
                skipped=skipped,
                probed=probed,
            )
        if failed:
            text += i18n.t("VIDEO_INFO_FAILED_SUFFIX").format(failed=failed)
        self.video_queue_summary_label.setText(text)

    def _refresh_video_queue_list(self) -> None:
        if not hasattr(self, "video_queue_list"):
            return
        selected_paths = {str(item.data(Qt.UserRole)) for item in self.video_queue_list.selectedItems()}
        self.video_queue_list.blockSignals(True)
        try:
            self.video_queue_list.clear()
            for source in self._selected_input_sources():
                key = self._source_key(source)
                item = QListWidgetItem(self._input_source_item_text(source))
                if source.kind == _SOURCE_KIND_IMAGE_SEQUENCE:
                    item.setIcon(image_folder_source_icon())
                else:
                    item.setIcon(video_source_icon())
                item.setData(Qt.UserRole, key)
                item.setToolTip(str(source.path))
                self.video_queue_list.addItem(item)
                if key in selected_paths:
                    item.setSelected(True)
        finally:
            self.video_queue_list.blockSignals(False)
        self._update_video_queue_summary_label()
        self._update_video_queue_buttons()

    def _update_video_queue_buttons(self) -> None:
        if not hasattr(self, "video_queue_list"):
            return
        has_videos = bool(self._selected_input_sources())
        self.remove_video_btn.setEnabled(bool(self.video_queue_list.selectedItems()))
        self.clear_video_btn.setEnabled(has_videos)

    def _is_multi_video_input(self) -> bool:
        return len(self._selected_video_paths()) > 1

    def _extract_output_mode(self) -> str:
        data = self.output_mode_combo.currentData()
        return str(data or "append")

    def _on_output_mode_changed(self) -> None:
        self._update_video_info_label()
        self._update_instant_estimate()
        self._update_ready_status()

    def _matching_video_sessions_for_path(self, video: Path) -> list[dict]:
        if not self.scene_dir:
            return []
        if not video.is_file():
            return []
        return matching_video_sessions(Path(self.scene_dir), video)

    def _matching_video_sessions(self) -> list[dict]:
        videos = self._selected_video_paths()
        if not videos:
            return []
        return self._matching_video_sessions_for_path(videos[0])

    def _autoload_videos_from_scene_if_empty(self) -> None:
        if not self.scene_dir or self._selected_input_sources():
            return
        scene = Path(self.scene_dir)
        if not scene.is_dir():
            return
        videos = self._source_video_paths_from_project(scene)
        if not videos:
            videos = self._source_video_paths_from_extract_manifest(scene)
        if not videos:
            videos = self._scan_video_paths_under_scene(scene)
        if videos:
            self._set_input_sources([InputSource(_SOURCE_KIND_VIDEO, video) for video in videos])

    def _prune_missing_selected_videos(self) -> bool:
        videos = self._selected_video_paths()
        if not videos:
            return False
        existing = [video for video in videos if video.is_file()]
        if len(existing) == len(videos):
            return False
        missing_keys = {self._video_key(video) for video in videos if not video.is_file()}
        self.video_infos = {key: value for key, value in self.video_infos.items() if key not in missing_keys}
        self.video_info_failures = {
            key: value for key, value in self.video_info_failures.items() if key not in missing_keys
        }
        if not existing:
            self.video_info = None
        existing_keys = {self._video_key(video) for video in existing}
        self._set_input_sources(
            [
                source
                for source in self._selected_input_sources()
                if source.kind != _SOURCE_KIND_VIDEO or self._video_key(source.path) in existing_keys
            ]
        )
        if not existing:
            self._autoload_videos_from_scene_if_empty()
        return True

    @staticmethod
    def _resolve_scene_or_absolute_path(scene: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else scene / path

    @staticmethod
    def _is_supported_video_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS

    def _unique_existing_video_paths(self, scene: Path, values: list[str]) -> list[Path]:
        videos: list[Path] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            path = self._resolve_scene_or_absolute_path(scene, value)
            if not self._is_supported_video_file(path):
                continue
            try:
                key = str(path.resolve()).casefold()
            except OSError:
                key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            videos.append(path)
        return videos

    def _source_video_paths_from_project(self, scene: Path) -> list[Path]:
        data = load_json(source_videos_path(scene), {"videos": []})
        records = data.get("videos")
        if not isinstance(records, list):
            return []
        values: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            source = record.get("source")
            if not isinstance(source, dict):
                continue
            values.append(str(source.get("path") or ""))
        return self._unique_existing_video_paths(scene, values)

    def _source_video_paths_from_extract_manifest(self, scene: Path) -> list[Path]:
        values: list[str] = []
        for session in load_manifest(scene).get("sessions", []):
            if not isinstance(session, dict):
                continue
            source = session.get("source_video")
            if isinstance(source, dict):
                values.append(str(source.get("path") or ""))
            else:
                values.append(str(session.get("source_video_path") or ""))
        return self._unique_existing_video_paths(scene, values)

    def _scan_video_paths_under_scene(self, scene: Path) -> list[Path]:
        videos: list[Path] = []
        try:
            for root, dirs, files in os.walk(scene):
                dirs[:] = sorted(
                    [
                        name
                        for name in dirs
                        if name.casefold() not in _VIDEO_SCAN_EXCLUDED_DIRS and not name.startswith(".")
                    ],
                    key=str.lower,
                )
                for name in sorted(files, key=str.lower):
                    path = Path(root) / name
                    if self._is_supported_video_file(path):
                        videos.append(path)
        except OSError:
            return []
        return videos

    def _video_queue_status_key(self, video: Path) -> str:
        if not video.is_file():
            return "missing"
        matching = self._matching_video_sessions_for_path(video)
        if self._extract_output_mode() == "replace-video":
            return "reextract" if matching else "new"
        return "skip" if matching else "new"

    def _video_queue_status_text(self, video: Path) -> str:
        key = self._video_queue_status_key(video)
        if key == "skip":
            return i18n.t("VIDEO_QUEUE_STATUS_SKIP")
        if key == "reextract":
            return i18n.t("VIDEO_QUEUE_STATUS_REEXTRACT")
        if key == "missing":
            return i18n.t("VIDEO_QUEUE_STATUS_MISSING")
        return i18n.t("VIDEO_QUEUE_STATUS_NEW")

    def _video_projection_text(self, info: dict | None) -> str:
        if not isinstance(info, dict):
            return i18n.t("VIDEO_PROJECTION_UNKNOWN")
        detected = infer_video_projection(info)
        projection = str(detected.get("projection") or "")
        if projection == "equirectangular":
            return i18n.t("VIDEO_PROJECTION_EQUIRECT")
        if projection == "normal":
            return i18n.t("VIDEO_PROJECTION_NORMAL")
        return i18n.t("VIDEO_PROJECTION_UNKNOWN")

    def _queued_selected_videos(self) -> tuple[list[Path], int]:
        videos = self._selected_video_paths()
        mode = self._extract_output_mode()
        if mode == "replace-video":
            return videos, 0
        queued: list[Path] = []
        skipped = 0
        for video in videos:
            if self._matching_video_sessions_for_path(video):
                skipped += 1
            else:
                queued.append(video)
        return queued, skipped

    def _effective_filename_prefix(self, video_path: Path | None = None) -> str:
        prefix = sanitize_filename_prefix(self.prefix_edit.text())
        if prefix and not self._is_multi_video_input():
            return prefix
        if video_path is not None:
            prefix = sanitize_filename_prefix(video_path.stem)
        else:
            video = self.video_browse.text()
            if video:
                prefix = sanitize_filename_prefix(Path(video).stem)
        return prefix or "frame"

    def _prefix_in_use(self, prefix: str) -> bool:
        if not self.scene_dir:
            return False
        scene = Path(self.scene_dir)
        manifest = load_manifest(scene)
        for session in manifest.get("sessions", []):
            if isinstance(session, dict) and session.get("filename_prefix") == prefix:
                return True
        images = scene_images_dir(scene)
        if images.exists():
            return any(images.glob(f"{prefix}_*"))
        return False

    def _unique_prefix(self, base: str, used_prefixes: set[str]) -> str:
        if base not in used_prefixes and not self._prefix_in_use(base):
            used_prefixes.add(base)
            return base
        for index in range(2, 1000):
            candidate = f"{base}_session{index}"
            if candidate not in used_prefixes and not self._prefix_in_use(candidate):
                used_prefixes.add(candidate)
                return candidate
        used_prefixes.add(f"{base}_session")
        return f"{base}_session"

    def _prefix_for_video(self, video_path: Path, used_prefixes: set[str]) -> str:
        mode = self._extract_output_mode()
        matching = self._matching_video_sessions_for_path(video_path)
        if not self._is_multi_video_input():
            prefix = sanitize_filename_prefix(self.prefix_edit.text())
            if prefix:
                return prefix
        base = self._effective_filename_prefix(video_path)
        if mode == "replace-video" and matching:
            prefix = str(matching[0].get("filename_prefix") or base)
            if prefix not in used_prefixes:
                used_prefixes.add(prefix)
                return prefix
            return self._unique_prefix(prefix, used_prefixes)
        return self._unique_prefix(base, used_prefixes)

    def _clear_input_videos(self) -> None:
        self.last_estimate_summary = None
        videos = self._selected_video_paths()
        if self._selected_input_sources():
            self._forget_source_videos(videos)
            self._set_input_sources([])
        self.video_info = None
        self.video_infos.clear()
        self.video_info_failures.clear()
        self._update_video_info_label()
        self._update_instant_estimate()
        self._update_ready_status()
        self.input_videos_cleared.emit()

    def _forget_source_videos(self, videos: list[Path]) -> None:
        if not self.scene_dir or not videos:
            return
        remove_source_videos(Path(self.scene_dir), videos)

    def _on_video_changed(self, _path: str) -> None:
        if self._syncing_input_source_widgets:
            return
        self._set_input_sources([InputSource(_SOURCE_KIND_VIDEO, video) for video in self._video_paths_from_text()])

    def _image_sequence_dir(self) -> Path | None:
        folders = self._selected_image_sequence_dirs()
        return folders[0] if folders else None

    def _image_sequence_files(self, folder: Path | None = None) -> list[Path]:
        root = folder or self._image_sequence_dir()
        if root is None or not root.is_dir():
            return []
        try:
            return sorted(
                (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SEQUENCE_EXTS),
                key=lambda path: path.name.lower(),
            )
        except OSError:
            return []

    def _suggest_scene_dir_from_sources(self, sources: list[InputSource]) -> None:
        if self.scene_dir or not sources:
            return
        roots: set[Path] = set()
        for source in sources:
            if source.kind == _SOURCE_KIND_VIDEO:
                if not source.path.is_file():
                    return
                roots.add(source.path.parent)
            elif source.kind == _SOURCE_KIND_IMAGE_SEQUENCE:
                if not source.path.is_dir():
                    return
                roots.add(source.path)
        try:
            resolved = {root.resolve() for root in roots}
        except OSError:
            return
        if len(resolved) == 1:
            self.scene_dir_suggested.emit(str(next(iter(resolved))))
