"""Step 4 preview rendering and image-count helpers."""

from __future__ import annotations

from pathlib import Path

from core.metashape_preview_targets import metashape_output_count_for_actions
from core.scene_layout import scene_images_dir
from gui import i18n
from gui.cubemap.view_config import _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS


class Step4PreviewCountsMixin:
    # -- ビュー --

    def _on_views_changed(self) -> None:
        self._update_output_count()
        self._schedule_render_preview()

    def _schedule_render_preview(self) -> None:
        if self._preview_render_timer.isActive():
            self._preview_render_pending = True
            self._preview_render_timer.start()
            return
        self._preview_render_pending = False
        self._render_preview()
        self._preview_render_timer.start()

    def _flush_scheduled_render_preview(self) -> None:
        if not self._preview_render_pending:
            return
        self._preview_render_pending = False
        self._render_preview()

    def _render_preview(self) -> None:
        try:
            views = (
                []
                if self._uses_direct_equirect_output()
                or (self._spheresfm_runs_conversion() and self._uses_spheresfm_3dgut_output())
                else self.view_config.collect_views(include_disabled=True)
            )
        except Exception:
            views = []
        mask_dir = str(self._mask_dir()) if self.scene_dir else ""
        self.preview.render(views, mask_dir)

    def _count_input_images(self) -> int:
        if not self.scene_dir:
            return 0
        scene = Path(self.scene_dir)
        images = scene_images_dir(scene)
        roots = [images] if images.is_dir() else [scene]
        exts = {".jpg", ".jpeg", ".png"}
        seen: set[str] = set()
        count = 0
        for root in roots:
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    key = str(p.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        count += 1
        return count

    def _refresh_input_image_count(self) -> None:
        self._input_image_count = len(getattr(self.preview, "preview_images", []) or [])

    def _update_output_count(self) -> None:
        label = i18n.t("OUTPUT_IMAGE_COUNT_LABEL")
        if self._uses_direct_equirect_output() or (
            self._spheresfm_runs_conversion() and self._uses_spheresfm_3dgut_output()
        ):
            action_counts = getattr(self, "_metashape_preview_action_counts", None)
            if self._uses_direct_equirect_output() and action_counts is not None:
                count = metashape_output_count_for_actions(
                    action_counts,
                    enabled_view_count=1,
                    direct_output=True,
                )
            else:
                count = self._input_image_count
            count_text = i18n.t("OUTPUT_IMAGE_COUNT_DIRECT_FORMAT").format(count=count)
            self.view_config.set_output_count_text(f"{label}: {count_text}")
            self._update_lfs_auto_steps_scaler()
            return
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            self.view_config.set_output_count_text(f"{label}: -")
            self._update_lfs_auto_steps_scaler()
            return
        enabled = sum(1 for v in views if v["enabled"])
        action_counts = getattr(self, "_metashape_preview_action_counts", None)
        if self._is_metashape_method() and action_counts is not None:
            total = metashape_output_count_for_actions(action_counts, enabled_view_count=enabled)
        else:
            sources = self._input_image_count
            total = sources * enabled
        warn = ""
        if enabled > _BLOCK_ENABLED_VIEWS:
            warn = " [超過]"
        elif enabled > _WARN_ENABLED_VIEWS:
            warn = " [多い]"
        count_text = i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=total)
        self.view_config.set_output_count_text(f"{label}: {count_text}{warn}")
        self._update_lfs_auto_steps_scaler()
