"""OpenGL direct-Cube6 preview widget for the AprilTag dev GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QImage, QPixmap, QVector3D, QVector4D
from PySide6.QtOpenGL import QOpenGLTexture
from PySide6.QtWidgets import QStackedLayout, QWidget

from devtools.apriltag.cubemap_preview import (
    CubemapFrameGroup,
    CubemapPreviewSamplerFace,
    cubemap_preview_sampler_faces,
)
from gui.common.perspective_image_view import (
    _GL_COLOR_BUFFER_BIT,
    _GL_TEXTURE0,
    _GL_TRIANGLE_STRIP,
    PerspectiveGLImageView,
    PerspectiveImageView,
    PerspectiveLabelOverlay,
    bgr_to_qimage,
)
from gui.common.perspective_preview import PerspectiveParams


class AprilTagCubemapGLView(PerspectiveGLImageView):
    """OpenGL perspective view that samples Cube6 pinhole face textures directly."""

    _FRAGMENT_SHADER = """
        #ifdef GL_ES
        precision highp float;
        #endif

        uniform int u_face_count;
        uniform sampler2D u_tex0;
        uniform sampler2D u_tex1;
        uniform sampler2D u_tex2;
        uniform sampler2D u_tex3;
        uniform sampler2D u_tex4;
        uniform sampler2D u_tex5;
        uniform sampler2D u_tex6;
        uniform sampler2D u_tex7;

        uniform vec3 u_face0_col0;
        uniform vec3 u_face0_col1;
        uniform vec3 u_face0_col2;
        uniform vec3 u_face1_col0;
        uniform vec3 u_face1_col1;
        uniform vec3 u_face1_col2;
        uniform vec3 u_face2_col0;
        uniform vec3 u_face2_col1;
        uniform vec3 u_face2_col2;
        uniform vec3 u_face3_col0;
        uniform vec3 u_face3_col1;
        uniform vec3 u_face3_col2;
        uniform vec3 u_face4_col0;
        uniform vec3 u_face4_col1;
        uniform vec3 u_face4_col2;
        uniform vec3 u_face5_col0;
        uniform vec3 u_face5_col1;
        uniform vec3 u_face5_col2;
        uniform vec3 u_face6_col0;
        uniform vec3 u_face6_col1;
        uniform vec3 u_face6_col2;
        uniform vec3 u_face7_col0;
        uniform vec3 u_face7_col1;
        uniform vec3 u_face7_col2;

        uniform vec4 u_face0_intr;
        uniform vec4 u_face1_intr;
        uniform vec4 u_face2_intr;
        uniform vec4 u_face3_intr;
        uniform vec4 u_face4_intr;
        uniform vec4 u_face5_intr;
        uniform vec4 u_face6_intr;
        uniform vec4 u_face7_intr;

        uniform vec2 u_face0_size;
        uniform vec2 u_face1_size;
        uniform vec2 u_face2_size;
        uniform vec2 u_face3_size;
        uniform vec2 u_face4_size;
        uniform vec2 u_face5_size;
        uniform vec2 u_face6_size;
        uniform vec2 u_face7_size;

        uniform float u_yaw_rad;
        uniform float u_pitch_rad;
        uniform float u_fov_rad;
        uniform vec2 u_viewport_origin;
        uniform vec2 u_viewport_size;

        vec3 faceLocal(int index, vec3 ray) {
            if (index == 0) {
                return vec3(dot(ray, u_face0_col0), dot(ray, u_face0_col1), dot(ray, u_face0_col2));
            }
            if (index == 1) {
                return vec3(dot(ray, u_face1_col0), dot(ray, u_face1_col1), dot(ray, u_face1_col2));
            }
            if (index == 2) {
                return vec3(dot(ray, u_face2_col0), dot(ray, u_face2_col1), dot(ray, u_face2_col2));
            }
            if (index == 3) {
                return vec3(dot(ray, u_face3_col0), dot(ray, u_face3_col1), dot(ray, u_face3_col2));
            }
            if (index == 4) {
                return vec3(dot(ray, u_face4_col0), dot(ray, u_face4_col1), dot(ray, u_face4_col2));
            }
            if (index == 5) {
                return vec3(dot(ray, u_face5_col0), dot(ray, u_face5_col1), dot(ray, u_face5_col2));
            }
            if (index == 6) {
                return vec3(dot(ray, u_face6_col0), dot(ray, u_face6_col1), dot(ray, u_face6_col2));
            }
            return vec3(dot(ray, u_face7_col0), dot(ray, u_face7_col1), dot(ray, u_face7_col2));
        }

        vec4 faceIntr(int index) {
            if (index == 0) return u_face0_intr;
            if (index == 1) return u_face1_intr;
            if (index == 2) return u_face2_intr;
            if (index == 3) return u_face3_intr;
            if (index == 4) return u_face4_intr;
            if (index == 5) return u_face5_intr;
            if (index == 6) return u_face6_intr;
            return u_face7_intr;
        }

        vec2 faceSize(int index) {
            if (index == 0) return u_face0_size;
            if (index == 1) return u_face1_size;
            if (index == 2) return u_face2_size;
            if (index == 3) return u_face3_size;
            if (index == 4) return u_face4_size;
            if (index == 5) return u_face5_size;
            if (index == 6) return u_face6_size;
            return u_face7_size;
        }

        vec4 sampleFace(int index, vec2 uv) {
            if (index == 0) return texture2D(u_tex0, uv);
            if (index == 1) return texture2D(u_tex1, uv);
            if (index == 2) return texture2D(u_tex2, uv);
            if (index == 3) return texture2D(u_tex3, uv);
            if (index == 4) return texture2D(u_tex4, uv);
            if (index == 5) return texture2D(u_tex5, uv);
            if (index == 6) return texture2D(u_tex6, uv);
            return texture2D(u_tex7, uv);
        }

        void main() {
            vec2 view_coord = (gl_FragCoord.xy - u_viewport_origin) / u_viewport_size;
            float view_x = view_coord.x * 2.0 - 1.0;
            float view_y = view_coord.y * 2.0 - 1.0;
            float focal = 1.0 / tan(u_fov_rad * 0.5);
            vec3 ray = normalize(vec3(view_x, view_y, focal));

            float cp = cos(u_pitch_rad);
            float sp = sin(u_pitch_rad);
            vec3 pitched = vec3(
                ray.x,
                cp * ray.y - sp * ray.z,
                sp * ray.y + cp * ray.z
            );

            float cy = cos(u_yaw_rad);
            float sy = sin(u_yaw_rad);
            vec3 preview_ray = vec3(
                cy * pitched.x + sy * pitched.z,
                pitched.y,
                -sy * pitched.x + cy * pitched.z
            );

            vec4 best_color = vec4(0.063, 0.063, 0.063, 1.0);
            float best_z = -1.0e20;
            for (int i = 0; i < 8; ++i) {
                if (i < u_face_count) {
                    vec3 local = faceLocal(i, preview_ray);
                    if (local.z > 1.0e-8 && local.z > best_z) {
                        vec4 intr = faceIntr(i);
                        vec2 size = faceSize(i);
                        float image_x = intr.x * (local.x / local.z) + intr.z;
                        float image_y = intr.w - intr.y * (local.y / local.z);
                        if (image_x >= 0.0 && image_y >= 0.0 && image_x <= size.x - 1.0 && image_y <= size.y - 1.0) {
                            vec2 uv = vec2((image_x + 0.5) / size.x, (image_y + 0.5) / size.y);
                            best_color = sampleFace(i, uv);
                            best_z = local.z;
                        }
                    }
                }
            }
            gl_FragColor = best_color;
        }
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._sampler_faces: tuple[CubemapPreviewSamplerFace, ...] = ()
        self._face_images: list[QImage] = []
        self._textures: list[QOpenGLTexture] = []

    def set_cubemap_faces(
        self,
        faces: tuple[CubemapPreviewSamplerFace, ...],
        params: PerspectiveParams,
        *,
        overlays: list[PerspectiveLabelOverlay] | None = None,
        logical_size: QSize | None = None,
    ) -> None:
        self._sampler_faces = tuple(faces[:8])
        self._face_images = [
            bgr_to_qimage(face.image_bgr).convertToFormat(QImage.Format_RGBA8888)
            for face in self._sampler_faces
        ]
        self._source_image = QImage(1, 1, QImage.Format_RGBA8888)
        size = logical_size or QSize(768, 768)
        self._logical_size = QSize(max(1, size.width()), max(1, size.height()))
        self._params = params
        self._overlays = list(overlays or [])
        if self._initialized and not self._failed:
            self.makeCurrent()
            try:
                self._upload_texture()
            finally:
                self.doneCurrent()
        self._clamp_pan()
        self.update()

    def _upload_texture(self) -> None:
        for texture in self._textures:
            texture.destroy()
        self._textures = []
        if self._failed:
            return
        for image in self._face_images:
            texture = QOpenGLTexture(image, QOpenGLTexture.MipMapGeneration.DontGenerateMipMaps)
            texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setWrapMode(
                QOpenGLTexture.CoordinateDirection.DirectionS,
                QOpenGLTexture.WrapMode.ClampToEdge,
            )
            texture.setWrapMode(
                QOpenGLTexture.CoordinateDirection.DirectionT,
                QOpenGLTexture.WrapMode.ClampToEdge,
            )
            self._textures.append(texture)

    def paintGL(self) -> None:  # noqa: N802 - Qt API
        if self._failed or self._functions is None:
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        full_w = max(1, int(round(self.width() * dpr)))
        full_h = max(1, int(round(self.height() * dpr)))
        self._functions.glViewport(0, 0, full_w, full_h)
        self._functions.glClearColor(0.0, 0.0, 0.0, 0.0)
        self._functions.glClear(_GL_COLOR_BUFFER_BIT)

        if not self._textures or self._program is None or self._vao is None:
            return
        target = self._target_rect()
        if target.width() <= 0 or target.height() <= 0:
            return

        viewport_x = int(round(target.left() * dpr))
        viewport_y = int(round((self.height() - target.bottom()) * dpr))
        viewport_w = max(1, int(round(target.width() * dpr)))
        viewport_h = max(1, int(round(target.height() * dpr)))
        self._functions.glViewport(viewport_x, viewport_y, viewport_w, viewport_h)

        self._program.bind()
        face_count = min(len(self._textures), len(self._sampler_faces), 8)
        self._program.setUniformValue1i(self._program.uniformLocation(b"u_face_count"), face_count)
        for index, texture in enumerate(self._textures[:face_count]):
            self._functions.glActiveTexture(_GL_TEXTURE0 + index)
            texture.bind(index)
            self._program.setUniformValue1i(self._program.uniformLocation(f"u_tex{index}".encode("ascii")), index)
        for index, face in enumerate(self._sampler_faces[:face_count]):
            self._set_face_uniforms(index, face)
        self._program.setUniformValue1f(
            self._program.uniformLocation(b"u_yaw_rad"),
            float(np.deg2rad(float(self._params.yaw_deg))),
        )
        self._program.setUniformValue1f(
            self._program.uniformLocation(b"u_pitch_rad"),
            float(np.deg2rad(float(self._params.pitch_deg))),
        )
        self._program.setUniformValue1f(
            self._program.uniformLocation(b"u_fov_rad"),
            float(np.deg2rad(float(self._params.fov_deg))),
        )
        self._program.setUniformValue(
            self._program.uniformLocation(b"u_viewport_origin"),
            float(viewport_x),
            float(viewport_y),
        )
        self._program.setUniformValue(
            self._program.uniformLocation(b"u_viewport_size"),
            float(viewport_w),
            float(viewport_h),
        )
        self._vao.bind()
        self._functions.glDrawArrays(_GL_TRIANGLE_STRIP, 0, 4)
        self._vao.release()
        for texture in self._textures[:face_count]:
            texture.release()
        self._program.release()

        self._functions.glViewport(0, 0, full_w, full_h)
        if self._overlays:
            self._paint_overlays(target)

    def _set_face_uniforms(self, index: int, face: CubemapPreviewSamplerFace) -> None:
        rotation = np.asarray(face.preview_to_face_rotation, dtype=np.float64)
        for column_index in range(3):
            column = rotation[:, column_index]
            self._program.setUniformValue(  # type: ignore[union-attr]
                self._program.uniformLocation(f"u_face{index}_col{column_index}".encode("ascii")),  # type: ignore[union-attr]
                QVector3D(float(column[0]), float(column[1]), float(column[2])),
            )
        frame = face.frame
        self._program.setUniformValue(  # type: ignore[union-attr]
            self._program.uniformLocation(f"u_face{index}_intr".encode("ascii")),  # type: ignore[union-attr]
            QVector4D(float(frame.fl_x), float(frame.fl_y), float(frame.cx), float(frame.cy)),
        )
        self._program.setUniformValue(  # type: ignore[union-attr]
            self._program.uniformLocation(f"u_face{index}_size".encode("ascii")),  # type: ignore[union-attr]
            float(frame.width),
            float(frame.height),
        )


class AprilTagCubemapPreviewView(QWidget):
    """Stacked preview view with direct Cube6 GPU sampling and equirect fallback."""

    look_dragged = Signal(float, float)
    image_clicked = Signal(float, float)
    gpu_failed = Signal()

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fallback_view = PerspectiveImageView(text)
        self._direct_view = AprilTagCubemapGLView(text)
        self._gpu_failed = False
        self._drag_mode = "pan"

        self._fallback_view.look_dragged.connect(self.look_dragged.emit)
        self._fallback_view.image_clicked.connect(self.image_clicked.emit)
        self._fallback_view.gpu_failed.connect(self.gpu_failed.emit)
        self._direct_view.look_dragged.connect(self.look_dragged.emit)
        self._direct_view.image_clicked.connect(self.image_clicked.emit)
        self._direct_view.gpu_failed.connect(self._on_direct_gpu_failed)

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._fallback_view)
        self._stack.addWidget(self._direct_view)
        self._stack.setCurrentWidget(self._fallback_view)

    @property
    def _source_pixmap(self):  # noqa: ANN001 - compatibility with existing preview tests
        return self._fallback_view._source_pixmap

    def setStyleSheet(self, style_sheet: str) -> None:  # noqa: N802 - Qt API
        super().setStyleSheet(style_sheet)
        self._fallback_view.setStyleSheet(style_sheet)
        self._direct_view.setStyleSheet(style_sheet)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._fallback_view.setText(text)
        self._stack.setCurrentWidget(self._fallback_view)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._fallback_view.set_source_pixmap(pixmap)
        self._stack.setCurrentWidget(self._fallback_view)

    def set_drag_mode(self, mode: str) -> None:
        if mode not in {"pan", "look"}:
            return
        self._drag_mode = mode
        self._fallback_view.set_drag_mode(mode)
        self._direct_view.set_drag_mode(mode)

    def reset_view(self) -> None:
        self._fallback_view.reset_view()
        self._direct_view.reset_view()

    def set_perspective_image_bgr(
        self,
        image: np.ndarray,
        params: PerspectiveParams,
        *,
        overlays: list[PerspectiveLabelOverlay] | None = None,
        logical_size: QSize | None = None,
    ) -> bool:
        shown = self._fallback_view.set_perspective_image_bgr(
            image,
            params,
            overlays=overlays,
            logical_size=logical_size,
        )
        if shown:
            self._stack.setCurrentWidget(self._fallback_view)
        return shown

    def set_cubemap_group(
        self,
        group: CubemapFrameGroup,
        params: PerspectiveParams,
        *,
        overlays: list[PerspectiveLabelOverlay] | None = None,
        logical_size: QSize | None = None,
        image_cache: dict[Path, np.ndarray] | None = None,
    ) -> bool:
        if self._gpu_failed or self._direct_view.failed():
            return False
        faces = cubemap_preview_sampler_faces(group, image_cache=image_cache)
        if not faces:
            return False
        self._direct_view.set_drag_mode(self._drag_mode)
        self._direct_view.set_cubemap_faces(
            faces,
            params,
            overlays=overlays,
            logical_size=logical_size,
        )
        self._stack.setCurrentWidget(self._direct_view)
        return True

    def set_perspective_params(self, params: PerspectiveParams) -> bool:
        if self._stack.currentWidget() is self._direct_view and not self._direct_view.failed():
            self._direct_view.set_perspective_params(params)
            return True
        return self._fallback_view.set_perspective_params(params)

    def set_perspective_label_overlays(self, overlays: list[PerspectiveLabelOverlay]) -> bool:
        if self._stack.currentWidget() is self._direct_view and not self._direct_view.failed():
            self._direct_view.set_label_overlays(overlays)
            return True
        return self._fallback_view.set_perspective_label_overlays(overlays)

    def perspective_logical_size(self) -> QSize:
        if self._stack.currentWidget() is self._direct_view:
            return self._direct_view.logical_image_size()
        return self._fallback_view.perspective_logical_size()

    def is_showing_gpu_perspective(self) -> bool:
        if self._stack.currentWidget() is self._direct_view:
            return not self._gpu_failed and not self._direct_view.failed()
        return self._fallback_view.is_showing_gpu_perspective()

    def _on_direct_gpu_failed(self) -> None:
        self._gpu_failed = True
        self._stack.setCurrentWidget(self._fallback_view)
        self.gpu_failed.emit()
