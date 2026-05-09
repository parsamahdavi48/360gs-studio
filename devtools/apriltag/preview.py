"""Preview and CSV generation for synthetic AprilTag injection runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.apriltag_detection import detect_apriltags
from core.apriltag_geometry import load_pinhole_frames, points_intersect_image, project_sfm_points, tag_corners_sfm
from core.image_io import imread_unicode, imwrite_unicode


@dataclass(frozen=True)
class SyntheticPreviewResult:
    preview_path: Path
    csv_path: Path
    detected_images: int
    projection_rows: int


def write_synthetic_projection_csv(synthetic_report: Path, output_csv: Path) -> int:
    report = json.loads(synthetic_report.read_text(encoding="utf-8"))
    frames = load_pinhole_frames(Path(report["input_transforms"]))
    corners = tag_corners_sfm(
        np.asarray(report["tag_center_sfm"], dtype=float),
        np.asarray(report["tag_normal_sfm"], dtype=float),
        np.asarray(report["tag_up_sfm"], dtype=float),
        float(report["tag_size_m"]),
        float(report["true_scale"]),
    )

    rows: list[dict[str, str | float]] = []
    for frame in frames:
        projected = project_sfm_points(frame, corners)
        if projected is None or not points_intersect_image(projected, frame.width, frame.height):
            continue
        xs = projected[:, 0]
        ys = projected[:, 1]
        edge_w = max(float(np.linalg.norm(projected[1] - projected[0])), float(np.linalg.norm(projected[2] - projected[3])))
        edge_h = max(float(np.linalg.norm(projected[2] - projected[1])), float(np.linalg.norm(projected[3] - projected[0])))
        rows.append(
            {
                "file_path": frame.file_path,
                "face": Path(frame.file_path).stem.rsplit("_", 1)[-1],
                "center_x_px": round(float(xs.mean()), 2),
                "center_y_px": round(float(ys.mean()), 2),
                "approx_width_px": round(edge_w, 2),
                "approx_height_px": round(edge_h, 2),
                "min_x_px": round(float(xs.min()), 2),
                "min_y_px": round(float(ys.min()), 2),
                "max_x_px": round(float(xs.max()), 2),
                "max_y_px": round(float(ys.max()), 2),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path",
        "face",
        "center_x_px",
        "center_y_px",
        "approx_width_px",
        "approx_height_px",
        "min_x_px",
        "min_y_px",
        "max_x_px",
        "max_y_px",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _draw_detection(image: np.ndarray, tag_id: int, corners: np.ndarray) -> None:
    pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, (0, 255, 0), 6, cv2.LINE_AA)
    x = int(corners[:, 0].min())
    y = int(corners[:, 1].min())
    cv2.putText(image, f"id {tag_id}", (x, max(32, y - 18)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(image, f"id {tag_id}", (x, max(32, y - 18)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3, cv2.LINE_AA)


def _fit_tile(image: np.ndarray, label: str, tile_size: int) -> np.ndarray:
    tile = np.full((tile_size, tile_size, 3), 24, dtype=np.uint8)
    label_h = 48
    content_h = tile_size - label_h
    height, width = image.shape[:2]
    scale = min(tile_size / float(max(1, width)), content_h / float(max(1, height)))
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    y = (content_h - resized.shape[0]) // 2
    x = (tile_size - resized.shape[1]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[:, :, :3]
    cv2.rectangle(tile, (0, content_h), (tile_size, tile_size), (12, 14, 18), -1)
    text = label[-52:]
    cv2.putText(tile, text, (12, content_h + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return tile


def create_detection_contact_sheet(
    transforms_json: Path,
    output_path: Path,
    *,
    tag_size_m: float,
    family: str = "tag36h11",
    tag_ids: set[int] | None = None,
    max_images: int = 12,
    tile_size: int = 520,
    columns: int = 3,
) -> int:
    frames = load_pinhole_frames(transforms_json)
    tiles: list[np.ndarray] = []
    for frame in frames:
        image = imread_unicode(frame.image_path)
        if image is None:
            continue
        detections = detect_apriltags(image, frame, tag_size_m=tag_size_m, family=family, tag_ids=tag_ids)
        if not detections:
            continue
        annotated = image[:, :, :3].copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for detection in detections:
            _draw_detection(annotated, detection.tag_id, detection.corners_px)
        tiles.append(_fit_tile(annotated, frame.file_path, tile_size))
        if len(tiles) >= max_images:
            break

    if not tiles:
        empty = np.full((tile_size, tile_size, 3), 24, dtype=np.uint8)
        cv2.putText(empty, "No AprilTag detections", (32, tile_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
        tiles.append(empty)

    columns = max(1, int(columns))
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * tile_size, columns * tile_size, 3), 18, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        y = row * tile_size
        x = col * tile_size
        sheet[y : y + tile_size, x : x + tile_size] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(output_path, sheet)
    return len(tiles)


def create_synthetic_preview(
    synthetic_dir: Path,
    *,
    family: str = "tag36h11",
    tag_size_m: float,
    tag_ids: set[int] | None,
    preview_path: Path | None = None,
    csv_path: Path | None = None,
    max_images: int = 12,
) -> SyntheticPreviewResult:
    transforms_json = synthetic_dir / "transforms.json"
    report_json = synthetic_dir / "synthetic_apriltag_report.json"
    if not transforms_json.is_file():
        raise FileNotFoundError(f"transforms.json not found: {transforms_json}")
    if not report_json.is_file():
        raise FileNotFoundError(f"synthetic_apriltag_report.json not found: {report_json}")
    preview_path = preview_path or (synthetic_dir / "preview_contact_sheet.jpg")
    csv_path = csv_path or (synthetic_dir / "synthetic_injection_frames.csv")
    projection_rows = write_synthetic_projection_csv(report_json, csv_path)
    detected_images = create_detection_contact_sheet(
        transforms_json,
        preview_path,
        tag_size_m=tag_size_m,
        family=family,
        tag_ids=tag_ids,
        max_images=max_images,
    )
    return SyntheticPreviewResult(
        preview_path=preview_path,
        csv_path=csv_path,
        detected_images=detected_images,
        projection_rows=projection_rows,
    )

