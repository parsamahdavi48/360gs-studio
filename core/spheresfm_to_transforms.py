"""Convert a SphereSfM sparse model to equirectangular transforms.json.

SphereSfM writes COLMAP sparse models with an additional SPHERE camera model
(model id 11). COLMAP's stock Python helper does not always know that model, so
this module uses the repository's small COLMAP reader instead of depending on
the vendored SphereSfM checkout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from core.colmap_sparse_model import Camera, Point3D, colmap_pose_to_c2w, read_model, resolve_image_path


def output_file_path(image_path: Path, images_dir: Path, output_dir: Path, mode: str) -> str:
    try:
        rel_to_images = image_path.resolve().relative_to(images_dir.resolve())
    except ValueError:
        rel_to_images = Path(image_path.name)

    if mode == "relative":
        return rel_to_images.as_posix()
    if mode == "images-prefix":
        return (Path("images") / rel_to_images).as_posix()
    if mode == "relative-to-output":
        return os.path.relpath(image_path.resolve(), output_dir.resolve()).replace(os.sep, "/")
    if mode == "absolute":
        return image_path.resolve().as_posix()
    raise ValueError(f"Unsupported image path mode: {mode}")


def camera_payload(camera: Camera) -> dict[str, float | int]:
    f = float(camera.params[0]) if camera.params else float(camera.width) / 2.0
    cx = float(camera.params[1]) if len(camera.params) > 1 else (camera.width - 1) / 2.0
    cy = float(camera.params[2]) if len(camera.params) > 2 else (camera.height - 1) / 2.0
    return {
        "w": int(camera.width),
        "h": int(camera.height),
        "fl_x": f,
        "fl_y": f,
        "cx": cx,
        "cy": cy,
    }


def write_ascii_ply(path: Path, points: dict[int, Point3D]) -> None:
    ordered = [points[key] for key in sorted(points)]
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(ordered)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in ordered:
            x, y, z = point.xyz
            r, g, b = point.rgb
            f.write(f"{x:.10g} {y:.10g} {z:.10g} {r:d} {g:d} {b:d}\n")


def convert(
    model_dir: Path,
    output_dir: Path,
    images_dir: Path,
    *,
    image_path_mode: str = "relative",
    opengl_camera: bool = True,
    write_pointcloud: bool = True,
) -> dict:
    cameras, images, points, resolved_model = read_model(model_dir)
    if not images:
        raise ValueError(f"No registered images found in sparse model: {resolved_model}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_images = sorted(images.values(), key=lambda image: (image.name.lower(), image.image_id))
    first_camera = cameras[ordered_images[0].camera_id]
    if any(cameras[image.camera_id].model != "SPHERE" for image in ordered_images):
        models = sorted({cameras[image.camera_id].model for image in ordered_images})
        raise ValueError(f"SphereSfM export requires SPHERE cameras, got: {', '.join(models)}")

    frames: list[dict] = []
    for image in ordered_images:
        camera = cameras[image.camera_id]
        image_path = resolve_image_path(images_dir, image.name)
        frame = {
            "file_path": output_file_path(image_path, images_dir, output_dir, image_path_mode),
            "transform_matrix": colmap_pose_to_c2w(image, opengl_camera=opengl_camera).tolist(),
        }
        frame.update(camera_payload(camera))
        frames.append(frame)

    data: dict = {
        "camera_model": "EQUIRECTANGULAR",
        **camera_payload(first_camera),
        "frames": frames,
        "source": {
            "type": "spheresfm_colmap_sparse",
            "model_dir": str(resolved_model),
            "images_dir": str(images_dir),
            "camera_convention": "opengl" if opengl_camera else "colmap",
        },
    }

    if write_pointcloud:
        ply_path = output_dir / "pointcloud.ply"
        write_ascii_ply(ply_path, points)
        data["ply_file_path"] = "pointcloud.ply"

    json_path = output_dir / "transforms.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {
        "model_dir": str(resolved_model),
        "output_dir": str(output_dir),
        "transforms": str(json_path),
        "pointcloud": str(output_dir / "pointcloud.ply") if write_pointcloud else "",
        "num_images": len(frames),
        "num_points": len(points),
    }


def parse_args():
    from core.spheresfm_to_transforms_cli import parse_args as _parse_args

    return _parse_args()


def main() -> None:
    from core.spheresfm_to_transforms_cli import main as _main

    _main()


if __name__ == "__main__":
    main()
