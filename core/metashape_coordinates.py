from __future__ import annotations

import numpy as np

from core.metashape_model import MetashapeCamera, MetashapeModel


def metashape_pointcloud_matrix(*, fix_upside_down: bool = True) -> np.ndarray:
    """Return the shared Metashape PLY-to-output coordinate transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix = matrix[[2, 0, 1, 3], :]
    if fix_upside_down:
        matrix = _rot_x_pos90() @ matrix
    return matrix


def metashape_camera_matrix_to_output_world(
    transform: np.ndarray,
    *,
    fix_upside_down: bool = True,
) -> np.ndarray:
    """Convert a Metashape camera-to-world matrix to this app's output convention."""
    converted = metashape_pointcloud_matrix(fix_upside_down=fix_upside_down) @ transform
    converted[:, 1:3] *= -1.0
    return _rot_y_180() @ converted


def metashape_camera_to_world(
    model: MetashapeModel,
    camera: MetashapeCamera,
    *,
    fix_upside_down: bool = True,
) -> np.ndarray:
    """Apply Metashape component transforms, then convert the camera pose."""
    transform = camera.transform.copy()
    component = model.components.get(camera.component_id)
    if component is not None:
        matrix, scale = component
        transform[:3, 3] *= float(scale)
        transform = matrix @ transform
    return metashape_camera_matrix_to_output_world(transform, fix_upside_down=fix_upside_down)


def metashape_pointcloud_file_matrix(*, fix_upside_down: bool = True, scale: float = 1.0) -> np.ndarray:
    """Return the full PLY file transform, including the user scale factor."""
    scale_matrix = np.diag([float(scale), float(scale), float(scale), 1.0]).astype(np.float64)
    return scale_matrix @ metashape_pointcloud_matrix(fix_upside_down=fix_upside_down)


def _rot_x_pos90() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rot_y_180() -> np.ndarray:
    return np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
