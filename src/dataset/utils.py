"""
Utility functions for dataset processing.
Extracted from dust3r to avoid dependencies.
"""
import numpy as np


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Convert depthmap to 3D points in camera coordinates.

    Args:
        depthmap: HxW depth array
        camera_intrinsics: 3x3 camera matrix
        pseudo_focal: Optional pseudo focal length (not used in our case)

    Returns:
        X_cam: HxWx3 array of 3D points in camera coordinates
        valid_mask: HxW boolean mask of valid points
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0

    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal

    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    valid_mask = depthmap > 0.0
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(
    depthmap, camera_intrinsics, camera_pose, **kw
):
    """
    Convert depthmap to 3D points in world coordinates.

    Args:
        depthmap: HxW depth array
        camera_intrinsics: 3x3 camera matrix
        camera_pose: 4x4 cam2world transformation matrix

    Returns:
        X_world: HxWx3 array of 3D points in world coordinates
        valid_mask: HxW boolean mask of valid points
    """
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        X_world = (
            np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        )

    return X_world, valid_mask


def colmap_to_opencv_intrinsics(K):
    """
    Convert COLMAP camera intrinsics to OpenCV convention.

    Coordinates of the center of the top-left pixels are:
    - (0.5, 0.5) in COLMAP
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Convert OpenCV camera intrinsics to COLMAP convention.

    Coordinates of the center of the top-left pixels are:
    - (0,0) in OpenCV
    - (0.5, 0.5) in COLMAP
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K
