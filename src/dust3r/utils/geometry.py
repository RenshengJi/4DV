# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
from einops import einsum, rearrange, reduce, repeat

from dust3r.utils.misc import invalid_to_zeros, invalid_to_nans
from dust3r.utils.device import to_numpy


def xy_grid(
    W,
    H,
    device=None,
    origin=(0, 0),
    unsqueeze=None,
    cat_dim=-1,
    homogeneous=False,
    **arange_kw,
):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:

        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:

        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:

                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:

                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
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
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
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
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def normalize_pointcloud(
    pts1, pts2, norm_mode="avg_dis", valid1=None, valid2=None, ret_factor=False
):
    """renorm pointmaps pts1, pts2 with norm_mode"""
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split("_")

    if norm_mode == "avg":

        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = (
            invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        )
        all_pts = (
            torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1
        )

        all_dis = all_pts.norm(dim=-1)
        if dis_mode == "dis":
            pass  # do nothing
        elif dis_mode == "log1p":
            all_dis = torch.log1p(all_dis)
        elif dis_mode == "warp-log1p":

            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:, : W1 * H1].view(-1, H1, W1, 1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:, W1 * H1 :].view(-1, H2, W2, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f"bad {dis_mode=}")

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:

        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = (
            torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1
        )

        all_dis = all_pts.norm(dim=-1)

        if norm_mode == "avg":
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == "median":
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == "sqrt":
            norm_factor = all_dis.sqrt().nanmean(dim=1) ** 2
        else:
            raise ValueError(f"bad {norm_mode=}")

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        res = res + (norm_factor,)
    return res


def normalize_pointcloud_group(
    pts_list,
    norm_mode="avg_dis",
    valid_list=None,
    conf_list=None,
    ret_factor=False,
    ret_factor_only=False,
):
    """renorm pointmaps pts1, pts2 with norm_mode"""
    for pts in pts_list:
        assert pts.ndim >= 3 and pts.shape[-1] == 3

    norm_mode, dis_mode = norm_mode.split("_")

    if norm_mode == "avg":

        nan_pts_list, nnz_list = zip(
            *[
                invalid_to_zeros(pts1, valid1, ndim=3)
                for pts1, valid1 in zip(pts_list, valid_list)
            ]
        )
        all_pts = torch.cat(nan_pts_list, dim=1)
        if conf_list is not None:
            nan_conf_list = [
                invalid_to_zeros(conf1[..., None], valid1, ndim=3)[0]
                for conf1, valid1 in zip(conf_list, valid_list)
            ]
            all_conf = torch.cat(nan_conf_list, dim=1)[..., 0]
        else:
            all_conf = torch.ones_like(all_pts[..., 0])

        all_dis = all_pts.norm(dim=-1)
        if dis_mode == "dis":
            pass  # do nothing
        elif dis_mode == "log1p":
            all_dis = torch.log1p(all_dis)
        elif dis_mode == "warp-log1p":

            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H_W_list = [pts.shape[1:-1] for pts in pts_list]
            pts_list = [
                pts
                * warp_factor[:, sum(H_W_list[:i]) : sum(H_W_list[: i + 1])].view(
                    -1, H, W, 1
                )
                for i, (pts, (H, W)) in enumerate(zip(pts_list, H_W_list))
            ]
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f"bad {dis_mode=}")

        norm_factor = (all_conf * all_dis).sum(dim=1) / (all_conf.sum(dim=1) + 1e-8)
    else:

        nan_pts_list = [
            invalid_to_nans(pts1, valid1, ndim=3)
            for pts1, valid1 in zip(pts_list, valid_list)
        ]

        all_pts = torch.cat(nan_pts_list, dim=1)

        all_dis = all_pts.norm(dim=-1)

        if norm_mode == "avg":
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == "median":
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == "sqrt":
            norm_factor = all_dis.sqrt().nanmean(dim=1) ** 2
        else:
            raise ValueError(f"bad {norm_mode=}")

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts_list[0].ndim:
        norm_factor.unsqueeze_(-1)

    if ret_factor_only:

        return norm_factor

    res = [pts / norm_factor for pts in pts_list]
    if ret_factor:
        return res, norm_factor
    return res


@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):

    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = (
        invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1)
        if z2 is not None
        else None
    )
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_group_pointcloud_depth(zs, valid_masks, quantile=0.5):

    _zs = [
        invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
        for z1, valid_mask1 in zip(zs, valid_masks)
    ]
    _z = torch.cat(_zs, dim=-1)

    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(
    pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True
):

    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = (
        invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3)
        if pts2 is not None
        else None
    )
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


@torch.no_grad()
def get_group_pointcloud_center_scale(pts, valid_masks=None, z_only=False, center=True):

    _pts = [
        invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
        for pts1, valid_mask1 in zip(pts, valid_masks)
    ]
    _pts = torch.cat(_pts, dim=1)

    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2))
    reciprocal_in_P2 = nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist

    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))


def weighted_procrustes(A, B, w, use_weights=True, eps=1e-16, return_T=False):
    """
    X: torch tensor B x N x 3
    Y: torch tensor B x N x 3
    w: torch tensor B x N
    """
    assert len(A) == len(B)
    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        a_mean = (w_norm * A).sum(dim=1, keepdim=True)
        b_mean = (w_norm * B).sum(dim=1, keepdim=True)

        A_c = A - a_mean
        B_c = B - b_mean

        H = torch.einsum("bni,bnj->bij", A_c, w_norm * B_c)

    else:
        a_mean = A.mean(axis=1, keepdim=True)
        b_mean = B.mean(axis=1, keepdim=True)

        A_c = A - a_mean
        B_c = B - b_mean

        H = torch.einsum("bij,bik->bjk", A_c, B_c)

    U, S, V = torch.svd(H)  # U: B x 3 x 3, S: B x 3, V: B x 3 x 3
    Z = torch.eye(3).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))  # B x 3 x 3
    R = V @ Z @ U.transpose(1, 2)  # B x 3 x 3
    t = b_mean - torch.einsum("bij,bjk->bik", R, a_mean.transpose(-2, -1)).transpose(
        -2, -1
    )
    if return_T:
        T = torch.eye(4).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)
        T[:, :3, :3] = R
        T[:, :3, 3] = t.squeeze()
        return T
    return R, t.squeeze()


def homogenize_points(points):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors):
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(homogeneous_coordinates, transformation):
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(homogeneous_coordinates, extrinsics):
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def unproject(coordinates, z, intrinsics):
    """
    Unproject 2D camera coordinates with the given Z values.
    Args:
        coordinates: 2D pixel coordinates in camera space.
            Shape: (*#batch, 2).
        z: Depth values for each pixel.
            Shape: (*#batch).
        intrinsics: Camera intrinsics matrix.
            Shape: (*#batch, 3, 3).
    Returns:
        Ray directions in camera coordinates.
            Shape: (*#batch, 3).
    """

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(coordinates, extrinsics, intrinsics):
    """
    Get the ray origins and directions in world coordinates.
    Args:
        coordinates: 2D pixel coordinates in camera space.
            Shape: (*#batch, 2).
        extrinsics: Camera-to-world transformation matrix.
            Shape: (*#batch, 4, 4).
        intrinsics: Camera intrinsics matrix.
            Shape: (*#batch, 3, 3).
    Returns:
        origins: Ray origins in world coordinates.
            Shape: (*#batch, 3).
        directions: Ray directions in world coordinates.
            Shape: (*#batch, 3).
    """
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_fov(intrinsics, shapes):
    """
    Calculate the field of view (FOV) from the camera intrinsics.
    Args:
        intrinsics: Camera intrinsics matrix.
            Shape: (*#batch, 3, 3).
        shapes: List of shapes for each image in the batch.
            Shape: (*#batch, 2).
    Returns:
        fov: Field of view in radians.
            Shape: (*#batch, 2).
    """
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = einsum(intrinsics_inv, vector, "b i j, b j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector(
        torch.cat(
            [torch.zeros_like(shapes[:, 1:2]), shapes[:, 0:1] / 2 - 0.5, torch.ones_like(shapes[:, 0:1])], dim=-1
        )
    )
    right = process_vector(
        torch.cat(
            [shapes[:, 1:2] - 1, shapes[:, 0:1] / 2 - 0.5, torch.ones_like(shapes[:, 0:1])], dim=-1
        )
    )
    top = process_vector(
        torch.cat(
            [shapes[:, 1:2] / 2 - 0.5, torch.zeros_like(shapes[:, 0:1]), torch.ones_like(shapes[:, 0:1])], dim=-1
        )
    )
    bottom = process_vector(
        torch.cat(
            [shapes[:, 1:2] / 2 - 0.5, shapes[:, 0:1] - 1, torch.ones_like(shapes[:, 0:1])], dim=-1
        )
    )
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)
