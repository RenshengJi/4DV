"""
Transform utilities for dynamic objects
"""

import torch


def object_exists_in_frame(obj_data, frame_idx):
    """Check if dynamic object exists in specified frame (including interpolated frames)"""
    frame_existence = obj_data.get('frame_existence')
    if frame_existence is not None and frame_idx < len(frame_existence):
        return frame_existence[frame_idx].item()

    # Fallback: check if frame_transforms exists for this frame
    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            return True

    return False


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = torch.stack([w, x, y, z])
    return q / torch.norm(q)  # Normalize


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = torch.zeros(3, 3, dtype=q.dtype, device=q.device)
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*w*z
    R[0, 2] = 2*x*z + 2*w*y
    R[1, 0] = 2*x*y + 2*w*z
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*w*x
    R[2, 0] = 2*x*z - 2*w*y
    R[2, 1] = 2*y*z + 2*w*x
    R[2, 2] = 1 - 2*x*x - 2*y*y

    return R


def slerp_quaternion(q1, q2, alpha):
    """Spherical linear interpolation between two quaternions"""
    dot = torch.dot(q1, q2)

    # Clamp dot product to avoid numerical issues
    dot = torch.clamp(dot, -1.0, 1.0)

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + alpha * (q2 - q1)
        return result / torch.norm(result)

    # Calculate angle between quaternions
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # SLERP formula
    w1 = torch.sin((1 - alpha) * theta) / sin_theta
    w2 = torch.sin(alpha * theta) / sin_theta

    result = w1 * q1 + w2 * q2
    return result / torch.norm(result)


def interpolate_transforms(transform1, transform2, alpha, device):
    """
    Interpolate between two 4x4 transformation matrices using LERP for translation and SLERP for rotation.

    Args:
        transform1: 4x4 transformation matrix at t=0
        transform2: 4x4 transformation matrix at t=1
        alpha: interpolation factor in [0, 1]
        device: torch device

    Returns:
        Interpolated 4x4 transformation matrix
    """
    # Extract rotation matrices (3x3) and translations (3x1)
    R1 = transform1[:3, :3]
    t1 = transform1[:3, 3]
    R2 = transform2[:3, :3]
    t2 = transform2[:3, 3]

    # LERP for translation
    t_interp = (1 - alpha) * t1 + alpha * t2

    # Convert rotation matrices to quaternions for SLERP
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)

    # Ensure shortest path interpolation
    if torch.dot(q1, q2) < 0:
        q2 = -q2

    # SLERP for rotation (quaternion interpolation)
    q_interp = slerp_quaternion(q1, q2, alpha)
    R_interp = quaternion_to_rotation_matrix(q_interp)

    # Construct interpolated transform
    transform_interp = torch.eye(4, dtype=transform1.dtype, device=device)
    transform_interp[:3, :3] = R_interp
    transform_interp[:3, 3] = t_interp

    return transform_interp


def get_object_transform_to_frame(obj_data, frame_idx):
    """
    Get transform matrix from canonical space to specified frame.
    Note: Transforms should be pre-computed by prepare_target_frame_transforms for target frames.
    """
    reference_frame = obj_data.get('reference_frame', 0)
    if frame_idx == reference_frame:
        return None

    if 'frame_transforms' not in obj_data:
        return None

    frame_transforms = obj_data['frame_transforms']

    # Direct lookup (works for both context and pre-computed target frames)
    if frame_idx in frame_transforms:
        frame_to_canonical = frame_transforms[frame_idx]
        canonical_to_frame = torch.inverse(frame_to_canonical)
        return canonical_to_frame

    return None


def apply_transform_to_gaussians(gaussians, transform):
    """Apply transform to Gaussian parameters"""
    if torch.allclose(transform, torch.zeros_like(transform), atol=1e-6):
        print(f"Warning: Zero transform matrix detected! Using identity matrix instead")
        transform = torch.eye(4, dtype=transform.dtype, device=transform.device)
    else:
        det_val = torch.det(transform[:3, :3].float()).abs()
        if det_val < 1e-8:
            print(f"Warning: Singular transform matrix (det={det_val:.2e})!")

    transformed_gaussians = gaussians.clone()

    positions = gaussians[:, :3]
    positions_homo = torch.cat([positions, torch.ones(
        positions.shape[0], 1, device=positions.device)], dim=1)
    transformed_positions = torch.mm(
        transform, positions_homo.T).T[:, :3]
    transformed_gaussians[:, :3] = transformed_positions

    return transformed_gaussians
