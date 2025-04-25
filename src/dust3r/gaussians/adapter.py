from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ..utils.geometry import get_world_rays
from .utils import build_covariance, RGB2SH


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]
    scales: Optional[Float[Tensor, "*batch 3"]] = None
    rotations: Optional[Float[Tensor, "*batch 4"]] = None


@dataclass
class GaussianAdapterCfg:
    sh_degree: int
    gaussian_scale_min: float = 1e-6
    gaussian_scale_max: float = 0.3
    only_rest: bool = False
    scale_factor: float = 0.01
    scale_activation: str = "softplus"


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=sh,
            opacities=opacities,
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        # 3 for means_offset, 1 for opacity, 3 for scales, 4 for rotations, and 3 * d_sh for spherical harmonics
        return 3 + 1 + 7 + 3 * self.d_sh


class UnifiedGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        rgbs: Float[Tensor, "*#batch 3"] = None,
        eps: float = 1e-8,
    ) -> Gaussians:
        mean_offsets, opacities, scales, rotations, sh = raw_gaussians.split((3, 1, 3, 4, 3 * self.d_sh), dim=-1)

        opacities = opacities.sigmoid()

        if self.cfg.scale_activation == "softplus":
            scales = self.cfg.scale_factor * F.softplus(scales)
            scales = scales.clamp_max(self.cfg.gaussian_scale_max)
        elif self.cfg.scale_activation == "sigmoid":
            scales = self.cfg.gaussian_scale_min + (self.cfg.gaussian_scale_max - self.cfg.gaussian_scale_min) * scales.sigmoid()
        elif self.cfg.scale_activation == "exp":
            scales = self.cfg.scale_factor * torch.exp(scales)
            scales = scales.clamp_max(self.cfg.gaussian_scale_max)
        else:
            raise ValueError(f"Unknown scale activation: {self.cfg.scale_activation}")

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3) * self.sh_mask

        if self.cfg.only_rest and rgbs is not None:
            sh = torch.cat(
                (
                    RGB2SH(rgbs * 0.5 + 0.5).unsqueeze(-1),
                    sh[..., 1:],
                ),
                dim=-1,
            )

        covariances = build_covariance(scales, rotations)

        return Gaussians(
            means=means+mean_offsets,
            covariances=covariances,
            harmonics=sh,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )
