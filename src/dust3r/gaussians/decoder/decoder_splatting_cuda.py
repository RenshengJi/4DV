from dataclasses import dataclass
from typing import Literal, List

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int32
from torch import Tensor

from ..adapter import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
from ..utils import pose_encoding_to_camera, build_covariance


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    near: float
    far: float


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: List[Gaussians] | Gaussians,
        extrinsics: List[Float[Tensor, "batch 7"]] | Float[Tensor, "batch 4 4"],
        intrinsics: List[Float[Tensor, "batch 3 3"]] | Float[Tensor, "batch 3 3"],
        image_shape: List[Int32[Tensor, "batch 2"]] | Int32[Tensor, "batch 2"],
        near: List[Float[Tensor, "batch"]] | Float[Tensor, "batch"] | None = None,
        far: List[Float[Tensor, "batch"]] | Float[Tensor, "batch"] | None = None,
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: List[Float[Tensor, "batch 3"]] | Float[Tensor, "batch 3"] | None = None,
        cam_trans_delta: List[Float[Tensor, "batch 3"]] | Float[Tensor, "batch 3"] | None = None,
    ) -> DecoderOutput:
        if isinstance(gaussians, List):
            gaussians = Gaussians(
                means=torch.cat([g.means for g in gaussians], dim=0),
                scales=torch.cat([g.scales for g in gaussians], dim=0),
                rotations=torch.cat([g.rotations for g in gaussians], dim=0),
                harmonics=torch.cat([g.harmonics for g in gaussians], dim=0),
                opacities=torch.cat([g.opacities for g in gaussians], dim=0),
            )
            extrinsics = torch.cat(extrinsics, dim=0)
            intrinsics = torch.cat(intrinsics, dim=0)
            image_shape = torch.cat(image_shape, dim=0)
            if near is not None:
                near = torch.cat(near, dim=0)
            if far is not None:
                far = torch.cat(far, dim=0)
            if cam_rot_delta is not None:
                cam_rot_delta = torch.cat(cam_rot_delta, dim=0)
            if cam_trans_delta is not None:
                cam_trans_delta = torch.cat(cam_trans_delta, dim=0)

        b, _, _ = intrinsics.shape
        extrinsics = pose_encoding_to_camera(extrinsics)
        gaussians.covariances = build_covariance(
            gaussians.scales,
            gaussians.rotations
        )

        if near is None:
            near = torch.full((b,), self.cfg.near, dtype=extrinsics.dtype, device=extrinsics.device)
        if far is None:
            far = torch.full((b,), self.cfg.far, dtype=extrinsics.dtype, device=extrinsics.device)

        color, depth = render_cuda(
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            repeat(self.background_color, "c -> b c", b=b),
            gaussians.means,
            gaussians.covariances,
            gaussians.harmonics,
            gaussians.opacities,
            scale_invariant=self.make_scale_invariant,
            cam_rot_delta=cam_rot_delta if cam_rot_delta is not None else None,
            cam_trans_delta=cam_trans_delta if cam_trans_delta is not None else None,
        )
        return DecoderOutput(color, depth)
