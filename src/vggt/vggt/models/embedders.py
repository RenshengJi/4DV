"""
Embedders migrated from storm.
"""
import collections.abc
from itertools import repeat
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def _ntuple(n):
    """
    Creates a parser that converts an input to a tuple of length n.

    Args:
        n (int): Length of the tuple.

    Returns:
        Callable: A function that parses the input into a tuple of length n.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PluckerEmbedder(nn.Module):
    """
    Convert rays to plucker embedding
    """

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 1,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])

        x, y = torch.meshgrid(
            torch.arange(self.grid_size[1]),
            torch.arange(self.grid_size[0]),
            indexing="xy",
        )
        x = x.float().reshape(1, -1) + 0.5
        y = y.float().reshape(1, -1) + 0.5
        self.register_buffer("x", x)
        self.register_buffer("y", y)

    def forward(
        self,
        intrinsics: Tensor,
        camtoworlds: Tensor,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Tensor:
        assert intrinsics.shape[-2:] == (3, 3), "intrinsics should be (B, 3, 3)"
        assert camtoworlds.shape[-2:] == (4, 4), "camtoworlds should be (B, 4, 4)"
        intrinsics_shape = intrinsics.shape
        intrinsics = intrinsics.reshape(-1, 3, 3)
        camtoworlds = camtoworlds.reshape(-1, 4, 4)
        if image_size is not None:
            image_size = to_2tuple(image_size)
        else:
            image_size = self.img_size
        if patch_size is not None:
            patch_size = to_2tuple(patch_size)
        else:
            patch_size = self.patch_size

        grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])

        if grid_size != self.grid_size:
            grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])
            x, y = torch.meshgrid(
                torch.arange(grid_size[1]),
                torch.arange(grid_size[0]),
                indexing="xy",
            )
            x = x.float().reshape(1, -1) + 0.5
            y = y.float().reshape(1, -1) + 0.5
            x = x.to(intrinsics.device)
            y = y.to(intrinsics.device)
            intrinsics = intrinsics.clone()
            # intrinsics should be scaled to the grid size
            intrinsics[..., 0, 0] = intrinsics[..., 0, 0] / patch_size[1]
            intrinsics[..., 0, 2] = intrinsics[..., 0, 2] / patch_size[1]
            intrinsics[..., 1, 1] = intrinsics[..., 1, 1] / patch_size[0]
            intrinsics[..., 1, 2] = intrinsics[..., 1, 2] / patch_size[0]
        else:
            x, y = self.x, self.y

        x = x.repeat(intrinsics.size(0), 1)
        y = y.repeat(intrinsics.size(0), 1)
        camera_dirs = torch.nn.functional.pad(
            torch.stack(
                [
                    (x - intrinsics[:, 0, 2][..., None] + 0.5) / intrinsics[:, 0, 0][..., None],
                    (y - intrinsics[:, 1, 2][..., None] + 0.5) / intrinsics[:, 1, 1][..., None],
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )
        directions = torch.sum(camera_dirs[:, :, None, :] * camtoworlds[:, None, :3, :3], dim=-1)
        origins = torch.broadcast_to(camtoworlds[:, :3, -1].unsqueeze(1), directions.shape)
        direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
        viewdirs = directions / (direction_norm + 1e-8)
        cross_prod = torch.cross(origins, viewdirs, dim=-1)
        plucker = torch.cat((cross_prod, viewdirs), dim=-1)
        origins = rearrange(origins, "b (h w) c -> b h w c", h=grid_size[0])
        viewdirs = rearrange(viewdirs, "b (h w) c -> b h w c", h=grid_size[0])
        directions = rearrange(directions, "b (h w) c -> b h w c", h=grid_size[0])
        plucker = rearrange(plucker, "b (h w) c -> b h w c", h=grid_size[0])
        return {
            "origins": origins.view(*intrinsics_shape[:-2], *grid_size, 3),
            "viewdirs": viewdirs.view(*intrinsics_shape[:-2], *grid_size, 3),
            "dirs": directions.view(*intrinsics_shape[:-2], *grid_size, 3),
            "plucker": plucker.view(*intrinsics_shape[:-2], *grid_size, 6),
        }
