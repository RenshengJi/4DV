from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager
from einops import reduce, rearrange, repeat
from functools import cache
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab, deltaE_cie76
from torch import Tensor


@contextmanager
def tf32_off():
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # disable tf32 temporarily
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original # restore original setting


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


def get_heatmap(info, var_min=0.0, var_max=10.0):
    """
    Extract confidence heatmap from RAFT model info output.

    Args:
        info: Info tensor from RAFT model output with shape (B, 4, H, W)
              where channels are [weight1, weight2, log_var1, log_var2]
        var_min: Minimum variance clamp value
        var_max: Maximum variance clamp value

    Returns:
        heatmap: Confidence heatmap tensor of shape (B, 1, H, W)
    """
    if info is None:
        return None

    raw_b = info[:, 2:]  # Extract variance channels
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)  # Softmax over weight channels

    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)

    return heatmap


@torch.no_grad()
def calc_flow(images, flow_model, chunk_size=16, check_consistency=False, geo_thresh=-1, photo_thresh=-1, interval=1, return_heatmap=False):
    """
    Calculate optical flow between consecutive images in a batch of views.

    Args:
        images: Input images tensor of shape (B, S, C, H, W)
        flow_model: RAFT flow model
        chunk_size: Batch size for flow computation
        check_consistency: Whether to compute consistency masks
        geo_thresh: Geometric consistency threshold
        photo_thresh: Photometric consistency threshold
        interval: Interval between frames for flow computation
        return_heatmap: Whether to return confidence heatmaps

    Returns:
        If return_heatmap=False:
            forward_flow, backward_flow (and consistency masks if check_consistency=True)
        If return_heatmap=True:
            forward_flow, backward_flow, forward_heatmap, backward_heatmap (and consistency masks if check_consistency=True)
    """
    # Shape: (batch_size, num_views, C, H, W)
    if torch.max(images) <= 1.0 + 1e-5:
        images = images * 255.0   # RAFT expects images in [0, 255]

    images_for_raft = images.clone()
    batch_size, num_views, C, H, W = images_for_raft.shape

    # Prepare images for RAFT
    images_raft_proc = rearrange(images_for_raft, "B S C H W -> (B S) C H W")
    flowmap_size = flow_model.args.image_size
    images_raft_proc = F.interpolate(images_raft_proc, size=flowmap_size, mode="bilinear", align_corners=False)
    images_raft_proc = rearrange(images_raft_proc, "(B S) C H W -> B S C H W", B=batch_size)

    images_part1, images_part2 = images_raft_proc[:, :-interval], images_raft_proc[:, interval:]
    images_part1 = rearrange(images_part1, "B S C H W -> (B S) C H W")
    images_part2 = rearrange(images_part2, "B S C H W -> (B S) C H W")

    forward_flow_raft = []
    backward_flow_raft = []
    forward_info_raft = []
    backward_info_raft = []

    for i in range(0, images_part1.shape[0], chunk_size):
        forward_output = flow_model(images_part1[i : i + chunk_size], images_part2[i : i + chunk_size], iters=flow_model.args.iters, test_mode=True)
        forward_flow_raft.append(forward_output["flow"][-1])
        if return_heatmap and "info" in forward_output:
            forward_info_raft.append(forward_output["info"][-1])

        backward_output = flow_model(images_part2[i : i + chunk_size], images_part1[i : i + chunk_size], iters=flow_model.args.iters, test_mode=True)
        backward_flow_raft.append(backward_output["flow"][-1])
        if return_heatmap and "info" in backward_output:
            backward_info_raft.append(backward_output["info"][-1])

    forward_flow_raft = torch.cat(forward_flow_raft, dim=0)
    backward_flow_raft = torch.cat(backward_flow_raft, dim=0)

    if return_heatmap and len(forward_info_raft) > 0 and len(backward_info_raft) > 0:
        forward_info_raft = torch.cat(forward_info_raft, dim=0)
        backward_info_raft = torch.cat(backward_info_raft, dim=0)

    scale_factor = torch.tensor([W / flowmap_size[1], H / flowmap_size[0]], device=forward_flow_raft.device, dtype=forward_flow_raft.dtype)

    forward_flow = F.interpolate(forward_flow_raft, size=(H, W), mode="bilinear", align_corners=False) * scale_factor[None, :, None, None]
    backward_flow = F.interpolate(backward_flow_raft, size=(H, W), mode="bilinear", align_corners=False) * scale_factor[None, :, None, None]

    forward_flow = rearrange(forward_flow, "(B S) C H W -> B S C H W", B=batch_size)
    forward_flow = torch.cat(
        [forward_flow, torch.zeros_like(forward_flow[:, 0:1]).repeat(1,interval,1,1,1)], dim=1
    ) # Pad last frame's forward flow

    backward_flow = rearrange(backward_flow, "(B S) C H W -> B S C H W", B=batch_size)
    backward_flow = torch.cat(
        [torch.zeros_like(backward_flow[:, 0:1]).repeat(1,interval,1,1,1), backward_flow], dim=1
    ) # Pad first frame's backward flow

    # Process heatmaps if requested
    if return_heatmap and len(forward_info_raft) > 0 and len(backward_info_raft) > 0:
        # Extract heatmaps from info tensors
        forward_heatmap_raft = get_heatmap(forward_info_raft, var_min=getattr(flow_model.args, 'var_min', 0.0), var_max=getattr(flow_model.args, 'var_max', 10.0))
        backward_heatmap_raft = get_heatmap(backward_info_raft, var_min=getattr(flow_model.args, 'var_min', 0.0), var_max=getattr(flow_model.args, 'var_max', 10.0))

        # Interpolate heatmaps to original resolution
        forward_heatmap = F.interpolate(forward_heatmap_raft, size=(H, W), mode="bilinear", align_corners=False)
        backward_heatmap = F.interpolate(backward_heatmap_raft, size=(H, W), mode="bilinear", align_corners=False)

        # Reshape and pad heatmaps
        forward_heatmap = rearrange(forward_heatmap, "(B S) C H W -> B S C H W", B=batch_size)
        forward_heatmap = torch.cat(
            [forward_heatmap, torch.zeros_like(forward_heatmap[:, 0:1]).repeat(1,interval,1,1,1)], dim=1
        ) # Pad last frame's forward heatmap

        backward_heatmap = rearrange(backward_heatmap, "(B S) C H W -> B S C H W", B=batch_size)
        backward_heatmap = torch.cat(
            [torch.zeros_like(backward_heatmap[:, 0:1]).repeat(1,interval,1,1,1), backward_heatmap], dim=1
        ) # Pad first frame's backward heatmap

    if check_consistency:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=forward_flow.device, dtype=forward_flow.dtype),
            torch.arange(W, device=forward_flow.device, dtype=forward_flow.dtype),
            indexing="ij",
        ) # H, W
        init_pos = torch.stack((grid_x, grid_y), dim=0) # 2, H, W
        init_pos = repeat(init_pos, "Crd H W -> BS Crd H W", BS=batch_size*(num_views-interval))

        images_for_consist = images.clone()
        forward_consist_mask, forward_in_bound_mask = get_consist_mask(
            rearrange(images_for_consist[:, :-interval], "B S C H W -> (B S) C H W"),
            rearrange(images_for_consist[:, interval:], "B S C H W -> (B S) C H W"),
            rearrange(forward_flow[:, :-interval], "B S C H W -> (B S) C H W"),
            rearrange(backward_flow[:, interval:], "B S C H W -> (B S) C H W"),
            init_pos=init_pos,
            geo_thresh=geo_thresh,
            photo_thresh=photo_thresh,
        )
        forward_consist_mask = rearrange(forward_consist_mask, "(B S) C H W -> B S C H W", B=batch_size)
        forward_consist_mask = torch.cat(
            [forward_consist_mask, torch.ones_like(forward_consist_mask[:, 0:1].repeat(1,interval,1,1,1))], dim=1
        )
        forward_in_bound_mask = rearrange(forward_in_bound_mask, "(B S) C H W -> B S C H W", B=batch_size)
        forward_in_bound_mask = torch.cat(
            [forward_in_bound_mask, torch.ones_like(forward_in_bound_mask[:, 0:1].repeat(1,interval,1,1,1))], dim=1
        )
        backward_consist_mask, backward_in_bound_mask = get_consist_mask(
            rearrange(images_for_consist[:, interval:], "B S C H W -> (B S) C H W"),
            rearrange(images_for_consist[:, :-interval], "B S C H W -> (B S) C H W"),
            rearrange(backward_flow[:, interval:], "B S C H W -> (B S) C H W"),
            rearrange(forward_flow[:, :-interval], "B S C H W -> (B S) C H W"),
            init_pos=init_pos,
            geo_thresh=geo_thresh,
            photo_thresh=photo_thresh,
        )
        backward_consist_mask = rearrange(backward_consist_mask, "(B S) C H W -> B S C H W", B=batch_size)
        backward_consist_mask = torch.cat(
            [torch.ones_like(backward_consist_mask[:, 0:1].repeat(1,interval,1,1,1)), backward_consist_mask], dim=1
        )
        backward_in_bound_mask = rearrange(backward_in_bound_mask, "(B S) C H W -> B S C H W", B=batch_size)
        backward_in_bound_mask = torch.cat(
            [torch.ones_like(backward_in_bound_mask[:, 0:1].repeat(1,interval,1,1,1)), backward_in_bound_mask], dim=1
        )

        if return_heatmap and len(forward_info_raft) > 0 and len(backward_info_raft) > 0:
            return forward_flow, backward_flow, forward_heatmap, backward_heatmap, forward_consist_mask, backward_consist_mask, forward_in_bound_mask, backward_in_bound_mask
        else:
            return forward_flow, backward_flow, forward_consist_mask, backward_consist_mask, forward_in_bound_mask, backward_in_bound_mask

    if return_heatmap and len(forward_info_raft) > 0 and len(backward_info_raft) > 0:
        return forward_flow, backward_flow, forward_heatmap, backward_heatmap
    else:
        return forward_flow, backward_flow


def get_consist_mask(image1, image2, flow1_2, flow2_1, init_pos=None, geo_thresh=-1, photo_thresh=-1):
    """
    Check the consistency of between two images.
    Args:
        image1: The first image tensor of shape (B, C, H, W).
        image2: The second image tensor of shape (B, C, H, W).
        flow1_2: The optical flow tensor from image1 to image2 of shape (B, 2, H, W).
        flow2_1: The optical flow tensor from image2 to image1 of shape (B, 2, H, W).
        init_pos: Optional initial position tensor of shape (B, 2, H, W).
    Returns:
        A boolean mask tensor of shape (B, 1, H, W) indicating the consistency.
    """
    assert image1.shape == image2.shape, "Image1 and Image2 must have the same shape."
    B, C, H, W = image1.shape
    if init_pos is None:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow1_2.device, dtype=flow1_2.dtype),
            torch.arange(W, device=flow1_2.device, dtype=flow1_2.dtype),
            indexing="ij",
        ) # H, W
        init_pos = torch.stack((grid_x, grid_y), dim=0) # 2, H, W
        init_pos = repeat(init_pos, "Crd H W -> B Crd H W", B=B)
    else:
        assert init_pos.shape == (B, 2, H, W), "init_pos must have shape (B, 2, H, W)."

    warped_pos = init_pos + flow1_2
    # Normalize coordinates for grid_sample. Grid sample expects [-1, 1] range.
    # align_corners=True means -1 maps to 0 and 1 maps to Dim-1.
    norm_warped_x = 2.0 * warped_pos[:, 0] / (W - 1) - 1.0
    norm_warped_y = 2.0 * warped_pos[:, 1] / (H - 1) - 1.0
    normalized_warped_grid = torch.stack((norm_warped_x, norm_warped_y), dim=-1) # B, H, W, 2

    sampled_flow2_1 = F.grid_sample(
        flow2_1, normalized_warped_grid,
        mode="bilinear", padding_mode='border', align_corners=True
    )
    reconstructed_pos = warped_pos + sampled_flow2_1

    # 1. In-bound mask
    in_bound_mask = warped_pos[:, 0].ge(0) & warped_pos[:, 0].lt(W) & \
        warped_pos[:, 1].ge(0) & warped_pos[:, 1].lt(H)

    # 2. Geometric consistency mask
    if geo_thresh >= 0:
        geo_consist_mask = (reconstructed_pos - init_pos).norm(dim=1, p=2) < geo_thresh
    else:
        geo_consist_mask = torch.ones_like(in_bound_mask, dtype=torch.bool)

    # 3. Photometric consistency mask
    if photo_thresh >= 0:
        warped_color = F.grid_sample(
            image2, normalized_warped_grid,
            mode="bilinear", padding_mode='border', align_corners=True
        )
        photo_consist_mask = color_distance(image1, warped_color) < photo_thresh
    else:
        photo_consist_mask = torch.ones_like(in_bound_mask, dtype=torch.bool)

    consist_mask = in_bound_mask & geo_consist_mask & photo_consist_mask
    return consist_mask.unsqueeze(1), in_bound_mask.unsqueeze(1)  # Add channel dimension for consistency


def color_distance(color1, color2):
    """
    Compute the color distance between two sets of colors under LAB color space.
    Args:
        color1: First set of colors of shape (batch, channel, height, width).
        color2: Second set of colors of shape (batch, channel, height, width).
    Returns:
        Color distance of shape (batch, height, width).
    """
    assert color1.shape == color2.shape, "color1 and color2 must have the same shape."
    if color1.shape[1] == 3:
        color1 = rearrange(color1, "b c h w -> b h w c")
        color2 = rearrange(color2, "b c h w -> b h w c")
    color1_lab = rgb2lab(color1.cpu().numpy() / 255.0)
    color2_lab = rgb2lab(color2.cpu().numpy() / 255.0)
    return torch.tensor(
        deltaE_cie76(color1_lab, color2_lab),
        device=color1.device,
        dtype=color1.dtype
    )


@dataclass
class RAFTCfg:
    name: str = "spring-M"
    dataset: str = "spring"
    path: str = "Tartan-C-T-TSKH-spring540x960-M.pth"
    use_var: bool = True
    var_min: float = 0.0
    var_max: float = 10.0
    pretrain: str = "resnet34"
    initial_dim: int = 64
    block_dims: list[int] = (64, 128, 256)
    radius: int = 4
    dim: int = 128
    num_blocks: int = 2
    iters: int = 4
    image_size: list[int] = (540, 960)
    offload: bool = False
    geo_thresh: float = -1
    photo_thresh: float = -1


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
