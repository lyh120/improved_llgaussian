"""Relative-depth priors and robust local correlation supervision."""

import math

import torch
import torch.nn.functional as F


def _finite_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value)).sum() * 0.0


def normalize_relative_depth(depth: torch.Tensor, valid_mask=None) -> torch.Tensor:
    """Min-max normalize valid pixels without allowing invalid rays to set scale."""
    valid = torch.isfinite(depth)
    if valid_mask is not None:
        valid = valid & valid_mask.bool()
    safe = torch.where(valid, depth, torch.zeros_like(depth))
    if not valid.any():
        return safe * 0.0
    samples = depth[valid]
    span = samples.max() - samples.min()
    normalized = (safe - samples.min()) / span.clamp_min(torch.finfo(depth.dtype).eps)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def pearson_depth_loss(depth_src, depth_target, eps=1e-6, valid_mask=None):
    """Compute 1-r on finite, nondegenerate depth pairs, including negative r."""
    target = depth_target.detach().to(depth_src)
    valid = torch.isfinite(depth_src) & torch.isfinite(target)
    if valid_mask is not None:
        valid &= valid_mask.bool()
    if valid.sum() < 2:
        return _finite_zero(depth_src)
    src = depth_src[valid]
    dst = target[valid]
    src = src - src.mean()
    dst = dst - dst.mean()
    src_var, dst_var = src.square().mean(), dst.square().mean()
    if src_var.detach() <= torch.finfo(src.dtype).eps or dst_var <= torch.finfo(dst.dtype).eps:
        return _finite_zero(depth_src)
    correlation = (src * dst).mean() / torch.sqrt((src_var + eps) * (dst_var + eps))
    return 1.0 - correlation.clamp(-1.0, 1.0)


def local_depth_similarity(depth_src, depth_target, box_p=128, p_corr=0.5, valid_mask=None):
    """Sample valid windows with inclusive boundaries and a safe small-image case."""
    if depth_src.ndim != 2 or depth_src.shape != depth_target.shape:
        raise ValueError("Depth correlation requires matching [H,W] tensors")
    if box_p < 1 or not math.isfinite(p_corr) or p_corr <= 0:
        raise ValueError("Depth window and sampling fraction must be positive")
    height, width = depth_src.shape
    if height == 0 or width == 0:
        return _finite_zero(depth_src)
    patch_h, patch_w = min(box_p, height), min(box_p, width)
    count = max(1, int(p_corr * max(1, height // box_p) * max(1, width // box_p)))
    rows = torch.randint(height - patch_h + 1, (count,), device=depth_src.device)
    cols = torch.randint(width - patch_w + 1, (count,), device=depth_src.device)
    total = _finite_zero(depth_src)
    for row, col in zip(rows.tolist(), cols.tolist()):
        region = (slice(row, row + patch_h), slice(col, col + patch_w))
        total = total + pearson_depth_loss(
            depth_src[region], depth_target[region],
            valid_mask=None if valid_mask is None else valid_mask[region],
        )
    return total / count


def relative_depth_supervision(depth, prior, coverage, patch_size=128, sample_fraction=0.5):
    """Keep LL-Gaussian's inverted accumulated-depth/local Pearson objective."""
    valid = torch.isfinite(depth) & torch.isfinite(prior)
    if coverage is not None:
        valid &= torch.isfinite(coverage) & (coverage > torch.finfo(depth.dtype).eps)
    source = 1.0 - normalize_relative_depth(depth, valid)
    return local_depth_similarity(
        source.squeeze(0), prior.detach().squeeze(0), patch_size, sample_fraction,
        valid_mask=valid.squeeze(0),
    )


@torch.no_grad()
def generate_depth_priors(model, cameras, source_path, images="images") -> dict:
    """Infer frozen relative-depth targets for the supplied training views only."""
    import cv2

    from utils.prior_utils import find_source_image

    model.eval().requires_grad_(False)
    result = {}
    for camera in cameras:
        path = find_source_image(source_path, images, camera.image_name)
        image = cv2.imread(str(path))  # Depth Anything's infer_image expects BGR.
        if image is None:
            raise ValueError(f"Could not read depth-prior input: {path}")
        depth = torch.as_tensor(model.infer_image(image), dtype=torch.float32).detach()
        if depth.ndim == 2:
            depth = depth[None]
        elif depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 3 or depth.shape[0] != 1:
            raise ValueError(f"Depth prior for {camera.image_name} must have shape [1,H,W]")
        finite = torch.isfinite(depth)
        size = (camera.image_height, camera.image_width)
        if depth.shape[-2:] != size:
            depth = F.interpolate(torch.where(finite, depth, 0.0)[None], size=size,
                                  mode="bilinear", align_corners=False)[0]
            finite = F.interpolate(finite.float()[None], size=size, mode="nearest")[0].bool()
        depth = depth.to(camera.original_image.device)
        finite = finite.to(depth.device)
        normalized = normalize_relative_depth(depth, finite)
        result[camera.uid] = normalized.masked_fill(~finite, float("nan")).detach()
    return result
