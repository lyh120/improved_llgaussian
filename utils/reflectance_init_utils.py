"""Exposure-normalized initialization using the project's camera convention."""

import torch


@torch.no_grad()
def estimate_initial_log_reflectance(anchors: torch.Tensor, cameras) -> torch.Tensor:
    """Average observed log chromaticity, weighted by observed signal strength.

    This is an initialization gauge, not a calibrated albedo measurement. Points
    outside the frustum or without signal retain neutral log-reflectance. Camera
    matrices are transposed for CUDA and multiply row vectors in Python.
    """
    totals = torch.zeros_like(anchors)
    weights = torch.zeros_like(anchors[:, :1])
    if cameras is None:
        return totals
    eps = torch.finfo(anchors.dtype).eps
    homogeneous = torch.cat([anchors, torch.ones_like(anchors[:, :1])], dim=-1)
    for camera in cameras:
        image = camera.original_image.to(device=anchors.device, dtype=anchors.dtype)
        height, width = image.shape[-2:]
        clip = homogeneous @ camera.full_proj_transform.to(anchors)
        ndc = clip[:, :3] / clip[:, 3:].clamp_min(eps)
        valid = (
            (clip[:, 3] > 0)
            & (ndc[:, :2].abs() <= 1).all(dim=-1)
            & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
            & torch.isfinite(ndc).all(dim=-1)
        )
        indices = valid.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        # CUDA ndc2Pix: ((ndc + 1) * size - 1) / 2, with no vertical flip.
        pixels = ((ndc[indices, :2] + 1) * anchors.new_tensor([width, height]) - 1) / 2
        px = pixels[:, 0].round().long().clamp(0, width - 1)
        py = pixels[:, 1].round().long().clamp(0, height - 1)
        samples = image[:, py, px].T
        signal = samples.amax(dim=-1, keepdim=True)
        chromaticity = (samples / signal.clamp_min(eps)).clamp(eps, 1)
        totals[indices] += chromaticity.log() * signal
        weights[indices] += signal
    return totals / weights.clamp_min(eps)
