"""Scene-relative scale diagnostics and regularization for 3D Gaussians."""

import math

import torch


def apply_3d_filter_to_scales(
    scales,
    filter_3d,
    apply_to_offsets=False,
):
    """Apply a scalar 3D filter to covariance scales, not offset scales.

    Scaffold anchors use the first three scale channels to parameterize offset
    displacement and the final three channels for Gaussian covariance. Adding
    the anti-aliasing filter to offset channels moves Gaussian centers away
    from their anchors, which creates floating geometry.
    """
    if scales.ndim != 2 or scales.shape[-1] < 3:
        raise ValueError("scales must have shape [N, C] with C >= 3")
    if filter_3d.ndim == 1:
        filter_3d = filter_3d.unsqueeze(-1)
    if filter_3d.ndim != 2 or filter_3d.shape[0] != scales.shape[0]:
        raise ValueError("filter_3d must have shape [N] or [N, 1]")

    if apply_to_offsets or scales.shape[-1] == 3:
        return torch.sqrt(torch.square(scales) + torch.square(filter_3d))

    offset_scales = scales[:, :3]
    covariance_scales = torch.sqrt(
        torch.square(scales[:, 3:]) + torch.square(filter_3d)
    )
    return torch.cat((offset_scales, covariance_scales), dim=-1)


def gaussian_scale_regularization(
    scaling,
    voxel_size,
    max_axis_voxel_ratio=16.0,
    max_anisotropy_ratio=10.0,
    eps=1e-8,
):
    """Penalize oversized or needle-like rendered Gaussian scales.

    The returned losses are dimensionless so that their weights remain useful
    when the scene voxel size changes. Both penalties are one-sided: scales
    below the configured limits receive no penalty.
    """
    if scaling.ndim != 2 or scaling.shape[-1] != 3:
        raise ValueError("scaling must have shape [N, 3]")
    if scaling.shape[0] == 0:
        zero = scaling.sum() * 0.0
        return zero, zero

    safe_scaling = scaling.abs().clamp_min(eps)
    axis_max = safe_scaling.max(dim=1).values
    axis_min = safe_scaling.min(dim=1).values
    zero = scaling.sum() * 0.0

    if max_axis_voxel_ratio > 0 and voxel_size > 0:
        max_axis_limit = float(voxel_size) * float(max_axis_voxel_ratio)
        max_axis_loss = torch.relu(axis_max / max_axis_limit - 1.0).square().mean()
    else:
        max_axis_loss = zero

    if max_anisotropy_ratio > 1:
        anisotropy = axis_max / axis_min
        log_limit = math.log(float(max_anisotropy_ratio))
        anisotropy_loss = torch.relu(torch.log(anisotropy) - log_limit).square().mean()
    else:
        anisotropy_loss = zero

    return max_axis_loss, anisotropy_loss
