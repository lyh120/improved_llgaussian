"""Apply the sampling footprint to emitted three-dimensional Gaussians."""

import torch


def filter_gaussian_covariance(scales: torch.Tensor, opacity: torch.Tensor,
                               footprint: torch.Tensor, max_footprint_ratio: float = 0.0):
    """Add isotropic sampling variance and preserve integrated Gaussian opacity.

    The footprint has world-space standard-deviation units. It is applied after
    covariance decoding so the decoder cannot shrink below the sampling floor.
    """
    if max_footprint_ratio > 0:
        # Define a bounded emitted Gaussian before applying the antialiasing
        # floor. This is a model support limit, not post-render image cleanup.
        scales = torch.minimum(scales, footprint * max_footprint_ratio)
    variance = scales.square()
    filtered_variance = variance + footprint.square()
    tiny = torch.finfo(scales.dtype).tiny
    # A covariance determinant has three spatial axes, not the six anchor scales.
    filtered_scales = filtered_variance.clamp_min(tiny).sqrt()
    # Multiplying squared ratios before sqrt can underflow to zero, whose
    # backward produces inf * 0. The equivalent per-axis ratios stay stable.
    coefficient = (scales.abs() / filtered_scales).prod(dim=-1, keepdim=True)
    return filtered_scales, opacity * coefficient
