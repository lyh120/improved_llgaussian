"""Utilities for detecting and limiting needle-like Gaussian scales."""

import math

import torch
import torch.nn.functional as F


def needle_loss(scales: torch.Tensor, ratio_threshold: float = 5.0) -> torch.Tensor:
    """Penalize a single scale axis that is much larger than the other two."""
    if scales.numel() == 0:
        return scales.sum() * 0.0
    if ratio_threshold <= 1.0:
        raise ValueError("ratio_threshold must be greater than 1")
    sorted_scales = torch.sort(scales.clamp_min(1e-8), dim=-1).values
    log_ratio = torch.log(sorted_scales[:, 2]) - torch.log(sorted_scales[:, 1])
    return F.relu(log_ratio - math.log(ratio_threshold)).square().mean()


def oblate_loss(scales: torch.Tensor, ratio_threshold: float = 20.0) -> torch.Tensor:
    """Penalize near-zero thickness in disk-like Gaussian scales."""
    if scales.numel() == 0:
        return scales.sum() * 0.0
    if ratio_threshold <= 1.0:
        raise ValueError("ratio_threshold must be greater than 1")
    sorted_scales = torch.sort(scales.clamp_min(1e-8), dim=-1).values
    log_ratio = torch.log(sorted_scales[:, 1]) - torch.log(sorted_scales[:, 0])
    return F.relu(log_ratio - math.log(ratio_threshold)).square().mean()


def clamp_needle_scales(
    scales: torch.Tensor,
    ratio_threshold: float = 5.0,
    oblate_ratio_threshold: float | None = None,
    max_scale: float = 0.0,
) -> torch.Tensor:
    """Limit prolate, oblate, and optional absolute scale outliers."""
    if scales.numel() == 0:
        return scales
    if ratio_threshold <= 1.0:
        raise ValueError("ratio_threshold must be greater than 1")
    sorted_scales, sorted_indices = torch.sort(scales.clamp_min(1e-8), dim=-1)
    shortest = sorted_scales[:, 0]
    middle = sorted_scales[:, 1]
    longest = torch.minimum(sorted_scales[:, 2], middle * ratio_threshold)
    if oblate_ratio_threshold is not None:
        if oblate_ratio_threshold <= 1.0:
            raise ValueError("oblate_ratio_threshold must be greater than 1")
        shortest = torch.maximum(shortest, middle / oblate_ratio_threshold)
    sorted_scales = torch.stack((shortest, middle, longest), dim=-1)
    if max_scale > 0.0:
        sorted_scales = sorted_scales.clamp_max(max_scale)
    return torch.zeros_like(scales).scatter(1, sorted_indices, sorted_scales)


@torch.no_grad()
def scaling_diagnostics(
    scales: torch.Tensor,
    radii: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return scalar diagnostics for effective rasterized Gaussian scales."""
    if scales.numel() == 0:
        zero = scales.new_tensor(0.0)
        return {
            "needle_loss_r5": zero,
            "needle_ratio_p95": zero,
            "needle_ratio_p99": zero,
            "needle_ratio_max": zero,
            "needle_ratio_gt5": zero,
            "needle_ratio_gt10": zero,
            "needle_ratio_gt20": zero,
            "oblate_ratio_p95": zero,
            "oblate_ratio_p99": zero,
            "oblate_ratio_max": zero,
            "oblate_ratio_gt10": zero,
            "oblate_ratio_gt20": zero,
            "oblate_ratio_gt50": zero,
            "condition_ratio_p95": zero,
            "condition_ratio_p99": zero,
            "condition_ratio_max": zero,
            "scale_max_p99": zero,
            "scale_max": zero,
            "scale_mid_p99": zero,
            "scale_min_p01": zero,
            "radii_p99": zero,
            "radii_max": zero,
            "radii_gt64": zero,
            "radii_gt128": zero,
            "radii_gt256": zero,
        }

    sorted_scales = torch.sort(scales.clamp_min(1e-8), dim=-1).values
    ratio = sorted_scales[:, 2] / sorted_scales[:, 1].clamp_min(1e-8)
    oblate_ratio = sorted_scales[:, 1] / sorted_scales[:, 0].clamp_min(1e-8)
    condition_ratio = sorted_scales[:, 2] / sorted_scales[:, 0].clamp_min(1e-8)
    longest = sorted_scales[:, 2]
    stats = {
        "needle_loss_r5": F.relu(torch.log(ratio) - math.log(5.0)).square().mean(),
        "needle_ratio_p95": torch.quantile(ratio, 0.95),
        "needle_ratio_p99": torch.quantile(ratio, 0.99),
        "needle_ratio_max": ratio.max(),
        "needle_ratio_gt5": (ratio > 5.0).float().mean(),
        "needle_ratio_gt10": (ratio > 10.0).float().mean(),
        "needle_ratio_gt20": (ratio > 20.0).float().mean(),
        "oblate_ratio_p95": torch.quantile(oblate_ratio, 0.95),
        "oblate_ratio_p99": torch.quantile(oblate_ratio, 0.99),
        "oblate_ratio_max": oblate_ratio.max(),
        "oblate_ratio_gt10": (oblate_ratio > 10.0).float().mean(),
        "oblate_ratio_gt20": (oblate_ratio > 20.0).float().mean(),
        "oblate_ratio_gt50": (oblate_ratio > 50.0).float().mean(),
        "condition_ratio_p95": torch.quantile(condition_ratio, 0.95),
        "condition_ratio_p99": torch.quantile(condition_ratio, 0.99),
        "condition_ratio_max": condition_ratio.max(),
        "scale_max_p99": torch.quantile(longest, 0.99),
        "scale_max": longest.max(),
        "scale_mid_p99": torch.quantile(sorted_scales[:, 1], 0.99),
        "scale_min_p01": torch.quantile(sorted_scales[:, 0], 0.01),
    }
    visible_radii = None if radii is None else radii[radii > 0].float()
    if visible_radii is None or visible_radii.numel() == 0:
        stats["radii_p99"] = scales.new_tensor(0.0)
        stats["radii_max"] = scales.new_tensor(0.0)
        stats["radii_gt64"] = scales.new_tensor(0.0)
        stats["radii_gt128"] = scales.new_tensor(0.0)
        stats["radii_gt256"] = scales.new_tensor(0.0)
    else:
        stats["radii_p99"] = torch.quantile(visible_radii, 0.99)
        stats["radii_max"] = visible_radii.max()
        stats["radii_gt64"] = (visible_radii > 64.0).float().mean()
        stats["radii_gt128"] = (visible_radii > 128.0).float().mean()
        stats["radii_gt256"] = (visible_radii > 256.0).float().mean()
    return stats
