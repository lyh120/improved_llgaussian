"""Normalized objectives for the explicit v2 reflectance/illumination model."""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F

from utils.model_format import NUMERICAL_EPS


# Canonical SSIM constants from Wang et al. for unit-range images. They are
# part of the metric definition, not trainable loss weights.
SSIM_K1 = 0.01
SSIM_K2 = 0.03
SSIM_WINDOW_SIGMA = 1.5


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(prediction - target).mean()


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    values = torch.tensor(
        [
            exp(-((position - window_size // 2) ** 2) / (2 * sigma**2))
            for position in range(window_size)
        ]
    )
    return values / values.sum()


def _ssim_window(
    window_size: int,
    channels: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    one_dimensional = _gaussian(window_size, SSIM_WINDOW_SIGMA).unsqueeze(1)
    two_dimensional = one_dimensional.mm(one_dimensional.t()).float()[None, None]
    return (
        two_dimensional.expand(channels, 1, window_size, window_size)
        .contiguous()
        .to(reference)
    )


def ssim(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """Canonical Scaffold-GS SSIM for CHW or BCHW unit-range images."""
    if image_a.ndim == 3:
        image_a = image_a.unsqueeze(0)
        image_b = image_b.unsqueeze(0)
    channels = image_a.shape[1]
    window = _ssim_window(window_size, channels, image_a)
    mean_a = F.conv2d(image_a, window, padding=window_size // 2, groups=channels)
    mean_b = F.conv2d(image_b, window, padding=window_size // 2, groups=channels)
    mean_a_sq = mean_a.square()
    mean_b_sq = mean_b.square()
    mean_ab = mean_a * mean_b
    variance_a = (
        F.conv2d(image_a.square(), window, padding=window_size // 2, groups=channels)
        - mean_a_sq
    )
    variance_b = (
        F.conv2d(image_b.square(), window, padding=window_size // 2, groups=channels)
        - mean_b_sq
    )
    covariance = (
        F.conv2d(image_a * image_b, window, padding=window_size // 2, groups=channels)
        - mean_ab
    )
    c1 = SSIM_K1**2
    c2 = SSIM_K2**2
    value = ((2 * mean_ab + c1) * (2 * covariance + c2)) / (
        (mean_a_sq + mean_b_sq + c1) * (variance_a + variance_b + c2)
    )
    return value.mean() if size_average else value.mean(dim=(1, 2, 3))


def photo_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float,
    ssim_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaffold photometric objective with its sole public mixture weight."""
    similarity = ssim(prediction, target) if ssim_value is None else ssim_value
    return (1.0 - lambda_dssim) * l1_loss(prediction, target) + lambda_dssim * (
        1.0 - similarity
    )


def _as_bchw(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(0) if image.ndim == 3 else image


def retinex_targets(lowlight_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached Max-RGB illumination and finite Retinex reflectance."""
    was_chw = lowlight_target.ndim == 3
    target = _as_bchw(lowlight_target.detach())
    max_rgb = target.amax(dim=1, keepdim=True)
    reflectance = target / max_rgb.clamp_min(NUMERICAL_EPS)
    if was_chw:
        return max_rgb.squeeze(0), reflectance.squeeze(0)
    return max_rgb, reflectance


def coverage_masked_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Replace uncovered prediction pixels by fixed targets without gradients."""
    was_chw = prediction.ndim == 3
    prediction_bchw = _as_bchw(prediction)
    target_bchw = _as_bchw(target.detach()).expand_as(prediction_bchw)
    coverage_bchw = (
        _as_bchw(coverage.detach())
        .mean(dim=1, keepdim=True)
        .clamp(0.0, 1.0)
        .expand_as(prediction_bchw)
    )
    masked = coverage_bchw * prediction_bchw + (1.0 - coverage_bchw) * target_bchw
    return masked.squeeze(0) if was_chw else masked


def edge_aware_tv_weights(
    max_rgb_target: torch.Tensor,
    coverage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached edge-aware horizontal/vertical coverage weights."""
    target = _as_bchw(max_rgb_target.detach()).mean(dim=1, keepdim=True)
    covered = _as_bchw(coverage.detach()).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    horizontal_coverage = torch.minimum(covered[..., :, :-1], covered[..., :, 1:])
    vertical_coverage = torch.minimum(covered[..., :-1, :], covered[..., 1:, :])
    horizontal_gradient = torch.abs(target[..., :, :-1] - target[..., :, 1:])
    vertical_gradient = torch.abs(target[..., :-1, :] - target[..., 1:, :])
    return (
        horizontal_coverage / (horizontal_gradient + NUMERICAL_EPS),
        vertical_coverage / (vertical_gradient + NUMERICAL_EPS),
    )


def edge_aware_illumination_tv_loss(
    illumination: torch.Tensor,
    max_rgb_target: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Normalized edge-aware TV in [0, 1] for unit-range illumination."""
    illumination_gray = _as_bchw(illumination).mean(dim=1, keepdim=True)
    horizontal_weight, vertical_weight = edge_aware_tv_weights(
        max_rgb_target, coverage
    )
    horizontal_gradient = torch.abs(
        illumination_gray[..., :, :-1] - illumination_gray[..., :, 1:]
    )
    vertical_gradient = torch.abs(
        illumination_gray[..., :-1, :] - illumination_gray[..., 1:, :]
    )
    horizontal = (horizontal_weight * horizontal_gradient).sum() / (
        horizontal_weight.sum() + NUMERICAL_EPS
    )
    vertical = (vertical_weight * vertical_gradient).sum() / (
        vertical_weight.sum() + NUMERICAL_EPS
    )
    return (horizontal + vertical) / 2.0


def _unit_interval(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = values[valid]
    minimum = selected.min()
    maximum = selected.max()
    return (values - minimum) / (maximum - minimum).clamp_min(NUMERICAL_EPS)


def inverse_depth_pearson_loss(
    rendered_depth: torch.Tensor,
    disparity_prior: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Pearson distance between rendered inverse depth and cached disparity."""
    depth = rendered_depth.squeeze()
    prior = disparity_prior.detach().squeeze()
    weight = coverage.detach().mean(dim=0).squeeze().clamp(0.0, 1.0)
    valid = (
        torch.isfinite(depth)
        & torch.isfinite(prior)
        & (depth > NUMERICAL_EPS)
        & (weight > NUMERICAL_EPS)
    )
    if int(valid.sum()) < 2:
        return rendered_depth.sum() * 0.0
    inverse_depth = _unit_interval(depth.reciprocal(), valid)
    normalized_prior = _unit_interval(prior, valid)
    sample_weight = weight[valid]
    sample_weight = sample_weight / sample_weight.sum().clamp_min(NUMERICAL_EPS)
    source = inverse_depth[valid]
    target = normalized_prior[valid]
    source = source - (sample_weight * source).sum()
    target = target - (sample_weight * target).sum()
    covariance = (sample_weight * source * target).sum()
    source_variance = (sample_weight * source.square()).sum()
    target_variance = (sample_weight * target.square()).sum()
    correlation = covariance / torch.sqrt(
        source_variance * target_variance + NUMERICAL_EPS
    )
    return ((1.0 - correlation.clamp(-1.0, 1.0)) / 2.0).clamp(0.0, 1.0)


def scaling_loss(scaling: torch.Tensor) -> torch.Tensor:
    return scaling.prod(dim=1).mean()
