"""Pure PyTorch losses for structure-preserving exposure enhancement."""

import torch
import torch.nn.functional as F


def _coverage_2d(coverage, reference):
    """Convert renderer coverage to a single-channel spatial weight."""
    if coverage is None:
        return torch.ones_like(reference)
    if coverage.ndim == 3:
        coverage = coverage.mean(dim=0, keepdim=True)
    elif coverage.ndim == 2:
        coverage = coverage.unsqueeze(0)
    if coverage.shape[-2:] != reference.shape[-2:]:
        coverage = F.interpolate(
            coverage.unsqueeze(0),
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return coverage.to(device=reference.device, dtype=reference.dtype)


def L_Enhancement_Gain_Smooth(
    illumination_enhanced,
    illumination_base,
    coverage=None,
    eps=1e-3,
):
    """Smooth exposure gain without suppressing structure in base illumination."""
    log_gain = (
        torch.log(illumination_enhanced.clamp_min(eps))
        - torch.log(illumination_base.detach().clamp_min(eps))
    )
    gain_gray = log_gain.mean(dim=0, keepdim=True)
    weight = _coverage_2d(coverage, gain_gray).detach()

    grad_y = torch.abs(gain_gray[:, 1:, :] - gain_gray[:, :-1, :])
    grad_x = torch.abs(gain_gray[:, :, 1:] - gain_gray[:, :, :-1])
    weight_y = torch.minimum(weight[:, 1:, :], weight[:, :-1, :])
    weight_x = torch.minimum(weight[:, :, 1:], weight[:, :, :-1])
    loss_y = (grad_y * weight_y).sum() / weight_y.sum().clamp_min(1.0)
    loss_x = (grad_x * weight_x).sum() / weight_x.sum().clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def _log_luminance_sobel(image, eps=1e-3):
    luminance = (
        0.2126 * image[0:1]
        + 0.7152 * image[1:2]
        + 0.0722 * image[2:3]
    )
    log_luminance = torch.log(luminance.clamp_min(eps)).unsqueeze(0)
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    log_luminance = F.pad(log_luminance, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(log_luminance, kernel_x)
    grad_y = F.conv2d(log_luminance, kernel_y)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8).squeeze(0)


def L_Enhancement_Edge_Preserve(
    enhanced_image,
    base_image,
    enhance_ratio,
    coverage=None,
    edge_threshold=0.02,
    target_ratio=0.9,
    saturation_threshold=0.95,
):
    """Prevent exposure enhancement from erasing edges present in reconstruction."""
    teacher = torch.clamp(base_image.detach() * float(enhance_ratio), 0.0, 1.0)
    teacher_grad = _log_luminance_sobel(teacher).detach()
    enhanced_grad = _log_luminance_sobel(enhanced_image)
    teacher_luminance = (
        0.2126 * teacher[0:1]
        + 0.7152 * teacher[1:2]
        + 0.0722 * teacher[2:3]
    )
    valid_coverage = (_coverage_2d(coverage, teacher_grad) >= 0.95).detach()
    valid_mask = (
        (teacher_grad >= edge_threshold)
        & (teacher_luminance < saturation_threshold)
        & valid_coverage
    ).to(enhanced_grad.dtype)
    missing_edge = F.relu(target_ratio * teacher_grad - enhanced_grad)
    return (missing_edge * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
