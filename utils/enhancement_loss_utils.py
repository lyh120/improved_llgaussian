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


def L_Enhancement_Target_Edge(enhanced_image, target_image, coverage=None):
    """Match coherent CIDNet edges while ignoring most single-pixel low-light noise.

    The target is smoothed before differentiation. Only its stronger edges
    supervise the prediction, so flat noisy regions do not ask the 3D model
    to reproduce frame-specific grain.
    """
    kernel_1d = enhanced_image.new_tensor(
        [0.036633, 0.111281, 0.216745, 0.270682, 0.216745, 0.111281, 0.036633]
    )
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 7, 7)
    sobel_x = enhanced_image.new_tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)

    def gradients(image):
        gray = (0.2126 * image[0:1] + 0.7152 * image[1:2]
                + 0.0722 * image[2:3]).unsqueeze(0)
        smooth = F.conv2d(F.pad(gray, (3, 3, 3, 3), mode="reflect"), kernel)
        smooth = F.pad(smooth, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(smooth, sobel_x), F.conv2d(smooth, sobel_y)

    target_dx, target_dy = gradients(target_image.detach())
    pred_dx, pred_dy = gradients(enhanced_image)
    target_magnitude = torch.sqrt(target_dx.square() + target_dy.square() + 1e-8)
    threshold = torch.maximum(
        target_magnitude.new_tensor(0.015),
        target_magnitude.mean() + target_magnitude.std(unbiased=False),
    )
    mask = (target_magnitude > threshold).detach().to(enhanced_image.dtype)
    mask = mask * (_coverage_2d(coverage, target_magnitude.squeeze(0)) >= 0.95).unsqueeze(0).detach()
    error = torch.abs(pred_dx - target_dx) + torch.abs(pred_dy - target_dy)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def L_Reflectance_Target_Detail(reflectance_image, target_image, coverage=None):
    """Transfer local structure from a 2D target into R without changing geometry.

    Local log luminance removes most smooth exposure changes. The target is
    gently blurred before comparison, and flat or saturated regions carry
    little weight, so isolated sensor noise does not become material texture.
    The caller supplies a geometry-detached reflectance raster.
    """
    def luminance(image):
        return (0.2126 * image[0:1] + 0.7152 * image[1:2]
                + 0.0722 * image[2:3]).unsqueeze(0)

    target_gray = luminance(target_image.detach())
    ref_gray = luminance(reflectance_image)
    target_gray = F.avg_pool2d(
        F.pad(target_gray, (1, 1, 1, 1), mode="reflect"), 3, stride=1
    )

    def local_log_detail(gray):
        log_gray = torch.log(gray.clamp_min(0.02))
        smooth = F.avg_pool2d(
            F.pad(log_gray, (4, 4, 4, 4), mode="reflect"), 9, stride=1
        )
        return log_gray - smooth

    target_detail = local_log_detail(target_gray).detach()
    ref_detail = local_log_detail(ref_gray)
    local_contrast = target_detail.abs()
    mask = ((local_contrast > 0.035) & (target_gray < 0.97)).to(ref_detail.dtype)
    mask = mask * (_coverage_2d(coverage, target_gray.squeeze(0)) >= 0.95).unsqueeze(0).detach()
    error = F.smooth_l1_loss(ref_detail, target_detail, reduction="none", beta=0.05)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def L_Reflectance_Target_Chroma(reflectance_image, target_image, coverage=None):
    """Keep material color in R while ignoring overall exposure differences."""
    target = target_image.detach()
    target_mean = target.mean(dim=0, keepdim=True)
    ref_mean = reflectance_image.mean(dim=0, keepdim=True)
    target_chroma = target / target_mean.clamp_min(0.05)
    ref_chroma = reflectance_image / ref_mean.clamp_min(0.05)
    mask = ((target_mean > 0.08) & (target_mean < 0.95)).to(ref_chroma.dtype)
    mask = mask * (_coverage_2d(coverage, ref_mean) >= 0.95).detach()
    error = F.smooth_l1_loss(ref_chroma, target_chroma, reduction="none", beta=0.1)
    return (error * mask).sum() / (3.0 * mask.sum().clamp_min(1.0))


def L_Illumination_Chroma_Consistency(illumination_image, coverage=None):
    """Make enhanced lighting spatially smooth in color, with room for a global tint."""
    mean = illumination_image.mean(dim=0, keepdim=True)
    chroma = illumination_image / mean.clamp_min(0.05)
    mask = ((mean > 0.08) & (_coverage_2d(coverage, mean) >= 0.95)).to(chroma.dtype).detach()
    denom = mask.sum().clamp_min(1.0)
    global_chroma = (chroma * mask).sum(dim=(1, 2), keepdim=True) / denom
    spatial = F.smooth_l1_loss(chroma, global_chroma.expand_as(chroma), reduction="none", beta=0.1)
    spatial = (spatial * mask).sum() / (3.0 * denom)
    neutral = (global_chroma - 1.0).square().mean()
    return spatial + 0.2 * neutral
