"""Confidence weighted appearance constraints with explicit gradient permissions."""

import math

import torch
import torch.nn.functional as F


STAGE_WEIGHTS = {
    "sharp": dict(edge=1e-3, contrast=5e-4, color=0., b0=0., detail=0., decoder=0., sg_tail=0.),
    "color": dict(edge=1e-3, contrast=5e-4, color=1e-3, b0=0., detail=0., decoder=0., sg_tail=0.),
    "stable": dict(edge=1e-3, contrast=5e-4, color=1e-3, b0=1e-5, detail=1e-6, decoder=2e-5, sg_tail=1e-4),
    "asg_paper": dict(edge=1e-3, contrast=5e-4, flat=1e-4, color=0., b0=1e-5,
                      detail=1e-6, decoder=2e-5, asg_energy=1e-4,
                      asg_sharpness=5e-5, asg_anisotropy=1e-5),
}

ILLUMINATION_TERMS = {"sg_tail", "asg_energy", "asg_sharpness", "asg_anisotropy"}


def compat_ramp(iteration: int) -> float:
    """Linearly activate the new objectives between steps 1000 and 2000."""
    return min(1., max(0., (iteration - 1000) / 1000))


def binomial_blur(image: torch.Tensor) -> torch.Tensor:
    """Blur CHW images without introducing black padding at their boundaries."""
    vector = image.new_tensor([1., 2., 1.])
    kernel = (vector[:, None] * vector[None, :] / 16).expand(image.shape[0], 1, 3, 3)
    return F.conv2d(F.pad(image[None], (1, 1, 1, 1), mode="replicate"), kernel,
                    groups=image.shape[0])[0]


def gray(image: torch.Tensor) -> torch.Tensor:
    return (image * image.new_tensor([.299, .587, .114])[:, None, None]).sum(0, keepdim=True)


def sobel(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    kernels = torch.stack([kernel, kernel.T])[:, None]
    return F.conv2d(F.pad(image[None], (1, 1, 1, 1), mode="replicate"), kernels)[0]


def local_std(image: torch.Tensor) -> torch.Tensor:
    # Center before second moments to avoid cancellation on constant log images.
    centered = image - image.mean().detach()
    padded = F.pad(centered[None], (2, 2, 2, 2), mode="replicate")
    mean = F.avg_pool2d(padded, 5, stride=1)
    variance = (F.avg_pool2d(padded.square(), 5, stride=1) - mean.square()).clamp_min(0)
    return (variance + 1e-10).sqrt()[0]


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep empty masks finite and independent of the number of valid pixels."""
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def structure_band(response: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.relu(.8 * target - response) + F.relu(response - 1.25 * target)


@torch.no_grad()
def prepare_compat_target(target: torch.Tensor) -> dict:
    """Precompute detached structure and signal confidence on train images only."""
    clean = binomial_blur(target)
    second = binomial_blur(clean)
    y = gray(clean)
    log_y = y.clamp_min(1 / 255).log()
    edges = sobel(log_y)
    edges2 = sobel(gray(second).clamp_min(1 / 255).log())
    magnitude = edges.square().sum(0, keepdim=True).clamp_min(1e-12).sqrt()
    magnitude2 = edges2.square().sum(0, keepdim=True).clamp_min(1e-12).sqrt()
    residual = gray(target) - y
    sigma = (residual - residual.median()).abs().median() / .67448975
    log_noise = sigma / y.clamp_min(1 / 255)
    confidence = ((clean.amax(0, keepdim=True) - 4 / 255) / (8 / 255)).clamp(0, 1)
    confidence *= (target.amax(0, keepdim=True) < 250 / 255)
    # Exclude neighborhoods touching saturation and the two-pixel image border.
    saturated = F.max_pool2d((target.amax(0, keepdim=True) >= 250 / 255).float()[None],
                            5, stride=1, padding=2)[0]
    confidence *= 1 - saturated
    confidence[:, :2] = confidence[:, -2:] = 0
    confidence[:, :, :2] = confidence[:, :, -2:] = 0
    direction_agreement = (edges * edges2).sum(0, keepdim=True) / (magnitude * magnitude2).clamp_min(1e-8)
    contrast = local_std(log_y)
    return dict(clean=clean, edge=magnitude, contrast=contrast, confidence=confidence,
                edge_confidence=confidence * (magnitude > (3 * log_noise).clamp_min(1e-4)) * (direction_agreement >= .8),
                contrast_confidence=confidence * (contrast > (3 * log_noise).clamp_min(1e-4)), sigma=sigma)


def image_compat_terms(reflectance: torch.Tensor, prepared: dict, coverage: torch.Tensor) -> dict:
    """Evaluate auxiliary image losses; callers must freeze all raster geometry."""
    valid = (coverage.detach().amin(0, keepdim=True) >= .95).to(reflectance.dtype)
    clean = binomial_blur(reflectance)
    log_y = gray(clean).clamp_min(1e-3).log()
    edge = sobel(log_y).square().sum(0, keepdim=True).clamp_min(1e-12).sqrt()
    contrast = local_std(log_y)
    edge_weight = valid * prepared["edge_confidence"]
    contrast_weight = valid * prepared["contrast_confidence"]
    color_weight = valid * prepared["confidence"]
    chroma = clean / clean.sum(0, keepdim=True).clamp_min(1e-6)
    target_chroma = prepared["clean"] / prepared["clean"].sum(0, keepdim=True).clamp_min(1e-6)
    return dict(edge=weighted_mean(structure_band(edge, prepared["edge"]), edge_weight),
                contrast=weighted_mean(structure_band(contrast, prepared["contrast"]), contrast_weight),
                color=weighted_mean((chroma - target_chroma).abs().mean(0, keepdim=True), color_weight),
                signal_mask_fraction=(prepared["confidence"] > 0).float().mean(),
                edge_mask_fraction=(edge_weight > 0).float().mean(),
                contrast_mask_fraction=(contrast_weight > 0).float().mean(),
                color_mask_fraction=(color_weight > 0).float().mean(),
                sigma=prepared["sigma"])


def chromaticity(image: torch.Tensor) -> torch.Tensor:
    return image / image.sum(0, keepdim=True).clamp_min(1e-6)


def chroma_gradient(image: torch.Tensor) -> torch.Tensor:
    """Return all RGB chromaticity Sobel components as six CHW channels."""
    return torch.cat([sobel(channel[None]) for channel in image], dim=0)


@torch.no_grad()
def prepare_asg_material_target(target: torch.Tensor) -> dict:
    """Identify heuristic flat regions and chromatic material-edge candidates."""
    prepared = prepare_compat_target(target)
    clean = prepared["clean"]
    q = chromaticity(clean)
    q_second = chromaticity(binomial_blur(clean))
    gradients = chroma_gradient(q)
    gradients_second = chroma_gradient(q_second)
    norm = gradients.square().sum(0, keepdim=True).clamp_min(1e-12).sqrt()
    norm_second = gradients_second.square().sum(0, keepdim=True).clamp_min(1e-12).sqrt()
    agreement = (gradients * gradients_second).sum(0, keepdim=True) / (norm * norm_second).clamp_min(1e-8)
    chroma_noise = prepared["sigma"] / clean.sum(0, keepdim=True).clamp_min(12 / 255)
    log_noise = prepared["sigma"] / gray(clean).clamp_min(1 / 255)
    threshold_c = (3 * chroma_noise).clamp_min(1e-4)
    threshold_y = (3 * log_noise).clamp_min(1e-4)
    material = (norm > threshold_c) & (agreement >= .8)
    material_neighborhood = F.max_pool2d(material.float()[None], 5, stride=1, padding=2)[0]
    flat = ((prepared["contrast"] <= threshold_y)
            & (local_std(q).square().sum(0, keepdim=True).sqrt() <= threshold_c))
    prepared.update(material_edge_confidence=prepared["edge_confidence"] * material,
                    material_contrast_confidence=prepared["contrast_confidence"] * material_neighborhood,
                    flat_confidence=prepared["confidence"] * flat,
                    material_confidence=prepared["confidence"] * material)
    return prepared


def asg_material_image_terms(reflectance: torch.Tensor, prepared: dict,
                             coverage: torch.Tensor) -> dict:
    """Constrain only reliable material structure and flat-region chromaticity."""
    valid = (coverage.detach().amin(0, keepdim=True) >= .95).to(reflectance.dtype)
    clean = binomial_blur(reflectance)
    log_y = gray(clean).clamp_min(1e-3).log()
    edge = sobel(log_y).square().sum(0, keepdim=True).clamp_min(1e-12).sqrt()
    edge_weight = valid * prepared["material_edge_confidence"]
    contrast_weight = valid * prepared["material_contrast_confidence"]
    flat_weight = valid * prepared["flat_confidence"]
    q = chromaticity(clean)
    weights_x = torch.minimum(flat_weight[:, :, 1:], flat_weight[:, :, :-1])
    weights_y = torch.minimum(flat_weight[:, 1:, :], flat_weight[:, :-1, :])
    differences_x = (q[:, :, 1:] - q[:, :, :-1]).abs().mean(0, keepdim=True)
    differences_y = (q[:, 1:, :] - q[:, :-1, :]).abs().mean(0, keepdim=True)
    flat_loss = ((differences_x * weights_x).sum() + (differences_y * weights_y).sum()) / (
        weights_x.sum() + weights_y.sum()).clamp_min(1e-8)
    return dict(edge=weighted_mean(structure_band(edge, prepared["edge"]), edge_weight),
                contrast=weighted_mean(structure_band(local_std(log_y), prepared["contrast"]), contrast_weight),
                flat=flat_loss, color=reflectance.new_zeros(()),
                signal_mask_fraction=(prepared["confidence"] > 0).float().mean(),
                edge_mask_fraction=(edge_weight > 0).float().mean(),
                contrast_mask_fraction=(contrast_weight > 0).float().mean(),
                flat_mask_fraction=(flat_weight > 0).float().mean(),
                material_mask_fraction=((valid * prepared["material_confidence"]) > 0).float().mean(),
                sigma=prepared["sigma"])


def asg_parameter_compat_terms(raw_amplitude: torch.Tensor, raw_sharpness: torch.Tensor,
                               lambda_min: float = 1.) -> dict:
    """Evaluate the paper's means over every explicit lobe, without rasterization."""
    sharpness = F.softplus(raw_sharpness) + lambda_min
    ratio = sharpness.amax(-1) / (sharpness.amin(-1) + 1e-6)
    return dict(asg_energy=raw_amplitude.sigmoid().mean(), asg_sharpness=sharpness.mean(),
                asg_anisotropy=ratio.mean())


def parameter_compat_terms(b0: torch.Tensor, detail: torch.Tensor,
                           decoder_squared: torch.Tensor, sg_tail: torch.Tensor) -> dict:
    return dict(b0=(F.relu(b0).square() + F.relu(math.log(.001) - b0).square()).mean(),
                detail=detail.tanh().square().mean(), decoder=decoder_squared, sg_tail=sg_tail)


def weighted_compat_losses(terms: dict, stage: str, iteration: int,
                           scale: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate R-only and illumination-only losses for white lists."""
    weights = STAGE_WEIGHTS[stage]
    ramp = compat_ramp(iteration) * scale
    zero = terms["edge"].new_zeros(())
    reflectance = sum((terms[key] * weight for key, weight in weights.items()
                      if key not in ILLUMINATION_TERMS and weight > 0), zero) * ramp
    sg = sum((terms[key] * weight for key, weight in weights.items()
              if key in ILLUMINATION_TERMS and weight > 0), zero) * ramp
    return reflectance, sg
