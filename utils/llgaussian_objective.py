"""Released LL-Gaussian supervision for the retained R/L representation."""

import torch

from utils.visualize_utils import minmax_normalize
from utils.loss_utils import L_Depth_similarity, L_Illu, L_Smooth, l1_plus_loss, ssim


def llgaussian_objective(render_pkg, target, depth_prior, iteration, opt,
                        enhance_ratio, refined_target=None, warmup=False,
                        ssim_fn=ssim):
    """Compute reference weights, detach directions and activation thresholds.

    The single residual has a sigmoid output, so its mean equals its L1 norm.
    Extra reflectance, color, SG and geometry objectives are deliberately absent.
    """
    reflectance = render_pkg["render_reflectance"]
    illumination = render_pkg["render_illumination"]
    enhanced = render_pkg["render_illumination_enhanced"]
    intrinsic = reflectance * illumination
    residual = (torch.zeros_like(target) if warmup else
                render_pkg.get("render_residual", torch.zeros_like(target)))
    prediction = (intrinsic + residual).clamp(0.0, 1.0)
    terms = {
        "l1": l1_plus_loss(prediction, target, phi=0.5 / 255).abs().mean(),
        "dssim": 1.0 - ssim_fn(intrinsic, target),
        "illumination": L_Illu(target, illumination),
        "volume": 0.01 * render_pkg["scaling"].prod(dim=1).mean(),
    }
    zero = target.new_zeros(())
    terms.update(smooth=zero, depth=zero, degree=zero, enhanced_smooth=zero,
                 prior_l=zero, prior_r=zero, residual=zero, residual_volume=zero)
    if iteration >= opt.update_from:
        terms["smooth"] = L_Smooth(illumination, target, kernel_size=9) * (1e-4 if warmup else 1e-3)
        depth = render_pkg["render_depth"]
        normalized_depth = minmax_normalize(depth)
        terms["depth"] = L_Depth_similarity(
            1 - normalized_depth.squeeze(0), depth_prior.squeeze(0), 128, 0.5) * 0.15
        if not warmup:
            terms["degree"] = (
                (enhanced.mean(0) - (illumination.mean(0).detach() * enhance_ratio).clamp(0, 1)).abs().mean() * 0.2
                + (enhanced.mean() - illumination.mean().detach() * enhance_ratio).abs() * 0.05)
            terms["enhanced_smooth"] = L_Smooth(enhanced / enhance_ratio, target, kernel_size=9) * 5e-4
    if not warmup and iteration >= opt.update_from * 2:
        if refined_target is None:
            raise ValueError("StableSR training prior is required after the reference activation threshold")
        terms["prior_l"] = (enhanced * reflectance.detach() - refined_target).abs().mean()
        terms["prior_r"] = (enhanced.detach() * reflectance - refined_target).abs().mean() * 0.2
    if not warmup and "render_residual" in render_pkg:
        weight = 2.0 + (0.5 - 2.0) * min(iteration / opt.iterations, 1.0)
        terms["residual"] = residual.mean() * weight
        terms["residual_volume"] = 0.05 * render_pkg["scaling_residual"].prod(dim=1).mean()
    loss = ((1 - opt.lambda_dssim) * terms["l1"] + opt.lambda_dssim * terms["dssim"]
            + sum(value for key, value in terms.items() if key not in {"l1", "dssim"}))
    return loss, terms
