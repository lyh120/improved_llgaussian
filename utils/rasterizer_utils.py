"""Autograd contracts for auxiliary rasterization, independent of CUDA imports."""

import torch


def rasterize_reflectance_aux(rasterizer, *, means3D, means2D, colors_precomp,
                              opacities, scales, rotations):
    """Render trainable appearance with every geometry input frozen."""
    return rasterizer(
        means3D=means3D.detach(), means2D=means2D.detach(), shs=None,
        colors_precomp=colors_precomp, opacities=opacities.detach(),
        scales=scales.detach(), rotations=rotations.detach(), cov3D_precomp=None,
    )[0]


def rasterize_training_depth(zero_background_rasterizer, *, means3D, means2D,
                             opacities, scales, rotations):
    """Return accumulated depth/coverage with the direct z derivative intact.

    The caller supplies a zero-background rasterizer with the main camera/settings.
    Encoding camera-space z as color retains both alpha and direct depth gradients,
    unlike the custom extension's incomplete depth-output backward.
    """
    z = means3D[:, 2:3]
    channels = torch.cat((z, torch.ones_like(z), torch.zeros_like(z)), dim=1)
    image, _, _ = zero_background_rasterizer(
        means3D=means3D, means2D=means2D, shs=None, colors_precomp=channels,
        opacities=opacities, scales=scales, rotations=rotations, cov3D_precomp=None,
    )
    return image[:1], image[1:2].detach()
