"""Autograd boundaries shared by auxiliary Gaussian rasterization passes."""


def rasterize_with_frozen_geometry(
    rasterizer,
    *,
    means3D,
    means2D,
    colors_precomp,
    opacities,
    scales,
    rotations,
):
    """Rasterize trainable colors without geometry or opacity gradients."""
    return rasterizer(
        means3D=means3D.detach(),
        means2D=means2D.detach(),
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacities.detach(),
        scales=scales.detach(),
        rotations=rotations.detach(),
        cov3D_precomp=None,
    )
