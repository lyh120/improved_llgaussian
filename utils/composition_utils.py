"""Image-space composition helpers shared by training and rendering."""

import torch


def compose_decomposed_render(
    render_pkg,
    enhanced=False,
    detach_reflectance=False,
):
    """Compose rasterized reflectance and illumination in image space.

    Args:
        render_pkg: Renderer output containing the decomposed image buffers.
        enhanced: Use the enhanced illumination buffer when ``True``.
        detach_reflectance: Treat reflectance as a fixed teacher. This is used
            by enhancement-prior losses so they cannot update reflectance or
            geometry through the reflectance rasterization branch.
    """
    illumination_key = "render_illumination_enhanced" if enhanced else "render_illumination"
    reflectance = render_pkg["render_reflectance"]
    if detach_reflectance:
        reflectance = reflectance.detach()
    return torch.clamp(
        reflectance * render_pkg[illumination_key],
        0.0,
        1.0,
    )
