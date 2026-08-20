"""Image-space composition helpers shared by training and rendering."""

import torch


def compose_decomposed_render(render_pkg, enhanced=False, detach_reflectance=False):
    """Compose rasterized reflectance and illumination in image space."""
    illumination_key = "render_illumination_enhanced" if enhanced else "render_illumination"
    reflectance = render_pkg["render_reflectance"]
    if detach_reflectance:
        reflectance = reflectance.detach()
    return torch.clamp(
        reflectance * render_pkg[illumination_key],
        0.0,
        1.0,
    )
