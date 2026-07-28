"""Image-space composition helpers shared by training and rendering."""

import torch


def compose_decomposed_render(render_pkg, enhanced=False):
    """Compose rasterized reflectance and illumination in image space."""
    illumination_key = "render_illumination_enhanced" if enhanced else "render_illumination"
    return torch.clamp(
        render_pkg["render_reflectance"] * render_pkg[illumination_key],
        0.0,
        1.0,
    )
