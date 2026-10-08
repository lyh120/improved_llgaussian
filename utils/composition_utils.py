"""Composition helpers shared by training and rendering."""

import torch


def compose_decomposed_render(render_pkg, enhanced=False):
    """Use radiance composited once; support older component-only packages."""
    radiance_key = "render_enhanced" if enhanced else "render"
    if radiance_key in render_pkg:
        return torch.clamp(render_pkg[radiance_key], 0.0, 1.0)
    illumination_key = "render_illumination_enhanced" if enhanced else "render_illumination"
    return torch.clamp(
        render_pkg["render_reflectance"] * render_pkg[illumination_key],
        0.0,
        1.0,
    )
