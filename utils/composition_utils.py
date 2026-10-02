"""Image-space composition helpers shared by training and rendering."""

import torch


def compose_decomposed_render(
    render_pkg,
    enhanced=False,
    detach_reflectance=False,
    detach_illumination=False,
    reflectance_key="render_reflectance",
):
    """Compose R and L with the model's selected image formation path.

    Direct composition rasterizes per-Gaussian R*L once. The legacy path
    multiplies separate image-space rasters. Gradient routing is selected
    independently for reflectance and illumination supervision.

    Args:
        render_pkg: Renderer output dictionary.
        enhanced: Use the enhanced illumination image when true.
        detach_reflectance: Stop the composed loss from updating the
            reflectance rasterization path. This is intended for enhancement
            supervision, which should train illumination without changing
            Gaussian geometry through the reflectance image.
        detach_illumination: Symmetric counterpart of ``detach_reflectance``;
            keeps the composed loss from updating the illumination path.
        reflectance_key: Which reflectance raster to compose, e.g.
            ``"render_reflectance_sharpen"`` for the geometry-isolated
            sharpening branch.
    """
    if render_pkg.get("direct_composition", False):
        if enhanced:
            if detach_reflectance:
                direct = render_pkg.get("render_enhanced_illumination")
                if direct is not None:
                    return torch.clamp(direct, 0.0, 1.0)
            if detach_illumination:
                direct = render_pkg.get("render_enhanced_reflectance")
                if direct is not None:
                    return torch.clamp(direct, 0.0, 1.0)
            elif reflectance_key == "render_reflectance":
                return torch.clamp(render_pkg["render_enhanced"], 0.0, 1.0)
        elif reflectance_key == "render_reflectance" and not detach_reflectance and not detach_illumination:
            return torch.clamp(render_pkg["render"], 0.0, 1.0)
    illumination_key = "render_illumination_enhanced" if enhanced else "render_illumination"
    reflectance = render_pkg[reflectance_key]
    illumination = render_pkg[illumination_key]
    if detach_reflectance:
        reflectance = reflectance.detach()
    if detach_illumination:
        illumination = illumination.detach()
    return torch.clamp(
        reflectance * illumination,
        0.0,
        1.0,
    )
