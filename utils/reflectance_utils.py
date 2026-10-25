"""Reflectance capacity settings and train-only appearance evaluation."""

import torch
import torch.nn.functional as F

from utils.decomposition_config import REPRESENTATION_DEFAULTS


def reflectance_config(source) -> dict:
    """Read representation settings from an argparse object or model."""
    return {key: getattr(source, key, default) for key, default in REPRESENTATION_DEFAULTS.items()}


def predict_aux_reflectance(pc, feat, offsets, visible_mask, distance) -> torch.Tensor:
    """Evaluate R while treating all shared/geometry inputs as constants."""
    if pc.reflectance_mode == "explicit":
        value = pc.get_reflectance_with_decoder(
            feat.detach(), offsets.detach(), visible_mask, record_stats=False,
        )
    else:
        inputs = feat.detach()
        if pc.add_reflectance_dist:
            inputs = torch.cat((inputs, distance.detach()), dim=-1)
        value = pc.mlp_reflectance(inputs)
    return value.reshape(-1, 3)


@torch.no_grad()
def estimate_initial_b0(anchors, cameras, config) -> torch.Tensor:
    """Initialize B0 from the first training view using CUDA's pixel convention."""
    result = anchors.new_zeros((anchors.shape[0], 3))
    if not cameras:
        return result
    camera = cameras[0]
    image = camera.original_image.detach().to(anchors)
    height, width = image.shape[-2:]
    signal_valid = torch.isfinite(image).all(dim=0) & (image.amax(dim=0) > 0)
    if not signal_valid.any():
        return result
    floor = config.reflectance_floor
    image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    reflectance = (image / image.amax(dim=0, keepdim=True).clamp_min(
        config.b0_init_intensity_floor,
    )).clamp(floor, 1.0)
    window = config.b0_init_window_size
    blur = F.avg_pool2d(reflectance[None], window, stride=1, padding=window // 2)[0]
    reflectance = (reflectance + config.b0_init_detail_gain * (reflectance - blur)).clamp(floor, 1.0)
    log_reflectance = reflectance.log()
    result[:] = log_reflectance[:, signal_valid].mean(dim=1)

    homogeneous = torch.cat((anchors, torch.ones_like(anchors[:, :1])), dim=1)
    # Camera transforms are already transposed for the CUDA/GLM layout.
    view = homogeneous @ camera.world_view_transform.to(anchors)
    clip = homogeneous @ camera.full_proj_transform.to(anchors)
    finite = torch.isfinite(clip).all(dim=1) & torch.isfinite(view).all(dim=1)
    valid_w = clip[:, 3] > torch.finfo(anchors.dtype).eps
    ndc = clip[:, :3] / torch.where(valid_w, clip[:, 3], torch.ones_like(clip[:, 3]))[:, None]
    # ndc2Pix(v, S) = ((v + 1) * S - 1) / 2, including CUDA's y direction.
    pixels = ((ndc[:, :2] + 1) * anchors.new_tensor((width, height)) - 1) / 2
    valid = finite & valid_w & (view[:, 2] > camera.znear)
    valid &= (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
    valid &= (pixels[:, 0] >= -0.5) & (pixels[:, 0] < width - 0.5)
    valid &= (pixels[:, 1] >= -0.5) & (pixels[:, 1] < height - 0.5)
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    xy = pixels[indices].round().long()
    x, y = xy[:, 0].clamp(0, width - 1), xy[:, 1].clamp(0, height - 1)
    accepted = signal_valid[y, x]
    result[indices[accepted]] = log_reflectance[:, y[accepted], x[accepted]].T
    return result
