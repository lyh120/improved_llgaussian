"""Named defaults preserving the original ASG experiment values."""

import math

REPRESENTATION_DEFAULTS = {
    "reflectance_detail_scale": 1.1,
    "reflectance_decoder_scale": 0.15,
    "reflectance_floor": 1e-3,
    "b0_init_intensity_floor": 0.1,
    "b0_init_detail_gain": 0.4,
    "b0_init_window_size": 5,
}

SUPERVISION_DEFAULTS = {
    "reflectance_edge_uplift_threshold": 0.15,
    "reflectance_edge_uplift_target_ratio": 0.85,
    "reflectance_contrast_threshold": 0.1,
    "reflectance_contrast_target_ratio": 0.8,
    "reflectance_contrast_window_size": 5,
    "reflectance_highfreq_threshold": 0.1,
    "reflectance_highfreq_target_ratio": 0.85,
    "depth_loss_weight": 0.15,
    "depth_patch_size": 128,
    "depth_sample_fraction": 0.5,
    "depth_prior_encoder": "vitl",
    "depth_prior_checkpoint": "./checkpoints/depth_anything_v2_vitl.pth",
    "stablesr_config": "./configs/config.gin",
    "enhancement_target_mean": 0.45,
    "enhancement_smooth_kernel_size": 0,
    "residual_reg_initial_weight": 2.0,
    "residual_reg_final_weight": 1.0,
    "stablesr_input_mode": "clamped",
}


def configure_reflectance(model, values=None) -> None:
    """Apply a saved representation, using historical defaults for missing fields."""
    config = dict(REPRESENTATION_DEFAULTS)
    if values is not None:
        config.update({key: values[key] for key in config if key in values})
    if not all(math.isfinite(value) for value in config.values()):
        raise ValueError("Reflectance representation settings must be finite")
    if config["reflectance_detail_scale"] < 0 or not 0 <= config["reflectance_decoder_scale"] < 1:
        raise ValueError("Reflectance detail scale must be nonnegative and decoder scale must be in [0,1)")
    if not 0 < config["reflectance_floor"] <= 1 or config["b0_init_intensity_floor"] <= 0:
        raise ValueError("Reflectance and initialization floors must be positive")
    window = config["b0_init_window_size"]
    if not isinstance(window, int) or window < 1 or window % 2 != 1:
        raise ValueError("B0 initialization window must be a positive odd integer")
    for key, value in config.items():
        setattr(model, key, value)
