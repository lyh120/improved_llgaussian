"""Frozen RGB prior loading and lazy StableSR generation."""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_source_image(source_path, images, image_name) -> Path:
    """Resolve an image in the configured input directory, including mixed-case suffixes."""
    directory = Path(source_path) / images
    exact = directory / image_name
    if exact.is_file():
        return exact
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    matches = sorted(path for path in directory.glob(image_name + ".*")
                     if path.suffix.lower() in allowed)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source image for {image_name} in {directory}, found {matches}")
    return matches[0]


def normalize_rgb_prior(value, camera) -> torch.Tensor:
    """Return a detached, finite CPU RGB target aligned to the training camera."""
    if isinstance(value, Image.Image):
        array = np.array(value.convert("RGB"), copy=True)
        value = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    else:
        value = torch.as_tensor(value).detach().cpu()
        if not value.is_floating_point():
            value = value.float() / 255.0
        value = value.float()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"RGB prior for {camera.image_name} must have shape [3,H,W]")
    if not torch.isfinite(value).all():
        raise ValueError(f"RGB prior for {camera.image_name} contains non-finite pixels")
    size = (camera.image_height, camera.image_width)
    if value.shape[-2:] != size:
        value = F.interpolate(value[None], size=size, mode="bilinear", align_corners=False,
                              antialias=True)[0]
    return value.clamp(0.0, 1.0).contiguous().detach()


def resolve_cached_prior(directory, camera, name_first=False):
    """Use the same legacy filename resolution for cache checks and loading."""
    uid = Path(directory) / f"{camera.uid}.png"
    name = Path(directory) / f"{camera.image_name}.png"
    candidates = (name, uid) if name_first else (uid, name)
    return next((path for path in candidates if path.is_file()), None)


def load_rgb_prior(directory, cameras, name_first=False) -> dict:
    """Load complete StableSR or CIDNet targets without importing either backend."""
    result = {}
    for camera in cameras:
        path = resolve_cached_prior(directory, camera, name_first)
        if path is None:
            raise FileNotFoundError(f"Missing prior for {camera.image_name} (uid={camera.uid}) in {directory}")
        with Image.open(path) as image:
            result[camera.uid] = normalize_rgb_prior(image, camera)
    return result


def _load_stablesr_backend(config_path):
    # StableSR's upstream configuration imports its vendored `ldm` package.
    root = str(PROJECT_ROOT / "submodules" / "StableSR")
    if root not in sys.path:
        sys.path.append(root)
    from utils.StableSR_utlis import SD_refine, get_SRModel

    model, vq_model, options = get_SRModel(config_path)
    return model, vq_model, options, SD_refine


def prepare_stablesr_prior(dataset, cameras, enhance_ratio, logger=None, backend_loader=None):
    """Reuse complete caches and generate only missing views with a frozen backend."""
    logger = logger or logging.getLogger(__name__)
    input_mode = getattr(dataset, "stablesr_input_mode", "clamped")
    if input_mode not in {"clamped", "llgaussian"}:
        raise ValueError("stablesr_input_mode must be clamped or llgaussian")
    directory = Path(dataset.source_path) / ("diffusion_prior_" + str(enhance_ratio))
    directory.mkdir(parents=True, exist_ok=True)
    missing = [camera for camera in cameras if resolve_cached_prior(directory, camera) is None]
    if not missing:
        logger.info("Loading cached StableSR prior from %s", directory)
        return load_rgb_prior(directory, cameras)
    loader = backend_loader or _load_stablesr_backend
    config = Path(getattr(dataset, "stablesr_config", "./configs/config.gin"))
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    model, vq_model, options, refine = loader(str(config))
    try:
        model.eval().requires_grad_(False)
        vq_model.eval().requires_grad_(False)
        with torch.no_grad():
            for camera in missing:
                inputs = camera.original_image * enhance_ratio
                if input_mode == "clamped":
                    inputs = inputs.clamp(0.0, 1.0)
                inputs = inputs[None]
                refined = normalize_rgb_prior(refine(model, vq_model, inputs, options), camera)
                array = (refined.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                path = directory / f"{camera.uid}.png"
                temporary = path.with_suffix(".tmp.png")
                Image.fromarray(array).save(temporary)
                os.replace(temporary, path)
    finally:
        model.cpu()
        vq_model.cpu()
        del model, vq_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Always use the saved target, so cache hits and freshly generated runs agree.
    return load_rgb_prior(directory, cameras)
