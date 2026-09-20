"""Strict, name-aligned loading for fixed v2 training priors."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

from utils.model_format import MODEL_FORMAT_VERSION


def resolve_required_file(
    candidates: list[str],
    description: str,
    override_argument: str,
) -> str:
    """Return the first existing file or raise one actionable path error."""
    checked = []
    for candidate in candidates:
        absolute = os.path.abspath(candidate)
        if absolute in checked:
            continue
        checked.append(absolute)
        if os.path.isfile(absolute):
            return absolute
    formatted = "\n  - ".join(checked)
    raise FileNotFoundError(
        f"{description} was not found. Checked:\n  - {formatted}\n"
        f"Download the checkpoint or provide its path with {override_argument}."
    )


def stablesr_prior_directory_name(input_gain: float) -> str:
    """Return the scene-local StableSR cache name keyed by explicit gain."""
    return f"diffusion_prior_{format(float(input_gain), 'g')}"


def prior_generation_required(path: str, prior_name: str) -> bool:
    """Decide whether a scene-local cache needs generation without overwriting data."""
    manifest = os.path.join(path, "manifest.json")
    if os.path.isfile(manifest):
        return False
    if os.path.exists(path) and not os.path.isdir(path):
        raise RuntimeError(f"{prior_name} prior path is not a directory: {path}")
    if os.path.isdir(path) and os.listdir(path):
        raise RuntimeError(
            f"{prior_name} prior directory is nonempty but has no v2 manifest: {path}. "
            "Move the old cache aside or provide an empty output directory; training will not overwrite it."
        )
    return True


def _read_manifest(root: str, prior_type: str) -> dict[str, dict]:
    if not root:
        raise ValueError(f"A fixed {prior_type} prior path is required")
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Prior manifest does not exist: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_manifest_fields = {"model_format_version", "prior_type", "entries"}
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        actual = set(manifest) if isinstance(manifest, dict) else set()
        raise ValueError(
            f"Invalid prior manifest fields; missing={sorted(expected_manifest_fields - actual)}, "
            f"extra={sorted(actual - expected_manifest_fields)}"
        )
    if manifest.get("model_format_version") != MODEL_FORMAT_VERSION:
        raise RuntimeError(
            f"Prior manifest {manifest_path} must use MODEL_FORMAT_VERSION={MODEL_FORMAT_VERSION}"
        )
    if manifest.get("prior_type") != prior_type:
        raise ValueError(
            f"Expected prior_type={prior_type!r}, found {manifest.get('prior_type')!r}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Prior manifest entries must be a list: {manifest_path}")
    indexed: dict[str, dict] = {}
    files = set()
    expected_entry_fields = {"image_name", "file", "width", "height"}
    for entry in entries:
        name = entry.get("image_name") if isinstance(entry, dict) else None
        if not isinstance(entry, dict) or set(entry) != expected_entry_fields:
            actual = set(entry) if isinstance(entry, dict) else set()
            raise ValueError(
                f"Invalid prior entry fields; missing={sorted(expected_entry_fields - actual)}, "
                f"extra={sorted(actual - expected_entry_fields)}"
            )
        if not name:
            raise ValueError(f"Every prior entry needs a non-empty image_name: {manifest_path}")
        if name in indexed:
            raise ValueError(f"Duplicate prior image_name {name!r}: {manifest_path}")
        filename = entry["file"]
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Prior file must be a non-empty relative path for {name!r}")
        if filename in files:
            raise ValueError(f"Duplicate prior file {filename!r}: {manifest_path}")
        files.add(filename)
        indexed[name] = entry
    return indexed


def _prior_file(root: str, filename: str) -> str:
    root_path = os.path.abspath(root)
    path = os.path.abspath(os.path.join(root_path, filename))
    try:
        inside_root = os.path.commonpath((root_path, path)) == root_path
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"Prior file escapes its manifest directory: {filename!r}")
    return path


def _validate_key_set(cameras, entries: dict[str, dict], prior_type: str) -> None:
    camera_names = [camera.image_name for camera in cameras]
    if len(camera_names) != len(set(camera_names)):
        duplicates = sorted({name for name in camera_names if camera_names.count(name) > 1})
        raise ValueError(f"Duplicate training camera image_name values: {duplicates[:5]}")
    expected = set(camera_names)
    actual = set(entries)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{prior_type} prior keys do not exactly match training cameras; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def load_enhancement_priors(cameras, root: str) -> dict[int, torch.Tensor]:
    """Load lossless RGB priors and attach them to their camera objects."""
    entries = _read_manifest(root, "enhancement_rgb")
    _validate_key_set(cameras, entries, "enhancement")
    loaded = {}
    for camera in tqdm(cameras, desc="Loading enhancement priors", leave=False):
        entry = entries[camera.image_name]
        path = _prior_file(root, entry["file"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Enhancement prior image is missing: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            expected = (int(camera.image_width), int(camera.image_height))
            manifest_size = (int(entry.get("width", -1)), int(entry.get("height", -1)))
            if (width, height) != expected or manifest_size != expected:
                raise ValueError(
                    f"Enhancement prior size mismatch for {camera.image_name!r}: "
                    f"file={(width, height)}, manifest={manifest_size}, camera={expected}"
                )
            tensor = pil_to_tensor(image).float().div_(255.0).to(camera.original_image.device)
        camera.enhancement_prior = tensor
        loaded[camera.uid] = tensor
    return loaded


def load_depth_priors(cameras, root: str) -> dict[int, torch.Tensor]:
    """Load float32 disparity maps without running DepthAnything in training."""
    entries = _read_manifest(root, "depth_disparity")
    _validate_key_set(cameras, entries, "depth")
    loaded = {}
    for camera in tqdm(cameras, desc="Loading depth priors", leave=False):
        entry = entries[camera.image_name]
        path = _prior_file(root, entry["file"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Depth prior array is missing: {path}")
        disparity = np.load(path, allow_pickle=False)
        if disparity.dtype != np.float32:
            raise ValueError(f"Depth prior must be float32 for {camera.image_name!r}, got {disparity.dtype}")
        if disparity.ndim == 3 and disparity.shape[0] == 1:
            disparity = disparity[0]
        expected_shape = (int(camera.image_height), int(camera.image_width))
        manifest_size = (int(entry.get("height", -1)), int(entry.get("width", -1)))
        if disparity.shape != expected_shape or manifest_size != expected_shape:
            raise ValueError(
                f"Depth prior size mismatch for {camera.image_name!r}: "
                f"array={disparity.shape}, manifest={manifest_size}, camera={expected_shape}"
            )
        if not np.isfinite(disparity).all():
            raise ValueError(f"Depth prior contains non-finite values: {path}")
        tensor = torch.from_numpy(disparity).unsqueeze(0).to(camera.original_image.device)
        loaded[camera.uid] = tensor
    return loaded


def load_training_priors(cameras, enhancement_root: str, depth_root: str):
    """Load fixed priors in the clean-reference startup order."""
    tqdm.write(f"[prior load 1/2] DepthAnything disparity: {depth_root}")
    depth = load_depth_priors(cameras, depth_root)
    tqdm.write(f"[prior load 2/2] Enhancement RGB: {enhancement_root}")
    enhancement = load_enhancement_priors(cameras, enhancement_root)
    return enhancement, depth
