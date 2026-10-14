"""LL-Gaussian prior protocol and cache identity, without loading StableSR."""

import ast
import hashlib
import json
from pathlib import Path

import torch


LLGAUSSIAN_SETTINGS = {
    "ddpm_steps": 4, "dec_w": 0.75, "seed": 42, "n_samples": 1,
    "colorfix_type": "adain", "precision": "autocast", "tile_overlap": 32,
    "upscale": 1.0, "vqgantile_stride": 1000, "vqgantile_size": 1280,
    "input_size": 512, "n_iter": 1, "C": 4, "f": 8,
}


def enhancement_gain(mean: torch.Tensor, prior: str) -> int:
    """Use the release's unbounded integer gain for StableSR only."""
    if prior not in {"stablesr", "cidnet"}:
        raise ValueError(f"Unsupported enhancement_prior: {prior}")
    if not torch.isfinite(mean) or mean <= 0:
        raise ValueError("Training-image mean must be finite and positive")
    gain = int(0.45 / mean)
    return gain if prior == "stablesr" else max(1, gain)


def stablesr_input(image: torch.Tensor, gain: int) -> torch.Tensor:
    """Multiply before SD_refine's bicubic resize and internal clamp, as released."""
    return (image * gain).unsqueeze(0)


def sha256_file(path: Path) -> str:
    """Hash weights in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prior_cache_identity(project_root, cameras, gain: int) -> dict:
    """Bind cached priors to the release protocol, weights and training inputs."""
    root = Path(project_root)
    config = root / "configs/config.gin"
    settings = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("Config.") and "=" in line:
            key, value = line.split("=", 1)
            settings[key.removeprefix("Config.").strip()] = ast.literal_eval(value.strip())
    for key, expected in LLGAUSSIAN_SETTINGS.items():
        if settings.get(key) != expected:
            raise ValueError(f"StableSR must use LL-Gaussian {key}={expected!r}; got {settings.get(key)!r}")
    files = {
        "generator": root / "utils/StableSR_utlis.py", "gin": config,
        "diffusion_config": root / settings["config"],
        "vqgan_config": root / "submodules/StableSR/configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml",
        "diffusion_weights": root / settings["ckpt"],
        "vqgan_weights": root / settings["vqgan_ckpt"],
    }
    frames = []
    for camera in cameras:
        image = camera.original_image.detach().cpu().contiguous()
        frames.append({"uid": int(camera.uid), "image_name": camera.image_name,
                       "shape": list(image.shape),
                       "input_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest()})
    return {"protocol": "llgaussian_release_stablesr_unclipped_input_v1",
            "gain": gain, "settings": settings,
            "files_sha256": {name: sha256_file(path) for name, path in files.items()},
            "train_frames": frames}


def prior_cache_path(source_path, identity: dict) -> Path:
    """Keep old clipped-input priors separate, without overwriting them."""
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return Path(source_path) / f"diffusion_prior_llgaussian_{identity['gain']}_{digest}"


def prior_cache_matches(directory: Path, identity: dict) -> bool:
    """Check metadata and all output hashes before reusing a complete cache."""
    manifest = directory.with_suffix(".manifest.json")
    if not directory.is_dir() or not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            return False
        expected = {f"{frame['uid']}.png" for frame in identity["train_frames"]}
        outputs = record["outputs_sha256"]
        return set(outputs) == expected and all(
            (directory / name).is_file() and sha256_file(directory / name) == digest
            for name, digest in outputs.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
