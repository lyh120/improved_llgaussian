#!/usr/bin/env python3
"""Generate strict v2 enhancement priors with the bundled StableSR inference script."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.model_format import MODEL_FORMAT_VERSION

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Fixed StableSR-Turbo prior recipe from the clean LL-Gaussian project. These
# are preprocessing settings, not appearance or training-loss multipliers.
STABLESR_STEPS = 4
STABLESR_DECODER_WEIGHT = 0.75
STABLESR_COLOR_FIX = "wavelet"
STABLESR_SEED = 42
STABLESR_INPUT_SIZE = 512
STABLESR_TILE_OVERLAP = 32
STABLESR_VQGAN_TILE_SIZE = 1280
STABLESR_VQGAN_TILE_STRIDE = 1000


def _images_by_name(image_root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(image_root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.stem in images:
            raise ValueError(
                f"Duplicate image_name {path.stem!r}: {images[path.stem]} and {path}"
            )
        images[path.stem] = path
    if not images:
        raise FileNotFoundError(f"No supported images found in {image_root}")
    return images


def _select_training_images(
    images: dict[str, Path],
    evaluate: bool,
    lod: int,
    llffhold: int,
) -> dict[str, Path]:
    if not evaluate:
        return images
    items = list(images.items())
    if lod > 0:
        items = items[lod + 1 :] if lod < 50 else items[: lod + 1]
    else:
        items = [item for index, item in enumerate(items) if index % llffhold != 0]
    if not items:
        raise ValueError("The requested evaluation split contains no training images")
    return dict(items)


def _save_gained_input(source_path: Path, destination: Path, gain: float) -> tuple[int, int]:
    with Image.open(source_path) as source:
        source = source.convert("RGB")
        size = source.size
        if gain != 1.0:
            source = source.point(lambda value: min(255, round(value * gain)))
        source.save(destination, format="PNG")
    return size


def _run_stablesr(args, input_root: Path, raw_output_root: Path) -> None:
    command = [
        args.stablesr_python,
        str(PROJECT_ROOT / "scripts" / "stablesr_compat_entrypoint.py"),
        args.inference_script,
        "--config",
        args.config,
        "--ckpt",
        args.checkpoint,
        "--vqgan_ckpt",
        args.vqgan_checkpoint,
        "--init-img",
        str(input_root),
        "--outdir",
        str(raw_output_root),
        "--ddpm_steps",
        str(args.steps),
        "--dec_w",
        str(args.decoder_weight),
        "--colorfix_type",
        args.colorfix_type,
        "--seed",
        str(args.seed),
        "--n_samples",
        "1",
        "--input_size",
        str(args.input_size),
        "--upscale",
        "1.0",
        "--tile_overlap",
        str(args.tile_overlap),
        "--vqgantile_size",
        str(args.vqgan_tile_size),
        "--vqgantile_stride",
        str(args.vqgan_tile_stride),
    ]
    environment = os.environ.copy()
    stable_root = str(Path(args.stablesr_root).resolve())
    stable_scripts = str(Path(args.inference_script).resolve().parent)
    existing_python_path = environment.get("PYTHONPATH", "")
    python_paths = [stable_root, stable_scripts]
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(
        command,
        cwd=stable_root,
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stablesr_root", required=True)
    parser.add_argument("--stablesr_python", default=sys.executable)
    parser.add_argument("--inference_script", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqgan_checkpoint", required=True)
    parser.add_argument("--input_gain", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=STABLESR_STEPS)
    parser.add_argument("--decoder_weight", type=float, default=STABLESR_DECODER_WEIGHT)
    parser.add_argument(
        "--colorfix_type",
        choices=("adain", "wavelet", "nofix"),
        default=STABLESR_COLOR_FIX,
    )
    parser.add_argument("--seed", type=int, default=STABLESR_SEED)
    parser.add_argument("--input_size", type=int, default=STABLESR_INPUT_SIZE)
    parser.add_argument("--tile_overlap", type=int, default=STABLESR_TILE_OVERLAP)
    parser.add_argument("--vqgan_tile_size", type=int, default=STABLESR_VQGAN_TILE_SIZE)
    parser.add_argument("--vqgan_tile_stride", type=int, default=STABLESR_VQGAN_TILE_STRIDE)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--lod", type=int, default=0)
    parser.add_argument("--llffhold", type=int, default=8)
    args = parser.parse_args()

    if args.input_gain <= 0.0:
        raise ValueError("StableSR input_gain must be positive")
    if args.steps <= 0:
        raise ValueError("StableSR steps must be positive")
    if args.vqgan_tile_stride >= args.vqgan_tile_size:
        raise ValueError("StableSR vqgan_tile_stride must be smaller than vqgan_tile_size")
    for label, path in (
        ("StableSR root", Path(args.stablesr_root)),
        ("StableSR inference script", Path(args.inference_script)),
        ("StableSR config", Path(args.config)),
        ("StableSR checkpoint", Path(args.checkpoint)),
        ("StableSR VQGAN checkpoint", Path(args.vqgan_checkpoint)),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    output_root = Path(args.output)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"StableSR prior output is not empty: {output_root}. Use a new output directory."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    images = _select_training_images(
        _images_by_name(Path(args.source_path) / args.images),
        args.eval,
        args.lod,
        args.llffhold,
    )

    with tempfile.TemporaryDirectory(prefix="llgaussian_stablesr_") as temporary:
        temporary_root = Path(temporary)
        input_root = temporary_root / "inputs"
        raw_output_root = temporary_root / "outputs"
        input_root.mkdir()
        raw_output_root.mkdir()
        original_sizes = {}
        input_progress = tqdm(images.items(), desc="StableSR input preparation progress")
        for image_name, source_path in input_progress:
            input_progress.set_description(
                f"StableSR input preparation progress ({image_name})"
            )
            original_sizes[image_name] = _save_gained_input(
                source_path,
                input_root / f"{image_name}.png",
                args.input_gain,
            )
        input_progress.close()
        print("[StableSR] Loading models and generating fixed enhancement cache")
        _run_stablesr(args, input_root, raw_output_root)

        output_root.mkdir(parents=True, exist_ok=True)
        entries = []
        output_progress = tqdm(images, desc="StableSR rendering progress")
        for index, image_name in enumerate(output_progress):
            output_progress.set_description(f"StableSR rendering progress ({image_name})")
            raw_path = raw_output_root / f"{image_name}.png"
            if not raw_path.is_file():
                raise FileNotFoundError(
                    f"StableSR did not produce the expected image for {image_name!r}: {raw_path}"
                )
            expected_size = original_sizes[image_name]
            with Image.open(raw_path) as enhanced:
                enhanced = enhanced.convert("RGB")
                if enhanced.size != expected_size:
                    enhanced = enhanced.resize(expected_size, Image.Resampling.LANCZOS)
                filename = f"{index:06d}.png"
                enhanced.save(output_root / filename, format="PNG")
            entries.append(
                {
                    "image_name": image_name,
                    "file": filename,
                    "width": expected_size[0],
                    "height": expected_size[1],
                }
            )
        output_progress.close()

    manifest = {
        "model_format_version": MODEL_FORMAT_VERSION,
        "prior_type": "enhancement_rgb",
        "entries": entries,
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(entries)} fixed StableSR priors to {output_root}")


if __name__ == "__main__":
    main()
