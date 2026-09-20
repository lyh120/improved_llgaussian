#!/usr/bin/env python3
"""Generate one fixed, lossless CIDNet enhancement prior per training image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.model_format import MODEL_FORMAT_VERSION

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CIDNET_PADDING_FACTOR = 8


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


def _load_model(cidnet_root: Path, weights: Path, device: torch.device):
    sys.path.insert(0, str(cidnet_root.resolve()))
    try:
        from net.CIDNet import CIDNet
    except ImportError as error:
        raise ImportError(f"Could not import CIDNet from {cidnet_root}") from error

    model = CIDNet().to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"CIDNet weights must contain a state dictionary: {weights}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    model.trans.alpha_s = 1.0
    model.trans.alpha = 1.0
    model.trans.gated = True
    model.trans.gated2 = True
    return model


@torch.inference_mode()
def _enhance(model, path: Path, device: torch.device) -> Image.Image:
    with Image.open(path) as source:
        source = source.convert("RGB")
        width, height = source.size
        tensor = pil_to_tensor(source).float().div_(255.0).unsqueeze(0).to(device)
    pad_height = (-height) % CIDNET_PADDING_FACTOR
    pad_width = (-width) % CIDNET_PADDING_FACTOR
    if pad_height or pad_width:
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height), mode=mode)
    enhanced = model(tensor).clamp(0.0, 1.0)[0, :, :height, :width].cpu()
    return to_pil_image(enhanced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cidnet_root", default="submodules/HVI-CIDNet")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--lod", type=int, default=0)
    parser.add_argument("--llffhold", type=int, default=8)
    args = parser.parse_args()

    image_root = Path(args.source_path) / args.images
    output_root = Path(args.output)
    manifest_path = output_root / "manifest.json"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"CIDNet prior output is not empty: {output_root}. Use a new output directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    images = _select_training_images(
        _images_by_name(image_root),
        args.eval,
        args.lod,
        args.llffhold,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for CIDNet preprocessing but is unavailable")
    print(f"[CIDNet] Loading model from {Path(args.weights).resolve()}")
    model = _load_model(Path(args.cidnet_root), Path(args.weights), device)
    print("[CIDNet] Model loaded; generating fixed enhancement cache")

    entries = []
    progress_bar = tqdm(images.items(), desc="CIDNet enhancement progress")
    for index, (image_name, path) in enumerate(progress_bar):
        progress_bar.set_description(f"CIDNet enhancement progress ({image_name})")
        enhanced = _enhance(model, path, device)
        filename = f"{index:06d}.png"
        enhanced.save(output_root / filename, format="PNG")
        width, height = enhanced.size
        entries.append(
            {
                "image_name": image_name,
                "file": filename,
                "width": width,
                "height": height,
            }
        )
    progress_bar.close()

    manifest = {
        "model_format_version": MODEL_FORMAT_VERSION,
        "prior_type": "enhancement_rgb",
        "entries": entries,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(entries)} fixed CIDNet priors to {output_root}")


if __name__ == "__main__":
    main()
