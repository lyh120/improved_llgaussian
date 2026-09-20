#!/usr/bin/env python3
"""Cache DepthAnything V2 relative disparity as strict float32 v2 priors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.model_format import MODEL_FORMAT_VERSION

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ENCODER_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
OFFICIAL_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}


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


def _load_model(root: Path, encoder: str, checkpoint: Path, device: torch.device):
    if not checkpoint.is_file():
        download = OFFICIAL_CHECKPOINT_URLS.get(encoder, "the official DepthAnything V2 release")
        raise FileNotFoundError(
            f"DepthAnything checkpoint does not exist: {checkpoint}. "
            f"Download it from {download}"
        )
    sys.path.insert(0, str(root.resolve()))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as error:
        raise ImportError(f"Could not import DepthAnything V2 from {root}") from error
    model = DepthAnythingV2(encoder=encoder, **ENCODER_CONFIGS[encoder])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth_anything_root", default="submodules/Depth-Anything-V2")
    parser.add_argument("--encoder", choices=sorted(ENCODER_CONFIGS), default="vitl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--lod", type=int, default=0)
    parser.add_argument("--llffhold", type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output)
    manifest_path = output_root / "manifest.json"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Depth prior output is not empty: {output_root}. Use a new output directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    images = _select_training_images(
        _images_by_name(Path(args.source_path) / args.images),
        args.eval,
        args.lod,
        args.llffhold,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for depth preprocessing but is unavailable")
    print(f"[DepthAnything] Loading {args.encoder} from {Path(args.checkpoint).resolve()}")
    model = _load_model(
        Path(args.depth_anything_root),
        args.encoder,
        Path(args.checkpoint),
        device,
    )
    print("[DepthAnything] Model loaded; generating fixed disparity cache")

    entries = []
    progress_bar = tqdm(images.items(), desc="DepthAnything inference progress")
    for index, (image_name, path) in enumerate(progress_bar):
        progress_bar.set_description(f"DepthAnything inference progress ({image_name})")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode {path}")
        height, width = image.shape[:2]
        with torch.inference_mode():
            disparity = model.infer_image(image, args.input_size)
        if isinstance(disparity, torch.Tensor):
            disparity = disparity.detach().cpu().numpy()
        disparity = np.asarray(disparity, dtype=np.float32).squeeze()
        if disparity.shape != (height, width):
            raise ValueError(
                f"DepthAnything output size mismatch for {image_name!r}: "
                f"{disparity.shape} != {(height, width)}"
            )
        if not np.isfinite(disparity).all():
            raise ValueError(f"DepthAnything produced non-finite values for {image_name!r}")
        filename = f"{index:06d}.npy"
        np.save(output_root / filename, disparity, allow_pickle=False)
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
        "prior_type": "depth_disparity",
        "entries": entries,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(entries)} float32 disparity priors to {output_root}")


if __name__ == "__main__":
    main()
