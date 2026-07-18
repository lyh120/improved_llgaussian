"""Evaluate PSNR, SSIM, and LPIPS between two image folders.

Images are paired by their relative path without the file extension. For
example, ``renders/00001.png`` is paired with ``gt/00001.jpg``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from tqdm import tqdm

from utils.image_utils import psnr
from utils.loss_utils import ssim


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate image-folder quality using the project's PSNR, SSIM, and LPIPS metrics."
    )
    parser.add_argument("--pred_dir", required=True, type=Path, help="Directory containing rendered or predicted images.")
    parser.add_argument("--gt_dir", required=True, type=Path, help="Directory containing ground-truth reference images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metric-report JSON path. Defaults to <pred_dir>/metrics.json.",
    )
    parser.add_argument("--device", default="auto", help="Torch device, such as cuda, cuda:0, or cpu. Default: auto.")
    parser.add_argument("--recursive", action="store_true", help="Also search nested image directories.")
    parser.add_argument(
        "--no_resize_gt",
        action="store_true",
        help="Fail on image-size mismatches instead of resizing GT to the prediction size.",
    )
    parser.add_argument("--skip_lpips", action="store_true", help="Compute only PSNR and SSIM.")
    parser.add_argument("--strict", action="store_true", help="Fail when either folder contains unmatched images.")
    return parser.parse_args()


def collect_images(root, recursive):
    pattern = "**/*" if recursive else "*"
    images = {}
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in images:
            raise ValueError(f"Ambiguous image key '{key}' in {root}: {images[key]} and {path}")
        images[key] = path
    return images


def load_rgb_tensor(path, device):
    with Image.open(path) as image:
        return tf.to_tensor(image.convert("RGB")).to(device)


def align_pair(prediction, target, resize_gt):
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction, target
    if not resize_gt:
        raise ValueError(
            f"Image sizes differ: prediction={tuple(prediction.shape[-2:])}, "
            f"GT={tuple(target.shape[-2:])}."
        )
    target = F.interpolate(
        target.unsqueeze(0),
        size=prediction.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return prediction, target


def compute_metrics(prediction, target, lpips_model):
    prediction = prediction.unsqueeze(0)
    target = target.unsqueeze(0)
    values = {
        "PSNR": float(psnr(prediction, target).mean().detach().cpu()),
        "SSIM": float(ssim(prediction, target).detach().cpu()),
    }
    if lpips_model is not None:
        values["LPIPS"] = float(lpips_model(prediction, target).detach().cpu())
    return values


def main():
    args = parse_args()
    if not args.pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {args.pred_dir}")
    if not args.gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {args.gt_dir}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    pred_images = collect_images(args.pred_dir, args.recursive)
    gt_images = collect_images(args.gt_dir, args.recursive)
    shared_keys = sorted(pred_images.keys() & gt_images.keys())
    only_prediction = sorted(pred_images.keys() - gt_images.keys())
    only_gt = sorted(gt_images.keys() - pred_images.keys())

    if not shared_keys:
        raise RuntimeError("No matching image pairs were found. Pairing uses relative paths without extensions.")
    if args.strict and (only_prediction or only_gt):
        raise RuntimeError(
            f"Found unmatched images: {len(only_prediction)} prediction-only and {len(only_gt)} GT-only."
        )

    lpips_model = None
    if not args.skip_lpips:
        try:
            import lpips
        except ImportError as error:
            raise ImportError(
                "LPIPS is unavailable. Install the project's requirements or pass --skip_lpips "
                "to evaluate PSNR and SSIM only."
            ) from error
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    per_image = {}
    with torch.no_grad():
        for key in tqdm(shared_keys, desc="Evaluating", unit="image"):
            prediction = load_rgb_tensor(pred_images[key], device)
            target = load_rgb_tensor(gt_images[key], device)
            prediction, target = align_pair(prediction, target, resize_gt=not args.no_resize_gt)
            per_image[key] = compute_metrics(prediction, target, lpips_model)

    metric_names = next(iter(per_image.values())).keys()
    summary = {
        metric_name: float(np.mean([values[metric_name] for values in per_image.values()]))
        for metric_name in metric_names
    }
    report = {
        "summary": summary,
        "num_pairs": len(per_image),
        "unmatched_prediction": only_prediction,
        "unmatched_gt": only_gt,
        "per_image": per_image,
    }
    output_path = args.output or args.pred_dir / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    metrics_text = ", ".join(f"{name}={value:.4f}" for name, value in summary.items())
    print(f"Evaluated {len(per_image)} image pairs on {device}: {metrics_text}")
    print(f"Saved report to {output_path}")
    if only_prediction or only_gt:
        print(f"Unmatched images: prediction-only={len(only_prediction)}, gt-only={len(only_gt)}")


if __name__ == "__main__":
    main()
