"""Saved-image evaluation and LL-Gaussian supplementary luminance alignment."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def align_lab_luminance(prediction: np.ndarray, target: np.ndarray) -> tuple:
    """Invert the paper's least-squares fit GT-L -> predicted-L, keeping a/b."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("Expected matching H x W x 3 RGB arrays")
    pred_lab = cv2.cvtColor(np.clip(prediction, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(np.clip(target, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    x = target_lab[..., 0].astype(np.float64)
    y = pred_lab[..., 0].astype(np.float64)
    variance = np.mean((x - x.mean()) ** 2)
    a = np.mean((x - x.mean()) * (y - y.mean())) / max(variance, 1e-12)
    b = y.mean() - a * x.mean()
    degenerate = variance < 1e-12 or abs(a) < 1e-6
    if not degenerate:
        pred_lab[..., 0] = np.clip((y - b) / a, 0, 100)
    aligned = np.clip(cv2.cvtColor(pred_lab, cv2.COLOR_LAB2RGB), 0, 1)
    return aligned, {"a": float(a), "b": float(b), "degenerate_identity_fallback": bool(degenerate)}


@torch.no_grad()
def evaluate_saved_test_set(folder: str, source_path: str, device: str = "cuda") -> dict:
    """Evaluate every saved test PNG; paired normal-light GT is evaluation-only."""
    import lpips
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    root = Path(folder)
    filenames = sorted(p.name for p in (root / "renders").glob("*.png"))
    if not filenames:
        raise RuntimeError(f"No test PNGs found in {root}")
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    results = {"lowlight": {}, "enhanced_raw": {}, "enhanced_lab_aligned": {}}
    alignment = {}
    normal_root = Path(source_path) / "gt" / "images"
    aligned_root = root / "render_enhanceds_lab_aligned"

    def read(path):
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if array is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    def metrics(pred, gt):
        if pred.shape != gt.shape:
            raise ValueError(f"GT/image sizes differ: {pred.shape} vs {gt.shape}")
        pred_tensor = torch.from_numpy(pred.copy()).permute(2, 0, 1).unsqueeze(0).to(device).contiguous()
        gt_tensor = torch.from_numpy(gt.copy()).permute(2, 0, 1).unsqueeze(0).to(device).contiguous()
        return {"PSNR": float(psnr(pred_tensor, gt_tensor).mean()),
                "SSIM": float(ssim(pred_tensor, gt_tensor)),
                "LPIPS": float(lpips_fn(pred_tensor, gt_tensor))}

    hashes = {}
    for name in filenames:
        paths = [root / "renders" / name, root / "gt" / name, root / "render_enhanceds" / name]
        for path in paths:
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        results["lowlight"][name] = metrics(read(paths[0]), read(paths[1]))
        normal = normal_root / name
        if normal_root.is_dir():
            # Missing paired views must never silently shrink the evaluated set.
            gt = read(normal)
            pred = read(paths[2])
            results["enhanced_raw"][name] = metrics(pred, gt)
            aligned, alignment[name] = align_lab_luminance(pred, gt)
            results["enhanced_lab_aligned"][name] = metrics(aligned, gt)
            aligned_root.mkdir(exist_ok=True)
            cv2.imwrite(str(aligned_root / name), cv2.cvtColor(np.round(aligned * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            hashes["normal_gt/" + name] = hashlib.sha256(normal.read_bytes()).hexdigest()
    payload = {}
    for kind, per_view in results.items():
        if not per_view:
            continue
        summary = {key: float(np.mean([row[key] for row in per_view.values()])) for key in ("PSNR", "SSIM", "LPIPS")}
        report = {"summary": summary, "per_view": per_view,
                  "protocol": {"test_views": len(per_view), "input": "saved 8-bit sRGB PNG, no crop or resize",
                               "SSIM": "RGB 11x11 Gaussian window sigma=1.5",
                               "LPIPS": "VGG, [0,1], normalize=False; original LL-Gaussian train.py evaluator",
                               "alignment": "per-view LAB L affine inversion, GT used only during evaluation" if kind.endswith("aligned") else "none"},
                  "input_sha256": hashes}
        if kind.endswith("aligned"):
            report["alignment_parameters"] = alignment
        filename = {"lowlight": "metrics_lowlight.json", "enhanced_raw": "metrics_enhanced_gt_raw.json",
                    "enhanced_lab_aligned": "metrics_enhanced_gt.json"}[kind]
        (root / filename).write_text(json.dumps(report, indent=2))
        payload[kind] = report
        print(f"[saved test {kind}] {summary}")
    return payload
