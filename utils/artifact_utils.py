"""Diagnostics for dark rendering artifacts and Gaussian coverage."""

import torch
import torch.nn.functional as F


def _luminance(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB CHW tensor to a single-channel luminance image."""
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("Expected an RGB tensor with shape [3, H, W].")
    weights = image.new_tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)
    return (image[:3] * weights).sum(dim=0)


def _coverage_to_2d(coverage: torch.Tensor) -> torch.Tensor:
    """Collapse renderer coverage to one spatial plane."""
    if coverage.ndim == 4:
        coverage = coverage.mean(dim=(0, 1))
    elif coverage.ndim == 3:
        coverage = coverage.mean(dim=0)
    elif coverage.ndim != 2:
        raise ValueError("Expected coverage with shape [H,W], [C,H,W], or [B,C,H,W].")
    return coverage


def _log_luminance_gradient(image: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    luminance = torch.log(_luminance(image).clamp_min(eps)).view(1, 1, *image.shape[-2:])
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    luminance = F.pad(luminance, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(luminance, kernel_x)
    grad_y = F.conv2d(luminance, kernel_y)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8).squeeze()


def compute_artifact_diagnostics(
    prediction: torch.Tensor,
    bright_gt: torch.Tensor,
    coverage: torch.Tensor,
    prediction_black_threshold: float = 0.05,
    gt_bright_threshold: float = 0.15,
    low_coverage_threshold: float = 0.95,
) -> dict:
    """Measure black predictions in regions that should be visibly bright."""
    if bright_gt.shape[-2:] != prediction.shape[-2:]:
        bright_gt = F.interpolate(
            bright_gt.unsqueeze(0),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if coverage.shape[-2:] != prediction.shape[-2:]:
        coverage = F.interpolate(
            coverage.unsqueeze(0),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    prediction_luma = _luminance(prediction)
    gt_luma = _luminance(bright_gt)
    coverage_2d = _coverage_to_2d(coverage)

    bright_gt_mask = gt_luma > gt_bright_threshold
    black_artifact_mask = bright_gt_mask & (prediction_luma < prediction_black_threshold)
    low_coverage_mask = coverage_2d < low_coverage_threshold
    artifact_low_coverage_mask = black_artifact_mask & low_coverage_mask

    num_pixels = int(prediction_luma.numel())
    bright_gt_pixels = int(bright_gt_mask.sum().item())
    black_artifact_pixels = int(black_artifact_mask.sum().item())
    low_coverage_pixels = int(low_coverage_mask.sum().item())
    artifact_low_coverage_pixels = int(artifact_low_coverage_mask.sum().item())
    if artifact_low_coverage_pixels > black_artifact_pixels:
        raise RuntimeError("Artifact/coverage intersection cannot exceed artifact count.")

    return {
        "num_pixels": num_pixels,
        "bright_gt_pixels": bright_gt_pixels,
        "black_artifact_pixels": black_artifact_pixels,
        "low_coverage_pixels": low_coverage_pixels,
        "artifact_low_coverage_pixels": artifact_low_coverage_pixels,
        "bright_gt_ratio": bright_gt_pixels / max(1, num_pixels),
        "black_artifact_ratio": black_artifact_pixels / max(1, bright_gt_pixels),
        "black_artifact_image_ratio": black_artifact_pixels / max(1, num_pixels),
        "low_coverage_ratio": low_coverage_pixels / max(1, num_pixels),
        "artifact_low_coverage_overlap_ratio": (
            artifact_low_coverage_pixels / max(1, black_artifact_pixels)
        ),
    }


def summarize_artifact_diagnostics(per_view: dict) -> dict:
    """Aggregate per-view artifact counts into dataset-level ratios."""
    if not per_view:
        return {}

    count_keys = (
        "num_pixels",
        "bright_gt_pixels",
        "black_artifact_pixels",
        "low_coverage_pixels",
        "artifact_low_coverage_pixels",
    )
    totals = {
        key: sum(int(item[key]) for item in per_view.values())
        for key in count_keys
    }
    totals.update(
        {
            "bright_gt_ratio": totals["bright_gt_pixels"] / max(1, totals["num_pixels"]),
            "black_artifact_ratio": (
                totals["black_artifact_pixels"] / max(1, totals["bright_gt_pixels"])
            ),
            "black_artifact_image_ratio": (
                totals["black_artifact_pixels"] / max(1, totals["num_pixels"])
            ),
            "low_coverage_ratio": (
                totals["low_coverage_pixels"] / max(1, totals["num_pixels"])
            ),
            "artifact_low_coverage_overlap_ratio": (
                totals["artifact_low_coverage_pixels"]
                / max(1, totals["black_artifact_pixels"])
            ),
        }
    )
    return totals


def compute_detail_diagnostics(
    prediction: torch.Tensor,
    bright_gt: torch.Tensor,
    coverage: torch.Tensor,
    edge_threshold: float = 0.02,
    retained_edge_ratio: float = 0.5,
    low_coverage_threshold: float = 0.95,
) -> dict:
    """Measure how much bright-GT edge structure survives enhancement."""
    if bright_gt.shape[-2:] != prediction.shape[-2:]:
        bright_gt = F.interpolate(
            bright_gt.unsqueeze(0),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if coverage.shape[-2:] != prediction.shape[-2:]:
        coverage = F.interpolate(
            coverage.unsqueeze(0),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    prediction_grad = _log_luminance_gradient(prediction)
    gt_grad = _log_luminance_gradient(bright_gt)
    valid_mask = _coverage_to_2d(coverage) >= low_coverage_threshold
    gt_edge_mask = (gt_grad >= edge_threshold) & valid_mask
    retained_mask = gt_edge_mask & (
        prediction_grad >= retained_edge_ratio * gt_grad
    )

    gt_edge_pixels = int(gt_edge_mask.sum().item())
    retained_edge_pixels = int(retained_mask.sum().item())
    gt_edge_strength = float((gt_grad * gt_edge_mask).sum().item())
    prediction_edge_strength = float(
        (prediction_grad * gt_edge_mask).sum().item()
    )
    edge_recall = retained_edge_pixels / max(1, gt_edge_pixels)
    return {
        "num_pixels": int(prediction_grad.numel()),
        "gt_edge_pixels": gt_edge_pixels,
        "retained_edge_pixels": retained_edge_pixels,
        "gt_edge_strength": gt_edge_strength,
        "prediction_edge_strength": prediction_edge_strength,
        "edge_recall": edge_recall,
        "edge_miss_ratio": 1.0 - edge_recall,
        "edge_strength_ratio": prediction_edge_strength / max(1e-8, gt_edge_strength),
    }


def summarize_detail_diagnostics(per_view: dict) -> dict:
    """Aggregate edge-retention diagnostics across all evaluated views."""
    if not per_view:
        return {}
    count_keys = ("num_pixels", "gt_edge_pixels", "retained_edge_pixels")
    totals = {
        key: sum(int(item[key]) for item in per_view.values())
        for key in count_keys
    }
    totals["gt_edge_strength"] = sum(
        float(item["gt_edge_strength"]) for item in per_view.values()
    )
    totals["prediction_edge_strength"] = sum(
        float(item["prediction_edge_strength"]) for item in per_view.values()
    )
    totals["edge_recall"] = (
        totals["retained_edge_pixels"] / max(1, totals["gt_edge_pixels"])
    )
    totals["edge_miss_ratio"] = 1.0 - totals["edge_recall"]
    totals["edge_strength_ratio"] = (
        totals["prediction_edge_strength"] / max(1e-8, totals["gt_edge_strength"])
    )
    return totals
