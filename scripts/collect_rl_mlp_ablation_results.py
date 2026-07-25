#!/usr/bin/env python3
"""Collect test enhanced-image metrics and core FPS for R/L MLP ablations."""

import argparse
import json
from pathlib import Path
from statistics import mean


SCENES = ("bike", "buu", "chair", "sofa")
VARIANTS = ("no_residual", "mlp_r", "mlp_l", "mlp_rl")
METRICS = ("PSNR", "SSIM", "LPIPS", "FPS")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing test result: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_variant(root: Path, scene: str, variant: str, iteration: int) -> dict:
    result_dir = root / f"{scene}_ablation_{variant}_8k" / "test" / f"ours_{iteration}"
    metrics = load_json(result_dir / "metrics_enhanced_gt.json")["summary"]
    timing = load_json(result_dir / "render_timing_profile.json")["summary"]
    return {
        "PSNR": float(metrics["PSNR"]),
        "SSIM": float(metrics["SSIM"]),
        "LPIPS": float(metrics["LPIPS"]),
        "FPS": float(timing["render_core_time_fps"]),
        "measured_views": int(timing["measured_views"]),
        "experiment": str(root / f"{scene}_ablation_{variant}_8k"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--scene", choices=("all", *SCENES), default="all")
    parser.add_argument("--iteration", type=int, default=8000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scenes = SCENES if args.scene == "all" else (args.scene,)
    results = {scene: {variant: read_variant(args.outputs_root, scene, variant, args.iteration) for variant in VARIANTS} for scene in scenes}
    averages = {variant: {metric: mean(results[scene][variant][metric] for scene in scenes) for metric in METRICS} for variant in VARIANTS}
    output = args.output or args.outputs_root / ("rl_mlp_ablation_results.json" if args.scene == "all" else f"rl_mlp_ablation_results_{args.scene}.json")
    payload = {
        "protocol": {
            "split": "test",
            "metric_source": "metrics_enhanced_gt.json",
            "fps_source": "render_timing_profile.json:summary.render_core_time_fps",
            "scenes": list(scenes),
            "variants": list(VARIANTS),
        },
        "scenes": results,
        "average_across_scenes": averages,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Saved test-only R/L ablation summary: {output}")


if __name__ == "__main__":
    main()
