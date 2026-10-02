#!/usr/bin/env python3
"""Collect enhanced-image metrics and render FPS for the sharpen ablation."""

import argparse
import json
from pathlib import Path
from statistics import mean


SCENES = ("buu", "chair", "sofa")
VARIANTS = (
    "a0_baseline",
    "no_contrast",
    "minimal_structure",
    "no_highfreq",
    "no_structure_all",
    "no_highlight",
    "no_tiny_regs",
    "isolate_off",
    "asg_lobes_2",
)
METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_result(output_root: Path, scene: str, variant: str, iteration: int) -> dict:
    experiment = output_root / f"{scene}_sharpen_{variant}_8k"
    result_dir = experiment / "test" / f"ours_{iteration}"
    metric_path = result_dir / "metrics_enhanced_gt.json"
    timing_path = result_dir / "render_timing_profile.json"

    metric_summary = load_json(metric_path).get("summary", {})
    timing_summary = load_json(timing_path).get("summary", {})
    missing_metrics = [key for key in METRIC_KEYS if key not in metric_summary]
    if missing_metrics:
        raise KeyError(f"Missing metrics {missing_metrics} in {metric_path}")
    if "render_core_time_fps" not in timing_summary:
        raise KeyError(f"Missing render_core_time_fps in {timing_path}")

    return {
        "PSNR": float(metric_summary["PSNR"]),
        "SSIM": float(metric_summary["SSIM"]),
        "LPIPS": float(metric_summary["LPIPS"]),
        "FPS": float(timing_summary["render_core_time_fps"]),
        "measured_views": int(timing_summary.get("measured_views", 0)),
        "experiment": str(experiment),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs", type=Path)
    parser.add_argument("--scene", default="all", choices=("all", *SCENES))
    parser.add_argument("--iteration", default=8000, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected_scenes = SCENES if args.scene == "all" else (args.scene,)
    output_path = args.output
    if output_path is None:
        suffix = "" if args.scene == "all" else f"_{args.scene}"
        output_path = args.outputs_root / f"sharpen_ablation_results{suffix}.json"

    scene_results = {
        scene: {
            variant: collect_result(args.outputs_root, scene, variant, args.iteration)
            for variant in VARIANTS
        }
        for scene in selected_scenes
    }
    averages = {
        variant: {
            key: mean(scene_results[scene][variant][key] for scene in selected_scenes)
            for key in (*METRIC_KEYS, "FPS")
        }
        for variant in VARIANTS
    }

    payload = {
        "protocol": {
            "iteration": args.iteration,
            "metric_source": "metrics_enhanced_gt.json",
            "fps_source": "render_timing_profile.json:summary.render_core_time_fps",
            "scenes": list(selected_scenes),
            "variants": list(VARIANTS),
        },
        "scenes": scene_results,
        "average_across_scenes": averages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Sharpen ablation summary saved to: {output_path}")
    for scene in selected_scenes:
        print(f"\n[{scene}] enhanced_gt metrics (PSNR / SSIM / LPIPS / FPS)")
        for variant in VARIANTS:
            row = scene_results[scene][variant]
            print(
                f"  {variant:>18}: {row['PSNR']:.4f} / {row['SSIM']:.4f} / "
                f"{row['LPIPS']:.4f} / {row['FPS']:.1f}"
            )


if __name__ == "__main__":
    main()
