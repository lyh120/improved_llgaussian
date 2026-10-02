"""Run an LLNeRF scene with corrected cameras and scene-scaled densification.

Examples:
  python scripts/run_llnerf.py room --gpu 1 --run-tag cap4
  python scripts/run_llnerf.py shrub --gpu 1 --run-tag cap8 --max-gaussian-anisotropy 8
  python scripts/run_llnerf.py room --gpu 1 --run-tag selected --enhanced-dir /path/to/2d/images
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_llnerf_scene import undistort_image
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--source-root", type=Path,
                        default=Path("/home/liuyuhao/datasets/llnerf-dataset"))
    parser.add_argument("--prepared-root", type=Path,
                        default=ROOT / "datasets" / "llnerf_prepared")
    parser.add_argument("--experiment-root", type=Path,
                        default=ROOT / "experiments" / "llnerf")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--port", default="45717")
    parser.add_argument("--run-tag", help="Name for an independent model and CIDNet cache")
    parser.add_argument("--iterations", type=int, default=6000,
                        help="Main training iterations, in addition to the warmup stage")
    parser.add_argument("--cidnet-gamma", type=float, default=0.9)
    parser.add_argument("--cidnet-alpha", type=float, default=0.9)
    parser.add_argument("--cidnet-mlp-steps", type=int, default=0)
    parser.add_argument("--max-gaussian-anisotropy", type=float, default=4.0)
    parser.add_argument("--reflectance-init-floor", type=float,
                        help="Floor for estimating initial reflectance from dark images; default: half of scene image q90")
    parser.add_argument("--depth-prior-gamma", type=float,
                        help="Gamma for Depth Anything inputs; default: lift scene image q90 to 0.35")
    parser.add_argument("--reflectance-texture-scale", type=float, default=0.0,
                        help="Scale reflectance edge uplift, contrast and high-frequency losses (default: 0 for noisy LLNeRF images)")
    parser.add_argument("--enhancement-reflectance-reg", type=float, default=0.0,
                        help="Route part of CIDNet image guidance into reflectance and geometry")
    parser.add_argument("--enhancement-target-edge-reg", type=float, default=0.0,
                        help="Match coherent CIDNet edges through reflectance and geometry")
    parser.add_argument("--reflectance-target-detail-reg", type=float, default=0.0,
                        help="Transfer local CIDNet texture into R with geometry detached")
    parser.add_argument("--reflectance-target-chroma-reg", type=float, default=0.0,
                        help="Align R material chromaticity with the selected 2D target")
    parser.add_argument("--enhancement-illumination-chroma-reg", type=float, default=0.0,
                        help="Discourage spatially varying color in enhanced illumination")
    parser.add_argument("--enhancement-degree-reg", type=float, default=0.12,
                        help="Weight tying enhanced illumination to scaled low-light illumination")
    parser.add_argument("--enhancement-degree-global-reg", type=float, default=0.02)
    parser.add_argument("--high-detail", action="store_true",
                        help="Use a finer voxel grid and larger, more frequent anchor growth")
    parser.add_argument("--max-anchors", type=int,
                        help="Override the anchor budget for longer high-detail training")
    parser.add_argument("--update-until", type=int,
                        help="Override the final anchor densification iteration")
    parser.add_argument("--enhancement-guidance-final-weight", type=float, default=0.75,
                        help="Final pixel guidance weight from the selected 2D enhancement")
    parser.add_argument("--direct-composition", action="store_true",
                        help="Train and render with a single rasterization of R times L")
    parser.add_argument("--pure-explicit-rl", action="store_true",
                        help="Use explicit offset reflectance and ASG/SG lighting without an R decoder or feature gate")
    parser.add_argument("--explicit-feature-conditioning", action="store_true",
                        help="Condition explicit R and enhancement on Scaffold-GS features")
    parser.add_argument("--enhanced-dir", type=Path)
    parser.add_argument("--enhanced-already-undistorted", action="store_true")
    parser.add_argument("--input-dir", type=Path,
                        help="Already undistorted images for direct bright-image reconstruction; must include every registered view")
    parser.add_argument("--pilot", action="store_true",
                        help="Short geometry and render trial (1200 iterations, no warmup)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stage-only", action="store_true",
                        help="Build the per-run scene and selected 2D targets without training")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.prepared_root = args.prepared_root.resolve()
    args.experiment_root = args.experiment_root.resolve()
    if args.enhanced_dir:
        args.enhanced_dir = args.enhanced_dir.resolve()
    if args.input_dir:
        args.input_dir = args.input_dir.resolve()
    if args.reflectance_texture_scale < 0:
        raise ValueError("--reflectance-texture-scale must be nonnegative")
    if any(value < 0 for value in (args.enhancement_reflectance_reg,
                                   args.enhancement_target_edge_reg,
                                   args.reflectance_target_detail_reg,
                                   args.reflectance_target_chroma_reg,
                                   args.enhancement_illumination_chroma_reg,
                                   args.enhancement_degree_reg,
                                   args.enhancement_degree_global_reg)):
        raise ValueError("Enhancement regularization weights must be nonnegative")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.max_anchors is not None and args.max_anchors < 1:
        raise ValueError("--max-anchors must be positive")
    if args.update_until is not None and not 1 <= args.update_until < args.iterations:
        raise ValueError("--update-until must be between 1 and iterations - 1")
    if args.enhancement_guidance_final_weight < 0:
        raise ValueError("--enhancement-guidance-final-weight must be nonnegative")
    if args.reflectance_init_floor is not None and args.reflectance_init_floor <= 0:
        raise ValueError("--reflectance-init-floor must be positive")
    if args.depth_prior_gamma is not None and args.depth_prior_gamma <= 0:
        raise ValueError("--depth-prior-gamma must be positive")

    prepared = args.prepared_root / args.scene
    metadata_path = prepared / "preparation.json"
    if not metadata_path.exists():
        prepare = [sys.executable, str(ROOT / "scripts" / "prepare_llnerf_scene.py"),
                   args.scene, "--source-root", str(args.source_root),
                   "--output-root", str(args.prepared_root)]
        if args.dry_run:
            print("PREPARE", " ".join(prepare))
            return
        subprocess.run(prepare, cwd=ROOT, check=True)
    metadata = json.loads(metadata_path.read_text())
    expected_source = str((args.source_root / args.scene).resolve())
    if metadata["source"] != expected_source:
        raise ValueError(f"Prepared source differs from {expected_source}")
    if args.prepare_only:
        print(metadata_path)
        return

    tag = args.run_tag or ("pilot" if args.pilot else "asg")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("--run-tag must use letters, digits, underscores or hyphens")
    model = args.experiment_root / (args.scene + "_" + tag)
    if model.exists() and any(model.iterdir()):
        raise FileExistsError(f"Refusing to reuse experiment directory {model}")
    scene_data = args.prepared_root / "_runs" / (args.scene + "_" + tag)
    if scene_data.exists() and any(scene_data.iterdir()):
        raise FileExistsError(f"Refusing to reuse scene data directory {scene_data}")
    input_images = args.input_dir or (prepared / "images")
    if not args.dry_run:
        scene_data.mkdir(parents=True)
        if args.input_dir:
            for reference in (prepared / "images").iterdir():
                if reference.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                matches = list(input_images.glob(reference.stem + ".*"))
                if len(matches) != 1:
                    raise ValueError(f"Expected one input image for {reference.stem}")
                with Image.open(reference) as original, Image.open(matches[0]) as selected:
                    if selected.size != original.size:
                        raise ValueError(f"Input size differs for {reference.stem}")
        (scene_data / "images").symlink_to(input_images, target_is_directory=True)
        (scene_data / "sparse").symlink_to(prepared / "sparse", target_is_directory=True)
        if args.enhanced_dir:
            sparse = args.source_root / args.scene / "sparse" / "0"
            camera = next(iter(read_intrinsics_binary(str(sparse / "cameras.bin")).values()))
            registered = read_extrinsics_binary(str(sparse / "images.bin"))
            names = sorted(Path(item.name).stem for item in registered.values())
            targets = scene_data / "cidnet_prior" / "round_000"
            image_dir = targets / "images"
            image_dir.mkdir(parents=True)
            for index, name in enumerate(names):
                if index % 8 == 0:
                    continue
                matches = list(args.enhanced_dir.glob(name + ".*"))
                if len(matches) != 1:
                    raise ValueError(f"Expected one selected 2D image for {name}")
                destination = image_dir / (name + ".png")
                if args.enhanced_already_undistorted:
                    selected_image = Image.open(matches[0]).convert("RGB")
                    if selected_image.size != (camera.width, camera.height):
                        raise ValueError(f"{matches[0]}: size disagrees with cameras.bin")
                    selected_image.save(destination)
                else:
                    undistort_image(matches[0], destination, camera)
            (targets / "params.json").write_text('{"round": 0, "images": {}}\n')
    if args.stage_only:
        print(scene_data)
        return
    voxel_size = metadata["recommended_voxel_size"] * (0.5 if args.high_detail else 1.0)
    image_files = sorted(path for path in input_images.iterdir()
                         if path.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not image_files:
        raise ValueError(f"No prepared images in {prepared / 'images'}")
    image = np.asarray(Image.open(image_files[min(1, len(image_files) - 1)]).convert("RGB"), dtype=np.float32) / 255.0
    image_q90 = float(np.quantile(image.max(axis=2), 0.9))
    if args.reflectance_init_floor is None:
        reflectance_init_floor = max(1.0 / 255.0, min(0.1, 0.5 * image_q90))
    else:
        reflectance_init_floor = args.reflectance_init_floor
    depth_prior_gamma = (float(np.clip(np.log(0.35) / np.log(np.clip(image_q90, 1e-6, 0.999)), 0.2, 1.0))
                         if args.depth_prior_gamma is None else args.depth_prior_gamma)
    iterations = 1200 if args.pilot else args.iterations
    checkpoint_iterations = ([4000, 5000, 6000, 9000, 14000, 17000, iterations]
                             if args.high_detail and iterations > 6000 else
                             [4000, 5000, iterations]
                             if args.high_detail and not args.pilot else [iterations])
    checkpoint_iterations = sorted({step for step in checkpoint_iterations if step <= iterations})
    evaluation_iterations = [6000, iterations] if iterations > 6000 else [iterations]
    update_until = (1200 if args.pilot else
                    min(8500, iterations - 500) if args.high_detail else
                    min(5000, iterations - 500))
    if args.update_until is not None:
        update_until = args.update_until
    command = [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(scene_data), "-m", str(model),
        "--port", args.port, "--eval", "--gpu", args.gpu,
        "--use_3D_filter", "--disable_reflectance_grad_isolation",
        "--use_asg_illumination", "--illumination_mode", "asg",
        "--asg_lobes", "1", "--asg_lambda_min", "1.0",
        "--asg_energy_reg", "1e-4", "--asg_sharpness_reg", "5e-5",
        "--asg_anisotropy_reg", "1e-5",
        "--iterations", str(iterations),
        "--save_iterations", *(str(step) for step in checkpoint_iterations),
        "--test_iterations", *(str(step) for step in evaluation_iterations),
        "--position_lr_max_steps", str(max(12000, iterations)),
        "--offset_lr_max_steps", str(max(12000, iterations)),
        "--mlp_opacity_lr_max_steps", str(max(12000, iterations)),
        "--mlp_cov_lr_max_steps", str(max(12000, iterations)),
        "--mlp_color_lr_max_steps", str(max(12000, iterations)),
        "--mlp_color_lr_init", "0.008", "--mlp_color_lr_final", "0.00025",
        "--mlp_enhance_lr_init", "0.02", "--mlp_enhance_lr_final", "0.00025",
        "--offset_lr_init", "0.001", "--offset_lr_final", "0.00001",
        "--voxel_size", str(voxel_size), "--prune_ratio", "1.0", "--feat_dim", "32",
        "--max_gaussian_anisotropy", str(args.max_gaussian_anisotropy),
        "--reflectance_init_floor", str(reflectance_init_floor),
        "--depth_prior_gamma", str(depth_prior_gamma),
        "--start_stat", "200", "--update_from", "800",
        "--update_until", str(update_until),
        "--update_interval", "50" if args.high_detail and not args.pilot else "100",
        "--success_threshold", "0.75", "--densify_grad_threshold", "0.00015",
        "--min_opacity", "0.002", "--max_anchors", str(args.max_anchors or (90000 if args.high_detail else 60000)),
        "--max_new_anchors_per_update", "640" if args.high_detail else "256",
        "--densify_level_caps", "320,200,120" if args.high_detail else "128,80,48",
        "--prune_from_iter", "6000" if args.high_detail else "3000",
        "--max_pruned_anchors_per_update", "128",
        "--anchor_prune_grace_iters", "1200",
        "--warmup_start_stat", "200", "--warmup_update_from", "600",
        "--warmup_update_until", "1900", "--warmup_update_interval", "100",
        "--warmup_max_new_anchors", "384" if args.high_detail else "192",
        "--warmup_level_caps", "192,120,72" if args.high_detail else "96,60,36",
        "--warmup_densify_grad_threshold", "0.00015",
        "--warmup_success_threshold", "0.7",
        "--illumination_smooth_reg", "1e-4", "--illumination_smooth_kernel_size", "5",
        "--warmup_illumination_smooth_reg", "5e-5",
        "--warmup_illumination_smooth_kernel_size", "9",
        "--reflectance_consistency_reg", "2e-5", "--reflectance_smooth_reg", "0.0",
        "--reflectance_edge_reg", "2e-4",
        "--reflectance_edge_uplift_reg", str(3e-3 * args.reflectance_texture_scale),
        "--reflectance_contrast_reg", str(2e-3 * args.reflectance_texture_scale),
        "--reflectance_highfreq_reg", str(3e-3 * args.reflectance_texture_scale),
        "--highlight_reflectance_reg", "1e-3", "--reflectance_detail_reg", "1e-6",
        "--reflectance_decoder_reg", "2e-5", "--reflectance_offset_lr", "0.008",
        "--reflectance_decoder_lr", "0.002", "--b0_spatial_smooth_reg", "0.0",
        "--enhancement_diff_start_iter", "3500",
        "--enhancement_guidance_ramp_iters", "1500",
        "--enhancement_smooth_reg", "0.0", "--enhancement_gain_smooth_reg", "5e-5",
        "--enhancement_edge_preserve_reg", "0.02",
        "--enhancement_guidance_final_weight", str(args.enhancement_guidance_final_weight),
        "--enhancement_reflectance_reg", str(args.enhancement_reflectance_reg),
        "--enhancement_target_edge_reg", str(args.enhancement_target_edge_reg),
        "--reflectance_target_detail_reg", str(args.reflectance_target_detail_reg),
        "--reflectance_target_chroma_reg", str(args.reflectance_target_chroma_reg),
        "--enhancement_illumination_chroma_reg", str(args.enhancement_illumination_chroma_reg),
        "--enhancement_degree_reg", str(args.enhancement_degree_reg),
        "--enhancement_degree_global_reg", str(args.enhancement_degree_global_reg),
        "--enhancement_color_reg", "0.03", "--enhancement_color_std_reg", "0.01",
        "--enhancement_green_bias_reg", "0.03", "--enhancement_grad_clip", "1.0",
        "--enhancement_prior", "cidnet", "--cidnet_conda_env", "CIDNet",
        "--cidnet_root", "./submodules/HVI-CIDNet",
        "--cidnet_weights", "./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth",
        "--cidnet_refresh_interval", "0", "--cidnet_mlp_steps", str(args.cidnet_mlp_steps),
        "--cidnet_target_exposure", "0.5", "--cidnet_refresh_reg", "0.5",
        "--cidnet_color_reg", "0.2", "--cidnet_param_reg", "0.1",
        "--cidnet_mv_reg", "0.5", "--cidnet_alpha_init", str(args.cidnet_alpha),
        "--cidnet_gamma_init", str(args.cidnet_gamma),
        "--wandb_monitor_camera", "1", "--wandb_monitor_split", "test",
        "--wandb_monitor_interval", "300",
    ]
    if not args.pilot:
        command.append("--warmup")
    if args.wandb:
        command.append("--use_wandb")
    if args.direct_composition:
        command.append("--direct_composition")
    if args.pure_explicit_rl:
        command.append("--pure_explicit_rl")
    if args.explicit_feature_conditioning:
        command.append("--explicit_feature_conditioning")
    print(" ".join(command), flush=True)
    if not args.dry_run:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
