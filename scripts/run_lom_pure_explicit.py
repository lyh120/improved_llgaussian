"""Run the five canonical LOM scenes with explicit R and ASG/SG illumination.

The recipe follows the previously successful 6000-step buu baseline. Outputs
are kept separate from legacy experiments under experiments/lom_pure_explicit.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("bike", "buu", "chair", "shrub", "sofa")


def command_for(scene, gpu, port):
    source = ROOT / "datasets" / scene
    model = ROOT / "experiments" / "lom_pure_explicit" / scene
    if not (source / "images").is_dir() or not (source / "sparse").is_dir():
        raise FileNotFoundError(f"Prepared LOM scene missing: {source}")
    return [
        sys.executable, str(ROOT / "train.py"), "-s", str(source), "-m", str(model),
        "--port", str(port), "--eval", "--gpu", str(gpu), "--warmup", "--use_3D_filter",
        "--pure_explicit_rl", "--use_asg_illumination", "--illumination_mode", "asg",
        "--asg_lobes", "1", "--asg_lambda_min", "1.0",
        "--asg_energy_reg", "1e-4", "--asg_sharpness_reg", "5e-5",
        "--asg_anisotropy_reg", "1e-5",
        "--iterations", "6000", "--save_iterations", "4000", "4500", "5000", "6000",
        "--test_iterations", "4000", "4500", "5000", "6000",
        "--position_lr_max_steps", "12000", "--offset_lr_max_steps", "12000",
        "--mlp_opacity_lr_max_steps", "12000", "--mlp_cov_lr_max_steps", "12000",
        "--mlp_color_lr_max_steps", "12000", "--mlp_color_lr_init", "0.008",
        "--mlp_color_lr_final", "0.00025", "--mlp_enhance_lr_init", "0.02",
        "--mlp_enhance_lr_final", "0.00025", "--offset_lr_init", "0.001",
        "--offset_lr_final", "0.00001", "--voxel_size", "0.0005",
        "--prune_ratio", "1.0", "--feat_dim", "32", "--start_stat", "200",
        "--update_from", "800", "--update_until", "8500", "--update_interval", "50",
        "--success_threshold", "0.75", "--densify_grad_threshold", "0.00015",
        "--min_opacity", "0.002", "--max_anchors", "90000",
        "--max_new_anchors_per_update", "640", "--densify_level_caps", "320,200,120",
        "--prune_from_iter", "6000", "--max_pruned_anchors_per_update", "128",
        "--anchor_prune_grace_iters", "1200", "--warmup_start_stat", "200",
        "--warmup_update_from", "600", "--warmup_update_until", "1900",
        "--warmup_update_interval", "100", "--warmup_max_new_anchors", "384",
        "--warmup_level_caps", "192,120,72", "--warmup_densify_grad_threshold", "0.00015",
        "--warmup_success_threshold", "0.7", "--illumination_smooth_reg", "1e-4",
        "--illumination_smooth_kernel_size", "5", "--warmup_illumination_smooth_reg", "5e-5",
        "--warmup_illumination_smooth_kernel_size", "9",
        "--reflectance_consistency_reg", "2e-5", "--reflectance_smooth_reg", "0.0",
        "--reflectance_edge_reg", "2e-4", "--reflectance_edge_uplift_reg", "3e-3",
        "--reflectance_contrast_reg", "2e-3", "--reflectance_highfreq_reg", "3e-3",
        "--highlight_reflectance_reg", "1e-3", "--reflectance_detail_reg", "1e-6",
        "--reflectance_decoder_reg", "0.0", "--reflectance_offset_lr", "0.008",
        "--b0_spatial_smooth_reg", "0.0", "--enhancement_diff_start_iter", "3500",
        "--enhancement_guidance_ramp_iters", "1500", "--enhancement_smooth_reg", "0.0",
        "--enhancement_gain_smooth_reg", "5e-5", "--enhancement_edge_preserve_reg", "0.02",
        "--enhancement_guidance_final_weight", "0.75",
        "--enhancement_reflectance_reg", "0.0", "--enhancement_degree_reg", "0.12",
        "--enhancement_degree_global_reg", "0.02", "--enhancement_color_reg", "0.03",
        "--enhancement_color_std_reg", "0.01", "--enhancement_green_bias_reg", "0.03",
        "--enhancement_grad_clip", "1.0", "--enhancement_prior", "cidnet",
        "--cidnet_conda_env", "CIDNet", "--cidnet_root", "./submodules/HVI-CIDNet",
        "--cidnet_weights", "./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth",
        "--cidnet_refresh_interval", "0", "--cidnet_mlp_steps", "100",
        "--cidnet_target_exposure", "0.5", "--cidnet_refresh_reg", "0.5",
        "--cidnet_color_reg", "0.2", "--cidnet_param_reg", "0.1", "--cidnet_mv_reg", "0.5",
        "--cidnet_alpha_init", "1.2", "--cidnet_gamma_init", "1.2",
        "--wandb_monitor_camera", "1", "--wandb_monitor_split", "test",
        "--wandb_monitor_interval", "300",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--port", type=int, default=45720)
    parser.add_argument("--explicit-feature-conditioning", action="store_true")
    args = parser.parse_args()
    if not args.scenes:
        args.scenes = list(SCENES)
    for scene in args.scenes:
        if scene not in SCENES:
            parser.error(f"Unknown LOM scene {scene}; choose from {', '.join(SCENES)}")
    for index, scene in enumerate(args.scenes):
        command = command_for(scene, args.gpu, args.port + index)
        if args.explicit_feature_conditioning:
            command.append("--explicit_feature_conditioning")
        model = ROOT / "experiments" / "lom_pure_explicit" / scene
        results = model / "results.json"
        if results.exists() and "ours_6000" in json.loads(results.read_text()):
            print(f"[skip] {scene}: completed", flush=True)
            continue
        if model.exists() and any(model.iterdir()):
            raise FileExistsError(f"Existing incomplete experiment: {model}")
        model.mkdir(parents=True, exist_ok=True)
        print(f"[start] {scene}: {' '.join(command)}", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["WANDB_MODE"] = "offline"
        with (model / "runner.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"[done] {scene}: {results.read_text()}", flush=True)


if __name__ == "__main__":
    main()
