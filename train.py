"""Training entry point for MODEL_FORMAT_VERSION=2 explicit R/L."""

from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    validate_training_schedule,
)
from gaussian_renderer import prefilter_voxel, render
from render import render_sets
from scene import Scene
from scene.explicit_appearance import decompose_enhanced_prior
from utils.model_format import MODEL_FORMAT_VERSION, training_stage_state
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import (
    coverage_masked_prediction,
    edge_aware_illumination_tv_loss,
    inverse_depth_pearson_loss,
    photo_loss,
    retinex_targets,
    scaling_loss,
    ssim,
)
from utils.model_format import NUMERICAL_EPS
from utils.pose_utils import get_tensor_from_camera, save_pose
from utils.prior_utils import (
    prior_generation_required,
    resolve_required_file,
    stablesr_prior_directory_name,
)

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


WANDB_CORE_IMAGE_FIELDS = {
    "image": "render",
    "image_enhanced": "render_enhanced",
    "reflectance": "render_reflectance",
    "illumination": "render_illumination",
    "illumination_enhanced": "render_illumination_enhanced",
}

SCENE_ENHANCEMENT_PRIOR_DIRECTORY = "cidnet_prior"
SCENE_DEPTH_PRIOR_DIRECTORY = "depth_maps"
ENHANCEMENT_PRIOR_BACKENDS = {"cidnet", "stablesr"}
# The clean LL-Gaussian monitor exposes dark renders by mapping the scene's
# mean training brightness to a visible mid-tone. This constant is display
# only: it never enters a prior, appearance evaluation, target, or loss.
WANDB_DISPLAY_TARGET_MEAN = 0.45


def _training_split_brightness(dataset) -> float:
    """Return mean RGB brightness using the prior preprocessing split."""
    from scripts.precompute_depth_prior import _images_by_name, _select_training_images

    images = _select_training_images(
        _images_by_name(Path(dataset.source_path) / dataset.images),
        bool(dataset.eval),
        int(dataset.lod),
        8,
    )
    if not images:
        raise RuntimeError(
            "No training-split images found at "
            f"{os.path.join(dataset.source_path, dataset.images)}"
        )
    total = 0.0
    for path in images.values():
        with Image.open(path) as image:
            total += float(pil_to_tensor(image.convert("RGB")).float().mean()) / 255.0
    return total / len(images)


def _wandb_display_gain(mean_brightness: float) -> float:
    """Match the clean-reference dark-map display transform, for W&B only."""
    return float(
        max(
            1,
            int(WANDB_DISPLAY_TARGET_MEAN / max(mean_brightness, NUMERICAL_EPS)),
        )
    )


@torch.no_grad()
def _scene_display_gain(cameras) -> tuple[float, float]:
    """Compute the W&B-only display gain from already-built cameras."""
    if not cameras:
        raise RuntimeError("Scene enhance ratio requires at least one training camera")
    means = torch.zeros(3, device=cameras[0].original_image.device)
    for camera in cameras:
        means += camera.original_image.mean(dim=(1, 2))
    brightness = float((means / len(cameras)).mean().item())
    return _wandb_display_gain(brightness), brightness


def _start_training_progress(phase: str, next_iteration: int, total_iterations: int):
    """Start one progress bar per training phase for a visible warmup/main split."""
    return tqdm(
        total=max(total_iterations, 0),
        initial=min(max(next_iteration, 0), max(total_iterations, 0)),
        desc=f"Training [{phase}]",
    )


def create_gaussian_model(dataset) -> GaussianModel:
    return GaussianModel(
        feat_dim=dataset.feat_dim,
        n_offsets=dataset.n_offsets,
        voxel_size=dataset.voxel_size,
        update_depth=dataset.update_depth,
        update_init_factor=dataset.update_init_factor,
        update_hierachy_factor=dataset.update_hierachy_factor,
        use_feat_bank=dataset.use_feat_bank,
        add_opacity_dist=dataset.add_opacity_dist,
        add_cov_dist=dataset.add_cov_dist,
        use_3D_filter=dataset.use_3D_filter,
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output", str(uuid.uuid4())[:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as handle:
        handle.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


def get_logger(path):
    logger = logging.getLogger(f"ll-gaussian-v2:{os.path.abspath(path)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    for handler in (logging.FileHandler(os.path.join(path, "outputs.log")), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _stablesr_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    candidate = os.path.join(root, path)
    return candidate if os.path.exists(candidate) else _project_path(path)


def _default_enhancement_prior_directory(dataset, input_gain: float) -> str:
    if dataset.enhancement_prior_backend == "cidnet":
        return SCENE_ENHANCEMENT_PRIOR_DIRECTORY
    # StableSR caches are scene-local and keyed by their explicit input gain.
    return stablesr_prior_directory_name(input_gain)


def _run_prior_command(command: list[str], output: str, prior_name: str, logger) -> None:
    logger.info("Generating fixed %s prior once at %s", prior_name, output)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    manifest = os.path.join(output, "manifest.json")
    if not os.path.isfile(manifest):
        raise RuntimeError(f"{prior_name} preprocessing completed without creating {manifest}")


def _ensure_training_priors(dataset, logger) -> None:
    """Reuse scene-local v2 priors or generate each missing cache exactly once.

    Loading order follows the clean reference LL-Gaussian: the DepthAnything
    prior is prepared first, the enhancement prior second, and an existing
    cache is loaded without regenerating anything.
    """
    backend = dataset.enhancement_prior_backend.lower()
    if backend not in ENHANCEMENT_PRIOR_BACKENDS:
        raise ValueError(
            f"enhancement_prior_backend must be one of {sorted(ENHANCEMENT_PRIOR_BACKENDS)}, "
            f"got {dataset.enhancement_prior_backend!r}"
        )
    dataset.enhancement_prior_backend = backend
    dataset.depth_prior_path = os.path.abspath(
        dataset.depth_prior_path
        or os.path.join(dataset.source_path, SCENE_DEPTH_PRIOR_DIRECTORY)
    )
    input_gain = float(dataset.stablesr_input_gain)
    if backend == "stablesr" and input_gain <= 0.0:
        raise ValueError("--stablesr_input_gain must be explicitly greater than zero")
    # Brightness is observed only for the clean-reference W&B display. It is
    # deliberately not reused as a training target or StableSR input gain.
    scene_brightness = _training_split_brightness(dataset)
    dataset.scene_mean_brightness = scene_brightness
    dataset.wandb_display_gain = _wandb_display_gain(scene_brightness)
    if backend == "stablesr":
        logger.info("[prior] StableSR input gain is explicit: %.4f", input_gain)
    logger.info(
        "[INFO] W&B-only clean-reference display: scene mean %.5f, gain %.1f",
        scene_brightness,
        dataset.wandb_display_gain,
    )
    dataset.enhancement_prior_path = os.path.abspath(
        dataset.enhancement_prior_path
        or os.path.join(
            dataset.source_path,
            _default_enhancement_prior_directory(dataset, input_gain),
        )
    )
    common = [
        "--source_path",
        dataset.source_path,
        "--images",
        dataset.images,
        "--lod",
        str(dataset.lod),
    ]
    if dataset.eval:
        common.append("--eval")

    if prior_generation_required(dataset.depth_prior_path, "DepthAnything"):
        depth_root = _project_path(dataset.depth_anything_root)
        checkpoint_name = os.path.basename(dataset.depth_anything_checkpoint)
        depth_checkpoint = resolve_required_file(
            [
                _project_path(dataset.depth_anything_checkpoint),
                os.path.join(depth_root, "checkpoints", checkpoint_name),
                os.path.join(PROJECT_ROOT, "checkpoints", checkpoint_name),
            ],
            f"DepthAnything {dataset.depth_anything_encoder} checkpoint",
            "--depth_anything_checkpoint",
        )
        logger.info("[prior 1/2] DepthAnything cache missing: loading model once")
        _run_prior_command(
            [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "precompute_depth_prior.py"),
                *common,
                "--output",
                dataset.depth_prior_path,
                "--depth_anything_root",
                depth_root,
                "--encoder",
                dataset.depth_anything_encoder,
                "--checkpoint",
                depth_checkpoint,
                "--input_size",
                str(dataset.depth_anything_input_size),
                "--device",
                "cuda",
            ],
            dataset.depth_prior_path,
            "DepthAnything",
            logger,
        )
    else:
        logger.info(
            "[prior 1/2] Reusing existing DepthAnything cache at %s",
            dataset.depth_prior_path,
        )

    enhancement_name = "CIDNet" if backend == "cidnet" else "StableSR"
    if prior_generation_required(dataset.enhancement_prior_path, enhancement_name):
        logger.info(
            "[prior 2/2] %s cache missing: loading model once",
            enhancement_name,
        )
        if backend == "cidnet":
            cidnet_weights = resolve_required_file(
                [
                    _project_path(dataset.cidnet_weights),
                    os.path.join(
                        _project_path(dataset.cidnet_root),
                        "weights",
                        "LOLv2_real",
                        os.path.basename(dataset.cidnet_weights),
                    ),
                ],
                "CIDNet checkpoint",
                "--cidnet_weights",
            )
            enhancement_command = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "precompute_cidnet_prior.py"),
                *common,
                "--output",
                dataset.enhancement_prior_path,
                "--cidnet_root",
                _project_path(dataset.cidnet_root),
                "--weights",
                cidnet_weights,
                "--device",
                "cuda",
            ]
        else:
            stablesr_root = _project_path(dataset.stablesr_root)
            stablesr_checkpoint = resolve_required_file(
                [
                    _project_path(dataset.stablesr_checkpoint),
                    os.path.join(
                        PROJECT_ROOT,
                        "checkpoints",
                        os.path.basename(dataset.stablesr_checkpoint),
                    ),
                    os.path.join(
                        stablesr_root,
                        "checkpoints",
                        os.path.basename(dataset.stablesr_checkpoint),
                    ),
                ],
                "StableSR checkpoint",
                "--stablesr_checkpoint",
            )
            stablesr_vqgan_checkpoint = resolve_required_file(
                [
                    _project_path(dataset.stablesr_vqgan_checkpoint),
                    os.path.join(
                        PROJECT_ROOT,
                        "checkpoints",
                        os.path.basename(dataset.stablesr_vqgan_checkpoint),
                    ),
                    os.path.join(
                        stablesr_root,
                        "checkpoints",
                        os.path.basename(dataset.stablesr_vqgan_checkpoint),
                    ),
                ],
                "StableSR VQGAN checkpoint",
                "--stablesr_vqgan_checkpoint",
            )
            logger.info(
                "[INFO] StableSR fixed-prior recipe: input_gain=%s, checkpoint=%s, vqgan=%s",
                input_gain,
                stablesr_checkpoint,
                stablesr_vqgan_checkpoint,
            )
            enhancement_command = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "precompute_stablesr_prior.py"),
                *common,
                "--output",
                dataset.enhancement_prior_path,
                "--stablesr_root",
                stablesr_root,
                "--stablesr_python",
                dataset.stablesr_python or sys.executable,
                "--inference_script",
                os.path.join(
                    stablesr_root,
                    "scripts",
                    "sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py",
                ),
                "--config",
                _stablesr_path(stablesr_root, dataset.stablesr_config),
                "--checkpoint",
                stablesr_checkpoint,
                "--vqgan_checkpoint",
                stablesr_vqgan_checkpoint,
                "--input_gain",
                str(input_gain),
            ]
        _run_prior_command(
            enhancement_command,
            dataset.enhancement_prior_path,
            enhancement_name,
            logger,
        )
    else:
        logger.info(
            "[prior 2/2] Reusing existing %s cache at %s",
            enhancement_name,
            dataset.enhancement_prior_path,
        )


def _checkpoint_payload(
    gaussians: GaussianModel,
    iteration: int,
    use_warmup: bool,
    warmup_iterations: int,
) -> dict:
    return {
        "model_format_version": MODEL_FORMAT_VERSION,
        "iteration": int(iteration),
        "training_stage": training_stage_state(
            iteration,
            use_warmup,
            warmup_iterations,
        ),
        "gaussians": gaussians.capture(),
    }


def _load_checkpoint(
    path: str,
    gaussians: GaussianModel,
    optimization,
    use_warmup: bool,
) -> int:
    state = torch.load(path, map_location="cuda")
    expected = {
        "model_format_version",
        "iteration",
        "training_stage",
        "gaussians",
    }
    if not isinstance(state, dict) or state.get("model_format_version") != MODEL_FORMAT_VERSION:
        raise RuntimeError(
            f"Checkpoint must use MODEL_FORMAT_VERSION={MODEL_FORMAT_VERSION}; legacy tuples are unsupported"
        )
    if set(state) != expected:
        raise ValueError(
            f"Invalid v2 checkpoint fields; missing={sorted(expected - set(state))}, "
            f"extra={sorted(set(state) - expected)}"
        )
    expected_stage = training_stage_state(
        int(state["iteration"]),
        use_warmup,
        optimization.warmup_iterations,
    )
    if state["training_stage"] != expected_stage:
        raise RuntimeError(
            "Checkpoint warmup schedule does not match this training command: "
            f"stored={state['training_stage']}, expected={expected_stage}. "
            "Resume with the original --warmup and --warmup_iterations values."
        )
    gaussians.restore(state["gaussians"], optimization)
    return int(state["iteration"])


@torch.no_grad()
def _evaluate(scene, pipeline, background, kernel_size, iteration, writer=None, wandb=None):
    cameras = scene.getTestCameras()
    if not cameras:
        return
    scene.gaussians.eval()
    l1_value = 0.0
    psnr_value = 0.0
    for camera in cameras:
        pose = get_tensor_from_camera(camera.world_view_transform.transpose(0, 1)).cuda()
        visible = prefilter_voxel(
            camera,
            scene.gaussians,
            pipeline,
            background,
            kernel_size,
            camera_pose=pose,
        )
        prediction = render(
            camera,
            scene.gaussians,
            pipeline,
            background,
            kernel_size,
            visible_mask=visible,
            camera_pose=pose,
        )["render"].clamp(0.0, 1.0)
        target = camera.original_image.cuda().clamp(0.0, 1.0)
        l1_value += float(l1_loss(prediction, target))
        psnr_value += float(psnr(prediction, target).mean())
    l1_value /= len(cameras)
    psnr_value /= len(cameras)
    if writer:
        writer.add_scalar("test/l1", l1_value, iteration)
        writer.add_scalar("test/psnr", psnr_value, iteration)
    if wandb:
        wandb.log({"test/l1": l1_value, "test/psnr": psnr_value, "iteration": iteration})
    scene.gaussians.train()


def _log_densification(logger, wandb, stage, iteration, statistics):
    logger.info("[%s densify %d] %s", stage, iteration, statistics)
    if wandb:
        values = {
            f"{stage}_densify/{key}": value
            for key, value in statistics.items()
            if key != "added_by_level"
        }
        values.update(
            {
                f"{stage}_densify/added_level_{level}": value
                for level, value in enumerate(statistics["added_by_level"])
            }
        )
        values["iteration"] = iteration
        wandb.log(values)


def _select_monitor_camera(scene, requested_name: str, requested_split: str, logger):
    """Select one deterministic train/test camera for compact W&B monitoring."""
    cameras_by_split = {
        "train": scene.getTrainCameras(),
        "test": scene.getTestCameras(),
    }
    order = (requested_split, "train" if requested_split == "test" else "test")
    for split in order:
        for camera in cameras_by_split[split]:
            if str(camera.image_name) == str(requested_name):
                return camera, split
    try:
        requested_index = int(requested_name)
    except ValueError:
        requested_index = -1
    for split in order:
        ordered = sorted(cameras_by_split[split], key=lambda item: str(item.image_name))
        if 0 <= requested_index < len(ordered):
            return ordered[requested_index], split
    for split in order:
        if cameras_by_split[split]:
            camera = sorted(cameras_by_split[split], key=lambda item: str(item.image_name))[0]
            logger.warning(
                "W&B monitor camera %s was not found; using %s camera %s",
                requested_name,
                split,
                camera.image_name,
            )
            return camera, split
    raise RuntimeError("No train or test camera is available for W&B monitoring")


def _wandb_image(tensor: torch.Tensor, caption: str):
    """Convert a render tensor into one finite W&B image."""
    from wandb import Image as WandbImage

    image = torch.nan_to_num(
        tensor.detach(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    return WandbImage(to_pil_image(image.cpu()), caption=caption)


@torch.no_grad()
def _build_wandb_core_images(
    scene,
    camera,
    split,
    pipeline,
    background,
    kernel_size,
    iteration,
    display_gain,
):
    """Build only the five v2 decomposition images requested for supervision.

    Dark maps (the low-light render and main illumination) use the clean
    reference's display-only scene gain; bright maps stay untouched.
    """
    was_training = scene.gaussians.mlp_opacity.training
    scene.gaussians.eval()
    try:
        pose = (
            scene.gaussians.get_RT(camera.uid)
            if split == "train"
            else get_tensor_from_camera(camera.world_view_transform.transpose(0, 1)).cuda()
        )
        visible = prefilter_voxel(
            camera,
            scene.gaussians,
            pipeline,
            background,
            kernel_size,
            camera_pose=pose,
        )
        package = render(
            camera,
            scene.gaussians,
            pipeline,
            background,
            kernel_size,
            visible_mask=visible,
            camera_pose=pose,
        )
        caption = f"{split}:{camera.image_name} iteration={iteration}"
        brightened_fields = {"image", "illumination"}
        images = {}
        for name, field in WANDB_CORE_IMAGE_FIELDS.items():
            tensor = package[field]
            if name in brightened_fields:
                tensor = torch.clamp(tensor * display_gain, 0.0, 1.0)
            images[name] = _wandb_image(tensor, caption)
        reflectance = torch.nan_to_num(package["render_reflectance"].detach())
        illumination = torch.nan_to_num(package["render_illumination"].detach())
        enhanced_illumination = torch.nan_to_num(
            package["render_illumination_enhanced"].detach()
        )
        images.update(
            {
                "monitor/reflectance_mean": float(reflectance.mean()),
                "monitor/reflectance_std": float(reflectance.std()),
                "monitor/illumination_mean": float(illumination.mean()),
                "monitor/illumination_enhanced_mean": float(
                    enhanced_illumination.mean()
                ),
                "monitor/illumination_enhanced_p95": float(
                    torch.quantile(enhanced_illumination.float(), 0.95)
                ),
                "monitor/illumination_enhanced_max": float(
                    enhanced_illumination.max()
                ),
                "monitor/illumination_enhanced_fraction_gt_one": float(
                    (enhanced_illumination > 1.0).float().mean()
                ),
            }
        )
        return images
    finally:
        if was_training:
            scene.gaussians.train()


def training(
    dataset,
    optimization,
    pipeline,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    start_checkpoint,
    debug_from,
    use_warmup,
    optimize_pose,
    wandb_monitor_camera,
    wandb_monitor_split,
    wandb_monitor_interval,
    wandb=None,
    logger=None,
):
    writer = prepare_output_and_logger(dataset)
    gaussians = create_gaussian_model(dataset)
    scene = Scene(
        dataset,
        gaussians,
        shuffle=False,
        require_priors=True,
    )
    gaussians.training_setup(optimization)
    first_iteration = 0
    if start_checkpoint:
        first_iteration = _load_checkpoint(
            start_checkpoint,
            gaussians,
            optimization,
            use_warmup,
        )
    previous_warmup = bool(
        use_warmup
        and optimization.warmup_iterations > 0
        and first_iteration <= optimization.warmup_iterations
    )
    # Reference two-run budget: warmup first counts its own
    # 1..warmup_iterations, then the main stage counts a fresh
    # 1..optimization.iterations on top; wall-clock total is the sum.
    warmup_total = (
        optimization.warmup_iterations
        if use_warmup and optimization.warmup_iterations > 0
        else 0
    )
    logger.info(
        "Training stages: warmup=%s warmup_iterations=%d + main_iterations=%d "
        "= total_iterations=%d warmup_geometry_lr_scale=%g resume_iteration=%d",
        bool(use_warmup),
        warmup_total,
        optimization.iterations,
        optimization.iterations + warmup_total,
        optimization.warmup_geometry_lr_scale,
        first_iteration,
    )
    logger.info("=" * 80)
    if warmup_total > 0 and first_iteration < warmup_total:
        logger.info(
            "[phase 1/2] Starting warmup: %d iterations; main starts afterward at 1/%d",
            warmup_total,
            optimization.iterations,
        )
    else:
        logger.info(
            "[phase] Starting main training: local iteration %d/%d",
            max(first_iteration - warmup_total + 1, 1),
            optimization.iterations,
        )
    logger.info("=" * 80)
    train_cameras = scene.getTrainCameras().copy()
    if not train_cameras:
        raise RuntimeError("Training requires at least one camera")
    monitor_camera = None
    monitor_split = None
    if wandb is not None and wandb_monitor_interval > 0:
        monitor_camera, monitor_split = _select_monitor_camera(
            scene,
            wandb_monitor_camera,
            wandb_monitor_split,
            logger,
        )
        logger.info(
            "W&B core-image monitor: split=%s camera=%s interval=%d",
            monitor_split,
            monitor_camera.image_name,
            wandb_monitor_interval,
        )
    os.makedirs(os.path.join(scene.model_path, "pose"), exist_ok=True)
    display_gain = float(getattr(dataset, "wandb_display_gain", 0.0) or 0.0)
    if display_gain <= 0.0:
        display_gain, scene_brightness = _scene_display_gain(train_cameras)
    else:
        scene_brightness = float(getattr(dataset, "scene_mean_brightness", 0.0) or 0.0)
    logger.info(
        "[INFO] W&B display gain %.1f (scene mean brightness %.5f); "
        "this transform is excluded from all training targets and losses",
        display_gain,
        scene_brightness,
    )
    original_pose_path = os.path.join(scene.model_path, "pose", "pose_org.npy")
    if not os.path.exists(original_pose_path):
        save_pose(original_pose_path, gaussians.P, train_cameras)
    gaussians.compute_3D_filter(train_cameras)
    if not optimize_pose:
        gaussians.P.requires_grad_(False)

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device="cuda",
    )
    camera_stack = []
    ema_loss = 0.0
    total_iterations = optimization.iterations + warmup_total
    warmup_display = bool(
        warmup_total > 0 and first_iteration < optimization.warmup_iterations
    )
    progress = _start_training_progress(
        "warmup" if warmup_display else "main",
        first_iteration if warmup_display else max(first_iteration - warmup_total - 1, 0),
        optimization.warmup_iterations if warmup_display else optimization.iterations,
    )
    gaussians.train()

    for iteration in range(first_iteration + 1, total_iterations + 1):
        warmup_phase = use_warmup and iteration <= optimization.warmup_iterations
        phase_iteration = iteration if warmup_phase else iteration - warmup_total
        densification_reset_event = 0.0
        if previous_warmup and not warmup_phase:
            progress.close()
            progress = _start_training_progress(
                "main", phase_iteration - 1, optimization.iterations
            )
            logger.info("=" * 80)
            logger.info(
                "[phase 2/2] Warmup finished! Switching to main training at global "
                "iteration %d (main iteration %d / %d)",
                iteration,
                phase_iteration,
                optimization.iterations,
            )
            logger.info("=" * 80)
            reset_summary = gaussians.reset_densification_gradient_statistics()
            densification_reset_event = 1.0
            logger.info(
                "[densification reset warmup->main at %d] "
                "observed_offsets=%d observation_count=%g",
                iteration,
                reset_summary["observed_offsets"],
                reset_summary["observation_count"],
            )
        previous_warmup = warmup_phase
        geometry_lr_scale = (
            optimization.warmup_geometry_lr_scale if warmup_phase else 1.0
        )
        stage_start_stat = (
            optimization.warmup_start_stat if warmup_phase else optimization.start_stat
        )
        stage_update_from = (
            optimization.warmup_update_from if warmup_phase else optimization.update_from
        )
        stage_update_until = (
            optimization.warmup_update_until if warmup_phase else optimization.update_until
        )
        stage_update_interval = (
            optimization.warmup_update_interval if warmup_phase else optimization.update_interval
        )
        densification_stat_collection_active = 0.0
        densification_adjust_event = 0.0
        gaussians.update_learning_rate(
            phase_iteration,
            geometry_lr_scale=geometry_lr_scale,
        )

        if not camera_stack:
            camera_stack = train_cameras.copy()
        viewpoint = camera_stack.pop(random.randrange(len(camera_stack)))
        pose = gaussians.get_RT(viewpoint.uid)
        if iteration - 1 == debug_from:
            pipeline.debug = True
        visible = prefilter_voxel(
            viewpoint,
            gaussians,
            pipeline,
            background,
            dataset.kernel_size,
            camera_pose=pose,
        )
        retain_grad = phase_iteration < stage_update_until
        package = render(
            viewpoint,
            gaussians,
            pipeline,
            background,
            dataset.kernel_size,
            visible_mask=visible,
            retain_grad=retain_grad,
            camera_pose=pose,
        )
        low_target = viewpoint.original_image.cuda()
        enhanced_prior = viewpoint.enhancement_prior
        depth_prior = scene.depth_prior_dict[viewpoint.uid]
        low_prediction = package["render"]
        enhanced_prediction = package["render_enhanced"]
        coverage = package["_training_coverage"]
        enhanced_target = enhanced_prior.detach()

        if FUSED_SSIM_AVAILABLE:
            low_ssim = fused_ssim(low_prediction.unsqueeze(0), low_target.unsqueeze(0))
            enhanced_ssim = fused_ssim(enhanced_prediction.unsqueeze(0), enhanced_target.unsqueeze(0))
        else:
            low_ssim = ssim(low_prediction, low_target)
            enhanced_ssim = ssim(enhanced_prediction, enhanced_target)
        low_photo = photo_loss(
            low_prediction,
            low_target,
            optimization.lambda_dssim,
            low_ssim,
        )
        max_rgb_target, reflectance_target = retinex_targets(low_target)
        reflectance_reconstruction_prediction = coverage_masked_prediction(
            package["render_reflectance"] * package["render_illumination"].detach(),
            low_target,
            coverage,
        )
        # Diagnostic rasterization has detached geometry and coverage; main L
        # is detached here as well. Therefore this objective can update only
        # explicit reflectance base/detail, never Gaussian geometry or ASG.
        reflectance_reconstruction = photo_loss(
            reflectance_reconstruction_prediction,
            low_target,
            optimization.lambda_dssim,
        )
        reflectance_objective = reflectance_reconstruction
        illumination_target = max_rgb_target.expand_as(package["render_illumination"])
        illumination_prediction = coverage_masked_prediction(
            package["render_illumination"],
            illumination_target,
            coverage,
        )
        illumination_photo = photo_loss(
            illumination_prediction,
            illumination_target,
            optimization.lambda_dssim,
        )
        illumination_edge_tv = edge_aware_illumination_tv_loss(
            package["render_illumination"],
            max_rgb_target,
            coverage,
        )
        illumination_objective = illumination_photo + illumination_edge_tv
        enhanced_photo = photo_loss(
            enhanced_prediction,
            enhanced_target,
            optimization.lambda_dssim,
            enhanced_ssim,
        )
        enhanced_diffuse_target, enhanced_fill_target = decompose_enhanced_prior(
            enhanced_target.detach(),
            reflectance_target,
        )
        enhanced_illumination_target = enhanced_diffuse_target + (
            1.0 - enhanced_diffuse_target
        ) * enhanced_fill_target
        enhanced_illumination_prediction = coverage_masked_prediction(
            package["render_illumination_enhanced"],
            enhanced_illumination_target,
            coverage,
        )
        enhanced_illumination_photo = photo_loss(
            enhanced_illumination_prediction,
            enhanced_illumination_target,
            optimization.lambda_dssim,
        )
        enhanced_objective = enhanced_photo + enhanced_illumination_photo
        depth_objective = inverse_depth_pearson_loss(
            package["render_depth"],
            depth_prior,
            coverage,
        )
        scale = scaling_loss(package["scaling"])
        loss = (
            low_photo
            + optimization.lambda_reflectance_reconstruction * reflectance_objective
            + optimization.lambda_illumination * illumination_objective
            + optimization.lambda_enhanced * enhanced_objective
            + optimization.lambda_depth * depth_objective
            + optimization.lambda_scaling * scale
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite total loss at iteration {iteration}")
        loss.backward()

        with torch.no_grad():
            if retain_grad and package["viewspace_points"].grad is not None:
                if stage_start_stat < phase_iteration < stage_update_until:
                    gaussians.training_statis(
                        package["viewspace_points"],
                        package["neural_opacity"],
                        package["visibility_filter"],
                        package["selection_mask"],
                        visible,
                    )
                    densification_stat_collection_active = 1.0
                    if (
                        phase_iteration > stage_update_from
                        and phase_iteration % stage_update_interval == 0
                    ):
                        statistics = gaussians.adjust_anchor(
                            check_interval=stage_update_interval,
                            success_threshold=(
                                optimization.warmup_success_threshold
                                if warmup_phase
                                else optimization.success_threshold
                            ),
                            grad_threshold=(
                                optimization.warmup_densify_grad_threshold
                                if warmup_phase
                                else optimization.densify_grad_threshold
                            ),
                            min_opacity=optimization.min_opacity,
                            max_anchors=optimization.max_anchors,
                            max_new_anchors=(
                                optimization.warmup_max_new_anchors
                                if warmup_phase
                                else optimization.max_new_anchors_per_update
                            ),
                            level_caps=(
                                optimization.warmup_level_caps
                                if warmup_phase
                                else optimization.densify_level_caps
                            ),
                            current_iteration=phase_iteration,
                            prune_grace_iters=optimization.anchor_prune_grace_iters,
                            prune_from_iter=optimization.prune_from_iter,
                            max_pruned_anchors=optimization.max_pruned_anchors_per_update,
                            allow_prune=not warmup_phase,
                        )
                        densification_adjust_event = 1.0
                        _log_densification(
                            logger,
                            wandb,
                            "warmup" if warmup_phase else "main",
                            phase_iteration,
                            statistics,
                        )
                        gaussians.compute_3D_filter(train_cameras)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            ema_loss = 0.4 * float(loss) + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress.set_postfix(
                    loss=f"{ema_loss:.4f}",
                    photo=f"{float(low_photo):.4f}",
                    rrec=f"{float(reflectance_reconstruction):.4f}",
                    enh=f"{float(enhanced_photo):.4f}",
                    enhL=f"{float(enhanced_illumination_photo):.4f}",
                    lphoto=f"{float(illumination_photo):.4f}",
                    ltv=f"{float(illumination_edge_tv):.4f}",
                    anchors=gaussians.get_anchor.shape[0],
                )
                progress.update(10)
            scalars = {
                "loss/total": float(loss),
                "loss/photo_low": float(low_photo),
                "loss/reflectance_reconstruction": float(reflectance_reconstruction),
                "loss/illumination_photo": float(illumination_photo),
                "loss/illumination_edge_tv": float(illumination_edge_tv),
                "loss/illumination": float(illumination_objective),
                "loss/photo_enhanced": float(enhanced_photo),
                "loss/illumination_enhanced_photo": float(enhanced_illumination_photo),
                "loss/enhanced": float(enhanced_objective),
                "loss/depth": float(depth_objective),
                "loss/scaling": float(scale),
                "anchors": gaussians.get_anchor.shape[0],
                "phase/is_warmup": float(warmup_phase),
                "phase/iteration": float(phase_iteration),
                "monitor/display_gain": float(display_gain),
                "phase/geometry_lr_scale": float(geometry_lr_scale),
                "densification/stat_collection_active": densification_stat_collection_active,
                "densification/adjust_event": densification_adjust_event,
                "densification/reset_event": densification_reset_event,
                "densification/check_interval": stage_update_interval,
                "densification/update_until": stage_update_until,
                "iteration": iteration,
            }
            for group in gaussians.optimizer.param_groups:
                scalars[f"lr/{group['name']}"] = float(group["lr"])
            if writer:
                for name, value in scalars.items():
                    writer.add_scalar(name, value, iteration)
            if wandb:
                wandb_payload = dict(scalars)
                if wandb_monitor_interval > 0 and iteration % wandb_monitor_interval == 0:
                    wandb_payload.update(
                        _build_wandb_core_images(
                            scene,
                            monitor_camera,
                            monitor_split,
                            pipeline,
                            background,
                            dataset.kernel_size,
                            iteration,
                            display_gain,
                        )
                    )
                wandb.log(wandb_payload)
            if phase_iteration in testing_iterations:
                _evaluate(
                    scene,
                    pipeline,
                    background,
                    dataset.kernel_size,
                    iteration,
                    writer,
                    wandb,
                )
            if phase_iteration in saving_iterations:
                logger.info(
                    "[global %d] Saving v2 model (%s iteration %d)",
                    iteration,
                    "warmup" if warmup_phase else "main",
                    phase_iteration,
                )
                scene.save(phase_iteration)
                save_pose(
                    os.path.join(scene.model_path, "pose", f"pose_{phase_iteration}.npy"),
                    gaussians.P,
                    train_cameras,
                )
            if phase_iteration in checkpoint_iterations:
                path = os.path.join(scene.model_path, f"chkpnt{phase_iteration}.pth")
                torch.save(
                    _checkpoint_payload(
                        gaussians,
                        iteration,
                        use_warmup,
                        optimization.warmup_iterations,
                    ),
                    path,
                )
    progress.close()
    return scene


def main() -> None:
    parser = ArgumentParser(description="Explicit R/L Scaffold-GS training")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_monitor_camera", type=str, default="1")
    parser.add_argument("--wandb_monitor_split", choices=("train", "test"), default="test")
    parser.add_argument("--wandb_monitor_interval", type=int, default=300)
    parser.add_argument("--disable_pose_optimization", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8_000, 10_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")
    args = parser.parse_args(sys.argv[1:])
    if args.model_format_version != MODEL_FORMAT_VERSION:
        raise RuntimeError(f"Training requires MODEL_FORMAT_VERSION={MODEL_FORMAT_VERSION}")
    validate_training_schedule(args)
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    os.makedirs(args.model_path, exist_ok=True)
    logger = get_logger(args.model_path)
    dataset = model.extract(args)
    _ensure_training_priors(dataset, logger)
    args.enhancement_prior_backend = dataset.enhancement_prior_backend
    args.enhancement_prior_path = dataset.enhancement_prior_path
    args.depth_prior_path = dataset.depth_prior_path
    logger.info("args: %s", args)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    wandb = None
    if args.use_wandb:
        import wandb as wandb_module

        wandb = wandb_module.init(
            project=f"Explicit-Scaffold-GS-{os.path.basename(os.path.normpath(args.source_path))}",
            name=os.path.basename(os.path.normpath(args.model_path)),
            config=vars(args),
        )
        wandb_module.define_metric("iteration")
        for metric_name in (
            "loss/*",
            "phase/*",
            "densification/*",
            "lr/*",
            "monitor/*",
            "test/*",
            "warmup_densify/*",
            "main_densify/*",
            "anchors",
            *WANDB_CORE_IMAGE_FIELDS.keys(),
        ):
            wandb_module.define_metric(metric_name, step_metric="iteration")
    training(
        dataset,
        optimization.extract(args),
        pipeline.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.warmup,
        not args.disable_pose_optimization,
        args.wandb_monitor_camera,
        args.wandb_monitor_split,
        args.wandb_monitor_interval,
        wandb,
        logger,
    )
    logger.info("Training complete")
    # Post-training render + evaluation, mirroring the clean reference
    # pipeline: render every train/test view of the final saved checkpoint,
    # then score renders and enhanced outputs against the captures.
    logger.info("\nStarting Rendering~")
    render_sets(
        model.extract(args),
        -1,
        pipeline.extract(args),
        wandb=wandb,
        logger=logger,
    )
    logger.info("\nRendering complete.")


if __name__ == "__main__":
    main()
