"""Render and evaluate MODEL_FORMAT_VERSION=2 explicit R/L models."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import prefilter_voxel, render
from scene import Scene
from utils.model_format import MODEL_FORMAT_VERSION, NUMERICAL_EPS
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.pose_utils import get_tensor_from_camera


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


def _original_pose(camera) -> torch.Tensor:
    return get_tensor_from_camera(camera.world_view_transform.transpose(0, 1)).cuda()


def _pose_for_view(scene: Scene, camera, split: str) -> torch.Tensor:
    if split == "train" and 0 <= camera.uid < scene.gaussians.P.shape[0]:
        return scene.gaussians.get_RT(camera.uid)
    return _original_pose(camera)


def _load_image(root: str | None, image_name: str, device) -> torch.Tensor | None:
    if not root:
        return None
    matches = []
    for pattern in (image_name, f"{image_name}.*"):
        matches.extend(glob.glob(os.path.join(root, pattern)))
    matches = sorted(set(matches))
    if not matches:
        return None
    with Image.open(matches[0]) as image:
        return TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0).to(device)


def _bright_gt_root(source_path: str) -> str | None:
    for candidate in (os.path.join(source_path, "gt", "images"), os.path.join(source_path, "gt")):
        if os.path.isdir(candidate):
            return candidate
    return None


def _align(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(
            target.unsqueeze(0),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return prediction, target


def _metrics(prediction, target, lpips_model):
    prediction, target = _align(prediction, target)
    pred_batch = prediction.unsqueeze(0)
    target_batch = target.unsqueeze(0)
    return {
        "PSNR": float(psnr(pred_batch, target_batch).mean().cpu()),
        "SSIM": float(ssim(pred_batch, target_batch).cpu()),
        "LPIPS": float(
            lpips_model(pred_batch * 2.0 - 1.0, target_batch * 2.0 - 1.0)
            .mean()
            .cpu()
        ),
    }


def _save_metrics(path: str, per_view: dict, label: str, wandb=None, logger=None) -> None:
    if not per_view:
        return
    keys = next(iter(per_view.values())).keys()
    summary = {key: float(np.mean([entry[key] for entry in per_view.values()])) for key in keys}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_view": per_view}, handle, indent=2)
    message = (
        f"[{label}] PSNR={summary['PSNR']:.4f}, SSIM={summary['SSIM']:.4f}, "
        f"LPIPS={summary['LPIPS']:.4f}"
    )
    print(message)
    if logger is not None:
        logger.info(message)
    if wandb is not None:
        wandb.log(
            {
                f"{label}_PSNR": summary["PSNR"],
                f"{label}_SSIM": summary["SSIM"],
                f"{label}_LPIPS": summary["LPIPS"],
            }
        )


def _save_depth(depth: torch.Tensor, raw_path: str, color_path: str) -> None:
    depth = depth.detach().squeeze().cpu().numpy()
    np.save(raw_path, depth.astype(np.float32))
    finite = np.isfinite(depth)
    normalized = np.zeros_like(depth, dtype=np.float32)
    if finite.any():
        low = float(depth[finite].min())
        high = float(depth[finite].max())
        normalized[finite] = (depth[finite] - low) / max(high - low, NUMERICAL_EPS)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(color_path, color)


def render_set(
    model_path,
    split,
    iteration,
    views,
    scene,
    pipeline,
    background,
    kernel_size,
    evaluate_metrics=False,
    bright_gt_root=None,
    profile_timing=False,
    wandb=None,
    logger=None,
):
    root = os.path.join(model_path, split, f"ours_{iteration}")
    directories = {
        "render": "renders",
        "enhanced": "render_enhanceds",
        "reflectance": "render_reflectances",
        "illumination": "render_illuminations",
        "illumination_enhanced": "render_illuminations_enhanced",
        "depth": "render_depths",
        "gt": "gt",
    }
    paths = {key: os.path.join(root, value) for key, value in directories.items()}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    lpips_model = None
    if evaluate_metrics:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg").cuda().eval()
    low_metrics = {}
    enhanced_metrics = {}
    timings = []
    for view in tqdm(views, desc=f"Rendering {split}"):
        pose = _pose_for_view(scene, view, split)
        visible = prefilter_voxel(
            view,
            scene.gaussians,
            pipeline,
            background,
            kernel_size,
            camera_pose=pose,
        )
        profile = {} if profile_timing else None
        if profile_timing:
            torch.cuda.synchronize()
            start = time.perf_counter()
        package = render(
            view,
            scene.gaussians,
            pipeline,
            background,
            kernel_size,
            visible_mask=visible,
            camera_pose=pose,
            profile_timings=profile,
        )
        if profile_timing:
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        name = f"{view.image_name}.png"
        outputs = {
            "render": package["render"].clamp(0.0, 1.0),
            "enhanced": package["render_enhanced"].clamp(0.0, 1.0),
            "reflectance": package["render_reflectance"].clamp(0.0, 1.0),
            "illumination": package["render_illumination"].clamp(0.0, 1.0),
            "illumination_enhanced": package["render_illumination_enhanced"].clamp(0.0, 1.0),
        }
        for key, tensor in outputs.items():
            torchvision.utils.save_image(tensor, os.path.join(paths[key], name))
        _save_depth(
            package["render_depth"],
            os.path.join(paths["depth"], f"{view.image_name}.npy"),
            os.path.join(paths["depth"], f"color_{view.image_name}.png"),
        )
        target = view.original_image.cuda().clamp(0.0, 1.0)
        torchvision.utils.save_image(target, os.path.join(paths["gt"], name))
        if lpips_model is not None:
            low_metrics[name] = _metrics(outputs["render"], target, lpips_model)
            bright = _load_image(bright_gt_root, view.image_name, target.device)
            if bright is not None:
                enhanced_metrics[name] = _metrics(outputs["enhanced"], bright, lpips_model)

    _save_metrics(
        os.path.join(root, "metrics_lowlight.json"),
        low_metrics,
        f"{split}/lowlight",
        wandb=wandb,
        logger=logger,
    )
    _save_metrics(
        os.path.join(root, "metrics_enhanced_gt.json"),
        enhanced_metrics,
        f"{split}/enhanced",
        wandb=wandb,
        logger=logger,
    )
    if timings:
        checkpoint_root = os.path.join(
            model_path,
            "point_cloud",
            f"iteration_{iteration}",
        )
        checkpoint_bytes = sum(
            os.path.getsize(path)
            for path in glob.glob(os.path.join(checkpoint_root, "**", "*"), recursive=True)
            if os.path.isfile(path)
        )
        profile = {
            "views": len(timings),
            "seconds_per_view": float(np.mean(timings)),
            "fps": float(1.0 / np.mean(timings)),
            "parameter_count": scene.gaussians.parameter_count(),
            "checkpoint_bytes": checkpoint_bytes,
        }
        with open(os.path.join(root, "profile.json"), "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
        fps_message = (
            f"[{split}] render FPS {profile['fps']:.5f} over {profile['views']} views"
        )
        print(fps_message)
        if logger is not None:
            logger.info(fps_message)
        if wandb is not None:
            wandb.log({f"{split}_fps": profile["fps"]})
    return paths["enhanced"]


def images_to_video(image_folder: str, output_path: str, fps: int = 30) -> None:
    files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"No rendered frames found in {image_folder}")
    first = cv2.imread(files[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for path in files:
            frame = cv2.imread(path)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


def render_sets(
    dataset,
    iteration,
    pipeline,
    skip_train=False,
    skip_test=False,
    infer_video=False,
    eval_train_metrics=False,
    profile_render_timing=False,
    wandb=None,
    logger=None,
):
    with torch.no_grad():
        gaussians = create_gaussian_model(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        background = torch.tensor(
            [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
            device="cuda",
        )
        bright_root = _bright_gt_root(dataset.source_path)
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                scene,
                pipeline,
                background,
                dataset.kernel_size,
                evaluate_metrics=eval_train_metrics,
                bright_gt_root=bright_root,
                profile_timing=profile_render_timing,
                wandb=wandb,
                logger=logger,
            )
        enhanced_path = None
        if not skip_test:
            enhanced_path = render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                scene,
                pipeline,
                background,
                dataset.kernel_size,
                evaluate_metrics=True,
                bright_gt_root=bright_root,
                profile_timing=profile_render_timing,
                wandb=wandb,
                logger=logger,
            )
        if infer_video and enhanced_path:
            images_to_video(enhanced_path, os.path.join(os.path.dirname(enhanced_path), "render_enhanced.mp4"))


def main() -> None:
    parser = ArgumentParser(description="Explicit R/L rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument("--eval_train_metrics", action="store_true")
    parser.add_argument("--profile_render_timing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if args.dataset_path:
        args.source_path = os.path.abspath(args.dataset_path)
    if args.model_format_version != MODEL_FORMAT_VERSION:
        raise RuntimeError(f"Rendering requires MODEL_FORMAT_VERSION={MODEL_FORMAT_VERSION}")
    safe_state(args.quiet)
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.infer_video,
        args.eval_train_metrics,
        args.profile_render_timing,
    )


if __name__ == "__main__":
    main()
