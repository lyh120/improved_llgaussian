#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from os import makedirs
import sys
import importlib.util
import torch
import glob

import numpy as np

from pathlib import Path
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import imageio.v2 as imageio
import cv2
from PIL import Image
from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel,render_fast
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from utils.visualize_utils import minmax_normalize, visualize_cmap
from utils.pose_utils import get_tensor_from_camera
from utils.camera_utils import visualizer, generate_interpolated_path
from scene.dataset_readers import loadCameras
import matplotlib.cm as cm
from time import perf_counter
from utils.loss_utils import l1_plus_loss, ssim
from utils.image_utils import psnr
import torchvision.transforms.functional as tf

try:
    from arguments import ModelParams, PipelineParams, get_combined_args
except ImportError:
    arguments_path = os.path.join(PROJECT_ROOT, "arguments", "__init__.py")
    spec = importlib.util.spec_from_file_location("arguments", arguments_path)
    arguments_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["arguments"] = arguments_module
    spec.loader.exec_module(arguments_module)
    ModelParams = arguments_module.ModelParams
    PipelineParams = arguments_module.PipelineParams
    get_combined_args = arguments_module.get_combined_args

def load_pose(path, train_cams):
    w2c_list = np.load(path)
    quat_pose = []
    for i in range(len(w2c_list)):
        bb= w2c_list[i]
        bb = get_tensor_from_camera(bb)
        quat_pose.append(bb)
    quat_pose = torch.stack(quat_pose).to('cuda')

    return quat_pose

def save_interpolate_pose(model_path, iter, n_views):

    org_pose = np.load(model_path / f"pose/pose_{iter}.npy")
    
       # 添加相机位置重排序逻辑
    positions = org_pose[:, :3, 3]  # 提取相机位置
    sorted_indices = []
    current_idx = 0  # 从第一个相机开始
    sorted_indices.append(current_idx)
    
    # 贪心算法：每次选择距离当前相机最近的下一个相机
    remaining_indices = set(range(n_views))
    remaining_indices.remove(current_idx)
    k = None
    while remaining_indices:
        current_pos = positions[current_idx]
        # 找到距离当前相机最近的下一个相机
        # import pdb; pdb.set_trace()
        if k is None:
            next_idx = min(remaining_indices, 
                      key=lambda i: np.linalg.norm(positions[i] - current_pos))
        else:
            next_idx = min(remaining_indices, 
                      key=lambda i: np.linalg.norm(positions[i] - current_pos))
        sorted_indices.append(next_idx)
        remaining_indices.remove(next_idx)
        current_idx = next_idx
        
    # 重新排序相机位姿
    org_pose = org_pose[sorted_indices] 
    visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/poses.png")
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):

        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2,:3,:], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)
    all_inter_pose = np.concatenate([all_inter_pose, org_pose[-1][:3, :].reshape(1, 3, 4)], axis=0)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path / f"pose/poses_interpolated.png")
    np.save(model_path / f"pose/pose_interpolated.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    frame_paths = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPG", ".PNG")):
            frame_paths.append(os.path.join(image_folder, filename))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {image_folder}")

    first = imageio.imread(frame_paths[0])
    if first.ndim == 2:
        first = np.stack([first, first, first], axis=-1)
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video_path}")

    try:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # imageio read is RGB; OpenCV writer expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def _load_image_tensor_from_pattern(pattern, device):
    matches = glob.glob(pattern)
    if not matches:
        return None
    image = Image.open(matches[0]).convert("RGB")
    return tf.to_tensor(image).to(device)


def _align_tensor_pair(pred, target):
    if pred.shape[-2:] == target.shape[-2:]:
        return pred, target
    target_resized = torch.nn.functional.interpolate(
        target.unsqueeze(0),
        size=pred.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return pred, target_resized


def _compute_metrics(pred, target, lpips_fn):
    pred, target = _align_tensor_pair(pred, target)
    pred_b = pred.unsqueeze(0)
    target_b = target.unsqueeze(0)
    return {
        "PSNR": float(psnr(pred_b, target_b).mean().detach().cpu()),
        "SSIM": float(ssim(pred_b, target_b).detach().cpu()),
        "LPIPS": float(lpips_fn(pred_b, target_b).detach().cpu()),
    }


def _dump_metric_report(metric_path, per_view_metrics, report_name):
    if not per_view_metrics:
        return
    summary = {}
    metric_names = next(iter(per_view_metrics.values())).keys()
    for metric_name in metric_names:
        summary[metric_name] = float(np.mean([item[metric_name] for item in per_view_metrics.values()]))
    payload = {
        "summary": summary,
        "per_view": per_view_metrics,
    }
    with open(metric_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"[{report_name}] saved metric report to {metric_path}")
    print(
        f"[{report_name}] summary: "
        f"PSNR={summary['PSNR']:.4f}, "
        f"SSIM={summary['SSIM']:.4f}, "
        f"LPIPS={summary['LPIPS']:.4f}"
    )


def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_reflectance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_reflectances")
    render_illumination_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations")
    render_enhanced_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_enhanceds")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depths")
    render_residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_residuals")
    render_noise_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_noises")
    render_artifact_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_artifacts")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(render_reflectance_path, exist_ok=True)
    makedirs(render_illumination_path, exist_ok=True)
    makedirs(render_enhanced_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_residual_path, exist_ok=True)
    makedirs(render_noise_path, exist_ok=True)
    makedirs(render_artifact_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

    gaussians._anchor.requires_grad_(False)
    gaussians._offset.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians.eval()




    from utils.pose_utils import get_tensor_from_camera
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = 50
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))


        camera_tensor_T = camera_pose[-3:].requires_grad_(True)
        camera_tensor_q = camera_pose[:4].requires_grad_(True)
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.003},
            {"params": [camera_tensor_q], "lr": 0.001}
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4
        )

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=num_iter, eta_min=0.0001)
        with tqdm(total=num_iter, desc=f"Tracking Time Step: {idx+1}", leave=True) as progress_bar:
            candidate_q = camera_tensor_q.clone().detach()
            candidate_T = camera_tensor_T.clone().detach()
            current_min_loss = float(1e20)
            gt = view.original_image[0:3, :, :]
            initial_loss = None

            for iteration in range(num_iter):
                # rendering = render(view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
                voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, kernel_size=kernel_size, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))
                render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size, visible_mask=voxel_visible_mask, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))
                rendering = render_pkg["render"]
                black_hole_threshold = 0.0
                mask = (rendering > black_hole_threshold).float()
                loss = torch.abs(l1_plus_loss(rendering, gt) * mask).mean()
                loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                    if iteration == 0:
                        initial_loss = loss.item()  # Capture initial loss

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_q = camera_tensor_q.clone().detach()
                        candidate_T = camera_tensor_T.clone().detach()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), initial_loss=initial_loss)
                scheduler.step()

            camera_tensor_q = candidate_q
            camera_tensor_T = candidate_T
        with torch.no_grad():
            optimal_pose = torch.cat([camera_tensor_q, camera_tensor_T])
            # print("optimal_pose-camera_pose: ", optimal_pose-camera_pose)
            #rendering_opt = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)["render"]
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, kernel_size=kernel_size, camera_pose=optimal_pose)
            render_pkg_opt = render(view, gaussians, pipeline, background, kernel_size=kernel_size, visible_mask=voxel_visible_mask, camera_pose=optimal_pose)

        
            
            rendering = torch.clamp(render_pkg_opt["render"], 0.0, 1.0)
            rendering_reflectance = torch.clamp(render_pkg_opt["render_reflectance"], 0.0, 1.0)
            rendering_illumination = torch.clamp(render_pkg_opt["render_illumination"] , 0.0, 1.0)
            rendering_enhanced = torch.clamp(render_pkg["render_illumination_enhanced"] * render_pkg["render_reflectance"], 0.0, 1.0)
            rendering_depth = 1 - minmax_normalize(render_pkg["render_depth"])
            if 'render_residual' in render_pkg:
                rendering_residual = torch.clamp(render_pkg["render_residual"], 0.0, 1.0)
                torchvision.utils.save_image(rendering_residual, os.path.join(render_residual_path, view.image_name + ".png"))
            if 'render_noise' in render_pkg:
                rendering_noise = torch.clamp(render_pkg["render_noise"], 0.0, 1.0)
                torchvision.utils.save_image(rendering_noise, os.path.join(render_noise_path, view.image_name + ".png"))
            if 'render_artifact' in render_pkg:
                rendering_artifact = torch.clamp(render_pkg["render_artifact"], 0.0, 1.0)
                torchvision.utils.save_image(rendering_artifact, os.path.join(render_artifact_path, view.image_name + ".png"))



            # gts
            gt = view.original_image[0:3, :, :]
            
            # error maps
            errormap = (rendering - gt).abs()


            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering_reflectance, os.path.join(render_reflectance_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering_illumination, os.path.join(render_illumination_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering_enhanced, os.path.join(render_enhanced_path, view.image_name + ".png"))

            torchvision.utils.save_image(rendering_depth, os.path.join(render_depth_path, view.image_name + ".png"))
            torchvision.utils.save_image(errormap, os.path.join(error_path, view.image_name + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

            depth_est = 1 - rendering_depth.squeeze().detach().cpu().numpy()
            depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
            depth_est = torch.as_tensor(depth_est).permute(2,0,1)
            torchvision.utils.save_image(depth_est, os.path.join(render_depth_path, 'color_{0:05d}'.format(idx) + ".png"))




def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, bright_gt_root=None, evaluate_metrics=False):


    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_path_enhanced = os.path.join(model_path, name, "ours_{}".format(iteration), "renders(enhanced)")
    render_reflectance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_reflectances")
    render_illumination_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations")
    render_illumination_path_enhanced = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations(enhanced)")
    render_illumination_path_enhance = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations_enhance")
    render_enhanced_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_enhanceds")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depths")
    render_residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_residuals")
    render_residual_path_fast = os.path.join(model_path, name, "ours_{}".format(iteration), "render_residuals(enhanced)")
    render_noise_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_noises")
    render_artifact_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_artifacts")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(render_path_enhanced, exist_ok=True)
    makedirs(render_reflectance_path, exist_ok=True)
    makedirs(render_illumination_path, exist_ok=True)
    makedirs(render_illumination_path_enhanced, exist_ok=True)
    makedirs(render_illumination_path_enhance, exist_ok=True)
    makedirs(render_enhanced_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_residual_path, exist_ok=True)
    makedirs(render_residual_path_fast, exist_ok=True)
    makedirs(render_noise_path, exist_ok=True)
    makedirs(render_artifact_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    sg_stats_dict = {}
    lowlight_metrics = {}
    enhanced_metrics = {}
    time_consume = 0
    lpips_fn = None
    if evaluate_metrics:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to("cuda").eval()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        
        if name == "interp":
            pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        else:
            pose = gaussians.get_RT(view.uid)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, kernel_size=kernel_size,camera_pose=pose)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, kernel_size=kernel_size,camera_pose=pose)
        torch.cuda.synchronize(); t1 = time.time()
        time_consume += t1 - t0
        sg_stats = render_pkg.get("sg_stats")
        if sg_stats:
            sg_stats_dict[view.image_name + ".png"] = {
                "sg_energy": float(sg_stats["sg_energy"].detach().cpu()),
                "sg_lambda_mean": float(sg_stats["sg_lambda_mean"].detach().cpu()),
                "sg_lambda_max": float(sg_stats["sg_lambda_max"].detach().cpu()),
            }

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendering_enhance = torch.clamp(render_pkg["render"] * 30, 0.0, 1.0)
        rendering_reflectance = torch.clamp(render_pkg["render_reflectance"], 0.0, 1.0)
        rendering_illumination = torch.clamp(render_pkg["render_illumination"] , 0.0, 1.0)
        rendering_illumination_enhanced = torch.clamp(render_pkg["render_illumination"] * 30 , 0.0, 1.0)
        rendering_illumination_enhance = torch.clamp(render_pkg["render_illumination_enhanced"] , 0.0, 1.0)
        rendering_enhanced = torch.clamp(render_pkg["render_illumination_enhanced"] * render_pkg["render_reflectance"], 0.0, 1.0)
        rendering_depth = 1 - minmax_normalize(render_pkg["render_depth"])
        if "render_residual" in render_pkg:
             rendering_residul_image = torch.clamp(render_pkg["render_residual"] * 30, 0.0, 1.0)
             torchvision.utils.save_image(rendering_residul_image, os.path.join(render_residual_path, view.image_name + ".png"))
             rendering_residul_image_enhance = torch.clamp(render_pkg["render_residual"] * 30, 0.0, 1.0)
             torchvision.utils.save_image(rendering_residul_image_enhance, os.path.join(render_residual_path_fast, view.image_name + ".png"))
        if "render_noise" in render_pkg:
             rendering_noise_image = torch.clamp(render_pkg["render_noise"] * 30, 0.0, 1.0)
             torchvision.utils.save_image(rendering_noise_image, os.path.join(render_noise_path, view.image_name + ".png"))
        if "render_artifact" in render_pkg:
             rendering_artifact_image = torch.clamp(render_pkg["render_artifact"] * 30, 0.0, 1.0)
             torchvision.utils.save_image(rendering_artifact_image, os.path.join(render_artifact_path, view.image_name + ".png"))

        gt = None
        if name != "interp":
            gt = view.original_image[0:3, :, :]
        name_list.append(view.image_name + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_enhance, os.path.join(render_path_enhanced, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_reflectance, os.path.join(render_reflectance_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_illumination, os.path.join(render_illumination_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_illumination_enhance , os.path.join(render_illumination_path_enhance, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_illumination_enhanced, os.path.join(render_illumination_path_enhanced, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_enhanced, os.path.join(render_enhanced_path, view.image_name + ".png"))

        torchvision.utils.save_image(rendering_depth, os.path.join(render_depth_path, view.image_name + ".png"))
        if gt is not None:
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
        depth_est = 1 - rendering_depth.squeeze().detach().cpu().numpy()
        depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
        depth_est = torch.as_tensor(depth_est).permute(2,0,1)
        torchvision.utils.save_image(depth_est, os.path.join(render_depth_path, 'color_{0:05d}'.format(idx) + ".png"))
        if gt is not None:
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
            if evaluate_metrics and lpips_fn is not None:
                lowlight_metrics[view.image_name + ".png"] = _compute_metrics(rendering, gt, lpips_fn)
        if evaluate_metrics and lpips_fn is not None and bright_gt_root is not None:
            bright_gt = _load_image_tensor_from_pattern(os.path.join(bright_gt_root, view.image_name + ".*"), rendering_enhanced.device)
            if bright_gt is not None:
                enhanced_metrics[view.image_name + ".png"] = _compute_metrics(rendering_enhanced, bright_gt, lpips_fn)

    img_num = idx + 1
    fps = img_num / time_consume
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
    if sg_stats_dict:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "sg_stats.json"), 'w') as fp:
            json.dump(sg_stats_dict, fp, indent=True)
    if evaluate_metrics:
        _dump_metric_report(
            os.path.join(model_path, name, "ours_{}".format(iteration), "metrics_lowlight.json"),
            lowlight_metrics,
            "lowlight",
        )
        _dump_metric_report(
            os.path.join(model_path, name, "ours_{}".format(iteration), "metrics_enhanced_gt.json"),
            enhanced_metrics,
            "enhanced_gt",
        )
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_optimize, infer_video : bool, include_residual_render: bool, eval_train_metrics: bool):
    with torch.no_grad():
        use_sg_illumination = getattr(dataset, "use_sg_illumination", getattr(dataset, "use_sg", True))
        illumination_mode = getattr(dataset, "illumination_mode", "sg")
        sg_lobes = getattr(dataset, "sg_lobes", getattr(dataset, "num_sg", 4))
        sg_lambda_min = getattr(dataset, "sg_lambda_min", 1.0)

        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_residual_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_reflectance_dist, dataset.add_illumination_dist, dataset.add_residual_dist, dataset.use_residual, dataset.use_dual_transient, dataset.use_3D_filter,
                              use_sg_illumination=use_sg_illumination, illumination_mode=illumination_mode, sg_lobes=sg_lobes, sg_lambda_min=sg_lambda_min)
        scene = Scene(dataset, gaussians, depth_piror_model=None, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()
        if not include_residual_render:
            gaussians.use_residual = False
        if gaussians.illumination_mode == "sg" and use_sg_illumination and gaussians.sg_illumination_available:
            print("Rendering with SG illumination and B0 reflectance.")
        else:
            print("Rendering in legacy compatibility mode.")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        bright_gt_root = os.path.join(dataset.source_path, "gt", "images")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.kernel_size,
                bright_gt_root=bright_gt_root,
                evaluate_metrics=eval_train_metrics,
            )

    if not skip_test:
        gaussians.init_RT_seq(scene.test_cameras)
        if skip_optimize:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size, bright_gt_root=bright_gt_root, evaluate_metrics=True)
        else:
            render_set_optimize(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size)

    with torch.no_grad():
        if infer_video :
            gaussians.use_residual = False
            save_interpolate_pose(Path(args.model_path), scene.loaded_iter, len(scene.getTrainCameras()))
            interp_pose = np.load(Path(args.model_path) / 'pose' / 'pose_interpolated.npy')
            viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras())
            render_set(
                dataset.model_path,
                "interp",
                scene.loaded_iter,
                viewpoint_stack,
                gaussians,
                pipeline,
                background,
                dataset.kernel_size,
                bright_gt_root=bright_gt_root,
                evaluate_metrics=False,
            )
            image_folder = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/render_enhanceds')
            output_video_file = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/interp_enhanced_view.mp4')
            images_to_video(image_folder, output_video_file, fps=30)
            image_folder = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/renders')
            output_video_file = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/interp_view.mp4')
            images_to_video(image_folder, output_video_file, fps=30)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--dataset_path", default='None', type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_optimize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument("--include_residual_render", action="store_true")
    parser.add_argument("--eval_train_metrics", action="store_true")

    args = get_combined_args(parser)
    if args.dataset_path:
        args.source_path = args.dataset_path
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_optimize,
        args.infer_video,
        args.include_residual_render,
        args.eval_train_metrics,
    )
