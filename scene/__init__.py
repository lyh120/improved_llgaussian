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
import random
import json
import importlib.util
import sys
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import cv2
from utils.visualize_utils import minmax_normalize, visualize_anchor
from utils.pose_utils import load_pose


try:
    from arguments import ModelParams
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arguments_path = os.path.join(project_root, "arguments", "__init__.py")
    spec = importlib.util.spec_from_file_location("arguments", arguments_path)
    arguments_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["arguments"] = arguments_module
    spec.loader.exec_module(arguments_module)
    ModelParams = arguments_module.ModelParams

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, depth_piror_model, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, only_ply=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.gaussians.max_gaussian_anisotropy = getattr(args, "max_gaussian_anisotropy", 0.0)
        self.gaussians.reflectance_init_floor = getattr(args, "reflectance_init_floor", 0.1)
        self.gaussians.direct_composition = (
            getattr(args, "direct_composition", False)
            or getattr(args, "pure_explicit_rl", False)
        )
        self.gaussians.pure_explicit_rl = getattr(args, "pure_explicit_rl", False)
        self.gaussians.explicit_feature_conditioning = getattr(
            args, "explicit_feature_conditioning", False
        )
        gain = tuple(float(value) for value in getattr(args, "enhancement_rgb_gain", "1,1,1").split(","))
        if len(gain) != 3 or any(value <= 0 for value in gain):
            raise ValueError("enhancement_rgb_gain must contain three positive RGB values")
        self.gaussians.enhancement_rgb_gain = gain
        self.gaussians.reflectance_target_detail_reg = getattr(args, "reflectance_target_detail_reg", 0.0)
        self.gaussians.reflectance_target_chroma_reg = getattr(args, "reflectance_target_chroma_reg", 0.0)
        self.depth_prior_gamma = getattr(args, "depth_prior_gamma", 1.0)
        self.depth_piror_model = depth_piror_model

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(os.path.join(args.source_path, "sparse"))
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"


        self.gaussians.set_appearance_residual(len(scene_info.train_cameras))
        
        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            for camera in self.test_cameras[resolution_scale]:
                print(f'camera.image_name: {camera.image_name}')

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if not only_ply:
                self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            if os.path.exists(os.path.join(self.model_path, "pose", f"pose_{self.loaded_iter}.npy")):
                pose = load_pose(os.path.join(self.model_path, "pose", f"pose_{self.loaded_iter}.npy"))
                self.gaussians.P = pose
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, num_sky_gaussians=1000, cameras=self.getTestCameras(),model_path=self.model_path, prune_ratio=args.prune_ratio, beta=args.beta)
            print(f'self.gaussians.get_anchor.shape: {self.gaussians.get_anchor.shape}')
        
####
            self.gaussians.init_RT_seq(self.train_cameras)
####
        if self.depth_piror_model:
            self.depth_piror_dict = self.depth_piror_generator(args.source_path)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def depth_piror_generator(self, source_path):
        depth_piror_dict = dict()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for camera in self.getTrainCameras().copy():
            gt_path = os.path.join(source_path, 'images', camera.image_name + '.*')
            import glob
            gt_path = glob.glob(gt_path)[0]
            gt_image = cv2.imread(gt_path)
            if self.depth_prior_gamma != 1.0:
                gt_image = np.clip(
                    (gt_image.astype(np.float32) / 255.0) ** self.depth_prior_gamma * 255.0,
                    0.0, 255.0,
                ).astype(np.uint8)
            depth_piror = self.depth_piror_model.infer_image(gt_image)
            if not isinstance(depth_piror, torch.Tensor):
                depth_piror = torch.as_tensor(depth_piror, dtype=torch.float32)
            else:
                depth_piror = depth_piror.float()
            if depth_piror.ndim == 2:
                depth_piror = depth_piror.unsqueeze(0)
            depth_piror = depth_piror.to(device)
            idx = camera.uid
            depth_piror_dict[idx] = minmax_normalize(depth_piror)
        return depth_piror_dict
    

