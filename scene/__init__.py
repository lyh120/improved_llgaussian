"""Scene loading and strict v2 model/prior orchestration."""

from __future__ import annotations

import json
import os
import random
import shutil

SCAFFOLD_SKY_POINT_COUNT = 1000


class Scene:
    def __init__(
        self,
        args,
        gaussians,
        load_iteration=None,
        shuffle=True,
        resolution_scales=(1.0,),
        ply_path=None,
        require_priors=False,
    ) -> None:
        # Keep package import light: explicit-appearance CPU tests must not
        # require compiled Scaffold CUDA extensions.
        from scene.dataset_readers import sceneLoadTypeCallbacks
        from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
        from utils.pose_utils import load_pose
        from utils.prior_utils import load_training_priors
        from utils.system_utils import searchForMaxIteration

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.enhancement_prior_dict = {}
        self.depth_prior_dict = {}

        if load_iteration is not None:
            self.loaded_iter = (
                searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                if load_iteration == -1
                else load_iteration
            )
            print(f"Loading trained model at iteration {self.loaded_iter}")

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.eval,
                args.lod,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                ply_path=ply_path,
            )
        else:
            raise RuntimeError(f"Could not recognize scene type at {args.source_path}")

        if not self.loaded_iter:
            source_ply = ply_path if ply_path is not None else scene_info.ply_path
            shutil.copyfile(source_ply, os.path.join(self.model_path, "input.ply"))
            cameras = list(scene_info.test_cameras) + list(scene_info.train_cameras)
            with open(os.path.join(self.model_path, "cameras.json"), "w", encoding="utf-8") as handle:
                json.dump([camera_to_JSON(index, camera) for index, camera in enumerate(cameras)], handle)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras = {}
        self.test_cameras = {}
        for scale in resolution_scales:
            self.train_cameras[scale] = cameraList_from_camInfos(scene_info.train_cameras, scale, args)
            self.test_cameras[scale] = cameraList_from_camInfos(scene_info.test_cameras, scale, args)

        train_cameras = self.getTrainCameras()
        if require_priors:
            self.enhancement_prior_dict, self.depth_prior_dict = load_training_priors(
                train_cameras,
                args.enhancement_prior_path,
                args.depth_prior_path,
            )

        if self.loaded_iter is not None:
            point_cloud_path = os.path.join(
                self.model_path,
                "point_cloud",
                f"iteration_{self.loaded_iter}",
            )
            self.gaussians.load_ply_sparse_gaussian(os.path.join(point_cloud_path, "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(point_cloud_path)
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud,
                self.cameras_extent,
                num_sky_gaussians=SCAFFOLD_SKY_POINT_COUNT,
                cameras=train_cameras,
                prune_ratio=args.prune_ratio,
                beta=args.beta,
            )

        self.gaussians.init_RT_seq(self.train_cameras)
        if self.loaded_iter is not None:
            pose_path = os.path.join(self.model_path, "pose", f"pose_{self.loaded_iter}.npy")
            if os.path.exists(pose_path):
                self.gaussians.P = load_pose(pose_path)

    def save(self, iteration) -> None:
        path = os.path.join(self.model_path, "point_cloud", f"iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
