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
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
import copy
import torch
try:
    import laspy
except:
    print("No laspy")
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_prior: np.array = None
    structure_prior: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def loadCameras(poses, viewpoint_stack):

    # load optimized poses
    if poses.shape[0] == len(viewpoint_stack):
        for idx, cam in enumerate(viewpoint_stack):
            R = np.transpose(poses[idx][:3, :3])
            T = poses[idx][:3, 3]
            cam.R = R
            cam.T = T
            cam.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]

    # load interpolated poses
    elif poses.shape[0] > len(viewpoint_stack):
        repeat_times = int(np.ceil(poses.shape[0] / len(viewpoint_stack)))
        # Create repeated list instead of using np.tile
        viewpoint_stack = [copy.deepcopy(vp) for vp in viewpoint_stack * repeat_times][:poses.shape[0]]
        for idx in range(poses.shape[0]):                                 
            R = np.transpose(poses[idx][:3, :3])
            T = poses[idx][:3, 3]
            viewpoint_stack[idx].uid = idx           
            viewpoint_stack[idx].colmap_id = idx+1    
            viewpoint_stack[idx].image_name = str(idx).zfill(5)    
            viewpoint_stack[idx].R = R
            viewpoint_stack[idx].T = T            
            viewpoint_stack[idx].world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            viewpoint_stack[idx].full_proj_transform = (viewpoint_stack[idx].world_view_transform.unsqueeze(0).bmm(viewpoint_stack[idx].projection_matrix.unsqueeze(0))).squeeze(0)
            viewpoint_stack[idx].camera_center = viewpoint_stack[idx].world_view_transform.inverse()[3, :3]
    return viewpoint_stack

def _load_prior_image(prior_path, image_name, expected_size, kind):
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"Missing {kind} prior for image '{image_name}': expected {prior_path}")
    prior = cv2.imread(prior_path, cv2.IMREAD_UNCHANGED)
    if prior is None:
        raise FileNotFoundError(f"Unable to read {kind} prior for image '{image_name}': {prior_path}")
    if prior.ndim == 3:
        prior = cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY)
    prior = prior.astype(np.float32)
    if prior.max() > 1.0:
        prior /= 255.0 if prior.max() <= 255.0 else prior.max()
    width, height = expected_size
    prior = cv2.resize(prior, (width, height), interpolation=cv2.INTER_LINEAR)
    return prior


def _resolve_depth_prior_path(depth_folder, image_name):
    candidates = [
        os.path.join(depth_folder, f"depth_{image_name}.png"),
        os.path.join(depth_folder, f"{image_name}.png"),
        os.path.join(depth_folder, f"depth_{image_name}.jpg"),
        os.path.join(depth_folder, f"{image_name}.jpg"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _resolve_structure_prior_path(structure_folder, image_basename, image_name):
    candidates = [
        os.path.join(structure_folder, image_basename),
        os.path.join(structure_folder, f"{image_name}.png"),
        os.path.join(structure_folder, f"{image_name}.jpg"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    images_folder,
    depth_prior_folder=None,
    structure_prior_folder=None,
):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            print(intr.model)
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # print(f'FovX: {FovX}, FovY: {FovY}')

        image_basename = os.path.basename(extr.name)
        image_path = os.path.join(images_folder, image_basename)
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        depth_prior = None
        structure_prior = None
        if depth_prior_folder is not None:
            depth_prior = _load_prior_image(
                _resolve_depth_prior_path(depth_prior_folder, image_name),
                image_name,
                image.size,
                "depth",
            )
        if structure_prior_folder is not None:
            structure_prior = _load_prior_image(
                _resolve_structure_prior_path(structure_prior_folder, image_basename, image_name),
                image_name,
                image.size,
                "structure",
            )

        # print(f'image: {image.size}')

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth_prior=depth_prior, structure_prior=structure_prior)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    if all(name in vertices.data.dtype.names for name in ("nx", "ny", "nz")):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def _resolve_colmap_sparse_dir(path):
    sparse_zero_dir = os.path.join(path, "sparse/0")
    sparse_dir = os.path.join(path, "sparse")
    if os.path.exists(os.path.join(sparse_zero_dir, "images.bin")) or os.path.exists(os.path.join(sparse_zero_dir, "images.txt")):
        return sparse_zero_dir
    if os.path.exists(os.path.join(sparse_dir, "images.bin")) or os.path.exists(os.path.join(sparse_dir, "images.txt")):
        return sparse_dir
    return sparse_zero_dir

def readColmapSceneInfo(
    path,
    images,
    eval,
    lod,
    llffhold=8,
    use_depth_prior_files=False,
    use_structure_prior_files=False,
    depth_prior_dir="depth_maps",
    structure_prior_dir="W_0.8",
):
    sparse_dir = _resolve_colmap_sparse_dir(path)
    try:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    depth_prior_folder = os.path.join(path, depth_prior_dir) if use_depth_prior_files else None
    structure_prior_folder = os.path.join(path, structure_prior_dir) if use_structure_prior_files else None
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        depth_prior_folder=depth_prior_folder,
        structure_prior_folder=structure_prior_folder,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if lod>0:
            print(f'using lod, using eval')
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f'test_cam_infos: {len(test_cam_infos)}')
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    ply_path = os.path.join(sparse_dir, "points3D.ply")
    fallback_ply_path = os.path.join(sparse_dir, "points.ply")

    if not os.path.exists(ply_path):
        if os.path.exists(bin_path) or os.path.exists(txt_path):
            print("Converting points3D.bin/txt to points3D.ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        elif os.path.exists(fallback_ply_path):
            ply_path = fallback_ply_path
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_debug=False, undistorted=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]], dtype=np.float32)
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
            if is_debug and idx > 50:
                break
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}
