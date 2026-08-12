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

import torch
import torch.nn.functional as F
import math
from functools import reduce
import numpy as np
from torch_scatter import scatter_max, scatter_mean
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.pose_utils import get_tensor_from_camera, get_camera_from_tensor
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from utils.graphics_utils import get_uniform_points_on_sphere_fibonacci
from tqdm import tqdm
from utils.camera_utils import camera_project
from utils.visualize_utils import  visualize_anchor, plot_point_cloud_projection

@torch.no_grad()
def get_sky_points(num_points, points3D, cameras):
    xnp = torch
    points = get_uniform_points_on_sphere_fibonacci(num_points, xnp=xnp)
    points = points.to(points3D.device)
    mean = points3D.mean(0)[None]
    sky_distance = xnp.quantile(xnp.linalg.norm(points3D - mean, 2, -1), 0.97) * 10
    points = points * sky_distance
    points = points + mean
    gmask = torch.zeros((points.shape[0],), dtype=xnp.bool, device=points.device)
    for cam in tqdm(cameras, desc="Generating skybox"):
        uv = camera_project(cam, points[xnp.logical_not(gmask)])
        mask = xnp.logical_not(xnp.isnan(uv).any(-1))
        # Only top 2/3 of the image
        assert cam.image_height is not None
        mask = xnp.logical_and(mask, uv[..., -1] < 2/3 * cam.image_height)
        gmask[xnp.logical_not(gmask)] = xnp.logical_or(gmask[xnp.logical_not(gmask)], mask)
    return points[gmask], sky_distance / 2

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_residual_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_reflectance_dist : bool = False,
                 add_illumination_dist : bool = False,
                 add_residual_dist : bool = False,
                 use_residual : bool = False,
                 use_dual_transient: bool = True,
                 use_3D_filter : bool = False,
                 use_undependent_illumination : bool = False,
                 use_sg_illumination: bool = True,
                 use_asg_illumination: bool = True,
                 illumination_mode: str = "asg",
                 sg_lobes: int = 4,
                 sg_lambda_min: float = 1.0,
                 asg_lobes: int = 1,
                 asg_lambda_min: float = 1.0,
                 clamp_needle_render: bool = False,
                 needle_ratio_threshold: float = 5.0,
                 oblate_ratio_threshold: float = 20.0,
                 render_scale_max: float = 0.0,
                 render_min_opacity: float = 0.0,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_residual_dim = appearance_residual_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_reflectance_dist = add_reflectance_dist
        self.add_illumination_dist = add_illumination_dist
        self.add_residual_dist = add_residual_dist

        self.use_3D_filter = use_3D_filter
        self.use_dual_transient = use_dual_transient

        self.use_undependent_illumination = use_undependent_illumination
        self.use_sg_illumination = use_sg_illumination
        self.use_asg_illumination = use_asg_illumination
        self.illumination_mode = illumination_mode
        self.sg_lobes = sg_lobes
        self.sg_lambda_min = sg_lambda_min
        self.asg_lobes = asg_lobes
        self.asg_lambda_min = asg_lambda_min
        self.clamp_needle_render = clamp_needle_render
        self.needle_ratio_threshold = needle_ratio_threshold
        self.oblate_ratio_threshold = oblate_ratio_threshold
        self.render_scale_max = max(0.0, float(render_scale_max))
        self.render_min_opacity = max(0.0, float(render_min_opacity))
        self.geometry_frozen = False
        self.sg_illumination_available = use_sg_illumination
        self.asg_illumination_available = use_asg_illumination
        self.legacy_compatibility_mode = illumination_mode == "legacy"
        self.reflectance_detail_scale = 1.1
        self._last_reflectance_decoder_mean = torch.tensor(0.0, device="cuda")
        self.enhancement_sg_init_sharpness = 8.0
        self.enhancement_sg_amplitude_init = 0.05
        
        ## residual
        self.use_residual = use_residual

        self.render_enhancement = False

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._base_log_reflectance = torch.empty(0)
        self._reflectance_offset_delta = torch.empty(0)
        self._enhancement_sg_axis = torch.empty(0)
        self._enhancement_sg_sharpness = torch.empty(0)
        self._enhancement_sg_amplitude = torch.empty(0)
        self._enhancement_feat_weight = torch.empty(0)
        self._enhancement_illum_weight = torch.empty(0)
        self._enhancement_context_bias = torch.empty(0)
        self._illum_asg_axis = torch.empty(0)
        self._illum_asg_tangent = torch.empty(0)
        self._illum_asg_sharpness = torch.empty(0)
        self._illum_asg_amplitude = torch.empty(0)
        self._illum_asg_bias = torch.empty(0)
        self._illum_asg_dist_weight = torch.empty(0)

        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank: # weight of anchor
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        ## residual
        if self.use_residual:
            self.n_offsets_residual = int(n_offsets )
            self._offset_residual = torch.empty(0)
            self._anchor_feat_residual = torch.empty(0)
            self._scaling_residual = torch.empty(0)



        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0 # take distant as input or not 
        self.mlp_opacity = nn.Sequential(  # output n_offsets's opacity
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        if self.use_residual:
            self.mlp_opacity_residual = nn.Sequential(  # output n_offsets's opacity
                nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, self.n_offsets_residual),
                nn.Sigmoid()
            ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0 # take distant as input or not 
        self.mlp_cov = nn.Sequential( # output n_offsets's covriance
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        if self.use_residual:
            self.mlp_cov_residual = nn.Sequential( # output n_offsets's covriance
                nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 7*self.n_offsets_residual),
            ).cuda()
        self.reflectance_dist_dim = 1 if self.add_reflectance_dist else 0 # legacy only
        self.mlp_reflectance = nn.Sequential(
            nn.Linear(feat_dim + self.reflectance_dist_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
        self.mlp_reflectance_decoder = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3),
        ).cuda()
        self.illumination_dist_dim = 1 if self.add_illumination_dist else 0 # take distant as input or not
        self.mlp_illumination= nn.Sequential(
            nn.Linear(feat_dim// 2+3+self.illumination_dist_dim, feat_dim // 2),
            nn.ReLU(True),
            nn.Linear(feat_dim// 2, 1*self.n_offsets)
        ).cuda()
        self.mlp_sg_illumination = nn.Sequential(
            nn.Linear(feat_dim// 2+3+self.illumination_dist_dim, feat_dim // 2),
            nn.ReLU(True),
            nn.Linear(feat_dim // 2, self.n_offsets * self.sg_lobes * 5)
        ).cuda()
        if self.use_residual:
            self.residual_dist_dim = 1 if self.add_residual_dist else 0 # take distant as input or not
            if self.use_dual_transient:
                self.noise_net = nn.Sequential(
                    nn.Linear(feat_dim+3+self.residual_dist_dim+self.appearance_residual_dim, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, 3*self.n_offsets_residual),
                    nn.Tanh()
                ).cuda()
                self.artifact_net = nn.Sequential(
                    nn.Linear(feat_dim+3+self.residual_dist_dim+self.appearance_residual_dim, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, 3*self.n_offsets_residual),
                    nn.Tanh()
                ).cuda()
                self.residual_net = self.artifact_net
            else:
                self.residual_net = nn.Sequential(
                nn.Linear(feat_dim+3+self.residual_dist_dim+self.appearance_residual_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*self.n_offsets_residual),
                    nn.Sigmoid()
            ).cuda()

    def _reset_noise_net(self):
        if not self.use_residual or not self.use_dual_transient:
            return
        for module in self.noise_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_enhancement_sg_params(self, anchor_count, device="cuda", dtype=torch.float):
        axis = F.normalize(torch.randn((anchor_count, self.n_offsets, 3), device=device, dtype=dtype), dim=-1)
        sharpness = torch.log(torch.expm1(torch.full(
            (anchor_count, self.n_offsets, 1),
            self.enhancement_sg_init_sharpness,
            device=device,
            dtype=dtype,
        )))
        amplitude = inverse_sigmoid(torch.full(
            (anchor_count, self.n_offsets, 3),
            self.enhancement_sg_amplitude_init,
            device=device,
            dtype=dtype,
        ))
        return axis, sharpness, amplitude

    def _init_illumination_asg_params(self, anchor_count, device="cuda", dtype=torch.float):
        axis = F.normalize(
            torch.randn((anchor_count, self.n_offsets, self.asg_lobes, 3), device=device, dtype=dtype),
            dim=-1,
        )
        tangent = F.normalize(
            torch.randn((anchor_count, self.n_offsets, self.asg_lobes, 3), device=device, dtype=dtype),
            dim=-1,
        )
        sharpness = torch.log(torch.expm1(torch.full(
            (anchor_count, self.n_offsets, self.asg_lobes, 2),
            6.0,
            device=device,
            dtype=dtype,
        )))
        amplitude = inverse_sigmoid(torch.full(
            (anchor_count, self.n_offsets, self.asg_lobes, 1),
            0.04,
            device=device,
            dtype=dtype,
        ))
        bias = inverse_sigmoid(torch.full(
            (anchor_count, self.n_offsets, 1),
            0.08,
            device=device,
            dtype=dtype,
        ))
        dist_weight = torch.zeros((anchor_count, self.n_offsets, 1), device=device, dtype=dtype)
        return axis, tangent, sharpness, amplitude, bias, dist_weight

    def _ensure_illumination_asg_params(self):
        expected = (self._anchor.shape[0], self.n_offsets, self.asg_lobes)
        has_asg = (
            torch.is_tensor(self._illum_asg_axis)
            and self._illum_asg_axis.numel() > 0
            and tuple(self._illum_asg_axis.shape[:3]) == expected
        )
        if not has_asg:
            params = self._init_illumination_asg_params(
                self._anchor.shape[0],
                device=self._anchor.device,
                dtype=self._anchor.dtype,
            )
            (
                self._illum_asg_axis,
                self._illum_asg_tangent,
                self._illum_asg_sharpness,
                self._illum_asg_amplitude,
                self._illum_asg_bias,
                self._illum_asg_dist_weight,
            ) = [nn.Parameter(t.requires_grad_(True)) for t in params]
            return

        for name in (
            "_illum_asg_axis",
            "_illum_asg_tangent",
            "_illum_asg_sharpness",
            "_illum_asg_amplitude",
            "_illum_asg_bias",
            "_illum_asg_dist_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, nn.Parameter):
                setattr(self, name, nn.Parameter(value.requires_grad_(True)))

    def _ensure_enhancement_sg_params(self):
        if (
            not torch.is_tensor(self._enhancement_feat_weight)
            or self._enhancement_feat_weight.numel() == 0
            or tuple(self._enhancement_feat_weight.shape) != (self.feat_dim, 3)
            or tuple(self._enhancement_illum_weight.shape) != (self.n_offsets, 3)
            or tuple(self._enhancement_context_bias.shape) != (1, self.n_offsets, 3)
        ):
            self._enhancement_feat_weight = nn.Parameter(
                torch.zeros((self.feat_dim, 3), device=self._anchor.device, dtype=self._anchor.dtype).requires_grad_(True)
            )
            self._enhancement_illum_weight = nn.Parameter(
                torch.zeros((self.n_offsets, 3), device=self._anchor.device, dtype=self._anchor.dtype).requires_grad_(True)
            )
            self._enhancement_context_bias = nn.Parameter(
                torch.zeros((1, self.n_offsets, 3), device=self._anchor.device, dtype=self._anchor.dtype).requires_grad_(True)
            )
        else:
            if not isinstance(self._enhancement_feat_weight, nn.Parameter):
                self._enhancement_feat_weight = nn.Parameter(self._enhancement_feat_weight.requires_grad_(True))
            if not isinstance(self._enhancement_illum_weight, nn.Parameter):
                self._enhancement_illum_weight = nn.Parameter(self._enhancement_illum_weight.requires_grad_(True))
            if not isinstance(self._enhancement_context_bias, nn.Parameter):
                self._enhancement_context_bias = nn.Parameter(self._enhancement_context_bias.requires_grad_(True))

        if (
            torch.is_tensor(self._enhancement_sg_axis)
            and self._enhancement_sg_axis.numel() > 0
            and self._enhancement_sg_axis.shape[0] == self._anchor.shape[0]
        ):
            if not isinstance(self._enhancement_sg_axis, nn.Parameter):
                self._enhancement_sg_axis = nn.Parameter(self._enhancement_sg_axis.requires_grad_(True))
            if not isinstance(self._enhancement_sg_sharpness, nn.Parameter):
                self._enhancement_sg_sharpness = nn.Parameter(self._enhancement_sg_sharpness.requires_grad_(True))
            if not isinstance(self._enhancement_sg_amplitude, nn.Parameter):
                self._enhancement_sg_amplitude = nn.Parameter(self._enhancement_sg_amplitude.requires_grad_(True))
            return
        axis, sharpness, amplitude = self._init_enhancement_sg_params(
            self._anchor.shape[0],
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        )
        self._enhancement_sg_axis = nn.Parameter(axis.requires_grad_(True))
        self._enhancement_sg_sharpness = nn.Parameter(sharpness.requires_grad_(True))
        self._enhancement_sg_amplitude = nn.Parameter(amplitude.requires_grad_(True))

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        # self.mlp_color.eval()
        if self.illumination_mode == "legacy":
            self.mlp_illumination.eval()
        if self.illumination_mode == "sg" and self.use_sg_illumination:
            self.mlp_sg_illumination.eval()
        self.mlp_reflectance_decoder.eval()
        if self.use_residual:
            if self.use_dual_transient:
                self.noise_net.eval()
                self.artifact_net.eval()
            else:
                self.residual_net.eval()
            self.mlp_cov_residual.eval()
            self.mlp_opacity_residual.eval()
        if self.appearance_residual_dim > 0: # use appearance embedding or not
            self.embedding_appearance.eval()
        if self.use_feat_bank: # use anchor feature or not (mutil-resolution)
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        # self.mlp_color.train()
        if self.illumination_mode == "legacy":
            self.mlp_illumination.train()
        if self.illumination_mode == "sg" and self.use_sg_illumination:
            self.mlp_sg_illumination.train()
        self.mlp_reflectance_decoder.train()
        if self.use_residual:
            if self.use_dual_transient:
                self.noise_net.train()
                self.artifact_net.train()
            else:
                self.residual_net.train()
            self.mlp_cov_residual.train()
            self.mlp_opacity_residual.train()
        if self.appearance_residual_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        # ``self.denom`` belonged to an older densification implementation and
        # is not created by the current anchor/offset statistics path.  Keep a
        # shape-compatible value in the legacy checkpoint slot so checkpoint
        # saving remains backward compatible.
        checkpoint_denom = getattr(self, "denom", None)
        if checkpoint_denom is None:
            checkpoint_denom = getattr(self, "anchor_demon", None)
        if checkpoint_denom is None or checkpoint_denom.numel() == 0:
            checkpoint_denom = torch.zeros(
                (self._anchor.shape[0], 1),
                dtype=self._anchor.dtype,
                device=self._anchor.device,
            )
        if self.use_residual:
            return (
                self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._enhancement_feat_weight,
                self._enhancement_illum_weight,
                self._enhancement_context_bias,
                self._illum_asg_axis,
                self._illum_asg_tangent,
                self._illum_asg_sharpness,
                self._illum_asg_amplitude,
                self._illum_asg_bias,
                self._illum_asg_dist_weight,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                checkpoint_denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._enhancement_feat_weight,
                self._enhancement_illum_weight,
                self._enhancement_context_bias,
                self._illum_asg_axis,
                self._illum_asg_tangent,
                self._illum_asg_sharpness,
                self._illum_asg_amplitude,
                self._illum_asg_bias,
                self._illum_asg_dist_weight,
                self._offset,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                checkpoint_denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
    
    def restore(self, model_args, training_args):
        if self.use_residual:
            has_asg_context = len(model_args) == 27
            has_enhancement_context = len(model_args) == 21
            has_enhancement_sg = len(model_args) == 18
            has_reflectance_detail = len(model_args) == 15
            has_b0 = len(model_args) == 14
            if has_asg_context:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._enhancement_feat_weight,
                self._enhancement_illum_weight,
                self._enhancement_context_bias,
                self._illum_asg_axis,
                self._illum_asg_tangent,
                self._illum_asg_sharpness,
                self._illum_asg_amplitude,
                self._illum_asg_bias,
                self._illum_asg_dist_weight,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_enhancement_context:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._enhancement_feat_weight,
                self._enhancement_illum_weight,
                self._enhancement_context_bias,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_enhancement_sg:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_reflectance_detail:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_b0:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
                self._reflectance_offset_delta = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], self.n_offsets, 3), device="cuda", dtype=torch.float).requires_grad_(True)
                )
            else:
                (self._anchor,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                _unused_local,
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
                self._anchor_feat = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], self.feat_dim), device="cuda", dtype=torch.float).requires_grad_(True)
                )
                self._base_log_reflectance = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], 3), device="cuda", dtype=torch.float).requires_grad_(True)
                )
                self._reflectance_offset_delta = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], self.n_offsets, 3), device="cuda", dtype=torch.float).requires_grad_(True)
                )
                self.illumination_mode = "legacy"
                self.legacy_compatibility_mode = True
            if not has_asg_context:
                self.asg_illumination_available = False
                if self.illumination_mode == "asg":
                    self.illumination_mode = "legacy"
                    self.legacy_compatibility_mode = True
            self._ensure_enhancement_sg_params()
            self._ensure_illumination_asg_params()
            self.training_setup(training_args)
            self.denom = denom
            try:
                self.optimizer.load_state_dict(opt_dict)
            except ValueError:
                print("Optimizer checkpoint is not compatible with SG parameter groups; optimizer state is reinitialized.")
        else:
            has_asg_context = len(model_args) == 24
            has_enhancement_context = len(model_args) == 18
            has_enhancement_sg = len(model_args) == 15
            has_reflectance_detail = len(model_args) == 12
            has_b0 = len(model_args) == 11
            if has_asg_context:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._enhancement_feat_weight,
                self._enhancement_illum_weight,
                self._enhancement_context_bias,
                self._illum_asg_axis,
                self._illum_asg_tangent,
                self._illum_asg_sharpness,
                self._illum_asg_amplitude,
                self._illum_asg_bias,
                self._illum_asg_dist_weight,
                self._offset,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_enhancement_context:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._enhancement_feat_weight,
                self._enhancement_illum_weight,
                self._enhancement_context_bias,
                self._offset,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_enhancement_sg:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._enhancement_sg_axis,
                self._enhancement_sg_sharpness,
                self._enhancement_sg_amplitude,
                self._offset,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_reflectance_detail:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._reflectance_offset_delta,
                self._offset,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
            elif has_b0:
                (self._anchor,
                self._anchor_feat,
                self._base_log_reflectance,
                self._offset,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
                self._reflectance_offset_delta = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], self.n_offsets, 3), device="cuda", dtype=torch.float).requires_grad_(True)
                )
            else:
                (self._anchor,
                self._offset,
                _unused_local,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                denom,
                opt_dict,
                self.spatial_lr_scale) = model_args
                self._anchor_feat = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], self.feat_dim), device="cuda", dtype=torch.float).requires_grad_(True)
                )
                self._base_log_reflectance = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], 3), device="cuda", dtype=torch.float).requires_grad_(True)
                )
                self._reflectance_offset_delta = nn.Parameter(
                    torch.zeros((self._anchor.shape[0], self.n_offsets, 3), device="cuda", dtype=torch.float).requires_grad_(True)
                )
                self.illumination_mode = "legacy"
                self.legacy_compatibility_mode = True
            if not has_asg_context:
                self.asg_illumination_available = False
                if self.illumination_mode == "asg":
                    self.illumination_mode = "legacy"
                    self.legacy_compatibility_mode = True
            self._ensure_enhancement_sg_params()
            self._ensure_illumination_asg_params()
            self.training_setup(training_args)
            self.denom = denom
            try:
                self.optimizer.load_state_dict(opt_dict)
            except ValueError:
                print("Optimizer checkpoint is not compatible with SG parameter groups; optimizer state is reinitialized.")

    def set_appearance_residual(self, num_cameras):
        if self.appearance_residual_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_residual_dim).cuda()

    @property
    def get_appearance_residual(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    

    def get_scaling_with_3D_filter(self, scales, visible_mask):
        scales = torch.square(scales) + torch.square(self.filter_3D[visible_mask])
        scales = torch.sqrt(scales)
        return scales  

    @property
    def get_scaling_residual(self):
        return 1.0*self.scaling_activation(self._scaling_residual)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    

    def get_opacity_with_3D_filter(self, opacity, visible_mask):
        # apply 3D filter
        scales = self.get_scaling[visible_mask]
        scales = scales.repeat(self.n_offsets, 1)
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D[visible_mask].repeat(self.n_offsets, 1)) 
        det2 = scales_after_square.prod(dim=1) 

        eps = 1e-10  # 小的常数值
        det1 = torch.clamp_min(det1, eps)
        det2 = torch.clamp_min(det2, eps)
        coef = torch.sqrt(det1 / det2)
        if torch.any(torch.isnan(coef)) or torch.any(torch.isinf(coef)):
            import pdb;pdb.set_trace()
        return opacity * coef[..., None]

    @property
    def get_opacity_residual_mlp(self):
        return self.mlp_opacity_residual

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_cov_residual_mlp(self):
        return self.mlp_cov_residual
    # @property
    # def get_color_mlp(self):
    #     return self.mlp_color

    @property
    def get_base_log_reflectance(self):
        return self._base_log_reflectance

    @property
    def get_reflectance(self):
        return torch.exp(self._base_log_reflectance)

    @property
    def get_reflectance_with_detail(self):
        if self._base_log_reflectance.shape[-1] != 3:
            base = self._base_log_reflectance[..., :3]
        else:
            base = self._base_log_reflectance
        detail = self.reflectance_detail_scale * torch.tanh(self._reflectance_offset_delta)
        return torch.exp(base.unsqueeze(1) + detail)

    def get_reflectance_with_decoder(self, feat, offsets, visible_mask):
        reflectance = self.get_reflectance_with_detail[visible_mask]
        feat_repeated = feat.unsqueeze(1).expand(-1, self.n_offsets, -1)
        decoder_input = torch.cat([feat_repeated, offsets], dim=-1).reshape(-1, self.feat_dim + 3)
        decoder_out = self.mlp_reflectance_decoder(decoder_input).reshape(-1, self.n_offsets, 3)
        decoder_refine = 1.0 + 0.15 * torch.tanh(decoder_out)
        self._last_reflectance_decoder_mean = torch.abs(torch.tanh(decoder_out)).mean()
        return torch.clamp(reflectance * decoder_refine, 1e-3, 1.0)

    def get_enhanced_illumination(self, feat, illumination_feat, view_dirs, visible_mask):
        self._ensure_enhancement_sg_params()
        axis = F.normalize(self._enhancement_sg_axis[visible_mask], dim=-1)
        sharpness = F.softplus(self._enhancement_sg_sharpness[visible_mask])
        amplitude = torch.sigmoid(self._enhancement_sg_amplitude[visible_mask])

        view_dirs = view_dirs.view(-1, 1, 3)
        cosine = torch.sum(axis * view_dirs, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        sg_term = amplitude * torch.exp(sharpness * (cosine - 1.0))

        illumination_context = torch.sigmoid(illumination_feat) if self.legacy_compatibility_mode else illumination_feat
        illumination_context = illumination_context.detach()
        feat_gain = feat.detach().matmul(self._enhancement_feat_weight).view(-1, 1, 3)
        illum_gain = illumination_context.view(-1, self.n_offsets, 1) * self._enhancement_illum_weight.view(1, self.n_offsets, 3)
        context_gain = feat_gain + illum_gain
        input_gate = torch.sigmoid(context_gain + self._enhancement_context_bias)

        base_illumination = illumination_context.view(-1, self.n_offsets, 1).expand(-1, -1, 3)
        enhanced = torch.clamp(base_illumination + input_gate * sg_term, 0.0, 1.0)
        return enhanced.reshape(-1, 3)

    @property
    def get_illumination_mlp(self):
        return self.mlp_illumination

    @property
    def get_sg_illumination_mlp(self):
        return self.mlp_sg_illumination
    
    @property
    def get_noise_net(self):
        if not self.use_dual_transient:
            raise AttributeError("noise_net is only available when use_dual_transient=True")
        return self.noise_net

    @property
    def get_artifact_net(self):
        return self.artifact_net if self.use_dual_transient else self.residual_net

    @property
    def get_residual_net(self):
        return self.artifact_net if self.use_dual_transient else self.residual_net
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

###

    def init_RT_seq(self, cam_list):
        poses = []
        for cam in cam_list[1.0]:
            p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1))
            poses.append(p)
        poses = torch.stack(poses)
        self.P = poses.cuda().requires_grad_(True)

    def get_RT(self, idx):
        pose = self.P[idx]
        return pose

    def get_RT_test(self, idx):
        pose = self.test_P[idx]
        return pose
    
    def get_closest_RT(self, pose):
        poses_list = self.P
        distances = torch.norm(poses_list[:, 4:] - pose[4:].unsqueeze(0), dim=1)
        index = torch.randint(1, 3, (1,)).item() 
        min_distance_idx = torch.argmin(distances)
        distances[min_distance_idx] = float('inf')
        
        # 选择最近的姿态
        closest_idx = torch.argmin(distances)
        return self.P[closest_idx], closest_idx
#####


    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    # def get_covariance_residual(self, scaling_modifier = 1):
    #     return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation_residual)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_anchor
        print("points number:", xyz.shape[0])
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size # resize to voxel space and remove the repeat parts
        
        return data

    def _estimate_initial_b0(self, anchors: torch.Tensor, cameras) -> torch.Tensor:
        N = anchors.shape[0]
        b0 = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        if cameras is None or len(cameras) == 0:
            return b0

        cam = cameras[0]
        image = cam.original_image  # (3, H, W)
        H, W = image.shape[1], image.shape[2]

        max_c_img = image.max(dim=0, keepdim=True)[0].clamp(min=1e-1)
        reflectance_map = (image / max_c_img).clamp(1e-3, 1.0)
        reflectance_map_bchw = reflectance_map.unsqueeze(0)
        reflectance_map_blur = F.avg_pool2d(reflectance_map_bchw, kernel_size=5, stride=1, padding=2).squeeze(0)
        detail = reflectance_map - reflectance_map_blur
        reflectance_map = (reflectance_map + 0.4 * detail).clamp(1e-3, 1.0)
        log_reflectance_map = torch.log(reflectance_map)
        global_mean_b0 = log_reflectance_map.mean(dim=(1, 2))

        ones = torch.ones((N, 1), dtype=torch.float, device="cuda")
        pts_h = torch.cat([anchors, ones], dim=1)  # (N, 4)
        proj = cam.full_proj_transform  # (4, 4)
        pts_clip = pts_h @ proj.T  # (N, 4)
        w = pts_clip[:, 3:].clamp(min=1e-6)
        pts_ndc = pts_clip[:, :3] / w  # (N, 3)

        px = ((pts_ndc[:, 0] + 1.0) * 0.5 * (W - 1)).long().clamp(0, W - 1)
        py = ((1.0 - pts_ndc[:, 1]) * 0.5 * (H - 1)).long().clamp(0, H - 1)

        valid = (pts_ndc[:, 2] > -1.0) & (pts_ndc[:, 2] < 1.0)
        if valid.any():
            b0[valid] = log_reflectance_map[:, py[valid], px[valid]].T
        if (~valid).any():
            b0[~valid] = global_mean_b0
        return b0

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, num_sky_gaussians=0, cameras=None, prune_ratio : float = 0.05,model_path=None, beta=1, skybox_scale_max_factor: float = 0.0):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points # 
        os.makedirs( os.path.join(model_path, 'dust3r'), exist_ok=True)
        plot_point_cloud_projection(torch.tensor(pcd.points).cuda(), cameras.copy()[0], os.path.join(model_path, 'dust3r', f"anchor_in_view0_dust3r.png"), alpha=0.2)
        # for i in range(len(cameras)):
        #     plot_point_cloud_projection(torch.tensor(pcd.points).cuda(), cameras.copy()[i], os.path.join(model_path, 'dust3r', f"anchor_in_view{i}_dust3r.png"), alpha=0.2)



        if self.voxel_size <= 0: # auto-obtain the voxel_size
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        ## dust3r + downsampling
        # points = self.voxelize_sample(points, voxel_size=self.voxel_size ) # resize to voxel space, turn to the voxel (N, x, y)
        # down_size = self.voxel_size
        # # points = self.voxelize_sample(points, voxel_size=down_size * 4) #50,4
        # while len(points) > 200000:
        #     down_size *= 1.5  # 逐步增大体素
        #     points = self.voxelize_sample(points, voxel_size=down_size) #50



        ## dust3r+ FPS(Farthest Point Sampling)
        # def farthest_point_sampling(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
        #     """
        #     xyz: (N, 3) input point cloud
        #     n_samples: number of points to sample
        #     return: (n_samples,) indices of sampled points
        #     """
        #     N, _ = xyz.shape
        #     centroids = torch.zeros(n_samples, dtype=torch.long, device=xyz.device)
        #     distance = torch.ones(N, device=xyz.device) * 1e10
        #     farthest = torch.randint(0, N, (1,), device=xyz.device).item()
        #     for i in range(n_samples):
        #         centroids[i] = farthest
        #         centroid = xyz[farthest].unsqueeze(0)  # (1, 3)
        #         dist = torch.sum((xyz - centroid) ** 2, dim=1)
        #         mask = dist < distance
        #         distance[mask] = dist[mask]
        #         farthest = torch.max(distance, dim=0)[1].item()
        #     return centroids
        # fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        # original_points_size = fused_point_cloud.shape[0]
        # target_points = int(original_points_size * prune_ratio)

        # if target_points < fused_point_cloud.shape[0]:
        #     print(f"Original point cloud size: {original_points_size}, Target size after FPS: {target_points}")
        #     sampled_indices = farthest_point_sampling(fused_point_cloud, target_points)
        #     fused_point_cloud = fused_point_cloud[sampled_indices]
        #     print(f"Sampled point cloud size: {fused_point_cloud.shape[0]}")


        # LLGIM
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        original_points_size = fused_point_cloud.shape[0]
        print(f"original points size: {original_points_size}")
        # import pdb; pdb.set_trace()
        # 根据距离随机裁剪点云
        tau= 1
        visualize_anchor(fused_point_cloud.detach().cpu().numpy(), os.path.join(model_path, 'anchor_without_prune.png'))
        print("prune_ratio:", prune_ratio)
        while fused_point_cloud.shape[0] > original_points_size * prune_ratio and prune_ratio < 1:  
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
            # tau = torch.max(torch.tensor(1), tau * torch.exp(- torch.tensor(fused_point_cloud.shape[0]/original_points_size)))
            tau *= torch.exp( 1.0 * torch.tensor(beta * fused_point_cloud.shape[0]/original_points_size))
            dist2_threshold = torch.tensor(self.voxel_size * tau)
            print("tau:", tau)
            
            
            # 计算保留概率,距离越小概率越小
            probs = dist2 / dist2_threshold
            probs = torch.clamp(probs, 0.5, 1)
        
            # # 随机采样生成mask
            rand = torch.rand_like(dist2)
            # rand_idx = torch.randint(0, dist2.shape[0], (dist2.shape[0]*9//10,))
            mask = rand < probs
            # mask[rand_idx] = True
            dist2 = dist2[mask]
            fused_point_cloud = fused_point_cloud[mask]
            print(fused_point_cloud.shape[0])
        




        

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        visualize_anchor(fused_point_cloud.detach().cpu().numpy(), os.path.join(model_path, 'anchor.png'))
        skybox_count = 0
        if num_sky_gaussians:
            th_cameras = cameras
            skybox, self._sky_distance = get_sky_points(num_sky_gaussians, fused_point_cloud, th_cameras)
            skybox = skybox
            print(f"Adding skybox with {skybox.shape[0]} points")
            fused_point_cloud = torch.cat((fused_point_cloud, skybox), dim=0)
            skybox_count = skybox.shape[0]
            opacities = torch.cat((opacities, inverse_sigmoid(torch.ones((skybox.shape[0], 1), dtype=torch.float, device="cuda"))), dim=0)

        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda() # use to caculate the position of 3d gaussians
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001) # get the distance of the voxel center and prune the overlapping voxel
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6) 
        if skybox_count > 0 and skybox_scale_max_factor > 0:
            max_sky_scale = max(float(self._sky_distance) * skybox_scale_max_factor, 1e-8)
            scales[-skybox_count:] = torch.minimum(
                scales[-skybox_count:],
                torch.full_like(scales[-skybox_count:], math.log(max_sky_scale)),
            )
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        

        if self.use_residual:
            anchors_feat_residual = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
            offsets_residual = torch.zeros((fused_point_cloud.shape[0], self.n_offsets_residual, 3)).float().cuda()
            scales_residual = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6) 
        base_log_reflectance = self._estimate_initial_b0(fused_point_cloud, cameras)
        reflectance_offset_delta = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3), dtype=torch.float, device="cuda")
        enhancement_sg_axis, enhancement_sg_sharpness, enhancement_sg_amplitude = self._init_enhancement_sg_params(
            fused_point_cloud.shape[0],
            device="cuda",
            dtype=torch.float,
        )
        illum_asg_axis, illum_asg_tangent, illum_asg_sharpness, illum_asg_amplitude, illum_asg_bias, illum_asg_dist_weight = self._init_illumination_asg_params(
            fused_point_cloud.shape[0],
            device="cuda",
            dtype=torch.float,
        )


        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._base_log_reflectance = nn.Parameter(base_log_reflectance.requires_grad_(True))
        self._reflectance_offset_delta = nn.Parameter(reflectance_offset_delta.requires_grad_(True))
        self._enhancement_sg_axis = nn.Parameter(enhancement_sg_axis.requires_grad_(True))
        self._enhancement_sg_sharpness = nn.Parameter(enhancement_sg_sharpness.requires_grad_(True))
        self._enhancement_sg_amplitude = nn.Parameter(enhancement_sg_amplitude.requires_grad_(True))
        self._illum_asg_axis = nn.Parameter(illum_asg_axis.requires_grad_(True))
        self._illum_asg_tangent = nn.Parameter(illum_asg_tangent.requires_grad_(True))
        self._illum_asg_sharpness = nn.Parameter(illum_asg_sharpness.requires_grad_(True))
        self._illum_asg_amplitude = nn.Parameter(illum_asg_amplitude.requires_grad_(True))
        self._illum_asg_bias = nn.Parameter(illum_asg_bias.requires_grad_(True))
        self._illum_asg_dist_weight = nn.Parameter(illum_asg_dist_weight.requires_grad_(True))
        if self.use_residual:
            self._anchor_feat_residual = nn.Parameter(anchors_feat_residual.requires_grad_(True))
            self._offset_residual = nn.Parameter(offsets_residual.requires_grad_(True))
            self._scaling_residual = nn.Parameter(scales_residual.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")




    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self._ensure_enhancement_sg_params()
        self._ensure_illumination_asg_params()

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")


        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._base_log_reflectance], 'lr': training_args.b0_lr, "name": "base_log_reflectance"},
                {'params': [self._reflectance_offset_delta], 'lr': training_args.reflectance_offset_lr, "name": "reflectance_offset_delta"},
                {'params': self.mlp_reflectance_decoder.parameters(), 'lr': training_args.reflectance_decoder_lr, "name": "mlp_reflectance_decoder"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_sg_illumination.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_sg_illumination"},
                {'params': [self._illum_asg_axis], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_axis"},
                {'params': [self._illum_asg_tangent], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_tangent"},
                {'params': [self._illum_asg_sharpness], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_sharpness"},
                {'params': [self._illum_asg_amplitude], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_amplitude"},
                {'params': [self._illum_asg_bias], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_bias"},
                {'params': [self._illum_asg_dist_weight], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_dist_weight"},
                {'params': [self._enhancement_sg_axis], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_axis"},
                {'params': [self._enhancement_sg_sharpness], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_sharpness"},
                {'params': [self._enhancement_sg_amplitude], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_amplitude"},
                {'params': [self._enhancement_feat_weight], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_feat"},
                {'params': [self._enhancement_illum_weight], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_illum"},
                {'params': [self._enhancement_context_bias], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_bias"},
                # {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_residual_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._base_log_reflectance], 'lr': training_args.b0_lr, "name": "base_log_reflectance"},
                {'params': [self._reflectance_offset_delta], 'lr': training_args.reflectance_offset_lr, "name": "reflectance_offset_delta"},
                {'params': self.mlp_reflectance_decoder.parameters(), 'lr': training_args.reflectance_decoder_lr, "name": "mlp_reflectance_decoder"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_sg_illumination.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_sg_illumination"},
                {'params': [self._illum_asg_axis], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_axis"},
                {'params': [self._illum_asg_tangent], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_tangent"},
                {'params': [self._illum_asg_sharpness], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_sharpness"},
                {'params': [self._illum_asg_amplitude], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_amplitude"},
                {'params': [self._illum_asg_bias], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_bias"},
                {'params': [self._illum_asg_dist_weight], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_dist_weight"},
                {'params': [self._enhancement_sg_axis], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_axis"},
                {'params': [self._enhancement_sg_sharpness], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_sharpness"},
                {'params': [self._enhancement_sg_amplitude], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_amplitude"},
                {'params': [self._enhancement_feat_weight], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_feat"},
                {'params': [self._enhancement_illum_weight], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_illum"},
                {'params': [self._enhancement_context_bias], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_bias"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._base_log_reflectance], 'lr': training_args.b0_lr, "name": "base_log_reflectance"},
                {'params': [self._reflectance_offset_delta], 'lr': training_args.reflectance_offset_lr, "name": "reflectance_offset_delta"},
                {'params': self.mlp_reflectance_decoder.parameters(), 'lr': training_args.reflectance_decoder_lr, "name": "mlp_reflectance_decoder"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_sg_illumination.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_sg_illumination"},
                {'params': [self._illum_asg_axis], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_axis"},
                {'params': [self._illum_asg_tangent], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_tangent"},
                {'params': [self._illum_asg_sharpness], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_sharpness"},
                {'params': [self._illum_asg_amplitude], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_amplitude"},
                {'params': [self._illum_asg_bias], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_bias"},
                {'params': [self._illum_asg_dist_weight], 'lr': training_args.mlp_color_lr_init, "name": "illum_asg_dist_weight"},
                {'params': [self._enhancement_sg_axis], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_axis"},
                {'params': [self._enhancement_sg_sharpness], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_sharpness"},
                {'params': [self._enhancement_sg_amplitude], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_sg_amplitude"},
                {'params': [self._enhancement_feat_weight], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_feat"},
                {'params': [self._enhancement_illum_weight], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_illum"},
                {'params': [self._enhancement_context_bias], 'lr': training_args.mlp_enhance_lr_init, "name": "enhancement_context_bias"},
            ]
        if self.use_residual:
            l.append({'params': [self._anchor_feat_residual], 'lr': training_args.feature_lr , "name": "anchor_feat_residual"})
            l.append({'params': [self._offset_residual], 'lr': training_args.offset_lr_init * self.spatial_lr_scale  , "name": "offset_residual"})
            l.append({'params': [self._scaling_residual], 'lr': training_args.scaling_lr, "name": "scaling_residual"})
            if self.use_dual_transient:
                l.append({'params': self.noise_net.parameters(), 'lr': training_args.mlp_color_lr_init , "name": "noise_net"})
                l.append({'params': self.artifact_net.parameters(), 'lr': training_args.mlp_color_lr_init , "name": "artifact_net"})
            else:
                l.append({'params': self.residual_net.parameters(), 'lr': training_args.mlp_color_lr_init , "name": "residual_net"})
            l.append({'params': self.mlp_opacity_residual.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity_residual"})
            l.append({'params': self.mlp_cov_residual.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov_residual"})
            
#####
        l_cam = [{'params':[self.P], 'lr':training_args.pose_lr_init, "name": "pose"},]
        
        l += l_cam
#####
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale ,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        # self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
        #                                             lr_final=training_args.mlp_color_lr_final,
        #                                             lr_delay_mult=training_args.mlp_color_lr_delay_mult,
        #                                             max_steps=training_args.mlp_color_lr_max_steps)
        self.mlp_sg_illumination_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        self.illum_asg_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        self.enhancement_sg_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_enhance_lr_init ,
                                                    lr_final=training_args.mlp_enhance_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        self.b0_scheduler_args = get_expon_lr_func(lr_init=training_args.b0_lr,
                                                    lr_final=training_args.b0_lr * 0.01,
                                                    lr_delay_mult=0.01,
                                                    max_steps=training_args.position_lr_max_steps)
        self.reflectance_offset_scheduler_args = get_expon_lr_func(lr_init=training_args.reflectance_offset_lr,
                                                    lr_final=training_args.reflectance_offset_lr * 0.01,
                                                    lr_delay_mult=0.01,
                                                    max_steps=training_args.position_lr_max_steps)
        self.reflectance_decoder_scheduler_args = get_expon_lr_func(lr_init=training_args.reflectance_decoder_lr,
                                                    lr_final=training_args.reflectance_decoder_lr * 0.01,
                                                    lr_delay_mult=0.01,
                                                    max_steps=training_args.position_lr_max_steps)
        if self.use_residual:
            self.residual_net_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
            if self.use_dual_transient:
                self.noise_net_scheduler_args = self.residual_net_scheduler_args
                self.artifact_net_scheduler_args = self.residual_net_scheduler_args
            
            self.offset_residual_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale * 5,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale * 5,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
            
            self.mlp_opacity_residual_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init ,
                                                        lr_final=training_args.mlp_opacity_lr_final ,
                                                        lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                        max_steps=training_args.mlp_opacity_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_residual_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
            
        self.pose_scheduler_args = get_expon_lr_func(lr_init=training_args.pose_lr_init,
                                                    lr_final=training_args.pose_lr_final,
                                                    lr_delay_mult=training_args.pose_lr_delay_mult,
                                                    max_steps=training_args.pose_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            # if param_group["name"] == "mlp_color":
            #     lr = self.mlp_color_scheduler_args(iteration)
            #     param_group['lr'] = lr
            if param_group["name"] == "mlp_sg_illumination":
                lr = self.mlp_sg_illumination_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] in {"illum_asg_axis", "illum_asg_tangent", "illum_asg_sharpness", "illum_asg_amplitude", "illum_asg_bias", "illum_asg_dist_weight"}:
                lr = self.illum_asg_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] in {"enhancement_sg_axis", "enhancement_sg_sharpness", "enhancement_sg_amplitude", "enhancement_context_feat", "enhancement_context_illum", "enhancement_context_bias"}:
                lr = self.enhancement_sg_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "base_log_reflectance":
                lr = self.b0_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "reflectance_offset_delta":
                lr = self.reflectance_offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_reflectance_decoder":
                lr = self.reflectance_decoder_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_residual_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group["name"] == "offset_residual":
                lr = self.offset_residual_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group["name"] == "noise_net":
                lr = self.noise_net_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group["name"] == "artifact_net":
                lr = self.artifact_net_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group["name"] == "residual_net":
                lr = self.residual_net_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group['name'] == "mlp_opacity_residual":
                lr = self.mlp_opacity_residual_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group['name'] == "mlp_cov_residual":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group['name'] == "pose":
                lr = self.pose_scheduler_args(iteration)
                param_group['lr'] = lr


    def freeze(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] not in {"enhancement_sg_axis", "enhancement_sg_sharpness", "enhancement_sg_amplitude", "enhancement_context_feat", "enhancement_context_illum", "enhancement_context_bias", "illum_asg_axis", "illum_asg_tangent", "illum_asg_sharpness", "illum_asg_amplitude", "illum_asg_bias", "illum_asg_dist_weight", "base_log_reflectance", "reflectance_offset_delta", "mlp_reflectance_decoder"}:
                param_group['lr'] = 0

    def freeze_geometry(self):
        """Freeze geometry and covariance after the geometry-first stage."""
        if self.geometry_frozen:
            return
        for tensor in (self._anchor, self._offset, self._anchor_feat, self._scaling, self._rotation, self.P):
            tensor.requires_grad_(False)
        for parameter in self.mlp_cov.parameters():
            parameter.requires_grad_(False)
        for parameter in self.mlp_opacity.parameters():
            parameter.requires_grad_(False)
        geometry_groups = {
            "anchor", "offset", "anchor_feat", "scaling", "rotation", "pose",
            "mlp_cov", "mlp_opacity",
        }
        if self.use_residual:
            for tensor in (self._anchor_feat_residual, self._offset_residual, self._scaling_residual):
                tensor.requires_grad_(False)
            for parameter in self.mlp_cov_residual.parameters():
                parameter.requires_grad_(False)
            for parameter in self.mlp_opacity_residual.parameters():
                parameter.requires_grad_(False)
            geometry_groups.update({
                "anchor_feat_residual", "offset_residual", "scaling_residual",
                "mlp_cov_residual", "mlp_opacity_residual",
            })
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in geometry_groups:
                param_group["lr"] = 0.0
        self.geometry_frozen = True

                

    ### need debug
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._base_log_reflectance.shape[1]):
            l.append('b0_{}'.format(i))
        for i in range(self._reflectance_offset_delta.shape[1] * self._reflectance_offset_delta.shape[2]):
            l.append('b0_detail_{}'.format(i))
        for i in range(self._enhancement_sg_axis.shape[1] * self._enhancement_sg_axis.shape[2]):
            l.append('enh_sg_axis_{}'.format(i))
        for i in range(self._enhancement_sg_sharpness.shape[1] * self._enhancement_sg_sharpness.shape[2]):
            l.append('enh_sg_sharpness_{}'.format(i))
        for i in range(self._enhancement_sg_amplitude.shape[1] * self._enhancement_sg_amplitude.shape[2]):
            l.append('enh_sg_amplitude_{}'.format(i))
        for i in range(self._illum_asg_axis.shape[1] * self._illum_asg_axis.shape[2] * self._illum_asg_axis.shape[3]):
            l.append('illum_asg_axis_{}'.format(i))
        for i in range(self._illum_asg_tangent.shape[1] * self._illum_asg_tangent.shape[2] * self._illum_asg_tangent.shape[3]):
            l.append('illum_asg_tangent_{}'.format(i))
        for i in range(self._illum_asg_sharpness.shape[1] * self._illum_asg_sharpness.shape[2] * self._illum_asg_sharpness.shape[3]):
            l.append('illum_asg_sharpness_{}'.format(i))
        for i in range(self._illum_asg_amplitude.shape[1] * self._illum_asg_amplitude.shape[2] * self._illum_asg_amplitude.shape[3]):
            l.append('illum_asg_amplitude_{}'.format(i))
        for i in range(self._illum_asg_bias.shape[1] * self._illum_asg_bias.shape[2]):
            l.append('illum_asg_bias_{}'.format(i))
        for i in range(self._illum_asg_dist_weight.shape[1] * self._illum_asg_dist_weight.shape[2]):
            l.append('illum_asg_dist_weight_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('filter_3D')
        if self.use_residual:
            for i in range(self._anchor_feat_residual.shape[1]):
                l.append('r_anchor_feat_residual_{}'.format(i))
            for i in range(self._scaling_residual.shape[1]):
                l.append('r_scale_residual_{}'.format(i))
            for i in range(self._offset_residual.shape[1]*self._offset_residual.shape[2]):
                l.append('r_offset_residual_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        base_log_reflectance = self._base_log_reflectance.detach().cpu().numpy()
        reflectance_offset_delta = self._reflectance_offset_delta.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        enhancement_sg_axis = self._enhancement_sg_axis.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        enhancement_sg_sharpness = self._enhancement_sg_sharpness.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        enhancement_sg_amplitude = self._enhancement_sg_amplitude.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        illum_asg_axis = self._illum_asg_axis.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        illum_asg_tangent = self._illum_asg_tangent.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        illum_asg_sharpness = self._illum_asg_sharpness.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        illum_asg_amplitude = self._illum_asg_amplitude.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        illum_asg_bias = self._illum_asg_bias.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        illum_asg_dist_weight = self._illum_asg_dist_weight.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filter_3D = self.filter_3D.detach().cpu().numpy()
        if self.use_residual:
            anchor_feat_residual = self._anchor_feat_residual.detach().cpu().numpy()
            scale_residual = self._scaling_residual.detach().cpu().numpy()
            offset_residual = self._offset_residual.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, base_log_reflectance, reflectance_offset_delta, enhancement_sg_axis, enhancement_sg_sharpness, enhancement_sg_amplitude, illum_asg_axis, illum_asg_tangent, illum_asg_sharpness, illum_asg_amplitude, illum_asg_bias, illum_asg_dist_weight, opacities, scale, rotation, filter_3D), axis=1)
        if self.use_residual:
            attributes = np.concatenate((anchor, normals, offset, anchor_feat, base_log_reflectance, reflectance_offset_delta, enhancement_sg_axis, enhancement_sg_sharpness, enhancement_sg_amplitude, illum_asg_axis, illum_asg_tangent, illum_asg_sharpness, illum_asg_amplitude, illum_asg_bias, illum_asg_dist_weight, opacities, scale, rotation, filter_3D, anchor_feat_residual, scale_residual, offset_residual), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis].astype(np.float32)
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        b0_names = [
            p.name for p in plydata.elements[0].properties
            if p.name.startswith("b0_") and not p.name.startswith("b0_detail_")
        ]
        b0_names = sorted(b0_names, key = lambda x: int(x.split('_')[-1]))
        if len(b0_names) > 0:
            base_log_reflectance = np.zeros((anchor.shape[0], len(b0_names)))
            for idx, attr_name in enumerate(b0_names):
                base_log_reflectance[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        else:
            base_log_reflectance = np.zeros((anchor.shape[0], 3), dtype=np.float32)
            self.illumination_mode = "legacy"
            self.legacy_compatibility_mode = True
        b0_detail_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("b0_detail_")]
        b0_detail_names = sorted(b0_detail_names, key=lambda x: int(x.split('_')[-1]))
        if len(b0_detail_names) > 0:
            reflectance_offset_delta = np.zeros((anchor.shape[0], len(b0_detail_names)), dtype=np.float32)
            for idx, attr_name in enumerate(b0_detail_names):
                reflectance_offset_delta[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            reflectance_offset_delta = reflectance_offset_delta.reshape((anchor.shape[0], 3, self.n_offsets)).transpose(0, 2, 1)
        else:
            reflectance_offset_delta = np.zeros((anchor.shape[0], self.n_offsets, 3), dtype=np.float32)

        enh_axis_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("enh_sg_axis_")]
        enh_axis_names = sorted(enh_axis_names, key=lambda x: int(x.split('_')[-1]))
        enh_sharpness_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("enh_sg_sharpness_")]
        enh_sharpness_names = sorted(enh_sharpness_names, key=lambda x: int(x.split('_')[-1]))
        enh_amplitude_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("enh_sg_amplitude_")]
        enh_amplitude_names = sorted(enh_amplitude_names, key=lambda x: int(x.split('_')[-1]))
        if len(enh_axis_names) > 0 and len(enh_sharpness_names) > 0 and len(enh_amplitude_names) > 0:
            enhancement_sg_axis = np.zeros((anchor.shape[0], len(enh_axis_names)), dtype=np.float32)
            enhancement_sg_sharpness = np.zeros((anchor.shape[0], len(enh_sharpness_names)), dtype=np.float32)
            enhancement_sg_amplitude = np.zeros((anchor.shape[0], len(enh_amplitude_names)), dtype=np.float32)
            for idx, attr_name in enumerate(enh_axis_names):
                enhancement_sg_axis[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            for idx, attr_name in enumerate(enh_sharpness_names):
                enhancement_sg_sharpness[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            for idx, attr_name in enumerate(enh_amplitude_names):
                enhancement_sg_amplitude[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            enhancement_sg_axis = enhancement_sg_axis.reshape((anchor.shape[0], 3, self.n_offsets)).transpose(0, 2, 1)
            enhancement_sg_sharpness = enhancement_sg_sharpness.reshape((anchor.shape[0], 1, self.n_offsets)).transpose(0, 2, 1)
            enhancement_sg_amplitude = enhancement_sg_amplitude.reshape((anchor.shape[0], 3, self.n_offsets)).transpose(0, 2, 1)
        else:
            axis, sharpness, amplitude = self._init_enhancement_sg_params(anchor.shape[0], device="cuda", dtype=torch.float)
            enhancement_sg_axis = axis.detach().cpu().numpy()
            enhancement_sg_sharpness = sharpness.detach().cpu().numpy()
            enhancement_sg_amplitude = amplitude.detach().cpu().numpy()

        def _load_flat_property(prefix):
            names = [p.name for p in plydata.elements[0].properties if p.name.startswith(prefix)]
            names = sorted(names, key=lambda x: int(x.split('_')[-1]))
            if len(names) == 0:
                return None
            values = np.zeros((anchor.shape[0], len(names)), dtype=np.float32)
            for idx, attr_name in enumerate(names):
                values[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            return values

        illum_asg_axis = _load_flat_property("illum_asg_axis_")
        illum_asg_tangent = _load_flat_property("illum_asg_tangent_")
        illum_asg_sharpness = _load_flat_property("illum_asg_sharpness_")
        illum_asg_amplitude = _load_flat_property("illum_asg_amplitude_")
        illum_asg_bias = _load_flat_property("illum_asg_bias_")
        illum_asg_dist_weight = _load_flat_property("illum_asg_dist_weight_")
        has_illum_asg = all(
            value is not None
            for value in (
                illum_asg_axis,
                illum_asg_tangent,
                illum_asg_sharpness,
                illum_asg_amplitude,
                illum_asg_bias,
                illum_asg_dist_weight,
            )
        )
        if has_illum_asg:
            illum_asg_axis = illum_asg_axis.reshape((anchor.shape[0], self.n_offsets, self.asg_lobes, 3))
            illum_asg_tangent = illum_asg_tangent.reshape((anchor.shape[0], self.n_offsets, self.asg_lobes, 3))
            illum_asg_sharpness = illum_asg_sharpness.reshape((anchor.shape[0], self.n_offsets, self.asg_lobes, 2))
            illum_asg_amplitude = illum_asg_amplitude.reshape((anchor.shape[0], self.n_offsets, self.asg_lobes, 1))
            illum_asg_bias = illum_asg_bias.reshape((anchor.shape[0], self.n_offsets, 1))
            illum_asg_dist_weight = illum_asg_dist_weight.reshape((anchor.shape[0], self.n_offsets, 1))
            self.asg_illumination_available = True
        else:
            params = self._init_illumination_asg_params(anchor.shape[0], device="cuda", dtype=torch.float)
            illum_asg_axis, illum_asg_tangent, illum_asg_sharpness, illum_asg_amplitude, illum_asg_bias, illum_asg_dist_weight = [
                value.detach().cpu().numpy() for value in params
            ]
            self.asg_illumination_available = False
            if self.illumination_mode == "asg":
                self.illumination_mode = "legacy"
                self.legacy_compatibility_mode = True

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        if self.use_residual:
            anchor_feat_residual_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("r_anchor_feat_residual")]
            anchor_feat_residual_names = sorted(anchor_feat_residual_names, key = lambda x: int(x.split('_')[-1]))
            anchor_feat_residuals = np.zeros((anchor.shape[0], len(anchor_feat_residual_names)))
            for idx, attr_name in enumerate(anchor_feat_residual_names):
                anchor_feat_residuals[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            
            offset_residual_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("r_offset_residual")]
            offset_residual_names = sorted(offset_residual_names, key = lambda x: int(x.split('_')[-1]))
            offset_residuals = np.zeros((anchor.shape[0], len(offset_residual_names)))
            for idx, attr_name in enumerate(offset_residual_names):
                offset_residuals[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            offset_residuals = offset_residuals.reshape((offset_residuals.shape[0], 3, -1))
            
            scaling_residual_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("r_scale_residual")]
            scaling_residual_names = sorted(scaling_residual_names, key = lambda x: int(x.split('_')[-1]))
            scaling_residuals = np.zeros((anchor.shape[0], len(scaling_residual_names)))
            for idx, attr_name in enumerate(scaling_residual_names):
                scaling_residuals[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

            

            

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._base_log_reflectance = nn.Parameter(torch.tensor(base_log_reflectance, dtype=torch.float, device="cuda").requires_grad_(True))
        self._reflectance_offset_delta = nn.Parameter(torch.tensor(reflectance_offset_delta, dtype=torch.float, device="cuda").requires_grad_(True))
        self._enhancement_sg_axis = nn.Parameter(torch.tensor(enhancement_sg_axis, dtype=torch.float, device="cuda").requires_grad_(True))
        self._enhancement_sg_sharpness = nn.Parameter(torch.tensor(enhancement_sg_sharpness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._enhancement_sg_amplitude = nn.Parameter(torch.tensor(enhancement_sg_amplitude, dtype=torch.float, device="cuda").requires_grad_(True))
        self._illum_asg_axis = nn.Parameter(torch.tensor(illum_asg_axis, dtype=torch.float, device="cuda").requires_grad_(True))
        self._illum_asg_tangent = nn.Parameter(torch.tensor(illum_asg_tangent, dtype=torch.float, device="cuda").requires_grad_(True))
        self._illum_asg_sharpness = nn.Parameter(torch.tensor(illum_asg_sharpness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._illum_asg_amplitude = nn.Parameter(torch.tensor(illum_asg_amplitude, dtype=torch.float, device="cuda").requires_grad_(True))
        self._illum_asg_bias = nn.Parameter(torch.tensor(illum_asg_bias, dtype=torch.float, device="cuda").requires_grad_(True))
        self._illum_asg_dist_weight = nn.Parameter(torch.tensor(illum_asg_dist_weight, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = nn.Parameter(torch.tensor(filter_3D, dtype=torch.float, device="cuda"))

        if self.use_residual:
            self._anchor_feat_residual = nn.Parameter(torch.tensor(anchor_feat_residuals, dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling_residual = nn.Parameter(torch.tensor(scaling_residuals, dtype=torch.float, device="cuda").requires_grad_(True))
            self._offset_residual = nn.Parameter(torch.tensor(offset_residuals, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'noise_net' in group['name'] or \
                'artifact_net' in group['name'] or \
                'residual_net' in group['name'] or \
                'embedding' in group['name'] or \
                'enhancement_context' in group['name'] or \
                'pose' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        if not hasattr(self, 'grad_variance'):
            self.grad_variance = torch.zeros_like(self.offset_gradient_accum)
            self.grad_mean = torch.zeros_like(self.offset_gradient_accum)

        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach() # [N * n_offsets, 1]
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets]) # [N, n_offsets]
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True) # [N, 1]
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1 # add for visiting anchor

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1) # [N * n_offsets]
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask # only update the visiable gaussians in the visiable anchors
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter 
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1 # add for visiting gaussians

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'noise_net' in group['name'] or \
                'artifact_net' in group['name'] or \
                'residual_net' in group['name'] or \
                'enhancement_context' in group['name'] or \
                'embedding' in group['name'] or \
                'pose' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                try:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                except:
                    print(group['name'])
                    import pdb; pdb.set_trace()

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._base_log_reflectance = optimizable_tensors["base_log_reflectance"]
        self._reflectance_offset_delta = optimizable_tensors["reflectance_offset_delta"]
        self._enhancement_sg_axis = optimizable_tensors["enhancement_sg_axis"]
        self._enhancement_sg_sharpness = optimizable_tensors["enhancement_sg_sharpness"]
        self._enhancement_sg_amplitude = optimizable_tensors["enhancement_sg_amplitude"]
        self._illum_asg_axis = optimizable_tensors["illum_asg_axis"]
        self._illum_asg_tangent = optimizable_tensors["illum_asg_tangent"]
        self._illum_asg_sharpness = optimizable_tensors["illum_asg_sharpness"]
        self._illum_asg_amplitude = optimizable_tensors["illum_asg_amplitude"]
        self._illum_asg_bias = optimizable_tensors["illum_asg_bias"]
        self._illum_asg_dist_weight = optimizable_tensors["illum_asg_dist_weight"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_residual:
            self._anchor_feat_residual = optimizable_tensors["anchor_feat_residual"]
            self._scaling_residual = optimizable_tensors["scaling_residual"]
            self._offset_residual = optimizable_tensors["offset_residual"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0: # if increased anchor number is zero, skip to next turn
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0) # add the increased anchor id

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1) # get all gaussians's location
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask] # get selected gaussians's location
            selected_grid_coords = torch.round(selected_xyz / cur_size).int() # get selected guassian's voxel location

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0) # get selected anchor voxel set, which will be used for the generation of new ancher's feature


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096 * 5
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
            
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates] # ues the big grad anchors to grow the new feature of the new anchors
                new_base_log_reflectance = torch.zeros((candidate_anchor.shape[0], 3), dtype=torch.float, device="cuda")
                if candidate_mask.any():
                    repeated_b0 = self._base_log_reflectance.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, 3])[candidate_mask]
                    new_base_log_reflectance = scatter_mean(
                        repeated_b0,
                        inverse_indices.unsqueeze(1).expand(-1, repeated_b0.size(1)),
                        dim=0,
                    )[remove_duplicates]
                new_reflectance_offset_delta = torch.zeros((candidate_anchor.shape[0], self.n_offsets, 3), dtype=torch.float, device="cuda")
                new_enhancement_sg_axis, new_enhancement_sg_sharpness, new_enhancement_sg_amplitude = self._init_enhancement_sg_params(
                    candidate_anchor.shape[0],
                    device="cuda",
                    dtype=torch.float,
                )
                new_illum_asg_axis, new_illum_asg_tangent, new_illum_asg_sharpness, new_illum_asg_amplitude, new_illum_asg_bias, new_illum_asg_dist_weight = self._init_illumination_asg_params(
                    candidate_anchor.shape[0],
                    device="cuda",
                    dtype=torch.float,
                )
                if candidate_mask.any():
                    def _inherit_offset_param(param):
                        flat_param = param.view(self.get_anchor.shape[0] * self.n_offsets, -1)[candidate_mask]
                        inherited = scatter_mean(
                            flat_param,
                            inverse_indices.unsqueeze(1).expand(-1, flat_param.size(1)),
                            dim=0,
                        )[remove_duplicates]
                        return inherited

                    new_illum_asg_axis = _inherit_offset_param(self._illum_asg_axis).view(candidate_anchor.shape[0], 1, self.asg_lobes, 3).repeat(1, self.n_offsets, 1, 1)
                    new_illum_asg_tangent = _inherit_offset_param(self._illum_asg_tangent).view(candidate_anchor.shape[0], 1, self.asg_lobes, 3).repeat(1, self.n_offsets, 1, 1)
                    new_illum_asg_sharpness = _inherit_offset_param(self._illum_asg_sharpness).view(candidate_anchor.shape[0], 1, self.asg_lobes, 2).repeat(1, self.n_offsets, 1, 1)
                    new_illum_asg_amplitude = _inherit_offset_param(self._illum_asg_amplitude).view(candidate_anchor.shape[0], 1, self.asg_lobes, 1).repeat(1, self.n_offsets, 1, 1)
                    new_illum_asg_bias = _inherit_offset_param(self._illum_asg_bias).view(candidate_anchor.shape[0], 1, 1).repeat(1, self.n_offsets, 1)
                    new_illum_asg_dist_weight = _inherit_offset_param(self._illum_asg_dist_weight).view(candidate_anchor.shape[0], 1, 1).repeat(1, self.n_offsets, 1)

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                if self.use_residual:
                    new_scaling_residual = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size
                    new_scaling_residual = torch.log(new_scaling_residual)
                    new_anchor_feat_residual = self._anchor_feat_residual.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                    new_anchor_feat_residual = scatter_max(new_anchor_feat_residual, inverse_indices.unsqueeze(1).expand(-1, new_anchor_feat_residual.size(1)), dim=0)[0][remove_duplicates] 

                    new_offset_residual = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets_residual,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "base_log_reflectance": new_base_log_reflectance,
                    "reflectance_offset_delta": new_reflectance_offset_delta,
                    "enhancement_sg_axis": new_enhancement_sg_axis,
                    "enhancement_sg_sharpness": new_enhancement_sg_sharpness,
                    "enhancement_sg_amplitude": new_enhancement_sg_amplitude,
                    "illum_asg_axis": new_illum_asg_axis,
                    "illum_asg_tangent": new_illum_asg_tangent,
                    "illum_asg_sharpness": new_illum_asg_sharpness,
                    "illum_asg_amplitude": new_illum_asg_amplitude,
                    "illum_asg_bias": new_illum_asg_bias,
                    "illum_asg_dist_weight": new_illum_asg_dist_weight,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                if self.use_residual:
                    d["scaling_residual"] = new_scaling_residual
                    d["offset_residual"] = new_offset_residual
                    d["anchor_feat_residual"] = new_anchor_feat_residual

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._base_log_reflectance = optimizable_tensors["base_log_reflectance"]
                self._reflectance_offset_delta = optimizable_tensors["reflectance_offset_delta"]
                self._enhancement_sg_axis = optimizable_tensors["enhancement_sg_axis"]
                self._enhancement_sg_sharpness = optimizable_tensors["enhancement_sg_sharpness"]
                self._enhancement_sg_amplitude = optimizable_tensors["enhancement_sg_amplitude"]
                self._illum_asg_axis = optimizable_tensors["illum_asg_axis"]
                self._illum_asg_tangent = optimizable_tensors["illum_asg_tangent"]
                self._illum_asg_sharpness = optimizable_tensors["illum_asg_sharpness"]
                self._illum_asg_amplitude = optimizable_tensors["illum_asg_amplitude"]
                self._illum_asg_bias = optimizable_tensors["illum_asg_bias"]
                self._illum_asg_dist_weight = optimizable_tensors["illum_asg_dist_weight"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                if self.use_residual:
                    self._scaling_residual = optimizable_tensors["scaling_residual"]
                    self._offset_residual = optimizable_tensors["offset_residual"]
                    self._anchor_feat_residual = optimizable_tensors["anchor_feat_residual"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005, mode="train", phi=0.5):
        # # adding anchors
        if mode =="warmup":
            old_anchor_num = self.anchor_demon.shape[0]
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold * phi).squeeze(dim=1) # choose the neural gaussians with high seen ratio as growing anchor candidates
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)  # choose the anchors with low opacity as prune candidates
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1] # choose the 
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 

        if mode == "warmup":
            unvisibility_mask = (self.anchor_demon == 0).squeeze(dim=1) 
            unvisibility_mask[old_anchor_num:] = False

            print("removed unvisibility anchor: ", sum(unvisibility_mask))
            prune_mask = torch.logical_or(prune_mask, unvisibility_mask)

            
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            # self.mlp_color.eval()
            # color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            # color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            # self.mlp_color.train()

            if self.illumination_mode == "sg" and self.use_sg_illumination:
                self.mlp_sg_illumination.eval()
                sg_illumination_mlp = torch.jit.trace(self.mlp_sg_illumination, (torch.rand(1, self.feat_dim//2+3+self.illumination_dist_dim).cuda()))
                sg_illumination_mlp.save(os.path.join(path, 'sg_illumination_mlp.pt'))
                self.mlp_sg_illumination.train()
            if self.illumination_mode == "legacy":
                self.mlp_reflectance.eval()
                reflectance_mlp = torch.jit.trace(self.mlp_reflectance, (torch.rand(1, self.feat_dim + self.reflectance_dist_dim).cuda()))
                reflectance_mlp.save(os.path.join(path, 'reflectance_mlp.pt'))
                self.mlp_reflectance.train()

                self.mlp_illumination.eval()
                illumination_mlp = torch.jit.trace(self.mlp_illumination, (torch.rand(1, self.feat_dim//2+3+self.illumination_dist_dim).cuda()))
                illumination_mlp.save(os.path.join(path, 'illumination_mlp.pt'))
                self.mlp_illumination.train()

            self.mlp_reflectance_decoder.eval()
            reflectance_decoder = torch.jit.trace(self.mlp_reflectance_decoder, (torch.rand(1, self.feat_dim + 3).cuda()))
            reflectance_decoder.save(os.path.join(path, 'reflectance_decoder.pt'))
            self.mlp_reflectance_decoder.train()

            torch.save({
                'enhancement_feat_weight': self._enhancement_feat_weight.detach(),
                'enhancement_illum_weight': self._enhancement_illum_weight.detach(),
                'enhancement_context_bias': self._enhancement_context_bias.detach(),
            }, os.path.join(path, 'enhancement_context.pth'))

            if self.use_residual:
                residual_input = torch.rand(1, self.feat_dim+3+self.residual_dist_dim + self.appearance_residual_dim).cuda()
                if self.use_dual_transient:
                    self.noise_net.eval()
                    noise_net = torch.jit.trace(self.noise_net, (residual_input,))
                    noise_net.save(os.path.join(path, 'noise_net.pt'))
                    self.noise_net.train()

                    self.artifact_net.eval()
                    artifact_net = torch.jit.trace(self.artifact_net, (residual_input,))
                    artifact_net.save(os.path.join(path, 'artifact_net.pt'))
                    self.artifact_net.train()
                else:
                    self.residual_net.eval()
                    residual_net = torch.jit.trace(self.residual_net, (residual_input,))
                    residual_net.save(os.path.join(path, 'residual_net.pt'))
                    self.residual_net.train()

                self.mlp_cov_residual.eval()
                cov_residual_mlp = torch.jit.trace(self.mlp_cov_residual, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
                cov_residual_mlp.save(os.path.join(path, 'cov_residual_mlp.pt'))
                self.mlp_cov_residual.train()

                self.mlp_opacity_residual.eval()
                opacity_residual_mlp = torch.jit.trace(self.mlp_opacity_residual, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
                opacity_residual_mlp.save(os.path.join(path, 'opacity_residual_mlp.pt'))
                self.mlp_opacity_residual.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_residual_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                checkpoint = {
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'reflectance_decoder': self.mlp_reflectance_decoder.state_dict(),
                    'enhancement_context': {
                        'feat_weight': self._enhancement_feat_weight.detach(),
                        'illum_weight': self._enhancement_illum_weight.detach(),
                        'bias': self._enhancement_context_bias.detach(),
                    },
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict(),
                    'illumination_mode': self.illumination_mode,
                    }
                if self.illumination_mode == "sg":
                    checkpoint['sg_illumination_mlp'] = self.mlp_sg_illumination.state_dict()
                if self.use_residual:
                    if self.use_dual_transient:
                        checkpoint['noise_net'] = self.noise_net.state_dict()
                        checkpoint['artifact_net'] = self.artifact_net.state_dict()
                    else:
                        checkpoint['residual_net'] = self.residual_net.state_dict()
                    checkpoint['cov_residual_mlp'] = self.mlp_cov_residual.state_dict()
                    checkpoint['opacity_residual_mlp'] = self.mlp_opacity_residual.state_dict()
                if self.illumination_mode == "legacy":
                    checkpoint['reflectance_mlp'] = self.mlp_reflectance.state_dict()
                    checkpoint['illumination_mlp'] = self.mlp_illumination.state_dict()
                torch.save(checkpoint, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_residual_dim > 0:
                checkpoint = {
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'reflectance_decoder': self.mlp_reflectance_decoder.state_dict(),
                    'enhancement_context': {
                        'feat_weight': self._enhancement_feat_weight.detach(),
                        'illum_weight': self._enhancement_illum_weight.detach(),
                        'bias': self._enhancement_context_bias.detach(),
                    },
                    'appearance': self.embedding_appearance.state_dict(),
                    'illumination_mode': self.illumination_mode,
                    }
                if self.illumination_mode == "sg":
                    checkpoint['sg_illumination_mlp'] = self.mlp_sg_illumination.state_dict()
                if self.use_residual:
                    if self.use_dual_transient:
                        checkpoint['noise_net'] = self.noise_net.state_dict()
                        checkpoint['artifact_net'] = self.artifact_net.state_dict()
                    else:
                        checkpoint['residual_net'] = self.residual_net.state_dict()
                    checkpoint['cov_residual_mlp'] = self.mlp_cov_residual.state_dict()
                    checkpoint['opacity_residual_mlp'] = self.mlp_opacity_residual.state_dict()
                if self.illumination_mode == "legacy":
                    checkpoint['reflectance_mlp'] = self.mlp_reflectance.state_dict()
                    checkpoint['illumination_mlp'] = self.mlp_illumination.state_dict()
                torch.save(checkpoint, os.path.join(path, 'checkpoints.pth'))
            else:
                checkpoint = {
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'reflectance_decoder': self.mlp_reflectance_decoder.state_dict(),
                    'enhancement_context': {
                        'feat_weight': self._enhancement_feat_weight.detach(),
                        'illum_weight': self._enhancement_illum_weight.detach(),
                        'bias': self._enhancement_context_bias.detach(),
                    },
                    'illumination_mode': self.illumination_mode,
                    }
                if self.illumination_mode == "sg":
                    checkpoint['sg_illumination_mlp'] = self.mlp_sg_illumination.state_dict()
                if self.use_residual:
                    if self.use_dual_transient:
                        checkpoint['noise_net'] = self.noise_net.state_dict()
                        checkpoint['artifact_net'] = self.artifact_net.state_dict()
                    else:
                        checkpoint['residual_net'] = self.residual_net.state_dict()
                    checkpoint['cov_residual_mlp'] = self.mlp_cov_residual.state_dict()
                    checkpoint['opacity_residual_mlp'] = self.mlp_opacity_residual.state_dict()
                if self.illumination_mode == "legacy":
                    checkpoint['reflectance_mlp'] = self.mlp_reflectance.state_dict()
                    checkpoint['illumination_mlp'] = self.mlp_illumination.state_dict()
                torch.save(checkpoint, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            sg_path = os.path.join(path, 'sg_illumination_mlp.pt')
            legacy_reflectance_path = os.path.join(path, 'reflectance_mlp.pt')
            legacy_illumination_path = os.path.join(path, 'illumination_mlp.pt')
            reflectance_decoder_path = os.path.join(path, 'reflectance_decoder.pt')
            enhancement_context_path = os.path.join(path, 'enhancement_context.pth')
            if self.illumination_mode == "asg" and self.asg_illumination_available:
                self.legacy_compatibility_mode = False
            elif self.use_sg_illumination and os.path.exists(sg_path):
                self.mlp_sg_illumination = torch.jit.load(sg_path).cuda()
                self.sg_illumination_available = True
                self.illumination_mode = "sg"
                self.legacy_compatibility_mode = False
            else:
                self.sg_illumination_available = False
                self.illumination_mode = "legacy"
                self.legacy_compatibility_mode = True
                print("SG illumination checkpoint not found; entering legacy compatibility mode.")
            if self.legacy_compatibility_mode and os.path.exists(legacy_reflectance_path):
                self.mlp_reflectance = torch.jit.load(legacy_reflectance_path).cuda()
            if os.path.exists(reflectance_decoder_path):
                self.mlp_reflectance_decoder = torch.jit.load(reflectance_decoder_path).cuda()
            if self.legacy_compatibility_mode and os.path.exists(legacy_illumination_path):
                self.mlp_illumination = torch.jit.load(legacy_illumination_path).cuda()
            if os.path.exists(enhancement_context_path):
                enhancement_context = torch.load(enhancement_context_path)
                self._enhancement_feat_weight = nn.Parameter(enhancement_context['enhancement_feat_weight'].cuda().requires_grad_(True))
                self._enhancement_illum_weight = nn.Parameter(enhancement_context['enhancement_illum_weight'].cuda().requires_grad_(True))
                self._enhancement_context_bias = nn.Parameter(enhancement_context['enhancement_context_bias'].cuda().requires_grad_(True))
            else:
                self._ensure_enhancement_sg_params()
            if self.use_residual:
                noise_path = os.path.join(path, 'noise_net.pt')
                artifact_path = os.path.join(path, 'artifact_net.pt')
                legacy_residual_path = os.path.join(path, 'residual_net.pt')
                if self.use_dual_transient and os.path.exists(noise_path):
                    self.noise_net = torch.jit.load(noise_path).cuda()
                else:
                    self._reset_noise_net()
                if self.use_dual_transient and os.path.exists(artifact_path):
                    self.artifact_net = torch.jit.load(artifact_path).cuda()
                    self.residual_net = self.artifact_net
                elif self.use_dual_transient and os.path.exists(legacy_residual_path):
                    self.artifact_net = torch.jit.load(legacy_residual_path).cuda()
                    self.residual_net = self.artifact_net
                elif not self.use_dual_transient and os.path.exists(legacy_residual_path):
                    self.residual_net = torch.jit.load(legacy_residual_path).cuda()
                elif not self.use_dual_transient and os.path.exists(artifact_path):
                    self.residual_net = torch.jit.load(artifact_path).cuda()
                self.mlp_cov_residual = torch.jit.load(os.path.join(path, 'cov_residual_mlp.pt')).cuda()
                self.mlp_opacity_residual = torch.jit.load(os.path.join(path, 'opacity_residual_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_residual_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            checkpoint_mode = checkpoint.get('illumination_mode', self.illumination_mode)
            if checkpoint_mode == "asg" and self.asg_illumination_available:
                self.illumination_mode = "asg"
                self.legacy_compatibility_mode = False
            elif self.use_sg_illumination and 'sg_illumination_mlp' in checkpoint:
                self.mlp_sg_illumination.load_state_dict(checkpoint['sg_illumination_mlp'])
                self.sg_illumination_available = True
                self.illumination_mode = checkpoint_mode
                self.legacy_compatibility_mode = self.illumination_mode == "legacy"
            else:
                self.sg_illumination_available = False
                self.illumination_mode = "legacy"
                self.legacy_compatibility_mode = True
                print("SG illumination checkpoint not found; entering legacy compatibility mode.")
            if self.legacy_compatibility_mode and 'reflectance_mlp' in checkpoint:
                self.mlp_reflectance.load_state_dict(checkpoint['reflectance_mlp'])
            if 'reflectance_decoder' in checkpoint:
                self.mlp_reflectance_decoder.load_state_dict(checkpoint['reflectance_decoder'])
            if self.legacy_compatibility_mode and 'illumination_mlp' in checkpoint:
                self.mlp_illumination.load_state_dict(checkpoint['illumination_mlp'])
            if 'enhancement_context' in checkpoint:
                enhancement_context = checkpoint['enhancement_context']
                self._enhancement_feat_weight = nn.Parameter(enhancement_context['feat_weight'].cuda().requires_grad_(True))
                self._enhancement_illum_weight = nn.Parameter(enhancement_context['illum_weight'].cuda().requires_grad_(True))
                self._enhancement_context_bias = nn.Parameter(enhancement_context['bias'].cuda().requires_grad_(True))
            else:
                self._ensure_enhancement_sg_params()
            if self.use_residual:
                if self.use_dual_transient and 'noise_net' in checkpoint:
                    self.noise_net.load_state_dict(checkpoint['noise_net'])
                else:
                    self._reset_noise_net()
                if self.use_dual_transient and 'artifact_net' in checkpoint:
                    self.artifact_net.load_state_dict(checkpoint['artifact_net'])
                    self.residual_net = self.artifact_net
                elif self.use_dual_transient and 'residual_net' in checkpoint:
                    self.artifact_net.load_state_dict(checkpoint['residual_net'])
                    self.residual_net = self.artifact_net
                elif not self.use_dual_transient and 'residual_net' in checkpoint:
                    self.residual_net.load_state_dict(checkpoint['residual_net'])
                elif not self.use_dual_transient and 'artifact_net' in checkpoint:
                    self.residual_net.load_state_dict(checkpoint['artifact_net'])
                self.mlp_cov_residual.load_state_dict(checkpoint['cov_residual_mlp'])
                self.mlp_opacity_residual.load_state_dict(checkpoint['opacity_residual_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_residual_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
        
