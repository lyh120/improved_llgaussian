"""CPU fixtures executing real model methods without importing CUDA-only modules."""

import ast
import json
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.decomposition_config import configure_reflectance
from utils.reflectance_utils import estimate_initial_b0, reflectance_config


def load_model_class():
    path = Path(__file__).resolve().parents[1] / "scene" / "gaussian_model.py"
    node = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
    namespace = dict(torch=torch, nn=nn, F=F, np=np, os=os, json=json, math=math,
                     BasicPointCloud=object, configure_reflectance=configure_reflectance,
                     estimate_initial_b0=estimate_initial_b0, reflectance_config=reflectance_config,
                     mkdir_p=lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["GaussianModel"]


def make_cpu_model(mode="explicit", distance=False):
    torch.manual_seed(7)
    model_class = load_model_class()
    model = model_class.__new__(model_class)
    configure_reflectance(model)
    model.feat_dim, model.n_offsets, model.asg_lobes = 8, 4, 1
    model.reflectance_mode = mode
    model.illumination_mode = "asg"
    model.add_reflectance_dist = distance
    model.use_residual = False
    model._anchor = nn.Parameter(torch.tensor([[-0.25, -0.2, 1.2], [0.3, 0.25, 1.6]]))
    model._offset = nn.Parameter(torch.randn(2, 4, 3) * 0.12)
    model._anchor_feat = nn.Parameter(torch.randn(2, 8))
    model._base_log_reflectance = nn.Parameter(-1.0 + torch.randn(2, 3) * 0.05)
    model._reflectance_offset_delta = nn.Parameter(torch.randn(2, 4, 3) * 0.03)
    model.mlp_reflectance_decoder = nn.Sequential(nn.Linear(11, 8), nn.Tanh(), nn.Linear(8, 3))
    model.mlp_reflectance = nn.Sequential(nn.Linear(8 + int(distance), 12), nn.Sigmoid())
    model.mlp_opacity = nn.Linear(8, 4)
    model.mlp_cov = nn.Linear(8, 7)
    model.mlp_illumination = nn.Linear(8, 4)
    model.mlp_sg_illumination = nn.Linear(8, 20)
    model._scaling = nn.Parameter(torch.full((2, 6), -2.0))
    model._rotation = nn.Parameter(torch.zeros(2, 4))
    model._opacity = nn.Parameter(torch.zeros(2, 1))
    model.P = torch.zeros(1, 7, requires_grad=True)
    model.filter_3D = torch.ones(2, 1) * 0.01
    model.spatial_lr_scale = 1.0
    model.max_radii2D = torch.zeros(2)
    model._last_reflectance_decoder_mean = torch.tensor(-1.0)
    model._enhancement_sg_axis = nn.Parameter(torch.ones(2, 4, 3))
    model._enhancement_sg_sharpness = nn.Parameter(torch.ones(2, 4, 1))
    model._enhancement_sg_amplitude = nn.Parameter(torch.ones(2, 4, 3))
    model._enhancement_feat_weight = nn.Parameter(torch.zeros(8, 3))
    model._enhancement_illum_weight = nn.Parameter(torch.zeros(4, 3))
    model._enhancement_context_bias = nn.Parameter(torch.zeros(1, 4, 3))
    model._illum_asg_axis = nn.Parameter(torch.ones(2, 4, 1, 3))
    model._illum_asg_tangent = nn.Parameter(torch.ones(2, 4, 1, 3))
    model._illum_asg_sharpness = nn.Parameter(torch.ones(2, 4, 1, 2))
    model._illum_asg_amplitude = nn.Parameter(torch.ones(2, 4, 1, 1))
    model._illum_asg_bias = nn.Parameter(torch.ones(2, 4, 1))
    model._illum_asg_dist_weight = nn.Parameter(torch.ones(2, 4, 1))

    def cpu_training_setup(self, training_args):
        self.offset_denom = torch.zeros(8, 1)
        self.anchor_birth_iteration = torch.zeros(2, dtype=torch.long)
        parameters = [value for value in vars(self).values() if isinstance(value, nn.Parameter)]
        parameters += [param for value in vars(self).values() if isinstance(value, nn.Module)
                       for param in value.parameters()]
        parameters.append(self.P)
        self.optimizer = torch.optim.Adam(parameters, lr=0.001)

    # Optimizer allocation is the sole CPU replacement; restore/capture/decoder are real methods.
    model.training_setup = MethodType(cpu_training_setup, model)
    model.training_setup(SimpleNamespace())
    return model


class SoftRasterizer:
    """Differentiable test double, so the real wrappers can be checked on CPU."""

    def __call__(self, *, means3D, means2D, colors_precomp, opacities, scales,
                 rotations, shs=None, cov3D_precomp=None):
        grid = torch.linspace(-1, 1, 24, device=means3D.device, dtype=means3D.dtype)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        centers = means3D + means2D
        spread = scales.mean(dim=1).clamp_min(0.1)
        distance = ((xx[None] - centers[:, 0, None, None]).square()
                    + (yy[None] - centers[:, 1, None, None]).square())
        weights = torch.exp(-distance / spread[:, None, None])
        weights = weights * opacities[:, 0, None, None] / (1 + centers[:, 2, None, None].square())
        weights = weights * (1 + 0.01 * rotations.sum(dim=1)[:, None, None])
        image = torch.einsum("nhw,nc->chw", weights, colors_precomp)
        return image, torch.ones(len(means3D), device=means3D.device), image[:1]
