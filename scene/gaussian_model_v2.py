"""Scaffold-GS geometry with a strictly explicit v2 appearance model."""

from __future__ import annotations

import os
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn
from tqdm import tqdm

try:
    from simple_knn._C import distCUDA2 as _cuda_nearest_distance
except ImportError:
    _cuda_nearest_distance = None

try:
    from torch_scatter import scatter_max as _torch_scatter_max
except ImportError:
    _torch_scatter_max = None

from scene.explicit_appearance import ExplicitAppearance
from utils.general_utils import (
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud, get_uniform_points_on_sphere_fibonacci
from utils.model_format import (
    EXPLICIT_APPEARANCE_FORMAT_VERSION,
    MODEL_FORMAT_VERSION,
    NUMERICAL_EPS,
)
from utils.system_utils import mkdir_p


# These constants preserve the Scaffold geometry baseline. They are named so
# that they cannot be mistaken for appearance or loss weights.  In particular,
# the 0.15 value is the Scaffold screen-border expansion used only while
# estimating the 3D filter; it is not an R/L gain or a loss coefficient.
SCAFFOLD_VISIBILITY_MARGIN = 0.15
# Scaffold's projected low-pass footprint, expressed as sqrt(0.2).
SCAFFOLD_FILTER_FOOTPRINT = 0.2 ** 0.5
SCAFFOLD_MIN_PRUNE_KEEP_PROBABILITY = 0.5
SCAFFOLD_DENSIFICATION_OBSERVATION_FRACTION = 0.5
SCAFFOLD_FILTER_MIN_CAMERA_DEPTH = 0.2
SCAFFOLD_FILTER_UNSEEN_DISTANCE = 100_000.0
SCAFFOLD_INITIAL_OPACITY = 0.1
SCAFFOLD_SKY_RADIUS_QUANTILE = 0.97
SCAFFOLD_SKY_RADIUS_MULTIPLIER = 10.0
SCAFFOLD_SKY_DISTANCE_RATIO = 0.5
SCAFFOLD_ADAM_EPS = 1e-15


def _nearest_distance(points: torch.Tensor) -> torch.Tensor:
    if _cuda_nearest_distance is not None and points.is_cuda:
        return _cuda_nearest_distance(points)
    if points.shape[0] < 2:
        return torch.ones((points.shape[0],), device=points.device, dtype=points.dtype)
    distances = torch.cdist(points, points).square()
    distances.fill_diagonal_(torch.inf)
    return distances.amin(dim=1)


def scatter_max(source: torch.Tensor, index: torch.Tensor, dim: int = 0):
    if _torch_scatter_max is not None:
        return _torch_scatter_max(source, index, dim=dim)
    if dim != 0:
        raise NotImplementedError("CPU test fallback only supports scatter dim=0")
    group_count = int(index.max().item()) + 1
    output_shape = list(source.shape)
    output_shape[0] = group_count
    output = torch.full(
        output_shape,
        -torch.inf,
        dtype=source.dtype,
        device=source.device,
    )
    output.scatter_reduce_(0, index, source, reduce="amax", include_self=True)
    return output, None


@torch.no_grad()
def get_sky_points(num_points, points3d, cameras):
    from utils.camera_utils import camera_project

    points = get_uniform_points_on_sphere_fibonacci(num_points, xnp=torch)
    points = points.to(points3d.device)
    mean = points3d.mean(0, keepdim=True)
    sky_distance = torch.quantile(
        torch.linalg.norm(points3d - mean, 2, -1),
        SCAFFOLD_SKY_RADIUS_QUANTILE,
    ) * SCAFFOLD_SKY_RADIUS_MULTIPLIER
    points = points * sky_distance + mean
    generated = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    for camera in tqdm(cameras, desc="Generating skybox"):
        uv = camera_project(camera, points[~generated])
        mask = ~torch.isnan(uv).any(-1)
        mask &= uv[..., -1] < 2 / 3 * camera.image_height
        generated[~generated] |= mask
    return points[generated], sky_distance * SCAFFOLD_SKY_DISTANCE_RATIO


class GaussianModel:
    """Scaffold anchor geometry plus :class:`ExplicitAppearance`."""

    def __init__(
        self,
        feat_dim: int = 32,
        n_offsets: int = 5,
        voxel_size: float = 0.01,
        update_depth: int = 3,
        update_init_factor: int = 100,
        update_hierachy_factor: int = 4,
        use_feat_bank: bool = False,
        add_opacity_dist: bool = False,
        add_cov_dist: bool = False,
        use_3D_filter: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.use_3D_filter = use_3D_filter

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.filter_3D = torch.empty(0)
        self.appearance = ExplicitAppearance(n_offsets)

        self.opacity_accum = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.anchor_visible_count = torch.empty(0)
        self.anchor_birth_iteration = torch.empty(0, dtype=torch.long)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self._setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(4, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1),
            ).to(self.device)

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, self.n_offsets),
            nn.Tanh(),
        ).to(self.device)

        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.n_offsets),
        ).to(self.device)

    def _setup_functions(self) -> None:
        def covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            matrix = build_scaling_rotation(scaling_modifier * scaling, rotation)
            return strip_symmetric(matrix @ matrix.transpose(1, 2))

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def eval(self) -> None:
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self) -> None:
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.appearance.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self) -> dict:
        if self.optimizer is None:
            raise RuntimeError("training_setup must run before capture")
        scaffold = {
            "anchor": self._anchor.detach(),
            "offset": self._offset.detach(),
            "anchor_feat": self._anchor_feat.detach(),
            "scaling": self._scaling.detach(),
            "rotation": self._rotation.detach(),
            "opacity": self._opacity.detach(),
            "filter_3D": self.filter_3D.detach(),
            "max_radii2D": self.max_radii2D.detach(),
            "opacity_accum": self.opacity_accum.detach(),
            "offset_gradient_accum": self.offset_gradient_accum.detach(),
            "offset_denom": self.offset_denom.detach(),
            "anchor_demon": self.anchor_demon.detach(),
            "anchor_visible_count": self.anchor_visible_count.detach(),
            "anchor_birth_iteration": self.anchor_birth_iteration.detach(),
            "spatial_lr_scale": self.spatial_lr_scale,
        }
        if hasattr(self, "P"):
            scaffold["poses"] = self.P.detach()
        core_mlps = {
            "opacity": self.mlp_opacity.state_dict(),
            "covariance": self.mlp_cov.state_dict(),
        }
        if self.use_feat_bank:
            core_mlps["feature_bank"] = self.mlp_feature_bank.state_dict()
        return {
            "model_format_version": MODEL_FORMAT_VERSION,
            "scaffold": scaffold,
            "explicit_appearance": self.appearance.state_dict_v2(),
            "core_mlps": core_mlps,
            "optimizer": self.optimizer.state_dict(),
        }

    def restore(self, state: dict, training_args) -> None:
        self._require_v2(state, "checkpoint")
        expected = {"model_format_version", "scaffold", "explicit_appearance", "core_mlps", "optimizer"}
        if set(state) != expected:
            raise ValueError(
                f"Invalid v2 checkpoint fields; missing={sorted(expected - set(state))}, "
                f"extra={sorted(set(state) - expected)}"
            )
        scaffold = state["scaffold"]
        required_scaffold = {
            "anchor",
            "offset",
            "anchor_feat",
            "scaling",
            "rotation",
            "opacity",
            "filter_3D",
            "max_radii2D",
            "opacity_accum",
            "offset_gradient_accum",
            "offset_denom",
            "anchor_demon",
            "anchor_visible_count",
            "anchor_birth_iteration",
            "spatial_lr_scale",
            "poses",
        }
        if set(scaffold) != required_scaffold:
            raise ValueError(
                "Invalid v2 Scaffold fields; "
                f"missing={sorted(required_scaffold - set(scaffold))}, "
                f"extra={sorted(set(scaffold) - required_scaffold)}"
            )
        expected_core = {"opacity", "covariance"}
        if self.use_feat_bank:
            expected_core.add("feature_bank")
        if set(state["core_mlps"]) != expected_core:
            raise ValueError(
                "Invalid v2 core MLP fields; "
                f"missing={sorted(expected_core - set(state['core_mlps']))}, "
                f"extra={sorted(set(state['core_mlps']) - expected_core)}"
            )
        for key in ("anchor", "offset", "anchor_feat", "scaling", "rotation", "opacity"):
            setattr(self, f"_{key}", nn.Parameter(scaffold[key].requires_grad_(True)))
        self.filter_3D = scaffold["filter_3D"]
        self.max_radii2D = scaffold["max_radii2D"]
        self.opacity_accum = scaffold["opacity_accum"]
        self.offset_gradient_accum = scaffold["offset_gradient_accum"]
        self.offset_denom = scaffold["offset_denom"]
        self.anchor_demon = scaffold["anchor_demon"]
        self.anchor_visible_count = scaffold["anchor_visible_count"]
        self.anchor_birth_iteration = scaffold["anchor_birth_iteration"]
        self.spatial_lr_scale = scaffold["spatial_lr_scale"]
        self.P = scaffold["poses"].requires_grad_(True)
        self.appearance.load_state_dict_v2(state["explicit_appearance"])
        self._validate_anchor_alignment(include_statistics=True)
        self.mlp_opacity.load_state_dict(state["core_mlps"]["opacity"])
        self.mlp_cov.load_state_dict(state["core_mlps"]["covariance"])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(state["core_mlps"]["feature_bank"])
        self.training_setup(training_args, reset_statistics=False)
        self.optimizer.load_state_dict(state["optimizer"])

    @staticmethod
    def _require_v2(state: dict, source: str) -> None:
        if not isinstance(state, dict) or state.get("model_format_version") != MODEL_FORMAT_VERSION:
            raise RuntimeError(
                f"{source} is not MODEL_FORMAT_VERSION={MODEL_FORMAT_VERSION}; "
                "legacy checkpoints and PLY files are intentionally unsupported"
            )

    def _validate_anchor_alignment(self, include_statistics: bool) -> None:
        count = self._anchor.shape[0]
        expected_shapes = {
            "anchor": (count, 3),
            "offset": (count, self.n_offsets, 3),
            "anchor_feat": (count, self.feat_dim),
            "scaling": (count, 6),
            "rotation": (count, 4),
            "opacity": (count, 1),
            "filter_3D": (count, 1),
        }
        geometry = {
            "offset": self._offset,
            "anchor_feat": self._anchor_feat,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "filter_3D": self.filter_3D,
        }
        if include_statistics:
            geometry.update(
                {
                    "max_radii2D": self.max_radii2D,
                    "opacity_accum": self.opacity_accum,
                    "anchor_demon": self.anchor_demon,
                    "anchor_visible_count": self.anchor_visible_count,
                    "anchor_birth_iteration": self.anchor_birth_iteration,
                }
            )
            statistic_shapes = {
                "max_radii2D": (count,),
                "opacity_accum": (count, 1),
                "anchor_demon": (count, 1),
                "anchor_visible_count": (count, 1),
                "anchor_birth_iteration": (count,),
                "offset_gradient_accum": (count * self.n_offsets, 1),
                "offset_denom": (count * self.n_offsets, 1),
            }
            for name, expected_shape in statistic_shapes.items():
                if tuple(getattr(self, name).shape) != expected_shape:
                    raise ValueError(
                        f"Invalid v2 Scaffold layout for {name}: "
                        f"{tuple(getattr(self, name).shape)} != {expected_shape}"
                    )
            expected_offsets = count * self.n_offsets
            for name in ("offset_gradient_accum", "offset_denom"):
                if getattr(self, name).shape[0] != expected_offsets:
                    raise ValueError(
                        f"Invalid v2 Scaffold shape for {name}: "
                        f"{getattr(self, name).shape[0]} != {expected_offsets}"
                    )
        for name, tensor in geometry.items():
            if tensor.shape[0] != count:
                raise ValueError(
                    f"Invalid v2 Scaffold shape for {name}: {tensor.shape[0]} != {count}"
                )
        layout_tensors = {
            "anchor": self._anchor,
            "offset": self._offset,
            "anchor_feat": self._anchor_feat,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "filter_3D": self.filter_3D,
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(layout_tensors[name].shape) != expected_shape:
                raise ValueError(
                    f"Invalid v2 Scaffold layout for {name}: "
                    f"{tuple(layout_tensors[name].shape)} != {expected_shape}"
                )
        if self.appearance.anchor_count != count:
            raise ValueError(
                "Explicit appearance and Scaffold anchor counts do not match: "
                f"{self.appearance.anchor_count} != {count}"
            )
        if hasattr(self, "P") and (self.P.ndim != 2 or self.P.shape[1] != 7):
            raise ValueError(f"Invalid v2 pose layout: {tuple(self.P.shape)}; expected [N, 7]")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    def get_scaling_with_3D_filter(self, scales, visible_mask):
        return torch.sqrt(torch.square(scales) + torch.square(self.filter_3D[visible_mask]))

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    def get_opacity_with_3D_filter(self, opacity, visible_mask):
        scales = self.get_scaling[visible_mask].repeat(self.n_offsets, 1)
        scales_square = torch.square(scales)
        det_before = scales_square.prod(dim=1).clamp_min(NUMERICAL_EPS)
        filtered = scales_square + torch.square(self.filter_3D[visible_mask].repeat(self.n_offsets, 1))
        det_after = filtered.prod(dim=1).clamp_min(NUMERICAL_EPS)
        return opacity * torch.sqrt(det_before / det_after)[..., None]

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def evaluate_appearance(self, view_dirs, visible_mask=None):
        return self.appearance.evaluate(view_dirs, visible_mask)

    def parameter_count(self) -> int:
        """Return the unique trainable tensor count without requiring an optimizer."""
        tensors = [
            self._anchor,
            self._offset,
            self._anchor_feat,
            self._scaling,
            self._rotation,
            self._opacity,
            *self.appearance.parameters(),
            *self.mlp_opacity.parameters(),
            *self.mlp_cov.parameters(),
        ]
        if self.use_feat_bank:
            tensors.extend(self.mlp_feature_bank.parameters())
        if hasattr(self, "P"):
            tensors.append(self.P)
        return sum(tensor.numel() for tensor in tensors)

    def init_RT_seq(self, camera_lists) -> None:
        from utils.pose_utils import get_tensor_from_camera

        poses = [
            get_tensor_from_camera(camera.world_view_transform.transpose(0, 1))
            for camera in camera_lists[1.0]
        ]
        self.P = torch.stack(poses).to(self.device).requires_grad_(True)

    def get_RT(self, index):
        return self.P[index]

    @torch.no_grad()
    def compute_3D_filter(self, cameras) -> None:
        """Cache the fixed Scaffold footprint without retaining an autograd graph."""
        device = self._anchor.device
        distance = torch.full(
            (self._anchor.shape[0],),
            SCAFFOLD_FILTER_UNSEEN_DISTANCE,
            device=device,
        )
        valid_points = torch.zeros((self._anchor.shape[0],), dtype=torch.bool, device=device)
        focal_length = 0.0
        margin = SCAFFOLD_VISIBILITY_MARGIN
        for camera in cameras:
            rotation = torch.tensor(camera.R, device=device, dtype=torch.float32)
            translation = torch.tensor(camera.T, device=device, dtype=torch.float32)
            xyz_camera = self._anchor @ rotation + translation[None]
            valid_depth = xyz_camera[:, 2] > SCAFFOLD_FILTER_MIN_CAMERA_DEPTH
            z = xyz_camera[:, 2].clamp_min(NUMERICAL_EPS)
            x = xyz_camera[:, 0] / z * camera.focal_x + camera.image_width / 2.0
            y = xyz_camera[:, 1] / z * camera.focal_y + camera.image_height / 2.0
            in_screen = (
                (x >= -margin * camera.image_width)
                & (x <= (1.0 + margin) * camera.image_width)
                & (y >= -margin * camera.image_height)
                & (y <= (1.0 + margin) * camera.image_height)
            )
            valid = valid_depth & in_screen
            distance[valid] = torch.minimum(distance[valid], z[valid])
            valid_points |= valid
            focal_length = max(focal_length, float(camera.focal_x))
        if not valid_points.any():
            raise RuntimeError("No anchors are visible while computing the Scaffold 3D filter")
        distance[~valid_points] = distance[valid_points].max()
        self.filter_3D = (distance / focal_length * SCAFFOLD_FILTER_FOOTPRINT)[..., None]

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        spatial_lr_scale: float,
        num_sky_gaussians=0,
        cameras=None,
        prune_ratio: float = 1.0,
        beta=1,
    ) -> None:
        self.spatial_lr_scale = spatial_lr_scale
        cameras = list(cameras or [])
        points = pcd.points
        if self.voxel_size <= 0:
            initial = torch.tensor(points, dtype=torch.float32, device=self.device)
            distances = _nearest_distance(initial).float()
            self.voxel_size = torch.median(distances).item()

        fused = torch.tensor(np.asarray(points), dtype=torch.float32, device=self.device)
        original_count = fused.shape[0]
        tau = torch.tensor(1.0, device=fused.device)
        while fused.shape[0] > original_count * prune_ratio and prune_ratio < 1:
            distances = _nearest_distance(fused).float().clamp_min(NUMERICAL_EPS)
            tau *= torch.exp(torch.tensor(float(beta) * fused.shape[0] / original_count, device=fused.device))
            probabilities = (distances / (self.voxel_size * tau)).clamp(
                SCAFFOLD_MIN_PRUNE_KEEP_PROBABILITY,
                1.0,
            )
            fused = fused[torch.rand_like(distances) < probabilities]

        opacities = inverse_sigmoid(
            torch.full(
                (fused.shape[0], 1),
                SCAFFOLD_INITIAL_OPACITY,
                device=fused.device,
            )
        )
        if num_sky_gaussians and cameras:
            skybox, self._sky_distance = get_sky_points(num_sky_gaussians, fused, cameras)
            fused = torch.cat((fused, skybox), dim=0)
            opacities = torch.cat(
                (opacities, inverse_sigmoid(torch.ones((skybox.shape[0], 1), device=fused.device))),
                dim=0,
            )

        offsets = torch.zeros((fused.shape[0], self.n_offsets, 3), device=fused.device)
        features = torch.zeros((fused.shape[0], self.feat_dim), device=fused.device)
        distances = _nearest_distance(fused).float().clamp_min(NUMERICAL_EPS)
        scales = torch.log(torch.sqrt(distances))[..., None].repeat(1, 6)
        rotations = torch.zeros((fused.shape[0], 4), device=fused.device)
        rotations[:, 0] = 1.0

        self._anchor = nn.Parameter(fused.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(features.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.filter_3D = torch.zeros((fused.shape[0], 1), device=fused.device)
        self.appearance.initialize(self._anchor, self._offset, self.get_scaling, cameras)
        self.max_radii2D = torch.zeros((fused.shape[0],), device=fused.device)

    def training_setup(self, training_args, reset_statistics: bool = True) -> None:
        device = self.get_anchor.device
        if reset_statistics:
            self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device=device)
            self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device=device)
            self.offset_denom = torch.zeros_like(self.offset_gradient_accum)
            self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device=device)
            self.anchor_visible_count = torch.zeros_like(self.anchor_demon)
            self.anchor_birth_iteration = torch.zeros((self.get_anchor.shape[0],), dtype=torch.long, device=device)

        groups = [
            {"params": [self._anchor], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {"params": [self._offset], "lr": training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {"params": [self._anchor_feat], "lr": training_args.feature_lr, "name": "anchor_feat"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": self.mlp_opacity.parameters(), "lr": training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {"params": self.mlp_cov.parameters(), "lr": training_args.mlp_cov_lr_init, "name": "mlp_cov"},
        ]
        groups.extend(self.appearance.optimizer_groups(training_args.explicit_appearance_lr_init))
        if self.use_feat_bank:
            groups.append({
                "params": self.mlp_feature_bank.parameters(),
                "lr": training_args.mlp_featurebank_lr_init,
                "name": "mlp_featurebank",
            })
        if hasattr(self, "P"):
            groups.append({"params": [self.P], "lr": training_args.pose_lr_init, "name": "pose"})
        self._fixed_geometry_lrs = {
            "anchor_feat": training_args.feature_lr,
            "opacity": training_args.opacity_lr,
            "scaling": training_args.scaling_lr,
            "rotation": training_args.rotation_lr,
        }
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=SCAFFOLD_ADAM_EPS)

        self.anchor_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.offset_scheduler_args = get_expon_lr_func(
            lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
            lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.offset_lr_delay_mult,
            max_steps=training_args.offset_lr_max_steps,
        )
        self.mlp_opacity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_opacity_lr_init,
            lr_final=training_args.mlp_opacity_lr_final,
            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
            max_steps=training_args.mlp_opacity_lr_max_steps,
        )
        self.mlp_cov_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_cov_lr_init,
            lr_final=training_args.mlp_cov_lr_final,
            lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
            max_steps=training_args.mlp_cov_lr_max_steps,
        )
        self.appearance_scheduler_args = get_expon_lr_func(
            lr_init=training_args.explicit_appearance_lr_init,
            lr_final=training_args.explicit_appearance_lr_final,
            max_steps=training_args.iterations,
        )
        self.pose_scheduler_args = get_expon_lr_func(
            lr_init=training_args.pose_lr_init,
            lr_final=training_args.pose_lr_final,
            lr_delay_mult=training_args.pose_lr_delay_mult,
            max_steps=training_args.pose_lr_max_steps,
        )
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(
                lr_init=training_args.mlp_featurebank_lr_init,
                lr_final=training_args.mlp_featurebank_lr_final,
                lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                max_steps=training_args.mlp_featurebank_lr_max_steps,
            )

    @torch.no_grad()
    def reset_densification_gradient_statistics(self) -> dict[str, int | float]:
        """Clear warmup-local gradient evidence before main densification.

        Opacity, visibility, birth iteration, and optimizer state are lifetime
        statistics and deliberately remain untouched.
        """
        self._validate_anchor_alignment(include_statistics=True)
        summary: dict[str, int | float] = {
            "observed_offsets": int((self.offset_denom > 0).sum().item()),
            "observation_count": float(self.offset_denom.sum().item()),
        }
        self.offset_gradient_accum.zero_()
        self.offset_denom.zero_()
        return summary

    def update_learning_rate(self, iteration, geometry_lr_scale=1.0):
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name == "anchor":
                group["lr"] = self.anchor_scheduler_args(iteration) * geometry_lr_scale
            elif name == "offset":
                group["lr"] = self.offset_scheduler_args(iteration) * geometry_lr_scale
            elif name == "mlp_opacity":
                group["lr"] = self.mlp_opacity_scheduler_args(iteration) * geometry_lr_scale
            elif name == "mlp_cov":
                group["lr"] = self.mlp_cov_scheduler_args(iteration) * geometry_lr_scale
            elif name == "mlp_featurebank":
                group["lr"] = self.mlp_featurebank_scheduler_args(iteration) * geometry_lr_scale
            elif name == "pose":
                group["lr"] = self.pose_scheduler_args(iteration)
            elif name.startswith("appearance_"):
                group["lr"] = self.appearance_scheduler_args(iteration)
            elif name in self._fixed_geometry_lrs:
                group["lr"] = self._fixed_geometry_lrs[name] * geometry_lr_scale

    def construct_list_of_attributes(self) -> list[str]:
        names = ["x", "y", "z", "nx", "ny", "nz", "model_format_version"]
        names.extend(f"f_offset_{index}" for index in range(self.n_offsets * 3))
        names.extend(f"f_anchor_feat_{index}" for index in range(self.feat_dim))
        names.extend(self.appearance.ply_attribute_names())
        names.append("opacity")
        names.extend(f"scale_{index}" for index in range(6))
        names.extend(f"rot_{index}" for index in range(4))
        names.append("filter_3D")
        return names

    def save_ply(self, path) -> None:
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu()
        count = anchor.shape[0]
        values = torch.cat(
            [
                anchor,
                torch.zeros_like(anchor),
                torch.full((count, 1), float(MODEL_FORMAT_VERSION)),
                self._offset.detach().transpose(1, 2).flatten(1).cpu(),
                self._anchor_feat.detach().cpu(),
                self.appearance.ply_values().cpu(),
                self._opacity.detach().cpu(),
                self._scaling.detach().cpu(),
                self._rotation.detach().cpu(),
                self.filter_3D.detach().cpu(),
            ],
            dim=1,
        ).numpy()
        dtype = [(name, "f4") for name in self.construct_list_of_attributes()]
        elements = np.empty(count, dtype=dtype)
        elements[:] = list(map(tuple, values))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply_sparse_gaussian(self, path) -> None:
        ply = PlyData.read(path, mmap=False)
        vertex = ply.elements[0]
        names = {prop.name for prop in vertex.properties}
        if "model_format_version" not in names:
            raise RuntimeError(
                f"PLY is not MODEL_FORMAT_VERSION={MODEL_FORMAT_VERSION}; legacy PLY files are unsupported"
            )
        versions = np.asarray(vertex["model_format_version"])
        if versions.size == 0 or not np.all(versions == MODEL_FORMAT_VERSION):
            raise RuntimeError(
                f"PLY model_format_version must be exactly {MODEL_FORMAT_VERSION}"
            )
        if "appearance_layout_version" not in names:
            raise RuntimeError("PLY has no explicit appearance layout version")
        appearance_versions = np.asarray(vertex["appearance_layout_version"])
        if appearance_versions.size == 0 or not np.all(
            appearance_versions == EXPLICIT_APPEARANCE_FORMAT_VERSION
        ):
            raise RuntimeError(
                "PLY explicit appearance parameterization is incompatible: "
                f"expected version {EXPLICIT_APPEARANCE_FORMAT_VERSION}; "
                "the unbounded enhanced-gain layout cannot be restored"
            )
        expected_names = set(self.construct_list_of_attributes())
        if names != expected_names:
            raise ValueError(
                "Invalid v2 PLY properties; "
                f"missing={sorted(expected_names - names)}, "
                f"extra={sorted(names - expected_names)}"
            )
        device = self.device

        def stack_properties(prefix: str, width: int) -> torch.Tensor:
            keys = [f"{prefix}{index}" for index in range(width)]
            missing = [key for key in keys if key not in names]
            if missing:
                raise ValueError(f"Missing v2 PLY properties: {missing[:3]}")
            array = np.stack([np.asarray(vertex[key]) for key in keys], axis=1).astype(np.float32)
            return torch.tensor(array, device=device)

        anchor = torch.tensor(
            np.stack([np.asarray(vertex[key]) for key in ("x", "y", "z")], axis=1).astype(np.float32),
            device=device,
        )
        count = anchor.shape[0]
        self._anchor = nn.Parameter(anchor.requires_grad_(True))
        self._offset = nn.Parameter(
            stack_properties("f_offset_", self.n_offsets * 3)
            .reshape(count, 3, self.n_offsets)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._anchor_feat = nn.Parameter(stack_properties("f_anchor_feat_", self.feat_dim).requires_grad_(True))
        appearance_properties = {
            name: torch.tensor(np.asarray(vertex[name]).astype(np.float32), device=device)
            for name in names
            if name.startswith("appearance_")
        }
        self.appearance.load_ply_values(appearance_properties)
        self._opacity = nn.Parameter(
            torch.tensor(np.asarray(vertex["opacity"])[..., None].astype(np.float32), device=device).requires_grad_(True)
        )
        self._scaling = nn.Parameter(stack_properties("scale_", 6).requires_grad_(True))
        self._rotation = nn.Parameter(stack_properties("rot_", 4).requires_grad_(True))
        self.filter_3D = torch.tensor(
            np.asarray(vertex["filter_3D"])[..., None].astype(np.float32),
            device=device,
        )
        self.max_radii2D = torch.zeros((count,), device=device)
        self._validate_anchor_alignment(include_statistics=False)

    def replace_tensor_to_optimizer(self, tensor, name):
        output = {}
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            state = self.optimizer.state.get(group["params"][0])
            if state is not None:
                state["exp_avg"] = torch.zeros_like(tensor)
                state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            output[name] = group["params"][0]
        return output

    @staticmethod
    def _is_growable_group(name: str) -> bool:
        return not name.startswith("mlp_") and name != "pose"

    def cat_tensors_to_optimizer(self, tensors):
        output = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if not self._is_growable_group(name):
                continue
            extension = tensors[name]
            old_parameter = group["params"][0]
            state = self.optimizer.state.get(old_parameter)
            if state is not None:
                state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(extension)), dim=0)
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
                del self.optimizer.state[old_parameter]
            parameter = nn.Parameter(torch.cat((old_parameter, extension), dim=0).requires_grad_(True))
            group["params"][0] = parameter
            if state is not None:
                self.optimizer.state[parameter] = state
            output[name] = parameter
        return output

    def _prune_anchor_optimizer(self, valid_mask):
        output = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if not self._is_growable_group(name):
                continue
            old_parameter = group["params"][0]
            state = self.optimizer.state.get(old_parameter)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][valid_mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][valid_mask]
                del self.optimizer.state[old_parameter]
            parameter = nn.Parameter(old_parameter[valid_mask].requires_grad_(True))
            group["params"][0] = parameter
            if state is not None:
                self.optimizer.state[parameter] = state
            output[name] = parameter
        return output

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        positive_opacity = opacity.detach().view(-1).clamp_min(0.0).view(-1, self.n_offsets)
        self.opacity_accum[anchor_visible_mask] += positive_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1
        self.anchor_visible_count[anchor_visible_mask] += 1
        flat_visible = anchor_visible_mask[:, None].expand(-1, self.n_offsets).reshape(-1)
        combined = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(1)
        combined[flat_visible] = offset_selection_mask
        selected = combined.clone()
        combined[selected] = update_filter
        gradient_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined] += gradient_norm
        self.offset_denom[combined] += 1

    def prune_anchor(self, prune_mask) -> None:
        valid = ~prune_mask
        tensors = self._prune_anchor_optimizer(valid)
        self._anchor = tensors["anchor"]
        self._offset = tensors["offset"]
        self._anchor_feat = tensors["anchor_feat"]
        self._opacity = tensors["opacity"]
        self._scaling = tensors["scaling"]
        self._rotation = tensors["rotation"]
        self.appearance.assign_optimizer_tensors(tensors)
        self.filter_3D = self.filter_3D[valid]

    def anchor_growing(
        self,
        grads,
        threshold,
        offset_mask,
        max_anchors=60_000,
        max_new_anchors=512,
        level_caps=(256, 160, 96),
        current_iteration=0,
    ):
        if isinstance(level_caps, str):
            level_caps = tuple(int(value) for value in level_caps.split(",") if value.strip())
        if not level_caps:
            level_caps = (max_new_anchors,)
        stats = {
            "candidates": 0,
            "added_by_level": [],
            "cap_hit": False,
            "selected_grad_min": 0.0,
            "appearance_params_inherited": 0,
        }
        total_added = 0
        selected_grad_min = None
        initial_length = self.get_anchor.shape[0] * self.n_offsets
        for level in range(self.update_depth):
            current_threshold = threshold * ((self.update_hierachy_factor // 2) ** level)
            candidate_mask = (grads >= current_threshold) & offset_mask
            length_increase = self.get_anchor.shape[0] * self.n_offsets - initial_length
            if length_increase == 0 and level > 0:
                continue
            if length_increase:
                candidate_mask = torch.cat(
                    [candidate_mask, torch.zeros(length_increase, dtype=torch.bool, device=candidate_mask.device)]
                )
            all_xyz = self.get_anchor[:, None] + self._offset * self.get_scaling[:, None, :3]
            size_factor = self.update_init_factor // (self.update_hierachy_factor ** level)
            current_size = self.voxel_size * size_factor
            grid_coords = torch.round(self.get_anchor / current_size).int()
            selected_xyz = all_xyz.reshape(-1, 3)[candidate_mask]
            selected_grid = torch.round(selected_xyz / current_size).int()
            if selected_grid.numel() == 0:
                stats["added_by_level"].append(0)
                continue
            unique_grid, inverse = torch.unique(selected_grid, return_inverse=True, dim=0)
            chunks = []
            for start in range(0, grid_coords.shape[0], 4096 * 5):
                chunks.append(
                    (unique_grid[:, None] == grid_coords[start : start + 4096 * 5]).all(-1).any(-1)
                )
            is_new = ~reduce(torch.logical_or, chunks)
            stats["candidates"] += int(is_new.sum().item())
            available = min(
                max(0, int(max_anchors) - self.get_anchor.shape[0]),
                max(0, int(max_new_anchors) - total_added),
            )
            level_cap = level_caps[level] if level < len(level_caps) else level_caps[-1]
            keep = min(int(is_new.sum()), int(level_cap), available)
            if keep <= 0:
                stats["added_by_level"].append(0)
                stats["cap_hit"] = True
                break
            source_scores = grads[candidate_mask[: grads.shape[0]]]
            voxel_scores = scatter_max(source_scores, inverse, dim=0)[0].view(-1)
            new_indices = torch.nonzero(is_new, as_tuple=False).squeeze(1)
            scores, top = torch.topk(voxel_scores[new_indices], k=keep, largest=True, sorted=False)
            selected_grad_min = min(
                float(scores.min().detach().cpu()),
                selected_grad_min if selected_grad_min is not None else float("inf"),
            )
            selected_voxels = torch.zeros_like(is_new)
            selected_voxels[new_indices[top]] = True
            candidate_anchor = unique_grid[selected_voxels] * current_size
            count = candidate_anchor.shape[0]

            source_features = self._anchor_feat[:, None].expand(-1, self.n_offsets, -1).reshape(-1, self.feat_dim)[candidate_mask]
            new_features = scatter_max(
                source_features,
                inverse[:, None].expand(-1, self.feat_dim),
                dim=0,
            )[0][selected_voxels]
            appearance = self.appearance.inherited_parameters(candidate_mask, inverse, selected_voxels)
            new_scaling = torch.log(torch.full((count, 6), current_size, device=candidate_anchor.device))
            new_rotation = torch.zeros((count, 4), device=candidate_anchor.device)
            new_rotation[:, 0] = 1.0
            new_opacity = inverse_sigmoid(
                torch.full(
                    (count, 1),
                    SCAFFOLD_INITIAL_OPACITY,
                    device=candidate_anchor.device,
                )
            )
            new_offsets = torch.zeros((count, self.n_offsets, 3), device=candidate_anchor.device)
            extension = {
                "anchor": candidate_anchor,
                "offset": new_offsets,
                "anchor_feat": new_features,
                "opacity": new_opacity,
                "scaling": new_scaling,
                "rotation": new_rotation,
                **appearance,
            }
            self.anchor_demon = torch.cat((self.anchor_demon, torch.zeros((count, 1), device=candidate_anchor.device)))
            self.opacity_accum = torch.cat((self.opacity_accum, torch.zeros((count, 1), device=candidate_anchor.device)))
            self.anchor_visible_count = torch.cat((self.anchor_visible_count, torch.zeros((count, 1), device=candidate_anchor.device)))
            self.anchor_birth_iteration = torch.cat(
                (
                    self.anchor_birth_iteration,
                    torch.full((count,), int(current_iteration), dtype=torch.long, device=candidate_anchor.device),
                )
            )
            self.filter_3D = torch.cat((self.filter_3D, torch.zeros((count, 1), device=candidate_anchor.device)))
            optimized = self.cat_tensors_to_optimizer(extension)
            self._anchor = optimized["anchor"]
            self._offset = optimized["offset"]
            self._anchor_feat = optimized["anchor_feat"]
            self._opacity = optimized["opacity"]
            self._scaling = optimized["scaling"]
            self._rotation = optimized["rotation"]
            self.appearance.assign_optimizer_tensors(optimized)
            total_added += count
            stats["appearance_params_inherited"] += count
            stats["added_by_level"].append(count)
        if selected_grad_min is not None:
            stats["selected_grad_min"] = selected_grad_min
        return stats

    def adjust_anchor(
        self,
        check_interval=100,
        success_threshold=0.8,
        grad_threshold=0.0002,
        min_opacity=0.005,
        max_anchors=60_000,
        max_new_anchors=512,
        level_caps=(256, 160, 96),
        current_iteration=0,
        prune_grace_iters=500,
        prune_from_iter=0,
        max_pruned_anchors=0,
        allow_prune=True,
    ):
        anchors_before = self.get_anchor.shape[0]
        grads = self.offset_gradient_accum / self.offset_denom
        grads[~torch.isfinite(grads)] = 0.0
        grads = torch.norm(grads, dim=-1)
        offset_mask = (
            self.offset_denom
            > check_interval
            * success_threshold
            * SCAFFOLD_DENSIFICATION_OBSERVATION_FRACTION
        ).squeeze(1)
        eligible = grads[offset_mask]
        if eligible.numel():
            quantiles = torch.quantile(eligible.float(), torch.tensor([0.5, 0.9, 0.99], device=eligible.device))
            grad_stats = {f"candidate_grad_q{q}": float(value.detach().cpu()) for q, value in zip((50, 90, 99), quantiles)}
        else:
            grad_stats = {f"candidate_grad_q{q}": 0.0 for q in (50, 90, 99)}
        stats = self.anchor_growing(
            grads,
            grad_threshold,
            offset_mask,
            max_anchors,
            max_new_anchors,
            level_caps,
            current_iteration,
        )
        stats.update(grad_stats)

        self.offset_denom[offset_mask] = 0
        self.offset_gradient_accum[offset_mask] = 0
        required = self.get_anchor.shape[0] * self.n_offsets
        padding = required - self.offset_denom.shape[0]
        if padding:
            zeros = torch.zeros((padding, 1), device=self.offset_denom.device)
            self.offset_denom = torch.cat((self.offset_denom, zeros), dim=0)
            self.offset_gradient_accum = torch.cat((self.offset_gradient_accum, zeros.clone()), dim=0)

        pruning_disabled = not allow_prune or max_pruned_anchors <= 0
        if pruning_disabled or current_iteration < prune_from_iter:
            self.max_radii2D = torch.zeros((self.get_anchor.shape[0],), device=self.get_anchor.device)
            stats.update({
                "anchors_before": anchors_before,
                "pruned": 0,
                "anchors_after": self.get_anchor.shape[0],
                "pruned_low_opacity": 0,
                "pruned_never_visible": 0,
                "prune_skipped": (
                    "disabled" if pruning_disabled else "before_prune_from_iter"
                ),
                "effective_mean_opacity_threshold": float(min_opacity) / self.n_offsets,
            })
            return stats

        low_opacity = (
            (self.opacity_accum < min_opacity * self.anchor_demon)
            & (self.anchor_demon > check_interval * success_threshold)
        ).squeeze(1)
        never_visible = (
            (current_iteration - self.anchor_birth_iteration >= prune_grace_iters)
            & (self.anchor_visible_count.squeeze(1) == 0)
            & ~low_opacity
        )
        prune = low_opacity | never_visible
        if int(prune.sum()) > max_pruned_anchors:
            candidates = torch.nonzero(prune, as_tuple=False).squeeze(1)
            mean_opacity = self.opacity_accum.squeeze(1) / self.anchor_demon.squeeze(1).clamp_min(1.0)
            _, order = torch.topk(mean_opacity[candidates], k=max_pruned_anchors, largest=False, sorted=False)
            limited = torch.zeros_like(prune)
            limited[candidates[order]] = True
            prune = limited

        self.offset_denom = self.offset_denom.view(-1, self.n_offsets)[~prune].reshape(-1, 1)
        self.offset_gradient_accum = self.offset_gradient_accum.view(-1, self.n_offsets)[~prune].reshape(-1, 1)
        self.opacity_accum = self.opacity_accum[~prune]
        self.anchor_demon = self.anchor_demon[~prune]
        self.anchor_visible_count = self.anchor_visible_count[~prune]
        self.anchor_birth_iteration = self.anchor_birth_iteration[~prune]
        pruned = int(prune.sum())
        pruned_low = int((prune & low_opacity).sum())
        pruned_unseen = int((prune & never_visible).sum())
        if pruned:
            self.prune_anchor(prune)
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0],), device=self.get_anchor.device)
        stats.update({
            "anchors_before": anchors_before,
            "pruned": pruned,
            "anchors_after": self.get_anchor.shape[0],
            "pruned_low_opacity": pruned_low,
            "pruned_never_visible": pruned_unseen,
            "prune_skipped": "",
            "effective_mean_opacity_threshold": float(min_opacity) / self.n_offsets,
        })
        return stats

    def save_mlp_checkpoints(self, path, mode="split") -> None:
        mkdir_p(path)
        if mode == "split":
            modules = {"opacity_mlp.pt": self.mlp_opacity, "cov_mlp.pt": self.mlp_cov}
            if self.use_feat_bank:
                modules["feature_bank_mlp.pt"] = self.mlp_feature_bank
            for filename, module in modules.items():
                was_training = module.training
                module.eval()
                input_width = 4 if filename == "feature_bank_mlp.pt" else (
                    self.feat_dim + 3 + (self.opacity_dist_dim if filename == "opacity_mlp.pt" else self.cov_dist_dim)
                )
                module_device = next(module.parameters()).device
                torch.jit.trace(module, torch.rand(1, input_width, device=module_device)).save(
                    os.path.join(path, filename)
                )
                module.train(was_training)
            return
        if mode == "unite":
            state = {
                "model_format_version": MODEL_FORMAT_VERSION,
                "opacity_mlp": self.mlp_opacity.state_dict(),
                "cov_mlp": self.mlp_cov.state_dict(),
            }
            if self.use_feat_bank:
                state["feature_bank_mlp"] = self.mlp_feature_bank.state_dict()
            torch.save(state, os.path.join(path, "checkpoints.pth"))
            return
        raise ValueError(f"Unknown MLP checkpoint mode: {mode}")

    def load_mlp_checkpoints(self, path, mode="split") -> None:
        if mode == "split":
            self.mlp_opacity = torch.jit.load(
                os.path.join(path, "opacity_mlp.pt"), map_location=self.device
            )
            self.mlp_cov = torch.jit.load(
                os.path.join(path, "cov_mlp.pt"), map_location=self.device
            )
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(
                    os.path.join(path, "feature_bank_mlp.pt"), map_location=self.device
                )
            return
        if mode == "unite":
            state = torch.load(os.path.join(path, "checkpoints.pth"), map_location=self.device)
            self._require_v2(state, "MLP checkpoint")
            expected = {"model_format_version", "opacity_mlp", "cov_mlp"}
            if self.use_feat_bank:
                expected.add("feature_bank_mlp")
            if set(state) != expected:
                raise ValueError(
                    "Invalid v2 MLP checkpoint fields; "
                    f"missing={sorted(expected - set(state))}, "
                    f"extra={sorted(set(state) - expected)}"
                )
            self.mlp_opacity.load_state_dict(state["opacity_mlp"])
            self.mlp_cov.load_state_dict(state["cov_mlp"])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(state["feature_bank_mlp"])
            return
        raise ValueError(f"Unknown MLP checkpoint mode: {mode}")
