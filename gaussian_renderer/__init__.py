"""Scaffold-GS rendering with explicit Gaussian-space R/L composition."""

from __future__ import annotations

import math
import time

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from einops import repeat

from scene.gaussian_model import GaussianModel
from utils.model_format import NUMERICAL_EPS
from utils.pose_utils import (
    get_camera_center_from_tensor,
    get_camera_from_tensor,
    get_tensor_from_camera,
    quadmultiply,
)


def generate_neural_gaussians(
    viewpoint_camera,
    pc: GaussianModel,
    visible_mask=None,
    effective_camera_center=None,
):
    """Generate Scaffold neural Gaussians and query one explicit appearance API."""
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    camera_center = (
        viewpoint_camera.camera_center
        if effective_camera_center is None
        else effective_camera_center
    )
    view_dirs = anchor - camera_center
    view_distance = view_dirs.norm(dim=1, keepdim=True).clamp_min(NUMERICAL_EPS)
    view_dirs = view_dirs / view_distance
    if pc.use_feat_bank:
        bank_weight = pc.get_featurebank_mlp(torch.cat([view_dirs, view_distance], dim=1)).unsqueeze(1)
        bank_feat = feat.unsqueeze(-1)
        feat = (
            bank_feat[:, ::4, :1].repeat(1, 4, 1) * bank_weight[:, :, :1]
            + bank_feat[:, ::2, :1].repeat(1, 2, 1) * bank_weight[:, :, 1:2]
            + bank_feat * bank_weight[:, :, 2:]
        ).squeeze(-1)

    local_without_distance = torch.cat([feat, view_dirs], dim=1)
    local_with_distance = torch.cat([local_without_distance, view_distance], dim=1)
    opacity_input = local_with_distance if pc.add_opacity_dist else local_without_distance
    neural_opacity = pc.get_opacity_mlp(opacity_input).reshape(-1, 1)
    if pc.use_3D_filter:
        neural_opacity = pc.get_opacity_with_3D_filter(neural_opacity, visible_mask)
        grid_scaling = pc.get_scaling_with_3D_filter(grid_scaling, visible_mask)
    selection_mask = neural_opacity.reshape(-1) > 0.0
    opacity = neural_opacity[selection_mask]

    # Appearance directions are observations, not a geometry-optimization
    # path. Geometry remains fully supervised by the main rasterization and
    # Scaffold opacity/covariance heads, while TV/enhanced losses cannot leak
    # into anchor positions through the directional lobes.
    appearance = pc.evaluate_appearance(
        view_dirs.detach(),
        visible_mask,
    )
    reflectance = appearance.reflectance.reshape(-1, 3)
    illumination = appearance.illumination.reshape(-1, 1)
    enhanced_color = appearance.enhanced_color.reshape(-1, 3)
    illumination_enhanced = appearance.illumination_enhanced.reshape(-1, 3)

    covariance_input = local_with_distance if pc.add_cov_dist else local_without_distance
    scale_rotation = pc.get_cov_mlp(covariance_input).reshape(-1, 7)
    offsets = grid_offsets.reshape(-1, 3)
    repeated = repeat(torch.cat([grid_scaling, anchor], dim=-1), "n c -> (n k) c", k=pc.n_offsets)
    packed = torch.cat(
        [
            repeated,
            reflectance,
            illumination,
            enhanced_color,
            illumination_enhanced,
            scale_rotation,
            offsets,
        ],
        dim=-1,
    )[selection_mask]
    scaling_repeat, repeated_anchor, reflectance, illumination, enhanced_color, illumination_enhanced, scale_rotation, offsets = packed.split(
        [6, 3, 3, 1, 3, 3, 7, 3],
        dim=-1,
    )
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rotation[:, :3])
    rotation = pc.rotation_activation(scale_rotation[:, 3:7])
    xyz = repeated_anchor + offsets * scaling_repeat[:, :3]
    return {
        "xyz": xyz,
        "reflectance": reflectance,
        "illumination": illumination,
        "enhanced_color": enhanced_color,
        "illumination_enhanced": illumination_enhanced,
        "opacity": opacity,
        "scaling": scaling,
        "rotation": rotation,
        "neural_opacity": neural_opacity,
        "selection_mask": selection_mask,
    }


def _camera_pose(viewpoint_camera, camera_pose):
    if camera_pose is not None:
        return camera_pose
    return get_tensor_from_camera(viewpoint_camera.world_view_transform.transpose(0, 1))


def _raster_settings(viewpoint_camera, pipe, background, kernel_size, scaling_modifier):
    identity = torch.eye(4, device=background.device, dtype=background.dtype)
    projection = identity.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        kernel_size=kernel_size,
        bg=background,
        scale_modifier=scaling_modifier,
        viewmatrix=identity,
        projmatrix=projection,
        sh_degree=1,
        campos=identity.inverse()[3, :3],
        prefiltered=False,
        debug=pipe.debug,
    )


def _transform_gaussians(xyz, rotation, camera_pose):
    relative = get_camera_from_tensor(camera_pose)
    homogeneous = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    transformed_xyz = (relative @ homogeneous.T).T[:, :3]
    transformed_rotation = quadmultiply(camera_pose[:4], rotation)
    return transformed_xyz, transformed_rotation


def _profile_now(enabled):
    if not enabled:
        return None
    torch.cuda.synchronize()
    return time.perf_counter()


def _normalize_by_coverage(
    accumulated: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Convert an alpha-premultiplied raster attribute to its covered mean."""
    return accumulated / coverage.clamp_min(NUMERICAL_EPS)


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size: float,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
    camera_pose=None,
    profile_timings=None,
):
    """Render all stable v2 outputs with explicit gradient ownership."""
    start = _profile_now(profile_timings is not None)
    pose = _camera_pose(viewpoint_camera, camera_pose)
    generated = generate_neural_gaussians(
        viewpoint_camera,
        pc,
        visible_mask,
        effective_camera_center=get_camera_center_from_tensor(pose),
    )
    if profile_timings is not None:
        after_generate = _profile_now(True)
        profile_timings["generate_neural_gaussians_time"] = profile_timings.get(
            "generate_neural_gaussians_time", 0.0
        ) + after_generate - start

    xyz = generated["xyz"]
    rotation = generated["rotation"]
    means3d, transformed_rotation = _transform_gaussians(xyz, rotation, pose)
    means2d = torch.zeros_like(xyz, requires_grad=True)
    if retain_grad:
        means2d.retain_grad()

    main_rasterizer = GaussianRasterizer(
        raster_settings=_raster_settings(
            viewpoint_camera,
            pipe,
            bg_color,
            kernel_size,
            scaling_modifier,
        )
    )
    diagnostic_rasterizer = GaussianRasterizer(
        raster_settings=_raster_settings(
            viewpoint_camera,
            pipe,
            torch.zeros_like(bg_color),
            kernel_size,
            scaling_modifier,
        )
    )
    raster_start = _profile_now(profile_timings is not None)
    reflectance = generated["reflectance"]
    illumination = generated["illumination"].expand(-1, 3)
    enhanced_illumination = generated["illumination_enhanced"]
    enhanced_color = generated["enhanced_color"]
    opacity = generated["opacity"]
    scaling = generated["scaling"]

    rendered, radii, _ = main_rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=None,
        colors_precomp=reflectance * illumination,
        opacities=opacity,
        scales=scaling,
        rotations=transformed_rotation,
        cov3D_precomp=None,
    )
    depth_coverage, _, depth_accumulated = diagnostic_rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=None,
        colors_precomp=torch.ones_like(reflectance),
        opacities=opacity,
        scales=scaling,
        rotations=transformed_rotation,
        cov3D_precomp=None,
    )
    depth_coverage_scalar = depth_coverage.mean(dim=0, keepdim=True)
    rendered_depth = _normalize_by_coverage(
        depth_accumulated,
        depth_coverage_scalar,
    )
    detached_coverage = depth_coverage.detach()
    detached_geometry = {
        "means3D": means3d.detach(),
        "means2D": means2d.detach(),
        "shs": None,
        "opacities": opacity.detach(),
        "scales": scaling.detach(),
        "rotations": transformed_rotation.detach(),
        "cov3D_precomp": None,
    }
    reflectance_accumulated, _, _ = diagnostic_rasterizer(
        colors_precomp=reflectance,
        **detached_geometry,
    )
    illumination_accumulated, _, _ = diagnostic_rasterizer(
        colors_precomp=illumination,
        **detached_geometry,
    )
    enhanced_illumination_accumulated, _, _ = diagnostic_rasterizer(
        colors_precomp=enhanced_illumination,
        **detached_geometry,
    )
    rendered_reflectance = _normalize_by_coverage(
        reflectance_accumulated,
        detached_coverage,
    )
    rendered_illumination = _normalize_by_coverage(
        illumination_accumulated,
        detached_coverage,
    )
    rendered_illumination_enhanced = _normalize_by_coverage(
        enhanced_illumination_accumulated,
        detached_coverage,
    )
    rendered_enhanced, _, _ = main_rasterizer(
        colors_precomp=enhanced_color,
        **detached_geometry,
    )
    if profile_timings is not None:
        after_raster = _profile_now(True)
        profile_timings["rasterize_main_time"] = profile_timings.get(
            "rasterize_main_time", 0.0
        ) + after_raster - raster_start

    return {
        "render": rendered,
        "render_enhanced": rendered_enhanced,
        "render_reflectance": rendered_reflectance,
        "render_illumination": rendered_illumination,
        "render_illumination_enhanced": rendered_illumination_enhanced,
        "render_depth": rendered_depth,
        "_training_coverage": detached_coverage,
        "viewspace_points": means2d,
        "visibility_filter": radii > 0,
        "radii": radii,
        "selection_mask": generated["selection_mask"],
        "neural_opacity": generated["neural_opacity"],
        "scaling": scaling,
    }

def prefilter_voxel(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    kernel_size: float,
    scaling_modifier=1.0,
    override_color=None,
    camera_pose=None,
):
    del override_color
    pose = _camera_pose(viewpoint_camera, camera_pose)
    means3d, rotation = _transform_gaussians(pc.get_anchor, pc.get_rotation, pose)
    rasterizer = GaussianRasterizer(
        raster_settings=_raster_settings(
            viewpoint_camera,
            pipe,
            bg_color,
            kernel_size,
            scaling_modifier,
        )
    )
    scales = pc.get_scaling
    if pc.use_3D_filter:
        visible = torch.ones(scales.shape[0], dtype=torch.bool, device=scales.device)
        scales = pc.get_scaling_with_3D_filter(scales, visible)
    radii = rasterizer.visible_filter(
        means3D=means3d,
        scales=scales[:, :3],
        rotations=rotation,
        cov3D_precomp=None,
    )
    return radii > 0
