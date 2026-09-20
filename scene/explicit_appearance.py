"""Explicit reflectance and lighting attached to Scaffold-GS anchor offsets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from utils.model_format import EXPLICIT_APPEARANCE_FORMAT_VERSION, NUMERICAL_EPS


@dataclass(frozen=True)
class AppearanceEvaluation:
    """Explicit per-offset appearance values consumed by the renderer."""

    reflectance: torch.Tensor
    illumination: torch.Tensor
    enhanced_diffuse: torch.Tensor
    enhanced_sg: torch.Tensor
    enhanced_color: torch.Tensor
    illumination_enhanced: torch.Tensor


def _safe_logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(NUMERICAL_EPS, 1.0 - NUMERICAL_EPS)
    return torch.log(value) - torch.log1p(-value)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(NUMERICAL_EPS)
    return value + torch.log(-torch.expm1(-value))


def decompose_enhanced_prior(
    enhanced_rgb: torch.Tensor,
    reflectance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an RGB prior into bounded diffuse and additive fill without P/R."""
    enhanced_base = torch.minimum(enhanced_rgb, reflectance)
    enhanced_diffuse = enhanced_base / reflectance.clamp_min(NUMERICAL_EPS)
    enhanced_fill = (enhanced_rgb - enhanced_base) / (
        1.0 - enhanced_base
    ).clamp_min(NUMERICAL_EPS)
    return enhanced_diffuse.clamp(0.0, 1.0), enhanced_fill.clamp(0.0, 1.0)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    canonical = torch.zeros_like(vector)
    canonical[..., 2] = 1.0
    vector = torch.where(
        vector.norm(dim=-1, keepdim=True) > NUMERICAL_EPS,
        vector,
        canonical,
    )
    return F.normalize(vector, dim=-1, eps=NUMERICAL_EPS)


def orthonormal_frame(
    axis: torch.Tensor,
    tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic orthonormal frame, including degenerate inputs."""
    canonical_axis = torch.zeros_like(axis)
    canonical_axis[..., 2] = 1.0
    z_axis = _normalize(torch.where(
        axis.norm(dim=-1, keepdim=True) > NUMERICAL_EPS,
        axis,
        canonical_axis,
    ))
    tangent = tangent - (tangent * z_axis).sum(dim=-1, keepdim=True) * z_axis

    basis_index = z_axis.abs().argmin(dim=-1)
    fallback = F.one_hot(basis_index, num_classes=3).to(z_axis.dtype)
    fallback = fallback - (fallback * z_axis).sum(dim=-1, keepdim=True) * z_axis
    tangent = torch.where(
        tangent.norm(dim=-1, keepdim=True) > NUMERICAL_EPS,
        tangent,
        fallback,
    )
    x_axis = _normalize(tangent)
    y_axis = _normalize(torch.cross(z_axis, x_axis, dim=-1))
    return z_axis, x_axis, y_axis


class ExplicitAppearance(nn.Module):
    """Own all non-neural reflectance, ASG, and enhanced-SG parameters."""

    PARAMETER_NAMES = (
        "reflectance_base",
        "reflectance_detail",
        "main_asg_axis",
        "main_asg_tangent",
        "main_asg_sharpness",
        "main_asg_energy",
        "main_asg_ambient",
        "enhanced_sg_axis",
        "enhanced_sg_sharpness",
        "enhanced_sg_energy",
        "enhanced_diffuse_raw",
    )

    def __init__(self, n_offsets: int) -> None:
        super().__init__()
        if n_offsets < 1:
            raise ValueError("n_offsets must be positive")
        self.n_offsets = int(n_offsets)
        for name in self.PARAMETER_NAMES:
            self.register_parameter(name, nn.Parameter(torch.empty(0)))

    @property
    def anchor_count(self) -> int:
        if self.reflectance_base.ndim == 0:
            return 0
        return int(self.reflectance_base.shape[0])

    @property
    def reflectance(self) -> torch.Tensor:
        centered_detail = self.reflectance_detail - self.reflectance_detail.mean(
            dim=1,
            keepdim=True,
        )
        return torch.sigmoid(self.reflectance_base[:, None, :] + centered_detail)

    def evaluate(
        self,
        view_dirs: torch.Tensor,
        visible_mask: torch.Tensor | None = None,
    ) -> AppearanceEvaluation:
        """Evaluate R, grayscale ASG, and bounded diffuse/additive RGB SG."""
        if visible_mask is None:
            visible_mask = torch.ones(
                self.anchor_count,
                dtype=torch.bool,
                device=self.reflectance_base.device,
            )
        view_dirs = _normalize(view_dirs).view(-1, 1, 3)
        enhanced_view_dirs = view_dirs.detach()

        reflectance = self.reflectance[visible_mask]
        main_axis, main_tangent, main_bitangent = orthonormal_frame(
            self.main_asg_axis[visible_mask],
            self.main_asg_tangent[visible_mask],
        )
        main_sharpness = F.softplus(self.main_asg_sharpness[visible_mask]) + NUMERICAL_EPS
        lambda_x = main_sharpness[..., :1]
        lambda_y = main_sharpness[..., 1:]
        dot_z = (view_dirs * main_axis).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        dot_x = (view_dirs * main_tangent).sum(dim=-1, keepdim=True)
        dot_y = (view_dirs * main_bitangent).sum(dim=-1, keepdim=True)
        asg = dot_z.clamp_min(0.0) * torch.exp(
            -lambda_x * dot_x.square() - lambda_y * dot_y.square()
        )
        main_ambient = torch.sigmoid(self.main_asg_ambient[visible_mask])
        main_energy = torch.sigmoid(self.main_asg_energy[visible_mask])
        illumination = main_ambient + (1.0 - main_ambient) * main_energy * asg

        enhanced_axis = _normalize(self.enhanced_sg_axis[visible_mask])
        enhanced_sharpness = (
            F.softplus(self.enhanced_sg_sharpness[visible_mask]) + NUMERICAL_EPS
        )
        enhanced_cosine = (enhanced_view_dirs * enhanced_axis).sum(
            dim=-1,
            keepdim=True,
        ).clamp(-1.0, 1.0)
        sg = torch.exp(enhanced_sharpness * (enhanced_cosine - 1.0))
        enhanced_diffuse = torch.sigmoid(self.enhanced_diffuse_raw[visible_mask])
        enhanced_energy = torch.sigmoid(self.enhanced_sg_energy[visible_mask])
        enhanced_sg = enhanced_energy * sg
        enhanced_base = reflectance.detach() * enhanced_diffuse
        enhanced_color = enhanced_base + (1.0 - enhanced_base) * enhanced_sg
        illumination_enhanced = (
            enhanced_diffuse + (1.0 - enhanced_diffuse) * enhanced_sg
        )
        return AppearanceEvaluation(
            reflectance=reflectance,
            illumination=illumination,
            enhanced_diffuse=enhanced_diffuse,
            enhanced_sg=enhanced_sg,
            enhanced_color=enhanced_color,
            illumination_enhanced=illumination_enhanced,
        )

    @torch.no_grad()
    def initialize(
        self,
        anchors: torch.Tensor,
        offsets: torch.Tensor,
        scaling: torch.Tensor,
        cameras: Iterable,
        chunk_size: int = 4096,
    ) -> None:
        """Initialize explicit parameters from robust multi-view projections."""
        cameras = list(cameras or [])
        device = anchors.device
        dtype = anchors.dtype
        count = anchors.shape[0]
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if count == 0:
            for name, value in self._empty_shapes(device, dtype).items():
                setattr(self, name, nn.Parameter(value.requires_grad_(True)))
            return
        if count and not cameras:
            raise ValueError(
                "Explicit appearance initialization requires training cameras; "
                "constant or random appearance fallback is intentionally unsupported"
            )
        positions = anchors[:, None, :] + offsets * scaling[:, None, :3]

        global_reflectance, global_low, global_diffuse, global_fill = self._global_statistics(
            cameras,
            device,
            dtype,
        )
        frontmost_masks = self._frontmost_visibility_masks(
            positions,
            cameras,
            chunk_size,
        )
        results = {name: [] for name in self.PARAMETER_NAMES}
        observed_offsets = []
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            chunk = positions[start:stop]
            low_rgb, enhanced_rgb, view_dirs, valid = self._sample_views(
                chunk,
                cameras,
                frontmost_masks[:, start:stop],
            )
            # A numerically black projection contains no usable Retinex ratio.
            valid = valid & (low_rgb.amax(dim=-1) > NUMERICAL_EPS)
            observed_offsets.append(valid.any(dim=0))
            values = self._initialize_chunk(
                low_rgb,
                enhanced_rgb,
                view_dirs,
                valid,
                global_reflectance,
                global_low,
                global_diffuse,
                global_fill,
            )
            for name, value in values.items():
                results[name].append(value)

        merged = {name: torch.cat(results[name], dim=0) for name in self.PARAMETER_NAMES}
        merged = self._fill_unobserved_with_global_medians(
            merged,
            torch.cat(observed_offsets, dim=0),
        )
        for name in self.PARAMETER_NAMES:
            value = merged[name]
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))

    @staticmethod
    def _fill_unobserved_with_global_medians(
        values: dict[str, torch.Tensor],
        observed_offsets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fill every missing offset from robust statistics of real projections."""
        if not observed_offsets.any():
            raise ValueError(
                "No anchor offset projects into any training view; "
                "explicit appearance cannot be initialized from data"
            )
        anchor_observed = observed_offsets.any(dim=1)
        base = values["reflectance_base"].clone()
        base[~anchor_observed] = base[anchor_observed].median(dim=0).values
        values["reflectance_base"] = base

        direction_names = {
            "main_asg_axis",
            "main_asg_tangent",
            "enhanced_sg_axis",
        }
        for name in ExplicitAppearance.PARAMETER_NAMES:
            if name == "reflectance_base":
                continue
            value = values[name].clone()
            observed_value = value[observed_offsets]
            if name in direction_names:
                reference = observed_value[:1]
                sign = torch.where(
                    (observed_value * reference).sum(dim=-1, keepdim=True) < 0.0,
                    -torch.ones_like(observed_value[..., :1]),
                    torch.ones_like(observed_value[..., :1]),
                )
                observed_value = observed_value * sign
            global_median = observed_value.median(dim=0).values
            value[~observed_offsets] = global_median
            values[name] = value

        detail = values["reflectance_detail"]
        values["reflectance_detail"] = detail - detail.mean(dim=1, keepdim=True)
        axis, tangent, _ = orthonormal_frame(
            values["main_asg_axis"],
            values["main_asg_tangent"],
        )
        values["main_asg_axis"] = axis
        values["main_asg_tangent"] = tangent
        values["enhanced_sg_axis"] = _normalize(values["enhanced_sg_axis"])
        return values

    def _empty_shapes(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        k = self.n_offsets
        return {
            "reflectance_base": torch.empty((0, 3), device=device, dtype=dtype),
            "reflectance_detail": torch.empty((0, k, 3), device=device, dtype=dtype),
            "main_asg_axis": torch.empty((0, k, 3), device=device, dtype=dtype),
            "main_asg_tangent": torch.empty((0, k, 3), device=device, dtype=dtype),
            "main_asg_sharpness": torch.empty((0, k, 2), device=device, dtype=dtype),
            "main_asg_energy": torch.empty((0, k, 1), device=device, dtype=dtype),
            "main_asg_ambient": torch.empty((0, k, 1), device=device, dtype=dtype),
            "enhanced_sg_axis": torch.empty((0, k, 3), device=device, dtype=dtype),
            "enhanced_sg_sharpness": torch.empty((0, k, 1), device=device, dtype=dtype),
            "enhanced_sg_energy": torch.empty((0, k, 3), device=device, dtype=dtype),
            "enhanced_diffuse_raw": torch.empty((0, k, 3), device=device, dtype=dtype),
        }

    @staticmethod
    def _camera_prior(camera, low_image: torch.Tensor) -> torch.Tensor:
        prior = getattr(camera, "enhancement_prior", None)
        if prior is None:
            return low_image
        return prior.to(device=low_image.device, dtype=low_image.dtype)

    def _global_statistics(
        self,
        cameras: list,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reflectance_values = []
        low_values = []
        diffuse_values = []
        fill_values = []
        for camera in cameras:
            image = camera.original_image[:3].to(device=device, dtype=dtype)
            low_raw = image.amax(dim=0, keepdim=True)
            valid = low_raw.squeeze(0) > NUMERICAL_EPS
            if not valid.any():
                continue
            low = low_raw.clamp_min(NUMERICAL_EPS)
            reflectance = image / low
            reflectance_values.append(reflectance[:, valid].median(dim=1).values)
            low_values.append(low_raw[:, valid].median())
            enhanced = self._camera_prior(camera, image)
            valid_enhanced = enhanced[:, valid]
            if (
                not torch.isfinite(valid_enhanced).all()
                or (valid_enhanced < 0.0).any()
                or (valid_enhanced > 1.0).any()
            ):
                raise ValueError(
                    "Enhanced initialization requires a finite prior in [0, 1]"
                )
            enhanced_diffuse, enhanced_fill = decompose_enhanced_prior(
                enhanced,
                reflectance,
            )
            diffuse_values.append(enhanced_diffuse[:, valid].median(dim=1).values)
            fill_values.append(enhanced_fill[:, valid].median(dim=1).values)
        if not reflectance_values:
            raise ValueError(
                "Global appearance statistics require at least one finite, "
                "nonblack training pixel"
            )
        return (
            torch.stack(reflectance_values).median(dim=0).values,
            torch.stack(low_values).median().reshape(1),
            torch.stack(diffuse_values).median(dim=0).values,
            torch.stack(fill_values).median(dim=0).values,
        )

    @staticmethod
    def _project_camera(
        flat_positions: torch.Tensor,
        camera,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project world positions and return rounded pixels plus monotonic NDC depth."""
        device = flat_positions.device
        dtype = flat_positions.dtype
        homogeneous = torch.cat(
            [flat_positions, torch.ones_like(flat_positions[:, :1])],
            dim=1,
        )
        # Camera stores world-view and projection matrices already transposed
        # for row-vector multiplication (the same convention as
        # geom_transform_points); do not transpose the composed matrix again.
        clip = homogeneous @ camera.full_proj_transform.to(
            device=device,
            dtype=dtype,
        )
        divisor = clip[:, 3]
        safe_divisor = torch.where(
            divisor.abs() > NUMERICAL_EPS,
            divisor,
            torch.ones_like(divisor),
        )
        ndc = clip[:, :3] / safe_divisor[:, None]
        x_float = (ndc[:, 0] + 1.0) * (width - 1) / 2.0
        y_float = (1.0 - ndc[:, 1]) * (height - 1) / 2.0
        valid = (
            (divisor > 0)
            & (ndc[:, 2] > -1.0)
            & (ndc[:, 2] < 1.0)
            & (x_float >= 0)
            & (x_float <= width - 1)
            & (y_float >= 0)
            & (y_float <= height - 1)
        )
        x = x_float.round().long().clamp(0, width - 1)
        y = y_float.round().long().clamp(0, height - 1)
        camera_center = camera.camera_center.to(device=device, dtype=dtype)
        return x, y, valid, ndc[:, 2], camera_center

    def _frontmost_visibility_masks(
        self,
        positions: torch.Tensor,
        cameras: list,
        chunk_size: int,
    ) -> torch.Tensor:
        """Compute a global per-view point z-buffer without materializing RGB samples."""
        count, offsets, _ = positions.shape
        device = positions.device
        dtype = positions.dtype
        masks = []
        for camera in cameras:
            height, width = camera.original_image.shape[-2:]
            nearest = torch.full(
                (height * width,),
                torch.inf,
                device=device,
                dtype=dtype,
            )
            for start in range(0, count, chunk_size):
                flat = positions[start : start + chunk_size].reshape(-1, 3)
                x, y, valid, depths, _ = self._project_camera(
                    flat,
                    camera,
                    height,
                    width,
                )
                if valid.any():
                    pixels = y[valid] * width + x[valid]
                    nearest.scatter_reduce_(
                        0,
                        pixels,
                        depths[valid],
                        reduce="amin",
                        include_self=True,
                    )
            camera_masks = []
            for start in range(0, count, chunk_size):
                flat = positions[start : start + chunk_size].reshape(-1, 3)
                x, y, valid, depths, _ = self._project_camera(
                    flat,
                    camera,
                    height,
                    width,
                )
                pixels = y * width + x
                frontmost = valid & (
                    depths <= nearest[pixels] + NUMERICAL_EPS
                )
                camera_masks.append(frontmost.reshape(-1, offsets))
            masks.append(torch.cat(camera_masks, dim=0))
        if not masks:
            return torch.empty(
                (0, count, offsets),
                device=device,
                dtype=torch.bool,
            )
        return torch.stack(masks)

    def _sample_views(
        self,
        positions: torch.Tensor,
        cameras: list,
        frontmost_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count, offsets, _ = positions.shape
        device = positions.device
        dtype = positions.dtype
        low_samples = []
        enhanced_samples = []
        directions = []
        valid_samples = []
        flat_positions = positions.reshape(-1, 3)
        if frontmost_masks is not None and tuple(frontmost_masks.shape) != (
            len(cameras),
            count,
            offsets,
        ):
            raise ValueError(
                "frontmost_masks shape must be "
                f"{(len(cameras), count, offsets)}, got {tuple(frontmost_masks.shape)}"
            )
        for camera_index, camera in enumerate(cameras):
            image = camera.original_image[:3].to(device=device, dtype=dtype)
            enhanced = self._camera_prior(camera, image)
            height, width = image.shape[-2:]
            x, y, valid, depths, camera_center = self._project_camera(
                flat_positions,
                camera,
                height,
                width,
            )
            if frontmost_masks is not None:
                valid = valid & frontmost_masks[camera_index].reshape(-1)
            else:
                valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
                if valid_indices.numel():
                    pixel_indices = y[valid_indices] * width + x[valid_indices]
                    _, inverse = torch.unique(pixel_indices, return_inverse=True)
                    nearest = torch.full(
                        (int(inverse.max().item()) + 1,),
                        torch.inf,
                        device=device,
                        dtype=dtype,
                    )
                    nearest.scatter_reduce_(
                        0,
                        inverse,
                        depths[valid_indices],
                        reduce="amin",
                        include_self=True,
                    )
                    frontmost = (
                        depths[valid_indices]
                        <= nearest[inverse] + NUMERICAL_EPS
                    )
                    visible = torch.zeros_like(valid)
                    visible[valid_indices] = frontmost
                    valid = visible
            low_samples.append(image[:, y, x].T.reshape(count, offsets, 3))
            enhanced_samples.append(enhanced[:, y, x].T.reshape(count, offsets, 3))
            directions.append(_normalize(flat_positions - camera_center).reshape(count, offsets, 3))
            valid_samples.append(valid.reshape(count, offsets))
        if not cameras:
            shape = (0, count, offsets)
            return (
                torch.empty((*shape, 3), device=device, dtype=dtype),
                torch.empty((*shape, 3), device=device, dtype=dtype),
                torch.empty((*shape, 3), device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=torch.bool),
            )
        return (
            torch.stack(low_samples),
            torch.stack(enhanced_samples),
            torch.stack(directions),
            torch.stack(valid_samples),
        )

    def _initialize_chunk(
        self,
        low_rgb: torch.Tensor,
        enhanced_rgb: torch.Tensor,
        view_dirs: torch.Tensor,
        valid: torch.Tensor,
        global_reflectance: torch.Tensor,
        global_low: torch.Tensor,
        global_diffuse: torch.Tensor,
        global_fill: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if low_rgb.shape[0] == 0:
            raise ValueError("Appearance initialization received no projected camera batch")
        low_intensity = low_rgb.amax(dim=-1, keepdim=True).clamp_min(NUMERICAL_EPS)
        reflectance_samples = low_rgb / low_intensity
        reflectance = self._masked_median(reflectance_samples, valid, global_reflectance)
        low = self._masked_median(low_intensity, valid, global_low)
        valid_enhanced = enhanced_rgb[valid]
        if (
            not torch.isfinite(valid_enhanced).all()
            or (valid_enhanced < 0.0).any()
            or (valid_enhanced > 1.0).any()
        ):
            raise ValueError(
                "Enhanced initialization requires a finite prior in [0, 1]"
            )
        enhanced_diffuse_samples, enhanced_fill_samples = decompose_enhanced_prior(
            enhanced_rgb,
            reflectance_samples,
        )
        enhanced_diffuse = self._masked_median(
            enhanced_diffuse_samples,
            valid,
            global_diffuse,
        )

        low_excess = (low_intensity - low.unsqueeze(0)).clamp_min(0.0)
        direction, tangent, main_variance = self._direction_statistics(
            view_dirs,
            low_excess,
            valid,
        )
        enhanced_excess = enhanced_fill_samples.mean(dim=-1, keepdim=True)
        enhanced_direction, _, enhanced_variance = self._direction_statistics(
            view_dirs,
            enhanced_excess,
            valid,
        )
        main_sharpness = 1.0 / main_variance.clamp_min(NUMERICAL_EPS)
        enhanced_sharpness = 1.0 / enhanced_variance.mean(
            dim=-1,
            keepdim=True,
        ).clamp_min(NUMERICAL_EPS)

        reflectance_logits = _safe_logit(reflectance)
        base = reflectance_logits.mean(dim=1)
        detail = reflectance_logits - base[:, None, :]
        main_axis, main_tangent, main_bitangent = orthonormal_frame(direction, tangent)
        main_dot_z = (view_dirs * main_axis.unsqueeze(0)).sum(dim=-1, keepdim=True)
        main_dot_x = (view_dirs * main_tangent.unsqueeze(0)).sum(dim=-1, keepdim=True)
        main_dot_y = (view_dirs * main_bitangent.unsqueeze(0)).sum(dim=-1, keepdim=True)
        main_basis = main_dot_z.clamp_min(0.0) * torch.exp(
            -main_sharpness.unsqueeze(0)[..., :1] * main_dot_x.square()
            -main_sharpness.unsqueeze(0)[..., 1:] * main_dot_y.square()
        )
        main_target = (low_intensity - low.unsqueeze(0)) / (
            1.0 - low.unsqueeze(0)
        ).clamp_min(NUMERICAL_EPS)
        valid_weight = valid[..., None].to(low_rgb.dtype)
        main_energy = (
            (main_basis * main_target * valid_weight).sum(dim=0)
            / (main_basis.square() * valid_weight).sum(dim=0).clamp_min(NUMERICAL_EPS)
        ).clamp(0.0, 1.0)

        enhanced_axis = _normalize(enhanced_direction)
        enhanced_cosine = (view_dirs * enhanced_axis.unsqueeze(0)).sum(
            dim=-1,
            keepdim=True,
        ).clamp(-1.0, 1.0)
        enhanced_basis = torch.exp(
            enhanced_sharpness.unsqueeze(0) * (enhanced_cosine - 1.0)
        )
        enhanced_energy = (
            (enhanced_basis * enhanced_fill_samples * valid_weight).sum(dim=0)
            / (enhanced_basis.square() * valid_weight).sum(dim=0).clamp_min(
                NUMERICAL_EPS
            )
        ).clamp(0.0, 1.0)
        has_observation = valid.any(dim=0, keepdim=False)[..., None]
        enhanced_energy = torch.where(
            has_observation,
            enhanced_energy,
            global_fill.view(1, 1, 3).expand_as(enhanced_energy),
        )

        return {
            "reflectance_base": base,
            "reflectance_detail": detail,
            "main_asg_axis": direction,
            "main_asg_tangent": tangent,
            "main_asg_sharpness": _inverse_softplus(main_sharpness),
            "main_asg_energy": _safe_logit(main_energy),
            "main_asg_ambient": _safe_logit(low),
            "enhanced_sg_axis": enhanced_direction,
            "enhanced_sg_sharpness": _inverse_softplus(enhanced_sharpness),
            "enhanced_sg_energy": _safe_logit(enhanced_energy),
            "enhanced_diffuse_raw": _safe_logit(enhanced_diffuse),
        }

    @staticmethod
    def _masked_median(
        values: torch.Tensor,
        valid: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        masked = values.masked_fill(~valid[..., None], torch.nan)
        median = torch.nanmedian(masked, dim=0).values
        fallback = fallback.view(*([1] * (median.ndim - 1)), -1).expand_as(median)
        return torch.where(torch.isfinite(median), median, fallback)

    @staticmethod
    def _direction_statistics(
        view_dirs: torch.Tensor,
        directional_signal: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = directional_signal.squeeze(-1) * valid.to(directional_signal.dtype)
        raw_weight_sum = weights.sum(dim=0, keepdim=False)
        weight_sum = raw_weight_sum.clamp_min(NUMERICAL_EPS)
        axis_raw = (view_dirs * weights[..., None]).sum(dim=0) / weight_sum[..., None]
        fallback = (view_dirs * valid[..., None]).sum(dim=0)
        axis_raw = torch.where(
            axis_raw.norm(dim=-1, keepdim=True) > NUMERICAL_EPS,
            axis_raw,
            fallback,
        )
        fallback_axis = torch.zeros_like(axis_raw)
        fallback_axis[..., 2] = 1.0
        axis = _normalize(torch.where(
            axis_raw.norm(dim=-1, keepdim=True) > NUMERICAL_EPS,
            axis_raw,
            fallback_axis,
        ))
        planar = view_dirs - (view_dirs * axis.unsqueeze(0)).sum(
            dim=-1,
            keepdim=True,
        ) * axis.unsqueeze(0)
        planar_mean = (planar * weights[..., None]).sum(dim=0) / weight_sum[..., None]
        centered = planar - planar_mean.unsqueeze(0)
        covariance = torch.einsum(
            "vnki,vnkj,vnk->nkij",
            centered,
            centered,
            weights,
        ) / weight_sum[..., None, None]
        covariance = (covariance + covariance.transpose(-1, -2)) / 2.0
        active = raw_weight_sum > NUMERICAL_EPS
        tangent = torch.zeros_like(axis)
        if active.any():
            _, eigenvectors = torch.linalg.eigh(covariance[active].double())
            tangent[active] = eigenvectors[..., -1].to(tangent.dtype)
        axis, tangent, bitangent = orthonormal_frame(axis, tangent)
        variance_x = (
            weights * (centered * tangent.unsqueeze(0)).sum(dim=-1).square()
        ).sum(dim=0) / weight_sum
        variance_y = (
            weights * (centered * bitangent.unsqueeze(0)).sum(dim=-1).square()
        ).sum(dim=0) / weight_sum
        variance = torch.stack([variance_x, variance_y], dim=-1).clamp_min(NUMERICAL_EPS)
        return axis, tangent, variance

    def optimizer_groups(self, learning_rate: float) -> list[dict]:
        return [
            {"params": [getattr(self, name)], "lr": learning_rate, "name": f"appearance_{name}"}
            for name in self.PARAMETER_NAMES
        ]

    def assign_optimizer_tensors(self, tensors: dict[str, nn.Parameter]) -> None:
        for name in self.PARAMETER_NAMES:
            key = f"appearance_{name}"
            if key in tensors:
                setattr(self, name, tensors[key])

    @torch.no_grad()
    def inherited_parameters(
        self,
        candidate_mask: torch.Tensor,
        inverse_indices: torch.Tensor,
        selected_voxels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Average complete source-anchor appearances for new anchor voxels."""
        source_anchor = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1) // self.n_offsets
        output: dict[str, torch.Tensor] = {}
        group_count = int(inverse_indices.max().item()) + 1
        selected_count = int(selected_voxels.sum().item())
        counts = torch.zeros(
            (group_count, 1),
            device=source_anchor.device,
            dtype=self.reflectance_base.dtype,
        )
        counts.index_add_(
            0,
            inverse_indices,
            torch.ones(
                (source_anchor.shape[0], 1),
                device=counts.device,
                dtype=counts.dtype,
            ),
        )

        def group_mean(value: torch.Tensor) -> torch.Tensor:
            trailing_shape = value.shape[1:]
            source = value.reshape(source_anchor.shape[0], -1)
            sums = torch.zeros(
                (group_count, source.shape[1]),
                device=source.device,
                dtype=source.dtype,
            )
            sums.index_add_(0, inverse_indices, source)
            return (sums / counts.clamp_min(1.0))[selected_voxels].reshape(
                selected_count,
                *trailing_shape,
            )

        def align_hemispheres(value: torch.Tensor) -> torch.Tensor:
            references = []
            for group_index in range(group_count):
                first = torch.nonzero(
                    inverse_indices == group_index,
                    as_tuple=False,
                )[0, 0]
                references.append(value[first])
            reference = torch.stack(references, dim=0)[inverse_indices]
            sign = torch.where(
                (value * reference).sum(dim=-1, keepdim=True) < 0.0,
                -torch.ones_like(value[..., :1]),
                torch.ones_like(value[..., :1]),
            )
            return value * sign

        # Reflectance base/detail are a coupled logit parameterization.  Mean
        # the decoded per-offset RGB reflectance and encode a new hierarchy.
        inherited_reflectance = group_mean(self.reflectance[source_anchor])
        inherited_reflectance_logits = _safe_logit(inherited_reflectance)
        inherited_base = inherited_reflectance_logits.mean(dim=1)
        output["appearance_reflectance_base"] = inherited_base
        output["appearance_reflectance_detail"] = (
            inherited_reflectance_logits - inherited_base[:, None, :]
        )

        source_main_axis, source_main_tangent, _ = orthonormal_frame(
            self.main_asg_axis[source_anchor],
            self.main_asg_tangent[source_anchor],
        )
        decoded_directions = {
            "main_asg_axis": source_main_axis,
            "main_asg_tangent": source_main_tangent,
            "enhanced_sg_axis": _normalize(self.enhanced_sg_axis[source_anchor]),
        }
        softplus_names = {
            "main_asg_sharpness",
            "enhanced_sg_sharpness",
        }
        sigmoid_names = {
            "main_asg_energy",
            "main_asg_ambient",
            "enhanced_sg_energy",
            "enhanced_diffuse_raw",
        }

        for name in self.PARAMETER_NAMES:
            if name in {"reflectance_base", "reflectance_detail"}:
                continue
            parameter = getattr(self, name)
            source_value = decoded_directions.get(name, parameter[source_anchor])
            if name in decoded_directions:
                source_value = align_hemispheres(source_value)
            elif name in softplus_names:
                source_value = F.softplus(source_value)
            elif name in sigmoid_names:
                source_value = torch.sigmoid(source_value)
            inherited = group_mean(source_value)
            if name in softplus_names:
                inherited = _inverse_softplus(inherited)
            elif name in sigmoid_names:
                inherited = _safe_logit(inherited)
            output[f"appearance_{name}"] = inherited

        axis, tangent, _ = orthonormal_frame(
            output["appearance_main_asg_axis"],
            output["appearance_main_asg_tangent"],
        )
        output["appearance_main_asg_axis"] = axis
        output["appearance_main_asg_tangent"] = tangent
        output["appearance_enhanced_sg_axis"] = _normalize(
            output["appearance_enhanced_sg_axis"]
        )
        return output

    def prune_without_optimizer(self, valid_mask: torch.Tensor) -> None:
        for name in self.PARAMETER_NAMES:
            value = getattr(self, name)[valid_mask].detach()
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))

    def state_dict_v2(self) -> dict[str, torch.Tensor | int]:
        state: dict[str, torch.Tensor | int] = {
            "explicit_appearance_format_version": EXPLICIT_APPEARANCE_FORMAT_VERSION,
        }
        state.update({name: getattr(self, name).detach() for name in self.PARAMETER_NAMES})
        return state

    def parameter_shapes(self, anchor_count: int) -> dict[str, tuple[int, ...]]:
        k = self.n_offsets
        return {
            "reflectance_base": (anchor_count, 3),
            "reflectance_detail": (anchor_count, k, 3),
            "main_asg_axis": (anchor_count, k, 3),
            "main_asg_tangent": (anchor_count, k, 3),
            "main_asg_sharpness": (anchor_count, k, 2),
            "main_asg_energy": (anchor_count, k, 1),
            "main_asg_ambient": (anchor_count, k, 1),
            "enhanced_sg_axis": (anchor_count, k, 3),
            "enhanced_sg_sharpness": (anchor_count, k, 1),
            "enhanced_sg_energy": (anchor_count, k, 3),
            "enhanced_diffuse_raw": (anchor_count, k, 3),
        }

    def load_state_dict_v2(self, state: dict[str, torch.Tensor | int]) -> None:
        version_key = "explicit_appearance_format_version"
        expected = {version_key, *self.PARAMETER_NAMES}
        missing = expected - set(state)
        extra = set(state) - expected
        if missing or extra:
            raise ValueError(
                f"Invalid v2 explicit appearance fields; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if state[version_key] != EXPLICIT_APPEARANCE_FORMAT_VERSION:
            raise RuntimeError(
                "Explicit appearance parameterization is incompatible: "
                f"expected version {EXPLICIT_APPEARANCE_FORMAT_VERSION}, "
                f"got {state[version_key]}. Retrain instead of reinterpreting "
                "the previous enhanced-SG parameterization."
            )
        anchor_count = int(state["reflectance_base"].shape[0])
        expected_shapes = self.parameter_shapes(anchor_count)
        for name in self.PARAMETER_NAMES:
            value = state[name]
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"Invalid v2 appearance shape for {name}: "
                    f"{tuple(value.shape)} != {expected_shapes[name]}"
                )
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))

    def ply_attribute_names(self) -> list[str]:
        names: list[str] = ["appearance_layout_version"]
        shapes = self.parameter_shapes(1)
        for parameter_name in self.PARAMETER_NAMES:
            width = int(torch.tensor(shapes[parameter_name][1:]).prod().item())
            names.extend(f"appearance_{parameter_name}_{index}" for index in range(width))
        return names

    def ply_values(self) -> torch.Tensor:
        version = torch.full(
            (self.anchor_count, 1),
            float(EXPLICIT_APPEARANCE_FORMAT_VERSION),
            device=self.reflectance_base.device,
            dtype=self.reflectance_base.dtype,
        )
        return torch.cat(
            [version]
            + [getattr(self, name).detach().reshape(self.anchor_count, -1) for name in self.PARAMETER_NAMES],
            dim=1,
        )

    def load_ply_values(self, properties: dict[str, torch.Tensor]) -> None:
        version_key = "appearance_layout_version"
        if version_key not in properties:
            raise RuntimeError(
                "PLY has no explicit appearance parameterization version; "
                "older enhanced-SG PLY layouts are unsupported"
            )
        versions = properties[version_key]
        if versions.numel() == 0 or not torch.all(
            versions == float(EXPLICIT_APPEARANCE_FORMAT_VERSION)
        ):
            raise RuntimeError(
                "PLY explicit appearance parameterization is incompatible: "
                f"expected version {EXPLICIT_APPEARANCE_FORMAT_VERSION}"
            )
        count = next(iter(properties.values())).shape[0]
        shapes = self.parameter_shapes(count)
        for name, shape in shapes.items():
            width = int(torch.tensor(shape[1:]).prod().item())
            keys = [f"appearance_{name}_{index}" for index in range(width)]
            missing = [key for key in keys if key not in properties]
            if missing:
                raise ValueError(f"Missing v2 PLY appearance properties: {missing[:3]}")
            value = torch.stack([properties[key] for key in keys], dim=1).reshape(shape)
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))
