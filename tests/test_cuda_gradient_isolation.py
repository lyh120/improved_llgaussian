import unittest
from types import SimpleNamespace

import torch

from scene.explicit_appearance import ExplicitAppearance
from utils.loss_utils import (
    coverage_masked_prediction,
    edge_aware_illumination_tv_loss,
    photo_loss,
)
from utils.model_format import NUMERICAL_EPS

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    GaussianRasterizationSettings = None
    GaussianRasterizer = None


CUDA_RASTERIZER_AVAILABLE = (
    torch.cuda.is_available()
    and GaussianRasterizationSettings is not None
    and GaussianRasterizer is not None
)


@unittest.skipUnless(CUDA_RASTERIZER_AVAILABLE, "CUDA rasterizer is not available")
class RasterGradientIsolationTest(unittest.TestCase):
    @staticmethod
    def _constant_attribute_render(opacity_value):
        device = torch.device("cuda")
        settings = GaussianRasterizationSettings(
            image_height=8,
            image_width=8,
            tanfovx=1.0,
            tanfovy=1.0,
            kernel_size=0.1,
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device),
            projmatrix=torch.eye(4, device=device),
            sh_degree=1,
            campos=torch.zeros(3, device=device),
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        geometry = {
            "means3D": torch.tensor([[0.0, 0.0, 1.0]], device=device),
            "means2D": torch.zeros((1, 3), device=device),
            "shs": None,
            "opacities": torch.tensor([[opacity_value]], device=device),
            "scales": torch.full((1, 3), 0.2, device=device),
            "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            "cov3D_precomp": None,
        }
        color = torch.tensor([[0.2, 0.5, 0.8]], device=device)
        accumulated, _, depth_accumulated = rasterizer(
            colors_precomp=color,
            **geometry,
        )
        coverage, _, _ = rasterizer(
            colors_precomp=torch.ones_like(color),
            **geometry,
        )
        scalar_coverage = coverage.mean(dim=0, keepdim=True)
        normalized = accumulated / coverage.clamp_min(NUMERICAL_EPS)
        normalized_depth = depth_accumulated / scalar_coverage.clamp_min(NUMERICAL_EPS)
        return normalized, normalized_depth, scalar_coverage

    def test_normalized_attributes_and_depth_are_opacity_invariant(self):
        low_opacity = self._constant_attribute_render(0.2)
        high_opacity = self._constant_attribute_render(0.8)
        mask = (low_opacity[2] > 1e-4) & (high_opacity[2] > 1e-4)
        color_mask = mask.expand_as(low_opacity[0])
        self.assertTrue(torch.allclose(
            low_opacity[0][color_mask],
            high_opacity[0][color_mask],
            atol=1e-5,
            rtol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            low_opacity[1][mask],
            high_opacity[1][mask],
            atol=1e-5,
            rtol=1e-5,
        ))

    def test_enhanced_raster_only_updates_enhanced_sg(self):
        device = torch.device("cuda")
        appearance = ExplicitAppearance(1).to(device)
        appearance.initialize(
            torch.tensor([[0.0, 0.0, 1.0]], device=device),
            torch.zeros((1, 1, 3), device=device),
            torch.ones((1, 6), device=device),
            cameras=[SimpleNamespace(
                original_image=torch.full((3, 5, 5), 0.5, device=device),
                enhancement_prior=torch.full((3, 5, 5), 0.5, device=device),
                full_proj_transform=torch.eye(4, device=device),
                camera_center=torch.tensor([0.0, 0.0, 0.0], device=device),
            )],
        )
        position = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
        screenspace = torch.zeros_like(position, requires_grad=True)
        opacity_raw = torch.zeros((1, 1), device=device, requires_grad=True)
        scale_raw = torch.zeros((1, 3), device=device, requires_grad=True)
        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
        evaluated = appearance.evaluate(
            torch.tensor([[0.0, 0.0, 1.0]], device=device)
        )
        settings = GaussianRasterizationSettings(
            image_height=8,
            image_width=8,
            tanfovx=1.0,
            tanfovy=1.0,
            kernel_size=0.1,
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device),
            projmatrix=torch.eye(4, device=device),
            sh_degree=1,
            campos=torch.zeros(3, device=device),
            prefiltered=False,
            debug=False,
        )
        image, _, _ = GaussianRasterizer(raster_settings=settings)(
            means3D=position.detach(),
            means2D=screenspace.detach(),
            shs=None,
            colors_precomp=evaluated.enhanced_color.reshape(-1, 3),
            opacities=torch.sigmoid(opacity_raw).detach(),
            scales=(torch.sigmoid(scale_raw) + 0.01).detach(),
            rotations=rotation.detach(),
            cov3D_precomp=None,
        )
        image.sum().backward()
        enhanced_names = {
            "enhanced_sg_axis",
            "enhanced_sg_sharpness",
            "enhanced_sg_energy",
            "enhanced_diffuse_raw",
        }
        for name, parameter in appearance.named_parameters():
            if name in enhanced_names:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertIsNone(position.grad)
        self.assertIsNone(screenspace.grad)
        self.assertIsNone(opacity_raw.grad)
        self.assertIsNone(scale_raw.grad)
        self.assertIsNone(rotation.grad)

    def test_expected_depth_retains_geometry_gradient(self):
        device = torch.device("cuda")
        settings = GaussianRasterizationSettings(
            image_height=8,
            image_width=8,
            tanfovx=1.0,
            tanfovy=1.0,
            kernel_size=0.1,
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device),
            projmatrix=torch.eye(4, device=device),
            sh_degree=1,
            campos=torch.zeros(3, device=device),
            prefiltered=False,
            debug=False,
        )
        position = torch.tensor(
            [[0.0, 0.0, 0.8], [0.0, 0.0, 1.2]],
            device=device,
            requires_grad=True,
        )
        screenspace = torch.zeros_like(position, requires_grad=True)
        opacity = torch.full((2, 1), 0.6, device=device, requires_grad=True)
        scale = torch.full((2, 3), 0.2, device=device, requires_grad=True)
        rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            device=device,
            requires_grad=True,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        geometry = {
            "means3D": position,
            "means2D": screenspace,
            "shs": None,
            "opacities": opacity,
            "scales": scale,
            "rotations": rotation,
            "cov3D_precomp": None,
        }
        colors = torch.ones((2, 3), device=device)
        coverage, _, depth_accumulated = rasterizer(
            colors_precomp=colors,
            **geometry,
        )
        expected_depth = depth_accumulated / coverage.mean(
            dim=0, keepdim=True
        ).clamp_min(NUMERICAL_EPS)
        expected_depth.mean().backward()
        self.assertIsNotNone(position.grad)
        self.assertGreater(float(position.grad.abs().sum()), 0.0)

    def test_appearance_losses_never_reach_scaffold_geometry(self):
        """Every appearance-supervising loss owns only its appearance groups.

        Mirrors the render() gradient boundary: the R/L/enhanced diagnostics
        and the enhanced rasterization run on fully detached geometry, so no
        reflectance-sharpening or illumination loss may move Gaussian
        positions, opacities, scales, or rotations through the rasterizer.
        """
        device = torch.device("cuda")
        appearance = ExplicitAppearance(1).to(device)
        appearance.initialize(
            torch.tensor([[0.0, 0.0, 1.0]], device=device),
            torch.zeros((1, 1, 3), device=device),
            torch.ones((1, 6), device=device),
            cameras=[SimpleNamespace(
                original_image=torch.full((3, 5, 5), 0.5, device=device),
                enhancement_prior=torch.full((3, 5, 5), 0.5, device=device),
                full_proj_transform=torch.eye(4, device=device),
                camera_center=torch.tensor([0.0, 0.0, 0.0], device=device),
            )],
        )
        position = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
        screenspace = torch.zeros((1, 3), device=device, requires_grad=True)
        opacity = torch.full((1, 1), 0.8, device=device, requires_grad=True)
        scales = torch.full((1, 3), 0.2, device=device, requires_grad=True)
        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
        geometry_tensors = (position, screenspace, opacity, scales, rotation)

        evaluated = appearance.evaluate(torch.tensor([[0.0, 0.0, 1.0]], device=device))
        settings = GaussianRasterizationSettings(
            image_height=8,
            image_width=8,
            tanfovx=1.0,
            tanfovy=1.0,
            kernel_size=0.1,
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device),
            projmatrix=torch.eye(4, device=device),
            sh_degree=1,
            campos=torch.zeros(3, device=device),
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        reflectance = evaluated.reflectance.reshape(-1, 3)
        illumination = evaluated.illumination.reshape(-1, 1).expand(-1, 3)
        rendered, _, _ = rasterizer(
            means3D=position,
            means2D=screenspace,
            shs=None,
            colors_precomp=reflectance * illumination,
            opacities=opacity,
            scales=scales,
            rotations=rotation,
            cov3D_precomp=None,
        )
        coverage, _, _ = rasterizer(
            means3D=position,
            means2D=screenspace,
            shs=None,
            colors_precomp=torch.ones_like(reflectance),
            opacities=opacity,
            scales=scales,
            rotations=rotation,
            cov3D_precomp=None,
        )
        detached_geometry = {
            "means3D": position.detach(),
            "means2D": screenspace.detach(),
            "shs": None,
            "opacities": opacity.detach(),
            "scales": scales.detach(),
            "rotations": rotation.detach(),
            "cov3D_precomp": None,
        }
        detached_coverage = coverage.detach()
        reflectance_accumulated, _, _ = rasterizer(
            colors_precomp=reflectance, **detached_geometry
        )
        illumination_accumulated, _, _ = rasterizer(
            colors_precomp=illumination, **detached_geometry
        )
        enhanced_illumination_accumulated, _, _ = rasterizer(
            colors_precomp=evaluated.illumination_enhanced.reshape(-1, 3),
            **detached_geometry,
        )
        enhanced, _, _ = rasterizer(
            colors_precomp=evaluated.enhanced_color.reshape(-1, 3),
            **detached_geometry,
        )

        def covered(accumulated):
            return accumulated / detached_coverage.clamp_min(NUMERICAL_EPS)

        low_target = torch.full((3, 8, 8), 0.5, device=device)
        max_rgb_target = torch.full((1, 8, 8), 0.5, device=device)
        losses = {
            "reflectance_reconstruction": photo_loss(
                coverage_masked_prediction(
                    covered(reflectance_accumulated) * covered(illumination_accumulated).detach(),
                    low_target,
                    detached_coverage,
                ),
                low_target,
                0.2,
            ),
            "illumination_photo": photo_loss(
                coverage_masked_prediction(
                    covered(illumination_accumulated),
                    max_rgb_target.expand((3, 8, 8)),
                    detached_coverage,
                ),
                max_rgb_target.expand((3, 8, 8)),
                0.2,
            ),
            "illumination_edge_tv": edge_aware_illumination_tv_loss(
                covered(illumination_accumulated), max_rgb_target, detached_coverage
            ),
            "enhanced_photo": photo_loss(enhanced, low_target, 0.2),
            "enhanced_illumination_photo": photo_loss(
                coverage_masked_prediction(
                    covered(enhanced_illumination_accumulated),
                    low_target,
                    detached_coverage,
                ),
                low_target,
                0.2,
            ),
        }
        owning_groups = {
            "reflectance_reconstruction": ("reflectance_base", "reflectance_detail"),
            "illumination_photo": ("main_asg_",),
            "illumination_edge_tv": ("main_asg_",),
            "enhanced_photo": ("enhanced_",),
            "enhanced_illumination_photo": ("enhanced_",),
        }
        for loss_name, loss in losses.items():
            appearance.zero_grad(set_to_none=True)
            for tensor in geometry_tensors:
                tensor.grad = None
            loss.backward(retain_graph=True)
            allowed_prefixes = owning_groups[loss_name]
            owned_total = 0.0
            for name, parameter in appearance.named_parameters():
                gradient = parameter.grad
                magnitude = 0.0 if gradient is None else float(gradient.abs().sum())
                if any(name.startswith(prefix) for prefix in allowed_prefixes):
                    owned_total += magnitude
                else:
                    self.assertEqual(
                        magnitude,
                        0.0,
                        f"{loss_name} leaked gradient into appearance parameter {name}",
                    )
            self.assertGreater(
                owned_total,
                0.0,
                f"{loss_name} reached none of its owning appearance parameters",
            )
            for index, tensor in enumerate(geometry_tensors):
                self.assertTrue(
                    tensor.grad is None or float(tensor.grad.abs().sum()) == 0.0,
                    f"{loss_name} moved geometry tensor index {index} through the rasterizer",
                )
        del rendered, coverage


if __name__ == "__main__":
    unittest.main()
