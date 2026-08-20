import unittest

import torch
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError:
    GaussianRasterizationSettings = None
    GaussianRasterizer = None


CUDA_RASTERIZER_AVAILABLE = (
    torch.cuda.is_available()
    and GaussianRasterizationSettings is not None
    and GaussianRasterizer is not None
)


@unittest.skipUnless(
    CUDA_RASTERIZER_AVAILABLE,
    "CUDA diff-gaussian-rasterization is not available",
)
class RasterGradientIsolationTest(unittest.TestCase):
    def test_auxiliary_raster_only_updates_reflectance_expression(self):
        device = torch.device("cuda")
        position = torch.tensor(
            [[0.0, 0.0, 1.0]], device=device, requires_grad=True
        )
        screenspace = torch.zeros_like(position, requires_grad=True)
        offset = torch.zeros((1, 3), device=device, requires_grad=True)
        feature = torch.zeros((1, 3), device=device, requires_grad=True)
        base = torch.zeros((1, 3), device=device, requires_grad=True)
        detail = torch.zeros((1, 3), device=device, requires_grad=True)

        decoder = torch.nn.Linear(3, 3, bias=True, device=device)
        opacity_mlp = torch.nn.Linear(1, 1, bias=True, device=device)
        covariance_mlp = torch.nn.Linear(1, 7, bias=True, device=device)
        network_input = torch.ones((1, 1), device=device)

        decoder_refine = 1.0 + 0.15 * torch.tanh(
            decoder(feature.detach() + offset.detach())
        )
        reflectance = torch.exp(base + detail) * decoder_refine
        opacity = torch.sigmoid(opacity_mlp(network_input))
        covariance = covariance_mlp(network_input)
        scaling = 0.01 + 0.05 * torch.sigmoid(covariance[:, :3])
        rotation_seed = covariance[:, 3:7] + covariance.new_tensor(
            [[1.0, 0.0, 0.0, 0.0]]
        )
        rotation = F.normalize(rotation_seed, dim=-1)

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
        image, _, _ = rasterizer(
            means3D=position.detach(),
            means2D=screenspace.detach(),
            shs=None,
            colors_precomp=reflectance,
            opacities=opacity.detach(),
            scales=scaling.detach(),
            rotations=rotation.detach(),
            cov3D_precomp=None,
        )
        image.sum().backward()

        self.assertIsNotNone(base.grad)
        self.assertIsNotNone(detail.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in decoder.parameters()))
        self.assertIsNone(position.grad)
        self.assertIsNone(screenspace.grad)
        self.assertIsNone(offset.grad)
        self.assertIsNone(feature.grad)
        self.assertTrue(all(parameter.grad is None for parameter in opacity_mlp.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in covariance_mlp.parameters()))


if __name__ == "__main__":
    unittest.main()
