import unittest

import torch

from utils.gradient_isolation import rasterize_with_frozen_geometry


class _FakeRasterizer:
    def __call__(self, **kwargs):
        value = kwargs["colors_precomp"].sum()
        for key in (
            "means3D",
            "means2D",
            "opacities",
            "scales",
            "rotations",
        ):
            value = value + kwargs[key].sum()
        return value, torch.zeros(1, device=value.device), torch.zeros(1, device=value.device)


class FrozenGeometryRasterizationTest(unittest.TestCase):
    def _run_gradient_check(self, device):
        inputs = {
            name: torch.ones(2, device=device, requires_grad=True)
            for name in (
                "means3D",
                "means2D",
                "colors_precomp",
                "opacities",
                "scales",
                "rotations",
            )
        }

        rendered, _, _ = rasterize_with_frozen_geometry(
            _FakeRasterizer(),
            **inputs,
        )
        rendered.backward()

        self.assertIsNotNone(inputs["colors_precomp"].grad)
        self.assertGreater(inputs["colors_precomp"].grad.abs().sum().item(), 0.0)
        for name in (
            "means3D",
            "means2D",
            "opacities",
            "scales",
            "rotations",
        ):
            self.assertIsNone(inputs[name].grad, name)

    def test_only_colors_receive_gradients_on_cpu(self):
        self._run_gradient_check(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_only_colors_receive_gradients_on_cuda(self):
        self._run_gradient_check(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
