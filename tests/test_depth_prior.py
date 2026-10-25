import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from model_fixture import SoftRasterizer
from utils.depth_utils import (
    generate_depth_priors,
    local_depth_similarity,
    pearson_depth_loss,
    relative_depth_supervision,
)
from utils.rasterizer_utils import rasterize_training_depth


class DepthPriorTest(unittest.TestCase):
    def test_small_and_exact_window_sizes_have_finite_gradients(self):
        for height, width, window in ((3, 5, 128), (8, 8, 8), (8, 16, 8)):
            with self.subTest(size=(height, width)):
                source = torch.rand(height, width, requires_grad=True)
                target = torch.rand(height, width, requires_grad=True)
                loss = local_depth_similarity(source, target, window)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(source.grad).all())
                self.assertIsNone(target.grad)

    def test_degenerate_and_invalid_windows_are_differentiable_zero(self):
        for kind in ("constant", "invalid", "empty_mask"):
            source = torch.ones(3, 3, requires_grad=True)
            if kind == "invalid":
                source = torch.full((3, 3), float("nan"), requires_grad=True)
            mask = torch.zeros(3, 3, dtype=torch.bool) if kind == "empty_mask" else None
            loss = local_depth_similarity(source, torch.rand(3, 3), valid_mask=mask)
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertTrue(torch.isfinite(source.grad).all())

    def test_anticorrelation_is_not_clipped_to_a_flat_unit_loss(self):
        source = torch.tensor([0.8, 0.6, 0.3, 0.2], requires_grad=True)
        target = torch.tensor([0.0, 0.4, 0.6, 1.0])
        loss = pearson_depth_loss(source, target)
        loss.backward()
        self.assertGreater(loss.item(), 1.0)
        self.assertGreater(source.grad.abs().sum().item(), 0)

    def test_relative_supervision_ignores_uncovered_and_nonfinite_pixels(self):
        depth = torch.tensor([[[0.2, 0.5, float("nan")], [0.6, 0.7, 100.0]]], requires_grad=True)
        prior = torch.tensor([[[1.0, 0.4, float("nan")], [0.3, 0.0, 1000.0]]])
        coverage = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]])
        loss = relative_depth_supervision(depth, prior, coverage)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertEqual(depth.grad[0, 1, 2].item(), 0)

    def test_generator_accepts_numpy_tensor_and_configured_input_directory(self):
        class DepthBackend(torch.nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.tensor = tensor

            def infer_image(self, image):
                self.assertions = (not self.training, not self.weight.requires_grad,
                                   not torch.is_grad_enabled(), image[0, 0].tolist() == [0, 0, 255])
                output = np.arange(8, dtype=np.float32).reshape(2, 4)
                return torch.from_numpy(output)[None, None] if self.tensor else output

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "low_frames"
            directory.mkdir()
            Image.new("RGB", (4, 2), (255, 0, 0)).save(directory / "frame.PNG")
            camera = SimpleNamespace(uid=8, image_name="frame", image_height=6, image_width=8,
                                     original_image=torch.zeros(3, 6, 8))
            for tensor in (False, True):
                backend = DepthBackend(tensor)
                result = generate_depth_priors(backend, [camera], temp, "low_frames")[8]
                self.assertEqual(result.shape, (1, 6, 8))
                self.assertTrue(all(backend.assertions))
                self.assertFalse(result.requires_grad)
                self.assertTrue(torch.isfinite(result).all())
                self.assertAlmostEqual(result.min().item(), 0.0)
                self.assertAlmostEqual(result.max().item(), 1.0)

    def test_depth_color_path_matches_finite_difference_and_keeps_screen_gradient(self):
        points = torch.tensor([[-0.3, 0.2, 1.0], [0.3, -0.2, 1.7]], dtype=torch.float64, requires_grad=True)
        screen = torch.zeros_like(points, requires_grad=True)
        kwargs = dict(means2D=screen, opacities=torch.full((2, 1), 0.5, dtype=torch.float64),
                      scales=torch.full((2, 3), 0.2, dtype=torch.float64),
                      rotations=torch.zeros(2, 4, dtype=torch.float64))

        def objective(position):
            depth, coverage = rasterize_training_depth(SoftRasterizer(), means3D=position, **kwargs)
            self.assertFalse(coverage.requires_grad)
            return depth.square().mean()

        value = objective(points)
        value.backward()
        self.assertGreater(screen.grad.abs().sum().item(), 0)
        delta = 1e-5
        for index in range(2):
            positive, negative = points.detach().clone(), points.detach().clone()
            positive[index, 2] += delta
            negative[index, 2] -= delta
            numeric = (objective(positive) - objective(negative)) / (2 * delta)
            self.assertAlmostEqual(points.grad[index, 2].item(), numeric.item(), places=7)


if __name__ == "__main__":
    unittest.main()
