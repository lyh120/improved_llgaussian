import unittest

import torch

from utils.scale_utils import apply_3d_filter_to_scales, gaussian_scale_regularization


class GaussianScaleRegularizationTest(unittest.TestCase):
    def test_3d_filter_does_not_move_offset_scales_by_default(self):
        scales = torch.tensor([[0.001, 0.002, 0.003, 0.004, 0.005, 0.006]])
        filtered = apply_3d_filter_to_scales(scales, torch.tensor([[0.01]]))

        self.assertTrue(torch.equal(filtered[:, :3], scales[:, :3]))
        self.assertTrue(torch.all(filtered[:, 3:] > scales[:, 3:]))

    def test_legacy_3d_filter_mode_modifies_all_scale_channels(self):
        scales = torch.full((1, 6), 0.002)
        filtered = apply_3d_filter_to_scales(
            scales,
            torch.tensor([[0.01]]),
            apply_to_offsets=True,
        )

        self.assertTrue(torch.all(filtered > scales))

    def test_safe_isotropic_scale_has_zero_loss(self):
        scaling = torch.full((4, 3), 0.004, requires_grad=True)

        max_loss, anisotropy_loss = gaussian_scale_regularization(
            scaling,
            voxel_size=0.0005,
            max_axis_voxel_ratio=16.0,
            max_anisotropy_ratio=10.0,
        )

        self.assertAlmostEqual(max_loss.item(), 0.0, places=7)
        self.assertAlmostEqual(anisotropy_loss.item(), 0.0, places=7)

    def test_oversized_needle_is_penalized_with_finite_gradients(self):
        scaling = torch.tensor(
            [[0.02, 0.001, 0.001]],
            dtype=torch.float32,
            requires_grad=True,
        )

        max_loss, anisotropy_loss = gaussian_scale_regularization(
            scaling,
            voxel_size=0.0005,
            max_axis_voxel_ratio=16.0,
            max_anisotropy_ratio=10.0,
        )
        loss = max_loss + anisotropy_loss
        loss.backward()

        self.assertGreater(max_loss.item(), 0.0)
        self.assertGreater(anisotropy_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(scaling.grad).all())

    def test_disabled_thresholds_preserve_differentiability(self):
        scaling = torch.full((2, 3), 0.1, requires_grad=True)

        max_loss, anisotropy_loss = gaussian_scale_regularization(
            scaling,
            voxel_size=0.0005,
            max_axis_voxel_ratio=0.0,
            max_anisotropy_ratio=0.0,
        )
        (max_loss + anisotropy_loss).backward()

        self.assertAlmostEqual(max_loss.item(), 0.0, places=7)
        self.assertAlmostEqual(anisotropy_loss.item(), 0.0, places=7)
        self.assertTrue(torch.isfinite(scaling.grad).all())


if __name__ == "__main__":
    unittest.main()
