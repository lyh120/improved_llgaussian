import unittest

import torch

from utils.filter_utils import filter_gaussian_covariance


class GaussianFilterTest(unittest.TestCase):
    def test_support_bound_precedes_antialiasing_and_remains_finite(self):
        scales = torch.tensor([[100., 0.1, 0.2]], requires_grad=True)
        footprint = torch.tensor([[0.05]])
        opacity = torch.tensor([[0.7]], requires_grad=True)
        filtered, adjusted = filter_gaussian_covariance(scales, opacity, footprint, 16.0)
        bounded = torch.minimum(scales, footprint * 16)
        self.assertTrue((filtered <= footprint * (16 ** 2 + 1) ** 0.5 + 1e-6).all())
        torch.testing.assert_close(adjusted * filtered.prod(dim=-1, keepdim=True), opacity * bounded.prod(dim=-1, keepdim=True))
        (filtered.sum() + adjusted.sum()).backward()
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertEqual(scales.grad[0, 0].item(), 0.0)

    def test_degenerate_scales_have_finite_gradients(self):
        scales = torch.tensor([[0.0, 0.02, 0.03], [1e-10, 1e-10, 1e-10]], requires_grad=True)
        opacity = torch.ones(2, 1, requires_grad=True)
        footprint = torch.full((2, 1), 0.05)
        filtered, adjusted = filter_gaussian_covariance(scales, opacity, footprint)
        (filtered.sum() + adjusted.sum()).backward()
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue(torch.isfinite(opacity.grad).all())
        torch.testing.assert_close(adjusted * filtered.prod(dim=-1, keepdim=True), opacity * scales.prod(dim=-1, keepdim=True))

    def test_filter_preserves_integrated_opacity(self):
        scales = torch.tensor([[0.01, 0.02, 0.03], [0.1, 0.2, 0.3]], requires_grad=True)
        opacity = torch.tensor([[0.6], [0.8]], requires_grad=True)
        footprint = torch.tensor([[0.05], [0.02]])
        filtered, adjusted = filter_gaussian_covariance(scales, opacity, footprint)
        self.assertTrue((filtered >= footprint).all())
        torch.testing.assert_close(adjusted * filtered.prod(dim=-1, keepdim=True), opacity * scales.prod(dim=-1, keepdim=True))
        (filtered.sum() + adjusted.sum()).backward()
        self.assertTrue(torch.isfinite(scales.grad).all())

    def test_zero_footprint_leaves_gaussian_unchanged(self):
        scales = torch.tensor([[0.1, 0.2, 0.3]])
        opacity = torch.tensor([[0.5]])
        filtered, adjusted = filter_gaussian_covariance(scales, opacity, torch.zeros(1, 1))
        torch.testing.assert_close(filtered, scales)
        torch.testing.assert_close(adjusted, opacity)

    def test_distinct_anchor_footprints_follow_offset_order(self):
        footprint = torch.tensor([[0.02], [0.2]]).repeat_interleave(2, dim=0)
        filtered, _ = filter_gaussian_covariance(torch.full((4, 3), 0.01), torch.ones(4, 1), footprint)
        torch.testing.assert_close(filtered[0], filtered[1])
        torch.testing.assert_close(filtered[2], filtered[3])
        self.assertTrue((filtered[2] > filtered[0]).all())


if __name__ == "__main__":
    unittest.main()
