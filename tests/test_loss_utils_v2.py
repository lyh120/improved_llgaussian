import unittest

import torch

from utils.loss_utils import (
    coverage_masked_prediction,
    edge_aware_illumination_tv_loss,
    edge_aware_tv_weights,
    inverse_depth_pearson_loss,
    photo_loss,
    retinex_targets,
    scaling_loss,
)


class LossUtilsV2Test(unittest.TestCase):
    def test_retinex_targets_are_finite_and_bounded_at_black_pixels(self):
        target = torch.tensor(
            [
                [[0.0, 1.0e-12], [0.2, 0.8]],
                [[0.0, 0.0], [0.4, 0.1]],
                [[0.0, 2.0e-12], [0.1, 0.3]],
            ]
        )
        max_rgb, reflectance = retinex_targets(target)
        self.assertTrue(torch.isfinite(max_rgb).all())
        self.assertTrue(torch.isfinite(reflectance).all())
        self.assertGreaterEqual(float(reflectance.min()), 0.0)
        self.assertLessEqual(float(reflectance.max()), 1.0)
        self.assertTrue(torch.equal(reflectance[:, 0, 0], torch.zeros(3)))

    def test_coverage_mask_removes_uncovered_background_supervision(self):
        target = torch.rand((3, 8, 8))
        coverage = torch.ones((1, 8, 8))
        coverage[:, :, 4:] = 0.0
        prediction_a = target.clone()
        prediction_b = target.clone()
        prediction_b[:, :, 4:] = 100.0
        masked_a = coverage_masked_prediction(prediction_a, target, coverage)
        masked_b = coverage_masked_prediction(prediction_b, target, coverage)
        self.assertTrue(torch.equal(masked_a, masked_b))
        self.assertEqual(float(photo_loss(masked_b, target, 0.2)), 0.0)

    def test_reflectance_reconstruction_detaches_illumination(self):
        reflectance = torch.rand((3, 12, 12), requires_grad=True)
        illumination = torch.rand((3, 12, 12), requires_grad=True)
        target = torch.rand((3, 12, 12))
        coverage = torch.ones((1, 12, 12))
        prediction = coverage_masked_prediction(
            reflectance * illumination.detach(), target, coverage
        )
        photo_loss(prediction, target, 0.2).backward()
        self.assertIsNotNone(reflectance.grad)
        self.assertIsNone(illumination.grad)

    def test_illumination_losses_only_update_illumination(self):
        illumination = torch.rand((3, 12, 12), requires_grad=True)
        unrelated = torch.rand((3, 12, 12), requires_grad=True)
        low_target = torch.rand((3, 12, 12))
        max_rgb, _ = retinex_targets(low_target)
        coverage = torch.ones((1, 12, 12))
        illumination_target = max_rgb.expand_as(illumination)
        masked = coverage_masked_prediction(illumination, illumination_target, coverage)
        loss = photo_loss(masked, illumination_target, 0.2)
        loss = loss + edge_aware_illumination_tv_loss(
            illumination, max_rgb, coverage
        )
        loss.backward()
        self.assertIsNotNone(illumination.grad)
        self.assertIsNone(unrelated.grad)

    def test_edge_tv_weight_is_lower_at_image_edges(self):
        max_rgb = torch.zeros((1, 5, 5))
        max_rgb[:, :, 3:] = 1.0
        coverage = torch.ones((1, 5, 5))
        horizontal, _ = edge_aware_tv_weights(max_rgb, coverage)
        self.assertTrue(torch.all(horizontal[..., 2] < horizontal[..., 0]))

    def test_edge_tv_is_normalized_and_finite_for_flat_guidance(self):
        illumination = torch.rand((3, 12, 12), requires_grad=True)
        target = torch.full((1, 12, 12), 0.5)
        coverage = torch.ones((1, 12, 12))
        loss = edge_aware_illumination_tv_loss(illumination, target, coverage)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertLessEqual(float(loss), 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(illumination.grad).all())

    def test_uncovered_tv_is_zero(self):
        loss = edge_aware_illumination_tv_loss(
            torch.rand((3, 8, 8)),
            torch.rand((1, 8, 8)),
            torch.zeros((1, 8, 8)),
        )
        self.assertEqual(float(loss), 0.0)

    def test_inverse_depth_pearson_is_normalized_and_has_finite_gradient(self):
        disparity = torch.linspace(0.1, 1.0, 64).reshape(1, 8, 8)
        depth = disparity.reciprocal().clone().requires_grad_(True)
        coverage = torch.ones((3, 8, 8))
        aligned = inverse_depth_pearson_loss(depth, disparity, coverage)
        anti_aligned = inverse_depth_pearson_loss(
            depth, 1.1 - disparity, coverage
        )
        self.assertLess(float(aligned), 0.01)
        self.assertGreater(float(anti_aligned), 0.99)
        aligned.backward()
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_depth_ignores_uncovered_pixels(self):
        depth = torch.rand((1, 8, 8), requires_grad=True) + 0.1
        prior = torch.rand((1, 8, 8))
        loss = inverse_depth_pearson_loss(
            depth, prior, torch.zeros((3, 8, 8))
        )
        self.assertEqual(float(loss), 0.0)

    def test_photo_and_scaling_losses_are_finite(self):
        prediction = torch.rand((3, 16, 16), requires_grad=True)
        target = torch.rand((3, 16, 16))
        loss = photo_loss(prediction, target, lambda_dssim=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(scaling_loss(torch.rand((5, 3)))))


if __name__ == "__main__":
    unittest.main()
