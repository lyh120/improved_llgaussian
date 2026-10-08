"""Verify the paper's affine direction, chroma preservation and degenerate case."""

import unittest

import cv2
import numpy as np

from utils.evaluation_utils import align_lab_luminance


class LuminanceAlignmentTest(unittest.TestCase):
    def test_inverts_gt_to_prediction_fit(self):
        lab = np.zeros((10, 12, 3), dtype=np.float32)
        lab[..., 0] = np.linspace(20, 65, 120).reshape(10, 12)
        target = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        prediction_lab = lab.copy()
        prediction_lab[..., 0] = 0.65 * lab[..., 0] + 12
        prediction = cv2.cvtColor(prediction_lab, cv2.COLOR_LAB2RGB)
        aligned, fit = align_lab_luminance(prediction, target)
        self.assertAlmostEqual(fit["a"], 0.65, delta=0.003)
        self.assertAlmostEqual(fit["b"], 12, delta=0.12)
        self.assertFalse(fit["degenerate_identity_fallback"])
        np.testing.assert_allclose(aligned, target, atol=0.004)
        aligned_lab = cv2.cvtColor(aligned, cv2.COLOR_RGB2LAB)
        pred_lab = cv2.cvtColor(prediction, cv2.COLOR_RGB2LAB)
        # OpenCV's RGB/LAB lookup incurs sub-unit chroma round-trip error.
        np.testing.assert_allclose(aligned_lab[..., 1:], pred_lab[..., 1:], atol=0.35)

    def test_flat_target_is_finite_and_reported(self):
        image = np.full((6, 7, 3), 0.2, dtype=np.float32)
        aligned, fit = align_lab_luminance(image, image)
        self.assertTrue(fit["degenerate_identity_fallback"])
        self.assertTrue(np.isfinite(aligned).all())

    def test_size_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            align_lab_luminance(np.zeros((2, 2, 3)), np.zeros((3, 3, 3)))


if __name__ == "__main__":
    unittest.main()
