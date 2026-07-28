import unittest

import torch

from utils.enhancement_loss_utils import (
    L_Enhancement_Edge_Preserve,
    L_Enhancement_Gain_Smooth,
)


class EnhancementDetailLossTest(unittest.TestCase):
    def test_uniform_gain_has_zero_smoothness(self):
        base = torch.full((3, 8, 8), 0.2, requires_grad=True)
        enhanced = torch.full((3, 8, 8), 0.4, requires_grad=True)

        loss = L_Enhancement_Gain_Smooth(enhanced, base)

        self.assertLess(loss.item(), 1e-6)
        loss.backward()
        self.assertIsNone(base.grad)

    def test_spatial_gain_change_is_penalized(self):
        base = torch.full((3, 8, 8), 0.2)
        enhanced = torch.full((3, 8, 8), 0.4)
        enhanced[:, :, 4:] = 0.8

        loss = L_Enhancement_Gain_Smooth(enhanced, base)

        self.assertGreater(loss.item(), 0.0)

    def test_edge_preservation_penalizes_missing_teacher_edge(self):
        base = torch.full((3, 8, 8), 0.1)
        base[:, :, 4:] = 0.4
        matching_enhanced = torch.clamp(base * 2.0, 0.0, 1.0)
        missing_enhanced = torch.full_like(base, 0.5)

        matching_loss = L_Enhancement_Edge_Preserve(
            matching_enhanced,
            base,
            enhance_ratio=2,
        )
        missing_loss = L_Enhancement_Edge_Preserve(
            missing_enhanced,
            base,
            enhance_ratio=2,
        )

        self.assertLess(matching_loss.item(), 1e-4)
        self.assertGreater(missing_loss.item(), matching_loss.item())


if __name__ == "__main__":
    unittest.main()
