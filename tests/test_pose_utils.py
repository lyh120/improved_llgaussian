import unittest

import torch

from utils.pose_utils import get_camera_center_from_tensor


class PoseUtilsTest(unittest.TestCase):
    def test_camera_center_comes_from_the_effective_world_to_camera_pose(self):
        pose = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, -2.0, 3.0],
            requires_grad=True,
        )
        center = get_camera_center_from_tensor(pose)
        self.assertTrue(torch.allclose(center, torch.tensor([-1.0, 2.0, -3.0])))
        center.square().sum().backward()
        self.assertIsNotNone(pose.grad)
        self.assertTrue(torch.isfinite(pose.grad).all())


if __name__ == "__main__":
    unittest.main()
