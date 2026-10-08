import unittest
from types import SimpleNamespace

import torch

from utils.graphics_utils import getProjectionMatrix
from utils.reflectance_init_utils import estimate_initial_log_reflectance


class ReflectanceInitializationTest(unittest.TestCase):
    def camera(self, image, translation=None):
        world_view = torch.eye(4)
        if translation is not None:
            world_view[3, :3] = torch.tensor(translation)
        projection = getProjectionMatrix(0.01, 100, 1.5, 1.5).T
        return SimpleNamespace(original_image=image, full_proj_transform=world_view @ projection)

    def test_dark_scene_initialization_is_exposure_invariant(self):
        image = torch.tensor([0.08, 0.04, 0.02]).view(3, 1, 1).expand(3, 5, 5)
        anchors = torch.tensor([[0.0, 0.0, 2.0]])
        bright = estimate_initial_log_reflectance(anchors, [self.camera(image)])
        dark = estimate_initial_log_reflectance(anchors, [self.camera(image * 0.1)])
        torch.testing.assert_close(bright, dark)
        torch.testing.assert_close(bright.exp(), torch.tensor([[1.0, 0.5, 0.25]]))

    def test_outside_frustum_behind_camera_and_black_signal_are_neutral(self):
        image = torch.full((3, 5, 5), 0.1)
        image[0] = 0.2
        anchors = torch.tensor([[100.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
        result = estimate_initial_log_reflectance(anchors, [self.camera(image)])
        torch.testing.assert_close(result, torch.zeros_like(anchors))
        black = estimate_initial_log_reflectance(torch.tensor([[0., 0., 2.]]), [self.camera(torch.zeros_like(image))])
        torch.testing.assert_close(black, torch.zeros_like(black))

    def test_camera_translation_and_image_vertical_direction(self):
        image = torch.ones(3, 5, 5)
        image[1, 3:, :] = 0.5
        camera = self.camera(image, [0.0, 0.0, -1.0])
        result = estimate_initial_log_reflectance(torch.tensor([[0.0, 0.6, 3.0]]), [camera])
        torch.testing.assert_close(result.exp(), torch.tensor([[1.0, 0.5, 1.0]]))

    def test_multiple_views_and_missing_views(self):
        anchors = torch.tensor([[0.0, 0.0, 2.0]])
        red = torch.tensor([1.0, 0.5, 0.5]).view(3, 1, 1)
        blue = torch.tensor([0.5, 0.5, 1.0]).view(3, 1, 1)
        result = estimate_initial_log_reflectance(anchors, [self.camera(red), self.camera(blue)])
        torch.testing.assert_close(result, (red.flatten().log() + blue.flatten().log()).view(1, 3) / 2)
        torch.testing.assert_close(estimate_initial_log_reflectance(anchors, None), torch.zeros_like(anchors))


if __name__ == "__main__":
    unittest.main()
