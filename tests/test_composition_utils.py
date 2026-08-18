import unittest

import torch

from utils.composition_utils import compose_decomposed_render


class DecomposedCompositionTest(unittest.TestCase):
    def test_main_and_enhanced_use_image_space_products(self):
        render_pkg = {
            "render_reflectance": torch.tensor([[[0.5, 2.0]]]),
            "render_illumination": torch.tensor([[[0.4, 0.8]]]),
            "render_illumination_enhanced": torch.tensor([[[0.9, 0.7]]]),
            "render": torch.zeros(1, 1, 2),
            "render_enhanced": torch.zeros(1, 1, 2),
        }

        main = compose_decomposed_render(render_pkg)
        enhanced = compose_decomposed_render(render_pkg, enhanced=True)

        torch.testing.assert_close(main, torch.tensor([[[0.2, 1.0]]]))
        torch.testing.assert_close(enhanced, torch.tensor([[[0.45, 1.0]]]))

    def test_main_composition_updates_reflectance_and_illumination(self):
        reflectance = torch.tensor([[[0.5]]], requires_grad=True)
        illumination = torch.tensor([[[0.4]]], requires_grad=True)
        render_pkg = {
            "render_reflectance": reflectance,
            "render_illumination": illumination,
            "render_illumination_enhanced": illumination,
        }

        compose_decomposed_render(render_pkg).sum().backward()

        self.assertIsNotNone(reflectance.grad)
        self.assertIsNotNone(illumination.grad)
        self.assertGreater(reflectance.grad.abs().sum().item(), 0.0)
        self.assertGreater(illumination.grad.abs().sum().item(), 0.0)

    def test_enhanced_composition_can_freeze_reflectance(self):
        reflectance = torch.tensor([[[0.5]]], requires_grad=True)
        illumination_enhanced = torch.tensor([[[0.9]]], requires_grad=True)
        render_pkg = {
            "render_reflectance": reflectance,
            "render_illumination": torch.tensor([[[0.4]]]),
            "render_illumination_enhanced": illumination_enhanced,
        }

        enhanced = compose_decomposed_render(
            render_pkg,
            enhanced=True,
            detach_reflectance=True,
        )
        enhanced.sum().backward()

        self.assertIsNone(reflectance.grad)
        self.assertIsNotNone(illumination_enhanced.grad)
        self.assertGreater(illumination_enhanced.grad.abs().sum().item(), 0.0)
        torch.testing.assert_close(enhanced, torch.tensor([[[0.45]]]))


if __name__ == "__main__":
    unittest.main()
