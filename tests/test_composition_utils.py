import unittest

import torch

from utils.composition_utils import compose_decomposed_render


class DecomposedCompositionTest(unittest.TestCase):
    def test_main_and_enhanced_use_image_space_products(self):
        render_pkg = {
            "render_reflectance": torch.tensor([[[0.5, 2.0]]]),
            "render_illumination": torch.tensor([[[0.4, 0.8]]]),
            "render_illumination_enhanced": torch.tensor([[[0.9, 0.7]]]),
        }

        main = compose_decomposed_render(render_pkg)
        enhanced = compose_decomposed_render(render_pkg, enhanced=True)

        torch.testing.assert_close(main, torch.tensor([[[0.2, 1.0]]]))
        torch.testing.assert_close(enhanced, torch.tensor([[[0.45, 1.0]]]))

    def test_enhancement_can_detach_reflectance_gradient(self):
        reflectance = torch.tensor([[[0.5, 0.4]]], requires_grad=True)
        illumination_enhanced = torch.tensor([[[0.6, 0.7]]], requires_grad=True)
        render_pkg = {
            "render_reflectance": reflectance,
            "render_illumination": torch.ones_like(reflectance),
            "render_illumination_enhanced": illumination_enhanced,
        }

        enhanced = compose_decomposed_render(
            render_pkg,
            enhanced=True,
            detach_reflectance=True,
        )
        enhanced.sum().backward()

        self.assertIsNone(reflectance.grad)
        torch.testing.assert_close(
            illumination_enhanced.grad,
            reflectance.detach(),
        )

    def test_enhancement_can_detach_illumination_gradient(self):
        reflectance = torch.tensor([[[0.5, 0.4]]], requires_grad=True)
        illumination_enhanced = torch.tensor([[[0.6, 0.7]]], requires_grad=True)
        render_pkg = {
            "render_reflectance": reflectance,
            "render_illumination": torch.ones_like(reflectance),
            "render_illumination_enhanced": illumination_enhanced,
        }

        enhanced = compose_decomposed_render(
            render_pkg,
            enhanced=True,
            detach_illumination=True,
        )
        enhanced.sum().backward()

        self.assertIsNone(illumination_enhanced.grad)
        torch.testing.assert_close(
            reflectance.grad,
            illumination_enhanced.detach(),
        )

    def test_reflectance_key_selects_sharpen_branch(self):
        render_pkg = {
            "render_reflectance": torch.tensor([[[0.5, 0.4]]]),
            "render_reflectance_sharpen": torch.tensor([[[0.8, 0.1]]]),
            "render_illumination": torch.tensor([[[1.0, 1.0]]]),
            "render_illumination_enhanced": torch.tensor([[[1.0, 1.0]]]),
        }

        sharpened = compose_decomposed_render(
            render_pkg,
            enhanced=True,
            reflectance_key="render_reflectance_sharpen",
            detach_illumination=True,
        )

        torch.testing.assert_close(sharpened, torch.tensor([[[0.8, 0.1]]]))


if __name__ == "__main__":
    unittest.main()
