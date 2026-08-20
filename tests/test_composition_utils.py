import unittest

import torch

from utils.composition_utils import compose_decomposed_render


class DecomposedCompositionTest(unittest.TestCase):
    def test_default_composition_keeps_both_gradients(self):
        reflectance = torch.tensor([[[0.5, 0.25]]], requires_grad=True)
        illumination = torch.tensor([[[0.4, 0.8]]], requires_grad=True)
        render_pkg = {
            "render_reflectance": reflectance,
            "render_illumination": illumination,
        }

        compose_decomposed_render(render_pkg).sum().backward()

        self.assertIsNotNone(reflectance.grad)
        self.assertIsNotNone(illumination.grad)
        self.assertGreater(torch.count_nonzero(reflectance.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(illumination.grad).item(), 0)

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

    def test_detached_reflectance_preserves_values_and_blocks_its_gradient(self):
        reflectance = torch.tensor([[[0.5, 0.25]]], requires_grad=True)
        illumination = torch.tensor([[[0.9, 0.7]]], requires_grad=True)
        render_pkg = {
            "render_reflectance": reflectance,
            "render_illumination_enhanced": illumination,
        }

        regular = compose_decomposed_render(render_pkg, enhanced=True)
        detached = compose_decomposed_render(
            render_pkg,
            enhanced=True,
            detach_reflectance=True,
        )
        torch.testing.assert_close(detached, regular)

        detached.sum().backward()
        self.assertIsNone(reflectance.grad)
        self.assertIsNotNone(illumination.grad)
        self.assertGreater(torch.count_nonzero(illumination.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
