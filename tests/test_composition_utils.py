import unittest

import torch

from utils.composition_utils import compose_decomposed_render


class DecomposedCompositionTest(unittest.TestCase):
    def test_main_and_enhanced_use_composited_radiance(self):
        render_pkg = {
            "render_reflectance": torch.tensor([[[0.5, 2.0]]]),
            "render_illumination": torch.tensor([[[0.4, 0.8]]]),
            "render_illumination_enhanced": torch.tensor([[[0.9, 0.7]]]),
            "render": torch.tensor([[[0.4, 1.2]]]),
            "render_enhanced": torch.tensor([[[0.7, 1.5]]]),
        }

        main = compose_decomposed_render(render_pkg)
        enhanced = compose_decomposed_render(render_pkg, enhanced=True)

        torch.testing.assert_close(main, torch.tensor([[[0.4, 1.0]]]))
        torch.testing.assert_close(enhanced, torch.tensor([[[0.7, 1.0]]]))

    def test_component_only_legacy_package(self):
        package = {"render_reflectance": torch.tensor(0.5), "render_illumination": torch.tensor(0.2)}
        torch.testing.assert_close(compose_decomposed_render(package), torch.tensor(0.1))

    def test_partial_opacity_is_applied_once(self):
        opacity = torch.tensor(0.5, requires_grad=True)
        reflectance, illumination = torch.tensor(0.8), torch.tensor(0.2)
        package = {
            "render": opacity * reflectance * illumination,
            "render_reflectance": opacity * reflectance,
            "render_illumination": opacity * illumination,
        }
        result = compose_decomposed_render(package)
        torch.testing.assert_close(result, torch.tensor(0.08))
        result.backward()
        torch.testing.assert_close(opacity.grad, torch.tensor(0.16))


if __name__ == "__main__":
    unittest.main()
