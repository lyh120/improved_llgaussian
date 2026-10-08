import unittest

import torch

from utils.gradient_utils import (
    REFLECTANCE_PARAM_GROUP_NAMES,
    accumulate_auxiliary_gradients,
    rasterize_frozen_geometry,
)
from utils.loss_utils import (
    L_Reflectance_Edge_Uplift,
    L_Reflectance_HighFreq,
    L_Reflectance_LocalContrast,
)


class GradientRoutingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_each_sharpening_loss_updates_appearance_only(self):
        for loss_function in (L_Reflectance_Edge_Uplift, L_Reflectance_LocalContrast, L_Reflectance_HighFreq):
            for mode in ("explicit", "mlp"):
                with self.subTest(loss=loss_function.__name__, mode=mode):
                    shared = torch.nn.Parameter(torch.rand(3, 8, 8))
                    geometry = {key: torch.nn.Parameter(torch.tensor(0.4)) for key in (
                        "means3D", "means2D", "opacities", "scales", "rotations",
                    )}
                    screen = geometry["means2D"] + 0
                    screen.retain_grad()
                    parameters = [torch.nn.Parameter(torch.zeros(3, 8, 8)) for _ in range(3)]
                    names = ("base_log_reflectance", "reflectance_offset_delta", "mlp_reflectance_decoder") if mode == "explicit" else ("mlp_reflectance",)
                    groups = [{"params": [parameter], "name": name} for parameter, name in zip(parameters, names)]
                    optimizer = torch.optim.SGD(groups + [{"params": [shared, *geometry.values()], "name": "geometry"}], lr=0.1)
                    appearance = 0.3 + 0.02 * shared.detach() + sum(parameters[:len(names)])

                    def differentiable_rasterizer(colors_precomp, **inputs):
                        return colors_precomp * sum(inputs.values()), None, None

                    image, _, _ = rasterize_frozen_geometry(
                        differentiable_rasterizer, appearance, **{**geometry, "means2D": screen},
                    )
                    target = torch.zeros(3, 8, 8)
                    target[:, :, 4:] = 0.6
                    auxiliary = loss_function(image, target)
                    accumulate_auxiliary_gradients(auxiliary, optimizer, REFLECTANCE_PARAM_GROUP_NAMES)
                    self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in parameters[:len(names)]), 0)
                    self.assertIsNone(shared.grad)
                    self.assertIsNone(screen.grad)
                    for parameter in geometry.values():
                        self.assertIsNone(parameter.grad)
                    # Reconstruction retains exactly its original geometry gradient.
                    main_image, _, _ = differentiable_rasterizer(appearance, **{**geometry, "means2D": screen})
                    main_loss = (main_image - target).square().mean()
                    expected = torch.autograd.grad(main_loss, geometry["scales"], retain_graph=True)[0]
                    main_loss.backward()
                    torch.testing.assert_close(geometry["scales"].grad, expected)
                    self.assertIsNotNone(screen.grad)

    def test_auxiliary_accumulates_without_overwriting_main_gradient(self):
        reflectance = torch.nn.Parameter(torch.tensor(0.3))
        other = torch.nn.Parameter(torch.tensor(0.4))
        unused = torch.nn.Parameter(torch.tensor(0.5))
        optimizer = torch.optim.SGD([
            {"params": [reflectance, unused], "name": "mlp_reflectance"},
            {"params": [other], "name": "geometry"},
        ], lr=0.1)
        reflectance.grad = torch.tensor(2.0)
        auxiliary = reflectance.square() + other.square()
        accumulate_auxiliary_gradients(auxiliary, optimizer, REFLECTANCE_PARAM_GROUP_NAMES)
        torch.testing.assert_close(reflectance.grad, torch.tensor(2.6))
        self.assertIsNone(other.grad)
        self.assertIsNone(unused.grad)
        (reflectance * other).backward()
        torch.testing.assert_close(reflectance.grad, torch.tensor(3.0))
        torch.testing.assert_close(other.grad, torch.tensor(0.3))

    def test_disabled_or_frozen_auxiliary_is_safe(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        optimizer = torch.optim.SGD([{"params": [parameter], "name": "mlp_reflectance"}], lr=0.1)
        accumulate_auxiliary_gradients(torch.tensor(0.0), optimizer, REFLECTANCE_PARAM_GROUP_NAMES)
        self.assertIsNone(parameter.grad)

    def test_enhancement_guidance_cannot_update_base_components(self):
        reflectance = torch.nn.Parameter(torch.tensor(0.4))
        illumination = torch.nn.Parameter(torch.tensor(0.2))
        enhancement = torch.nn.Parameter(torch.tensor(0.5))
        geometry = torch.nn.Parameter(torch.tensor(0.7))
        optimizer = torch.optim.SGD([
            {"params": [reflectance], "name": "base_log_reflectance"},
            {"params": [illumination], "name": "mlp_sg_illumination"},
            {"params": [enhancement], "name": "enhancement_sg_amplitude"},
            {"params": [geometry], "name": "geometry"},
        ], lr=0.1)

        def rasterizer(colors_precomp, opacities):
            return colors_precomp * opacities, None, None

        image, _, _ = rasterize_frozen_geometry(rasterizer, reflectance.detach() * enhancement,
                                                opacities=geometry)
        accumulate_auxiliary_gradients((image - 1).square(), optimizer, {"enhancement_sg_amplitude"})
        self.assertGreater(enhancement.grad.abs().item(), 0)
        for parameter in (reflectance, illumination, geometry):
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
