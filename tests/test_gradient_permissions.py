import unittest

import torch

from utils.composition_utils import compose_decomposed_render
from utils.gradient_utils import accumulate_auxiliary_gradients, parameters_in_groups
from utils.sg_utils import (
    _orthonormalize_tangent, asg_parameter_regularization,
    evaluate_anisotropic_spherical_gaussians, normalize_asg_axes,
)


class GradientPermissionsTest(unittest.TestCase):
    def test_auxiliary_does_not_accumulate_geometry_or_screen(self):
        material = torch.tensor(0.3, requires_grad=True)
        geometry = torch.tensor(0.7, requires_grad=True)
        screen = torch.tensor(0.1, requires_grad=True)
        image = material * geometry + screen
        stats = accumulate_auxiliary_gradients(
            {"edge": (image.square(), 0.25)}, [material], audit=True,
        )
        self.assertGreater(stats["edge/grad_norm"], 0)
        self.assertIsNone(geometry.grad)
        self.assertIsNone(screen.grad)
        aux_gradient = material.grad.clone()
        image.backward()
        torch.testing.assert_close(material.grad, aux_gradient + geometry.detach())
        self.assertGreater(geometry.grad.item(), 0)
        self.assertGreater(screen.grad.item(), 0)

    def test_zero_weight_leaves_main_gradient_unchanged(self):
        parameter = torch.tensor(0.7, requires_grad=True)
        main = parameter.square()
        accumulate_auxiliary_gradients({"off": (main * 100, 0)}, [parameter], audit=True)
        self.assertIsNone(parameter.grad)
        main.backward()
        torch.testing.assert_close(parameter.grad, torch.tensor(1.4))

    def test_whitelist_follows_replaced_densification_parameters(self):
        old = torch.nn.Parameter(torch.tensor([0.3]))
        optimizer = torch.optim.Adam([{"params": [old], "name": "base_log_reflectance"}])
        grown = torch.nn.Parameter(torch.tensor([0.3, 0.4]))
        optimizer.param_groups[0]["params"] = [grown]
        current = parameters_in_groups(optimizer, {"base_log_reflectance"})
        self.assertIs(current[0], grown)
        accumulate_auxiliary_gradients({"detail": (grown.square().mean(), 1)}, current)
        self.assertIsNone(old.grad)
        self.assertTrue((grown.grad != 0).all())

    def test_detached_and_wrong_whitelist_are_detected(self):
        parameter = torch.tensor(0.7, requires_grad=True)
        unrelated = torch.tensor(0.3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "Detached"):
            accumulate_auxiliary_gradients({"bad": (parameter.detach(), 1)}, [parameter])
        with self.assertRaisesRegex(RuntimeError, "whitelisted"):
            accumulate_auxiliary_gradients({"bad": (unrelated.square(), 1)}, [parameter], audit=True)

    def test_connected_zero_derivative_is_inactive(self):
        parameter = torch.tensor(0.0, requires_grad=True)
        audit = accumulate_auxiliary_gradients({"detail": (parameter.square(), 1)}, [parameter], audit=True)
        self.assertEqual(audit["detail/inactive"], 1)

    def test_enhancement_teacher_keeps_main_reflectance_trainable(self):
        reflectance = torch.tensor([[[0.4]]], requires_grad=True)
        light = torch.tensor([[[0.2]]], requires_grad=True)
        enhanced = torch.tensor([[[0.8]]], requires_grad=True)
        package = {"render_reflectance": reflectance, "render_illumination": light,
                   "render_illumination_enhanced": enhanced}
        compose_decomposed_render(package, enhanced=True, detach_reflectance=True).sum().backward()
        self.assertIsNone(reflectance.grad)
        self.assertIsNone(light.grad)
        self.assertIsNotNone(enhanced.grad)
        compose_decomposed_render(package).sum().backward()
        self.assertIsNotNone(reflectance.grad)
        self.assertIsNotNone(light.grad)

    def test_all_lobes_get_regularizer_gradients(self):
        sharpness = torch.tensor([[[[1.0, 3.0]]], [[[2.0, 4.0]]]], requires_grad=True)
        amplitude = torch.zeros(2, 1, 1, 1, requires_grad=True)
        stats = asg_parameter_regularization(sharpness, amplitude, 1.0)
        (stats["asg_energy"] + stats["asg_lambda_mean"] + stats["asg_anisotropy"]).backward()
        self.assertTrue((amplitude.grad != 0).all())
        self.assertTrue((sharpness.grad != 0).all())

    def test_zero_and_collinear_axes_form_orthogonal_frames(self):
        axis = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        z = normalize_asg_axes(axis)
        x, y = _orthonormalize_tangent(axis, axis)
        frame = torch.stack([x, y, z], dim=-2)
        torch.testing.assert_close(frame @ frame.transpose(-1, -2), torch.eye(3).expand(3, 3, 3))

    def test_asg_formula_and_sharpness_axis_exchange(self):
        axis = torch.tensor([[[[0., 0., 1.]]]])
        tangent = torch.tensor([[[[1., 0., 0.]]]])
        sharp = torch.tensor([[[[1., 4.]]]])
        amp = torch.zeros(1, 1, 1, 1)
        bias = torch.zeros(1, 1, 1)
        distance = torch.zeros_like(bias)
        direction = torch.tensor([[0.3, 0.4, 0.8660254]])
        actual, _, _ = evaluate_anisotropic_spherical_gaussians(
            axis, tangent, sharp, amp, bias, distance, direction, torch.ones(1, 1), 1.0,
        )
        lambdas = torch.nn.functional.softplus(sharp) + 1.0
        expected = 0.5 + 0.5 * direction[0, 2] * torch.exp(
            -lambdas[..., 0] * 0.3 ** 2 - lambdas[..., 1] * 0.4 ** 2)
        torch.testing.assert_close(actual.flatten(), expected.flatten())
        swapped, _, _ = evaluate_anisotropic_spherical_gaussians(
            axis, torch.tensor([[[[0., 1., 0.]]]]), sharp.flip(-1), amp,
            bias, distance, direction, torch.ones(1, 1), 1.0,
        )
        torch.testing.assert_close(actual, swapped)


if __name__ == "__main__":
    unittest.main()
