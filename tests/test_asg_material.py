"""Explicit ASG formula, degenerate frames and material-aware objectives."""
import math
import unittest

import torch
import torch.nn.functional as F

from utils.rl_compat_utils import (asg_material_image_terms, asg_parameter_compat_terms,
                                   prepare_asg_material_target, weighted_compat_losses)
from utils.sg_utils import _orthonormalize_tangent, evaluate_anisotropic_spherical_gaussians


class ASGMaterialTest(unittest.TestCase):
    def test_degenerate_frames_are_orthonormal_and_finite(self):
        axis = torch.tensor([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [1e-12, 0., 0.]], requires_grad=True)
        tangent = axis.detach().clone().requires_grad_()
        x, y, z = _orthonormalize_tangent(axis, tangent)
        frame = torch.stack([x, y, z], dim=-1)
        torch.testing.assert_close(frame.transpose(-1, -2) @ frame, torch.eye(3).expand(4, 3, 3))
        self.assertTrue(torch.isfinite(frame).all())
        frame.square().sum().backward()
        self.assertTrue(torch.isfinite(axis.grad).all() and torch.isfinite(tangent.grad).all())

    def test_formula_and_axis_bandwidth_swap(self):
        axis = torch.tensor([[[[0., 0., 1.]]]])
        tangent = torch.tensor([[[[1., 0., 0.]]]])
        raw = torch.log(torch.expm1(torch.tensor([[[[2., 5.]]]])))
        amplitude = torch.tensor([[[[.3]]]])
        bias = torch.tensor([[[-2.]]])
        direction = F.normalize(torch.tensor([[.2, .4, .9]]), dim=-1)
        distance = torch.ones(1, 1)
        result, _, _ = evaluate_anisotropic_spherical_gaussians(axis, tangent, raw, amplitude, bias,
                torch.zeros_like(bias), direction, distance, 1.)
        expected = bias.sigmoid().reshape(-1) + amplitude.sigmoid().reshape(-1) * direction[:, 2] * torch.exp(
                -3 * direction[:, 0].square() -6 * direction[:, 1].square())
        torch.testing.assert_close(result.reshape(-1), expected)
        swapped, _, _ = evaluate_anisotropic_spherical_gaussians(axis, torch.tensor([[[[0., 1., 0.]]]]),
                raw.flip(-1), amplitude, bias, torch.zeros_like(bias), direction, distance, 1.)
        torch.testing.assert_close(result, swapped)

    def test_means_match_paper_and_extreme_values_remain_finite(self):
        amplitude = torch.tensor([[-20., 0., 20.]], requires_grad=True)
        raw = torch.tensor([[[-100., 1000.], [2., 2.], [3., -100.]]], requires_grad=True)
        terms = asg_parameter_compat_terms(amplitude, raw)
        widths = F.softplus(raw) + 1
        torch.testing.assert_close(terms['asg_energy'], amplitude.sigmoid().mean())
        torch.testing.assert_close(terms['asg_sharpness'], widths.mean())
        torch.testing.assert_close(terms['asg_anisotropy'], (widths.max(-1).values / (widths.min(-1).values + 1e-6)).mean())
        sum(terms.values()).backward()
        self.assertTrue(torch.isfinite(raw.grad).all() and torch.isfinite(amplitude.grad).all())

    def test_achromatic_shadow_rejected_but_material_edge_selected(self):
        shadow = torch.tensor([.12, .06, .03])[:, None, None].expand(3, 32, 32).clone()
        shadow[:, :, 16:] *= .4
        material = shadow.clone()
        material[:, :, 16:] = torch.tensor([.025, .08, .04])[:, None, None]
        a = prepare_asg_material_target(shadow)
        b = prepare_asg_material_target(material)
        self.assertEqual(float(a['material_confidence'].sum()), 0)
        self.assertGreater(float(b['material_confidence'].sum()), 0)
        self.assertGreater(float(b['material_edge_confidence'].sum()), 0)
        self.assertEqual(float(b['flat_confidence'][:, :, 15:17].sum()), 0)

    def test_empty_masks_and_flat_color_noise(self):
        for target, coverage in [(torch.zeros(3, 24, 24), torch.ones(3, 24, 24)),
                                 (torch.ones(3, 24, 24), torch.ones(3, 24, 24)),
                                 (torch.full((3, 24, 24), .2), torch.zeros(3, 24, 24))]:
            r = torch.rand_like(target, requires_grad=True)
            terms = asg_material_image_terms(r, prepare_asg_material_target(target), coverage)
            self.assertEqual(float(terms['edge'] + terms['contrast'] + terms['flat']), 0)
            (terms['edge'] + terms['contrast'] + terms['flat']).backward()
            self.assertEqual(int(torch.count_nonzero(r.grad)), 0)
        target = torch.tensor([.12, .06, .03])[:, None, None].expand(3, 24, 24)
        prepared = prepare_asg_material_target(target)
        self.assertGreater(float(prepared['flat_confidence'].sum()), 0)
        self.assertEqual(float(asg_material_image_terms(target * 3, prepared, torch.ones_like(target))['flat']), 0)
        noisy = target.clone(); noisy[0, 8:16, 8:16] *= 1.5
        self.assertGreater(float(asg_material_image_terms(noisy, prepared, torch.ones_like(target))['flat']), 0)

    def test_asg_ramp_zero_scale_and_two_sided_structure(self):
        terms = {key: torch.tensor(1., requires_grad=True) for key in
                ('edge', 'contrast', 'flat', 'b0', 'detail', 'decoder', 'asg_energy', 'asg_sharpness', 'asg_anisotropy')}
        for iteration, scale in [(1000, 1.), (8000, 0.)]:
            r, l = weighted_compat_losses(terms, 'asg_paper', iteration, scale)
            self.assertEqual(float(r + l), 0)
        r_half, l_half = weighted_compat_losses(terms, 'asg_paper', 1500)
        r_full, l_full = weighted_compat_losses(terms, 'asg_paper', 2000)
        torch.testing.assert_close(r_half * 2, r_full)
        torch.testing.assert_close(l_half * 2, l_full)


if __name__ == '__main__':
    unittest.main()
