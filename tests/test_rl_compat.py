"""Numerical behavior of the new appearance constraints."""

import unittest

import torch

from utils.rl_compat_utils import (compat_ramp, image_compat_terms, parameter_compat_terms,
                                   prepare_compat_target, structure_band, weighted_compat_losses)


class RLCompatTest(unittest.TestCase):
    def test_empty_dark_saturated_and_uncovered_masks(self):
        for target, coverage in [(torch.zeros(3, 20, 20), torch.ones(3, 20, 20)),
                                 (torch.ones(3, 20, 20), torch.ones(3, 20, 20)),
                                 (torch.full((3, 20, 20), .2), torch.zeros(3, 20, 20))]:
            r = torch.rand_like(target, requires_grad=True)
            terms = image_compat_terms(r, prepare_compat_target(target), coverage)
            loss, _ = weighted_compat_losses(terms, "color", 2000)
            self.assertEqual(float(loss), 0.)
            loss.backward()
            self.assertEqual(int(torch.count_nonzero(r.grad)), 0)

    def test_color_is_exposure_invariant_and_detects_cast(self):
        target = torch.tensor([.08, .04, .02])[:, None, None].expand(3, 20, 20)
        prepared = prepare_compat_target(target)
        correct = image_compat_terms(target * 4, prepared, torch.ones_like(target))
        wrong = image_compat_terms(target.flip(0), prepared, torch.ones_like(target))
        self.assertLess(float(correct["color"]), 1e-6)
        self.assertGreater(float(wrong["color"]), .1)
        self.assertEqual(float(correct["edge_mask_fraction"]), 0)
        self.assertEqual(float(correct["contrast_mask_fraction"]), 0)

    def test_band_has_both_correction_directions(self):
        response = torch.tensor([.4, 1., 2.], requires_grad=True)
        structure_band(response, torch.ones(3)).sum().backward()
        torch.testing.assert_close(response.grad, torch.tensor([-1., 0., 1.]))

    def test_confidence_rejects_noise_and_recognizes_clean_edge(self):
        target = torch.full((3, 32, 32), .04)
        target[:, :, 16:] = .2
        prepared = prepare_compat_target(target)
        self.assertGreater(float((prepared["edge_confidence"] > 0).float().mean()), 0)
        self.assertTrue(all(not value.requires_grad for value in prepared.values()))
        self.assertEqual(float(prepared["confidence"][:, :2].sum()), 0)

    def test_stability_penalties_only_act_as_defined(self):
        b0 = torch.tensor([[-8., -1., .2]], requires_grad=True)
        detail = torch.zeros(1, 2, 3, requires_grad=True)
        decoder = torch.tensor(.1, requires_grad=True)
        tail = torch.tensor(0., requires_grad=True)
        terms = parameter_compat_terms(b0, detail, decoder, tail)
        terms["b0"].backward()
        self.assertLess(float(b0.grad[0, 0]), 0)
        self.assertEqual(float(b0.grad[0, 1]), 0)
        self.assertGreater(float(b0.grad[0, 2]), 0)
        self.assertEqual(float(terms["detail"]), 0)

    def test_activation_and_disabled_scale(self):
        self.assertEqual([compat_ramp(i) for i in (999, 1000, 1500, 2000, 8000)], [0, 0, .5, 1, 1])
        terms = {key: torch.tensor(1., requires_grad=True) for key in
                 ("edge", "contrast", "color", "b0", "detail", "decoder", "sg_tail")}
        for stage in ("sharp", "color", "stable"):
            r, sg = weighted_compat_losses(terms, stage, 2000, scale=0)
            self.assertEqual(float(r + sg), 0)


if __name__ == "__main__":
    unittest.main()
