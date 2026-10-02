import unittest

import torch

from utils.sg_utils import evaluate_anisotropic_spherical_gaussians


def _asg_inputs(lobes):
    axis = torch.zeros(1, 1, lobes, 3)
    axis[..., 2] = 1.0
    tangent = torch.zeros(1, 1, lobes, 3)
    tangent[..., 0] = 1.0
    sharpness = torch.full((1, 1, lobes, 2), -10.0)
    amplitude = torch.full((1, 1, lobes, 1), -2.0)
    bias = torch.full((1, 1, 1), -20.0)
    dist_weight = torch.zeros(1, 1, 1)
    return axis, tangent, sharpness, amplitude, bias, dist_weight


class AnisotropicSphericalGaussianTest(unittest.TestCase):
    def test_lobes_combine_additively(self):
        view_dirs = torch.tensor([[0.0, 0.0, 1.0]])
        view_dist = torch.tensor([[1.0]])

        illumination_one, _, _ = evaluate_anisotropic_spherical_gaussians(
            *_asg_inputs(1), view_dirs, view_dist, lambda_min=1.0, use_distance=False
        )
        illumination_two, _, _ = evaluate_anisotropic_spherical_gaussians(
            *_asg_inputs(2), view_dirs, view_dist, lambda_min=1.0, use_distance=False
        )

        # Two identical lobes double the lobe response instead of averaging
        # it back down to a single-lobe value.
        torch.testing.assert_close(illumination_two, 2.0 * illumination_one)


if __name__ == "__main__":
    unittest.main()
