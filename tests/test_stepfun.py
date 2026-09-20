import unittest

import numpy as np
import torch

from utils.stepfun import sample, sample_np


class StepFunctionTest(unittest.TestCase):
    def test_uniform_intervals_return_uniform_quantiles(self):
        result = sample_np(None, np.array([0.0, 1.0, 2.0]), np.zeros(2), 5)
        np.testing.assert_allclose(result, np.linspace(0.0, 2.0, 5))

    def test_torch_wrapper_and_validation(self):
        result = sample(None, torch.tensor([0.0, 1.0]), torch.tensor([0.0]), 3)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "one more"):
            sample_np(None, np.array([0.0, 1.0]), np.zeros(2), 3)


if __name__ == "__main__":
    unittest.main()
