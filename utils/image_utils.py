"""Small image metrics shared by training and rendering."""

import torch

from utils.model_format import NUMERICAL_EPS


def psnr(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-image PSNR for images represented in the ``[0, 1]`` range."""
    squared_error = (image - target).square().reshape(image.shape[0], -1)
    mean_squared_error = squared_error.mean(dim=1, keepdim=True)
    return -10.0 * torch.log10(mean_squared_error.clamp_min(NUMERICAL_EPS))
