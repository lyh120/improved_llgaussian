"""Small deterministic inverse-CDF helpers for constant-speed camera paths."""

from __future__ import annotations

import numpy as np
import torch


def sample_np(rng, x, log_weights, num_samples):
    """Resample interval edges ``x`` according to log-space interval weights."""
    del rng  # The quantiles are deterministic; retained for the historical API.
    edges = np.asarray(x)
    log_weights = np.asarray(log_weights)
    if edges.ndim != 1 or log_weights.ndim != 1:
        raise ValueError("stepfun sampling expects one-dimensional arrays")
    if edges.size != log_weights.size + 1:
        raise ValueError(
            "stepfun edges must contain exactly one more value than interval weights: "
            f"edges={edges.size}, weights={log_weights.size}"
        )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not np.isfinite(edges).all() or not np.isfinite(log_weights).all():
        raise ValueError("stepfun inputs must be finite")

    weights = np.exp(log_weights - log_weights.max())
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("stepfun interval weights must have positive finite mass")
    cumulative = np.concatenate(([0.0], np.cumsum(weights / total)))
    cumulative[-1] = 1.0
    quantiles = np.linspace(0.0, 1.0, int(num_samples), endpoint=True)
    return np.interp(quantiles, cumulative, edges)


def sample(rng, x, log_weights, num_samples):
    """Torch-compatible wrapper around :func:`sample_np`."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(log_weights, torch.Tensor):
        log_weights = log_weights.detach().cpu().numpy()
    return sample_np(rng, x, log_weights, num_samples)
