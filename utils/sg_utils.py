import torch
import torch.nn.functional as F


def normalize_sg_directions(directions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize SG lobe directions with a small numerical guard."""
    return F.normalize(directions, dim=-1, eps=eps)


def evaluate_spherical_gaussians(
    raw_params: torch.Tensor,
    view_dirs: torch.Tensor,
    n_offsets: int,
    sg_lobes: int,
    lambda_min: float,
) -> tuple[torch.Tensor, dict]:
    """Evaluate per-offset SG illumination from raw network parameters.

    Args:
        raw_params: Tensor shaped (N, n_offsets * sg_lobes * 5).
        view_dirs: Unit directions shaped (N, 3).
        n_offsets: Number of neural Gaussians per anchor.
        sg_lobes: Number of SG lobes per offset.
        lambda_min: Minimum sharpness added after softplus.

    Returns:
        Illumination shaped (N * n_offsets, 1) and scalar/stat tensors.
    """
    anchor_count = raw_params.shape[0]
    sg = raw_params.view(anchor_count, n_offsets, sg_lobes, 5)
    mu = normalize_sg_directions(sg[..., :3])
    lambdas = F.softplus(sg[..., 3:4]) + lambda_min
    amplitudes = torch.sigmoid(sg[..., 4:5])

    view_dirs = normalize_sg_directions(view_dirs).view(anchor_count, 1, 1, 3)
    cosine = (mu * view_dirs).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    lobes = amplitudes * torch.exp(lambdas * (cosine - 1.0))
    illumination = lobes.mean(dim=2).reshape(anchor_count * n_offsets, 1)

    stats = {
        "sg_energy": amplitudes.mean(),
        "sg_lambda_mean": lambdas.mean(),
        "sg_lambda_max": lambdas.max(),
    }
    return illumination, stats


def sg_energy_regularization(stats: dict | None) -> torch.Tensor:
    if not stats or "sg_energy" not in stats:
        return torch.tensor(0.0, device="cuda")
    return stats["sg_energy"]


def sg_sharpness_regularization(stats: dict | None) -> torch.Tensor:
    if not stats or "sg_lambda_mean" not in stats:
        return torch.tensor(0.0, device="cuda")
    return stats["sg_lambda_mean"]


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise FloatingPointError(f"{name} contains NaN or Inf values")
