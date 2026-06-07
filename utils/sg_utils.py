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


def _orthonormalize_tangent(axis: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    z_axis = normalize_sg_directions(axis, eps=eps)
    tangent = tangent - (tangent * z_axis).sum(dim=-1, keepdim=True) * z_axis
    tangent_norm = tangent.norm(dim=-1, keepdim=True)

    fallback = torch.zeros_like(tangent)
    fallback[..., 0] = 1.0
    alt = torch.zeros_like(tangent)
    alt[..., 1] = 1.0
    fallback = torch.where(torch.abs(z_axis[..., :1]) > 0.9, alt, fallback)
    fallback = fallback - (fallback * z_axis).sum(dim=-1, keepdim=True) * z_axis
    tangent = torch.where(tangent_norm > eps, tangent, fallback)

    x_axis = normalize_sg_directions(tangent, eps=eps)
    y_axis = normalize_sg_directions(torch.cross(z_axis, x_axis, dim=-1), eps=eps)
    return x_axis, y_axis


def evaluate_anisotropic_spherical_gaussians(
    axis: torch.Tensor,
    tangent: torch.Tensor,
    sharpness: torch.Tensor,
    amplitude: torch.Tensor,
    bias: torch.Tensor,
    dist_weight: torch.Tensor,
    view_dirs: torch.Tensor,
    view_dist: torch.Tensor,
    lambda_min: float,
    use_distance: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Evaluate per-offset ASG illumination from point-cloud parameters.

    Args:
        axis: Raw ASG lobe axes shaped (N, n_offsets, asg_lobes, 3).
        tangent: Raw ASG tangent axes shaped (N, n_offsets, asg_lobes, 3).
        sharpness: Raw two-axis bandwidths shaped (N, n_offsets, asg_lobes, 2).
        amplitude: Raw lobe amplitudes shaped (N, n_offsets, asg_lobes, 1).
        bias: Raw per-offset base illumination shaped (N, n_offsets, 1).
        dist_weight: Raw per-offset distance falloff shaped (N, n_offsets, 1).
        view_dirs: Unit view directions shaped (N, 3).
        view_dist: View distances shaped (N, 1).
        lambda_min: Minimum sharpness added after softplus.
        use_distance: Whether to apply the learned distance falloff.

    Returns:
        Illumination shaped (N * n_offsets, 1), per-offset illumination shaped
        (N, n_offsets), and scalar/stat tensors.
    """
    z_axis = normalize_sg_directions(axis)
    x_axis, y_axis = _orthonormalize_tangent(z_axis, tangent)
    bandwidth = F.softplus(sharpness) + lambda_min
    lambda_x = bandwidth[..., :1]
    lambda_y = bandwidth[..., 1:2]
    amplitudes = torch.sigmoid(amplitude)

    view_dirs = normalize_sg_directions(view_dirs).view(view_dirs.shape[0], 1, 1, 3)
    dot_z = (view_dirs * z_axis).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    dot_x = (view_dirs * x_axis).sum(dim=-1, keepdim=True)
    dot_y = (view_dirs * y_axis).sum(dim=-1, keepdim=True)
    upper_hemi = torch.clamp(dot_z, min=0.0)
    lobes = amplitudes * upper_hemi * torch.exp(-lambda_x * dot_x.square() - lambda_y * dot_y.square())
    asg_term = lobes.mean(dim=2)

    if use_distance:
        falloff = torch.exp(-F.softplus(dist_weight) * torch.log1p(view_dist.view(-1, 1, 1)))
    else:
        falloff = 1.0
    illumination = torch.clamp(torch.sigmoid(bias) + asg_term * falloff, 0.0, 1.0)
    illumination_feat = illumination.squeeze(-1)

    anisotropy = torch.maximum(lambda_x, lambda_y) / (torch.minimum(lambda_x, lambda_y) + 1e-6)
    stats = {
        "illumination_energy": amplitudes.mean(),
        "illumination_lambda_mean": bandwidth.mean(),
        "illumination_lambda_max": bandwidth.max(),
        "illumination_anisotropy": anisotropy.mean(),
        "asg_energy": amplitudes.mean(),
        "asg_lambda_mean": bandwidth.mean(),
        "asg_lambda_max": bandwidth.max(),
        "asg_anisotropy": anisotropy.mean(),
        "sg_energy": amplitudes.mean(),
        "sg_lambda_mean": bandwidth.mean(),
        "sg_lambda_max": bandwidth.max(),
    }
    return illumination.reshape(-1, 1), illumination_feat, stats


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
