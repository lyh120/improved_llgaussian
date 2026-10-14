"""Accumulate auxiliary losses without granting them geometry permissions."""

import torch


REFLECTANCE_GROUPS = {
    "base_log_reflectance", "reflectance_offset_delta", "mlp_reflectance_decoder",
}
ASG_REGULARIZER_GROUPS = {"illum_asg_amplitude", "illum_asg_sharpness"}


def parameters_in_groups(optimizer, names: set[str]) -> list[torch.Tensor]:
    """Select live trainable optimizer parameters, including after densification."""
    return [p for group in optimizer.param_groups if group.get("name") in names
            for p in group["params"] if p.requires_grad]


def accumulate_auxiliary_gradients(
    terms: dict, parameters: list[torch.Tensor], audit: bool = False,
) -> dict:
    """Add only whitelisted gradients; preserve the graph for the main backward.

    Each term is (unweighted scalar loss, effective weight). Zero-weight terms
    do not traverse the graph. A disconnected nonzero loss is a programming
    error; a connected loss with a zero derivative is reported as inactive.
    """
    active = [(name, loss, float(weight)) for name, (loss, weight) in terms.items()
              if float(weight) != 0.0]
    if not active:
        return {"enabled_terms": 0, "grad_norm": 0.0}
    if not parameters:
        raise RuntimeError("Enabled auxiliary losses have no trainable parameter group")
    stats = {"enabled_terms": len(active)}
    total = None
    for name, loss, weight in active:
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(f"Nonfinite auxiliary loss: {name}")
        if not loss.requires_grad:
            if loss.detach().abs().item() != 0.0:
                raise RuntimeError(f"Detached nonzero auxiliary loss: {name}")
            if audit:
                stats[f"{name}/grad_norm"] = 0.0
                stats[f"{name}/inactive"] = 1
            continue
        weighted = loss * weight
        total = weighted if total is None else total + weighted
        if audit:
            gradients = torch.autograd.grad(weighted, parameters, retain_graph=True,
                                            allow_unused=True)
            connected = [g for g in gradients if g is not None]
            if not connected and loss.detach().abs().item() != 0.0:
                raise RuntimeError(f"Auxiliary loss has no whitelisted path: {name}")
            norm = sum(g.detach().square().sum() for g in connected)
            norm = float(torch.sqrt(norm).item()) if connected else 0.0
            stats.update({f"{name}/raw": float(loss.detach().item()),
                          f"{name}/weight": weight,
                          f"{name}/weighted": float(weighted.detach().item()),
                          f"{name}/grad_norm": norm,
                          f"{name}/inactive": int(norm == 0.0)})
    if total is None:
        stats["grad_norm"] = 0.0
        return stats
    gradients = torch.autograd.grad(total, parameters, retain_graph=True, allow_unused=True)
    norm = 0.0
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite auxiliary gradient")
            detached = gradient.detach()
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            parameter.grad.add_(detached)
            if audit:
                norm += float(detached.square().sum().item())
    if audit:
        stats["grad_norm"] = norm ** 0.5
    return stats
