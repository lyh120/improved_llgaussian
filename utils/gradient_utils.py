"""Restrict auxiliary supervision to the parameter blocks it describes."""

import torch


REFLECTANCE_PARAM_GROUP_NAMES = frozenset({
    "base_log_reflectance", "reflectance_offset_delta",
    "mlp_reflectance_decoder", "mlp_reflectance",
})


def accumulate_auxiliary_gradients(loss: torch.Tensor, optimizer, group_names,
                                  return_norms: bool = False):
    """Accumulate an auxiliary gradient without updating other parameter blocks.

    The main reconstruction still uses ordinary backward. Shared anchor features
    are deliberately excluded: changing them also changes opacity and covariance.
    Frozen rasterizer inputs are required when using image-space supervision.
    """
    named_parameters = [
        (group.get("name"), parameter)
        for group in optimizer.param_groups
        if group.get("name") in group_names
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters or not loss.requires_grad:
        return {} if return_norms else None
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True,
    )
    norms = {}
    for (name, parameter), gradient in zip(named_parameters, gradients):
        if gradient is not None:
            gradient = gradient.detach()
            if return_norms:
                norms[name] = norms.get(name, 0.) + float(gradient.abs().sum())
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)
    return norms if return_norms else None


def rasterize_frozen_geometry(rasterizer, colors: torch.Tensor, **geometry):
    """Rasterize differentiable appearance with constant geometry and visibility."""
    return rasterizer(
        colors_precomp=colors,
        **{
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in geometry.items()
        },
    )
