"""Stage policies shared by the CLI and CPU regression checks."""

import copy


def validate_training_args(args) -> None:
    """Reject inconsistent branch and initial-retention settings."""
    if args.use_dual_transient and not args.use_residual:
        raise ValueError("--use_dual_transient requires --use_residual")
    if not 0 < args.prune_ratio <= 1:
        raise ValueError("--prune_ratio is the initial retention fraction and must be in (0,1]")


def prepare_warmup_args(args):
    """Keep the user's initial retention/residual choices during geometry warmup."""
    warmup = copy.deepcopy(args)
    warmup.iterations = 2000
    warmup.save_iterations.append(warmup.iterations)
    warmup.offset_lr_init = 0.0001
    warmup.opacity_lr = 0.1
    warmup.mlp_color_lr_init = 0.1
    return warmup


def residual_stage_state(use_residual, use_dual_transient, mode, iteration, start_iter, ramp_iters):
    """Single residual starts with main training; dual transient keeps its ramp."""
    if not use_residual or mode == "warmup":
        return False, 0.0
    if not use_dual_transient:
        return True, 1.0
    if iteration < start_iter:
        return False, 0.0
    return True, min(1.0, float(iteration - start_iter + 1) / max(1, ramp_iters))
