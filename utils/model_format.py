"""Shared constants for the strict explicit-appearance model format."""

MODEL_FORMAT_VERSION = 2
# Keeps the top-level v2 contract while rejecting the unstable unbounded-gain
# enhanced appearance layout used by early v2 experiments.
EXPLICIT_APPEARANCE_FORMAT_VERSION = 2
NUMERICAL_EPS = 1e-6


def training_stage_state(
    iteration: int,
    use_warmup: bool,
    warmup_iterations: int,
) -> dict[str, bool | int]:
    """Return the strict checkpoint metadata for warmup/main restoration."""
    enabled = bool(use_warmup and warmup_iterations > 0)
    boundary = int(warmup_iterations) if enabled else 0
    return {
        "warmup_enabled": enabled,
        "warmup_iterations": boundary,
        "transition_completed": not enabled or int(iteration) > boundary,
    }
