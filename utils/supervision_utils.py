"""Reference supervision profiles share the same forward and base objective."""


def is_reference_supervision(profile: str) -> bool:
    """Return whether a profile uses the released LL-Gaussian base behavior."""
    return profile in {"llgaussian", "llgaussian_rl"}
