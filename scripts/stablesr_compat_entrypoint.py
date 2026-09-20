#!/usr/bin/env python3
"""Run bundled StableSR with compatibility aliases for current torchvision."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


def _install_torchvision_compatibility() -> None:
    """Expose the removed private functional_tensor import used by old BasicSR."""
    try:
        __import__("torchvision.transforms.functional_tensor")
    except ModuleNotFoundError:
        from torchvision.transforms.functional import rgb_to_grayscale

        compatibility_module = types.ModuleType(
            "torchvision.transforms.functional_tensor"
        )
        compatibility_module.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[compatibility_module.__name__] = compatibility_module


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: stablesr_compat_entrypoint.py INFERENCE_SCRIPT [ARGS...]")
    inference_script = str(Path(sys.argv[1]).resolve())
    inference_script_directory = str(Path(inference_script).parent)
    if inference_script_directory not in sys.path:
        # Match `python StableSR/scripts/inference.py`: sibling helpers such
        # as util_image.py and wavelet_color_fix.py must be importable.
        sys.path.insert(0, inference_script_directory)
    sys.argv = [inference_script, *sys.argv[2:]]
    _install_torchvision_compatibility()
    runpy.run_path(inference_script, run_name="__main__")


if __name__ == "__main__":
    main()
