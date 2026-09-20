import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from unittest.mock import patch

import utils.prior_utils as prior_utils

from utils.prior_utils import (
    load_depth_priors,
    load_enhancement_priors,
    prior_generation_required,
    resolve_required_file,
    stablesr_prior_directory_name,
)


def camera(name="frame", width=6, height=4, uid=1):
    return SimpleNamespace(
        image_name=name,
        image_width=width,
        image_height=height,
        uid=uid,
        original_image=torch.zeros((3, height, width)),
    )


def write_manifest(root: Path, prior_type: str, entries: list[dict], version=2):
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "model_format_version": version,
                "prior_type": prior_type,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


class PriorUtilsTest(unittest.TestCase):
    def test_combined_loader_uses_depth_then_enhancement_order(self):
        order = []
        with patch.object(
            prior_utils,
            "load_depth_priors",
            side_effect=lambda cameras, root: order.append(("depth", root)) or {1: "d"},
        ), patch.object(
            prior_utils,
            "load_enhancement_priors",
            side_effect=lambda cameras, root: order.append(("enhancement", root)) or {1: "e"},
        ):
            enhancement, depth = prior_utils.load_training_priors(
                [camera()], "enhancement-root", "depth-root"
            )
        self.assertEqual(order, [("depth", "depth-root"), ("enhancement", "enhancement-root")])
        self.assertEqual(enhancement, {1: "e"})
        self.assertEqual(depth, {1: "d"})

    def test_required_file_resolves_fallback_and_reports_all_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / "fallback.pth"
            fallback.touch()
            resolved = resolve_required_file(
                [str(root / "missing.pth"), str(fallback)],
                "Depth checkpoint",
                "--depth-checkpoint",
            )
            self.assertEqual(resolved, str(fallback.resolve()))
            with self.assertRaisesRegex(FileNotFoundError, "--depth-checkpoint"):
                resolve_required_file(
                    [str(root / "first.pth"), str(root / "second.pth")],
                    "Depth checkpoint",
                    "--depth-checkpoint",
                )

    def test_valid_name_aligned_priors_load_on_camera_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (6, 4), (10, 20, 30)).save(root / "enhanced.png")
            write_manifest(
                root,
                "enhancement_rgb",
                [{"image_name": "frame", "file": "enhanced.png", "width": 6, "height": 4}],
            )
            view = camera()
            result = load_enhancement_priors([view], str(root))
            self.assertEqual(tuple(result[1].shape), (3, 4, 6))
            self.assertEqual(result[1].device, view.original_image.device)
            self.assertIs(view.enhancement_prior, result[1])

            disparity = np.arange(24, dtype=np.float32).reshape(4, 6)
            np.save(root / "depth.npy", disparity, allow_pickle=False)
            write_manifest(
                root,
                "depth_disparity",
                [{"image_name": "frame", "file": "depth.npy", "width": 6, "height": 4}],
            )
            depth = load_depth_priors([view], str(root))[1]
            self.assertEqual(tuple(depth.shape), (1, 4, 6))
            self.assertEqual(depth.dtype, torch.float32)

    def test_missing_extra_duplicate_and_size_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (6, 4)).save(root / "a.png")
            base = {"image_name": "frame", "file": "a.png", "width": 6, "height": 4}

            write_manifest(root, "enhancement_rgb", [base, dict(base)])
            with self.assertRaisesRegex(ValueError, "Duplicate prior"):
                load_enhancement_priors([camera()], str(root))

            write_manifest(root, "enhancement_rgb", [{**base, "image_name": "extra"}])
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                load_enhancement_priors([camera()], str(root))

            write_manifest(root, "enhancement_rgb", [{**base, "width": 7}])
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                load_enhancement_priors([camera()], str(root))

            write_manifest(root, "enhancement_rgb", [base], version=1)
            with self.assertRaisesRegex(RuntimeError, "MODEL_FORMAT_VERSION=2"):
                load_enhancement_priors([camera()], str(root))

    def test_depth_dtype_and_finite_checks_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {"image_name": "frame", "file": "depth.npy", "width": 6, "height": 4}
            write_manifest(root, "depth_disparity", [entry])
            np.save(root / "depth.npy", np.zeros((4, 6), dtype=np.float64), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "float32"):
                load_depth_priors([camera()], str(root))
            value = np.zeros((4, 6), dtype=np.float32)
            value[0, 0] = np.nan
            np.save(root / "depth.npy", value, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_depth_priors([camera()], str(root))

    def test_stablesr_cache_directory_matches_clean_reference_format(self):
        self.assertEqual(stablesr_prior_directory_name(1.0), "diffusion_prior_1")
        self.assertEqual(stablesr_prior_directory_name(2.125), "diffusion_prior_2.125")
        self.assertEqual(stablesr_prior_directory_name(15.0), "diffusion_prior_15")

    def test_scene_local_generation_reuses_manifest_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing"
            self.assertTrue(prior_generation_required(str(missing), "CIDNet"))

            empty = root / "empty"
            empty.mkdir()
            self.assertTrue(prior_generation_required(str(empty), "DepthAnything"))

            cached = root / "cached"
            cached.mkdir()
            (cached / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertFalse(prior_generation_required(str(cached), "CIDNet"))

            invalid = root / "legacy"
            invalid.mkdir()
            (invalid / "old.png").write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "nonempty but has no v2 manifest"):
                prior_generation_required(str(invalid), "CIDNet")


if __name__ == "__main__":
    unittest.main()
