import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from arguments import ModelParams
from utils.stablesr_prior_utils import (
    enhancement_gain, prior_cache_identity, prior_cache_matches,
    prior_cache_path, sha256_file, stablesr_input,
)


class StableSRPriorProtocolTest(unittest.TestCase):
    def test_prior_is_optional_with_cidnet_default(self):
        parser = argparse.ArgumentParser()
        ModelParams(parser)
        self.assertEqual(parser.parse_args([]).enhancement_prior, "cidnet")
        self.assertEqual(parser.parse_args(["--enhancement_prior", "stablesr"]).enhancement_prior, "stablesr")

    def test_stablesr_does_not_clip_gain_or_input_early(self):
        mean = torch.tensor(0.8)
        self.assertEqual(enhancement_gain(mean, "stablesr"), 0)
        self.assertEqual(enhancement_gain(mean, "cidnet"), 1)
        image = torch.tensor([[[0.2, 0.8]]])
        torch.testing.assert_close(stablesr_input(image, 3), torch.tensor([[[[0.6, 2.4]]]]))

    def test_bad_training_mean_is_rejected(self):
        for value in (0., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                enhancement_gain(torch.tensor(value), "stablesr")

    def test_cache_tracks_inputs_weights_and_output_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings_path = root / "configs/config.gin"
            settings_path.parent.mkdir()
            settings_path.write_text((Path(__file__).resolve().parents[1] / "configs/config.gin").read_text())
            for name in ("utils/StableSR_utlis.py", "checkpoints/stablesr_turbo.ckpt",
                         "checkpoints/vqgan_cfw_00011.ckpt",
                         "submodules/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml",
                         "submodules/StableSR/configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            camera = SimpleNamespace(uid=0, image_name="train", original_image=torch.full((3, 4, 4), 0.1))
            identity = prior_cache_identity(root, [camera], 4)
            cache = prior_cache_path(root, identity)
            cache.mkdir()
            image = cache / "0.png"
            image.write_bytes(b"prior")
            self.assertFalse(prior_cache_matches(cache, identity))
            cache.with_suffix(".manifest.json").write_text(json.dumps({
                "identity": identity, "outputs_sha256": {"0.png": sha256_file(image)},
            }))
            self.assertTrue(prior_cache_matches(cache, identity))
            camera.original_image.fill_(0.2)
            changed = prior_cache_identity(root, [camera], 4)
            self.assertNotEqual(prior_cache_path(root, changed), cache)
            self.assertFalse(prior_cache_matches(cache, changed))
            (root / "checkpoints/stablesr_turbo.ckpt").write_bytes(b"changed weights")
            self.assertNotEqual(prior_cache_identity(root, [camera], 4)["files_sha256"], identity["files_sha256"])
            image.write_bytes(b"corrupted")
            self.assertFalse(prior_cache_matches(cache, identity))

    def test_modified_sampler_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "configs/config.gin"
            config.parent.mkdir()
            config.write_text("Config.ddpm_steps = 200\n")
            with self.assertRaisesRegex(ValueError, "ddpm_steps"):
                prior_cache_identity(root, [], 4)


if __name__ == "__main__":
    unittest.main()
