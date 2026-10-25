import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from utils.prior_utils import load_rgb_prior, normalize_rgb_prior, prepare_stablesr_prior, resolve_cached_prior


def camera(uid, name):
    return SimpleNamespace(uid=uid, image_name=name, image_height=8, image_width=10,
                           original_image=torch.full((3, 8, 10), 0.2))


class PriorCompatibilityTest(unittest.TestCase):
    def test_stablesr_loader_resolves_configs_and_weights_against_project_root(self):
        path = Path(__file__).resolve().parents[1] / "utils" / "StableSR_utlis.py"
        function = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                        if isinstance(node, ast.FunctionDef) and node.name == "get_SRModel")
        loaded = []
        config = SimpleNamespace(config="./submodules/StableSR/configs/model.yaml",
                                 ckpt="./checkpoints/sr.ckpt", vqgan_ckpt="./checkpoints/vq.ckpt",
                                 seed=42, colorfix_type="adain", dec_w=0.75)

        def load_model(options, checkpoint):
            loaded.append(checkpoint)
            model = torch.nn.Linear(1, 1)
            model.decoder = SimpleNamespace(fusion_w=None)
            return model

        def load_yaml(value):
            loaded.append(value)
            return SimpleNamespace(model=object())

        namespace = dict(__file__=str(path), Path=Path, torch=torch,
                         gin=SimpleNamespace(clear_config=lambda: None,
                                             parse_config_file=lambda value: loaded.append(value)),
                         Config=lambda: config, seed_everything=lambda seed: None,
                         OmegaConf=SimpleNamespace(load=load_yaml), load_model_from_config=load_model)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        model, vq, _ = namespace["get_SRModel"]("./configs/config.gin")
        self.assertTrue(all(Path(value).is_absolute() for value in loaded))
        self.assertIn(str(path.parents[1] / "submodules" / "StableSR" / "configs" / "autoencoder" /
                          "autoencoder_kl_64x64x4_resi.yaml"), loaded)
        self.assertFalse(model.training)
        self.assertFalse(vq.training)

    def test_both_legacy_names_are_used_for_cache_check_and_load(self):
        with tempfile.TemporaryDirectory() as temp:
            views = [camera(2, "frame_a"), camera(3, "frame_b")]
            Image.new("RGB", (5, 4), "red").save(Path(temp) / "2.png")
            Image.new("RGB", (20, 16), "blue").save(Path(temp) / "frame_b.png")
            for view in views:
                self.assertIsNotNone(resolve_cached_prior(temp, view))
            targets = load_rgb_prior(temp, views)
            self.assertEqual(set(targets), {2, 3})
            for value in targets.values():
                self.assertEqual(value.shape, (3, 8, 10))
                self.assertFalse(value.requires_grad)
                self.assertTrue(torch.isfinite(value).all())

    def test_partial_cache_generates_only_missing_views_and_reuses_disk_values(self):
        with tempfile.TemporaryDirectory() as temp:
            views = [camera(2, "frame_a"), camera(3, "frame_b")]
            prior = Path(temp) / "diffusion_prior_2"
            prior.mkdir()
            existing = prior / "frame_a.png"
            Image.new("RGB", (10, 8), "red").save(existing)
            before = existing.read_bytes()
            dataset = SimpleNamespace(source_path=temp, stablesr_config="./configs/config.gin")
            model, vq = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
            calls = []

            def refine(model, vq, inputs, options):
                self.assertFalse(model.training)
                self.assertFalse(vq.training)
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                calls.append(inputs.shape)
                return inputs[:, :, ::2, ::2]

            def loader(config):
                self.assertTrue(Path(config).is_absolute())
                return model, vq, None, refine

            first = prepare_stablesr_prior(dataset, views, 2, backend_loader=loader)
            self.assertEqual(len(calls), 1)
            self.assertEqual(existing.read_bytes(), before)
            second = prepare_stablesr_prior(dataset, views, 2, backend_loader=lambda _: self.fail("Backend loaded for complete cache"))
            for uid in first:
                torch.testing.assert_close(first[uid], second[uid], rtol=0, atol=0)

    def test_cidnet_loader_and_cached_stablesr_do_not_import_stablesr(self):
        with tempfile.TemporaryDirectory() as temp:
            view = camera(7, "frame")
            directory = Path(temp) / "diffusion_prior_1"
            directory.mkdir()
            Image.new("RGB", (10, 8), "white").save(directory / "frame.png")
            with patch("utils.prior_utils._load_stablesr_backend", side_effect=AssertionError("Unexpected backend import")):
                load_rgb_prior(directory, [view], name_first=True)
                prepare_stablesr_prior(SimpleNamespace(source_path=temp), [view], 1)

    def test_nonfinite_targets_fail_and_tensor_targets_are_detached_and_clamped(self):
        view = camera(0, "frame")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_rgb_prior(torch.full((3, 8, 10), float("nan")), view)
        target = normalize_rgb_prior(torch.full((1, 3, 2, 2), 1.5, requires_grad=True), view)
        self.assertFalse(target.requires_grad)
        self.assertEqual(target.shape, (3, 8, 10))
        self.assertEqual(target.max().item(), 1.0)

    def test_llgaussian_mode_keeps_the_original_unclipped_gain_input(self):
        for mode, expected in (("llgaussian", 1.6), ("clamped", 1.0)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                view = camera(7, "frame")
                view.original_image.fill_(0.8)
                dataset = SimpleNamespace(source_path=temp, stablesr_input_mode=mode)

                def refine(model, vq, inputs, options):
                    self.assertAlmostEqual(inputs.max().item(), expected, places=6)
                    return inputs.clamp(0.0, 1.0)

                def loader(config):
                    return torch.nn.Linear(1, 1), torch.nn.Linear(1, 1), None, refine

                prepare_stablesr_prior(dataset, [view], 2, backend_loader=loader)


if __name__ == "__main__":
    unittest.main()
