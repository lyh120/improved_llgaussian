import io
import tempfile
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

import torch

from arguments import ModelParams, _backfill_model_compatibility
from model_fixture import make_cpu_model
from utils.decomposition_config import REPRESENTATION_DEFAULTS, configure_reflectance
from utils.reflectance_utils import estimate_initial_b0, reflectance_config


class DecompositionCompatibilityTest(unittest.TestCase):
    def test_cli_and_old_config_backfill_keep_the_historical_defaults(self):
        parser = ArgumentParser()
        ModelParams(parser)
        parsed = parser.parse_args([])
        legacy = _backfill_model_compatibility({"num_sg": 2, "use_sg": True})
        for name, value in REPRESENTATION_DEFAULTS.items():
            self.assertEqual(getattr(parsed, name), value)
            self.assertEqual(legacy[name], value)
        self.assertEqual(legacy["sg_lobes"], 2)
        self.assertEqual(parsed.enhancement_prior, "cidnet")
        self.assertEqual(parsed.depth_patch_size, 128)

    def test_representation_sidecar_roundtrip_and_absent_sidecar_fallback(self):
        model = make_cpu_model()
        model.reflectance_detail_scale = 0.7
        model.reflectance_decoder_scale = 0.08
        with tempfile.TemporaryDirectory() as temp:
            model.save_representation_config(temp)
            restored = make_cpu_model()
            restored.load_representation_config(temp)
            self.assertEqual(reflectance_config(restored), reflectance_config(model))
        with tempfile.TemporaryDirectory() as temp:
            restored.load_representation_config(temp)
            self.assertEqual(reflectance_config(restored), REPRESENTATION_DEFAULTS)

    def test_training_checkpoint_preserves_networks_settings_pose_and_statistics(self):
        source = make_cpu_model()
        source.reflectance_decoder_scale = 0.07
        source.anchor_birth_iteration[:] = 43
        source.training_prior_state = {"provider": "cidnet", "cidnet_round_idx": 3}
        with torch.no_grad():
            source.mlp_reflectance_decoder[-1].bias.fill_(0.4)
            source.P.fill_(0.15)
        # Create an actual Adam state before serialization.
        source.get_reflectance_with_decoder(source._anchor_feat, source._offset, torch.ones(2, dtype=torch.bool)).sum().backward()
        source.optimizer.step()
        source.optimizer.zero_grad(set_to_none=True)
        buffer = io.BytesIO()
        torch.save(source.capture(), buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=False)
        restored = make_cpu_model()
        restored.restore(payload, SimpleNamespace())
        self.assertEqual(restored.reflectance_decoder_scale, 0.07)
        self.assertEqual(restored.training_prior_state, source.training_prior_state)
        torch.testing.assert_close(restored.P, source.P)
        torch.testing.assert_close(restored.anchor_birth_iteration, source.anchor_birth_iteration)
        torch.testing.assert_close(restored.filter_3D, source.filter_3D)
        self.assertEqual(len(restored.optimizer.state), len(source.optimizer.state))
        visible = torch.ones(2, dtype=torch.bool)
        torch.testing.assert_close(
            restored.get_reflectance_with_decoder(restored._anchor_feat, restored._offset, visible),
            source.get_reflectance_with_decoder(source._anchor_feat, source._offset, visible),
        )

    def test_legacy_training_tuple_still_restores_tensors_and_numeric_defaults(self):
        source, restored = make_cpu_model(), make_cpu_model()
        restored.reflectance_detail_scale = 0.2
        restored.restore(source._capture_tensors(), SimpleNamespace())
        torch.testing.assert_close(restored._base_log_reflectance, source._base_log_reflectance)
        self.assertEqual(reflectance_config(restored), REPRESENTATION_DEFAULTS)

    def test_b0_projection_uses_row_vector_matrices_and_excludes_invalid_observations(self):
        model = make_cpu_model()
        model.b0_init_detail_gain = 0.0
        image = torch.zeros(3, 3, 3)
        image[:, 0, 0] = torch.tensor([0.2, 0.4, 0.8])
        image[:, 2, 2] = torch.tensor([0.8, 0.2, 0.4])
        # A row-vector translation makes a transpose error observable.
        matrix = torch.eye(4)
        matrix[3, 0] = 0.1
        camera = SimpleNamespace(original_image=image, world_view_transform=torch.eye(4),
                                 full_proj_transform=matrix, znear=0.01)
        anchors = torch.tensor([[-2/3 - 0.1, -2/3, 0.5], [2/3 - 0.1, 2/3, 0.5],
                                [9.0, 0.0, 0.5], [0.0, 0.0, -0.5],
                                [float("nan"), 0.0, 0.5]])
        result = estimate_initial_b0(anchors, [camera], model)
        torch.testing.assert_close(result[0], torch.log(torch.tensor([0.25, 0.5, 1.0])))
        torch.testing.assert_close(result[1], torch.log(torch.tensor([1.0, 0.25, 0.5])))
        for index in (2, 3, 4):
            torch.testing.assert_close(result[index], result[:2].mean(dim=0))
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(result.requires_grad)

    def test_invalid_capacity_settings_are_rejected(self):
        for settings in ({"reflectance_decoder_scale": 1.0}, {"b0_init_window_size": 4}):
            with self.assertRaises(ValueError):
                configure_reflectance(SimpleNamespace(), settings)


if __name__ == "__main__":
    unittest.main()
