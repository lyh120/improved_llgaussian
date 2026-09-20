import tempfile
import unittest
from contextlib import redirect_stderr
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from arguments import (
    ModelParams,
    OptimizationParams,
    get_combined_args,
    validate_training_schedule,
)
from scene.explicit_appearance import orthonormal_frame
from scene.gaussian_model import GaussianModel
from utils.model_format import training_stage_state


def initialized_model():
    model = GaussianModel(feat_dim=8, n_offsets=2, device="cpu")
    count = 3
    model._anchor = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    model._offset = nn.Parameter(torch.zeros((count, 2, 3)))
    model._anchor_feat = nn.Parameter(torch.zeros((count, 8)))
    model._scaling = nn.Parameter(torch.zeros((count, 6)))
    model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1))
    model._opacity = nn.Parameter(torch.zeros((count, 1)))
    model.filter_3D = torch.zeros((count, 1))
    camera = SimpleNamespace(
        original_image=torch.full((3, 5, 5), 0.5),
        enhancement_prior=torch.full((3, 5, 5), 0.5),
        full_proj_transform=torch.eye(4),
        camera_center=torch.tensor([0.0, 0.0, -1.0]),
    )
    model.appearance.initialize(model._anchor, model._offset, model.get_scaling, cameras=[camera])
    model.max_radii2D = torch.zeros(count)
    model.spatial_lr_scale = 1.0
    model.P = torch.zeros((1, 7), requires_grad=True)
    parser = ArgumentParser()
    group = OptimizationParams(parser)
    optimization = group.extract(parser.parse_args([]))
    model.training_setup(optimization)
    return model, optimization


class ModelFormatV2Test(unittest.TestCase):
    def test_loss_cli_uses_only_the_six_v2_names(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        defaults = group.extract(parser.parse_args([]))
        self.assertEqual(defaults.lambda_reflectance_reconstruction, 1.0)
        self.assertEqual(defaults.lambda_illumination, 1.0)
        for removed_name in ("--lambda_reflectance_gradient", "--lambda_illum_tv"):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parser.parse_args([removed_name, "1.0"])

    def test_scaffold_3d_filter_cache_has_no_autograd_history(self):
        model, _ = initialized_model()
        camera = SimpleNamespace(
            R=np.eye(3, dtype=np.float32),
            T=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            focal_x=10.0,
            focal_y=10.0,
            image_width=16,
            image_height=16,
        )
        model.compute_3D_filter([camera])
        self.assertFalse(model.filter_3D.requires_grad)
        self.assertIsNone(model.filter_3D.grad_fn)

    def test_scaffold_initial_pruning_arguments_are_available(self):
        parser = ArgumentParser()
        group = ModelParams(parser)
        defaults = group.extract(parser.parse_args([]))
        self.assertEqual(defaults.prune_ratio, 1.0)
        arguments = group.extract(parser.parse_args([
            "--prune_ratio", "1.0",
            "--beta", "1.0",
        ]))
        self.assertEqual(arguments.prune_ratio, 1.0)
        self.assertEqual(arguments.beta, 1.0)

    def test_invalid_densification_schedule_is_rejected(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        invalid = group.extract(parser.parse_args([
            "--iterations", "6000",
            "--update_from", "800",
            "--update_until", "8500",
            "--prune_from_iter", "6000",
            "--max_pruned_anchors_per_update", "128",
        ]))
        with self.assertRaisesRegex(ValueError, "refinement tail"):
            validate_training_schedule(invalid)

        valid = group.extract(parser.parse_args([
            "--iterations", "6000",
            "--update_from", "800",
            "--update_until", "4500",
            "--prune_from_iter", "3500",
            "--max_pruned_anchors_per_update", "128",
        ]))
        validate_training_schedule(valid)

        growth_disabled = group.extract(parser.parse_args([
            "--iterations", "6000",
            "--update_from", "800",
            "--update_until", "8500",
            "--max_new_anchors_per_update", "0",
        ]))
        validate_training_schedule(growth_disabled)

        pruning_after_window = group.extract(parser.parse_args([
            "--iterations", "8000",
            "--update_from", "2000",
            "--update_until", "6000",
            "--update_interval", "100",
            "--max_new_anchors_per_update", "0",
            "--max_pruned_anchors_per_update", "128",
            "--prune_from_iter", "6000",
        ]))
        with self.assertRaisesRegex(ValueError, "pruning is enabled"):
            validate_training_schedule(pruning_after_window)

        pruning_enabled = group.extract(parser.parse_args([
            "--iterations", "8000",
            "--update_from", "2000",
            "--update_until", "6000",
            "--update_interval", "100",
            "--max_new_anchors_per_update", "0",
            "--max_pruned_anchors_per_update", "128",
            "--prune_from_iter", "5900",
        ]))
        validate_training_schedule(pruning_enabled)

    def test_format_version_is_not_cli_overridable_and_old_cfg_is_rejected(self):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)
        self.assertNotIn("--model_format_version", parser._option_string_actions)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "cfg_args").write_text(
                f"Namespace(model_path={directory!r})",
                encoding="utf-8",
            )
            with patch("sys.argv", ["render.py", "-m", directory]):
                with self.assertRaisesRegex(RuntimeError, "cannot upgrade an old cfg_args"):
                    get_combined_args(parser)

    def test_zero_prune_budget_disables_training_time_pruning(self):
        model, _ = initialized_model()
        model.anchor_demon.fill_(100.0)
        model.opacity_accum.zero_()
        disabled = model.adjust_anchor(
            check_interval=1,
            max_new_anchors=0,
            current_iteration=10,
            prune_from_iter=0,
            max_pruned_anchors=0,
        )
        self.assertEqual(disabled["prune_skipped"], "disabled")
        self.assertEqual(model.get_anchor.shape[0], 3)

        enabled = model.adjust_anchor(
            check_interval=1,
            max_new_anchors=0,
            current_iteration=10,
            prune_from_iter=0,
            max_pruned_anchors=1,
        )
        self.assertEqual(enabled["pruned"], 1)
        self.assertEqual(model.get_anchor.shape[0], 2)

    def test_warmup_scales_only_scaffold_learning_rates(self):
        model, _ = initialized_model()
        model.update_learning_rate(1, geometry_lr_scale=0.1)
        warmup = {group["name"]: group["lr"] for group in model.optimizer.param_groups}
        model.update_learning_rate(1, geometry_lr_scale=1.0)
        main = {group["name"]: group["lr"] for group in model.optimizer.param_groups}
        for name in warmup:
            if name.startswith("appearance_") or name == "pose":
                self.assertEqual(warmup[name], main[name], name)
            else:
                self.assertAlmostEqual(warmup[name], main[name] * 0.1, places=12)

    def test_warmup_transition_resets_only_interval_gradient_statistics(self):
        model, _ = initialized_model()
        model.offset_gradient_accum.fill_(3.0)
        model.offset_denom.fill_(4.0)
        model.opacity_accum.fill_(5.0)
        model.anchor_demon.fill_(6.0)
        model.anchor_visible_count.fill_(7.0)
        model.anchor_birth_iteration.copy_(torch.tensor([8, 9, 10]))
        preserved = {
            "opacity_accum": model.opacity_accum.clone(),
            "anchor_demon": model.anchor_demon.clone(),
            "anchor_visible_count": model.anchor_visible_count.clone(),
            "anchor_birth_iteration": model.anchor_birth_iteration.clone(),
        }
        summary = model.reset_densification_gradient_statistics()
        self.assertEqual(summary["observed_offsets"], 6)
        self.assertEqual(summary["observation_count"], 24.0)
        self.assertEqual(int(torch.count_nonzero(model.offset_gradient_accum)), 0)
        self.assertEqual(int(torch.count_nonzero(model.offset_denom)), 0)
        for name, expected in preserved.items():
            self.assertTrue(torch.equal(getattr(model, name), expected), name)

    def test_checkpoint_training_stage_marks_boundary_transition_exactly_once(self):
        at_boundary = training_stage_state(2000, True, 2000)
        after_boundary = training_stage_state(2001, True, 2000)
        disabled = training_stage_state(2000, False, 2000)
        self.assertFalse(at_boundary["transition_completed"])
        self.assertTrue(after_boundary["transition_completed"])
        self.assertEqual(disabled, {
            "warmup_enabled": False,
            "warmup_iterations": 0,
            "transition_completed": True,
        })

    def test_model_level_grow_and_prune_keep_appearance_aligned(self):
        model, _ = initialized_model()
        with torch.no_grad():
            model._offset[0, 0] = torch.tensor([2.0, 0.0, 0.0])
        gradients = torch.zeros(6)
        gradients[0] = 1.0
        offset_mask = torch.zeros(6, dtype=torch.bool)
        offset_mask[0] = True
        statistics = model.anchor_growing(
            gradients,
            threshold=0.1,
            offset_mask=offset_mask,
            max_anchors=10,
            max_new_anchors=1,
            level_caps=(1,),
        )
        self.assertEqual(statistics["appearance_params_inherited"], 1)
        self.assertEqual(model.get_anchor.shape[0], 4)
        self.assertTrue(all(
            getattr(model.appearance, name).shape[0] == 4
            for name in model.appearance.PARAMETER_NAMES
        ))
        axis, tangent, _ = orthonormal_frame(
            model.appearance.main_asg_axis,
            model.appearance.main_asg_tangent,
        )
        self.assertTrue(torch.allclose(axis.norm(dim=-1), torch.ones((4, 2)), atol=1e-6))
        self.assertTrue(torch.allclose((axis * tangent).sum(-1), torch.zeros((4, 2)), atol=1e-6))

        model.prune_anchor(torch.tensor([False, True, False, False]))
        self.assertEqual(model.get_anchor.shape[0], 3)
        self.assertTrue(all(
            getattr(model.appearance, name).shape[0] == 3
            for name in model.appearance.PARAMETER_NAMES
        ))

    def test_ply_and_checkpoint_round_trip(self):
        model, optimization = initialized_model()
        model.anchor_birth_iteration.copy_(torch.tensor([1, 2, 3]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply_path = root / "point_cloud.ply"
            model.save_ply(str(ply_path))
            loaded = GaussianModel(feat_dim=8, n_offsets=2, device="cpu")
            loaded.load_ply_sparse_gaussian(str(ply_path))
            self.assertTrue(torch.equal(model._anchor, loaded._anchor))
            self.assertTrue(torch.equal(model.appearance.reflectance, loaded.appearance.reflectance))

            mlp_root = root / "core_mlps"
            model.save_mlp_checkpoints(str(mlp_root))
            loaded.load_mlp_checkpoints(str(mlp_root))
            scaffold_input = torch.rand((2, 11))
            self.assertTrue(torch.equal(
                model.mlp_opacity(scaffold_input),
                loaded.mlp_opacity(scaffold_input),
            ))
            self.assertTrue(torch.equal(
                model.mlp_cov(scaffold_input),
                loaded.mlp_cov(scaffold_input),
            ))

            state = model.capture()
            restored = GaussianModel(feat_dim=8, n_offsets=2, device="cpu")
            restored.restore(state, optimization)
            self.assertTrue(torch.equal(model._anchor, restored._anchor))
            self.assertTrue(torch.equal(model.appearance.reflectance, restored.appearance.reflectance))
            self.assertTrue(torch.equal(
                model.anchor_birth_iteration,
                restored.anchor_birth_iteration,
            ))
            resume_loss = (
                restored._anchor.sum()
                + restored.appearance.reflectance.sum()
                + restored.mlp_opacity(torch.rand((2, 11))).sum()
                + restored.mlp_cov(torch.rand((2, 11))).sum()
            )
            resume_loss.backward()
            restored.optimizer.step()
            self.assertEqual(set(state), {
                "model_format_version",
                "scaffold",
                "explicit_appearance",
                "core_mlps",
                "optimizer",
            })

            missing_pose = dict(state)
            missing_pose["scaffold"] = dict(state["scaffold"])
            missing_pose["scaffold"].pop("poses")
            with self.assertRaisesRegex(ValueError, "Scaffold fields"):
                GaussianModel(feat_dim=8, n_offsets=2, device="cpu").restore(
                    missing_pose,
                    optimization,
                )

    def test_legacy_checkpoint_and_ply_are_rejected(self):
        model = GaussianModel(feat_dim=8, n_offsets=2, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "MODEL_FORMAT_VERSION=2"):
            model._require_v2(("legacy",), "checkpoint")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.ply"
            vertex = np.empty(1, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            vertex[0] = (0.0, 0.0, 0.0)
            PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))
            with self.assertRaisesRegex(RuntimeError, "legacy PLY"):
                model.load_ply_sparse_gaussian(str(path))


if __name__ == "__main__":
    unittest.main()
