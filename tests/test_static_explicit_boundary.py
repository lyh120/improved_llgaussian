import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticExplicitBoundaryTest(unittest.TestCase):
    def test_only_scaffold_allowlist_contains_linear_layers(self):
        model_source = (ROOT / "scene" / "gaussian_model_v2.py").read_text(encoding="utf-8")
        explicit_source = (ROOT / "scene" / "explicit_appearance.py").read_text(encoding="utf-8")
        self.assertNotIn("nn.Linear", explicit_source)
        tree = ast.parse(model_source)
        allowed = {"mlp_opacity", "mlp_cov", "mlp_feature_bank"}
        linear_owners = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "Linear"
                for child in ast.walk(node.value)
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    linear_owners.add(target.attr)
        self.assertEqual(linear_owners, allowed)

    def test_removed_appearance_networks_and_modes_are_absent(self):
        paths = [
            ROOT / "scene" / "gaussian_model_v2.py",
            ROOT / "gaussian_renderer" / "__init__.py",
            ROOT / "train.py",
            ROOT / "arguments" / "__init__.py",
        ]
        forbidden = (
            "mlp_reflectance",
            "mlp_illumination",
            "mlp_color",
            "reflectance_decoder",
            "appearance_embedding",
            "illumination_mode",
            "legacy_compatibility",
            "render_residual",
        )
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for identifier in forbidden:
            self.assertNotIn(identifier, source)

    def test_renderer_detaches_geometry_from_appearance_view_directions(self):
        renderer = (ROOT / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("view_dirs.detach()", renderer)
        self.assertIn(
            "effective_camera_center=get_camera_center_from_tensor(pose)",
            renderer,
        )

    def test_renderer_unpremultiplies_diagnostic_attributes_and_depth(self):
        renderer = (ROOT / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")
        for value in (
            "reflectance_accumulated",
            "illumination_accumulated",
            "enhanced_illumination_accumulated",
            "depth_accumulated",
        ):
            self.assertIn(f"_normalize_by_coverage(\n        {value},", renderer)
        self.assertIn('"render_depth": rendered_depth', renderer)
        self.assertIn('"_training_coverage": detached_coverage', renderer)

    def test_wandb_logs_only_the_five_core_decomposition_images(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fields = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "WANDB_CORE_IMAGE_FIELDS"
                for target in node.targets
            ):
                fields = ast.literal_eval(node.value)
        self.assertEqual(fields, {
            "image": "render",
            "image_enhanced": "render_enhanced",
            "reflectance": "render_reflectance",
            "illumination": "render_illumination",
            "illumination_enhanced": "render_illumination_enhanced",
        })
        for removed in (
            "gt_lowlight",
            "base_brightened",
            "cidnet_target",
            "residual_image",
            "artifact_image",
            "coverage_image",
        ):
            self.assertNotIn(removed, source)

    def test_diagnostic_rasterization_is_fully_detached(self):
        """Every appearance-supervising rasterization must use detached geometry."""
        renderer_source = (
            ROOT / "gaussian_renderer" / "__init__.py"
        ).read_text(encoding="utf-8")
        # reflectance / illumination / enhanced-illumination diagnostics + the
        # enhanced rasterization; the main composite and depth passes stay live
        # because they own geometry supervision by design.
        self.assertEqual(renderer_source.count("**detached_geometry"), 4)
        for fragment in (
            '"means3D": means3d.detach()',
            '"means2D": means2d.detach()',
            '"opacities": opacity.detach()',
            '"scales": scaling.detach()',
            '"rotations": transformed_rotation.detach()',
        ):
            self.assertIn(fragment, renderer_source)

    def test_warmup_runs_on_top_of_the_main_budget(self):
        """Warmup and main each count their own 1..N (reference two-run mode)."""
        train_source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn(
            "total_iterations = optimization.iterations + warmup_total",
            train_source,
        )
        self.assertIn(
            "range(first_iteration + 1, total_iterations + 1)",
            train_source,
        )
        self.assertIn(
            "phase_iteration = iteration if warmup_phase else iteration - warmup_total",
            train_source,
        )
        for fragment in (
            "gaussians.update_learning_rate(\n            phase_iteration,",
            "retain_grad = phase_iteration < stage_update_until",
            "if stage_start_stat < phase_iteration < stage_update_until:",
            "current_iteration=phase_iteration,",
            "if phase_iteration in testing_iterations:",
            "if phase_iteration in saving_iterations:",
            "scene.save(phase_iteration)",
        ):
            self.assertIn(fragment, train_source)

    def test_warmup_phase_and_conservative_8k_schedule_are_observable(self):
        train_source = (ROOT / "train.py").read_text(encoding="utf-8")
        for scalar in (
            '"phase/is_warmup"',
            '"phase/geometry_lr_scale"',
            '"densification/stat_collection_active"',
            '"densification/adjust_event"',
            '"densification/reset_event"',
        ):
            self.assertIn(scalar, train_source)
        self.assertIn("reset_densification_gradient_statistics()", train_source)
        script = (ROOT / "scripts" / "train.sh").read_text(encoding="utf-8")
        for fragment in (
            "--warmup_iterations 2000",
            "--warmup_update_from 1200",
            "--warmup_update_until 1900",
            "--warmup_update_interval 200",
            "--warmup_max_new_anchors 128",
            "--max_anchors 30000",
            "--max_new_anchors_per_update 256",
            "--densify_level_caps 128,80,48",
            "--max_pruned_anchors_per_update 0",
        ):
            self.assertIn(fragment, script)

    def test_enhancement_prior_backends_stay_outside_the_training_graph(self):
        train_source = (ROOT / "train.py").read_text(encoding="utf-8")
        args_source = (ROOT / "arguments" / "__init__.py").read_text(encoding="utf-8")
        stable_source = (
            ROOT / "scripts" / "precompute_stablesr_prior.py"
        ).read_text(encoding="utf-8")
        self.assertIn('ENHANCEMENT_PRIOR_BACKENDS = {"cidnet", "stablesr"}', train_source)
        self.assertIn('self.enhancement_prior_backend = "cidnet"', args_source)
        self.assertIn('self.stablesr_input_gain = 15.0', args_source)
        # StableSR uses only the explicit input gain. The clean-reference
        # brightness mapping is isolated to W&B visualization.
        self.assertNotIn("0.45 /", stable_source)
        self.assertNotIn("0.45 /", train_source)
        self.assertNotIn("scene_enhance_ratio", train_source)
        self.assertIn("WANDB_DISPLAY_TARGET_MEAN", train_source)
        self.assertIn('"prior_type": "enhancement_rgb"', stable_source)
        self.assertIn(
            '"--stablesr_checkpoint"',
            train_source,
        )
        self.assertIn(
            "dataset.stablesr_vqgan_checkpoint",
            train_source,
        )

    def test_training_objective_has_no_hidden_reference_loss_stack(self):
        train_source = (ROOT / "train.py").read_text(encoding="utf-8")
        loss_source = (ROOT / "utils" / "loss_utils.py").read_text(encoding="utf-8")
        for removed in (
            "DEPTH_SIMILARITY_WEIGHT",
            "REFLECTANCE_SMOOTH_WEIGHT",
            "CHROMA_CONSISTENCY_WEIGHT",
            "ENHANCED_SG_SPARSE_WEIGHT",
            "inverse_intensity_weighted_l1",
            "enhanced_prior_blend_target",
        ):
            self.assertNotIn(removed, train_source)
            self.assertNotIn(removed, loss_source)


if __name__ == "__main__":
    unittest.main()
