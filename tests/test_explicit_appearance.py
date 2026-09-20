import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scene.explicit_appearance import (
    ExplicitAppearance,
    _inverse_softplus,
    _safe_logit,
    decompose_enhanced_prior,
    orthonormal_frame,
)
from utils.model_format import EXPLICIT_APPEARANCE_FORMAT_VERSION, NUMERICAL_EPS
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.loss_utils import (
    coverage_masked_prediction,
    edge_aware_illumination_tv_loss,
    photo_loss,
    retinex_targets,
)


def fallback_camera():
    return SimpleNamespace(
        original_image=torch.full((3, 5, 5), 0.5),
        enhancement_prior=torch.full((3, 5, 5), 0.5),
        full_proj_transform=torch.eye(4),
        camera_center=torch.tensor([0.0, 0.0, -1.0]),
    )


def make_appearance(anchor_count=2, n_offsets=3):
    appearance = ExplicitAppearance(n_offsets)
    anchors = torch.zeros((anchor_count, 3))
    offsets = torch.zeros((anchor_count, n_offsets, 3))
    scaling = torch.ones((anchor_count, 6))
    appearance.initialize(anchors, offsets, scaling, cameras=[fallback_camera()])
    return appearance


class ExplicitAppearanceTest(unittest.TestCase):
    def test_reflectance_is_bounded_and_detail_is_zero_mean(self):
        appearance = make_appearance()
        with torch.no_grad():
            appearance.reflectance_detail.copy_(
                torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
            )
        reflectance = appearance.reflectance
        centered = appearance.reflectance_detail - appearance.reflectance_detail.mean(1, keepdim=True)
        self.assertEqual(tuple(reflectance.shape), (2, 3, 3))
        self.assertTrue(torch.all((reflectance >= 0.0) & (reflectance <= 1.0)))
        self.assertTrue(torch.allclose(centered.mean(1), torch.zeros((2, 3))))

    def test_asg_and_enhanced_outputs_are_bounded_and_deterministic(self):
        appearance = make_appearance()
        with torch.no_grad():
            appearance.enhanced_diffuse_raw.fill_(2.0)
            appearance.enhanced_sg_energy.fill_(2.0)
        directions = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        first = appearance.evaluate(directions)
        second = appearance.evaluate(directions)
        for field in first.__dataclass_fields__:
            value = getattr(first, field)
            repeated = getattr(second, field)
            self.assertTrue(torch.equal(value, repeated))
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.all((value >= 0.0) & (value <= 1.0)))

        sum(getattr(first, field).sum() for field in first.__dataclass_fields__).backward()
        for parameter in appearance.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_prior_decomposition_is_finite_for_zero_reflectance(self):
        reflectance = torch.tensor([0.0, 0.25, 1.0])
        prior = torch.tensor([0.8, 0.8, 0.8])
        diffuse, fill = decompose_enhanced_prior(prior, reflectance)
        base = reflectance * diffuse
        reconstructed = base + (1.0 - base) * fill
        illumination_target = diffuse + (1.0 - diffuse) * fill
        for value in (diffuse, fill, reconstructed, illumination_target):
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.all((value >= 0.0) & (value <= 1.0)))
        self.assertTrue(torch.allclose(reconstructed, prior, atol=NUMERICAL_EPS))

    def test_inverse_softplus_is_finite_across_the_physical_gain_range(self):
        values = torch.tensor([NUMERICAL_EPS, 1e-4, 0.5, 1.0, 10.0, 1e6])
        raw = _inverse_softplus(values)
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue(
            torch.allclose(
                torch.nn.functional.softplus(raw),
                values,
                rtol=1e-5,
                atol=NUMERICAL_EPS,
            )
        )

    def test_degenerate_frame_has_deterministic_orthonormal_fallback(self):
        zeros = torch.zeros((4, 3))
        axis, tangent, bitangent = orthonormal_frame(zeros, zeros)
        for vector in (axis, tangent, bitangent):
            self.assertTrue(torch.allclose(vector.norm(dim=-1), torch.ones(4)))
        self.assertTrue(torch.allclose((axis * tangent).sum(-1), torch.zeros(4)))
        self.assertTrue(torch.allclose((axis * bitangent).sum(-1), torch.zeros(4)))
        self.assertTrue(torch.equal(axis, axis[0].expand_as(axis)))
        self.assertTrue(torch.equal(tangent, tangent[0].expand_as(tangent)))

    def test_multiview_initialization_and_unobserved_fallback(self):
        low_a = torch.tensor([0.2, 0.4, 0.8]).view(3, 1, 1).expand(3, 5, 5)
        low_b = torch.tensor([0.3, 0.6, 0.9]).view(3, 1, 1).expand(3, 5, 5)
        enhanced = torch.tensor([0.7, 0.8, 0.9]).view(3, 1, 1).expand(3, 5, 5)
        cameras = [
            SimpleNamespace(
                original_image=image,
                enhancement_prior=enhanced,
                full_proj_transform=torch.eye(4),
                camera_center=torch.tensor([0.0, 0.0, -1.0]),
            )
            for image in (low_a, low_b)
        ]
        appearance = ExplicitAppearance(2)
        anchors = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        appearance.initialize(
            anchors,
            torch.zeros((2, 2, 3)),
            torch.ones((2, 6)),
            cameras,
        )
        self.assertTrue(torch.isfinite(appearance.reflectance).all())
        self.assertTrue(torch.isfinite(appearance.main_asg_axis).all())
        self.assertTrue(torch.allclose(
            appearance.reflectance_detail.mean(dim=1),
            torch.zeros((2, 3)),
            atol=1e-6,
        ))
        self.assertGreater(float(appearance.main_asg_axis[0, 0, 2]), 0.99)
        for name in appearance.PARAMETER_NAMES:
            self.assertTrue(
                torch.allclose(getattr(appearance, name)[1], getattr(appearance, name)[0]),
                name,
            )

    def test_projection_visibility_keeps_only_the_frontmost_sample_per_pixel(self):
        appearance = ExplicitAppearance(1)
        positions = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.5]],
            ]
        )
        camera = fallback_camera()
        frontmost = appearance._frontmost_visibility_masks(
            positions,
            [camera],
            chunk_size=1,
        )
        _, _, _, valid = appearance._sample_views(
            positions,
            [camera],
            frontmost,
        )
        self.assertEqual(valid.shape, (1, 2, 1))
        self.assertTrue(torch.equal(valid, frontmost))
        self.assertTrue(bool(valid[0, 0, 0]))
        self.assertFalse(bool(valid[0, 1, 0]))

    def test_projection_uses_camera_row_vector_matrix_convention(self):
        world_view = torch.tensor(
            getWorld2View2(
                np.eye(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )
        ).T
        projection = getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fovX=math.pi / 2.0,
            fovY=math.pi / 2.0,
        ).T
        camera = SimpleNamespace(
            full_proj_transform=world_view @ projection,
            camera_center=torch.linalg.inv(world_view)[3, :3],
        )
        x, y, valid, projected_depth, _ = ExplicitAppearance._project_camera(
            torch.tensor([[0.0, 0.0, 1.0]]),
            camera,
            height=5,
            width=5,
        )
        self.assertTrue(bool(valid[0]))
        self.assertEqual((int(x[0]), int(y[0])), (2, 2))
        self.assertGreater(float(projected_depth[0]), 0.0)
        self.assertLess(float(projected_depth[0]), 1.0)

    def test_visibility_uses_projected_depth_instead_of_euclidean_distance(self):
        world_view = torch.tensor(
            getWorld2View2(
                np.eye(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )
        ).T
        projection = getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fovX=math.pi / 2.0,
            fovY=math.pi / 2.0,
        ).T
        camera = SimpleNamespace(
            original_image=torch.ones((3, 3, 3)),
            full_proj_transform=world_view @ projection,
            camera_center=torch.linalg.inv(world_view)[3, :3],
        )
        # Both points round to the center pixel.  The first has smaller camera
        # depth but a larger Euclidean distance because of its lateral offset.
        positions = torch.tensor(
            [
                [[0.49, 0.0, 1.0]],
                [[0.0, 0.0, 1.05]],
            ]
        )
        euclidean = (positions[:, 0] - camera.camera_center).norm(dim=-1)
        self.assertGreater(float(euclidean[0]), float(euclidean[1]))
        visible = ExplicitAppearance(1)._frontmost_visibility_masks(
            positions,
            [camera],
            chunk_size=1,
        )
        self.assertTrue(bool(visible[0, 0, 0]))
        self.assertFalse(bool(visible[0, 1, 0]))

    def test_enhanced_initialization_reconstructs_prior_without_gain_division(self):
        low = torch.tensor([0.2, 0.1, 0.05]).view(3, 1, 1).expand(3, 5, 5)
        prior = torch.full((3, 5, 5), 0.8)
        camera = SimpleNamespace(
            original_image=low,
            enhancement_prior=prior,
            full_proj_transform=torch.eye(4),
            camera_center=torch.tensor([0.0, 0.0, -1.0]),
        )
        appearance = ExplicitAppearance(1)
        appearance.initialize(
            torch.zeros((1, 3)),
            torch.zeros((1, 1, 3)),
            torch.ones((1, 6)),
            [camera],
        )
        diffuse = torch.sigmoid(appearance.enhanced_diffuse_raw)
        self.assertTrue(torch.allclose(
            diffuse,
            torch.tensor([0.8, 1.0, 1.0]).view(1, 1, 3),
            atol=5e-6,
        ))
        evaluated = appearance.evaluate(
            torch.tensor([[0.0, 0.0, 1.0]])
        )
        self.assertTrue(
            torch.allclose(
                evaluated.enhanced_color,
                torch.full_like(evaluated.reflectance, 0.8),
                atol=5e-6,
            )
        )

    def test_gradient_ownership_of_explicit_branches(self):
        appearance = make_appearance(anchor_count=1, n_offsets=2)
        view = torch.tensor([[0.2, 0.0, 1.0]], requires_grad=True)
        evaluated = appearance.evaluate(view)
        (evaluated.enhanced_color.sum() + evaluated.illumination_enhanced.sum()).backward()
        enhanced_names = {
            "enhanced_sg_axis",
            "enhanced_sg_sharpness",
            "enhanced_sg_energy",
            "enhanced_diffuse_raw",
        }
        for name, parameter in appearance.named_parameters():
            if name in enhanced_names:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertIsNone(view.grad)

        appearance.zero_grad(set_to_none=True)
        view.grad = None
        evaluated = appearance.evaluate(view)
        (evaluated.reflectance * evaluated.illumination).sum().backward()
        for name, parameter in appearance.named_parameters():
            if name in enhanced_names:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
        self.assertIsNotNone(view.grad)

    def test_reflectance_reconstruction_and_illumination_own_disjoint_parameters(self):
        appearance = make_appearance(anchor_count=1, n_offsets=2)
        view = torch.tensor([[0.0, 0.0, 1.0]])
        evaluated = appearance.evaluate(view)
        reflectance_image = evaluated.reflectance[0].T.unsqueeze(1)
        illumination_image = evaluated.illumination[0].T.unsqueeze(1).expand(3, -1, -1)
        coverage = torch.ones_like(reflectance_image)
        target = torch.rand_like(reflectance_image)
        reflectance_prediction = coverage_masked_prediction(
            reflectance_image * illumination_image.detach(), target, coverage
        )
        photo_loss(reflectance_prediction, target, 0.2).backward()
        reflectance_names = {"reflectance_base", "reflectance_detail"}
        main_names = {
            "main_asg_axis",
            "main_asg_tangent",
            "main_asg_sharpness",
            "main_asg_energy",
            "main_asg_ambient",
        }
        for name, parameter in appearance.named_parameters():
            if name in reflectance_names:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)

        appearance.zero_grad(set_to_none=True)
        evaluated = appearance.evaluate(view)
        illumination_image = evaluated.illumination[0].T.unsqueeze(1).expand(3, -1, -1)
        coverage = torch.ones_like(illumination_image)
        target = torch.rand_like(illumination_image)
        max_rgb, _ = retinex_targets(target)
        illumination_target = max_rgb.expand_as(illumination_image)
        illumination_prediction = coverage_masked_prediction(
            illumination_image, illumination_target, coverage
        )
        illumination_loss = photo_loss(illumination_prediction, illumination_target, 0.2)
        illumination_loss = illumination_loss + edge_aware_illumination_tv_loss(
            illumination_image, max_rgb, coverage
        )
        illumination_loss.backward()
        for name, parameter in appearance.named_parameters():
            if name in main_names:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_grow_prune_and_property_round_trip(self):
        appearance = make_appearance(anchor_count=3, n_offsets=2)
        with torch.no_grad():
            appearance.main_asg_axis[1].neg_()
            appearance.reflectance_base[0].fill_(-2.0)
            appearance.reflectance_base[1].fill_(2.0)
            appearance.reflectance_detail[0].zero_()
            appearance.reflectance_detail[1].zero_()
            appearance.main_asg_sharpness[0].fill_(-10.0)
            appearance.main_asg_sharpness[1].fill_(10.0)
            appearance.main_asg_energy[0].fill_(-10.0)
            appearance.main_asg_energy[1].fill_(10.0)
            appearance.enhanced_sg_axis[0].copy_(
                torch.tensor([100.0, 0.0, 0.0]).expand(2, 3)
            )
            appearance.enhanced_sg_axis[1].copy_(
                torch.tensor([0.0, 1.0, 0.0]).expand(2, 3)
            )
            appearance.enhanced_sg_sharpness[0].fill_(-10.0)
            appearance.enhanced_sg_sharpness[1].fill_(10.0)
            physical_diffuse = torch.tensor([0.2, 0.6, 0.8]).view(3, 1, 1)
            physical_energy = torch.tensor([0.1, 0.5, 0.9]).view(3, 1, 1)
            appearance.enhanced_diffuse_raw.copy_(
                _safe_logit(physical_diffuse.expand(3, 2, 3))
            )
            appearance.enhanced_sg_energy.copy_(
                _safe_logit(physical_energy.expand(3, 2, 3))
            )
        inherited = appearance.inherited_parameters(
            torch.tensor([True, False, True, False, False, True]),
            torch.tensor([0, 0, 1]),
            torch.tensor([True, True]),
        )
        detail = inherited["appearance_reflectance_detail"]
        axis = inherited["appearance_main_asg_axis"]
        tangent = inherited["appearance_main_asg_tangent"]
        self.assertTrue(torch.allclose(detail.mean(1), torch.zeros((2, 3)), atol=1e-6))
        self.assertTrue(torch.allclose(axis.norm(dim=-1), torch.ones((2, 2)), atol=1e-6))
        self.assertTrue(torch.allclose((axis * tangent).sum(-1), torch.zeros((2, 2)), atol=1e-6))
        inherited_reflectance = torch.sigmoid(
            inherited["appearance_reflectance_base"][:, None, :]
            + inherited["appearance_reflectance_detail"]
        )
        expected_reflectance = torch.stack(
            [
                appearance.reflectance[[0, 1]].mean(dim=0),
                appearance.reflectance[2],
            ]
        )
        self.assertTrue(
            torch.allclose(inherited_reflectance, expected_reflectance, atol=1e-6)
        )
        expected_axis = torch.tensor([2.0**-0.5, 2.0**-0.5, 0.0])
        self.assertTrue(
            torch.allclose(
                inherited["appearance_enhanced_sg_axis"][0, 0],
                expected_axis,
                atol=1e-6,
            )
        )
        expected_sharpness = (
            torch.nn.functional.softplus(torch.tensor(-10.0))
            + torch.nn.functional.softplus(torch.tensor(10.0))
        ) / 2.0
        self.assertTrue(
            torch.allclose(
                torch.nn.functional.softplus(
                    inherited["appearance_enhanced_sg_sharpness"][0]
                ),
                torch.full((2, 1), expected_sharpness),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.nn.functional.softplus(
                    inherited["appearance_main_asg_sharpness"][0]
                ),
                torch.full((2, 2), expected_sharpness),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.sigmoid(inherited["appearance_main_asg_energy"][0]),
                torch.full((2, 1), 0.5),
                atol=1e-6,
            )
        )
        inherited_diffuse = torch.sigmoid(
            inherited["appearance_enhanced_diffuse_raw"]
        )
        inherited_energy = torch.sigmoid(
            inherited["appearance_enhanced_sg_energy"]
        )
        self.assertTrue(
            torch.allclose(
                inherited_diffuse[:, 0, 0],
                torch.tensor([0.4, 0.8]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                inherited_energy[:, 0, 0],
                torch.tensor([0.3, 0.9]),
                atol=1e-6,
            )
        )

        names = appearance.ply_attribute_names()
        values = appearance.ply_values()
        loaded = ExplicitAppearance(2)
        properties = {name: values[:, index] for index, name in enumerate(names)}
        loaded.load_ply_values(properties)
        for name in appearance.PARAMETER_NAMES:
            self.assertTrue(torch.equal(getattr(appearance, name), getattr(loaded, name)))
        old_properties = dict(properties)
        old_properties.pop("appearance_layout_version")
        with self.assertRaisesRegex(RuntimeError, "no explicit appearance"):
            ExplicitAppearance(2).load_ply_values(old_properties)
        old_layout = dict(properties)
        old_layout["appearance_layout_version"] = torch.ones_like(
            old_layout["appearance_layout_version"]
        )
        with self.assertRaisesRegex(RuntimeError, "parameterization is incompatible"):
            ExplicitAppearance(2).load_ply_values(old_layout)
        loaded.prune_without_optimizer(torch.tensor([True, False, True]))
        self.assertEqual(loaded.anchor_count, 2)

    def test_v2_state_rejects_missing_or_extra_fields(self):
        appearance = make_appearance(anchor_count=1, n_offsets=2)
        state = appearance.state_dict_v2()
        appearance.load_state_dict_v2(state)
        old_bounded_state = dict(state)
        old_bounded_state.pop("explicit_appearance_format_version")
        with self.assertRaisesRegex(ValueError, "appearance fields"):
            appearance.load_state_dict_v2(old_bounded_state)
        wrong_parameterization = dict(state)
        wrong_parameterization["explicit_appearance_format_version"] = 1
        with self.assertRaisesRegex(RuntimeError, "parameterization is incompatible"):
            appearance.load_state_dict_v2(wrong_parameterization)
        with self.assertRaises(ValueError):
            appearance.load_state_dict_v2({"reflectance_base": state["reflectance_base"]})
        bad = dict(state)
        bad["legacy_decoder"] = torch.empty(0)
        with self.assertRaises(ValueError):
            appearance.load_state_dict_v2(bad)
        wrong_shape = dict(state)
        wrong_shape["enhanced_sg_energy"] = torch.empty((1, 2, 1))
        with self.assertRaisesRegex(ValueError, "appearance shape"):
            appearance.load_state_dict_v2(wrong_shape)

    def test_missing_training_views_are_rejected(self):
        appearance = ExplicitAppearance(1)
        with self.assertRaisesRegex(ValueError, "requires training cameras"):
            appearance.initialize(
                torch.zeros((1, 3)),
                torch.zeros((1, 1, 3)),
                torch.ones((1, 6)),
                cameras=[],
            )


if __name__ == "__main__":
    unittest.main()
