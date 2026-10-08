"""Native rasterizer tests for profile parity and exact auxiliary permissions."""

from types import SimpleNamespace
import unittest

import torch

from utils.gradient_utils import accumulate_auxiliary_gradients, REFLECTANCE_PARAM_GROUP_NAMES
from utils.graphics_utils import getProjectionMatrix
from utils.rl_compat_utils import (image_compat_terms, parameter_compat_terms,
                                   prepare_compat_target, weighted_compat_losses)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and native rasterizer")
class RLCompatCudaTest(unittest.TestCase):
    def create(self, profile, mode="sg"):
        from scene.gaussian_model import GaussianModel
        torch.manual_seed(23)
        model = GaussianModel(feat_dim=8, n_offsets=2, appearance_residual_dim=0,
                              illumination_mode=mode, reflectance_mode="explicit", supervision_profile=profile)
        model._anchor = torch.nn.Parameter(torch.tensor([[-.2, 0., 2.], [.2, .1, 2.5]], device="cuda"))
        model._anchor_feat = torch.nn.Parameter(torch.rand(2, 8, device="cuda"))
        model._offset = torch.nn.Parameter(torch.zeros(2, 2, 3, device="cuda"))
        model._scaling = torch.nn.Parameter(torch.full((2, 6), -1., device="cuda"))
        model._base_log_reflectance = torch.nn.Parameter(torch.full((2, 3), -.7, device="cuda"))
        model._reflectance_offset_delta = torch.nn.Parameter(torch.full((2, 2, 3), .1, device="cuda"))
        if mode == "asg":
            model._ensure_illumination_asg_params()
        with torch.no_grad():
            model.mlp_opacity[-2].weight.zero_(); model.mlp_opacity[-2].bias.fill_(.8)
        pose = torch.nn.Parameter(torch.tensor([1., 0., 0., 0., 0., 0., 0.], device="cuda"))
        camera = SimpleNamespace(camera_center=torch.zeros(3, device="cuda"), FoVx=1.5, FoVy=1.5,
                                 image_height=20, image_width=20,
                                 projection_matrix=getProjectionMatrix(.01, 100, 1.5, 1.5).T.cuda())
        return model, pose, camera

    def render(self, model, pose, camera, auxiliary=False):
        from gaussian_renderer import render
        return render(camera, model, SimpleNamespace(debug=False), torch.zeros(3, device="cuda"), .1,
                      camera_pose=pose, retain_grad=True, return_reflectance_aux=auxiliary,
                      return_sg_aux=auxiliary, return_coverage=auxiliary)

    def test_new_profile_matches_base_forward_and_gradients(self):
        records = []
        for profile in ("llgaussian", "llgaussian_rl"):
            model, pose, camera = self.create(profile)
            pkg = self.render(model, pose, camera)
            objective = pkg["render"].square().mean() + pkg["render_enhanced"].square().mean()
            objective.backward()
            records.append((pkg, model, pose))
        for key in ("render", "render_enhanced", "render_reflectance", "render_illumination", "render_depth"):
            torch.testing.assert_close(records[0][0][key], records[1][0][key], rtol=0, atol=0)
        for attribute in ("_anchor", "_offset", "_scaling", "_anchor_feat", "_base_log_reflectance", "_reflectance_offset_delta"):
            torch.testing.assert_close(getattr(records[0][1], attribute).grad,
                                       getattr(records[1][1], attribute).grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(records[0][2].grad, records[1][2].grad)

    def test_auxiliary_images_and_losses_freeze_geometry_and_other_branches(self):
        model, pose, camera = self.create("llgaussian_rl")
        pkg = self.render(model, pose, camera, True)
        torch.testing.assert_close(pkg["render_reflectance_aux"], pkg["render_reflectance"], rtol=0, atol=0)
        target = torch.full((3, 20, 20), .2, device="cuda")
        target[0] = .1
        # A full-coverage fixture tests the objective, independent of sparse scene coverage.
        terms = image_compat_terms(pkg["render_reflectance_aux"], prepare_compat_target(target), torch.ones_like(target))
        terms.update(parameter_compat_terms(model._base_log_reflectance, model._reflectance_offset_delta,
                                            model._last_reflectance_decoder_squared,
                                            pkg["illumination_aux_stats"]["sg_lambda_tail"]))
        r_loss, _ = weighted_compat_losses(terms, "stable", 2000)
        r_loss.backward(retain_graph=True)
        for p in (model._anchor, model._offset, model._scaling, model._anchor_feat, pose):
            self.assertIsNone(p.grad)
        self.assertIsNone(pkg["viewspace_points"].grad)
        for network in (model.mlp_cov, model.mlp_opacity, model.mlp_sg_illumination, model.enhancement_net):
            self.assertTrue(all(p.grad is None for p in network.parameters()))
        self.assertGreater(float(model._base_log_reflectance.grad.abs().sum()), 0)
        self.assertGreater(float(model._reflectance_offset_delta.grad.abs().sum()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.mlp_reflectance_decoder.parameters()), 0)
        (pkg["render"] - target).square().mean().backward()
        self.assertGreater(float(model._anchor.grad.abs().sum()), 0)
        self.assertGreater(float(pkg["viewspace_points"].grad.abs().sum()), 0)

    def test_sg_tail_updates_only_sg_head(self):
        model, pose, camera = self.create("llgaussian_rl")
        with torch.no_grad():
            model.mlp_sg_illumination[-1].bias.view(2, 4, 5)[:, :, 3] = 80.
        pkg = self.render(model, pose, camera, True)
        pkg["illumination_aux_stats"]["sg_lambda_tail"].backward()
        for p in (model._anchor, model._offset, model._scaling, model._anchor_feat, pose,
                  model._base_log_reflectance, model._reflectance_offset_delta):
            self.assertIsNone(p.grad)
        for network in (model.mlp_cov, model.mlp_opacity, model.mlp_reflectance_decoder, model.enhancement_net):
            self.assertTrue(all(p.grad is None for p in network.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.mlp_sg_illumination.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
