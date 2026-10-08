"""Native CUDA smoke test; CPU-only environments report an explicit skip."""

import unittest
from types import SimpleNamespace

import torch

from utils.graphics_utils import getProjectionMatrix
from utils.loss_utils import L_Illu, L_Smooth, L_Reflectance_Edge_Uplift, L_Reflectance_HighFreq, L_Reflectance_LocalContrast


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA PyTorch and project rasterizer extensions")
class NativeRasterizerGradientTest(unittest.TestCase):
    def test_reference_enhancement_network_survives_anchor_growth_and_prune(self):
        from scene.gaussian_model import GaussianModel

        model = GaussianModel(feat_dim=8, n_offsets=2, appearance_residual_dim=0,
                              illumination_mode="sg", supervision_profile="llgaussian")
        network_parameters = list(model.enhancement_net.parameters())
        model.optimizer = torch.optim.Adam([
            {"params": [torch.nn.Parameter(torch.rand(2, 3, device="cuda"))], "name": "anchor"},
            {"params": network_parameters, "name": "enhancement_net"},
        ])
        grown = model.cat_tensors_to_optimizer({"anchor": torch.zeros(1, 3, device="cuda")})
        self.assertEqual(grown["anchor"].shape[0], 3)
        pruned = model._prune_anchor_optimizer(torch.tensor([True, False, True], device="cuda"))
        self.assertEqual(pruned["anchor"].shape[0], 2)
        for expected, actual in zip(network_parameters, model.optimizer.param_groups[1]["params"]):
            self.assertIs(expected, actual)

    def test_auxiliary_freezes_geometry_and_main_reconstruction_trains_it(self):
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel

        for mode in ("explicit", "mlp"):
            with self.subTest(reflectance_mode=mode):
                torch.manual_seed(19)
                model = GaussianModel(feat_dim=8, n_offsets=2, appearance_residual_dim=0,
                                      illumination_mode="mlp", reflectance_mode=mode)
                model._anchor = torch.nn.Parameter(torch.tensor([[-0.2, 0., 2.], [0.2, 0.1, 2.5]], device="cuda"))
                model._anchor_feat = torch.nn.Parameter(torch.rand(2, 8, device="cuda"))
                model._offset = torch.nn.Parameter(torch.zeros(2, 2, 3, device="cuda"))
                model._scaling = torch.nn.Parameter(torch.full((2, 6), -1.5, device="cuda"))
                model._base_log_reflectance = torch.nn.Parameter(torch.full((2, 3), -0.7, device="cuda"))
                model._reflectance_offset_delta = torch.nn.Parameter(torch.zeros(2, 2, 3, device="cuda"))
                with torch.no_grad():
                    model.mlp_opacity[-2].weight.zero_()
                    model.mlp_opacity[-2].bias.fill_(0.5)
                pose = torch.nn.Parameter(torch.tensor([1., 0., 0., 0., 0., 0., 0.], device="cuda"))
                camera = SimpleNamespace(
                    camera_center=torch.zeros(3, device="cuda"), FoVx=1.5, FoVy=1.5,
                    image_height=16, image_width=16,
                    projection_matrix=getProjectionMatrix(0.01, 100, 1.5, 1.5).T.cuda(),
                )
                package = render(camera, model, SimpleNamespace(debug=False), torch.zeros(3, device="cuda"),
                                 kernel_size=0.1, camera_pose=pose, retain_grad=True,
                                 return_reflectance_aux=True, return_enhanced_reflectance_aux=True)
                torch.testing.assert_close(package["render_reflectance_aux"], package["render_reflectance"])
                torch.testing.assert_close(package["render_enhanced_reflectance_aux"], package["render_enhanced"])
                target = torch.rand(3, 16, 16, device="cuda")
                auxiliary = sum(function(package["render_reflectance_aux"], target) for function in (
                    L_Reflectance_Edge_Uplift, L_Reflectance_LocalContrast, L_Reflectance_HighFreq,
                ))
                auxiliary.backward(retain_graph=True)
                for parameter in (model._anchor, model._offset, model._scaling, model._anchor_feat, pose):
                    self.assertIsNone(parameter.grad)
                self.assertIsNone(package["viewspace_points"].grad)
                appearance = [model._base_log_reflectance, model._reflectance_offset_delta] if mode == "explicit" else list(model.mlp_reflectance.parameters())
                self.assertGreater(sum(p.grad.abs().sum().item() for p in appearance if p.grad is not None), 0)
                (package["render"] - target).square().mean().backward()
                self.assertIsNotNone(package["viewspace_points"].grad)
                self.assertGreater(model._anchor.grad.abs().sum().item(), 0)

    def test_illumination_image_prior_freezes_geometry_in_all_modes(self):
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel

        for mode in ("sg", "asg", "mlp"):
            with self.subTest(illumination_mode=mode):
                torch.manual_seed(29)
                model = GaussianModel(feat_dim=8, n_offsets=2, appearance_residual_dim=0,
                                      illumination_mode=mode, reflectance_mode="explicit")
                model._anchor = torch.nn.Parameter(torch.tensor([[-0.2, 0., 2.], [0.2, 0.1, 2.5]], device="cuda"))
                model._anchor_feat = torch.nn.Parameter(torch.rand(2, 8, device="cuda"))
                model._offset = torch.nn.Parameter(torch.zeros(2, 2, 3, device="cuda"))
                model._scaling = torch.nn.Parameter(torch.full((2, 6), -1.5, device="cuda"))
                model._base_log_reflectance = torch.nn.Parameter(torch.full((2, 3), -0.7, device="cuda"))
                model._reflectance_offset_delta = torch.nn.Parameter(torch.zeros(2, 2, 3, device="cuda"))
                model._ensure_illumination_asg_params()
                with torch.no_grad():
                    model.mlp_opacity[-2].weight.zero_()
                    model.mlp_opacity[-2].bias.fill_(0.5)
                pose = torch.nn.Parameter(torch.tensor([1., 0., 0., 0., 0., 0., 0.], device="cuda"))
                camera = SimpleNamespace(camera_center=torch.zeros(3, device="cuda"), FoVx=1.5, FoVy=1.5,
                    image_height=16, image_width=16,
                    projection_matrix=getProjectionMatrix(0.01, 100, 1.5, 1.5).T.cuda())
                package = render(camera, model, SimpleNamespace(debug=False), torch.zeros(3, device="cuda"),
                    kernel_size=0.1, camera_pose=pose, retain_grad=True, return_illumination_aux=True,
                    return_reflectance_aux=True, geometry_only=True)
                torch.testing.assert_close(package["render_reconstruction_aux"], package["render"])
                torch.testing.assert_close(package["render_illumination_aux"], package["render_illumination"])
                target = torch.rand(3, 16, 16, device="cuda")
                (L_Illu(target, package["render_illumination_aux"])
                 + L_Smooth(package["render_illumination_aux"], target, kernel_size=5)).backward(retain_graph=True)
                for parameter in (model._anchor, model._offset, model._scaling, model._anchor_feat, pose,
                                  model._base_log_reflectance, model._reflectance_offset_delta):
                    self.assertIsNone(parameter.grad)
                for network in (model.mlp_cov, model.mlp_opacity, model.mlp_reflectance_decoder):
                    self.assertTrue(all(parameter.grad is None for parameter in network.parameters()))
                self.assertIsNone(package["viewspace_points"].grad)
                active = (list(model.mlp_sg_illumination.parameters()) if mode == "sg" else
                          [model._illum_asg_bias, model._illum_asg_amplitude] if mode == "asg" else
                          list(model.mlp_illumination.parameters()))
                self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in active if parameter.grad is not None), 0)
                for parameter in active:
                    parameter.grad = None
                photo = (package["render_reconstruction_aux"] - target).square().mean()
                photo.backward(retain_graph=True)
                for parameter in (model._anchor, model._offset, model._scaling, model._anchor_feat, pose):
                    self.assertIsNone(parameter.grad)
                self.assertIsNone(package["viewspace_points"].grad)
                self.assertGreater(model._base_log_reflectance.grad.abs().sum().item(), 0)
                model._base_log_reflectance.grad = None
                model._reflectance_offset_delta.grad = None
                for parameter in active:
                    parameter.grad = None
                (package["render"] - target).square().mean().backward()
                # The renderer packs colors/covariances into one cat/split
                # tensor; its unused color slices can receive exact zeros.
                self.assertTrue(model._base_log_reflectance.grad is None or
                                torch.count_nonzero(model._base_log_reflectance.grad) == 0)
                self.assertTrue(all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                                    for parameter in active))
                self.assertGreater(model._anchor.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
