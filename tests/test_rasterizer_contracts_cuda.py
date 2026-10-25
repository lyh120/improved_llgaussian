"""Real CUDA integration checks; intentionally skipped on CPU-only machines."""

import importlib.util
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import torch

from model_fixture import make_cpu_model
from utils.composition_utils import compose_decomposed_render
from utils.rasterizer_utils import rasterize_training_depth


CUDA_AVAILABLE = torch.cuda.is_available() and all(importlib.util.find_spec(name) is not None for name in (
    "diff_gaussian_rasterization", "diff_gaussian_rasterization_fast",
    "diff_gaussian_rasterization_residual", "simple_knn", "torch_scatter",
))


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA rasterizers and model extensions are unavailable")
class RasterizerContractsCudaTest(unittest.TestCase):
    def make_scene(self, illumination="asg", reflectance="explicit", distance=False):
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import getProjectionMatrix

        model = GaussianModel(feat_dim=8, n_offsets=4, appearance_residual_dim=0,
                              illumination_mode=illumination, reflectance_mode=reflectance,
                              add_reflectance_dist=distance, add_cov_dist=distance,
                              add_opacity_dist=distance)
        fixture = make_cpu_model()
        for name, value in vars(fixture).items():
            if torch.is_tensor(value):
                converted = value.detach().cuda()
                if isinstance(value, torch.nn.Parameter):
                    converted = torch.nn.Parameter(converted, requires_grad=value.requires_grad)
                setattr(model, name, converted)
        model.P = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0]], device="cuda", requires_grad=True)
        with torch.no_grad():
            model.mlp_opacity[-2].weight.zero_()
            model.mlp_opacity[-2].bias.fill_(0.8)
            model.mlp_cov[-1].weight.zero_()
            model.mlp_cov[-1].bias.copy_(torch.tensor([0, 0, 0, 1, 0, 0, 0], device="cuda").repeat(4))
        model.train()
        projection = getProjectionMatrix(0.01, 100, 1.0, 1.0).T.cuda()
        camera = SimpleNamespace(uid=0, image_name="fixture", image_height=32, image_width=32,
                                 FoVx=1.0, FoVy=1.0, camera_center=torch.zeros(3, device="cuda"),
                                 projection_matrix=projection)
        return model, camera, SimpleNamespace(debug=False), torch.zeros(3, device="cuda")

    def test_full_renderer_sharpening_permissions_and_no_inference_auxiliary(self):
        from gaussian_renderer import render
        from utils.loss_utils import L_Reflectance_Edge_Uplift, L_Reflectance_LocalContrast, L_Reflectance_HighFreq

        for mode in ("explicit", "mlp"):
            for loss_fn in (L_Reflectance_Edge_Uplift, L_Reflectance_LocalContrast, L_Reflectance_HighFreq):
                with self.subTest(mode=mode, loss=loss_fn.__name__):
                    model, camera, pipe, bg = self.make_scene(reflectance=mode, distance=True)
                    package = render(camera, model, pipe, bg, 0.1, camera_pose=model.P[0],
                                     retain_grad=True, return_reflectance_aux=True)
                    torch.testing.assert_close(package["render_reflectance"], package["render_reflectance_aux"])
                    loss_fn(package["render_reflectance_aux"], torch.rand(3, 32, 32, device="cuda")).backward()
                    parameters = (list(model.mlp_reflectance.parameters()) if mode == "mlp" else
                                  [model._base_log_reflectance, model._reflectance_offset_delta,
                                   *model.mlp_reflectance_decoder.parameters()])
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))
                    for parameter in (model._anchor, model._offset, model._scaling, model._anchor_feat,
                                      model.P, package["viewspace_points"],
                                      *model.mlp_opacity.parameters(), *model.mlp_cov.parameters()):
                        self.assertIsNone(parameter.grad)
                    model.eval()
                    with torch.enable_grad():
                        inference = render(camera, model, pipe, bg, 0.1, camera_pose=model.P[0],
                                           return_reflectance_aux=True, return_training_depth=True)
                    self.assertIsNone(inference["render_reflectance_aux"])
                    self.assertIsNone(inference["render_depth_for_loss"])

    def test_split_unite_ply_and_fast_render_agree(self):
        from gaussian_renderer import render, render_fast

        for illumination in ("asg", "sg"):
            for reflectance in ("explicit", "mlp"):
                model, camera, pipe, bg = self.make_scene(illumination, reflectance)
                model.reflectance_detail_scale, model.reflectance_decoder_scale = 0.7, 0.08
                model.eval()
                with torch.no_grad():
                    expected = compose_decomposed_render(render(camera, model, pipe, bg, 0.1, camera_pose=model.P[0]), enhanced=True)
                    fast = render_fast(camera, model, pipe, bg, 0.1, camera_pose=model.P[0])["render"]
                    torch.testing.assert_close(fast, expected, atol=2e-5, rtol=2e-4)
                for mode in ("split", "unite"):
                    with self.subTest(illumination=illumination, reflectance=reflectance, checkpoint=mode):
                        with tempfile.TemporaryDirectory() as temp:
                            model.save_ply(str(Path(temp) / "point_cloud.ply"))
                            model.save_mlp_checkpoints(temp, mode=mode)
                            restored, _, _, _ = self.make_scene(illumination, reflectance)
                            restored.load_ply_sparse_gaussian(str(Path(temp) / "point_cloud.ply"))
                            restored.load_mlp_checkpoints(temp, mode=mode)
                            restored.eval()
                            self.assertEqual(restored.reflectance_detail_scale, 0.7)
                            self.assertEqual(restored.reflectance_decoder_scale, 0.08)
                            with torch.no_grad():
                                actual = compose_decomposed_render(render(camera, restored, pipe, bg, 0.1, camera_pose=restored.P[0]), enhanced=True)
                            torch.testing.assert_close(actual, expected)

    def test_real_accumulated_depth_forward_and_direct_z_finite_difference(self):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        from utils.graphics_utils import getProjectionMatrix

        settings = GaussianRasterizationSettings(
            image_height=32, image_width=32, tanfovx=0.55, tanfovy=0.55, kernel_size=0.1,
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0, viewmatrix=torch.eye(4, device="cuda"),
            projmatrix=getProjectionMatrix(0.01, 100, 1.0, 1.0).T.cuda(), sh_degree=1,
            campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        points = torch.tensor([[-0.15, 0.0, 1.2], [0.2, 0.1, 1.7]], device="cuda", requires_grad=True)
        kwargs = dict(means2D=torch.zeros_like(points), opacities=torch.full((2, 1), 0.6, device="cuda"),
                      scales=torch.full((2, 3), 0.09, device="cuda"),
                      rotations=torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device="cuda"))
        depth, _ = rasterize_training_depth(rasterizer, means3D=points, **kwargs)
        _, _, old_depth = rasterizer(means3D=points, colors_precomp=torch.ones_like(points), **kwargs)
        torch.testing.assert_close(depth, old_depth, atol=1e-5, rtol=1e-5)
        depth.mean().backward()
        delta = 1e-3
        for index in range(2):
            positive, negative = points.detach().clone(), points.detach().clone()
            positive[index, 2] += delta
            negative[index, 2] -= delta
            plus = rasterize_training_depth(rasterizer, means3D=positive, **kwargs)[0].mean()
            minus = rasterize_training_depth(rasterizer, means3D=negative, **kwargs)[0].mean()
            numeric = (plus - minus) / (2 * delta)
            torch.testing.assert_close(points.grad[index, 2], numeric, atol=1e-3, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
