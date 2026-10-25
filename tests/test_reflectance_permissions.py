import unittest

import torch

from model_fixture import SoftRasterizer, make_cpu_model
from utils.loss_utils import (
    L_Reflectance_Edge_Uplift,
    L_Reflectance_HighFreq,
    L_Reflectance_LocalContrast,
)
from utils.rasterizer_utils import rasterize_reflectance_aux
from utils.reflectance_utils import predict_aux_reflectance


class ReflectancePermissionsTest(unittest.TestCase):
    def render_pair(self, mode, distance=False):
        model = make_cpu_model(mode, distance)
        visible = torch.ones(2, dtype=torch.bool)
        distances = model._anchor.norm(dim=1, keepdim=True)
        if mode == "explicit":
            colors = model.get_reflectance_with_decoder(model._anchor_feat, model._offset, visible).reshape(-1, 3)
        else:
            inputs = model._anchor_feat
            if distance:
                inputs = torch.cat((inputs, distances), dim=1)
            colors = model.mlp_reflectance(inputs).reshape(-1, 3)
        statistic = model._last_reflectance_decoder_mean
        aux_colors = predict_aux_reflectance(model, model._anchor_feat, model._offset, visible, distances)
        self.assertIs(model._last_reflectance_decoder_mean, statistic)
        pose = torch.zeros(3, requires_grad=True)
        points = model._anchor.repeat_interleave(4, dim=0) + model._offset.reshape(-1, 3) + pose
        screenspace = torch.zeros_like(points, requires_grad=True)
        opacity = torch.sigmoid(model.mlp_opacity(model._anchor_feat)).reshape(-1, 1)
        covariance = model.mlp_cov(model._anchor_feat).repeat_interleave(4, dim=0)
        kwargs = dict(means3D=points, means2D=screenspace, opacities=opacity,
                      scales=torch.sigmoid(covariance[:, :3]), rotations=covariance[:, 3:])
        main = SoftRasterizer()(colors_precomp=colors, **kwargs)[0]
        auxiliary = rasterize_reflectance_aux(SoftRasterizer(), colors_precomp=aux_colors, **kwargs)
        torch.testing.assert_close(main, auxiliary)
        return model, main, auxiliary, screenspace, pose

    def test_each_sharpening_loss_updates_only_the_allowed_parameters(self):
        for mode, distance in (("explicit", False), ("mlp", False), ("mlp", True)):
            for loss_fn in (L_Reflectance_Edge_Uplift, L_Reflectance_LocalContrast, L_Reflectance_HighFreq):
                with self.subTest(mode=mode, distance=distance, loss=loss_fn.__name__):
                    model, _, auxiliary, screen, pose = self.render_pair(mode, distance)
                    torch.manual_seed(31)
                    target = torch.rand(3, 24, 24)
                    loss = loss_fn(auxiliary, target)
                    self.assertGreater(loss.item(), 0)
                    loss.backward()
                    allowed = (list(model.mlp_reflectance.parameters()) if mode == "mlp" else
                               [model._base_log_reflectance, model._reflectance_offset_delta,
                                *model.mlp_reflectance_decoder.parameters()])
                    self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in allowed))
                    for parameter in (model._anchor, model._offset, model._anchor_feat, model._scaling,
                                      model._rotation, model.P, screen, pose,
                                      *model.mlp_opacity.parameters(), *model.mlp_cov.parameters(),
                                      *model.mlp_illumination.parameters()):
                        self.assertIsNone(parameter.grad)

    def test_main_reconstruction_keeps_geometry_and_shared_feature_gradients(self):
        model, main, _, screen, pose = self.render_pair("explicit")
        main.square().mean().backward()
        for parameter in (model._anchor, model._offset, model._anchor_feat, screen, pose,
                          *model.mlp_opacity.parameters(), *model.mlp_cov.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
