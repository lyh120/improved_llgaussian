"""Run real Gaussian generation on CPU, comparing regular and fast paths."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from einops import rearrange, repeat
except ImportError:
    # CPU fixtures need only these two layout primitives; CUDA integration uses
    # the real module and its installed dependencies without substitution.
    def repeat(value, pattern, *, k):
        if pattern != 'n (c) -> (n k) (c)':
            raise ValueError(pattern)
        return value.repeat_interleave(k, dim=0)

    def rearrange(value, pattern):
        if pattern != 'n k c -> (n k) c':
            raise ValueError(pattern)
        return value.reshape(-1, value.shape[-1])

from model_fixture import make_cpu_model
from utils.reflectance_utils import predict_aux_reflectance
from utils.sg_utils import evaluate_anisotropic_spherical_gaussians, evaluate_spherical_gaussians


def load_generators(model_class):
    path = Path(__file__).resolve().parents[1] / "gaussian_renderer" / "__init__.py"
    names = {"_compute_illumination", "_view_adapted_features", "generate_neural_gaussians",
             "generate_neural_gaussians_fast"}
    functions = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, rearrange=rearrange, repeat=repeat, GaussianModel=model_class,
                     predict_aux_reflectance=predict_aux_reflectance,
                     evaluate_anisotropic_spherical_gaussians=evaluate_anisotropic_spherical_gaussians,
                     evaluate_spherical_gaussians=evaluate_spherical_gaussians)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["generate_neural_gaussians"], namespace["generate_neural_gaussians_fast"]


class RendererGenerationTest(unittest.TestCase):
    def test_regular_fast_and_aux_values_agree_for_supported_feature_inputs(self):
        for illumination in ("asg", "sg"):
            for reflectance in ("explicit", "mlp"):
                for distance in (False, True):
                    for feature_bank in (False, True):
                        with self.subTest(illumination=illumination, reflectance=reflectance,
                                          distance=distance, feature_bank=feature_bank):
                            model = make_cpu_model(reflectance, distance)
                            model.illumination_mode = illumination
                            model.use_feat_bank = feature_bank
                            model.use_3D_filter = False
                            model.add_opacity_dist = model.add_cov_dist = model.add_illumination_dist = distance
                            model.sg_illumination_available = model.use_sg_illumination = True
                            model.asg_illumination_available = model.use_asg_illumination = True
                            model.sg_lobes, model.sg_lambda_min, model.asg_lambda_min = 2, 1.0, 1.0
                            model.scaling_activation = torch.exp
                            model.rotation_activation = torch.nn.functional.normalize
                            model.mlp_feature_bank = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Softmax(dim=-1))
                            model.mlp_opacity = torch.nn.Sequential(torch.nn.Linear(11 + int(distance), 4), torch.nn.Sigmoid())
                            model.mlp_cov = torch.nn.Linear(11 + int(distance), 28)
                            model.mlp_sg_illumination = torch.nn.Linear(7 + int(distance), 40)
                            camera = SimpleNamespace(camera_center=torch.zeros(3))
                            visible = torch.ones(2, dtype=torch.bool)
                            regular, fast = load_generators(type(model))
                            main = regular(camera, model, visible, is_training=True, return_reflectance_aux=True)
                            quick = fast(camera, model, visible, return_components=True)
                            for main_index, quick_index in ((0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 5)):
                                torch.testing.assert_close(main[main_index], quick[quick_index])
                            torch.testing.assert_close(main[1], main[-1])
                            main[-1].sum().backward()
                            for parameter in (model._anchor, model._offset, model._anchor_feat,
                                              *model.mlp_feature_bank.parameters(),
                                              *model.mlp_opacity.parameters(), *model.mlp_cov.parameters()):
                                self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
