"""CUDA regression audit using a real scene, PLY and differentiable rasterizer."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import prefilter_voxel, render
from scene import GaussianModel, Scene
from utils.composition_utils import compose_decomposed_render
from utils.gradient_utils import (
    ASG_REGULARIZER_GROUPS, REFLECTANCE_GROUPS,
    accumulate_auxiliary_gradients, parameters_in_groups,
)
from utils.loss_utils import (
    L_Reflectance_Edge, L_Reflectance_Edge_Uplift, L_Reflectance_HighFreq,
    L_Reflectance_LocalContrast, L_ASG_Energy, L_ASG_Sharpness, L_ASG_Anisotropy,
)
from utils.sg_utils import asg_parameter_regularization


def main():
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iteration", type=int, default=8000)
    args = parser.parse_args()
    torch.manual_seed(0)
    data = model_params.extract(args)
    pipe = pipeline_params.extract(args)
    model = GaussianModel(n_offsets=data.n_offsets, appearance_residual_dim=0,
                          use_residual=False, illumination_mode="asg",
                          use_3D_filter=data.use_3D_filter)
    scene = Scene(data, model, None, load_iteration=args.iteration,
                  shuffle=False, only_ply=True)
    assert model.P.is_leaf and model.P.requires_grad
    model.spatial_lr_scale = scene.cameras_extent
    model.training_setup(optimization_params.extract(args))
    model.train()
    camera = scene.getTrainCameras()[0]
    background = torch.zeros(3, device="cuda")
    pose = model.get_RT(camera.uid)
    visible = prefilter_voxel(camera, model, pipe, background, data.kernel_size, camera_pose=pose)
    package = render(camera, model, pipe, background, data.kernel_size,
                     visible_mask=visible, camera_pose=pose, retain_grad=True,
                     return_reflectance_aux=True)
    main_image = compose_decomposed_render(package)
    aux = package["render_reflectance_aux"]
    error = (aux - package["render_reflectance"]).detach().abs().max().item()
    assert error == 0, error
    allowed = parameters_in_groups(model.optimizer, REFLECTANCE_GROUPS)
    all_parameters = [p for group in model.optimizer.param_groups for p in group["params"] if p.requires_grad]
    allowed_ids = {id(p) for p in allowed}
    forbidden = [p for p in all_parameters if id(p) not in allowed_ids]
    screen = package["viewspace_points"]
    targets = forbidden + [screen]
    gt = camera.original_image
    terms = {
        "edge": (L_Reflectance_Edge(aux, gt), 2e-4),
        "uplift": (L_Reflectance_Edge_Uplift(aux, gt), 3e-3),
        "contrast": (L_Reflectance_LocalContrast(aux, gt), 2e-3),
        "highfreq": (L_Reflectance_HighFreq(aux, gt), 3e-3),
        "decoder": (model._last_reflectance_decoder_mean, 2e-5),
    }
    for name, (loss, _) in terms.items():
        gradients = torch.autograd.grad(loss, targets, allow_unused=True, retain_graph=True)
        assert all(g is None or torch.count_nonzero(g) == 0 for g in gradients), name
    main_loss = (main_image - gt).square().mean()
    baseline_gradients = torch.autograd.grad(main_loss, targets, allow_unused=True, retain_graph=True)
    # retain_grad hooks on this non-leaf also run during the baseline VJP.
    screen.grad = None
    record = {"r_aux_max_error": error, "reloaded_pose_is_trainable_leaf": True,
              "R": accumulate_auxiliary_gradients(terms, allowed, audit=True)}
    assert all(p.grad is None for p in targets)
    main_loss.backward(retain_graph=True)
    max_main_gradient_difference = 0.0
    for p, expected in zip(targets, baseline_gradients):
        if expected is None:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0
        else:
            # CUDA rasterizer reductions use atomic adds; independent backwards
            # can differ at the last few floating-point bits.
            max_main_gradient_difference = max(
                max_main_gradient_difference, (p.grad - expected).abs().max().item(),
            )
            torch.testing.assert_close(p.grad, expected, rtol=1e-3, atol=1e-6)
    record["main_gradient_max_absolute_difference"] = max_main_gradient_difference
    for label, parameter in (("offset", model._offset), ("scale", model._scaling),
                             ("feature", model._anchor_feat), ("pose", model.P), ("screen", screen)):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, label
    record["main_geometry_and_screen_gradients_unchanged"] = True
    model.optimizer.zero_grad(set_to_none=True)
    screen.grad = None
    accumulate_auxiliary_gradients({"off": (aux.mean(), 0)}, allowed, audit=True)
    assert all(p.grad is None for p in all_parameters)
    record["zero_weight_no_gradients"] = True
    enhanced = compose_decomposed_render(package, enhanced=True, detach_reflectance=True)
    enhanced_loss = enhanced.mean() + package["render_illumination_enhanced"].square().mean()
    enh_allowed = parameters_in_groups(model.optimizer, {
        "enhancement_sg_axis", "enhancement_sg_sharpness", "enhancement_sg_amplitude",
        "enhancement_context_feat", "enhancement_context_illum", "enhancement_context_bias",
    })
    enh_ids = {id(p) for p in enh_allowed}
    enh_forbidden = [p for p in all_parameters if id(p) not in enh_ids] + [screen]
    enh_grads = torch.autograd.grad(enhanced_loss, enh_forbidden, allow_unused=True, retain_graph=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in enh_grads)
    assert any(g is not None and g.abs().sum() > 0 for g in torch.autograd.grad(
        enhanced_loss, enh_allowed, allow_unused=True, retain_graph=True))
    record["enhancement_only_updates_enhancement_branch"] = True
    stats = asg_parameter_regularization(model._illum_asg_sharpness, model._illum_asg_amplitude, 1.0)
    l_allowed = parameters_in_groups(model.optimizer, ASG_REGULARIZER_GROUPS)
    l_terms = {"energy": (L_ASG_Energy(stats), 1e-4),
               "sharpness": (L_ASG_Sharpness(stats), 5e-5),
               "anisotropy": (L_ASG_Anisotropy(stats), 1e-5)}
    record["L"] = accumulate_auxiliary_gradients(l_terms, l_allowed, audit=True)
    assert model._illum_asg_axis.grad is None and model._illum_asg_bias.grad is None
    assert all(record["L"][f"{name}/grad_norm"] > 0 for name in l_terms)
    record["ASG_regularizers_have_live_gradients"] = True
    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        model.save_ply(str(Path(directory) / "point_cloud.ply"))
        model.save_mlp_checkpoints(directory)
        restored = GaussianModel(n_offsets=data.n_offsets, appearance_residual_dim=0,
                                 use_residual=False, illumination_mode="asg",
                                 use_3D_filter=data.use_3D_filter)
        restored.load_ply_sparse_gaussian(str(Path(directory) / "point_cloud.ply"))
        restored.load_mlp_checkpoints(directory)
        restored.train()
        second = render(camera, restored, pipe, background, data.kernel_size,
                        visible_mask=visible, camera_pose=pose, return_reflectance_aux=True)
        for key in ("render_reflectance", "render_illumination", "render_illumination_enhanced"):
            torch.testing.assert_close(package[key], second[key], rtol=0, atol=0)
    record["PLY_MLP_roundtrip_exact"] = True
    record["passed"] = True
    Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
