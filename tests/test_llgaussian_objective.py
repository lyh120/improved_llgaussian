"""Parity with an unchanged excerpt of the user's original LL-Gaussian."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from utils import loss_utils
from utils.llgaussian_objective import llgaussian_objective
from utils.visualize_utils import minmax_normalize


@unittest.skipUnless(torch.cuda.is_available(), "Reference depth loss requires CUDA")
class ReferenceObjectiveTest(unittest.TestCase):
    def test_loss_values_and_all_gradient_directions_match_reference(self):
        excerpt = (Path(__file__).parent / 'fixtures/llgaussian_loss_excerpt.txt').read_text(encoding='utf-8')
        opt = SimpleNamespace(update_from=1000, iterations=8000, lambda_dssim=0.3)
        torch.manual_seed(12)
        original = dict(render_reflectance=torch.rand(3, 272, 272),
                        render_illumination=torch.rand(3, 272, 272) * 0.04,
                        render_illumination_enhanced=torch.rand(3, 272, 272),
                        render_depth=torch.rand(1, 272, 272), scaling=torch.rand(8, 3),
                        render_residual=torch.rand(3, 272, 272) * 0.001,
                        scaling_residual=torch.rand(8, 3))
        target = torch.rand(3, 272, 272) * 0.03
        prior = torch.rand(3, 272, 272)
        depth_prior = torch.rand(1, 272, 272)
        for mode, iteration in [('warmup', 999), ('warmup', 1000), ('train', 999),
                                ('train', 1000), ('train', 1999), ('train', 2000), ('train', 8000)]:
            with self.subTest(mode=mode, iteration=iteration):
                actual = {k: v.clone().requires_grad_() for k, v in original.items()}
                reference = {k: v.clone().requires_grad_() for k, v in original.items()}
                r, l, e = [reference[k] for k in ('render_reflectance', 'render_illumination', 'render_illumination_enhanced')]
                residual = reference['render_residual'] if mode != 'warmup' else torch.zeros_like(target)
                scope = dict(vars(loss_utils), torch=torch, FUSED_SSIM_AVAILABLE=False,
                             image_tmp=(r * l + residual).clamp(0, 1), gt_image=target,
                             reflectance_image=r, illumination_image=l, illumination_enhanced_image=e,
                             depth_image=reference['render_depth'], depth_piror_norm=depth_prior,
                             scaling=reference['scaling'], scaling_residual=reference['scaling_residual'],
                             residual_image=residual, minmax_normalize=minmax_normalize, mode=mode,
                             iteration=iteration, opt=opt, dataset=SimpleNamespace(use_residual=True),
                             enhance_ratio=29, viewpoint_cam=SimpleNamespace(uid=0),
                             refined_image_dict={0: prior}, weight_scheduler=lambda i: max(2-1.5*i/8000, 0.5))
                # Excerpt's .cuda() is an image-device operation, not part of the loss.
                if torch.cuda.is_available():
                    scope['refined_image_dict'] = {0: prior.cuda()}
                    for key in actual:
                        actual[key] = actual[key].detach().cuda().requires_grad_()
                        reference[key] = reference[key].detach().cuda().requires_grad_()
                    r, l, e = [reference[k] for k in ('render_reflectance', 'render_illumination', 'render_illumination_enhanced')]
                    residual = reference['render_residual'] if mode != 'warmup' else torch.zeros_like(target.cuda())
                    scope.update(image_tmp=(r*l+residual).clamp(0,1), gt_image=target.cuda(),
                                 reflectance_image=r, illumination_image=l, illumination_enhanced_image=e,
                                 depth_image=reference['render_depth'], depth_piror_norm=depth_prior.cuda(),
                                 scaling=reference['scaling'], scaling_residual=reference['scaling_residual'], residual_image=residual)
                code = excerpt if torch.cuda.is_available() else excerpt.replace('.cuda()', '')
                torch.manual_seed(33)
                exec(compile(code, 'original_LL-Gaussian_train.py', 'exec'), scope)
                torch.manual_seed(33)
                device = actual['render_depth'].device
                loss, _ = llgaussian_objective(actual, target.to(device), depth_prior.to(device),
                                               iteration, opt, 29, prior.to(device), warmup=(mode == 'warmup'))
                torch.testing.assert_close(loss, scope['loss'], rtol=1e-6, atol=1e-6)
                loss.backward(); scope['loss'].backward()
                for key in actual:
                    if reference[key].grad is None:
                        self.assertIsNone(actual[key].grad, key)
                    else:
                        torch.testing.assert_close(actual[key].grad, reference[key].grad, rtol=1e-5, atol=1e-7, msg=key)


if __name__ == '__main__':
    unittest.main()
