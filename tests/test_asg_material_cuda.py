"""Native ASG rasterization and isolated gradient permissions."""
from types import SimpleNamespace
import unittest

import torch

import test_rl_compat_cuda as fixture
from utils.llgaussian_objective import llgaussian_objective
from utils.rl_compat_utils import (asg_material_image_terms, asg_parameter_compat_terms,
                                   parameter_compat_terms, prepare_asg_material_target,
                                   weighted_compat_losses)


@unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA rasterizers')
class ASGMaterialCudaTest(unittest.TestCase):
    def create(self, profile):
        helper = fixture.RLCompatCudaTest()
        model, pose, camera = helper.create(profile, 'asg')
        with torch.no_grad():
            model._illum_asg_axis.zero_(); model._illum_asg_axis[..., 2] = 1
            model._illum_asg_tangent.zero_(); model._illum_asg_tangent[..., 0] = 1
        return helper, model, pose, camera

    def all_parameters(self, model, pose):
        result = {'pose': pose}
        for key, value in vars(model).items():
            if isinstance(value, torch.nn.Parameter):
                result[key] = value
            elif isinstance(value, torch.nn.Module):
                result.update({key + '.' + name: p for name, p in value.named_parameters()})
        return result

    def test_zero_scale_full_objective_outputs_and_all_gradients_match(self):
        records = []
        for profile in ('llgaussian', 'llgaussian_rl'):
            helper, model, pose, camera = self.create(profile)
            camera.image_height = camera.image_width = 256
            # Production scale=0 bypasses every auxiliary render/target calculation.
            pkg = helper.render(model, pose, camera, False)
            target = torch.full_like(pkg['render'], .06)
            target[0, :, 128:] *= 1.4
            opt = SimpleNamespace(update_from=1000, iterations=8000, lambda_dssim=.3)
            torch.manual_seed(51)
            loss, _ = llgaussian_objective(pkg, target, target[:1], 2000, opt, 29, target * 4)
            loss.backward()
            records.append((pkg, loss, self.all_parameters(model, pose)))
        for key in ('render', 'render_enhanced', 'render_reflectance', 'render_illumination', 'render_depth'):
            torch.testing.assert_close(records[0][0][key], records[1][0][key], rtol=0, atol=0)
        torch.testing.assert_close(records[0][1], records[1][1], rtol=0, atol=0)
        self.assertEqual(set(records[0][2]), set(records[1][2]))
        for key, p in records[0][2].items():
            other = records[1][2][key]
            self.assertEqual(p.grad is None, other.grad is None, key)
            if p.grad is not None:
                # CUDA rasterizer reductions use floating point atomics.
                torch.testing.assert_close(p.grad, other.grad, rtol=1e-4, atol=1e-6, msg=lambda m: key + ': ' + m)

    def test_auxiliary_r_geometry_freeze_and_main_geometry_permission(self):
        helper, model, pose, camera = self.create('llgaussian_rl')
        pkg = helper.render(model, pose, camera, True)
        torch.testing.assert_close(pkg['render_reflectance_aux'], pkg['render_reflectance'], rtol=0, atol=0)
        target = torch.tensor([.1, .05, .025], device='cuda')[:, None, None].expand(3, 20, 20).clone()
        target[:, :, 10:] = target.flip(0)[:, :, 10:]
        terms = asg_material_image_terms(pkg['render_reflectance_aux'], prepare_asg_material_target(target), torch.ones_like(target))
        terms.update(parameter_compat_terms(model._base_log_reflectance, model._reflectance_offset_delta,
                                            model._last_reflectance_decoder_squared, target.new_zeros(())))
        terms.update(asg_parameter_compat_terms(model._illum_asg_amplitude, model._illum_asg_sharpness))
        r_loss, _ = weighted_compat_losses(terms, 'asg_paper', 2000)
        r_loss.backward(retain_graph=True)
        allowed = ('_base_log_reflectance', '_reflectance_offset_delta', 'mlp_reflectance_decoder.')
        for name, p in self.all_parameters(model, pose).items():
            if not name.startswith(allowed): self.assertIsNone(p.grad, name)
        self.assertIsNone(pkg['viewspace_points'].grad)
        self.assertGreater(float(model._base_log_reflectance.grad.abs().sum()), 0)
        (pkg['render'] - target).square().mean().backward()
        self.assertGreater(float(model._anchor.grad.abs().sum()), 0)
        self.assertGreater(float(pkg['viewspace_points'].grad.abs().sum()), 0)
        self.assertGreater(float(model._illum_asg_bias.grad.abs().sum()), 0)

    def test_paper_regularizers_only_update_amplitude_and_bandwidth(self):
        helper, model, pose, camera = self.create('llgaussian_rl')
        pkg = helper.render(model, pose, camera, True)
        terms = asg_parameter_compat_terms(model._illum_asg_amplitude, model._illum_asg_sharpness)
        for name, term in terms.items():
            params = self.all_parameters(model, pose)
            grads = torch.autograd.grad(term, list(params.values()), allow_unused=True, retain_graph=True)
            active = {key for key, grad in zip(params, grads) if grad is not None}
            expected = {'_illum_asg_amplitude'} if name == 'asg_energy' else {'_illum_asg_sharpness'}
            self.assertEqual(active, expected)
        self.assertIsNone(pkg['viewspace_points'].grad)

    def test_explicit_parameter_adam_states_survive_growth_and_prune(self):
        _, model, _, _ = self.create('llgaussian_rl')
        names = ('anchor', 'base_log_reflectance', 'reflectance_offset_delta',
                 'illum_asg_axis', 'illum_asg_tangent', 'illum_asg_sharpness',
                 'illum_asg_amplitude', 'illum_asg_bias', 'illum_asg_dist_weight')
        groups = [{'name': name, 'params': [getattr(model, '_' + name)]} for name in names]
        enhancement = list(model.enhancement_net.parameters())
        decoder = list(model.mlp_reflectance_decoder.parameters())
        groups += [{'name': 'enhancement_net', 'params': enhancement},
                   {'name': 'mlp_reflectance_decoder', 'params': decoder}]
        model.optimizer = torch.optim.Adam(groups, lr=.001)
        sum(p.square().sum() for group in groups for p in group['params']).backward()
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        originals = {name: getattr(model, '_' + name).detach().clone() for name in names}
        moments = {name: model.optimizer.state[getattr(model, '_' + name)]['exp_avg'].clone() for name in names}
        grown = model.cat_tensors_to_optimizer({name: originals[name][:1].clone() for name in names})
        for name in names:
            torch.testing.assert_close(grown[name][:2], originals[name], rtol=0, atol=0)
            state = model.optimizer.state[grown[name]]
            torch.testing.assert_close(state['exp_avg'][:2], moments[name], rtol=0, atol=0)
            self.assertEqual(int(state['exp_avg'][2:].count_nonzero()), 0)
            self.assertEqual(int(state['exp_avg_sq'][2:].count_nonzero()), 0)
            self.assertEqual(float(state['step']), 1)
        mask = torch.tensor([True, False, True], device='cuda')
        pruned = model._prune_anchor_optimizer(mask)
        for name in names:
            self.assertEqual(pruned[name].shape[0], 2)
            torch.testing.assert_close(pruned[name][0], originals[name][0], rtol=0, atol=0)
            torch.testing.assert_close(model.optimizer.state[pruned[name]]['exp_avg'][0], moments[name][0], rtol=0, atol=0)
        for group, expected in zip(model.optimizer.param_groups[-2:], (enhancement, decoder)):
            self.assertTrue(all(p is q for p, q in zip(group['params'], expected)))
            self.assertTrue(all(float(model.optimizer.state[p]['step']) == 1 for p in expected))


if __name__ == '__main__':
    unittest.main()
