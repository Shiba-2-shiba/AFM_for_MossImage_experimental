import json
import os
from types import SimpleNamespace
import unittest

import torch

from tests.integration import test_native_hooks as native_tests
from tests.unit.test_runtime import FakeExecutor


@unittest.skipUnless(os.environ.get('MOSS_COMFY_ROOT'), 'requires actual ComfyUI')
class BlockTuneWindowNativeTests(unittest.TestCase):
    model = native_tests.NativeTests.model
    setUpClass = classmethod(native_tests.NativeTests.setUpClass.__func__)
    tearDownClass = classmethod(native_tests.NativeTests.tearDownClass.__func__)

    def quantized_qkv(self, base, *, full_precision=False):
        import comfy.ops

        operation = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
        module = operation.Linear(64, 192, bias=False, device='cpu')
        module.load_state_dict({
            'weight': torch.ones((192, 64), dtype=torch.int8),
            'weight_scale': torch.linspace(.01, .04, 192).reshape(-1, 1),
            'comfy_quant': torch.tensor(list(json.dumps({
                'format': 'int8_tensorwise', 'convrot': True,
                'convrot_groupsize': 4,
            }).encode()), dtype=torch.uint8),
        })
        module._full_precision_mm = full_precision
        base.model.diffusion_model.layers[0].proj['qkv'] = module
        return module

    def test_int8_convrot_window_matches_scaled_and_identity_dequantized_references(self):
        from afm.block_tune import install_block_tune
        from afm.config import AFMConfig
        from afm.runtime import RUN_KEY, RunState
        from comfy.patcher_extension import WrappersMP

        for full_precision in (False, True):
            with self.subTest(full_precision=full_precision), torch.inference_mode():
                base = self.model(True, num_layers=42)
                module = self.quantized_qkv(base, full_precision=full_precision)
                original = {key: value.clone() for key, value in module.state_dict().items()}
                x = torch.randn(2, 3, 64)
                identity_weight = module.weight.dequantize().clone()
                scaled_weight = identity_weight.clone()
                for part, factor in zip(scaled_weight.chunk(3), (1.7, 1., .8)):
                    part.mul_(factor)
                identity = torch.nn.functional.linear(x, identity_weight)
                scaled = torch.nn.functional.linear(x, scaled_weight)

                tuned, _ = install_block_tune(base, 'image_2_global_qkv', 'middle_30_70')
                diffusion, = tuned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
                tuned.patch_model(device_to=torch.device('cpu'), load_weights=True)
                try:
                    outputs = {}
                    for step in (0, 5, 10):
                        state = RunState(11, AFMConfig(logging='off'), step_index=step)
                        outputs[step] = diffusion(
                            FakeExecutor(None, lambda *args, **kwargs: module(x)),
                            None, None, None, {RUN_KEY: state})
                    torch.testing.assert_close(outputs[0], identity, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(outputs[5], scaled, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(outputs[10], identity, rtol=1e-5, atol=1e-6)
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                finally:
                    tuned.unpatch_model(device_to=torch.device('cpu'))

    def test_real_euler_and_er_sde_drive_installed_middle_window_and_clean_up(self):
        import comfy.model_sampling
        from afm.block_tune import install_block_tune
        from afm.runtime import RUN_KEY
        from comfy.k_diffusion import sampling
        from comfy.patcher_extension import WrappersMP

        class FlowSampling(comfy.model_sampling.CONST):
            def percent_to_sigma(self, percent):
                return 1 - percent

        for function in (sampling.sample_euler, sampling.sample_er_sde):
            with self.subTest(sampler=function.__name__), torch.inference_mode():
                base = self.model(True, num_layers=42)
                module = self.quantized_qkv(base)
                x_proj = torch.randn(2, 3, 64)
                identity_weight = module.weight.dequantize().clone()
                scaled_weight = identity_weight.clone()
                for part, factor in zip(scaled_weight.chunk(3), (1.7, 1., .8)):
                    part.mul_(factor)
                expected = {
                    False: torch.nn.functional.linear(x_proj, identity_weight),
                    True: torch.nn.functional.linear(x_proj, scaled_weight),
                }
                tuned, _ = install_block_tune(base, 'image_2_global_qkv', 'middle_30_70')
                outer, = tuned.get_all_wrappers(WrappersMP.OUTER_SAMPLE)
                diffusion, = tuned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
                original_options = tuned.model_options
                guider = SimpleNamespace(model_options=original_options)
                observed = []
                tuned.patch_model(device_to=torch.device('cpu'), load_weights=True)
                try:
                    class Model:
                        def __init__(self):
                            self.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
                                get_model_object=lambda key: FlowSampling()))

                        def __call__(self, latent, sigma, **kwargs):
                            options = guider.model_options['transformer_options']

                            def evaluate(*args, **inner_kwargs):
                                state = options[RUN_KEY]
                                observed.append((state.step_index, module(x_proj).clone()))
                                return torch.zeros_like(latent)

                            return diffusion(FakeExecutor(None, evaluate), latent, sigma, None,
                                             options)

                    sampler = SimpleNamespace(sampler_function=function)

                    def execute(noise, latent, sampler_arg, sigmas, mask, callback,
                                disabled, seed, shapes):
                        return function(
                            Model(), torch.ones(1, 2, 2, 2), sigmas,
                            callback=lambda data: callback(data['i'], data['denoised'],
                                                           data['x'], 3),
                            disable=True, s_noise=0)

                    def fail_after_evaluation(noise, latent, sampler_arg, sigmas, mask,
                                              callback, disabled, seed, shapes):
                        Model()(torch.ones(1, 2, 2, 2), sigmas[:1])
                        raise RuntimeError('injected sampler failure')

                    with self.assertRaisesRegex(RuntimeError, 'injected sampler failure'):
                        outer(FakeExecutor(guider, fail_after_evaluation), None, None, sampler,
                              torch.tensor([1.0, .7, .3, 0.0]))
                    self.assertIs(guider.model_options, original_options)
                    with self.assertRaisesRegex(RuntimeError, 'active text-to-image'):
                        module(x_proj)
                    observed.clear()
                    outer(FakeExecutor(guider, execute), None, None, sampler,
                          torch.tensor([1.0, .7, .3, 0.0]))
                    self.assertIs(guider.model_options, original_options)
                    self.assertEqual(sorted({step for step, _ in observed}), [0, 1, 2])
                    for step, output in observed:
                        torch.testing.assert_close(output, expected[step == 1],
                                                   rtol=1e-5, atol=1e-6)
                    with self.assertRaisesRegex(RuntimeError, 'active text-to-image'):
                        module(x_proj)
                finally:
                    tuned.unpatch_model(device_to=torch.device('cpu'))

    def test_reference_kv_rejection_does_not_leak_gate_state(self):
        from afm.block_tune import install_block_tune
        from afm.config import AFMConfig
        from afm.runtime import RUN_KEY, RunState
        from comfy.patcher_extension import WrappersMP

        with torch.inference_mode():
            base = self.model(True, num_layers=42)
            module = self.quantized_qkv(base)
            x = torch.randn(2, 3, 64)
            tuned, _ = install_block_tune(base, 'image_2_global_qkv', 'middle_30_70')
            diffusion, = tuned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
            state = RunState(3, AFMConfig(logging='off'), step_index=1)
            options = {RUN_KEY: state}
            executor = FakeExecutor(None, lambda *args, **kwargs: module(x))
            tuned.patch_model(device_to=torch.device('cpu'), load_weights=True)
            try:
                with self.assertRaisesRegex(ValueError, 'without reference images'):
                    diffusion(executor, None, None, None, options,
                              ref_kv=torch.ones(1))
                with self.assertRaisesRegex(RuntimeError, 'active text-to-image'):
                    module(x)
                valid = diffusion(executor, None, None, None, options)
                self.assertTrue(torch.isfinite(valid).all())
                with self.assertRaisesRegex(RuntimeError, 'active text-to-image'):
                    module(x)
            finally:
                tuned.unpatch_model(device_to=torch.device('cpu'))
