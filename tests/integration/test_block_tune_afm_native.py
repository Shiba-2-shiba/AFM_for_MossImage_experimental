import os
from types import SimpleNamespace
import unittest

import torch

from tests.integration import test_native_hooks as native_tests
from tests.unit.test_runtime import FakeExecutor


@unittest.skipUnless(os.environ.get('MOSS_COMFY_ROOT'), 'requires actual ComfyUI')
class BlockTuneAFMNativeTests(unittest.TestCase):
    model = native_tests.NativeTests.model
    setUpClass = classmethod(native_tests.NativeTests.setUpClass.__func__)
    tearDownClass = classmethod(native_tests.NativeTests.tearDownClass.__func__)

    def evaluate(self, model, config, step, x, text):
        from afm.runtime import RUN_KEY, RunState

        options = dict(model.model_options['transformer_options'])
        options.update(wrappers=model.wrappers,
                       cond_or_uncond=[0, 1] if x.shape[0] == 2 else [0])
        state = RunState(5, config, step_index=step)
        options[RUN_KEY] = state
        model.patch_model(device_to=torch.device('cpu'), load_weights=True)
        try:
            output = model.model.diffusion_model(
                x.clone(), torch.ones(x.shape[0]), text, options)
        finally:
            model.unpatch_model(device_to=torch.device('cpu'))
        return output, state

    def test_middle_window_and_afm_compose_independently_with_cfg(self):
        from afm.adapter import install_afm
        from afm.block_tune import install_block_tune
        from afm.config import AFMConfig

        base = self.model(True, num_layers=42)
        x, text = torch.randn(2, 8, 2, 3), torch.randn(2, 4, 16)
        config = AFMConfig(mode='edit', strength=.2, block_scope='24-26',
                           branch_mode='positive_only', logging='off')
        prepared = {}
        for application in ('full_run', 'middle_30_70', 'identity_control'):
            tuned, _ = install_block_tune(base, 'image_2_global_qkv', application)
            prepared[application] = (tuned, install_afm(tuned, config))

        for step, expected_application in ((0, 'identity_control'), (2, 'full_run'),
                                           (4, 'identity_control')):
            with self.subTest(step=step):
                actual, state = self.evaluate(prepared['middle_30_70'][1], config,
                                              step, x, text)
                expected, expected_state = self.evaluate(
                    prepared[expected_application][1], config, step, x, text)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(state.edited_calls, 3)
                self.assertEqual(expected_state.edited_calls, 3)

                tuned_only, _ = self.evaluate(prepared['middle_30_70'][0], config,
                                              step, x, text)
                torch.testing.assert_close(actual[1], tuned_only[1], rtol=0, atol=0)

    def test_noop_and_observe_afm_preserve_timed_block_tune_output(self):
        from afm.adapter import install_afm
        from afm.block_tune import install_block_tune
        from afm.config import AFMConfig
        from comfy.patcher_extension import WrappersMP

        base = self.model(True, num_layers=42)
        x, text = torch.randn(1, 8, 2, 3), torch.randn(1, 4, 16)
        for application in ('full_run', 'middle_30_70', 'identity_control'):
            tuned, _ = install_block_tune(base, 'image_2_b34_41_v100', application)
            for mode, strength in (('off', .2), ('edit', 0), ('observe', .2)):
                with self.subTest(application=application, mode=mode):
                    config = AFMConfig(mode=mode, strength=strength,
                                       block_scope='24-26', logging='off')
                    combined = install_afm(tuned, config)
                    actual, state = self.evaluate(combined, config, 2, x, text)
                    expected, _ = self.evaluate(tuned, config, 2, x, text)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if mode == 'observe':
                        self.assertEqual(state.observed_calls, 3)
                        self.assertEqual(
                            len(combined.get_all_wrappers(WrappersMP.OUTER_SAMPLE)), 1)
                        self.assertEqual(
                            len(combined.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)), 2)

    def test_afm_two_intervals_and_gap_do_not_change_tune_window(self):
        from afm.adapter import install_afm
        from afm.block_tune import install_block_tune
        from afm.config import AFMConfig

        base = self.model(True, num_layers=42)
        tuned, _ = install_block_tune(base, 'image_2_global_qkv', 'middle_30_70')
        x, text = torch.randn(1, 8, 2, 3), torch.randn(1, 4, 16)
        config = AFMConfig(mode='edit', strength=.2, block_scope='24-26',
                           start_percent=0, end_percent=.25,
                           two_interval_mode=True, second_block_scope='0-2',
                           second_start_percent=.75, second_end_percent=1,
                           logging='off')
        combined = install_afm(tuned, config)

        for step, edited in ((0, 3), (2, 0), (4, 3)):
            with self.subTest(step=step):
                actual, state = self.evaluate(combined, config, step, x, text)
                if edited:
                    scope = '24-26' if step == 0 else '0-2'
                    single = AFMConfig(mode='edit', strength=.2, block_scope=scope,
                                       logging='off')
                    expected, _ = self.evaluate(install_afm(tuned, single),
                                                single, step, x, text)
                else:
                    expected, _ = self.evaluate(tuned, config, step, x, text)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(state.edited_calls, edited)

    def test_real_euler_and_er_sde_share_one_clock_and_keep_source_standalone(self):
        import comfy.model_sampling
        from afm.adapter import install_afm
        from afm.block_tune import install_block_tune
        from afm.config import AFMConfig
        from afm.runtime import RUN_KEY
        from comfy.k_diffusion import sampling
        from comfy.patcher_extension import WrappersMP

        class FlowSampling(comfy.model_sampling.CONST):
            def percent_to_sigma(self, percent):
                return 1 - percent

        for function in (sampling.sample_euler, sampling.sample_er_sde):
            with self.subTest(sampler=function.__name__), torch.inference_mode():
                base = self.model(True, num_layers=42)
                tuned, _ = install_block_tune(base, 'image_2_global_qkv',
                                              'middle_30_70')
                config = AFMConfig(mode='observe', block_scope='24-26', logging='off')
                combined = install_afm(tuned, config)
                self.assertEqual(len(tuned.get_all_wrappers(WrappersMP.OUTER_SAMPLE)), 1)
                self.assertEqual(len(tuned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)), 1)
                self.assertEqual(len(combined.get_all_wrappers(WrappersMP.OUTER_SAMPLE)), 1)
                self.assertEqual(len(combined.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)), 2)

                outer, = combined.get_all_wrappers(WrappersMP.OUTER_SAMPLE)
                original_options = combined.model_options
                guider = SimpleNamespace(model_options=original_options)
                observed = []
                sampler = SimpleNamespace(sampler_function=function)
                native = combined.model.diffusion_model
                text = torch.randn(1, 4, 16)

                class Model:
                    def __init__(self):
                        self.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
                            get_model_object=lambda key: FlowSampling()))

                    def __call__(self, latent, sigma, **kwargs):
                        options = guider.model_options['transformer_options']
                        options = dict(options, wrappers=combined.wrappers,
                                       cond_or_uncond=[0])
                        state = options[RUN_KEY]
                        observed.append((state.step_index, state.progress))
                        return native(latent, sigma, text, options)

                def execute(noise, latent, sampler_arg, sigmas, mask, callback,
                            disabled, seed, shapes):
                    return function(
                        Model(), torch.ones(1, 8, 2, 3), sigmas,
                        callback=lambda data: callback(data['i'], data['denoised'],
                                                       data['x'], 3),
                        disable=True, s_noise=0)

                def fail_after_evaluation(noise, latent, sampler_arg, sigmas, mask,
                                          callback, disabled, seed, shapes):
                    Model()(torch.ones(1, 8, 2, 3), sigmas[:1])
                    raise RuntimeError('injected combined sampler failure')

                combined.patch_model(device_to=torch.device('cpu'), load_weights=True)
                try:
                    with self.assertRaisesRegex(RuntimeError, 'injected combined'):
                        outer(FakeExecutor(guider, fail_after_evaluation), None, None,
                              sampler, torch.tensor([1., .7, .3, 0.]))
                    self.assertIs(guider.model_options, original_options)
                    qkv = native.layers[0].proj['qkv']
                    with self.assertRaisesRegex(RuntimeError, 'active text-to-image'):
                        qkv(torch.randn(1, 3, 64))
                    observed.clear()
                    output = outer(FakeExecutor(guider, execute), None, None, sampler,
                                   torch.tensor([1., .7, .3, 0.]))
                finally:
                    combined.unpatch_model(device_to=torch.device('cpu'))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(sorted(set(observed)), [(0, 0.), (1, .5), (2, 1.)])
                self.assertIs(guider.model_options, original_options)

                # Creating and running the AFM clone must not consume the source
                # model's standalone clock/registration.
                source_output, source_state = self.evaluate(
                    tuned, config, 1, torch.randn(1, 8, 2, 3), text)
                self.assertTrue(torch.isfinite(source_output).all())
                self.assertEqual(source_state.step_index, 1)

