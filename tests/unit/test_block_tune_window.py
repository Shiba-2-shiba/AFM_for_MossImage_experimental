from contextvars import ContextVar, copy_context
from types import SimpleNamespace
import unittest

import torch

from afm.block_tune import ScaleRows
from afm.config import AFMConfig
from afm.runtime import RUN_KEY, RunState, make_outer_wrapper
from tests.unit.test_runtime import FakeExecutor


class BlockTuneWindowTests(unittest.TestCase):
    def test_window_uses_sampler_steps_and_repeated_evaluations_not_timestep(self):
        from afm.block_tune_window import GatedScaleRows, make_window_wrapper
        gate = ContextVar('test_gate', default=None)
        scale = GatedScaleRows(ScaleRows((1.7, 1., .8)), gate)
        diffusion = make_window_wrapper('middle_30_70', gate)
        original = {'transformer_options': {}}
        guider = SimpleNamespace(model_options=original)
        observed = []
        weight = torch.ones(3, 2)

        def evaluate(x, t, c, options, **kwargs):
            observed.append(scale(weight)[:, 0].tolist())
            return x

        def sample(noise, latent, sampler, sigmas, mask, callback, disabled, seed, shapes):
            options = guider.model_options['transformer_options']
            for step in range(11):
                for _ in range(2):
                    diffusion(FakeExecutor(None, evaluate), None, 999, None, options)
                    self.assertIsNone(gate.get())
                callback(step, None, None, 11)
            return 'done'

        outer = make_outer_wrapper(AFMConfig(logging='off'), lambda s: None, lambda: None)
        result = outer(FakeExecutor(guider, sample), None, None, None, torch.linspace(1, 0, 12))
        self.assertEqual(result, 'done')
        self.assertIs(guider.model_options, original)
        for index, row in enumerate(observed):
            expected = torch.tensor([1.7, 1., .8] if 3 <= index // 2 <= 7 else [1., 1., 1.])
            torch.testing.assert_close(torch.tensor(row), expected, rtol=0, atol=0)

    def test_full_identity_and_window_boundaries(self):
        from afm.block_tune_window import GatedScaleRows, make_window_wrapper
        gate = ContextVar('test_gate', default=None)
        scale = GatedScaleRows(ScaleRows((.8,)), gate)
        weight = torch.ones(2, 3)
        for mode in ('full_run', 'middle_30_70', 'identity_control'):
            wrapper = make_window_wrapper(mode, gate)
            for index in range(25):
                state = RunState(25, AFMConfig(logging='off'), step_index=index)
                actual = wrapper(FakeExecutor(None, lambda *a, **k: scale(weight)),
                                 None, None, None, {RUN_KEY: state})
                active = mode == 'full_run' or (mode == 'middle_30_70' and 8 <= index <= 16)
                torch.testing.assert_close(actual, weight * (.8 if active else 1), rtol=0, atol=0)

    def test_cleanup_missing_run_and_context_isolation(self):
        from afm.block_tune_window import GatedScaleRows, make_window_wrapper
        gate = ContextVar('test_gate', default=None)
        other_context = copy_context()
        scale = GatedScaleRows(ScaleRows((.8,)), gate)
        wrapper = make_window_wrapper('full_run', gate)
        state = RunState(3, AFMConfig(logging='off'))

        def fail(*args, **kwargs):
            self.assertTrue(gate.get())
            self.assertIsNone(other_context.run(gate.get))
            raise RuntimeError('injected projection failure')

        with self.assertRaisesRegex(RuntimeError, 'injected'):
            wrapper(FakeExecutor(None, fail), None, None, None, {RUN_KEY: state})
        self.assertIsNone(gate.get())
        with self.assertRaises(RuntimeError):
            scale(torch.ones(2, 3))
        with self.assertRaises(RuntimeError):
            wrapper(FakeExecutor(None, fail), None, None, None, {})
        state.finished = True
        with self.assertRaises(RuntimeError):
            wrapper(FakeExecutor(None, fail), None, None, None, {RUN_KEY: state})
        with self.assertRaises(ValueError):
            make_window_wrapper('invalid', gate)
