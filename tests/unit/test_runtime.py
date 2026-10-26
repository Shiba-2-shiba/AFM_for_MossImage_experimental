import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from afm.config import AFMConfig, TokenLayout
from afm.runtime import RunState, branch_rows, make_outer_wrapper, make_diffusion_wrapper, RUN_KEY
from afm.adapter import AttentionOverride


class Container:
    def __init__(self, tensor):
        self.tensor = tensor

    def peek(self):
        if self.tensor is None:
            raise RuntimeError("consumed")
        return self.tensor

    def take(self):
        tensor = self.peek()
        self.tensor = None
        return tensor


class RuntimeTests(unittest.TestCase):
    def test_multiple_rows_clone_once_and_failure_keeps_native_baseline(self):
        from unittest.mock import patch
        from afm.attention import edit_image_rows
        config = AFMConfig(mode="edit", branch_mode="both", logging="off",
                           fail_mode="fallback_original")
        flat = [torch.randn(3, 10, 12) for _ in range(3)]
        baseline = torch.randn(3, 10, 12)
        original = baseline.clone()
        clones = []
        native_clone = torch.Tensor.clone

        def clone(tensor, *args, **kwargs):
            clones.append(tuple(tensor.shape))
            return native_clone(tensor, *args, **kwargs)

        for fail in (False, True):
            state = RunState(2, config)
            hook = AttentionOverride(config, state,
                    {"layout": TokenLayout((2, 3), 4), "branches": [0], "block": 0},
                    None, lambda *a, **kw: baseline)
            calls = []

            def kernel(*args, **kwargs):
                calls.append(1)
                if fail and len(calls) == 2:
                    raise FloatingPointError("second row failure")
                return edit_image_rows(*args, **kwargs)

            clones.clear()
            with patch.object(torch.Tensor, "clone", clone), \
                    patch("afm.adapter.edit_image_rows", kernel):
                got = hook.container_function(*flat, 3)
            self.assertEqual(clones.count(tuple(baseline.shape)), 1)
            self.assertTrue(torch.equal(baseline, original))
            if fail:
                self.assertIs(got, baseline)
                self.assertEqual(state.fallback_calls, 1)
            else:
                self.assertTrue(torch.equal(got[:, 6:], baseline[:, 6:]))
                self.assertEqual(state.edited_calls, 1)

    def test_cfg_layout_uses_metadata_not_batch_position(self):
        self.assertEqual(branch_rows([1, 0], 4, "positive_only"), (2, 3))
        self.assertEqual(branch_rows([0], 2, "positive_only"), (0, 1))
        self.assertEqual(branch_rows([1], 2, "positive_only"), ())
        self.assertEqual(branch_rows([0, 1], 4, "both"), (0, 1, 2, 3))
        for branches,batch in [(None, 2), ([2], 1), ([0, 1], 3)]:
            with self.assertRaises(ValueError):
                branch_rows(branches, batch, "both")

    def test_observe_preserves_dispatch_preference_and_consumes_once(self):
        qkv = [torch.randn(2, 10, 12) for _ in range(3)]
        containers = [Container(t) for t in qkv]
        preferred = object()
        state = RunState(2, AFMConfig(mode="observe", logging="off"))
        calls = []
        expected = torch.randn(2, 10, 12)

        def dispatcher(q, k, v, heads, **kw):
            self.assertIs(kw["preferred_attention"], preferred)
            self.assertNotIn("optimized_attention_override", kw["transformer_options"])
            self.assertNotIn("_inside_attn_wrapper", kw)
            calls.append(tuple(c.take().shape for c in (q, k, v)))
            return expected

        context = {"layout": TokenLayout((2, 3), 4), "branches": [0, 1], "block": 0}
        hook = AttentionOverride(AFMConfig(mode="observe", logging="off"),
                                 state, context, preferred, dispatcher)
        output = hook.container_function(*containers, 3,
                 transformer_options={"optimized_attention_override": hook},
                 _inside_attn_wrapper=True)
        self.assertIs(output, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state.observed_calls, 1)
        self.assertTrue(all(c.tensor is None for c in containers))

    def test_edit_is_atomic_on_kernel_failure_and_backend_error_is_not_retried(self):
        from unittest.mock import patch
        config = AFMConfig(mode="edit", fail_mode="fallback_original", logging="off")
        state = RunState(2, config)
        context = {"layout": TokenLayout((2, 3), 4), "branches": [0], "block": 0}
        expected = torch.randn(1, 10, 12)
        calls = []

        def dispatcher(*args, **kwargs):
            calls.append(1)
            for c in args[:3]:
                c.take()
            return expected

        hook = AttentionOverride(config, state, context, None, dispatcher)
        with patch("afm.adapter.edit_image_rows", side_effect=FloatingPointError("test")):
            got = hook.container_function(*[Container(torch.randn(1, 10, 12)) for _ in range(3)], 3)
        self.assertIs(got, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state.fallback_calls, 1)
        self.assertEqual(state.edited_calls, 0)

        def broken(*args, **kwargs):
            raise RuntimeError("native failure")
        hook = AttentionOverride(config, state, context, None, broken)
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            hook.container_function(*[Container(torch.randn(1, 10, 12)) for _ in range(3)], 3)

    def test_callback_owns_step_and_cleanup_even_on_interrupt(self):
        config = AFMConfig(mode="observe", logging="off")
        guider = SimpleNamespace(model_options={"transformer_options": {"sentinel": True}})
        original = guider.model_options
        steps = []

        def execute(noise, latent, sampler, sigmas, denoise_mask, callback,
                    disable_pbar, seed, latent_shapes):
            state = guider.model_options["transformer_options"][RUN_KEY]
            for i in range(3):
                self.assertEqual(state.step_index, i)
                state.model_evaluations += 2
                callback(i, None, None, 3)
            return "done"

        executor = SimpleNamespace(class_obj=guider)
        executor = FakeExecutor(guider, execute)
        wrapper = make_outer_wrapper(config, lambda sampler: None, lambda: None)
        got = wrapper(executor, None, None, object(), torch.tensor([1, .7, .3, 0]),
                      callback=lambda i,*_: steps.append(i))
        self.assertEqual(got, "done")
        self.assertEqual(steps, [0, 1, 2])
        self.assertIs(guider.model_options, original)

        def interrupt(*args):
            self.assertIn(RUN_KEY, guider.model_options["transformer_options"])
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            wrapper(FakeExecutor(guider, interrupt), None, None, object(),
                    torch.tensor([1, 0]))
        self.assertIs(guider.model_options, original)

    def test_missing_callback_is_not_success(self):
        config = AFMConfig(mode="edit", logging="off")
        guider = SimpleNamespace(model_options={"transformer_options": {}})
        wrapper = make_outer_wrapper(config, lambda sampler: None, lambda: None)
        with self.assertRaisesRegex(RuntimeError, "callback"):
            wrapper(FakeExecutor(guider, lambda *a: "ignored"), None, None,
                    object(), torch.tensor([1, 0]))

    def test_positive_edit_keeps_text_and_negative_rows_and_logs_actual_branch(self):
        config = AFMConfig(mode="edit", logging="off", branch_mode="positive_only")
        state = RunState(2, config)
        torch.manual_seed(9)
        flat = [torch.randn(2, 10, 12) for _ in range(3)]
        q, k, v = [x.reshape(2, 10, 3, 4).transpose(1, 2) for x in flat]
        expected = ((q @ k.transpose(-2, -1) * 0.5).softmax(-1) @ v)
        expected = expected.transpose(1, 2).reshape(2, 10, 12)
        def dispatch(*args, **kwargs):
            for c in args[:3]:
                c.take()
            return expected
        hook = AttentionOverride(config, state,
                {"layout": TokenLayout((2, 3), 4), "branches": [0, 1], "block": 0},
                None, dispatch)
        got = hook.container_function(*[Container(x) for x in flat], 3)
        self.assertTrue(torch.equal(got[1], expected[1]))
        self.assertTrue(torch.equal(got[0, 6:], expected[0, 6:]))
        self.assertFalse(torch.equal(got[0, :6], expected[0, :6]))
        self.assertEqual(state.calls[(0, 0, 0, "edit")], 1)
        self.assertEqual(state.calls[(0, 0, 1, "edit")], 0)

    def test_layout_log_uses_actual_grid_and_deduplicates_within_run(self):
        import json
        config = AFMConfig(mode="observe")
        state = RunState(2, config)
        with tempfile.TemporaryDirectory() as directory:
            state.log_path = Path(directory) / "run.jsonl"
            opts = {RUN_KEY: state, "cond_or_uncond": [0]}
            wrapper = make_diffusion_wrapper(config)
            executor = FakeExecutor(None, lambda x,t,c,o,**kw: x)
            for _ in range(2):
                wrapper(executor, torch.zeros(1, 8, 3, 5), torch.ones(1),
                        torch.zeros(1, 4, 16), opts)
            records = [json.loads(line) for line in state.log_path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["grid"], [3, 5])
        self.assertEqual(records[0]["text_tokens"], 4)
        self.assertEqual(records[0]["branches"], [0])


class FakeExecutor:
    def __init__(self, owner, func):
        self.class_obj, self.func = owner, func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
