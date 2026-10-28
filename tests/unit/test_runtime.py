import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    def test_residual_afm_repeated_runs_cleanup(self):
        import json
        from afm.residual import ResidualConfig, RESIDUAL_CONFIG_KEY
        residual = ResidualConfig(0.75, 1.15)
        original = {'transformer_options': {RESIDUAL_CONFIG_KEY: residual}}
        guider = SimpleNamespace(model_options=original)
        states = []
        def execute(*args):
            state = guider.model_options['transformer_options'][RUN_KEY]
            states.append(state)
            self.assertEqual(state.step_index, 0)
            self.assertIs(guider.model_options['transformer_options'][RESIDUAL_CONFIG_KEY], residual)
            if len(states) == 1:
                raise RuntimeError('injected')
            args[5](0, None, None, 1)
            return 'done'
        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_outer_wrapper(AFMConfig(), lambda s: None, lambda: directory)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                wrapper(FakeExecutor(guider, execute), None, None, None, torch.tensor([1., 0.]))
            self.assertIs(guider.model_options, original)
            self.assertEqual(wrapper(FakeExecutor(guider, execute), None, None, None,
                                     torch.tensor([1., 0.])), 'done')
            self.assertIs(guider.model_options, original)
            logs = list((Path(directory) / 'moss_afm').glob('*.jsonl'))
            self.assertEqual(len(logs), 2)
            for log in logs:
                first = json.loads(log.read_text().splitlines()[0])
                self.assertEqual(first['residual_config'],
                                 {'mid_multiplier': 0.75, 'late_multiplier': 1.15})
        self.assertNotEqual(states[0].run_id, states[1].run_id)
        self.assertTrue(all(s.finished for s in states))

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
                    {"layout": TokenLayout((2, 3), 4), "branches": [0], "block": 0,
                     "interval_ids": (1,)},
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

        context = {"layout": TokenLayout((2, 3), 4), "branches": [0, 1], "block": 0,
                   "interval_ids": (1,)}
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
        context = {"layout": TokenLayout((2, 3), 4), "branches": [0], "block": 0,
                   "interval_ids": (1,)}
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
                {"layout": TokenLayout((2, 3), 4), "branches": [0, 1], "block": 0,
                 "interval_ids": (1,)},
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


class IntervalRuntimeTests(unittest.TestCase):
    def test_two_interval_run_logs_config_and_resets(self):
        import json
        config = AFMConfig(mode="edit", block_scope="0", end_percent=0.5,
                           two_interval_mode=True, second_block_scope="2")
        guider = SimpleNamespace(model_options={"transformer_options": {"sentinel": True}})
        original = guider.model_options
        beginnings = []
        flat = [torch.randn(1, 10, 8) for _ in range(3)]
        baseline = torch.zeros(1, 10, 8)

        def execute(noise, latent, sampler, sigmas, mask, callback, *args):
            state = guider.model_options["transformer_options"][RUN_KEY]
            beginnings.append(state.step_index)
            for step in range(5):
                for block, ids in ((0, (1,)), (2, (2,))):
                    context = {"layout": TokenLayout((2, 3), 4), "branches": [0],
                               "block": block, "interval_ids": ids}
                    hook = AttentionOverride(config, state, context, None,
                                             lambda *a, **kw: baseline)
                    hook.container_function(*flat, 2)
                callback(step, None, None, 5)
            return "done"

        def broken(*args):
            state = guider.model_options["transformer_options"][RUN_KEY]
            beginnings.append(state.step_index)
            raise RuntimeError("interrupt run")

        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_outer_wrapper(config, lambda sampler: None, lambda: directory)
            for _ in range(2):
                self.assertEqual(wrapper(FakeExecutor(guider, execute), None, None, object(),
                                         torch.linspace(1, 0, 6)), "done")
                self.assertIs(guider.model_options, original)
            with self.assertRaisesRegex(RuntimeError, "interrupt run"):
                wrapper(FakeExecutor(guider, broken), None, None, object(), torch.tensor([1, 0]))
            self.assertIs(guider.model_options, original)
            runs = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                    for path in (Path(directory) / "moss_afm").glob("*.jsonl")]
        self.assertEqual(beginnings, [0, 0, 0])
        self.assertEqual(len({rows[0]["run_id"] for rows in runs}), 3)
        successes = [rows for rows in runs if rows[-1]["status"] == "success"]
        self.assertEqual(len(successes), 2)
        for rows in successes:
            settings = rows[0]["config"]
            self.assertEqual((settings["two_interval_mode"], settings["second_block_scope"],
                              settings["second_start_percent"], settings["second_end_percent"]),
                             (True, "2", 0.5, 1.0))
            summary = rows[-1]
            self.assertEqual((summary["callbacks"], summary["edited_calls"],
                              summary["skipped_calls"], summary["fallback_calls"]), (5, 5, 5, 0))
            self.assertEqual({(c["step"], c["block"]) for c in summary["calls"]
                              if c["action"] == "edit"}, {(0, 0), (1, 0), (2, 0), (3, 2), (4, 2)})

    def hook(self, *, step=3, interval_ids=(2,), **settings):
        config = AFMConfig(mode="edit", strength=0.2, end_percent=0.5,
                           two_interval_mode=True, logging="off", **settings)
        state = RunState(5, config, step_index=step)
        generator = torch.Generator().manual_seed(37)
        flat = [torch.randn(2, 10, 8, generator=generator) for _ in range(3)]
        q, k, v = [x.reshape(2, 10, 2, 4).transpose(1, 2) for x in flat]
        baseline = ((q @ k.transpose(-2, -1) * 0.5).softmax(-1) @ v)
        baseline = baseline.transpose(1, 2).reshape(2, 10, 8)
        context = {"layout": TokenLayout((2, 3), 4), "branches": [0, 1],
                   "block": 0, "interval_ids": interval_ids}
        hook = AttentionOverride(config, state, context, None, lambda *a, **kw: baseline)
        return hook, state, baseline, flat

    def test_second_interval_uses_global_progress(self):
        from afm.attention import edit_image_rows
        hook, state, baseline, flat = self.hook()
        with patch("afm.adapter.edit_image_rows", wraps=edit_image_rows) as kernel:
            got = hook.container_function(*flat, 2)
        self.assertEqual(state.edited_calls, 1)
        self.assertEqual(kernel.call_args.args[5], 0.75)
        self.assertFalse(torch.equal(got[0, :6], baseline[0, :6]))

    def test_overlap_suppresses_secondary_blocks(self):
        for mode in ("edit", "observe"):
            for interval_ids, expected in (((1,), "active"), ((2,), "skipped")):
                with self.subTest(mode=mode, interval_ids=interval_ids):
                    hook, state, baseline, flat = self.hook(step=2, interval_ids=interval_ids)
                    from dataclasses import replace
                    hook.config = replace(hook.config, mode=mode,
                                          second_start_percent=0.25)
                    got = hook.container_function(*flat, 2)
                    if expected == "skipped":
                        self.assertIs(got, baseline)
                        self.assertEqual((state.edited_calls, state.observed_calls,
                                          state.skipped_calls), (0, 0, 1))
                    elif mode == "edit":
                        self.assertEqual(state.edited_calls, 1)
                    else:
                        self.assertEqual(state.observed_calls, 1)

    def test_shared_blocks_edit_once_per_call(self):
        from afm.attention import edit_image_rows
        for step in (2, 3):
            hook, state, _, flat = self.hook(step=step, interval_ids=(1, 2))
            with patch("afm.adapter.edit_image_rows", wraps=edit_image_rows) as kernel:
                hook.container_function(*flat, 2)
            self.assertEqual(kernel.call_count, 1)
            self.assertEqual(state.edited_calls, 1)

    def test_gap_dispatches_native(self):
        hook, state, baseline, flat = self.hook(step=3, second_start_percent=0.9)
        with patch("afm.adapter.edit_image_rows", side_effect=AssertionError("gap edited")):
            self.assertIs(hook.container_function(*flat, 2), baseline)
        self.assertEqual(state.skipped_calls, 1)

    def test_multiple_evaluations_keep_interval(self):
        hook, state, _, flat = self.hook()
        for _ in range(2):
            hook.container_function(*flat, 2)
        self.assertEqual(state.step_index, 3)
        self.assertEqual(state.edited_calls, 2)
        self.assertEqual(state.calls[(3, 0, 0, "edit")], 2)

    def test_second_interval_preserves_cfg_rows_and_fallback(self):
        hook, state, baseline, flat = self.hook(fail_mode="fallback_original")
        got = hook.container_function(*flat, 2)
        self.assertEqual(state.edited_calls, 1)
        self.assertTrue(torch.equal(got[1], baseline[1]))
        self.assertTrue(torch.equal(got[0, 6:], baseline[0, 6:]))
        with patch("afm.adapter.edit_image_rows", side_effect=FloatingPointError("failure")):
            self.assertIs(hook.container_function(*flat, 2), baseline)
        self.assertEqual(state.fallback_calls, 1)
        self.assertEqual(state.calls[(3, 0, 1, "edit")], 0)


class FakeExecutor:
    def __init__(self, owner, func):
        self.class_obj, self.func = owner, func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
