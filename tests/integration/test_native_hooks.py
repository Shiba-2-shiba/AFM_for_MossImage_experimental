import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


COMFY_ROOT = os.environ.get("MOSS_COMFY_ROOT")


@unittest.skipUnless(COMFY_ROOT, "set MOSS_COMFY_ROOT for actual ComfyUI contract tests")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, COMFY_ROOT)
        from comfy.cli_args import args
        args.cpu = True
        import comfy.model_management
        cls.training = patch.object(comfy.model_management, "in_training", True)
        cls.training.start()

    @classmethod
    def tearDownClass(cls):
        cls.training.stop()

    def model(self, fused, num_layers=3):
        import comfy.ops
        from comfy.ldm.moss_image.model import MOSSImage
        from comfy.model_patcher import ModelPatcher
        torch.manual_seed(33)
        native = MOSSImage(dim=64, ffn_dim=96, num_heads=2, num_layers=num_layers,
                           text_dim=16, latent_channels=8, fused_qkv=fused,
                           device="cpu", dtype=torch.float32,
                           operations=comfy.ops.disable_weight_init)
        for param in native.parameters():
            param.data.normal_(0, 0.05)
        base = torch.nn.Module()
        base.diffusion_model = native
        return ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))

    def test_native_observe_matches_original_and_only_joint_blocks_are_observed(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.runtime import RUN_KEY, RunState
        from comfy.patcher_extension import WrappersMP
        for fused in [False, True]:
            model = self.model(fused)
            native = model.model.diffusion_model
            x, t, text = torch.randn(1, 8, 3, 5), torch.ones(1), torch.randn(1, 4, 16)
            original = native(x.clone(), t, text, {})
            config = AFMConfig(mode="observe", block_scope="all", logging="off")
            clone = install_afm(model, config)
            options = dict(clone.model_options["transformer_options"])
            options["wrappers"] = clone.wrappers
            state = RunState(1, config)
            options[RUN_KEY] = state
            options["cond_or_uncond"] = [0]
            observed = native(x.clone(), t, text, options)
            torch.testing.assert_close(observed, original, atol=0, rtol=0)
            self.assertEqual(state.observed_calls, 3)
            self.assertEqual(state.model_evaluations, 1)
            self.assertNotIn("patches_replace", model.model_options["transformer_options"])
            self.assertTrue(clone.get_all_wrappers(WrappersMP.OUTER_SAMPLE))

    def test_noop_has_no_hooks_and_duplicate_patch_is_rejected(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        model = self.model(True)
        for config in [AFMConfig(mode="off"), AFMConfig(mode="edit", strength=0)]:
            clone = install_afm(model, config)
            self.assertEqual(clone.model_options, model.model_options)
            self.assertEqual(clone.wrappers, model.wrappers)
        clone = install_afm(model, AFMConfig(logging="off"))
        with self.assertRaises(ValueError):
            install_afm(clone, AFMConfig(logging="off"))

    def test_two_interval_native_outputs_match_single_interval(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.runtime import RUN_KEY, RunState
        for fused in (False, True):
            model = self.model(fused)
            native = model.model.diffusion_model
            x, t, text = torch.randn(1, 8, 3, 5), torch.ones(1), torch.randn(1, 4, 16)

            def evaluate(config, step):
                clone = install_afm(model, config)
                options = dict(clone.model_options["transformer_options"])
                options.update(wrappers=clone.wrappers, cond_or_uncond=[0])
                state = RunState(5, config, step_index=step)
                options[RUN_KEY] = state
                return native(x.clone(), t, text, options), state

            cases = [("0", 0.5, 0.5, (0, 2, 4), ("0", "0", "2")),
                     ("0", 0.25, 0.75, (2,), (None,)),
                     ("2", 0.5, 0.25, (2, 4), ("2", "2"))]
            for primary, end, start, steps, wanted in cases:
                config = AFMConfig(mode="edit", strength=0.2, block_scope=primary,
                                   end_percent=end, two_interval_mode=True,
                                   second_block_scope="2", second_start_percent=start,
                                   logging="off")
                for step, scope in zip(steps, wanted):
                    with self.subTest(fused=fused, primary=primary, step=step, scope=scope):
                        actual, state = evaluate(config, step)
                        if scope is None:
                            expected = native(x.clone(), t, text, {"cond_or_uncond": [0]})
                        else:
                            expected, _ = evaluate(AFMConfig(mode="edit", strength=0.2,
                                                            block_scope=scope, logging="off"), step)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        self.assertEqual(state.edited_calls, 0 if scope is None else 1)

    def test_two_interval_presets_register_one_wrapper_and_union(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from comfy.patcher_extension import WrappersMP
        model = self.model(True, num_layers=33)
        config = AFMConfig(block_scope="30-32", end_percent=0.5, two_interval_mode=True,
                           logging="off")
        clone = install_afm(model, config)
        patches = clone.model_options["transformer_options"]["patches_replace"]["dit"]
        self.assertEqual(set(patches), {("single_block", n) for n in (0, 1, 2, 30, 31, 32)})
        for kind in (WrappersMP.OUTER_SAMPLE, WrappersMP.DIFFUSION_MODEL):
            self.assertEqual(len(clone.get_all_wrappers(kind)), 1)
        self.assertNotIn("patches_replace", model.model_options["transformer_options"])
        with self.assertRaises(ValueError):
            install_afm(clone, config)

    def test_disabled_secondary_scope_is_not_parsed(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        model = self.model(True)
        install_afm(model, AFMConfig(second_block_scope="invalid", logging="off"))
        for scope in ("invalid", "30-32"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                install_afm(model, AFMConfig(two_interval_mode=True,
                                             second_block_scope=scope, logging="off"))
        for config in (AFMConfig(mode="off", two_interval_mode=True),
                       AFMConfig(mode="edit", strength=0, two_interval_mode=True)):
            clone = install_afm(model, config)
            self.assertEqual(clone.model_options, model.model_options)
            self.assertEqual(clone.wrappers, model.wrappers)

    def node_class(self, node_id='MossImageAFM'):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("moss_afm_test_pack", root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        pack = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pack
        spec.loader.exec_module(pack)
        import asyncio
        extension = asyncio.run(pack.comfy_entrypoint())
        nodes = asyncio.run(extension.get_node_list())
        return next(node for node in nodes if node.define_schema().node_id == node_id)

    def test_residual_v3_schema(self):
        node = self.node_class('MossImageResidual')
        schema = node.define_schema()
        self.assertEqual(schema.category, 'model/patches/moss_image')
        self.assertEqual([n.id for n in schema.inputs], ['model', 'mid_multiplier', 'late_multiplier'])
        self.assertEqual(len(schema.outputs), 1)
        for control in schema.inputs[1:]:
            self.assertEqual((control.default, control.min, control.max, control.step), (1, 0, 2, 0.01))
        model = self.model(True, num_layers=42)
        output = node.execute(model).result[0]
        self.assertIsNot(output, model)
        self.assertEqual(output.model_options, model.model_options)
        output = node.execute(model, 0.75, 1.15).result[0]
        self.assertEqual(len(output.model_options['transformer_options']['patches_replace']['dit']), 25)

    def test_native_lora_residual_afm_prompt_validation(self):
        import asyncio
        import execution
        import folder_paths
        import nodes as comfy_nodes

        class Source:
            RETURN_TYPES = ('MODEL', 'CLIP')
            @classmethod
            def INPUT_TYPES(cls):
                return {'required': {}}

        class Output:
            RETURN_TYPES = ()
            OUTPUT_NODE = True
            @classmethod
            def INPUT_TYPES(cls):
                return {'required': {'model': ('MODEL',)}}

        residual_node = self.node_class('MossImageResidual')
        afm_node = self.node_class()
        mapping = {'ResidualTestSource': Source, 'ResidualTestOutput': Output,
                   'MossImageResidual': residual_node, 'MossImageAFM': afm_node}
        with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, mapping), \
                patch.object(folder_paths, 'get_filename_list', return_value=['test.safetensors']):
            for loader_type in ('LoraLoaderModelOnly', 'LoraLoader'):
                inputs = {'model': ['1', 0], 'lora_name': 'test.safetensors', 'strength_model': 0.6}
                if loader_type == 'LoraLoader':
                    inputs.update(clip=['1', 1], strength_clip=0.5)
                prompt = {'1': {'class_type': 'ResidualTestSource', 'inputs': {}},
                          '2': {'class_type': loader_type, 'inputs': inputs},
                          '3': {'class_type': 'MossImageResidual', 'inputs': {
                              'model': ['2', 0], 'mid_multiplier': 0.75, 'late_multiplier': 1.15}},
                          '4': {'class_type': 'MossImageAFM', 'inputs': {
                              'model': ['3', 0], 'mode': 'observe', 'strength': 0.05,
                              'block_scope': '24-26', 'branch_mode': 'positive_only'}},
                          '5': {'class_type': 'ResidualTestOutput', 'inputs': {'model': ['4', 0]}}}
                # Validate the actual shipped example's settings, not reconstructed defaults.
                import json
                example = json.loads((Path(__file__).resolve().parents[2] /
                                      'examples/moss_lora_residual_afm_api.json').read_text(encoding='utf-8'))
                settings = next(n['inputs'] for n in example.values() if n['class_type'] == 'MossImageAFM')
                prompt['4']['inputs'] = {**settings, 'model': ['3', 0]}
                result = asyncio.run(execution.validate_prompt('residual-api', prompt, None))
                self.assertTrue(result[0], result)
                for value in (-0.1, float('nan'), float('inf'), 'nan'):
                    prompt['3']['inputs']['mid_multiplier'] = value
                    with self.assertLogs(level='ERROR'):
                        self.assertFalse(asyncio.run(execution.validate_prompt('residual-invalid', prompt, None))[0])
                # ComfyUI itself canonicalizes FLOAT literals before node validation.
                for value, expected in ((True, 1.0), (False, 0.0), ('0.75', 0.75)):
                    prompt['3']['inputs']['mid_multiplier'] = value
                    self.assertTrue(asyncio.run(execution.validate_prompt('residual-coercion', prompt, None))[0])
                    self.assertEqual(prompt['3']['inputs']['mid_multiplier'], expected)

    def test_residual_native_noop_and_branch_isolation(self):
        from afm.residual import ResidualConfig, install_residual
        for fused in (False, True):
            model = self.model(fused, num_layers=42)
            native = model.model.diffusion_model
            x, t, text = torch.randn(2, 8, 2, 3), torch.ones(2), torch.randn(2, 4, 16)
            before = native(x.clone(), t, text, {})
            clone = install_residual(model, ResidualConfig())
            self.assertIsNot(clone, model)
            self.assertEqual(clone.model_options, model.model_options)
            torch.testing.assert_close(native(x.clone(), t, text, clone.model_options['transformer_options']),
                                       before, atol=0, rtol=0)
            patched = install_residual(model, ResidualConfig(0.75, 1.15))
            result = native(x.clone(), t, text, patched.model_options['transformer_options'])
            self.assertFalse(torch.equal(result, before))
            torch.testing.assert_close(native(x.clone(), t, text, {}), before, atol=0, rtol=0)
            self.assertNotIn('patches_replace', model.model_options['transformer_options'])
            for config, wanted in ((ResidualConfig(0.75, 1), range(9, 26)),
                                   (ResidualConfig(1, 1.15), range(34, 42))):
                clone = install_residual(model, config)
                self.assertEqual(set(clone.model_options['transformer_options']['patches_replace']['dit']),
                                 {('single_block', i) for i in wanted})

    def test_residual_matches_projection_scaling(self):
        from afm.residual import ResidualConfig, install_residual
        for fused in (False, True):
            for factor in (0, 0.75, 1.15, 2):
                with self.subTest(fused=fused, factor=factor), torch.inference_mode():
                    model = self.model(fused, num_layers=42)
                    reference = self.model(fused, num_layers=42)
                    config = ResidualConfig(factor, factor)
                    for i, block in enumerate(reference.model.diffusion_model.layers):
                        for name in ('o', 'ffn_down'):
                            block.proj[name].weight.mul_(config.multiplier_for(i))
                    clone = install_residual(model, config)
                    x, t, text = torch.randn(1, 8, 2, 3), torch.ones(1), torch.randn(1, 4, 16)
                    actual = model.model.diffusion_model(x.clone(), t, text,
                                                         clone.model_options['transformer_options'])
                    expected = reference.model.diffusion_model(x.clone(), t, text, {})
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_residual_rejects_invalid_models(self):
        from types import SimpleNamespace
        from afm.residual import ResidualConfig, install_residual
        for size in (41, 43):
            for config in (ResidualConfig(), ResidualConfig(0.75, 1.15)):
                with self.assertRaises(ValueError):
                    install_residual(self.model(True, num_layers=size), config)
        for native in (torch.nn.Linear(2, 2), None):
            with self.assertRaises(ValueError):
                install_residual(SimpleNamespace(model=SimpleNamespace(diffusion_model=native)),
                                 ResidualConfig())
        model = self.model(True, num_layers=42)
        model.model.diffusion_model._orig_mod = True
        with self.assertRaises(ValueError):
            install_residual(model, ResidualConfig())

    def test_residual_lora_is_scaled_after_application(self):
        from types import SimpleNamespace
        import comfy.sd
        from afm.residual import ResidualConfig, install_residual
        for fused in (False, True):
            for count in (1, 2):
                with self.subTest(fused=fused, count=count), torch.inference_mode():
                    model = self.model(fused, num_layers=42)
                    model.model.model_config = SimpleNamespace(unet_config={})
                    reference = self.model(fused, num_layers=42)
                    native = model.model.diffusion_model
                    keys = [(9, 'o'), (34, 'ffn_down'), (9, 'qkv' if fused else 'q')]
                    loaded = model
                    for strength in (0.5, -0.3)[:count]:
                        lora = {}
                        for index, name in keys:
                            weight = reference.model.diffusion_model.layers[index].proj[name].weight
                            up = torch.randn(weight.shape[0], 2) * 0.1
                            down = torch.randn(2, weight.shape[1]) * 0.1
                            key = f'diffusion_model.layers.{index}.proj.{name}'
                            lora[key + '.lora_up.weight'] = up
                            lora[key + '.lora_down.weight'] = down
                            weight.add_(strength * (up @ down))
                        loaded, _ = comfy.sd.load_lora_for_models(loaded, None, lora, strength, 0)
                    self.assertEqual(len(loaded.patches), 3)
                    clone = install_residual(loaded, ResidualConfig(0.75, 1.15))
                    self.assertEqual(set(clone.patches), set(loaded.patches))
                    for key in clone.patches:
                        self.assertEqual(len(clone.patches[key]), count)
                        clone.patch_weight_to_device(key, torch.device('cpu'))
                    x, t, text = torch.randn(1, 8, 2, 3), torch.ones(1), torch.randn(1, 4, 16)
                    try:
                        noop = install_residual(loaded, ResidualConfig())
                        torch.testing.assert_close(native(x.clone(), t, text, noop.model_options['transformer_options']),
                                                   native(x.clone(), t, text, {}), rtol=0, atol=0)
                        for i, block in enumerate(reference.model.diffusion_model.layers):
                            factor = 0.75 if 9 <= i <= 25 else 1.15 if 34 <= i <= 41 else 1
                            for name in ('o', 'ffn_down'):
                                block.proj[name].weight.mul_(factor)
                        actual = native(x.clone(), t, text, clone.model_options['transformer_options'])
                        expected = reference.model.diffusion_model(x.clone(), t, text, {})
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    finally:
                        clone.unpatch_model()
                    self.assertEqual(model.patches, {})
                    self.assertNotIn('moss_residual_config', loaded.model_options['transformer_options'])

    def evaluate_patched(self, model, x, text, config=None, step=0):
        from afm.runtime import RUN_KEY, RunState
        options = dict(model.model_options['transformer_options'])
        options.update(wrappers=model.wrappers, cond_or_uncond=[0, 1] if x.shape[0] == 2 else [0])
        state = RunState(5, config, step_index=step) if config is not None else None
        if state is not None:
            options[RUN_KEY] = state
        output = model.model.diffusion_model(x.clone(), torch.ones(x.shape[0]), text, options)
        return output, state

    def test_residual_afm_overlap_calls_native_once(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.residual import ResidualConfig, install_residual
        from comfy.patcher_extension import WrappersMP
        for fused in (False, True):
            model = self.model(fused, num_layers=42)
            reference = self.model(fused, num_layers=42)
            with torch.no_grad():
                for i, block in enumerate(reference.model.diffusion_model.layers):
                    factor = 0.75 if 9 <= i <= 25 else 1.15 if 34 <= i <= 41 else 1
                    for name in ('o', 'ffn_down'):
                        block.proj[name].weight.mul_(factor)
            config = AFMConfig(mode='edit', block_scope='24-26', strength=0.2, logging='off')
            residual = install_residual(model, ResidualConfig(0.75, 1.15))
            combined = install_afm(residual, config)
            counts = [0] * 42
            def count_for(i):
                def count(*args):
                    counts[i] += 1
                return count
            handles = [b.register_forward_hook(count_for(i))
                       for i, b in enumerate(model.model.diffusion_model.layers)]
            x, text = torch.randn(2, 8, 2, 3), torch.randn(2, 4, 16)
            try:
                actual, state = self.evaluate_patched(combined, x, text, config)
            finally:
                for handle in handles:
                    handle.remove()
            expected, _ = self.evaluate_patched(install_afm(reference, config), x, text, config)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertEqual(counts, [1] * 42)
            self.assertEqual(state.edited_calls, 3)
            residual_output, _ = self.evaluate_patched(residual, x, text)
            torch.testing.assert_close(actual[1], residual_output[1], rtol=0, atol=0)
            self.assertEqual(len(combined.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)), 1)
            self.assertEqual(len(combined.get_all_wrappers(WrappersMP.OUTER_SAMPLE)), 1)

    def test_residual_afm_noop_modes_keep_residual(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.residual import ResidualConfig, install_residual
        residual = install_residual(self.model(True, num_layers=42), ResidualConfig(0.75, 1.15))
        x, text = torch.randn(2, 8, 2, 3), torch.randn(2, 4, 16)
        baseline, _ = self.evaluate_patched(residual, x, text)
        for scope in ('0-2', '24-26', '30-32'):
            for mode, strength in (('off', 0.2), ('edit', 0), ('observe', 0.2)):
                config = AFMConfig(mode=mode, strength=strength, block_scope=scope, logging='off')
                result, _ = self.evaluate_patched(install_afm(residual, config), x, text, config)
                torch.testing.assert_close(result, baseline, rtol=0, atol=0)

    def test_residual_afm_two_intervals_keep_residual_in_gaps(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.residual import ResidualConfig, install_residual
        residual = install_residual(self.model(True, num_layers=42), ResidualConfig(0.75, 1.15))
        x, text = torch.randn(1, 8, 2, 3), torch.randn(1, 4, 16)
        for end, start, step, selected in ((0.5, 0.5, 2, '24-26'), (0.5, 0.5, 3, '0-2'),
                                           (0.25, 0.75, 2, None)):
            config = AFMConfig(mode='edit', block_scope='24-26', end_percent=end,
                               two_interval_mode=True, second_start_percent=start,
                               logging='off')
            actual, state = self.evaluate_patched(install_afm(residual, config), x, text, config, step)
            if selected is None:
                expected, _ = self.evaluate_patched(residual, x, text)
            else:
                single = AFMConfig(mode='edit', block_scope=selected, logging='off')
                expected, _ = self.evaluate_patched(install_afm(residual, single), x, text, single, step)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(state.edited_calls, 0 if selected is None else 3)

    def test_residual_afm_cfg_and_fallback(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.residual import ResidualConfig, install_residual
        residual = install_residual(self.model(True, num_layers=42), ResidualConfig(0.75, 1.15))
        x, text = torch.randn(2, 8, 2, 3), torch.randn(2, 4, 16)
        baseline, _ = self.evaluate_patched(residual, x, text)
        config = AFMConfig(mode='edit', block_scope='24-26', fail_mode='fallback_original', logging='off')
        combined = install_afm(residual, config)
        with patch('afm.adapter.edit_image_rows', side_effect=FloatingPointError('test')):
            result, state = self.evaluate_patched(combined, x, text, config)
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)
        self.assertEqual(state.fallback_calls, 3)
        recovered, state = self.evaluate_patched(combined, x, text, config)
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertEqual(state.edited_calls, 3)

    def test_only_known_residual_patches_are_accepted(self):
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.residual import ResidualBlockPatch, ResidualConfig, install_residual, RESIDUAL_CONFIG_KEY
        from comfy.patcher_extension import WrappersMP
        model = self.model(True, num_layers=42)
        settings = ResidualConfig(0.75, 1.15)
        residual = install_residual(model, settings)
        afm = install_afm(model, AFMConfig(logging='off'))
        for original in (residual, afm):
            with self.assertRaises(ValueError):
                install_residual(original, settings)
            clone = install_residual(original, ResidualConfig())
            self.assertEqual(clone.model_options, original.model_options)
        combined = install_afm(residual, AFMConfig(logging='off'))
        with self.assertRaises(ValueError):
            install_afm(combined, AFMConfig(logging='off'))
        invalid = []
        for replacement in (lambda args, extra: extra['original_block'](args),
                            ResidualBlockPatch(9, 0.5), ResidualBlockPatch(10, 0.75)):
            clone = residual.clone()
            clone.set_model_patch_replace(replacement, 'dit', 'single_block', 9)
            invalid.append(clone)
        clone = residual.clone()
        clone.model_options['transformer_options'].pop(RESIDUAL_CONFIG_KEY)
        invalid.append(clone)
        clone = residual.clone()
        clone.model_options['transformer_options']['patches_replace']['dit'].pop(('single_block', 41))
        invalid.append(clone)
        clone = residual.clone()
        clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, 'other', lambda *args: None)
        invalid.append(clone)
        for original in invalid:
            with self.assertRaises(ValueError):
                install_afm(original, AFMConfig(logging='off'))

    def test_smoke_baseline_keeps_lora_and_residual(self):
        from types import SimpleNamespace
        import comfy.sd
        from afm.config import AFMConfig
        from afm.residual import ResidualConfig, RESIDUAL_CONFIG_KEY
        from tools.smoke_contract import (ModelPipeline, ModelOnlyLoraSpec,
                                          prepare_model_pipeline, model_for_mode)
        model = self.model(True, num_layers=42)
        model.model.model_config = SimpleNamespace(unet_config={})
        key = 'diffusion_model.layers.24.proj.o'
        lora = {key + '.lora_up.weight': torch.ones(64, 2) * 0.1,
                key + '.lora_down.weight': torch.ones(2, 64) * 0.1}
        def loader(model, spec):
            return comfy.sd.load_lora_for_models(model, None, lora, spec.strength_model, 0)[0]
        config = AFMConfig(mode='edit', block_scope='24-26', logging='off')
        pipeline = ModelPipeline((ModelOnlyLoraSpec('synthetic', 0.6),),
                                 ResidualConfig(0.75, 1.15), config)
        prepared = prepare_model_pipeline(model, pipeline, loader)
        self.assertEqual(len(prepared.patches), 1)
        self.assertEqual(prepared.model_options['transformer_options'][RESIDUAL_CONFIG_KEY], pipeline.residual)
        prepared.patch_weight_to_device(key + '.weight', torch.device('cpu'))
        x, text = torch.randn(1, 8, 2, 3), torch.randn(1, 4, 16)
        try:
            baseline, _ = self.evaluate_patched(prepared, x, text)
            for mode in ('baseline', 'off', 'zero', 'observe'):
                actual, _ = self.evaluate_patched(model_for_mode(prepared, config, mode), x, text, config)
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        finally:
            prepared.unpatch_model()

    def test_smoke_rejects_partial_lora_application(self):
        from types import SimpleNamespace
        import comfy.sd
        from tools.smoke_contract import ModelOnlyLoraSpec, load_lora_checked
        model = self.model(True, num_layers=42)
        model.model.model_config = SimpleNamespace(unet_config={})
        spec = ModelOnlyLoraSpec('synthetic', 0.6)
        def lora_for(index):
            key = f'diffusion_model.layers.{index}.proj.o'
            return {key + '.lora_up.weight': torch.ones(64, 2) * 0.1,
                    key + '.lora_down.weight': torch.ones(2, 64) * 0.1}
        valid = lora_for(9)
        def loader(data):
            return lambda model, spec: comfy.sd.load_lora_for_models(
                model, None, data, spec.strength_model, 0)[0]
        loaded, evidence = load_lora_checked(model, spec, loader(valid))
        self.assertEqual(evidence['applied_keys'], ['diffusion_model.layers.9.proj.o.weight'])
        again, evidence = load_lora_checked(loaded, spec, loader(valid))
        self.assertEqual(evidence['applied_keys'], ['diffusion_model.layers.9.proj.o.weight'])
        self.assertEqual(len(again.patches['diffusion_model.layers.9.proj.o.weight']), 2)
        with self.assertLogs(level='WARNING'), self.assertRaisesRegex(ValueError, 'not.*loaded|unapplied'):
            load_lora_checked(model, spec, loader({**valid, **lora_for(999)}))
        self.assertEqual(model.patches, {})

    def test_smoke_uses_native_model_only_loader_entrypoint(self):
        import tempfile
        from types import SimpleNamespace
        from safetensors.torch import save_file
        import folder_paths
        from tools.smoke_contract import ModelOnlyLoraSpec, load_model_only_lora
        model = self.model(True, num_layers=42)
        model.model.model_config = SimpleNamespace(unet_config={})
        key = 'diffusion_model.layers.24.proj.o'
        lora = {key + '.lora_up.weight': torch.ones(64, 2) * 0.1,
                key + '.lora_down.weight': torch.ones(2, 64) * 0.1}
        with tempfile.TemporaryDirectory() as directory:
            save_file(lora, str(Path(directory) / 'native.safetensors'))
            with patch.dict(folder_paths.folder_names_and_paths,
                            {'loras': ([directory], {'.safetensors'})}):
                loaded = load_model_only_lora(model, ModelOnlyLoraSpec('native.safetensors', 0.6))
        self.assertEqual(set(loaded.patches), {key + '.weight'})
        self.assertEqual(loaded.patches[key + '.weight'][0][0], 0.6)
        self.assertEqual(model.patches, {})

    def test_old_api_prompt_validation(self):
        import asyncio
        from dataclasses import asdict
        import execution
        import nodes as comfy_nodes
        from afm.config import AFMConfig

        class Loader:
            RETURN_TYPES = ("MODEL",)

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

        class Output:
            RETURN_TYPES = ()
            OUTPUT_NODE = True

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"model": ("MODEL",)}}

        settings = asdict(AFMConfig())
        for name in ("two_interval_mode", "second_block_scope", "second_start_percent",
                     "second_end_percent"):
            settings.pop(name)
        prompt = {"28": {"class_type": "AFMTestLoader", "inputs": {}},
                  "38": {"class_type": "MossImageAFM",
                         "inputs": {"model": ["28", 0], **settings}},
                  "39": {"class_type": "AFMTestOutput", "inputs": {"model": ["38", 0]}}}
        with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS,
                        {"AFMTestLoader": Loader, "MossImageAFM": self.node_class(),
                         "AFMTestOutput": Output}):
            result = asyncio.run(execution.validate_prompt("afm-old-api", prompt, None))
            self.assertTrue(result[0], result)
            self.assertEqual(result[3], {})
            prompt["38"]["inputs"].update(two_interval_mode=True, end_percent=0.5,
                                           second_block_scope="30-32",
                                           second_start_percent=0.5, second_end_percent=1.0)
            result = asyncio.run(execution.validate_prompt("afm-two-intervals", prompt, None))
            self.assertTrue(result[0], result)
            self.assertEqual(result[3], {})

    def test_v3_node_schema_and_registration(self):
        node = self.node_class()
        schema = node.define_schema()
        self.assertEqual(schema.node_id, "MossImageAFM")
        additions = schema.inputs[-4:]
        self.assertEqual([n.id for n in additions],
                         ["two_interval_mode", "second_block_scope", "second_start_percent",
                          "second_end_percent"])
        self.assertTrue(all(n.optional for n in additions))
        model = self.model(True)
        output = node.execute(model, mode="off")
        self.assertIsNot(output.result[0], model)
        self.assertEqual(output.result[0].model_options, model.model_options)
        output = node.execute(model, end_percent=0.4, two_interval_mode=True,
                              second_block_scope="0-2", second_start_percent=0.6,
                              second_end_percent=0.9, logging="off")
        from afm.runtime import CONFIG_KEY
        config = output.result[0].model_options["transformer_options"][CONFIG_KEY]
        self.assertEqual((config.two_interval_mode, config.end_percent,
                          config.second_block_scope, config.second_start_percent,
                          config.second_end_percent), (True, 0.4, "0-2", 0.6, 0.9))
