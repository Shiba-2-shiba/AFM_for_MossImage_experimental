import os
import unittest

import torch

from tests.integration import test_native_hooks as native_tests


@unittest.skipUnless(os.environ.get('MOSS_COMFY_ROOT'), 'set MOSS_COMFY_ROOT for native contracts')
class BlockTuneNativeTests(unittest.TestCase):
    model = native_tests.NativeTests.model
    node_class = native_tests.NativeTests.node_class
    setUpClass = classmethod(native_tests.NativeTests.setUpClass.__func__)
    tearDownClass = classmethod(native_tests.NativeTests.tearDownClass.__func__)

    def test_fused_and_separate_match_scaled_checkpoint_and_restore(self):
        from afm.block_tune import install_block_tune, load_preset
        for fused in (False, True):
            with self.subTest(fused=fused), torch.inference_mode():
                base = self.model(fused, num_layers=42)
                reference = self.model(fused, num_layers=42)
                for row, block in zip(load_preset()['blocks'], reference.model.diffusion_model.layers):
                    for name in ('o', 'ffn_down'):
                        block.proj[name].weight.mul_(row['strength'])
                    if fused:
                        for weight, key in zip(block.proj['qkv'].weight.chunk(3), ('q', 'k', 'v')):
                            weight.mul_(row[key])
                    else:
                        for key in ('q', 'k', 'v'):
                            block.proj[key].weight.mul_(row[key])
                x, t, text = torch.randn(1, 8, 2, 3), torch.ones(1), torch.randn(1, 4, 16)
                native = base.model.diffusion_model
                before = native(x.clone(), t, text, {})
                tuned, report = install_block_tune(base)
                self.assertFalse(base.weight_wrapper_patches)
                self.assertFalse(base.clone_has_same_weights(tuned))
                self.assertEqual(report['unobserved_qkv_blocks'], [*range(14, 21), *range(35, 42)])
                tuned.patch_model(device_to=torch.device('cpu'), load_weights=True)
                try:
                    actual = native(x.clone(), t, text, {})
                    expected = reference.model.diffusion_model(x.clone(), t, text, {})
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                finally:
                    tuned.unpatch_model(device_to=torch.device('cpu'))
                # ComfyUI reconfigures weight functions when the original branch is loaded.
                base.patch_model(device_to=torch.device('cpu'), load_weights=True)
                try:
                    torch.testing.assert_close(native(x.clone(), t, text, {}), before, rtol=0, atol=0)
                finally:
                    base.unpatch_model(device_to=torch.device('cpu'))

    def test_afm_order_duplicate_and_bypass(self):
        from afm.block_tune import install_block_tune
        from afm.adapter import install_afm
        from afm.config import AFMConfig
        from afm.residual import install_residual, ResidualConfig
        base = self.model(True, num_layers=42)
        off, report = install_block_tune(base, 'off')
        self.assertFalse(off.weight_wrapper_patches)
        self.assertEqual(report['status'], 'bypass')
        tuned, _ = install_block_tune(base)
        combined = install_afm(tuned, AFMConfig(mode='observe', logging='off'))
        self.assertEqual(combined.weight_wrapper_patches, tuned.weight_wrapper_patches)
        with self.assertRaises(ValueError):
            install_block_tune(tuned)
        with self.assertRaises(ValueError):
            install_block_tune(install_afm(base, AFMConfig(logging='off')))
        with self.assertRaises(ValueError):
            install_block_tune(install_residual(base, ResidualConfig(.9, 1)))
        with self.assertRaises(ValueError):
            install_block_tune(base, 'invalid')

    def test_second_preset_schema_and_execution(self):
        node = self.node_class('MossImageBlockTunePreset')
        schema = node.define_schema()
        self.assertEqual(schema.node_id, 'MossImageBlockTunePreset')
        self.assertEqual(schema.inputs[1].options,
                         ['off', 'image_1_individual_qkv', 'image_2_global_qkv'])
        result = node.execute(self.model(True, num_layers=42), 'image_2_global_qkv')
        tuned = result.result[0]
        for index in range(42):
            wrapper = tuned.weight_wrapper_patches[f'diffusion_model.layers.{index}.proj.qkv.weight'][0]
            self.assertEqual(wrapper.factors, (1.7, 1., .8))

    def test_legacy_diffusion_wrapper_is_rejected(self):
        from comfy.patcher_extension import WrappersMP, add_wrapper
        from afm.block_tune import install_block_tune
        base = self.model(True, num_layers=42)
        add_wrapper(WrappersMP.DIFFUSION_MODEL, lambda executor, *a, **kw: executor(*a, **kw),
                    base.model_options['transformer_options'])
        with self.assertRaises(ValueError):
            install_block_tune(base)

    def test_k_balance_metadata_is_rejected(self):
        from afm.block_tune import install_block_tune
        for fused in (False, True):
            base = self.model(fused, num_layers=42)
            # No dependency on the separate, optional K Row Balance implementation.
            base.model_options['transformer_options']['moss_k_row_balance'] = {'blocks': [41]}
            with self.assertRaises(ValueError):
                install_block_tune(base)

    def test_int8_convrot_projection_matches_scaled_dequantized_reference(self):
        import json
        import comfy.ops
        from afm.block_tune import install_block_tune
        for full_precision in (False, True):
            with self.subTest(full_precision=full_precision), torch.inference_mode():
                base = self.model(True, num_layers=42)
                operation = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
                module = operation.Linear(64, 192, bias=False, device='cpu')
                module.load_state_dict({
                    'weight': torch.ones((192, 64), dtype=torch.int8),
                    'weight_scale': torch.linspace(.01, .04, 192).reshape(-1, 1),
                    'comfy_quant': torch.tensor(list(json.dumps({
                        'format': 'int8_tensorwise', 'convrot': True,
                        'convrot_groupsize': 4}).encode()), dtype=torch.uint8),
                })
                module._full_precision_mm = full_precision
                base.model.diffusion_model.layers[0].proj['qkv'] = module
                original = {key: value.clone() for key, value in module.state_dict().items()}
                weight = module.weight.dequantize().clone()
                for part, factor in zip(weight.chunk(3), (1.45, 1.54, 1.39)):
                    part.mul_(factor)
                x = torch.randn(2, 3, 64)
                expected = torch.nn.functional.linear(x, weight)
                tuned, _ = install_block_tune(base)
                tuned.patch_model(device_to=torch.device('cpu'), load_weights=True)
                try:
                    torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                finally:
                    tuned.unpatch_model(device_to=torch.device('cpu'))
