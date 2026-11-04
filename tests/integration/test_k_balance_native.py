import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


COMFY_ROOT = os.environ.get("MOSS_COMFY_ROOT")


@unittest.skipUnless(COMFY_ROOT, "set MOSS_COMFY_ROOT for CPU ComfyUI contracts")
class KBalanceNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, COMFY_ROOT)
        from comfy.cli_args import args
        args.cpu = True
        import comfy.ops
        from comfy.model_patcher import ModelPatcher
        from comfy.ldm.moss_image.model import MOSSImage
        cls.ops, cls.patcher_class, cls.native_class = comfy.ops, ModelPatcher, MOSSImage
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def linear(self, dim=4, group=4, full_precision=False, dtype=torch.float32):
        operation = self.ops.mixed_precision_ops(compute_dtype=dtype)
        module = operation.Linear(dim, 3 * dim, bias=False, device="cpu")
        state = {
            "weight": torch.ones((3 * dim, dim), dtype=torch.int8),
            "weight_scale": torch.linspace(.01, .04, 3 * dim).reshape(-1, 1),
            "comfy_quant": torch.tensor(list(json.dumps({
                "format": "int8_tensorwise", "convrot": True,
                "convrot_groupsize": group}).encode()), dtype=torch.uint8),
        }
        module.load_state_dict(state)
        module._full_precision_mm = full_precision
        return module

    def model(self, large=True):
        dim, heads, ffn, device = (3072, 24, 8192, "meta") if large else (8, 2, 16, "cpu")
        native = self.native_class(dim=dim, ffn_dim=ffn, num_heads=heads, num_layers=42,
                                   text_dim=16, latent_channels=128, fused_qkv=True,
                                   device=device, dtype=torch.float32,
                                   operations=self.ops.disable_weight_init)
        native.layers[41].proj["qkv"] = self.linear(dim, 256 if large else 4)
        base = torch.nn.Module()
        base.diffusion_model = native
        return self.patcher_class(base, torch.device("cpu"), torch.device("cpu"), size=1)

    def test_zero_is_clone_only_even_without_target_weight_inspection(self):
        from afm.k_balance import install_k_balance
        from afm.k_balance_math import KBalanceConfig
        base = self.model(large=False)
        result, report = install_k_balance(base, KBalanceConfig())
        self.assertIsNot(result, base)
        self.assertEqual(result.object_patches, base.object_patches)
        self.assertEqual(result.patches_uuid, base.patches_uuid)
        self.assertEqual(report["status"], "bypass")

    def test_scale_only_projection_preserves_codes_qv_and_native_compute_mode(self):
        from afm.k_balance import balance_projection
        from afm.k_balance_math import KBalanceConfig
        from comfy.quant_ops import QuantizedTensor
        for full_precision in (False, True):
            with self.subTest(full_precision=full_precision):
                original = self.linear(full_precision=full_precision)
                before = {k: v.clone() for k, v in original.state_dict().items()}
                edited, report = balance_projection(original, KBalanceConfig(1, .03), 4,
                                                     torch.device("cpu"))
                after = edited.state_dict()
                self.assertIsInstance(edited.weight, QuantizedTensor)
                self.assertTrue(torch.equal(before["weight"], after["weight"]))
                self.assertNotEqual(before["weight"].data_ptr(), after["weight"].data_ptr())
                self.assertTrue(torch.equal(before["weight_scale"][:4], after["weight_scale"][:4]))
                self.assertTrue(torch.equal(before["weight_scale"][8:], after["weight_scale"][8:]))
                self.assertFalse(torch.equal(before["weight_scale"][4:8], after["weight_scale"][4:8]))
                self.assertTrue(torch.equal(before["comfy_quant"], after["comfy_quant"]))
                self.assertEqual(edited._full_precision_mm, original._full_precision_mm)
                self.assertEqual(edited.weight_function, [])
                x = torch.tensor([[1., 2., 3., 4.], [-1., 2., 0., 1.]])
                with torch.inference_mode():
                    a, b = original(x), edited(x)
                self.assertTrue(torch.equal(a[:, :4], b[:, :4]))
                self.assertTrue(torch.equal(a[:, 8:], b[:, 8:]))
                gains = after["weight_scale"][4:8] / before["weight_scale"][4:8]
                torch.testing.assert_close(b[:, 4:8], a[:, 4:8] * gains.T,
                                           rtol=2e-6, atol=1e-7)
                for key, value in before.items():
                    self.assertTrue(torch.equal(original.state_dict()[key], value))
                self.assertTrue(report["qv_scales_unchanged"])
                self.assertTrue(report["full_precision_mm_preserved"])
                self.assertEqual(report["full_precision_mm"], full_precision)

    def test_strict_adapter_rejects_noncanonical_native_geometry(self):
        from afm.k_balance import install_k_balance
        from afm.k_balance_math import KBalanceConfig
        with self.assertRaisesRegex(ValueError, "3072|geometry|geometry|24"):
            install_k_balance(self.model(large=False), KBalanceConfig(1))

    def test_native_42_block_adapter_clone_restore_and_noncompounding(self):
        from afm.k_balance import TARGET_MODULE, install_k_balance
        from afm.k_balance_math import KBalanceConfig
        base = self.model()
        original = base.get_model_object(TARGET_MODULE)
        original_scale = original.state_dict()["weight_scale"].clone()
        rng_state = torch.get_rng_state().clone()
        plus, report = install_k_balance(base, KBalanceConfig(1))
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        self.assertEqual(set(plus.object_patches), {TARGET_MODULE})
        self.assertEqual(base.object_patches, {})
        self.assertNotEqual(plus.patches_uuid, base.patches_uuid)
        self.assertFalse(base.clone_has_same_weights(plus))
        self.assertTrue(plus.clone_has_same_weights(plus.clone()))
        plus.patch_model(load_weights=False)
        self.assertIsNot(base.model.diffusion_model.layers[41].proj["qkv"], original)
        self.assertIs(base.get_model_object(TARGET_MODULE), original)
        minus, _ = install_k_balance(base, KBalanceConfig(-1))
        self.assertFalse(plus.clone_has_same_weights(minus))
        plus.unpatch_model(unpatch_weights=False)
        self.assertIs(base.model.diffusion_model.layers[41].proj["qkv"], original)
        self.assertTrue(torch.equal(original.state_dict()["weight_scale"], original_scale))
        minus.patch_model(load_weights=False)
        minus.unpatch_model(unpatch_weights=False)
        self.assertIs(base.model.diffusion_model.layers[41].proj["qkv"], original)
        self.assertEqual(base.object_patches_backup, {})
        with self.assertRaisesRegex(ValueError, "already|conflict|patch"):
            install_k_balance(plus, KBalanceConfig(1))
        self.assertEqual(report["block_index"], 41)

    def test_nonzero_strength_zero_cap_is_same_path_identity_control(self):
        from afm.k_balance import TARGET_MODULE, install_k_balance
        from afm.k_balance_math import KBalanceConfig
        base = self.model()
        original = base.get_model_object(TARGET_MODULE)
        controlled, report = install_k_balance(base, KBalanceConfig(1, 0))
        self.assertEqual(set(controlled.object_patches), {TARGET_MODULE})
        self.assertEqual(report["status"], "rebuild_control")
        self.assertEqual(report["changed_rows"], 0)
        replacement = controlled.get_model_object(TARGET_MODULE)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, replacement.state_dict()[key]))
        x = torch.linspace(-.2, .3, 3072).reshape(1, -1)
        with torch.inference_mode():
            self.assertTrue(torch.equal(original(x), replacement(x)))

    def test_bf16_compute_metadata_and_qv_outputs_are_preserved_without_requant(self):
        from afm.k_balance import balance_projection
        from afm.k_balance_math import KBalanceConfig
        from comfy.quant_ops import QuantizedTensor
        original = self.linear(dtype=torch.bfloat16)
        with patch.object(QuantizedTensor, "requantize_from_float",
                          side_effect=AssertionError("requantization is forbidden")):
            edited, report = balance_projection(original, KBalanceConfig(1), 4,
                                                 torch.device("cpu"))
            x = torch.tensor([[.17, -.31, .74, 1.13]], dtype=torch.bfloat16)
            with torch.inference_mode():
                before, after = original(x), edited(x)
        self.assertEqual(edited.weight.dtype, torch.bfloat16)
        self.assertEqual(edited.factory_kwargs["dtype"], torch.bfloat16)
        self.assertTrue(torch.equal(before[:, :4], after[:, :4]))
        self.assertTrue(torch.equal(before[:, 8:], after[:, 8:]))
        self.assertFalse(torch.equal(before[:, 4:8], after[:, 4:8]))
        self.assertEqual(report["compute_dtype"], "torch.bfloat16")

    def test_projection_preserves_eval_and_explicit_cast_flags(self):
        from afm.k_balance import balance_projection
        from afm.k_balance_math import KBalanceConfig
        original = self.linear().eval()
        original.comfy_force_cast_weights = True
        edited, _ = balance_projection(original, KBalanceConfig(1), 4, torch.device("cpu"))
        self.assertFalse(edited.training)
        self.assertTrue(edited.comfy_force_cast_weights)

    def test_registered_target_hooks_are_rejected_without_being_dropped(self):
        from afm.k_balance import balance_projection
        from afm.k_balance_math import KBalanceConfig
        for hook_kind in ("forward", "pre", "backward"):
            with self.subTest(hook_kind=hook_kind):
                original = self.linear()
                if hook_kind == "forward":
                    handle = original.register_forward_hook(lambda m, a, out: out * 2)
                elif hook_kind == "pre":
                    handle = original.register_forward_pre_hook(lambda m, a: None)
                else:
                    handle = original.register_full_backward_hook(lambda m, a, b: None)
                before = original.state_dict()["weight_scale"].clone()
                try:
                    with self.assertRaisesRegex(ValueError, "hook"):
                        balance_projection(original, KBalanceConfig(1), 4, torch.device("cpu"))
                    self.assertTrue(torch.equal(before, original.state_dict()["weight_scale"]))
                finally:
                    handle.remove()

    def test_instance_runtime_override_is_rejected(self):
        from afm.k_balance import balance_projection
        from afm.k_balance_math import KBalanceConfig
        original = self.linear()
        original._forward = lambda x, weight, bias: x
        with self.assertRaisesRegex(ValueError, "override|patch"):
            balance_projection(original, KBalanceConfig(1), 4, torch.device("cpu"))

    def test_unqualified_projection_contracts_fail_before_modification(self):
        from afm.k_balance import balance_projection
        from afm.k_balance_math import KBalanceConfig
        for invalid in ("scale_axis", "nonpositive_scale", "no_rotation", "extra_scale"):
            with self.subTest(invalid=invalid):
                module = self.linear()
                state = dict(module.state_dict())
                if invalid == "scale_axis":
                    state["weight_scale"] = state["weight_scale"].flatten()
                    module.load_state_dict(state)
                elif invalid == "nonpositive_scale":
                    state["weight_scale"] = state["weight_scale"].clone()
                    state["weight_scale"][4] = 0
                    module.load_state_dict(state)
                elif invalid == "no_rotation":
                    conf = {"format": "int8_tensorwise", "convrot": False}
                    state["comfy_quant"] = torch.tensor(list(json.dumps(conf).encode()),
                                                        dtype=torch.uint8)
                    module.load_state_dict(state)
                else:
                    module.input_scale = torch.tensor(1.)
                before = {k: v.clone() for k, v in module.state_dict().items()}
                with self.assertRaises((ValueError, FloatingPointError)):
                    balance_projection(module, KBalanceConfig(1), 4, torch.device("cpu"))
                for key, value in before.items():
                    self.assertTrue(torch.equal(module.state_dict()[key], value))

    def test_target_lora_and_ancestor_object_patch_are_rejected(self):
        from afm.k_balance import TARGET_MODULE, TARGET_WEIGHT, install_k_balance
        from afm.k_balance_math import KBalanceConfig
        base = self.model()
        for conflict in ("weight_patch", "weight_wrapper", "object", "hook"):
            with self.subTest(conflict=conflict):
                other = base.clone()
                if conflict == "weight_patch":
                    other.add_patches({TARGET_WEIGHT: ("diff", (torch.zeros(1),))})
                elif conflict == "weight_wrapper":
                    other.add_weight_wrapper(TARGET_WEIGHT, lambda x: x)
                elif conflict == "object":
                    other.add_object_patch("diffusion_model.layers.41", base.model.diffusion_model.layers[41])
                else:
                    other.hook_patches["unknown"] = {}
                with self.assertRaisesRegex(ValueError, "patch|hook|conflict"):
                    install_k_balance(other, KBalanceConfig(1))

    def test_small_cpu_patcher_switch_reloads_correct_variant(self):
        from afm.k_balance import TARGET_MODULE, _attach_replacement, balance_projection
        from afm.k_balance_math import KBalanceConfig
        base = self.model(large=False)
        original = base.get_model_object(TARGET_MODULE)
        changed, report = balance_projection(original, KBalanceConfig(1), 4, torch.device("cpu"))
        edited = _attach_replacement(base, changed, report)
        base.partially_load(torch.device("cpu"), extra_memory=10**9)
        self.assertEqual(base.model.current_weight_patches_uuid, base.patches_uuid)
        edited.partially_load(torch.device("cpu"), extra_memory=10**9)
        self.assertIs(base.model.diffusion_model.layers[41].proj["qkv"], changed)
        self.assertEqual(base.model.current_weight_patches_uuid, edited.patches_uuid)
        base.partially_load(torch.device("cpu"), extra_memory=10**9)
        self.assertIs(base.model.diffusion_model.layers[41].proj["qkv"], original)
        self.assertEqual(base.model.current_weight_patches_uuid, base.patches_uuid)
        base.unpatch_model(unpatch_weights=True)

    def test_v3_registration_and_zero_execution(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("moss_k_test_pack", root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        pack = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pack
        spec.loader.exec_module(pack)
        extension = asyncio.run(pack.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        found = [n for n in node_classes if n.define_schema().node_id == "MossImageKRowBalance"]
        self.assertEqual(len(found), 1)
        node = found[0]
        schema = node.define_schema()
        self.assertEqual([n.id for n in schema.inputs], ["model", "strength", "max_change"])
        self.assertEqual([n.get_io_type() for n in schema.outputs], ["MODEL", "STRING"])
        base = self.model(large=False)
        result = node.execute(base).result
        self.assertEqual(len(result), 2)
        self.assertEqual(json.loads(result[1])["status"], "bypass")
        self.assertEqual(result[0].object_patches, {})
