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

    def model(self, large=True, blocks=(41,)):
        dim, heads, ffn, device = (3072, 24, 8192, "meta") if large else (8, 2, 16, "cpu")
        native = self.native_class(dim=dim, ffn_dim=ffn, num_heads=heads, num_layers=42,
                                   text_dim=16, latent_channels=128, fused_qkv=True,
                                   device=device, dtype=torch.float32,
                                   operations=self.ops.disable_weight_init)
        for block in blocks:
            native.layers[block].proj["qkv"] = self.linear(dim, 256 if large else 4)
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
        from afm.k_balance import TARGET_MODULE, _attach_replacements, balance_projection
        from afm.k_balance_math import KBalanceConfig
        base = self.model(large=False)
        original = base.get_model_object(TARGET_MODULE)
        changed, report = balance_projection(original, KBalanceConfig(1), 4, torch.device("cpu"))
        edited = _attach_replacements(base, {TARGET_MODULE: changed}, report)
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
        self.assertEqual([n.id for n in schema.inputs], ["model", "strength", "max_change", "blocks"])
        self.assertEqual((schema.inputs[1].min, schema.inputs[1].max), (-10, 10))
        self.assertEqual(schema.inputs[2].max, .95)
        self.assertEqual(schema.inputs[3].default, "41")
        self.assertEqual([n.get_io_type() for n in schema.outputs], ["MODEL", "STRING"])
        base = self.model(large=False)
        result = node.execute(base).result
        self.assertEqual(len(result), 2)
        self.assertEqual(json.loads(result[1])["status"], "bypass")
        self.assertEqual(result[0].object_patches, {})

    def test_normal_patcher_subclass_is_supported(self):
        from afm.k_balance import install_k_balance
        from afm.k_balance_math import KBalanceConfig
        class Derived(self.patcher_class):
            pass
        base = self.model()
        derived = Derived(base.model, base.load_device, base.offload_device, size=1)
        edited, report = install_k_balance(derived, KBalanceConfig(2, .3))
        self.assertIsInstance(edited, Derived)
        self.assertFalse(report["dynamic_delegate_used"])
        self.assertEqual(report["block_indices"], [41])

    def test_native_dynamic_delegate_contract_in_cpu_harness(self):
        from comfy.model_patcher import ModelPatcherDynamic
        from afm.k_balance import install_k_balance, TARGET_MODULE
        from afm.k_balance_math import KBalanceConfig
        # Actual native delegate/clone methods; bypass only GPU-specific construction.
        class Harness(self.patcher_class):
            def is_dynamic(self):
                return True
            get_non_dynamic_delegate = ModelPatcherDynamic.get_non_dynamic_delegate
        base = self.model()
        dynamic = Harness(base.model, base.load_device, base.offload_device, size=1)
        dynamic.non_dynamic_delegate_model = None
        reloads = []
        def reload_model(disable_dynamic=False):
            self.assertTrue(disable_dynamic)
            reloads.append(disable_dynamic)
            return self.model()
        dynamic.cached_patcher_init = (reload_model, ())
        dynamic.model_options["test_option"] = "preserve"
        zero, zero_report = install_k_balance(dynamic, KBalanceConfig())
        self.assertEqual(reloads, [])
        first, report = install_k_balance(dynamic, KBalanceConfig(2, .3))
        self.assertFalse(first.is_dynamic())
        self.assertTrue(report["dynamic_delegate_used"])
        self.assertEqual(first.model_options["test_option"], "preserve")
        original = first.get_model_object(TARGET_MODULE)
        first.patch_model(load_weights=False)
        second, second_report = install_k_balance(dynamic, KBalanceConfig(-2, .3))
        self.assertEqual(len(reloads), 1)
        self.assertIs(first.model, second.model)
        self.assertEqual(second_report["block_indices"], [41])
        first.unpatch_model(unpatch_weights=False)
        self.assertEqual(dynamic.object_patches, {})
        self.assertIsNot(second.get_model_object(TARGET_MODULE), original)

    def test_dynamic_without_loader_keeps_clear_failure(self):
        from comfy.model_patcher import ModelPatcherDynamic
        from afm.k_balance import install_k_balance
        from afm.k_balance_math import KBalanceConfig
        class Harness(self.patcher_class):
            def is_dynamic(self):
                return True
            get_non_dynamic_delegate = ModelPatcherDynamic.get_non_dynamic_delegate
        base = self.model()
        dynamic = Harness(base.model, base.load_device, base.offload_device, size=1)
        dynamic.non_dynamic_delegate_model = None
        with self.assertRaisesRegex(RuntimeError, "cached_patcher_init"):
            install_k_balance(dynamic, KBalanceConfig(1))
        self.assertEqual(dynamic.object_patches, {})

    def test_selected_blocks_only_and_atomic_failure(self):
        from afm.k_balance import install_k_balance, target_module_path
        from afm.k_balance_math import KBalanceConfig
        base = self.model(blocks=(0, 22, 41))
        originals = {i: base.get_model_object(target_module_path(i)) for i in (0, 22, 41)}
        edited, report = install_k_balance(base, KBalanceConfig(2, .3), blocks="0,22")
        self.assertEqual(set(edited.object_patches), {target_module_path(0), target_module_path(22)})
        self.assertEqual(report["block_indices"], [0, 22])
        self.assertEqual([r["block_index"] for r in report["block_reports"]], [0, 22])
        edited.patch_model(load_weights=False)
        self.assertIs(base.model.diffusion_model.layers[41].proj["qkv"], originals[41])
        edited.unpatch_model(unpatch_weights=False)
        for i, module in originals.items():
            self.assertIs(base.get_model_object(target_module_path(i)), module)
        handle = originals[22].register_forward_hook(lambda m, a, out: out)
        try:
            with self.assertRaisesRegex(ValueError, "[Bb]lock 22.*hook"):
                install_k_balance(base, KBalanceConfig(2, .3), blocks="0,22")
            self.assertEqual(base.object_patches, {})
            self.assertIs(base.model.diffusion_model.layers[0].proj["qkv"], originals[0])
        finally:
            handle.remove()

    def test_conflict_checks_cover_selected_non41_blocks(self):
        from afm.k_balance import install_k_balance, target_module_path
        from afm.k_balance_math import KBalanceConfig
        base = self.model(blocks=(0, 41))
        base.add_weight_wrapper(target_module_path(0) + ".weight", lambda x: x)
        with self.assertRaisesRegex(ValueError, "conflict"):
            install_k_balance(base, KBalanceConfig(1, .3), blocks="0")

    def test_all_ten_requested_blocks_apply_and_restore_together(self):
        from afm.k_balance import install_k_balance, target_module_path
        from afm.k_balance_math import KBalanceConfig
        wanted = (0, 1, 2, 22, 24, 26, 30, 31, 40, 41)
        base = self.model(blocks=wanted)
        originals = {i: base.get_model_object(target_module_path(i)) for i in wanted}
        unselected = base.model.diffusion_model.layers[3].proj["qkv"]
        edited, report = install_k_balance(base, KBalanceConfig(2, .3),
                                           blocks=",".join(map(str, wanted)))
        self.assertEqual(report["block_indices"], list(wanted))
        self.assertEqual(len(edited.object_patches), 10)
        self.assertTrue(report["quantized_codes_unchanged"])
        self.assertTrue(report["qv_scales_unchanged"])
        edited.patch_model(load_weights=False)
        for i, original in originals.items():
            self.assertIsNot(base.model.diffusion_model.layers[i].proj["qkv"], original)
        self.assertIs(base.model.diffusion_model.layers[3].proj["qkv"], unselected)
        edited.unpatch_model(unpatch_weights=False)
        for i, original in originals.items():
            self.assertIs(base.model.diffusion_model.layers[i].proj["qkv"], original)

    def test_selection_parser_requested_list_and_legacy_zero(self):
        from afm.k_balance import parse_k_blocks, install_k_balance
        from afm.k_balance_math import KBalanceConfig
        wanted = (0, 1, 2, 22, 24, 26, 30, 31, 40, 41)
        self.assertEqual(parse_k_blocks("0-2,22,24,26,30,31,40,41"), wanted)
        self.assertEqual(parse_k_blocks("41,0,41"), (0, 41))
        self.assertEqual(parse_k_blocks("all"), tuple(range(42)))
        for value in ("", "42", "-1", "2-0", "30.31", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_k_blocks(value)
        base = self.model(large=False)
        zero, report = install_k_balance(base, KBalanceConfig(), blocks="0,41")
        self.assertEqual(zero.object_patches, {})
        self.assertEqual(report["block_indices"], [0, 41])
