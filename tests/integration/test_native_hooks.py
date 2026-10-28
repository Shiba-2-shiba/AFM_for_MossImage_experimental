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

    def node_class(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("moss_afm_test_pack", root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        pack = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pack
        spec.loader.exec_module(pack)
        import asyncio
        extension = asyncio.run(pack.comfy_entrypoint())
        nodes = asyncio.run(extension.get_node_list())
        return nodes[0]

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
