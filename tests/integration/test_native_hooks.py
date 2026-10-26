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

    def model(self, fused):
        import comfy.ops
        from comfy.ldm.moss_image.model import MOSSImage
        from comfy.model_patcher import ModelPatcher
        torch.manual_seed(33)
        native = MOSSImage(dim=64, ffn_dim=96, num_heads=2, num_layers=2,
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
            self.assertEqual(state.observed_calls, 2)
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

    def test_v3_node_schema_and_registration(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("moss_afm_test_pack", root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        pack = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pack
        spec.loader.exec_module(pack)
        import asyncio
        extension = asyncio.run(pack.comfy_entrypoint())
        nodes = asyncio.run(extension.get_node_list())
        schema = nodes[0].define_schema()
        self.assertEqual(schema.node_id, "MossImageAFM")
        model = self.model(True)
        output = nodes[0].execute(model, mode="off")
        self.assertIsNot(output.result[0], model)
        self.assertEqual(output.result[0].model_options, model.model_options)
