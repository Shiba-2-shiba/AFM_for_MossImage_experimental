import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from afm.config import AFMConfig
from afm.runtime import RUN_KEY, make_outer_wrapper
from tests.unit.test_runtime import FakeExecutor


@unittest.skipUnless(os.environ.get("MOSS_COMFY_ROOT"), "requires actual ComfyUI")
class SamplerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.environ["MOSS_COMFY_ROOT"])
        from comfy.cli_args import args
        args.cpu = True

    def test_real_euler_and_er_sde_callbacks_preserve_step_identity(self):
        from afm.adapter import validate_sampler
        from comfy.k_diffusion import sampling
        import comfy.model_sampling
        class FlowSampling(comfy.model_sampling.CONST):
            def percent_to_sigma(self, p):
                return 1 - p
        for function in [sampling.sample_euler, sampling.sample_er_sde]:
            original = {"transformer_options": {}}
            guider = SimpleNamespace(model_options=original)
            observed = []
            config = AFMConfig(logging="off")
            sampler = SimpleNamespace(sampler_function=function)

            class Model:
                def __init__(self):
                    self.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
                        get_model_object=lambda key: FlowSampling()))

                def __call__(self, x, sigma, **kwargs):
                    state = guider.model_options["transformer_options"][RUN_KEY]
                    observed.append((state.step_index, state.progress, float(sigma[0])))
                    return torch.zeros_like(x)

            def execute(noise, latent, sampler, sigmas, mask, callback, disabled, seed, shapes):
                return function(Model(), torch.ones(1, 2, 2, 2), sigmas,
                                callback=lambda d: callback(d["i"], d["denoised"], d["x"], 3),
                                disable=True, s_noise=0)
            wrapper = make_outer_wrapper(config, validate_sampler, lambda: None)
            wrapper(FakeExecutor(guider, execute), None, None, sampler,
                    torch.tensor([1.0, 0.7, 0.3, 0.0]))
            self.assertEqual([v[0] for v in observed], [0, 1, 2])
            self.assertEqual([v[1] for v in observed], [0.0, 0.5, 1.0])
            if function is sampling.sample_er_sde:
                self.assertLess(observed[0][2], 1.0)
            self.assertIs(guider.model_options, original)

    def test_unqualified_sampler_rejected(self):
        from afm.adapter import validate_sampler
        from comfy.k_diffusion import sampling
        with self.assertRaises(ValueError):
            validate_sampler(SimpleNamespace(sampler_function=sampling.sample_heun))
