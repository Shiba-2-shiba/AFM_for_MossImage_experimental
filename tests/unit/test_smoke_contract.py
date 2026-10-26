from dataclasses import asdict
import importlib
import unittest

import torch

from afm.config import AFMConfig


class SmokeContractTests(unittest.TestCase):
    def helpers(self):
        try:
            return importlib.import_module("tools.smoke_contract")
        except ModuleNotFoundError:
            self.fail("Missing import-safe smoke configuration/result contract")

    def workflow(self, **settings):
        return {"31": {"inputs": {"model": ["38", 0]}},
                "38": {"class_type": "MossImageAFM", "inputs":
                       {"model": ["28", 0], "mode": "observe", **settings}}}

    def test_workflow_settings_survive_mode_variants(self):
        h = self.helpers()
        w = self.workflow(block_scope="24-26", strength=0.12, cutoff=0.4,
                          branch_mode="both", query_chunk_size=32)
        c = h.workflow_config(w)
        self.assertEqual(c.block_scope, "24-26")
        for mode in ("off", "observe", "edit", "zero"):
            selected = h.mode_config(c, mode)
            expected = asdict(c)
            expected["mode"] = "edit" if mode == "zero" else mode
            if mode == "zero":
                expected["strength"] = 0
            self.assertEqual(asdict(selected), expected)

    def test_default_and_invalid_workflow_settings(self):
        h = self.helpers()
        self.assertEqual(h.workflow_config({"31": {"inputs": {"model": ["28", 0]}}}), AFMConfig())
        for settings in ({"block_scope": "0"}, {"block_scope": "all"},
                         {"unknown": 1}, {"strength": ["99", 0]}):
            with self.assertRaises(ValueError):
                h.workflow_config(self.workflow(**settings))

    def test_mode_list_requires_a_unique_baseline_first(self):
        h = self.helpers()
        self.assertEqual(h.parse_modes("baseline,off,zero,observe,edit"),
                         ("baseline", "off", "zero", "observe", "edit"))
        for text in ("edit", "off,baseline", "baseline,edit,edit", "baseline,bad", ""):
            with self.assertRaises(ValueError):
                h.parse_modes(text)

    def test_noop_regression_is_a_failure_not_just_a_report_value(self):
        h = self.helpers()
        base = torch.zeros(1, 2, 3, 4)
        for mode in ("off", "zero", "observe"):
            result = h.compare_latents(mode, base.clone(), base)
            self.assertTrue(result["equal_to_baseline"])
            with self.assertRaisesRegex(RuntimeError, "baseline"):
                h.compare_latents(mode, base + 1, base)
        self.assertFalse(h.compare_latents("edit", base + 1, base)["equal_to_baseline"])
        for values in (torch.full_like(base, float("nan")), torch.zeros(2)):
            with self.assertRaises((ValueError, FloatingPointError)):
                h.compare_latents("edit", values, base)

    def test_edit_with_zero_workflow_strength_is_checked_as_a_noop(self):
        h = self.helpers()
        config = h.mode_config(h.workflow_config(self.workflow(strength=0)), "edit")
        self.assertTrue(config.bypass)
        base = torch.zeros(1, 2, 3, 4)
        try:
            h.compare_latents("edit", base + 1, base, require_equal=config.bypass)
        except RuntimeError as exc:
            self.assertIn("baseline", str(exc))
        except TypeError:
            self.fail("Smoke comparison cannot yet assert a configured bypass")
        else:
            self.fail("Edit-strength-zero regression was silently accepted")
