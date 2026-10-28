from dataclasses import asdict
import importlib
import unittest

import torch

from afm.config import AFMConfig


class SmokeContractTests(unittest.TestCase):
    def test_old_api_omits_second_interval(self):
        config = self.helpers().workflow_config(self.workflow())
        self.assertFalse(config.two_interval_mode)
        self.assertEqual([config.active_interval(p) for p in (0, 0.5, 1)], [1, 1, 1])

    def test_two_interval_settings_survive_mode_variants(self):
        helpers = self.helpers()
        config = helpers.workflow_config(self.workflow(
            block_scope="30-32", end_percent=0.5, two_interval_mode=True,
            second_block_scope="24-26", second_start_percent=0.4, second_end_percent=0.9))
        for mode in ("baseline", "off", "zero", "observe", "edit"):
            selected = helpers.mode_config(config, mode)
            self.assertTrue(selected.two_interval_mode)
            self.assertEqual((selected.block_scope, selected.end_percent,
                              selected.second_block_scope, selected.second_start_percent,
                              selected.second_end_percent), ("30-32", 0.5, "24-26", 0.4, 0.9))

    def test_secondary_preset_validation_respects_enabled_mode(self):
        helpers = self.helpers()
        for scope in ("0-2", "24-26", "30-32"):
            config = helpers.workflow_config(self.workflow(two_interval_mode=True,
                                                           second_block_scope=scope))
            self.assertEqual(config.second_block_scope, scope)
        for scope in ("all", "0", "invalid"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                helpers.workflow_config(self.workflow(two_interval_mode=True,
                                                       second_block_scope=scope))
            helpers.workflow_config(self.workflow(two_interval_mode=False,
                                                   second_block_scope=scope))

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
