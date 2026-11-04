import json
import math
import unittest

import torch

from afm.k_balance_math import KBalanceConfig, compute_k_gains


def _codes_from_norms(norms: list[int]) -> torch.Tensor:
    return torch.tensor([[value, 0] for value in norms], dtype=torch.int8)


class KBalanceMathTests(unittest.TestCase):
    def test_config_validates_controls_and_bypass(self):
        self.assertEqual(KBalanceConfig(), KBalanceConfig(strength=0.0, max_change=0.03))
        self.assertTrue(KBalanceConfig().bypass)
        self.assertTrue(KBalanceConfig(strength=0.5, max_change=0.0).bypass)
        self.assertFalse(KBalanceConfig(strength=0.5, max_change=0.03).bypass)

        for kwargs in (
            {"strength": True},
            {"strength": float("nan")},
            {"strength": -1.01},
            {"strength": 1.01},
            {"max_change": False},
            {"max_change": float("inf")},
            {"max_change": -0.001},
            {"max_change": 0.101},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KBalanceConfig(**kwargs)

    def test_positive_compresses_and_negative_expands_within_head_dispersion(self):
        codes = _codes_from_norms([1, 2, 4, 8])
        scales = torch.ones(4, 1, dtype=torch.float32)

        compressed, compressed_report = compute_k_gains(
            codes, scales, 4, KBalanceConfig(strength=1.0, max_change=0.10)
        )
        expanded, expanded_report = compute_k_gains(
            codes, scales, 4, KBalanceConfig(strength=-1.0, max_change=0.10)
        )

        self.assertGreater(float(compressed[0]), 1.0)
        self.assertLess(float(compressed[-1]), 1.0)
        self.assertLess(float(expanded[0]), 1.0)
        self.assertGreater(float(expanded[-1]), 1.0)
        self.assertLess(
            compressed_report["per_head_p99_over_p50"]["median_after"],
            compressed_report["per_head_p99_over_p50"]["median_before"],
        )
        self.assertGreater(
            expanded_report["per_head_p99_over_p50"]["median_after"],
            expanded_report["per_head_p99_over_p50"]["median_before"],
        )

    def test_gains_are_bounded_by_three_percent(self):
        gains, report = compute_k_gains(
            _codes_from_norms([1, 2, 4, 8]),
            torch.ones(4, 1, dtype=torch.float32),
            4,
            KBalanceConfig(strength=1.0),
        )

        self.assertGreaterEqual(float(gains.min()), 0.97)
        self.assertLessEqual(float(gains.max()), 1.03)
        self.assertAlmostEqual(report["gain_min"], 0.97, places=6)
        self.assertAlmostEqual(report["gain_max"], 1.03, places=6)

    def test_zero_strength_and_zero_cap_return_exact_identity_without_mutation(self):
        codes = _codes_from_norms([1, 2, 3, 4])
        scales = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
        original_codes = codes.clone()
        original_scales = scales.clone()

        for config in (KBalanceConfig(), KBalanceConfig(strength=0.8, max_change=0.0)):
            with self.subTest(config=config):
                gains, report = compute_k_gains(codes, scales, 2, config)
                torch.testing.assert_close(gains, torch.ones_like(gains), rtol=0, atol=0)
                self.assertEqual(report["changed_rows"], 0)
                self.assertEqual(report["gain_min"], 1.0)
                self.assertEqual(report["gain_max"], 1.0)

        torch.testing.assert_close(codes, original_codes, rtol=0, atol=0)
        torch.testing.assert_close(scales, original_scales, rtol=0, atol=0)

    def test_each_heads_uniform_scale_factor_does_not_change_gains(self):
        codes = _codes_from_norms([1, 2, 3, 4, 1, 2, 3, 4])
        base_scales = torch.ones(8, 1, dtype=torch.float32)
        shifted_scales = base_scales.clone()
        shifted_scales[:4] *= 0.125
        shifted_scales[4:] *= 32.0
        config = KBalanceConfig(strength=0.75, max_change=0.10)

        base, base_report = compute_k_gains(codes, base_scales, 4, config)
        shifted, shifted_report = compute_k_gains(codes, shifted_scales, 4, config)

        torch.testing.assert_close(base, shifted, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(
            base_report["median_per_head_log_norm_std"]["before"],
            shifted_report["median_per_head_log_norm_std"]["before"],
            places=12,
        )

    def test_even_head_uses_exact_linear_median(self):
        codes = _codes_from_norms([100, 101, 102, 103])
        gains, _ = compute_k_gains(
            codes,
            torch.ones(4, 1, dtype=torch.float32),
            4,
            KBalanceConfig(strength=1.0, max_change=0.10),
        )

        expected = torch.tensor(
            [[1.015], [1.015 / 1.01], [1.015 / 1.02], [1.015 / 1.03]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(gains, expected, rtol=1e-6, atol=1e-7)

    def test_report_is_compact_json_safe_and_tracks_actual_float32_after_state(self):
        gains, report = compute_k_gains(
            _codes_from_norms([1, 2, 3, 4, 2, 3, 4, 5]),
            torch.full((8, 1), 0.25, dtype=torch.float32),
            4,
            KBalanceConfig(strength=0.5, max_change=0.03),
            chunk_rows=3,
        )

        json.dumps(report, allow_nan=False)
        self.assertEqual(gains.shape, (8, 1))
        self.assertEqual(gains.dtype, torch.float32)
        self.assertEqual(gains.device.type, "cpu")
        self.assertEqual(report["rows"], 8)
        self.assertEqual(report["heads"], 2)
        self.assertEqual(report["head_dim"], 4)
        self.assertEqual(report["strength"], 0.5)
        self.assertEqual(report["max_change"], 0.03)
        self.assertGreater(report["workspace_estimate_bytes"], 0)
        self.assertEqual(
            set(report),
            {
                "rows",
                "heads",
                "head_dim",
                "strength",
                "max_change",
                "changed_rows",
                "gain_min",
                "gain_max",
                "row_p99_over_p50",
                "median_per_head_log_norm_std",
                "per_head_p99_over_p50",
                "workspace_estimate_bytes",
            },
        )

    def test_chunk_size_does_not_change_results(self):
        codes = _codes_from_norms(list(range(1, 17)))
        scales = torch.linspace(0.5, 1.5, 16, dtype=torch.float32).reshape(-1, 1)
        config = KBalanceConfig(strength=0.4, max_change=0.08)

        gains_one, report_one = compute_k_gains(codes, scales, 4, config, chunk_rows=1)
        gains_seven, report_seven = compute_k_gains(codes, scales, 4, config, chunk_rows=7)

        torch.testing.assert_close(gains_one, gains_seven, rtol=0, atol=0)
        for key in (
            "row_p99_over_p50",
            "median_per_head_log_norm_std",
            "per_head_p99_over_p50",
        ):
            self.assertEqual(report_one[key], report_seven[key])

    def test_rejects_invalid_tensor_contracts_and_chunk_size(self):
        valid_codes = _codes_from_norms([1, 2, 3, 4])
        valid_scales = torch.ones(4, 1, dtype=torch.float32)
        config = KBalanceConfig(strength=0.5)
        invalid_calls = (
            (valid_codes.float(), valid_scales, 2, 128),
            (valid_codes.reshape(2, 2, 2), valid_scales, 2, 128),
            (valid_codes, valid_scales.double(), 2, 128),
            (valid_codes, valid_scales.flatten(), 2, 128),
            (valid_codes, torch.ones(3, 1, dtype=torch.float32), 2, 128),
            (valid_codes, valid_scales, 3, 128),
            (valid_codes, valid_scales, True, 128),
            (valid_codes, valid_scales, 0, 128),
            (valid_codes, valid_scales, 2, True),
            (valid_codes, valid_scales, 2, 0),
        )
        for codes, scales, head_dim, chunk_rows in invalid_calls:
            with self.subTest(
                codes_shape=codes.shape,
                scales_shape=scales.shape,
                head_dim=head_dim,
                chunk_rows=chunk_rows,
            ), self.assertRaises(ValueError):
                compute_k_gains(codes, scales, head_dim, config, chunk_rows=chunk_rows)

        with self.assertRaises(ValueError):
            compute_k_gains("codes", valid_scales, 2, config)

    def test_rejects_zero_rows_nonpositive_scales_and_nonfinite_scales(self):
        config = KBalanceConfig(strength=0.5)
        with self.assertRaises(FloatingPointError):
            compute_k_gains(
                torch.tensor([[1, 0], [0, 0]], dtype=torch.int8),
                torch.ones(2, 1, dtype=torch.float32),
                2,
                config,
            )
        for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            scales = torch.ones(2, 1, dtype=torch.float32)
            scales[0, 0] = value
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                compute_k_gains(_codes_from_norms([1, 2]), scales, 2, config)

    def test_scale_inspection_detaches_autograd_without_mutating_input(self):
        scales = torch.tensor(
            [[0.5], [1.0], [1.5], [2.0]], dtype=torch.float32, requires_grad=True
        )
        original = scales.detach().clone()

        gains, _ = compute_k_gains(
            _codes_from_norms([1, 2, 3, 4]),
            scales,
            4,
            KBalanceConfig(strength=0.5, max_change=0.03),
        )

        self.assertFalse(gains.requires_grad)
        self.assertTrue(scales.requires_grad)
        torch.testing.assert_close(scales.detach(), original, rtol=0, atol=0)

    def test_rejects_float32_scale_overflow_after_gain(self):
        largest = torch.finfo(torch.float32).max
        with self.assertRaises(FloatingPointError):
            compute_k_gains(
                _codes_from_norms([1, 2]),
                torch.full((2, 1), largest, dtype=torch.float32),
                2,
                KBalanceConfig(strength=1.0, max_change=0.10),
            )

    def test_changed_rows_counts_only_represented_float32_scale_changes(self):
        smallest = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
        scales = smallest.expand(2, 1).clone()

        gains, report = compute_k_gains(
            _codes_from_norms([1, 2]),
            scales,
            2,
            KBalanceConfig(strength=1.0, max_change=0.03),
        )

        self.assertTrue(torch.all(gains != 1))
        torch.testing.assert_close(scales * gains, scales, rtol=0, atol=0)
        self.assertEqual(report["changed_rows"], 0)


if __name__ == "__main__":
    unittest.main()
