import math
import unittest

import torch

from afm.config import AFMConfig, TokenLayout, parse_block_scope
from afm.spectral import filter_logits, schedule_alphas


class SpectralTests(unittest.TestCase):
    def test_curve_uses_sampler_progress(self):
        self.assertEqual(schedule_alphas(0.2, 0), (1.2, 1.0))
        self.assertEqual(schedule_alphas(0.2, 1), (1.0, 1.2))
        self.assertEqual(schedule_alphas(-0.2, 0.5), (0.9, 0.9))
        for progress in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                schedule_alphas(0.2, progress)

    def test_spatial_filter_scales_known_bands_and_preserves_dc(self):
        row = torch.arange(16, dtype=torch.float32)
        low = torch.cos(2 * math.pi * row / 16)[None, :].expand(16, -1)
        high = torch.cos(2 * math.pi * row * 6 / 16)[None, :].expand(16, -1)
        x = (3 + low + high).reshape(1, 1, 256, 1)
        got = filter_logits(x, (16, 16), 2, 4, 0.3, False, 0.05, True)
        expected = (3 + 2 * low + 4 * high).reshape_as(x)
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)

    def test_rectangular_and_same_token_count_grids_remain_distinct(self):
        torch.manual_seed(14)
        for grid in [(3, 5), (5, 3), (36, 64), (48, 48)]:
            x = torch.randn(1, 2, math.prod(grid), 3)
            got = filter_logits(x, grid, 1, 1, 0.25, True, 0.05, True)
            torch.testing.assert_close(got, x, atol=2e-5, rtol=2e-5)
        x = torch.randn(1, 1, 2304, 2)
        a = filter_logits(x, (36, 64), 1.2, 1, 0.25, True, 0.05, True)
        b = filter_logits(x, (48, 48), 1.2, 1, 0.25, True, 0.05, True)
        self.assertFalse(torch.allclose(a, b))

    def test_invalid_inputs_rejected_before_fft(self):
        for kwargs in [{"strength": float("inf")}, {"cutoff": 0},
                       {"mask_width": 0}, {"start_percent": 0.8, "end_percent": 0.2}]:
            with self.assertRaises(ValueError):
                AFMConfig(**kwargs)
        with self.assertRaises(ValueError):
            filter_logits(torch.zeros(1, 1, 15, 2), (3, 4), 1, 1, 0.25, True, 0.05, True)
        with self.assertRaises(FloatingPointError):
            filter_logits(torch.full((1, 1, 15, 2), float("nan")),
                          (3, 5), 1, 1, 0.25, True, 0.05, True)

    def test_image_first_layout_uses_actual_grid(self):
        layout = TokenLayout((3, 5), 7)
        self.assertEqual(layout.image_slice, (0, 15))
        self.assertEqual(layout.text_slice, (15, 22))
        self.assertEqual(layout.total_tokens, 22)
        with self.assertRaises(ValueError):
            TokenLayout((3, 0), 7)

    def test_scope_parser_rejects_silent_empty_or_out_of_range(self):
        self.assertEqual(parse_block_scope("0,2-4", 6), frozenset({0, 2, 3, 4}))
        self.assertEqual(parse_block_scope("all", 3), frozenset({0, 1, 2}))
        for scope in ["", "-1", "4-2", "6", "0,,1"]:
            with self.assertRaises(ValueError):
                parse_block_scope(scope, 6)
