from dataclasses import asdict
import unittest

from afm.config import AFMConfig


class IntervalTests(unittest.TestCase):
    def test_primary_interval_wins_shared_boundary(self):
        config = AFMConfig(end_percent=0.5, two_interval_mode=True)
        self.assertEqual([config.active_interval(p) for p in (0, 0.25, 0.5, 0.75, 1)],
                         [1, 1, 1, 2, 2])

    def test_disabled_mode_preserves_primary_window(self):
        config = AFMConfig(start_percent=0.25, end_percent=0.5)
        self.assertEqual([config.active_interval(p) for p in (0, 0.25, 0.5, 0.75, 1)],
                         [None, 1, 1, None, None])
        self.assertFalse(config.two_interval_mode)

    def test_overlap_gap_and_zero_width(self):
        overlap = AFMConfig(start_percent=0.2, end_percent=0.8, two_interval_mode=True,
                            second_start_percent=0.1, second_end_percent=0.9)
        self.assertEqual([overlap.active_interval(p) for p in (0, 0.1, 0.5, 0.9, 1)],
                         [None, 2, 1, 2, None])
        gap = AFMConfig(end_percent=0.25, two_interval_mode=True,
                        second_start_percent=0.75)
        self.assertIsNone(gap.active_interval(0.5))
        point = AFMConfig(start_percent=0.5, end_percent=0.5)
        self.assertEqual([point.active_interval(p) for p in (0.49, 0.5, 0.51)],
                         [None, 1, None])
        shadowed = AFMConfig(two_interval_mode=True)
        self.assertEqual([shadowed.active_interval(p) for p in (0, 0.5, 1)], [1, 1, 1])

    def test_interval_validation(self):
        for enabled in (False, True):
            for settings in ({"second_start_percent": float("nan")},
                             {"second_end_percent": float("inf")},
                             {"second_start_percent": True},
                             {"second_end_percent": "1"},
                             {"second_start_percent": -0.1},
                             {"second_end_percent": 1.1},
                             {"second_start_percent": 0.9, "second_end_percent": 0.8}):
                with self.subTest(enabled=enabled, settings=settings), self.assertRaises(ValueError):
                    AFMConfig(two_interval_mode=enabled, **settings)
        for enabled in (0, 1, None, "true"):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                AFMConfig(two_interval_mode=enabled)
        for progress in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                AFMConfig().active_interval(progress)

    def test_executed_step_schedule_boundaries(self):
        config = AFMConfig(end_percent=0.5, two_interval_mode=True)
        for steps, first, second in ((25, 13, 12), (24, 12, 12), (1, 1, 0)):
            chosen = [config.active_interval(i / max(steps - 1, 1)) for i in range(steps)]
            self.assertEqual((chosen.count(1), chosen.count(2)), (first, second))

    def test_legacy_positional_arguments_keep_their_meaning(self):
        config = AFMConfig("edit", 0.12, "24-26", "both", 0.1, 0.8,
                           0.4, False, 0.1, False, 2, 64, 256, "raise", "off")
        expected = dict(mode="edit", strength=0.12, block_scope="24-26", branch_mode="both",
                        start_percent=0.1, end_percent=0.8, cutoff=0.4, soft_mask=False,
                        mask_width=0.1, preserve_dc=False, head_chunk_size=2,
                        query_chunk_size=64, max_workspace_mib=256, fail_mode="raise",
                        logging="off", two_interval_mode=False, second_block_scope="0-2",
                        second_start_percent=0.5, second_end_percent=1.0)
        self.assertEqual(asdict(config), expected)
