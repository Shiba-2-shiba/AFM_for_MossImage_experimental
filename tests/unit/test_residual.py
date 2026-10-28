import unittest

import torch


class ResidualTests(unittest.TestCase):
    def test_fixed_ranges_and_boundaries(self):
        from afm.residual import ResidualConfig
        config = ResidualConfig(0.75, 1.15)
        self.assertEqual([config.multiplier_for(i) for i in (8, 9, 25, 26, 33, 34, 41, 42)],
                         [1, 0.75, 0.75, 1, 1, 1.15, 1.15, 1])
        self.assertTrue(ResidualConfig().bypass)
        self.assertFalse(config.bypass)

    def test_invalid_multiplier(self):
        from afm.residual import ResidualConfig
        for value in (True, '1', None, float('nan'), float('inf'), -float('inf'), -0.01, 2.01):
            for name in ('mid_multiplier', 'late_multiplier'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ResidualConfig(**{name: value})
        for value in (0, 1, 2, 0.755):
            self.assertEqual(ResidualConfig(value).mid_multiplier, value)

    def test_modulation_is_local(self):
        from afm.residual import ResidualBlockPatch
        mod = tuple(torch.full((2, 1, 4), float(i + 1)) for i in range(6))
        before = tuple(x.clone() for x in mod)
        args = {'mod': mod, 'img': torch.zeros(2, 3, 4), 'custom': object()}
        calls = []

        def original(local):
            calls.append(local)
            self.assertIs(local['custom'], args['custom'])
            self.assertIs(local['img'], args['img'])
            for i, value in enumerate(local['mod']):
                if i in (2, 5):
                    torch.testing.assert_close(value, before[i] * 0.75, rtol=0, atol=0)
                else:
                    self.assertIs(value, mod[i])
            return {'img': local['img'], 'extra': 7}

        for index in (9, 10):
            result = ResidualBlockPatch(index, 0.75)(args, {'original_block': original})
            self.assertEqual(result['extra'], 7)
        self.assertEqual(len(calls), 2)
        self.assertIs(args['mod'], mod)
        for actual, expected in zip(mod, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bad_mod_contract(self):
        from afm.residual import scale_residual_mod, ResidualBlockPatch
        mod = tuple(torch.ones(2, 1, 4) for _ in range(6))
        for invalid in (None, mod[:5], (*mod[:5], 'gate'),
                        (*mod[:5], torch.ones(2, 2, 4)),
                        (*mod[:5], torch.ones(2, 1, 4, dtype=torch.float64))):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                scale_residual_mod(invalid, 0.75)
        with self.assertRaises(ValueError):
            ResidualBlockPatch(9, 0.75)({}, {'original_block': lambda args: args})
