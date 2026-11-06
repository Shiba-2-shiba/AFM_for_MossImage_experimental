import unittest

import torch


class BlockTuneTests(unittest.TestCase):
    def test_third_screenshot_tracks_visible_and_missing_values_separately(self):
        from afm.block_tune import PRESET_NAMES, load_preset
        name = 'image_3_individual_qkv_partial'
        self.assertIn(name, PRESET_NAMES)
        data = load_preset(name)
        self.assertEqual(data['blocks'][0], {
            'block': 0, 'strength': 1.05, 'q': 1.02, 'k': .99, 'v': 1.01,
            'qkv_observed': True, 'strength_observed': True})
        self.assertEqual(data['blocks'][21]['q'], .80)
        self.assertEqual(data['blocks'][21]['k'], .79)
        self.assertEqual(data['blocks'][29]['q'], .91)
        self.assertEqual(data['blocks'][35]['v'], 1.05)
        self.assertEqual(data['blocks'][15]['strength'], .85)
        self.assertEqual(data['blocks'][36]['strength'], 1.05)
        self.assertEqual([r['block'] for r in data['blocks'] if not r['qkv_observed']],
                         [*range(15, 21), *range(36, 42)])
        self.assertEqual([r['block'] for r in data['blocks'] if not r['strength_observed']],
                         [*range(16, 21), *range(37, 42)])
        for row in data['blocks']:
            if not row['qkv_observed']:
                self.assertEqual([row[k] for k in ('q', 'k', 'v')], [1., 1., 1.])
            if not row['strength_observed']:
                self.assertEqual(row['strength'], 1.)
        self.assertFalse(data['head_observed'])
        self.assertEqual(data['head'], 1.)

    def test_image2_variants_isolate_requested_blocks_and_projections(self):
        from afm.block_tune import PRESET_NAMES, load_preset, projection_scales
        baseline = load_preset('image_2_global_qkv')
        cases = (
            ('image_2_global_v090', 0, .9, None),
            ('image_2_global_v100', 0, 1., None),
            ('image_2_b34_41_v090', 34, .9, None),
            ('image_2_b34_41_v100', 34, 1., None),
            ('image_2_b34_41_strength100', 34, None, 1.),
            ('image_2_b34_41_v100_strength100', 34, 1., 1.),
        )
        for name, first, v, strength in cases:
            with self.subTest(preset=name):
                self.assertIn(name, PRESET_NAMES)
                data = load_preset(name)
                self.assertEqual(data['name'], name)
                self.assertEqual(data['head'], 1.)
                self.assertEqual(data['blocks'][:first], baseline['blocks'][:first])
                for row, original in zip(data['blocks'], baseline['blocks']):
                    self.assertEqual((row['q'], row['k']), (1.7, 1.))
                    self.assertEqual(row['v'], v if row['block'] >= first and v is not None
                                     else original['v'])
                    self.assertEqual(row['strength'], strength
                                     if row['block'] >= first and strength is not None
                                     else original['strength'])
                for fused in (False, True):
                    base_plan = projection_scales(baseline, fused)
                    plan = projection_scales(data, fused)
                    for index in range(first):
                        prefix = f'layers.{index}.proj.'
                        self.assertEqual({k: v for k, v in plan.items() if k.startswith(prefix)},
                                         {k: v for k, v in base_plan.items() if k.startswith(prefix)})
                    if strength == 1:
                        self.assertEqual(plan['layers.34.proj.o'], (1.,))
                        self.assertEqual(plan['layers.41.proj.ffn_down'], (1.,))
                    if v == 1 and not fused:
                        self.assertEqual(plan['layers.34.proj.v'], (1.,))
                        self.assertEqual(plan['layers.41.proj.v'], (1.,))
                    self.assertEqual(set(plan), set(base_plan))
                # Returned data is private; editing one comparison cannot change another.
                data['blocks'][0]['v'] = 0
                self.assertEqual(load_preset('image_2_global_qkv'), baseline)

    def test_screenshot_values_and_unknowns(self):
        from afm.block_tune import load_preset
        data = load_preset()
        self.assertEqual(len(data['blocks']), 42)
        self.assertEqual(data['blocks'][0], {'block': 0, 'strength': 1.0,
                                           'q': 1.45, 'k': 1.54, 'v': 1.39,
                                           'qkv_observed': True, 'strength_observed': True})
        self.assertEqual(data['blocks'][3]['q'], .58)
        self.assertEqual(data['blocks'][28]['k'], 1.51)
        self.assertEqual(data['blocks'][34]['v'], 1.33)
        for index in (*range(14, 21), *range(35, 42)):
            row = data['blocks'][index]
            self.assertFalse(row['qkv_observed'])
            self.assertEqual([row[k] for k in ('q', 'k', 'v')], [1.7, 1., .8])
            self.assertEqual(row['strength'], 1.)
        self.assertEqual(data['head'], 1.)
        second = load_preset('image_2_global_qkv')
        for row in second['blocks']:
            self.assertEqual([row[k] for k in ('q', 'k', 'v')], [1.7, 1., .8])
        self.assertEqual(second['blocks'][35]['strength'], 1.1)
        self.assertEqual(second['blocks'][41]['strength'], .95)

    def test_projection_plan_uses_individual_values_without_global_multiplication(self):
        from afm.block_tune import load_preset, projection_scales
        data = load_preset()
        for fused in (False, True):
            plan = projection_scales(data, fused)
            if fused:
                self.assertEqual(plan['layers.0.proj.qkv'], (1.45, 1.54, 1.39))
                self.assertEqual(plan['layers.14.proj.qkv'], (1.7, 1., .8))
            else:
                self.assertEqual(plan['layers.0.proj.q'], (1.45,))
                self.assertEqual(plan['layers.14.proj.q'], (1.7,))
                self.assertNotIn('layers.14.proj.k', plan)
                self.assertEqual(plan['layers.14.proj.v'], (.8,))
            self.assertNotIn('layers.35.proj.o', plan)
            self.assertNotIn('layers.41.proj.ffn_down', plan)
            self.assertFalse(any('norm' in key or 'head' in key for key in plan))

    def test_scaling_preserves_source_and_supports_fused_weights_and_bias(self):
        from afm.block_tune import ScaleRows
        for shape in ((12,), (12, 8)):
            for dtype in (torch.float32, torch.bfloat16):
                source = torch.arange(1, 1 + torch.Size(shape).numel()).reshape(shape).to(dtype)
                before = source.clone()
                actual = ScaleRows((1.45, 1., .8))(source)
                expected = torch.cat([(part.float() * scale).to(dtype)
                                      for part, scale in zip(source.chunk(3), (1.45, 1., .8))])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(source, before, rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
        with self.assertRaises(ValueError):
            ScaleRows((1., 2., 1.))(torch.ones(4, 2))

    def test_schema_is_registered(self):
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[2] / 'nodes.py').read_text('utf-8'))
        names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn('MossImageBlockTunePreset', names)
        extension = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == 'MossAFMExtension')
        self.assertTrue(any(isinstance(node, ast.Name) and node.id == 'MossImageBlockTunePreset'
                            for node in ast.walk(extension)))
