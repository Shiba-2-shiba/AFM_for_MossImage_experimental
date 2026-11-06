import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BlockTuneExampleTests(unittest.TestCase):
    def test_examples_preserve_generation_settings_and_route_the_selected_preset(self):
        script = ROOT / 'tools/build_block_tune_texture_examples.py'
        self.assertTrue(script.exists(), 'Missing comparison workflow builder')
        source = json.loads((ROOT / 'examples/block_tune_texture_source_api.json').read_text('utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(script), '--output', directory], check=True)
            paths = list(Path(directory).glob('*_api.json'))
            self.assertEqual(len(paths), 32)
            for path in paths:
                api = json.loads(path.read_text('utf-8'))
                ui = json.loads(path.with_name(path.name.replace('_api.json', '_ui.json')).read_text('utf-8'))
                preset, application = path.stem.removesuffix('_api').split('__')
                self.assertEqual(api['42']['inputs'], {
                    'model': ['28', 0], 'preset': preset, 'application': application})
                self.assertEqual(api['31']['inputs']['model'], ['42', 0])
                for key in source:
                    if key not in ('42', '9'):
                        self.assertEqual(api[key], source[key])
                self.assertEqual(api['32']['inputs']['vae_name'], 'flux2-vae.safetensors')
                self.assertEqual(api['19']['inputs']['aspect_ratio'], '16:9 (Widescreen)')
                self.assertNotIn('41', api)
                self.assertFalse(any(n['class_type'] == 'MossImageAFM' for n in api.values()))
                node = next(n for n in ui['nodes'] if n['id'] == 42)
                self.assertEqual(node['widgets_values'], [preset, application])
                self.assertEqual(node['widgets_values_named'],
                                 {'preset': preset, 'application': application})
                self.assertTrue(any(l[1:5] == [42, 0, 31, 0] for l in ui['links']))
                self.assertEqual({str(n['id']) for n in ui['nodes']}, set(api))
                link_ids = {l[0] for l in ui['links']}
                for n in ui['nodes']:
                    for inp in n.get('inputs', []):
                        if inp.get('link') is not None:
                            self.assertIn(inp['link'], link_ids)
                    for out in n.get('outputs', []):
                        self.assertTrue(set(out.get('links') or []).issubset(link_ids))
