"""Build matched comparisons from the supplied volleyball generation metadata."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afm.block_tune import IMAGE2_VARIANTS
from afm.block_tune_window import APPLICATION_MODES


def build_examples(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_api = json.loads((ROOT / 'examples/block_tune_texture_source_api.json').read_text('utf-8'))
    source_ui = json.loads((ROOT / 'examples/block_tune_texture_source_ui.json').read_text('utf-8'))
    for preset in ('image_2_global_qkv', *IMAGE2_VARIANTS, 'image_3_individual_qkv_partial'):
        for application in APPLICATION_MODES:
            stem = f'{preset}__{application}'
            prefix = f'moss_block_tune_texture/{stem}'
            api, ui = copy.deepcopy(source_api), copy.deepcopy(source_ui)
            api['42']['inputs'].update(preset=preset, application=application)
            api['9']['inputs']['filename_prefix'] = prefix
            for node in ui['nodes']:
                if node['id'] == 42:
                    node['widgets_values'] = [preset, application]
                    node['widgets_values_named'] = {'preset': preset, 'application': application}
                    node['title'] = f'Block Tune: {preset} / {application}'
                    node['size'] = [470, 150]
                elif node['id'] == 9:
                    node['widgets_values'] = [prefix]
                    node['widgets_values_named'] = {'filename_prefix': prefix}
            ui['extra']['block_tune_comparison'] = {'preset': preset, 'application': application}
            for kind, data in (('api', api), ('ui', ui)):
                (output / f'{stem}_{kind}.json').write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'examples/block_tune_texture'))
    build_examples(parser.parse_args().output)
