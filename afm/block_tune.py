"""Screenshot presets applied through native ComfyUI weight wrappers."""
from dataclasses import dataclass
from contextvars import ContextVar
import json
import math
from pathlib import Path
import uuid

import torch


# (first block through 41 inclusive, V override, O/FFN-down strength override).
# None preserves the source value. These are experiments, not screenshot readings.
IMAGE2_VARIANTS = {
    'image_2_global_v090': (0, .9, None),
    'image_2_global_v100': (0, 1., None),
    'image_2_b34_41_v090': (34, .9, None),
    'image_2_b34_41_v100': (34, 1., None),
    'image_2_b34_41_strength100': (34, None, 1.),
    'image_2_b34_41_v100_strength100': (34, 1., 1.),
}
PRESET_NAMES = ('image_1_individual_qkv', 'image_2_global_qkv', *IMAGE2_VARIANTS,
                'image_3_individual_qkv_partial')
CONFIG_KEY = 'moss_block_tune_preset'


def load_preset(name=PRESET_NAMES[0]):
    if name not in PRESET_NAMES:
        raise ValueError(f'Unknown MOSS block tune preset: {name}')
    source = 'image_2_global_qkv' if name in IMAGE2_VARIANTS else name
    data = json.loads((Path(__file__).parent / 'presets' / f'{source}.json').read_text('utf-8'))
    if name in IMAGE2_VARIANTS:
        first, v, strength = IMAGE2_VARIANTS[name]
        data['name'] = name
        data['experiment'] = {'base_preset': source, 'blocks': list(range(first, 42)),
                              'v_override': v, 'strength_override': strength}
        data['notes'] = 'Experimental image2 variant; image quality is not validated.'
        data['qkv_policy'] = 'image2 reference with explicit experimental overrides'
        for row in data['blocks'][first:]:
            if v is not None:
                row['v'] = v
                row['qkv_observed'] = False
            if strength is not None:
                row['strength'] = strength
                row['strength_observed'] = False
    if data['mode'] != 'residual' or data['head'] != 1 or len(data['blocks']) != 42:
        raise ValueError('Preset requires 42 blocks, residual mode and unchanged head')
    for index, row in enumerate(data['blocks']):
        if row['block'] != index:
            raise ValueError('Preset blocks must be ordered 0..41')
        for key in ('strength', 'q', 'k', 'v'):
            value = row[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= 2):
                raise ValueError(f'Invalid preset multiplier: block {index} {key}')
    return data


def projection_scales(data, fused):
    """Return final per-projection factors; global UI values are never multiplied twice."""
    plan = {}
    for row in data['blocks']:
        prefix = f"layers.{row['block']}.proj."
        values = {'o': (row['strength'],), 'ffn_down': (row['strength'],)}
        if fused:
            values['qkv'] = tuple(row[key] for key in ('q', 'k', 'v'))
        else:
            values.update({key: (row[key],) for key in ('q', 'k', 'v')})
        plan.update({prefix + key: factors for key, factors in values.items()
                     if any(value != 1 for value in factors)})
    if 'experiment' in data:
        # Keep the same INT8 dequantization path even when an experimental factor
        # returns to 1.0. Otherwise a changed kernel could confound the comparison.
        base = load_preset(data['experiment']['base_preset'])
        for path, factors in projection_scales(base, fused).items():
            plan.setdefault(path, tuple(1. for _ in factors))
    return plan


@dataclass(frozen=True)
class ScaleRows:
    factors: tuple[float, ...]

    def __call__(self, weight):
        if weight.ndim not in (1, 2) or weight.shape[0] % len(self.factors):
            raise ValueError('Projection rows must divide evenly into Q/K/V components')
        if len(self.factors) == 1:
            return (weight.float() * self.factors[0]).to(weight.dtype)
        return torch.cat([(part.float() * factor).to(weight.dtype) if factor != 1 else part
                          for part, factor in zip(weight.chunk(len(self.factors)), self.factors)])


def install_block_tune(model, preset=PRESET_NAMES[0], application='static'):
    from .block_tune_window import APPLICATION_MODES, GatedScaleRows, install_window
    if application not in APPLICATION_MODES:
        raise ValueError(f'Unknown Block Tune application: {application}')
    if preset == 'off':
        return model.clone(), {'status': 'bypass', 'preset': preset, 'application': application}
    data = load_preset(preset)
    from .adapter import detect_native_moss
    from .runtime import CONFIG_KEY as AFM_CONFIG_KEY
    from .residual import RESIDUAL_CONFIG_KEY
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrappersMP, get_all_wrappers

    if not isinstance(model, ModelPatcher):
        raise ValueError('Block Tune requires a native ComfyUI ModelPatcher')
    caps = detect_native_moss(model)
    if caps.blocks != 42:
        raise ValueError('Block Tune presets require exactly 42 native MOSS blocks')
    options = model.model_options.get('transformer_options', {})
    if (any(key in options for key in (CONFIG_KEY, AFM_CONFIG_KEY, RESIDUAL_CONFIG_KEY,
                                       'moss_k_row_balance', 'optimized_attention_override'))
            or options.get('patches_replace') or any(model.wrappers.values())
            or get_all_wrappers(WrappersMP.DIFFUSION_MODEL, options)
            or any(model.injections.values())):
        raise ValueError('Apply Block Tune once, after LoRA and before AFM; do not stack Residual/K Balance')
    if (model.hook_patches or model.current_hooks is not None or model.forced_hooks is not None
            or model.is_multigpu_base_clone or 'multigpu' in model.additional_models
            or any(label in str(options).lower() for label in ('cache', 'compile', 'multigpu'))):
        raise ValueError('Model hooks/cache/compile/multi-GPU are not qualified for Block Tune')
    delegated = False
    if application != 'static' and model.is_dynamic():
        delegate = getattr(model, 'get_non_dynamic_delegate', None)
        if not callable(delegate):
            raise ValueError('Timed Block Tune requires the native non-dynamic delegate API')
        model = delegate()
        if not isinstance(model, ModelPatcher) or model.is_dynamic():
            raise ValueError('Timed Block Tune delegate must return a non-dynamic ModelPatcher')
        detect_native_moss(model)
        delegated = True
    native = model.model.diffusion_model
    fused = 'qkv' in native.layers[0].proj
    if any(('qkv' in layer.proj) != fused for layer in native.layers):
        raise ValueError('Mixed fused/separate QKV layout is unsupported')
    plan = projection_scales(data, fused)
    gate = ContextVar('moss_block_tune_active', default=None) if application != 'static' else None
    wrappers = {}
    for path, factors in plan.items():
        target = 'diffusion_model.' + path
        if any(target == key or target.startswith(key + '.') or key.startswith(target + '.')
               for key in model.object_patches):
            raise ValueError(f'Existing object patch conflicts with {target}')
        module = model.get_model_object(target)
        if (not hasattr(module, 'weight_function') or not hasattr(module, 'bias_function')
                or module.weight.ndim != 2 or module.weight.shape[0] % len(factors)):
            raise ValueError(f'Unsupported native projection: {target}')
        for parameter in ('weight', 'bias'):
            key = target + '.' + parameter
            if getattr(module, parameter, None) is None:
                continue
            if key in model.weight_wrapper_patches:
                raise ValueError(f'Existing weight wrapper conflicts with {key}')
            scale = ScaleRows(factors)
            wrappers[key] = GatedScaleRows(scale, gate) if gate is not None else scale
    clone = model.clone()
    for key, wrapper in wrappers.items():
        clone.add_weight_wrapper(key, wrapper)
    if gate is not None:
        install_window(clone, application, gate)
    report = {
        'status': 'applied', 'preset': preset, 'blocks': data['blocks'],
        'unobserved_qkv_blocks': [row['block'] for row in data['blocks'] if not row['qkv_observed']],
        'unobserved_strength_blocks': [row['block'] for row in data['blocks']
                                       if not row['strength_observed']],
        'qkv_policy': data['qkv_policy'], 'global_multiplied_again': False,
        'head': 1.0, 'weight_wrappers': len(wrappers), 'checkpoint_written': False,
        'application': application, 'dynamic_delegate_used': delegated,
    }
    if gate is not None:
        report['sampling_window'] = {
            'start': .3 if application == 'middle_30_70' else 0.,
            'end': .7 if application == 'middle_30_70' else 1.,
            'inclusive': True, 'progress': 'step_index / max(total_steps - 1, 1)',
            'identity_control': application == 'identity_control',
            'outside_window': 'identity multipliers with the same weight-wrapper backend',
            'afm_compatible': False, 'reference_images_supported': False,
        }
    if 'experiment' in data:
        report['experiment'] = data['experiment']
    if 'source_image' in data:
        report.update(source_image=data['source_image'], head_observed=data['head_observed'])
    # Native clone comparison ignores wrapper-only differences when no LoRA is present.
    # A unique attachment forces the normal unload/reload path when switching branches.
    clone.set_attachments(CONFIG_KEY + ':' + uuid.uuid4().hex, report)
    clone.model_options.setdefault('transformer_options', {})[CONFIG_KEY] = preset
    return clone, report
