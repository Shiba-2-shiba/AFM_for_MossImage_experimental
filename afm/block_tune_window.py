"""Sampler-step windows for Block Tune, with a per-execution scaling gate."""
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable

from .runtime import RUN_KEY, RunState


APPLICATION_MODES = ('static', 'full_run', 'middle_30_70', 'identity_control')
WINDOW_KEY = 'moss_block_tune_window'


@dataclass(frozen=True)
class WindowRegistration:
    outer: Callable
    diffusion: Callable


def qualified_window(model):
    """Recognize only the intact standalone window pair before sharing its clock."""
    from comfy.patcher_extension import WrappersMP

    registration = model.get_attachment(WINDOW_KEY)
    kinds = (WrappersMP.OUTER_SAMPLE, WrappersMP.DIFFUSION_MODEL)
    if registration is None and not any(model.get_wrappers(kind, WINDOW_KEY) for kind in kinds):
        return None
    if type(registration) is not WindowRegistration:
        raise ValueError('Block Tune window registration is missing or unsupported')
    expected = {WrappersMP.OUTER_SAMPLE: {WINDOW_KEY: [registration.outer]},
                WrappersMP.DIFFUSION_MODEL: {WINDOW_KEY: [registration.diffusion]}}
    actual = {kind: {key: values for key, values in groups.items() if values}
              for kind, groups in model.wrappers.items() if any(groups.values())}
    legacy = model.model_options.get('transformer_options', {}).get('wrappers', {})
    if actual != expected or any(values for groups in legacy.values() for values in groups.values()):
        raise ValueError('AFM requires the intact Block Tune window pair without extra wrappers')
    return registration


@dataclass(frozen=True)
class GatedScaleRows:
    scale: object
    gate: ContextVar

    def __call__(self, weight):
        active = self.gate.get()
        if active is None:
            raise RuntimeError('Timed Block Tune requires an active text-to-image sampling run; '
                               'reference-image preprocessing is unsupported')
        return self.scale(weight) if active else weight


def make_window_wrapper(application, gate):
    if application not in APPLICATION_MODES[1:]:
        raise ValueError(f'Invalid Block Tune window: {application}')

    def diffusion(executor, x, timestep, context, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        state = options.get(RUN_KEY)
        if not isinstance(state, RunState) or state.finished:
            raise RuntimeError('Timed Block Tune requires an active sampling run')
        if any(kwargs.get(key) is not None for key in ('ref_kv', 'ref_latents')):
            raise ValueError('Timed Block Tune supports text-to-image without reference images only')
        progress = state.progress
        active = application == 'full_run' or (
            application == 'middle_30_70' and .3 <= progress <= .7)
        token = gate.set(active)
        try:
            return executor(x, timestep, context, options, **kwargs)
        finally:
            gate.reset(token)
    return diffusion


def install_window(clone, application, gate):
    from comfy.patcher_extension import WrappersMP
    from .adapter import validate_sampler
    from .config import AFMConfig
    from .runtime import make_outer_wrapper

    # Reuse the qualified Euler/ER-SDE callback clock and its cleanup, without AFM edits.
    outer = make_outer_wrapper(AFMConfig(logging='off'), validate_sampler, lambda: None)
    diffusion = make_window_wrapper(application, gate)
    clone.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, WINDOW_KEY, outer)
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WINDOW_KEY, diffusion)
    clone.set_attachments(WINDOW_KEY, WindowRegistration(outer, diffusion))
