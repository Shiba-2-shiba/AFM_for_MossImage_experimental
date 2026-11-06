"""Sampler-step windows for Block Tune, with a per-execution scaling gate."""
from contextvars import ContextVar
from dataclasses import dataclass

from .runtime import RUN_KEY, RunState


APPLICATION_MODES = ('static', 'full_run', 'middle_30_70', 'identity_control')


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
    clone.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'moss_block_tune_window',
                               make_outer_wrapper(AFMConfig(logging='off'), validate_sampler,
                                                  lambda: None))
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, 'moss_block_tune_window',
                               make_window_wrapper(application, gate))
