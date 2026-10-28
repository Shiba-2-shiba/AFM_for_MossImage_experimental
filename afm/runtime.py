from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from uuid import uuid4

import torch

from . import __version__
from .config import TokenLayout
from .residual import RESIDUAL_CONFIG_KEY

RUN_KEY = "moss_afm_run"
CONTEXT_KEY = "moss_afm_context"
CONFIG_KEY = "moss_afm_config"


def branch_rows(branches, batch, mode):
    if (not isinstance(branches, (list, tuple)) or not branches or batch <= 0
            or any(type(b) is not int or b not in (0, 1) for b in branches)
            or batch % len(branches)):
        raise ValueError("Unsupported cond_or_uncond layout")
    width = batch // len(branches)
    wanted = {"positive_only": (0,), "negative_only": (1,), "both": (0, 1)}[mode]
    return tuple(row for index, branch in enumerate(branches) if branch in wanted
                 for row in range(index * width, (index + 1) * width))


@dataclass
class RunState:
    total_steps: int
    config: object
    run_id: str = field(default_factory=lambda: uuid4().hex)
    step_index: int = 0
    callbacks: int = 0
    model_evaluations: int = 0
    observed_calls: int = 0
    edited_calls: int = 0
    fallback_calls: int = 0
    skipped_calls: int = 0
    reasons: Counter = field(default_factory=Counter)
    calls: Counter = field(default_factory=Counter)
    evaluations_per_step: Counter = field(default_factory=Counter)
    layouts: set = field(default_factory=set)
    log_path: Path | None = None
    finished: bool = False

    @property
    def progress(self):
        if not 0 <= self.step_index < self.total_steps:
            raise RuntimeError("AFM call outside sampler step")
        return self.step_index / max(self.total_steps - 1, 1)

    def record(self, event, **values):
        if self.config.logging == "off":
            return
        row = {"schema_version": 1, "run_id": self.run_id, "event": event, **values}
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def make_outer_wrapper(config, validate_sampler, output_directory):
    def outer(executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
              callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        validate_sampler(sampler)
        if sigmas.ndim != 1 or len(sigmas) < 2 or not torch.isfinite(sigmas).all():
            raise ValueError("AFM requires a finite sampler sigma schedule")
        guider = executor.class_obj
        original = guider.model_options
        options = original.copy()
        options["transformer_options"] = original.get("transformer_options", {}).copy()
        if RUN_KEY in options["transformer_options"]:
            raise RuntimeError("Nested AFM sampling is unsupported")
        if original.get("multigpu_thread_pool") is not None:
            raise ValueError("Multi-GPU AFM is not qualified")
        state = RunState(len(sigmas) - 1, config)
        if config.logging != "off":
            directory = Path(output_directory()) / "moss_afm"
            directory.mkdir(parents=True, exist_ok=True)
            state.log_path = directory / (state.run_id + ".jsonl")
        options["transformer_options"][RUN_KEY] = state
        guider.model_options = options

        def step_done(index, denoised, latent, total):
            if index != state.step_index or total != state.total_steps:
                raise RuntimeError("AFM sampler callback order changed")
            if callback is not None:
                callback(index, denoised, latent, total)
            state.callbacks += 1
            state.step_index += 1

        status, error_type = "error", None
        try:
            residual = options["transformer_options"].get(RESIDUAL_CONFIG_KEY)
            metadata = {} if residual is None else {"residual_config": asdict(residual)}
            state.record("run_start", config=asdict(config), total_steps=state.total_steps,
                         implementation_version=__version__,
                         torch_version=torch.__version__,
                         hip_version=getattr(torch.version, "hip", None),
                         sigmas=sigmas.detach().cpu().tolist(), seed=seed, **metadata)
            result = executor(noise, latent_image, sampler, sigmas, denoise_mask,
                              step_done, disable_pbar, seed, latent_shapes)
            if state.callbacks != state.total_steps:
                raise RuntimeError("AFM sampler callback missing")
            status = "success"
            return result
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            guider.model_options = original
            options["transformer_options"].pop(RUN_KEY, None)
            state.finished = True
            try:
                state.record("run_summary", status=status, error_type=error_type,
                             callbacks=state.callbacks, model_evaluations=state.model_evaluations,
                             observed_calls=state.observed_calls, edited_calls=state.edited_calls,
                             skipped_calls=state.skipped_calls, fallback_calls=state.fallback_calls,
                             reasons=dict(state.reasons),
                             calls=[{"step": k[0], "block": k[1], "branch": k[2],
                                     "action": k[3], "count": v}
                                    for k, v in sorted(state.calls.items())])
            except OSError:
                if error_type is None:
                    raise
                logging.getLogger(__name__).exception("AFM summary write failed")
    return outer


def make_diffusion_wrapper(config):
    def diffusion(executor, x, timestep, context, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        state = options.get(RUN_KEY)
        if not isinstance(state, RunState) or state.finished:
            raise RuntimeError("MOSS AFM requires an active sampling run")
        if x.ndim != 4 or context.ndim != 3 or x.shape[0] != context.shape[0]:
            raise ValueError("MOSS AFM requires native 4D image latent and 3D text context")
        branches = options.get("cond_or_uncond")
        branch_rows(branches, x.shape[0], "both")
        layout = TokenLayout(tuple(x.shape[-2:]), int(context.shape[1]))
        state.progress
        signature = (tuple(x.shape), tuple(context.shape), tuple(branches), str(x.dtype))
        if signature not in state.layouts:
            state.layouts.add(signature)
            state.record("layout", grid=list(layout.grid), text_tokens=layout.text_tokens,
                         batch=x.shape[0], branches=list(branches), dtype=str(x.dtype),
                         device=str(x.device))
        evaluation = state.evaluations_per_step[state.step_index]
        state.evaluations_per_step[state.step_index] += 1
        state.model_evaluations += 1
        options[CONTEXT_KEY] = {"layout": layout, "branches": tuple(branches),
                               "evaluation": evaluation}
        return executor(x, timestep, context, options, **kwargs)
    return diffusion
