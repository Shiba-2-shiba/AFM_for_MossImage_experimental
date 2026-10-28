from dataclasses import dataclass
import math

import torch

from .attention import edit_image_rows, write_image_rows
from .config import parse_block_scope
from .memory import MemoryGuardError, plan_chunks
from .runtime import (CONFIG_KEY, CONTEXT_KEY, RUN_KEY, RunState, branch_rows,
                      make_diffusion_wrapper, make_outer_wrapper)


@dataclass(frozen=True)
class Capabilities:
    blocks: int
    heads: int
    head_dim: int
    latent_channels: int


class AttentionOverride:
    def __init__(self, config, state, context, preferred, dispatcher):
        self.config, self.state, self.context = config, state, context
        self.preferred, self.dispatcher = preferred, dispatcher

    def __call__(self, original_func, q, k, v, heads, **kwargs):
        return self._call(q, k, v, heads, **kwargs)

    def container_function(self, q, k, v, heads, **kwargs):
        return self._call(q, k, v, heads, **kwargs)

    def _call(self, q, k, v, heads, **kwargs):
        config, state, context = self.config, self.state, self.context
        layout = context["layout"]
        call_kwargs = dict(kwargs)
        options = dict(call_kwargs.get("transformer_options") or {})
        options.pop("optimized_attention_override", None)
        call_kwargs["transformer_options"] = options
        call_kwargs.pop("_inside_attn_wrapper", None)
        call_kwargs["preferred_attention"] = self.preferred
        tensors = tuple(x.peek() if hasattr(x, "peek") else x for x in (q, k, v))
        step, block = state.step_index, context["block"]
        selected = branch_rows(context["branches"], tensors[0].shape[0], config.branch_mode)
        active = config.active_interval(state.progress) in context["interval_ids"]
        action = "observe" if config.mode == "observe" else "edit"
        if not selected or not active:
            state.skipped_calls += 1
            action = "skipped"
        def record_action(result):
            width = tensors[0].shape[0] // len(context["branches"])
            for position, branch in enumerate(context["branches"]):
                chosen = any(row in selected for row in range(position * width, (position + 1) * width))
                branch_action = result if chosen else "skipped"
                state.calls[(step, block, branch, branch_action)] += 1
        if action != "edit":
            if action == "observe":
                state.observed_calls += 1
            record_action(action)
            return self.dispatcher(q, k, v, heads, **call_kwargs)

        try:
            allowed = {"transformer_options", "_inside_attn_wrapper", "attn_precision",
                       "skip_reshape", "skip_output_reshape", "mask", "is_causal",
                       "scale", "preferred_attention"}
            if set(kwargs) - allowed:
                raise ValueError("Unsupported MOSS attention options")
            if kwargs.get("mask") is not None or kwargs.get("is_causal", False):
                raise ValueError("Masked/causal AFM is unsupported")
            if not all(isinstance(x, torch.Tensor) for x in tensors):
                raise ValueError("Invalid QKV tensors")
            if kwargs.get("skip_reshape", False):
                shaped = tensors
            else:
                if any(x.ndim != 3 or x.shape[-1] % heads for x in tensors):
                    raise ValueError("Expected native flat QKV")
                shaped = tuple(x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)
                               for x in tensors)
            if (any(x.ndim != 4 for x in shaped) or shaped[0].shape != shaped[1].shape
                    or shaped[0].shape != shaped[2].shape
                    or shaped[0].shape[2] != layout.total_tokens
                    or shaped[0].shape[1] != heads):
                raise ValueError("Native MOSS attention layout changed")
            chunks = plan_chunks(shaped[0].shape, shaped[0].element_size(), layout,
                                 config.max_workspace_mib, config.head_chunk_size,
                                 config.query_chunk_size)
            scale = kwargs.get("scale", shaped[0].shape[-1] ** -0.5)
            if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid attention scale")
        except (ValueError, MemoryGuardError) as exc:
            if config.fail_mode == "raise":
                raise
            state.fallback_calls += 1
            record_action("fallback")
            state.reasons[str(exc)] += 1
            return self.dispatcher(q, k, v, heads, **call_kwargs)

        # Backend errors propagate. Only an edit failure may return this baseline.
        baseline = self.dispatcher(q, k, v, heads, **call_kwargs)
        try:
            candidate = baseline.clone()
            for row in selected:
                images = edit_image_rows(*(x[row:row+1] for x in shaped),
                                         layout, config, state.progress, chunks, scale)
                write_image_rows(candidate, images, [row], layout, heads,
                                 kwargs.get("skip_output_reshape", False))
            if not torch.isfinite(candidate).all():
                raise FloatingPointError("Nonfinite native or edited output")
        except (FloatingPointError, MemoryGuardError, torch.OutOfMemoryError, ValueError) as exc:
            if config.fail_mode == "raise":
                raise
            state.fallback_calls += 1
            record_action("fallback")
            state.reasons[type(exc).__name__] += 1
            return baseline
        state.edited_calls += 1
        record_action("edit")
        return candidate


def detect_native_moss(model):
    from comfy.ldm.moss_image.model import MOSSImage
    native = model.model.diffusion_model
    if not isinstance(native, MOSSImage) or hasattr(native, "_orig_mod"):
        raise ValueError("MOSS Image AFM requires an uncompiled native MOSS MODEL")
    if not native.layers:
        raise ValueError("Native MOSS has no layers")
    heads = {layer.num_heads for layer in native.layers}
    dims = {layer.q_norm.weight.shape[0] for layer in native.layers}
    if len(heads) != 1 or len(dims) != 1:
        raise ValueError("Heterogeneous MOSS heads are unsupported")
    return Capabilities(len(native.layers), heads.pop(), dims.pop(), native.latent_channels)


def validate_sampler(sampler):
    from comfy.k_diffusion import sampling
    if sampler.sampler_function not in (sampling.sample_euler, sampling.sample_er_sde):
        raise ValueError("MOSS AFM currently supports standard Euler and ER-SDE only")


def install_afm(model, config):
    import folder_paths
    from comfy.ldm.modules.attention import optimized_attention
    from comfy.patcher_extension import WrappersMP, get_all_wrappers

    caps = detect_native_moss(model)
    first_scope = parse_block_scope(config.block_scope, caps.blocks)
    second_scope = (parse_block_scope(config.second_block_scope, caps.blocks)
                    if config.two_interval_mode else frozenset())
    if config.bypass:
        return model.clone()
    options = model.model_options.get("transformer_options", {})
    if CONFIG_KEY in options or "optimized_attention_override" in options:
        raise ValueError("Conflicting or duplicate AFM attention patch")
    if options.get("patches_replace", {}).get("dit"):
        raise ValueError("Existing DiT block replacement conflicts with MOSS AFM")
    if (model.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
            or get_all_wrappers(WrappersMP.DIFFUSION_MODEL, options)):
        raise ValueError("Existing diffusion wrapper requires separate AFM qualification")
    labels = " ".join(str(k).lower() for k in options)
    if any(key in labels for key in ("cache", "spectrum", "compile", "multigpu")):
        raise ValueError("Model skip-cache/compile/multi-GPU is not qualified for AFM")
    clone = model.clone()
    clone.model_options["transformer_options"][CONFIG_KEY] = config
    clone.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "moss_afm",
                               make_outer_wrapper(config, validate_sampler,
                                                  folder_paths.get_output_directory))
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "moss_afm",
                               make_diffusion_wrapper(config))
    native = model.model.diffusion_model

    def replacement_for(index):
        preferred = native.layers[index].comfy_attention
        interval_ids = tuple(n for n, scope in ((1, first_scope), (2, second_scope))
                             if index in scope)

        def replacement(args, extra):
            local = dict(args)
            opts = dict(args["transformer_options"])
            state = opts.get(RUN_KEY)
            if not isinstance(state, RunState) or state.finished:
                raise RuntimeError("MOSS AFM block outside sampling run")
            context = dict(opts[CONTEXT_KEY], block=index, interval_ids=interval_ids)
            opts["optimized_attention_override"] = AttentionOverride(
                config, state, context, preferred, optimized_attention)
            local["transformer_options"] = opts
            return extra["original_block"](local)
        return replacement

    for index in first_scope | second_scope:
        clone.set_model_patch_replace(replacement_for(index), "dit", "single_block", index)
    return clone
