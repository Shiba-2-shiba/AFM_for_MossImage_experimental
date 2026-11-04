"""Reversible selected-block K-row scale edits for native INT8 MOSS."""

import json
import uuid

import torch

from .k_balance_math import KBalanceConfig, compute_k_gains
from .config import parse_block_scope


TARGET_MODULE = "diffusion_model.layers.41.proj.qkv"
TARGET_WEIGHT = TARGET_MODULE + ".weight"
CONFIG_KEY = "moss_k_row_balance"
ATTACHMENT_PREFIX = "moss_k_row_balance:"
DEFAULT_BLOCKS = "41"


def parse_k_blocks(blocks):
    if not isinstance(blocks, str):
        raise ValueError("blocks must be a comma-separated string of indices/ranges")
    return tuple(sorted(parse_block_scope(blocks, 42)))


def target_module_path(block_index):
    if type(block_index) is not int or not 0 <= block_index < 42:
        raise ValueError("block index must be in 0..41")
    return f"diffusion_model.layers.{block_index}.proj.qkv"


def _patcher_type(model):
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _resolve_patcher(model):
    from comfy.model_patcher import ModelPatcher

    if not isinstance(model, ModelPatcher):
        raise ValueError(f"Unsupported K Row Balance patcher: {_patcher_type(model)}")
    if not model.is_dynamic():
        return model, False
    delegate = getattr(model, "get_non_dynamic_delegate", None)
    if not callable(delegate):
        raise ValueError(f"{_patcher_type(model)} lacks the native non-dynamic delegate API")
    # Native API caches the reloaded model. Loader/OOM failures must propagate.
    standard = delegate()
    if not isinstance(standard, ModelPatcher) or standard.is_dynamic():
        raise ValueError("Native delegate did not return a non-dynamic ModelPatcher")
    return standard, True


def _canonical_state(module):
    from comfy.ops import MixedPrecisionOp
    from comfy.quant_ops import QuantizedTensor

    if not isinstance(module, MixedPrecisionOp) or not isinstance(module.weight, QuantizedTensor):
        raise ValueError("K Row Balance requires native quantized mixed-precision Linear")
    hook_registries = (
        "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
        "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    if any(getattr(module, name, None) for name in hook_registries):
        raise ValueError("Existing target module hooks are not qualified for K Row Balance")
    overrides = ("forward", "_forward", "forward_comfy_cast_weights", "state_dict",
                 "_load_from_state_dict", "_save_to_state_dict", "convert_weight", "set_weight",
                 "_apply", "_call_impl", "_wrapped_call_impl")
    if any(name in vars(module) for name in overrides):
        raise ValueError("Existing target runtime override/patch conflicts with K Row Balance")
    if getattr(module, "_compiled_call_impl", None) is not None:
        raise ValueError("Compiled target projection is not qualified for K Row Balance")
    if (module.quant_format != "int8_tensorwise"
            or module.layout_type != "TensorWiseINT8Layout"
            or module.weight._layout_cls != "TensorWiseINT8Layout"
            or module.bias is not None):
        raise ValueError("Unsupported QKV quantization layout or bias")
    params = module.weight._params
    group = params.convrot_groupsize
    if (params.convrot is not True or type(group) is not int or group < 4
            or group & (group - 1) or group.bit_length() % 2 != 1):
        raise ValueError("Unsupported ConvRot group size")
    dim = module.in_features
    if (type(dim) is not int or not 0 < dim <= 3072 or module.out_features != 3 * dim
            or dim % group or tuple(params.orig_shape) != (3 * dim, dim)):
        raise ValueError("Unsupported fused QKV shape")
    if module.factory_kwargs["dtype"] != module.weight.dtype:
        raise ValueError("QKV factory/weight compute dtype mismatch")
    if module.weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Only FP32/BF16 compute metadata is qualified")
    if (module.weight_function or module.bias_function
            or getattr(module, "weight_lowvram_function", None) is not None
            or getattr(module, "bias_lowvram_function", None) is not None):
        raise ValueError("Existing target weight function or forward patch conflicts")
    if module.comfy_gemv_optimized:
        raise ValueError("Packed/GEMV-optimized target is not qualified")
    state = dict(module.state_dict())
    if set(state) != {"weight", "weight_scale", "comfy_quant"}:
        raise ValueError("Unexpected auxiliary QKV state; no silent parameter dropping")
    codes, scales = state["weight"], state["weight_scale"]
    if (codes.dtype != torch.int8 or tuple(codes.shape) != (3 * dim, dim)
            or codes.device.type == "meta" or scales.device.type == "meta"):
        raise ValueError("QKV requires materialized canonical signed INT8 codes")
    if scales.dtype != torch.float32 or tuple(scales.shape) != (3 * dim, 1):
        raise ValueError("QKV requires FP32 per-output-row scales")
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise FloatingPointError("QKV scales must be finite and positive")
    conf = json.loads(state["comfy_quant"].cpu().numpy().tobytes())
    if (conf.get("format") != "int8_tensorwise" or conf.get("convrot") is not True
            or conf.get("convrot_groupsize") != group):
        raise ValueError("Serialized QKV quantization metadata disagrees with live layout")
    return {key: value.detach().to(device="cpu", copy=True) for key, value in state.items()}


def balance_projection(module, config: KBalanceConfig, head_dim: int, device, block_index=41):
    """Build one independent native Linear; never call a requantizing setter."""
    state = _canonical_state(module)
    dim = module.in_features
    old_scales = state["weight_scale"]
    gains, report = compute_k_gains(state["weight"][dim:2 * dim],
                                   old_scales[dim:2 * dim], head_dim, config)
    new_scales = old_scales.clone()
    new_scales[dim:2 * dim] *= gains
    replacement = type(module)(dim, 3 * dim, bias=False, device=device,
                               dtype=module.factory_kwargs["dtype"])
    replacement.load_state_dict({**state, "weight_scale": new_scales}, strict=True)
    replacement._full_precision_mm = module._full_precision_mm
    replacement.train(module.training)
    if hasattr(module, "comfy_force_cast_weights"):
        replacement.comfy_force_cast_weights = module.comfy_force_cast_weights
    after = replacement.state_dict()
    codes_same = torch.equal(state["weight"], after["weight"].cpu())
    qv_same = (torch.equal(old_scales[:dim], after["weight_scale"][:dim].cpu())
               and torch.equal(old_scales[2 * dim:], after["weight_scale"][2 * dim:].cpu()))
    if not codes_same or not qv_same:
        raise RuntimeError("Native reconstruction changed INT8 codes or Q/V scales")
    report.update({
        "status": "rebuild_control" if config.bypass else "edited",
        "block_index": block_index, "scope": "K_weight_row_scale_only",
        "quantized_codes_unchanged": codes_same, "qv_scales_unchanged": qv_same,
        "convrot_groupsize": module.weight._params.convrot_groupsize,
        "compute_dtype": str(module.weight.dtype),
        "full_precision_mm": bool(replacement._full_precision_mm),
        "full_precision_mm_preserved": replacement._full_precision_mm == module._full_precision_mm,
        "training_mode_preserved": replacement.training == module.training,
        "additional_projection_storage_bytes": state["weight"].numel() + new_scales.numel() * 4,
        "estimated_peak_working_bytes": (report["workspace_estimate_bytes"]
                                         + 3 * state["weight"].numel()
                                         + 4 * new_scales.numel() * 4),
        "memory_estimate_scope": "construction only; excludes model and forward backend workspace",
        "checkpoint_written": False, "image_quality_effect": "unverified",
    })
    return replacement, report


def _overlaps_target(path, target):
    return (path == target or path.startswith(target + ".")
            or target.startswith(path + "."))


def _reject_conflicts(model, targets):
    options = model.model_options.get("transformer_options", {})
    if any(key.startswith(ATTACHMENT_PREFIX) for key in model.attachments):
        raise ValueError("K Row Balance is already applied; use the original input MODEL")
    for patches in (model.patches, model.weight_wrapper_patches, model.object_patches):
        if any(_overlaps_target(key, target) for key in patches for target in targets):
            raise ValueError("An existing target weight/object patch conflicts with K Row Balance")
    if model.hook_patches or model.current_hooks is not None or model.forced_hooks is not None:
        raise ValueError("Existing model hooks are not qualified for K Row Balance")
    if (any(model.injections.values()) or any(model.wrappers.values())
            or options.get("patches_replace") or "optimized_attention_override" in options
            or any(key in options for key in ("moss_afm_config", "moss_residual_config", CONFIG_KEY))):
        raise ValueError("Test K Row Balance before other model patches/wrappers")
    labels = " ".join(str(key).lower() for key in options)
    if (any(label in labels for label in ("cache", "compile", "multigpu"))
            or model.is_multigpu_base_clone or "multigpu" in model.additional_models):
        raise ValueError("Compile/cache/multi-GPU models are not qualified for K Row Balance")


def _attach_replacements(model, replacements, report):
    clone = model.clone()
    for target, replacement in replacements.items():
        clone.add_object_patch(target, replacement)
    token = uuid.uuid4()
    # partially_load compares this UUID; object replacement alone does not change it.
    clone.patches_uuid = token
    # clone_has_same_weights otherwise ignores object_patches when patches is empty.
    clone.set_attachments(ATTACHMENT_PREFIX + token.hex, dict(report))
    clone.model_options.setdefault("transformer_options", {})[CONFIG_KEY] = {
        "block_indices": report.get("block_indices", [report["block_index"]]),
        "strength": report["strength"], "max_change": report["max_change"]}
    return clone


def install_k_balance(model, config: KBalanceConfig, blocks=DEFAULT_BLOCKS):
    if not isinstance(config, KBalanceConfig):
        raise ValueError("Expected KBalanceConfig")
    selected = parse_k_blocks(blocks)
    input_type = _patcher_type(model)
    if config.strength == 0:
        return model.clone(), {"status": "bypass", "block_index": selected[0] if len(selected) == 1 else None,
                               "block_indices": list(selected), "implementation_version": 2,
                               "strength": float(config.strength),
                               "max_change": float(config.max_change), "changed_rows": 0,
                               "input_patcher_type": input_type, "effective_patcher_type": input_type,
                               "dynamic_delegate_used": False,
                               "checkpoint_written": False, "image_quality_effect": "unverified"}
    from comfy.model_patcher import ModelPatcher
    from .adapter import detect_native_moss

    if not isinstance(model, ModelPatcher):
        raise ValueError(f"Unsupported K Row Balance patcher: {input_type}")
    targets = [target_module_path(index) for index in selected]
    _reject_conflicts(model, targets)
    working, delegated = _resolve_patcher(model)
    _reject_conflicts(working, targets)
    caps = detect_native_moss(working)
    if (caps.blocks, caps.heads, caps.head_dim, caps.latent_channels) != (42, 24, 128, 128):
        raise ValueError("K Row Balance requires native 42-block, 3072-dim/24-head MOSS geometry")
    replacements, reports = {}, []
    retained, peak = 0, 0
    for index, target in zip(selected, targets):
        try:
            module = working.get_model_object(target)
        except AttributeError as error:
            raise ValueError(f"Block {index}: K Row Balance requires fused QKV") from error
        if (module.in_features != 3072 or module.out_features != 9216
                or getattr(getattr(module.weight, "_params", None), "convrot_groupsize", None) != 256):
            raise ValueError(f"Block {index}: requires canonical fused 3072-dim QKV with group256")
        try:
            replacement, row = balance_projection(module, config, caps.head_dim,
                                                   working.offload_device, block_index=index)
        except (ValueError, FloatingPointError) as error:
            raise type(error)(f"Block {index}: {error}") from error
        replacements[target] = replacement
        reports.append(row)
        peak = max(peak, retained + row["estimated_peak_working_bytes"])
        retained += row["additional_projection_storage_bytes"]
    report = dict(reports[0]) if len(reports) == 1 else {
        "status": "rebuild_control" if config.bypass else "edited", "block_index": None,
        "scope": "K_weight_row_scale_only", "strength": float(config.strength),
        "max_change": float(config.max_change), "checkpoint_written": False,
        "image_quality_effect": "unverified",
    }
    clipped = sum(row["clipped_rows"] for row in reports)
    report.update({
        "implementation_version": 2, "block_indices": list(selected), "block_reports": reports,
        "input_patcher_type": input_type, "effective_patcher_type": _patcher_type(working),
        "dynamic_delegate_used": delegated,
        "changed_rows": sum(row["changed_rows"] for row in reports),
        "clipped_rows": clipped, "clipped_fraction": clipped / sum(row["rows"] for row in reports),
        "gain_min": min(row["gain_min"] for row in reports),
        "gain_max": max(row["gain_max"] for row in reports),
        "additional_projection_storage_bytes": retained, "estimated_peak_working_bytes": peak,
        "memory_estimate_scope": "replacement construction only; excludes native delegate/model/forward workspace",
        "quantized_codes_unchanged": all(row["quantized_codes_unchanged"] for row in reports),
        "qv_scales_unchanged": all(row["qv_scales_unchanged"] for row in reports),
    })
    return _attach_replacements(working, replacements, report), report
