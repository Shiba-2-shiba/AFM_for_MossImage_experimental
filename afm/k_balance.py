"""Reversible block-41 K-row scale edits for canonical native INT8 MOSS."""

import json
import uuid

import torch

from .k_balance_math import KBalanceConfig, compute_k_gains


TARGET_MODULE = "diffusion_model.layers.41.proj.qkv"
TARGET_WEIGHT = TARGET_MODULE + ".weight"
CONFIG_KEY = "moss_k_row_balance"
ATTACHMENT_PREFIX = "moss_k_row_balance:"


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


def balance_projection(module, config: KBalanceConfig, head_dim: int, device):
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
        "block_index": 41, "scope": "K_weight_row_scale_only",
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


def _overlaps_target(path):
    return (path == TARGET_MODULE or path.startswith(TARGET_MODULE + ".")
            or TARGET_MODULE.startswith(path + "."))


def _reject_conflicts(model):
    options = model.model_options.get("transformer_options", {})
    if any(key.startswith(ATTACHMENT_PREFIX) for key in model.attachments):
        raise ValueError("K Row Balance is already applied; use the original input MODEL")
    for patches in (model.patches, model.weight_wrapper_patches, model.object_patches):
        if any(_overlaps_target(key) for key in patches):
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


def _attach_replacement(model, replacement, report):
    clone = model.clone()
    clone.add_object_patch(TARGET_MODULE, replacement)
    token = uuid.uuid4()
    # partially_load compares this UUID; object replacement alone does not change it.
    clone.patches_uuid = token
    # clone_has_same_weights otherwise ignores object_patches when patches is empty.
    clone.set_attachments(ATTACHMENT_PREFIX + token.hex, dict(report))
    clone.model_options.setdefault("transformer_options", {})[CONFIG_KEY] = {
        "block_index": 41, "strength": report["strength"], "max_change": report["max_change"]}
    return clone


def install_k_balance(model, config: KBalanceConfig):
    if not isinstance(config, KBalanceConfig):
        raise ValueError("Expected KBalanceConfig")
    if config.strength == 0:
        return model.clone(), {"status": "bypass", "block_index": 41,
                               "strength": float(config.strength),
                               "max_change": float(config.max_change), "changed_rows": 0,
                               "checkpoint_written": False, "image_quality_effect": "unverified"}
    from comfy.model_patcher import ModelPatcher
    from .adapter import detect_native_moss

    if type(model) is not ModelPatcher or model.is_dynamic():
        raise ValueError("K Row Balance currently requires the standard ModelPatcher")
    _reject_conflicts(model)
    caps = detect_native_moss(model)
    if (caps.blocks, caps.heads, caps.head_dim, caps.latent_channels) != (42, 24, 128, 128):
        raise ValueError("K Row Balance requires native 42-block, 3072-dim/24-head MOSS geometry")
    try:
        module = model.get_model_object(TARGET_MODULE)
    except AttributeError as error:
        raise ValueError("K Row Balance requires fused QKV in native block 41") from error
    if (module.in_features != 3072 or module.out_features != 9216
            or getattr(getattr(module.weight, "_params", None), "convrot_groupsize", None) != 256):
        raise ValueError("K Row Balance requires canonical fused 3072-dim QKV with group256")
    replacement, report = balance_projection(module, config, caps.head_dim, model.offload_device)
    return _attach_replacement(model, replacement, report), report
