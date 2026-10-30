"""Local residual-gate scaling for native, 42-block MOSS models."""
from dataclasses import dataclass
import math

import torch


RESIDUAL_CONFIG_KEY = "moss_residual_config"


def _validate_multiplier(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 2):
        raise ValueError("Residual multipliers must be finite numbers between 0 and 2")


@dataclass(frozen=True)
class ResidualConfig:
    mid_multiplier: float = 1.0
    late_multiplier: float = 1.0

    def __post_init__(self):
        _validate_multiplier(self.mid_multiplier)
        _validate_multiplier(self.late_multiplier)

    @property
    def bypass(self):
        return self.mid_multiplier == self.late_multiplier == 1

    def multiplier_for(self, index: int) -> float:
        if 9 <= index <= 25:
            return self.mid_multiplier
        if 34 <= index <= 41:
            return self.late_multiplier
        return 1.0


@dataclass(frozen=True)
class ResidualTestConfig:
    """Independent multipliers for consecutive groups of three joint blocks."""

    blocks_00_02: float = 1.0
    blocks_03_05: float = 1.0
    blocks_06_08: float = 1.0
    blocks_09_11: float = 1.0
    blocks_12_14: float = 1.0
    blocks_15_17: float = 1.0
    blocks_18_20: float = 1.0
    blocks_21_23: float = 1.0
    blocks_24_26: float = 1.0
    blocks_27_29: float = 1.0
    blocks_30_32: float = 1.0
    blocks_33_35: float = 1.0
    blocks_36_38: float = 1.0
    blocks_39_41: float = 1.0

    def __post_init__(self):
        for value in vars(self).values():
            _validate_multiplier(value)

    @property
    def bypass(self):
        return all(value == 1 for value in vars(self).values())

    def multiplier_for(self, index: int) -> float:
        if 0 <= index < 42:
            start = index // 3 * 3
            return getattr(self, f"blocks_{start:02d}_{start + 2:02d}")
        return 1.0


def scale_residual_mod(mod, multiplier: float) -> tuple[torch.Tensor, ...]:
    _validate_multiplier(multiplier)
    if (not isinstance(mod, (tuple, list)) or len(mod) != 6
            or any(not isinstance(value, torch.Tensor) or value.ndim != 3
                   or value.shape[1] != 1 for value in mod)):
        raise ValueError("Residual requires the native six-tensor MOSS modulation")
    if any((value.shape, value.dtype, value.device) !=
           (mod[0].shape, mod[0].dtype, mod[0].device) for value in mod[1:]):
        raise ValueError("MOSS modulation shapes, dtypes and devices must match")
    return tuple(value * multiplier if i in (2, 5) else value
                 for i, value in enumerate(mod))


@dataclass(frozen=True)
class ResidualBlockPatch:
    block_index: int
    multiplier: float

    def __post_init__(self):
        if type(self.block_index) is not int or not 0 <= self.block_index < 42:
            raise ValueError("Invalid MOSS Residual block index")
        _validate_multiplier(self.multiplier)

    def __call__(self, args: dict, extra: dict) -> dict:
        local = dict(args)
        local["mod"] = scale_residual_mod(args.get("mod"), self.multiplier)
        return extra["original_block"](local)


def install_residual(model, config: ResidualConfig | ResidualTestConfig):
    # Keep the math/config module importable without a ComfyUI runtime.
    from comfy.patcher_extension import WrappersMP, get_all_wrappers
    from .adapter import detect_native_moss
    from .runtime import CONFIG_KEY

    caps = detect_native_moss(model)
    if caps.blocks != 42:
        raise ValueError("MOSS Residual requires exactly 42 native joint blocks")
    if config.bypass:
        return model.clone()
    options = model.model_options.get("transformer_options", {})
    if CONFIG_KEY in options or RESIDUAL_CONFIG_KEY in options:
        raise ValueError("Apply MOSS Residual once, before MOSS AFM")
    if ("optimized_attention_override" in options
            or options.get("patches_replace", {}).get("dit")):
        raise ValueError("Existing block/attention patch conflicts with MOSS Residual")
    if (model.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
            or get_all_wrappers(WrappersMP.DIFFUSION_MODEL, options)):
        raise ValueError("Existing diffusion wrapper is not qualified for MOSS Residual")
    labels = " ".join(str(k).lower() for k in options)
    if (any(key in labels for key in ("cache", "spectrum", "compile", "multigpu"))
            or model.model_options.get("multigpu_thread_pool") is not None):
        raise ValueError("Model skip-cache/compile/multi-GPU is not qualified for Residual")
    clone = model.clone()
    clone.model_options["transformer_options"][RESIDUAL_CONFIG_KEY] = config
    for index in range(caps.blocks):
        multiplier = config.multiplier_for(index)
        if multiplier != 1:
            clone.set_model_patch_replace(ResidualBlockPatch(index, multiplier),
                                          "dit", "single_block", index)
    return clone
