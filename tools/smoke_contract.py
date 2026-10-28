"""Import-safe contracts for the separate-process native smoke helper."""
from dataclasses import dataclass, fields, replace
import math
import logging

import torch

from afm.config import AFMConfig
from afm.residual import ResidualConfig, install_residual
from afm.adapter import install_afm


@dataclass(frozen=True)
class ModelOnlyLoraSpec:
    lora_name: str
    strength_model: float

    def __post_init__(self):
        if not isinstance(self.lora_name, str) or not self.lora_name.strip():
            raise ValueError("Expected a literal LoRA filename")
        if (isinstance(self.strength_model, bool)
                or not isinstance(self.strength_model, (int, float))
                or not math.isfinite(self.strength_model)):
            raise ValueError("Expected a finite literal LoRA strength")


@dataclass(frozen=True)
class ModelPipeline:
    loras: tuple[ModelOnlyLoraSpec, ...]
    residual: ResidualConfig | None
    afm: AFMConfig


def _literal_config(config_type, values):
    allowed = {f.name for f in fields(config_type)}
    if values.keys() - allowed or any(isinstance(v, (list, dict)) for v in values.values()):
        raise ValueError("Unsupported or linked model configuration input")
    try:
        return config_type(**values)
    except TypeError as exc:
        raise ValueError("Missing model configuration input") from exc


def workflow_pipeline(workflow) -> ModelPipeline:
    """Parse only the documented MODEL path into KSampler 31, rooted at loader 28."""
    link = workflow.get("31", {}).get("inputs", {}).get("model")
    loras, residual, config = [], None, AFMConfig()
    seen, previous = set(), 4
    ranks = {"MossImageAFM": 3, "MossImageResidual": 2, "LoraLoaderModelOnly": 1}
    while True:
        if (not isinstance(link, list) or len(link) != 2
                or type(link[1]) is not int or link[1] != 0
                or type(link[0]) not in (str, int)):
            raise ValueError("Expected a MODEL link with output index 0")
        node_id = str(link[0])
        if node_id in seen or node_id not in workflow:
            raise ValueError("Cyclic or missing node in smoke MODEL path")
        seen.add(node_id)
        node = workflow[node_id]
        kind = node.get("class_type")
        if node_id == "28" and kind == "UNETLoader":
            break
        if kind not in ranks:
            raise ValueError("Smoke supports UNETLoader -> LoraLoaderModelOnly -> Residual -> AFM; "
                             "use the ComfyUI API/UI for CLIP-changing LoraLoader workflows")
        rank = ranks[kind]
        if rank > previous or (rank == previous and rank != 1):
            raise ValueError("Duplicate or incorrectly ordered smoke MODEL patch")
        previous = rank
        values = dict(node.get("inputs", {}))
        link = values.pop("model", None)
        if kind == "MossImageAFM":
            config = _literal_config(AFMConfig, values)
        elif kind == "MossImageResidual":
            residual = _literal_config(ResidualConfig, values)
        else:
            loras.append(_literal_config(ModelOnlyLoraSpec, values))
    if config.block_scope not in ("0-2", "24-26", "30-32"):
        raise ValueError("block_scope must match a current node preset")
    if config.two_interval_mode and config.second_block_scope not in ("0-2", "24-26", "30-32"):
        raise ValueError("second_block_scope must match a current node preset")
    return ModelPipeline(tuple(reversed(loras)), residual, config)


def workflow_config(workflow):
    return workflow_pipeline(workflow).afm


def prepare_model_pipeline(model, pipeline: ModelPipeline, load_lora):
    for spec in pipeline.loras:
        model = load_lora(model, spec)
    return install_residual(model, pipeline.residual) if pipeline.residual is not None else model


def load_lora_checked(model, spec: ModelOnlyLoraSpec, loader):
    """Require complete native mapping for smoke runs, including successive LoRAs."""
    unmatched = []

    class UnmatchedKeys(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.lower().startswith(("lora key not loaded:", "not loaded ")):
                unmatched.append(message)

    before = {key: len(patches) for key, patches in model.patches.items()}
    handler = UnmatchedKeys(logging.WARNING)
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        loaded = loader(model, spec)
    finally:
        logger.removeHandler(handler)
        handler.close()
    if unmatched:
        raise ValueError(f"LoRA {spec.lora_name} has unapplied keys: " + "; ".join(unmatched))
    applied = sorted(key for key, patches in loaded.patches.items()
                     if len(patches) > before.get(key, 0))
    if spec.strength_model != 0 and not applied:
        raise ValueError(f"LoRA {spec.lora_name} did not apply any model patches")
    return loaded, {"applied_keys": applied, "skipped_zero_strength": spec.strength_model == 0}


def load_model_only_lora(model, spec: ModelOnlyLoraSpec):
    import nodes
    loader = nodes.LoraLoaderModelOnly()
    return getattr(loader, loader.FUNCTION)(model=model, lora_name=spec.lora_name,
                                           strength_model=spec.strength_model)[0]


def model_for_mode(model, config: AFMConfig, mode):
    selected = mode_config(config, mode)
    return model if mode == "baseline" else install_afm(model, selected)


def mode_config(config, mode):
    if mode == "zero":
        return replace(config, mode="edit", strength=0)
    if mode not in ("baseline", "off", "observe", "edit"):
        raise ValueError("Unsupported smoke mode")
    return replace(config, mode="off" if mode == "baseline" else mode)


def parse_modes(text):
    modes = tuple(part.strip() for part in text.split(","))
    allowed = {"baseline", "off", "zero", "observe", "edit"}
    if (not modes or modes[0] != "baseline" or len(set(modes)) != len(modes)
            or set(modes) - allowed):
        raise ValueError("Smoke modes must be unique and start with baseline")
    return modes


def compare_latents(mode, values, baseline, *, require_equal=False):
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite sampled latent")
    if baseline is None or values.shape != baseline.shape or values.dtype != baseline.dtype:
        raise ValueError("A same-shape/dtype baseline is required for comparison")
    equal = torch.equal(values, baseline)
    if (require_equal or mode in ("off", "zero", "observe")) and not equal:
        raise RuntimeError(f"{mode} must be bitwise equal to baseline")
    return {"max_abs_vs_baseline": float((values - baseline).abs().max()),
            "equal_to_baseline": equal}
