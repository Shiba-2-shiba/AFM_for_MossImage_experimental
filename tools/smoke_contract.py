"""Import-safe contracts for the separate-process native smoke helper."""
from dataclasses import fields, replace

import torch

from afm.config import AFMConfig


def workflow_config(workflow):
    """Read the AFM node feeding the supplied workflow's KSampler (node 31).

    The supplied baseline graph has no AFM node and uses current defaults.
    Linked/dynamic widget inputs are intentionally unsupported by this helper.
    """
    link = workflow["31"]["inputs"]["model"]
    if not isinstance(link, list) or len(link) != 2 or link[1] != 0:
        raise ValueError("Expected KSampler MODEL link with output index 0")
    if str(link[0]) == "28":
        return AFMConfig()
    node = workflow.get(str(link[0]), {})
    if node.get("class_type") != "MossImageAFM":
        raise ValueError("Smoke helper supports direct loader or MOSS Image AFM only")
    values = dict(node.get("inputs", {}))
    if values.pop("model", None) != ["28", 0]:
        raise ValueError("MOSS Image AFM must use the supplied native loader")
    allowed = {f.name for f in fields(AFMConfig)}
    if values.keys() - allowed or any(isinstance(v, (list, dict)) for v in values.values()):
        raise ValueError("Unsupported or linked AFM configuration input")
    config = AFMConfig(**values)
    if config.block_scope not in ("0-2", "24-26", "30-32"):
        raise ValueError("block_scope must match a current node preset")
    return config


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
