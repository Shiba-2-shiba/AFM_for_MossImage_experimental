"""Build controlled MossImage K-row balance workflow examples.

This module is intentionally standalone: it uses only the Python standard library
and can transform either ComfyUI API JSON or ComfyUI UI workflow JSON (version 0.4).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afm.config import parse_block_scope


NODE_TYPE = "MossImageKRowBalance"
DISPLAY_NAME = "MOSS Image K Row Balance (Experimental)"
SCOPE = "fixed block 41 INT8 K-scale correction"
MAX_CHANGE = 0.03
BLOCK_COUNT = 42
STRENGTH_LIMIT = 10.0
MAX_CHANGE_LIMIT = 0.95


def _settings(blocks: str, strength: float, max_change: float) -> tuple[str, float, float]:
    if not isinstance(blocks, str):
        raise ValueError("blocks must be a string")
    parse_block_scope(blocks, BLOCK_COUNT)
    for name, value in (("strength", strength), ("max_change", max_change)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= strength <= STRENGTH_LIMIT:
        raise ValueError(f"strength magnitude must be in [0, {STRENGTH_LIMIT:g}]")
    if not 0 <= max_change <= MAX_CHANGE_LIMIT:
        raise ValueError(f"max_change must be in [0, {MAX_CHANGE_LIMIT:g}]")
    return blocks, float(strength), float(max_change)


def _variant_settings(strength: float, max_change: float):
    return (
        ("identity", 0.0, max_change),
        ("rebuild_control", strength, 0.0),
        ("compress", strength, max_change),
        ("reverse", -strength, max_change),
    )


def _scope(blocks: str) -> str:
    if blocks == "41":
        return SCOPE
    return f"selected blocks {blocks} INT8 K-scale correction"


def _format(workflow: dict[str, Any]) -> str:
    if isinstance(workflow.get("nodes"), list) and workflow.get("version") == 0.4:
        return "ui"
    if workflow and all(isinstance(value, dict) and "class_type" in value
                        for value in workflow.values()):
        return "api"
    raise ValueError("workflow must be ComfyUI API JSON or UI workflow JSON version 0.4")


def _next_api_node_id(workflow: dict[str, Any]) -> str:
    try:
        return str(max(int(node_id) for node_id in workflow) + 1)
    except (TypeError, ValueError) as error:
        raise ValueError("API workflow node IDs must be integer strings") from error


def _api_sampler(workflow: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    samplers = [(node_id, node) for node_id, node in workflow.items()
                if node.get("class_type") == "KSampler"]
    if len(samplers) != 1:
        raise ValueError("workflow must contain exactly one KSampler")
    sampler_id, sampler = samplers[0]
    model = sampler.get("inputs", {}).get("model")
    if not (isinstance(model, list) and len(model) == 2
            and isinstance(model[0], (str, int)) and model[1] == 0):
        raise ValueError("KSampler model input must be a slot-0 workflow link")
    source_id = str(model[0])
    source = workflow.get(source_id)
    if not isinstance(source, dict):
        raise ValueError("KSampler model link references a missing node")
    if source.get("class_type") != "UNETLoader":
        raise ValueError(
            "KSampler must link directly from UNETLoader; existing AFM, Residual, "
            "KBalance, and LoRA model paths are not accepted"
        )
    return sampler_id, sampler, source_id


def _set_api_prefixes(workflow: dict[str, Any], variant: str) -> None:
    for node in workflow.values():
        if node.get("class_type") == "SaveImage" and "filename_prefix" in node.get("inputs", {}):
            node["inputs"]["filename_prefix"] = f"moss_k_balance/{variant}"


def _build_api(
    workflow: dict[str, Any], variant: str, strength: float, max_change: float,
    blocks: str,
) -> dict[str, Any]:
    result = copy.deepcopy(workflow)
    sampler_id, sampler, source_id = _api_sampler(result)
    node_id = _next_api_node_id(result)
    result[node_id] = {
        "class_type": NODE_TYPE,
        "inputs": {
            "model": [source_id, 0],
            "strength": strength,
            "max_change": max_change,
            "blocks": blocks,
        },
        "_meta": {
            "title": DISPLAY_NAME,
            "description": f"{_scope(blocks)}; generated comparison variant: {variant}",
        },
    }
    sampler["inputs"]["model"] = [node_id, 0]
    _set_api_prefixes(result, variant)
    return result


def _ui_nodes(workflow: dict[str, Any]) -> dict[int, dict[str, Any]]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("UI workflow nodes must be a list")
    by_id: dict[int, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), int):
            raise ValueError("UI workflow nodes must have integer IDs")
        if node["id"] in by_id:
            raise ValueError("UI workflow node IDs must be unique")
        by_id[node["id"]] = node
    return by_id


def _ui_sampler_path(
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Any], int, int]:
    by_id = _ui_nodes(workflow)
    samplers = [node for node in by_id.values()
                if node.get("type") == "KSampler" and node.get("mode", 0) == 0]
    if len(samplers) != 1:
        raise ValueError("UI workflow must contain exactly one mode-0 KSampler")
    sampler = samplers[0]
    model_inputs = [(index, item) for index, item in enumerate(sampler.get("inputs", []))
                    if item.get("name") == "model"]
    if len(model_inputs) != 1 or not isinstance(model_inputs[0][1].get("link"), int):
        raise ValueError("KSampler must have one linked model input")
    input_slot, model_input = model_inputs[0]
    old_link_id = model_input["link"]
    links = workflow.get("links")
    if not isinstance(links, list):
        raise ValueError("UI workflow links must be a list")
    matching = [link for link in links
                if isinstance(link, list) and len(link) == 6 and link[0] == old_link_id
                and link[3] == sampler["id"] and link[4] == input_slot]
    if len(matching) != 1:
        raise ValueError("KSampler model input link is missing or ambiguous")
    old_link = matching[0]
    source = by_id.get(old_link[1])
    if source is None:
        raise ValueError("KSampler model link references a missing node")
    if source.get("type") != "UNETLoader" or source.get("mode", 0) != 0 or old_link[2] != 0:
        raise ValueError(
            "KSampler must link directly from a mode-0 UNETLoader; existing AFM, "
            "Residual, KBalance, and LoRA model paths are not accepted"
        )
    outputs = source.get("outputs", [])
    if not isinstance(outputs, list) or old_link[2] >= len(outputs):
        raise ValueError("UNETLoader output slot referenced by model link is missing")
    source_output = outputs[old_link[2]]
    if old_link_id not in (source_output.get("links") or []):
        raise ValueError("UNETLoader endpoint does not contain the KSampler model link")
    return sampler, model_input, source, old_link, input_slot, old_link_id


def _next_ui_ids(workflow: dict[str, Any]) -> tuple[int, int, int]:
    nodes = _ui_nodes(workflow)
    links = workflow.get("links", [])
    link_ids = [link[0] for link in links
                if isinstance(link, list) and len(link) == 6 and isinstance(link[0], int)]
    node_id = max([workflow.get("last_node_id", 0), *nodes], default=0) + 1
    first_link = max([workflow.get("last_link_id", 0), *link_ids], default=0) + 1
    return node_id, first_link, first_link + 1


def _set_ui_prefixes(workflow: dict[str, Any], variant: str) -> None:
    for node in workflow["nodes"]:
        if node.get("type") == "SaveImage" and node.get("widgets_values"):
            node["widgets_values"][0] = f"moss_k_balance/{variant}"


def _build_ui(
    workflow: dict[str, Any], variant: str, strength: float, max_change: float,
    blocks: str,
) -> dict[str, Any]:
    result = copy.deepcopy(workflow)
    sampler, model_input, source, old_link, input_slot, old_link_id = _ui_sampler_path(result)
    node_id, input_link_id, output_link_id = _next_ui_ids(result)
    source_pos = source.get("pos", [0, 0])
    sampler_pos = sampler.get("pos", [source_pos[0] + 700, source_pos[1]])
    pos = [(source_pos[0] + sampler_pos[0]) / 2, (source_pos[1] + sampler_pos[1]) / 2]
    order = max((node.get("order", 0) for node in result["nodes"]), default=-1) + 1
    new_node = {
        "id": node_id,
        "type": NODE_TYPE,
        "title": DISPLAY_NAME,
        "pos": pos,
        "size": [350, 180],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": input_link_id}],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": [output_link_id], "slot_index": 0},
            {"name": "report", "type": "STRING", "links": None, "slot_index": 1},
        ],
        "properties": {"Node name for S&R": NODE_TYPE},
        "widgets_values": [strength, max_change, blocks],
    }
    source_links = source["outputs"][old_link[2]]["links"]
    source["outputs"][old_link[2]]["links"] = [
        input_link_id if link_id == old_link_id else link_id for link_id in source_links
    ]
    model_input["link"] = output_link_id
    result["links"] = [link for link in result["links"] if link is not old_link]
    result["links"].extend([
        [input_link_id, source["id"], old_link[2], node_id, 0, "MODEL"],
        [output_link_id, node_id, 0, sampler["id"], input_slot, "MODEL"],
    ])
    result["nodes"].append(new_node)
    result["last_node_id"] = node_id
    result["last_link_id"] = output_link_id
    extra = result.setdefault("extra", {})
    extra["workflow_description"] = (
        f"{_scope(blocks)}; generated {variant} comparison. Actual quality unverified."
    )
    _set_ui_prefixes(result, variant)
    return result


def build_variants(
    workflow: dict[str, Any], *, include_baseline: bool = False, blocks: str = "41",
    strength: float = 1.0, max_change: float = MAX_CHANGE,
) -> dict[str, dict[str, Any]]:
    """Return independent controlled variants without modifying ``workflow``."""
    if not isinstance(workflow, dict):
        raise ValueError("workflow JSON root must be an object")
    blocks, strength, max_change = _settings(blocks, strength, max_change)
    workflow_format = _format(workflow)
    # Validate before producing even the optional baseline copy.
    if workflow_format == "api":
        _api_sampler(workflow)
        build = _build_api
    else:
        _ui_sampler_path(workflow)
        build = _build_ui
    variants: dict[str, dict[str, Any]] = {}
    if include_baseline:
        baseline = copy.deepcopy(workflow)
        if workflow_format == "api":
            _set_api_prefixes(baseline, "baseline")
        else:
            _set_ui_prefixes(baseline, "baseline")
            baseline.setdefault("extra", {})["workflow_description"] = (
                f"Unmodified model path baseline for {_scope(blocks)}. Actual quality unverified."
            )
        variants["baseline"] = baseline
    for name, variant_strength, variant_max_change in _variant_settings(strength, max_change):
        variants[name] = build(
            workflow, name, variant_strength, variant_max_change, blocks
        )
    return variants


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_examples(
    workflow: dict[str, Any],
    output: Path | str,
    *,
    source_name: str,
    source_bytes: bytes,
    include_baseline: bool = False,
    blocks: str = "41",
    strength: float = 1.0,
    max_change: float = MAX_CHANGE,
) -> dict[str, Any]:
    """Write variants and a hash manifest to a new output directory."""
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    workflow_format = _format(workflow)
    blocks, strength, max_change = _settings(blocks, strength, max_change)
    variants = build_variants(
        workflow, include_baseline=include_baseline, blocks=blocks,
        strength=strength, max_change=max_change,
    )
    suffix = "api" if workflow_format == "api" else "ui"
    payloads: dict[str, bytes] = {}
    variant_records: dict[str, dict[str, Any]] = {}
    settings = {name: (variant_strength, variant_max_change)
                for name, variant_strength, variant_max_change
                in _variant_settings(strength, max_change)}
    for name, variant in variants.items():
        filename = f"moss_k_balance_{name}_{suffix}.json"
        payloads[filename] = _json_bytes(variant)
        variant_records[name] = {
            "file": filename,
            "strength": None if name == "baseline" else settings[name][0],
            "max_change": None if name == "baseline" else settings[name][1],
            "blocks": None if name == "baseline" else blocks,
        }
    manifest = {
        "source": {
            "file": source_name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "format": suffix,
        },
        "scope": _scope(blocks),
        "settings": {
            "blocks": blocks, "strength": strength, "max_change": max_change,
        },
        "quality": "actual quality unverified",
        "variants": variant_records,
        "files": {name: hashlib.sha256(payload).hexdigest()
                  for name, payload in payloads.items()},
    }
    output_path.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads.items():
        (output_path / filename).write_bytes(payload)
    (output_path / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--blocks", default="41")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-change", type=float, default=MAX_CHANGE)
    options = parser.parse_args(argv)
    try:
        source_bytes = options.workflow.read_bytes()
        workflow = json.loads(source_bytes.decode("utf-8-sig"))
        write_examples(
            workflow,
            options.output,
            source_name=options.workflow.name,
            source_bytes=source_bytes,
            include_baseline=options.include_baseline,
            blocks=options.blocks,
            strength=options.strength,
            max_change=options.max_change,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
