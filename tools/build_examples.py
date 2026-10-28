"""Create API variants from the user's unchanged native MOSS workflow."""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afm.config import AFMConfig


def add_node(workflow, kind, inputs):
    node_id = str(max(int(key) for key in workflow) + 1)
    workflow[node_id] = {"class_type": kind, "inputs": inputs}
    return [node_id, 0]

parser = argparse.ArgumentParser()
parser.add_argument("--workflow", required=True)
parser.add_argument("--output", default="examples")
parser.add_argument("--model-lora", nargs=2, action="append", default=[],
                    metavar=("NAME", "STRENGTH"))
opts = parser.parse_args()
loras = []
for name, text in opts.model_lora:
    strength = float(text)
    if not name.strip() or not math.isfinite(strength):
        parser.error("LoRA requires a filename and a finite strength")
    loras.append((name, strength))
original = json.loads(Path(opts.workflow).read_text(encoding="utf-8-sig"))
assert original["31"]["inputs"]["model"] == ["28", 0]
assert "38" not in original
out = Path(opts.output)
out.mkdir(parents=True, exist_ok=True)
for variant in ["baseline", "observe", "edit", "two_intervals"]:
    mode = "edit" if variant == "two_intervals" else variant
    workflow = copy.deepcopy(original)
    workflow["9"]["inputs"]["filename_prefix"] = "moss_afm/" + variant
    if mode != "baseline":
        workflow["38"] = {
            "class_type": "MossImageAFM",
            "inputs": {"model": ["28", 0], "mode": mode, "strength": 0.05,
                       "block_scope": "0-2", "branch_mode": "positive_only"},
            "_meta": {"title": "MOSS Image AFM"}}
        workflow["31"]["inputs"]["model"] = ["38", 0]
        if variant == "two_intervals":
            workflow["38"]["inputs"].update(
                block_scope="30-32", start_percent=0.0, end_percent=0.5,
                cutoff=0.25, soft_mask=True, mask_width=0.05, preserve_dc=True,
                head_chunk_size=1, query_chunk_size=128, max_workspace_mib=512.0,
                fail_mode="raise", logging="summary", two_interval_mode=True,
                second_block_scope="0-2", second_start_percent=0.5, second_end_percent=1.0)
    (out / ("moss_afm_" + variant + "_api.json")).write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

for with_lora in ([False, True] if loras else [False]):
    workflow = copy.deepcopy(original)
    link = ["28", 0]
    if with_lora:
        for name, strength in loras:
            link = add_node(workflow, "LoraLoaderModelOnly", {"model": link, "lora_name": name,
                                                   "strength_model": strength})
    link = add_node(workflow, "MossImageResidual", {"model": link, "mid_multiplier": 0.75,
                                          "late_multiplier": 1.15})
    link = add_node(workflow, "MossImageAFM", {"model": link,
                    **asdict(AFMConfig(mode="edit", block_scope="24-26"))})
    workflow["31"]["inputs"]["model"] = link
    name = "moss_lora_residual_afm" if with_lora else "moss_residual_afm"
    workflow["9"]["inputs"]["filename_prefix"] = "moss_afm/" + name
    (out / (name + "_api.json")).write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
