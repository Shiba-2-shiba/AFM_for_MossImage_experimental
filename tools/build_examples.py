"""Create API variants from the user's unchanged native MOSS workflow."""
import argparse
import copy
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--workflow", required=True)
parser.add_argument("--output", default="examples")
opts = parser.parse_args()
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
