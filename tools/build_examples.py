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
for mode in ["baseline", "observe", "edit"]:
    workflow = copy.deepcopy(original)
    workflow["9"]["inputs"]["filename_prefix"] = "moss_afm/" + mode
    if mode != "baseline":
        workflow["38"] = {
            "class_type": "MossImageAFM",
            "inputs": {"model": ["28", 0], "mode": mode, "strength": 0.05,
                       "block_scope": "0-2", "branch_mode": "positive_only"},
            "_meta": {"title": "MOSS Image AFM"}}
        workflow["31"]["inputs"]["model"] = ["38", 0]
    (out / ("moss_afm_" + mode + "_api.json")).write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
