import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from afm.config import AFMConfig

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ["0-2", "24-26", "30-32"]


class RepositoryContractTests(unittest.TestCase):
    def test_core_and_node_defaults_match_validated_presets(self):
        self.assertEqual(AFMConfig().block_scope, "0-2")
        tree = ast.parse((ROOT / "nodes.py").read_text())
        execute = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "execute")
        defaults = dict(zip([a.arg for a in execute.args.args][-len(execute.args.defaults):],
                            execute.args.defaults))
        self.assertEqual(ast.literal_eval(defaults["block_scope"]), "0-2")
        combo = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and n.args and isinstance(n.args[0], ast.Constant)
                     and n.args[0].value == "block_scope")
        options = {k.arg: k.value for k in combo.keywords}
        self.assertEqual(ast.literal_eval(options["options"]), PRESETS)

    def test_shipped_api_and_ui_use_current_default(self):
        for mode in ("observe", "edit"):
            data = json.loads((ROOT / "examples" / f"moss_afm_{mode}_api.json").read_text())
            self.assertEqual(data["38"]["inputs"]["block_scope"], "0-2")
        data = json.loads((ROOT / "examples/moss_afm_edit_ui.json").read_text())
        node = next(n for n in data["nodes"] if n["type"] == "MossImageAFM")
        self.assertEqual(node["widgets_values"][2], "0-2")

    def test_example_generator_preserves_generation_inputs_and_default(self):
        source = ROOT / "examples/moss_afm_baseline_api.json"
        original = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(ROOT / "tools/build_examples.py"),
                            "--workflow", str(source), "--output", directory], check=True)
            generated = json.loads((Path(directory) / "moss_afm_edit_api.json").read_text())
        self.assertEqual(generated["38"]["inputs"]["block_scope"], "0-2")
        for key in ("28", "29", "30", "32", "35", "36"):
            self.assertEqual(generated[key], original[key])
        for key, value in original["31"]["inputs"].items():
            if key != "model":
                self.assertEqual(generated["31"]["inputs"][key], value)
