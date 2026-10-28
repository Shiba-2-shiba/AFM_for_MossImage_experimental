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
    def test_residual_example_preserves_sampling_and_lora_order(self):
        source = ROOT / 'examples/moss_afm_baseline_api.json'
        original = json.loads(source.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(ROOT / 'tools/build_examples.py'),
                            '--workflow', str(source), '--output', directory,
                            '--model-lora', 'first.safetensors', '0.6',
                            '--model-lora', 'second.safetensors', '-0.2'], check=True)
            simple = json.loads((Path(directory) / 'moss_residual_afm_api.json').read_text(encoding='utf-8'))
            combined = json.loads((Path(directory) / 'moss_lora_residual_afm_api.json').read_text(encoding='utf-8'))
            legacy = json.loads((Path(directory) / 'moss_afm_edit_api.json').read_text(encoding='utf-8'))
        from tools.smoke_contract import workflow_pipeline
        self.assertEqual(legacy['38']['inputs']['model'], ['28', 0])
        self.assertEqual(workflow_pipeline(simple).loras, ())
        pipeline = workflow_pipeline(combined)
        self.assertEqual([s.lora_name for s in pipeline.loras], ['first.safetensors', 'second.safetensors'])
        self.assertEqual(pipeline.afm.block_scope, '24-26')
        self.assertEqual((pipeline.residual.mid_multiplier, pipeline.residual.late_multiplier), (0.75, 1.15))
        for workflow in (simple, combined):
            for key in ('28', '29', '30', '32', '35', '36'):
                self.assertEqual(workflow[key], original[key])
            for key, value in original['31']['inputs'].items():
                if key != 'model':
                    self.assertEqual(workflow['31']['inputs'][key], value)

    def test_two_interval_example_preserves_sampling_inputs(self):
        source = ROOT / "examples/moss_afm_baseline_api.json"
        original = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(ROOT / "tools/build_examples.py"),
                            "--workflow", str(source), "--output", directory], check=True)
            generated = json.loads((Path(directory) / "moss_afm_two_intervals_api.json")
                                   .read_text(encoding="utf-8"))
        settings = generated["38"]["inputs"]
        self.assertEqual((settings["block_scope"], settings["start_percent"],
                          settings["end_percent"], settings["two_interval_mode"],
                          settings["second_block_scope"], settings["second_start_percent"],
                          settings["second_end_percent"]), ("30-32", 0.0, 0.5, True, "0-2", 0.5, 1.0))
        self.assertEqual((settings["mode"], settings["strength"]), ("edit", 0.05))
        self.assertEqual(settings["model"], ["28", 0])
        self.assertEqual(generated["31"]["inputs"]["model"], ["38", 0])
        for key in ("28", "29", "30", "32", "35", "36"):
            self.assertEqual(generated[key], original[key])
        for key, value in original["31"]["inputs"].items():
            if key != "model":
                self.assertEqual(generated["31"]["inputs"][key], value)

    def test_new_inputs_are_optional_and_appended(self):
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MossImageAFM")
        schema = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "define_schema")
        inputs = next(k.value for n in ast.walk(schema) if isinstance(n, ast.Call)
                      for k in n.keywords if k.arg == "inputs")
        names = [ast.literal_eval(n.args[0]) for n in inputs.elts]
        legacy = ["model", "mode", "strength", "block_scope", "branch_mode",
                  "start_percent", "end_percent", "cutoff", "soft_mask", "mask_width",
                  "preserve_dc", "head_chunk_size", "query_chunk_size",
                  "max_workspace_mib", "fail_mode", "logging"]
        added = ["two_interval_mode", "second_block_scope", "second_start_percent",
                 "second_end_percent"]
        self.assertEqual(names, legacy + added)
        for node, default in zip(inputs.elts[len(legacy):], (False, "0-2", 0.5, 1.0)):
            keywords = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            self.assertTrue(keywords["optional"])
            self.assertEqual(keywords["default"], default)
            if ast.literal_eval(node.args[0]) == "second_block_scope":
                self.assertEqual(keywords["options"], PRESETS)
        execute = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == "execute")
        self.assertEqual([a.arg for a in execute.args.args][-4:], added)
        self.assertEqual([ast.literal_eval(n) for n in execute.args.defaults][-4:],
                         [False, "0-2", 0.5, 1.0])

    def test_core_and_node_defaults_match_validated_presets(self):
        self.assertEqual(AFMConfig().block_scope, "0-2")
        tree = ast.parse((ROOT / "nodes.py").read_text())
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MossImageAFM")
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
