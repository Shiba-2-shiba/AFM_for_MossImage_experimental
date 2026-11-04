import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.build_k_balance_examples import build_variants, write_examples


ROOT = Path(__file__).resolve().parents[2]


def api_workflow():
    return {
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "model.safetensors"}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "public prompt"}},
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0], "seed": 123, "steps": 25, "cfg": 3.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "positive": ["11", 0],
            },
        },
        "13": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
        "14": {
            "class_type": "SaveImage",
            "inputs": {"images": ["15", 0], "filename_prefix": "original"},
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["13", 0]}},
    }


def ui_workflow():
    return {
        "last_node_id": 14,
        "last_link_id": 5,
        "nodes": [
            {
                "id": 10, "type": "UNETLoader", "pos": [0, 0], "size": [300, 200],
                "flags": {}, "order": 0, "mode": 0, "inputs": [],
                "outputs": [{"name": "MODEL", "type": "MODEL", "links": [5], "slot_index": 0}],
                "properties": {"Node name for S&R": "UNETLoader"},
                "widgets_values": ["model.safetensors", "default"],
            },
            {
                "id": 12, "type": "KSampler", "pos": [700, 0], "size": [300, 200],
                "flags": {}, "order": 1, "mode": 0,
                "inputs": [{"name": "model", "type": "MODEL", "link": 5}],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": None, "slot_index": 0}],
                "properties": {"Node name for S&R": "KSampler"},
                "widgets_values": [123, "fixed", 25, 3.0, "euler", "simple", 1.0],
            },
            {
                "id": 14, "type": "SaveImage", "pos": [1100, 0], "size": [300, 200],
                "flags": {}, "order": 2, "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "outputs": [], "properties": {"Node name for S&R": "SaveImage"},
                "widgets_values": ["original"],
            },
        ],
        "links": [[5, 10, 0, 12, 0, "MODEL"]],
        "groups": [], "config": {}, "extra": {"workflow_description": "original"}, "version": 0.4,
    }


class BuildKBalanceExamplesTests(unittest.TestCase):
    def test_api_variants_only_change_requested_fields(self):
        source = api_workflow()
        untouched = copy.deepcopy(source)
        variants = build_variants(source)
        self.assertEqual(
            set(variants), {"identity", "rebuild_control", "compress", "reverse"}
        )
        self.assertEqual(source, untouched)
        settings = {
            "identity": (0.0, 0.03),
            "rebuild_control": (1.0, 0.0),
            "compress": (1.0, 0.03),
            "reverse": (-1.0, 0.03),
        }
        for name, generated in variants.items():
            self.assertEqual(generated["16"]["class_type"], "MossImageKRowBalance")
            self.assertEqual(generated["16"]["inputs"], {
                "model": ["10", 0],
                "strength": settings[name][0],
                "max_change": settings[name][1],
                "blocks": "41",
            })
            self.assertEqual(generated["12"]["inputs"]["model"], ["16", 0])
            for node_id in ("10", "11", "13", "15"):
                self.assertEqual(generated[node_id], source[node_id])
            for key, value in source["12"]["inputs"].items():
                if key != "model":
                    self.assertEqual(generated["12"]["inputs"][key], value)
            self.assertEqual(generated["14"]["inputs"]["images"], source["14"]["inputs"]["images"])

    def test_optional_baseline_is_an_independent_copy(self):
        source = api_workflow()
        variants = build_variants(source, include_baseline=True)
        self.assertEqual(
            set(variants),
            {"baseline", "identity", "rebuild_control", "compress", "reverse"},
        )
        self.assertNotIn("16", variants["baseline"])
        variants["baseline"]["10"]["inputs"]["unet_name"] = "changed"
        self.assertEqual(source["10"]["inputs"]["unet_name"], "model.safetensors")

    def test_ui_rewires_both_endpoints_with_fresh_links(self):
        source = ui_workflow()
        original = copy.deepcopy(source)
        generated = build_variants(source)["compress"]
        self.assertEqual(source, original)
        self.assertEqual((generated["last_node_id"], generated["last_link_id"]), (15, 7))
        node = next(node for node in generated["nodes"] if node["id"] == 15)
        self.assertEqual(node["type"], "MossImageKRowBalance")
        self.assertEqual(node["mode"], 0)
        self.assertEqual(node["widgets_values"], [1.0, 0.03, "41"])
        self.assertEqual(node["inputs"], [{"name": "model", "type": "MODEL", "link": 6}])
        self.assertEqual(node["outputs"], [
            {"name": "MODEL", "type": "MODEL", "links": [7], "slot_index": 0},
            {"name": "report", "type": "STRING", "links": None, "slot_index": 1},
        ])
        loader = next(node for node in generated["nodes"] if node["id"] == 10)
        sampler = next(node for node in generated["nodes"] if node["id"] == 12)
        self.assertEqual(loader["outputs"][0]["links"], [6])
        self.assertEqual(sampler["inputs"][0]["link"], 7)
        self.assertNotIn(5, [link[0] for link in generated["links"]])
        self.assertIn([6, 10, 0, 15, 0, "MODEL"], generated["links"])
        self.assertIn([7, 15, 0, 12, 0, "MODEL"], generated["links"])
        endpoint_ids = {link[0] for link in generated["links"]}
        self.assertTrue({6, 7}.issubset(endpoint_ids))
        sampler_widgets = next(
            node for node in generated["nodes"] if node["id"] == 12
        )["widgets_values"]
        self.assertEqual(sampler_widgets, original["nodes"][1]["widgets_values"])

    def test_ui_rebuild_control_uses_nonzero_strength_and_zero_cap(self):
        generated = build_variants(ui_workflow())["rebuild_control"]
        node = next(node for node in generated["nodes"] if node["id"] == 15)
        self.assertEqual(node["widgets_values"], [1.0, 0.0, "41"])

    def test_custom_scope_strength_and_cap_drive_all_variants(self):
        blocks = "0-2,22,24,26,30-31,40-41"
        variants = build_variants(
            api_workflow(), blocks=blocks, strength=7.5, max_change=0.4
        )
        expected = {
            "identity": (0.0, 0.4),
            "rebuild_control": (7.5, 0.0),
            "compress": (7.5, 0.4),
            "reverse": (-7.5, 0.4),
        }
        for name, generated in variants.items():
            inputs = generated["16"]["inputs"]
            self.assertEqual(inputs["blocks"], blocks)
            self.assertEqual((inputs["strength"], inputs["max_change"]), expected[name])

    def test_rejects_invalid_custom_settings_before_building(self):
        cases = (
            {"blocks": "42"},
            {"blocks": "2-1"},
            {"strength": -1.0},
            {"strength": 10.1},
            {"max_change": -0.01},
            {"max_change": 0.951},
        )
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    build_variants(api_workflow(), **options)

    def test_rejects_malformed_or_ambiguous_workflows(self):
        cases = []
        malformed = api_workflow()
        malformed["12"]["inputs"]["model"] = "10"
        cases.append(malformed)
        no_sampler = api_workflow()
        del no_sampler["12"]
        cases.append(no_sampler)
        multiple = api_workflow()
        multiple["17"] = copy.deepcopy(multiple["12"])
        cases.append(multiple)
        indirect = api_workflow()
        indirect["16"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10", 0]}}
        indirect["12"]["inputs"]["model"] = ["16", 0]
        cases.append(indirect)
        existing = api_workflow()
        existing["16"] = {"class_type": "MossImageKRowBalance", "inputs": {"model": ["10", 0]}}
        existing["12"]["inputs"]["model"] = ["16", 0]
        cases.append(existing)
        for workflow in cases:
            with self.subTest(workflow=workflow):
                with self.assertRaises(ValueError):
                    build_variants(workflow)

    def test_ui_ignores_bypassed_sampler_but_rejects_other_active_sampler(self):
        source = ui_workflow()
        bypassed = copy.deepcopy(source["nodes"][1])
        bypassed.update(id=20, mode=4)
        source["nodes"].append(bypassed)
        self.assertIn("identity", build_variants(source))
        source["nodes"][-1]["mode"] = 0
        with self.assertRaises(ValueError):
            build_variants(source)

    def test_write_examples_refuses_existing_output_before_writing(self):
        source = api_workflow()
        payload = json.dumps(source).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_examples(source, output, source_name="source.json", source_bytes=payload)
            self.assertEqual(list(output.iterdir()), [marker])

    def test_write_examples_creates_hash_manifest_and_cli_requires_output(self):
        source = api_workflow()
        source_bytes = json.dumps(source, sort_keys=True).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated"
            manifest = write_examples(
                source, output, source_name="fixture.json", source_bytes=source_bytes
            )
            self.assertEqual(
                manifest["source"]["sha256"], hashlib.sha256(source_bytes).hexdigest()
            )
            self.assertEqual(manifest["scope"], "fixed block 41 INT8 K-scale correction")
            self.assertEqual(manifest["settings"], {
                "blocks": "41", "strength": 1.0, "max_change": 0.03,
            })
            self.assertEqual(manifest["quality"], "actual quality unverified")
            self.assertEqual(
                set(manifest["variants"]),
                {"identity", "rebuild_control", "compress", "reverse"},
            )
            self.assertEqual(
                manifest["variants"]["rebuild_control"],
                {
                    "file": "moss_k_balance_rebuild_control_api.json",
                    "strength": 1.0,
                    "max_change": 0.0,
                    "blocks": "41",
                },
            )
            for filename, digest in manifest["files"].items():
                actual_digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
                self.assertEqual(actual_digest, digest)
            recorded = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded, manifest)

            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/build_k_balance_examples.py"),
                 "--workflow", str(output / "moss_k_balance_identity_api.json")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--output", result.stderr)

    def test_write_examples_and_cli_record_requested_settings(self):
        source = api_workflow()
        source_bytes = json.dumps(source).encode("utf-8")
        blocks = "0,1,2,22,24,26,30,31,40,41"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow_path = root / "source.json"
            workflow_path.write_bytes(source_bytes)
            output = root / "generated"
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "tools/build_k_balance_examples.py"),
                    "--workflow", str(workflow_path), "--output", str(output),
                    "--blocks", blocks, "--strength", "8", "--max-change", "0.5",
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["settings"], {
                "blocks": blocks, "strength": 8.0, "max_change": 0.5,
            })
            self.assertEqual(manifest["scope"], f"selected blocks {blocks} INT8 K-scale correction")
            generated = json.loads(
                (output / "moss_k_balance_reverse_api.json").read_text(encoding="utf-8")
            )
            self.assertEqual(generated["16"]["inputs"], {
                "model": ["10", 0], "strength": -8.0,
                "max_change": 0.5, "blocks": blocks,
            })


if __name__ == "__main__":
    unittest.main()
