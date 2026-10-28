from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]


class ReleaseTests(unittest.TestCase):
    def test_release_is_reproducible_and_contains_current_sources(self):
        script = ROOT / "tools/build_release.py"
        self.assertTrue(script.exists(), "Missing reproducible release builder")
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("first.zip", "second.zip")]
            for path in paths:
                subprocess.run([sys.executable, str(script), "--output", str(path)], check=True)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            with zipfile.ZipFile(paths[0]) as archive:
                for name in ("nodes.py", "afm/config.py", "tools/smoke_contract.py",
                             "afm/residual.py", "examples/moss_residual_afm_api.json",
                             "examples/moss_lora_residual_afm_api.json",
                             "examples/moss_afm_edit_api.json", "README.md"):
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
                self.assertFalse(any("__pycache__" in n or n.endswith('.safetensors')
                                     or n.startswith((".git/", "dist/", "test-results/", "参考/"))
                                     for n in archive.namelist()))
