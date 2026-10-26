"""Build a deterministic source archive; never include local validation images."""
import argparse
from pathlib import Path
import zipfile


def build_release(root, output):
    root, output = Path(root), Path(output)
    files = [root / name for name in ("__init__.py", "nodes.py", "pyproject.toml",
                                      "README.md", "NOTICE.md")]
    for directory in ("afm", "tools", "tests", "docs", "examples"):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".md", ".json")
                     and "__pycache__" not in p.parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/AFM_for_MossImage-0.1.1.zip")
    options = parser.parse_args()
    build_release(Path(__file__).resolve().parents[1], options.output)
