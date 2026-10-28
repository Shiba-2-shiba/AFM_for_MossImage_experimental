"""Run native MOSS inference in a separate process without restarting the active server."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--comfy-root", required=True)
parser.add_argument("--workflow", required=True)
parser.add_argument("--model-root", default="D:/StabilityMatrix-win-x64/Data/Models")
parser.add_argument("--extra-model-paths-config", action="append", default=[])
parser.add_argument("--output", required=True)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--cfg", type=float)
parser.add_argument("--modes", default="baseline,off,zero,observe,edit")
options = parser.parse_args()
sys.argv = [sys.argv[0]]
out = Path(options.output).resolve()
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, options.comfy_root)
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args
args.base_directory = str(out)
args.preview_method = None

import numpy as np
from PIL import Image
import torch
import comfy.sd
import comfy.utils
import comfy.model_management
import folder_paths
import nodes

from tools.smoke_contract import (compare_latents, mode_config, parse_modes,
                                  workflow_pipeline, prepare_model_pipeline, model_for_mode,
                                  load_lora_checked, load_model_only_lora)

from utils.extra_config import load_extra_path_config
default_paths = Path(options.comfy_root) / "extra_model_paths.yaml"
if default_paths.is_file():
    load_extra_path_config(str(default_paths))
for path in options.extra_model_paths_config:
    load_extra_path_config(path)

folder_paths.set_output_directory(str(out))
workflow_path = Path(options.workflow)
workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
inputs = workflow["31"]["inputs"]
pipeline = workflow_pipeline(workflow)
base_config = pipeline.afm
modes = parse_modes(options.modes)
if base_config.logging == "off":
    raise ValueError("Smoke verification requires logging=summary")
model_root = Path(options.model_root)
model_path = model_root / "DiffusionModels" / workflow["28"]["inputs"]["unet_name"]
clip_path = model_root / "TextEncoders" / workflow["30"]["inputs"]["clip_name"]
vae_path = model_root / "VAE" / workflow["32"]["inputs"]["vae_name"]
package = Path(__file__).resolve().parents[1]
source_hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [*sorted((package / "afm").glob("*.py")), package / "nodes.py", package / "__init__.py"]}
report = {"source_sha256": source_hashes, "status": "running", "workflow_sha256": hashlib.sha256(workflow_path.read_bytes()).hexdigest(),
          "model": str(model_path), "width": options.width, "height": options.height,
          "steps": options.steps, "batch": options.batch,
          "cfg": options.cfg if options.cfg is not None else inputs["cfg"],
          "sampler": inputs["sampler_name"], "scheduler": inputs["scheduler"],
          "torch_version": torch.__version__, "hip": torch.version.hip, "runs": [],
          "afm_config": asdict(base_config),
          "model_pipeline": asdict(pipeline), "lora_files": [],
          "scope": "Separate native ComfyUI process, not active server node registration"}

def save_report():
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def load_model_lora(model, spec):
    path = Path(folder_paths.get_full_path_or_raise("loras", spec.lora_name))
    loaded, evidence = load_lora_checked(model, spec, load_model_only_lora)
    stat = path.stat()
    report["lora_files"].append({"name": spec.lora_name, "path": str(path),
                                 "strength_model": spec.strength_model,
                                 "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                 **evidence})
    return loaded

save_report()
try:
    with torch.inference_mode():
        print("Loading text encoder", flush=True)
        clip = comfy.sd.load_clip([str(clip_path)], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(workflow["29"]["inputs"]["text"]))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(workflow["36"]["inputs"]["text"]))
        print("Loading diffusion model", flush=True)
        model = comfy.sd.load_diffusion_model(str(model_path))
        model = prepare_model_pipeline(model, pipeline, load_model_lora)
        print("Loading VAE", flush=True)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(vae_path)))
        baseline = None
        for mode in modes:
            print("RUN " + mode, flush=True)
            config = mode_config(base_config, mode)
            patched = model_for_mode(model, base_config, mode)
            latent = {"samples": torch.zeros(options.batch, 128, options.height // 16,
                                             options.width // 16)}
            before_logs = set((out / "moss_afm").glob("*.jsonl"))
            started = time.perf_counter()
            samples = nodes.common_ksampler(
                patched, inputs["seed"], options.steps, report["cfg"], inputs["sampler_name"],
                inputs["scheduler"], positive, negative, latent, denoise=inputs["denoise"])[0]
            values = samples["samples"].detach().cpu()
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite sampled latent")
            if mode == "baseline":
                baseline = values.clone()
            row = {"mode": mode, "sample_seconds": time.perf_counter() - started,
                   "finite": True, "latent_shape": list(values.shape),
                   "afm_config": asdict(config)}
            row.update(compare_latents(mode, values, baseline, require_equal=config.bypass))
            new_logs = set((out / "moss_afm").glob("*.jsonl")) - before_logs
            if not config.bypass:
                if len(new_logs) != 1:
                    raise RuntimeError("Expected one AFM run log")
                entries = [json.loads(x) for x in next(iter(new_logs)).read_text().splitlines()]
                summary = entries[-1]
                if summary.get("status") != "success" or summary.get("callbacks") != options.steps:
                    raise RuntimeError("AFM did not complete every sampler step")
                if summary.get("fallback_calls") != 0:
                    raise RuntimeError("AFM fallback is not a smoke-test success")
                if mode == "edit" and summary.get("edited_calls", 0) <= 0:
                    raise RuntimeError("AFM did not edit any attention calls")
                row["summary"] = summary
            images = vae.decode(samples["samples"])
            for i, image in enumerate(images):
                Image.fromarray(np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(
                    out / (mode + "_" + str(i) + ".png"))
            torch.save(values, out / (mode + ".pt"))
            report["runs"].append(row)
            save_report()
        report["status"] = "success"
except BaseException as exc:
    report["status"] = "error"
    report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    raise
finally:
    save_report()
