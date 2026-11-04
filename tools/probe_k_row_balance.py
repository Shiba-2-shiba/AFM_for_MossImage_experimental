"""CPU-only real-QKV probe; does not load the full model or generate images."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afm.k_balance import balance_projection
from afm.k_balance_math import KBalanceConfig


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(options.comfy_root.resolve()))
    from comfy.cli_args import args
    args.cpu = True
    import comfy.ops

    original_threads = torch.get_num_threads()
    started = time.monotonic()
    result = {"status": "running", "scope": "one native block41 QKV projection on CPU",
              "checkpoint": options.checkpoint.name, "checkpoint_written": False,
              "image_generation": False, "image_quality_effect": "unverified",
              "torch": torch.__version__, "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
              "before_sha256": sha256(options.checkpoint), "variants": {}}
    try:
        torch.set_num_threads(4)
        prefix = "model.diffusion_model.layers.41.proj.qkv"
        with safe_open(options.checkpoint, framework="pt", device="cpu") as tensors:
            metadata = json.loads(tensors.metadata()["_quantization_metadata"])
            conf = metadata["layers"][prefix]
            if conf != {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}:
                raise ValueError("unexpected real-QKV quantization contract")
            codes = tensors.get_tensor(prefix + ".weight")
            scales = tensors.get_tensor(prefix + ".weight_scale")
            if list(codes.shape) != [9216, 3072] or list(scales.shape) != [9216, 1]:
                raise ValueError("unexpected real-QKV geometry")
            operation = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
            original = operation.Linear(3072, 9216, bias=False, device="cpu").eval()
            original.load_state_dict({
                "weight": codes, "weight_scale": scales,
                "comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8),
            })
            probe_input = torch.linspace(-.2, .3, 3072).reshape(1, -1)
            with torch.inference_mode():
                before = original(probe_input)
            result["variants"]["identity"] = {
                "strength": 0, "max_change": .03, "node_behavior": "clone-only bypass",
                "full_model_forward_tested": False,
            }
            for name, strength, cap in (("rebuild_control", 1, 0),
                                        ("compress", 1, .03), ("reverse", -1, .03)):
                edited, report = balance_projection(original, KBalanceConfig(strength, cap),
                                                     128, torch.device("cpu"))
                with torch.inference_mode():
                    after = edited(probe_input)
                report["projection_q_output_equal"] = torch.equal(before[:, :3072], after[:, :3072])
                report["projection_v_output_equal"] = torch.equal(before[:, 6144:], after[:, 6144:])
                report["projection_k_changed_elements"] = int(torch.count_nonzero(
                    before[:, 3072:6144] != after[:, 3072:6144]).item())
                if not report["projection_q_output_equal"] or not report["projection_v_output_equal"]:
                    raise RuntimeError("Q/V output changed in CPU projection probe")
                if cap == 0 and not torch.equal(before, after):
                    raise RuntimeError("Reconstruction control changed CPU projection output")
                if not torch.equal(original.state_dict()["weight_scale"], scales):
                    raise RuntimeError("original in-memory scale changed")
                result["variants"][name] = report
                del edited
        result["status"] = "cpu_projection_probe_complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["after_sha256"] = sha256(options.checkpoint)
        result["checkpoint_unchanged"] = result["before_sha256"] == result["after_sha256"]
        if not result["checkpoint_unchanged"]:
            result["status"] = "integrity_failure"
        result["elapsed_seconds"] = time.monotonic() - started
        result["source_sha256"] = {
            "afm/k_balance.py": sha256(ROOT / "afm/k_balance.py"),
            "afm/k_balance_math.py": sha256(ROOT / "afm/k_balance_math.py"),
            "comfy/ops.py": sha256(options.comfy_root / "comfy/ops.py"),
            "comfy/model_patcher.py": sha256(options.comfy_root / "comfy/model_patcher.py"),
        }
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        temporary = options.output / "probe.json.tmp"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(options.output / "probe.json")
        torch.set_num_threads(original_threads)
    if not result["checkpoint_unchanged"]:
        raise RuntimeError("checkpoint integrity failure")
    print(json.dumps({"status": result["status"], "checkpoint_unchanged": True,
                      "seconds": result["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
