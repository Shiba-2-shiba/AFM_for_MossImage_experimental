"""CPU-only real-QKV probe; does not load the full model or generate images."""

import argparse
import gc
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

from afm.k_balance import balance_projection, parse_k_blocks
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
    parser.add_argument("--blocks", default="41")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-change", type=float, default=0.03)
    options = parser.parse_args()
    try:
        selected_blocks = tuple(sorted(parse_k_blocks(options.blocks)))
        requested_config = KBalanceConfig(options.strength, options.max_change)
        if requested_config.strength < 0:
            raise ValueError("strength must be a nonnegative magnitude")
    except ValueError as error:
        parser.error(str(error))
    options.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(options.comfy_root.resolve()))
    from comfy.cli_args import args
    args.cpu = True
    import comfy.ops

    original_threads = torch.get_num_threads()
    started = time.monotonic()
    result = {"status": "running", "scope": "selected native QKV projections on CPU",
              "checkpoint": options.checkpoint.name, "checkpoint_written": False,
              "full_model_loaded": False, "projection_probe_only": True, "device": "cpu",
              "image_generation": False, "image_quality_effect": "unverified",
              "torch": torch.__version__,
              "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
              "before_sha256": sha256(options.checkpoint),
              "requested": {
                  "blocks": options.blocks,
                  "selected_blocks": list(selected_blocks),
                  "strength": float(requested_config.strength),
                  "max_change": float(requested_config.max_change),
                  "strength_limits": [-10.0, 10.0],
                  "max_change_limits": [0.0, 0.95],
              },
              "block_order": list(selected_blocks), "blocks": {}}
    try:
        torch.set_num_threads(4)
        with safe_open(options.checkpoint, framework="pt", device="cpu") as tensors:
            metadata = json.loads(tensors.metadata()["_quantization_metadata"])
            operation = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
            for block_index in selected_blocks:
                prefix = f"model.diffusion_model.layers.{block_index}.proj.qkv"
                conf = metadata["layers"][prefix]
                if conf != {"format": "int8_tensorwise", "convrot": True,
                            "convrot_groupsize": 256}:
                    raise ValueError(
                        f"unexpected real-QKV quantization contract for block {block_index}"
                    )
                codes = tensors.get_tensor(prefix + ".weight")
                scales = tensors.get_tensor(prefix + ".weight_scale")
                if list(codes.shape) != [9216, 3072] or list(scales.shape) != [9216, 1]:
                    raise ValueError(f"unexpected real-QKV geometry for block {block_index}")
                original = operation.Linear(3072, 9216, bias=False, device="cpu").eval()
                original.load_state_dict({
                    "weight": codes, "weight_scale": scales,
                    "comfy_quant": torch.tensor(
                        list(json.dumps(conf).encode()), dtype=torch.uint8
                    ),
                })
                probe_input = torch.linspace(-.2, .3, 3072).reshape(1, -1)
                with torch.inference_mode():
                    before = original(probe_input)
                block_result = {"block_index": block_index, "tensor_prefix": prefix,
                                "variants": {}}
                settings = (
                    ("identity", 0.0, requested_config.max_change),
                    ("rebuild_control", requested_config.strength, 0.0),
                    ("compress", requested_config.strength, requested_config.max_change),
                    ("reverse", -requested_config.strength, requested_config.max_change),
                )
                for name, strength, cap in settings:
                    edited, report = balance_projection(
                        original, KBalanceConfig(strength, cap), 128,
                        torch.device("cpu"), block_index=block_index,
                    )
                    with torch.inference_mode():
                        after = edited(probe_input)
                    report["projection_q_output_equal"] = torch.equal(
                        before[:, :3072], after[:, :3072]
                    )
                    report["projection_v_output_equal"] = torch.equal(
                        before[:, 6144:], after[:, 6144:]
                    )
                    report["projection_k_changed_elements"] = int(torch.count_nonzero(
                        before[:, 3072:6144] != after[:, 3072:6144]
                    ).item())
                    report["full_projection_output_equal"] = torch.equal(before, after)
                    report["full_model_forward_tested"] = False
                    if (not report["projection_q_output_equal"]
                            or not report["projection_v_output_equal"]):
                        raise RuntimeError(f"Q/V output changed in block {block_index} CPU probe")
                    if (name in ("identity", "rebuild_control")
                            and not report["full_projection_output_equal"]):
                        raise RuntimeError(
                            f"{name} changed block {block_index} CPU projection output"
                        )
                    if not torch.equal(original.state_dict()["weight_scale"], scales):
                        raise RuntimeError(f"original block {block_index} in-memory scale changed")
                    block_result["variants"][name] = report
                    del after, edited
                block_result["checks"] = {
                    "qv_output_equal_all_variants": all(
                        report["projection_q_output_equal"]
                        and report["projection_v_output_equal"]
                        for report in block_result["variants"].values()
                    ),
                    "identity_output_equal": block_result["variants"]["identity"][
                        "full_projection_output_equal"
                    ],
                    "rebuild_control_output_equal": block_result["variants"][
                        "rebuild_control"
                    ]["full_projection_output_equal"],
                }
                result["blocks"][str(block_index)] = block_result
                del before, probe_input, original, codes, scales
                gc.collect()
            result["checks"] = {
                "qv_output_equal_all_blocks": all(
                    block["checks"]["qv_output_equal_all_variants"]
                    for block in result["blocks"].values()
                ),
                "identity_output_equal_all_blocks": all(
                    block["checks"]["identity_output_equal"]
                    for block in result["blocks"].values()
                ),
                "rebuild_control_output_equal_all_blocks": all(
                    block["checks"]["rebuild_control_output_equal"]
                    for block in result["blocks"].values()
                ),
            }
            if options.blocks == "41" and selected_blocks == (41,):
                result["variants"] = result["blocks"]["41"]["variants"]
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
