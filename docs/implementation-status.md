# Implementation status — 2026-10-24

Source staging: `参考/implementation/AFM_for_MossImage`.
Planned deployment: `C:/ComfyUI/custom_nodes/AFM_for_MossImage`.

## Implemented

- V3 `MossImageAFM` MODEL patch node.
- Native MOSS image-first partition; text refinement excluded.
- FP32 spatial FFT on full H×W, complementary radial masks and optional DC preservation.
- Curve schedule, joint all-key softmax, head/query chunks and workspace guard.
- Off/zero no-hook path; observe preserves the native dispatcher and preferred backend.
- One native baseline call; selected CFG image rows replaced atomically.
- Run-scoped state, callback-owned Euler/ER-SDE steps, cleanup on exceptions and interrupt.
- Summary JSONL with configuration, versions, actual grid/text length/branches and results.
- Baseline/observe/edit API examples preserving the specified workflow.
- UI graph draft; browser load/save/reload is pending installation.

## Verified

Actual runtime: Python 3.12.11, PyTorch 2.12.0+rocm7.14.0,
AMD Radeon AI PRO R9700, local ComfyUI source
8f04fd6ecf4a548937971e840af2a985529fab76.

All tests in ComfyUI venv: **23 passed** using unittest discovery.
Global CPU pytest: **18 passed, 5 native tests skipped**; the skipped native
tests are included in the 23 passing venv tests. These are overlapping counts.

Native contract tests cover fused/unfused QKV, actual model forward,
no-op cloning, duplicate patch rejection, V3 schema, real Euler/ER-SDE
callback loops including first-sigma adjustment.
Numerical tests cover known frequency bands, DC, rectangle topology,
joint normalization, chunk equivalence, CFG/text preservation, failure
atomicity, container consumption, progress/cleanup and workspace rejection.

### GPU generation

All runs use the supplied INT8 MoodyMoss model, Qwen3.5 INT8 encoder and Flux2 VAE.

| Configuration | Controls and results |
|---|---|
| 512×512, 3 steps, CFG 2.5, ER-SDE/simple | baseline/off/zero/observe/edit succeeded |
| **1024×1024, 25 steps, CFG 2.5, ER-SDE/simple** | baseline/off/zero/observe/edit succeeded |
| 768×1024, 3 steps, batch 2, CFG 1, ER-SDE/simple | baseline/observe/edit succeeded; actual grid 64×48 and BF16 attention recorded |

For both square cases, off/zero/observe latent tensors are **bitwise equal**
to baseline. Rectangular observe is also bitwise equal.
Nonzero edit produces finite, changed latent tensors; edit reached every
target step, with **zero fallback calls**.
The normal 25-step run records 25 edited calls on the selected positive branch.

Evidence is under `test-results/gpu-small`, `gpu-standard`,
`gpu-rect-batch2`, and `test-results-native.log`.
The rectangle run also records source-file SHA-256 values.
The normal run predates the addition of deduplicated layout metadata; its
numerical kernel and hook implementation are unchanged.

These are separate-process native ComfyUI inference checks.
They do not establish active-server registration or browser workflow round-trip.
The helper records elapsed time only as execution evidence, not a controlled
performance benchmark. No image-quality benefit is claimed.

### Nonblocking environment observations

- Standalone model loading emits upstream unexpected `.comfy_quant` key
  warnings. Native generation completed with finite outputs.
- Python interpreter shutdown emits existing ModelPatcher destructor warnings
  after the completed separate-process runs. Terminal reports and images
  are saved before shutdown.
- Global Python cannot run native contracts because its comfy_aimdo package differs;
  those tests run successfully in the actual ComfyUI venv.
- Ruff/typecheck tooling is not installed. AST parsing, whitespace checks and
  compileall pass; a full linter/typechecker was not run.

## Remaining deployment gate

This session can write only under `C:/ComfyUI/custom_nodes/AFM_for_MossImage/参考`.
The intended parent directory is outside that filesystem write scope.
No workaround was used to write outside the granted scope.

Parent deployment, restart with the existing launch arguments,
`GET /api/object_info/MossImageAFM`, active-server prompt execution and
browser UI load/save/reload remain pending. These are the remaining A5/A6
installation checks, not verified successes.

## Exclusions

Quality/spectral/statistical/performance evaluation remains in the separate
evaluation plan. HF loader plugin, Slider/other model patches, skip-cache,
compile, multiple GPUs and unqualified samplers remain unsupported.

## Review

A native code-reviewer agent was attempted, but its configured gpt-5.4 model
was unavailable for this account. Final review was therefore a separate
author self-review, supported by the unit/native/GPU evidence above.
No unresolved code correctness issue was identified in that pass.
Independent review remains a stated verification gap.
