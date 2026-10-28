# Implementation status — updated 2026-10-28

## Current owner-confirmed status

- The node is in use in the owner's real ComfyUI environment.
- All layers were compared in three-layer groups; `0-2`, `24-26`, `30-32` were
  selected. Default `0-2` and these three choices are authoritative.
- Owner-observed uses are overall improvement, emotional-expression emphasis,
  and pose changes respectively. These are empirical selections, not universal
  guarantees across every prompt/model/seed.
- Validation images exist locally and are deliberately excluded to keep the
  repository lightweight. No image upload or additional deployment gate is needed.
- The exact model/prompt/seed/count matrix for that full-layer sweep has not been
  supplied in the repository. Do not substitute the historical smoke matrix below
  for the preset-selection experiment.

These statements are the owner's 2026-10-26 clarification. The current maintenance
pass did not inspect the local image collection or rerun the owner's GPU workflow.
No particular browser load/save/reload procedure is inferred from general real-device use.

## Implemented foundations

Native V3 MODEL patch, image-first partition excluding text-refinement blocks,
FP32 actual-grid FFT, complementary radial masks and optional DC preservation,
progress-aligned gains, joint all-key softmax, head/query chunking and workspace
budget checks. Off/zero install no new hooks; observe keeps native dispatch.
The native output is retained for unselected CFG/text rows and recognized edit
failures. Run-scoped callbacks and cleanup protect sampler state.

## 2026-10-28 two-interval block selection

- `two_interval_mode=False` preserves the existing single-interval behavior.
  Enabling it adds a second block scope and start/end window in the same node.
- The primary interval takes priority over the entire secondary interval at every
  overlapping progress value, including the shared endpoint. Blocks are patched
  once over the union of enabled scopes; only the chosen interval is active.
- The second interval shares strength, CFG selection, frequency controls, memory
  limits and failure handling. Frequency gains continue to use whole-run progress.
- New node inputs are optional and appended after existing widgets. Existing
  API/UI examples are retained. The new `moss_afm_two_intervals_api.json` example
  switches from `30-32` to `0-2` after progress 0.5.
- The native ComfyUI prompt validator accepted both a complete legacy-input
  payload with no new fields and a two-interval payload. Native miniature-model
  tests verify fused/unfused output equivalence to the selected single interval,
  gaps, shared scopes, one wrapper per type, and all six selected preset blocks.

Fresh verification in this implementation environment:

- Python 3.10.11, PyTorch 2.10.0+cpu, local workspace ComfyUI checkout.
  The published Python requirement remains 3.12+; a Python 3.12 runtime was not
  available for a separate run in this session.
- With `PYTHONUTF8=1`, `PYTHONDONTWRITEBYTECODE=1` and `MOSS_COMFY_ROOT` pointing
  to the local ComfyUI checkout, `python -B -m pytest -q -p no:cacheprovider`:
  **57 passed, 45 subtests passed, no skipped tests**.
- `python -m compileall -q afm tools tests nodes.py __init__.py`: passed.
- `git diff --check`: passed. No typecheck command is configured and Ruff is not
  installed in this environment; no new validation dependencies were added.
- Log tests verify both interval settings, step/block/action selection, separate
  run IDs, step reset and cleanup after an exception.
- Browser UI load/save/reload was not performed. No ComfyUI server was listening
  on the checked local ports 8188/8189 and Playwright was not installed. Schema
  ordering and optional-input tests are automated checks, not a browser round-trip.
- Real-model GPU/image smoke was not run: this Python environment has CPU-only
  PyTorch and reports no available GPU. Historical GPU evidence below remains
  separate from this new feature's verification.

## 2026-10-26 maintenance changes

- Core default, node default, API/UI examples and API generator agree on `0-2`.
- Native miniature fixture has three layers, sufficient for the shipped default.
  Off-mode validation policy is otherwise unchanged.
- Smoke helper consumes the workflow's AFM settings rather than silently testing
  a different default. No-op equality is an assertion, not just a report field.
- One private candidate clone per edited attention call replaces repeated full
  output clones. Partial failure still returns the untouched native baseline.
- Query chunks compute image-key logits only; filtered text-key logits are reused.
  FFT gains/masks, schedule, branch selection and softmax semantics are unchanged.
- Stale 0.1.0 distribution ZIP replaced by a deterministic source-archive builder.

Current CPU verification is recorded in [maintenance review](final-review.md).
Numerical equivalence is tolerance-based, not a claim of bitwise-identical edited
GPU outputs. Actual speed/VRAM improvements are not measured in this maintenance pass.

## Historical implementation evidence (2026-10-24)

The original implementation record reports Python 3.12.11, PyTorch 2.12.0+rocm7.14.0,
AMD Radeon AI PRO R9700 and native ComfyUI commit
`8f04fd6ecf4a548937971e840af2a985529fab76`.
It reports 23 passing tests in that ComfyUI venv, including the overlapping CPU
subset of 18 passed / 5 native skipped. Those counts are historical and must not
be interpreted as a rerun of today's source.

Reported separate-process generation used an INT8 MoodyMoss model, Qwen3.5 INT8
encoder and Flux2 VAE:

| Configuration | Historical result |
|---|---|
| 512×512, 3 steps, CFG 2.5, ER-SDE/simple | baseline/off/zero/observe/edit succeeded |
| 1024×1024, 25 steps, CFG 2.5, ER-SDE/simple | baseline/off/zero/observe/edit succeeded |
| 768×1024, 3 steps, batch 2, CFG 1, ER-SDE/simple | baseline/observe/edit succeeded |

The record states no-op latent equality, finite changed edit latents, and zero
fallback calls. Its normal run recorded 25 edited calls using the old smoke
configuration (single block `0`), not the new three-block default. These smoke
checks do not describe the later full-layer preset-selection sweep.
Original `test-results` logs/images are local, not included in this repository;
this maintenance pass does not independently re-certify those raw artifacts.

Old staging/write-scope restrictions were session-specific and are no longer a
current deployment blocker, given the owner's real-device operation confirmation.

## Remaining optional work / unsupported combinations

Measured performance/memory profiling and controlled frequency-attribution studies
can extend the evidence, but are not prerequisites for current use. HF loader
plugins, Slider/other patches, skip-cache, compilation and multi-GPU remain
unqualified. The workspace estimate is not a total-device-memory guarantee.
