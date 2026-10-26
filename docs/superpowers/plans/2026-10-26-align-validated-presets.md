# Align validated MOSS AFM presets

**Goal:** Keep the owner's selected `0-2`, `24-26`, `30-32` presets, default `0-2`, consistent across the repository without changing AFM behavior.

**Architecture:** Existing native attention and FFT design remain intact. The smoke harness reads the workflow's AFM configuration and verifies mode contracts. Small allocation optimizations preserve baseline rollback and numerical equivalence.

**Scope:** The owner confirmed a three-layer-group sweep across all layers, local validation images intentionally excluded for size, and actual ComfyUI use. This change does not recreate those images, claim new GPU validation, or require another deployment gate.

## Tasks

- [x] Add failing default/example/generator consistency tests; align configuration, examples and native fixture size.
- [x] Add failing smoke configuration/result validation tests; extract import-safe helpers and consume the actual AFM node inputs.
- [x] Add numerical/atomicity regression tests; remove redundant text-key multiplication and repeated candidate cloning only if equivalent.
- [x] Update README/status/review/progress provenance, record current verification separately, and replace stale distribution with reproducible packaging.
- [x] Run full CPU suite, compileall, whitespace checks and review. Native/GPU tests require unavailable ComfyUI/model environment and must be reported as skipped.

Publication to main follows successful review and the owner's explicit approval.
