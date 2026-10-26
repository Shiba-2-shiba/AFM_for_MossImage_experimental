# Maintenance review — 2026-10-26

Scope: alignment with owner-selected three-layer presets, current examples and
documentation, smoke verification contracts, and bounded allocation optimizations.

- Current default is `0-2`; only node presets `0-2`, `24-26`, `30-32` are offered.
- The owner confirmed real ComfyUI use and a full-layer, three-layer-group comparison.
  Local images are intentionally not committed. No new deployment gate is imposed.
- Previous UI/API/config mismatch is covered by CPU contract tests.
- Native test fixture is now large enough for the default; native tests remain
  environment-dependent and were not rerun in this maintenance cloud environment.
- Smoke reads AFM node settings, preserves them across variants, requires a baseline,
  and fails when off/zero/observe differ from baseline.
- Candidate output is cloned once. A simulated failure after one successful edited
  row verifies that fallback returns the original baseline, not partial output.
- The attention kernel reuses filtered text logits. Dense joint-oracle tests cover
  rectangular topology, several chunk sizes, FP32/BF16/FP16, non-default scale and
  start/mid/end progress. These are numerical-tolerance checks on CPU.
- No FFT mask/gain/schedule changes or new GPU/image-quality claims are introduced.

## Current verification

- Python 3.12.14, PyTorch 2.14.1+cpu, pytest 9.1.1 (maintenance environment).
- Full CPU test command: `python -m pytest -q`: **30 passed, 5 skipped**.
- The five skipped tests require actual ComfyUI; no native/GPU run was performed
  during this maintenance pass. Owner-confirmed prior real-device use is separate.
- `python -m compileall -q afm tools tests nodes.py __init__.py`: passed.
- `git diff --check`: passed.
- Independent code review and a separate CPU rerun found no remaining important
  correctness issue in this scoped change. The configured edit-strength-zero
  bypass is also required to equal baseline by the smoke harness.
- Historical GPU checks and old counts are separately identified in
  implementation-status.md. No measured speedup, VRAM reduction or new image-quality
  result is claimed.
