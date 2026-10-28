# Maintenance and feature review

## 2026-10-28 two-interval implementation

Independent read-only whole-branch review covered all 15 changed files from
`f47831f` through `ce9fb74`, plus supporting runtime and attention code.
Result: **approved, zero Critical / Important / Minor findings**.
The reviewer independently reran the native-enabled suite:
**57 passed, 45 subtests passed, no skipped tests**.
All 10 changed Python files parsed, and the diff whitespace check passed.

The review confirms primary-window precedence before block membership, one
replacement per block over the enabled union, global frequency progress,
appended optional inputs, complete legacy-payload validation, and native
fused/unfused equivalence. No separate architecture change is required: the
existing single sampler wrapper, diffusion wrapper and RunState are retained.

The following review boundaries remain explicit:

- Browser UI save/reload: not executed; schema ordering and real API validation
  passed. This is a validation gap, not a claim of a completed browser round-trip.
- Real-model GPU output and image quality: not executed in CPU-only PyTorch;
  no image-quality improvement is claimed for the new switch.
- Python 3.12-specific runtime behavior: not rerun; the available environment
  was Python 3.10.11. The published Python 3.12+ requirement is retained.
- Production performance and VRAM: not profiled; no speed or memory benefit
  is claimed for union-of-scopes registration.
- Compile, multi-GPU, skip-cache, other patches and Slider: remain outside the
  qualified integrations, as before this feature.
- Historical sparse API examples: their omitted required advanced inputs are
  a pre-existing limitation, not a new optional-input regression. Existing
  examples were retained, and the new two-interval API example includes all
  required AFM inputs. Legacy API validation uses complete old settings with
  the new optional fields absent.

Execution decisions: work stayed in the user-named checkout on feature branch
`feat/two-interval-block-selection`; a plan-scoped PowerShell ledger recorded
task evidence; UTF-8 mode resolves pre-existing locale-default test reads.
The planned GPU command now explicitly supplies 1024×1024 / 25 steps because
the smoke helper itself defaults to 512×512 / 3 steps. These decisions do not
alter the requested AFM frequency behavior.

## Maintenance review — 2026-10-26

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
