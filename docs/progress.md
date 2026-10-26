# Historical implementation ledger (2026-10-24)

This is a historical snapshot, not the current deployment or verification status.
The owner confirmed real ComfyUI use and validated presets on 2026-10-26.
Old staging/permission blockers below are superseded; see implementation-status.md.

# Original implementation notes

- Ruling: isolated package staged under 参考/implementation/AFM_for_MossImage because the planned parent deployment directory is outside the session write scope. No writes to parent or ComfyUI core. Cost: live installation remains gated by actual workspace authority.
- Ruling: project is new and has no git repository; use this ledger and test evidence rather than skill scripts that require an existing git history. No existing main branch is modified.
- Pre-flight A1→A2: image-first TokenLayout and FP32 filter must have the same axis semantics.
- Pre-flight A2→A3: kernel returns image rows only; adapter retains native text/CFG rows.
- Pre-flight A3→A4: run context passes through native dictionary copies; step index is callback-owned.
- Pre-flight A4→A5: node installs immutable config and creates fresh mutable state per sampling run.
- Ruling: ComfyUI venv has no pytest; run CPU tests with installed global pytest, and native-runtime checks with stdlib unittest in ComfyUI venv. No dependencies installed.

- A1 complete: test_spectral.py RED (missing afm.config) -> GREEN, 6 passed.
- A2 complete: test_attention.py RED (missing kernel) -> GREEN; suite 11 passed.
- A3/A4: core unit suite 16 passed; actual native MOSS fused/unfused observe and clone contracts passed. V3 registration test RED before node files.
- A3/A4/A5: actual ComfyUI venv tests 5 passed, including fused/unfused native forward, V3 registration and real Euler/ER-SDE callback loops.
- Unit branch-recording regression reproduced then fixed: negative branch is no longer labeled edited. Global CPU suite: 17 passed / 5 native tests skipped.
- Environment note: running native tests under global Python failed because comfy_aimdo.storage is only present in ComfyUI venv; separate environments intentionally used.
- A4 layout metadata regression: missing metadata RED -> GREEN; CPU suite 18 passed.
- A1-A4 complete for the scoped contracts; all 23 tests passed in actual ComfyUI venv.
- A5 package/schema/API examples complete; API generation inputs verified unchanged. UI graph generated but browser round-trip awaits deployment.
- A6 separate-process native GPU checks passed: small/standard/rectangular-batch2, all no-ops exactly equal, edit finite and fallback=0. Live-server installation checks pending actual parent-directory write authority.
- Final review: self-review because configured native reviewer model gpt-5.4 is unavailable. No unresolved Critical/Important code finding; independent review gap recorded.
- Ruling: full normal generation tested in a separate process to avoid writing outside the workspace or restarting the active server; cost: server registration and UI behavior remain unverified.
- Final validation: 23/23 native-environment tests passed; Python AST and API input preservation passed; all 3 GPU matrices passed. Package archive generated. Deployment remains blocked by parent write scope.
- Ruling: GPU verification uses tools/run_gpu_smoke.py instead of adding pytest to ComfyUI venv; cost: execution uses the documented helper rather than the plan's illustrative pytest command.
