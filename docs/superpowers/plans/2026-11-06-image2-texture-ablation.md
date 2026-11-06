# Image2 texture comparison implementation plan

User scope: add presets based on image2; isolate blocks 34–41 (zero-based,
inclusive), and also support sampling progress 0.30–0.70 only. Preserve the
existing two presets and their defaults. No model-file changes or dependencies.

Design: derive six variants from the existing image2 JSON, without duplicating
42-row tables. Global V=.90/1.00; blocks34–41 V=.90/1.00; blocks34–41
strength=1.00; blocks34–41 V=1.00 plus strength=1.00. Q/K stay unchanged.
The latter four form local comparisons against image2; the .80/1.00 V and
original/1.00 strength conditions provide a 2×2 comparison. Strength affects
both O and FFN-down, not just one branch. These are experimental controls,
not a claim that late blocks exclusively encode texture.

Timing is an independent axis. Add explicit full-interval and middle-interval
controls using the same execution path, preserving legacy static behavior.
Use sampler step progress, not layer index or raw diffusion timestep. Verify
outside-window bypass, boundaries, multiple model evaluations per step, cleanup,
and restoration after switching to the original model. Runtime design is being
checked against native ComfyUI before implementation.

- [x] Add regression tests for variant isolation and unchanged source presets.
- [x] Implement derived variants and expose them in the existing preset menu.
- [x] Add/test independent application-window control and native execution.
- [x] Document the comparison matrix and supply loadable example workflows.
- [x] Run unit/native CPU checks and independent code review (no findings).

Final timing design: keep `static` as the old default. `full_run`,
`middle_30_70`, and `identity_control` use the same projection wrapper targets
and a ContextVar gate inside the diffusion wrapper. Reuse the qualified
Euler/ER-SDE callback clock and cleanup. Dynamic inputs use the native classic
delegate. Timed modes reject AFM and reference-image paths. Outside the window,
multipliers are identity on the wrapper/dequant path, not a promise of equality
to the original INT8 kernel.

User addition: screenshot `7WMYV4XJ4M584P309VBZ1Q3QH0.png` is included as
`image_3_individual_qkv_partial`. Independent transcription check found 30
visible QKV rows and 32 visible strengths. Unknown QKV/strength/head use 1.0,
marked as unobserved. Block29 Q was independently checked as 0.91.
Comparison examples: image2 baseline + six variants + image3, four applications
each, 32 UI/API pairs from the supplied volleyball PNG metadata.

Validation: `PYTHONUTF8=1 MOSS_COMFY_ROOT=../ComfyUI OMP_NUM_THREADS=1 python -m
pytest -q`: 155 passed, 393 subtests passed. Compileall and diff whitespace
checks passed. Ruff is not installed; no dependency was added. GPU/image-quality
and browser UI import remain unverified.

Work directly in the user's named directory: it contains existing uncommitted
implementation needed by this task. Do not revert those changes or initialize
another checkout. No commit/push. No GPU/image-quality success claim.
