# Block Tune application windows + AFM compatibility repair

The supplied log stops in `install_afm` with `Existing diffusion wrapper requires
separate AFM qualification`. Timed Block Tune installs a diffusion wrapper and a
sampler clock; AFM currently rejects it. Removing that check alone would install
two clocks using the same run key and cause nested-run errors.

Preserve static behavior and saved application names. Qualify only the exact
known Block Tune window registration, remove its outer clock on the AFM clone,
and use the AFM outer wrapper as the single shared sampler-step clock. Keep both
diffusion wrappers: one controls weight factors, the other AFM attention context.
Their selected blocks and time intervals remain independent. Preserve the input
model's standalone behavior and reject unknown, tampered, or legacy wrappers.

- [x] Reproduce the logged install failure with a regression test.
- [x] Register/validate the known wrapper pair and share one outer clock.
- [x] Test native edit/observe/bypass, CFG, independent intervals, cleanup and restoration.
- [x] Clarify UI/help: static and full_run both apply the preset throughout;
      full_run carries the comparison clock; identity_control changes no Block
      Tune factors but does not disable AFM. Keep INT8-backend caveat explicit.
- [x] Add an AFM-combined workflow example, run full CPU checks and independent review.

Full validation with `PYTHONUTF8=1`, `MOSS_COMFY_ROOT=../ComfyUI`,
`OMP_NUM_THREADS=1`: 161 passed, 419 subtests passed. Compileall and whitespace
checks passed. Independent review: no findings; targeted native review run
21 tests / 36 subtests passed. Ruff unavailable, no dependency installed.

No model-file edits or dependency additions. Reference-image support remains
outside timed mode. GPU image quality cannot be established by CPU regression tests.
